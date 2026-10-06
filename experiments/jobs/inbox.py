"""Shared, gated fleet requests. Host workers never need inter-host SSH.

All runtime state is external to the checkout. Directory creation arbitrates
ownership on NFS; a missing heartbeat never authorizes taking another owner's job.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import math
import os
import resource
from pathlib import Path
import socket
import subprocess
import sys
import time
import tarfile
import tempfile
from typing import Any
import uuid

from experiments.jobs import fleet, monitor, staging
from experiments.jobs.claims import get_process_start_time, is_pid_alive

TERMINAL = {"complete", "failed", "cancelled", "contended", "interrupted", "budget_exhausted", "expired", "attention"}
SCHEMA = 1
DEVICES = ("gpu", "cpu")
POLL_SECONDS = 60


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise fleet.FleetError(f"{label} must be positive and finite")
    return float(value)


def argv(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value or any(not isinstance(v, str) or not v or "\0" in v for v in value):
        raise fleet.FleetError(f"{label} must be a nonempty argv array")
    return value


def validate_spec(spec: Any, *, now: float | None = None) -> dict[str, Any]:
    if not isinstance(spec, dict) or spec.get("schema") != SCHEMA:
        raise fleet.FleetError("job specification requires schema: 1")
    required = {"hosts", "gpu_models", "gpu_memory_mib", "cpu_threads", "entrypoint", "arguments", "scale", "inputs", "smoke", "full", "budget_seconds", "deadline", "validator", "required_commands"}
    if required - spec.keys():
        raise fleet.FleetError(f"missing job fields: {sorted(required - spec.keys())}")
    if not spec["hosts"] or any(h not in fleet.DEFAULT_HOSTS for h in spec["hosts"]):
        raise fleet.FleetError("hosts must name configured GPU servers")
    if not isinstance(spec["gpu_models"], list) or any(not isinstance(n, str) or not n.strip() for n in spec["gpu_models"]):
        raise fleet.FleetError("gpu_models must be GPU-name substrings (empty means any model)")
    device = spec.get("device", "gpu")
    if device not in DEVICES:
        raise fleet.FleetError("device must be gpu or cpu")
    if device == "cpu":
        # A cpu job claims no GPU, so it must not describe one, nor react to its use.
        if spec["gpu_models"] or spec["gpu_memory_mib"] != 0 or isinstance(spec["gpu_memory_mib"], bool):
            raise fleet.FleetError("a cpu job declares gpu_models: [] and gpu_memory_mib: 0")
        # Validation is re-run on the normalized spec, which carries the default policy.
        if contention_policy(spec.get("contention")) != {"policy": "stop", "resume_attempts": 0}:
            raise fleet.FleetError("contention applies to a claimed GPU; a cpu job declares none")
    for key in ("gpu_memory_mib", "cpu_threads") if device == "gpu" else ("cpu_threads",):
        number(spec[key], key)
        if not isinstance(spec[key], int):
            raise fleet.FleetError(f"{key} must be an integer")
    # One Python module/script, never two unrelated commands or an inline shell.
    entry = argv(spec["entrypoint"], "entrypoint")
    if not ((len(entry) == 2 and entry[0] == "-m" and all(p.isidentifier() for p in entry[1].split("."))) or
            (len(entry) == 1 and entry[0].endswith(".py") and not Path(entry[0]).is_absolute() and ".." not in Path(entry[0]).parts)):
        raise fleet.FleetError("entrypoint must be ['-m', 'module'] or ['relative/script.py']")
    if not isinstance(spec["arguments"], list) or any(not isinstance(v, str) or "\0" in v for v in spec["arguments"]):
        raise fleet.FleetError("arguments must be strings")
    scale = spec["scale"]
    if not isinstance(scale, dict) or not scale:
        raise fleet.FleetError("scale must declare named bounded parameters")
    for name, values in scale.items():
        if not name.isidentifier() or not isinstance(values, dict) or set(values) != {"smoke", "full"}:
            raise fleet.FleetError("each scale parameter requires smoke and full values")
        if any(not isinstance(v, (str, int, float)) or isinstance(v, bool) for v in values.values()):
            raise fleet.FleetError("scale values must be finite scalar strings/numbers")
        if any(isinstance(v, float) and not math.isfinite(v) for v in values.values()):
            raise fleet.FleetError("scale values must be finite")
        if "{" + name + "}" not in spec["arguments"]:
            raise fleet.FleetError(f"scale parameter {name} must occupy a whole argument")
    staged = spec.get("staged_inputs", [])
    if not isinstance(staged, list):
        raise fleet.FleetError("staged_inputs must be an array")
    for item in staged:
        if not isinstance(item, dict) or set(item) - {"name", "path", "sha256", "extract"}:
            raise fleet.FleetError("staged inputs declare only name, path, sha256 and extract")
        if not isinstance(item.get("name"), str) or not item["name"].isidentifier():
            raise fleet.FleetError("staged input names must be identifiers")
        if not isinstance(item.get("extract", False), bool):
            raise fleet.FleetError("staged input extract must be a boolean")
        if item.get("extract") and not spec.get("local_storage_gib"):
            raise fleet.FleetError("extracted staged inputs require local_storage_gib")
    names = [item["name"] for item in staged]
    if len(names) != len(set(names)):
        raise fleet.FleetError("staged input names must be unique")
    local_storage = spec.get("local_storage_gib", 0)
    if not isinstance(local_storage, int) or isinstance(local_storage, bool) or local_storage < 0:
        raise fleet.FleetError("local_storage_gib must be a nonnegative integer")
    packed = spec.get("environment")
    if packed is not None:
        if not isinstance(packed, dict) or set(packed) != {"path", "sha256"}:
            raise fleet.FleetError("environment declares exactly path and sha256")
        if not local_storage:
            raise fleet.FleetError("a staged environment requires local_storage_gib")
    placeholders = {"{" + k + "}" for k in scale} | {"{staged:" + n + "}" for n in names}
    for arg in spec["arguments"]:
        if "{" in arg or "}" in arg:
            if arg not in placeholders:
                raise fleet.FleetError("only whole-argument scale and staged-input placeholders are supported")
    inputs = spec["inputs"]
    if not isinstance(inputs, list) or not inputs:
        raise fleet.FleetError("inputs must include immutable manifest/file identities")
    for item in [*inputs, *spec.get("staged_inputs", []), *([packed] if packed is not None else [])]:
        if not isinstance(item, dict) or not Path(item.get("path", "")).is_absolute():
            raise fleet.FleetError("input identities require absolute paths")
        sha = item.get("sha256", "")
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise fleet.FleetError("input identities require lowercase SHA256")
    for stage in ("smoke", "full"):
        number(spec[stage].get("seconds"), stage + ".seconds")
    if not spec["smoke"].get("representative_basis"):
        raise fleet.FleetError("smoke.representative_basis must explain the representative input")
    if spec["smoke"]["seconds"] > 600:
        raise fleet.FleetError("smoke must be bounded to at most 600 seconds")
    if number(spec["budget_seconds"], "budget_seconds") < spec["smoke"]["seconds"] + spec["full"]["seconds"]:
        raise fleet.FleetError("budget must reserve smoke plus the estimated full duration")
    argv(spec["validator"], "validator")
    number(spec.get("validator_seconds", 60), "validator_seconds")
    if not isinstance(spec["required_commands"], list) or any(not isinstance(v, str) or not v for v in spec["required_commands"]):
        raise fleet.FleetError("required_commands must be an array of executables")
    try:
        date = datetime.fromisoformat(spec["deadline"].replace("Z", "+00:00"))
        if date.tzinfo is None:
            raise ValueError("timezone missing")
    except (ValueError, TypeError, AttributeError) as exc:
        raise fleet.FleetError("deadline must be an ISO timestamp with timezone") from exc
    deadline = date.timestamp()
    if now is not None and deadline <= now:
        raise fleet.FleetError("deadline has already expired")
    contention = contention_policy(spec.get("contention"))
    if contention.get("pause_seconds", 0) > spec["budget_seconds"]:
        raise fleet.FleetError("contention.pause_seconds cannot exceed the budget")
    return {**spec, "device": device, "contention": contention, "deadline_epoch": deadline, "validator_seconds": spec.get("validator_seconds", 60), "stall_seconds": number(spec.get("stall_seconds", 1800), "stall_seconds")}


def contention_policy(value: Any) -> dict[str, Any]:
    """What the supervisor does when another process starts using the claimed GPU.

    ``stop`` ends the attempt (the default). ``continue`` keeps running and marks
    timing contaminated; ``basis`` must say why the results do not depend on
    timing. ``pause`` suspends the owned process group for up to ``pause_seconds``
    and continues if the GPU clears, otherwise stops. ``resume_attempts`` declares
    how many numbered attempts may restart a contended stage from its checkpoint.
    """
    if value is None:
        return {"policy": "stop", "resume_attempts": 0}
    if not isinstance(value, dict) or set(value) - {"policy", "pause_seconds", "basis", "resume_attempts"}:
        raise fleet.FleetError("contention declares only policy, pause_seconds, basis and resume_attempts")
    policy = value.get("policy")
    if policy not in monitor.CONTENTION_POLICIES:
        raise fleet.FleetError("contention.policy must be stop, continue or pause")
    result: dict[str, Any] = {"policy": policy}
    if policy == "pause":
        result["pause_seconds"] = number(value.get("pause_seconds"), "contention.pause_seconds")
    elif "pause_seconds" in value:
        raise fleet.FleetError("contention.pause_seconds applies only to the pause policy")
    if policy == "continue":
        if not isinstance(value.get("basis"), str) or not value["basis"].strip():
            raise fleet.FleetError("contention policy continue requires a basis explaining why results do not depend on timing")
        result["basis"] = value["basis"]
    resume = value.get("resume_attempts", 0)
    if not isinstance(resume, int) or isinstance(resume, bool) or resume < 0:
        raise fleet.FleetError("contention.resume_attempts must be a nonnegative integer")
    if resume and policy == "continue":
        raise fleet.FleetError("the continue policy never stops for contention, so it has nothing to resume")
    result["resume_attempts"] = resume
    return result


STAGES = ("smoke", "full")


def resume_record(directory: Path) -> dict[str, Any]:
    """The current attempt's resume declaration; attempt 1 has none."""
    return monitor.read_json(directory / "resume.json", {}) or {}


def pending_stages(directory: Path) -> list[str]:
    completed = resume_record(directory).get("completed_stages", [])
    return [stage for stage in STAGES if stage not in completed]


def stopped_stage(directory: Path) -> str | None:
    """The stage a stopped attempt was in: the last one started and not released."""
    for stage in reversed(pending_stages(directory)):
        record = monitor.read_json(directory / stage / "staging.json")
        if (directory / stage).is_dir():
            return None if record and record.get("released") else stage
    return None


def resume_decision(directory: Path, spec: dict[str, Any], state: dict[str, Any], status: dict[str, Any], now: float) -> tuple[dict[str, Any] | None, str]:
    """Whether a contended attempt may continue as a numbered new attempt, and why."""
    attempt = state.get("attempt", 1)
    allowed = spec.get("contention", {}).get("resume_attempts", 0)
    if attempt > allowed:
        return None, f"no declared resume attempts remain ({allowed} declared)"
    stage = stopped_stage(directory)
    if stage is None:
        return None, "the stopped stage is unknown"
    staged = monitor.read_json(directory / stage / "staging.json", {}) or {}
    checkpoint = staged.get("checkpoint")
    if not checkpoint or not checkpoint.get("sha256"):
        return None, f"{stage} published no checkpoint"
    consumed = state.get("consumed_seconds", 0.0) + max(0.0, status.get("ended", now) - status.get("started", now))
    stages = pending_stages(directory)
    needed = sum(spec[name]["seconds"] for name in stages[stages.index(stage):])
    remaining = min(spec["budget_seconds"] - consumed, spec["deadline_epoch"] - now)
    if remaining < needed:
        return None, f"remaining budget {remaining:.0f} s cannot support {needed:.0f} s of stages"
    return {"attempt": attempt, "stage": stage, "checkpoint": checkpoint, "consumed_seconds": consumed}, "declared resume"


def resume_attempt(directory: Path, decision: dict[str, Any], status: dict[str, Any]) -> None:
    """Archive the contended attempt and queue the next numbered attempt.

    This is not a replay: the specification declared it, it starts from the
    stopped stage's published checkpoint, and the budget and deadline span every
    attempt. Ownership is released only after the archive and record exist.
    """
    attempt, stage = decision["attempt"], decision["stage"]
    archive = directory / "resumes" / str(attempt)
    archive.mkdir(parents=True)
    for name in ("run", stage, "campaign-error.json", "environment.json"):
        if (directory / name).exists():
            (directory / name).rename(archive / name)
    if stage == "smoke":
        for name in ("validation.json", "gate.json"):
            if (directory / name).exists():
                (directory / name).rename(archive / name)
    checkpoint = {**decision["checkpoint"], "path": str(archive / stage / "checkpoint.tar")}
    previous = resume_record(directory)
    record = {
        "attempt": attempt + 1,
        "completed_stages": [s for s in STAGES if s not in pending_stages(directory) or (stage == "full" and s == "smoke")],
        "previous": {"attempt": attempt, "archive": str(archive), "stage": stage, "checkpoint": checkpoint,
                     "status": status.get("status"), "error": status.get("error"), "contention": status.get("contention")},
        "consumed_seconds": decision["consumed_seconds"],
        "declared": "contention.resume_attempts",
        "created": time.time(),
    }
    if previous:
        monitor.write_json(archive / "resume.json", previous)
    monitor.write_json(directory / "resume.json", record)
    history = [*monitor.read_json(directory / "state.json", {}).get("attempts", []), record["previous"]]
    transition(directory, "pending", attempt=attempt + 1, consumed_seconds=decision["consumed_seconds"], attempts=history)
    event(directory, f"resume:attempt-{attempt + 1}", record["previous"])
    (directory / "owner").rename(archive / "owner")


def salvage(directory: Path, reason: str) -> int:
    """Publish what a stopped stage finished, labelled partial, with its checkpoint."""
    failed = 0
    for stage in STAGES:
        output = directory / stage
        record = monitor.read_json(output / "staging.json")
        if not record or record.get("released") or record.get("salvaged"):
            continue
        scratch = Path(record["scratch"])
        salvaged: dict[str, Any] = {"reason": reason, "time": time.time()}
        if scratch.is_dir():
            try:
                record["partial"] = staging.publish_scratch(scratch, output, partial=True)
                record["checkpoint"] = staging.publish_checkpoint(scratch / "checkpoint", output)
            except (OSError, tarfile.TarError) as exc:
                salvaged["error"] = str(exc)
                failed = 1
        else:
            salvaged["error"] = "scratch is not present on this host"
            failed = 1
        record["salvaged"] = salvaged
        monitor.write_json(output / "staging.json", record)
    return failed


def verify_inputs(spec: dict[str, Any], data_root: Path) -> None:
    for item in spec["inputs"]:
        path = Path(item["path"]).resolve()
        if not path.is_relative_to(data_root.resolve()) or not path.is_file():
            raise fleet.FleetError(f"input identity must be a file under the external data root: {path}")
        if file_digest(path) != item["sha256"]:
            raise fleet.FleetError(f"input identity changed: {path}")


def event(directory: Path, kind: str, detail: Any = None) -> None:
    identity = uuid.uuid5(uuid.NAMESPACE_URL, str(directory) + kind + digest(detail)).hex
    target = directory / "events" / f"{identity}.json"
    if not target.exists():
        monitor.write_json(target, {"id": identity, "kind": kind, "detail": detail, "timestamp": time.time()})


def transition(directory: Path, status: str, **fields: Any) -> dict[str, Any]:
    value = {**monitor.read_json(directory / "state.json", {}), **fields, "status": status, "updated": time.time()}
    monitor.write_json(directory / "state.json", value)
    if status in TERMINAL:
        event(directory, "terminal:" + status, fields.get("error"))
    return value


def acquire_request(directory: Path, alias: str) -> bool:
    """Permanent execution ownership. Never steal it based on elapsed time."""
    try:
        (directory / "owner").mkdir()
    except FileExistsError:
        return False
    monitor.write_json(directory / "owner" / "identity.json", {
        "alias": alias, "host": socket.getfqdn().lower(), "pid": os.getpid(),
        "proc_start_time": get_process_start_time(os.getpid()), "token": uuid.uuid4().hex,
    })
    return True


def local_probe(alias: str) -> dict[str, Any]:
    result = subprocess.run([sys.executable, "-c", fleet._PROBE_PYTHON], capture_output=True, text=True, timeout=60)
    value = json.loads(result.stdout.strip().splitlines()[-1])
    value.update(alias=alias, available=result.returncode == 0 and not value.get("probe_error"))
    return value


def eligible(spec: dict[str, Any], host: dict[str, Any]) -> dict[str, Any] | None:
    """The GPU this job claims on this host; None for an admitted cpu job."""
    if host["alias"] not in spec["hosts"]:
        raise fleet.FleetError("host is not compatible")
    host = {**host, "gpus": [g for g in host.get("gpus", []) if not spec["gpu_models"] or any(m in g["name"] for m in spec["gpu_models"])]}
    for command in spec["required_commands"]:
        import shutil
        if not shutil.which(command):
            raise fleet.FleetError("required executable unavailable: " + command)
    storage = staging.admission_error(spec)
    if storage:
        raise fleet.FleetError(storage)
    if spec.get("device", "gpu") == "cpu":
        fleet.select_cpu([host], cpu_threads=spec["cpu_threads"])
        return None
    _, gpu = fleet.select_gpu([host], required_memory_mib=spec["gpu_memory_mib"], cpu_threads=spec["cpu_threads"])
    return gpu


def reconcile(directory: Path, alias: str) -> None:
    state = monitor.read_json(directory / "state.json", {})
    owner = monitor.read_json(directory / "owner" / "identity.json", {})
    if owner.get("alias") != alias or state.get("status") in TERMINAL:
        return
    run = directory / "run"
    status = monitor.read_json(run / "status.json", {})
    supervisor = monitor.read_json(run / "supervisor.json", {})
    if status.get("admission_rejected") and not status.get("pid"):
        if supervisor.get("pid") and not is_pid_alive(supervisor["pid"], supervisor.get("proc_start_time")):
            # No child ever existed, and the supervisor released any acquired
            # resources before recording rejection. Archive, then release only
            # this request's ownership; this is admission retry, not job replay.
            attempt = directory / "attempts" / uuid.uuid4().hex
            attempt.parent.mkdir(exist_ok=True)
            run.rename(attempt)
            transition(directory, "pending", admission_retry=status.get("error"))
            (directory / "owner").rename(attempt / "owner")
        return
    if status.get("status") in TERMINAL:
        if supervisor.get("pid") and is_pid_alive(supervisor["pid"], supervisor.get("proc_start_time")):
            return  # still salvaging or delivering; decide once it has exited
        fields: dict[str, Any] = {"remote_status": status}
        if status["status"] == "contended":
            spec = monitor.read_json(directory / "spec.json")
            decision, reason = resume_decision(directory, spec, state, status, time.time())
            if decision is not None:
                resume_attempt(directory, decision, status)
                return
            fields["resume_declined"] = reason
        transition(directory, status["status"], **fields)
        return
    identity = supervisor or owner
    if identity.get("pid") and is_pid_alive(identity["pid"], identity.get("proc_start_time")):
        return
    transition(directory, "attention", error="Execution owner disappeared; inspect supervisor and owned processes. No replay is permitted.")


def start_request(directory: Path, spec: dict[str, Any], gpu: dict[str, Any] | None, host: dict[str, Any], base: Path) -> None:
    if not acquire_request(directory, host["alias"]):
        return
    state = monitor.read_json(directory / "state.json", {})
    if state.get("status") != "pending":
        return
    if (directory / "cancel.json").exists():
        transition(directory, "cancelled")
        return
    # Budget spans every declared attempt; waiting between attempts is bounded by the deadline.
    remaining = min(spec["budget_seconds"] - state.get("consumed_seconds", 0.0), spec["deadline_epoch"] - time.time())
    if remaining < sum(spec[stage]["seconds"] for stage in pending_stages(directory)):
        transition(directory, "expired", error="Insufficient time remains before the deadline")
        return
    if digest(spec) != monitor.read_json(directory / "ready.json")["spec_sha256"]:
        transition(directory, "attention", error="Saved specification identity changed")
        return
    verify_inputs(spec, base.parent.parent)
    run = directory / "run"
    run.mkdir()
    (run / "events").mkdir()
    source = directory / "source"
    monitor.write_json(run / "request.json", {
        "command": [sys.executable, "-m", "experiments.jobs.inbox", "campaign", str(directory)],
        "cwd": str(source), "budget_seconds": remaining, "deadline_epoch": spec["deadline_epoch"],
        "thread": None, "codex": "codex", "cancel_path": str(directory / "cancel.json"),
        "contention": spec.get("contention") or {"policy": "stop"}, "attempt": state.get("attempt", 1),
        "salvage": [sys.executable, "-m", "experiments.jobs.inbox", "salvage", str(directory)],
        # Without a GPU claim the supervisor claims only CPU threads and hides every GPU.
        "claims": {"gpu": gpu["uuid"] if gpu else None, "cpu_threads": spec["cpu_threads"], "claims_dir": str(base.parent / "claims"), "min_free_memory_mb": spec["gpu_memory_mib"] + fleet.DEFAULT_MEMORY_MARGIN_MIB},
    })
    monitor.write_json(run / "policy.json", {"stall_seconds": spec["stall_seconds"], "report_at": None, "quiet_until": 0})
    monitor.write_json(directory / "environment.json", {"host": host["host"], "alias": host["alias"], "gpu": gpu, "python": host.get("python_version"), "tools": host.get("tool_versions"), "started": time.time()})
    transition(directory, "launching", host_alias=host["alias"], gpu=gpu)
    env = {**os.environ, "PYTHONPATH": str(source), "PS_DATA_ROOT": str(base.parent.parent)}
    with (run / "supervisor.log").open("a") as output:
        child = subprocess.Popen([sys.executable, "-m", "experiments.jobs.monitor", "serve", str(run)], cwd=source, env=env, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    monitor.write_json(run / "supervisor.json", {"pid": child.pid, "proc_start_time": get_process_start_time(child.pid), "host": host["host"]})
    transition(directory, "running", supervisor_pid=child.pid)


def worker_tick(base: Path, alias: str) -> None:
    # Reap only our own completed supervisors before checking their liveness.
    while True:
        try:
            reaped, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            break
        if reaped == 0:
            break
    worker = base / "workers" / alias
    monitor.write_json(worker / "heartbeat.json", {"pid": os.getpid(), "proc_start_time": get_process_start_time(os.getpid()), "updated": time.time(), "alias": alias})
    directories = sorted((base / "inbox").glob("*"))
    host = None
    for directory in directories:
        if not (directory / "ready.json").exists():
            continue
        if (directory / "owner").exists():
            reconcile(directory, alias)
            continue
        state = monitor.read_json(directory / "state.json", {})
        if state.get("status") != "pending":
            continue
        try:
            spec = monitor.read_json(directory / "spec.json")
            if time.time() >= spec["deadline_epoch"] or (directory / "cancel.json").exists():
                if acquire_request(directory, alias):
                    transition(directory, "cancelled" if (directory / "cancel.json").exists() else "expired")
                continue
            if alias not in spec["hosts"]:
                continue
            if host is None:
                host = local_probe(alias)
            gpu = eligible(spec, host)
            start_request(directory, spec, gpu, host, base)
            # Do not reuse the pre-launch occupancy snapshot for a second job.
            host = None
        except (OSError, ValueError, subprocess.SubprocessError, fleet.FleetError) as exc:
            identity = monitor.read_json(directory / "owner" / "identity.json", {})
            if identity.get("alias") == alias:
                transition(directory, "attention", error=str(exc))
            else:
                monitor.write_json(worker / "admission.json", {"job_id": directory.name, "reason": str(exc), "updated": time.time()})


def worker(base: Path, alias: str) -> int:
    location = base / "workers" / alias
    location.mkdir(parents=True, exist_ok=True)
    # Serialize bootstrap/recovery as well as the full worker lifetime. Two
    # starters must never both reclaim the same verified-dead worker slot.
    with (location / "worker.lock").open("a") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return worker_loop(base, alias)


def worker_loop(base: Path, alias: str) -> int:
    location = base / "workers" / alias
    location.mkdir(parents=True, exist_ok=True)
    lock = location / "active"
    try:
        lock.mkdir()
    except FileExistsError:
        old = monitor.read_json(lock / "identity.json", {})
        if not old.get("pid") or is_pid_alive(old["pid"], old.get("proc_start_time")):
            raise fleet.FleetError("worker ownership is active or unresolved")
        # Only a process on this configured host may reclaim its dead worker slot.
        if old.get("host") != socket.getfqdn().lower():
            raise fleet.FleetError("worker hostname changed; inspect the existing ownership")
        (lock / "identity.json").unlink()
        lock.rmdir()
        lock.mkdir()
    monitor.write_json(lock / "identity.json", {"pid": os.getpid(), "proc_start_time": get_process_start_time(os.getpid()), "host": socket.getfqdn().lower()})
    try:
        while True:
            try:
                worker_tick(base, alias)
            except Exception as exc:
                monitor.write_json(location / "error.json", {"error": str(exc), "updated": time.time()})
            time.sleep(POLL_SECONDS)
    finally:
        (lock / "identity.json").unlink(missing_ok=True)
        lock.rmdir()


def phase_command(spec: dict[str, Any], stage: str, staged: list[dict[str, Any]] | None = None, python: str = sys.executable) -> list[str]:
    replacements = {"{" + name + "}": str(values[stage]) for name, values in spec["scale"].items()}
    replacements.update({"{staged:" + item["name"] + "}": item["path"] for item in staged or []})
    return [python, *spec["entrypoint"], *[replacements.get(arg, arg) for arg in spec["arguments"]]]


def run_bounded(command: list[str], *, cwd: Path, env: dict[str, str], log: Any, seconds: float, budget_end: float, run: Path) -> int:
    """Run one stage or validator command under its allowance.

    Time that a declared contention pause held the job stopped extends only the
    command's own allowance; the budget and deadline still bound it.
    """
    started = time.time()
    child = subprocess.Popen(command, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    while True:
        try:
            return child.wait(timeout=1)
        except subprocess.TimeoutExpired:
            now = time.time()
            if now < budget_end and now - started < seconds:
                continue
            if now < budget_end and now - started < seconds + monitor.paused_seconds(monitor.read_json(run / "status.json", {}) or {}, started, now):
                continue
            # Let the workload finish a checkpoint before it is killed.
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise fleet.FleetError(f"command exceeded its {seconds:.0f} s allowance or the budget")


def contention_during(run: Path, started: float, ended: float) -> dict[str, Any]:
    """The supervisor's contention episodes that overlapped one command."""
    status = monitor.read_json(run / "status.json", {}) or {}
    episodes = [e for e in status.get("contention") or [] if e["detected"] <= ended and e.get("ended", e.get("cleared", ended)) >= started]
    return {"timing_contaminated": bool(episodes), "contention": episodes, "paused_seconds": monitor.paused_seconds(status, started, ended)}


def campaign(directory: Path) -> int:
    """The monitor owns this process group across smoke, validation, and full."""
    previous = monitor.read_json(directory / "run" / "status.json", {})
    resume = resume_record(directory)
    attempt = resume.get("attempt", 1)
    completed = resume.get("completed_stages", [])
    leftovers = [name for name in (*pending_stages(directory), "campaign-error.json") if (directory / name).exists()]
    if "smoke" not in completed and (directory / "gate.json").exists():
        leftovers.append("gate.json")
    if previous.get("status") in TERMINAL or leftovers or (monitor.read_json(directory / "run" / "request.json", {}) or {}).get("attempt", 1) != attempt:
        print("campaign: existing attempt is preserved; create a new request instead of replaying", file=sys.stderr)
        return 1
    spec = monitor.read_json(directory / "spec.json")
    manifest = monitor.read_json(directory / "ready.json")
    source = directory / "source"
    run = directory / "run"
    started = time.time()
    supervisor_state = monitor.read_json(run / "status.json", {})
    supervisor_request = monitor.read_json(run / "request.json", {})
    budget_end = min(supervisor_state.get("started", started) + supervisor_request.get("budget_seconds", spec["budget_seconds"]), spec["deadline_epoch"])
    try:
        if digest(spec) != manifest["spec_sha256"]:
            raise fleet.FleetError("saved specification identity changed")
        code_identity = source_identity(source)
        if code_identity != manifest["source_sha256"]:
            raise fleet.FleetError("frozen code identity changed")
        verify_inputs(spec, directory.parents[3])
        for item in [*spec.get("staged_inputs", []), *([spec["environment"]] if spec.get("environment") else [])]:
            path = Path(item["path"]).resolve()
            if not path.is_relative_to(directory.parents[3].resolve()) or not path.is_file():
                raise fleet.FleetError(f"staged input must be a file under the external data root: {path}")
        if "smoke" in completed:
            # A resumed attempt keeps an earlier attempt's gate only under identical identities.
            gate = monitor.read_json(directory / "gate.json", {}) or {}
            if gate.get("passed") is not True or gate.get("spec_sha256") != digest(spec) or gate.get("source_sha256") != code_identity or gate.get("inputs") != spec["inputs"]:
                raise fleet.FleetError("a resumed attempt requires the earlier smoke gate under unchanged identities")
        local_root = staging.local_root()
        for stage in pending_stages(directory):
            verify_inputs(spec, directory.parents[3])
            if digest(monitor.read_json(directory / "spec.json")) != manifest["spec_sha256"]:
                raise fleet.FleetError("specification changed after smoke")
            if stage == "full" and source_identity(source) != code_identity:
                raise fleet.FleetError("code changed after smoke")
            if time.time() + spec[stage]["seconds"] > budget_end:
                raise fleet.FleetError("Remaining budget cannot support the saved stage estimate")
            output = directory / stage
            output.mkdir()
            # Re-verified per stage, so promotion also requires unchanged staged bytes.
            staged = staging.stage_inputs(spec.get("staged_inputs", []), local_root)
            scratch_name = directory.name if attempt == 1 else f"{directory.name}-attempt{attempt}"
            scratch = staging.scratch_directory(local_root, scratch_name, stage, output, declared=bool(spec.get("local_storage_gib")))
            resumed_from = resume["previous"] if resume.get("previous", {}).get("stage") == stage else None
            if resumed_from:
                staging.restore_checkpoint(resumed_from["checkpoint"], scratch / "checkpoint")
            packed = staging.stage_environment(spec["environment"], local_root) if spec.get("environment") else None
            python = packed["python"] if packed else sys.executable
            staging_record: dict[str, Any] = {
                "local_root": str(local_root) if local_root else None, "inputs": staged, "environment": packed,
                "scratch": str(scratch), "attempt": attempt,
                "restored_checkpoint": resumed_from["checkpoint"] if resumed_from else None,
            }
            monitor.write_json(output / "staging.json", staging_record)
            command = phase_command(spec, stage, staged, python)
            env = {
                **os.environ, "PS_STAGE": stage, "PS_STAGE_DIR": str(output), "PS_SCRATCH_DIR": str(scratch),
                "PS_CHECKPOINT_DIR": str(scratch / "checkpoint"), "PS_ATTEMPT": str(attempt),
                "PS_VALIDATION_PATH": str(directory / "validation.json"),
            }
            if packed:
                env.update(PATH=os.pathsep.join([packed["bin"], env.get("PATH", "")]), CONDA_PREFIX=packed["prefix"])
            monitor.write_json(output / "dispatch.json", {
                "command": command, "input_identity": spec["inputs"], "staged_inputs": staged, "staged_environment": packed,
                "code": manifest, "environment": monitor.read_json(directory / "environment.json", {}), "started": time.time(),
                "citable": False if stage == "smoke" else None, "attempt": attempt, "resumed_from": resumed_from,
                "contention_policy": spec.get("contention"),
            })
            monitor.publish_progress(stage, 0)
            phase_started = time.time()
            with (output / "command.log").open("w") as log:
                returncode = run_bounded(command, cwd=source, env=env, log=log, seconds=spec[stage]["seconds"], budget_end=budget_end, run=run)
            phase_ended = time.time()
            monitor.write_json(output / "execution.json", {
                "exit_code": returncode, "seconds": phase_ended - phase_started,
                "children_resource_usage": list(resource.getrusage(resource.RUSAGE_CHILDREN)),
                **contention_during(run, phase_started, phase_ended),
            })
            if returncode:
                # Failed scratch stays on the host for inspection; staging.json names it.
                raise fleet.FleetError(f"{stage} failed with exit code {returncode}")
            staging_record["published"] = staging.publish_scratch(scratch, output)
            staging.release_scratch(scratch, output)
            staging_record["released"] = time.time()
            monitor.write_json(output / "staging.json", staging_record)
            monitor.publish_progress(stage, 1)
            if stage == "smoke":
                validation_command = [python if a == "{python}" else a for a in spec["validator"]]
                validator_started = time.time()
                with (output / "validator.log").open("w") as log:
                    checked = run_bounded(validation_command, cwd=source, env=env, log=log, seconds=spec["validator_seconds"], budget_end=budget_end, run=run)
                validation = monitor.read_json(directory / "validation.json", {})
                if checked or validation.get("passed") is not True or not validation.get("checks"):
                    raise fleet.FleetError("smoke validator must exit zero and publish passed:true with substantive checks")
                monitor.write_json(directory / "gate.json", {"passed": True, "spec_sha256": digest(spec), "source_sha256": code_identity, "inputs": spec["inputs"], "validation": validation, "validator_seconds": time.time() - validator_started, "smoke_seconds": time.time() - started, "remaining_seconds": budget_end - time.time(), "attempt": attempt})
        return 0
    except Exception as exc:
        monitor.write_json(directory / "campaign-error.json", {"error": str(exc), "updated": time.time()})
        monitor.publish_progress("attention", 0, decision=str(exc))
        return 1


def source_identity(source: Path) -> str:
    # DirEntry caches directory metadata. Path.rglob followed by repeated stat
    # calls is prohibitively slow on the fleet's shared NFS mount.
    records = []
    def scan(directory: Path) -> None:
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                rel = path.relative_to(source)
                if entry.name == "__pycache__" or path.suffix == ".pyc":
                    continue
                if entry.is_symlink():
                    if not path.resolve().is_relative_to(source.resolve()):
                        raise fleet.FleetError(f"snapshot symlink escapes frozen source: {rel}")
                    records.append([str(rel), "symlink", os.readlink(path)])
                elif entry.is_dir(follow_symlinks=False):
                    scan(path)
                elif entry.is_file(follow_symlinks=False):
                    records.append([str(rel), file_digest(path)])
    scan(source)
    records.sort(key=lambda record: Path(record[0]).parts)
    return digest(records)


def job_directory(base: Path, job_id: str) -> Path:
    if not fleet.JOB_ID_RE.fullmatch(job_id):
        raise fleet.FleetError("invalid job ID")
    return base / "inbox" / job_id


def rpc(base: Path, action: str, payload: dict[str, Any]) -> Any:
    """Small remote operations, executed from the installed frozen release."""
    if action == "selftest_input":
        path = base / "checks" / ("smoke-" + uuid.uuid4().hex) / "input.json"
        monitor.write_json(path, {"infrastructure_smoke": True})
        path.chmod(0o444)
        return {"path": str(path), "sha256": file_digest(path)}
    if action == "prepare":
        directory = job_directory(base, payload["job_id"])
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "source").mkdir()
        (directory / "events").mkdir()
        return {"directory": str(directory)}
    if action == "publish":
        directory = job_directory(base, payload["job_id"])
        if (directory / "ready.json").exists():
            raise fleet.FleetError("request is already published")
        spec = validate_spec(payload["spec"], now=time.time())
        verify_inputs(spec, base.parent.parent)
        for rel in payload["snapshot"].get("deleted_paths", []):
            path = Path(rel)
            if path.is_absolute() or ".." in path.parts:
                raise fleet.FleetError("unsafe deleted snapshot path")
            (directory / "source" / path).unlink(missing_ok=True)
        identity = source_identity(directory / "source")
        monitor.write_json(directory / "spec.json", spec)
        (directory / "spec.json").chmod(0o444)
        transition(directory, "pending", job_id=directory.name, created=time.time())
        # Publish last: workers cannot observe an incompletely transferred job.
        monitor.write_json(directory / "ready.json", {**payload["snapshot"], "spec_sha256": digest(spec), "source_sha256": identity, "published": time.time()})
        return {"job_id": directory.name, "status": "pending", "source_sha256": identity}
    if action in {"status", "events", "cancel"}:
        directories = [job_directory(base, payload["job_id"])] if payload.get("job_id") else sorted((base / "inbox").glob("*"))
        results = []
        for directory in directories:
            if not (directory / "ready.json").exists():
                if payload.get("job_id"):
                    raise fleet.FleetError("job is unpublished or unknown; inspect transfer before retrying")
                continue
            state = monitor.read_json(directory / "state.json", {})
            remote = monitor.read_json(directory / "run" / "status.json", {})
            if remote.get("status"):
                state = {**state, "status": remote["status"], "remote_status": remote}
                if remote.get("admission_rejected") and not remote.get("pid"):
                    state.update(status="pending", admission_reason=remote.get("error"))
                spec = monitor.read_json(directory / "spec.json", {}) or {}
                declared = (spec.get("contention") or {}).get("resume_attempts", 0)
                if remote["status"] == "contended" and state.get("attempt", 1) <= declared and monitor.read_json(directory / "state.json", {}).get("status") not in TERMINAL:
                    # The owning worker has not yet decided on the declared resume.
                    state.update(status="resuming")
            if state.get("status") == "pending" and (directory / "owner").exists() and not remote:
                state.update(status="preparing", execution_owner=monitor.read_json(directory / "owner" / "identity.json", {}))
            if action == "cancel":
                if state.get("status") not in TERMINAL:
                    request = {"requested_at": time.time(), "request_id": uuid.uuid4().hex}
                    monitor.write_json(directory / "cancel.json", request)
                    if (directory / "run").is_dir():
                        monitor.write_json(directory / "run" / "cancel.json", request)
                    state = {**state, "cancellation_requested": True}
            if action == "events":
                records = [monitor.read_json(path) for root in (directory / "events", directory / "run" / "events") for path in sorted(root.glob("*.json"))]
                unique = {}
                for record in records:
                    if record["kind"].startswith("terminal:"):
                        record["id"] = uuid.uuid5(uuid.NAMESPACE_URL, str(directory) + record["kind"]).hex
                    unique[record["id"]] = record
                results.append({"job_id": directory.name, "status": state.get("status"), "events": list(unique.values())})
            else:
                results.append({**state, "job_id": directory.name, "directory": str(directory), "gate": monitor.read_json(directory / "gate.json"), "campaign_error": monitor.read_json(directory / "campaign-error.json")})
        return results[0] if payload.get("job_id") else results
    if action == "worker_start":
        alias = payload["alias"]
        if alias not in fleet.DEFAULT_HOSTS:
            raise fleet.FleetError("unconfigured worker host")
        location = base / "workers" / alias
        location.mkdir(parents=True, exist_ok=True)
        previous = monitor.read_json(location / "active" / "identity.json", {})
        if isinstance(previous, dict) and previous.get("pid") and previous.get("host") == socket.getfqdn().lower() and is_pid_alive(previous["pid"], previous.get("proc_start_time")):
            return {"alias": alias, "pid": previous["pid"], "status": "already_running"}
        # Worker itself arbitrates singleton startup. This acknowledgement is
        # provisional until its heartbeat and ownership match the new PID.
        with (location / f"worker-{int(time.time())}-{uuid.uuid4().hex[:8]}.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-m", "experiments.jobs.inbox", "worker", str(base), alias], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=Path(__file__).resolve().parents[2], env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2]), "PS_DATA_ROOT": str(base.parent.parent)})
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise fleet.FleetError(f"worker {alias} exited during startup; inspect its log")
            identity = monitor.read_json(location / "active" / "identity.json", {})
            if isinstance(identity, dict) and identity.get("pid") == process.pid:
                return {"alias": alias, "pid": process.pid, "status": "running", "release": str(Path(__file__).resolve().parents[2])}
            time.sleep(0.1)
        raise fleet.FleetError("worker startup acknowledgement missing; inspect before retrying")
    if action == "health":
        return {"workers": {h: monitor.read_json(base / "workers" / h / "heartbeat.json") for h in payload["hosts"]}, "base": str(base)}
    raise fleet.FleetError("unknown remote operation")


def remote_rpc(config: dict[str, Any], action: str, payload: dict[str, Any], *, alias: str | None = None) -> Any:
    alias = alias or config["hosts"][0]
    script = "import json,sys; sys.path.insert(0," + repr(config["release"]) + "); from pathlib import Path; from experiments.jobs.inbox import rpc; print(json.dumps(rpc(Path(" + repr(config["base"]) + ")," + repr(action) + ",json.loads(" + repr(json.dumps(payload)) + "))))"
    result = fleet._ssh(alias, [config["pythons"][alias], "-c", script], timeout=120)
    if result.returncode:
        raise fleet.FleetError(f"{alias}: {action} failed: {(result.stderr or result.stdout)[-4000:]}")
    return json.loads(result.stdout.strip().splitlines()[-1])


def load_config(state_dir: Path) -> dict[str, Any]:
    config = monitor.read_json(state_dir / "inbox.json")
    if not config:
        raise fleet.FleetError("workers are not configured; run workers start first")
    if not config.get("hosts") or any(h not in fleet.DEFAULT_HOSTS for h in config["hosts"]):
        raise fleet.FleetError("invalid fleet configuration")
    return config


def doctor(hosts: list[str]) -> dict[str, Any]:
    """Verify common visibility AND one-winner mkdir, retaining the report."""
    probes = fleet.inspect_fleet(tuple(hosts))
    failures = [h for h in probes if not h.get("available") or not h.get("data_root_available") or not h.get("python_available")]
    if failures:
        return {"passed": False, "probes": probes, "error": "all participating hosts must pass environment probes"}
    roots = {h["data_root"] for h in probes}
    if len(roots) != 1:
        return {"passed": False, "probes": probes, "error": "hosts report different data roots"}
    base = str(Path(probes[0]["data_root"]) / "jobs" / "fleet")
    nonce = uuid.uuid4().hex
    path = str(Path(base) / "checks" / nonce)
    token = uuid.uuid4().hex
    init = "import pathlib; p=pathlib.Path(" + repr(path) + "); p.mkdir(parents=True); (p/'token').write_text(" + repr(token) + ")"
    fleet._remote_json(hosts[0], probes[0]["python"], init + "; print('{}')")
    script = "import pathlib,json,socket; p=pathlib.Path(" + repr(path) + "); visible=(p/'token').read_text()==" + repr(token) + "; won=False\nif visible:\n try:\n  (p/'claim').mkdir(); won=True\n except FileExistsError: pass\nprint(json.dumps({'visible':visible,'won':won,'host':socket.getfqdn()}))"
    with ThreadPoolExecutor(max_workers=len(hosts)) as pool:
        futures = {h["alias"]: pool.submit(fleet._remote_json, h["alias"], h["python"], script) for h in probes}
        checks = {}
        for alias, future in futures.items():
            try:
                checks[alias] = future.result()
            except Exception as exc:
                checks[alias] = {"visible": False, "error": str(exc)}
    passed = all(c.get("visible") for c in checks.values()) and sum(c.get("won", False) for c in checks.values()) == 1
    report = {"passed": passed, "base": base, "probes": probes, "atomic_checks": checks, "checked_at": time.time(), "evidence_path": path}
    fleet._remote_json(hosts[0], probes[0]["python"], "import pathlib,json; pathlib.Path(" + repr(path + "/report.json") + ").write_text(" + repr(json.dumps(report)) + "); print('{}')")
    return report


def worker_bundle(snapshot: Path) -> tuple[Path, dict[str, str]]:
    """Freeze the manager's dependency closure, without unrelated experiment files."""
    def included(name: str) -> bool:
        return name in {"src/__init__.py", "experiments/__init__.py"} or name.startswith(("src/contracts/", "experiments/jobs/"))
    with tempfile.NamedTemporaryFile(prefix="pointstream-worker-", suffix=".tar", delete=False) as temporary:
        bundle = Path(temporary.name)
    identities = {}
    try:
        with tarfile.open(snapshot) as original, tarfile.open(bundle, "w") as target:
            for member in original:
                if included(member.name):
                    data = original.extractfile(member) if member.isfile() else None
                    target.addfile(member, data)
                    if data is not None:
                        data.close()
                        content = original.extractfile(member)
                        if content is None:
                            raise OSError(f"could not read snapshot member {member.name}")
                        with content:
                            identities[member.name] = hashlib.sha256(content.read()).hexdigest()
        return bundle, identities
    except Exception:
        bundle.unlink(missing_ok=True)
        raise


def stop_worker(config: dict[str, Any], alias: str) -> Any:
    """Replace only a verified fleet worker, preserving detached supervisors."""
    script = """import json,pathlib,os,signal,socket,time,sys
sys.path.insert(0, RELEASE)
from experiments.jobs.claims import is_pid_alive
location=pathlib.Path(BASE)/'workers'/ALIAS
path=location/'active'/'identity.json'
if not path.exists(): print('{}'); raise SystemExit(0)
identity=json.loads(path.read_text()); pid=identity.get('pid'); start=identity.get('proc_start_time')
if identity.get('host')!=socket.getfqdn().lower() or not pid or start is None: raise SystemExit('unresolved worker ownership; no signal sent')
if not is_pid_alive(pid,start): print('{}'); raise SystemExit(0)
command=pathlib.Path('/proc')/str(pid)/'cmdline'
args=command.read_bytes().split(bytes([0]))
expected=[b'-m',b'experiments.jobs.inbox',b'worker',BASE.encode(),ALIAS.encode()]
if args[1:6]!=expected: raise SystemExit('worker process identity differs; no signal sent')
os.kill(pid,signal.SIGTERM)
deadline=time.monotonic()+30
while is_pid_alive(pid,start) and time.monotonic()<deadline: time.sleep(.1)
if is_pid_alive(pid,start): raise SystemExit('worker did not stop; inspect before retrying')
print(json.dumps({'stopped_worker_pid':pid}))
""".replace("RELEASE", repr(config["release"])).replace("BASE", repr(config["base"])).replace("ALIAS", repr(alias))
    return fleet._remote_json(alias, config["pythons"][alias], script, timeout=60)


def workers_start(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    report = doctor(args.hosts)
    if not report["passed"]:
        raise fleet.FleetError(json.dumps(report))
    release_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    release = str(Path(report["base"]) / "releases" / release_id)
    first = report["probes"][0]
    config = {"schema": SCHEMA, "hosts": args.hosts, "base": report["base"], "release": release, "pythons": {h["alias"]: h["python"] for h in report["probes"]}, "doctor": report["evidence_path"]}
    snapshot, sha, metadata = fleet._build_snapshot(root, include_changes=tuple(args.include_change), include_untracked=tuple(args.include_untracked))
    original_snapshot = snapshot
    snapshot, identities = worker_bundle(original_snapshot)
    original_snapshot.unlink(missing_ok=True)
    sha = file_digest(snapshot)
    metadata.update(worker_bundle_sha256=sha, worker_files_sha256=identities)
    try:
        fleet._remote_json(first["alias"], first["python"], "import pathlib; pathlib.Path(" + repr(release) + ").mkdir(parents=True,exist_ok=False); print('{}')")
        fleet._send_snapshot(first["alias"], snapshot, release, sha)
        # Apply explicit deletions, then preserve the installed revision record.
        script = "import pathlib,json; root=pathlib.Path(" + repr(release) + "); meta=json.loads(" + repr(json.dumps(metadata)) + ")\nfor rel in meta['deleted_paths']: root.joinpath(rel).unlink(missing_ok=True)\n(root/'worker-revision.json').write_text(json.dumps(meta)); print('{}')"
        fleet._remote_json(first["alias"], first["python"], script)
    finally:
        snapshot.unlink(missing_ok=True)
    # Save before starting: an ambiguous SSH acknowledgement must be recoverable.
    monitor.write_json(args.state_dir / "inbox.json", config)
    started = []
    for alias in args.hosts:
        if args.operation == "restart":
            stop_worker(config, alias)
        started.append(remote_rpc(config, "worker_start", {"alias": alias}, alias=alias))
    return {"config": config, "workers": started, "snapshot": metadata}


def snapshot_root(root: Path, selected: Path | None) -> Path:
    """Permit an explicit checkout of this repository, never another repo."""
    if selected is None:
        return root
    selected = selected.resolve(strict=True)
    def common(path: Path) -> Path:
        value = fleet._git(path, "rev-parse", "--git-common-dir").stdout.decode().strip()
        return (path / value).resolve()
    top = fleet._git(selected, "rev-parse", "--show-toplevel").stdout.decode().strip()
    if Path(top).resolve() != selected or common(root) != common(selected):
        raise fleet.FleetError("source worktree must be a checkout of this same repository")
    return selected


def submit(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    source_root = snapshot_root(root, getattr(args, "source_worktree", None))
    transfer_seconds = getattr(args, "snapshot_transfer_seconds", None)
    if transfer_seconds is not None:
        number(transfer_seconds, "snapshot transfer seconds")
        if transfer_seconds > 90:
            raise fleet.FleetError("bounded snapshot transfer seconds must be <=90")
    config = load_config(args.state_dir)
    spec = validate_spec(json.loads(args.spec.read_text()), now=time.time())
    if not set(spec["hosts"]).issubset(config["hosts"]):
        raise fleet.FleetError("job hosts must have verified workers")
    health = remote_rpc(config, "health", {"hosts": config["hosts"]})
    for alias in spec["hosts"]:
        heartbeat = health["workers"].get(alias)
        if not heartbeat or time.time() - heartbeat["updated"] > 3 * POLL_SECONDS:
            raise fleet.FleetError(f"worker {alias} has no fresh heartbeat; run doctor and inspect before submission")
    job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    archive_paths = tuple(getattr(args, "snapshot_path", []))
    archive_options = {"include_paths": archive_paths} if archive_paths else {}
    snapshot, sha, metadata = fleet._build_snapshot(source_root, include_changes=tuple(args.include_change), include_untracked=tuple(args.include_untracked), **archive_options)
    record = {"job_id": job_id, "status": "preparing", "config": config, "spec_sha256": digest(spec), "snapshot": metadata, "chat_id": args.chat_id}
    monitor.write_json(args.state_dir / f"{job_id}.json", record)
    try:
        prepared = remote_rpc(config, "prepare", {"job_id": job_id})
        transfer_options = {"timeout": transfer_seconds} if transfer_seconds is not None else {}
        fleet._send_snapshot(config["hosts"][0], snapshot, str(Path(prepared["directory"]) / "source"), sha, **transfer_options)
        response = remote_rpc(config, "publish", {"job_id": job_id, "spec": spec, "snapshot": metadata})
        record.update(response)
        monitor.write_json(args.state_dir / f"{job_id}.json", record)
        return {**record, "monitoring": {"chat_id": args.chat_id, "interval_minutes": 5, "events_command": str(root / "scripts" / "ps-fleet") + " events " + job_id, "heartbeat_required": bool(args.chat_id)}}
    except Exception as exc:
        # Publication/launch may have succeeded even when the reply was lost.
        monitor.write_json(args.state_dir / f"{job_id}.json", {**record, "status": "submission_unknown", "error": str(exc)})
        raise fleet.FleetError(f"job {job_id}: submission outcome unknown; inspect this ID, never resubmit automatically: {exc}") from exc
    finally:
        snapshot.unlink(missing_ok=True)


def remote_job(args: argparse.Namespace) -> Any:
    if args.job_id:
        path = args.state_dir / f"{args.job_id}.json"
        if not fleet.JOB_ID_RE.fullmatch(args.job_id):
            raise fleet.FleetError("invalid job ID")
        record = monitor.read_json(path)
        if not record:
            raise fleet.FleetError("no local job record")
        if "config" not in record:
            if args.action == "events":
                raise fleet.FleetError("legacy job events are in the recorded remote run directory")
            return fleet.cancel_job(args.job_id, args.state_dir) if args.action == "cancel" else fleet.status_job(args.job_id, args.state_dir)
        # Preserve the original execution config as provenance, but use the
        # current compatible management release for this same shared inbox.
        current = monitor.read_json(args.state_dir / "inbox.json", {})
        management = current if current.get("base") == record["config"].get("base") and current.get("hosts") else record["config"]
        result = remote_rpc(management, args.action, {"job_id": args.job_id})
        if args.action == "events":
            # Acknowledge only after the consumer actually reports the event.
            seen = monitor.read_json(args.state_dir / "event-acks.json", {})
            result["events"] = [e for e in result["events"] if e["id"] not in seen and not (args.state_dir / "event-acks" / (e["id"] + ".json")).exists()]
        return result
    return remote_rpc(load_config(args.state_dir), "status", {})


def selftest_submit(args: argparse.Namespace, root: Path) -> Any:
    config = load_config(args.state_dir)
    identity = remote_rpc(config, "selftest_input", {})
    spec = {
        "schema": SCHEMA, "hosts": args.hosts or config["hosts"], "gpu_models": [],
        "gpu_memory_mib": 1024, "cpu_threads": 1,
        "entrypoint": ["-m", "experiments.jobs.fleet_smoke"],
        "arguments": ["--input", identity["path"], "--iterations", "{iterations}"],
        "scale": {"iterations": {"smoke": 2, "full": 8}}, "inputs": [identity],
        "smoke": {"seconds": 60, "representative_basis": "Infrastructure only: same input and CUDA operation in both stages"},
        "full": {"seconds": 60}, "budget_seconds": 300,
        "validator": ["{python}", "-m", "experiments.jobs.fleet_smoke", "--validate"],
        "validator_seconds": 30, "required_commands": [],
        "deadline": datetime.fromtimestamp(time.time() + 900, timezone.utc).isoformat(),
    }
    args.spec = args.state_dir / "smoke-specs" / (uuid.uuid4().hex + ".json")
    monitor.write_json(args.spec, spec)
    return submit(args, root)


def watch_chat(state_dir: Path, chat_id: str) -> dict[str, Any]:
    """One native heartbeat watches every request submitted by the same chat."""
    from argparse import Namespace

    jobs = []
    for path in sorted(state_dir.glob("*.json")):
        record = monitor.read_json(path, {})
        if record.get("chat_id") != chat_id or not record.get("job_id"):
            continue
        args = Namespace(state_dir=state_dir, job_id=record["job_id"], action="status")
        try:
            status = remote_job(args)
            args.action = "events"
            events = remote_job(args)
            jobs.append({"job_id": args.job_id, "status": status, "events": events["events"]})
        except Exception as exc:
            jobs.append({"job_id": args.job_id, "status": {"status": "unreachable"}, "error": str(exc)})
    return {"chat_id": chat_id, "jobs": jobs, "all_terminal": all(j["status"].get("status") in TERMINAL for j in jobs)}


def main(argv_values: list[str] | None = None, *, public_only: bool = False) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=fleet.DEFAULT_STATE_DIR)
    actions = parser.add_subparsers(dest="action", required=True)
    for name in ("inspect", "doctor"):
        command = actions.add_parser(name)
        command.add_argument("--hosts", nargs="+", choices=fleet.DEFAULT_HOSTS, default=list(fleet.DEFAULT_HOSTS))
    workers = actions.add_parser("workers")
    workers.add_argument("operation", choices=["start", "status", "restart"])
    workers.add_argument("--hosts", nargs="+", choices=fleet.DEFAULT_HOSTS, default=list(fleet.DEFAULT_HOSTS))
    submission = actions.add_parser("submit")
    submission.add_argument("spec", type=Path)
    submission.add_argument("--source-worktree", type=Path, help="snapshot an explicit checkout of this same repository; default is the canonical checkout")
    submission.add_argument("--snapshot-transfer-seconds", type=float, help="bound each archive transfer/extraction to at most 90 seconds; default preserves the fleet timeout")
    submission.add_argument("--snapshot-path", action="append", default=[], help="archive only this reviewed tracked file/directory from HEAD; repeat for a complete workload dependency set")
    submission.add_argument("--chat-id", default=os.environ.get("CODEX_THREAD_ID"), help="submitting Codex chat; register its five-minute heartbeat")
    for command in (workers, submission):
        command.add_argument("--include-change", action="append", default=[])
        command.add_argument("--include-untracked", action="append", default=[])
    selftest = actions.add_parser("selftest", help="submit a bounded non-citable CUDA infrastructure campaign")
    selftest.add_argument("--hosts", nargs="+", choices=fleet.DEFAULT_HOSTS, help="limit the bounded selftest to verified compatible hosts")
    selftest.add_argument("--chat-id", default=os.environ.get("CODEX_THREAD_ID"))
    selftest.add_argument("--include-change", action="append", default=[])
    selftest.add_argument("--include-untracked", action="append", default=[])
    status = actions.add_parser("status")
    status.add_argument("job_id", nargs="?")
    status.add_argument("--artifact", action="append", default=[], help="bounded read-only export of a selected job artifact")
    status.add_argument("--output", type=Path, help="new local artifact export directory")
    for name in ("events", "cancel"):
        command = actions.add_parser(name)
        command.add_argument("job_id")
    watch = actions.add_parser("watch")
    watch.add_argument("chat_id")
    ack = actions.add_parser("ack")
    ack.add_argument("event_ids", nargs="+")
    for name in (() if public_only else ("worker", "campaign", "salvage", "rpc")):
        command = actions.add_parser(name, help=argparse.SUPPRESS)
        command.add_argument("directory", type=Path)
        if name == "salvage":
            command.add_argument("reason")
        if name == "worker":
            command.add_argument("alias", choices=fleet.DEFAULT_HOSTS)
        if name == "rpc":
            command.add_argument("operation")
            command.add_argument("payload")
    args = parser.parse_args(argv_values)
    args.state_dir = args.state_dir.expanduser()
    root = Path(__file__).resolve().parents[2]
    try:
        if args.action == "worker":
            return worker(args.directory.resolve(), args.alias)
        if args.action == "campaign":
            return campaign(args.directory.resolve())
        if args.action == "salvage":
            return salvage(args.directory.resolve(), args.reason)
        if args.action == "rpc":
            value = rpc(args.directory.resolve(), args.operation, json.loads(args.payload))
        elif args.action == "inspect":
            value = fleet.inspect_fleet(tuple(args.hosts))
        elif args.action == "doctor":
            value = doctor(args.hosts)
            report_path = args.state_dir / "checks" / (uuid.uuid4().hex + ".json")
            monitor.write_json(report_path, value)
            value["local_report"] = str(report_path)
        elif args.action == "workers":
            if args.operation == "status":
                config = load_config(args.state_dir)
                value = remote_rpc(config, "health", {"hosts": config["hosts"]})
                value["fresh"] = {alias: bool(heartbeat and time.time() - heartbeat["updated"] <= 3 * POLL_SECONDS) for alias, heartbeat in value["workers"].items()}
            else:
                value = workers_start(args, root)
        elif args.action == "submit":
            value = submit(args, root)
        elif args.action == "selftest":
            value = selftest_submit(args, root)
        elif args.action == "watch":
            value = watch_chat(args.state_dir, args.chat_id)
        elif args.action == "ack":
            for event_id in args.event_ids:
                if len(event_id) != 32 or any(c not in "0123456789abcdef" for c in event_id):
                    raise fleet.FleetError("invalid event ID")
                monitor.write_json(args.state_dir / "event-acks" / (event_id + ".json"), {"acknowledged": time.time()})
            value = {"acknowledged": args.event_ids}
        else:
            if getattr(args, "artifact", []):
                if not args.job_id or args.output is None:
                    raise fleet.FleetError("artifact export needs a recorded job ID and --output")
                if not fleet.JOB_ID_RE.fullmatch(args.job_id):
                    raise fleet.FleetError("invalid job ID")
                from experiments.jobs.artifacts import export
                record = monitor.read_json(args.state_dir / f"{args.job_id}.json")
                if not record:
                    raise fleet.FleetError("no local job record")
                value = export(record, args.artifact, args.output)
            else:
                value = remote_job(args)
        print(json.dumps(value, indent=2, allow_nan=False))
        return 1 if isinstance(value, dict) and value.get("passed") is False else 0
    except (fleet.FleetError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        print(f"ps-fleet: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
