"""Inspect the GPU fleet and dispatch isolated, supervised PointStream jobs.

Run from the local PointStream checkout. Other users may still allocate a GPU
after launch-time checks; see ``docs/workflow/long-jobs.md`` for limits.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tarfile
import tempfile
import uuid
from typing import Any


DEFAULT_HOSTS = tuple(f"gpu{n}" for n in range(1, 7))
DEFAULT_IDLE_MEMORY_MIB = 256
DEFAULT_MEMORY_MARGIN_MIB = 4096
DEFAULT_IDLE_UTILIZATION_PCT = 5
DEFAULT_STATE_DIR = Path.home() / ".pointstream" / "fleet" / "jobs"
SNAPSHOT_TRANSFER_TIMEOUT_SECONDS = 300
DEVICE_RANK = ("RTX 6000 Ada", "RTX A6000", "RTX 8000", "GV100")
SNAPSHOT_EXCLUDED_PATHS = ("demo/outputs",)
JOB_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$")


class FleetError(RuntimeError):
    """A fleet probe, admission or remote operation failed."""


def _ssh(host: str, remote_argv: list[str], *, timeout: float = 30) -> subprocess.CompletedProcess[str]:
    """Run one noninteractive SSH command and return text output."""
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, shlex.join(remote_argv)],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _probe_script(remote_python: str) -> str:
    return f"""set -eu
exec {shlex.quote(remote_python)} -c {shlex.quote(_PROBE_PYTHON)}
"""


_PROBE_PYTHON = r'''import json, math, os, pathlib, shutil, subprocess
def query(args):
    p = subprocess.run(["nvidia-smi", *args], check=True, capture_output=True, text=True, timeout=12)
    return p.stdout.strip().splitlines()
try:
    gpu_lines = query(["--query-gpu=index,uuid,name,memory.used,memory.total,memory.free,utilization.gpu", "--format=csv,noheader,nounits"])
    app_lines = query(["--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"])
except (OSError, subprocess.SubprocessError) as exc:
    print(json.dumps({"probe_error": str(exc)})); raise SystemExit(0)
apps = {}
app_query_error = None
for line in app_lines:
    fields = [x.strip() for x in line.split(",")]
    if len(fields) < 2:
        app_query_error = "malformed GPU compute-process response"
        break
    try: pid = int(fields[1])
    except ValueError:
        app_query_error = "malformed GPU process ID"
        break
    try: used = int(fields[2]) if len(fields) > 2 else None
    except ValueError: used = None
    apps.setdefault(fields[0], []).append({"pid": pid, "memory_mib": used})
if app_query_error:
    print(json.dumps({"probe_error": app_query_error})); raise SystemExit(0)
gpus = []
for line in gpu_lines:
    fields = [x.strip() for x in line.split(",")]
    if len(fields) != 7: continue
    try:
        index, used, total, free, util = int(fields[0]), int(fields[3]), int(fields[4]), int(fields[5]), int(fields[6])
        gpus.append({"index": index, "uuid": fields[1], "name": fields[2], "memory_used_mib": used, "memory_total_mib": total, "memory_free_mib": free, "utilization_pct": util, "compute_processes": apps.get(fields[1], [])})
    except ValueError: continue
if len(gpus) != len(gpu_lines):
    print(json.dumps({"probe_error": "GPU query returned an incomplete or malformed device list", "gpus": gpus})); raise SystemExit(0)
try: affinity = len(os.sched_getaffinity(0))
except (AttributeError, OSError): affinity = os.cpu_count() or 1
capacity = float(affinity)
for p in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
    try:
        if p.endswith("cpu.max"):
            q, period = pathlib.Path(p).read_text().split()[:2]
            if q != "max": capacity = min(capacity, float(q) / float(period))
        else:
            q = float(pathlib.Path(p).read_text().strip())
            period = float(pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text().strip())
            if q > 0 and period > 0: capacity = min(capacity, q / period)
    except (OSError, ValueError, IndexError, ZeroDivisionError): pass
try: load1 = os.getloadavg()[0]
except (AttributeError, OSError): load1 = 0.0
data = os.environ.get("PS_DATA_ROOT", "").strip()
if not data:
    marker = pathlib.Path.home() / "pointstream" / ".ps-data-root"
    try: data = marker.read_text().strip()
    except OSError: data = ""
if not data: data = str(pathlib.Path.home() / "pointstream-data")
data = str(pathlib.Path(data).expanduser().resolve())
host_key = __import__("socket").getfqdn().strip().lower()
claim_root = pathlib.Path(data) / "jobs" / "claims" / "devices"
try:
    for gpu in gpus:
        safe_uuid = "".join(ch if ch.isalnum() or ch in "_.-" else "_" for ch in gpu["uuid"])
        gpu["pointstream_claimed"] = (claim_root / (host_key + "_" + safe_uuid)).is_dir()
except OSError as exc:
    print(json.dumps({"probe_error": "resource-claim path could not be checked: " + str(exc)})); raise SystemExit(0)
env_python = os.path.expanduser("~/.conda/envs/pointstream/bin/python")
tools = {name: shutil.which(name) for name in ("ffmpeg", "vvencapp", "vvdecapp", "codex")}
tool_versions = {}
for name in ("ffmpeg", "vvencapp", "vvdecapp"):
    path = tools[name]
    if path:
        try:
            version = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=10, check=False)
            tool_versions[name] = {"path": path, "version": (version.stdout or version.stderr).splitlines()[:3], "returncode": version.returncode}
        except (OSError, subprocess.SubprocessError) as exc:
            tool_versions[name] = {"path": path, "version_error": str(exc)}
python_version = None
if os.path.isfile(env_python):
    try:
        version = subprocess.run([env_python, "--version"], capture_output=True, text=True, timeout=10, check=False)
        python_version = (version.stdout or version.stderr).strip()
    except (OSError, subprocess.SubprocessError): pass
print(json.dumps({"host": __import__("socket").getfqdn(), "gpus": gpus, "cpu_capacity": capacity, "load1": load1, "cpu_headroom": max(0, math.floor(capacity - load1)), "data_root": data, "data_root_available": os.path.isdir(data) and os.access(data, os.W_OK), "python": env_python, "python_available": os.path.isfile(env_python) and os.access(env_python, os.X_OK), "python_version": python_version, "tools": tools, "tool_versions": tool_versions}))'''


def probe_host(host: str, *, remote_python: str = "/usr/bin/python3", timeout: float = 20) -> dict[str, Any]:
    """Query one server; a failed or incomplete response is never admitted."""
    if host not in DEFAULT_HOSTS:
        return {"alias": host, "available": False, "error": "server is not in the configured GPU fleet", "gpus": []}
    try:
        result = _ssh(host, ["bash", "-lc", _probe_script(remote_python)], timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"alias": host, "available": False, "error": f"SSH/environment probe timed out after {timeout:g} seconds", "gpus": []}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"alias": host, "available": False, "error": str(exc), "gpus": []}
    if result.returncode:
        return {"alias": host, "available": False, "error": (result.stderr or f"ssh exited {result.returncode}").strip(), "gpus": []}
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"alias": host, "available": False, "error": "malformed GPU probe response", "gpus": []}
    if not isinstance(payload, dict) or payload.get("probe_error") or not isinstance(payload.get("gpus"), list) or not payload["gpus"]:
        return {"alias": host, "available": False, "error": payload.get("probe_error", "incomplete GPU probe response") if isinstance(payload, dict) else "incomplete GPU probe response", "gpus": []}
    required_gpu_fields = {"uuid", "name", "memory_used_mib", "memory_total_mib", "memory_free_mib", "utilization_pct", "compute_processes"}
    for gpu in payload["gpus"]:
        if not isinstance(gpu, dict) or not required_gpu_fields.issubset(gpu):
            return {"alias": host, "available": False, "error": "incomplete GPU status record", "gpus": []}
    payload.update(alias=host, available=True, error=None)
    return payload


def inspect_fleet(hosts: tuple[str, ...], *, remote_python: str = "/usr/bin/python3") -> list[dict[str, Any]]:
    """Probe all hosts concurrently so selection reflects one short time window."""
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(8, len(hosts))) as pool:
        futures = {pool.submit(probe_host, host, remote_python=remote_python): host for host in hosts}
        for future in as_completed(futures):
            host = futures[future]
            try:
                results[host] = future.result()
            except Exception as exc:  # defensive boundary around each host
                results[host] = {"alias": host, "available": False, "error": str(exc), "gpus": []}
    return [results[host] for host in hosts]


def _device_rank(name: str, preferred_names: tuple[str, ...] = ()) -> tuple[int, str]:
    order = (*preferred_names, *(family for family in DEVICE_RANK if family not in preferred_names))
    for index, family in enumerate(order):
        if family.casefold() in name.casefold():
            return (index, name.casefold())
    return (len(order), name.casefold())


def select_gpu(
    hosts: list[dict[str, Any]],
    *,
    required_memory_mib: int,
    cpu_threads: int,
    idle_memory_mib: int = DEFAULT_IDLE_MEMORY_MIB,
    idle_utilization_pct: int = DEFAULT_IDLE_UTILIZATION_PCT,
    required_paths: tuple[str, ...] = (),
    required_commands: tuple[str, ...] = (),
    prefer_gpu_names: tuple[str, ...] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pick a free compatible device; reject hosts with any failed prerequisite."""
    if required_memory_mib <= 0 or cpu_threads <= 0:
        raise FleetError("GPU memory estimate and CPU thread allowance must be positive")
    if any(not name.strip() for name in prefer_gpu_names):
        raise FleetError("preferred GPU name substrings must not be empty")
    candidates: list[tuple[tuple[int, int, int, str], dict[str, Any], dict[str, Any]]] = []
    reasons: list[str] = []
    for host in hosts:
        if not host.get("available"):
            reasons.append(f"{host.get('alias')}: {host.get('error', 'unavailable')}")
            continue
        if not host.get("python_available") or not host.get("data_root_available"):
            reasons.append(f"{host['alias']}: PointStream Python or writable data root is unavailable")
            continue
        headroom = host.get("cpu_headroom")
        if not isinstance(headroom, int) or cpu_threads > math.floor(0.90 * headroom):
            reasons.append(f"{host['alias']}: CPU allowance exceeds 90% of current headroom")
            continue
        missing_paths = [p for p in required_paths if not os.path.isabs(p)]
        if missing_paths:
            raise FleetError(f"--require-path values must be absolute remote paths: {missing_paths}")
        command_status = host.get("required_commands_available")
        missing_commands = [
            name
            for name in required_commands
            if not (
                command_status.get(name)
                if isinstance(command_status, dict)
                else host.get("tools", {}).get(name)
            )
        ]
        if missing_commands:
            reasons.append(f"{host['alias']}: missing commands {', '.join(missing_commands)}")
            continue
        for gpu in host["gpus"]:
            if gpu.get("pointstream_claimed"):
                continue
            if gpu["compute_processes"]:
                continue
            if gpu["memory_used_mib"] > idle_memory_mib:
                continue
            if gpu["utilization_pct"] > idle_utilization_pct:
                continue
            if gpu["memory_free_mib"] < required_memory_mib + DEFAULT_MEMORY_MARGIN_MIB:
                continue
            order = _device_rank(gpu["name"], prefer_gpu_names)
            candidates.append(((order[0], -gpu["memory_free_mib"], gpu["utilization_pct"], host["alias"]), host, gpu))
    if not candidates:
        detail = "; ".join(reasons) if reasons else "all inspected GPUs are busy, non-idle, or below the memory requirement"
        raise FleetError(f"No eligible GPU is available: {detail}")
    _, host, gpu = min(candidates, key=lambda item: item[0])
    return host, gpu


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def _build_snapshot(
    root: Path,
    *,
    include_changes: tuple[str, ...] = (),
    include_untracked: tuple[str, ...] = (),
    include_paths: tuple[str, ...] = (),
) -> tuple[Path, str, dict[str, Any]]:
    """Create a tar of HEAD plus explicitly selected changes and new files."""
    root = root.resolve()
    for rel in include_paths:
        pure = PurePosixPath(rel)
        if not rel or rel == "." or pure.is_absolute() or ".." in pure.parts or rel.startswith(":"):
            raise FleetError(f"invalid selected archive path: {rel}")
        if any(rel == excluded or rel.startswith(excluded + "/") for excluded in SNAPSHOT_EXCLUDED_PATHS):
            raise FleetError(f"snapshot path is excluded by policy: {rel}")
    temp = tempfile.NamedTemporaryFile(prefix="pointstream-fleet-", suffix=".tar", delete=False)
    snapshot = Path(temp.name)
    try:
        archive = subprocess.Popen(
            [
                "git", "-C", str(root), "archive", "--format=tar", "HEAD", "--", *(include_paths or (".",)),
                ":(exclude)demo/outputs/**",
            ],
            stdout=temp,
            stderr=subprocess.PIPE,
        )
        stderr = archive.communicate()[1]
        if archive.returncode:
            raise FleetError(f"git archive failed: {stderr.decode(errors='replace')}")
        temp.close()
        changed = _git(root, "diff", "HEAD", "--name-status", "-z").stdout.decode().split("\0")
        entries = [entry for entry in changed if entry]
        all_names: list[tuple[str, str]] = []
        index = 0
        while index < len(entries):
            status = entries[index]
            index += 1
            if status.startswith(("R", "C")):
                old, new = entries[index:index + 2]
                all_names.extend((("D", old), ("M", new)))
                index += 2
            else:
                all_names.append((status[0], entries[index]))
                index += 1
        changed_paths = {path for _, path in all_names}
        selected_changes: set[str] = set()
        for rel in include_changes:
            pure = PurePosixPath(rel)
            if pure.is_absolute() or ".." in pure.parts or rel not in changed_paths:
                raise FleetError(f"invalid or unchanged tracked snapshot path: {rel}")
            if any(rel == excluded or rel.startswith(excluded + "/") for excluded in SNAPSHOT_EXCLUDED_PATHS):
                raise FleetError(f"snapshot path is excluded by policy: {rel}")
            selected_changes.add(rel)
        names = [item for item in all_names if item[1] in selected_changes]
        excluded_changes = sorted(
            path for path in changed_paths
            if path not in selected_changes
            and not any(path == excluded or path.startswith(excluded + "/") for excluded in SNAPSHOT_EXCLUDED_PATHS)
        )
        excluded_changes = [path for path in excluded_changes if path not in include_changes]
        ignored_untracked: list[str] = []
        for rel in include_untracked:
            pure = PurePosixPath(rel)
            if pure.is_absolute() or ".." in pure.parts or not (root / rel).is_file():
                raise FleetError(f"invalid or missing explicit snapshot file: {rel}")
            if any(rel == excluded or rel.startswith(excluded + "/") for excluded in SNAPSHOT_EXCLUDED_PATHS):
                raise FleetError(f"snapshot path is excluded by policy: {rel}")
            names.append(("A", rel))
        if any(status != "D" for status, _ in names):
            with tarfile.open(snapshot, mode="a") as tar:
                for status, rel in names:
                    pure = PurePosixPath(rel)
                    if pure.is_absolute() or ".." in pure.parts:
                        raise FleetError(f"unsafe path in git status: {rel}")
                    if status == "D":
                        continue
                    source = root / rel
                    if not source.exists():
                        continue
                    tar.add(source, arcname=str(pure), recursive=source.is_dir())
        raw = _git(root, "rev-parse", "HEAD").stdout.decode().strip()
        patch_args = ["diff", "HEAD", "--binary", "--"]
        patch_args.extend(include_changes)
        patch = _git(root, *patch_args).stdout if include_changes else b""
        patch_hash = hashlib.sha256(patch).hexdigest()
        status_output = _git(root, "status", "--short", "--untracked-files=all").stdout.decode().splitlines()
        untracked = [line[3:] for line in status_output if line.startswith("?? ")]
        ignored_untracked = [path for path in untracked if path not in include_untracked]
        deleted_paths = [path for status, path in names if status == "D"]
        digest_state = hashlib.sha256()
        with snapshot.open("rb") as archive_stream:
            for block in iter(lambda: archive_stream.read(1024 * 1024), b""):
                digest_state.update(block)
        digest = digest_state.hexdigest()
        included_untracked = {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in include_untracked
        if (root / path).is_file()
        }
        excluded_paths_hash = hashlib.sha256("\n".join(sorted(ignored_untracked)).encode()).hexdigest()
        metadata = {
            "git_head": raw,
            "tracked_worktree_patch_sha256": patch_hash,
            "tracked_changes_included": sorted(selected_changes),
            "tracked_changes_excluded": excluded_changes,
            "included_untracked_sha256": included_untracked,
            "untracked_excluded_count": len(ignored_untracked),
            "untracked_excluded_paths_sha256": excluded_paths_hash,
            "untracked_excluded_sample": ignored_untracked[:20],
            "deleted_paths": deleted_paths,
            "snapshot_sha256": digest,
        }
        if include_paths:
            metadata["tracked_archive_paths_selected"] = list(include_paths)
        return snapshot, digest, metadata
    except Exception:
        snapshot.unlink(missing_ok=True)
        raise
    finally:
        try:
            temp.close()
        except OSError:
            pass


def _remote_path(host: dict[str, Any], *parts: str) -> str:
    return str(Path(host["data_root"]).joinpath("jobs", "fleet", *parts))


def _send_snapshot(host: str, source: Path, target: str, expected_sha256: str, *, timeout: float | None = None) -> None:
    timeout = SNAPSHOT_TRANSFER_TIMEOUT_SECONDS if timeout is None else timeout
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise FleetError("snapshot transfer timeout must be positive and finite")
    remote_tar = str(Path(target).with_suffix(".tar"))
    receive = f"umask 077; set -o noclobber; cat > {shlex.quote(remote_tar)}"
    with source.open("rb") as stream:
        proc = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, shlex.join(["bash", "-lc", receive])],
            stdin=stream,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    if proc.returncode:
        raise FleetError(f"{host}: snapshot transfer failed: {proc.stderr.decode(errors='replace').strip()}")
    checksum = _ssh(host, ["sha256sum", "--", remote_tar])
    if checksum.returncode or checksum.stdout.split(maxsplit=1)[0] != expected_sha256:
        raise FleetError(f"{host}: transferred snapshot checksum does not match the local archive")
    extract = _ssh(
        host,
        ["tar", "-xf", remote_tar, "-C", target],
        timeout=timeout,
    )
    if extract.returncode:
        raise FleetError(f"{host}: snapshot extraction failed: {extract.stderr.strip()}")
    remove_archive = _ssh(host, ["/usr/bin/python3", "-c", "import pathlib,sys; pathlib.Path(sys.argv[1]).unlink()", remote_tar])
    if remove_archive.returncode:
        raise FleetError(f"{host}: snapshot extracted but its temporary archive could not be removed")


def _remote_json(host: str, python: str, script: str, *, timeout: float = 30) -> dict[str, Any]:
    result = _ssh(host, [python, "-c", script], timeout=timeout)
    if result.returncode:
        raise FleetError(f"{host}: remote operation failed: {result.stderr.strip()}")
    try:
        value = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise FleetError(f"{host}: malformed response from remote operation") from exc
    if not isinstance(value, dict):
        raise FleetError(f"{host}: malformed response from remote operation")
    return value


def inspect_required_commands(
    hosts: list[dict[str, Any]], required_commands: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Check bare PATH names and absolute executable paths on every candidate."""
    if not required_commands:
        return hosts
    invalid = [
        name
        for name in required_commands
        if not name or (os.path.sep in name and not os.path.isabs(name))
    ]
    if invalid:
        raise FleetError(f"--require-command values must be bare names or absolute paths: {invalid}")
    script = (
        "import json,os,shutil; names="
        + repr(list(required_commands))
        + "; found={n:(os.path.isfile(n) and os.access(n,os.X_OK)) "
        "if os.path.isabs(n) else bool(shutil.which(n)) for n in names}; "
        "print(json.dumps({'commands':found}))"
    )
    futures: dict[Any, dict[str, Any]] = {}
    reachable = [host for host in hosts if host.get("available")]
    if not reachable:
        return hosts
    with ThreadPoolExecutor(max_workers=min(8, len(reachable))) as pool:
        for host in reachable:
            future = pool.submit(_remote_json, host["alias"], host["python"], script)
            futures[future] = host
        for future, host in futures.items():
            try:
                result = future.result()
                status = result.get("commands")
                if not isinstance(status, dict) or any(name not in status or not isinstance(status[name], bool) for name in required_commands):
                    raise FleetError("incomplete required-command probe response")
                host["required_commands_available"] = {name: status[name] for name in required_commands}
            except Exception as exc:
                host.update(available=False, error=f"required-command probe failed: {exc}")
    return hosts


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def launch_job(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    hosts = inspect_fleet(tuple(args.hosts), remote_python=args.probe_python)
    hosts = inspect_required_commands(hosts, tuple(args.require_command))
    for host in hosts:
        if host.get("available"):
            absent = [path for path in args.require_path if not os.path.isabs(path)]
            if absent:
                raise FleetError(f"required remote paths must be absolute: {absent}")
            if not host.get("python_available"):
                host["required_paths_available"] = False
                host["missing_required_paths"] = ["PointStream Python environment unavailable"]
                continue
            check = _remote_json(host["alias"], host["python"], _path_check_script(args.require_path))
            host["required_paths_available"] = check.get("available") is True
            host["missing_required_paths"] = check.get("missing", [])
    viable = [host for host in hosts if not args.require_path or host.get("required_paths_available")]
    selected_host, selected_gpu = select_gpu(
        viable,
        required_memory_mib=args.gpu_memory_mib,
        cpu_threads=args.cpu_threads,
        idle_memory_mib=args.idle_memory_mib,
        idle_utilization_pct=args.idle_utilization_pct,
        required_commands=tuple(args.require_command),
        prefer_gpu_names=tuple(args.prefer_gpu_name),
    )
    job_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    run_dir = _remote_path(selected_host, "runs", job_id)
    source_dir = _remote_path(selected_host, "snapshots", job_id)
    local_manifest_path = Path(args.state_dir).expanduser() / f"{job_id}.json"
    snapshot: Path | None = None
    manifest: dict[str, Any] = {
        "job_id": job_id,
        "status": "preparing",
        "host_alias": selected_host["alias"],
        "host": selected_host["host"],
        "gpu": selected_gpu,
        "command": args.command,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "estimated_gpu_memory_mib": args.gpu_memory_mib,
        "cpu_threads": args.cpu_threads,
        "run_dir": run_dir,
        "source_dir": source_dir,
        "data_root": selected_host["data_root"],
        "runtime_environment": {"pointstream_python": selected_host.get("python_version"), "native_tools": selected_host.get("tool_versions", {})},
        "selection": {"policy": "no compute processes; memory use <= idle threshold; utilization <= idle threshold; free memory >= estimate + 4 GiB; CPU allowance <= 90% headroom", "idle_memory_mib": args.idle_memory_mib, "idle_utilization_pct": args.idle_utilization_pct, "memory_margin_mib": DEFAULT_MEMORY_MARGIN_MIB, "hardware_order": list(args.prefer_gpu_name) + [name for name in DEVICE_RANK if name not in args.prefer_gpu_name], "hardware_order_is_heuristic": not bool(args.prefer_gpu_name), "preferred_gpu_names": list(args.prefer_gpu_name)},
    }
    _write_manifest(local_manifest_path, manifest)
    try:
        include_untracked = ("experiments/jobs/fleet.py", *args.include_untracked)
        snapshot, digest, git_meta = _build_snapshot(
            root,
            include_changes=tuple(args.include_change),
            include_untracked=include_untracked,
        )
        manifest.update(git_meta, status="transferring")
        manifest["source_dir"] = source_dir
        _write_manifest(local_manifest_path, manifest)
        parent = _ssh(selected_host["alias"], ["mkdir", "-p", "--", str(Path(source_dir).parent)])
        if parent.returncode:
            raise FleetError(f"{selected_host['alias']}: could not create snapshot parent: {parent.stderr.strip()}")
        mkdir = _ssh(selected_host["alias"], ["mkdir", "-m", "700", "--", source_dir])
        if mkdir.returncode:
            raise FleetError(f"{selected_host['alias']}: could not create unique source directory: {mkdir.stderr.strip()}")
        _send_snapshot(selected_host["alias"], snapshot, source_dir, digest)
        remote_run = _launch_script(
            data_root=selected_host["data_root"],
            job_id=job_id,
            source_dir=source_dir,
            run_dir=run_dir,
            remote_python=selected_host["python"],
            gpu_uuid=selected_gpu["uuid"],
            cpu_threads=args.cpu_threads,
            budget_hours=args.budget_hours,
            command=args.command,
            requirements=args.require_command,
            require_paths=args.require_path,
            snapshot_sha256=digest,
            git_metadata=git_meta,
            selected_gpu={
                **selected_gpu,
                "pointstream_python_version": selected_host.get("python_version"),
                "tool_versions": selected_host.get("tool_versions", {}),
            },
            estimated_gpu_memory_mib=args.gpu_memory_mib,
            idle_memory_mib=args.idle_memory_mib,
            idle_utilization_pct=args.idle_utilization_pct,
        )
        result = _ssh(selected_host["alias"], ["bash", "-lc", remote_run], timeout=90)
        if result.returncode:
            raise FleetError(f"{selected_host['alias']}: launch failed: {result.stderr.strip() or result.stdout.strip()}")
        response = json.loads(result.stdout.strip().splitlines()[-1])
        manifest.update(status="running", supervisor_pid=response.get("supervisor_pid"), launched_at=datetime.now(timezone.utc).isoformat())
        _write_manifest(local_manifest_path, manifest)
        return manifest
    except Exception as exc:
        manifest.update(status="launch_failed", error=str(exc))
        _write_manifest(local_manifest_path, manifest)
        raise
    finally:
        if snapshot is not None:
            snapshot.unlink(missing_ok=True)


def _path_check_script(paths: tuple[str, ...]) -> str:
    return "import json,os,sys; p=" + repr(list(paths)) + "; m=[x for x in p if not os.path.exists(x)]; print(json.dumps({'available':not m,'missing':m}))"


def _launch_script(
    *, data_root: str, job_id: str, source_dir: str, run_dir: str, remote_python: str,
    gpu_uuid: str, cpu_threads: int, budget_hours: float, command: list[str],
    requirements: list[str], require_paths: list[str], snapshot_sha256: str,
    git_metadata: dict[str, Any], selected_gpu: dict[str, Any], estimated_gpu_memory_mib: int,
    idle_memory_mib: int, idle_utilization_pct: int,
) -> str:
    for value in (job_id,):
        if not JOB_ID_RE.fullmatch(value):
            raise FleetError("invalid generated job ID")
    payload = {
        "job_id": job_id, "command": command, "gpu_uuid": gpu_uuid,
        "cpu_threads": cpu_threads, "budget_hours": budget_hours,
        "required_commands": requirements, "required_paths": require_paths,
        "snapshot_sha256": snapshot_sha256, "source_dir": source_dir,
        "data_root": data_root, "run_dir": run_dir,
        "git_metadata": git_metadata, "selected_gpu": selected_gpu,
        "estimated_gpu_memory_mib": estimated_gpu_memory_mib,
        "resource_policy": {"idle_memory_mib": idle_memory_mib, "idle_utilization_pct": idle_utilization_pct, "additional_free_memory_margin_mib": DEFAULT_MEMORY_MARGIN_MIB},
        "runtime_environment": {"pointstream_python": selected_gpu.get("pointstream_python_version"), "native_tools": selected_gpu.get("tool_versions", {})},
    }
    python = r'''import datetime,json,os,pathlib,shutil,subprocess,sys
spec=json.loads(sys.argv[1]); source=pathlib.Path(spec["source_dir"]); run=pathlib.Path(spec["run_dir"]); root=pathlib.Path(spec["data_root"])
if not root.is_dir() or not os.access(root,os.W_OK): raise SystemExit("external data root is unavailable or read-only")
missing=[p for p in spec["required_paths"] if not pathlib.Path(p).exists()]
if missing: raise SystemExit("required input paths unavailable: "+", ".join(missing))
missing=[n for n in spec["required_commands"] if not ((os.path.isfile(n) and os.access(n,os.X_OK)) if os.path.isabs(n) else shutil.which(n))]
if missing: raise SystemExit("required commands unavailable: "+", ".join(missing))
if not source.is_dir() or run.exists(): raise SystemExit("snapshot missing or run directory already exists")
if not pathlib.Path(spec["python"]).is_file(): raise SystemExit("PointStream Python environment disappeared")
for rel in spec["git_metadata"].get("deleted_paths", []):
 relpath=pathlib.PurePosixPath(rel)
 if relpath.is_absolute() or ".." in relpath.parts: raise SystemExit("unsafe deleted path in local snapshot")
 source.joinpath(*relpath.parts).unlink(missing_ok=True)
(source/"dispatch.json").write_text(json.dumps({**spec,"created_at":__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()},indent=2,sort_keys=True)+"\n")
cmd=[spec["python"],"-m","experiments.jobs.monitor","start",str(run),"--budget-hours",str(spec["budget_hours"]),"--claim-gpu",spec["gpu_uuid"],"--min-free-gpu-memory-mib",str(spec["estimated_gpu_memory_mib"]+spec["resource_policy"]["additional_free_memory_margin_mib"]),"--cpu-threads",str(spec["cpu_threads"]),"--claims-dir",str(root/"jobs"/"claims"),"--command",*spec["command"]]
env=os.environ.copy(); env["PS_DATA_ROOT"]=str(root); env["PYTHONPATH"]=str(source)+(os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
p=subprocess.run(cmd,cwd=source,env=env,capture_output=True,text=True,timeout=60)
if p.returncode: raise SystemExit(p.stderr or p.stdout or "PointStream monitor refused launch")
try: pid=int(p.stdout.strip().split("supervisor=")[1].split()[0])
except (ValueError,IndexError): pid=None
deadline=__import__("time").monotonic()+60
status_path=run/"status.json"
while __import__("time").monotonic()<deadline:
 try: state=json.loads(status_path.read_text())
 except (OSError,json.JSONDecodeError): state={}
 if state.get("pid"): break
 if state.get("status") not in (None,"running"): raise SystemExit("supervisor rejected launch before child start: "+json.dumps(state))
 __import__("time").sleep(.1)
if not state.get("pid"): raise SystemExit("supervisor did not confirm child launch within 60 seconds")
(run/"dispatch.json").write_text((source/"dispatch.json").read_text())
print(json.dumps({"supervisor_pid":pid,"monitor_output":p.stdout.strip()}))'''
    payload["python"] = remote_python
    return f"exec {shlex.quote(remote_python)} -c {shlex.quote(python)} {shlex.quote(json.dumps(payload))}"


def _load_manifest(job_id: str, state_dir: Path) -> tuple[Path, dict[str, Any]]:
    if not JOB_ID_RE.fullmatch(job_id):
        raise FleetError("invalid job ID")
    path = state_dir.expanduser() / f"{job_id}.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FleetError(f"no local fleet record for job {job_id}: {exc}") from exc
    if value.get("job_id") != job_id:
        raise FleetError("local fleet record does not match the requested job ID")
    if value.get("host_alias") not in DEFAULT_HOSTS:
        raise FleetError("local fleet record names an unconfigured server")
    expected = Path(value.get("data_root", "/invalid")) / "jobs" / "fleet" / "runs" / job_id
    if Path(value.get("run_dir", "/invalid")) != expected:
        raise FleetError("local fleet record contains an unexpected remote run path")
    return path, value


def status_job(job_id: str, state_dir: Path) -> dict[str, Any]:
    path, manifest = _load_manifest(job_id, state_dir)
    host = manifest["host_alias"]
    status_path = str(Path(manifest["run_dir"]) / "status.json")
    result = _ssh(host, ["cat", "--", status_path])
    if result.returncode:
        raise FleetError(f"{host}: status unavailable: {result.stderr.strip()}")
    try:
        value = json.loads(result.stdout)
    except ValueError as exc:
        raise FleetError(f"{host}: status file is malformed") from exc
    manifest.update(status=value.get("status", "unknown"), remote_status=value)
    _write_manifest(path, manifest)
    return manifest


def cancel_job(job_id: str, state_dir: Path) -> dict[str, Any]:
    path, manifest = _load_manifest(job_id, state_dir)
    status = status_job(job_id, state_dir)
    if status.get("status") not in {"running", "preparing", "transferring"}:
        raise FleetError(f"job {job_id} is already {status.get('status')}; no cancellation was sent")
    cancel_path = str(Path(manifest["run_dir"]) / "cancel.json")
    script = "import json,pathlib,sys,time,uuid; p=pathlib.Path(sys.argv[1]); t=p.with_suffix('.tmp'); t.write_text(json.dumps({'requested_at':time.time(),'request_id':uuid.uuid4().hex})); t.replace(p); print('cancellation_requested')"
    # Send the path as a shell-quoted argument; the host monitor owns process-group termination.
    result = _ssh(manifest["host_alias"], ["/usr/bin/python3", "-c", script, cancel_path])
    if result.returncode:
        raise FleetError(f"{manifest['host_alias']}: could not request cancellation: {result.stderr.strip()}")
    manifest.update(status="cancellation_requested")
    _write_manifest(path, manifest)
    return manifest


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be positive and finite") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def _percentage(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer from 0 to 100") from exc
    if not 0 <= number <= 100:
        raise argparse.ArgumentTypeError("must be an integer from 0 to 100")
    return number


def main(argv: list[str] | None = None) -> int:
    """Compatibility module entry point; all submissions use the smoke gate."""
    from experiments.jobs.inbox import main as inbox_main

    return inbox_main(argv, public_only=True)


if __name__ == "__main__":
    raise SystemExit(main())
