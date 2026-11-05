"""Bounded CUDA profiler diagnostic using synthetic tensors and opt-in isolation.

This diagnostic is not a receiver policy, model qualification, or dispatcher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import time
import traceback
from typing import Any

from experiments.tennis import w8_cuda_system_leaves as policy
from experiments.tennis import w8_profiler_clock_oracle as base
from experiments.tennis import w8_receiver_bootstrap as bootstrap

SOURCE = (*base.SOURCE, "experiments/tennis/w8_cuda_profiler_synthetic.py")
SCOPE = "synthetic CUDA profiler diagnostic; no model, receiver or quality qualification"
ARMS = ("strict", "unrestricted", "clock")
KINDS = ("trace", "config", "receipt", "stdout", "stderr")
TRACE_CAP = 20_000_000
OTHER_CAP = 1_000_000
RESERVE_SECONDS = 25.0
SMOKE_SECONDS = 150.0


def source_origins() -> dict[str, str]:
    root = Path(__file__).resolve(strict=True).parents[2]
    if Path.cwd().resolve(strict=True) != root:
        raise ValueError("project source root differs")
    origins = {"__main__": str(Path(__file__).resolve(strict=True).relative_to(root))}
    for name, mod in sys.modules.items():
        if name == "experiments" or name.startswith("experiments."):
            filename = getattr(mod, "__file__", None)
            if filename:
                origins[name] = str(Path(filename).resolve(strict=True).relative_to(root))
    if set(origins.values()) != set(SOURCE):
        raise ValueError("loaded project source closure differs")
    return origins


def worker(config: Path, expected_sha: str, arm: str) -> None:
    if base.sha(config) != expected_sha or arm not in ARMS:
        raise ValueError("invalid config or arm")
    cfg = json.loads(config.read_text())
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or policy.canonical_gpu_uuid(visible) != policy.canonical_gpu_uuid(
        cfg["gpu_uuid"]
    ):
        raise ValueError("visible device does not match claim")
    origins = source_origins()
    output, canary = Path(cfg["output"]), Path(cfg["canary"])
    denied = [Path(p) for p in cfg["denied"]]
    namespaces = {n: os.readlink("/proc/self/ns/" + n) for n in ("user", "net")}
    if any(namespaces[n] == cfg["parent_ns"][n] for n in namespaces):
        raise ValueError("namespace mismatch")
    if sorted(n for _, n in socket.if_nameindex()) != ["lo"]:
        raise ValueError("network isolation mismatch")
    if len(os.sched_getaffinity(0)) != 1 or len(os.listdir("/proc/self/task")) != 1:
        raise ValueError("worker must start on one core and thread")
    if any(
        int(fd) > 2 and Path("/proc/self/fd", fd).exists() for fd in os.listdir("/proc/self/fd")
    ):
        raise ValueError("unexpected inherited descriptor")
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():
        raise ValueError("CUDA initialized before isolation")
    before = torch.arange(4).sum().item()
    if len(os.listdir("/proc/self/task")) != 1:
        raise ValueError("threads created before isolation")
    receipt: dict[str, Any] = {
        "arm": arm,
        "scope": SCOPE,
        "source_origins": origins,
        "namespaces": namespaces,
        "parent_ns": cfg["parent_ns"],
        "visible_uuid": visible,
        "cuda_initialized_before": False,
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda),
        "provider": "torch CUDA",
    }
    if arm != "unrestricted":
        extra, leaves = policy.approved_leaves(denied, cfg["loader"])
        paths = [output, Path(sys.prefix), *(Path(p) for p in policy.SYSTEM_GRANTS), *extra]
        if arm == "clock":
            for item in cfg["clock_identity"]:
                path = Path(item["path"])
                current = path.stat()
                if path.resolve(strict=True) != path or (
                    current.st_dev,
                    current.st_ino,
                    current.st_mode,
                ) != (item["device"], item["inode"], item["mode"]):
                    raise ValueError("clock identity differs")
            original = policy.allowed_rights

            def rights(path: Path, output: Path, abi: int) -> int:
                return policy.READ_DIR if path == base.CLOCK else original(path, output, abi)

            policy.allowed_rights = rights
            paths.extend((base.CLOCK, *base.LEAVES))
            receipt["experimental_clock_grants"] = [
                dict(
                    item,
                    rights="READ_DIR recursive" if i == 0 else "READ_FILE",
                    child_uid=Path(item["path"]).stat().st_uid,
                )
                for i, item in enumerate(cfg["clock_identity"])
            ]
        sys.addaudithook(bootstrap._deny_processes)
        abi, mask = policy.restrict(paths, output)
        receipt.update(
            {
                "landlock_abi": abi,
                "handled_mask": mask,
                "approved_leaves": leaves,
                "denials": bootstrap._check_denials(denied),
                "mutations": policy.mutation_probes(canary, output),
            }
        )
    receipt["cpu_bootstrap_equal"] = before == torch.arange(4).sum().item()
    # Keep failures after isolation, including CUDA setup failures, as diagnostics.
    receipt["cuda"] = {"success": False}
    receipt["profiler"] = {"attempted": False, "success": False, "cuda_kernels": []}
    try:
        count = torch.cuda.device_count()
        if count != 1:
            raise RuntimeError(f"expected one CUDA device, got {count}")
        props = torch.cuda.get_device_properties(0)
        uuid = str(getattr(props, "uuid", ""))
        if policy.canonical_gpu_uuid(uuid) != policy.canonical_gpu_uuid(visible):
            raise ValueError("physical CUDA UUID differs from claim")
        if props.major != 8 or props.minor != 9 or "RTX 6000 Ada" not in props.name:
            raise ValueError("actual GPU is not the admitted Ada class")
        x = torch.ones((64, 64), dtype=torch.float32, device="cuda:0")
        y = x @ x
        torch.cuda.synchronize()
        value = float(y.sum().item())
        if not (x.is_cuda and y.is_cuda and value == 262144.0):
            raise ValueError("synthetic CUDA matmul failed correctness or fell back")
        receipt["cuda"] = {
            "success": True,
            "device_count": count,
            "uuid": uuid,
            "name": props.name,
            "capability": [props.major, props.minor],
            "sum": value,
            "inputs_cuda": bool(x.is_cuda),
            "output_cuda": bool(y.is_cuda),
            "cuda_initialized": bool(torch.cuda.is_initialized()),
        }
        os.write(2, b"W8_CUDA_PROFILE_BEGIN\n")
        try:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                z = x @ x
                torch.cuda.synchronize()
            events = [
                str(e.name)
                for e in prof.events()
                if str(getattr(e, "device_type", "")).endswith("CUDA") and e.name
            ]
            kernels = [
                name
                for name in events
                if "memcpy" not in name.lower() and "memset" not in name.lower()
            ]
            if not z.is_cuda or not kernels:
                raise ValueError("profiler succeeded without CUDA output and kernel events")
            receipt["profiler"] = {
                "attempted": True,
                "success": True,
                "cuda_events": events,
                "cuda_kernels": kernels,
            }
        except Exception as exc:
            receipt["profiler"] = {
                "attempted": True,
                "success": False,
                "cuda_kernels": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            os.write(2, b"W8_CUDA_PROFILE_END\n")
    except Exception as exc:
        receipt["cuda"] = {
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    receipt["cuda_initialized_after"] = bool(torch.cuda.is_initialized())
    receipt["cpu_seconds"] = sum(resource.getrusage(resource.RUSAGE_SELF)[:2])
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True))


def run() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    claimed = os.environ.get("PS_CLAIMED_GPU_UUID", "")
    if not visible or policy.canonical_gpu_uuid(visible) != policy.canonical_gpu_uuid(claimed):
        raise ValueError("missing exclusive GPU claim")
    if len(os.sched_getaffinity(0)) != 1:
        raise ValueError("one CPU affinity required")
    stage = Path(os.environ["PS_STAGE_DIR"])
    dispatch = json.loads((stage / "dispatch.json").read_text())
    gpu = dispatch["environment"]["gpu"]
    if policy.canonical_gpu_uuid(gpu["uuid"]) != policy.canonical_gpu_uuid(claimed):
        raise ValueError("dispatch device differs from claim")
    clock = dispatch["campaign_clock"]
    if not clock["started"] <= clock["phase_started"] < clock["budget_end"]:
        raise ValueError("invalid fleet campaign clock")
    resource.setrlimit(resource.RLIMIT_CPU, (50, 50))
    resource.setrlimit(resource.RLIMIT_FSIZE, (TRACE_CAP, TRACE_CAP))
    root = Path(os.environ["PS_SCRATCH_DIR"]) / "cuda-profiler-oracle"
    root.mkdir()
    canary, protected = root / "canary", root / "protected"
    canary.mkdir()
    protected.mkdir()
    for n in ("write", "truncate", "remove", "rename", "link"):
        (canary / n).write_bytes(b"canary")
    denied: list[str] = []
    for i in range(5):
        p = protected / str(i)
        p.write_bytes(b"synthetic private canary")
        alias = protected / f"alias-{i}"
        alias.symlink_to(p)
        denied.extend((str(p), str(alias)))
    env = {
        "CUDA_VISIBLE_DEVICES": visible,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
    }
    pub = Path(os.environ["PS_SCRATCH_DIR"]) / "publish"
    pub.mkdir()
    rows: list[dict[str, Any]] = []
    projection: dict[str, Any] = {"controls_admitted": False}
    for arm in ARMS:
        if arm != "strict" and not projection["controls_admitted"]:
            break
        if arm == "clock":
            now = time.monotonic()
            remaining = min(clock["budget_end"], clock["phase_started"] + SMOKE_SECONDS) - now
            observed = rows[-1]["ended"] - rows[-1]["started"]
            required = (
                max(projection["measured_first_stage_seconds"], observed) * 1.5 + RESERVE_SECONDS
            )
            projection["clock_gate"] = {
                "evaluated_at": now,
                "remaining_seconds": remaining,
                "required_seconds": required,
                "admitted": required <= remaining,
            }
            if not projection["clock_gate"]["admitted"]:
                break
        output = root / arm
        output.mkdir()
        config = root / f"{arm}.json"
        cfg = {
            "output": str(output),
            "canary": str(canary),
            "denied": denied,
            "gpu_uuid": claimed,
            "loader": base.parent_loader_identity(Path("/etc/ld.so.cache")),
            "parent_ns": {n: os.readlink("/proc/self/ns/" + n) for n in ("user", "net")},
            "clock_identity": [
                {
                    "path": str(p),
                    "device": p.stat().st_dev,
                    "inode": p.stat().st_ino,
                    "mode": p.stat().st_mode,
                    "uid": p.stat().st_uid,
                    "sha256": None if p == base.CLOCK else base.sha(p),
                }
                for p in (base.CLOCK, *base.LEAVES)
            ],
        }
        config.write_text(json.dumps(cfg, sort_keys=True))
        trace = root / f"{arm}.trace"
        cmd = [
            "/usr/bin/strace",
            "-f",
            "-yy",
            "-s",
            "256",
            "-o",
            str(trace),
            "-e",
            "trace=openat,openat2,newfstatat,readlink,readlinkat,getdents64,write",
            "/usr/bin/unshare",
            "--user",
            "--map-root-user",
            "--net",
            sys.executable,
            "-m",
            "experiments.tennis.w8_cuda_profiler_synthetic",
            "worker",
            "--config",
            str(config),
            "--sha",
            base.sha(config),
            "--arm",
            arm,
        ]
        arm_env = {
            **env,
            **{k: str(output) for k in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "TORCH_HOME")},
        }
        arm_started = time.monotonic()
        time_left = (
            min(clock["budget_end"], clock["phase_started"] + SMOKE_SECONDS)
            - arm_started
            - RESERVE_SECONDS
        )
        if time_left <= 0:
            raise RuntimeError("no time for strict arm and preservation")
        proc = subprocess.run(
            cmd,
            cwd=env["PYTHONPATH"],
            env=arm_env,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=min(60.0, time_left),
            check=False,
        )
        arm_finished = time.monotonic()
        if proc.returncode:
            (root / f"{arm}.stderr").write_bytes(proc.stderr)
            raise RuntimeError(
                f"{arm} worker failed: " + proc.stderr.decode(errors="replace")[-2000:]
            )
        receipt = json.loads((output / "receipt.json").read_text())
        if (
            trace.stat().st_size > TRACE_CAP
            or len(proc.stdout) > OTHER_CAP
            or len(proc.stderr) > OTHER_CAP
        ):
            raise ValueError("raw evidence exceeds cap")
        artifacts = {"trace": trace, "config": config, "receipt": output / "receipt.json"}
        for kind, data in (("stdout", proc.stdout), ("stderr", proc.stderr)):
            path = root / f"{arm}.{kind}"
            path.write_bytes(data)
            artifacts[kind] = path
        digests = {}
        for kind, path in artifacts.items():
            dst = pub / f"{arm}.{kind}"
            shutil.copyfile(path, dst)
            digests[kind] = base.sha(dst)
        rows.append(
            {
                "arm": arm,
                "started": arm_started,
                "ended": arm_finished,
                "artifacts": digests,
                "receipt": receipt,
            }
        )
        if arm == "strict":
            now = time.monotonic()
            measured = now - clock["started"]
            remaining = min(clock["budget_end"], clock["phase_started"] + SMOKE_SECONDS) - now
            required = measured * 1.5 * 2 + RESERVE_SECONDS
            projection = {
                "controls_admitted": bool(required <= remaining),
                "measured_first_stage_seconds": measured,
                "remaining_seconds": remaining,
                "required_seconds": required,
                "reserve_seconds": RESERVE_SECONDS,
                "multiplier": 1.5,
                "evaluated_at": now,
            }
    result = {
        "scope": SCOPE,
        "arms": rows,
        "projection": projection,
        "gpu_uuid": claimed,
        "dispatch_gpu": gpu,
        "campaign_clock": clock,
        "code_sha256": {n: base.sha(Path(__file__).resolve().parents[2] / n) for n in SOURCE},
        "strace_sha256": base.sha(Path("/usr/bin/strace")),
        "parent_cpu_seconds": sum(resource.getrusage(resource.RUSAGE_SELF)[:2]),
        "child_cpu_seconds": sum(resource.getrusage(resource.RUSAGE_CHILDREN)[:2]),
    }
    (pub / "cuda-profiler-oracle.json").write_text(json.dumps(result, sort_keys=True))


def validate() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (15, 15))
    stage = Path(os.environ["PS_STAGE_DIR"])
    output = Path(os.environ["PS_VALIDATION_PATH"])
    try:
        dispatch = json.loads((stage / "dispatch.json").read_text())
        staging = json.loads((stage / "staging.json").read_text())
        spec = json.loads((stage.parent / "spec.json").read_text())
        ready = json.loads((stage.parent / "ready.json").read_text())
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        claimed = os.environ.get("PS_CLAIMED_GPU_UUID", "")
        if (
            not visible
            or not claimed
            or policy.canonical_gpu_uuid(visible) != policy.canonical_gpu_uuid(claimed)
            or policy.canonical_gpu_uuid(visible)
            != policy.canonical_gpu_uuid(dispatch["environment"]["gpu"]["uuid"])
        ):
            raise ValueError("validator device claim differs")
        if (
            spec["environment"]["sha256"]
            != "bf738ce596552875e81d067de933ba1ae610c970dba915570efdb2784add8a7e"
        ):
            raise ValueError("environment identity differs")
        inputs = staging["inputs"]
        if (
            len(inputs) != 1
            or inputs[0]["name"] != "synthetic_reference"
            or inputs[0]["sha256"]
            != "5452a5d6bb865152556a0e26dc99798a0ee88dff1c1c86b9a4fd1f5c904413dd"
            or base.sha(Path(inputs[0]["path"])) != inputs[0]["sha256"]
        ):
            raise ValueError("synthetic reference identity differs")
        source = stage.parent / "source.tar"
        if (
            base.sha(source) != ready["snapshot_sha256"]
            or base.sha(source) != dispatch["code"]["snapshot_sha256"]
        ):
            raise ValueError("source archive mismatch")
        published = stage / "published.tar"
        if base.sha(published) != staging["published"]["sha256"]:
            raise ValueError("published archive mismatch")
        with tarfile.open(published) as tar:
            members = tar.getmembers()
            files = [m for m in members if m.isfile()]
            if any(
                not m.isfile() and not (m.isdir() and m.name.rstrip("/") == "publish")
                for m in members
            ):
                raise ValueError("unexpected publication entry")
            if any(m.size > (TRACE_CAP if m.name.endswith(".trace") else OTHER_CAP) for m in files):
                raise ValueError("publication exceeds cap")
            raw = {}
            for member in files:
                stream = tar.extractfile(member)
                if stream is None or member.name in raw:
                    raise ValueError("missing or duplicate archive entry")
                raw[member.name] = stream.read()
        result = json.loads(raw["publish/cuda-profiler-oracle.json"])
        if result["scope"] != SCOPE or result["strace_sha256"] != base.sha(Path("/usr/bin/strace")):
            raise ValueError("scope or tracer identity differs")
        arms = [row["arm"] for row in result["arms"]]
        if arms not in ([ARMS[0]], list(ARMS[:2]), list(ARMS)):
            raise ValueError("expected strict-only or all three arms")
        expected = {"publish/cuda-profiler-oracle.json"} | {
            f"publish/{arm}.{kind}" for arm in arms for kind in KINDS
        }
        if set(raw) != expected:
            raise ValueError("publication membership differs")
        if (
            result["campaign_clock"] != dispatch["campaign_clock"]
            or result["dispatch_gpu"] != dispatch["environment"]["gpu"]
        ):
            raise ValueError("dispatch identity differs")
        if policy.canonical_gpu_uuid(result["gpu_uuid"]) != policy.canonical_gpu_uuid(visible):
            raise ValueError("published device differs")
        if set(result["code_sha256"]) != set(SOURCE):
            raise ValueError("source closure differs")
        with tarfile.open(source) as tar:
            for name, digest in result["code_sha256"].items():
                stream = tar.extractfile(name)
                if stream is None or hashlib.sha256(stream.read()).hexdigest() != digest:
                    raise ValueError("source closure hash mismatch")
        clock = result["campaign_clock"]
        projection = result["projection"]
        measured = projection["evaluated_at"] - clock["started"]
        remaining = (
            min(clock["budget_end"], clock["phase_started"] + SMOKE_SECONDS)
            - projection["evaluated_at"]
        )
        if not 0 < measured or abs(projection["measured_first_stage_seconds"] - measured) > 1e-5:
            raise ValueError("first stage cost differs")
        if (
            abs(projection["remaining_seconds"] - remaining) > 1e-5
            or abs(projection["required_seconds"] - (measured * 3 + RESERVE_SECONDS)) > 1e-5
        ):
            raise ValueError("control projection differs")
        if projection["controls_admitted"] != (
            projection["required_seconds"] <= remaining
        ) or projection["controls_admitted"] != (len(arms) >= 2):
            raise ValueError("control promotion differs")
        if projection["multiplier"] != 1.5 or projection["reserve_seconds"] != RESERVE_SECONDS:
            raise ValueError("control margin differs")
        if len(arms) >= 2:
            second = projection["clock_gate"]
            expected_remaining = (
                min(clock["budget_end"], clock["phase_started"] + SMOKE_SECONDS)
                - second["evaluated_at"]
            )
            observed = result["arms"][1]["ended"] - result["arms"][1]["started"]
            expected_required = max(measured, observed) * 1.5 + RESERVE_SECONDS
            if (
                abs(second["remaining_seconds"] - expected_remaining) > 1e-5
                or abs(second["required_seconds"] - expected_required) > 1e-5
                or second["admitted"] != (expected_required <= expected_remaining)
                or second["admitted"] != (len(arms) == 3)
            ):
                raise ValueError("second control budget differs")
        for row in result["arms"]:
            arm = row["arm"]
            data = {k: raw[f"publish/{arm}.{k}"] for k in KINDS}
            if row["artifacts"] != {k: hashlib.sha256(v).hexdigest() for k, v in data.items()}:
                raise ValueError("raw artifact hash differs")
            receipt = json.loads(data["receipt"])
            config = json.loads(data["config"])
            if receipt != row["receipt"] or receipt["scope"] != SCOPE or receipt["arm"] != arm:
                raise ValueError("receipt differs")
            if set(receipt["source_origins"].values()) != set(SOURCE):
                raise ValueError("source origins differ")
            if policy.canonical_gpu_uuid(config["gpu_uuid"]) != policy.canonical_gpu_uuid(
                visible
            ) or policy.canonical_gpu_uuid(receipt["visible_uuid"]) != policy.canonical_gpu_uuid(
                visible
            ):
                raise ValueError("worker device claim differs")
            if (
                receipt["provider"] != "torch CUDA"
                or receipt["torch_version"] != "2.7.0+cu128"
                or receipt["torch_cuda_version"] != "12.8"
            ):
                raise ValueError("CUDA provider or pinned runtime differs")
            if (
                receipt["cuda_initialized_before"] is not False
                or receipt["cpu_bootstrap_equal"] is not True
            ):
                raise ValueError("pre-isolation bootstrap differs")
            if any(
                receipt["namespaces"][n] == receipt["parent_ns"][n]
                or receipt["parent_ns"][n] != config["parent_ns"][n]
                for n in ("user", "net")
            ):
                raise ValueError("namespace identity differs")
            if arm != "unrestricted":
                if receipt["landlock_abi"] < 3 or receipt["handled_mask"] != policy.handled_rights(
                    receipt["landlock_abi"]
                ):
                    raise ValueError("Landlock differs")
                policy.validate_leaf_receipt(receipt["approved_leaves"], config["loader"])
                if (
                    set(receipt["denials"]) != set(config["denied"])
                    or len(receipt["denials"]) != 10
                    or any(v not in (1, 13) for v in receipt["denials"].values())
                ):
                    raise ValueError("private read denial differs")
                if set(receipt["mutations"]) != {
                    "write",
                    "truncate",
                    "create",
                    "remove",
                    "mkdir",
                    "rename",
                    "link",
                } or any(v not in (1, 13, 18) for v in receipt["mutations"].values()):
                    raise ValueError("mutation denial differs")
            if arm == "clock":
                grants = receipt["experimental_clock_grants"]
                expected_paths = [str(base.CLOCK), *(str(p) for p in base.LEAVES)]
                if [g["path"] for g in grants] != expected_paths or [
                    {key: value for key, value in grant.items() if key != "child_uid"}
                    for grant in grants
                ] != [
                    dict(item, rights="READ_DIR recursive" if i == 0 else "READ_FILE")
                    for i, item in enumerate(config["clock_identity"])
                ]:
                    raise ValueError("experimental grants differ")
                for i, item in enumerate(grants):
                    if (
                        item["uid"] != 0
                        or item["child_uid"] != 65534
                        or item["mode"] & stat.S_IWOTH
                        or not (
                            stat.S_ISDIR(item["mode"]) if i == 0 else stat.S_ISREG(item["mode"])
                        )
                    ):
                        raise ValueError("experimental grant parent identity differs")
                    if i and (not isinstance(item["sha256"], str) or len(item["sha256"]) != 64):
                        raise ValueError("experimental leaf digest missing")
            cuda = receipt["cuda"]
            profile = receipt["profiler"]
            if cuda["success"]:
                if (
                    cuda["device_count"] != 1
                    or policy.canonical_gpu_uuid(cuda["uuid"]) != policy.canonical_gpu_uuid(visible)
                    or cuda["capability"] != [8, 9]
                    or "RTX 6000 Ada" not in cuda["name"]
                    or cuda["sum"] != 262144.0
                    or not cuda["inputs_cuda"]
                    or not cuda["output_cuda"]
                    or not cuda["cuda_initialized"]
                    or receipt["cuda_initialized_after"] is not True
                ):
                    raise ValueError("physical CUDA proof differs")
            elif not cuda.get("error"):
                raise ValueError("CUDA failure missing diagnostic")
            if profile["success"]:
                if (
                    not cuda["success"]
                    or not profile["attempted"]
                    or not profile["cuda_kernels"]
                    or any(not isinstance(x, str) or not x for x in profile["cuda_kernels"])
                    or profile["cuda_kernels"]
                    != [
                        name
                        for name in profile["cuda_events"]
                        if "memcpy" not in name.lower() and "memset" not in name.lower()
                    ]
                ):
                    raise ValueError("actual kernel events missing")
            elif profile["attempted"] and not profile.get("error"):
                raise ValueError("profiler failure missing diagnostic")
            trace = data["trace"]
            if profile["attempted"] and not (
                trace.count(b"W8_CUDA_PROFILE_BEGIN") == trace.count(b"W8_CUDA_PROFILE_END") == 1
                and trace.find(b"W8_CUDA_PROFILE_BEGIN") < trace.find(b"W8_CUDA_PROFILE_END")
            ):
                raise ValueError("profiler trace markers missing")
        receipt = {
            "passed": True,
            "scope": SCOPE,
            "checks": {
                "source_and_synthetic_identity": True,
                "publication_and_raw_traces": True,
                "gpu_claim_and_reported_device": True,
                "namespace_and_isolation_receipts": True,
                "cuda_outcomes_and_kernels": True,
                "measured_control_budget": True,
            },
            "arms": arms,
            "controls_admitted": projection["controls_admitted"],
            "cuda_proof_arms": [
                r["arm"] for r in result["arms"] if r["receipt"]["cuda"]["success"]
            ],
        }
    except Exception as exc:
        output.write_text(
            json.dumps(
                {
                    "passed": False,
                    "checks": {"oracle_validation": False},
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            + "\n"
        )
        raise
    output.write_text(json.dumps(receipt) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "worker", "validate", "marker"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--sha")
    parser.add_argument("--arm")
    args = parser.parse_args()
    if args.mode == "worker":
        if args.config is None or args.sha is None or args.arm is None:
            parser.error("worker requires config, sha and arm")
        worker(args.config, args.sha, args.arm)
    elif args.mode == "run":
        run()
    elif args.mode == "validate":
        validate()


if __name__ == "__main__":
    main()
