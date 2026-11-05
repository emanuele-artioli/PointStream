"""CPU-only synthetic W8 Kineto isolation diagnosis; never a GPU qualification."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import sys
import tarfile
import warnings
from experiments.tennis import w8_cuda_system_leaves as policy
from experiments.tennis import w8_receiver_bootstrap as bootstrap
from experiments.tennis.w8_folder_oracle import parent_loader_identity

CLOCK = Path("/sys/devices/system/clocksource")
LEAVES = (CLOCK / "clocksource0/current_clocksource", CLOCK / "clocksource0/available_clocksource")
SOURCE = (
    "experiments/tennis/w8_profiler_clock_oracle.py",
    "experiments/tennis/w8_cuda_system_leaves.py",
    "experiments/tennis/w8_receiver_bootstrap.py",
    "experiments/tennis/w8_folder_oracle.py",
    "experiments/tennis/w8_public_folder_materialize.py",
    "experiments/tennis/k6b_generate.py",
    "experiments/__init__.py",
    "experiments/tennis/__init__.py",
)


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def worker(cfg: Path, expected: str, arm: str) -> None:
    if sha(cfg) != expected or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("config or device mismatch")
    c = json.loads(cfg.read_text())
    output, canary = Path(c["output"]), Path(c["canary"])
    denied = [Path(p) for p in c["denied"]]
    ns = {n: os.readlink("/proc/self/ns/" + n) for n in ("user", "net")}
    if any(ns[n] == c["parent_ns"][n] for n in ns) or sorted(
        n for _, n in socket.if_nameindex()
    ) != ["lo"]:
        raise ValueError("namespace mismatch")
    if len(os.sched_getaffinity(0)) != 1 or len(os.listdir("/proc/self/task")) != 1:
        raise ValueError("must start with one thread and core")
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.cuda.is_initialized():
        raise ValueError("unexpected CUDA init")
    before = torch.arange(4).sum().item()
    if len(os.listdir("/proc/self/task")) != 1:
        raise ValueError("threads before restriction")
    receipt = {
        "arm": arm,
        "cuda_visible_devices": "",
        "cuda_before": torch.cuda.is_initialized(),
        "namespaces": ns,
        "parent_ns": c["parent_ns"],
        "torch_version": str(torch.__version__),
    }
    if arm != "unrestricted":
        extra, leaf_receipt = policy.approved_leaves(denied, c["loader"])
        paths = [output, Path(sys.prefix), *(Path(p) for p in policy.SYSTEM_GRANTS), *extra]
        if arm == "clock":
            # Experimental only: directory enumeration recursively and two read-only leaf files.
            # READ_DIR is inheritable, so this is not a claim of exact nonrecursive access.
            if CLOCK.resolve(strict=True) != CLOCK or any(
                p.resolve(strict=True) != p for p in LEAVES
            ):
                raise ValueError("clock path redirected")
            for item in c["clock_identity"]:
                st = Path(item["path"]).stat()
                if (st.st_dev, st.st_ino, st.st_mode) != (
                    item["device"],
                    item["inode"],
                    item["mode"],
                ):
                    raise ValueError("clock identity changed")
            original = policy.allowed_rights

            def clock_rights(path: Path, output: Path, abi: int) -> int:
                return policy.READ_DIR if path == CLOCK else original(path, output, abi)

            policy.allowed_rights = clock_rights
            receipt["experimental_clock_grants"] = [
                {
                    "path": str(p),
                    "device": p.stat().st_dev,
                    "inode": p.stat().st_ino,
                    "mode": p.stat().st_mode,
                    "uid": p.stat().st_uid,
                    "rights": "READ_DIR recursive" if p == CLOCK else "READ_FILE",
                }
                for p in (CLOCK, *LEAVES)
            ]
            paths.extend((CLOCK, *LEAVES))
        sys.addaudithook(bootstrap._deny_processes)
        abi, mask = policy.restrict(paths, output)
        receipt.update(
            {
                "landlock_abi": abi,
                "handled_mask": mask,
                "approved_leaves": leaf_receipt,
                "denials": bootstrap._check_denials(denied),
                "mutations": policy.mutation_probes(canary, output),
            }
        )
    receipt["bootstrap_equal"] = before == torch.arange(4).sum().item()
    results = {}
    for label, activities in [
        ("cpu", [torch.profiler.ProfilerActivity.CPU]),
        (
            "requested_cpu_cuda",
            [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        ),
    ]:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with torch.profiler.profile(activities=activities):
                    torch.arange(8).sum().item()
            results[label] = {"success": True, "warnings": [str(w.message) for w in caught]}
        except Exception as exc:
            results[label] = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
    receipt["results"] = results
    receipt["cuda_after"] = torch.cuda.is_initialized()
    receipt["cpu_seconds"] = sum(resource.getrusage(resource.RUSAGE_SELF)[:2])
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True))


def run() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or len(os.sched_getaffinity(0)) != 1:
        raise ValueError("requires GPU-hidden one-core fleet")
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    root = Path(os.environ["PS_SCRATCH_DIR"]) / "profiler-oracle"
    root.mkdir()
    canary = root / "canary"
    canary.mkdir()
    protected = root / "protected"
    protected.mkdir()
    for n in ("write", "truncate", "remove", "rename", "link"):
        (canary / n).write_bytes(b"canary")
    denied: list[str] = []
    for i in range(5):
        p = protected / str(i)
        p.write_bytes(b"synthetic private canary")
        a = protected / f"alias-{i}"
        a.symlink_to(p)
        denied.extend((str(p), str(a)))
    env = {
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
    }
    rows = []
    for arm in ("unrestricted", "strict", "clock"):
        output = root / arm
        output.mkdir()
        cfg = root / f"{arm}.json"
        c = {
            "output": str(output),
            "canary": str(canary),
            "denied": denied,
            "loader": parent_loader_identity(Path("/etc/ld.so.cache")),
            "parent_ns": {n: os.readlink("/proc/self/ns/" + n) for n in ("user", "net")},
            "clock_identity": [
                {
                    "path": str(p),
                    "device": p.stat().st_dev,
                    "inode": p.stat().st_ino,
                    "mode": p.stat().st_mode,
                    "uid": p.stat().st_uid,
                }
                for p in (CLOCK, *LEAVES)
            ],
        }
        cfg.write_text(json.dumps(c, sort_keys=True))
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
            "trace=openat,openat2,newfstatat,readlink,readlinkat,getdents64",
            "/usr/bin/unshare",
            "--user",
            "--map-root-user",
            "--net",
            sys.executable,
            "-m",
            "experiments.tennis.w8_profiler_clock_oracle",
            "worker",
            "--config",
            str(cfg),
            "--sha",
            sha(cfg),
            "--arm",
            arm,
        ]

        proc = subprocess.run(
            cmd,
            env=env,
            cwd=env["PYTHONPATH"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=30,
            check=False,
        )
        if proc.returncode:
            (root / f"{arm}.stderr").write_bytes(proc.stderr)
            raise RuntimeError(f"{arm} failed: " + proc.stderr.decode(errors="replace")[-2000:])
        receipt = json.loads((output / "receipt.json").read_text())
        trace_lines = [
            line
            for line in trace.read_text(errors="replace").splitlines()
            if "/sys/devices/system/clocksource" in line
        ]
        receipt["clocksource_trace"] = trace_lines
        receipt["trace_sha256"] = sha(trace)
        rows.append(receipt)
    pub = Path(os.environ["PS_SCRATCH_DIR"]) / "publish"
    pub.mkdir()
    result = {
        "scope": "CPU only; no model or GPU qualification",
        "arms": rows,
        "code_sha256": {n: sha(Path(__file__).resolve().parents[2] / n) for n in SOURCE},
        "strace_sha256": sha(Path("/usr/bin/strace")),
        "child_cpu_seconds": sum(resource.getrusage(resource.RUSAGE_CHILDREN)[:2]),
        "parent_cpu_seconds": sum(resource.getrusage(resource.RUSAGE_SELF)[:2]),
    }
    (pub / "profiler-oracle.json").write_text(json.dumps(result, sort_keys=True))


def validate() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    stage = Path(os.environ["PS_STAGE_DIR"])
    out = Path(os.environ["PS_VALIDATION_PATH"])
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("GPU visible")
        staging = json.loads((stage / "staging.json").read_text())
        if sha(stage / "published.tar") != staging["published"]["sha256"]:
            raise ValueError("publication digest")
        with tarfile.open(stage / "published.tar") as t:
            files = [m for m in t.getmembers() if m.isfile()]
            if len(files) != 1 or files[0].name != "publish/profiler-oracle.json":
                raise ValueError("publication membership")
            stream = t.extractfile(files[0])
            assert stream is not None
            r = json.load(stream)
        if r["scope"] != "CPU only; no model or GPU qualification" or [
            x["arm"] for x in r["arms"]
        ] != ["unrestricted", "strict", "clock"]:
            raise ValueError("scope/arms mismatch")
        if not r["strace_sha256"] or not 0 <= r["child_cpu_seconds"] <= 60:
            raise ValueError("tracer identity or CPU budget")
        for x in r["arms"]:
            if (
                x["cuda_visible_devices"] != ""
                or x["cuda_before"]
                or x["cuda_after"]
                or not x["bootstrap_equal"]
            ):
                raise ValueError("GPU or bootstrap")
            if any(x["namespaces"][n] == x["parent_ns"][n] for n in ("user", "net")):
                raise ValueError("namespace")
            if not x["trace_sha256"] or not isinstance(x["clocksource_trace"], list):
                raise ValueError("trace absent")
            if set(x["results"]) != {"cpu", "requested_cpu_cuda"}:
                raise ValueError("missing profiles")
            if x["arm"] != "unrestricted":
                if x["landlock_abi"] < 3 or x["handled_mask"] != policy.handled_rights(
                    x["landlock_abi"]
                ):
                    raise ValueError("landlock")
                if len(x["denials"]) != 10 or any(v not in (1, 13) for v in x["denials"].values()):
                    raise ValueError("denials")
                if any(v not in (1, 13, 18) for v in x["mutations"].values()):
                    raise ValueError("mutation denial errno")
                if set(x["mutations"]) != {
                    "write",
                    "truncate",
                    "create",
                    "remove",
                    "mkdir",
                    "rename",
                    "link",
                }:
                    raise ValueError("mutations")
        clock = r["arms"][2]["experimental_clock_grants"]
        if [x["path"] for x in clock] != [str(CLOCK), *(str(p) for p in LEAVES)]:
            raise ValueError("experimental grants differ")
        if [x["rights"] for x in clock] != ["READ_DIR recursive", "READ_FILE", "READ_FILE"]:
            raise ValueError("experimental rights differ")
        dispatch = json.loads((stage / "dispatch.json").read_text())
        source = stage.parent / "source.tar"
        if sha(source) != dispatch["code"]["snapshot_sha256"] or set(r["code_sha256"]) != set(
            SOURCE
        ):
            raise ValueError("source identity")
        with tarfile.open(source) as t:
            for name, digest in r["code_sha256"].items():
                stream = t.extractfile(name)
                if stream is None or hashlib.sha256(stream.read()).hexdigest() != digest:
                    raise ValueError("source closure")
        result = {
            "passed": True,
            "scope": r["scope"],
            "arms_complete": True,
            "policy_denials": True,
            "source_verified": True,
        }
    except Exception as exc:
        out.write_text(
            json.dumps({"passed": False, "error": f"{type(exc).__name__}: {exc}"}) + "\n"
        )
        raise
    out.write_text(json.dumps(result) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["run", "worker", "validate", "marker"])
    p.add_argument("--config", type=Path)
    p.add_argument("--sha")
    p.add_argument("--arm")
    a = p.parse_args()
    if a.mode == "worker":
        worker(a.config, a.sha, a.arm)
    elif a.mode == "run":
        run()
    elif a.mode == "validate":
        validate()


if __name__ == "__main__":
    main()
