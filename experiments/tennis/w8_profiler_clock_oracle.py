"""CPU-only synthetic W8 Kineto isolation diagnosis; never a GPU qualification."""

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
    source_root = Path(__file__).resolve(strict=True).parents[2]
    if Path.cwd().resolve(strict=True) != source_root:
        raise ValueError("worker source root differs")
    origins = {"__main__": str(Path(__file__).resolve(strict=True).relative_to(source_root))}
    for name, module in sys.modules.items():
        if name == "experiments" or name.startswith("experiments."):
            filename = getattr(module, "__file__", None)
            if filename:
                origins[name] = str(Path(filename).resolve(strict=True).relative_to(source_root))
    if set(origins.values()) != set(SOURCE):
        raise ValueError("project source closure differs")
    output, canary = Path(c["output"]), Path(c["canary"])
    denied = [Path(p) for p in c["denied"]]
    ns = {n: os.readlink("/proc/self/ns/" + n) for n in ("user", "net")}
    if any(ns[n] == c["parent_ns"][n] for n in ns) or sorted(
        n for _, n in socket.if_nameindex()
    ) != ["lo"]:
        raise ValueError("namespace mismatch")
    if len(os.sched_getaffinity(0)) != 1 or len(os.listdir("/proc/self/task")) != 1:
        raise ValueError("must start with one thread and core")
    if any(
        int(fd) > 2 and Path("/proc/self/fd", fd).exists() for fd in os.listdir("/proc/self/fd")
    ):
        raise ValueError("unexpected inherited file descriptor")
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
        "source_origins": origins,
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
                    "sha256_before": next(
                        x["sha256"] for x in c["clock_identity"] if x["path"] == str(p)
                    ),
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
        os.write(2, f"W8_PROFILE_{label}_BEGIN\n".encode())
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with torch.profiler.profile(activities=activities):
                    torch.arange(8).sum().item()
            results[label] = {"success": True, "warnings": [str(w.message) for w in caught]}
        except Exception as exc:
            results[label] = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            os.write(2, f"W8_PROFILE_{label}_END\n".encode())
    receipt["results"] = results
    receipt["cuda_after"] = torch.cuda.is_initialized()
    receipt["cpu_seconds"] = sum(resource.getrusage(resource.RUSAGE_SELF)[:2])
    (output / "receipt.json").write_text(json.dumps(receipt, sort_keys=True))


def run() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or len(os.sched_getaffinity(0)) != 1:
        raise ValueError("requires GPU-hidden one-core fleet")
    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_FSIZE, (20_000_000, 20_000_000))
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
    pub = Path(os.environ["PS_SCRATCH_DIR"]) / "publish"
    pub.mkdir()
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
                    "sha256": None if p == CLOCK else sha(p),
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
            "trace=openat,openat2,newfstatat,readlink,readlinkat,getdents64,write",
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

        arm_env = {
            **env,
            **{
                key: str(output)
                for key in ("TMPDIR", "TMP", "TEMP", "XDG_CACHE_HOME", "TORCH_HOME")
            },
        }
        proc = subprocess.run(
            cmd,
            env=arm_env,
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
        if (
            trace.stat().st_size > 20_000_000
            or len(proc.stdout) > 1_000_000
            or len(proc.stderr) > 1_000_000
        ):
            raise ValueError("trace or stream exceeds publication cap")
        artifacts = {"trace": trace, "config": cfg, "receipt": output / "receipt.json"}
        for name, data in (("stdout", proc.stdout), ("stderr", proc.stderr)):
            path = root / f"{arm}.{name}"
            path.write_bytes(data)
            artifacts[name] = path
        artifact_hashes = {}
        for name, path in artifacts.items():
            destination = pub / f"{arm}.{name}"
            shutil.copyfile(path, destination)
            artifact_hashes[name] = sha(destination)
        receipt["clocksource_trace"] = trace_lines
        receipt["trace_sha256"] = sha(trace)
        receipt["artifacts"] = artifact_hashes
        rows.append(receipt)
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
        inputs = staging["inputs"]
        if (
            len(inputs) != 1
            or inputs[0]["name"] != "synthetic_reference"
            or inputs[0]["sha256"]
            != "5452a5d6bb865152556a0e26dc99798a0ee88dff1c1c86b9a4fd1f5c904413dd"
            or sha(Path(inputs[0]["path"])) != inputs[0]["sha256"]
        ):
            raise ValueError("synthetic input identity")
        if sha(stage / "published.tar") != staging["published"]["sha256"]:
            raise ValueError("publication digest")
        arms = ("unrestricted", "strict", "clock")
        kinds = ("trace", "config", "receipt", "stdout", "stderr")
        expected = {"publish/profiler-oracle.json"} | {
            f"publish/{arm}.{kind}" for arm in arms for kind in kinds
        }
        raw: dict[str, bytes] = {}
        with tarfile.open(stage / "published.tar") as t:
            members = t.getmembers()
            files = [m for m in members if m.isfile()]
            if len(files) != len(expected) or {m.name for m in files} != expected:
                raise ValueError("publication membership")
            if any(
                not m.isfile() and not (m.isdir() and m.name.rstrip("/") == "publish")
                for m in members
            ):
                raise ValueError("unexpected archive entry")
            if any(
                m.size > (20_000_000 if m.name.endswith(".trace") else 1_000_000) for m in files
            ):
                raise ValueError("publication size cap")
            for member in files:
                stream = t.extractfile(member)
                if stream is None:
                    raise ValueError("missing archive content")
                raw[member.name] = stream.read()
        r = json.loads(raw["publish/profiler-oracle.json"])
        if r["scope"] != "CPU only; no model or GPU qualification" or [
            x["arm"] for x in r["arms"]
        ] != ["unrestricted", "strict", "clock"]:
            raise ValueError("scope/arms mismatch")
        if (
            r["strace_sha256"] != sha(Path("/usr/bin/strace"))
            or not 0 <= r["child_cpu_seconds"] <= 60
        ):
            raise ValueError("tracer identity or CPU budget")
        for x in r["arms"]:
            arm = x["arm"]
            saved = {kind: raw[f"publish/{arm}.{kind}"] for kind in kinds}
            if x["artifacts"] != {
                kind: hashlib.sha256(data).hexdigest() for kind, data in saved.items()
            }:
                raise ValueError("raw artifact digest")
            receipt = json.loads(saved["receipt"])
            if receipt != {
                key: value
                for key, value in x.items()
                if key not in ("artifacts", "clocksource_trace", "trace_sha256")
            }:
                raise ValueError("worker receipt mismatch")
            config = json.loads(saved["config"])
            if config["parent_ns"] != x["parent_ns"]:
                raise ValueError("config namespace mismatch")
            filtered = [
                line
                for line in saved["trace"].decode(errors="replace").splitlines()
                if "/sys/devices/system/clocksource" in line
            ]
            markers = [
                f"W8_PROFILE_{label}_{phase}".encode()
                for label in ("cpu", "requested_cpu_cuda")
                for phase in ("BEGIN", "END")
            ]
            positions = [saved["trace"].find(marker) for marker in markers]
            if any(p < 0 for p in positions) or positions != sorted(positions):
                raise ValueError("profile trace markers")
            if filtered != x["clocksource_trace"] or x["trace_sha256"] != x["artifacts"]["trace"]:
                raise ValueError("trace filter mismatch")
            if set(x["source_origins"].values()) != set(SOURCE):
                raise ValueError("project source origins")
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
            for result in x["results"].values():
                if (
                    type(result.get("success")) is not bool
                    or (result["success"] and not isinstance(result.get("warnings"), list))
                    or (not result["success"] and not result.get("error"))
                ):
                    raise ValueError("invalid profiler outcome")
            if x["arm"] != "unrestricted":
                if x["landlock_abi"] < 3 or x["handled_mask"] != policy.handled_rights(
                    x["landlock_abi"]
                ):
                    raise ValueError("landlock")
                policy.validate_leaf_receipt(x["approved_leaves"], config["loader"])
                if (
                    set(x["denials"]) != set(config["denied"])
                    or len(x["denials"]) != 10
                    or any(v not in (1, 13) for v in x["denials"].values())
                ):
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
        config_clock = json.loads(raw["publish/clock.config"])["clock_identity"]
        if [x["path"] for x in config_clock] != [str(CLOCK), *(str(p) for p in LEAVES)]:
            raise ValueError("parent clock paths differ")
        for i, (parent, child) in enumerate(zip(config_clock, clock, strict=True)):
            if parent["uid"] != 0 or parent["mode"] & stat.S_IWOTH:
                raise ValueError("clock identity ownership")
            if not (stat.S_ISDIR(parent["mode"]) if i == 0 else stat.S_ISREG(parent["mode"])):
                raise ValueError("clock identity type")
            if (
                any(parent[key] != child[key] for key in ("path", "device", "inode", "mode"))
                or child["sha256_before"] != parent["sha256"]
            ):
                raise ValueError("clock identity differs")
            if (
                i == 0
                and parent["sha256"] is not None
                or i > 0
                and (not isinstance(parent["sha256"], str) or len(parent["sha256"]) != 64)
            ):
                raise ValueError("clock leaf SHA")
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
            "checks": {
                "source_identity": True,
                "publication_identity_and_raw_traces": True,
                "synthetic_input": True,
                "namespaces": True,
                "strict_denials_and_mutations": True,
                "cpu_profile_arms": True,
                "experimental_clock_grants": True,
            },
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
