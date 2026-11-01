"""CPU-only qualification of packet-derived inputs against the original prepare.

The decode subprocess receives selected packet files and a declared tick recipe;
only the validator reads the prepared archive. The decoder requires Linux
Landlock and proves denial of the original and retained prepared archives,
the source video and a separate probe before reading packet inputs.
"""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import tempfile
import subprocess
import sys
import time
from typing import Any

# The fleet reserves 650 CPU-s for the 650-second campaign under enforced
# one-core affinity. Per-process bounds below are secondary safeguards.
if len(sys.argv) > 1 and sys.argv[1] in {"run", "worker", "validate"}:
    _cpu = {"run": 120, "worker": 60, "validate": 120}[sys.argv[1]]
    resource.setrlimit(resource.RLIMIT_CPU, (_cpu, 650))

# Set before importing NumPy/OpenCV, including in the fleet validator process.
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_env] = "1"

import numpy as np  # noqa: E402

from experiments.tennis.w8_inputs import archive_members, decode_inputs, validate_golden  # noqa: E402

WINDOW = "nadal_kyrgios/000@120"
UNITS = tuple(f"nadal_kyrgios__000_at120__person_{i:012d}" for i in range(2))
PREFIX = "publish/packets/nadal_kyrgios__000@120/"
RECIPE = "0:240:3"
PREPARED_SHA = "94a5d6318fc113557fb24c8a32f2edffdf368f7a289484fddd4b19a749302bfa"
PACKET_SHAS = {
    "keypoints.kps": "2e8452e4c458cf0998b6b8e004812575bdb2a638dd927b087dbc330d11d67070",
    f"{UNITS[0]}.png.ref": "63d214f5631fd5557d86a9c8c7c2936e2ef45afe539e882001d60e6115896559",
    f"{UNITS[1]}.png.ref": "c8a4d5355f06d72281751a0d936d5e28c1d80d189021b20000ccfc1dc62e7d6a",
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def judged_bitmap(recipe: str) -> bytes:
    if recipe != RECIPE:
        raise ValueError("Unexpected paid judged-frame recipe")
    ticks = np.zeros(240, dtype=np.uint8)
    ticks[0:240:3] = 1
    return np.packbits(ticks).tobytes()


def cpu_limit(seconds: int) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (seconds, 650))


def cpu_seconds() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def _verified_denials(paths: list[Path]) -> None:
    if len(paths) < 2 or len({str(path.resolve()) for path in paths}) != len(paths):
        raise ValueError("Distinct existing golden and probe files are required")
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Denied-read probe is not a file: {path}")
        with path.open("rb") as handle:
            if not handle.read(1):
                raise ValueError(f"Denied-read probe is empty: {path}")


def _allowed_read_paths(packet_dir: Path, output: Path, forbidden: list[Path]) -> list[Path]:
    """Narrow code/runtime trees; fail when any forbidden file is within one."""
    candidates = [Path(sys.prefix), Path(sys.base_prefix), Path(__file__).resolve().parents[2],
                  packet_dir, output, Path("/usr/lib"), Path("/lib"), Path("/lib64"),
                  Path("/etc/ld.so.cache"), Path("/dev/null"), Path("/dev/zero"), Path("/dev/urandom")]
    denied_roots = {Path("/"), Path("/usr"), Path("/home"), Path("/local"), Path("/tmp"), Path("/var"), Path("/dev")}
    allowed = list(dict.fromkeys(p.resolve() for p in candidates if p.exists()))
    if not Path(sys.prefix).resolve().is_dir() or not Path(__file__).resolve().parents[2].is_dir():
        raise ValueError("Runtime and source trees must exist")
    for path in allowed:
        if path in denied_roots:
            raise ValueError(f"Overbroad read allowance: {path}")
        for target in forbidden:
            if target.resolve() == path or target.resolve().is_relative_to(path):
                raise ValueError(f"Read allowance includes forbidden input: {target}")
    return allowed


def worker(packet_dir: Path, output: Path, recipe: str, forbidden: list[Path]) -> None:
    cpu_limit(60)
    started = time.monotonic()
    import cv2

    cv2.setNumThreads(1)
    output.mkdir(parents=True, exist_ok=False)
    allowed = _allowed_read_paths(packet_dir, output, forbidden)
    from experiments.tennis.k6b_receiver_sam import restrict_reads

    abi = restrict_reads(allowed)
    denials = {}
    for path in forbidden:
        try:
            with path.open("rb") as handle:
                handle.read(1)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EPERM):
                raise RuntimeError(f"Forbidden probe did not prove access denial: {path}") from exc
            denials[str(path)] = exc.errno
        else:
            raise RuntimeError(f"Worker read forbidden file: {path}")
    blob = (packet_dir / "keypoints.kps").read_bytes()
    references = {n: (packet_dir / f"{n}.png.ref").read_bytes() for n in UNITS}
    arrays, bindings = decode_inputs(blob, references, WINDOW, judged_bitmap(recipe))
    hashes = {}
    for name in UNITS:
        path = output / f"{name}.npz"
        np.savez_compressed(path, **arrays[name])
        hashes[name] = digest(path.read_bytes())
    write_json(output / "worker.json", {"bindings": bindings, "hashes": hashes, "numpy": np.__version__,
                                      "opencv": cv2.__version__, "cpu_s": cpu_seconds(),
                                      "wall_s": time.monotonic() - started, "recipe": recipe,
                                      "isolation": {"landlock_abi": abi, "allowed": [str(p) for p in allowed],
                                                    "denied": denials, "packet_read_sha256": digest(blob)}})


def run(transport: Path, prepared: Path, stage: Path, scratch: Path, recipe: str, source: Path) -> None:
    cpu_limit(120)
    started = time.monotonic()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only environment required")
    judged_bitmap(recipe)
    stage.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=False)
    packet_dir = stage / "packets"
    packet_dir.mkdir()
    requested = [PREFIX + "keypoints.kps", *(PREFIX + f"{n}.png.ref" for n in UNITS)]
    data = archive_members(transport, requested)
    for member, content in data.items():
        (packet_dir / Path(member).name).write_bytes(content)
    archived_golden = stage / "prepared.tar"
    shutil.copyfile(prepared, archived_golden)
    if digest(archived_golden.read_bytes()) != PREPARED_SHA:
        raise ValueError("Prepared archive identity mismatch")
    probe = stage / "forbidden-probe.bin"
    probe.write_bytes(b"landlock-negative-read-probe")
    forbidden = [prepared.resolve(), archived_golden.resolve(), probe.resolve(), source.resolve()]
    _verified_denials(forbidden)
    out = stage / "decoded"
    cmd = [sys.executable, "-m", "experiments.tennis.w8_input_job", "worker", "--packet-dir", str(packet_dir),
           "--out", str(out), "--recipe", recipe, *[v for path in forbidden for v in ("--deny", str(path))]]
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120, check=False)
        returncode, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        returncode = -1
        stdout = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or b"").decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        stderr += "\nWorker exceeded 120-second wall bound.\n"
    (stage / "worker.stdout").write_text(stdout)
    (stage / "worker.stderr").write_text(stderr)
    receipt = {"worker_returncode": returncode, "packet_dir": "packets", "prepared": "prepared.tar",
               "packets": {Path(k).name: digest(v) for k, v in data.items()}, "recipe": recipe,
               "parent_cpu_s": cpu_seconds(), "wall_s": time.monotonic() - started,
               "forbidden": [str(p) for p in forbidden], "access_scope": "Landlock required in worker; parent verifies probes before launch"}
    write_json(stage / "receipt.json", receipt)
    if returncode:
        raise ValueError("Packet decoder failed; logs and receipt retained")
    result = validate(stage)
    result["manifest"] = evidence_manifest(stage)
    write_json(stage / "full_validation.json", result)
    if not result["passed"]:
        raise ValueError("Strict in-stage validation failed; evidence retained")


def evidence_manifest(stage: Path) -> dict[str, str]:
    paths = ["receipt.json", "prepared.tar", "decoded/worker.json", "packets/keypoints.kps"]
    paths += [f"packets/{n}.png.ref" for n in UNITS]
    paths += [f"decoded/{n}.npz" for n in UNITS]
    result = {}
    for name in paths:
        path = stage / name
        if path.is_symlink() or not path.resolve().is_relative_to(stage.resolve()) or not path.is_file():
            raise ValueError(f"Invalid stage evidence path: {name}")
        result[name] = digest(path.read_bytes())
    return result


def validate(stage: Path) -> dict[str, Any]:
    cpu_limit(120)
    started = time.monotonic()
    import cv2

    cv2.setNumThreads(1)
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    try:
        receipt = json.loads((stage / "receipt.json").read_text())
        worker_result = json.loads((stage / "decoded" / "worker.json").read_text())
        checks["cpu_only"] = os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        checks["worker_success"] = receipt["worker_returncode"] == 0
        isolation = worker_result["isolation"]
        checks["landlock_active"] = (isinstance(isolation["landlock_abi"], int) and isolation["landlock_abi"] >= 1 and
                                     set(isolation["denied"]) == set(receipt["forbidden"]) and len(isolation["denied"]) == 4 and
                                     all(code in (errno.EACCES, errno.EPERM) for code in isolation["denied"].values()))
        details["isolation"] = isolation
        checks["declared_recipe"] = receipt["recipe"] == RECIPE == worker_result["recipe"]
        manifest = evidence_manifest(stage)
        checks["relative_paths"] = receipt["packet_dir"] == "packets" and receipt["prepared"] == "prepared.tar"
        packets = stage / "packets"
        checks["packet_hashes"] = receipt["packets"] == PACKET_SHAS and all(manifest["packets/" + n] == sha for n, sha in PACKET_SHAS.items())
        checks["prepared_hash"] = manifest["prepared.tar"] == PREPARED_SHA
        golden_bytes = archive_members(stage / "prepared.tar", [f"publish/units/{n}.npz" for n in UNITS])
        tmpdir = Path(os.environ["TMPDIR"])
        if not tmpdir.is_dir():
            raise ValueError("Host-local TMPDIR is unavailable")
        golden_dir = Path(tempfile.mkdtemp(prefix="w8-golden-validation-", dir=tmpdir))
        details["golden_directory"] = str(golden_dir)
        golden = {}
        decoded = {}
        for name in UNITS:
            golden[name] = golden_dir / f"{name}.npz"
            golden[name].write_bytes(golden_bytes[f"publish/units/{name}.npz"])
            path = stage / "decoded" / f"{name}.npz"
            checks[name + ":output_hash"] = digest(path.read_bytes()) == worker_result["hashes"][name]
            with np.load(path, allow_pickle=False) as data:
                decoded[name] = {k: data[k] for k in data.files}
            checks[name + ":fields"] = set(decoded[name]) == {"reference", "reference_box", "t", "run", "box", "pose", "judged"}
            checks[name + ":coverage"] = (len(decoded[name]["t"]) == 240 and int(decoded[name]["judged"].sum()) == 80)
        try:
            validate_golden(decoded, golden, (packets / "keypoints.kps").read_bytes())
            checks["exact_golden_and_native_equivalence"] = True
        except (ValueError, KeyError) as exc:
            checks["exact_golden_and_native_equivalence"] = False
            details["mismatch"] = str(exc)
        details["golden_hashes"] = {name: digest(golden_bytes[f"publish/units/{name}.npz"]) for name in UNITS}
        details["versions"] = {"numpy": np.__version__, "opencv": cv2.__version__, "worker_numpy": worker_result["numpy"], "worker_opencv": worker_result["opencv"]}
        checks["same_environment"] = details["versions"]["numpy"] == worker_result["numpy"] and details["versions"]["opencv"] == worker_result["opencv"]
        details["worker_cpu_s"] = worker_result["cpu_s"]
        details["parent_cpu_s"] = receipt["parent_cpu_s"]
        checks["process_cpu_limits"] = (0 <= details["parent_cpu_s"] <= 120 and
                                        0 <= details["worker_cpu_s"] <= 60 and cpu_seconds() <= 120)
    except Exception as exc:
        checks["validator_completed"] = False
        details["error"] = f"{type(exc).__name__}: {exc}"
    return {"passed": bool(checks) and all(checks.values()), "checks": checks, "details": details,
            "cpu_seconds": cpu_seconds(), "wall_seconds": time.monotonic() - started, "gpu_hours": 0}


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--transport", type=Path, required=True)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--recipe", required=True)
    p.add_argument("--windows", type=int, required=True, choices=[1])
    p.add_argument("--deny-source", type=Path, required=True)
    p = sub.add_parser("worker")
    p.add_argument("--packet-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--deny", type=Path, action="append", required=True)
    p.add_argument("--recipe", required=True)
    sub.add_parser("validate")
    args = parser.parse_args()
    if args.command == "worker":
        worker(args.packet_dir, args.out, args.recipe, args.deny)
    elif args.command == "run":
        run(args.transport, args.prepared, Path(os.environ["PS_STAGE_DIR"]), Path(os.environ["PS_SCRATCH_DIR"]) / ("w8-input-packets-" + os.environ["PS_STAGE"]), args.recipe, args.deny_source)
    else:
        stage = Path(os.environ["PS_STAGE_DIR"])
        result = validate(stage)
        try:
            sealed = json.loads((stage / "full_validation.json").read_text())
            result["checks"]["sealed_pass"] = sealed.get("passed") is True
            result["checks"]["sealed_manifest"] = sealed.get("manifest") == evidence_manifest(stage)
        except Exception as exc:
            result["checks"]["sealed_manifest"] = False
            result["details"]["seal_error"] = f"{type(exc).__name__}: {exc}"
        result["passed"] = all(result["checks"].values())
        write_json(Path(os.environ["PS_VALIDATION_PATH"]), result)
        print(json.dumps(result))
        return int(not result["passed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
