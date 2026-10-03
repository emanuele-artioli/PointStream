"""Bounded, artifact-only diagnostics for the background smoke plan.

This module deliberately has no training path. It validates identities and
scores already produced eight- or 32-frame artifacts; codec inference remains
an explicit, version-checked operation in the installed codec environment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import struct
import time
from typing import Any

import numpy as np


FPS = 30
PRIMARY_START = 120
SHORT_LENGTH = 8
DRIFT_LENGTH = 32
MAX_HASH_TIMEOUT_SECONDS = 90
MAX_CPU_PREPARATION_SECONDS = 600


def _hash_child(path: Path) -> list[str]:
    """Return a small isolated Python command for a bounded streaming hash."""
    script = r"""import hashlib,json,os,sys
p=sys.argv[1]
before=os.stat(p)
h=hashlib.sha256()
n=0
with open(p,'rb') as f:
    while True:
        block=f.read(1048576)
        if not block: break
        h.update(block)
        n+=len(block)
after=os.stat(p)
stable=(before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino)
print(json.dumps({'sha256':h.hexdigest(),'bytes':n,'stat_bytes':after.st_size,'stable':stable}))"""
    return [sys.executable, "-c", script, str(path)]


def sha256_file(path: Path, *, timeout: float = 90.0) -> dict[str, Any]:
    """Hash one file with a hard child-process timeout and stable-file check."""
    if timeout <= 0 or timeout > MAX_HASH_TIMEOUT_SECONDS:
        raise ValueError(f"timeout must be in (0, {MAX_HASH_TIMEOUT_SECONDS}] seconds")
    path = Path(path)
    try:
        result = subprocess.run(
            _hash_child(path), capture_output=True, text=True, timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"hash timed out for {path}") from exc
    if result.returncode != 0:
        raise OSError(f"hash worker failed for {path}: {result.stderr[-500:]}")
    try:
        payload = json.loads(result.stdout)
    except (ValueError, TypeError) as exc:
        raise OSError(f"hash worker returned an incomplete receipt for {path}") from exc
    if payload.get("stable") is not True:
        raise RuntimeError(f"file changed while hashing: {path}")
    if payload.get("bytes") != payload.get("stat_bytes") or len(payload.get("sha256", "")) != 64:
        raise OSError(f"hash worker returned an incomplete receipt for {path}")
    return {
        "path": str(path.absolute()), "bytes": payload["bytes"],
        "sha256": payload["sha256"], "complete": True,
    }


def read_json_bounded(path: Path, *, deadline: float | None = None) -> Any:
    """Read a small manifest in a killable child with the plan's time limit."""
    timeout = MAX_HASH_TIMEOUT_SECONDS
    if deadline is not None:
        timeout = min(timeout, deadline - time.monotonic())
    if timeout <= 0:
        raise TimeoutError("aggregate CPU preparation budget expired before JSON read")
    script = (
        "import sys; f=open(sys.argv[1],'rb'); data=f.read(16777217); "
        "sys.stdout.buffer.write(data) if len(data)<=16777216 else sys.exit(3)"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, str(path)], capture_output=True,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"JSON read timed out for {path}") from exc
    if result.returncode != 0:
        raise OSError(f"manifest read failed or exceeded 16 MiB: {path}")
    return json.loads(result.stdout)


def state_shapes(state: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    """Read tensor-like shapes without loading or coercing checkpoint weights."""
    shapes: dict[str, tuple[int, ...]] = {}
    for name, value in state.items():
        shape = getattr(value, "shape", None)
        if shape is None:
            raise ValueError(f"checkpoint entry {name!r} has no tensor shape")
        shapes[str(name)] = tuple(int(dim) for dim in shape)
    return shapes


def canonical_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    """Remove one documented DataParallel prefix and reject ambiguous keys."""
    result: dict[str, Any] = {}
    for key, value in state.items():
        canonical = key[len("module."):] if key.startswith("module.") else key
        if not canonical or canonical in result:
            raise ValueError(f"state-dict key collision after module-prefix removal: {canonical!r}")
        result[canonical] = value
    return result


def compare_state_dict_shapes(expected: dict[str, Any], saved: dict[str, Any]) -> dict[str, Any]:
    """Require exact state keys and shapes before a strict model load."""
    expected_shapes = state_shapes(expected)
    saved_shapes = state_shapes(canonical_state_dict(saved))
    missing = sorted(set(expected_shapes) - set(saved_shapes))
    extra = sorted(set(saved_shapes) - set(expected_shapes))
    mismatched = {
        name: {"expected": expected_shapes[name], "saved": saved_shapes[name]}
        for name in sorted(set(expected_shapes) & set(saved_shapes))
        if expected_shapes[name] != saved_shapes[name]
    }
    if missing or extra or mismatched:
        raise ValueError(json.dumps({"missing": missing, "extra": extra, "shape_mismatch": mismatched}))
    return {"keys": len(expected_shapes), "strict_compatible": True}


def compare_runtime_config(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    """Require the codec configuration fields that affect loaded weights."""
    required = (
        "structure", "normalization", "lambdas", "qp_mapping", "precision",
        "extension_path", "extension_version",
    )
    missing = [key for key in required if key not in expected or key not in actual]
    if missing:
        raise ValueError(f"runtime configuration is incomplete: {', '.join(missing)}")
    differences = {
        key: {"expected": expected[key], "actual": actual[key]}
        for key in required if expected[key] != actual[key]
    }
    if differences:
        raise ValueError(json.dumps({"runtime_config_mismatch": differences}, sort_keys=True))


def state_dicts_equal(first: dict[str, Any], second: dict[str, Any]) -> bool:
    """Compare corresponding status/checkpoint weights exactly on CPU."""
    left, right = canonical_state_dict(first), canonical_state_dict(second)
    if set(left) != set(right):
        return False
    for key in left:
        a, b = left[key], right[key]
        a_shape, b_shape = tuple(getattr(a, "shape", ())), tuple(getattr(b, "shape", ()))
        if a_shape != b_shape:
            return False
        if hasattr(a, "detach"):
            a = a.detach().cpu().numpy()
        else:
            a = np.asarray(a)
        if hasattr(b, "detach"):
            b = b.detach().cpu().numpy()
        else:
            b = np.asarray(b)
        if a.dtype != b.dtype or not np.array_equal(a, b):
            return False
    return True


def stage_epochs(stage0: dict[str, Any], stage1: dict[str, Any]) -> dict[str, int]:
    """Keep zero-based stage epochs separate; never infer a total epoch count."""
    result = {}
    for name, status in (("stage0", stage0), ("stage1", stage1)):
        epoch = status.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError(f"{name} requires a nonnegative embedded epoch")
        result[name] = epoch
    return result


def validate_disjoint_frames(
    training_ids: list[str], holdout_ids: list[str], *, excluded_ids: tuple[str, ...] = (),
) -> None:
    """Reject train/hold-out overlap and excluded identities in training."""
    train, held, excluded = set(training_ids), set(holdout_ids), set(excluded_ids)
    overlap = sorted(train & held)
    contaminated = sorted(train & excluded)
    if overlap or contaminated:
        raise ValueError(json.dumps({"train_holdout_overlap": overlap, "excluded_in_training": contaminated}))


def validate_same_room_sequence(room_ids: list[str], second_ids: list[str]) -> None:
    """Reject a prediction segment that crosses a room or sampled-second reset."""
    if not room_ids or len(room_ids) != len(second_ids):
        raise ValueError("room and source-second identities must be nonempty and aligned")
    if len(set(room_ids)) != 1 or len(set(second_ids)) != 1:
        raise ValueError("temporal sequence crosses a room or independent sampled second")


def read_rgb(path: Path, *, timeout: float = MAX_HASH_TIMEOUT_SECONDS) -> np.ndarray:
    if timeout <= 0 or timeout > MAX_HASH_TIMEOUT_SECONDS:
        raise ValueError(f"image read timeout must be in (0, {MAX_HASH_TIMEOUT_SECONDS}] seconds")
    script = (
        "import struct,sys,warnings; import numpy as np; from PIL import Image; "
        "Image.MAX_IMAGE_PIXELS=2073600; warnings.simplefilter('error',Image.DecompressionBombWarning); "
        "im=Image.open(sys.argv[1]).convert('RGB'); a=np.asarray(im,dtype=np.uint8); "
        "sys.stdout.buffer.write(struct.pack('>III',a.shape[0],a.shape[1],3)); "
        "sys.stdout.buffer.write(a.tobytes(order='C'))"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, str(path)], capture_output=True,
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"RGB read/decode timed out for {path}") from exc
    if result.returncode != 0 or len(result.stdout) < 12:
        raise OSError(f"RGB read/decode failed for {path}: {result.stderr[-500:].decode(errors='replace')}")
    height, width, channels = struct.unpack(">III", result.stdout[:12])
    if channels != 3 or len(result.stdout) != 12 + height * width * channels:
        raise OSError(f"RGB reader returned incomplete image bytes for {path}")
    return np.frombuffer(result.stdout, dtype=np.uint8, offset=12).reshape(height, width, channels).copy()


def rgb_sequence_metrics(
    reference: list[Path], reconstruction: list[Path], *, deadline: float | None = None,
    expected_shape: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    """Calculate framewise RGB PSNR and temporal reconstruction error."""
    if not reference or len(reference) != len(reconstruction):
        raise ValueError("reference and reconstruction frame lists must be nonempty and aligned")
    frame_mse: list[float] = []
    ref_arrays: list[np.ndarray] = []
    rec_arrays: list[np.ndarray] = []
    for ref_path, rec_path in zip(reference, reconstruction):
        timeout = MAX_HASH_TIMEOUT_SECONDS
        if deadline is not None:
            timeout = min(timeout, deadline - time.monotonic())
            if timeout <= 0:
                raise TimeoutError("aggregate CPU preparation budget expired during RGB scoring")
        ref = read_rgb(ref_path, timeout=timeout)
        if deadline is not None:
            timeout = min(MAX_HASH_TIMEOUT_SECONDS, deadline - time.monotonic())
            if timeout <= 0:
                raise TimeoutError("aggregate CPU preparation budget expired during RGB scoring")
        rec = read_rgb(rec_path, timeout=timeout)
        if ref.shape != rec.shape:
            raise ValueError(f"RGB dimensions differ: {ref_path} versus {rec_path}")
        if expected_shape is not None and ref.shape != expected_shape:
            raise ValueError(f"RGB dimensions do not match expected {expected_shape}: {ref_path}")
        ref_arrays.append(ref.astype(np.float64))
        rec_arrays.append(rec.astype(np.float64))
        frame_mse.append(float(np.mean((ref_arrays[-1] - rec_arrays[-1]) ** 2)))
    psnr = [99.0 if mse == 0.0 else 10.0 * math.log10(255.0**2 / mse) for mse in frame_mse]
    pooled = float(np.mean(frame_mse))
    temporal = 0.0
    if len(reference) > 1:
        temporal_terms = [
            np.abs((rec_arrays[i] - rec_arrays[i - 1]) - (ref_arrays[i] - ref_arrays[i - 1]))
            for i in range(1, len(reference))
        ]
        temporal = float(np.mean(np.stack(temporal_terms)) / 255.0)
    return {
        "frame_count": len(reference), "frame_mse": frame_mse,
        "mean_frame_psnr_db": float(np.mean(psnr)),
        "pooled_mse_psnr_db": 99.0 if pooled == 0.0 else 10.0 * math.log10(255.0**2 / pooled),
        "temporal_reconstruction_error": temporal,
    }


def validate_cut(
    records: list[dict[str, Any]], *, start: int, length: int,
    deadline: float | None = None,
) -> None:
    """Check exact ordered frame identities and content hashes for one cut."""
    if length not in (SHORT_LENGTH, DRIFT_LENGTH):
        raise ValueError(f"only {SHORT_LENGTH}- or {DRIFT_LENGTH}-frame cuts are allowed")
    if len(records) != length:
        raise ValueError(f"cut has {len(records)} records, expected {length}")
    for offset, record in enumerate(records):
        remaining = MAX_HASH_TIMEOUT_SECONDS
        if deadline is not None:
            remaining = min(remaining, deadline - time.monotonic())
            if remaining <= 0:
                raise TimeoutError("aggregate CPU preparation budget expired while validating frames")
        if record.get("index") != start + offset:
            raise ValueError(f"frame order/index mismatch at offset {offset}")
        identity = sha256_file(Path(record["path"]), timeout=remaining)
        if identity["sha256"] != record.get("sha256"):
            raise ValueError(f"frame content changed: {record['path']}")


def rate_kbps(stream_bytes: int, *, frames: int, fps: int = FPS) -> float:
    if stream_bytes < 0 or frames <= 0 or fps <= 0:
        raise ValueError("stream bytes, frame count, and FPS must be valid")
    return 8.0 * stream_bytes * fps / (1000.0 * frames)


def stream_file_bytes(paths: list[Path], *, deadline: float | None = None) -> int:
    """Count packaged bytes from explicit bitstream files, including headers."""
    if not paths:
        raise ValueError("at least one actual stream file is required")
    names = [str(path.absolute()) for path in paths]
    if len(set(names)) != len(names):
        raise ValueError("stream file appears more than once")
    timeout = MAX_HASH_TIMEOUT_SECONDS
    if deadline is not None:
        timeout = min(timeout, deadline - time.monotonic())
    if timeout <= 0:
        raise TimeoutError("aggregate CPU preparation budget expired before stream inventory")
    script = r"""import json,os,stat,sys
result=[]
for path in sys.argv[1:]:
    info=os.stat(path)
    if not stat.S_ISREG(info.st_mode): sys.exit(4)
    result.append(info.st_size)
print(json.dumps(result))"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", script, *names], capture_output=True,
            text=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError("stream file inventory timed out") from exc
    if result.returncode != 0:
        raise OSError(f"stream file inventory failed: {result.stderr[-500:]}")
    sizes = json.loads(result.stdout)
    if len(sizes) != len(paths) or any(not isinstance(size, int) or size < 0 for size in sizes):
        raise OSError("stream file inventory returned incomplete sizes")
    return sum(sizes)


def summarize_drift(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize an ordered 32-frame metric record without extrapolation."""
    if len(rows) != DRIFT_LENGTH or [row.get("index") for row in rows] != list(range(PRIMARY_START, PRIMARY_START + DRIFT_LENGTH)):
        raise ValueError("drift report requires exactly the ordered 32 frames 120..151")
    values = [float(row["psnr_db"]) for row in rows]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("PSNR contains a nonfinite value")
    return {
        "frame_count": DRIFT_LENGTH,
        "mean_frame_psnr_db": float(np.mean(values)),
        "first8_mean_psnr_db": float(np.mean(values[:8])),
        "last8_mean_psnr_db": float(np.mean(values[-8:])),
        "last8_minus_first8_db": float(np.mean(values[-8:]) - np.mean(values[:8])),
    }


def _write_json_new(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial")
    if temp.exists():
        raise FileExistsError(f"refusing to overwrite partial evidence: {temp}")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.link(temp, path)
        temp.unlink()
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def _require_job_path(path: Path) -> Path:
    job_root = os.environ.get("PS_JOB_DIR")
    if not job_root:
        raise RuntimeError("PS_JOB_DIR is required for new experiment outputs")
    resolved_root = Path(job_root).resolve()
    resolved_path = Path(path).resolve()
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise ValueError(f"experiment output must be inside PS_JOB_DIR: {resolved_path}")
    return resolved_path


def _score_provenance(manifest: dict[str, Any]) -> dict[str, Any]:
    required = (
        "code_revision", "source_tree_sha256", "patch_sha256",
        "checkpoint_sha256", "environment", "codec_revision", "gpu_uuid",
        "command", "encoder", "decoder", "peak_memory_mib",
    )
    missing = [name for name in required if not manifest.get(name)]
    if missing:
        raise ValueError(f"score manifest lacks provenance fields: {', '.join(missing)}")
    for name in ("code_revision", "codec_revision"):
        value = manifest[name]
        if not isinstance(value, str) or len(value) not in (40, 64) or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError(f"{name} must be an exact lowercase Git revision")
    for name in ("source_tree_sha256", "patch_sha256", "checkpoint_sha256"):
        value = manifest[name]
        if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError(f"{name} must be a complete lowercase SHA-256")
    if not isinstance(manifest["environment"], dict) or not manifest["environment"]:
        raise ValueError("environment must record the selected runtime versions")
    if not isinstance(manifest["command"], list) or not manifest["command"] or any(not isinstance(item, str) for item in manifest["command"]):
        raise ValueError("command must be the exact argv used for the codec run")
    if not isinstance(manifest["gpu_uuid"], str) or not manifest["gpu_uuid"]:
        raise ValueError("gpu_uuid must identify the assigned device")
    for name in ("encoder", "decoder"):
        value = manifest[name]
        if not isinstance(value, dict) or not value.get("path") or not value.get("version"):
            raise ValueError(f"{name} must record its native path and version")
    peak = manifest["peak_memory_mib"]
    if isinstance(peak, bool) or not isinstance(peak, (int, float)) or not math.isfinite(peak) or peak <= 0:
        raise ValueError("peak_memory_mib must be a positive measured value")
    return {name: manifest[name] for name in required}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    inventory = commands.add_parser("inventory", help="hash explicit files only")
    inventory.add_argument("--file", type=Path, action="append", required=True)
    inventory.add_argument("--output", type=Path, required=True)
    drift = commands.add_parser("drift", help="summarize exactly 32 indexed frame scores")
    drift.add_argument("--scores", type=Path, required=True)
    drift.add_argument("--output", type=Path, required=True)
    packet = commands.add_parser("latent", help="package an already extracted latent array")
    packet.add_argument("--codes", type=Path, required=True)
    packet.add_argument("--metadata", type=Path, required=True)
    packet.add_argument("--output-dir", type=Path, required=True)
    codec = commands.add_parser("codec", help="score an exact eight-frame decoded artifact manifest")
    codec.add_argument("--manifest", type=Path, required=True)
    codec.add_argument("--output", type=Path, required=True)
    summarize = commands.add_parser("summarize", help="combine completed diagnostic records")
    summarize.add_argument("--record", type=Path, action="append", required=True)
    summarize.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    cpu_deadline = time.monotonic() + MAX_CPU_PREPARATION_SECONDS
    if args.action == "inventory":
        receipts = []
        for path in args.file:
            remaining = min(MAX_HASH_TIMEOUT_SECONDS, cpu_deadline - time.monotonic())
            if remaining <= 0:
                raise SystemExit("aggregate CPU preparation budget reached during inventory")
            receipts.append(sha256_file(path, timeout=remaining))
        payload = {"files": receipts}
        _write_json_new(_require_job_path(args.output), payload)
        return 0
    if args.action == "drift":
        source = read_json_bounded(args.scores, deadline=cpu_deadline)
        provenance = _score_provenance(source)
        payload = {**provenance, **summarize_drift(source["frames"])}
        _write_json_new(_require_job_path(args.output), payload)
        return 0
    if args.action == "latent":
        from demo.experiments.hnerv_latent_packet import write_segment_packets

        metadata = read_json_bounded(args.metadata, deadline=cpu_deadline)
        codes = np.load(args.codes, allow_pickle=False)
        output_dir = _require_job_path(args.output_dir)
        payload = write_segment_packets(codes, metadata, output_dir)
        payload["provenance"] = _score_provenance(metadata["run_provenance"])
        _write_json_new(output_dir / "manifest.json", payload)
        print(json.dumps(payload, indent=2))
        return 0
    if args.action == "codec":
        manifest = read_json_bounded(args.manifest, deadline=cpu_deadline)
        if manifest.get("start") != PRIMARY_START or manifest.get("frame_count") != SHORT_LENGTH:
            raise SystemExit("codec scoring accepts only primary indices 120..127")
        provenance = _score_provenance(manifest)
        reference, reconstruction = manifest["reference"], manifest["reconstruction"]
        validate_cut(reference, start=PRIMARY_START, length=SHORT_LENGTH, deadline=cpu_deadline)
        validate_cut(reconstruction, start=PRIMARY_START, length=SHORT_LENGTH, deadline=cpu_deadline)
        if manifest.get("source_fps") != {"numerator": 30, "denominator": 1}:
            raise ValueError("codec source timebase must be the recorded 30/1 fps")
        actual_stream_bytes = stream_file_bytes(
            [Path(path) for path in manifest.get("stream_files", [])], deadline=cpu_deadline,
        )
        if isinstance(manifest.get("stream_bytes"), bool) or not isinstance(manifest.get("stream_bytes"), int) or manifest.get("stream_bytes") != actual_stream_bytes:
            raise ValueError("manifest stream_bytes differs from the actual packaged file sizes")
        setup_bytes = manifest.get("decoder_setup_bytes")
        if isinstance(setup_bytes, bool) or not isinstance(setup_bytes, int) or setup_bytes < 0:
            raise ValueError("decoder_setup_bytes must be a separate nonnegative integer")
        result = rgb_sequence_metrics(
            [Path(row["path"]) for row in reference],
            [Path(row["path"]) for row in reconstruction], deadline=cpu_deadline,
            expected_shape=(1080, 1920, 3),
        )
        result.update({
            **provenance,
            "codec": manifest["codec"],
            "structure": manifest["structure"],
            "qp": manifest["qp"],
            "stream_bytes": actual_stream_bytes,
            "rate_kbps": rate_kbps(actual_stream_bytes, frames=SHORT_LENGTH),
            "source_hashes": [row["sha256"] for row in reference],
            "reconstruction_hashes": [row["sha256"] for row in reconstruction],
            "decoder_setup_bytes": setup_bytes,
            "status": "inconclusive",
            "metrics_missing": ["LPIPS Alex", "aligned-mask regional errors when masks exist"],
        })
        _write_json_new(_require_job_path(args.output), result)
        return 0
    if args.action == "summarize":
        rows = [read_json_bounded(path, deadline=cpu_deadline) for path in args.record]
        _write_json_new(_require_job_path(args.output), {"records": rows, "record_count": len(rows)})
        return 0
    raise AssertionError(args.action)


if __name__ == "__main__":
    raise SystemExit(main())
