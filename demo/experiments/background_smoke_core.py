"""Torch-free identities, metrics and provenance for the background smoke.

Everything here runs in the fleet worker interpreter (Python 3.10) and in
local tests. Model code lives in ``dcvc_uf_adapter.py`` (DCVC interpreter)
and ``hnerv_frozen.py`` (torch, imported lazily).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable
import zipfile

import numpy as np

from demo.experiments.background_smoke_io import _stat_identity, _stream_hash


DATA_ROOT = Path("/home/itec/emanuele/Datasets/pointstream-data")
DEMO_ROOT = Path("/home/itec/emanuele/Datasets/pointstream-demo")
WORK_ROOT = DATA_ROOT / "jobs" / "factory-bg-rd"
DCVC_ROOT = DATA_ROOT / "jobs" / "neural-bg" / "src" / "DCVC"
HNERV_ROOT = DATA_ROOT / "jobs" / "neural-bg" / "src" / "HNeRV"
DCVC_PYTHON = Path("/home/itec/emanuele/.conda/envs/pointstream-dcvc/bin/python")
HNERV_PYTHON = Path("/home/itec/emanuele/.conda/envs/pointstream/bin/python")
FFMPEG = "/opt/local/bin/ffmpeg"


def resolve_ffmpeg() -> str | None:
    """Prefer the lab binary, then PATH. Missing ffmpeg blocks only the AV1 arm."""
    override = os.environ.get("FFMPEG")
    for candidate in (override, FFMPEG):
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return shutil.which("ffmpeg")
PREVIEW_JOB = "20261003T101656Z-61d46aea"
PREVIEW_ROOT = DATA_ROOT / "jobs" / "fleet" / "runs" / PREVIEW_JOB

FPS = 30
WIDTH, HEIGHT = 1920, 1080
FRAME_SHAPE = (HEIGHT, WIDTH, 3)
PRIMARY_START = 120
SHORT_LENGTH = 8
DRIFT_LENGTH = 32
ALLOWED_LENGTHS = (SHORT_LENGTH, DRIFT_LENGTH)
QP = 21
MAX_READ_SECONDS = 90.0
CPU_PREPARATION_SECONDS = 600.0
CASE_SECONDS = 240.0
STRUCTURES = ("ld", "hts", "htl")
PSNR_IDENTICAL_DB = 99.0

# Cut name -> (factory, hold-out stem, expected hold-out frames).
CUTS = {
    "f001c3": ("factory001", "clip_03_factory001_worker001_00000_last10s", 300),
    "f002": ("factory002", "factory002_worker001_00000_last10s", 299),
}
# Factory001 trains on clips 1 and 3; clip 3 seconds 210/240/420 stay out.
TRAINING_SOURCES = {
    "factory001": (
        ("clip_01_factory001_worker001_00001", "f000000-f035129", ()),
        ("clip_03_factory001_worker001_00000", "f000000-f012629", (210, 240, 420)),
    ),
    "factory002": (
        ("factory002_worker001_00000", "f000000-f035129", ()),
    ),
}
UF_IMAGE = DCVC_ROOT / "checkpoints" / "cvpr2026_image.pth.tar"
UF_VIDEO = {s: DCVC_ROOT / "checkpoints" / f"cvpr2026_video_{s}.pth.tar" for s in STRUCTURES}

# Installed DCVC files the adapter mirrors, hashed at the public revision
# cbdae87a5445114cdc7f48816da63ea80bdeac40. A mismatch blocks codec runs.
DCVC_REVISION = "cbdae87a5445114cdc7f48816da63ea80bdeac40"
DCVC_REFERENCE_SHA256 = {
    "test_video.py": "6597af5cb9a86b6be1d8d3ed8534b88e45c9ef882cc2de14d11ed07dee892d7b",
    "src/models/image_model.py": "95ff7158ffa592b608f346f3585e6111380bee08e50096a1e7c0355acf01089e",
    "src/models/video_model_ld.py": "0a810363912e0fe298f84494693a9817c45e943e4f2d117365d59a85cf618fc5",
    "src/models/video_model_ht.py": "0ac7fe2f6c5e6520b0c452bf16105533ce80f02ac911d3975648be5b5f8b7d11",
    "src/models/common_model.py": "e309a7144514836f29453f8d1fe65df8f7ab72d17ed1efea08ac35ee9f67469a",
    "src/models/entropy_models.py": "578eda1f5941dfc55cf2d4d3a353ba60c8ce0e483d0a10151641ae96c25cfc2b",
    "src/utils/common.py": "1f1bcb84b4ef42be7e48967675b85875301a287c55970d6f8fb6bd17f1c1f009",
    "src/utils/stream_helper.py": "2d470e3692209be0301b5da48c33c9b7a372be865ccb458d4a5358c7bc55fac4",
    "src/utils/transforms.py": "85713c55f217f348d5824cf493e7df9cb8ce9402f27c9f34fd95a4b799aaab25",
}
HNERV_REFERENCE_SHA256 = {
    "model_all.py": "a79a8e5115e89965c0584a63c09b937a219d92ec8a3cb2ac69f5ce9809227cc1",
    "hnerv_utils.py": "2fcda4399c2b05ac0680d375612c17fc58a236b9530008b0ef6cf0ae852f1b62",
}


# ---------------------------------------------------------------- identities

def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def sha256_file(path: Path, *, timeout: float = MAX_READ_SECONDS) -> dict[str, Any]:
    """Complete SHA-256 receipt from a killable child, or an exception."""
    if not 0 < timeout <= MAX_READ_SECONDS:
        raise ValueError(f"timeout must be in (0, {MAX_READ_SECONDS:g}] seconds")
    command = [sys.executable, str(Path(__file__).with_name("background_smoke_io.py")), "hash-one", str(path)]
    started = time.monotonic()
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"hash timed out after {timeout:g}s: {path}") from exc
    if result.returncode != 0:
        raise OSError(f"hash child failed for {path}: {result.stderr[-500:]}")
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise OSError(f"hash child returned no complete receipt for {path}") from exc
    if not is_sha256(payload.get("sha256")) or not isinstance(payload.get("bytes"), int):
        raise OSError(f"hash child returned an incomplete receipt for {path}")
    return {
        "path": str(Path(path).absolute()), **payload, "complete": True,
        "seconds": round(time.monotonic() - started, 3), "timeout_seconds": timeout,
    }


def copy_verified(source: Path, destination: Path, expected_sha256: str, *, timeout: float = MAX_READ_SECONDS) -> dict[str, Any]:
    """One bounded read of ``source`` that both hashes and copies it.

    The copy is accepted only when the streamed bytes match the expected
    identity; the source is never moved or rewritten.
    """
    if not is_sha256(expected_sha256):
        raise ValueError("expected identity must be a complete lowercase SHA-256")
    if not 0 < timeout <= MAX_READ_SECONDS:
        raise ValueError(f"timeout must be in (0, {MAX_READ_SECONDS:g}] seconds")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite scratch copy: {destination}")
    command = [
        sys.executable, str(Path(__file__).with_name("background_smoke_io.py")), "copy-one",
        str(source), str(destination),
    ]
    started = time.monotonic()
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        destination.unlink(missing_ok=True)
        raise TimeoutError(f"verified copy timed out after {timeout:g}s: {source}") from exc
    if result.returncode != 0:
        destination.unlink(missing_ok=True)
        raise OSError(f"verified copy failed for {source}: {result.stderr[-500:]}")
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    if payload.get("sha256") != expected_sha256:
        destination.unlink(missing_ok=True)
        raise ValueError(f"checkpoint identity changed: {source} is {payload.get('sha256')}, expected {expected_sha256}")
    remaining = timeout - (time.monotonic() - started)
    if remaining <= 0:
        destination.unlink(missing_ok=True)
        raise TimeoutError(f"verified copy exhausted its allowance: {source}")
    copy = sha256_file(destination, timeout=remaining)
    if copy["sha256"] != expected_sha256:
        destination.unlink(missing_ok=True)
        raise ValueError(f"scratch copy differs from its verified source: {destination}")
    return {
        "source": str(source), "copy": str(destination), "sha256": expected_sha256,
        "bytes": payload["bytes"], "seconds": round(time.monotonic() - started, 3),
    }


def cached_identity(path: Path, cache: dict[str, Any]) -> dict[str, Any] | None:
    """Reuse a complete receipt only when every stat field is unchanged."""
    receipt = cache.get(str(Path(path).absolute()))
    if not receipt or receipt.get("complete") is not True:
        return None
    try:
        current = _stat_identity(path)
    except OSError:
        return None
    if any(receipt.get(key) != value for key, value in current.items()):
        return None
    return receipt


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def rgb_identity(pixels: np.ndarray) -> str:
    """Identity of decoded RGB samples, independent of the file container."""
    array = np.ascontiguousarray(pixels)
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("RGB identity requires a uint8 HxWx3 array")
    header = f"rgb8:{array.shape[0]}x{array.shape[1]}:".encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


# --------------------------------------------------------------- bounded IO

class StageClock:
    """Monotonic allowance shared by every operation inside one stage."""

    def __init__(self, seconds: float, *, reserve: float = 0.0) -> None:
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("stage allowance must be positive")
        self.started = time.monotonic()
        self.deadline = self.started + seconds - reserve

    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    def bounded(self, cap: float) -> float:
        value = min(cap, self.remaining())
        if value <= 0:
            raise TimeoutError("stage allowance exhausted")
        return value


def read_json_bounded(path: Path, *, max_bytes: int = 16 << 20) -> Any:
    with Path(path).open("rb") as stream:
        data = stream.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"JSON record exceeds {max_bytes} bytes: {path}")
    return json.loads(data)


def stage_dir() -> Path:
    value = os.environ.get("PS_STAGE_DIR")
    if not value:
        raise RuntimeError("PS_STAGE_DIR is required for new experiment outputs")
    return Path(value).resolve()


def require_stage_path(path: Path) -> Path:
    root = stage_dir()
    resolved = Path(path).resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"experiment output must be inside PS_STAGE_DIR: {resolved}")
    return resolved


def write_json_new(path: Path, payload: Any) -> Path:
    """Append-only evidence write; never replaces an existing record."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"refusing to overwrite partial evidence: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


class Ledger:
    """Wall-clock record of every operation after GPU admission."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.rows: list[dict[str, Any]] = []
        self.started = time.time()

    def record(self, operation: str, started: float, status: str, **detail: Any) -> None:
        self.rows.append({
            "operation": operation, "status": status,
            "seconds": round(time.time() - started, 3), "started": started, **detail,
        })
        self.flush()

    def flush(self) -> None:
        payload = {
            "started": self.started, "elapsed_seconds": round(time.time() - self.started, 3),
            "operations": self.rows,
        }
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
        temporary.replace(self.path)


# ------------------------------------------------------------------ metrics

def psnr_from_mse(mse: float) -> float:
    if not math.isfinite(mse) or mse < 0:
        raise ValueError("MSE must be finite and nonnegative")
    return PSNR_IDENTICAL_DB if mse == 0.0 else 10.0 * math.log10(255.0**2 / mse)


def require_rgb_frame(pixels: np.ndarray, *, expected_shape: tuple[int, int, int] | None = None, channel_order: str = "RGB") -> np.ndarray:
    """Accept only uint8 HxWx3 RGB; never convert by guessing conventions."""
    if channel_order != "RGB":
        raise ValueError(f"frames must be declared RGB, got {channel_order}")
    array = np.asarray(pixels)
    if array.dtype != np.uint8:
        raise ValueError(f"frames must be uint8 samples, got {array.dtype}")
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"frames must be HxWx3, got {array.shape}")
    if expected_shape is not None and array.shape != tuple(expected_shape):
        raise ValueError(f"frame shape {array.shape} differs from expected {expected_shape}")
    return array


def unit_float_to_uint8(values: np.ndarray, *, layout: str, value_range: str) -> np.ndarray:
    """One documented conversion from a declared float range to uint8 HxWx3.

    ``value_range`` is the model's declared output convention ("unit" for
    [0, 1]); values outside it are an error, not a cue to rescale.
    """
    array = np.asarray(values, dtype=np.float64)
    if layout == "CHW":
        if array.ndim != 3 or array.shape[0] != 3:
            raise ValueError(f"declared CHW but got {array.shape}")
        array = array.transpose(1, 2, 0)
    elif layout != "HWC" or array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"declared {layout} but got {array.shape}")
    if value_range != "unit":
        raise ValueError(f"unsupported declared output range: {value_range}")
    if not np.isfinite(array).all():
        raise ValueError("model output contains nonfinite values")
    low, high = float(array.min()), float(array.max())
    if low < -1e-6 or high > 1.0 + 1e-6:
        raise ValueError(f"declared unit-range output spans [{low}, {high}]")
    return np.round(np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8)


def lpips_input(pixels: np.ndarray) -> np.ndarray:
    """LPIPS expects CHW RGB in [-1, 1]; only uint8 RGB is accepted."""
    array = require_rgb_frame(pixels)
    return (array.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)


def sequence_metrics(
    reference: list[np.ndarray], reconstruction: list[np.ndarray], *,
    expected_shape: tuple[int, int, int] | None = FRAME_SHAPE,
    masks: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Per-frame RGB MSE/PSNR, both aggregates, and temporal error."""
    if not reference or len(reference) != len(reconstruction):
        raise ValueError("reference and reconstruction must be nonempty and aligned")
    if masks is not None and len(masks) != len(reference):
        raise ValueError("masks must align with frames")
    frame_mse: list[float] = []
    inside: list[float | None] = []
    outside: list[float | None] = []
    previous: tuple[np.ndarray, np.ndarray] | None = None
    temporal_sum, temporal_count = 0.0, 0
    for index, (ref8, rec8) in enumerate(zip(reference, reconstruction)):
        ref8 = require_rgb_frame(ref8, expected_shape=expected_shape)
        rec8 = require_rgb_frame(rec8, expected_shape=ref8.shape)
        ref, rec = ref8.astype(np.float64), rec8.astype(np.float64)
        squared = (ref - rec) ** 2
        frame_mse.append(float(squared.mean()))
        if masks is not None:
            mask = np.asarray(masks[index], dtype=bool)
            if mask.shape != ref.shape[:2]:
                raise ValueError("mask is not aligned with the frame")
            inside.append(float(squared[mask].mean()) if mask.any() else None)
            outside.append(float(squared[~mask].mean()) if (~mask).any() else None)
        if previous is not None:
            difference = np.abs((rec - previous[1]) - (ref - previous[0]))
            temporal_sum += float(difference.sum())
            temporal_count += difference.size
        previous = (ref, rec)
    if not all(math.isfinite(value) for value in frame_mse):
        raise ValueError("nonfinite frame MSE")
    psnr = [psnr_from_mse(value) for value in frame_mse]
    pooled = float(np.mean(frame_mse))
    result: dict[str, Any] = {
        "frame_count": len(reference),
        "frame_mse": frame_mse,
        "frame_psnr_db": psnr,
        "mean_frame_psnr_db": float(np.mean(psnr)),
        "pooled_mse_psnr_db": psnr_from_mse(pooled),
        "temporal_reconstruction_error": (temporal_sum / temporal_count / 255.0) if temporal_count else 0.0,
        "temporal_error_definition": "mean(abs((rec[t]-rec[t-1])-(ref[t]-ref[t-1])))/255, not motion-compensated flicker",
    }
    if masks is not None:
        def pooled_psnr(values: list[float | None]) -> float | None:
            present = [v for v in values if v is not None]
            return psnr_from_mse(float(np.mean(present))) if present else None
        result["mask_region"] = {
            "inside_mask_pooled_psnr_db": pooled_psnr(inside),
            "outside_mask_pooled_psnr_db": pooled_psnr(outside),
            "meaning": "fill reconstruction error inside/outside the hand-arm union, not distance to an empty room",
        }
    return result


def rate_kbps(stream_bytes: int, *, frames: int, fps: int = FPS) -> float:
    if isinstance(stream_bytes, bool) or not isinstance(stream_bytes, int) or stream_bytes < 0 or frames <= 0 or fps <= 0:
        raise ValueError("stream bytes, frame count and FPS must be valid")
    return 8.0 * stream_bytes * fps / (1000.0 * frames)


def stream_file_bytes(paths: list[Path]) -> int:
    """Packaged bytes from explicit stream files, headers included."""
    if not paths:
        raise ValueError("at least one actual stream file is required")
    names = [str(Path(path).absolute()) for path in paths]
    if len(set(names)) != len(names):
        raise ValueError("stream file appears more than once")
    total = 0
    for path in paths:
        if not Path(path).is_file():
            raise OSError(f"missing stream file: {path}")
        total += Path(path).stat().st_size
    return total


def validate_cut(records: list[dict[str, Any]], *, start: int, length: int, clock: StageClock | None = None) -> None:
    """Exact ordered identities and unchanged bytes for one selected cut."""
    if length not in ALLOWED_LENGTHS:
        raise ValueError(f"only 8- or 32-frame cuts are allowed, got {length}")
    if len(records) != length:
        raise ValueError(f"cut has {len(records)} records, expected {length}")
    for offset, record in enumerate(records):
        if record.get("index") != start + offset:
            raise ValueError(f"frame order/index mismatch at offset {offset}")
        deadline = time.monotonic() + (clock.bounded(MAX_READ_SECONDS) if clock else MAX_READ_SECONDS)
        if _stream_hash(Path(record["path"]), deadline=deadline)["sha256"] != record.get("sha256"):
            raise ValueError(f"frame content changed: {record['path']}")


def compare_frame_identities(expected: list[str], actual: list[str]) -> None:
    """Swapped, dropped, padded or substituted frames fail a comparison."""
    if len(expected) != len(actual):
        raise ValueError(f"frame count differs: expected {len(expected)}, got {len(actual)}")
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            raise ValueError(f"frame identity differs at display index {index}")


def summarize_drift(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Ordered 32-frame PSNR trend without extrapolation."""
    if len(rows) != DRIFT_LENGTH or [row.get("index") for row in rows] != list(range(PRIMARY_START, PRIMARY_START + DRIFT_LENGTH)):
        raise ValueError("drift report requires exactly the ordered 32 frames 120..151")
    values = [float(row["psnr_db"]) for row in rows]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("PSNR contains a nonfinite value")
    return {
        "frame_count": DRIFT_LENGTH,
        "mean_frame_psnr_db": float(np.mean(values)),
        "first_frame_psnr_db": values[0],
        "last_frame_psnr_db": values[-1],
        "first8_mean_psnr_db": float(np.mean(values[:8])),
        "last8_mean_psnr_db": float(np.mean(values[-8:])),
        "last8_minus_first8_db": float(np.mean(values[-8:]) - np.mean(values[:8])),
    }


# ------------------------------------------------- checkpoint/state metadata

class StorageRef:
    def __init__(self, dtype: str, key: str, location: str, numel: int) -> None:
        self.dtype, self.key, self.location, self.numel = dtype, key, location, numel


class TensorRef:
    """Shape/storage description of a serialized tensor; no data is read."""

    def __init__(self, storage: StorageRef, offset: int, size: tuple[int, ...], stride: tuple[int, ...]) -> None:
        self.storage, self.offset = storage, offset
        self.shape = tuple(int(dim) for dim in size)
        self.stride = tuple(int(dim) for dim in stride)

    @property
    def dtype(self) -> str:
        return self.storage.dtype


class _MetadataUnpickler(pickle.Unpickler):
    _STORAGE_DTYPES = {
        "FloatStorage": "float32", "HalfStorage": "float16", "DoubleStorage": "float64",
        "BFloat16Storage": "bfloat16", "LongStorage": "int64", "IntStorage": "int32",
        "ShortStorage": "int16", "CharStorage": "int8", "ByteStorage": "uint8", "BoolStorage": "bool",
    }

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if module == "torch" and name in self._STORAGE_DTYPES:
            return ("storage-type", self._STORAGE_DTYPES[name])
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            def rebuild(storage, offset, size, stride, *_rest):
                return TensorRef(storage, offset, size, stride)
            return rebuild
        if (module, name) == ("torch._utils", "_rebuild_parameter"):
            return lambda data, *_rest: data
        if (module, name) == ("torch", "device"):
            return lambda *args: ("device",) + tuple(args)
        raise pickle.UnpicklingError(f"unsupported global in checkpoint metadata: {module}.{name}")

    def persistent_load(self, pid: Any) -> Any:
        if not isinstance(pid, tuple) or not pid or pid[0] != "storage":
            raise pickle.UnpicklingError("unsupported persistent id")
        _tag, storage_type, key, location, numel = pid
        if not (isinstance(storage_type, tuple) and storage_type[0] == "storage-type"):
            raise pickle.UnpicklingError("unsupported storage type")
        return StorageRef(storage_type[1], str(key), str(location), int(numel))


def read_torch_zip_metadata(path: Path, *, max_pickle_bytes: int = 64 << 20) -> tuple[Any, dict[str, zipfile.ZipInfo], str]:
    """Object graph of a Torch ZIP checkpoint with tensors as references.

    Only ``data.pkl`` and the central directory are read; storages are not.
    """
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        pickles = [info for info in infos if info.filename.endswith("/data.pkl")]
        if len(pickles) != 1:
            raise ValueError(f"expected one data.pkl in {path}, found {len(pickles)}")
        prefix = pickles[0].filename[: -len("data.pkl")]
        if pickles[0].file_size > max_pickle_bytes:
            raise ValueError("checkpoint metadata pickle is unexpectedly large")
        payload = archive.read(pickles[0])
        storages = {
            info.filename[len(prefix) + len("data/"):]: info
            for info in infos if info.filename.startswith(prefix + "data/")
        }
    return _MetadataUnpickler(io.BytesIO(payload)).load(), storages, prefix


def checkpoint_state(obj: Any) -> dict[str, Any]:
    """The weights mapping DCVC's get_state_dict would load."""
    if not isinstance(obj, dict):
        raise ValueError("checkpoint root is not a mapping")
    if "state_dict" in obj:
        obj = obj["state_dict"]
    if "net" in obj:
        obj = obj["net"]
    if not isinstance(obj, dict) or not obj:
        raise ValueError("checkpoint contains no state mapping")
    return obj


def state_shapes(state: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for name, value in state.items():
        shape = getattr(value, "shape", None)
        if shape is None:
            raise ValueError(f"checkpoint entry {name!r} has no tensor shape")
        shapes[str(name)] = tuple(int(dim) for dim in shape)
    return shapes


def canonical_state_dict(state: dict[str, Any]) -> dict[str, Any]:
    """Remove only a leading DataParallel ``module.`` prefix."""
    result: dict[str, Any] = {}
    for key, value in state.items():
        canonical = key[len("module."):] if key.startswith("module.") else key
        if not canonical or canonical in result:
            raise ValueError(f"state-dict key collision after module-prefix removal: {canonical!r}")
        result[canonical] = value
    return result


def compare_state_dict_shapes(expected: dict[str, Any], saved: dict[str, Any]) -> dict[str, Any]:
    """Exact keys and shapes, as a strict load requires."""
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


def state_signature(state: dict[str, Any]) -> str:
    """Digest of canonical key names, shapes and dtypes."""
    rows = sorted(
        (key, list(getattr(value, "shape", ())), str(getattr(value, "dtype", "")))
        for key, value in canonical_state_dict(state).items()
    )
    return sha256_bytes(json.dumps(rows).encode())


def compare_runtime_config(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    required = ("structure", "normalization", "lambdas", "qp_mapping", "precision", "extension_path", "extension_version")
    missing = [key for key in required if key not in expected or key not in actual]
    if missing:
        raise ValueError(f"runtime configuration is incomplete: {', '.join(missing)}")
    differences = {key: {"expected": expected[key], "actual": actual[key]} for key in required if expected[key] != actual[key]}
    if differences:
        raise ValueError(json.dumps({"runtime_config_mismatch": differences}, sort_keys=True))


def state_dicts_equal(first: dict[str, Any], second: dict[str, Any]) -> bool:
    left, right = canonical_state_dict(first), canonical_state_dict(second)
    if set(left) != set(right):
        return False
    for key in left:
        a, b = left[key], right[key]
        a = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
        b = b.detach().cpu().numpy() if hasattr(b, "detach") else np.asarray(b)
        if a.shape != b.shape or a.dtype != b.dtype or not np.array_equal(a, b):
            return False
    return True


def stage_epochs(stage0: dict[str, Any], stage1: dict[str, Any]) -> dict[str, int]:
    """Zero-based epochs belong to their own stage; never sum them."""
    result = {}
    for name, status in (("stage0", stage0), ("stage1", stage1)):
        epoch = status.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError(f"{name} requires a nonnegative embedded epoch")
        result[name] = epoch
    return result


def serialized_state_equality(
    first: tuple[dict[str, Any], dict[str, zipfile.ZipInfo]],
    second: tuple[dict[str, Any], dict[str, zipfile.ZipInfo]],
) -> dict[str, Any]:
    """Compare two serialized state mappings by layout and storage CRC-32.

    This reads no tensor data. Equal CRC-32 over equal-size storages is a
    strong indicator, not a byte-exact proof; callers label it as such.
    """
    left, left_infos = canonical_state_dict(first[0]), first[1]
    right, right_infos = canonical_state_dict(second[0]), second[1]
    if set(left) != set(right):
        return {"layout_equal": False, "reason": "key sets differ"}
    mismatched = []
    for key in sorted(left):
        a, b = left[key], right[key]
        if not isinstance(a, TensorRef) or not isinstance(b, TensorRef):
            return {"layout_equal": False, "reason": f"{key} is not a tensor"}
        if (a.shape, a.stride, a.offset, a.dtype, a.storage.numel) != (b.shape, b.stride, b.offset, b.dtype, b.storage.numel):
            return {"layout_equal": False, "reason": f"{key} layout differs"}
        ia, ib = left_infos[a.storage.key], right_infos[b.storage.key]
        if (ia.file_size, ia.CRC) != (ib.file_size, ib.CRC):
            mismatched.append(key)
    return {
        "layout_equal": True, "keys": len(left), "crc_mismatched_keys": mismatched,
        "all_storage_crc32_equal": not mismatched,
        "method": "zip central-directory CRC-32 and size of each mapped storage; not byte-exact",
    }


# ----------------------------------------------- training/hold-out identity

def validate_disjoint_frames(training_ids: Iterable[str], holdout_ids: Iterable[str], *, excluded_ids: Iterable[str] = ()) -> None:
    train, held, excluded = set(training_ids), set(holdout_ids), set(excluded_ids)
    overlap = sorted(train & held)
    contaminated = sorted(train & excluded)
    if overlap or contaminated:
        raise ValueError(json.dumps({"train_holdout_overlap": overlap[:20], "excluded_in_training": contaminated[:20]}))


def validate_same_room_sequence(room_ids: list[str], second_ids: list[str]) -> None:
    if not room_ids or len(room_ids) != len(second_ids):
        raise ValueError("room and source-second identities must be nonempty and aligned")
    if len(set(room_ids)) != 1 or len(set(second_ids)) != 1:
        raise ValueError("temporal sequence crosses a room or independent sampled second")


def parse_training_sequence(name: str, factory: str) -> tuple[str, int]:
    """``<stem>_t<seconds:06d>`` from factory_bg_rd.factory_sequences."""
    stem, separator, second = name.rpartition("_t")
    if not separator or not second.isdigit() or len(second) != 6:
        raise ValueError(f"unrecognized training sequence name: {name}")
    stems = {source[0] for source in TRAINING_SOURCES[factory]}
    if stem not in stems:
        raise ValueError(f"training sequence {name} is not from {factory}")
    return stem, int(second)


def training_identity_report(description: dict[str, Any], factory: str, holdout_frames: dict[str, range]) -> dict[str, Any]:
    """Check one factory's DCVC training description against its policy.

    ``holdout_frames`` maps a source stem to its final hold-out frame range in
    that source's frame numbering.
    """
    sequences = description.get("seqs")
    frames = description.get("frames")
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("training description has no sequences")
    if frames != [f"{index:06d}.jpg" for index in range(FPS)]:
        raise ValueError("every training sequence must list exactly 000000..000029")
    other = {s[0] for f, sources in TRAINING_SOURCES.items() if f != factory for s in sources}
    counts: dict[str, int] = collections.Counter()
    seconds: dict[str, set[int]] = collections.defaultdict(set)
    training_ids: list[str] = []
    for row in sequences:
        if row.get("seq_length") != FPS or row.get("height") != HEIGHT or row.get("width") != WIDTH:
            raise ValueError(f"sequence {row.get('path')} is not a 30-frame 1080p second")
        stem, second = parse_training_sequence(row["path"], factory)
        if stem in other:
            raise ValueError(f"cross-room sequence {row['path']}")
        if second in seconds[stem]:
            raise ValueError(f"duplicate sampled second {row['path']}")
        seconds[stem].add(second)
        counts[stem] += 1
        training_ids.extend(f"{stem}/f{second * FPS + offset}" for offset in range(FPS))
    excluded = [
        f"{stem}/f{second * FPS + offset}"
        for stem, _span, drop in TRAINING_SOURCES[factory] for second in drop for offset in range(FPS)
    ]
    holdout_ids = [f"{stem}/f{index}" for stem, span in holdout_frames.items() for index in span]
    validate_disjoint_frames(training_ids, holdout_ids, excluded_ids=excluded)
    return {
        "factory": factory,
        "sequence_count": len(sequences),
        "frames_per_sequence": FPS,
        "training_frames": len(training_ids),
        "sequences_by_stem": dict(counts),
        "excluded_seconds_absent": {stem: list(drop) for stem, _span, drop in TRAINING_SOURCES[factory] if drop},
        "clip3_second0_present": (0 in seconds.get("clip_03_factory001_worker001_00000", set())) if factory == "factory001" else None,
        "holdout_overlap": 0,
        "each_sequence_is_one_room_and_one_second": True,
    }


# -------------------------------------------------------------- provenance

def job_provenance() -> dict[str, Any]:
    """Dispatcher identities for the running stage, read from the job tree."""
    directory = stage_dir().parent
    ready = read_json_bounded(directory / "ready.json") if (directory / "ready.json").is_file() else {}
    environment = read_json_bounded(directory / "environment.json") if (directory / "environment.json").is_file() else {}
    spec = read_json_bounded(directory / "spec.json") if (directory / "spec.json").is_file() else {}
    gpu = environment.get("gpu") or {}
    return {
        "job_id": directory.name,
        "stage": os.environ.get("PS_STAGE"),
        "code_revision": ready.get("git_head"),
        "snapshot_sha256": ready.get("snapshot_sha256"),
        "source_tree_sha256": ready.get("source_sha256"),
        "tracked_patch_sha256": ready.get("tracked_worktree_patch_sha256"),
        "included_untracked_sha256": ready.get("included_untracked_sha256"),
        "spec_sha256": ready.get("spec_sha256"),
        "inputs": spec.get("inputs"),
        "host": environment.get("host"),
        "gpu_uuid": gpu.get("uuid"),
        "gpu_name": gpu.get("name"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "worker_python": sys.executable,
        "citable": False,
        "label": "diagnostic smoke; not paper evidence",
    }


def require_score_provenance(record: dict[str, Any]) -> dict[str, Any]:
    """Every new score carries code, input and loaded-checkpoint identities."""
    provenance = record.get("provenance") or {}
    missing = [name for name in ("code_revision", "source_tree_sha256", "gpu_uuid") if not provenance.get(name)]
    if missing:
        raise ValueError(f"score lacks provenance fields: {', '.join(missing)}")
    if not isinstance(provenance["code_revision"], str) or len(provenance["code_revision"]) != 40:
        raise ValueError("code_revision must be an exact Git revision")
    checkpoints = record.get("checkpoints")
    if not isinstance(checkpoints, dict) or not checkpoints or not all(is_sha256(v.get("sha256")) for v in checkpoints.values()):
        raise ValueError("score must identify every loaded checkpoint by complete SHA-256")
    if not record.get("source_frames") or not all(is_sha256(f.get("sha256")) for f in record["source_frames"]):
        raise ValueError("score must identify every source frame")
    codec = record.get("codec_source")
    if codec is not None and not is_sha256(codec.get("tracked_diff_sha256", "")):
        raise ValueError("codec source diff identity is required")
    return provenance


# ------------------------------------------------------------- child entry

def _child(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="bounded child reads for background_smoke")
    commands = parser.add_subparsers(dest="action", required=True)
    one = commands.add_parser("hash-one")
    one.add_argument("path", type=Path)
    copy = commands.add_parser("copy-one")
    copy.add_argument("source", type=Path)
    copy.add_argument("destination", type=Path)
    args = parser.parse_args(argv)
    if args.action == "hash-one":
        print(json.dumps(_stream_hash(args.path)))
        return 0
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(_stream_hash(args.source, copy_to=args.destination)))
    return 0


if __name__ == "__main__":
    raise SystemExit(_child(sys.argv[1:]))
