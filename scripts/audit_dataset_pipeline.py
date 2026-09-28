"""Create a SAM3.1 quality dataset or run the bounded codec audit.

All inputs and outputs resolve under the external PointStream data root. The
script writes a diagnostic versioned dataset, never changes an active training
manifest, and decodes the serialized PointStream client payload in a fresh
process that receives no source-frame argument. The existing-dataset build
mode is sharded and writes a new, inactive training tree plus an audit of every
retained and rejected legacy-track observation.
"""

from __future__ import annotations

import argparse
from collections.abc import Collection, Mapping
from collections import Counter, defaultdict
import fcntl
import gc
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, cast

from src.components.segmentation.sam31 import Policy, Role
from src.shared.dataset_quality import (
    QualityDecision,
    filter_player_candidates,
    filter_racket_candidates,
)

import numpy as np

try:
    import cv2
except ImportError:  # SAM worker uses PIL; the parent audit uses OpenCV.
    cv2 = cast(Any, None)

ROLES: tuple[Role, Role] = ("player", "racket")
OFFLINE_POLICY: Policy = "offline_bidirectional"

PILOT_SCHEMA = "pointstream.sam31-pilot.v1"
OBSERVATION_SCHEMA = "pointstream.observation.v1"
QUALITY_DATASET_SCHEMA = "pointstream.sam31-quality-dataset.v1"
EXISTING_DATASET_SCHEMA = "pointstream.sam31-existing-training-set.v1"
QUALITY_CHUNK_SIZE = 48


def validate_pilot_manifest(
    pilot: dict[str, Any],
    scene_manifest: dict[str, Any],
    *,
    reserved_source_ids: Collection[str] = (),
) -> list[dict[str, Any]]:
    """Validate a frozen three-scene development sample and its frame ranges."""
    if pilot.get("schema") != PILOT_SCHEMA:
        raise ValueError(f"unsupported pilot schema {pilot.get('schema')!r}")
    if pilot.get("split") != "development_exposed_bp46":
        raise ValueError("SAM3.1 pilot must use explicitly exposed BP46 development data")
    if pilot.get("reserved_sources_excluded") is not True:
        raise ValueError("pilot must explicitly exclude reserved confirmation sources")
    frame_count = int(pilot.get("frame_count_per_scene", 0))
    if frame_count != 16 or int(pilot.get("working_fps", 0)) != 24:
        raise ValueError("pilot requires exactly 16 consecutive frames at the pinned 24 fps grid")
    selected = pilot.get("scenes")
    if not isinstance(selected, list) or len(selected) != 3:
        raise ValueError("pilot manifest must contain exactly three scenes")
    by_key = {
        (str(item.get("video")), str(item.get("scene"))): item
        for item in scene_manifest.get("scenes", [])
    }
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in selected:
        key = (str(item.get("video", "")), str(item.get("scene", "")))
        if not all(key) or key in seen:
            raise ValueError(f"invalid or duplicate pilot scene {key!r}")
        seen.add(key)
        source_id = str(item.get("source_id", ""))
        if not source_id:
            raise ValueError(f"pilot scene {key!r} lacks a stable source_id")
        frame_directory = str(item.get("frame_directory", "extract_24"))
        if frame_directory != "extract_24":
            raise ValueError(f"pilot scene {key!r} has unsupported frame directory {frame_directory!r}")
        if source_id in reserved_source_ids or any(
            reserved.casefold() in source_id.casefold() for reserved in reserved_source_ids
        ):
            raise ValueError(f"reserved confirmation source entered pilot: {source_id}")
        catalog = by_key.get(key)
        if catalog is None:
            raise ValueError(f"pilot scene {key!r} is absent from the BP46 manifest")
        interval = catalog.get("intervals", {}).get("48", {})
        start = int(item.get("frame_start", -1))
        low = int(interval.get("start_frame", -1))
        high = int(interval.get("end_frame", -1))
        if interval.get("status") != "eligible" or start < low or start + frame_count > high:
            raise ValueError(
                f"pilot frame span {key}/{start}:{start + frame_count} is outside its eligible BP46 interval"
            )
        result.append(
            {
                **item,
                "frame_directory": frame_directory,
                "frame_count": frame_count,
                "working_fps": 24,
                "catalog_interval": {
                    "start_frame": low,
                    "end_frame": high,
                    "frame_hashes": dict(interval.get("frame_hashes", {})),
                },
            }
        )
    return result


def validate_quality_dataset_manifest(
    selection: dict[str, Any],
    scene_manifest: dict[str, Any],
    *,
    reserved_source_ids: Collection[str] = (),
) -> list[dict[str, Any]]:
    """Validate the complete eligible BP46 development set for dataset build."""
    if selection.get("schema") != QUALITY_DATASET_SCHEMA:
        raise ValueError(f"unsupported quality dataset schema {selection.get('schema')!r}")
    if selection.get("split") != "development_exposed_bp46":
        raise ValueError("quality dataset must preserve the exposed BP46 development split")
    if selection.get("reserved_sources_excluded") is not True:
        raise ValueError("quality dataset must explicitly exclude reserved confirmation sources")
    if int(selection.get("frame_count_per_scene", 0)) != 48 or int(selection.get("working_fps", 0)) != 24:
        raise ValueError("quality dataset requires the catalogued 48-frame interval on the 24 fps grid")
    source_hash = selection.get("source_scene_manifest_sha256")
    catalog_path = Path(str(selection.get("source_scene_manifest", "")))
    if catalog_path.is_file() and source_hash != _sha256(catalog_path):
        raise ValueError("quality dataset selection was made from a different scene catalog revision")
    by_key = {
        (str(item.get("video")), str(item.get("scene"))): item
        for item in scene_manifest.get("scenes", [])
    }
    expected = {
        key
        for key, row in by_key.items()
        if row.get("role") == "development_candidate"
        and row.get("intervals", {}).get("48", {}).get("status") == "eligible"
    }
    selected = selection.get("scenes")
    if not isinstance(selected, list) or not selected:
        raise ValueError("quality dataset must select at least one development scene")
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, Any]] = []
    for item in selected:
        key = (str(item.get("video", "")), str(item.get("scene", "")))
        if not all(key) or key in seen:
            raise ValueError(f"invalid or duplicate quality dataset scene {key!r}")
        seen.add(key)
        if key not in expected:
            raise ValueError(f"quality dataset scene {key!r} is not an eligible BP46 development candidate")
        source_id = str(item.get("source_id", ""))
        if source_id != f"{key[0]}_{key[1]}":
            raise ValueError(f"quality dataset scene {key!r} has a non-canonical source_id")
        if source_id in reserved_source_ids or key[0] in reserved_source_ids:
            raise ValueError(f"reserved confirmation source entered quality dataset: {source_id}")
        catalog = by_key[key]
        interval = catalog.get("intervals", {}).get("48", {})
        frame_start = int(item.get("frame_start", -1))
        frame_count = int(item.get("frame_count", 0))
        if frame_start != int(interval.get("start_frame", -2)) or frame_count != 48:
            raise ValueError(f"quality dataset interval for {source_id} does not match the eligible catalog interval")
        result.append(
            {
                **item,
                "frame_directory": "extract_24",
                "frame_count": frame_count,
                "working_fps": 24,
                "split": selection["split"],
                "role": catalog["role"],
                "catalog_interval": {
                    "start_frame": int(interval["start_frame"]),
                    "end_frame": int(interval["end_frame"]),
                    "frame_hashes": dict(interval.get("frame_hashes", {})),
                },
            }
        )
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(f"quality dataset must cover all eligible development scenes; missing={missing}, extra={extra}")
    return result


def _sam_runtime_inventory(sam_python: Path, sam_source: Path) -> dict[str, Any]:
    """Verify the isolated SAM runtime without allocating model memory."""
    if not sam_python.is_file() or not os.access(sam_python, os.X_OK):
        raise FileNotFoundError(f"SAM3.1 Python interpreter is unavailable: {sam_python}")
    code = (
        "import importlib.metadata as metadata,json,sys,torch; "
        f"sys.path.insert(0,{str(sam_source)!r}); "
        "from sam3.model_builder import build_sam3_multiplex_video_predictor; "
        "from src.components.segmentation.sam31 import Sam31SequenceSegmenter; "
        "import PIL; "
        "import inspect; "
        "print(json.dumps({'python':sys.executable,'python_version':sys.version,'torch':torch.__version__,"
        "'cuda_available':bool(torch.cuda.is_available()),'cuda_version':torch.version.cuda,"
        "'predictor_signature':str(inspect.signature(build_sam3_multiplex_video_predictor)),"
        "'sam_adapter':'src.components.segmentation.sam31:Sam31SequenceSegmenter',"
        "'packages':{name:metadata.version(name) for name in ('timm','iopath','Pillow','torchvision')},"
        "'pillow_version':PIL.__version__}))"
    )
    result = subprocess.run(
        [str(sam_python), "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode:
        raise RuntimeError(
            "pinned SAM3.1 predictor import failed in its configured environment: "
            + (result.stderr.strip() or result.stdout.strip())
        )
    try:
        inventory = json.loads(result.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise RuntimeError("SAM3.1 runtime probe returned malformed JSON") from exc
    if not inventory.get("cuda_available"):
        raise RuntimeError("SAM3.1 environment has no CUDA runtime")
    return inventory


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _external_root(path: Path, repo_root: Path) -> Path:
    resolved = path.expanduser().resolve()
    root = repo_root.resolve()
    if resolved == root or root in resolved.parents:
        raise ValueError(f"dataset inputs and outputs must be outside the code tree: {resolved}")
    return resolved


def _source_frame_directory(data_root: Path, scene: dict[str, Any]) -> tuple[Path, str]:
    """Resolve the extraction used by the BP46 loader, retaining its explicit BP21 fallback."""
    relative = Path("clips") / scene["video"] / scene["scene"] / "extract_24"
    candidates = (
        (data_root / "outputs" / "bp46-long-scenes" / relative, "bp46-long-scenes"),
        (data_root / "outputs" / "bp21-headroom" / relative, "bp21-headroom-fallback"),
    )
    for directory, dataset in candidates:
        if directory.is_dir():
            return directory, dataset
    checked = ", ".join(str(path) for path, _ in candidates)
    raise FileNotFoundError(f"{scene['source_id']}: no BP46 or documented BP21 extraction at {checked}")


def _sorted_frame_paths(directory: Path) -> list[Path]:
    indexed: dict[int, Path] = {}
    for path in directory.glob("frame_*.png"):
        try:
            frame_id = int(path.stem.removeprefix("frame_"))
        except ValueError as exc:
            raise ValueError(f"unexpected extracted frame filename {path.name!r}") from exc
        if frame_id in indexed:
            raise ValueError(f"duplicate extracted frame file ID {frame_id} in {directory}")
        indexed[frame_id] = path
    ids = sorted(indexed)
    if ids and ids != list(range(ids[0], ids[0] + len(ids))):
        raise ValueError(f"non-consecutive extracted frame file IDs in {directory}")
    return [indexed[frame_id] for frame_id in ids]


def _frame_paths_by_position(directory: Path, start: int, count: int) -> list[Path]:
    """The BP46 catalog indexes decoded frames from zero; PNG names start at one."""
    paths = _sorted_frame_paths(directory)
    selected = paths[start : start + count]
    if len(selected) != count:
        raise ValueError(
            f"missing extracted frame positions {start}:{start + count} in {directory} (has {len(paths)} frames)"
        )
    return selected


def _verify_catalog_frame_anchors(directory: Path, scene: dict[str, Any]) -> list[Path]:
    interval = scene["catalog_interval"]
    start = int(interval["start_frame"])
    end = int(interval["end_frame"])
    count = end - start
    if count != 48:
        raise ValueError(f"{scene['source_id']}: catalog interval is not the pinned 48-frame span")
    interval_paths = _frame_paths_by_position(directory, start, count)
    anchor_indices = {"first": 0, "mid": count // 2, "last": count - 1}
    expected = interval.get("frame_hashes", {})
    for anchor, offset in anchor_indices.items():
        expected_hash = expected.get(anchor)
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"{scene['source_id']}: catalog interval lacks {anchor} frame hash")
        path = interval_paths[offset]
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"{scene['source_id']}: {anchor} catalog frame hash mismatch at position {start + offset} "
                f"({path.name}): expected {expected_hash}, found {actual_hash}"
            )
    return interval_paths


def _source_frames(
    data_root: Path,
    scene: dict[str, Any],
    *,
    frame_count: int,
) -> tuple[np.ndarray, list[dict[str, Any]], Path]:
    directory = Path(scene.get("resolved_source_frame_directory", ""))
    if not directory.is_dir():
        directory, _ = _source_frame_directory(data_root, scene)
    _verify_catalog_frame_anchors(directory, scene)
    start = int(scene["frame_start"])
    paths = _frame_paths_by_position(directory, start, frame_count)
    frame_ids = list(range(start, start + frame_count))
    frames: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    for path, frame_id in zip(paths, frame_ids, strict=True):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"could not decode source frame {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
        records.append(
            {
                "source_id": scene["source_id"],
                "frame_index": frame_id,
                "source_frame_file_id": int(path.stem.removeprefix("frame_")),
                "pts": {"value": frame_id, "timebase_num": 1, "timebase_den": 24},
                "source_path": str(path),
                "source_file_sha256": _sha256(path),
                "rgb_sha256": _sha256_array(rgb),
            }
        )
    if len({frame.shape for frame in frames}) != 1:
        raise ValueError(f"{scene['source_id']}: source frames change dimensions")
    return np.stack(frames), records, directory


def _frame_paths_by_id(directory: Path, start: int, count: int) -> list[Path]:
    indexed: dict[int, Path] = {}
    for path in directory.glob("frame_*.png"):
        try:
            frame_id = int(path.stem.removeprefix("frame_"))
        except ValueError as exc:
            raise ValueError(f"unexpected BP46 frame filename {path.name!r}") from exc
        if frame_id in indexed:
            raise ValueError(f"duplicate BP46 frame ID {frame_id} in {directory}")
        indexed[frame_id] = path
    expected = range(start, start + count)
    missing = [frame_id for frame_id in expected if frame_id not in indexed]
    if missing:
        raise ValueError(f"missing BP46 frame IDs {missing[:5]} in {directory}")
    return [indexed[frame_id] for frame_id in expected]


def _scene_local_frame_index(frame_index: int, scene: dict[str, Any]) -> int:
    """Convert catalog-global observation indices to chunk-local wire indices."""
    local_index = int(frame_index) - int(scene["frame_start"])
    if local_index < 0:
        raise ValueError("observation frame precedes its selected scene interval")
    return local_index


def _reserved_ids(repo_root: Path) -> set[str]:
    ids: set[str] = set()
    for relative in (
        "manifests/gate_b_confirmation.json",
        "manifests/evaluation_20260916_coordinator_confirmation_reservation.json",
    ):
        path = repo_root / relative
        if not path.is_file():
            continue
        payload = _read_json(path)
        for key in ("confirmation_sources_forbidden", "reserved_source_ids", "source_ids"):
            values = payload.get(key, [])
            if isinstance(values, list):
                ids.update(str(item) for item in values)
        for item in payload.get("sources", []):
            if isinstance(item, dict):
                for key in ("source_id", "id", "video", "match_id"):
                    if item.get(key):
                        ids.add(str(item[key]))
    return ids


def inspect_inputs(args: argparse.Namespace, *, repo_root: Path) -> dict[str, Any]:
    """Check selected source frames and model provenance without loading weights."""
    data_root = _external_root(Path(args.data_root), repo_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"PointStream data root does not exist: {data_root}")
    selection_path = Path(args.dataset_manifest or args.pilot_manifest).resolve()
    catalog_path = Path(args.scene_manifest).resolve()
    selection = _read_json(selection_path)
    catalog = _read_json(catalog_path)
    if selection.get("schema") == QUALITY_DATASET_SCHEMA:
        scenes = validate_quality_dataset_manifest(
            selection,
            catalog,
            reserved_source_ids=_reserved_ids(repo_root),
        )
        quality_policy = dict(selection.get("quality_policy", {}))
        selection_version = str(selection.get("version", "sam31-quality-dataset"))
    else:
        scenes = validate_pilot_manifest(
            selection,
            catalog,
            reserved_source_ids=_reserved_ids(repo_root),
        )
        quality_policy = {}
        selection_version = str(selection.get("version", "sam31-observations"))
    external_scenes: list[dict[str, Any]] = []
    for scene in scenes:
        directory, source_dataset = _source_frame_directory(data_root, scene)
        _verify_catalog_frame_anchors(directory, scene)
        start = int(scene["frame_start"])
        frame_count = int(scene["frame_count"])
        exact = _frame_paths_by_position(directory, start, frame_count)
        external_scenes.append(
            {
                **scene,
                "resolved_source_frame_directory": str(directory),
                "source_frame_dataset": source_dataset,
                "frame_paths": [str(path) for path in exact],
                "source_frame_file_ids": [int(path.stem.removeprefix("frame_")) for path in exact],
                "frame_file_sha256": [_sha256(path) for path in exact],
            }
        )
    sam_checkpoint = Path(args.sam_checkpoint).expanduser().resolve()
    sam_source = Path(args.sam_source_root).expanduser().resolve()
    sam_python = Path(args.sam_python).expanduser().resolve()
    dwpose_detector = Path(args.dwpose_detector).expanduser().resolve()
    dwpose_pose = Path(args.dwpose_pose).expanduser().resolve()
    required = (sam_checkpoint, dwpose_detector, dwpose_pose, sam_python)
    missing = [str(path) for path in required if not path.is_file()]
    if not sam_source.is_dir():
        missing.append(str(sam_source))
    if missing:
        raise FileNotFoundError("required local model/source artifacts are missing: " + ", ".join(missing))
    revision = subprocess.run(
        ["git", "-C", str(sam_source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(sam_source), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    if dirty:
        raise ValueError(f"SAM3.1 source checkout is dirty: {sam_source}")
    expected_revision = args.sam_source_revision
    if expected_revision and revision != expected_revision:
        raise ValueError(f"SAM3.1 source revision mismatch: expected {expected_revision}, found {revision}")
    hashes = {
        "sam31_checkpoint": _sha256(sam_checkpoint),
        "dwpose_detector": _sha256(dwpose_detector),
        "dwpose_pose": _sha256(dwpose_pose),
    }
    if args.sam_checkpoint_sha256 and hashes["sam31_checkpoint"] != args.sam_checkpoint_sha256:
        raise ValueError("SAM3.1 checkpoint SHA-256 does not match the pinned manifest value")
    native = _native_codec_inventory()
    runtime = _runtime_dependency_inventory()
    if not runtime["cuda_available"]:
        raise RuntimeError("SAM3.1 pilot requires CUDA in the selected Python environment")
    if runtime["missing_packages"]:
        raise RuntimeError(
            "SAM3.1/DWPose runtime packages are unavailable: "
            + ", ".join(runtime["missing_packages"])
        )
    sam_runtime = _sam_runtime_inventory(sam_python, sam_source)
    return {
        "data_root": str(data_root),
        "pilot_manifest": str(selection_path),
        "pilot_manifest_sha256": _sha256(selection_path),
        "selection_manifest": str(selection_path),
        "selection_manifest_sha256": _sha256(selection_path),
        "dataset_version": selection_version,
        "quality_policy": quality_policy,
        "scene_manifest": str(catalog_path),
        "scene_manifest_sha256": _sha256(catalog_path),
        "split": selection["split"],
        "selected_scenes": external_scenes,
        "sam31_source_root": str(sam_source),
        "sam31_python": str(sam_python),
        "sam31_source_revision": revision,
        "sam31_checkpoint_path": str(sam_checkpoint),
        "model_hashes": hashes,
        "dwpose_detector_path": str(dwpose_detector),
        "dwpose_pose_path": str(dwpose_pose),
        "native_codec_inventory": native,
        "runtime_dependencies": runtime,
        "sam31_runtime": sam_runtime,
    }


def _runtime_dependency_inventory() -> dict[str, Any]:
    import importlib.metadata

    missing: list[str] = []
    versions: dict[str, str | None] = {}
    for name in ("numpy", "rtmlib", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
            missing.append(name)
    opencv_version = None
    for distribution in ("opencv-python", "opencv-python-headless", "opencv-contrib-python"):
        try:
            opencv_version = importlib.metadata.version(distribution)
            break
        except importlib.metadata.PackageNotFoundError:
            continue
    versions["opencv"] = opencv_version
    if opencv_version is None:
        missing.append("opencv-python")
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        cuda_version = torch.version.cuda
    except Exception as exc:
        cuda_available = False
        cuda_version = None
        if "torch" not in missing:
            missing.append(f"torch runtime ({type(exc).__name__})")
    try:
        import onnxruntime

        versions["onnxruntime"] = importlib.metadata.version("onnxruntime-gpu")
        providers = list(onnxruntime.get_available_providers())
    except importlib.metadata.PackageNotFoundError:
        try:
            versions["onnxruntime"] = importlib.metadata.version("onnxruntime")
            import onnxruntime

            providers = list(onnxruntime.get_available_providers())
        except (importlib.metadata.PackageNotFoundError, ImportError):
            versions["onnxruntime"] = None
            providers = []
    except ImportError:
        versions["onnxruntime"] = None
        providers = []
    if versions.get("onnxruntime") is None:
        missing.append("onnxruntime")
    if "CUDAExecutionProvider" not in providers:
        missing.append("onnxruntime CUDAExecutionProvider")
    return {
        "packages": versions,
        "missing_packages": missing,
        "cuda_available": cuda_available,
        "cuda_runtime_version": cuda_version,
        "onnxruntime_providers": providers,
    }


def _native_codec_inventory() -> dict[str, Any]:
    import shutil

    result: dict[str, Any] = {
        "ffmpeg_path": shutil.which("ffmpeg"),
        "vvencapp_path": shutil.which("vvencapp"),
        "vvdecapp_path": shutil.which("vvdecapp"),
        "ffmpeg_vvc_decoder": None,
    }
    if result["ffmpeg_path"]:
        version = subprocess.run(
            [result["ffmpeg_path"], "-version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        result["ffmpeg_version"] = (version.stdout or version.stderr).splitlines()[0]
        decoders = subprocess.run(
            [result["ffmpeg_path"], "-hide_banner", "-decoders"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
        result["ffmpeg_vvc_decoder"] = any(
            " vvc " in f" {line.lower()} " or "vvdec" in line.lower()
            for line in (decoders.stdout + decoders.stderr).splitlines()
        )
    return result


def _xyxy(box: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _overlay_mask(frame: np.ndarray, masks: list[np.ndarray], color: tuple[int, int, int]) -> np.ndarray:
    image = np.asarray(frame, dtype=np.uint8).copy()
    if not masks:
        return image
    union = np.logical_or.reduce([np.asarray(mask) != 0 for mask in masks])
    tint = np.asarray(color, dtype=np.float32)
    image[union] = np.clip(0.42 * image[union] + 0.58 * tint, 0, 255).astype(np.uint8)
    return image


def _bbox_points(mask: np.ndarray) -> tuple[tuple[int, int, int, int], tuple[float, float]] | None:
    bbox = _mask_bbox(mask)
    if bbox is None:
        return None
    ys, xs = np.nonzero(mask)
    point = (float(np.median(xs)), float(np.median(ys)))
    return bbox, point


def _jsonable_provenance(value: Any) -> dict[str, Any]:
    return {
        "name": value.name,
        "model_revision": value.model_revision,
        "checkpoint_sha256": value.checkpoint_sha256,
        "config_sha256": value.config_sha256,
        "policy": value.policy,
    }


def _sam_worker(config_path: Path) -> int:
    """Run SAM3.1 only, so its supported CUDA runtime never overlaps DWPose."""
    from src.components.segmentation.sam31 import Sam31SequenceSegmenter

    config = _read_json(config_path)
    scene = config["scene"]
    frame_dir = Path(config["frames_dir"])
    metadata_path = Path(config["metadata_path"])
    arrays_path = Path(config["arrays_path"])
    count = int(config["frame_count"])
    torch = __import__("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("the SAM3.1 worker requires CUDA")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    segmenter = Sam31SequenceSegmenter(
        checkpoint_path=config["sam_checkpoint"],
        source_root=config["sam_source_root"],
        source_revision=config["sam_source_revision"],
        checkpoint_sha256=config["sam_checkpoint_sha256"],
        prob_threshold=float(config["prob_threshold"]),
    )
    model_load_seconds = time.perf_counter() - started
    policy = OFFLINE_POLICY
    provenance = segmenter.provenance(policy)
    width, height = int(config["frame_width"]), int(config["frame_height"])
    outputs: dict[str, dict[tuple[int, str], Any]] = {"player": {}, "racket": {}}
    prompt_records: list[dict[str, Any]] = []
    retry_records: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    try:
        for role in ROLES:
            segmenter.start_session(
                role,
                frame_dir,
                frame_width=width,
                frame_height=height,
                policy=policy,
            )
            prompt_start = time.perf_counter()
            prompt = segmenter.add_prompt(
                role,
                frame_index=0,
                object_id=f"{scene['source_id']}:{role}:prompt",
                text="tennis player" if role == "player" else "tennis racket",
            )
            prompt_seconds = time.perf_counter() - prompt_start
            for item in prompt:
                outputs[role][(item.frame_index, item.object_id)] = item
            propagation_start = time.perf_counter()
            propagated = segmenter.propagate(
                role,
                policy=policy,
                direction="both",
                frame_count=count,
            )
            propagation_seconds = time.perf_counter() - propagation_start
            for item in propagated:
                key = (item.frame_index, item.object_id)
                prior = outputs[role].get(key)
                if prior is None or item.mask is not None:
                    outputs[role][key] = item
            segmenter.close_session(role)
            prompt_records.append(
                {
                    "role": role,
                    "text": "tennis player" if role == "player" else "tennis racket",
                    "prompt_seconds": prompt_seconds,
                    "propagation_seconds": propagation_seconds,
                    "initial_object_ids": sorted({item.object_id for item in prompt}),
                }
            )

        for role_name, records in outputs.items():
            role = cast(Role, role_name)
            by_object: dict[str, list[Any]] = defaultdict(list)
            for item in records.values():
                by_object[item.object_id].append(item)
            for object_id, object_records in sorted(by_object.items()):
                for missing in sorted(object_records, key=lambda value: value.frame_index):
                    if missing.mask is not None:
                        continue
                    seeds = [item for item in object_records if item.mask is not None]
                    if not seeds:
                        retry_records.append(
                            {
                                "role": role,
                                "object_id": object_id,
                                "frame_index": missing.frame_index,
                                "attempted": False,
                                "reason": "no_observed_mask_available_for_spatial_guidance",
                            }
                        )
                        continue
                    seed = min(
                        seeds,
                        key=lambda item: (abs(item.frame_index - missing.frame_index), item.frame_index),
                    )
                    spatial = _bbox_points(np.asarray(seed.mask))
                    if spatial is None:
                        continue
                    bbox, point = spatial
                    retry_key = f"retry-{role}-{hashlib.sha1(object_id.encode()).hexdigest()[:8]}-{missing.frame_index}"
                    segmenter.start_session(
                        role,
                        frame_dir,
                        frame_width=width,
                        frame_height=height,
                        policy=policy,
                        session_key=retry_key,
                    )
                    try:
                        retry_prompt = segmenter.add_prompt(
                            role,
                            frame_index=missing.frame_index,
                            object_id=object_id,
                            points=[point],
                            point_labels=[1],
                            tracker_id=seed.tracker_id,
                            session_key=retry_key,
                        )
                        retry_result = next(
                            (
                                item
                                for item in retry_prompt
                                if item.object_id == object_id and item.mask is not None
                            ),
                            None,
                        )
                        if retry_result is None:
                            retry_propagated = segmenter.propagate(
                                role,
                                policy=policy,
                                direction="both",
                                frame_count=count,
                                start_frame_index=missing.frame_index,
                                session_key=retry_key,
                            )
                            retry_result = next(
                                (
                                    item
                                    for item in retry_propagated
                                    if item.object_id == object_id
                                    and item.frame_index == missing.frame_index
                                ),
                                None,
                            )
                        if retry_result is not None and retry_result.mask is not None:
                            records[(missing.frame_index, object_id)] = retry_result
                        retry_records.append(
                            {
                                "role": role,
                                "object_id": object_id,
                                "frame_index": missing.frame_index,
                                "attempted": True,
                                "guidance": "nearest_observed_mask_median_foreground_point",
                                "prompt_form": "positive_point_only",
                                "seed_frame_index": seed.frame_index,
                                "bbox_xyxy": list(bbox),
                                "positive_point_xy": list(point),
                                "accepted": retry_result is not None and retry_result.mask is not None,
                                "ambiguous_extra_masks": max(0, len(retry_prompt) - 1),
                            }
                        )
                    finally:
                        segmenter.close_session(role, session_key=retry_key)
    finally:
        for session_role, session_key in list(segmenter.sessions):
            segmenter.close_session(session_role, session_key=session_key)

    masks: dict[str, np.ndarray] = {}
    serialized: list[dict[str, Any]] = []
    for role_name, role_records in outputs.items():
        for (frame_index, object_id), item in sorted(role_records.items()):
            mask_key = None
            if item.mask is not None:
                mask_key = f"mask_{len(masks):06d}"
                masks[mask_key] = np.asarray(item.mask, dtype=np.uint8)
            serialized.append(
                {
                    "role": role_name,
                    "frame_index": int(frame_index),
                    "object_id": object_id,
                    "tracker_id": item.tracker_id,
                    "score": item.score,
                    "status": item.status.value,
                    "reason": item.reason,
                    "mask_key": mask_key,
                }
            )
    np.savez_compressed(arrays_path, **masks)
    result = {
        "schema": "pointstream.sam31-worker-result.v1",
        "source_id": scene["source_id"],
        "provenance": _jsonable_provenance(provenance),
        "outputs": serialized,
        "prompts": prompt_records,
        "retries": retry_records,
        "model_load_seconds": model_load_seconds,
        "inference_seconds": time.perf_counter() - total_started,
        "gpu": torch.cuda.get_device_name(0),
        "sam_runtime": {
            "python": sys.executable,
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
        },
        "gpu_memory_bytes": {
            "peak_allocated": int(torch.cuda.max_memory_allocated()),
            "peak_reserved": int(torch.cuda.max_memory_reserved()),
            "current_allocated": int(torch.cuda.memory_allocated()),
            "current_reserved": int(torch.cuda.memory_reserved()),
        },
        "arrays_path": str(arrays_path),
        "arrays_sha256": _sha256(arrays_path),
    }
    metadata_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(metadata_path), "arrays": str(arrays_path)}))
    return 0


def _load_sam_worker(metadata_path: Path) -> dict[str, Any]:
    from types import SimpleNamespace
    from src.contracts.observation import EstimatorProvenance, ObservationStatus

    payload = _read_json(metadata_path)
    arrays_path = Path(payload["arrays_path"])
    if _sha256(arrays_path) != payload["arrays_sha256"]:
        raise ValueError(f"SAM3.1 worker mask archive hash mismatch: {arrays_path}")
    with np.load(arrays_path, allow_pickle=False) as archive:
        masks = {name: archive[name] for name in archive.files}
    outputs: dict[str, dict[tuple[int, str], Any]] = {"player": {}, "racket": {}}
    for row in payload["outputs"]:
        item = SimpleNamespace(
            frame_index=int(row["frame_index"]),
            object_id=row["object_id"],
            tracker_id=row["tracker_id"],
            score=row["score"],
            status=ObservationStatus(row["status"]),
            reason=row["reason"],
            mask=masks[row["mask_key"]] if row["mask_key"] is not None else None,
        )
        outputs[row["role"]][(item.frame_index, item.object_id)] = item
    payload["outputs"] = outputs
    payload["provenance"] = EstimatorProvenance(**payload["provenance"])
    return payload


def _sequence_perception(
    scene: dict[str, Any],
    frames: np.ndarray,
    *,
    sam_result: dict[str, Any],
    pose_estimator: Any,
    quality_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Join isolated SAM3.1 output with DWPose and the shared geometry path."""
    from src.components.detection.geometry import Box
    from src.components.detection.types import Detection
    from src.components.perception.association import RacketPlayerAssociator
    from src.components.perception.coordinates import render_object_view
    from src.components.perception.model_adapters import render_model_adapter_views
    from src.components.rigid.racket import extract_racket_cross
    from src.components.rigid.types import ObservedObject, PlayerPose
    from src.contracts.observation import Observation, ObservationStatus, Pts
    from src.runner.perception import render_runtime_conditioning_view

    count, height, width, _ = frames.shape
    policy = OFFLINE_POLICY
    selection_policy = dict(quality_policy or {})
    sam_provenance = sam_result["provenance"]
    player_provenance = pose_estimator.provenance(policy=policy)
    outputs = sam_result["outputs"]
    prompt_records = sam_result["prompts"]
    retry_records = sam_result["retries"]

    poses_by_frame: dict[int, list[tuple[str, Any, Any]]] = defaultdict(list)
    pose_records: list[dict[str, Any]] = []
    pose_visibility: Counter[str] = Counter()
    player_ids = sorted({object_id for _, object_id in outputs["player"]})
    for frame_index in range(count):
        for object_id in player_ids:
            item = outputs["player"].get((frame_index, object_id))
            if item is None or item.mask is None:
                continue
            mask = np.asarray(item.mask, dtype=np.uint8)
            bbox = _mask_bbox(mask)
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            detection = Detection(
                class_name="person",
                bbox=Box(float(x0), float(y0), float(x1), float(y1)),
                track_id=object_id,
            )
            pose, transform = pose_estimator.estimate_with_transform(
                frames[frame_index],
                detection,
                mask=mask,
            )
            if pose is None:
                continue
            PlayerPose(
                object_id=object_id,
                frame_index=frame_index,
                keypoints=pose.values,
                schema_name=pose.schema.name,
            )
            poses_by_frame[frame_index].append((object_id, pose, transform))
            pose_visibility.update(
                {
                    "visible": int(np.count_nonzero(pose.visibility == 2)),
                    "low_confidence_or_occluded": int(np.count_nonzero(pose.visibility == 1)),
                    "unavailable": int(np.count_nonzero(pose.visibility == 0)),
                }
            )
            pose_records.append(
                {
                    "source_id": scene["source_id"],
                    "frame_index": int(scene["frame_start"]) + frame_index,
                    "object_id": object_id,
                    "schema": pose.schema.name,
                    "values": pose.values.tolist(),
                    "present": pose.present.tolist(),
                    "visibility": pose.visibility.tolist(),
                    "estimator_provenance": _jsonable_provenance(player_provenance),
                    "crop_transform": transform.to_record() if transform is not None else None,
                }
            )

    player_quality, player_track_quality = filter_player_candidates(
        outputs["player"],
        poses_by_frame,
        frames,
        max_players_per_frame=int(selection_policy.get("max_players_per_frame", 2)),
        minimum_track_score=float(selection_policy.get("player_track_min_score", 0.58)),
        minimum_sam_confidence=float(selection_policy.get("player_min_sam_confidence", 0.40)),
        minimum_pose_support=float(selection_policy.get("player_min_pose_support", 0.08)),
    )
    retained_poses_by_frame = {
        frame_index: [
            row
            for row in rows
            if player_quality.get((frame_index, row[0]), QualityDecision(False, 0.0, (), {})).eligible
        ]
        for frame_index, rows in poses_by_frame.items()
    }

    associator = RacketPlayerAssociator()
    associated_rackets: dict[tuple[int, str], ObservedObject] = {}
    geometry_by_key: dict[tuple[int, str], Any] = {}
    for frame_index in range(count):
        racket_objects: list[ObservedObject] = []
        player_poses = [
            PlayerPose(object_id, frame_index, pose.values, pose.schema.name)
            for object_id, pose, _transform in retained_poses_by_frame.get(frame_index, ())
        ]
        for (record_frame, object_id), item in outputs["racket"].items():
            if record_frame != frame_index or item.mask is None:
                continue
            bbox = _mask_bbox(np.asarray(item.mask))
            if bbox is None:
                continue
            racket_objects.append(
                ObservedObject(
                    object_id=object_id,
                    object_class="racket",
                    frame_index=frame_index,
                    bbox=_xyxy(bbox),
                    mask=item.mask,
                )
            )
        linked = associator.associate(racket_objects, player_poses)
        for racket in linked:
            key = (frame_index, racket.object_id)
            associated_rackets[key] = racket
            geometry_by_key[key] = extract_racket_cross(racket, player_poses)

    racket_quality, racket_track_quality = filter_racket_candidates(
        outputs["racket"],
        outputs["player"],
        player_quality,
        associated_rackets,
        retained_poses_by_frame,
        (height, width),
        minimum_score=float(selection_policy.get("racket_track_min_score", 0.68)),
        minimum_sam_confidence=float(selection_policy.get("racket_min_sam_confidence", 0.50)),
        maximum_player_area_fraction=float(selection_policy.get("racket_area_must_be_less_than_player_area_fraction", 0.85)),
        maximum_player_extent_fraction=float(selection_policy.get("racket_bbox_diagonal_must_not_exceed_player_bbox_diagonal_fraction", 1.0)),
        maximum_player_mask_overlap_fraction=float(selection_policy.get("racket_player_mask_overlap_fraction_max", 0.90)),
    )

    all_records: list[dict[str, Any]] = []
    masks: dict[str, np.ndarray] = {}
    view_records: list[dict[str, Any]] = []
    track_switches: Counter[str] = Counter()
    expected: Counter[str] = Counter()
    observed: Counter[str] = Counter()
    fallback_counts: Counter[str] = Counter()
    coordinate_errors: list[float] = []
    quarantined: list[dict[str, Any]] = []
    frame_roles: list[dict[str, Any]] = []
    object_ids_by_role = {
        role: sorted({object_id for _, object_id in records})
        for role, records in outputs.items()
    }
    previous_tracker: dict[tuple[str, str], int | None] = {}
    for frame_index in range(count):
        frame_records: dict[str, list[tuple[str, np.ndarray, tuple[int, int, int, int], Any]]] = {
            "player": [],
            "racket": [],
        }
        for role in ROLES:
            for object_id in object_ids_by_role[role]:
                expected[role] += 1
                item = outputs[role].get((frame_index, object_id))
                if item is not None and item.mask is not None:
                    mask = (np.asarray(item.mask) != 0).astype(np.uint8)
                    bbox = _mask_bbox(mask)
                    if bbox is None:
                        item = None
                    else:
                        observed[role] += 1
                        frame_records[role].append((object_id, mask, bbox, item))
                if item is not None:
                    current_tracker = item.tracker_id
                    prior = previous_tracker.get((role, object_id))
                    if prior is not None and current_tracker is not None and prior != current_tracker:
                        track_switches[role] += 1
                    if current_tracker is not None:
                        previous_tracker[(role, object_id)] = current_tracker

                linked_racket = associated_rackets.get((frame_index, object_id)) if role == "racket" else None
                reason = "mask_unavailable" if item is None or item.mask is None else None
                quality_decisions = player_quality if role == "player" else racket_quality
                quality_decision = quality_decisions.get(
                    (frame_index, object_id),
                    QualityDecision(False, 0.0, ("quality_observation_unavailable",), {}),
                )
                view_transform = None
                if item is not None and item.mask is not None:
                    view_box = _mask_bbox(np.asarray(item.mask))
                    if view_box is not None:
                        _crop, _crop_mask, transform = render_object_view(
                            frames[frame_index],
                            np.asarray(item.mask),
                            _xyxy(view_box),
                        )
                        view_transform = transform.to_record()
                        x0, y0, x1, y1 = view_box
                        probes = np.asarray(
                            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                            dtype=np.float64,
                        )
                        returned = transform.canvas_to_source(
                            transform.source_to_canvas(probes)
                        )
                        coordinate_errors.extend(
                            np.linalg.norm(returned - probes, axis=1).tolist()
                        )
                if role == "racket" and linked_racket is not None:
                    associated_player_id = linked_racket.associated_player_id
                    associated_wrist = linked_racket.associated_wrist
                else:
                    associated_player_id = None
                    associated_wrist = None
                pts = Pts(int(scene["frame_start"]) + frame_index, 1, 24)
                if item is not None and item.mask is not None:
                    observation = Observation(
                        source_id=scene["source_id"],
                        frame_index=int(scene["frame_start"]) + frame_index,
                        pts=pts,
                        object_id=object_id,
                        object_class="player" if role == "player" else "racket",
                        coordinate_system="pixel_xy_top_left_full_frame",
                        frame_width=width,
                        frame_height=height,
                        provenance=sam_provenance,
                        status=ObservationStatus.OBSERVED,
                        mask=np.asarray(item.mask, dtype=np.uint8),
                        transforms=(view_transform,) if view_transform is not None else (),
                        associated_player_id=associated_player_id,
                        associated_wrist=associated_wrist,
                    )
                    mask_key = f"m{len(masks):06d}"
                    observed_mask = observation.mask
                    if observed_mask is None:
                        raise RuntimeError("observed record is missing its mask")
                    masks[mask_key] = observed_mask
                    raw_record = observation.to_record()
                    raw_record["mask_key"] = mask_key
                    raw_record["mask_sha256"] = _sha256_array(observed_mask)
                    record = {**raw_record, "tracker_id": item.tracker_id}
                else:
                    observation = Observation.missing(
                        source_id=scene["source_id"],
                        frame_index=int(scene["frame_start"]) + frame_index,
                        pts=pts,
                        object_id=object_id,
                        object_class="player" if role == "player" else "racket",
                        frame_width=width,
                        frame_height=height,
                        provenance=sam_provenance,
                        reason=(item.reason if item is not None and item.reason else reason or "mask_unavailable"),
                        associated_player_id=associated_player_id,
                        associated_wrist=associated_wrist,
                    )
                    record = {**observation.to_record(), "tracker_id": item.tracker_id if item else None}
                    if role == "racket":
                        quarantined.append(
                            {
                                "source_id": scene["source_id"],
                                "frame_index": int(scene["frame_start"]) + frame_index,
                                "object_id": object_id,
                                "reason": "racket_mask_missing",
                            }
                        )
                if role == "racket" and linked_racket is not None:
                    shape = geometry_by_key.get((frame_index, object_id))
                    record["racket_geometry"] = shape.to_record() if shape is not None else None
                    if shape is not None and shape.kind == "hull_fallback_v1":
                        fallback_counts[str(shape.fallback_reason)] += 1
                        quarantined.append(
                            {
                                "source_id": scene["source_id"],
                                "frame_index": int(scene["frame_start"]) + frame_index,
                                "object_id": object_id,
                                "reason": "racket_hull_fallback_excluded_from_cross_training",
                                "fallback_reason": shape.fallback_reason,
                            }
                        )
                record["quality"] = quality_decision.to_record()
                record["training_eligible"] = bool(quality_decision.eligible)
                if not quality_decision.eligible:
                    quarantined.append(
                        {
                            "source_id": scene["source_id"],
                            "frame_index": int(scene["frame_start"]) + frame_index,
                            "object_id": object_id,
                            "role": role,
                            "reason": "quality_filter_rejected",
                            "quality_reasons": list(quality_decision.reasons),
                            "quality_score": float(quality_decision.score),
                        }
                    )
                all_records.append(record)

        player_masks = [
            record[1]
            for record in frame_records["player"]
            if player_quality.get((frame_index, record[0]), QualityDecision(False, 0.0, (), {})).eligible
        ]
        racket_masks = [
            record[1]
            for record in frame_records["racket"]
            if racket_quality.get((frame_index, record[0]), QualityDecision(False, 0.0, (), {})).eligible
        ]
        frame_roles.append(
            {
                "frame_index": frame_index,
                "player_mask_count": len(player_masks),
                "racket_mask_count": len(racket_masks),
                "failed": not player_masks or not racket_masks,
            }
        )
        for role in ROLES:
            color = (35, 220, 70) if role == "player" else (240, 130, 35)
            for object_id, mask, bbox, item in frame_records[role]:
                quality_decisions = player_quality if role == "player" else racket_quality
                quality_decision = quality_decisions.get(
                    (frame_index, object_id),
                    QualityDecision(False, 0.0, ("quality_observation_unavailable",), {}),
                )
                pose_data = next(
                    (pose for pose_id, pose, _transform in retained_poses_by_frame.get(frame_index, ()) if pose_id == object_id),
                    None,
                ) if role == "player" else None
                view = render_runtime_conditioning_view(
                    frames[frame_index],
                    mask,
                    tuple(float(value) for value in bbox),
                    pose=pose_data,
                )
                view_records.append(
                    {
                        "frame_index": frame_index,
                        "object_id": object_id,
                        "role": role,
                        "appearance": view.appearance,
                        "mask": view.mask,
                        "pose": view.pose,
                        "bbox": bbox,
                        "source_mask": mask,
                        "overlay": _overlay_mask(frames[frame_index], [mask], color),
                        "transform": view.appearance_transform,
                        "_pose_object": pose_data,
                        "_player_source_mask": mask if role == "player" else np.zeros_like(mask),
                        "_racket_source_mask": mask if role == "racket" else np.zeros_like(mask),
                        "_racket_geometry": (
                            geometry_by_key.get((frame_index, object_id))
                            if role == "racket"
                            else None
                        ),
                        "quality_eligible": bool(quality_decision.eligible),
                        "quality_score": float(quality_decision.score),
                        "quality_reasons": list(quality_decision.reasons),
                    }
                )

        # The associated view is built from exactly the player+racket observations
        # above; a racket without a visible wrist remains a quarantined diagnostic.
        for (linked_frame, racket_id), racket in associated_rackets.items():
            if (
                linked_frame != frame_index
                or racket.associated_player_id is None
                or not racket_quality.get((frame_index, racket_id), QualityDecision(False, 0.0, (), {})).eligible
                or not player_quality.get((frame_index, racket.associated_player_id), QualityDecision(False, 0.0, (), {})).eligible
            ):
                continue
            player_entry = next(
                (row for row in frame_records["player"] if row[0] == racket.associated_player_id),
                None,
            )
            racket_entry = next((row for row in frame_records["racket"] if row[0] == racket_id), None)
            if player_entry is None or racket_entry is None:
                continue
            joint_mask = np.logical_or(player_entry[1] != 0, racket_entry[1] != 0).astype(np.uint8)
            joint_bbox = (
                min(player_entry[2][0], racket_entry[2][0]),
                min(player_entry[2][1], racket_entry[2][1]),
                max(player_entry[2][2], racket_entry[2][2]),
                max(player_entry[2][3], racket_entry[2][3]),
            )
            linked_pose = next(
                (pose for pose_id, pose, _transform in retained_poses_by_frame.get(frame_index, ()) if pose_id == racket.associated_player_id),
                None,
            )
            joint = render_runtime_conditioning_view(
                frames[frame_index],
                joint_mask,
                tuple(float(value) for value in joint_bbox),
                pose=linked_pose,
            )
            shape = geometry_by_key.get((frame_index, racket_id))
            cross = np.zeros_like(joint.appearance)
            if shape is not None and shape.kind == "racket_cross_v1":
                converted = joint.appearance_transform.source_to_canvas(np.asarray(shape.points))
                for start, end in ((0, 1), (2, 3)):
                    cv2.line(
                        cross,
                        tuple(np.rint(converted[start]).astype(int)),
                        tuple(np.rint(converted[end]).astype(int)),
                        (255, 180, 20),
                        2,
                        cv2.LINE_AA,
                    )
                for point in converted:
                    cv2.circle(cross, tuple(np.rint(point).astype(int)), 3, (255, 255, 255), -1)
            pose_image = joint.pose if joint.pose is not None else np.zeros_like(cross)
            condition = np.maximum(pose_image, cross)
            view_records.append(
                {
                    "frame_index": frame_index,
                    "object_id": f"{racket.associated_player_id}+{racket_id}",
                    "role": "joint",
                    "appearance": joint.appearance,
                    "mask": joint.mask,
                    "pose": condition,
                    "bbox": joint_bbox,
                    "source_mask": joint_mask,
                    "overlay": condition,
                    "transform": joint.appearance_transform,
                    "eligible_for_cross_training": shape is not None and shape.kind == "racket_cross_v1",
                    "_pose_object": linked_pose,
                    "_player_source_mask": player_entry[1],
                    "_racket_source_mask": racket_entry[1],
                    "_racket_geometry": shape,
                    "quality_eligible": bool(shape is not None and shape.kind == "racket_cross_v1"),
                    "quality_score": min(
                        float(player_quality[(frame_index, racket.associated_player_id)].score),
                        float(racket_quality[(frame_index, racket_id)].score),
                    ),
                    "quality_reasons": [],
                }
            )

    for view in view_records:
        transform = view["transform"]
        adapters = render_model_adapter_views(
            view=view["role"],
            transform=transform,
            player_mask=transform.resize_mask(view["_player_source_mask"]),
            racket_mask=transform.resize_mask(view["_racket_source_mask"]),
            pose=view["_pose_object"],
            racket_geometry=view["_racket_geometry"],
        )
        view["_model_adapters"] = adapters

    identity_discontinuities = dict(track_switches)
    coverage = {
        role: {
            "track_count": len(object_ids_by_role[role]),
            "expected_track_frames": int(expected[role]),
            "observed_track_frames": int(observed[role]),
            "missing_track_frames": int(expected[role] - observed[role]),
            "track_frame_coverage": float(observed[role] / expected[role]) if expected[role] else 0.0,
        }
        for role in ("player", "racket")
    }
    return {
        "masks": masks,
        "observations": all_records,
        "poses": pose_records,
        "views": view_records,
        "prompts": prompt_records,
        "retries": retry_records,
        "coverage": coverage,
        "identity_discontinuities": identity_discontinuities,
        "pose_visibility_counts": dict(pose_visibility),
        "racket_fallback_counts": dict(fallback_counts),
        "coordinate_roundtrip_error_px": {
            "max": max(coordinate_errors, default=0.0),
            "mean": float(np.mean(coordinate_errors)) if coordinate_errors else 0.0,
            "n_points": len(coordinate_errors),
        },
        "quarantined": quarantined,
        "quality": {
            "player_tracks": player_track_quality,
            "racket_tracks": racket_track_quality,
            "player_eligible_observations": sum(item.eligible for item in player_quality.values()),
            "player_rejected_observations": sum(not item.eligible for item in player_quality.values()),
            "racket_eligible_observations": sum(item.eligible for item in racket_quality.values()),
            "racket_rejected_observations": sum(not item.eligible for item in racket_quality.values()),
        },
        "frame_roles": frame_roles,
        "estimator_provenance": {
            "sam31": _jsonable_provenance(sam_provenance),
            "dwpose": _jsonable_provenance(player_provenance),
        },
        "sam_runtime": sam_result["sam_runtime"],
        "sam_gpu_memory_bytes": sam_result["gpu_memory_bytes"],
        "sam_model_load_seconds": sam_result["model_load_seconds"],
        "sam_inference_seconds": sam_result["inference_seconds"],
        "sam_arrays_sha256": sam_result["arrays_sha256"],
    }


def _reference_agreement(scene: dict[str, Any], new_player_masks: list[np.ndarray]) -> dict[str, Any]:
    """Compare with old masks as reference agreement, never as accuracy."""
    try:
        from experiments.long_scenes.loader import load_long_scene_clip

        cached = load_long_scene_clip(
            scene["video"],
            scene["scene"],
            48,
            full_trajectory=False,
        )
    except Exception as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    offset = int(scene["frame_start"]) - cached.start_frame
    if offset < 0 or offset + len(new_player_masks) > len(cached.masks):
        return {"status": "unavailable", "reason": "cached mask interval does not cover exact pilot frames"}
    ious: list[float] = []
    for index, mask in enumerate(new_player_masks):
        reference = np.asarray(cached.masks[offset + index], dtype=bool)
        candidate = np.asarray(mask, dtype=bool)
        if reference.shape != candidate.shape:
            return {"status": "unavailable", "reason": "cached reference mask dimensions differ"}
        union = np.count_nonzero(reference | candidate)
        ious.append(float(np.count_nonzero(reference & candidate) / union) if union else 1.0)
    return {
        "status": "measured_reference_agreement",
        "metric": "framewise_union_mask_IoU",
        "mean": float(np.mean(ious)) if ious else None,
        "min": float(np.min(ious)) if ious else None,
        "max": float(np.max(ious)) if ious else None,
        "n_frames": len(ious),
        "interpretation": "agreement with cached historical masks; not ground-truth accuracy",
    }


def _encode_client_payload(
    scene: dict[str, Any],
    frames: np.ndarray,
    result: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    from src.contracts.config import BackendConfig, LatticeConfig, PointstreamConfig
    from src.pipeline.reconstruction.reconstruct import ObjectRequest
    from src.runner.run import run
    from src.runner.mask_wire import decode_mask

    count, height, width, _ = frames.shape
    from src.components.perception.coordinates import render_object_view

    objects: list[ObjectRequest] = []
    expected_masks: dict[tuple[str, int], np.ndarray] = {}
    for record in result["observations"]:
        if record.get("status") != "observed":
            continue
        object_id = str(record["object_id"])
        frame_index = _scene_local_frame_index(record["frame_index"], scene)
        mask = np.asarray(result["masks"][record["mask_key"]], dtype=np.uint8)
        bbox = _mask_bbox(mask)
        if bbox is None:
            continue
        appearance, _condition_mask, _transform = render_object_view(
            frames[frame_index], mask, _xyxy(bbox)
        )
        objects.append(
            ObjectRequest(
                object_id=object_id,
                appearance=appearance,
                bbox=bbox,
                mask=mask,
                frame_index=frame_index,
                object_class=str(record["object_class"]),
            )
        )
        expected_masks[(object_id, frame_index)] = mask.copy()

    lattice = LatticeConfig(
        scene_classification=False,
        detection=True,
        selection=False,
        tracking=False,
        appearance=True,
        motion=False,
        temporal_policy=False,
        pose=False,
        segmentation=True,
        rigid_objects=False,
        background=False,
        generation=False,
        residual=False,
    )
    config = PointstreamConfig(
        lattice=lattice,
        segmenter=BackendConfig(backend="none"),
    )
    runner_output = run(
        config,
        (frames,),
        objects=(tuple(objects),),
        heartbeat_interval=None,
        sync_fn=None,
    )
    chunk = runner_output.chunks[0]
    raw_payload = chunk.bag["wire_request"]
    if not isinstance(raw_payload, (bytes, bytearray)):
        raise TypeError("wire_request payload must be bytes")
    payload = bytes(raw_payload)
    payload_path = output_dir / "pointstream-client-payload.npz"
    payload_path.write_bytes(payload)
    decoded_path = output_dir / "fresh-process-decoded.npy"
    decode_cmd = [sys.executable, str(Path(__file__).resolve()), "--decode-worker", str(payload_path), str(decoded_path)]
    child_env = {key: value for key, value in os.environ.items() if key not in {"PS_DATA_ROOT", "SAM31_CHECKPOINT", "SAM31_SOURCE_ROOT"}}
    child_env["OMP_NUM_THREADS"] = "1"
    child = subprocess.run(
        decode_cmd,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
        env=child_env,
    )
    if child.returncode != 0 or not decoded_path.is_file():
        raise RuntimeError(
            f"fresh-process client decode failed: {child.stderr[-2000:] or child.stdout[-2000:]}"
        )
    decoded = np.load(decoded_path, allow_pickle=False)
    encoder_prediction = runner_output.delivered_frames
    if not np.array_equal(decoded, encoder_prediction):
        raise AssertionError("fresh-process decode differs from PointStream runner's client reconstruction")
    transport_masks_match = True
    with np.load(io_bytes(payload), allow_pickle=False) as arrays:
        metadata = json.loads(np.asarray(arrays["metadata"], dtype=np.uint8).tobytes())
        for placement in metadata["placements"]:
            expected = expected_masks[(placement["object_id"], int(placement["frame_index"]))]
            actual = decode_mask(
                np.asarray(arrays[placement["mask_key"]], dtype=np.uint8).tobytes()
            )
            if not np.array_equal(actual, expected):
                transport_masks_match = False
                break
    if not transport_masks_match:
        raise AssertionError("PSM1 mask did not survive the transport roundtrip")
    motion_errors: list[int] = []
    source_boxes = {
        (
            str(record["object_id"]),
            _scene_local_frame_index(record["frame_index"], scene),
        ): _mask_bbox(
            np.asarray(result["masks"][record["mask_key"]])
        )
        for record in result["observations"]
        if record.get("status") == "observed"
    }
    for placement in metadata["placements"]:
        original_bbox = source_boxes[(placement["object_id"], int(placement["frame_index"]))]
        if original_bbox is None:
            continue
        motion_errors.extend(
            abs(int(actual) - int(original))
            for actual, original in zip(placement["bbox"], original_bbox, strict=True)
        )
    size_record = chunk.sizes.as_dict()
    if int(size_record["transport_total"]) != len(payload):
        raise AssertionError("PointStream byte ledger does not match serialized client payload length")
    return {
        "payload_path": str(payload_path),
        "payload_sha256": _sha256(payload_path),
        "payload_bytes": len(payload),
        "pointstream_runner_size_ledger": size_record,
        "pointstream_runner_encode": True,
        "fresh_process_decode_command": decode_cmd,
        "fresh_process_decode_pid": json.loads(child.stdout.strip().splitlines()[-1])["pid"],
        "decoded_frames_path": str(decoded_path),
        "decoded_frames_sha256": _sha256_array(decoded),
        "encoder_client_parity": True,
        "psm1_mask_roundtrip_exact": transport_masks_match,
        "decoded_motion_error_px": {
            "max_abs_xyxy": max(motion_errors, default=0),
            "mean_abs_xyxy": float(np.mean(motion_errors)) if motion_errors else 0.0,
            "basis": "serialized bbox metadata versus source observation bboxes",
        },
        "source_paths_passed_to_decoder": [],
        "decoded": decoded,
    }


def io_bytes(payload: bytes):
    import io

    return io.BytesIO(payload)


def _decode_worker(payload_path: Path, output_path: Path) -> int:
    """Byte-only child path. Arguments contain the serialized client payload only."""
    from src.runner.client import reconstruct_serialized_client

    payload = payload_path.read_bytes()
    decoded = reconstruct_serialized_client(payload, require_compressed=True)
    np.save(output_path, np.asarray(decoded, dtype=np.uint8), allow_pickle=False)
    print(json.dumps({"pid": os.getpid(), "payload_sha256": hashlib.sha256(payload).hexdigest()}))
    return 0


def _ssim_rgb(reference: np.ndarray, predicted: np.ndarray) -> float:
    scores: list[float] = []
    c1, c2 = 6.5025, 58.5225
    for ref, pred in zip(reference, predicted, strict=True):
        x = np.asarray(ref, dtype=np.float32)
        y = np.asarray(pred, dtype=np.float32)
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        var_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
        var_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
        cov = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
        score = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
            (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
        )
        scores.append(float(np.mean(score)))
    return float(np.mean(scores)) if scores else 1.0


def _psnr_rgb(reference: np.ndarray, predicted: np.ndarray) -> float:
    mse = float(np.mean((np.asarray(reference, dtype=np.float64) - np.asarray(predicted, dtype=np.float64)) ** 2))
    return float("inf") if mse == 0.0 else 20.0 * float(np.log10(255.0 / np.sqrt(mse)))


def _draw_cross(image: np.ndarray, shape: Any, transform: Any) -> np.ndarray:
    result = np.asarray(image).copy()
    if shape is None or shape.kind != "racket_cross_v1":
        return result
    points = transform.source_to_canvas(np.asarray(shape.points))
    for start, end in ((0, 1), (2, 3)):
        cv2.line(
            result,
            tuple(np.rint(points[start]).astype(int)),
            tuple(np.rint(points[end]).astype(int)),
            (255, 180, 20),
            2,
            cv2.LINE_AA,
        )
    for point in points:
        cv2.circle(result, tuple(np.rint(point).astype(int)), 3, (255, 255, 255), -1)
    return result


def _save_view_manifest(scene: dict[str, Any], result: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for view in result["views"]:
        frame_index = int(view["frame_index"])
        safe_id = hashlib.sha1(str(view["object_id"]).encode("utf-8")).hexdigest()[:12]
        stem = f"{int(scene['frame_start']) + frame_index:06d}_{safe_id}"
        directory = output_dir / "views" / view["role"] / scene["source_id"]
        directory.mkdir(parents=True, exist_ok=True)
        appearance_path = directory / f"{stem}_appearance.png"
        mask_path = directory / f"{stem}_mask.png"
        condition_path = directory / f"{stem}_condition.png"
        cv2.imwrite(str(appearance_path), cv2.cvtColor(view["appearance"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_path), np.asarray(view["mask"], dtype=np.uint8) * 255)
        condition = view["pose"] if view["pose"] is not None else np.asarray(view["mask"])[:, :, None].repeat(3, axis=2) * 255
        cv2.imwrite(str(condition_path), cv2.cvtColor(np.asarray(condition, dtype=np.uint8), cv2.COLOR_RGB2BGR))
        adapters = view["_model_adapters"]
        adapter_directory = output_dir / "adapters" / view["role"] / scene["source_id"]
        adapter_directory.mkdir(parents=True, exist_ok=True)
        adapter_paths: dict[str, str] = {}

        def save_adapter(name: str, image: np.ndarray | None) -> None:
            if image is None:
                return
            path = adapter_directory / f"{stem}_{name}.png"
            array = np.asarray(image, dtype=np.uint8)
            if array.ndim == 2:
                ok = cv2.imwrite(str(path), array)
            else:
                ok = cv2.imwrite(str(path), cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
            if not ok:
                raise OSError(f"could not write model adapter image {path}")
            adapter_paths[name] = str(path)

        save_adapter("animate_anyone_openpose18", adapters.animate_anyone_pose_rgb)
        save_adapter("spade_condition_rgb", adapters.spade_condition_rgb)
        save_adapter("controlnet_condition_rgb", adapters.controlnet_condition_rgb)
        for channel_name, channel in adapters.controlnet_channels.items():
            save_adapter(f"controlnet_{channel_name}", channel)
        quality_eligible = bool(view.get("quality_eligible", False))
        adapter_eligible = bool(view.get("eligible_for_cross_training", view["role"] != "racket"))
        eligible = quality_eligible and adapter_eligible
        reasons = list(view.get("quality_reasons", ()))
        if not adapter_eligible:
            reasons.append(str(view.get("adapter_exclusion_reason") or "racket_cross_fallback_or_missing_geometry"))
        row = {
            "schema": "pointstream.observation-view.v1",
            "sample_id": f"{scene['source_id']}:{int(scene['frame_start']) + frame_index}:{view['object_id']}:{view['role']}",
            "source_id": scene["source_id"],
            "split": scene.get("split", "development_exposed_bp46"),
            "frame_index": int(scene["frame_start"]) + frame_index,
            "object_id": view["object_id"],
            "object_class": view["role"],
            "view": view["role"],
            "appearance_path": str(appearance_path),
            "appearance_sha256": _sha256(appearance_path),
            "mask_path": str(mask_path),
            "mask_sha256": _sha256(mask_path),
            "condition_path": str(condition_path),
            "condition_sha256": _sha256(condition_path),
            "model_adapter_schema": adapters.schema,
            "model_adapter_paths": adapter_paths,
            "model_adapter_sha256": {
                name: _sha256(Path(path)) for name, path in adapter_paths.items()
            },
            "adapter_eligible_for_cross_training": adapters.eligible_for_cross_training,
            "adapter_exclusion_reason": adapters.exclusion_reason,
            "quality_score": float(view.get("quality_score", 0.0)),
            "quality_eligible": quality_eligible,
            "quality_reasons": list(view.get("quality_reasons", ())),
            "transform": view["transform"].to_record(),
            "eligible_for_training": eligible,
            "training_exclusion_reason": None if eligible else ";".join(dict.fromkeys(reasons)),
        }
        rows.append(row)
    return rows


def _panel(frame: np.ndarray, title: str, *, width: int = 240, height: int = 135) -> np.ndarray:
    image = cv2.resize(np.asarray(frame, dtype=np.uint8), (width, height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((height + 24, width, 3), dtype=np.uint8)
    canvas[24:] = image
    cv2.putText(canvas, title[:36], (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 245), 1, cv2.LINE_AA)
    return canvas


def _write_contact_and_preview(
    scene_rows: list[dict[str, Any]],
    run_dir: Path,
) -> list[dict[str, Any]]:
    contact_dir = run_dir / "panels"
    contact_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[np.ndarray] = []
    html_rows: list[str] = []
    preview_frames: list[np.ndarray] = []
    for scene_row in scene_rows:
        source = scene_row["frames"]
        decoded = scene_row["decoded"]
        result = scene_row["result"]
        failed = {int(item["frame_index"]) for item in result["frame_roles"] if item["failed"]}
        selected = sorted({0, len(source) // 2, len(source) - 1, *failed})
        for frame_index in selected:
            player_masks = [
                np.asarray(item["source_mask"])
                for item in result["views"]
                if item["frame_index"] == frame_index and item["role"] == "player" and item.get("quality_eligible")
            ]
            rejected_player_masks = [
                np.asarray(item["source_mask"])
                for item in result["views"]
                if item["frame_index"] == frame_index and item["role"] == "player" and not item.get("quality_eligible")
            ]
            racket_masks = [
                np.asarray(item["source_mask"])
                for item in result["views"]
                if item["frame_index"] == frame_index and item["role"] == "racket" and item.get("quality_eligible")
            ]
            rejected_racket_masks = [
                np.asarray(item["source_mask"])
                for item in result["views"]
                if item["frame_index"] == frame_index and item["role"] == "racket" and not item.get("quality_eligible")
            ]
            player_pose = next(
                (item["pose"] for item in result["views"] if item["frame_index"] == frame_index and item["role"] == "player" and item["pose"] is not None),
                np.zeros((512, 512, 3), dtype=np.uint8),
            )
            appearance = next(
                (item["appearance"] for item in result["views"] if item["frame_index"] == frame_index and item["role"] == "player"),
                np.zeros((512, 512, 3), dtype=np.uint8),
            )
            racket_appearance = next(
                (item["appearance"] for item in result["views"] if item["frame_index"] == frame_index and item["role"] == "racket"),
                np.zeros((512, 512, 3), dtype=np.uint8),
            )
            joint_condition = next(
                (item["pose"] for item in result["views"] if item["frame_index"] == frame_index and item["role"] == "joint"),
                np.zeros((512, 512, 3), dtype=np.uint8),
            )
            panel_images = [
                _panel(source[frame_index], "source frame"),
                _panel(_overlay_mask(source[frame_index], player_masks, (35, 220, 70)), "accepted player masks"),
                _panel(_overlay_mask(source[frame_index], rejected_player_masks, (235, 45, 45)), "rejected player masks"),
                _panel(_overlay_mask(source[frame_index], racket_masks, (240, 130, 35)), "accepted racket masks"),
                _panel(_overlay_mask(source[frame_index], rejected_racket_masks, (235, 45, 45)), "rejected racket masks"),
                _panel(appearance, "blackened player crop"),
                _panel(racket_appearance, "blackened racket crop"),
                _panel(player_pose, "DWPose condition"),
                _panel(joint_condition, "player+racket geometry"),
                _panel(decoded[frame_index], "fresh-process decoded conditioning"),
            ]
            row = np.hstack(panel_images)
            all_rows.append(row)
            basename = f"{scene_row['scene']['source_id']}_{frame_index:03d}.png"
            path = contact_dir / basename
            cv2.imwrite(str(path), cv2.cvtColor(row, cv2.COLOR_RGB2BGR))
            html_rows.append(
                f"<section><h2>{html.escape(scene_row['scene']['source_id'])} frame {frame_index}</h2>"
                f"<img src='panels/{html.escape(basename)}' alt='perception audit panels'></section>"
            )
        for index in range(len(source)):
            joint_condition = next(
                (item["pose"] for item in result["views"] if item["frame_index"] == index and item["role"] == "joint"),
                np.zeros((512, 512, 3), dtype=np.uint8),
            )
            tiles = [
                _panel(source[index], "source"),
                _panel(_overlay_mask(source[index], [np.asarray(item["source_mask"]) for item in result["views"] if item["frame_index"] == index and item["role"] == "player" and item.get("quality_eligible")], (35, 220, 70)), "accepted players"),
                _panel(_overlay_mask(source[index], [np.asarray(item["source_mask"]) for item in result["views"] if item["frame_index"] == index and item["role"] == "racket" and item.get("quality_eligible")], (240, 130, 35)), "accepted rackets"),
                _panel(joint_condition, "joint geometry"),
                _panel(decoded[index], "decoded"),
            ]
            preview_frames.append(np.hstack(tiles))
    if all_rows:
        cv2.imwrite(str(run_dir / "contact_sheet.png"), np.vstack(all_rows))
    (run_dir / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>PointStream SAM3.1 audit</title>"
        "<style>body{font:14px sans-serif;background:#111;color:#eee}img{max-width:100%;height:auto}section{margin:20px 0}</style>"
        + "\n".join(html_rows),
        encoding="utf-8",
    )
    preview_path = run_dir / "sequence_preview.mp4"
    if preview_frames:
        height, width = preview_frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(preview_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            4.0,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not create the contact preview mp4")
        for frame in preview_frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
    return [{"path": str(run_dir / "contact_sheet.png"), "sha256": _sha256(run_dir / "contact_sheet.png")}]


def _write_training_dataset(scene_rows: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    """Write a TennisSkeletonDataset-compatible, quality-gated v2 data tree."""
    from src.components.perception.model_adapters import ModelAdapterViews

    root = run_dir / "train_dataset"
    records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for scene_row in scene_rows:
        scene = scene_row["scene"]
        result = scene_row["result"]
        split = str(scene.get("split", "development_exposed_bp46"))
        base = root / scene["video"] / "segmentations" / scene["scene"]
        for view in result["views"]:
            role = str(view["role"])
            if role not in {"player", "racket"} or not view.get("quality_eligible"):
                continue
            adapters = view.get("_model_adapters")
            if not isinstance(adapters, ModelAdapterViews):
                raise TypeError("training dataset export expected typed model adapter views")
            if role == "racket" and not adapters.eligible_for_cross_training:
                continue
            if role == "player" and view.get("pose") is None:
                continue

            source_frame_index = int(scene["frame_start"]) + int(view["frame_index"])
            stable_key = f"{scene['source_id']}:{role}:{view['object_id']}"
            track_name = "track_sam31_" + hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:16]
            color_dir = base / track_name
            condition_dirs: dict[str, Path] = {}
            if role == "player":
                condition_dirs["pose_body"] = base / f"{track_name}_pose_body"
                condition_dirs["pose_racket"] = base / f"{track_name}_pose_racket"
            else:
                condition_dirs["pose_racket"] = base / f"{track_name}_pose_racket"
            for directory in (color_dir, *condition_dirs.values()):
                directory.mkdir(parents=True, exist_ok=True)

            color_path = color_dir / f"frame_{source_frame_index:06d}.png"
            if not cv2.imwrite(
                str(color_path),
                cv2.cvtColor(np.asarray(view["appearance"], dtype=np.uint8), cv2.COLOR_RGB2BGR),
            ):
                raise OSError(f"could not write SAM3.1 training crop {color_path}")
            condition_paths: dict[str, str] = {}
            local_name = f"frame_{source_frame_index:06d}.png"
            for condition_name, directory in condition_dirs.items():
                if condition_name == "pose_body":
                    condition = np.asarray(view["pose"], dtype=np.uint8)
                else:
                    condition = np.asarray(adapters.spade_condition_rgb, dtype=np.uint8)
                path = directory / local_name
                if not cv2.imwrite(str(path), cv2.cvtColor(condition, cv2.COLOR_RGB2BGR)):
                    raise OSError(f"could not write SAM3.1 training condition {path}")
                condition_paths[condition_name] = str(path)

            record = {
                "schema": "pointstream.sam31-training-sample.v1",
                "sample_id": f"{scene['source_id']}:{source_frame_index}:{role}:{view['object_id']}",
                "source_id": scene["source_id"],
                "split": split,
                "video": scene["video"],
                "scene": scene["scene"],
                "frame_index": source_frame_index,
                "frame_position": int(view["frame_index"]),
                "object_id": str(view["object_id"]),
                "object_class": role,
                "quality_score": float(view["quality_score"]),
                "appearance_path": str(color_path),
                "appearance_sha256": _sha256(color_path),
                "condition_paths": condition_paths,
                "condition_sha256": {name: _sha256(Path(path)) for name, path in condition_paths.items()},
            }
            records.append(record)
            counts[role] += 1

    samples_path = run_dir / "training_samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    return {
        "root": str(root),
        "loader": "src.shared.tennis_dataset.TennisSkeletonDataset",
        "samples_jsonl": str(samples_path),
        "samples_jsonl_sha256": _sha256(samples_path),
        "eligible_samples_by_class": dict(counts),
        "eligible_sample_count": len(records),
        "conditions": {
            "pose_body": "high-confidence retained player poses",
            "pose_racket": "retained player pose or wrist-linked racket mask and geometry",
        },
    }


def _read_existing_track_metadata(dataset_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Inventory the complete previous training tree without editing it."""
    scenes: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    for video_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        segmentations = video_dir / "segmentations"
        if not segmentations.is_dir():
            continue
        for scene_dir in sorted(path for path in segmentations.glob("scene_*") if path.is_dir()):
            tracks: list[dict[str, Any]] = []
            sample_frame_ids: set[int] = set()
            for track_dir in sorted(
                path for path in scene_dir.iterdir()
                if path.is_dir() and re.fullmatch(r"track_\d+", path.name)
            ):
                metadata_path = scene_dir / f"{track_dir.name}_metadata.json"
                if not metadata_path.is_file():
                    raise FileNotFoundError(f"previous track lacks its metadata file: {track_dir}")
                payload = json.loads(metadata_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    payload = payload.get("frames") or payload.get("entries") or []
                if not isinstance(payload, list):
                    raise ValueError(f"expected a metadata row list at {metadata_path}")
                crop_paths = sorted(track_dir.glob("frame_*.png"))
                if len(crop_paths) != len(payload):
                    raise ValueError(
                        f"{track_dir.name}: {len(crop_paths)} crops vs {len(payload)} metadata rows; "
                        "refusing to pair a truncated old track"
                    )
                if not payload:
                    continue
                records: list[dict[str, Any]] = []
                seen_ids: set[int] = set()
                for position, row in enumerate(payload):
                    if not isinstance(row, dict) or row.get("frame_id") is None:
                        raise ValueError(f"{metadata_path} row {position} lacks a frame_id")
                    frame_id = int(row["frame_id"])
                    bbox = row.get("bbox")
                    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                        raise ValueError(f"{metadata_path} row {position} has no xyxy bbox")
                    box = tuple(int(value) for value in bbox)
                    if box[2] <= box[0] or box[3] <= box[1]:
                        raise ValueError(f"{metadata_path} row {position} has an empty player bbox")
                    if frame_id < 0 or frame_id in seen_ids:
                        raise ValueError(f"{metadata_path} has a negative or repeated frame_id {frame_id}")
                    seen_ids.add(frame_id)
                    racket = row.get("racket_bbox_crop")
                    racket_box = None
                    if racket is not None:
                        if not isinstance(racket, (list, tuple)) or len(racket) != 4:
                            raise ValueError(f"{metadata_path} row {position} has a malformed racket bbox")
                        racket_box = tuple(int(value) for value in racket)
                        if racket_box[2] <= racket_box[0] or racket_box[3] <= racket_box[1]:
                            racket_box = None
                    records.append(
                        {
                            "frame_id": frame_id,
                            "bbox": list(box),
                            "racket_bbox_crop": list(racket_box) if racket_box else None,
                            "racket_mask_points": row.get("racket_mask_points")
                            if isinstance(row.get("racket_mask_points"), dict)
                            else None,
                            "metadata_row": position,
                        }
                    )
                    sample_frame_ids.add(frame_id)
                tracks.append(
                    {
                        "object_id": track_dir.name,
                        "track_dir": str(track_dir),
                        "metadata_path": str(metadata_path),
                        "metadata_sha256": _sha256(metadata_path),
                        "crop_paths": [str(path) for path in crop_paths],
                        "records": records,
                        "track_frame_count": len(records),
                        "racket_seed_frame_count": sum(row["racket_bbox_crop"] is not None for row in records),
                    }
                )
            if not tracks:
                continue
            source_id = f"{video_dir.name}_{scene_dir.name}"
            row = {
                "source_id": source_id,
                "video": video_dir.name,
                "scene": scene_dir.name,
                "scene_dir": str(scene_dir),
                "scene_metadata_path": str(video_dir / "scene_metadata.json"),
                "tracks": tracks,
                "sample_frame_ids": sorted(sample_frame_ids),
                "track_count": len(tracks),
                "track_frame_count": sum(track["track_frame_count"] for track in tracks),
                "racket_seed_frame_count": sum(track["racket_seed_frame_count"] for track in tracks),
            }
            scenes.append(row)
            totals["scene_count"] += 1
            totals["track_count"] += len(tracks)
            totals["track_frame_count"] += row["track_frame_count"]
            totals["unique_frame_count"] += len(sample_frame_ids)
            totals["racket_seed_frame_count"] += row["racket_seed_frame_count"]
    if not scenes:
        raise ValueError(f"no previous tracks found under {dataset_root}")
    return scenes, dict(totals)


def _resolve_existing_scene_frames(
    data_root: Path,
    scene: dict[str, Any],
    shard_dir: Path,
) -> tuple[Path, list[Path], dict[str, Any]]:
    relative = Path("clips") / scene["video"] / scene["scene"] / "extract_24"
    for dataset_name in ("bp46-long-scenes", "bp21-headroom"):
        directory = data_root / "outputs" / dataset_name / relative
        if directory.is_dir():
            paths = _sorted_frame_paths(directory)
            return directory, paths, {"source_kind": dataset_name, "source_directory": str(directory)}

    dataset_root = data_root / "assets" / "dataset"
    metadata_path = dataset_root / scene["video"] / "scene_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"no prior frame cache or scene timing metadata for {scene['source_id']}")
    scene_index = int(scene["scene"].removeprefix("scene_"))
    catalog = _read_json(metadata_path)
    rows = catalog.get("scenes") or catalog.get("segments") or []
    if scene_index >= len(rows) or not isinstance(rows[scene_index], dict):
        raise ValueError(f"{scene['source_id']}: scene index is absent from {metadata_path}")
    timing = rows[scene_index]
    start = float(timing.get("t_start", timing.get("start", -1.0)))
    end = float(timing.get("t_end", timing.get("end", -1.0)))
    duration = float(timing.get("duration", end - start))
    raw_video = data_root / "assets" / "raw_4k" / f"{scene['video']}.mp4"
    if start < 0 or duration <= 0 or not raw_video.is_file():
        raise FileNotFoundError(f"{scene['source_id']}: cannot reconstruct missing 24 fps source frames")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise FileNotFoundError("ffmpeg is required to reconstruct missing legacy frame caches")
    directory = shard_dir / "reconstructed_sources" / scene["video"] / scene["scene"] / "extract_24"
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.6f}", "-i", str(raw_video), "-t", f"{duration:.6f}",
        "-r", "24", str(directory / "frame_%06d.png"),
    ]
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=False)
        subprocess.run(command, check=True, timeout=max(600, int(duration * 20)))
    elif not directory.is_dir():
        raise NotADirectoryError(f"reconstructed frame path is not a directory: {directory}")
    paths = _sorted_frame_paths(directory)
    if not paths or max(int(frame_id) for frame_id in scene["sample_frame_ids"]) >= len(paths):
        raise ValueError(f"{scene['source_id']}: reconstructed source frames are incomplete in {directory}")
    version = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True, check=False).stdout.splitlines()[:1]
    return directory, paths, {
        "source_kind": "reconstructed_from_raw_4k",
        "source_directory": str(directory),
        "raw_video": str(raw_video),
        "raw_video_sha256": _sha256(raw_video),
        "scene_timing": {"t_start_seconds": start, "duration_seconds": duration, "scene_metadata_path": str(metadata_path)},
        "ffmpeg_path": ffmpeg,
        "ffmpeg_version": version,
        "ffmpeg_command": command,
    }


def _crop_frame_alignment(
    frame_rgb: np.ndarray,
    crop_rgba: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> float:
    height, width = crop_rgba.shape[:2]
    x0, y0, x1, y1 = bbox
    box_width, box_height = x1 - x0, y1 - y0
    if (box_height, box_width) == (height, width):
        x_stop, y_stop = x1, y1
    elif (box_height + 1, box_width + 1) == (height, width):
        x_stop, y_stop = x1 + 1, y1 + 1
    else:
        x_stop, y_stop = x0 + width, y0 + height
    if x0 < 0 or y0 < 0 or x_stop > frame_rgb.shape[1] or y_stop > frame_rgb.shape[0]:
        raise ValueError(f"old crop bbox {bbox} lies outside the source frame {frame_rgb.shape[1]}x{frame_rgb.shape[0]}")
    patch = frame_rgb[y0:y_stop, x0:x_stop]
    if patch.shape[:2] != (height, width):
        raise ValueError("old crop dimensions do not match its stored xyxy bbox")
    alpha = crop_rgba[:, :, 3] >= 128
    if not np.any(alpha):
        raise ValueError("old track crop has no opaque pixels to verify")
    return float(np.abs(patch.astype(np.int16) - crop_rgba[:, :, :3].astype(np.int16))[alpha].mean())


def _verify_previous_scene_alignment(
    scene: dict[str, Any],
    source_paths: list[Path],
    *,
    maximum_mae: float = 2.0,
) -> tuple[list[dict[str, Any]], tuple[int, int]]:
    """Require old RGBA crops to match the selected full-frame cache at the stored bbox."""
    if cv2 is None:
        raise RuntimeError("OpenCV is required to verify old crop alignment")
    checks: list[dict[str, Any]] = []
    frame_shape: tuple[int, int] | None = None
    for track in scene["tracks"]:
        records = track["records"]
        positions = sorted(set((0, len(records) // 2, len(records) - 1)))
        for position in positions:
            row = records[position]
            frame_id = int(row["frame_id"])
            if frame_id >= len(source_paths):
                raise ValueError(
                    f"{scene['source_id']}/{track['object_id']}: frame_id {frame_id} "
                    f"exceeds its {len(source_paths)}-frame source cache"
                )
            source_path = source_paths[frame_id]
            frame_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            crop_raw = cv2.imread(track["crop_paths"][position], cv2.IMREAD_UNCHANGED)
            if frame_bgr is None or crop_raw is None:
                raise ValueError(f"could not decode crop/source pair at {source_path}")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if crop_raw.ndim == 3 and crop_raw.shape[2] == 4:
                crop_rgba = cv2.cvtColor(crop_raw, cv2.COLOR_BGRA2RGBA)
            elif crop_raw.ndim == 3 and crop_raw.shape[2] == 3:
                rgb = cv2.cvtColor(crop_raw, cv2.COLOR_BGR2RGB)
                crop_rgba = np.dstack((rgb, np.where(np.any(rgb > 8, axis=2), 255, 0).astype(np.uint8)))
            else:
                raise ValueError(f"old crop is not RGB(A): {track['crop_paths'][position]}")
            mae = _crop_frame_alignment(frame_rgb, crop_rgba, tuple(int(v) for v in row["bbox"]))
            if mae > maximum_mae:
                raise ValueError(
                    f"{scene['source_id']}/{track['object_id']} frame {frame_id}: "
                    f"old crop/source MAE {mae:.3f} exceeds {maximum_mae:.3f}"
                )
            frame_shape = tuple(int(value) for value in frame_rgb.shape[:2])
            checks.append(
                {
                    "video": scene["video"], "scene": scene["scene"],
                    "track_id": track["object_id"], "frame_id": frame_id,
                    "crop_path": track["crop_paths"][position],
                    "crop_sha256": _sha256(Path(track["crop_paths"][position])),
                    "source_path": str(source_path), "source_sha256": _sha256(source_path),
                    "opaque_pixel_mae": mae,
                }
            )
    if frame_shape is None:
        raise ValueError(f"{scene['source_id']} has no alignment evidence")
    return checks, frame_shape


def _clip_prompt_box(box: tuple[int, int, int, int], width: int, height: int) -> list[int] | None:
    x0, y0, x1, y1 = box
    clipped = [max(0, min(width, x0)), max(0, min(height, y0)), max(0, min(width, x1)), max(0, min(height, y1))]
    return clipped if clipped[2] > clipped[0] and clipped[3] > clipped[1] else None


def _racket_guidance_point(
    record: Mapping[str, Any],
    racket_box: tuple[int, int, int, int],
) -> tuple[float, float] | None:
    """Translate a legacy racket-mask interior guide point to source coordinates."""
    player_box = record.get("bbox")
    racket_points = record.get("racket_mask_points")
    if not isinstance(player_box, (list, tuple)) or len(player_box) != 4:
        return None
    if not isinstance(racket_points, Mapping):
        return None
    points: list[tuple[float, float]] = []
    for name in ("p1", "p2", "p3", "p4"):
        value = racket_points.get(name)
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            continue
        try:
            point = (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            continue
        if np.isfinite(point).all():
            points.append(point)
    if len(points) < 3:
        return None
    origin_x, origin_y = int(player_box[0]), int(player_box[1])
    point_x = origin_x + float(np.mean([point[0] for point in points]))
    point_y = origin_y + float(np.mean([point[1] for point in points]))
    x0, y0, x1, y1 = racket_box
    if not (x0 <= point_x < x1 and y0 <= point_y < y1):
        return None
    return point_x, point_y


def _build_quality_chunks(
    scene: dict[str, Any],
    source_paths: list[Path],
    frame_shape: tuple[int, int],
    *,
    chunk_size: int = QUALITY_CHUNK_SIZE,
) -> list[dict[str, Any]]:
    height, width = frame_shape
    by_id = {
        track["object_id"]: {int(row["frame_id"]): row for row in track["records"]}
        for track in scene["tracks"]
    }
    groups: dict[int, set[int]] = defaultdict(set)
    for frame_id in scene["sample_frame_ids"]:
        groups[int(frame_id) // chunk_size].add(int(frame_id))
    chunks: list[dict[str, Any]] = []
    for chunk_number, frame_ids in sorted(groups.items()):
        start = chunk_number * chunk_size
        count = min(chunk_size, len(source_paths) - start)
        if count <= 0:
            raise ValueError(f"{scene['source_id']}: empty source chunk at frame {start}")
        target_frames: dict[str, dict[str, list[str]]] = {}
        frame_prompts: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for frame_id in sorted(frame_ids):
            local_frame = str(frame_id - start)
            target_frames[local_frame] = {"player": [], "racket": []}
            frame_prompts[local_frame] = {"player": [], "racket": []}
            for track_id, rows in by_id.items():
                row = rows.get(frame_id)
                if row is None:
                    continue
                target_frames[local_frame]["player"].append(track_id)
                player_box = _clip_prompt_box(
                    tuple(int(value) for value in row["bbox"]), width, height
                )
                if player_box is not None:
                    frame_prompts[local_frame]["player"].append(
                        {
                            "object_id": track_id,
                            "frame_index": frame_id - start,
                            "source_frame_id": frame_id,
                            "bbox": player_box,
                        }
                    )
                if row["racket_bbox_crop"] is not None:
                    target_frames[local_frame]["racket"].append(track_id)
                    local_box = row["racket_bbox_crop"]
                    racket_box = (
                        int(row["bbox"][0]) + int(local_box[0]),
                        int(row["bbox"][1]) + int(local_box[1]),
                        int(row["bbox"][0]) + int(local_box[2]),
                        int(row["bbox"][1]) + int(local_box[3]),
                    )
                    clipped_racket_box = _clip_prompt_box(racket_box, width, height)
                    if clipped_racket_box is not None:
                        racket_prompt: dict[str, Any] = {
                            "object_id": track_id,
                            "frame_index": frame_id - start,
                            "source_frame_id": frame_id,
                            "bbox": clipped_racket_box,
                        }
                        guide_point = _racket_guidance_point(row, racket_box)
                        if guide_point is not None:
                            racket_prompt["point_xy"] = [guide_point[0], guide_point[1]]
                        frame_prompts[local_frame]["racket"].append(racket_prompt)

        chunks.append(
            {
                "source_id": scene["source_id"], "video": scene["video"], "scene": scene["scene"],
                "frame_start": start, "frame_count": count,
                "frame_paths": [str(path) for path in source_paths[start:start + count]],
                "target_frames": target_frames, "frame_prompts": frame_prompts,
            }
        )
    return chunks


def _assign_quality_shards(scenes: list[dict[str, Any]], shard_count: int) -> list[list[dict[str, Any]]]:
    """Deterministically balance scenes by 48-frame work units."""
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    loads = [0] * shard_count
    shards: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    for scene in sorted(scenes, key=lambda item: (-len(item["chunks"]), item["source_id"])):
        index = min(range(shard_count), key=lambda item: (loads[item], item))
        shards[index].append(scene)
        loads[index] += len(scene["chunks"])
    if any(not shard for shard in shards):
        raise ValueError(f"{shard_count} shards exceed the number of active scenes")
    return shards


def _sam_input_dimensions(width: int, height: int, max_side: int) -> tuple[int, int]:
    """Keep the image grid aspect ratio while bounding SAM's spatial workload."""
    if width <= 0 or height <= 0 or max_side <= 0:
        raise ValueError("SAM input dimensions and maximum side must be positive")
    scale = min(1.0, max_side / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def _has_tracking_seed(prompt_rows: Collection[Any], object_id: str) -> bool:
    """Only the requested object can seed tracking; detector extras cannot."""
    return any(
        item.object_id == object_id and item.mask is not None and item.tracker_id is not None
        for item in prompt_rows
    )


def _mask_tracking_point(mask: Any) -> tuple[float, float] | None:
    """Choose a positive click on a foreground pixel to initialize video tracking."""
    if mask is None:
        return None
    ys, xs = np.nonzero(np.asarray(mask) != 0)
    if xs.size == 0:
        return None
    center_x, center_y = float(np.median(xs)), float(np.median(ys))
    nearest = int(np.argmin((xs - center_x) ** 2 + (ys - center_y) ** 2))
    return float(xs[nearest]), float(ys[nearest])


def _tracking_refinement_points(
    seed: Mapping[str, Any],
    target: Any,
    *,
    scale_x: float,
    scale_y: float,
) -> tuple[list[tuple[float, float]], list[int], tuple[float, float] | None, str | None]:
    """Prefer legacy racket clicks and reject a clearly drifting SAM mask center."""
    if target is None or getattr(target, "tracker_id", None) is None:
        return [], [], None, None
    mask_center = _mask_tracking_point(getattr(target, "mask", None))
    guided = seed.get("point_xy")
    guided_point = None
    if isinstance(guided, (list, tuple)) and len(guided) == 2:
        values = (float(guided[0]), float(guided[1]))
        if np.isfinite(values).all():
            guided_point = (values[0] * scale_x, values[1] * scale_y)
    tracking_point = guided_point if guided_point is not None else mask_center
    if tracking_point is None:
        return [], [], None, None

    points = [tracking_point]
    labels = [1]
    negative_point = None
    if guided_point is not None and mask_center is not None:
        mask_center_source = (mask_center[0] / scale_x, mask_center[1] / scale_y)
        seed_box = seed["bbox"]
        center_inside_seed = (
            seed_box[0] <= mask_center_source[0] < seed_box[2]
            and seed_box[1] <= mask_center_source[1] < seed_box[3]
        )
        separation = float(np.hypot(mask_center[0] - guided_point[0], mask_center[1] - guided_point[1]))
        seed_diagonal = float(np.hypot(
            (seed_box[2] - seed_box[0]) * scale_x,
            (seed_box[3] - seed_box[1]) * scale_y,
        ))
        if not center_inside_seed and separation >= max(12.0, seed_diagonal * 0.12):
            negative_point = mask_center
            points.append(negative_point)
            labels.append(0)
    return points, labels, negative_point, "legacy_racket_mask_points" if guided_point is not None else "sam_mask_interior"


def _resize_mask_nearest(mask: Any, width: int, height: int) -> np.ndarray:
    """Restore a binary inference mask without requiring OpenCV in SAM's env."""
    from PIL import Image

    array = np.asarray(mask)
    if array.ndim != 2 or width <= 0 or height <= 0:
        raise ValueError("mask must be 2D and destination dimensions positive")
    if array.shape == (height, width):
        return array
    resized = Image.fromarray(array.astype(np.uint8)).resize(
        (width, height), Image.Resampling.NEAREST
    )
    return np.asarray(resized)


def _safe_propagate_sam3(
    segmenter: Any,
    role: str,
    **kwargs: Any,
) -> tuple[tuple[Any, ...], str | None]:
    """Withhold one role when SAM3.1 rejects its tracking points; surface all other failures."""
    try:
        return tuple(segmenter.propagate(role, **kwargs)), None
    except RuntimeError as exc:
        if str(exc).strip() == "No points are provided; please add points first":
            return (), "sam3_1_tracking_state_has_no_accepted_points"
        raise


def _validated_quality_chunk_summary(
    chunk: dict[str, Any],
    *,
    expected_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Validate a completed chunk before allowing it to satisfy a resumed shard."""
    mask_dir = Path(chunk["mask_dir"])
    metadata_path = mask_dir / "chunk.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"completed chunk metadata is missing: {metadata_path}")
    payload = _read_json(metadata_path)
    expected_fields = {
        "schema": "pointstream.sam31-existing-dataset-chunk.v1",
        "source_id": chunk["source_id"],
        "video": chunk["video"],
        "scene": chunk["scene"],
        "frame_start": int(chunk["frame_start"]),
        "frame_count": int(chunk["frame_count"]),
        "source_frame_size": [int(chunk["frame_width"]), int(chunk["frame_height"])],
    }
    for key, expected in expected_fields.items():
        if payload.get(key) != expected:
            raise ValueError(f"completed SAM3.1 chunk {metadata_path} has a different {key}")
    if payload.get("sam_provenance") != expected_provenance:
        raise ValueError(f"completed SAM3.1 chunk {metadata_path} has different model provenance")
    expected_rows = {
        (role, int(frame_index), str(object_id))
        for frame_index, targets in chunk["target_frames"].items()
        for role in ROLES
        for object_id in targets.get(role, [])
    }
    rows = payload.get("outputs")
    if not isinstance(rows, list):
        raise ValueError(f"completed SAM3.1 chunk {metadata_path} has no output rows")
    observed_rows: set[tuple[str, int, str]] = set()
    referenced_masks: set[Path] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"completed SAM3.1 chunk {metadata_path} has a malformed output row")
        key = (str(row.get("role")), int(row.get("frame_index", -1)), str(row.get("object_id")))
        if key in observed_rows:
            raise ValueError(f"completed SAM3.1 chunk {metadata_path} repeats output {key}")
        observed_rows.add(key)
        mask_value = row.get("mask_path")
        if mask_value:
            mask_path = Path(mask_value).resolve()
            if not mask_path.is_relative_to(mask_dir.resolve()) or not mask_path.is_file():
                raise ValueError(f"completed SAM3.1 mask is missing or outside its chunk: {mask_value}")
            if _sha256(mask_path) != row.get("mask_sha256"):
                raise ValueError(f"completed SAM3.1 mask checksum mismatch: {mask_path}")
            referenced_masks.add(mask_path)
    if observed_rows != expected_rows:
        raise ValueError(f"completed SAM3.1 chunk {metadata_path} does not cover its configured target rows")
    saved_masks = {path.resolve() for path in mask_dir.glob("*.png")}
    if saved_masks != referenced_masks:
        raise ValueError(f"completed SAM3.1 chunk {metadata_path} has unreferenced or missing mask files")
    return {
        "source_id": chunk["source_id"],
        "frame_start": int(chunk["frame_start"]),
        "mask_dir": str(mask_dir),
        "metadata_path": str(metadata_path),
        "metadata_sha256": _sha256(metadata_path),
        "output_count": len(rows),
        "seconds": float(payload.get("chunk_seconds", 0.0)),
        "suppressed_nonlegacy_sam_masks": payload.get("suppressed_nonlegacy_sam_masks", {}),
        "gpu": payload.get("sam_runtime", {}).get("gpu"),
        "gpu_uuid": payload.get("sam_runtime", {}).get("gpu_uuid"),
    }


def _quarantine_incomplete_quality_chunk(mask_dir: Path, shard_dir: Path) -> Path:
    """Move an unfinished chunk aside without deleting partial model output."""
    relative = mask_dir.relative_to(shard_dir)
    destination = shard_dir / "incomplete_chunks" / f"{relative.as_posix().replace('/', '__')}.{time.time_ns()}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(mask_dir, destination)
    return destination


def _sam_quality_dataset_worker(config_path: Path, *, resume: bool = False) -> int:
    """Generate framewise box-guided SAM3.1 masks for one balanced shard."""
    from PIL import Image
    import torch
    from src.components.segmentation.sam31 import Sam31SequenceSegmenter

    config = _read_json(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("the SAM3.1 dataset worker requires CUDA")
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    segmenter = Sam31SequenceSegmenter(
        checkpoint_path=config["sam_checkpoint"],
        source_root=config["sam_source_root"],
        source_revision=config["sam_source_revision"],
        checkpoint_sha256=config["sam_checkpoint_sha256"],
        prob_threshold=float(config["prob_threshold"]),
    )
    model_load_seconds = time.perf_counter() - started
    policy = OFFLINE_POLICY
    provenance = _jsonable_provenance(segmenter.provenance(policy))
    max_input_side = int(config["sam_inference"]["max_input_side_px"])
    chunk_summaries: list[dict[str, Any]] = []
    shard_dir = Path(config["temporary_root"]).parent
    for chunk_index, chunk in enumerate(config["chunks"], start=1):
        chunk_started = time.perf_counter()
        mask_dir = Path(chunk["mask_dir"])
        if resume and mask_dir.exists():
            metadata_path = mask_dir / "chunk.json" if mask_dir.is_dir() else None
            if metadata_path is not None and metadata_path.is_file():
                cached_summary = _validated_quality_chunk_summary(
                    chunk, expected_provenance=provenance
                )
                chunk_summaries.append(cached_summary)
                print(
                    json.dumps({"sam_worker_chunk_resumed": chunk_index, "chunks": len(config["chunks"]),
                                "source_id": chunk["source_id"], "frame_start": chunk["frame_start"]}),
                    flush=True,
                )
                continue
            quarantined = _quarantine_incomplete_quality_chunk(mask_dir, shard_dir)
            print(json.dumps({"sam_worker_incomplete_chunk_preserved": str(quarantined)}), flush=True)
        mask_dir.mkdir(parents=True, exist_ok=False)
        outputs: dict[str, dict[tuple[int, str], Any]] = {"player": {}, "racket": {}}
        prompts_seen: list[dict[str, Any]] = []
        source_width = int(chunk["frame_width"])
        source_height = int(chunk["frame_height"])
        inference_width, inference_height = _sam_input_dimensions(source_width, source_height, max_input_side)
        scale_x = inference_width / source_width
        scale_y = inference_height / source_height
        with tempfile.TemporaryDirectory(prefix="sam31-input-", dir=config["temporary_root"]) as temporary:
            input_dir = Path(temporary)
            for index, source_path in enumerate(chunk["frame_paths"]):
                target = input_dir / f"{index:06d}.png"
                if (inference_width, inference_height) == (source_width, source_height):
                    target.symlink_to(source_path)
                else:
                    with Image.open(source_path) as source_image:
                        source_image.convert("RGB").resize(
                            (inference_width, inference_height), Image.Resampling.LANCZOS
                        ).save(target, format="PNG")
            for role in ROLES:
                role_prompts = [
                    prompt
                    for local_frame in sorted(chunk["frame_prompts"], key=int)
                    for prompt in chunk["frame_prompts"][local_frame][role]
                ]
                if not role_prompts:
                    continue
                session_key = f"chunk{chunk_index}-{role}"
                segmenter.start_session(
                    role, input_dir, frame_width=inference_width,
                    frame_height=inference_height, policy=policy, session_key=session_key,
                )
                try:
                    for seed in role_prompts:
                        bbox = seed["bbox"]
                        inference_bbox = (
                            float(bbox[0]) * scale_x, float(bbox[1]) * scale_y,
                            float(bbox[2]) * scale_x, float(bbox[3]) * scale_y,
                        )
                        prompt_rows = segmenter.add_prompt(
                            role, frame_index=int(seed["frame_index"]), object_id=str(seed["object_id"]),
                            text="tennis player" if role == "player" else "tennis racket",
                            bbox=inference_bbox, session_key=session_key,
                        )
                        for item in prompt_rows:
                            outputs[role][(item.frame_index, item.object_id)] = item
                        target = next((item for item in prompt_rows if item.object_id == seed["object_id"]), None)
                        tracking_points, tracking_labels, negative_point, tracking_point_source = (
                            _tracking_refinement_points(seed, target, scale_x=scale_x, scale_y=scale_y)
                        )
                        tracking_point = tracking_points[0] if tracking_points else None
                        tracking_rows: tuple[Any, ...] = ()
                        if tracking_points:
                            tracking_rows = segmenter.add_prompt(
                                role,
                                frame_index=int(seed["frame_index"]),
                                object_id=str(seed["object_id"]),
                                points=tracking_points,
                                point_labels=tracking_labels,
                                tracker_id=int(target.tracker_id),
                                session_key=session_key,
                            )
                            for item in tracking_rows:
                                key = (item.frame_index, item.object_id)
                                prior = outputs[role].get(key)
                                if prior is None or item.mask is not None:
                                    outputs[role][key] = item
                        prompts_seen.append(
                            {
                                "role": role, "object_id": seed["object_id"],
                                "source_frame_id": seed["source_frame_id"],
                                "frame_index": seed["frame_index"], "bbox_xyxy": seed["bbox"],
                                "returned_status": target.status.value if target else "missing",
                                "sam_score": target.score if target else None,
                                "tracking_point_xy": list(tracking_point) if tracking_point is not None else None,
                                "tracking_point_source": tracking_point_source,
                                "negative_drift_point_xy": list(negative_point) if negative_point is not None else None,
                                "tracker_seeded": _has_tracking_seed(
                                    tracking_rows, str(seed["object_id"])
                                ),
                            }
                        )
                finally:
                    segmenter.close_session(role, session_key=session_key)

            rows: list[dict[str, Any]] = []
            suppressed_extra_masks = Counter()
            for local_frame, role_targets in sorted(chunk["target_frames"].items(), key=lambda item: int(item[0])):
                frame_index = int(local_frame)
                for role in ROLES:
                    object_ids = role_targets.get(role, [])
                    known = {object_id for frame, object_id in outputs[role] if frame == frame_index}
                    suppressed_extra_masks[role] += len(known - set(object_ids))
                    for object_id in object_ids:
                        item = outputs[role].get((frame_index, object_id))
                        if item is None:
                            rows.append(
                                {
                                    "role": role, "frame_index": frame_index, "object_id": object_id,
                                    "tracker_id": None, "score": None, "status": "missing",
                                    "reason": "no_frame_prompt_was_configured",
                                    "mask_path": None, "bbox_xyxy": None,
                                }
                            )
                            continue
                        mask = np.asarray(item.mask) if item.mask is not None else None
                        if mask is not None and mask.shape != (source_height, source_width):
                            mask = _resize_mask_nearest(mask, source_width, source_height)
                        bbox = _mask_bbox(mask) if mask is not None else None
                        mask_path = None
                        mask_hash = None
                        if bbox is not None:
                            x0, y0, x1, y1 = bbox
                            relative = Path(f"{role}_{hashlib.sha1(object_id.encode()).hexdigest()[:12]}_{frame_index:06d}.png")
                            target_path = mask_dir / relative
                            Image.fromarray((np.asarray(mask[y0:y1, x0:x1]) != 0).astype(np.uint8) * 255, mode="L").save(target_path, optimize=True)
                            mask_path = str(target_path)
                            mask_hash = _sha256(target_path)
                        rows.append(
                            {
                                "role": role, "frame_index": frame_index, "object_id": object_id,
                                "tracker_id": item.tracker_id, "score": item.score,
                                "status": item.status.value if bbox is not None else "missing",
                                "reason": item.reason if bbox is not None else (item.reason or "sam_returned_empty_mask"),
                                "mask_path": mask_path, "mask_sha256": mask_hash,
                                "bbox_xyxy": list(bbox) if bbox else None,
                            }
                        )
        metadata = {
            "schema": "pointstream.sam31-existing-dataset-chunk.v1",
            "source_id": chunk["source_id"], "video": chunk["video"], "scene": chunk["scene"],
            "frame_start": chunk["frame_start"], "frame_count": chunk["frame_count"],
            "source_frame_size": [source_width, source_height],
            "sam_input_frame_size": [inference_width, inference_height],
            "sam_input_scale_xy": [scale_x, scale_y], "sam_input_max_side_px": max_input_side,
            "target_frame_count": len(chunk["target_frames"]), "outputs": rows,
            "mask_generation_strategy": "framewise_sam31_box_and_point_prompts_no_propagation",
            "prompts": prompts_seen, "suppressed_nonlegacy_sam_masks": dict(suppressed_extra_masks),
            "propagation_failures": [],
            "sam_provenance": provenance,
            "sam_runtime": {
                "python": sys.executable, "python_version": sys.version,
                "torch_version": torch.__version__, "cuda_runtime_version": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "gpu_memory_bytes": {
                "peak_allocated": int(torch.cuda.max_memory_allocated()),
                "peak_reserved": int(torch.cuda.max_memory_reserved()),
            },
            "model_load_seconds": model_load_seconds,
            "chunk_seconds": time.perf_counter() - chunk_started,
        }
        metadata_path = mask_dir / "chunk.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        chunk_summaries.append(
            {
                "source_id": chunk["source_id"], "frame_start": chunk["frame_start"],
                "mask_dir": str(mask_dir), "metadata_path": str(metadata_path),
                "metadata_sha256": _sha256(metadata_path),
                "output_count": len(rows), "seconds": metadata["chunk_seconds"],
                "suppressed_nonlegacy_sam_masks": dict(suppressed_extra_masks),
                "gpu": metadata["sam_runtime"]["gpu"],
                "gpu_uuid": metadata["sam_runtime"]["gpu_uuid"],
            }
        )
        print(
            json.dumps({"sam_worker_chunk": chunk_index, "chunks": len(config["chunks"]),
                        "source_id": chunk["source_id"], "frame_start": chunk["frame_start"],
                        "seconds": round(metadata["chunk_seconds"], 3)}),
            flush=True,
        )
    worker_result = {
        "schema": "pointstream.sam31-existing-dataset-worker.v1",
        "sam_provenance": provenance, "sam_runtime": {
            "python": sys.executable, "python_version": sys.version,
            "torch_version": torch.__version__, "cuda_runtime_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "model_load_seconds": model_load_seconds,
        "gpu_memory_bytes": {
            "peak_allocated": int(torch.cuda.max_memory_allocated()),
            "peak_reserved": int(torch.cuda.max_memory_reserved()),
        },
        "chunks": chunk_summaries,
        "runtime_devices": [
            {"gpu": gpu, "gpu_uuid": gpu_uuid,
             "chunk_count": sum(item.get("gpu") == gpu and item.get("gpu_uuid") == gpu_uuid for item in chunk_summaries)}
            for gpu, gpu_uuid in sorted({(item.get("gpu"), item.get("gpu_uuid")) for item in chunk_summaries})
        ],
    }
    result_path = Path(config["result_path"])
    result_path.write_text(json.dumps(worker_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"sam_worker_result": str(result_path), "chunks": len(chunk_summaries)}), flush=True)
    return 0


def _load_quality_chunk(chunk: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[tuple[int, str], Any]]]:
    from PIL import Image
    from src.components.segmentation.sam31 import MaskObservation
    from src.contracts.observation import EstimatorProvenance, ObservationStatus

    payload = _read_json(Path(chunk["metadata_path"]))
    if _sha256(Path(chunk["metadata_path"])) != chunk["metadata_sha256"]:
        raise ValueError(f"SAM3.1 chunk metadata checksum mismatch: {chunk['metadata_path']}")
    outputs: dict[str, dict[tuple[int, str], Any]] = {"player": {}, "racket": {}}
    frame_shape = tuple(int(value) for value in chunk["frame_shape"])
    prompt_scores = {
        (str(prompt["role"]), int(prompt["frame_index"]), str(prompt["object_id"])): prompt.get("sam_score")
        for prompt in payload.get("prompts", [])
    }
    for row in payload["outputs"]:
        mask = None
        if row["mask_path"]:
            mask_path = Path(row["mask_path"])
            if _sha256(mask_path) != row["mask_sha256"]:
                raise ValueError(f"SAM3.1 crop mask checksum mismatch: {mask_path}")
            crop = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
            full = np.zeros(frame_shape, dtype=np.uint8)
            x0, y0, x1, y1 = (int(value) for value in row["bbox_xyxy"])
            if crop.shape != (y1 - y0, x1 - x0):
                raise ValueError(f"SAM3.1 cropped mask dimensions disagree with bbox: {mask_path}")
            full[y0:y1, x0:x1] = crop
            mask = full
        item = MaskObservation(
            role=row["role"], object_id=row["object_id"], tracker_id=row["tracker_id"],
            frame_index=int(row["frame_index"]), mask=mask,
            score=prompt_scores.get(
                (str(row["role"]), int(row["frame_index"]), str(row["object_id"])),
                row["score"],
            ),
            status=ObservationStatus(row["status"]), reason=row["reason"],
        )
        outputs[row["role"]][(item.frame_index, item.object_id)] = item
    payload["outputs"] = outputs
    payload["prompts"] = payload.get("prompts", [])
    payload["retries"] = []
    payload["provenance"] = EstimatorProvenance(**payload["sam_provenance"])
    payload["arrays_sha256"] = chunk["metadata_sha256"]
    payload["model_load_seconds"] = float(payload.get("model_load_seconds", 0.0))
    payload["inference_seconds"] = float(payload.get("chunk_seconds", 0.0))
    payload["gpu_memory_bytes"] = payload.get("gpu_memory_bytes", {})
    return payload, outputs


def _write_quality_training_chunk(
    scene: dict[str, Any],
    result: dict[str, Any],
    *,
    output_root: Path,
    samples_stream: Any,
) -> list[dict[str, Any]]:
    from src.components.perception.model_adapters import ModelAdapterViews

    saved: list[dict[str, Any]] = []
    scene_dir = output_root / "train_dataset" / scene["video"] / "segmentations" / scene["scene"]
    quality_mask_root = output_root / "segmentation_masks" / scene["video"] / scene["scene"]
    for view in result["views"]:
        role = str(view["role"])
        if role not in {"player", "racket"} or not view.get("quality_eligible"):
            continue
        adapters = view.get("_model_adapters")
        if not isinstance(adapters, ModelAdapterViews):
            raise TypeError("quality dataset export expected typed model adapter views")
        if role == "racket" and not adapters.eligible_for_cross_training:
            continue
        if role == "player" and view.get("pose") is None:
            continue
        frame_id = int(scene["frame_start"]) + int(view["frame_index"])
        stable_key = f"{scene['source_id']}:{role}:{view['object_id']}"
        track_name = "track_sam31_" + hashlib.sha256(stable_key.encode("utf-8")).hexdigest()[:16]
        color_dir = scene_dir / track_name
        condition_dirs: dict[str, Path] = {}
        if role == "player":
            condition_dirs["pose_body"] = scene_dir / f"{track_name}_pose_body"
            condition_dirs["pose_racket"] = scene_dir / f"{track_name}_pose_racket"
        else:
            condition_dirs["pose_racket"] = scene_dir / f"{track_name}_pose_racket"
        for directory in (color_dir, *condition_dirs.values()):
            directory.mkdir(parents=True, exist_ok=True)
        color_path = color_dir / f"frame_{frame_id:06d}.png"
        if not cv2.imwrite(str(color_path), cv2.cvtColor(np.asarray(view["appearance"], dtype=np.uint8), cv2.COLOR_RGB2BGR)):
            raise OSError(f"could not write quality training crop {color_path}")
        condition_paths: dict[str, str] = {}
        for condition_name, directory in condition_dirs.items():
            image = np.asarray(view["pose"] if condition_name == "pose_body" else adapters.spade_condition_rgb, dtype=np.uint8)
            path = directory / f"frame_{frame_id:06d}.png"
            if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
                raise OSError(f"could not write quality condition {path}")
            condition_paths[condition_name] = str(path)

        source_mask = np.asarray(view["source_mask"], dtype=np.uint8)
        bbox = _mask_bbox(source_mask)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        mask_path = quality_mask_root / track_name / f"frame_{frame_id:06d}.png"
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(mask_path), source_mask[y0:y1, x0:x1] * 255):
            raise OSError(f"could not write quality segmentation mask {mask_path}")
        sample = {
            "schema": "pointstream.sam31-training-sample.v2",
            "sample_id": f"{scene['source_id']}:{frame_id}:{role}:{view['object_id']}",
            "source_id": scene["source_id"], "split": scene.get("split", "development_exposed_existing_train"),
            "video": scene["video"], "scene": scene["scene"], "frame_index": frame_id,
            "object_id": str(view["object_id"]), "legacy_track_id": str(view["object_id"]),
            "object_class": role, "quality_score": float(view["quality_score"]),
            "quality_reasons": list(view.get("quality_reasons", ())),
            "appearance_path": str(color_path), "appearance_sha256": _sha256(color_path),
            "mask_path": str(mask_path), "mask_sha256": _sha256(mask_path),
            "mask_bbox_xyxy": [x0, y0, x1, y1],
            "condition_paths": condition_paths,
            "condition_sha256": {name: _sha256(Path(path)) for name, path in condition_paths.items()},
        }
        samples_stream.write(json.dumps(sample, sort_keys=True, separators=(",", ":")) + "\n")
        saved.append(sample)
    return saved


def _quality_review_row(
    frame: np.ndarray,
    result: dict[str, Any],
    frame_index: int,
    output_path: Path,
) -> dict[str, Any]:
    player_good = [np.asarray(row["source_mask"]) for row in result["views"] if row["frame_index"] == frame_index and row["role"] == "player" and row.get("quality_eligible")]
    player_bad = [np.asarray(row["source_mask"]) for row in result["views"] if row["frame_index"] == frame_index and row["role"] == "player" and not row.get("quality_eligible")]
    racket_good = [np.asarray(row["source_mask"]) for row in result["views"] if row["frame_index"] == frame_index and row["role"] == "racket" and row.get("quality_eligible")]
    racket_bad = [np.asarray(row["source_mask"]) for row in result["views"] if row["frame_index"] == frame_index and row["role"] == "racket" and not row.get("quality_eligible")]
    panels = [
        _panel(frame, f"source frame {frame_index}"),
        _panel(_overlay_mask(frame, player_good, (35, 220, 70)), f"accepted players {len(player_good)}"),
        _panel(_overlay_mask(frame, player_bad, (235, 45, 45)), f"rejected players {len(player_bad)}"),
        _panel(_overlay_mask(frame, racket_good, (240, 130, 35)), f"accepted rackets {len(racket_good)}"),
        _panel(_overlay_mask(frame, racket_bad, (235, 45, 45)), f"rejected rackets {len(racket_bad)}"),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row = cv2.hconcat(panels)
    if not cv2.imwrite(str(output_path), cv2.cvtColor(row, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise OSError(f"could not write SAM3.1 quality review row {output_path}")
    return {
        "path": str(output_path), "sha256": _sha256(output_path), "frame_position": int(frame_index),
        "accepted_player_count": len(player_good), "rejected_player_count": len(player_bad),
        "accepted_racket_count": len(racket_good), "rejected_racket_count": len(racket_bad),
    }


def _finalize_quality_dataset(
    output_root: Path,
    *,
    shard_count: int,
    selection: dict[str, Any],
    selection_manifest_path: Path,
    repo_root: Path,
) -> dict[str, Any] | None:
    """Merge completed shards once, atomically, into one loader-ready dataset index."""
    lock_path = output_root / ".finalize.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        shard_dirs = [output_root / "shards" / f"shard-{index:02d}" for index in range(shard_count)]
        manifests = []
        for directory in shard_dirs:
            path = directory / "shard_manifest.json"
            if not path.is_file():
                return None
            payload = _read_json(path)
            if payload.get("status") != "complete":
                return None
            manifests.append(payload)
        for name in ("training_samples.jsonl", "quality_audit.jsonl"):
            temporary = output_root / f".{name}.tmp"
            sample_ids: set[str] = set()
            with temporary.open("w", encoding="utf-8") as merged:
                for directory in shard_dirs:
                    source = directory / name
                    with source.open(encoding="utf-8") as stream:
                        for line in stream:
                            if not line.strip():
                                continue
                            row = json.loads(line)
                            if name == "training_samples.jsonl":
                                sample_id = str(row["sample_id"])
                                if sample_id in sample_ids:
                                    raise ValueError(f"duplicate accepted training sample {sample_id}")
                                sample_ids.add(sample_id)
                                for field in ("appearance_path", "mask_path", *row.get("condition_paths", {}).values()):
                                    path = Path(field)
                                    if not path.is_file() or path.stat().st_size == 0:
                                        raise FileNotFoundError(f"training sample artifact is missing: {path}")
                            merged.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            temporary.replace(output_root / name)

        combined_samples = output_root / "training_samples.jsonl"
        class_counts: Counter[str] = Counter()
        with combined_samples.open(encoding="utf-8") as stream:
            for line in stream:
                class_counts[str(json.loads(line)["object_class"])] += 1
        review_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
        scene_reviews: list[dict[str, Any]] = []
        for manifest in manifests:
            for review in manifest.get("scene_reviews", []):
                scene_reviews.append(review)
                video = str(review["video"])
                rows = review.get("rows", [])
                with_racket = [row for row in rows if row.get("accepted_racket_count", 0) or row.get("rejected_racket_count", 0)]
                if with_racket:
                    best_accept = max(with_racket, key=lambda row: (row.get("accepted_racket_count", 0), -row.get("rejected_racket_count", 0)))
                    best_reject = max(with_racket, key=lambda row: (row.get("rejected_racket_count", 0), -row.get("accepted_racket_count", 0)))
                    review_candidates[video].append(best_accept)
                    if best_reject["path"] != best_accept["path"]:
                        review_candidates[video].append(best_reject)
                elif rows:
                    review_candidates[video].append(rows[0])
        chosen_rows: list[dict[str, Any]] = []
        for video in sorted(review_candidates):
            seen: set[str] = set()
            for row in review_candidates[video]:
                if row["path"] not in seen:
                    chosen_rows.append(row)
                    seen.add(row["path"])
                if len(seen) >= 2:
                    break
        review_dir = output_root / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        contact_path = review_dir / "summary_contact_sheet.jpg"
        contact_rows = [cv2.imread(row["path"], cv2.IMREAD_COLOR) for row in chosen_rows]
        contact_rows = [row for row in contact_rows if row is not None]
        if contact_rows:
            if not cv2.imwrite(str(contact_path), cv2.vconcat(contact_rows), [cv2.IMWRITE_JPEG_QUALITY, 88]):
                raise OSError(f"could not write summary contact sheet {contact_path}")
        scene_links = []
        for review in sorted(scene_reviews, key=lambda row: (row["video"], row["scene"])):
            scene_contact = Path(review["contact_sheet"])
            scene_links.append(
                f"<li><a href='../{html.escape(str(scene_contact.relative_to(output_root)))}'>"
                f"{html.escape(review['video'])}/{html.escape(review['scene'])}</a> "
                f"({review['chunk_count']} chunks, {review['accepted_by_class']})</li>"
            )
        review_index = review_dir / "index.html"
        review_index.write_text(
            "<!doctype html><meta charset='utf-8'><title>SAM3.1 quality review</title>"
            "<h1>SAM3.1 existing training set: quality-gated labels</h1><ul>"
            + "\n".join(scene_links) + "</ul>",
            encoding="utf-8",
        )
        source_rows: dict[str, Any] = {}
        metadata_hashes: dict[str, str] = {}
        for manifest in manifests:
            for item in manifest.get("source_scenes", []):
                source_rows[item["source_id"]] = item
            metadata_hashes.update(manifest.get("track_metadata_sha256", {}))
        try:
            selection_manifest_ref = selection_manifest_path.resolve().relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            selection_manifest_ref = str(selection_manifest_path.resolve())
        manifest = {
            "schema": "pointstream.sam31-quality-dataset.v3",
            "version": selection["version"], "status": "complete",
            "active": False,
            "split": selection["split"],
            "selection_manifest": selection_manifest_ref,
            "selection_manifest_sha256": _sha256(selection_manifest_path),
            "source_dataset": str(manifests[0]["source_dataset"]),
            "source_dataset_scene_count": sum(int(row["scene_count"]) for row in manifests),
            "source_dataset_track_count": sum(int(row["track_count"]) for row in manifests),
            "source_dataset_track_frame_count": sum(int(row["track_frame_count"]) for row in manifests),
            "source_dataset_unique_frame_count": sum(int(row["unique_frame_count"]) for row in manifests),
            "source_track_metadata_sha256": metadata_hashes,
            "source_scenes": sorted(source_rows.values(), key=lambda row: row["source_id"]),
            "sam_inference": selection["sam_inference"],
            "quality_policy": selection["quality_policy"],
            "sampling_policy": selection.get("sampling_policy", selection["selection_policy"]),
            "estimators": {
                "sam31": manifests[0]["sam31"],
                "dwpose": manifests[0]["dwpose"],
            },
            "shards": [
                {
                    "shard_index": row["shard_index"], "shard_count": row["shard_count"],
                    "host": row["sam_runtime"].get("host"), "gpu": row["sam_runtime"].get("gpu"),
                    "scene_count": row["scene_count"], "chunk_count": row["chunk_count"],
                    "accepted_by_class": row["accepted_by_class"],
                    "rejected_by_class": row["rejected_by_class"],
                    "elapsed_seconds": row["elapsed_seconds"],
                }
                for row in manifests
            ],
            "training_export": {
                "root": str(output_root / "train_dataset"),
                "loader": "src.shared.tennis_dataset.TennisSkeletonDataset",
                "samples_jsonl": str(combined_samples),
                "samples_jsonl_sha256": _sha256(combined_samples),
                "eligible_sample_count": sum(class_counts.values()),
                "eligible_samples_by_class": dict(class_counts),
                "conditions": {
                    "pose_body": "retained high-confidence player masks with DWPose body poses",
                    "pose_racket": "retained player or wrist-associated racket masks with geometry",
                },
            },
            "quality_audit": {
                "jsonl": str(output_root / "quality_audit.jsonl"),
                "sha256": _sha256(output_root / "quality_audit.jsonl"),
                "contact_sheet": str(contact_path) if contact_rows else None,
                "review_index": str(review_index),
            },
            "promotion": {
                "eligible": False,
                "reason": "existing training data refreshed with SAM3.1 pseudo-labels and conservative automatic gates; visual/manual review remains required",
            },
        }
        manifest_path = output_root / "dataset_manifest.json"
        manifest_tmp = manifest_path.with_suffix(".json.tmp")
        manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest_tmp.replace(manifest_path)
        return manifest


def _prepare_quality_build_root(
    output_root: Path,
    *,
    plan: dict[str, Any],
) -> None:
    """Create or validate a shared output root for independent dataset shards."""
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".build_plan.lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        manifest_path = output_root / "dataset_manifest.json"
        if manifest_path.is_file():
            raise FileExistsError(f"refusing to overwrite completed quality dataset: {output_root}")
        plan_path = output_root / "build_plan.json"
        if plan_path.is_file():
            existing = _read_json(plan_path)
            if existing != plan:
                raise ValueError(f"quality-dataset shard plan does not match existing output root: {output_root}")
            return
        temporary = plan_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(plan_path)


def build_existing_quality_dataset(args: argparse.Namespace, *, repo_root: Path) -> dict[str, Any]:
    """Refresh every non-empty old track from its source video with SAM3.1 labels."""
    from src.components.pose.dwpose import DWPoseEstimator

    output_root = _external_root(Path(args.output_dir), repo_root)
    data_root = _external_root(Path(args.data_root), repo_root)
    manifest_path = (repo_root / args.quality_dataset_manifest).resolve()
    selection = _read_json(manifest_path)
    if selection.get("schema") != EXISTING_DATASET_SCHEMA:
        raise ValueError(f"unsupported existing-dataset selection schema {selection.get('schema')!r}")
    if int(selection.get("chunk_size_frames", 0)) != QUALITY_CHUNK_SIZE or int(selection.get("working_fps", 0)) != 24:
        raise ValueError("the existing-dataset manifest must pin the 48-frame, 24 fps source grid")
    sam_inference = selection.get("sam_inference")
    if not isinstance(sam_inference, dict) or int(sam_inference.get("max_input_side_px", 0)) <= 0:
        raise ValueError("the existing-dataset manifest must pin a positive SAM input maximum side")
    if int(args.shard_count) <= 0 or not 0 <= int(args.shard_index) < int(args.shard_count):
        raise ValueError("shard index must be within the positive shard count")
    if int(args.smoke_chunks) < 0:
        raise ValueError("smoke chunk count cannot be negative")
    dataset_root = data_root / str(selection["source_dataset"])
    all_scenes, totals = _read_existing_track_metadata(dataset_root)
    for scene in all_scenes:
        scene["chunks"] = [None] * len({int(frame_id) // QUALITY_CHUNK_SIZE for frame_id in scene["sample_frame_ids"]})
    shards = _assign_quality_shards(all_scenes, int(args.shard_count))
    selected_scenes = shards[int(args.shard_index)]
    if args.smoke_chunks:
        remaining = int(args.smoke_chunks)
        smoke_scenes = []
        for scene in selected_scenes:
            planned_count = len(scene["chunks"])
            if remaining <= 0:
                break
            scene["smoke_chunk_limit"] = min(remaining, planned_count)
            smoke_scenes.append(scene)
            remaining -= scene["smoke_chunk_limit"]
        if remaining > 0:
            raise ValueError(
                f"requested {args.smoke_chunks} smoke chunks but shard {args.shard_index} has only "
                f"{int(args.smoke_chunks) - remaining}"
            )
        selected_scenes = smoke_scenes
    if not selected_scenes:
        raise ValueError(f"shard {args.shard_index} has no existing scenes")
    sam_checkpoint = Path(args.sam_checkpoint).expanduser().resolve()
    dwpose_detector = Path(args.dwpose_detector).expanduser().resolve()
    dwpose_pose = Path(args.dwpose_pose).expanduser().resolve()
    missing = [str(path) for path in (sam_checkpoint, dwpose_detector, dwpose_pose) if not path.is_file()]
    if missing:
        raise FileNotFoundError("required model artifacts are missing: " + ", ".join(missing))
    if not args.sam_checkpoint_sha256:
        raise ValueError("SAM3.1 checkpoint SHA-256 must be pinned")
    revision = subprocess.run(
        ["git", "-C", str(args.sam_source_root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True, timeout=10,
    ).stdout.strip()
    if revision != args.sam_source_revision:
        raise ValueError(f"SAM3.1 source revision mismatch: expected {args.sam_source_revision}, found {revision}")

    build_plan = {
        "schema": "pointstream.sam31-existing-training-build-plan.v1",
        "selection_sha256": _sha256(manifest_path),
        "source_dataset": str(selection["source_dataset"]),
        "shard_count": int(args.shard_count),
        "smoke_chunks": int(args.smoke_chunks),
        "sam_checkpoint_sha256": args.sam_checkpoint_sha256,
        "sam_source_revision": revision,
        "sam_inference": sam_inference,
        "dwpose_detector_sha256": _sha256(dwpose_detector),
        "dwpose_pose_sha256": _sha256(dwpose_pose),
        "dwpose_device": args.dwpose_device,
    }
    _prepare_quality_build_root(output_root, plan=build_plan)

    shard_dir = output_root / "shards" / f"shard-{int(args.shard_index):02d}"
    if args.resume_sam_output:
        if not shard_dir.is_dir():
            raise FileNotFoundError(f"quality-dataset shard to resume does not exist: {shard_dir}")
    else:
        shard_dir.mkdir(parents=True, exist_ok=False)
    for scene in selected_scenes:
        directory, source_paths, source_provenance = _resolve_existing_scene_frames(data_root, scene, shard_dir)
        if max(scene["sample_frame_ids"]) >= len(source_paths):
            raise ValueError(
                f"{scene['source_id']}: old frame id {max(scene['sample_frame_ids'])} "
                f"exceeds {len(source_paths)} source frames"
            )
        scene["source_directory"] = str(directory)
        scene["source_frame_count"] = len(source_paths)
        scene["source_provenance"] = source_provenance
        scene["source_paths"] = source_paths
        scene["alignment_checks"], scene["frame_shape"] = _verify_previous_scene_alignment(scene, source_paths)
        scene["chunks"] = _build_quality_chunks(scene, source_paths, scene["frame_shape"])
        if args.smoke_chunks:
            scene["chunks"] = scene["chunks"][: int(scene["smoke_chunk_limit"])]
    temporary_root = shard_dir / "temporary"
    temporary_root.mkdir(parents=True, exist_ok=bool(args.resume_sam_output))
    quality_policy = dict(selection["quality_policy"])
    selected_chunks: list[dict[str, Any]] = []
    track_metadata_sha256: dict[str, str] = {}
    source_scene_inputs: list[dict[str, Any]] = []
    for scene in selected_scenes:
        track_metadata_sha256.update({track["metadata_path"]: track["metadata_sha256"] for track in scene["tracks"]})
        scene_chunks = []
        for chunk in scene["chunks"]:
            item = dict(chunk)
            item["frame_width"] = int(scene["frame_shape"][1])
            item["frame_height"] = int(scene["frame_shape"][0])
            item["mask_dir"] = str(shard_dir / "sam_masks" / scene["video"] / scene["scene"] / f"chunk_{chunk['frame_start']:06d}")
            scene_chunks.append(item)
        selected_chunks.extend(scene_chunks)
        source_scene_inputs.append(
            {
                "source_id": scene["source_id"], "video": scene["video"], "scene": scene["scene"],
                "source_frame_count": scene["source_frame_count"],
                "source_provenance": scene["source_provenance"],
                "track_count": scene["track_count"], "track_frame_count": scene["track_frame_count"],
                "unique_frame_count": len(scene["sample_frame_ids"]),
                "racket_seed_frame_count": scene["racket_seed_frame_count"],
                "alignment_checks": scene["alignment_checks"],
                "chunks": len(scene["chunks"]),
            }
        )

    sam_config_path = shard_dir / "sam_worker_config.json"
    sam_result_path = shard_dir / "sam_worker_result.json"
    sam_config = {
        "sam_checkpoint": str(sam_checkpoint), "sam_source_root": str(Path(args.sam_source_root).resolve()),
        "sam_source_revision": revision, "sam_checkpoint_sha256": args.sam_checkpoint_sha256,
        "prob_threshold": float(args.prob_threshold), "sam_inference": sam_inference,
        "temporary_root": str(temporary_root),
        "result_path": str(sam_result_path), "chunks": selected_chunks,
    }
    if args.resume_sam_output:
        if not sam_config_path.is_file():
            raise FileNotFoundError(f"SAM3.1 shard has no saved worker configuration: {sam_config_path}")
        if _read_json(sam_config_path) != sam_config:
            raise ValueError("saved SAM3.1 worker configuration differs from this shard plan")
        sam_result = _read_json(sam_result_path) if sam_result_path.is_file() else None
        if sam_result is not None:
            saved_provenance = sam_result.get("sam_provenance", {})
            if saved_provenance.get("checkpoint_sha256") != args.sam_checkpoint_sha256:
                raise ValueError("completed SAM3.1 worker result has a different checkpoint")
            if len(sam_result.get("chunks", [])) != len(selected_chunks):
                raise ValueError("completed SAM3.1 worker result does not match the selected chunk count")
            validated_summaries = [
                _validated_quality_chunk_summary(chunk, expected_provenance=saved_provenance)
                for chunk in selected_chunks
            ]
            sam_result["chunks"] = validated_summaries
        else:
            worker_command = [
                str(Path(args.sam_python).expanduser().resolve()),
                "-m", "scripts.audit_dataset_pipeline", "--sam-quality-worker", str(sam_config_path),
                "--resume-sam-output",
            ]
            worker = subprocess.Popen(worker_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if worker.stdout is None:
                raise RuntimeError("could not attach to the SAM3.1 worker log")
            for line in worker.stdout:
                print("sam31-worker: " + line.rstrip(), flush=True)
            return_code = worker.wait()
            if return_code:
                raise RuntimeError(f"SAM3.1 dataset worker exited with status {return_code}")
            sam_result = _read_json(sam_result_path)
    else:
        sam_config_path.write_text(json.dumps(sam_config, separators=(",", ":")) + "\n", encoding="utf-8")
        worker_command = [
            str(Path(args.sam_python).expanduser().resolve()),
            "-m", "scripts.audit_dataset_pipeline", "--sam-quality-worker", str(sam_config_path),
        ]
        worker = subprocess.Popen(worker_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if worker.stdout is None:
            raise RuntimeError("could not attach to the SAM3.1 worker log")
        for line in worker.stdout:
            print("sam31-worker: " + line.rstrip(), flush=True)
        return_code = worker.wait()
        if return_code:
            raise RuntimeError(f"SAM3.1 dataset worker exited with status {return_code}")
        sam_result = _read_json(sam_result_path)
    if sam_result.get("sam_provenance", {}).get("checkpoint_sha256") != args.sam_checkpoint_sha256:
        raise ValueError("completed SAM3.1 output checkpoint SHA-256 does not match the requested pin")
    del sam_config, selected_chunks
    gc.collect()

    os.environ["PS_DWPOSE_DET"] = str(dwpose_detector)
    os.environ["PS_DWPOSE_POSE"] = str(dwpose_pose)
    pose_estimator = DWPoseEstimator(device=args.dwpose_device)
    sam_chunk_index = {
        (chunk["source_id"], int(chunk["frame_start"])): chunk
        for chunk in sam_result["chunks"]
    }
    accepted_by_class: Counter[str] = Counter()
    rejected_by_class: Counter[str] = Counter()
    scene_reviews: list[dict[str, Any]] = []
    input_frames_path = shard_dir / "source_frames.jsonl"
    samples_path = shard_dir / "training_samples.jsonl"
    audit_path = shard_dir / "quality_audit.jsonl"
    start_time = time.perf_counter()
    with input_frames_path.open("w", encoding="utf-8") as input_stream, samples_path.open("w", encoding="utf-8") as samples_stream, audit_path.open("w", encoding="utf-8") as audit_stream:
        for scene in selected_scenes:
            review_rows: list[dict[str, Any]] = []
            scene_accepted: Counter[str] = Counter()
            for chunk in scene["chunks"]:
                chunk_meta = sam_chunk_index[(scene["source_id"], int(chunk["frame_start"]))]
                frame_records: list[dict[str, Any]] = []
                frames_list: list[np.ndarray] = []
                for local_index in range(int(chunk["frame_count"])):
                    frame_id = int(chunk["frame_start"]) + local_index
                    source_path = scene["source_paths"][frame_id]
                    frame_bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
                    if frame_bgr is None:
                        raise ValueError(f"could not decode source frame {source_path}")
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frames_list.append(frame_rgb)
                    frame_record = {
                        "source_id": scene["source_id"], "video": scene["video"], "scene": scene["scene"],
                        "frame_id": frame_id, "source_frame_file_id": int(source_path.stem.removeprefix("frame_")),
                        "source_path": str(source_path), "source_file_sha256": _sha256(source_path),
                    }
                    frame_records.append(frame_record)
                    input_stream.write(json.dumps(frame_record, sort_keys=True, separators=(",", ":")) + "\n")
                frames = np.stack(frames_list)
                scene_chunk = {
                    "source_id": scene["source_id"], "video": scene["video"], "scene": scene["scene"],
                    "split": selection["split"], "frame_start": int(chunk["frame_start"]),
                    "frame_count": int(chunk["frame_count"]),
                }
                chunk_payload, _raw_outputs = _load_quality_chunk(
                    {**chunk_meta, "frame_shape": list(frames.shape[1:3])}
                )
                chunk_result = _sequence_perception(
                    scene_chunk, frames, sam_result=chunk_payload,
                    pose_estimator=pose_estimator, quality_policy=quality_policy,
                )
                saved = _write_quality_training_chunk(
                    scene_chunk, chunk_result, output_root=output_root,
                    samples_stream=samples_stream,
                )
                accepted_by_class.update(sample["object_class"] for sample in saved)
                scene_accepted.update(sample["object_class"] for sample in saved)
                accepted_keys = {(sample["frame_index"], sample["legacy_track_id"], sample["object_class"]) for sample in saved}
                for observation in chunk_result["observations"]:
                    role = "player" if observation["object_class"] == "player" else "racket"
                    frame_id = int(observation["frame_index"])
                    local_index = frame_id - int(chunk["frame_start"])
                    targets = chunk["target_frames"].get(str(local_index), {}).get(role, [])
                    if observation["object_id"] not in targets:
                        continue
                    row = {
                        "source_id": scene["source_id"], "video": scene["video"], "scene": scene["scene"],
                        "frame_id": frame_id, "object_id": observation["object_id"], "object_class": role,
                        "status": observation["status"], "quality_score": observation["quality"]["score"],
                        "eligible": bool(observation["training_eligible"]),
                        "quality_reasons": observation["quality"]["reasons"],
                        "quality_features": observation["quality"]["features"],
                        "training_sample_emitted": (frame_id, observation["object_id"], role) in accepted_keys,
                    }
                    if not row["eligible"]:
                        rejected_by_class[role] += 1
                    audit_stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                target_local = sorted(int(value) for value in chunk["target_frames"])
                best_local = max(
                    target_local,
                    key=lambda frame_index: (
                        sum(1 for view in chunk_result["views"] if view["frame_index"] == frame_index and view["role"] == "racket" and view.get("quality_eligible")),
                        sum(1 for view in chunk_result["views"] if view["frame_index"] == frame_index and view["role"] == "racket"),
                        sum(1 for view in chunk_result["views"] if view["frame_index"] == frame_index and view["role"] == "player" and view.get("quality_eligible")),
                    ),
                )
                review_path = shard_dir / "review" / scene["video"] / scene["scene"] / f"chunk_{int(chunk['frame_start']):06d}.jpg"
                review_row = _quality_review_row(frames[best_local], chunk_result, best_local, review_path)
                review_row["chunk_start"] = int(chunk["frame_start"])
                review_rows.append(review_row)
                del frames, frames_list, frame_records, chunk_result, chunk_payload, _raw_outputs
                gc.collect()

            scene_review_path = shard_dir / "review" / scene["video"] / scene["scene"] / "contact_sheet.jpg"
            scene_review_path.parent.mkdir(parents=True, exist_ok=True)
            contact_images = [cv2.imread(row["path"], cv2.IMREAD_COLOR) for row in review_rows]
            contact_images = [image for image in contact_images if image is not None]
            if contact_images and not cv2.imwrite(str(scene_review_path), cv2.vconcat(contact_images), [cv2.IMWRITE_JPEG_QUALITY, 88]):
                raise OSError(f"could not write scene contact sheet {scene_review_path}")
            scene_report = {
                "video": scene["video"], "scene": scene["scene"], "source_id": scene["source_id"],
                "contact_sheet": str(scene_review_path), "chunk_count": len(scene["chunks"]),
                "accepted_by_class": dict(scene_accepted),
                "rows": review_rows,
            }
            (scene_review_path.parent / "scene_quality.json").write_text(
                json.dumps(scene_report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            scene_reviews.append(scene_report)
            print(json.dumps({"quality_scene_complete": scene["source_id"], "chunks": len(scene["chunks"]), "accepted": dict(scene_accepted)}), flush=True)

    pose_provenance = _jsonable_provenance(pose_estimator.provenance(policy=OFFLINE_POLICY))
    torch = __import__("torch")
    dispatch_path = repo_root / "dispatch.json"
    dispatch_record = _read_json(dispatch_path) if dispatch_path.is_file() else {}
    dispatch_provenance = {
        key: dispatch_record.get(key)
        for key in (
            "job_id", "host_alias", "host", "gpu_uuid", "snapshot_sha256", "git_head",
            "tracked_worktree_patch_sha256", "tracked_changes_included", "included_untracked_sha256",
            "runtime_environment", "selection",
        )
    }
    checkpoint_hashes = {
        "sam31_checkpoint": _sha256(sam_checkpoint),
        "dwpose_detector": _sha256(dwpose_detector),
        "dwpose_pose": _sha256(dwpose_pose),
    }
    shard_manifest = {
        "schema": "pointstream.sam31-quality-dataset-shard.v2",
        "status": "partial_smoke" if args.smoke_chunks else "complete", "version": selection["version"],
        "shard_index": int(args.shard_index), "shard_count": int(args.shard_count),
        "smoke_chunks": int(args.smoke_chunks),
        "host": subprocess.run(["hostname", "-f"], capture_output=True, text=True, check=False).stdout.strip(),
        "source_dataset": str(dataset_root),
        "scene_count": len(selected_scenes), "track_count": sum(scene["track_count"] for scene in selected_scenes),
        "track_frame_count": sum(scene["track_frame_count"] for scene in selected_scenes),
        "unique_frame_count": sum(len(scene["sample_frame_ids"]) for scene in selected_scenes),
        "chunk_count": sum(len(scene["chunks"]) for scene in selected_scenes),
        "accepted_by_class": dict(accepted_by_class), "rejected_by_class": dict(rejected_by_class),
        "track_metadata_sha256": track_metadata_sha256,
        "source_scenes": source_scene_inputs,
        "source_frame_manifest": str(input_frames_path), "source_frame_manifest_sha256": _sha256(input_frames_path),
        "training_samples": str(samples_path), "training_samples_sha256": _sha256(samples_path),
        "quality_audit": str(audit_path), "quality_audit_sha256": _sha256(audit_path),
        "scene_reviews": scene_reviews,
        "sam31": {
            "source_revision": revision, "source_root": str(Path(args.sam_source_root).resolve()),
            "python": sam_result["sam_runtime"]["python"], "checkpoint_path": str(sam_checkpoint),
            "checkpoint_sha256": checkpoint_hashes["sam31_checkpoint"],
            "provenance": sam_result["sam_provenance"],
            "runtime": sam_result["sam_runtime"], "gpu_memory_bytes": sam_result["gpu_memory_bytes"],
        },
        "sam_inference": selection["sam_inference"],
        "dwpose": {
            "detector_path": str(dwpose_detector), "detector_sha256": checkpoint_hashes["dwpose_detector"],
            "pose_path": str(dwpose_pose), "pose_sha256": checkpoint_hashes["dwpose_pose"],
            "provenance": pose_provenance,
        },
        "quality_policy": quality_policy,
        "compute_runtime": {
            "python": sys.executable, "python_version": sys.version,
            "torch_version": torch.__version__, "cuda_runtime_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "command_argv": sys.argv,
        },
        "dispatch_provenance": dispatch_provenance,
        "timings_seconds": {"sam_model_load": sam_result["model_load_seconds"], "sam_chunks": [row["seconds"] for row in sam_result["chunks"]], "parent_quality_and_export": time.perf_counter() - start_time},
        "elapsed_seconds": time.perf_counter() - start_time,
    }
    shard_manifest_path = shard_dir / "shard_manifest.json"
    shard_manifest_path.write_text(json.dumps(shard_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.smoke_chunks:
        review_dir = output_root / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        contact_path = review_dir / "smoke_contact_sheet.jpg"
        review_paths = [row["path"] for scene in scene_reviews for row in scene["rows"]]
        contact_rows = [cv2.imread(path, cv2.IMREAD_COLOR) for path in review_paths]
        contact_rows = [row for row in contact_rows if row is not None]
        if contact_rows and not cv2.imwrite(str(contact_path), cv2.vconcat(contact_rows), [cv2.IMWRITE_JPEG_QUALITY, 88]):
            raise OSError(f"could not write smoke contact sheet {contact_path}")
        smoke_manifest = {
            "schema": "pointstream.sam31-quality-smoke.v1",
            "status": "partial_smoke_not_a_complete_dataset",
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "selection_manifest": str(manifest_path),
            "selection_manifest_sha256": _sha256(manifest_path),
            "source_dataset": str(dataset_root),
            "old_dataset_inventory": totals,
            "processed_shard": {"index": int(args.shard_index), "planned_count": int(args.shard_count)},
            "processed_chunk_count": int(args.smoke_chunks),
            "accepted_by_class": dict(accepted_by_class),
            "rejected_by_class": dict(rejected_by_class),
            "sam31": shard_manifest["sam31"],
            "dwpose": shard_manifest["dwpose"],
            "shard_manifest": str(shard_manifest_path),
            "training_samples": str(samples_path),
            "training_samples_sha256": _sha256(samples_path),
            "quality_audit": str(audit_path),
            "quality_audit_sha256": _sha256(audit_path),
            "review_contact_sheet": str(contact_path) if contact_rows else None,
        }
        smoke_path = output_root / "smoke_manifest.json"
        smoke_path.write_text(json.dumps(smoke_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"status": smoke_manifest["status"], "smoke_manifest": str(smoke_path), "contact_sheet": smoke_manifest["review_contact_sheet"], "totals": totals}
    merged = _finalize_quality_dataset(
        output_root,
        shard_count=int(args.shard_count),
        selection=selection,
        selection_manifest_path=manifest_path,
        repo_root=repo_root,
    )
    return {"shard_manifest": str(shard_manifest_path), "status": "complete", "dataset_manifest": str(output_root / "dataset_manifest.json") if merged else None, "totals": totals}


def run_audit(args: argparse.Namespace, *, repo_root: Path) -> dict[str, Any]:
    from src.components.pose.dwpose import DWPoseEstimator

    preflight = inspect_inputs(args, repo_root=repo_root)
    if args.sam_source_revision and preflight["sam31_source_revision"] != args.sam_source_revision:
        raise ValueError("SAM3.1 source revision was not pinned to the inspected checkout")
    if not args.sam_checkpoint_sha256:
        raise ValueError("the pilot requires an explicit --sam-checkpoint-sha256 pin")
    data_root = Path(preflight["data_root"])
    run_dir = _external_root(Path(args.output_dir), repo_root)
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing audit output: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
    dispatch_path = repo_root / "dispatch.json"
    dispatch_record = _read_json(dispatch_path) if dispatch_path.is_file() else {}
    dispatch_provenance = {
        key: dispatch_record.get(key)
        for key in (
            "job_id",
            "host_alias",
            "host",
            "gpu_uuid",
            "snapshot_sha256",
            "git_head",
            "tracked_worktree_patch_sha256",
            "tracked_changes_included",
            "included_untracked_sha256",
            "runtime_environment",
            "selection",
        )
    }
    os.environ["SAM31_CHECKPOINT"] = preflight["sam31_checkpoint_path"]
    os.environ["SAM31_SOURCE_ROOT"] = preflight["sam31_source_root"]
    os.environ["SAM31_SOURCE_REVISION"] = preflight["sam31_source_revision"]
    os.environ["SAM31_CHECKPOINT_SHA256"] = args.sam_checkpoint_sha256
    os.environ["PS_DWPOSE_DET"] = preflight["dwpose_detector_path"]
    os.environ["PS_DWPOSE_POSE"] = preflight["dwpose_pose_path"]
    if cv2 is None:
        raise RuntimeError("the PointStream audit environment requires OpenCV")

    torch = __import__("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("the approved SAM3.1 and DWPose pilot requires CUDA")
    device_name = torch.cuda.get_device_name(0)
    torch.cuda.reset_peak_memory_stats()

    sam_results: dict[str, dict[str, Any]] = {}
    scene_provenance: dict[str, dict[str, Any]] = {}
    stage_timings: dict[str, float] = {}
    sam_subprocess = Path(__file__).resolve()
    for scene in preflight["selected_scenes"]:
        frames, frame_sources, source_dir = _source_frames(
            data_root,
            scene,
            frame_count=int(scene["frame_count"]),
        )
        scene_dir = run_dir / scene["source_id"]
        scene_dir.mkdir(parents=True)
        frame_dir = scene_dir / "sam31_frames"
        frame_dir.mkdir()
        jpeg_records: list[dict[str, Any]] = []
        for index, frame in enumerate(frames):
            jpeg = frame_dir / f"{index:05d}.jpg"
            ok = cv2.imwrite(
                str(jpeg),
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            if not ok:
                raise OSError(f"could not stage SAM3.1 frame {jpeg}")
            jpeg_records.append({"path": str(jpeg), "sha256": _sha256(jpeg), "quality": 95})
        metadata_path = scene_dir / "sam31_result.json"
        arrays_path = scene_dir / "sam31_masks.npz"
        config_path = scene_dir / "sam31_worker_config.json"
        config_path.write_text(
            json.dumps(
                {
                    "scene": scene,
                    "frames_dir": str(frame_dir),
                    "metadata_path": str(metadata_path),
                    "arrays_path": str(arrays_path),
                    "frame_count": len(frames),
                    "frame_width": int(frames.shape[2]),
                    "frame_height": int(frames.shape[1]),
                    "sam_checkpoint": preflight["sam31_checkpoint_path"],
                    "sam_source_root": preflight["sam31_source_root"],
                    "sam_source_revision": preflight["sam31_source_revision"],
                    "sam_checkpoint_sha256": preflight["model_hashes"]["sam31_checkpoint"],
                    "prob_threshold": args.prob_threshold,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        worker_command = [
            preflight["sam31_python"],
            str(sam_subprocess),
            "--sam-worker-config",
            str(config_path),
        ]
        sam_start = time.perf_counter()
        sam_process = subprocess.run(
            worker_command,
            check=False,
            capture_output=True,
            text=True,
            cwd=repo_root,
            env=os.environ.copy(),
            timeout=3600,
        )
        sam_seconds = time.perf_counter() - sam_start
        if sam_process.returncode:
            detail = sam_process.stderr.strip() or sam_process.stdout.strip()
            raise RuntimeError(
                f"SAM3.1 worker failed for {scene['source_id']} "
                f"(exit {sam_process.returncode}): {detail[-6000:]}"
            )
        sam_result = _load_sam_worker(metadata_path)
        sam_result["sam_runtime"] = preflight["sam31_runtime"]
        sam_result["worker_command"] = worker_command
        sam_results[scene["source_id"]] = sam_result
        scene_provenance[scene["source_id"]] = {
            "scene": scene,
            "source_frame_records": frame_sources,
            "source_frame_dir": str(source_dir),
            "sam_input_frames": jpeg_records,
            "sam_worker_command": worker_command,
            "sam_worker_seconds": sam_seconds,
        }
        stage_timings[f"{scene['source_id']}_sam31_worker_s"] = sam_seconds
        stage_timings[f"{scene['source_id']}_sam31_inference_s"] = sam_result["inference_seconds"]
        shutil.rmtree(frame_dir, ignore_errors=True)

    dwpose_start = time.perf_counter()
    pose_estimator = DWPoseEstimator(device="cuda")
    dwpose_load_seconds = time.perf_counter() - dwpose_start

    scene_rows: list[dict[str, Any]] = []
    observation_lines: list[str] = []
    pose_lines: list[str] = []
    view_lines: list[str] = []
    combined_masks: dict[str, np.ndarray] = {}
    stage_timings["sam31_model_load_s"] = max(
        (float(row["model_load_seconds"]) for row in sam_results.values()), default=0.0
    )
    stage_timings["dwpose_model_init_s"] = dwpose_load_seconds
    for scene in preflight["selected_scenes"]:
        frame_count = int(scene["frame_count"])
        frames, frame_sources, source_dir = _source_frames(
            data_root,
            scene,
            frame_count=frame_count,
        )
        scene_dir = run_dir / scene["source_id"]
        scene_meta = scene_provenance[scene["source_id"]]
        sam_result = sam_results[scene["source_id"]]
        stage_start = time.perf_counter()
        result = _sequence_perception(
            scene,
            frames,
            sam_result=sam_result,
            pose_estimator=pose_estimator,
            quality_policy=preflight["quality_policy"],
        )
        stage_timings[f"{scene['source_id']}_pose_geometry_views_s"] = time.perf_counter() - stage_start
        player_union = [
            np.logical_or.reduce(
                [
                    result["masks"][record["mask_key"]]
                    for record in result["observations"]
                    if record["status"] == "observed"
                    and record["object_class"] == "player"
                    and record.get("training_eligible")
                    and int(record["frame_index"]) == int(scene["frame_start"]) + frame_index
                ]
            )
            if any(
                record["status"] == "observed"
                and record["object_class"] == "player"
                and record.get("training_eligible")
                and int(record["frame_index"]) == int(scene["frame_start"]) + frame_index
                for record in result["observations"]
            )
            else np.zeros(frames.shape[1:3], dtype=bool)
            for frame_index in range(frame_count)
        ]
        reference_start = time.perf_counter()
        agreement = _reference_agreement(scene, player_union)
        stage_timings[f"{scene['source_id']}_cached_reference_agreement_s"] = time.perf_counter() - reference_start
        # Each observation is stored as a JSON record plus a referenced binary mask array.
        for record in result["observations"]:
            observation_lines.append(json.dumps(record, sort_keys=True, separators=(",", ":")))
        for record in result["poses"]:
            pose_lines.append(json.dumps(record, sort_keys=True, separators=(",", ":")))
        view_rows = _save_view_manifest(scene, result, run_dir)
        for row in view_rows:
            view_lines.append(json.dumps(row, sort_keys=True, separators=(",", ":")))
        for key, mask in result["masks"].items():
            combined_masks[f"{scene['source_id']}__{key}"] = mask
        codec_start = time.perf_counter()
        client = _encode_client_payload(scene, frames, result, scene_dir)
        client_seconds = time.perf_counter() - codec_start
        decoded = client.pop("decoded")
        stage_timings[f"{scene['source_id']}_client_encode_and_decode_s"] = client_seconds
        metrics = {
            "rgb_psnr_db": _psnr_rgb(frames, decoded),
            "rgb_ssim_gaussian_11_sigma_1_5": _ssim_rgb(frames, decoded),
            "frames": int(len(frames)),
            "resolution": [int(frames.shape[2]), int(frames.shape[1])],
        }
        scene_rows.append(
            {
                "scene": scene,
                "frames": frames,
                "decoded": decoded,
                "result": result,
                "frame_count": frame_count,
                "source_frame_records": scene_meta["source_frame_records"],
                "source_frame_dir": scene_meta["source_frame_dir"],
                "sam_input_frames": scene_meta["sam_input_frames"],
                "sam_worker_command": scene_meta["sam_worker_command"],
                "sam_worker_seconds": scene_meta["sam_worker_seconds"],
                "reference_agreement": agreement,
                "client_roundtrip": client,
                "reconstruction_metrics": metrics,
            }
        )

    _write_contact_and_preview(scene_rows, run_dir)
    training_export = _write_training_dataset(scene_rows, run_dir)
    with (run_dir / "observations.jsonl").open("w", encoding="utf-8") as stream:
        stream.write("\n".join(observation_lines) + "\n")
    with (run_dir / "poses.jsonl").open("w", encoding="utf-8") as stream:
        stream.write("\n".join(pose_lines) + "\n")
    with (run_dir / "views.jsonl").open("w", encoding="utf-8") as stream:
        stream.write("\n".join(view_lines) + "\n")
    np.savez_compressed(run_dir / "observation_masks.npz", **combined_masks)
    artifacts = {
        "observations_jsonl": run_dir / "observations.jsonl",
        "pose_observations_jsonl": run_dir / "poses.jsonl",
        "training_views_jsonl": run_dir / "views.jsonl",
        "mask_arrays_npz": run_dir / "observation_masks.npz",
        "training_samples_jsonl": Path(training_export["samples_jsonl"]),
    }
    dataset_manifest = {
        "schema": "pointstream.dataset-manifest.v1",
        "version": preflight["dataset_version"],
        "active": False,
        "code_provenance": dispatch_provenance,
        "split": preflight["split"],
        "selection_manifest": preflight["selection_manifest"],
        "selection_manifest_sha256": preflight["selection_manifest_sha256"],
        "scene_manifest": preflight["scene_manifest"],
        "scene_manifest_sha256": preflight["scene_manifest_sha256"],
        "quality_policy": preflight["quality_policy"],
        "training_export": training_export,
        "sam31_source_revision": preflight["sam31_source_revision"],
        "estimators": {
            "sam31": {
                "source_revision": preflight["sam31_source_revision"],
                "python": preflight["sam31_runtime"]["python"],
                "checkpoint_path": preflight["sam31_checkpoint_path"],
                "checkpoint_sha256": preflight["model_hashes"]["sam31_checkpoint"],
                "provenance": scene_rows[0]["result"]["estimator_provenance"]["sam31"],
                "policy": "offline_bidirectional",
            },
            "dwpose": {
                "detector_path": preflight["dwpose_detector_path"],
                "detector_sha256": preflight["model_hashes"]["dwpose_detector"],
                "pose_path": preflight["dwpose_pose_path"],
                "pose_sha256": preflight["model_hashes"]["dwpose_pose"],
                "provenance": scene_rows[0]["result"]["estimator_provenance"]["dwpose"],
                "policy": "offline_bidirectional",
            },
        },
        "model_adapters": {
            "schema": "pointstream.model-adapter-views.v1",
            "animate_anyone": {
                "pose_schema": "openpose-18",
                "path_key": "animate_anyone_openpose18",
                "supported_runtime_input": "RGB pose condition image",
            },
            "spade": {
                "path_key": "spade_condition_rgb",
                "player_view": "OpenPose-18 skeleton",
                "racket_view": "racket silhouette plus wrist-tip and transverse-width lines",
                "joint_view": "OpenPose-18 skeleton plus racket geometry",
                "checkpoint_compatibility": "requires racket-aware training; existing checkpoint compatibility unverified",
            },
            "controlnet": {
                "path_key": "controlnet_condition_rgb",
                "channel_order": ["player_mask_red", "racket_mask_green", "axis_or_width_blue"],
                "named_channel_paths": [
                    "controlnet_player_mask",
                    "controlnet_racket_mask",
                    "controlnet_racket_axis",
                    "controlnet_racket_width",
                ],
                "checkpoint_compatibility": "requires a checkpoint trained for this geometry contract",
            },
            "mttf": {
                "status": "not_integrated",
                "reason": "no MTTF model implementation or conditioning contract exists in this repository",
            },
        },
        "sources": [
            {
                "source_id": row["scene"]["source_id"],
                "split": row["scene"].get("split", preflight["split"]),
                "video": row["scene"]["video"],
                "scene": row["scene"]["scene"],
                "frame_ids": [frame["frame_index"] for frame in row["source_frame_records"]],
                "frames": row["source_frame_records"],
                "coverage_by_class": row["result"]["coverage"],
                "quality": row["result"]["quality"],
                "prompt_retries": row["result"]["retries"],
                "quarantined_samples": row["result"]["quarantined"],
                "sam_masks": {
                    "path": str(run_dir / row["scene"]["source_id"] / "sam31_masks.npz"),
                    "sha256": row["result"]["sam_arrays_sha256"],
                },
            }
            for row in scene_rows
        ],
        "samples": [json.loads(line) for line in view_lines],
        "artifacts": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in artifacts.items()
        },
        "runtime_environment": {
            "pointstream_python": sys.executable,
            "pointstream_dependencies": preflight["runtime_dependencies"],
            "sam31_runtime": preflight["sam31_runtime"],
        },
        "dispatch_provenance": dispatch_provenance,
        "promotion": {
            "eligible": False,
            "reason": "development-only SAM3.1 pseudo-labels; review the quality-filter contact sheets before promotion",
        },
    }
    dataset_manifest_path = run_dir / "dataset_manifest.json"
    dataset_manifest_path.write_text(
        json.dumps(dataset_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_scenes: list[dict[str, Any]] = []
    for row in scene_rows:
        result = row["result"]
        report_scenes.append(
            {
                "source_id": row["scene"]["source_id"],
                "scene": row["scene"],
                "frame_ids": [item["frame_index"] for item in row["source_frame_records"]],
                "frame_sha256": [item["rgb_sha256"] for item in row["source_frame_records"]],
                "source_video_sha256": next(
                    (
                        item.get("source_metadata", {}).get("sha256")
                        for item in _read_json(Path(args.scene_manifest)).get("scenes", [])
                        if item.get("video") == row["scene"]["video"] and item.get("scene") == row["scene"]["scene"]
                    ),
                    None,
                ),
                "coverage_by_class": result["coverage"],
                "quality": result["quality"],
                "missing_observations": {
                    "count": sum(1 for item in result["observations"] if item["status"] != "observed"),
                    "examples": [item for item in result["observations"] if item["status"] != "observed"][:50],
                },
                "identity_discontinuities": result["identity_discontinuities"],
                "prompt_retries": result["retries"],
                "pose_visibility_counts": result["pose_visibility_counts"],
                "racket_fallback_counts": result["racket_fallback_counts"],
                "quarantined_samples": result["quarantined"],
                "cached_mask_agreement": row["reference_agreement"],
                "client_roundtrip": row["client_roundtrip"],
                "reconstruction_metrics": row["reconstruction_metrics"],
                "frame_roles": result["frame_roles"],
                "sam31_runtime": result["sam_runtime"],
                "sam_worker_command": row["sam_worker_command"],
                "sam_worker_seconds": row["sam_worker_seconds"],
                "sam_gpu_memory_bytes": result["sam_gpu_memory_bytes"],
                "sam_mask_archive": {
                    "path": str(run_dir / row["scene"]["source_id"] / "sam31_masks.npz"),
                    "sha256": result["sam_arrays_sha256"],
                },
            }
        )
    sam_peak_allocated = max(
        (int(row["result"]["sam_gpu_memory_bytes"]["peak_allocated"]) for row in scene_rows),
        default=0,
    )
    sam_peak_reserved = max(
        (int(row["result"]["sam_gpu_memory_bytes"]["peak_reserved"]) for row in scene_rows),
        default=0,
    )
    dwpose_peak_allocated = int(torch.cuda.max_memory_allocated())
    dwpose_peak_reserved = int(torch.cuda.max_memory_reserved())
    report = {
        "schema": "pointstream.sam31-audit-report.v1",
        "status": "complete_quality_dataset" if preflight["quality_policy"] else "complete_bounded_development_pilot",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_dir": str(run_dir),
        "source_split": "development_exposed_bp46",
        "selection_manifest": preflight["selection_manifest"],
        "training_export": training_export,
        "gpu": device_name,
        "gpu_memory_bytes": {
            "peak_allocated": max(sam_peak_allocated, dwpose_peak_allocated),
            "peak_reserved": max(sam_peak_reserved, dwpose_peak_reserved),
            "sam_worker_peak_allocated": sam_peak_allocated,
            "sam_worker_peak_reserved": sam_peak_reserved,
            "dwpose_peak_allocated": dwpose_peak_allocated,
            "dwpose_peak_reserved": dwpose_peak_reserved,
            "current_allocated": int(torch.cuda.memory_allocated()),
            "current_reserved": int(torch.cuda.memory_reserved()),
        },
        "estimators": {
            "sam31_source_root": preflight["sam31_source_root"],
            "sam31_python": preflight["sam31_python"],
            "sam31_source_revision": preflight["sam31_source_revision"],
            "sam31_checkpoint_path": preflight["sam31_checkpoint_path"],
            "sam31_checkpoint_sha256": preflight["model_hashes"]["sam31_checkpoint"],
            "dwpose_detector_path": preflight["dwpose_detector_path"],
            "dwpose_detector_sha256": preflight["model_hashes"]["dwpose_detector"],
            "dwpose_pose_path": preflight["dwpose_pose_path"],
            "dwpose_pose_sha256": preflight["model_hashes"]["dwpose_pose"],
        },
        "native_codec_inventory": preflight["native_codec_inventory"],
        "codec_evidence_scope": (
            "PointStream semantic client transport measured (JPEG appearance references, bbox "
            "metadata, and exact PSM1 mask roundtrip). Pose/motion payloads, background and residual "
            "video codecs, and native video codec execution were not exercised."
        ),
        "training_conditioning_transport_scope": {
            "appearance_and_masks": "decoded through the fresh-process client path",
            "pose_and_racket_geometry": "derived from retained unquantized observations",
            "pose_or_geometry_transmitted": False,
            "training_views_identical_to_decoded_client_conditioning": False,
        },
        "runtime_environment": {
            "python_executable": sys.executable,
            "python_version": sys.version,
            "command_argv": sys.argv,
            "hostname": __import__("socket").gethostname(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES")
            or getattr(torch.cuda.get_device_properties(0), "uuid", None),
            "runtime_dependencies": preflight["runtime_dependencies"],
            "sam31_runtime": preflight["sam31_runtime"],
        },
        "dispatch_provenance": dispatch_provenance,
        "stage_timings_seconds": stage_timings,
        "scenes": report_scenes,
        "artifacts": {
            "observations_jsonl": str(run_dir / "observations.jsonl"),
            "pose_observations_jsonl": str(run_dir / "poses.jsonl"),
            "training_views_jsonl": str(run_dir / "views.jsonl"),
            "dataset_manifest_json": str(dataset_manifest_path),
            "mask_arrays_npz": str(run_dir / "observation_masks.npz"),
            "html_contact_sheet": str(run_dir / "index.html"),
            "contact_sheet_png": str(run_dir / "contact_sheet.png"),
            "sequence_preview_mp4": str(run_dir / "sequence_preview.mp4"),
            "training_samples_jsonl": training_export["samples_jsonl"],
            "training_dataset_root": training_export["root"],
        },
        "active_dataset_manifest_changed": False,
        "promotion": {
            "eligible": False,
            "reasons": [
                "dataset uses development-only SAM3.1 pseudo-labels rather than manual ground truth",
                "review accepted and rejected quality-filter overlays before promotion",
            ],
        },
    }
    report_path = run_dir / "audit.json"
    report["artifacts"]["dataset_manifest_sha256"] = _sha256(dataset_manifest_path)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["artifacts"]["audit_json"] = str(report_path)
    report["artifacts"]["audit_json_sha256"] = _sha256(report_path)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-manifest", type=Path, default=Path("manifests/sam31_pilot_v1.json"))
    parser.add_argument("--dataset-manifest", type=Path, help="full validated SAM3.1 quality-dataset selection")
    parser.add_argument("--quality-dataset-manifest", type=Path, default=Path("manifests/sam31_quality_dataset_v3.json"))
    parser.add_argument("--build-existing-quality-dataset", action="store_true", help="refresh all tracks in the existing external training dataset")
    parser.add_argument("--shard-index", type=int, default=0, help="deterministic dataset shard index")
    parser.add_argument("--shard-count", type=int, default=1, help="total deterministic dataset shard count")
    parser.add_argument("--smoke-chunks", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--resume-sam-output", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--scene-manifest", type=Path, default=Path("manifests/bp46_long_tennis_scenes.json"))
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("PS_DATA_ROOT", ".")))
    parser.add_argument("--output-dir", type=Path, required=False)
    parser.add_argument("--sam-checkpoint", type=Path, default=os.environ.get("SAM31_CHECKPOINT"))
    parser.add_argument("--sam-source-root", type=Path, default=os.environ.get("SAM31_SOURCE_ROOT", "/home/itec/emanuele/.cache/sam3-meta"))
    parser.add_argument("--sam-source-revision", default=os.environ.get("SAM31_SOURCE_REVISION", ""))
    parser.add_argument("--sam-checkpoint-sha256", default=os.environ.get("SAM31_CHECKPOINT_SHA256", ""))
    parser.add_argument(
        "--sam-python",
        type=Path,
        default=os.environ.get(
            "SAM31_PYTHON", "/home/itec/emanuele/.conda/envs/pointstream-sam31/bin/python"
        ),
    )
    parser.add_argument("--dwpose-detector", type=Path, default=os.environ.get("PS_DWPOSE_DET", "/home/itec/emanuele/Models/DWPose/yolox_l.onnx"))
    parser.add_argument("--dwpose-pose", type=Path, default=os.environ.get("PS_DWPOSE_POSE", "/home/itec/emanuele/Models/DWPose/dw-ll_ucoco_384.onnx"))
    parser.add_argument("--dwpose-device", choices=("cpu", "cuda"), default="cpu", help="DWPose execution device; SAM3.1 mask generation still runs on CUDA")
    parser.add_argument("--prob-threshold", type=float, default=0.35)
    parser.add_argument("--inspect-only", action="store_true", help="verify data/model inputs and print identities without loading models")
    parser.add_argument("--preflight-output", type=Path, help="write inspect-only provenance JSON outside the repository")
    parser.add_argument("--decode-worker", nargs=2, metavar=("PAYLOAD", "OUTPUT"), help=argparse.SUPPRESS)
    parser.add_argument("--sam-worker-config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--sam-quality-worker", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.sam_quality_worker:
        return _sam_quality_dataset_worker(args.sam_quality_worker, resume=args.resume_sam_output)
    if args.sam_worker_config:
        return _sam_worker(args.sam_worker_config)
    if args.decode_worker:
        return _decode_worker(Path(args.decode_worker[0]), Path(args.decode_worker[1]))
    repo_root = Path(__file__).resolve().parents[1]
    try:
        if args.build_existing_quality_dataset:
            if args.output_dir is None:
                raise ValueError("--output-dir is required for an existing quality-dataset build")
            result = build_existing_quality_dataset(args, repo_root=repo_root)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.sam_checkpoint is None:
            raise FileNotFoundError("set --sam-checkpoint or SAM31_CHECKPOINT to the pinned SAM3.1 checkpoint")
        details = inspect_inputs(args, repo_root=repo_root)
        if args.inspect_only:
            if args.preflight_output is not None:
                output = _external_root(args.preflight_output, repo_root)
                if output.exists():
                    raise FileExistsError(f"refusing to overwrite preflight output: {output}")
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(details, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps(details, indent=2, sort_keys=True))
            return 0
        if args.output_dir is None:
            raise ValueError("--output-dir is required for an audit run")
        report = run_audit(args, repo_root=repo_root)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(f"audit_dataset_pipeline: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
