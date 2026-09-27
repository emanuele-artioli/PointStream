"""Run the bounded shared SAM3.1/DWPose dataset and client-path audit.

All inputs and outputs resolve under the external PointStream data root. The
script writes a diagnostic versioned dataset, never changes an active training
manifest, and decodes the serialized PointStream client payload in a fresh
process that receives no source-frame argument.
"""

from __future__ import annotations

import argparse
from collections.abc import Collection
from collections import Counter, defaultdict
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # SAM worker uses PIL; the parent audit uses OpenCV.
    cv2 = None

PILOT_SCHEMA = "pointstream.sam31-pilot.v1"
OBSERVATION_SCHEMA = "pointstream.observation.v1"


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
    """Check exact pilot inputs and model provenance without loading weights."""
    data_root = _external_root(Path(args.data_root), repo_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"PointStream data root does not exist: {data_root}")
    pilot_path = Path(args.pilot_manifest).resolve()
    catalog_path = Path(args.scene_manifest).resolve()
    pilot = _read_json(pilot_path)
    catalog = _read_json(catalog_path)
    scenes = validate_pilot_manifest(
        pilot,
        catalog,
        reserved_source_ids=_reserved_ids(repo_root),
    )
    external_scenes: list[dict[str, Any]] = []
    for scene in scenes:
        directory, source_dataset = _source_frame_directory(data_root, scene)
        _verify_catalog_frame_anchors(directory, scene)
        start = int(scene["frame_start"])
        exact = _frame_paths_by_position(directory, start, 16)
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
        "pilot_manifest": str(pilot_path),
        "pilot_manifest_sha256": _sha256(pilot_path),
        "scene_manifest": str(catalog_path),
        "scene_manifest_sha256": _sha256(catalog_path),
        "split": pilot["split"],
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
    policy = "offline_bidirectional"
    provenance = segmenter.provenance(policy)
    width, height = int(config["frame_width"]), int(config["frame_height"])
    outputs: dict[str, dict[tuple[int, str], Any]] = {"player": {}, "racket": {}}
    prompt_records: list[dict[str, Any]] = []
    retry_records: list[dict[str, Any]] = []
    total_started = time.perf_counter()
    try:
        for role in ("player", "racket"):
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

        for role, records in outputs.items():
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
        for role, session_key in list(segmenter.sessions):
            segmenter.close_session(role, session_key=session_key)

    masks: dict[str, np.ndarray] = {}
    serialized: list[dict[str, Any]] = []
    for role, role_records in outputs.items():
        for (frame_index, object_id), item in sorted(role_records.items()):
            mask_key = None
            if item.mask is not None:
                mask_key = f"mask_{len(masks):06d}"
                masks[mask_key] = np.asarray(item.mask, dtype=np.uint8)
            serialized.append(
                {
                    "role": role,
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
    policy = "offline_bidirectional"
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

    associator = RacketPlayerAssociator()
    associated_rackets: dict[tuple[int, str], ObservedObject] = {}
    geometry_by_key: dict[tuple[int, str], Any] = {}
    for frame_index in range(count):
        racket_objects: list[ObservedObject] = []
        player_poses = [
            PlayerPose(object_id, frame_index, pose.values, pose.schema.name)
            for object_id, pose, _transform in poses_by_frame.get(frame_index, ())
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
                    bbox=tuple(float(value) for value in bbox),
                    mask=item.mask,
                )
            )
        linked = associator.associate(racket_objects, player_poses)
        for racket in linked:
            key = (frame_index, racket.object_id)
            associated_rackets[key] = racket
            geometry_by_key[key] = extract_racket_cross(racket, player_poses)

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
        for role in ("player", "racket"):
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

                linked = associated_rackets.get((frame_index, object_id)) if role == "racket" else None
                reason = "mask_unavailable" if item is None or item.mask is None else None
                view_transform = None
                if item is not None and item.mask is not None:
                    _crop, _crop_mask, transform = render_object_view(
                        frames[frame_index],
                        np.asarray(item.mask),
                        tuple(float(value) for value in _mask_bbox(item.mask)),
                    )
                    view_transform = transform.to_record()
                    box = _mask_bbox(item.mask)
                    if box is not None:
                        x0, y0, x1, y1 = box
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
                if role == "racket" and linked is not None:
                    associated_player_id = linked.associated_player_id
                    associated_wrist = linked.associated_wrist
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
                    key = f"m{len(masks):06d}"
                    masks[key] = observation.mask
                    raw_record = observation.to_record()
                    raw_record["mask_key"] = key
                    raw_record["mask_sha256"] = _sha256_array(observation.mask)
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
                if role == "racket" and linked is not None:
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
                all_records.append(record)

        player_masks = [record[1] for record in frame_records["player"]]
        racket_masks = [record[1] for record in frame_records["racket"]]
        frame_roles.append(
            {
                "frame_index": frame_index,
                "player_mask_count": len(player_masks),
                "racket_mask_count": len(racket_masks),
                "failed": not player_masks or not racket_masks,
            }
        )
        for role in ("player", "racket"):
            color = (35, 220, 70) if role == "player" else (240, 130, 35)
            for object_id, mask, bbox, item in frame_records[role]:
                pose_data = next(
                    (pose for pose_id, pose, _transform in poses_by_frame.get(frame_index, ()) if pose_id == object_id),
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
                    }
                )

        # The associated view is built from exactly the player+racket observations
        # above; a racket without a visible wrist remains a quarantined diagnostic.
        for (linked_frame, racket_id), racket in associated_rackets.items():
            if linked_frame != frame_index or racket.associated_player_id is None:
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
                (pose for pose_id, pose, _transform in poses_by_frame.get(frame_index, ()) if pose_id == racket.associated_player_id),
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
            frames[frame_index], mask, tuple(float(value) for value in bbox)
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
    payload = bytes(chunk.bag["wire_request"])
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
        eligible = bool(view.get("eligible_for_cross_training", view["role"] != "racket"))
        row = {
            "schema": "pointstream.observation-view.v1",
            "sample_id": f"{scene['source_id']}:{int(scene['frame_start']) + frame_index}:{view['object_id']}:{view['role']}",
            "source_id": scene["source_id"],
            "split": "development_exposed_bp46",
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
                name: _sha256(path) for name, path in adapter_paths.items()
            },
            "adapter_eligible_for_cross_training": adapters.eligible_for_cross_training,
            "adapter_exclusion_reason": adapters.exclusion_reason,
            "transform": view["transform"].to_record(),
            "eligible_for_training": eligible,
            "training_exclusion_reason": None if eligible else "racket_cross_fallback_or_missing_geometry",
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
                if item["frame_index"] == frame_index and item["role"] == "player"
            ]
            racket_masks = [
                np.asarray(item["source_mask"])
                for item in result["views"]
                if item["frame_index"] == frame_index and item["role"] == "racket"
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
                _panel(_overlay_mask(source[frame_index], player_masks, (35, 220, 70)), "SAM3.1 player masks"),
                _panel(_overlay_mask(source[frame_index], racket_masks, (240, 130, 35)), "SAM3.1 racket masks"),
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
                _panel(_overlay_mask(source[index], [np.asarray(item["source_mask"]) for item in result["views"] if item["frame_index"] == index and item["role"] == "player"], (35, 220, 70)), "player"),
                _panel(_overlay_mask(source[index], [np.asarray(item["source_mask"]) for item in result["views"] if item["frame_index"] == index and item["role"] == "racket"], (240, 130, 35)), "racket"),
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
            frame_count=16,
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
        frames, frame_sources, source_dir = _source_frames(
            data_root,
            scene,
            frame_count=16,
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
        )
        stage_timings[f"{scene['source_id']}_pose_geometry_views_s"] = time.perf_counter() - stage_start
        player_union = [
            np.logical_or.reduce(
                [
                    result["masks"][record["mask_key"]]
                    for record in result["observations"]
                    if record["status"] == "observed"
                    and record["object_class"] == "player"
                    and int(record["frame_index"]) == int(scene["frame_start"]) + frame_index
                ]
            )
            if any(
                record["status"] == "observed"
                and record["object_class"] == "player"
                and int(record["frame_index"]) == int(scene["frame_start"]) + frame_index
                for record in result["observations"]
            )
            else np.zeros(frames.shape[1:3], dtype=bool)
            for frame_index in range(16)
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
    }
    dataset_manifest = {
        "schema": "pointstream.dataset-manifest.v1",
        "version": "sam31-observations-v1",
        "active": False,
        "code_provenance": dispatch_provenance,
        "split": "development_exposed_bp46",
        "pilot_manifest": preflight["pilot_manifest"],
        "pilot_manifest_sha256": preflight["pilot_manifest_sha256"],
        "scene_manifest": preflight["scene_manifest"],
        "scene_manifest_sha256": preflight["scene_manifest_sha256"],
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
                "split": "development_exposed_bp46",
                "video": row["scene"]["video"],
                "scene": row["scene"]["scene"],
                "frame_ids": [frame["frame_index"] for frame in row["source_frame_records"]],
                "frames": row["source_frame_records"],
                "coverage_by_class": row["result"]["coverage"],
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
        "promotion": {"eligible": False, "reason": "pilot and visual review are incomplete"},
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
        "status": "complete_bounded_development_pilot",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "output_dir": str(run_dir),
        "source_split": "development_exposed_bp46",
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
        },
        "active_dataset_manifest_changed": False,
        "promotion": {
            "eligible": False,
            "reasons": [
                "pilot sample includes a racket-scale/occlusion candidate that still needs visual review",
                "all observed-object failures must be reviewed in contact panels",
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
    parser.add_argument("--prob-threshold", type=float, default=0.35)
    parser.add_argument("--inspect-only", action="store_true", help="verify data/model inputs and print identities without loading models")
    parser.add_argument("--preflight-output", type=Path, help="write inspect-only provenance JSON outside the repository")
    parser.add_argument("--decode-worker", nargs=2, metavar=("PAYLOAD", "OUTPUT"), help=argparse.SUPPRESS)
    parser.add_argument("--sam-worker-config", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.sam_worker_config:
        return _sam_worker(args.sam_worker_config)
    if args.decode_worker:
        return _decode_worker(Path(args.decode_worker[0]), Path(args.decode_worker[1]))
    repo_root = Path(__file__).resolve().parents[1]
    try:
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
