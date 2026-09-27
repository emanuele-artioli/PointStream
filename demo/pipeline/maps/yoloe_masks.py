"""YOLOE-26 instance masks for the maps gallery.

Native payload is a COCO-RLE JSON stream (``masks.rle.zst`` when zstd is
available, else ``payload.bin``). Colored overlay PNGs live under
``preview_masks/`` and are never counted as payload.

Empty frames use ``skip_frame``: ``instances: []`` and a zero union mask.
Never fill whole-frame True (the PRESLEY trap).

CLI::

    PYTHONPATH=. python -m demo.pipeline.maps.yoloe_masks --clip PATH \\
        --out demo/outputs/maps/<stem>/yoloe_masks/ [--max-frames N]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from demo.evaluation.profile_map import profile_map
from demo.pipeline.maps.av1_crf import CLASS_COLORS_BGR, PAINT_ORDER
from demo.pipeline.maps.contract import MapStream, write_sidecar
from demo.pipeline.maps.encode import (
    decode_coco_rle,
    frame_record,
    instance_record,
    pack_rle_stream,
    payload_filename,
    unpack_rle_stream,
    write_zstd_blob,
)
from demo.pipeline.maps.model_paths import MODELS, require

MAP_NAME = "yoloe_masks"
MODELS_KEY = "yoloe26_seg"
PREVIEW_DIRNAME = "preview_masks"
SIDECAR_NAME = "sidecar.json"
CLASSES_NAME = "classes.yaml"
PROMPTS_YAML = Path(__file__).with_name("prompts.yaml")

YOLOE_INSTALL_HINT = (
    "ultralytics.YOLOE is not importable. Install the project pin:\n"
    "  pip install ultralytics==8.4.6\n"
    "Load with YOLOE(str(require('yoloe26_seg'))) — never a Hub id, and do not auto-download."
)

_PALETTE_BGR = (
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 128, 255),
    (128, 0, 255),
    (255, 128, 0),
    (0, 255, 128),
)

PredictFn = Callable[[np.ndarray], list[dict[str, Any]]]


def load_classes(path: Path | str | None = None) -> list[str]:
    """Read open-vocab class names from prompts.yaml."""
    yaml_path = Path(path) if path is not None else PROMPTS_YAML
    text = yaml_path.read_text()
    classes: list[str] = []
    try:
        import yaml

        data = yaml.safe_load(text) or {}
        raw = data.get("classes") or []
        classes = [str(item) for item in raw]
    except ImportError:
        in_list = False
        for line in text.splitlines():
            stripped = line.strip()
            if stripped == "classes:":
                in_list = True
                continue
            if in_list:
                if stripped.startswith("- "):
                    classes.append(stripped[2:].strip().strip("'\""))
                elif stripped and not stripped.startswith("#"):
                    break
    if not classes:
        raise ValueError(f"no classes listed in {yaml_path}")
    return classes


def import_yoloe() -> Any:
    try:
        from ultralytics import YOLOE
    except ImportError as exc:
        raise ImportError(YOLOE_INSTALL_HINT) from exc
    return YOLOE


def bind_local_text_encoder(ts_path: Path | None) -> dict[str, str]:
    """Point YOLOE at MODELS['mobileclip2'] without copying into the git repo."""
    extra: dict[str, str] = {}
    if ts_path is None or not Path(ts_path).is_file():
        extra["text_encoder"] = (
            "MODELS['mobileclip2'] missing. first set_classes() may need network "
            "unless mobileclip2_b.ts is in cwd. Copy into the git repo is not allowed."
        )
        return extra
    path = Path(ts_path).resolve()
    extra["text_encoder"] = str(path)
    extra["text_encoder_note"] = (
        "Bound MobileCLIPTS to MODELS['mobileclip2'] without copying into the git repo. "
        "If the bind is ignored, run with cwd that can see the .ts; first set_classes() "
        "may hit the network."
    )
    try:
        import ultralytics.nn.text_model as text_model

        orig = text_model.MobileCLIPTS.__init__

        def _init(self: Any, device: Any, weight: str = "mobileclip_blt.ts") -> None:
            name = Path(str(weight)).name
            if name in {path.name, "mobileclip2_b.ts"}:
                weight = str(path)
            orig(self, device, weight)

        text_model.MobileCLIPTS.__init__ = _init  # type: ignore[method-assign]
    except Exception as exc:  # pragma: no cover - depends on ultralytics internals
        extra["text_encoder_bind"] = (
            f"failed ({exc}); prefer cwd that can see {path.name}. "
            "Copy into the git repo is not allowed."
        )
    return extra


def _as_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros((0,), dtype=np.float32)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask
    return cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)


def instances_from_result(
    result: Any,
    height: int,
    width: int,
    names: dict[int, str] | Sequence[str],
) -> list[dict[str, Any]]:
    """Duck-typed ultralytics Result → instance dicts with numpy masks (no RLE yet)."""
    if result is None:
        return []
    if isinstance(result, (list, tuple)):
        result = result[0] if result else None
    if result is None:
        return []
    masks_obj = getattr(result, "masks", None)
    data = getattr(masks_obj, "data", None) if masks_obj is not None else None
    if data is None:
        return []
    array = _as_numpy(data)
    if array.size == 0:
        return []
    if array.ndim == 2:
        array = array[None, ...]
    n = int(array.shape[0])
    boxes = getattr(result, "boxes", None)
    cls = _as_numpy(getattr(boxes, "cls", None)) if boxes is not None else np.zeros((0,))
    conf = _as_numpy(getattr(boxes, "conf", None)) if boxes is not None else np.zeros((0,))
    xyxy = _as_numpy(getattr(boxes, "xyxy", None)) if boxes is not None else np.zeros((0, 4))
    result_names = getattr(result, "names", None) or names
    out: list[dict[str, Any]] = []
    for i in range(n):
        binary = (_resize_mask(array[i], height, width) > 0.5).astype(np.uint8)
        if not binary.any():
            continue
        class_id = int(cls[i]) if i < cls.shape[0] else 0
        class_name = _class_name(class_id, result_names)
        score = float(conf[i]) if i < conf.shape[0] else 1.0
        if i < xyxy.shape[0] and xyxy[i].size >= 4:
            bbox = [float(v) for v in xyxy[i][:4].tolist()]
        else:
            ys, xs = np.nonzero(binary)
            bbox = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
        out.append(
            {
                "mask": binary,
                "class_id": class_id,
                "class_name": class_name,
                "score": score,
                "bbox": bbox,
            }
        )
    return out


def _class_name(class_id: int, names: dict[int, str] | Sequence[str] | Any) -> str:
    if isinstance(names, dict) and class_id in names:
        return str(names[class_id])
    if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
        if 0 <= class_id < len(names):
            return str(names[class_id])
    return f"class_{class_id}"


def records_from_instances(index: int, instances: Sequence[dict[str, Any]]) -> dict[str, Any]:
    packed: list[dict[str, Any]] = []
    for inst in instances:
        raw_bbox = inst.get("bbox") or []
        rec = instance_record(
            inst["mask"],
            class_id=int(inst["class_id"]),
            class_name=str(inst["class_name"]),
            score=float(inst.get("score", 1.0)),
            bbox=list(raw_bbox) if len(raw_bbox) >= 4 else None,
        )
        if rec is not None:
            packed.append(rec)
    return frame_record(index, packed)


def color_overlay(frame: np.ndarray, frame_rec: dict[str, Any], *, alpha: int = 180) -> np.ndarray:
    """Masks-only BGRA. Empty instances leave a fully transparent frame (no RGB under)."""
    if frame.ndim == 2:
        height, width = int(frame.shape[0]), int(frame.shape[1])
    else:
        height, width = int(frame.shape[0]), int(frame.shape[1])
    vis = np.zeros((height, width, 4), dtype=np.uint8)
    for inst in frame_rec.get("instances") or []:
        mask = decode_coco_rle(inst["rle"])
        if mask.shape != (height, width):
            mask = _resize_mask(mask, height, width) > 0.5
        color = _PALETTE_BGR[int(inst.get("class_id", 0)) % len(_PALETTE_BGR)]
        region = mask.astype(bool)
        if not region.any():
            continue
        vis[region, 0] = color[0]
        vis[region, 1] = color[1]
        vis[region, 2] = color[2]
        vis[region, 3] = int(alpha)
    return vis


def write_preview_masks(
    frames: Sequence[np.ndarray],
    frame_recs: Sequence[dict[str, Any]],
    preview_dir: Path,
) -> int:
    preview_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for frame, rec in zip(frames, frame_recs):
        path = preview_dir / f"{int(rec['index']):06d}.png"
        vis = color_overlay(frame, rec)
        if vis.ndim != 3 or vis.shape[2] != 4:
            raise RuntimeError("YOLOE preview must be BGRA (masks only, no RGB under)")
        if not cv2.imwrite(str(path), vis):
            raise RuntimeError(f"failed to write preview {path}")
        total += path.stat().st_size
    return total


def write_yoloe_map(
    frames: Sequence[np.ndarray],
    out_dir: Path | str,
    *,
    predict_fn: PredictFn,
    classes: Sequence[str] | None = None,
    backend: str = "yoloe-26n-seg.pt",
    fps: float = 30.0,
    extra: dict[str, Any] | None = None,
    n_warmup: int = 2,
    n_runs: int = 8,
) -> MapStream:
    """In-memory writer for tests. Does not load YOLOE."""
    return write_mask_map(
        frames,
        out_dir,
        predict_fn=predict_fn,
        classes=list(classes) if classes is not None else load_classes(),
        map_name=MAP_NAME,
        backend=backend,
        fps=fps,
        extra=extra,
        n_warmup=n_warmup,
        n_runs=n_runs,
    )


def write_mask_map(
    frames: Sequence[np.ndarray],
    out_dir: Path | str,
    *,
    predict_fn: PredictFn,
    classes: Sequence[str],
    map_name: str,
    backend: str,
    fps: float = 30.0,
    extra: dict[str, Any] | None = None,
    n_warmup: int = 2,
    n_runs: int = 8,
    profile_extract: Callable[[np.ndarray], Any] | None = None,
) -> MapStream:
    """Extract, pack RLE, write preview_masks/, sidecar. Works with a stub predict_fn."""
    if not frames:
        raise ValueError("no frames")
    if fps <= 0:
        raise ValueError("fps must be positive")
    class_list = [str(c) for c in classes]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    height, width = int(frames[0].shape[0]), int(frames[0].shape[1])
    frame_recs: list[dict[str, Any]] = []
    n_skipped = 0
    n_instances = 0
    for index, frame in enumerate(frames):
        instances = predict_fn(np.asarray(frame))
        rec = records_from_instances(index, instances)
        if not rec["instances"]:
            n_skipped += 1
        n_instances += len(rec["instances"])
        frame_recs.append(rec)

    blob, codec = pack_rle_stream(
        frame_recs, height=height, width=width, fps=float(fps), classes=class_list
    )
    payload_path = out / payload_filename(codec)
    write_zstd_blob(blob, payload_path)

    preview_dir = out / PREVIEW_DIRNAME
    preview_bytes = write_preview_masks(frames, frame_recs, preview_dir)

    extract_for_profile = profile_extract or predict_fn
    names = {i: name for i, name in enumerate(class_list)}

    def _to_instance_dicts(extracted: Any) -> list[dict[str, Any]]:
        if extracted is None:
            return []
        if isinstance(extracted, list) and (
            not extracted or (isinstance(extracted[0], dict) and "mask" in extracted[0])
        ):
            return extracted
        return instances_from_result(extracted, height, width, names)

    def pack_fn(extracted: Any) -> bytes:
        rec = records_from_instances(0, _to_instance_dicts(extracted))
        packed, _codec = pack_rle_stream(
            [rec], height=height, width=width, fps=float(fps), classes=class_list
        )
        return packed

    def decode_fn(payload: bytes) -> dict[str, Any]:
        return unpack_rle_stream(payload)

    stats = profile_map(
        extract_for_profile,
        np.asarray(frames[0]),
        pack_fn=pack_fn,
        decode_fn=decode_fn,
        n_warmup=n_warmup,
        n_runs=n_runs,
    )

    n_frames = len(frame_recs)
    duration_s = n_frames / float(fps)
    merged_extra: dict[str, Any] = {
        "rle": "coco",
        "payload_codec": f"{codec}+json",
        "payload_schema": "pointstream.maps.coco_rle.v1",
        "n_skipped": n_skipped,
        "n_instances": n_instances,
        "classes": class_list,
        "overlay": "rgba",
    }
    if extra:
        merged_extra.update(extra)

    stream = MapStream(
        map=map_name,
        backend=str(backend),
        payload_path=str(payload_path),
        payload_bytes=payload_path.stat().st_size,
        preview_path=str(preview_dir),
        preview_bytes=int(preview_bytes),
        duration_s=duration_s,
        n_frames=n_frames,
        fps=float(fps),
        extract_ms_p50=float(stats["extract_ms_p50"]),
        extract_ms_p95=float(stats["extract_ms_p95"]),
        pack_ms_p50=float(stats["pack_ms_p50"]),
        codec_ms_p50=float(stats["codec_ms_p50"]),
        decode_ms_p50=float(stats["decode_ms_p50"]),
        gpu=str(stats["gpu"]),
        kind="native",
        extra=merged_extra,
    )
    write_sidecar(stream, out / SIDECAR_NAME)
    return stream


def _probe_fps(clip: Path) -> float:
    cap = cv2.VideoCapture(str(clip))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    return fps if fps > 1e-6 else 30.0


def _infer_device() -> Any:
    try:
        import torch

        if torch.cuda.is_available():
            return 0
    except Exception:
        pass
    return "cpu"


def load_prompt_map(
    model_name: str,
    clip_id: str,
    path: Path | str | None = None,
) -> dict[str, str]:
    """Role name to the text that model should hear for this clip.

    Missing prompt entries fall back to the role name, so older prompts.yaml
    files that only list ``classes`` still load.
    """
    yaml_path = Path(path) if path is not None else PROMPTS_YAML
    roles = load_classes(yaml_path)
    try:
        import yaml

        data = yaml.safe_load(yaml_path.read_text()) or {}
    except ImportError:
        data = {}
    model_map = (data.get("prompts") or {}).get(model_name) or {}
    clip_map = model_map.get(clip_id) or model_map.get("default") or {}
    if not isinstance(clip_map, dict):
        clip_map = {}
    return {role: str(clip_map.get(role) or role) for role in roles}


def _enable_cuda_fast_math() -> None:
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


# Long side of a 1920x1080 frame. 1280 letterboxes that frame to 1280x720.
# yoloe-26x on the RTX 6000 Ada was already compute-bound at 640 and 1280.
YOLOE_IMGSZ = 1920
YOLOE_CONF = 0.10
YOLOE_WEIGHTS_KEY = "yoloe26_seg_x"


def load_yoloe_model(class_texts: Sequence[str]) -> tuple[Any, Path, dict[str, str]]:
    YOLOE = import_yoloe()
    weights = require(YOLOE_WEIGHTS_KEY)
    extra = bind_local_text_encoder(MODELS.get("mobileclip2"))
    extra["models_key"] = YOLOE_WEIGHTS_KEY
    extra["imgsz"] = str(YOLOE_IMGSZ)
    extra["conf"] = str(YOLOE_CONF)
    extra["half"] = "true"
    model = YOLOE(str(weights))
    # set_classes embeds text in fp32. half=True on predict/track comes after this.
    model.set_classes(list(class_texts))
    return model, weights, extra


def make_yoloe_predict_fn(model: Any, classes: Sequence[str], device: Any) -> PredictFn:
    names = {i: name for i, name in enumerate(classes)}

    def predict_fn(frame: np.ndarray) -> list[dict[str, Any]]:
        height, width = int(frame.shape[0]), int(frame.shape[1])
        results = model.predict(
            source=frame,
            verbose=False,
            retina_masks=True,
            device=device,
            imgsz=YOLOE_IMGSZ,
        )
        result = results[0] if results else None
        return instances_from_result(result, height, width, names)

    return predict_fn


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def paint_class_masks(
    class_masks: dict[str, np.ndarray],
    height: int,
    width: int,
) -> np.ndarray:
    """BGR mask video frame. Background stays black so the inspector can key it."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for name in PAINT_ORDER:
        mask = class_masks.get(name)
        if mask is None:
            continue
        binary = np.asarray(mask).astype(bool)
        if binary.shape != (height, width):
            binary = cv2.resize(
                binary.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        if not binary.any():
            continue
        canvas[binary] = CLASS_COLORS_BGR.get(name, (255, 255, 255))
    return canvas


def class_masks_from_result(
    result: Any,
    height: int,
    width: int,
    names: dict[int, str],
) -> dict[str, np.ndarray]:
    masks = {name: np.zeros((height, width), dtype=bool) for name in names.values()}
    if result is None or getattr(result, "masks", None) is None or getattr(result, "boxes", None) is None:
        return masks
    data = result.masks.data
    clss = result.boxes.cls
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    if hasattr(clss, "detach"):
        clss = clss.detach().cpu().numpy()
    data = np.asarray(data)
    clss = np.asarray(clss).astype(int)
    for mask, cls_id in zip(data, clss):
        name = names.get(int(cls_id))
        if name not in masks:
            continue
        binary = np.asarray(mask) > 0.5
        if binary.shape != (height, width):
            binary = cv2.resize(
                binary.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        masks[name] |= binary
    return masks


def run_yoloe_clip(
    clip: Path | str,
    out_dir: Path | str,
    *,
    max_frames: int | None = None,
    prompts: Path | str | None = None,
) -> MapStream:
    """Track YOLOE-26x across the clip and bill the shared CRF AV1 mask video.

    Model load is recorded separately and is not part of extract_ms_p50.
    The first tracked frame is a warmup and is also excluded.
    """
    import time

    from demo.evaluation.profile_map import gpu_name
    from demo.pipeline.maps.av1_crf import AV1_CRF, AV1_PRESET, AV1_SCALE, pipe_bgr_av1

    clip_path = Path(clip)
    role_prompts = load_prompt_map("yoloe", clip_path.stem, prompts)
    class_list = list(role_prompts)
    class_texts = [role_prompts[role] for role in class_list]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    src_yaml = Path(prompts) if prompts is not None else PROMPTS_YAML
    (out / CLASSES_NAME).write_text(src_yaml.read_text())

    _enable_cuda_fast_math()
    load_t0 = time.perf_counter()
    model, weights, encoder_extra = load_yoloe_model(class_texts)
    model_load_s = time.perf_counter() - load_t0
    device = _infer_device()
    fps = _probe_fps(clip_path)
    names = {i: role for i, role in enumerate(class_list)}

    probe = cv2.VideoCapture(str(clip_path))
    width = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    probe.release()
    if width <= 0 or height <= 0:
        raise RuntimeError(f"could not read frame size from {clip_path}")

    payload_path = out / "masks.av1.mp4"
    proc = pipe_bgr_av1(width, height, fps, payload_path)
    assert proc.stdin is not None
    step_ms: list[float] = []
    n_frames = 0
    n_nonempty = 0
    try:
        import torch
    except Exception:
        torch = None  # type: ignore[assignment]
    t0 = time.perf_counter()
    try:
        for result in model.track(
            source=str(clip_path),
            stream=True,
            persist=True,
            retina_masks=True,
            verbose=False,
            device=device,
            half=True,
            imgsz=YOLOE_IMGSZ,
            conf=YOLOE_CONF,
        ):
            if torch is not None and torch.cuda.is_available():
                torch.cuda.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if n_frames > 0:
                step_ms.append(dt_ms)
            if max_frames is not None and n_frames >= max_frames:
                break
            orig = getattr(result, "orig_shape", None)
            frame_h = int(orig[0]) if orig is not None else height
            frame_w = int(orig[1]) if orig is not None else width
            painted = paint_class_masks(
                class_masks_from_result(result, frame_h, frame_w, names),
                frame_h,
                frame_w,
            )
            if painted.shape[1] != width or painted.shape[0] != height:
                painted = cv2.resize(painted, (width, height), interpolation=cv2.INTER_NEAREST)
            if int(painted.max()) > 0:
                n_nonempty += 1
            proc.stdin.write(painted.tobytes())
            n_frames += 1
            t0 = time.perf_counter()
    finally:
        proc.stdin.close()
        stderr = proc.stderr.read() if proc.stderr is not None else b""
        code = proc.wait()
        if code != 0 or not payload_path.is_file() or payload_path.stat().st_size == 0:
            tail = stderr.decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"YOLOE mask AV1 encode failed ({code}): {tail}")

    if n_frames <= 0:
        raise RuntimeError(f"YOLOE track produced no frames for {clip_path}")
    duration_s = n_frames / float(fps)
    merged_extra: dict[str, Any] = {
        "payload_format": "av1",
        "payload_schema": "pointstream.maps.mask_av1.v1",
        "av1_scale": AV1_SCALE,
        "av1_preset": AV1_PRESET,
        "av1_crf": AV1_CRF,
        "classes": class_list,
        "class_prompts": role_prompts,
        "track": True,
        "imgsz": YOLOE_IMGSZ,
        "conf": YOLOE_CONF,
        "half": True,
        "model_load_s": round(model_load_s, 3),
        "latency_excludes_load": True,
        "n_nonempty": n_nonempty,
        "overlay_path": str(payload_path),
        "overlay_key": "black",
        "overlay": "av1-black-key",
    }
    merged_extra.update(encoder_extra)
    stream = MapStream(
        map=MAP_NAME,
        backend=weights.name,
        payload_path=str(payload_path),
        payload_bytes=payload_path.stat().st_size,
        preview_path=str(payload_path),
        preview_bytes=payload_path.stat().st_size,
        duration_s=duration_s,
        n_frames=n_frames,
        fps=float(fps),
        extract_ms_p50=_percentile(step_ms, 50),
        extract_ms_p95=_percentile(step_ms, 95),
        pack_ms_p50=0.0,
        codec_ms_p50=0.0,
        decode_ms_p50=0.0,
        gpu=str(gpu_name()),
        kind="native",
        extra=merged_extra,
    )
    write_sidecar(stream, out / SIDECAR_NAME)
    return stream


def load_boxes_from(path: Path | str) -> dict[int, list[dict[str, Any]]]:
    """Boxes keyed by frame index from a yoloe_masks dir, sidecar, or RLE payload."""
    target = Path(path)
    payload: Path | None = None
    if target.is_dir():
        for name in ("masks.rle.zst", "payload.bin"):
            candidate = target / name
            if candidate.is_file():
                payload = candidate
                break
        sidecar = target / SIDECAR_NAME
        if payload is None and sidecar.is_file():
            import json

            payload = Path(json.loads(sidecar.read_text())["payload_path"])
    elif target.suffix.lower() == ".json":
        import json

        data = json.loads(target.read_text())
        if "payload_path" in data:
            payload = Path(data["payload_path"])
        elif "frames" in data:
            return _boxes_from_doc(data)
        else:
            raise ValueError(f"no boxes in {target}")
    else:
        payload = target
    if payload is None or not payload.is_file():
        raise FileNotFoundError(f"no mask payload under {path}")
    return _boxes_from_doc(unpack_rle_stream(payload.read_bytes()))


def _boxes_from_doc(doc: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = {}
    for frame_rec in doc.get("frames") or []:
        boxes: list[dict[str, Any]] = []
        for inst in frame_rec.get("instances") or []:
            boxes.append(
                {
                    "xyxy": [float(v) for v in inst["bbox"][:4]],
                    "class_id": int(inst.get("class_id", 0)),
                    "class_name": str(inst.get("class_name", "object")),
                    "score": float(inst.get("score", 1.0)),
                }
            )
        out[int(frame_rec["index"])] = boxes
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract YOLOE-26 instance masks as a native COCO-RLE maps-gallery stream."
    )
    parser.add_argument("--clip", type=Path, required=True, help="Input video path")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory (masks.rle.zst, preview_masks/, sidecar.json)",
    )
    parser.add_argument("--max-frames", type=int, default=None, dest="max_frames")
    parser.add_argument("--prompts", type=Path, default=PROMPTS_YAML)
    args = parser.parse_args(argv)
    try:
        run_yoloe_clip(args.clip, args.out, max_frames=args.max_frames, prompts=args.prompts)
    except ImportError as exc:
        print(exc, file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
