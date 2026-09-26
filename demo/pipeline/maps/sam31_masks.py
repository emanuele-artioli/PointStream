"""SAM quality masks for the maps gallery.

SAM 3.1 multiplex is not in Models. Backend is MODELS['sam3'] (sam3.pt);
the inspector map name stays ``sam31_masks``. Frame-wise only
(``temporal_id=false``).

Prompt order:
1. YOLOE boxes from ``--boxes-from`` sidecar/dir
2. text concepts hand,tool via SAM3SemanticPredictor if that API exists
3. else a box/point PVS attempt; unprompted failures stay skip_frame

Native RLE schema matches YOLOE so the inspector can swap backends.

CLI::

    PYTHONPATH=. python -m demo.pipeline.maps.sam31_masks --clip PATH \\
        --out demo/outputs/maps/<stem>/sam31_masks/ [--boxes-from DIR] [--max-frames N]
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from demo.pipeline.maps.contract import MapStream
from demo.pipeline.maps.model_paths import require
from demo.pipeline.maps.yoloe_masks import (
    PredictFn,
    _infer_device,
    _probe_fps,
    instances_from_result,
    load_boxes_from,
    write_mask_map,
)

MAP_NAME = "sam31_masks"
MODELS_KEY = "sam3"
TEXT_CONCEPTS = ("hand", "tool")
SAM_NOTE = "sam3.pt; sam3.1_multiplex.pt not in Models"

SAM_INSTALL_HINT = (
    "ultralytics.SAM is not importable. Install the project pin:\n"
    "  pip install ultralytics==8.4.6\n"
    "Load with SAM(str(require('sam3'))) — never a Hub id, and do not auto-download."
)


def import_sam() -> Any:
    try:
        from ultralytics import SAM
    except ImportError as exc:
        raise ImportError(SAM_INSTALL_HINT) from exc
    return SAM


def _semantic_predictor_cls() -> Any | None:
    try:
        from ultralytics.models.sam.predict import SAM3SemanticPredictor

        return SAM3SemanticPredictor
    except Exception:
        return None


def _box_xyxy_list(boxes: Sequence[dict[str, Any]]) -> list[list[float]]:
    out: list[list[float]] = []
    for box in boxes:
        xyxy = box.get("xyxy") or box.get("bbox") or []
        if len(xyxy) < 4:
            continue
        out.append([float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])])
    return out


def _instances_for_boxes(
    result: Any,
    height: int,
    width: int,
    boxes: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    names = {int(b.get("class_id", 0)): str(b.get("class_name", "object")) for b in boxes}
    instances = instances_from_result(result, height, width, names)
    if not instances:
        return []
    n = min(len(instances), len(boxes))
    aligned: list[dict[str, Any]] = []
    for i in range(n):
        inst = dict(instances[i])
        inst["class_id"] = int(boxes[i].get("class_id", inst["class_id"]))
        inst["class_name"] = str(boxes[i].get("class_name", inst["class_name"]))
        inst["score"] = float(boxes[i].get("score", inst.get("score", 1.0)))
        aligned.append(inst)
    return aligned


def _point_pvs_predict(model: Any, frame: np.ndarray, device: Any) -> Any:
    """Last-resort geometric prompts: center point + full-frame box."""
    height, width = int(frame.shape[0]), int(frame.shape[1])
    cx, cy = width / 2.0, height / 2.0
    try:
        return model.predict(
            source=frame,
            points=[[cx, cy]],
            labels=[1],
            verbose=False,
            device=device,
        )
    except TypeError:
        pass
    except Exception:
        pass
    try:
        return model.predict(
            source=frame,
            bboxes=[[0.0, 0.0, float(width), float(height)]],
            verbose=False,
            device=device,
        )
    except Exception:
        return None


def make_sam_predict_fn(
    model: Any,
    *,
    device: Any,
    boxes_by_index: dict[int, list[dict[str, Any]]] | None,
    semantic: Any | None,
    prompt_mode: str,
) -> PredictFn:
    frame_i = {"n": 0}
    names = {i: name for i, name in enumerate(TEXT_CONCEPTS)}

    def predict_fn(frame: np.ndarray) -> list[dict[str, Any]]:
        index = frame_i["n"]
        frame_i["n"] += 1
        height, width = int(frame.shape[0]), int(frame.shape[1])
        if boxes_by_index is not None:
            boxes = boxes_by_index.get(index) or []
            if not boxes:
                return []
            xyxy = _box_xyxy_list(boxes)
            if not xyxy:
                return []
            results = model.predict(
                source=frame,
                bboxes=xyxy,
                verbose=False,
                device=device,
            )
            result = results[0] if results else None
            return _instances_for_boxes(result, height, width, boxes)

        if semantic is not None:
            kwargs: dict[str, Any] = {"conf": 0.25, "text": list(TEXT_CONCEPTS)}
            results = semantic(frame, **kwargs)
            result = results[0] if isinstance(results, (list, tuple)) else results
            return instances_from_result(result, height, width, names)

        if prompt_mode == "point_pvs":
            results = _point_pvs_predict(model, frame, device)
            result = results[0] if results else None
            instances = instances_from_result(result, height, width, {0: "object"})
            # A whole-frame True from a full-frame box is a skip, not a fill.
            kept: list[dict[str, Any]] = []
            for inst in instances:
                mask = inst["mask"]
                if mask.any() and not bool(mask.all()):
                    kept.append(inst)
            return kept
        return []

    return predict_fn


def load_sam_backend(device: Any) -> tuple[Any, Path, Any | None, str]:
    SAM = import_sam()
    weights = require(MODELS_KEY)
    model = SAM(str(weights))
    predictor_cls = _semantic_predictor_cls()
    semantic = None
    if predictor_cls is not None:
        overrides = {
            "conf": 0.25,
            "task": "segment",
            "mode": "predict",
            "imgsz": 1008,
            "model": str(weights),
            "verbose": False,
        }
        semantic = predictor_cls(overrides=overrides)
        semantic.setup_model()
        if device != "cpu":
            try:
                semantic.args.device = device
            except Exception:
                pass
    return model, weights, semantic, SAM_NOTE


def run_sam31_clip(
    clip: Path | str,
    out_dir: Path | str,
    *,
    max_frames: int | None = None,
    boxes_from: Path | str | None = None,
) -> MapStream:
    from demo.pipeline.background_codec import read_video_frames_robust

    clip_path = Path(clip)
    frames = read_video_frames_robust(clip_path, max_frames=max_frames)
    if not frames:
        raise ValueError(f"no frames decoded from {clip_path}")

    device = _infer_device()
    model, weights, semantic, note = load_sam_backend(device)
    boxes_by_index: dict[int, list[dict[str, Any]]] | None = None
    if boxes_from is not None:
        boxes_by_index = load_boxes_from(boxes_from)
        prompt_mode = "boxes"
    elif semantic is not None:
        prompt_mode = "text:hand,tool"
    else:
        prompt_mode = "point_pvs"

    predict_fn = make_sam_predict_fn(
        model,
        device=device,
        boxes_by_index=boxes_by_index,
        semantic=semantic if boxes_by_index is None else None,
        prompt_mode=prompt_mode,
    )

    extra: dict[str, Any] = {
        "note": note,
        "temporal_id": False,
        "prompt_mode": prompt_mode,
        "models_key": MODELS_KEY,
        "frame_wise": True,
    }
    if boxes_from is not None:
        extra["boxes_from"] = str(Path(boxes_from))

    def profile_extract(frame: np.ndarray) -> Any:
        if prompt_mode == "text:hand,tool":
            return semantic(frame, conf=0.25, text=list(TEXT_CONCEPTS))
        return model.predict(source=frame, verbose=False, device=device)

    stream = write_mask_map(
        frames,
        out_dir,
        predict_fn=predict_fn,
        classes=list(TEXT_CONCEPTS),
        map_name=MAP_NAME,
        backend=weights.name,
        fps=_probe_fps(clip_path),
        extra=extra,
        profile_extract=profile_extract,
    )
    return stream


def write_sam31_map(
    frames: Sequence[np.ndarray],
    out_dir: Path | str,
    *,
    predict_fn: PredictFn,
    backend: str = "sam3.pt",
    fps: float = 30.0,
    extra: dict[str, Any] | None = None,
    n_warmup: int = 2,
    n_runs: int = 8,
) -> MapStream:
    """In-memory writer for tests. Does not load SAM."""
    merged = {"note": SAM_NOTE, "temporal_id": False, "frame_wise": True}
    if extra:
        merged.update(extra)
    return write_mask_map(
        frames,
        out_dir,
        predict_fn=predict_fn,
        classes=list(TEXT_CONCEPTS),
        map_name=MAP_NAME,
        backend=backend,
        fps=fps,
        extra=merged,
        n_warmup=n_warmup,
        n_runs=n_runs,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract SAM 3 quality masks as a native COCO-RLE maps-gallery stream."
    )
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory (masks.rle.zst, preview_masks/, sidecar.json)",
    )
    parser.add_argument(
        "--boxes-from",
        type=Path,
        default=None,
        dest="boxes_from",
        help="YOLOE sidecar.json or yoloe_masks/ directory",
    )
    parser.add_argument("--max-frames", type=int, default=None, dest="max_frames")
    args = parser.parse_args(argv)
    try:
        run_sam31_clip(
            args.clip,
            args.out,
            max_frames=args.max_frames,
            boxes_from=args.boxes_from,
        )
    except ImportError as exc:
        print(exc, file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
