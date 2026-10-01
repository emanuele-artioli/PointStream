"""Save the YOLOE hand-ladder intermediates for a few clip_03 frames.

Same load and ``model.track`` call as ``encode_seg_av1_ladder.render_yoloe_hands``.
Our code does not resize the clip before that call. Ultralytics letterboxes
inside ``BasePredictor.preprocess``; this script keeps that tensor.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from demo.pipeline.maps.yoloe_masks import (
    YOLOE_CONF,
    YOLOE_IMGSZ,
    class_masks_from_result,
    load_yoloe_model,
    paint_class_masks,
)

SAVE_AT = {0, 8, 15, 75, 180, 232}
LAST_FRAME = max(SAVE_AT)


def _as_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _model_input_bgr(tensor) -> np.ndarray:
    chw = tensor[0].detach().float().cpu().numpy()
    image = np.transpose(chw, (1, 2, 0))
    if float(image.max()) <= 1.5:
        image = image * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image[:, :, ::-1]


def _write(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write {path}")


def _boxes_panel(orig: np.ndarray, result) -> np.ndarray:
    canvas = orig.copy()
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        cv2.putText(canvas, "no boxes", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        return canvas
    xyxy = _as_numpy(boxes.xyxy)
    conf = _as_numpy(boxes.conf)
    cls = _as_numpy(boxes.cls)
    for index, (box, score, cls_id) in enumerate(zip(xyxy, conf, cls)):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(
            canvas,
            f"{index} cls={int(cls_id)} {float(score):.2f}",
            (x1, max(30, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
    return canvas


def _mask_panel(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask) > 0.5
    panel = np.zeros((*binary.shape, 3), dtype=np.uint8)
    panel[binary] = (255, 255, 255)
    return panel


def dump_frame(out: Path, index: int, result, model_input: np.ndarray | None) -> dict:
    frame_dir = out / f"frame_{index:06d}"
    orig = result.orig_img
    if orig is None:
        raise RuntimeError(f"frame {index} has no orig_img")
    _write(frame_dir / "1_source.png", orig)
    if model_input is not None:
        _write(frame_dir / "2_model_input.png", model_input)
    _write(frame_dir / "3_boxes.png", _boxes_panel(orig, result))

    masks = getattr(result, "masks", None)
    data = _as_numpy(masks.data) if masks is not None else None
    records = []
    if data is not None:
        for mask_index, mask in enumerate(data[:6]):
            _write(frame_dir / f"4_raw_mask_{mask_index}.png", _mask_panel(mask))
            records.append({"index": mask_index, "shape": list(mask.shape), "foreground": int((mask > 0.5).sum())})
    painted = paint_class_masks(
        class_masks_from_result(result, orig.shape[0], orig.shape[1], {0: "hand"}),
        orig.shape[0],
        orig.shape[1],
    )
    _write(frame_dir / "5_painted.png", painted)
    boxes = getattr(result, "boxes", None)
    summary = {
        "frame": index,
        "orig_shape": list(result.orig_shape) if result.orig_shape is not None else None,
        "model_input_shape": None if model_input is None else list(model_input.shape),
        "names": {str(k): str(v) for k, v in dict(getattr(result, "names", {}) or {}).items()},
        "n_boxes": 0 if boxes is None else len(boxes),
        "boxes": [],
        "raw_masks": records,
        "painted_foreground": int(np.any(painted > 0, axis=2).sum()),
    }
    if boxes is not None and len(boxes):
        for box, score, cls_id in zip(_as_numpy(boxes.xyxy), _as_numpy(boxes.conf), _as_numpy(boxes.cls)):
            summary["boxes"].append({
                "xyxy": [round(float(v), 1) for v in box],
                "conf": round(float(score), 4),
                "cls": int(cls_id),
            })
    (frame_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    return summary


def main() -> None:
    clip = Path("/home/itec/emanuele/tmp/maps-clips/clip_03.mp4")
    out = Path("/home/itec/emanuele/pointstream-data/jobs/yoloe-hand-debug")
    out.mkdir(parents=True, exist_ok=True)
    model, weights, extra = load_yoloe_model(["hand"])
    print("weights", weights, "extra", extra, "names", getattr(model, "names", None), flush=True)

    from ultralytics.engine.predictor import BasePredictor

    captured: dict[str, np.ndarray] = {}
    original = BasePredictor.preprocess

    def preprocess(self, im):
        tensor = original(self, im)
        captured["input"] = _model_input_bgr(tensor)
        captured["imgsz"] = list(getattr(self, "imgsz", []) or [])
        return tensor

    BasePredictor.preprocess = preprocess  # type: ignore[method-assign]
    summaries = []
    for index, result in enumerate(model.track(
        source=str(clip),
        stream=True,
        persist=True,
        retina_masks=True,
        verbose=False,
        device="cuda:0",
        half=True,
        imgsz=YOLOE_IMGSZ,
        conf=YOLOE_CONF,
    )):
        if index in SAVE_AT:
            summaries.append(dump_frame(out, index, result, captured.get("input")))
        if index >= LAST_FRAME:
            break
    (out / "summaries.json").write_text(json.dumps({
        "imgsz_arg": YOLOE_IMGSZ,
        "conf": YOLOE_CONF,
        "predictor_imgsz": captured.get("imgsz"),
        "frames": summaries,
    }, indent=2) + "\n")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
