"""Gallery and per-model timing for the SAM-crop pose comparison.

Reads an existing report, re-runs only the frames shown in the gallery, and
times each pose head on its own. Columns are separate models. The crop is
zoomed, pixels outside the SAM mask are dimmed, and each finger has its own
color.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from demo.experiments.compare_pose_on_sam_crops import (
    DWPOSE_L,
    FOLDERS,
    HAND_ONNX,
    RTM_L,
    RTMW_L,
    VITPOSE_L,
    choose_hand,
    parse_people,
)
from demo.pipeline.hand_keypoints import FINGER_COLORS, HAND_CONNECTIONS

ORDER = ("rtmpose-m", "rtmpose-l", "dwpose-l", "rtmw-l", "vitpose-l")
BONE_COLOR = []
for start, _end in HAND_CONNECTIONS:
    if start < 5:
        BONE_COLOR.append(FINGER_COLORS[0])
    elif start < 9:
        BONE_COLOR.append(FINGER_COLORS[1])
    elif start < 13:
        BONE_COLOR.append(FINGER_COLORS[2])
    elif start < 17:
        BONE_COLOR.append(FINGER_COLORS[3])
    else:
        BONE_COLOR.append(FINGER_COLORS[4])


def load_models():
    from rtmlib import RTMPose, ViTPose

    specs = {
        "rtmpose-m": (RTMPose, HAND_ONNX, (256, 256)),
        "rtmpose-l": (RTMPose, RTM_L, (288, 384)),
        "dwpose-l": (RTMPose, DWPOSE_L, (288, 384)),
        "rtmw-l": (RTMPose, RTMW_L, (288, 384)),
        "vitpose-l": (ViTPose, VITPOSE_L, None),
    }
    models = {}
    for name, (cls, path, size) in specs.items():
        if size is None:
            models[name] = cls(str(path), to_openpose=False, device="cuda")
        else:
            models[name] = cls(
                str(path),
                model_input_size=size,
                to_openpose=False,
                backend="onnxruntime",
                device="cuda",
            )
        print("loaded", name, flush=True)
    return models


def pick_rows(rows: list[dict]) -> list[dict]:
    """Two moments per clip, far apart, plus two frames where the models disagree."""
    chosen = []
    used = set()
    press = [row for row in rows if not row["look"]]
    for clip in ("clip_01", "clip_03", "factory002"):
        pool = [row for row in press if row["clip"] == clip and row["area"] >= 8000]
        pool.sort(key=lambda row: row["frame"])
        if len(pool) < 2:
            continue
        for row in (pool[len(pool) // 5], pool[(4 * len(pool)) // 5]):
            key = (row["clip"], row["frame"], tuple(row["box"]))
            if key not in used:
                used.add(key)
                chosen.append(row)
    disagree = [
        row for row in press
        if row["area"] >= 8000
        and row["models"].get("vitpose-l", {}).get("inside", 0) >= 0.8
        and row["models"].get("rtmpose-m", {}).get("inside", 1) <= 0.35
    ]
    disagree.sort(key=lambda row: row["frame"])
    for row in disagree[:: max(1, len(disagree) // 2)][:2]:
        key = (row["clip"], row["frame"], tuple(row["box"]))
        if key not in used:
            used.add(key)
            chosen.append(row)
    return chosen


def draw_skeleton(canvas: np.ndarray, points: list[list[float]], origin, scale: float) -> None:
    ox, oy = origin
    mapped = []
    for x, y in points:
        mapped.append((int((x - ox) * scale), int((y - oy) * scale)))
    for (start, end), color in zip(HAND_CONNECTIONS, BONE_COLOR):
        if start >= len(mapped) or end >= len(mapped):
            continue
        cv2.line(canvas, mapped[start], mapped[end], color, 2, cv2.LINE_AA)
    for x, y in mapped:
        cv2.circle(canvas, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 3, (0, 0, 0), 1, cv2.LINE_AA)


def panel(image, mask, box, hand, title: str, size: int = 320) -> np.ndarray:
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    crop = image[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((size, size, 3), np.uint8)
    else:
        dim = crop.copy()
        dim[crop_mask <= 8] = (dim[crop_mask <= 8] * 0.35).astype(np.uint8)
        crop = dim
    height, width = crop.shape[:2]
    scale = size / max(height, width, 1)
    resized = cv2.resize(crop, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 3), np.uint8)
    top = (size - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    if hand and hand.get("points"):
        draw_skeleton(canvas, hand["points"], (x1 - left / scale, y1 - top / scale), scale)
    bar = np.zeros((36, size, 3), np.uint8)
    inside = hand["inside"] if hand else 0.0
    text = f"{title}  {int(round(inside * 21))}/21 in mask"
    cv2.putText(bar, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, canvas], axis=0)


def legend(width: int) -> np.ndarray:
    bar = np.full((48, width, 3), 24, np.uint8)
    labels = [("thumb", FINGER_COLORS[0]), ("index", FINGER_COLORS[1]), ("middle", FINGER_COLORS[2]), ("ring", FINGER_COLORS[3]), ("pinky", FINGER_COLORS[4])]
    x = 12
    cv2.putText(bar, "SAM crop, dimmed outside the mask", (x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    x = 430
    for name, color in labels:
        cv2.rectangle(bar, (x, 14), (x + 18, 32), color, -1)
        cv2.putText(bar, name, (x + 24, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        x += 110
    return bar


def read_crop(row: dict):
    folder = FOLDERS[row["clip"]]
    image = cv2.imread(str(folder / "original" / row["file"]))
    mask = cv2.imread(str(folder / "masks" / f"{Path(row['file']).stem}.png"), cv2.IMREAD_GRAYSCALE)
    return image, mask


def infer(models, image, mask, box) -> dict:
    found = {}
    for name, model in models.items():
        keypoints, scores = model(image, [box])
        hand = choose_hand(parse_people(keypoints, scores), mask)
        found[name] = hand
    return found


def time_models(models, samples: list[tuple]) -> dict:
    timings = {}
    for name, model in models.items():
        for image, _mask, box in samples[:3]:
            model(image, [box])
        started = time.perf_counter()
        count = 0
        for image, _mask, box in samples:
            model(image, [box])
            count += 1
        elapsed = time.perf_counter() - started
        timings[name] = {
            "device": "cpu",
            "crops": count,
            "ms_per_crop": 1000.0 * elapsed / count,
            "note": "ONNX Runtime fell back to CPU. This host's build requires cuDNN 9, which is not installed.",
        }
        print("time", name, timings[name]["ms_per_crop"], flush=True)
    return timings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--time-crops", type=int, default=40)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    rows = pick_rows(report["crops"])
    print("gallery rows", len(rows), flush=True)
    models = load_models()
    strips = []
    shown = []
    for row in rows:
        image, mask = read_crop(row)
        if image is None or mask is None:
            continue
        hands = infer(models, image, mask, row["box"])
        cells = [panel(image, mask, row["box"], hands[name], name) for name in ORDER]
        label = np.full((cells[0].shape[0], 150, 3), 16, np.uint8)
        second = row["frame"] / 30.0
        cv2.putText(label, row["clip"], (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(label, f"t={second:.0f}s", (8, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        strips.append(np.concatenate([label, *cells], axis=1))
        shown.append({"clip": row["clip"], "frame": row["frame"], "second": second, "box": row["box"], "area": row["area"]})
    gallery = np.concatenate([legend(strips[0].shape[1]), *strips], axis=0)
    args.out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out / "gallery.jpg"), gallery, [cv2.IMWRITE_JPEG_QUALITY, 90])

    timed = []
    for row in report["crops"]:
        if row["look"] or row["area"] < 8000:
            continue
        image, mask = read_crop(row)
        if image is None:
            continue
        timed.append((image, mask, row["box"]))
        if len(timed) >= args.time_crops:
            break
    timing = time_models(models, timed)
    payload = {
        "schema": "pointstream.sam_crop_pose_gallery.v1",
        "source_report": str(args.report),
        "shown": shown,
        "timing_ms_per_crop": timing,
        "gallery": str(args.out / "gallery.jpg"),
    }
    (args.out / "gallery.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("WROTE", args.out / "gallery.jpg", flush=True)


if __name__ == "__main__":
    main()
