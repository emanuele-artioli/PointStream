"""Pose heads on tight SAM hand/arm crops.

The box comes from a mask component, not from the pose detector. Whole-body
models receive that box as their only person box. Every joint score is kept,
including values below 0.25. A crop is a miss when the chosen hand's joints
do not land inside the mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

DEST = Path("/home/itec/emanuele/Datasets/pointstream-demo")
FOLDERS = {
    "clip_01": DEST / "clip_01_factory001_worker001_00001/f000000-f035129",
    "clip_03": DEST / "clip_03_factory001_worker001_00000/f000000-f012629",
    "factory002": DEST / "factory002_worker001_00000/f000000-f035129",
}
# Clip 3 source times, in frames at 30 fps. 210 s and 240 s are the aisle.
AISLE = {"clip_03": ((6300, 6600), (7200, 7500))}
LOOK = {"clip_03": ((0, 30), (12600, 12630))}
MIN_AREA = 1500
MARGIN = 0.2
LEFT = slice(91, 112)
RIGHT = slice(112, 133)

HAND_ONNX = Path(
    "/home/itec/emanuele/.cache/rtmlib/hub/checkpoints/"
    "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.onnx"
)
RTM_L = Path(
    "/home/itec/emanuele/pointstream-data/weights/rtm_scales/rtmpose-l-384/"
    "20230831/rtmpose_onnx/rtmpose-l_simcc-ucoco_dw-ucoco_270e-384x288-2438fd99_20230728/end2end.onnx"
)
RTMW_L = Path("/home/itec/emanuele/pointstream-data/weights/rtm_scales/rtmw-l-384/end2end.onnx")
DWPOSE_L = Path("/home/itec/emanuele/Models/DWPose/dw-ll_ucoco_384.onnx")
VITPOSE_L = Path(
    "/home/itec/emanuele/pointstream-data/weights/vitpose/onnx/wholebody/vitpose-l-wholebody.onnx"
)


def components(mask: np.ndarray, min_area: int = MIN_AREA) -> list[dict]:
    binary = (mask > 8).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    found = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x, y, w, h = (int(v) for v in stats[index, :4])
        found.append({"label": index, "x": x, "y": y, "w": w, "h": h, "area": area, "labels": labels})
    return found


def expand_box(comp: dict, width: int, height: int, margin: float = MARGIN) -> list[float]:
    pad_x = comp["w"] * margin
    pad_y = comp["h"] * margin
    x1 = max(0.0, comp["x"] - pad_x)
    y1 = max(0.0, comp["y"] - pad_y)
    x2 = min(float(width), comp["x"] + comp["w"] + pad_x)
    y2 = min(float(height), comp["y"] + comp["h"] + pad_y)
    return [x1, y1, x2, y2]


def inside_fraction(points: np.ndarray, mask: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    height, width = mask.shape[:2]
    hits = 0
    for x, y in points:
        xi, yi = int(round(float(x))), int(round(float(y)))
        if 0 <= xi < width and 0 <= yi < height and mask[yi, xi] > 8:
            hits += 1
    return hits / float(len(points))


def parse_people(keypoints, scores) -> list[tuple[str, np.ndarray, np.ndarray]]:
    keypoints = np.asarray(keypoints)
    scores = np.asarray(scores) if scores is not None else np.ones(keypoints.shape[:-1])
    if keypoints.size == 0:
        return []
    if keypoints.ndim == 2:
        keypoints = keypoints[None, ...]
        if scores.ndim == 1:
            scores = scores[None, ...]
    hands = []
    for person, person_scores in zip(keypoints, scores):
        if person.shape[0] >= 133:
            for side, sl in (("Left", LEFT), ("Right", RIGHT)):
                pts = np.asarray(person[sl])[:, :2]
                conf = np.asarray(person_scores[sl]).reshape(-1)[:21]
                hands.append((side, pts, conf))
        elif person.shape[0] >= 21:
            pts = np.asarray(person[:21])[:, :2]
            conf = np.asarray(person_scores).reshape(-1)[:21]
            hands.append(("Unknown", pts, conf))
    return hands


def choose_hand(hands, mask: np.ndarray) -> dict | None:
    best = None
    for side, pts, conf in hands:
        if len(pts) < 21 or len(conf) < 21:
            continue
        inside = inside_fraction(pts[:21], mask)
        mean_conf = float(np.mean(conf[:21]))
        span = float(max(pts[:21, 0]) - min(pts[:21, 0])) + float(max(pts[:21, 1]) - min(pts[:21, 1]))
        row = {
            "side": side,
            "inside": inside,
            "confidence": mean_conf,
            "span_px": span,
            "points": [[float(p[0]), float(p[1])] for p in pts[:21]],
        }
        if best is None or (inside, mean_conf) > (best["inside"], best["confidence"]):
            best = row
    return best


def in_ranges(frame: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= frame < stop for start, stop in ranges)


def iter_crops(max_frames: int | None):
    seen = 0
    for clip, folder in FOLDERS.items():
        original = folder / "original"
        masks = folder / "masks"
        for path in sorted(original.glob("*.jpg")):
            frame = int(path.stem)
            if clip in AISLE and in_ranges(frame, AISLE[clip]):
                continue
            mask = cv2.imread(str(masks / f"{path.stem}.png"), cv2.IMREAD_GRAYSCALE)
            image = cv2.imread(str(path))
            if mask is None or image is None:
                continue
            if mask.shape[:2] != image.shape[:2]:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
            for comp in components(mask):
                yield {
                    "clip": clip,
                    "frame": frame,
                    "file": path.name,
                    "look": clip in LOOK and in_ranges(frame, LOOK[clip]),
                    "area": comp["area"],
                    "box": expand_box(comp, image.shape[1], image.shape[0]),
                    "image": image,
                    "mask": mask,
                }
                seen += 1
                if max_frames is not None and seen >= max_frames:
                    return


def summarize(rows: list[dict], press_only: bool) -> dict:
    chosen = [row for row in rows if not (press_only and row["look"])]
    by_model: dict[str, dict] = {}
    names = sorted({name for row in chosen for name in row["models"]})
    for name in names:
        picked = [row["models"][name] for row in chosen if row["models"].get(name)]
        inside = [item["inside"] for item in picked]
        conf = [item["confidence"] for item in picked]
        by_model[name] = {
            "crops": len(chosen),
            "returned": len(picked),
            "inside_half": sum(value >= 0.5 for value in inside),
            "inside_half_and_conf_025": sum(
                item["inside"] >= 0.5 and item["confidence"] >= 0.25 for item in picked
            ),
            "mean_inside": float(np.mean(inside)) if inside else 0.0,
            "mean_confidence": float(np.mean(conf)) if conf else 0.0,
            "mean_span_px": float(np.mean([item["span_px"] for item in picked])) if picked else 0.0,
        }
    ranking = sorted(
        by_model,
        key=lambda name: (
            by_model[name]["inside_half"],
            by_model[name]["mean_inside"],
            by_model[name]["mean_confidence"],
        ),
        reverse=True,
    )
    return {"crops": len(chosen), "models": by_model, "ranking": ranking}


def draw_sheet(samples: list[dict], dest: Path) -> None:
    if not samples:
        return
    panels = []
    for sample in samples:
        canvas = sample["image"].copy()
        x1, y1, x2, y2 = (int(v) for v in sample["box"])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        colors = {
            "rtmpose-m": (0, 255, 255),
            "rtmpose-l": (255, 128, 0),
            "dwpose-l": (255, 0, 255),
            "rtmw-l": (0, 128, 255),
            "vitpose-l": (255, 255, 0),
        }
        for name, hand in sample["models"].items():
            if not hand:
                continue
            color = colors.get(name, (255, 255, 255))
            for x, y in hand["points"]:
                cv2.circle(canvas, (int(x), int(y)), 3, color, -1)
        canvas = cv2.resize(canvas, (640, 360), interpolation=cv2.INTER_AREA)
        panels.append(canvas)
    cv2.imwrite(str(dest), np.concatenate(panels, axis=0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-crops", type=int, default=None)
    args = parser.parse_args()
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
        if not path.is_file():
            raise FileNotFoundError(path)
        kwargs = {"to_openpose": False, "backend": "onnxruntime", "device": "cuda"}
        if size is not None:
            kwargs["model_input_size"] = size
        else:
            kwargs = {"to_openpose": False, "device": "cuda"}
        models[name] = cls(str(path), **kwargs)
        print("loaded", name, path, flush=True)

    rows = []
    sheets = []
    for crop in iter_crops(args.max_crops):
        record = {
            "clip": crop["clip"],
            "frame": crop["frame"],
            "file": crop["file"],
            "look": crop["look"],
            "area": crop["area"],
            "box": crop["box"],
            "models": {},
        }
        for name, model in models.items():
            keypoints, scores = model(crop["image"], [crop["box"]])
            hand = choose_hand(parse_people(keypoints, scores), crop["mask"])
            if hand is not None:
                record["models"][name] = {key: value for key, value in hand.items() if key != "points"}
                record["models"][name]["points"] = hand["points"]
        rows.append(record)
        if len(sheets) < 8 and not crop["look"]:
            sheets.append(record | {"image": crop["image"], "box": crop["box"]})
        if len(rows) % 50 == 0:
            print("crops", len(rows), crop["clip"], crop["frame"], flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    draw_sheet(sheets, args.out / "sheet.jpg")
    stored = []
    for row in rows:
        slim = dict(row)
        slim["models"] = {
            name: {key: value for key, value in hand.items() if key != "points"}
            for name, hand in row["models"].items()
        }
        stored.append(slim)
    report = {
        "schema": "pointstream.sam_crop_pose.v1",
        "min_area_px": MIN_AREA,
        "margin": MARGIN,
        "excluded_clip3_aisle_frames": list(AISLE["clip_03"]),
        "press": summarize(rows, press_only=True),
        "including_clip3_0s_and_420s": summarize(rows, press_only=False),
        "crops": stored,
    }
    (args.out / "report.json").write_text(json.dumps(report))
    press = report["press"]
    print("RANK", press["ranking"], flush=True)
    for name in press["ranking"]:
        print(name, press["models"][name], flush=True)
    print("WROTE", args.out / "report.json", flush=True)


if __name__ == "__main__":
    main()
