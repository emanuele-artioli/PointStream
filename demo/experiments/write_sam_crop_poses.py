"""Write every pose model's joints for each SAM crop into the dataset folder.

The comparison report kept scores and dropped coordinates. This pass keeps
both hands when a whole-body model returns two, and marks the one whose
joints fall furthest inside the mask.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from demo.experiments.compare_pose_on_sam_crops import (
    AISLE,
    FOLDERS,
    LOOK,
    components,
    expand_box,
    in_ranges,
    inside_fraction,
    parse_people,
)
from demo.experiments.pose_crop_gallery import load_models

FRAME_W = 1920
FRAME_H = 1080


def hands_for(model, image, mask, box) -> list[dict]:
    keypoints, scores = model(image, [box])
    parsed = parse_people(keypoints, scores)
    rows = []
    for side, pts, conf in parsed:
        if len(pts) < 21 or len(conf) < 21:
            continue
        points = [[float(p[0]), float(p[1])] for p in pts[:21]]
        mean_conf = float(conf[:21].mean())
        inside = inside_fraction(pts[:21], mask)
        span = float(max(pts[:21, 0]) - min(pts[:21, 0])) + float(max(pts[:21, 1]) - min(pts[:21, 1]))
        rows.append({
            "side": side,
            "confidence": mean_conf,
            "inside": inside,
            "span_px": span,
            "landmarks_pixel": points,
            "landmarks_norm": [[p[0] / FRAME_W, p[1] / FRAME_H, 0.0] for p in points],
        })
    if rows:
        best = max(range(len(rows)), key=lambda i: (rows[i]["inside"], rows[i]["confidence"]))
        for i, row in enumerate(rows):
            row["selected"] = i == best
    return rows


def main() -> None:
    models = load_models()
    for clip, folder in FOLDERS.items():
        original = folder / "original"
        masks = folder / "masks"
        by_model = {name: [] for name in models}
        for path in sorted(original.glob("*.jpg")):
            frame = int(path.stem)
            image = cv2.imread(str(path))
            mask = cv2.imread(str(masks / f"{path.stem}.png"), cv2.IMREAD_GRAYSCALE)
            crops = []
            if image is not None and mask is not None:
                if mask.shape[:2] != image.shape[:2]:
                    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                crops = components(mask)
            base = {
                "frame_idx": frame,
                "file": path.name,
                "aisle": clip in AISLE and in_ranges(frame, AISLE[clip]),
                "look": clip in LOOK and in_ranges(frame, LOOK[clip]),
            }
            found = {name: [] for name in models}
            for comp in crops:
                box = expand_box(comp, image.shape[1], image.shape[0])
                crop_meta = {"box": box, "area": comp["area"]}
                for name, model in models.items():
                    for hand in hands_for(model, image, mask, box):
                        found[name].append({**crop_meta, **hand})
            for name in models:
                by_model[name].append({**base, "hands": found[name]})
            if frame % 300 == 0:
                print(clip, frame, {name: len(found[name]) for name in models}, flush=True)
        dest = folder / "sam_poses"
        dest.mkdir(parents=True, exist_ok=True)
        for name, rows in by_model.items():
            (dest / f"{name}.json").write_text(json.dumps(rows))
            print("WROTE", dest / f"{name}.json", len(rows), flush=True)
        (dest / "manifest.json").write_text(json.dumps({
            "schema": "pointstream.sam_crop_poses.v1",
            "clip": clip,
            "models": list(models),
            "selected_means": "the hand whose joints land furthest inside this SAM component",
            "train_with": "rtmw-l.json",
            "note": "Joints are in the original frame. aisle and look frames are included and tagged.",
        }, indent=2) + "\n")


if __name__ == "__main__":
    main()
