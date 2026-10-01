"""RTMPose-m on frames with everything outside the hand/arm mask removed.

Writes poses_segmented.json and pose_mask_compare.json beside each sampled
folder. The full-frame poses.json is left in place.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from demo.evaluation.pose_backends import _cuda_onnx_ok
from demo.experiments.pose_sampled_frames import hands_for
from demo.pipeline.hand_keypoints import SingleHand

DEST = Path("/home/itec/emanuele/Datasets/pointstream-demo")
FOLDERS = (
    "clip_01_factory001_worker001_00001/f000000-f035129",
    "clip_03_factory001_worker001_00000/f000000-f012629",
    "factory002_worker001_00000/f000000-f035129",
)


def box_iou(a: list[int], b: list[int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter) / float(area) if area else 0.0


def row_for(hands: list[SingleHand], name: str, idx: int) -> dict:
    return {
        "frame_idx": idx,
        "file": name,
        "hands": [
            {
                "handedness": h.handedness,
                "confidence": h.confidence,
                "bbox": h.bbox,
                "landmarks_norm": h.landmarks_norm,
                "landmarks_pixel": h.landmarks_pixel,
            }
            for h in hands
        ],
    }


def compare(folder: Path, model) -> None:
    original = folder / "original"
    masks = folder / "masks"
    full = json.loads((folder / "poses.json").read_text())
    frames = sorted(original.glob("*.jpg"))
    segmented = []
    same = gained = lost = moved = 0
    for idx, path in enumerate(frames):
        image = cv2.imread(str(path))
        mask = cv2.imread(str(masks / f"{path.stem}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            segmented.append(row_for([], path.name, idx))
            continue
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        kept = image.copy()
        kept[mask <= 8] = 0
        hands = hands_for(model, kept)
        segmented.append(row_for(hands, path.name, idx))
        before = full[idx]["hands"] if idx < len(full) else []
        if len(hands) == len(before):
            matched = 0
            used = set()
            for hand in hands:
                best_i, best = -1, 0.0
                for j, old in enumerate(before):
                    if j in used:
                        continue
                    score = box_iou(hand.bbox, old["bbox"])
                    if score > best:
                        best, best_i = score, j
                if best >= 0.5:
                    matched += 1
                    used.add(best_i)
            if matched == len(hands):
                same += 1
            else:
                moved += 1
        elif len(hands) > len(before):
            gained += 1
        else:
            lost += 1
        if idx % 100 == 0:
            print("pose", folder.parent.name, idx, len(before), len(hands), flush=True)
    (folder / "poses_segmented.json").write_text(json.dumps(segmented))
    report = {
        "frames": len(frames),
        "same_boxes": same,
        "moved_boxes": moved,
        "gained_hands": gained,
        "lost_hands": lost,
        "full_hands": sum(len(row["hands"]) for row in full),
        "segmented_hands": sum(len(row["hands"]) for row in segmented),
    }
    (folder / "pose_mask_compare.json").write_text(json.dumps(report, indent=2))
    print("WROTE", folder / "pose_mask_compare.json", report, flush=True)


def main() -> None:
    from rtmlib import Hand

    device = "cuda" if _cuda_onnx_ok() else "cpu"
    model = Hand(mode="lightweight", to_openpose=False, backend="onnxruntime", device=device)
    for rel in FOLDERS:
        compare(DEST / rel, model)


if __name__ == "__main__":
    main()
