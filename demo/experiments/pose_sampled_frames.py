"""RTMPose-m on the sampled original frames.

Drops the wide short border strips. Writes poses.json beside each finished
original folder. Waits while a folder is still being copied.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from demo.evaluation.pose_backends import _cuda_onnx_ok, is_border_strip
from demo.pipeline.hand_keypoints import FrameHandPose, SingleHand

DEST = Path("/home/itec/emanuele/Datasets/pointstream-demo")
EXPECTED = {
    "clip_01_factory001_worker001_00001/f000000-f035129": 1080,
    "clip_03_factory001_worker001_00000/f000000-f012629": 450,
    "factory002_worker001_00000": None,
}


def hands_for(model, frame: np.ndarray) -> list[SingleHand]:
    if (frame.shape[1], frame.shape[0]) != (1920, 1080):
        frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
    kpts, scores = model(frame)
    if kpts is None:
        return []
    kpts = np.asarray(kpts)
    scores = np.asarray(scores) if scores is not None else np.ones(kpts.shape[:2])
    if kpts.ndim == 2:
        kpts = kpts[None, ...]
        scores = scores[None, ...] if scores.ndim == 1 else scores
    hands = []
    for hand_i, pts in enumerate(kpts):
        if pts.shape[0] < 21:
            continue
        conf = float(np.mean(scores[hand_i][:21])) if scores.ndim >= 2 else float(np.mean(scores))
        if conf < 0.25:
            continue
        xs, ys = pts[:21, 0], pts[:21, 1]
        if is_border_strip(xs, ys, frame_w=frame.shape[1], frame_h=frame.shape[0]):
            continue
        pad_x = (float(xs.max()) - float(xs.min())) * 0.25
        pad_y = (float(ys.max()) - float(ys.min())) * 0.25
        lms_px = [[float(p[0]), float(p[1])] for p in pts[:21]]
        hands.append(SingleHand(
            handedness="Unknown",
            confidence=conf,
            bbox=[
                int(max(0, xs.min() - pad_x)),
                int(max(0, ys.min() - pad_y)),
                int(min(1920, xs.max() + pad_x)),
                int(min(1080, ys.max() + pad_y)),
            ],
            landmarks_norm=[[p[0] / 1920.0, p[1] / 1080.0, 0.0] for p in lms_px],
            landmarks_pixel=lms_px,
        ))
    return hands


def folder_for(rel: str) -> Path | None:
    direct = DEST / rel / "original"
    if direct.is_dir():
        return direct
    root = DEST / rel
    found = sorted(p for p in root.glob("f*/original") if p.is_dir())
    return found[-1] if found else None


def main() -> None:
    from rtmlib import Hand

    device = "cuda" if _cuda_onnx_ok() else "cpu"
    model = Hand(mode="lightweight", to_openpose=False, backend="onnxruntime", device=device)
    deadline = time.monotonic() + 6 * 3600
    for rel, expected in EXPECTED.items():
        while True:
            folder = folder_for(rel)
            paths = sorted(folder.glob("*.jpg")) if folder else []
            if folder and (expected is None or len(paths) >= expected) and len(paths) >= 30:
                break
            if time.monotonic() > deadline:
                raise SystemExit(f"frames missing for {rel}")
            print("waiting", rel, len(paths), flush=True)
            time.sleep(30)
        out = folder.parent / "poses.json"
        rows = []
        for idx, path in enumerate(paths):
            frame = cv2.imread(str(path))
            pose = FrameHandPose(frame_idx=idx, hands=hands_for(model, frame) if frame is not None else [])
            rows.append({
                "frame_idx": idx,
                "file": path.name,
                "hands": [
                    {
                        "handedness": h.handedness,
                        "confidence": h.confidence,
                        "bbox": h.bbox,
                        "landmarks_norm": h.landmarks_norm,
                        "landmarks_pixel": h.landmarks_pixel,
                    }
                    for h in pose.hands
                ],
            })
            if idx % 100 == 0:
                print("pose", rel, idx, len(pose.hands), flush=True)
        out.write_text(json.dumps(rows))
        print("WROTE", out, "frames", len(rows), flush=True)


if __name__ == "__main__":
    main()
