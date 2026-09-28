"""Archived clip-1 comparison of hand RTMPose-m with RTMPose-l, RTMW-l, and DWPose-l.

Hand RTMPose-m stayed the encoder and the judge. This script remains for ablations.

Compare the hand RTMPose-m encoder with RTMPose-l, RTMW-l, and DWPose-l.

Eight frames of clip 1. Whole-body models are person-cropped with the same YOLOX
detector, then the 21 COCO-WholeBody hand joints are packed with the shipped
keypoint compressor. The hand model keeps its own hand detector.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.experiments.archive.compare_vitpose_densepose_clip1 import (
    CLIP,
    draw_pose,
    encoder_report,
    judge_matrix,
    montage,
    pose_from_hands,
    read_frames,
    sample_clip,
)
from demo.pipeline.hand_keypoints import FrameHandPose

logger = logging.getLogger("rtm_scale_compare")

WEIGHTS = Path("/home/itec/emanuele/pointstream-data/weights/rtm_scales")
LEFT_HAND = slice(91, 112)
RIGHT_HAND = slice(112, 133)
FRAME_W = 1920
FRAME_H = 1080


def first_onnx(root: Path) -> Path:
    found = sorted(root.rglob("*.onnx"))
    if not found:
        raise FileNotFoundError(root)
    return found[0]


def person_boxes(detector, frame: np.ndarray) -> list[list[float]]:
    boxes = detector(frame)
    if boxes is None or len(np.asarray(boxes)) == 0:
        return []
    return [box.tolist() for box in np.asarray(boxes).reshape(-1, 4)]


def wholebody_hands(model, detector, frames: list[np.ndarray], name: str) -> tuple[list[FrameHandPose], int]:
    poses: list[FrameHandPose] = []
    fallback = 0
    for index, frame in enumerate(frames):
        boxes = person_boxes(detector, frame)
        if not boxes:
            boxes = [[0, 0, frame.shape[1], frame.shape[0]]]
            fallback += 1
        keypoints, scores = model(frame, boxes)
        keypoints = np.asarray(keypoints)
        scores = np.asarray(scores)
        if keypoints.ndim == 2:
            keypoints = keypoints[None, ...]
            scores = scores[None, ...]
        hands: list[tuple[str, np.ndarray]] = []
        for person, person_scores in zip(keypoints, scores):
            if person.shape[0] < 133:
                raise RuntimeError(f"{name} returned {person.shape[0]} joints, expected 133")
            for side, sl in (("Left", LEFT_HAND), ("Right", RIGHT_HAND)):
                pts = person[sl]
                conf = person_scores[sl]
                if float(np.mean(conf)) < 0.25:
                    continue
                hands.append((side, pts[:, :2]))
        poses.append(pose_from_hands(index, hands))
    logger.info("%s full-frame fallback on %s/%s frames", name, fallback, len(frames))
    return poses, fallback


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, default=CLIP)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    from demo.evaluation.pose_backends import extract_rtm_hand
    from demo.pipeline.maps.model_paths import require
    from rtmlib import RTMPose, YOLOX

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    sampled = out / "clip1_sample.mp4"
    sample_clip(args.video, sampled, args.frames)
    frames = read_frames(sampled)

    specs = {
        "rtmpose-l": (first_onnx(WEIGHTS / "rtmpose-l-384"), (288, 384)),
        "rtmw-l": (first_onnx(WEIGHTS / "rtmw-l-384"), (288, 384)),
        "dwpose-l": (require("dwpose_pose"), (288, 384)),
    }
    detector = YOLOX(str(require("dwpose_det")), det_mode="human", score_thr=0.3, device="cuda")
    logger.info("running hand RTMPose-m")
    named = {"rtmpose-m": extract_rtm_hand(sampled)}
    fallbacks = {}
    for name, (path, size) in specs.items():
        logger.info("running %s from %s", name, path)
        model = RTMPose(str(path), model_input_size=size, to_openpose=False, backend="onnxruntime", device="cuda")
        named[name], fallbacks[name] = wholebody_hands(model, detector, frames, name)
        del model

    columns = [
        (name, [draw_pose(frame, pose, name) for frame, pose in zip(frames, poses)])
        for name, poses in named.items()
    ]
    import cv2

    sheets = out / "sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    for label, panels in columns:
        strip = np.concatenate(
            [cv2.resize(panel, (480, 270), interpolation=cv2.INTER_AREA) for panel in panels],
            axis=0,
        )
        cv2.imwrite(str(sheets / f"{label}.png"), strip)
    montage_path = out / "montage.png"
    montage(columns, montage_path)
    report = {
        "clip": str(args.video),
        "frames": len(frames),
        "checkpoints": {name: str(path) for name, (path, _) in specs.items()},
        "hand_counts": {name: sum(len(pose.hands) for pose in poses) for name, poses in named.items()},
        "full_frame_fallbacks": fallbacks,
        "encoders": {name: encoder_report(poses) for name, poses in named.items()},
        "judges": judge_matrix(named),
        "notes": {
            "rtmpose-m": "rtmlib Hand lightweight: RTMDet-nano hand detector and rtmpose-m Hand5, 21 joints. Not a whole-body model.",
            "wholebody": "RTMPose-l, RTMW-l, and DWPose-l share YOLOX-l person boxes. Hands are COCO-WholeBody joints 91:112 and 112:133.",
            "rtmpose-l": "Public ONNX is rtmpose-l ucoco_dw 384x288. The earlier coco-wholebody RTMPose-l has no ONNX zip.",
            "rtmw-l": "rtmw-dw-x-l cocktail14 384x288, the RTMW-l release at 70.1 whole-body AP.",
            "dwpose-l": "Local dw-ll_ucoco_384.onnx.",
            "rate": "keypoint_kbps is the 21-point compressor at 30 fps, the same packet the encoder sends.",
        },
        "montage": str(montage_path),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    logger.info("wrote %s", out / "report.json")


if __name__ == "__main__":
    main()
