"""Offline and live hand-pose backends used as GT or reconstruction evaluators."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from demo.pipeline.hand_keypoints import (
    FrameHandPose,
    HandPoseEstimator,
    HandTrajectoryFilter,
    SingleHand,
)

logger = logging.getLogger(__name__)

PoseExtractor = Callable[[Path, int | None], list[FrameHandPose]]


def extract_mediapipe_live(video_path: Path, max_frames: int | None = None) -> list[FrameHandPose]:
    """Realtime encoder settings: video-mode MediaPipe Hands, complexity 1, smoothed."""
    from demo.pipeline.hand_keypoints import extract_video_hand_poses

    return extract_video_hand_poses(video_path, max_frames=max_frames, smooth=True)


def extract_mediapipe_offline_gt(video_path: Path, max_frames: int | None = None) -> list[FrameHandPose]:
    """Denser MediaPipe pass: video mode, lower det threshold, no 1-Euro smoothing.

    ``static_image_mode=True`` and ``model_complexity=2`` abort this Mediapipe
    0.10.5 build (C2__PACKET). Those knobs are intentionally not used.
    """
    return _run_mediapipe_video(
        video_path,
        max_frames=max_frames,
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
        smooth=False,
    )


_RTM_HAND = None


def extract_rtm_hand(video_path: Path, max_frames: int | None = None) -> list[FrameHandPose]:
    """RTMPose hand (rtmlib). Same 21-point count; topology is COCO-WholeBody, not MediaPipe."""
    global _RTM_HAND
    try:
        from rtmlib import Hand
    except ImportError as exc:
        raise RuntimeError("rtmlib is not installed. pip install rtmlib") from exc

    from demo.pipeline.background_codec import read_video_frames_robust

    if _RTM_HAND is None:
        device = "cpu"
        if _cuda_onnx_ok():
            device = "cuda"
        _RTM_HAND = Hand(mode="lightweight", to_openpose=False, backend="onnxruntime", device=device)
    model = _RTM_HAND
    frames = read_video_frames_robust(video_path, max_frames=max_frames)
    poses: list[FrameHandPose] = []
    for idx, frame in enumerate(frames):
        if (frame.shape[1], frame.shape[0]) != (1920, 1080):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
        kpts, scores = model(frame)
        hands: list[SingleHand] = []
        if kpts is None:
            poses.append(FrameHandPose(frame_idx=idx, hands=[]))
            continue
        kpts = np.asarray(kpts)
        scores = np.asarray(scores) if scores is not None else np.ones(kpts.shape[:2])
        if kpts.ndim == 2:
            kpts = kpts[None, ...]
            scores = scores[None, ...] if scores.ndim == 1 else scores
        for hand_i, pts in enumerate(kpts):
            if pts.shape[0] < 21:
                continue
            conf = float(np.mean(scores[hand_i][:21])) if scores.ndim >= 2 else float(np.mean(scores))
            if conf < 0.25:
                continue
            xs = pts[:21, 0]
            ys = pts[:21, 1]
            pad_x = (float(xs.max()) - float(xs.min())) * 0.25
            pad_y = (float(ys.max()) - float(ys.min())) * 0.25
            x1 = int(max(0, xs.min() - pad_x))
            y1 = int(max(0, ys.min() - pad_y))
            x2 = int(min(1920, xs.max() + pad_x))
            y2 = int(min(1080, ys.max() + pad_y))
            lms_px = [[float(p[0]), float(p[1])] for p in pts[:21]]
            lms_nm = [[p[0] / 1920.0, p[1] / 1080.0, 0.0] for p in lms_px]
            hands.append(
                SingleHand(
                    handedness="Unknown",
                    confidence=conf,
                    bbox=[x1, y1, x2, y2],
                    landmarks_norm=lms_nm,
                    landmarks_pixel=lms_px,
                )
            )
        poses.append(FrameHandPose(frame_idx=idx, hands=hands))
        if idx % 50 == 0:
            logger.info("%s rtm frame %s/%s hands=%s", video_path.name, idx, len(frames), len(hands))
    logger.info("%s: %s rtm hands over %s frames", video_path.name, sum(len(p.hands) for p in poses), len(poses))
    return poses


def extract_rtm_wholebody_hands(video_path: Path, max_frames: int | None = None) -> list[FrameHandPose]:
    """Second independent judge: RTMPose whole-body, last 42 COCO-WholeBody hand joints."""
    global _RTM_WB
    try:
        from rtmlib import Wholebody
    except ImportError as exc:
        raise RuntimeError("rtmlib is not installed. pip install rtmlib") from exc

    from demo.pipeline.background_codec import read_video_frames_robust

    if "_RTM_WB" not in globals() or globals().get("_RTM_WB") is None:
        device = "cuda" if _cuda_onnx_ok() else "cpu"
        globals()["_RTM_WB"] = Wholebody(mode="lightweight", to_openpose=False, backend="onnxruntime", device=device)
    model = globals()["_RTM_WB"]
    frames = read_video_frames_robust(video_path, max_frames=max_frames)
    poses: list[FrameHandPose] = []
    for idx, frame in enumerate(frames):
        if (frame.shape[1], frame.shape[0]) != (1920, 1080):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
        kpts, scores = model(frame)
        hands: list[SingleHand] = []
        if kpts is None:
            poses.append(FrameHandPose(frame_idx=idx, hands=[]))
            continue
        kpts = np.asarray(kpts)
        scores = np.asarray(scores) if scores is not None else None
        if kpts.ndim == 3:
            kpts = kpts[0]
            if scores is not None and scores.ndim == 2:
                scores = scores[0]
        if kpts.shape[0] < 133:
            poses.append(FrameHandPose(frame_idx=idx, hands=[]))
            continue
        for side, sl in (("Left", slice(91, 112)), ("Right", slice(112, 133))):
            pts = kpts[sl]
            sc = scores[sl] if scores is not None and len(scores) >= 133 else np.ones(21)
            conf = float(np.mean(sc[:21]))
            if conf < 0.3:
                continue
            xs, ys = pts[:21, 0], pts[:21, 1]
            if float(np.max(xs) - np.min(xs)) < 8:
                continue
            pad_x = (float(xs.max()) - float(xs.min())) * 0.25
            pad_y = (float(ys.max()) - float(ys.min())) * 0.25
            x1 = int(max(0, xs.min() - pad_x))
            y1 = int(max(0, ys.min() - pad_y))
            x2 = int(min(1920, xs.max() + pad_x))
            y2 = int(min(1080, ys.max() + pad_y))
            lms_px = [[float(p[0]), float(p[1])] for p in pts[:21]]
            lms_nm = [[p[0] / 1920.0, p[1] / 1080.0, 0.0] for p in lms_px]
            hands.append(SingleHand(side, conf, [x1, y1, x2, y2], lms_nm, lms_px))
        poses.append(FrameHandPose(frame_idx=idx, hands=hands))
        if idx % 50 == 0:
            logger.info("%s wholebody frame %s/%s hands=%s", video_path.name, idx, len(frames), len(hands))
    return poses


BACKENDS: dict[str, PoseExtractor] = {
    "mp_live": extract_mediapipe_live,
    "mp_offline_gt": extract_mediapipe_offline_gt,
    "rtm_hand": extract_rtm_hand,
    "rtm_wholebody": extract_rtm_wholebody_hands,
}


def _cuda_onnx_ok() -> bool:
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def _run_mediapipe_video(
    video_path: Path,
    max_frames: int | None,
    static_image_mode: bool,
    model_complexity: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    smooth: bool,
) -> list[FrameHandPose]:
    from demo.pipeline.background_codec import read_video_frames_robust

    frames = read_video_frames_robust(video_path, max_frames=max_frames)
    estimator = HandPoseEstimator(
        static_image_mode=static_image_mode,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    trajectory_filter = HandTrajectoryFilter() if smooth else None
    poses: list[FrameHandPose] = []
    for frame_idx, frame in enumerate(frames):
        if (frame.shape[1], frame.shape[0]) != (1920, 1080):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
        pose = estimator.process_frame(frame, frame_idx=frame_idx)
        if trajectory_filter is not None and pose.hands:
            pose = FrameHandPose(
                frame_idx=frame_idx,
                hands=[trajectory_filter.filter_hand(h, frame_idx=frame_idx, fps=30.0) for h in pose.hands],
            )
        poses.append(pose)
    estimator.close()
    logger.info("%s: %s hands over %s frames", video_path.name, sum(len(p.hands) for p in poses), len(poses))
    return poses
