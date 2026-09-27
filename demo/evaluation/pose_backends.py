"""Offline and live hand-pose backends used as GT or reconstruction evaluators."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from demo.pipeline.hand_keypoints import (
    FrameHandPose,
    HandPoseEstimator,
    HandTrajectoryFilter,
    OneEuroFilter,
    SingleHand,
)

logger = logging.getLogger(__name__)

PoseExtractor = Callable[[Path, int | None], Any]

# COCO-WholeBody-133 (mmpose / rtmlib to_openpose=False).
COCO_WB_BODY = slice(0, 17)
COCO_WB_FEET = slice(17, 23)
COCO_WB_FACE = slice(23, 91)
COCO_WB_LEFT_HAND = slice(91, 112)
COCO_WB_RIGHT_HAND = slice(112, 133)


@dataclass
class WholeBodyPerson:
    """One COCO-WholeBody-133 instance as x,y,score rows."""

    keypoints: np.ndarray  # (133, 3)

    @property
    def body(self) -> np.ndarray:
        return self.keypoints[COCO_WB_BODY]

    @property
    def feet(self) -> np.ndarray:
        return self.keypoints[COCO_WB_FEET]

    @property
    def face(self) -> np.ndarray:
        return self.keypoints[COCO_WB_FACE]

    @property
    def left_hand(self) -> np.ndarray:
        return self.keypoints[COCO_WB_LEFT_HAND]

    @property
    def right_hand(self) -> np.ndarray:
        return self.keypoints[COCO_WB_RIGHT_HAND]


@dataclass
class FrameWholeBody:
    frame_idx: int
    people: list[WholeBodyPerson]
    width: int = 1920
    height: int = 1080


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


def people_from_rtm(kpts: np.ndarray | None, scores: np.ndarray | None) -> list[WholeBodyPerson]:
    """Split rtmlib Wholebody output into COCO-WholeBody-133 people."""
    if kpts is None:
        return []
    kpts_arr = np.asarray(kpts)
    scores_arr = np.asarray(scores) if scores is not None else None
    if kpts_arr.size == 0:
        return []
    if kpts_arr.ndim == 2:
        kpts_arr = kpts_arr[None, ...]
        if scores_arr is not None and scores_arr.ndim == 1:
            scores_arr = scores_arr[None, ...]
    people: list[WholeBodyPerson] = []
    for inst_i, inst in enumerate(kpts_arr):
        if inst.shape[0] < 133:
            continue
        xyz = np.zeros((133, 3), dtype=np.float32)
        xyz[:, :2] = inst[:133, :2]
        if scores_arr is not None:
            sc = scores_arr[inst_i]
            n = min(133, int(np.asarray(sc).shape[0]))
            xyz[:n, 2] = np.asarray(sc, dtype=np.float32)[:n]
        else:
            xyz[:, 2] = 1.0
        people.append(WholeBodyPerson(keypoints=xyz))
    return people


def _hand_from_kpts21(
    pts: np.ndarray,
    side: str,
    frame_w: int,
    frame_h: int,
    conf_thr: float = 0.4,
    min_visible: int = 8,
    min_span: float = 16.0,
) -> SingleHand | None:
    scores = pts[:21, 2] if pts.shape[1] >= 3 else np.ones(21, dtype=np.float32)
    vis = np.asarray(scores) >= conf_thr
    if int(np.count_nonzero(vis)) < int(min_visible):
        return None
    xs, ys = pts[:21][vis, 0], pts[:21][vis, 1]
    span_x = float(np.max(xs) - np.min(xs))
    span_y = float(np.max(ys) - np.min(ys))
    if span_x < min_span and span_y < min_span:
        return None
    if span_x > 0.7 * frame_w and span_y > 0.7 * frame_h:
        return None
    conf = float(np.mean(scores[vis]))
    pad_x = span_x * 0.25
    pad_y = span_y * 0.25
    x1 = int(max(0, xs.min() - pad_x))
    y1 = int(max(0, ys.min() - pad_y))
    x2 = int(min(frame_w, xs.max() + pad_x))
    y2 = int(min(frame_h, ys.max() + pad_y))
    lms_px = [[float(p[0]), float(p[1])] for p in pts[:21]]
    lms_nm = [[p[0] / float(frame_w), p[1] / float(frame_h), 0.0] for p in lms_px]
    return SingleHand(side, conf, [x1, y1, x2, y2], lms_nm, lms_px)


def wholebody_to_frame_hands(frame: FrameWholeBody, conf_thr: float = 0.3) -> FrameHandPose:
    """Project WholeBody people to the 21-point hand wire used by KeypointCompressor."""
    hands: list[SingleHand] = []
    for person in frame.people:
        left = _hand_from_kpts21(person.left_hand, "Left", frame.width, frame.height, conf_thr)
        right = _hand_from_kpts21(person.right_hand, "Right", frame.width, frame.height, conf_thr)
        if left is not None:
            hands.append(left)
        if right is not None:
            hands.append(right)
    return FrameHandPose(frame_idx=frame.frame_idx, hands=hands)


_DWPOSE_WB = None


def _get_dwpose_wholebody():
    """rtmlib Wholebody with local DWPose ONNX — never a Hub id / auto-download."""
    global _DWPOSE_WB
    if _DWPOSE_WB is not None:
        return _DWPOSE_WB
    from rtmlib import Wholebody

    from demo.pipeline.maps.model_paths import require

    det = require("dwpose_det")
    pose = require("dwpose_pose")
    device = "cuda" if _cuda_onnx_ok() else "cpu"
    _DWPOSE_WB = Wholebody(
        det=str(det),
        det_input_size=(640, 640),
        pose=str(pose),
        pose_input_size=(288, 384),
        to_openpose=False,
        backend="onnxruntime",
        device=device,
    )
    return _DWPOSE_WB


def extract_dwpose_hands(video_path: Path, max_frames: int | None = None) -> list[FrameHandPose]:
    """DW-Pose hands in the same FrameHandPose the generator and the compressor use."""
    return [wholebody_to_frame_hands(frame) for frame in extract_dwpose_wholebody(video_path, max_frames)]


def extract_dwpose_wholebody(video_path: Path, max_frames: int | None = None) -> list[FrameWholeBody]:
    """DWPose via rtmlib Wholebody; returns body 0:17, feet 17:23, face 23:91, hands 91:133.

    Keypoints are One-Euro smoothed across frames. The pose network stays at
    288×384 and the person detector at its exported 640×640; neither ONNX graph
    accepts a 1080p input.
    """
    try:
        from rtmlib import Wholebody  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("rtmlib is not installed. pip install rtmlib") from exc

    from demo.pipeline.background_codec import read_video_frames_robust

    model = _get_dwpose_wholebody()
    frames = read_video_frames_robust(video_path, max_frames=max_frames)
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if fps < 1.0:
        fps = 30.0
    smoother = WholeBodyEuroSmoother()
    poses: list[FrameWholeBody] = []
    for idx, frame in enumerate(frames):
        if (frame.shape[1], frame.shape[0]) != (1920, 1080):
            frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LANCZOS4)
        h, w = frame.shape[:2]
        kpts, scores = model(frame)
        people = smoother.smooth(people_from_rtm(kpts, scores), idx, fps)
        poses.append(FrameWholeBody(frame_idx=idx, people=people, width=w, height=h))
        if idx % 50 == 0:
            logger.info("%s dwpose frame %s/%s people=%s", video_path.name, idx, len(frames), len(people))
    return poses


def extract_hamer(video_path: Path, max_frames: int | None = None) -> list[FrameHandPose]:
    from demo.evaluation.hamer_backend import extract_hamer as _extract

    return _extract(video_path, max_frames=max_frames)


BACKENDS: dict[str, PoseExtractor] = {
    "mp_live": extract_mediapipe_live,
    "mp_offline_gt": extract_mediapipe_offline_gt,
    "rtm_hand": extract_rtm_hand,
    "rtm_wholebody": extract_rtm_wholebody_hands,
    "hamer": extract_hamer,
    "dwpose": extract_dwpose_wholebody,
    "dwpose_hands": extract_dwpose_hands,
}


def _preload_ort_cuda_libs() -> None:
    """Load cuDNN 9 and the CUDA 12 runtime before ONNX Runtime opens its CUDA provider.

    onnxruntime-gpu 1.23 lists CUDAExecutionProvider even when libcudnn.so.9 is
    not on the default linker path. Torch ships that library; the NVIDIA pip
    wheels ship libcudart and libcublas. Preloading them makes the provider
    actually initialize on the RTX 6000 Ada.
    """
    import ctypes

    candidates: list[Path] = []
    try:
        import torch

        candidates.append(Path(torch.__file__).resolve().parent / "lib" / "libcudnn.so.9")
    except Exception:
        pass
    try:
        import nvidia.cuda_runtime

        nvidia_root = Path(nvidia.cuda_runtime.__file__).resolve().parent.parent
        for rel in (
            "cuda_runtime/lib/libcudart.so.12",
            "cublas/lib/libcublas.so.12",
            "cublas/lib/libcublasLt.so.12",
            "cuda_nvrtc/lib/libnvrtc.so.12",
            "cudnn/lib/libcudnn.so.9",
        ):
            candidates.append(nvidia_root / rel)
    except Exception:
        pass
    for path in candidates:
        if path.is_file():
            ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)


def dwpose_ort_device() -> str:
    return "cuda" if _cuda_onnx_ok() else "cpu"


def _cuda_onnx_ok() -> bool:
    try:
        _preload_ort_cuda_libs()
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def _person_box(keypoints: np.ndarray) -> tuple[float, float, float, float] | None:
    visible = np.asarray(keypoints[:, 2]) >= 0.3
    if int(np.count_nonzero(visible)) < 4:
        return None
    xs = keypoints[visible, 0]
    ys = keypoints[visible, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _box_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    area_l = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_r = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return inter / max(area_l + area_r - inter, 1e-6)


class WholeBodyEuroSmoother:
    """One-Euro filter per joint, matched across frames by person box.

    The MediaPipe hand path already does this. DW-Pose was estimating every
    frame independently, which is the jitter. A gap of more than five frames
    starts a new filter so a reappearing person is not dragged toward the old pose.
    """

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.007, gap: int = 5) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.gap = gap
        self._tracks: list[dict[str, Any]] = []

    def smooth(
        self,
        people: list[WholeBodyPerson],
        frame_idx: int,
        fps: float = 30.0,
    ) -> list[WholeBodyPerson]:
        self._tracks = [track for track in self._tracks if frame_idx - int(track["last"]) <= self.gap]
        used: set[int] = set()
        smoothed: list[WholeBodyPerson] = []
        for person in people:
            box = _person_box(person.keypoints)
            best_i = -1
            best_iou = 0.3
            for index, track in enumerate(self._tracks):
                if index in used or track["box"] is None or box is None:
                    continue
                score = _box_iou(box, track["box"])
                if score > best_iou:
                    best_i = index
                    best_iou = score
            if best_i < 0:
                track = {
                    "fx": [OneEuroFilter(self.min_cutoff, self.beta) for _ in range(133)],
                    "fy": [OneEuroFilter(self.min_cutoff, self.beta) for _ in range(133)],
                    "box": box,
                    "last": frame_idx,
                }
                self._tracks.append(track)
            else:
                track = self._tracks[best_i]
                used.add(best_i)
            keypoints = self._apply(track, person.keypoints, frame_idx, fps)
            track["box"] = _person_box(keypoints) or box
            track["last"] = frame_idx
            smoothed.append(WholeBodyPerson(keypoints=keypoints))
        return smoothed

    def _apply(self, track: dict[str, Any], keypoints: np.ndarray, frame_idx: int, fps: float) -> np.ndarray:
        out = np.array(keypoints, dtype=np.float32, copy=True)
        t = frame_idx / max(fps, 1e-3)
        for joint in range(min(133, out.shape[0])):
            if float(out[joint, 2]) < 0.3:
                continue
            out[joint, 0] = track["fx"][joint](float(out[joint, 0]), t)
            out[joint, 1] = track["fy"][joint](float(out[joint, 1]), t)
        return out


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
