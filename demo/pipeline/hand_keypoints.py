"""MediaPipe hand pose estimator, 21-joint wire serializer, and skeleton visualizer."""

from __future__ import annotations

import sqlite3  # noqa: F401
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

# 21 Hand Landmarks defined by MediaPipe
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky finger
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm base cross-links
    (5, 9), (9, 13), (13, 17),
]

FINGER_COLORS = [
    (0, 255, 255),   # Thumb: Yellow
    (0, 255, 0),     # Index: Green
    (255, 255, 0),   # Middle: Cyan
    (255, 0, 0),     # Ring: Blue
    (255, 0, 255),   # Pinky: Magenta
]


@dataclass
class SingleHand:
    handedness: str  # "Left" or "Right"
    confidence: float
    bbox: list[int]  # [x1, y1, x2, y2] in pixel coords
    landmarks_norm: list[list[float]]  # 21 x [x, y, z] normalized [0, 1]
    landmarks_pixel: list[list[float]]  # 21 x [x, y] in pixel coords


@dataclass
class FrameHandPose:
    frame_idx: int
    hands: list[SingleHand]


class OneEuroFilter:
    """1-Euro filter for adaptive low-pass smoothing with low lag on human motion."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: float | None = None
        self.dx_prev: float = 0.0
        self.t_prev: float | None = None

    def __call__(self, x: float, t: float) -> float:
        if self.t_prev is None or self.x_prev is None:
            self.x_prev = float(x)
            self.t_prev = float(t)
            self.dx_prev = 0.0
            return float(x)

        dt = max(t - self.t_prev, 1e-4)
        dx = (float(x) - self.x_prev) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        edx = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(edx)
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * float(x) + (1.0 - alpha) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = edx
        self.t_prev = t
        return x_hat

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * max(cutoff, 1e-5))
        return 1.0 / (1.0 + tau / dt)


class HandTrajectoryFilter:
    """Filters 21 3D joint trajectories per hand across video frames."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.005) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        # {handedness: [21 x 3 OneEuroFilter instances]}
        self.filters: dict[str, list[list[OneEuroFilter]]] = {}
        self.last_seen_frame: dict[str, int] = {}

    def filter_hand(self, hand: SingleHand, frame_idx: int, fps: float = 30.0) -> SingleHand:
        side = hand.handedness
        t = frame_idx / fps

        # Reset filters if hand disappeared for more than 5 frames
        if side in self.last_seen_frame and (frame_idx - self.last_seen_frame[side]) > 5:
            self.filters.pop(side, None)

        self.last_seen_frame[side] = frame_idx

        if side not in self.filters:
            self.filters[side] = [
                [OneEuroFilter(self.min_cutoff, self.beta) for _ in range(3)]
                for _ in range(21)
            ]

        smoothed_norm = []
        for j_idx, pt in enumerate(hand.landmarks_norm):
            sx = self.filters[side][j_idx][0](pt[0], t)
            sy = self.filters[side][j_idx][1](pt[1], t)
            sz = self.filters[side][j_idx][2](pt[2], t) if len(pt) > 2 else 0.0
            smoothed_norm.append([sx, sy, sz])

        smoothed_px = [[p[0] * 1920.0, p[1] * 1080.0] for p in smoothed_norm]

        return SingleHand(
            handedness=hand.handedness,
            confidence=hand.confidence,
            bbox=hand.bbox,
            landmarks_norm=smoothed_norm,
            landmarks_pixel=smoothed_px,
        )


class HandPoseEstimator:
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1,
    ) -> None:
        import mediapipe as mp

        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int = 0) -> FrameHandPose:
        import cv2

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(rgb)

        hands: list[SingleHand] = []
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label
                score = float(handedness.classification[0].score)

                lms_norm: list[list[float]] = []
                lms_pixel: list[list[float]] = []
                xs, ys = [], []

                for lm in hand_lms.landmark:
                    lms_norm.append([float(lm.x), float(lm.y), float(lm.z)])
                    px, py = float(lm.x * w), float(lm.y * h)
                    lms_pixel.append([px, py])
                    xs.append(px)
                    ys.append(py)

                # Compute bounding box with 25% padding
                min_x, max_x = max(0, min(xs)), min(w, max(xs))
                min_y, max_y = max(0, min(ys)), min(h, max(ys))
                pad_x = (max_x - min_x) * 0.25
                pad_y = (max_y - min_y) * 0.25

                x1 = int(max(0, min_x - pad_x))
                y1 = int(max(0, min_y - pad_y))
                x2 = int(min(w, max_x + pad_x))
                y2 = int(min(h, max_y + pad_y))

                hands.append(
                    SingleHand(
                        handedness=label,
                        confidence=score,
                        bbox=[x1, y1, x2, y2],
                        landmarks_norm=lms_norm,
                        landmarks_pixel=lms_pixel,
                    )
                )

        return FrameHandPose(frame_idx=frame_idx, hands=hands)

    def close(self) -> None:
        self.mp_hands.close()


def render_skeleton_on_canvas(
    frame_pose: FrameHandPose,
    width: int,
    height: int,
    crop_bbox: list[int] | None = None,
) -> np.ndarray:
    """Render colored hand skeletons on a black canvas.

    If crop_bbox is specified ([x1, y1, x2, y2]), coordinates are mapped to the crop.
    """
    import cv2

    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    for hand in frame_pose.hands:
        lms = hand.landmarks_pixel
        if crop_bbox is not None:
            bx1, by1, bx2, by2 = crop_bbox
            bw = max(1, bx2 - bx1)
            bh = max(1, by2 - by1)
            scale = float(width) / max(bw, bh)
            new_w = int(round(bw * scale))
            new_h = int(round(bh * scale))
            pad_x = (width - new_w) // 2
            pad_y = (height - new_h) // 2
            mapped_lms = [
                [(pt[0] - bx1) * scale + pad_x, (pt[1] - by1) * scale + pad_y]
                for pt in lms
            ]
        else:
            mapped_lms = lms

        # Draw bones
        for i, (p1_idx, p2_idx) in enumerate(HAND_CONNECTIONS):
            pt1 = (int(mapped_lms[p1_idx][0]), int(mapped_lms[p1_idx][1]))
            pt2 = (int(mapped_lms[p2_idx][0]), int(mapped_lms[p2_idx][1]))
            # Assign color according to finger group
            color = FINGER_COLORS[min(i // 4, 4)]
            cv2.line(canvas, pt1, pt2, color, thickness=2, lineType=cv2.LINE_AA)

        # Draw joint keypoints
        for pt in mapped_lms:
            cv2.circle(canvas, (int(pt[0]), int(pt[1])), radius=3, color=(0, 0, 255), thickness=-1)

    return canvas


def extract_video_hand_poses(
    video_path: Path,
    max_frames: int | None = None,
    target_resolution: tuple[int, int] | None = (1920, 1080),
    smooth: bool = True,
    fps: float = 30.0,
) -> list[FrameHandPose]:
    from demo.pipeline.background_codec import read_video_frames_robust
    import cv2

    frames = read_video_frames_robust(video_path, max_frames=max_frames)
    estimator = HandPoseEstimator()
    trajectory_filter = HandTrajectoryFilter() if smooth else None
    poses: list[FrameHandPose] = []

    for frame_idx, frame in enumerate(frames):
        if target_resolution is not None and (frame.shape[1], frame.shape[0]) != target_resolution:
            frame = cv2.resize(frame, target_resolution, interpolation=cv2.INTER_LANCZOS4)
        pose = estimator.process_frame(frame, frame_idx=frame_idx)
        if trajectory_filter is not None and pose.hands:
            smoothed_hands = [
                trajectory_filter.filter_hand(h, frame_idx=frame_idx, fps=fps)
                for h in pose.hands
            ]
            pose = FrameHandPose(frame_idx=frame_idx, hands=smoothed_hands)
        poses.append(pose)

    estimator.close()
    return poses


def serialize_poses_to_json(poses: list[FrameHandPose], output_path: Path) -> None:
    data = []
    for p in poses:
        data.append(
            {
                "frame_idx": p.frame_idx,
                "hands": [
                    {
                        "handedness": h.handedness,
                        "confidence": h.confidence,
                        "bbox": h.bbox,
                        "landmarks_norm": h.landmarks_norm,
                        "landmarks_pixel": h.landmarks_pixel,
                    }
                    for h in p.hands
                ],
            }
        )
    with open(output_path, "w") as f:
        json.dump(data, f)

