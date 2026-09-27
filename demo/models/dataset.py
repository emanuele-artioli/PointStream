"""Dataset for training the overfit hand generator from pose keypoints and appearance anchor."""

from __future__ import annotations

import sqlite3  # noqa: F401
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from demo.models.matte import letterbox_alpha, matte_bgr
from demo.pipeline.foreground_segmenter import letterbox_crop
from demo.pipeline.hand_keypoints import (
    FrameHandPose,
    render_skeleton_on_canvas,
)


def to_torch_tensor(img_bgr: np.ndarray) -> torch.Tensor:
    """Converts BGR uint8 [H, W, 3] to RGB float32 [-1, 1] [3, H, W]."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    float_img = (rgb.astype(np.float32) / 127.5) - 1.0
    return torch.from_numpy(float_img).permute(2, 0, 1)


class EgocentricHandDataset(Dataset):
    """Pairs (appearance anchor, pose skeleton) with target hand crops."""

    def __init__(
        self,
        samples: list[dict[str, Any]],
        image_size: int = 256,
    ) -> None:
        self.samples = samples
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.samples[idx]
        appearance_crop = item["appearance_crop"]  # [image_size, image_size, 3]
        skeleton_crop = item["skeleton_crop"]      # [image_size, image_size, 3]
        target_crop = item["target_crop"]          # [image_size, image_size, 3]

        app_tensor = to_torch_tensor(appearance_crop)
        skel_tensor = to_torch_tensor(skeleton_crop)
        tgt_tensor = to_torch_tensor(target_crop)
        if "target_alpha" in item:
            alpha = torch.from_numpy(np.asarray(item["target_alpha"], dtype=np.float32)).unsqueeze(0)
            tgt_tensor = torch.cat([tgt_tensor, alpha], dim=0)

        # 6-channel input. Alpha is a training target, not an input.
        model_input = torch.cat([app_tensor, skel_tensor], dim=0)

        return {
            "input": model_input,
            "target": tgt_tensor,
            "clip_id": torch.tensor(item.get("clip_id", 0)),
            "frame_idx": torch.tensor(item.get("frame_idx", 0)),
        }


def build_curated_samples(
    video_path: Path,
    poses: list[FrameHandPose],
    image_size: int = 256,
    max_frames: int | None = None,
    clip_id: int = 0,
    frame_alphas: list[np.ndarray] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray], dict[str, int]]:
    """Extracts paired hand crops across video frames and identifies appearance anchors."""
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    raw_frames: list[np.ndarray] = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
        frame_idx += 1
        if max_frames and frame_idx >= max_frames:
            break
    cap.release()

    appearance_anchors: dict[str, np.ndarray] = {}
    anchor_bytes: dict[str, int] = {}  # compressed WebP payload size per side
    anchor_score: dict[str, float] = {}
    samples: list[dict[str, Any]] = []

    def _alpha_for(idx: int, frame: np.ndarray, bbox: list[int]) -> np.ndarray | None:
        if frame_alphas is None or idx >= len(frame_alphas):
            return None
        full = frame_alphas[idx]
        if full.shape[:2] != frame.shape[:2]:
            full = cv2.resize(full, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        return letterbox_alpha(full, bbox, target_size=image_size)

    def _store_anchor(side: str, crop: np.ndarray, score: float) -> None:
        # WebP roundtrip so the anchor matches the bytes that would be sent.
        ok, webp_buf = cv2.imencode(".webp", crop, [cv2.IMWRITE_WEBP_QUALITY, 90])
        if ok:
            anchor_bytes[side] = len(webp_buf)
            crop = cv2.imdecode(webp_buf, cv2.IMREAD_COLOR)
        appearance_anchors[side] = crop
        anchor_score[side] = score

    for idx, frame in enumerate(raw_frames):
        if idx >= len(poses):
            break
        pose = poses[idx]
        for hand in pose.hands:
            side = hand.handedness
            alpha = _alpha_for(idx, frame, hand.bbox)
            if alpha is not None and int(np.count_nonzero(alpha)) < 200:
                continue
            if side in appearance_anchors and hand.confidence <= anchor_score[side]:
                continue
            crop, _ = letterbox_crop(frame, hand.bbox, target_size=image_size)
            if alpha is not None:
                crop = matte_bgr(crop, alpha)
            _store_anchor(side, crop, hand.confidence)

    for idx, frame in enumerate(raw_frames):
        if idx >= len(poses):
            break
        pose = poses[idx]
        for hand in pose.hands:
            side = hand.handedness
            if side not in appearance_anchors:
                continue
            alpha = _alpha_for(idx, frame, hand.bbox)
            if alpha is not None and int(np.count_nonzero(alpha)) < 200:
                continue

            app_anchor = appearance_anchors[side]
            tgt_crop, _ = letterbox_crop(frame, hand.bbox, target_size=image_size)
            if alpha is not None:
                tgt_crop = matte_bgr(tgt_crop, alpha)

            single_hand_pose = FrameHandPose(frame_idx=idx, hands=[hand])
            skel_crop = render_skeleton_on_canvas(
                single_hand_pose,
                width=image_size,
                height=image_size,
                crop_bbox=hand.bbox,
            )

            sample: dict[str, Any] = {
                "appearance_crop": app_anchor,
                "skeleton_crop": skel_crop,
                "target_crop": tgt_crop,
                "clip_id": clip_id,
                "frame_idx": idx,
                "handedness": side,
                "bbox": hand.bbox,
            }
            if alpha is not None:
                sample["target_alpha"] = (alpha > 127).astype(np.float32)
            samples.append(sample)

    return samples, appearance_anchors, anchor_bytes
