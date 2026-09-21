"""Foreground adaptive keyframing driven by pose drift (OKS) and wire budget."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.components.appearance.compressed import CompressedImageAppearance
from src.components.metrics.pose import compute_oks as compute_pose_oks
from src.contracts.objectstream import CompressedImage


@dataclass(frozen=True)
class AdaptiveCropConfig:
    """Configuration for adaptive keyframe selection and compression."""

    max_crops: int = 3
    """Maximum number of keyframe crops: initial keyframe + up to 2 adaptive updates."""

    max_total_bytes: int = 12000
    """Strict wire budget in bytes (default 12 kB)."""

    min_frame_interval: int = 24
    """Cooldown interval in frames between keyframe updates."""

    oks_threshold: float = 0.80
    """OKS drift threshold to trigger an update."""

    format: str = "webp"
    """Compression format ('webp' or 'jpeg')."""

    quality: int = 75
    """Base compression quality (1-100)."""

    downscale: float = 1.0
    """Crop downscale factor in (0, 1]."""


@dataclass
class AdaptiveKeyframeResult:
    """Result of adaptive keyframe selection and encoding."""

    keyframe_indices: list[int]
    crops: list[np.ndarray]
    payloads: list[bytes]
    descriptors: list[CompressedImage]
    total_bytes: int
    frame_to_crop_idx: list[int]


class AdaptiveKeyframeSelector:
    """Selects and encodes foreground keyframes based on pose drift and budget."""

    def __init__(
        self,
        config: AdaptiveCropConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if config is not None:
            if kwargs:
                import dataclasses

                self.config = dataclasses.replace(config, **kwargs)
            else:
                self.config = config
        else:
            self.config = AdaptiveCropConfig(**kwargs)

        self.encoder = CompressedImageAppearance(
            quality=self.config.quality,
            downscale=self.config.downscale,
            format=self.config.format,
        )

    @staticmethod
    def _extract_crop(
        frame: np.ndarray,
        bbox: tuple[int, int, int, int] | tuple[float, float, float, float] | None,
    ) -> np.ndarray:
        """Extract a bounding box crop from a frame, clamped to boundaries."""
        if bbox is None:
            return frame
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        ix1 = max(0, min(w, int(round(x1))))
        ix2 = max(0, min(w, int(round(x2))))
        iy1 = max(0, min(h, int(round(y1))))
        iy2 = max(0, min(h, int(round(y2))))
        if ix2 > ix1 and iy2 > iy1:
            return np.ascontiguousarray(frame[iy1:iy2, ix1:ix2])
        return frame

    def _encode_within_budget(
        self,
        crop: np.ndarray,
        budget: int,
    ) -> tuple[CompressedImage, bytes] | None:
        """Encode crop, stepping down quality if necessary to respect budget."""
        if budget <= 0:
            return None

        # First attempt: base configured quality
        desc, payload = self.encoder.encode(
            crop,
            quality=self.config.quality,
            downscale=self.config.downscale,
            format=self.config.format,
        )
        if len(payload) <= budget:
            return desc, payload

        # Step down quality to fit within remaining budget
        candidate_qualities = list(range(self.config.quality - 5, 4, -5))
        if 1 not in candidate_qualities:
            candidate_qualities.append(1)

        for q in candidate_qualities:
            desc, payload = self.encoder.encode(
                crop,
                quality=q,
                downscale=self.config.downscale,
                format=self.config.format,
            )
            if len(payload) <= budget:
                return desc, payload

        return None

    def compute_oks(
        self,
        pts_curr: np.ndarray | None,
        pts_active: np.ndarray | None,
        bbox_active: tuple[int, int, int, int] | tuple[float, float, float, float] | None = None,
    ) -> float:
        """Compute Object Keypoint Similarity between current and active keypoints."""
        if pts_curr is None or pts_active is None:
            if pts_curr is None and pts_active is None:
                return 1.0
            return 0.0

        bbox_gt = (
            (float(bbox_active[0]), float(bbox_active[1]), float(bbox_active[2]), float(bbox_active[3]))
            if bbox_active is not None
            else None
        )
        oks, _, _, _, _ = compute_pose_oks(
            pts_pred=pts_curr,
            pts_gt=pts_active,
            bbox_gt=bbox_gt,
        )
        return float(oks)

    def select_and_encode(
        self,
        frames: Sequence[np.ndarray],
        keypoints: Sequence[np.ndarray | None],
        bboxes: Sequence[tuple[int, int, int, int] | None],
    ) -> AdaptiveKeyframeResult:
        """Select and encode keyframes across a sequence.

        Frame 0 is always encoded as keyframe 0. Subsequent frames trigger an
        adaptive keyframe update when:
        1. Pose drift OKS < oks_threshold
        2. Frame index >= last_keyframe + min_frame_interval (cooldown)
        3. len(crops) < max_crops (crop budget)
        4. The new crop can fit within max_total_bytes (possibly with reduced quality)
        """
        t_len = len(frames)
        if t_len == 0:
            return AdaptiveKeyframeResult(
                keyframe_indices=[],
                crops=[],
                payloads=[],
                descriptors=[],
                total_bytes=0,
                frame_to_crop_idx=[],
            )

        if len(keypoints) != t_len or len(bboxes) != t_len:
            raise ValueError(
                f"Mismatched input lengths: {t_len} frames, "
                f"{len(keypoints)} keypoints, {len(bboxes)} bboxes."
            )

        # Always encode frame 0 (first detection) as keyframe 0
        crop_0 = self._extract_crop(frames[0], bboxes[0])
        enc_0 = self._encode_within_budget(crop_0, budget=self.config.max_total_bytes)
        if enc_0 is None:
            # Fallback: encode at quality 1 so frame 0 is always present
            desc_0, payload_0 = self.encoder.encode(
                crop_0,
                quality=1,
                downscale=self.config.downscale,
                format=self.config.format,
            )
        else:
            desc_0, payload_0 = enc_0

        keyframe_indices: list[int] = [0]
        crops: list[np.ndarray] = [crop_0]
        payloads: list[bytes] = [payload_0]
        descriptors: list[CompressedImage] = [desc_0]
        total_bytes: int = len(payload_0)
        frame_to_crop_idx: list[int] = [0]

        last_keyframe: int = 0
        active_crop_idx: int = 0

        for t in range(1, t_len):
            curr_kpts = keypoints[t]
            active_kpts = keypoints[last_keyframe]

            should_update = False
            if (
                curr_kpts is not None
                and t >= last_keyframe + self.config.min_frame_interval
                and len(crops) < self.config.max_crops
            ):
                oks = self.compute_oks(curr_kpts, active_kpts, bboxes[last_keyframe])
                if oks < self.config.oks_threshold:
                    should_update = True

            if should_update:
                remaining_budget = self.config.max_total_bytes - total_bytes
                candidate_crop = self._extract_crop(frames[t], bboxes[t])
                enc = self._encode_within_budget(candidate_crop, budget=remaining_budget)
                if enc is not None:
                    desc, payload = enc
                    crops.append(candidate_crop)
                    payloads.append(payload)
                    descriptors.append(desc)
                    keyframe_indices.append(t)
                    total_bytes += len(payload)
                    last_keyframe = t
                    active_crop_idx = len(crops) - 1

            frame_to_crop_idx.append(active_crop_idx)

        return AdaptiveKeyframeResult(
            keyframe_indices=keyframe_indices,
            crops=crops,
            payloads=payloads,
            descriptors=descriptors,
            total_bytes=total_bytes,
            frame_to_crop_idx=frame_to_crop_idx,
        )
