from __future__ import annotations

import cv2
import numpy as np
import torch

from src.shared.tags import gpu_bound


class GenAICompositor:
    """Mock GenAI compositor interface for future Animate Anyone integration."""

    def __init__(self, confidence_threshold: float = 0.2) -> None:
        self._confidence_threshold = float(confidence_threshold)

    @gpu_bound
    def process(
        self,
        reference_crop_tensor: torch.Tensor,
        dense_dwpose_tensor: torch.Tensor,
        warped_background_frame: torch.Tensor,
    ) -> torch.Tensor:
        frame_np = self._to_frame_numpy(warped_background_frame)
        pose_np = self._to_pose_numpy(dense_dwpose_tensor)
        crop_np = self._to_crop_numpy(reference_crop_tensor)

        x1, y1, x2, y2 = self._estimate_bbox_from_pose(pose_np=pose_np, frame_height=frame_np.shape[0], frame_width=frame_np.shape[1])

        # Draw a filled placeholder silhouette to prove GenAI stage integration end-to-end.
        cv2.rectangle(frame_np, (x1, y1), (x2, y2), color=(35, 80, 210), thickness=-1)

        crop_h = max(1, y2 - y1)
        crop_w = max(1, x2 - x1)
        resized_crop = cv2.resize(crop_np, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
        frame_np[y1:y2, x1:x2] = resized_crop

        return torch.from_numpy(frame_np).permute(2, 0, 1).contiguous().to(torch.uint8)

    def _to_frame_numpy(self, frame_tensor: torch.Tensor) -> np.ndarray:
        if frame_tensor.ndim != 3:
            raise ValueError(f"Expected frame tensor [C,H,W], got shape {tuple(frame_tensor.shape)}")
        frame_np = frame_tensor.detach().cpu().permute(1, 2, 0).numpy()
        if frame_np.dtype != np.uint8:
            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
        return frame_np.copy()

    def _to_pose_numpy(self, pose_tensor: torch.Tensor) -> np.ndarray:
        pose_np = pose_tensor.detach().cpu().numpy()
        if pose_np.ndim == 3:
            # Use first frame when a temporal tensor [Frames, 18, 3] is provided.
            pose_np = pose_np[0]
        if pose_np.shape != (18, 3):
            raise ValueError(f"Expected pose tensor shape (18, 3), got {tuple(pose_np.shape)}")
        return pose_np.astype(np.float32, copy=False)

    def _to_crop_numpy(self, crop_tensor: torch.Tensor) -> np.ndarray:
        if crop_tensor.ndim != 3:
            raise ValueError(f"Expected crop tensor [C,H,W], got shape {tuple(crop_tensor.shape)}")
        crop_np = crop_tensor.detach().cpu().permute(1, 2, 0).numpy()
        if crop_np.dtype != np.uint8:
            crop_np = np.clip(crop_np, 0, 255).astype(np.uint8)
        if crop_np.shape[2] != 3:
            raise ValueError(f"Expected crop tensor with 3 channels, got shape {tuple(crop_np.shape)}")
        return crop_np

    def _estimate_bbox_from_pose(self, pose_np: np.ndarray, frame_height: int, frame_width: int) -> tuple[int, int, int, int]:
        valid = pose_np[:, 2] >= self._confidence_threshold
        if not np.any(valid):
            cx = frame_width // 2
            cy = frame_height // 2
            half_w = max(8, frame_width // 10)
            half_h = max(12, frame_height // 6)
            return (
                max(0, cx - half_w),
                max(0, cy - half_h),
                min(frame_width, cx + half_w),
                min(frame_height, cy + half_h),
            )

        xs = pose_np[valid, 0]
        ys = pose_np[valid, 1]
        x1 = int(np.floor(np.min(xs)))
        y1 = int(np.floor(np.min(ys)))
        x2 = int(np.ceil(np.max(xs)))
        y2 = int(np.ceil(np.max(ys)))

        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        pad_x = max(3, int(round(width * 0.15)))
        pad_y = max(4, int(round(height * 0.20)))

        bx1 = max(0, x1 - pad_x)
        by1 = max(0, y1 - pad_y)
        bx2 = min(frame_width, x2 + pad_x)
        by2 = min(frame_height, y2 + pad_y)

        if bx2 <= bx1:
            bx2 = min(frame_width, bx1 + 1)
        if by2 <= by1:
            by2 = min(frame_height, by1 + 1)
        return bx1, by1, bx2, by2
