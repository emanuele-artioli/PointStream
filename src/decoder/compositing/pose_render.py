"""Rendering pose tensors into the conditioning images backends consume.

Shared by every engine (ControlNet, pix2pix, SPADE), which is why these
live apart from any one strategy."""

from __future__ import annotations
import logging
import numpy as np
import torch
from src.shared.dwpose_draw import draw_dwpose_canvas
from src.shared.racket_heuristic import render_pose_with_racket
_LOGGER = logging.getLogger(__name__)


def _to_numpy_bgr(image_tensor: torch.Tensor) -> np.ndarray:
    if image_tensor.ndim != 3:
        raise ValueError(f"Expected image tensor [C,H,W], got shape {tuple(image_tensor.shape)}")
    image_np = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    if image_np.dtype != np.uint8:
        image_np = np.asarray(np.clip(image_np, 0, 255), dtype=np.uint8)
    if image_np.shape[2] != 3:
        raise ValueError(f"Expected BGR image tensor with 3 channels, got {tuple(image_np.shape)}")
    return image_np
def _render_pose_condition(pose_tensor: torch.Tensor, output_height: int, output_width: int) -> np.ndarray:
    pose_np = pose_tensor.detach().cpu().numpy()
    if pose_np.shape != (18, 3):
        raise ValueError(f"Expected pose tensor shape (18, 3), got {tuple(pose_np.shape)}")

    return draw_dwpose_canvas(
        height=int(output_height),
        width=int(output_width),
        people_dw=pose_np[np.newaxis, ...],
        confidence_threshold=0.2,
    )
def _render_pose_with_racket(
    pose_tensor: torch.Tensor,
    racket_bbox: tuple[float, float, float, float] | list[float] | None,
    output_height: int,
    output_width: int,
) -> np.ndarray:
    pose_np = pose_tensor.detach().cpu().numpy()
    if pose_np.ndim == 3:
        pose_np = pose_np[-1]
    racket_bbox_tuple: tuple[float, float, float, float] | None = None
    if isinstance(racket_bbox, (tuple, list)) and len(racket_bbox) == 4:
        racket_bbox_tuple = (float(racket_bbox[0]), float(racket_bbox[1]), float(racket_bbox[2]), float(racket_bbox[3]))
    # render_pose_with_racket returns a BGR/RGB canvas
    return render_pose_with_racket(pose_np, racket_bbox_tuple, output_height, output_width)
