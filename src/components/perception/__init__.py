"""Shared spatial transforms and conditioning renderers for perception paths."""

from src.components.perception.coordinates import (
    CropTransform,
    crop_and_mask,
    make_crop_transform,
    render_object_view,
)
from src.components.perception.association import RacketPlayerAssociator
from src.components.perception.conditioning import (
    ConditioningView,
    render_conditioning_view,
    render_pose_condition,
)

__all__ = [
    "CropTransform",
    "ConditioningView",
    "RacketPlayerAssociator",
    "crop_and_mask",
    "make_crop_transform",
    "render_object_view",
    "render_conditioning_view",
    "render_pose_condition",
]
