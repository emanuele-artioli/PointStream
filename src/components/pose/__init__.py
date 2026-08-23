"""Pose estimators.

Implementations live in sibling modules; this module holds the registry.
Construction targets are import strings, so importing this module does not load
torch, cv2, or encoder binaries. Do not change ``REGISTRY`` or its axis string
— the parent package and the shared smoke test key on both.
"""

from src.contracts.registry import BackendSpec, Registry

REGISTRY: Registry[object] = Registry("pose")

REGISTRY.register(
    BackendSpec(
        name="yolo",
        target="src.components.pose.yolo:YoloPoseEstimator",
        aliases=("yolo-pose",),
        defaults={"model_name": "yolo26n-pose.pt"},
        summary="YOLO26 pose; emits COCO-17, stored as canonical WholeBody-133.",
    )
)
