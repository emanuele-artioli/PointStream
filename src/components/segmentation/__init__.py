"""Segmenters.

Implementations live in sibling modules; this module holds the registry.
Construction targets are import strings, so importing this module does not load
torch, cv2, or encoder binaries. Do not change ``REGISTRY`` or its axis string
— the parent package and the shared smoke test key on both.
"""

from src.contracts.capabilities import CAP_INSTANCE_MASKS
from src.contracts.registry import BackendSpec, Registry

REGISTRY: Registry[object] = Registry("segmenter")

REGISTRY.register(
    BackendSpec(
        name="yolo",
        target="src.components.segmentation.yolo:YoloSegmenter",
        aliases=("yolo-seg",),
        capabilities=frozenset({CAP_INSTANCE_MASKS}),
        defaults={"model_name": "yolo26n-seg.pt"},
        summary="YOLO26 instance segmenter; the default.",
    )
)
REGISTRY.register(
    BackendSpec(
        name="sam3",
        target="src.components.segmentation.sam3:Sam3Segmenter",
        capabilities=frozenset({CAP_INSTANCE_MASKS}),
        defaults={"model_name": "sam3.pt"},
        summary="SAM3 box-prompted segmenter (supersedes SAM2).",
    )
)
