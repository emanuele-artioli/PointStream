"""Subject detectors.

Implementations live in sibling modules; this module holds the registry.
Construction targets are import strings, so importing this module does not load
torch, cv2, or encoder binaries. Do not change ``REGISTRY`` or its axis string
— the parent package and the shared smoke test key on both.
"""

from src.contracts.capabilities import CAP_INSTANCE_MASKS, CAP_NATIVE_TRACKING, CAP_OPEN_VOCABULARY
from src.contracts.registry import BackendSpec, Registry

REGISTRY: Registry[object] = Registry("detector")

REGISTRY.register(
    BackendSpec(
        name="yolo",
        target="src.components.detection.yolo:YoloDetector",
        aliases=("yolo26",),
        capabilities=frozenset({CAP_NATIVE_TRACKING}),
        defaults={"model_name": "yolo26n.pt"},
        summary="YOLO26 closed-vocabulary detector; the default and the fallback comparator.",
    )
)
REGISTRY.register(
    BackendSpec(
        name="sam3",
        target="src.components.detection.sam3:Sam3Detector",
        capabilities=frozenset({CAP_OPEN_VOCABULARY, CAP_INSTANCE_MASKS}),
        defaults={"model_name": "sam3.pt"},
        summary="SAM3 open-vocabulary detector (supersedes SAM2).",
    )
)
REGISTRY.register(
    BackendSpec(
        name="rf-detr",
        target="src.components.detection.rfdetr:RfDetrDetector",
        summary=(
            "RF-DETR detector. Not installed: rfdetr needs transformers>=5.1, "
            "this env pins 4.46.3."
        ),
    )
)
