"""Segmenter registry for the runner's ``segmenter`` config axis.

The implementations live in `src.segmentation`; this module only names them so
config validation and `build_backend` keep one registry per axis. Construction
targets are import strings, so importing this module loads no model code.
"""

from src.contracts.capabilities import CAP_INSTANCE_MASKS
from src.contracts.registry import BackendSpec, Registry

REGISTRY: Registry[object] = Registry("segmenter")

REGISTRY.register(
    BackendSpec(
        name="sam31",
        target="src.segmentation.sam31:Sam31Segmenter",
        aliases=("sam3.1", "sam3.1-multiplex"),
        capabilities=frozenset({CAP_INSTANCE_MASKS}),
        summary="SAM 3.1 multiplex video segmentation; the offline reference.",
    )
)
for _size in ("n", "s", "m", "l", "x"):
    REGISTRY.register(
        BackendSpec(
            name=f"yoloe-26{_size}",
            target="src.segmentation.yoloe:YoloeSegmenter",
            capabilities=frozenset({CAP_INSTANCE_MASKS}),
            defaults={"size": _size, "weights": f"yoloe-26{_size}-seg.pt"},
            summary=f"YOLOE-26{_size} open-vocabulary instance segmenter.",
        )
    )
