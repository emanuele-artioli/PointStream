"""Background-model backends.

Implementations live in sibling modules; this module holds the registry.
Construction targets are import strings, so importing this module does not load
torch, cv2, or encoder binaries. Do not change ``REGISTRY`` or its axis string
— the parent package and the shared smoke test key on both.
"""

from src.contracts.domain import (
    BACKGROUND_NONE,
    BACKGROUND_PANORAMA_DELTA,
    BACKGROUND_PANORAMA_FULL,
)
from src.contracts.registry import BackendSpec, Registry

REGISTRY: Registry[object] = Registry("background")

REGISTRY.register(
    BackendSpec(
        name=BACKGROUND_PANORAMA_FULL,
        target="src.components.background.strategy:PanoramaFull",
        summary="Transmit a full panorama plate once per chunk.",
    )
)
REGISTRY.register(
    BackendSpec(
        name=BACKGROUND_PANORAMA_DELTA,
        target="src.components.background.strategy:PanoramaDelta",
        summary="Full plate on the first chunk of a scene, signed diffs after.",
    )
)
REGISTRY.register(
    BackendSpec(
        name=BACKGROUND_NONE,
        target="src.components.background.strategy:BackgroundNone",
        summary="No background model; the residual carries the background.",
    )
)
