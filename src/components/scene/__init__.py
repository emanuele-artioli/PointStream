"""Scene classifiers.

Implementations live in sibling modules; this module holds the registry.
Construction targets are import strings, so importing this module does not load
torch, cv2, or encoder binaries. Do not change ``REGISTRY`` or its axis string
— the parent package and the shared smoke test key on both.
"""

from src.contracts.registry import BackendSpec, Registry

REGISTRY: Registry[object] = Registry("scene")

REGISTRY.register(
    BackendSpec(
        name="hsv-histogram",
        target="src.components.scene.hsv:HsvHistogramClassifier",
        summary="Split on HSV-histogram cuts; route point→semantic, interlude→fallback.",
    )
)
