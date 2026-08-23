"""Domain profiles and dataset plumbing.

Implementations live in sibling modules; this module holds the registry.
Construction targets are import strings, so importing this module does not load
torch, cv2, or encoder binaries. Do not change ``REGISTRY`` or its axis string
— the parent package and the shared smoke test key on both.
"""

from src.contracts.registry import BackendSpec, Registry

REGISTRY: Registry[object] = Registry("domain")

REGISTRY.register(
    BackendSpec(
        name="tennis",
        target="src.components.domain.profiles:build_tennis",
        defaults={"selector": "heuristic"},
        summary=(
            "Broadcast tennis: players, racket, ball; pan-tilt-zoom camera; "
            "a panorama background is valid."
        ),
    )
)
REGISTRY.register(
    BackendSpec(
        name="general",
        target="src.components.domain.profiles:build_general",
        defaults={"selector": "identity"},
        summary=(
            "General human video, evaluated on DAVIS clips that contain people; "
            "free-moving camera, so a panorama background is invalid."
        ),
    )
)
