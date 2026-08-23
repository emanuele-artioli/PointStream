"""Identity trackers.

Implementations live in sibling modules; this module holds the registry.
Construction targets are import strings, so importing this module does not load
torch, cv2, or encoder binaries. Do not change ``REGISTRY`` or its axis string
— the parent package and the shared smoke test key on both.
"""

from src.contracts.registry import BackendSpec, Registry

REGISTRY: Registry[object] = Registry("tracking")

REGISTRY.register(
    BackendSpec(
        name="tracker",
        target="src.components.tracking.tracker:IdentityTracker",
        summary="IoU identity plus composed track-recovery (not a detector base class).",
    )
)
