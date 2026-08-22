"""Motion representations.

Implementations live in sibling modules; this module holds the registry.
Construction targets are import strings, so importing this module does not load
torch, cv2, or encoder binaries. Do not change ``REGISTRY`` or its axis string
— the parent package and the shared smoke test key on both.
"""

from src.contracts.capabilities import (
    MOTION_ENCODED_VIDEO,
    MOTION_KEYPOINTS,
    MOTION_SPARSE_TRAJECTORIES,
    motion,
)
from src.contracts.registry import BackendSpec, Registry

REGISTRY: Registry[object] = Registry("motion")

REGISTRY.register(
    BackendSpec(
        name=MOTION_KEYPOINTS,
        target="src.components.motion.keypoints:KeypointMotionEncoder",
        capabilities=motion(MOTION_KEYPOINTS),
        summary="Per-frame pose vector under a declared wire schema.",
    )
)
REGISTRY.register(
    BackendSpec(
        name=MOTION_SPARSE_TRAJECTORIES,
        target="src.components.motion.trajectories:SparseTrajectoryEncoder",
        capabilities=motion(MOTION_SPARSE_TRAJECTORIES),
        summary="A handful of tracked points. Dense flow is refused.",
    )
)
REGISTRY.register(
    BackendSpec(
        name=MOTION_ENCODED_VIDEO,
        target="src.components.motion.encoded_video:EncodedVideoMotionEncoder",
        capabilities=motion(MOTION_ENCODED_VIDEO),
        summary="Object crop encoded as a literal video after the appearance keyframe.",
    )
)
