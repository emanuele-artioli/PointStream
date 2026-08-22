"""Rigid-object strategies.

Implementations live in sibling modules; this module holds the registry.
Construction targets are import strings, so importing this module does not load
torch, cv2, or encoder binaries. Do not change ``REGISTRY`` or its axis string
— the parent package and the shared smoke test key on both.
"""

from src.contracts.registry import BackendSpec, Registry

REGISTRY: Registry[object] = Registry("rigid")

REGISTRY.register(
    BackendSpec(
        name="tennis",
        target="src.components.rigid.strategy:TennisRigid",
        summary="Racket convex hull plus a ball blob; either class switchable off.",
    )
)
REGISTRY.register(
    BackendSpec(
        name="racket-hull",
        target="src.components.rigid.strategy:RacketHull",
        summary="Racket only: convex hull anchored to a player wrist.",
    )
)
REGISTRY.register(
    BackendSpec(
        name="ball-difference",
        target="src.components.rigid.strategy:BallDifference",
        summary="Ball only: difference against a background plate.",
    )
)
REGISTRY.register(
    BackendSpec(
        name="ball-segmentation",
        target="src.components.rigid.strategy:BallSegmentation",
        summary="Ball only: centroid of a provided mask or box.",
    )
)
REGISTRY.register(
    BackendSpec(
        name="none",
        target="src.components.rigid.strategy:RigidNone",
        summary="No rigid objects; racket and ball land in the residual.",
    )
)
