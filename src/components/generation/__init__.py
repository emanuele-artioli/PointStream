"""Frame and sequence generators.

Implementations live in sibling modules; this module holds the registry.
Construction targets are import strings, so importing this module does not load
torch, cv2, or encoder binaries. Do not change ``REGISTRY`` or its axis string
— the parent package and the shared smoke test key on both.
"""

from __future__ import annotations

from typing import Any

from src.contracts.capabilities import (
    APPEARANCE_COMPRESSED_IMAGE,
    APPEARANCE_DIFFUSION_LATENT,
    APPEARANCE_IMAGE_EMBEDDING,
    CAP_PER_FRAME,
    CAP_TEMPORAL_SEQUENCE,
    CONDITION_APPEARANCE,
    CONDITION_CANNY,
    CONDITION_MASK,
    CONDITION_MOTION_FIELD,
    CONDITION_POSE,
    MOTION_ENCODED_VIDEO,
    MOTION_KEYPOINTS,
    MOTION_SPARSE_TRAJECTORIES,
    appearance,
    domains,
    motion,
)
from src.contracts.registry import BackendSpec, Registry

REGISTRY: Registry[object] = Registry("generator")


def _add(
    name: str,
    target: str,
    *,
    summary: str,
    capabilities: frozenset[str],
    requires: frozenset[str] = frozenset(),
    aliases: tuple[str, ...] = (),
    defaults: dict[str, Any] | None = None,
) -> None:
    REGISTRY.register(
        BackendSpec(
            name=name,
            target=target,
            aliases=aliases,
            capabilities=capabilities,
            requires=requires,
            defaults=defaults or {},
            summary=summary,
        )
    )


_PER_FRAME_IMAGE_POSE = (
    appearance(APPEARANCE_COMPRESSED_IMAGE)
    | motion(MOTION_KEYPOINTS)
    | {CAP_PER_FRAME}
)

_add(
    "canny-controlnet",
    "src.components.generation.controlnet:ControlNetGenerator",
    summary="ControlNet canny. Structure from edges, colour from the appearance crop.",
    capabilities=_PER_FRAME_IMAGE_POSE,
    requires=frozenset({CONDITION_CANNY, CONDITION_APPEARANCE}),
    defaults={"variant": "canny"},
)
_add(
    "seg-controlnet",
    "src.components.generation.controlnet:ControlNetGenerator",
    summary="ControlNet segmentation. Structure from the object mask.",
    capabilities=_PER_FRAME_IMAGE_POSE,
    requires=frozenset({CONDITION_MASK, CONDITION_APPEARANCE}),
    defaults={"variant": "seg"},
)
_add(
    "pose-controlnet",
    "src.components.generation.controlnet:ControlNetGenerator",
    summary="ControlNet OpenPose. Structure from a rendered skeleton.",
    capabilities=_PER_FRAME_IMAGE_POSE,
    requires=frozenset({CONDITION_POSE, CONDITION_APPEARANCE}),
    aliases=("caption-controlnet", "controlnet"),
    defaults={"variant": "pose"},
)
_add(
    "trajectory-controlnet",
    "src.components.generation.controlnet:ControlNetGenerator",
    summary=(
        "ControlNet OpenPose driven by a rendered trajectory image. "
        "Same backbone as pose-controlnet; the control image changes."
    ),
    capabilities=(
        appearance(APPEARANCE_COMPRESSED_IMAGE)
        | motion(MOTION_SPARSE_TRAJECTORIES)
        | {CAP_PER_FRAME}
    ),
    requires=frozenset({CONDITION_MOTION_FIELD, CONDITION_APPEARANCE}),
    aliases=("trajectory-render",),
    defaults={"variant": "trajectory"},
)
_add(
    "ip-adapter-controlnet",
    "src.components.generation.controlnet:ControlNetGenerator",
    summary=(
        "Stock SD-1.5 + h94/IP-Adapter + stock OpenPose ControlNet. "
        "Appearance through the adapter, pose through ControlNet. "
        "The tennis directory named ip-adapter-controlnet is a mislabelled "
        "segmentation ControlNet and is not loaded."
    ),
    capabilities=(
        appearance(APPEARANCE_COMPRESSED_IMAGE, APPEARANCE_IMAGE_EMBEDDING)
        | motion(MOTION_KEYPOINTS)
        | {CAP_PER_FRAME}
    ),
    requires=frozenset({CONDITION_APPEARANCE, CONDITION_POSE}),
    defaults={"variant": "ip-adapter"},
)
_add(
    "multi-controlnet",
    "src.components.generation.controlnet:ControlNetGenerator",
    summary="Multi-ControlNet. Pose and mask as separate conditions, not a tuple in one slot.",
    capabilities=_PER_FRAME_IMAGE_POSE,
    requires=frozenset({CONDITION_POSE, CONDITION_MASK, CONDITION_APPEARANCE}),
    defaults={"variant": "multi"},
)
_add(
    "pix2pix",
    "src.components.generation.pix2pix:Pix2PixGenerator",
    summary="Pix2Pix UNet. Pose RGB concatenated with the appearance crop.",
    capabilities=_PER_FRAME_IMAGE_POSE,
    requires=frozenset({CONDITION_POSE, CONDITION_APPEARANCE}),
)
_add(
    "spade4tennis",
    "src.components.generation.spade:Spade4TennisGenerator",
    summary="SPADE4Tennis. Tennis-specific SPADE generator; not a general human model.",
    capabilities=_PER_FRAME_IMAGE_POSE | domains("tennis"),
    requires=frozenset({CONDITION_POSE, CONDITION_APPEARANCE}),
)
_add(
    "animate-anyone",
    "src.components.generation.animate_anyone:AnimateAnyoneGenerator",
    summary=(
        "Animate-Anyone pose-to-video. Fine-tuned on 7 matches, 114 tracks "
        "(assets/dataset/pointstream_aa_meta.json); not a single match."
    ),
    capabilities=_PER_FRAME_IMAGE_POSE | {CAP_TEMPORAL_SEQUENCE},
    requires=frozenset({CONDITION_POSE, CONDITION_APPEARANCE}),
    aliases=("animate_anyone", "animateanyone"),
)
_add(
    "stable-animator",
    "src.components.generation.stable_animator:StableAnimatorGenerator",
    summary=(
        "StableAnimator pose-to-video. Adapter Apache-2.0 on HF card "
        "FrancisRing/StableAnimator (checked 2026-08-22); inference needs "
        "SVD-XT (Stability AI, not bundled). GitHub code is MIT."
    ),
    capabilities=_PER_FRAME_IMAGE_POSE | {CAP_TEMPORAL_SEQUENCE},
    requires=frozenset({CONDITION_POSE, CONDITION_APPEARANCE}),
    aliases=("stableanimator", "stable_animator"),
)
_add(
    "mofa-video",
    "src.components.generation.mofa:MofaVideoGenerator",
    summary=(
        "MOFA-Video (candidate). Sparse-trajectory SVD adapter. Construction "
        "refused: SVD weights are Stability-AI-licensed and not bundled."
    ),
    capabilities=(
        appearance(APPEARANCE_COMPRESSED_IMAGE, APPEARANCE_DIFFUSION_LATENT)
        | motion(MOTION_SPARSE_TRAJECTORIES)
        | {CAP_TEMPORAL_SEQUENCE}
    ),
    requires=frozenset({CONDITION_APPEARANCE, CONDITION_MOTION_FIELD}),
    aliases=("mofa-trajectories",),
)
_add(
    "upscale-refine",
    "src.components.generation.upscale:UpscaleRefineGenerator",
    summary="No diffusion: bicubic upsample plus unsharp refine. Cheap baseline.",
    capabilities=(
        appearance(APPEARANCE_COMPRESSED_IMAGE)
        | motion(MOTION_ENCODED_VIDEO)
        | {CAP_PER_FRAME}
    ),
    requires=frozenset({CONDITION_APPEARANCE}),
)


def validate(config: Any) -> None:
    """Wire ``config.validate_backends`` for generation, appearance and motion.

    Passes ``generators=REGISTRY`` so an undecodable appearance/motion pair is
    rejected with the workable pairings named. ``"none"`` is skipped by
    ``validate_backends``; this axis does not register a generator of that name.
    """
    from src.components.appearance import REGISTRY as APPEARANCE
    from src.components.motion import REGISTRY as MOTION
    from src.contracts.config import validate_backends

    validate_backends(
        config,
        generators=REGISTRY,
        registries={
            "generator": REGISTRY,
            "appearance": APPEARANCE,
            "motion": MOTION,
        },
    )
