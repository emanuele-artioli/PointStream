"""Assemble a reconstructed clip from the enabled lattice corner.

Each concern stays in its own module. This file only sequences them and
honours the lattice: a disabled background is zeros (residual absorbs it);
disabled generation means subjects are not composited; the all-off corner
is the source video, proven by bit-identity, not by a special codec path.

Quality is scored on every return. There is no reconstruction path that
skips evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.contracts.conditioning import ConditioningBundle, GenerationParams
from src.contracts.errors import ConfigValueError
from src.contracts.lattice import (
    STAGE_BACKGROUND,
    STAGE_GENERATION,
    STAGE_SEGMENTATION,
    StageLattice,
)
from src.pipeline.reconstruction.background import (
    BackgroundModelView,
    BackgroundResolver,
)
from src.pipeline.reconstruction.clips import Clip, as_clip
from src.pipeline.reconstruction.compositor import Placement, composite_clip, heuristic_mask
from src.pipeline.reconstruction.device import DeviceDecision, DevicePolicy
from src.pipeline.reconstruction.dispatch import GeneratorRef, dispatch
from src.pipeline.reconstruction.quality import (
    NumpyPsnrEvaluator,
    QualityEvaluator,
    QualityReport,
    union_object_mask,
)


@dataclass(frozen=True)
class ObjectRequest:
    """One object the reconstruction may generate and place.

    ``appearance`` is the crop the generator (or a supplied-pixel path) uses.
    ``conditioning`` is what the generator declared it needs; when omitted,
    a bundle is built from appearance, mask, bbox and identity.
    """

    object_id: str
    appearance: np.ndarray
    bbox: tuple[int, int, int, int]
    mask: np.ndarray | None = None
    frame_index: int = 0
    conditioning: ConditioningBundle | None = None
    supplied_crop: np.ndarray | None = None
    """When set, skip generation for this object and composite these pixels.
    Rigid shapes and already-decoded appearance crops use this."""


@dataclass(frozen=True)
class ReconstructionRequest:
    """Everything reconstruction needs, injected rather than looked up."""

    lattice: StageLattice
    source: np.ndarray
    background: BackgroundModelView | None = None
    objects: tuple[ObjectRequest, ...] = ()
    generator: GeneratorRef | None = None
    params: GenerationParams | None = None
    seed: int = 1337
    policy: DevicePolicy = field(default_factory=DevicePolicy)
    evaluator: QualityEvaluator | None = None
    resolver: BackgroundResolver | None = None


@dataclass(frozen=True)
class ReconstructionResult:
    """Reconstructed clip, the quality record, and which path produced it."""

    frames: Clip
    quality: QualityReport
    path: str
    device: DeviceDecision
    object_mask: np.ndarray | None = None


def reconstruct(request: ReconstructionRequest) -> ReconstructionResult:
    """Rebuild frames for this lattice corner and score them against ``source``."""
    source = as_clip(request.source, path="source")
    evaluator = request.evaluator if request.evaluator is not None else NumpyPsnrEvaluator()
    policy = request.policy
    frame_count, height, width, _ = source.shape

    if request.lattice.is_source_passthrough:
        frames = source.copy()
        quality = evaluator.evaluate(source, frames)
        return ReconstructionResult(
            frames=frames,
            quality=quality,
            path="source-passthrough",
            device=DeviceDecision("cpu"),
        )

    resolver = request.resolver if request.resolver is not None else BackgroundResolver()
    background_on = request.lattice.is_enabled(STAGE_BACKGROUND)
    if background_on:
        background_frames, bg_decision = resolver.frames_for(
            request.background,
            frame_count=frame_count,
            height=height,
            width=width,
            policy=policy,
        )
    else:
        background_frames, bg_decision = resolver.frames_for(
            None,
            frame_count=frame_count,
            height=height,
            width=width,
            policy=policy,
        )

    generation_on = request.lattice.is_enabled(STAGE_GENERATION)
    use_heuristic = not request.lattice.is_enabled(STAGE_SEGMENTATION)
    placements: list[Placement] = []
    gen_decision = bg_decision

    to_generate = [item for item in request.objects if item.supplied_crop is None]
    supplied = [item for item in request.objects if item.supplied_crop is not None]

    if generation_on and to_generate:
        if request.generator is None:
            raise ConfigValueError(
                "reconstruction.generation",
                "the generation stage is enabled but no generator was injected. "
                "The runner binds the registry; reconstruction does not look backends up.",
            )
        bundles = tuple(_bundle_for(item) for item in to_generate)
        crops, gen_decision = dispatch(
            request.generator,
            bundles,
            seed=request.seed,
            params=request.params,
            policy=policy,
        )
        for item, crop in zip(to_generate, crops, strict=True):
            placements.append(_placement(item, crop))
    elif generation_on and not to_generate and not supplied:
        raise ConfigValueError(
            "reconstruction.generation",
            "generation is enabled but no objects were supplied, so the generator "
            "has nothing to draw. Disable generation, or pass objects — whatever "
            "generation would have contributed lands in the residual.",
        )

    for item in supplied:
        assert item.supplied_crop is not None
        placements.append(_placement(item, item.supplied_crop))

    frames = composite_clip(
        background_frames,
        tuple(placements),
        use_heuristic_mask=use_heuristic,
    )

    object_masks = []
    for item in request.objects:
        if use_heuristic or item.mask is None:
            object_masks.append(heuristic_mask(item.bbox, height, width))
        else:
            object_masks.append(np.asarray(item.mask, dtype=bool))
    combined = union_object_mask(object_masks, frames=frame_count, height=height, width=width)
    quality = evaluator.evaluate(source, frames, object_mask=combined)
    path = _path_label(request.lattice, generation_on=generation_on, background_on=background_on)
    return ReconstructionResult(
        frames=frames,
        quality=quality,
        path=path,
        device=gen_decision,
        object_mask=combined,
    )


def _bundle_for(item: ObjectRequest) -> ConditioningBundle:
    if item.conditioning is not None:
        return item.conditioning
    return ConditioningBundle(
        appearance=item.appearance,
        mask=item.mask,
        bbox=item.bbox,
        frame_index=item.frame_index,
        object_id=item.object_id,
    )


def _placement(item: ObjectRequest, crop: np.ndarray) -> Placement:
    return Placement(
        crop=crop,
        bbox=item.bbox,
        mask=item.mask,
        object_id=item.object_id,
        frame_index=item.frame_index,
    )


def _path_label(lattice: StageLattice, *, generation_on: bool, background_on: bool) -> str:
    parts = []
    if background_on:
        parts.append("background")
    if generation_on:
        parts.append("generation")
    if lattice.is_enabled(STAGE_SEGMENTATION):
        parts.append("masks")
    return "+".join(parts) if parts else "unaided"
