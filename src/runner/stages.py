"""Default ``StageCallable`` bindings the runner injects into C2.

Perception stages are pass-throughs: they forward artifacts the caller put
in the bag. They do not load detectors. Generation calls C1 ``dispatch``.
Residual composes an encoder-side reconstruction from bag artifacts (using
generated crops as ``supplied_crop`` so it does not dispatch again) and
runs ``compute_residual``. Codec / transport / metrics are identity-roundtrip
enough to score delivered pixels — they do not shell out.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from src.contracts.conditioning import ConditioningBundle, GenerationParams
from src.contracts.errors import ConfigValueError
from src.contracts.lattice import (
    ART_APPEARANCE_PAYLOAD,
    ART_BACKGROUND_MODEL,
    ART_BITSTREAM,
    ART_DELIVERED,
    ART_GENERATED_FRAMES,
    ART_RESIDUAL_STREAM,
    ART_SUBJECTS,
    STAGE_APPEARANCE,
    STAGE_BACKGROUND,
    STAGE_CODEC,
    STAGE_DETECTION,
    STAGE_GENERATION,
    STAGE_METRICS,
    STAGE_MOTION,
    STAGE_POSE,
    STAGE_RESIDUAL,
    STAGE_RIGID,
    STAGE_SCENE,
    STAGE_SEGMENTATION,
    STAGE_SELECTION,
    STAGE_TEMPORAL,
    STAGE_TRACKING,
    STAGE_TRANSPORT,
    StageLattice,
)
from src.contracts.config import ResidualConfig
from src.pipeline.dag.graph import StageCallable
from src.pipeline.encoder.encoder import SOURCE
from src.pipeline.reconstruction.background import BackgroundModelView, BackgroundResolver
from src.pipeline.reconstruction.clips import as_clip
from src.pipeline.reconstruction.dispatch import GeneratorRef, dispatch
from src.pipeline.reconstruction.quality import QualityEvaluator, QualityReport
from src.pipeline.reconstruction.reconstruct import (
    ObjectRequest,
    ReconstructionRequest,
    reconstruct,
)
from src.pipeline.residual.signal import ResidualResult, compute_residual
from src.runner.accounting import SizesBytes, measured, sizes_bytes

#: Objects the runner placed on the bag, before detection names them subjects.
OBJECTS = "objects"


@dataclass
class StageContext:
    """Shared bindings for the default stage callables of one run."""

    lattice: StageLattice
    residual: ResidualConfig
    generator: GeneratorRef | None
    evaluator: QualityEvaluator
    resolver: BackgroundResolver
    seed: int
    params: GenerationParams


def _subjects(bag: Mapping[str, Any]) -> tuple[ObjectRequest, ...]:
    raw = bag.get(ART_SUBJECTS)
    if raw is None:
        raw = bag.get(STAGE_DETECTION)
    if raw is None:
        raw = bag.get(OBJECTS)
    if raw is None:
        return ()
    if isinstance(raw, ObjectRequest):
        return (raw,)
    return tuple(item for item in raw)


def _as_background(value: object) -> BackgroundModelView | None:
    if value is None:
        return None
    if isinstance(value, BackgroundModelView):
        return value
    raise ConfigValueError(
        "runner.background",
        "the background stage must return a BackgroundModelView; "
        f"got {type(value).__name__}. The runner does not unpack a components artifact.",
    )


def _delivered_frames(delivered: object) -> np.ndarray:
    if isinstance(delivered, Mapping) and "frames" in delivered:
        return as_clip(delivered["frames"], path="delivered")
    return as_clip(np.asarray(delivered), path="delivered")


def detection(bag: Mapping[str, Any]) -> tuple[ObjectRequest, ...]:
    """Forward pre-supplied objects. Does not run a detector."""
    return _subjects(bag)


def background(bag: Mapping[str, Any]) -> BackgroundModelView:
    """A static plate from the first source frame. Identity warp."""
    source = as_clip(bag[SOURCE], path=SOURCE)
    return BackgroundModelView(
        plate=source[0],
        homographies=(),
        mode="full",
        width=int(source.shape[2]),
        height=int(source.shape[1]),
        scene_id=None,
    )


def make_generation(ctx: StageContext) -> StageCallable:
    """Encoder-side generation: C1 ``dispatch`` with the run's ``GeneratorRef``."""

    def generation(bag: Mapping[str, Any]) -> tuple[np.ndarray, ...]:
        if ctx.generator is None:
            raise ConfigValueError(
                "runner.generation",
                "the generation stage is enabled but no GeneratorRef was bound.",
            )
        subjects = _subjects(bag)
        bundles = tuple(_bundle_for(item) for item in subjects)
        crops, _decision = dispatch(
            ctx.generator,
            bundles,
            seed=ctx.seed,
            params=ctx.params,
        )
        return crops

    return generation


def make_residual(ctx: StageContext) -> StageCallable:
    """Encoder-side residual. Uses generated crops; does not dispatch again."""

    def residual(bag: Mapping[str, Any]) -> ResidualResult:
        source = as_clip(bag[SOURCE], path=SOURCE)
        view = _as_background(bag.get(ART_BACKGROUND_MODEL) or bag.get(STAGE_BACKGROUND))
        generated = bag.get(ART_GENERATED_FRAMES)
        subjects = _subjects(bag)
        objects = _with_supplied_crops(subjects, generated)
        encoder_side = reconstruct(
            ReconstructionRequest(
                lattice=ctx.lattice,
                source=source,
                background=view,
                objects=objects,
                generator=None,
                evaluator=ctx.evaluator,
                resolver=ctx.resolver,
                seed=ctx.seed,
                params=ctx.params,
            )
        )
        return compute_residual(
            source,
            encoder_side.frames,
            lattice=ctx.lattice,
            residual=ctx.residual,
            actor_mask=encoder_side.object_mask,
        )

    return residual


def codec(bag: Mapping[str, Any]) -> dict[str, Any]:
    """Identity encode: pixels that will be delivered, plus a byte count."""
    source = as_clip(bag[SOURCE], path=SOURCE)
    residual = bag.get(ART_RESIDUAL_STREAM)
    if isinstance(residual, ResidualResult):
        frames = residual.reconstructed
        byte_count = residual.payload.byte_count
    else:
        frames = source
        byte_count = int(source.nbytes)
    return {"frames": frames, "byte_count": int(byte_count)}


def transport(bag: Mapping[str, Any]) -> dict[str, Any]:
    """Deliver the bitstream as-is. No subprocess, no disk."""
    bits = bag[ART_BITSTREAM]
    if isinstance(bits, Mapping):
        return dict(bits)
    return {"frames": bits, "byte_count": int(np.asarray(bits).nbytes)}


def make_metrics(ctx: StageContext) -> StageCallable:
    """Score delivered pixels against the source. Required on every path."""

    def metrics(bag: Mapping[str, Any]) -> QualityReport:
        source = as_clip(bag[SOURCE], path=SOURCE)
        delivered = bag.get(ART_DELIVERED)
        if delivered is None:
            raise ConfigValueError(
                "runner.metrics",
                "metrics ran without ART_DELIVERED. Transport must produce the payload.",
            )
        return ctx.evaluator.evaluate(source, _delivered_frames(delivered))

    return metrics


def default_backends(ctx: StageContext) -> dict[str, StageCallable]:
    """A callable for every catalogue stage. C2 ignores the disabled ones."""
    roster: dict[str, StageCallable] = {
        STAGE_SCENE: _empty,
        STAGE_DETECTION: detection,
        STAGE_SELECTION: detection,
        STAGE_TRACKING: _empty,
        STAGE_APPEARANCE: _empty,
        STAGE_MOTION: _empty,
        STAGE_TEMPORAL: _empty,
        STAGE_POSE: _empty,
        STAGE_SEGMENTATION: _empty,
        STAGE_RIGID: _empty,
        STAGE_BACKGROUND: background,
        STAGE_GENERATION: make_generation(ctx),
        STAGE_RESIDUAL: make_residual(ctx),
        STAGE_CODEC: codec,
        STAGE_TRANSPORT: transport,
        STAGE_METRICS: make_metrics(ctx),
    }
    return roster


def _empty(bag: Mapping[str, Any]) -> tuple[()]:
    _ = bag
    return ()


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


def _with_supplied_crops(
    subjects: Sequence[ObjectRequest], generated: object
) -> tuple[ObjectRequest, ...]:
    if generated is None:
        return tuple(subjects)
    if not isinstance(generated, Sequence) or isinstance(generated, (str, bytes)):
        raise ConfigValueError(
            "runner.generation",
            "ART_GENERATED_FRAMES must be a sequence of crops.",
        )
    crops: tuple[object, ...] = tuple(generated)
    if len(crops) != len(subjects):
        raise ConfigValueError(
            "runner.generation",
            f"generation returned {len(crops)} crops for {len(subjects)} subjects.",
        )
    return tuple(
        replace(item, supplied_crop=np.asarray(crop))
        for item, crop in zip(subjects, crops, strict=True)
    )


def ledger_from_bag(bag: Mapping[str, Any], source: np.ndarray) -> SizesBytes:
    """One chunk's sizes from the artifacts the DAG actually produced.

    Actor-reference bytes are not inferred from ``ObjectRequest.appearance`` —
    that array is a source crop, not a transmitted payload. Count appearance
    only when the appearance stage left a measured byte count on the bag.
    """
    clip = as_clip(source, path=SOURCE)
    residual_bytes = 0
    residual = bag.get(ART_RESIDUAL_STREAM)
    if isinstance(residual, ResidualResult):
        # The stated `WireCost` first, not `payload.byte_count`. For a lossy
        # residual `byte_count` is the *dense* array size, which does not shrink
        # when the block gate zeroes a block — so a ledger reading it reports the
        # same payload for a coarse residual as for a fine one and makes
        # coarseness look free. `src/pipeline/residual/signal.py` says as much in
        # its own docstring. The cost carries the information content (nonzero
        # bytes); for a lossless residual the two are equal, so nothing moves.
        cost = residual.payload.cost
        if cost.byte_count is not None:
            residual_bytes = measured(cost)
        else:
            residual_bytes = residual.payload.byte_count
    panorama_bytes = 0
    view = bag.get(ART_BACKGROUND_MODEL) or bag.get(STAGE_BACKGROUND)
    if isinstance(view, BackgroundModelView) and view.plate is not None:
        if not view.deferred_to_residual and view.mode != "none":
            panorama_bytes = int(np.asarray(view.plate).nbytes)
    return sizes_bytes(
        source=int(clip.nbytes),
        residual=residual_bytes,
        panorama=panorama_bytes,
        actor_reference=_measured_actor_bytes(bag),
    )


def _measured_actor_bytes(bag: Mapping[str, Any]) -> int:
    payload = bag.get(ART_APPEARANCE_PAYLOAD)
    if isinstance(payload, Mapping) and "byte_count" in payload:
        return int(payload["byte_count"])
    return 0
