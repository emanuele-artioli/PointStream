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


def encoder_side(ctx: StageContext, bag: Mapping[str, Any]) -> Any:
    """The encoder's own copy of what the client will build.

    Shared by the residual stage and the codec stage so there is one definition
    of it. Generated crops arrive as ``supplied_crop``, so this never dispatches
    a generator a second time.
    """
    source = as_clip(bag[SOURCE], path=SOURCE)
    view = _as_background(bag.get(ART_BACKGROUND_MODEL) or bag.get(STAGE_BACKGROUND))
    objects = _with_supplied_crops(_subjects(bag), bag.get(ART_GENERATED_FRAMES))
    return reconstruct(
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


def make_residual(ctx: StageContext) -> StageCallable:
    """Encoder-side residual. Uses generated crops; does not dispatch again."""

    def residual(bag: Mapping[str, Any]) -> ResidualResult:
        source = as_clip(bag[SOURCE], path=SOURCE)
        built = encoder_side(ctx, bag)
        return compute_residual(
            source,
            built.frames,
            lattice=ctx.lattice,
            residual=ctx.residual,
            actor_mask=built.object_mask,
        )

    return residual


def make_codec(ctx: StageContext) -> StageCallable:
    """Identity encode: the pixels that will be delivered, plus a byte count.

    Three cases, and the middle one is the one that was wrong. With a residual
    the delivered clip is the reconstruction plus that residual. On the all-off
    corner it is the source, because all-off *is* the source. Between those two
    sits the corner with semantic stages on and the residual switched off — and
    that used to fall through to "deliver the source", so a residual-absent run
    reported an infinite PSNR and a perfect copy of the video it was supposed to
    be approximating. It has to deliver the unaided reconstruction, which is
    exactly what "nothing corrects generation error" means in the catalogue.

    One corner it still cannot serve: generation on *and* residual off. The
    catalogue does not list `generated-frames` among the codec stage's inputs
    (`src/contracts/lattice.py`, `STAGE_CODEC.optional_inputs`), so the DAG is
    free to run the codec before the generator and the crops are not there to
    composite. Rather than dispatch a second generator — which would be a
    different sample, not the encoder's copy — that case says so on the
    artifact it returns.
    """

    def codec(bag: Mapping[str, Any]) -> dict[str, Any]:
        source = as_clip(bag[SOURCE], path=SOURCE)
        residual = bag.get(ART_RESIDUAL_STREAM)
        if isinstance(residual, ResidualResult) and not residual.payload.is_absent:
            return {
                "frames": residual.reconstructed,
                "byte_count": int(residual.payload.byte_count),
            }
        if ctx.lattice.is_source_passthrough:
            return {"frames": source, "byte_count": int(source.nbytes)}
        if ctx.lattice.is_enabled(STAGE_GENERATION) and bag.get(ART_GENERATED_FRAMES) is None:
            return {
                "frames": source,
                "byte_count": int(source.nbytes),
                "fallback_reason": (
                    "generation is on and the residual is off, but no "
                    "ART_GENERATED_FRAMES reached the codec stage: the catalogue "
                    "does not declare it as an input, so the DAG may order codec "
                    "before generation. This clip is the SOURCE, not a "
                    "reconstruction — do not read its quality as a result."
                ),
            }
        built = encoder_side(ctx, bag)
        # No encoder runs here, so there is no coded size to report. The
        # measured semantic parts are what this corner actually transmits, and
        # inventing anything else would be a modelled byte count wearing a
        # measurement's clothes.
        return {"frames": built.frames, "byte_count": _semantic_bytes(bag)}

    return codec


def _semantic_bytes(bag: Mapping[str, Any]) -> int:
    return _panorama_bytes(bag) + _measured_actor_bytes(bag)


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
        STAGE_CODEC: make_codec(ctx),
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
    return sizes_bytes(
        source=int(clip.nbytes),
        residual=residual_bytes,
        panorama=_panorama_bytes(bag),
        actor_reference=_measured_actor_bytes(bag),
    )


def _panorama_bytes(bag: Mapping[str, Any]) -> int:
    """Raw plate size.

    Not a coded size: `background.codec` and `background.jpeg_quality` reach
    nothing on this path, so a plate counted here is uncompressed pixels. The
    number is honest about what it is rather than modelling a JPEG that was
    never made.
    """
    view = bag.get(ART_BACKGROUND_MODEL) or bag.get(STAGE_BACKGROUND)
    if not isinstance(view, BackgroundModelView) or view.plate is None:
        return 0
    if view.deferred_to_residual or view.mode == "none":
        return 0
    return int(np.asarray(view.plate).nbytes)


def _measured_actor_bytes(bag: Mapping[str, Any]) -> int:
    payload = bag.get(ART_APPEARANCE_PAYLOAD)
    if isinstance(payload, Mapping) and "byte_count" in payload:
        return int(payload["byte_count"])
    return 0
