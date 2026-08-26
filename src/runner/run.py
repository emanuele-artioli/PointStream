"""One run path: chunk loop, routing, accounting, both quality views.

A single-chunk clip is this loop with one iteration. There is no preview
path and no flag that skips evaluation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.contracts.conditioning import GenerationParams
from src.contracts.config import PointstreamConfig, validate
from src.contracts.errors import ConfigValueError
from src.contracts.lattice import (
    ART_BACKGROUND_MODEL,
    ART_DELIVERED,
    ART_QUALITY,
    ART_RESIDUAL_STREAM,
    STAGE_BACKGROUND,
    STAGE_GENERATION,
    StageLattice,
)
from src.pipeline.dag.graph import StageCallable
from src.pipeline.encoder.encoder import SOURCE, Encoder
from src.pipeline.reconstruction.background import BackgroundResolver
from src.pipeline.reconstruction.clips import as_clip
from src.pipeline.reconstruction.dispatch import GeneratorRef
from src.pipeline.reconstruction.quality import (
    Closeness,
    QualityEvaluator,
    QualityReport,
    measure_symmetry,
)
from src.pipeline.reconstruction.reconstruct import (
    ObjectRequest,
    ReconstructionRequest,
    ReconstructionResult,
    reconstruct,
)
from src.pipeline.residual.signal import ResidualResult, apply_residual
from src.runner.accounting import SizesBytes
from src.runner.routing import (
    bind_backends,
    bind_evaluator,
    bind_generator,
    generation_params,
)
from src.runner.stages import (
    OBJECTS,
    StageContext,
    _as_background,
    _delivered_frames,
    _subjects_for_reconstruct,
    ledger_from_bag,
)


@dataclass(frozen=True)
class ChunkResult:
    """One chunk after the single run path."""

    frames: np.ndarray
    """The client's clip, after the residual is applied."""
    encoder_frames: np.ndarray
    """The encoder's copy of the same thing, at the same point in the pipeline.
    Kept so symmetry over a whole run is the concatenation of the per-chunk
    pairs rather than a second, differently-derived comparison."""
    reconstruction: ReconstructionResult
    quality: QualityReport
    delivered_quality: QualityReport
    sizes: SizesBytes
    symmetry: Closeness
    bag: dict[str, object]


@dataclass(frozen=True)
class RunResult:
    """What a run returns. Importable; nothing here is scraped from stdout."""

    frames: np.ndarray
    quality: QualityReport
    delivered_quality: QualityReport
    sizes: SizesBytes
    symmetry: Closeness
    chunks: tuple[ChunkResult, ...]
    lattice: StageLattice

    @property
    def sizes_bytes(self) -> dict[str, int | float]:
        return self.sizes.as_dict()


def run(
    config: PointstreamConfig,
    chunks: Sequence[np.ndarray],
    *,
    backends: Mapping[str, StageCallable] | None = None,
    generator: GeneratorRef | None = None,
    bind_generator_fn: Callable[[], GeneratorRef] | None = None,
    evaluator: QualityEvaluator | None = None,
    objects: Sequence[tuple[ObjectRequest, ...]] | None = None,
    components: Mapping[str, object] | None = None,
    builders: Mapping[str, Callable[..., Any]] | None = None,
) -> RunResult:
    """Encode, reconstruct, score, and account every chunk.

    Args:
        config: The lattice corner and residual knobs. ``validate`` runs, but
            this function does not call ``assert_coherent`` — C2 does that
            inside ``Encoder.build`` once the generator's ``requires`` are
            known.
        chunks: Source clips, already arrays, in track order. A lone clip is
            still this loop. Do not rebuild filenames.
        backends: Optional roster overlay. Disabled-stage callables that sit
            in this mapping must not be invoked.
        generator: Injected ``GeneratorRef``. Ignored when generation is off.
        bind_generator_fn: Constructs the ref only if generation is on and
            ``generator`` was not passed. All-off must not call this.
        evaluator: Injected scorer. Defaults to C1's numpy PSNR floor.
        objects: Per-chunk ``ObjectRequest`` tuples, aligned with ``chunks``.
        components: Optional already-built perception backends (``detector``,
            ``pose``, ``segmenter``, ``appearance``, ``motion``, ``temporal``).
            Tests use this so a name swap does not load YOLO. A real run leaves
            it empty and routing builds from the config the first time a stage
            needs the backend.
        builders: Optional per-axis factory ``(name, **kwargs) -> backend``.
            Changing a config name must change which factory argument is
            passed; that is the proof the name reaches the run.
    """
    if not chunks:
        raise ValueError("run needs at least one source chunk; a reconstruction of nothing cannot be scored.")
    if objects is not None and len(objects) != len(chunks):
        raise ValueError(
            f"objects has {len(objects)} entries for {len(chunks)} chunks. "
            "Pair by track position, one tuple per chunk."
        )

    validate(config)
    lattice = config.stages
    generation_on = lattice.is_enabled(STAGE_GENERATION)
    ref = bind_generator(config, injected=generator, factory=bind_generator_fn)
    scorer = bind_evaluator(evaluator, config)
    resolver = BackgroundResolver()
    bound = dict(components or {})
    ctx = StageContext(
        lattice=lattice,
        residual=config.residual,
        generator=ref,
        evaluator=scorer,
        resolver=resolver,
        seed=config.run.seed,
        params=generation_params(config),
        config=config,
        builders=builders,
        detector=bound.get("detector"),
        pose_estimator=bound.get("pose"),
        segmenter=bound.get("segmenter"),
        appearance_encoder=bound.get("appearance"),
        motion_encoder=bound.get("motion"),
        temporal_policy=bound.get("temporal"),
    )
    roster = bind_backends(ctx, backends)
    conditioning = tuple(ref.requires) if ref is not None else ()
    encoder = Encoder.build(lattice, roster, conditioning=conditioning)

    results: list[ChunkResult] = []
    for index, raw in enumerate(chunks):
        source = as_clip(raw, path=f"{SOURCE}[{index}]")
        if config.run.max_frames is not None:
            source = source[: config.run.max_frames]
        chunk_objects = objects[index] if objects is not None else ()
        bag = encoder.encode({SOURCE: source, OBJECTS: chunk_objects})
        results.append(
            _finish_chunk(
                bag=bag,
                source=source,
                lattice=lattice,
                generation_on=generation_on,
                ref=ref,
                scorer=scorer,
                resolver=resolver,
                seed=config.run.seed,
                params=ctx.params,
                objects=chunk_objects,
            )
        )

    return _assemble(results, lattice=lattice, scorer=scorer)


def _finish_chunk(
    *,
    bag: dict[str, object],
    source: np.ndarray,
    lattice: StageLattice,
    generation_on: bool,
    ref: GeneratorRef | None,
    scorer: QualityEvaluator,
    resolver: BackgroundResolver,
    seed: int,
    params: GenerationParams,
    objects: tuple[ObjectRequest, ...],
) -> ChunkResult:
    delivered_quality = bag.get(ART_QUALITY)
    if not isinstance(delivered_quality, QualityReport):
        raise ConfigValueError(
            "runner.metrics",
            "ART_QUALITY is missing or is not a QualityReport. Metrics is a "
            "required stage; a run that skipped it is a failed run.",
        )

    view = _as_background(bag.get(ART_BACKGROUND_MODEL) or bag.get(STAGE_BACKGROUND))
    client_objects = _subjects_for_reconstruct(bag) or objects
    client = reconstruct(
        ReconstructionRequest(
            lattice=lattice,
            source=source,
            background=view,
            objects=client_objects,
            generator=ref if generation_on else None,
            evaluator=scorer,
            resolver=resolver,
            seed=seed,
            params=params,
        )
    )
    if client.quality is None:
        raise ConfigValueError(
            "runner.reconstruction",
            "reconstruct() returned no QualityReport. Every path must score.",
        )

    residual = bag.get(ART_RESIDUAL_STREAM)
    if isinstance(residual, ResidualResult):
        frames = apply_residual(client.frames, residual.payload)
        encoder_frames = residual.reconstructed
    else:
        frames = client.frames
        encoder_frames = _encoder_frames(bag, client.frames)

    # Both sides at the same point in the pipeline. `residual.reconstructed` is
    # the encoder's own copy *after* the residual is applied, so it belongs
    # against the client's clip after the residual, not against the client's
    # unaided reconstruction. Comparing across that step measured the residual
    # rather than the encoder/client gap, and reported a mismatch on every
    # corner where the residual does anything at all.
    symmetry = measure_symmetry(encoder_frames, frames)
    sizes = ledger_from_bag(bag, source)
    return ChunkResult(
        frames=frames,
        encoder_frames=encoder_frames,
        reconstruction=client,
        quality=client.quality,
        delivered_quality=delivered_quality,
        sizes=sizes,
        symmetry=symmetry,
        bag=bag,
    )


def _encoder_frames(bag: Mapping[str, object], fallback: np.ndarray) -> np.ndarray:
    delivered = bag.get(ART_DELIVERED)
    if delivered is not None:
        return _delivered_frames(delivered)
    return fallback


def _assemble(
    results: Sequence[ChunkResult],
    *,
    lattice: StageLattice,
    scorer: QualityEvaluator,
) -> RunResult:
    frames = np.concatenate([item.frames for item in results], axis=0)
    if len(results) == 1:
        quality = results[0].quality
        delivered_quality = results[0].delivered_quality
    else:
        sources = np.concatenate(
            [as_clip(np.asarray(item.bag[SOURCE]), path=SOURCE) for item in results],
            axis=0,
        )
        recon = np.concatenate([item.reconstruction.frames for item in results], axis=0)
        masks = [item.reconstruction.object_mask for item in results]
        present = [mask for mask in masks if mask is not None]
        object_mask: np.ndarray | None = (
            np.concatenate(present, axis=0) if present and len(present) == len(masks) else None
        )
        quality = scorer.evaluate(sources, recon, object_mask=object_mask)
        delivered = np.concatenate(
            [_delivered_frames(item.bag[ART_DELIVERED]) for item in results],
            axis=0,
        )
        delivered_quality = scorer.evaluate(sources, delivered, object_mask=object_mask)
    sizes = results[0].sizes
    for extra in results[1:]:
        sizes = sizes + extra.sizes
    if not sizes.parts_fit():
        raise ConfigValueError(
            "runner.sizes",
            f"payload parts sum to {sizes.parts_sum} bytes, more than "
            f"transport_total {sizes.transport_total}. One ledger, and it must add up.",
        )
    return RunResult(
        frames=frames,
        quality=quality,
        delivered_quality=delivered_quality,
        sizes=sizes,
        symmetry=_combined_symmetry(results),
        chunks=tuple(results),
        lattice=lattice,
    )


def _combined_symmetry(results: Sequence[ChunkResult]) -> Closeness:
    """The run's encoder/client closeness: the per-chunk pairs, concatenated.

    Deriving it a second way here is how the two comparisons drift apart, so
    it reuses exactly the clips `_finish_chunk` already paired.
    """
    if len(results) == 1:
        return results[0].symmetry
    encoder = np.concatenate([item.encoder_frames for item in results], axis=0)
    client = np.concatenate([item.frames for item in results], axis=0)
    return measure_symmetry(encoder, client)
