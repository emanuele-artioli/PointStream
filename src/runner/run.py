"""One run path: chunk loop, routing, accounting, both quality views.

A single-chunk clip is this loop with one iteration. There is no preview
path and no flag that skips evaluation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
import time
from typing import Any

import numpy as np

from src.contracts.conditioning import GenerationParams
from src.contracts.config import PointstreamConfig, validate
from src.contracts.errors import ConfigValueError
from src.contracts.lattice import (
    ART_APPEARANCE_PAYLOAD,
    ART_BACKGROUND_MODEL,
    ART_BITSTREAM,
    ART_DELIVERED,
    ART_QUALITY,
    ART_RESIDUAL_STREAM,
    STAGE_BACKGROUND,
    STAGE_GENERATION,
    STAGE_RESIDUAL,
    STAGE_SEGMENTATION,
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
    ReconstructionResult,
)
from src.pipeline.residual.signal import ResidualResult
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


def sync_gpu() -> None:
    """Synchronize CUDA if torch is imported and CUDA is available.

    Never imports torch if it has not been imported.
    """
    import sys

    torch_mod = sys.modules.get("torch")
    if torch_mod is not None:
        try:
            if torch_mod.cuda.is_available():
                torch_mod.cuda.synchronize()
        except Exception:
            pass


def get_gpu_info() -> str | None:
    """Return primary CUDA device name if torch is loaded and CUDA is available."""
    import sys

    torch_mod = sys.modules.get("torch")
    if torch_mod is not None:
        try:
            if torch_mod.cuda.is_available():
                return str(torch_mod.cuda.get_device_name(0))
        except Exception:
            pass
    return None


def get_cpu_info() -> str:
    """Return CPU processor or architecture description."""
    import platform

    proc = platform.processor()
    return proc if proc else platform.machine()


def get_peak_memory_bytes() -> int:
    """Return peak memory in bytes for the calling process."""
    import resource

    # On Linux ru_maxrss is in KiB
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


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
    encoder_seconds: float = 0.0
    client_seconds: float = 0.0
    evaluation_seconds: float = 0.0


@dataclass(frozen=True)
class RunResult:
    """What a run returns. Importable; nothing here is scraped from stdout."""

    frames: np.ndarray
    """The client's clip with the residual applied **as the residual stage
    produced it** — before any codec ran on that residual.

    Since BP24 codes the residual this is no longer the clip the pipeline
    delivers: `make_codec` round-trips the residual payload through
    `residual.codec` and rebuilds from what came back, and *that* is what
    reaches transport. Use `delivered_frames` for anything paired with a byte
    count. Scoring this array beside a coded size is the trap
    `plans/BP24-findings.md` §4 describes — two real numbers belonging to two
    different operating points."""

    quality: QualityReport
    delivered_quality: QualityReport
    sizes: SizesBytes
    symmetry: Closeness
    chunks: tuple[ChunkResult, ...]
    lattice: StageLattice
    stage_seconds: tuple[dict[str, float], ...] = ()
    timing: dict[str, Any] = field(default_factory=dict)
    phase_seconds: dict[str, float] = field(default_factory=dict)
    encoder_seconds: float = 0.0
    client_seconds: float = 0.0
    evaluation_seconds: float = 0.0

    @property
    def sizes_bytes(self) -> dict[str, int | float]:
        return self.sizes.as_dict()

    @property
    def delivered_frames(self) -> np.ndarray:
        """What transport handed the client — the clip `sizes` is the cost of.

        `delivered_quality` is already scored on this, but the array itself was
        reachable only through `chunks[i].bag[ART_DELIVERED]`, which is how a
        caller ends up reaching for `frames` instead and pairing a coded rate
        with a pre-codec reconstruction.
        """
        return np.concatenate(
            [_delivered_frames(chunk.bag[ART_DELIVERED]) for chunk in self.chunks],
            axis=0,
        )


def run(
    config: PointstreamConfig,
    chunks: Sequence[np.ndarray],
    *,
    context_ids: Sequence[str] | None = None,
    checkpoint_dir: Path | str | None = None,
    heartbeat_interval: float | None = 600.0,
    checkpoint_identity: str | None = None,
    clock: Callable[[], float] = time.perf_counter,
    sync_fn: Callable[[], None] | None = sync_gpu,
    **kwargs: Any,
) -> RunResult:
    """Run with identity-checked recovery and whole-invocation progress.

    Injected backends/evaluators must provide a stable checkpoint_identity
    describing their implementation/configuration; opaque state is not guessed.
    A hard-killed attempt makes cumulative time a labelled lower bound.
    """
    import platform
    from src.pipeline.dag.heartbeat import Heartbeat
    from src.runner.recovery import RecoverySession, runner_identity

    started = clock()
    session = None
    with Heartbeat(
        "runner (including preparation, recovery and scoring)", interval_s=heartbeat_interval
    ):
        if checkpoint_dir is not None:
            injected = ("backends", "generator", "evaluator", "components", "builders")
            if any(kwargs.get(key) is not None for key in injected) and not checkpoint_identity:
                raise ValueError("injected implementations require checkpoint_identity")
            if config.lattice.generation:
                raise ValueError(
                    "generative RNG recovery is not supported; disable generation for checkpointed runs"
                )
            contexts = (
                tuple(context_ids)
                if context_ids is not None
                else tuple(config.background.context_id or "run" for _ in chunks)
            )
            identity = runner_identity(
                config, chunks, kwargs.get("objects"), contexts, checkpoint_identity
            )
            session = RecoverySession(
                Path(checkpoint_dir), identity, started_at=started, clock=clock
            )
        try:
            result = _run(
                config,
                chunks,
                context_ids=context_ids,
                checkpoint_dir=checkpoint_dir,
                heartbeat_interval=heartbeat_interval,
                recovery_session=session,
                clock=clock,
                sync_fn=sync_fn,
                **kwargs,
            )
        except BaseException:
            if session is not None:
                session.finish(success=False)
            raise
        elapsed = clock() - started

        lost_work_lower_bound = 0.0
        timing: dict[str, Any] = {}
        if session is not None:
            recovery_timing = session.finish(success=True)
            lost_work_lower_bound = sum(
                float(item["seconds"])
                for item in session.attempts
                if item.get("status") == "interrupted"
            )
            timing.update(recovery_timing)
        else:
            timing = {
                "invocation_seconds": elapsed,
                "run_seconds": elapsed,
                "timing_complete": True,
                "run_seconds_lower_bound": elapsed,
                "attempts": 1,
            }

        steady_state: dict[str, float] = {}
        if len(result.chunks) > 1:
            steady_state = {
                "encoder_seconds": sum(c.encoder_seconds for c in result.chunks[1:]),
                "client_seconds": sum(c.client_seconds for c in result.chunks[1:]),
                "evaluation_seconds": sum(c.evaluation_seconds for c in result.chunks[1:]),
            }
        elif result.chunks:
            steady_state = {
                "encoder_seconds": result.chunks[0].encoder_seconds,
                "client_seconds": result.chunks[0].client_seconds,
                "evaluation_seconds": result.chunks[0].evaluation_seconds,
            }

        timing.update(
            {
                "encoder_seconds": result.encoder_seconds,
                "client_seconds": result.client_seconds,
                "evaluation_seconds": result.evaluation_seconds,
                "cold_initialization": result.phase_seconds.get("cold_initialization", 0.0),
                "preparation": result.phase_seconds.get("preparation", 0.0),
                "steady_state": steady_state,
                "attempt_wall": elapsed,
                "checkpoint_io_seconds": result.phase_seconds.get("checkpoint_io", 0.0),
                "recovery_lost_work_lower_bound": lost_work_lower_bound,
                "host": platform.node(),
                "gpu": get_gpu_info(),
                "cpu": get_cpu_info(),
                "peak_memory_bytes": get_peak_memory_bytes(),
            }
        )
        return replace(result, timing=timing)


def _run(
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
    context_ids: Sequence[str] | None = None,
    checkpoint_dir: Path | str | None = None,
    heartbeat_interval: float | None = 600.0,
    recovery_session: Any = None,
    clock: Callable[[], float] = time.perf_counter,
    sync_fn: Callable[[], None] | None = sync_gpu,
) -> RunResult:
    """Encode, reconstruct, score, and account every chunk."""
    if not chunks:
        raise ValueError(
            "run needs at least one source chunk; a reconstruction of nothing cannot be scored."
        )
    if objects is not None and len(objects) != len(chunks):
        raise ValueError(
            f"objects has {len(objects)} entries for {len(chunks)} chunks. "
            "Pair by track position, one tuple per chunk."
        )
    if context_ids is not None and len(context_ids) != len(chunks):
        raise ValueError(
            f"context_ids has {len(context_ids)} entries for {len(chunks)} chunks. "
            "Pair by track position, one id per chunk."
        )

    cold_init_start = clock()
    validate(config)
    lattice = config.stages
    generation_on = lattice.is_enabled(STAGE_GENERATION)
    ref = bind_generator(config, injected=generator, factory=bind_generator_fn)
    scorer = bind_evaluator(evaluator, config)
    resolver = BackgroundResolver()
    bound = dict(components or {})
    prepared: list[np.ndarray] = []
    for index, raw in enumerate(chunks):
        source = as_clip(raw, path=f"{SOURCE}[{index}]")
        if config.run.max_frames is not None:
            source = source[: config.run.max_frames]
        prepared.append(source)
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
        source_chunks=prepared,
        context_ids=(
            tuple(str(item) for item in context_ids)
            if context_ids is not None
            else tuple((config.background.context_id or "run") for _ in prepared)
        ),
    )
    from src.runner.chunk_checkpoint import (
        completed_indices,
        load_background,
        load_chunk,
        save_background,
        save_chunk,
    )

    results: list[ChunkResult] = []
    all_stage_seconds: list[dict[str, float]] = []
    checkpoint_io_seconds = 0.0
    ckpt = Path(checkpoint_dir) if checkpoint_dir is not None else None
    done = completed_indices(ckpt) if ckpt is not None else ()
    if len(done) > len(prepared):
        raise ValueError("checkpoint has more scenes than this input")
    restore_state = None
    if ckpt is not None:
        resume_root = ckpt
        for index in done:
            t_io = clock()
            chunk, seconds, background_state, bg_index = load_chunk(resume_root, index)
            checkpoint_io_seconds += clock() - t_io
            results.append(chunk)
            all_stage_seconds.append(seconds)
            print(f"resume chunk {index} ({seconds})", flush=True)
            restore_state = background_state
            ctx.background_chunk_index = bg_index

    if not done and ckpt is not None and (ckpt / "prepared").exists():
        t_io = clock()
        restore_state = load_background(ckpt)
        checkpoint_io_seconds += clock() - t_io
    ctx.background_restore_state = restore_state

    if sync_fn is not None:
        sync_fn()
    preparation_started = clock()
    roster = bind_backends(ctx, backends)
    if sync_fn is not None:
        sync_fn()
    preparation_seconds = clock() - preparation_started
    phase_seconds = {"preparation": preparation_seconds}
    print(f"runner preparation {phase_seconds['preparation']:.1f}s", flush=True)

    if ckpt is not None and not (ckpt / "prepared").exists():
        t_io = clock()
        model = ctx.background_model
        save_background(ckpt, model.export_stream_state() if model is not None else None)
        checkpoint_io_seconds += clock() - t_io
        if recovery_session is not None:
            recovery_session.checkpoint()
    conditioning = tuple(ref.requires) if ref is not None else ()
    encoder = Encoder.build(lattice, roster, conditioning=conditioning)
    cold_init_seconds = clock() - cold_init_start
    phase_seconds["cold_initialization"] = cold_init_seconds

    for index, source in enumerate(prepared):
        if index in done:
            continue
        chunk_objects = objects[index] if objects is not None else ()
        stage_seconds: dict[str, float] = {}

        def _on_stage(name: str, elapsed: float, *, _index: int = index) -> None:
            stage_seconds[name] = elapsed
            print(f"chunk {_index} stage {name} {elapsed:.1f}s", flush=True)

        bag = encoder.encode(
            {SOURCE: source, OBJECTS: chunk_objects},
            on_stage=_on_stage,
            heartbeat_interval=heartbeat_interval,
            clock=clock,
            sync_fn=sync_fn,
        )

        chunk_encoder_stages = [v for k, v in stage_seconds.items() if k != "metrics"]
        chunk_encoder_seconds = sum(chunk_encoder_stages)
        metrics_stage_seconds = stage_seconds.get("metrics", 0.0)

        chunk, chunk_client_seconds, chunk_eval_extra = _finish_chunk(
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
            clock=clock,
            sync_fn=sync_fn,
            chunk_encoder_seconds=chunk_encoder_seconds,
            checkpoint=config.generator.checkpoint,
        )
        chunk_eval_seconds = metrics_stage_seconds + chunk_eval_extra
        chunk = replace(
            chunk,
            encoder_seconds=chunk_encoder_seconds,
            client_seconds=chunk_client_seconds,
            evaluation_seconds=chunk_eval_seconds,
        )
        stage_seconds["finish_chunk"] = chunk_client_seconds + chunk_eval_extra
        print(f"chunk {index} finish/scoring {stage_seconds['finish_chunk']:.1f}s", flush=True)
        results.append(chunk)
        all_stage_seconds.append(stage_seconds)
        if ckpt is not None:
            t_io = clock()
            model = ctx.background_model
            state = model.export_stream_state() if model is not None else None
            save_chunk(
                ckpt,
                index,
                chunk,
                stage_seconds=stage_seconds,
                background_state=state,
                background_chunk_index=ctx.background_chunk_index,
            )
            checkpoint_io_seconds += clock() - t_io
            print(f"checkpointed chunk {index}", flush=True)
            if recovery_session is not None:
                recovery_session.checkpoint()

    if sync_fn is not None:
        sync_fn()
    import gc

    gc.collect()
    assembly_started = clock()
    result = _assemble(
        results,
        lattice=lattice,
        scorer=scorer,
        stage_seconds=tuple(all_stage_seconds),
        preparation_seconds=phase_seconds.get("preparation", 0.0),
    )
    if sync_fn is not None:
        sync_fn()
    assembly_eval_seconds = clock() - assembly_started
    phase_seconds["assembly_scoring"] = assembly_eval_seconds
    phase_seconds["checkpoint_io"] = checkpoint_io_seconds
    return replace(
        result,
        phase_seconds=phase_seconds,
        evaluation_seconds=result.evaluation_seconds + assembly_eval_seconds,
    )


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
    clock: Callable[[], float] = time.perf_counter,
    sync_fn: Callable[[], None] | None = None,
    chunk_encoder_seconds: float = 0.0,
    checkpoint: str | Path | None = None,
) -> tuple[ChunkResult, float, float]:
    delivered_quality = bag.get(ART_QUALITY)
    if not isinstance(delivered_quality, QualityReport):
        raise ConfigValueError(
            "runner.metrics",
            "ART_QUALITY is missing or is not a QualityReport. Metrics is a "
            "required stage; a run that skipped it is a failed run.",
        )

    view = _as_background(bag.get(ART_BACKGROUND_MODEL) or bag.get(STAGE_BACKGROUND))
    client_objects = _subjects_for_reconstruct(bag) or objects
    residual = bag.get(ART_RESIDUAL_STREAM)
    residual_payload = residual.payload if isinstance(residual, ResidualResult) else None

    # Build the transport envelope on the encoder side. The measured client
    # interval starts only once those bytes are ready to receive.
    wire_request: bytes | None = None
    placements: tuple[Any, ...] = ()
    transmitted_residual: Any = None
    if not lattice.is_source_passthrough:
        from src.runner.client import (
            ClientPlacement,
            reconstruct_serialized_client,
            serialize_client_request,
        )

        appearance_by_id: dict[str, bytes] = {}
        appearance_artifact = bag.get(ART_APPEARANCE_PAYLOAD)
        if (
            isinstance(appearance_artifact, dict)
            and appearance_artifact.get("representation") == "compressed-image"
        ):
            appearance_items = appearance_artifact.get("items", ())
            appearance_by_id = {
                str(appearance_item["object_id"]): bytes(appearance_item["payload"])
                for appearance_item in appearance_items
                if isinstance(appearance_item, dict)
                and isinstance(
                    appearance_item.get("payload"),
                    (bytes, bytearray, memoryview),
                )
            }

        if appearance_by_id:
            import cv2

            for encoded_crop in appearance_by_id.values():
                decoded_crop = cv2.imdecode(
                    np.frombuffer(encoded_crop, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if decoded_crop is None:
                    raise ValueError("JPEG appearance payload did not decode")
            # Do not write decoded appearance into supplied_crop. That field is
            # an encoder residual shortcut; generation intent on the wire is
            # ClientPlacement.is_generated from STAGE_GENERATION, not "crop is
            # empty". Appearance bytes still travel as references and condition
            # the generator.

        placements_list = []
        seen_keys: set[tuple[str, int]] = set()
        for item in client_objects:
            key = (str(item.object_id), int(item.frame_index))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            is_gen = bool(generation_on and ref is not None)
            crop_to_use = None
            if not is_gen:
                crop_to_use = item.supplied_crop if item.supplied_crop is not None else item.appearance

            pose_arr = None
            motion_arr = None
            if item.conditioning is not None:
                if item.conditioning.pose is not None:
                    pose_arr = np.asarray(item.conditioning.pose, dtype=np.uint8)
                if item.conditioning.motion_field is not None:
                    motion_arr = np.asarray(item.conditioning.motion_field, dtype=np.float32)

            placements_list.append(
                ClientPlacement(
                    crop=crop_to_use,
                    bbox=item.bbox,
                    encoded_crop=appearance_by_id.get(item.object_id),
                    frame_index=item.frame_index,
                    mask=item.mask,
                    object_id=item.object_id,
                    is_generated=is_gen,
                    pose=pose_arr,
                    motion_field=motion_arr,
                )
            )
        placements = tuple(placements_list)

        transmitted_residual = None
        if lattice.is_enabled(STAGE_RESIDUAL):
            bitstream = bag.get(ART_BITSTREAM)
            if isinstance(bitstream, Mapping):
                transmitted_residual = bitstream.get("transmitted_residual")
                if transmitted_residual is None and "frames" in bitstream:
                    from src.pipeline.residual.codec import TransmittedResidual

                    codec_frames = np.asarray(bitstream["frames"], dtype=np.uint8)
                    base = (
                        residual.base
                        if isinstance(residual, ResidualResult) and residual.base is not None
                        else np.zeros_like(codec_frames)
                    )
                    diff = codec_frames.astype(np.int16) - base[: codec_frames.shape[0]].astype(
                        np.int16
                    )
                    transmitted_residual = TransmittedResidual(
                        bitstream=b"",
                        codec_name="raw",
                        shape=(
                            int(diff.shape[0]),
                            int(diff.shape[1]),
                            int(diff.shape[2]),
                            int(diff.shape[3]),
                        ),
                        is_coded=False,
                        raw_frames=diff,
                    )
            elif isinstance(residual, ResidualResult):
                transmitted_residual = residual.transmitted

            if transmitted_residual is None and residual_payload is not None:
                transmitted_residual = residual_payload

        gen_meta = None
        if generation_on and ref is not None:
            from src.runner.generation_identity import identity_from_ref, params_as_dict

            gen_meta = identity_from_ref(
                ref,
                seed=seed,
                params=params_as_dict(params),
                checkpoint=checkpoint,
            )

        wire_request = serialize_client_request(
            background=view,
            frame_count=int(source.shape[0]),
            height=int(source.shape[1]),
            width=int(source.shape[2]),
            placements=placements,
            residual_payload=transmitted_residual,
            references=appearance_by_id if appearance_by_id else None,
            generator_meta=gen_meta,
        )
        bag["wire_request"] = wire_request
        if transmitted_residual is not None:
            bag["transmitted_residual"] = transmitted_residual

    # --- Client Phase: bytes received through delivered frames ---
    if sync_fn is not None:
        sync_fn()
    client_start = clock()
    if lattice.is_source_passthrough:
        client_delivered_frames = source.copy()
        client_base_frames = source.copy()
    else:
        if wire_request is None:
            raise RuntimeError("serialized client request was not prepared")
        client_delivered, client_base = reconstruct_serialized_client(
            wire_request,
            resolver=resolver,
            return_base=True,
            generator=ref,
            seed=seed,
        )
        client_delivered_frames = np.asarray(client_delivered, dtype=np.uint8)
        client_base_frames = np.asarray(client_base, dtype=np.uint8)
    if sync_fn is not None:
        sync_fn()
    client_seconds = clock() - client_start

    # --- Evaluation Phase: metric evaluation of independently decoded frames ---
    if sync_fn is not None:
        sync_fn()
    eval_start = clock()

    delivered_art = bag.get(ART_DELIVERED)
    if isinstance(delivered_art, Mapping) and delivered_art.get("fallback_reason"):
        raise ConfigValueError(
            "codec.fallback",
            f"Source fallback cannot earn a valid quality score: {delivered_art['fallback_reason']}",
        )

    use_heuristic = not lattice.is_enabled(STAGE_SEGMENTATION)
    from src.pipeline.reconstruction.compositor import heuristic_mask
    from src.pipeline.reconstruction.quality import union_object_mask

    object_masks = []
    for item in client_objects:
        if use_heuristic or item.mask is None:
            object_masks.append(
                heuristic_mask(item.bbox, int(source.shape[1]), int(source.shape[2]))
            )
        else:
            object_masks.append(np.asarray(item.mask, dtype=bool))
    combined_mask = (
        union_object_mask(
            object_masks,
            frames=int(source.shape[0]),
            height=int(source.shape[1]),
            width=int(source.shape[2]),
        )
        if object_masks
        else None
    )

    reconstruction_quality = scorer.evaluate(source, client_base_frames, object_mask=combined_mask)
    if reconstruction_quality is None:
        raise ConfigValueError(
            "runner.reconstruction",
            "reconstruct() returned no QualityReport. Every path must score.",
        )

    from src.pipeline.reconstruction.device import DeviceDecision

    client = ReconstructionResult(
        frames=as_clip(client_base_frames, path="client_base"),
        quality=reconstruction_quality,
        path="independent_client",
        device=DeviceDecision("cpu"),
        object_mask=combined_mask,
    )

    if isinstance(residual, ResidualResult):
        from src.pipeline.residual.signal import apply_residual

        frames = apply_residual(client.frames, residual.payload)
        encoder_frames = residual.reconstructed
    else:
        frames = client.frames
        encoder_frames = _encoder_frames(bag, client.frames)

    delivered_val = bag.get(ART_DELIVERED)
    if isinstance(delivered_val, Mapping):
        delivered_dict = dict(delivered_val)
        delivered_dict["frames"] = client_delivered_frames
        bag[ART_DELIVERED] = delivered_dict
    else:
        bag[ART_DELIVERED] = {"frames": client_delivered_frames}

    delivered_quality = scorer.evaluate(source, client_delivered_frames, object_mask=combined_mask)
    if delivered_quality is None:
        raise ConfigValueError(
            "runner.metrics",
            "scorer.evaluate() returned no QualityReport for delivered frames. Every path must score.",
        )

    symmetry = measure_symmetry(encoder_frames, frames)
    if sync_fn is not None:
        sync_fn()
    eval_seconds = clock() - eval_start

    sizes = ledger_from_bag(bag, source)
    if transmitted_residual is not None and getattr(transmitted_residual, "is_coded", False):
        wire_bytes = len(transmitted_residual.bitstream)
        if wire_bytes != sizes.residual:
            raise ValueError(
                f"Residual wire byte mismatch: wire has {wire_bytes} bytes but ledger has {sizes.residual} bytes"
            )
    if wire_request is not None and not sizes.raw_parts:
        if len(wire_request) != sizes.transport_total:
            raise ValueError(
                f"Wire request byte mismatch: wire has {len(wire_request)} bytes but ledger has {sizes.transport_total} bytes"
            )

    chunk = ChunkResult(
        frames=frames,
        encoder_frames=encoder_frames,
        reconstruction=client,
        quality=client.quality,
        delivered_quality=delivered_quality,
        sizes=sizes,
        symmetry=symmetry,
        bag=bag,
        encoder_seconds=chunk_encoder_seconds,
        client_seconds=client_seconds,
        evaluation_seconds=eval_seconds,
    )
    return chunk, client_seconds, eval_seconds


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
    stage_seconds: tuple[dict[str, float], ...] = (),
    preparation_seconds: float = 0.0,
) -> RunResult:
    frames = np.concatenate([item.frames for item in results], axis=0)
    if len(results) == 1:
        quality = results[0].quality
        delivered_quality = results[0].delivered_quality
    else:
        import gc

        gc.collect()
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
        del recon
        gc.collect()
        delivered = np.concatenate(
            [_delivered_frames(item.bag[ART_DELIVERED]) for item in results],
            axis=0,
        )
        delivered_quality = scorer.evaluate(sources, delivered, object_mask=object_mask)
        del delivered, sources
        gc.collect()
    sizes = results[0].sizes
    for extra in results[1:]:
        sizes = sizes + extra.sizes
    if not sizes.parts_fit():
        raise ConfigValueError(
            "runner.sizes",
            f"payload parts sum to {sizes.parts_sum} bytes, more than "
            f"transport_total {sizes.transport_total}. One ledger, and it must add up.",
        )
    total_encoder_seconds = preparation_seconds + sum(item.encoder_seconds for item in results)
    total_client_seconds = sum(item.client_seconds for item in results)
    total_eval_seconds = sum(item.evaluation_seconds for item in results)
    return RunResult(
        frames=frames,
        quality=quality,
        delivered_quality=delivered_quality,
        sizes=sizes,
        symmetry=_combined_symmetry(results),
        chunks=tuple(results),
        lattice=lattice,
        stage_seconds=stage_seconds,
        encoder_seconds=total_encoder_seconds,
        client_seconds=total_client_seconds,
        evaluation_seconds=total_eval_seconds,
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
