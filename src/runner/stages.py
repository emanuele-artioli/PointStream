"""Default ``StageCallable`` bindings the runner injects into C2.

Perception and representation stages bind the named config backend the first
time they have work to do. Injected objects still win over a detector, so a
test that supplies subjects never loads YOLO. Generation calls C1
``dispatch``. Residual composes an encoder-side reconstruction from bag
artifacts (using generated crops as ``supplied_crop`` so it does not dispatch
again) and runs ``compute_residual``. Codec / transport / metrics are
identity-roundtrip enough to score delivered pixels — they do not shell out.

``make_codec`` is owned by BP24; this stream does not edit it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
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
    ART_KEYPOINTS,
    ART_MASKS,
    ART_RESIDUAL_STREAM,
    ART_SCHEDULE,
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
from src.contracts.config import PointstreamConfig, ResidualConfig
from src.pipeline.dag.graph import StageCallable
from src.pipeline.encoder.encoder import SOURCE
from src.pipeline.reconstruction.background import BackgroundModelView, BackgroundResolver
from src.pipeline.reconstruction.clips import as_clip
from src.pipeline.reconstruction.compositor import heuristic_mask
from src.pipeline.reconstruction.dispatch import GeneratorRef, dispatch
from src.pipeline.reconstruction.quality import QualityEvaluator, QualityReport
from src.pipeline.reconstruction.reconstruct import (
    ObjectRequest,
    ReconstructionRequest,
    reconstruct,
)
from src.pipeline.residual.signal import (
    ResidualResult,
    ResidualVariant,
    compute_residual,
    decode_lossy,
)
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
    config: PointstreamConfig
    builders: Mapping[str, Callable[..., Any]] | None = None
    detector: Any = None
    pose_estimator: Any = None
    segmenter: Any = None
    appearance_encoder: Any = None
    motion_encoder: Any = None
    temporal_policy: Any = None


def _subjects(bag: Mapping[str, Any]) -> tuple[ObjectRequest, ...]:
    from src.runner.perception import subjects_from_bag

    return subjects_from_bag(bag)


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


def _injected_objects(bag: Mapping[str, Any]) -> tuple[ObjectRequest, ...]:
    raw = bag.get(OBJECTS)
    if not raw:
        return ()
    if isinstance(raw, ObjectRequest):
        return (raw,)
    return tuple(item for item in raw if isinstance(item, ObjectRequest))


def _as_detection(item: ObjectRequest) -> Any:
    from src.components.detection.geometry import Box
    from src.components.detection.types import Detection

    x1, y1, x2, y2 = item.bbox
    return Detection(
        class_name="person",
        bbox=Box(float(x1), float(y1), float(x2), float(y2)),
        track_id=item.object_id,
    )


def _object_from_detection(
    detection: Any, source: np.ndarray, frame_index: int, order: int
) -> ObjectRequest | None:
    frames, height, width = int(source.shape[0]), int(source.shape[1]), int(source.shape[2])
    box = detection.bbox
    x1 = max(0, int(np.floor(box.x1)))
    y1 = max(0, int(np.floor(box.y1)))
    x2 = min(width, int(np.ceil(box.x2)))
    y2 = min(height, int(np.ceil(box.y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = np.ascontiguousarray(source[frame_index, y1:y2, x1:x2])
    if crop.size == 0:
        return None
    object_id = detection.track_id or f"{detection.class_name}-{frame_index}-{order}"
    mask = np.zeros((frames, height, width), dtype=bool)
    mask[frame_index] = heuristic_mask((x1, y1, x2, y2), height, width)
    return ObjectRequest(
        object_id=str(object_id),
        appearance=crop,
        bbox=(x1, y1, x2, y2),
        mask=mask,
        frame_index=frame_index,
    )


def _frame_mask(
    crop_mask: np.ndarray, bbox: tuple[int, int, int, int], *, height: int, width: int
) -> np.ndarray:
    """Paste a crop-local mask into a frame-sized boolean array."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(x1, min(width, x2))
    y2 = max(y1, min(height, y2))
    region_h, region_w = y2 - y1, x2 - x1
    full = np.zeros((height, width), dtype=bool)
    if region_h < 1 or region_w < 1:
        return full
    array = np.asarray(crop_mask)
    if array.shape != (region_h, region_w):
        import cv2

        array = cv2.resize(
            array.astype(np.uint8),
            (region_w, region_h),
            interpolation=cv2.INTER_NEAREST,
        )
    full[y1:y2, x1:x2] = array.astype(bool)
    return full


def _perception_on(bag: Mapping[str, Any], object_id: str, frame_index: int) -> bool:
    """Honour pipeline sparsity when a schedule has already been planned."""
    schedule = bag.get(ART_SCHEDULE)
    perception = getattr(schedule, "perception", None)
    if not isinstance(perception, Mapping) or not perception:
        return True
    indices = perception.get(object_id)
    if indices is None:
        first = next(iter(perception.values()), None)
        if first is None:
            return True
        return frame_index in first
    return frame_index in indices


def _subjects_for_reconstruct(bag: Mapping[str, Any]) -> tuple[ObjectRequest, ...]:
    subjects = _subjects(bag)
    masks = bag.get(ART_MASKS)
    if not isinstance(masks, Mapping) or not subjects:
        return subjects
    source = as_clip(bag[SOURCE], path=SOURCE)
    height, width = int(source.shape[1]), int(source.shape[2])
    updated: list[ObjectRequest] = []
    for item in subjects:
        mask = masks.get(item.object_id)
        if mask is None:
            updated.append(item)
            continue
        array = np.asarray(mask)
        if array.shape != (height, width):
            array = _frame_mask(array, item.bbox, height=height, width=width)
        updated.append(replace(item, mask=array))
    return tuple(updated)


def make_detection(ctx: StageContext) -> StageCallable:
    """Named detector, or the objects the caller already put on the bag."""

    def detection(bag: Mapping[str, Any]) -> tuple[ObjectRequest, ...]:
        injected = _injected_objects(bag)
        if injected:
            return injected
        from src.runner.routing import ensure_detector

        detector = ensure_detector(ctx)
        if detector is None:
            return ()
        source = as_clip(bag[SOURCE], path=SOURCE)
        found: list[ObjectRequest] = []
        for index in range(int(source.shape[0])):
            for order, item in enumerate(detector.detect(source[index])):
                converted = _object_from_detection(item, source, index, order)
                if converted is not None:
                    found.append(converted)
        return tuple(found)

    return detection


def make_selection(ctx: StageContext) -> StageCallable:
    def selection(bag: Mapping[str, Any]) -> tuple[ObjectRequest, ...]:
        from src.runner.perception import build_backend, filter_selected

        subjects = _subjects(bag)
        if _injected_objects(bag) or not subjects:
            return subjects
        selector = build_backend(ctx, "selection", ctx.config.selection.backend)
        if selector is None:
            return subjects
        source = as_clip(bag[SOURCE], path=SOURCE)
        return filter_selected(subjects, selector, (int(source.shape[1]), int(source.shape[2])))

    return selection


def make_pose(ctx: StageContext) -> StageCallable:
    """Named pose backend. Empty when the estimator skips a class."""

    def pose(bag: Mapping[str, Any]) -> tuple[Any, ...]:
        from src.runner.routing import ensure_pose

        estimator = ensure_pose(ctx)
        subjects = _subjects(bag)
        if estimator is None or not subjects or _injected_objects(bag):
            return ()
        source = as_clip(bag[SOURCE], path=SOURCE)
        poses: list[Any] = []
        for item in subjects:
            if not _perception_on(bag, item.object_id, item.frame_index):
                poses.append(None)
                continue
            frame = source[min(item.frame_index, int(source.shape[0]) - 1)]
            poses.append(estimator.estimate(frame, _as_detection(item)))
        return tuple(poses)

    return pose


def make_segmentation(ctx: StageContext) -> StageCallable:
    """Named segmenter. Returns a map of object_id → frame-sized mask."""

    def segmentation(bag: Mapping[str, Any]) -> dict[str, np.ndarray]:
        from src.runner.routing import ensure_segmenter

        segmenter = ensure_segmenter(ctx)
        subjects = _subjects(bag)
        if segmenter is None or not subjects:
            return {}
        source = as_clip(bag[SOURCE], path=SOURCE)
        masks: dict[str, np.ndarray] = {}
        for item in subjects:
            if not _perception_on(bag, item.object_id, item.frame_index):
                continue
            frame = source[min(item.frame_index, int(source.shape[0]) - 1)]
            mask = segmenter.segment(frame, _as_detection(item))
            if mask is not None:
                masks[item.object_id] = _frame_mask(
                    np.asarray(mask),
                    item.bbox,
                    height=int(source.shape[1]),
                    width=int(source.shape[2]),
                )
        return masks

    return segmentation


def make_appearance(ctx: StageContext) -> StageCallable:
    """Named appearance representation. Byte count is the output that moves."""

    def appearance(bag: Mapping[str, Any]) -> dict[str, Any]:
        from src.runner.routing import ensure_appearance

        backend = ensure_appearance(ctx)
        subjects = _subjects(bag)
        if backend is None or not subjects:
            return {"byte_count": 0, "items": ()}
        total = 0
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in subjects:
            if item.object_id in seen:
                continue
            seen.add(item.object_id)
            crop = np.asarray(item.appearance)
            if crop.size == 0:
                continue
            encoded = backend.encode(crop)
            nbytes = _encoded_bytes(encoded)
            total += nbytes
            items.append({"object_id": item.object_id, "byte_count": nbytes})
        return {"byte_count": total, "items": tuple(items)}

    return appearance


def make_motion(ctx: StageContext) -> StageCallable:
    """Named motion representation. Byte count / tag is the output that moves."""

    def motion(bag: Mapping[str, Any]) -> dict[str, Any]:
        from src.runner.routing import ensure_motion

        backend = ensure_motion(ctx)
        if backend is None:
            return {"byte_count": 0, "representation": None}
        kind = getattr(backend, "kind", None)
        if kind is None:
            kind = ctx.config.motion.representation
        poses = bag.get(ART_KEYPOINTS) or ()
        subjects = _subjects(bag)
        byte_count = _motion_bytes(
            backend,
            kind,
            poses=poses,
            subjects=subjects,
            max_points=ctx.config.motion.max_points,
        )
        return {"byte_count": byte_count, "representation": kind}

    return motion


def make_temporal(ctx: StageContext) -> StageCallable:
    """Plan from ``TemporalConfig``. Keyframe count is the output that moves."""

    def temporal(bag: Mapping[str, Any]) -> Any:
        from src.runner.routing import ensure_temporal

        policy = ensure_temporal(ctx)
        if policy is None:
            return ()
        source = as_clip(bag[SOURCE], path=SOURCE)
        subjects = _subjects(bag)
        object_ids = tuple(dict.fromkeys(item.object_id for item in subjects)) or ("object",)
        motion = _frame_luma_motion(source)
        return policy.plan(
            frame_count=int(source.shape[0]),
            object_ids=object_ids,
            motion=motion,
        )

    return temporal


def _encoded_bytes(encoded: object) -> int:
    if isinstance(encoded, tuple) and len(encoded) == 2:
        payload = encoded[1]
        if isinstance(payload, (bytes, bytearray, memoryview)):
            return len(payload)
        array = np.asarray(payload)
        return int(array.nbytes)
    measured_size = getattr(encoded, "measured_bytes", None)
    if measured_size is not None:
        return int(measured_size)
    per_frame = getattr(encoded, "measured_bytes_per_frame", None)
    if per_frame is not None:
        return int(per_frame)
    return 0


def _motion_bytes(
    backend: Any,
    kind: object,
    *,
    poses: object,
    subjects: Sequence[ObjectRequest],
    max_points: int,
) -> int:
    from src.contracts.capabilities import (
        MOTION_ENCODED_VIDEO,
        MOTION_KEYPOINTS,
        MOTION_SPARSE_TRAJECTORIES,
    )

    pose_list = (
        poses if isinstance(poses, Sequence) and not isinstance(poses, (str, bytes)) else ()
    )
    if kind == MOTION_KEYPOINTS:
        total = 0
        for pose in pose_list:
            if pose is None:
                continue
            schema = getattr(pose, "schema", None)
            values = getattr(pose, "values", None)
            if values is None:
                values = np.asarray(pose)
            encoder = backend
            if schema is not None and getattr(backend, "schema", None) is not None:
                encoder = type(backend)(schema=schema)
            total += _encoded_bytes(encoder.encode(values))
        return total
    if kind == MOTION_SPARSE_TRAJECTORIES:
        points: list[list[float]] = []
        for pose in pose_list:
            if pose is None:
                continue
            values = np.asarray(getattr(pose, "values", pose))
            present = np.asarray(getattr(pose, "present", np.ones(len(values), dtype=bool)))
            if values.ndim == 2 and values.shape[0]:
                for row, flag in zip(values, present, strict=False):
                    if flag:
                        points.append([float(row[0]), float(row[1])])
        if not points:
            for item in subjects:
                x1, y1, x2, y2 = item.bbox
                points.extend(
                    [
                        [float(x1), float(y1)],
                        [float(x2), float(y1)],
                        [float(x2), float(y2)],
                        [float(x1), float(y2)],
                    ]
                )
        if not points:
            return 0
        capped = np.asarray(points[: max(1, max_points)], dtype=np.float32)
        return _encoded_bytes(backend.encode(capped))
    if kind == MOTION_ENCODED_VIDEO:
        if not subjects:
            return 0
        nbytes = int(np.asarray(subjects[0].appearance).nbytes)
        try:
            encoded = backend.encode(measured_bytes_per_frame=nbytes)
        except TypeError:
            encoded = backend.encode(np.asarray(subjects[0].appearance))
        return _encoded_bytes(encoded)
    return 0


def _frame_luma_motion(source: np.ndarray) -> list[float]:
    if source.shape[0] == 0:
        return []
    magnitudes = [0.0]
    prev = source[0].astype(np.float32)
    for index in range(1, int(source.shape[0])):
        current = source[index].astype(np.float32)
        magnitudes.append(float(np.mean(np.abs(current - prev))))
        prev = current
    return magnitudes


def background(bag: Mapping[str, Any]) -> BackgroundModelView:
    """A static plate from the first source frame. Identity warp.

    Kept for callers that bind stages directly without a config. Any run that
    has one should use `make_background`, which transmits the plate through the
    configured sidecar instead of handing over raw pixels.
    """
    source = as_clip(bag[SOURCE], path=SOURCE)
    return BackgroundModelView(
        plate=source[0],
        homographies=(),
        mode="full",
        width=int(source.shape[2]),
        height=int(source.shape[1]),
        scene_id=None,
    )


def make_background(ctx: StageContext) -> StageCallable:
    """Transmit the plate through `background.codec` and hand back what the
    client will actually hold (BP24).

    Before this, the runner's background stage was a stub: it put `source[0]`
    on the bag as raw pixels, so `background.method`, `background.codec` and
    `background.jpeg_quality` all reached nothing, and the plate the
    reconstruction used was never the plate a client would decode. BP23
    measured that raw plate at 24.9 MB — 95% of `tier_fast`'s whole figure.

    Two things change together, which is the point. The view now carries the
    **decoded** plate, so quality belongs to the same operating point as the
    rate; and `ART_BACKGROUND_BYTES` carries the **real payload length**, not a
    re-encode of already-decoded pixels.

    Still a stub: the plate itself is the first source frame rather than a
    stitched panorama, so `background.method` selects a transmission strategy
    over a one-frame plate. That is the next piece, not this one.
    """

    def background_stage(bag: Mapping[str, Any]) -> BackgroundModelView:
        from src.components.background.strategy import bind as bind_background

        source = as_clip(bag[SOURCE], path=SOURCE)
        model = bind_background(ctx.config)
        artifact = model.transmit(np.asarray(source[0], dtype=np.uint8))
        decoded = model.decode_payload(artifact)
        return BackgroundModelView(
            plate=source[0] if decoded is None else decoded,
            homographies=(),
            mode=artifact.mode,
            deferred_to_residual=artifact.deferred_to_residual,
            width=int(source.shape[2]),
            height=int(source.shape[1]),
            scene_id=artifact.scene_id,
            payload_bytes=int(len(artifact.payload)),
        )

    return background_stage


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
    objects = _with_supplied_crops(_subjects_for_reconstruct(bag), bag.get(ART_GENERATED_FRAMES))
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
            coded = _coded_residual(ctx, residual)
            if coded is not None:
                frames, coded_bytes = coded
                return {
                    "frames": frames,
                    "byte_count": coded_bytes,
                    "raw_byte_count": int(residual.payload.byte_count),
                    "residual_is_coded": True,
                }
            return {
                "frames": residual.reconstructed,
                "byte_count": int(residual.payload.byte_count),
                "raw_byte_count": int(residual.payload.byte_count),
                # The array's size, not a bitstream's. Nothing may divide this
                # by the source and call the result a compression ratio.
                "residual_is_coded": False,
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


def _coded_residual(
    ctx: StageContext, residual: ResidualResult
) -> tuple[np.ndarray, int] | None:
    """Send the residual through `residual.codec` and rebuild from what returns.

    Returns ``(frames, coded_bytes)``, or ``None`` when no encoder could run —
    in which case the caller must keep reporting the raw array size and say so.

    Both halves move together on purpose. `residual.reconstructed` was built
    from the **pre-codec** residual, so reporting a coded size beside it would
    put the rate and the quality at different operating points (BP24 finding 4).
    The correction is exact rather than approximate: the delivered clip is
    ``reconstructed - r + r_coded``, so the base reconstruction never has to be
    recovered or recomputed.
    """
    payload = residual.payload
    if payload.frames is None or payload.variant is not ResidualVariant.LOSSY:
        return None
    from src.components.codec.measure import coded_roundtrip

    try:
        coded_bytes, decoded = coded_roundtrip(
            np.asarray(payload.frames, dtype=np.uint8),
            request=ctx.config.residual.encode_request(),
        )
    except (FileNotFoundError, RuntimeError, ValueError):
        # No encoder on this host, or it refused this payload. Fall back to the
        # honest raw number rather than inventing a coded one.
        return None
    before = decode_lossy(np.asarray(payload.frames, dtype=np.uint8))
    after = decode_lossy(decoded[: before.shape[0]])
    frames = np.asarray(residual.reconstructed).astype(np.int16) - before + after
    return np.clip(frames, 0, 255).astype(np.uint8), int(coded_bytes)


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
        STAGE_DETECTION: make_detection(ctx),
        STAGE_SELECTION: make_selection(ctx),
        STAGE_TRACKING: _empty,
        STAGE_APPEARANCE: make_appearance(ctx),
        STAGE_MOTION: make_motion(ctx),
        STAGE_TEMPORAL: make_temporal(ctx),
        STAGE_POSE: make_pose(ctx),
        STAGE_SEGMENTATION: make_segmentation(ctx),
        STAGE_RIGID: _empty,
        STAGE_BACKGROUND: make_background(ctx),
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
    from src.runner.perception import metadata_bytes

    return sizes_bytes(
        source=int(clip.nbytes),
        residual=residual_bytes,
        panorama=_panorama_bytes(bag),
        actor_reference=_measured_actor_bytes(bag),
        metadata=metadata_bytes(bag),
    )


def _plate_of(bag: Mapping[str, Any]) -> np.ndarray | None:
    """The plate this run actually transmits, or None when it transmits none."""
    view = bag.get(ART_BACKGROUND_MODEL) or bag.get(STAGE_BACKGROUND)
    if not isinstance(view, BackgroundModelView) or view.plate is None:
        return None
    if view.deferred_to_residual or view.mode == "none":
        return None
    return np.asarray(view.plate)


def _panorama_bytes(bag: Mapping[str, Any]) -> int:
    """Raw plate size — uncompressed pixels, kept for comparison with BP23.

    This is deliberately *not* the transmitted cost once a sidecar is
    configured; see `_panorama_coded_bytes`. It stays so the BP23 table remains
    readable next to the coded one and the change in meaning is visible.
    """
    plate = _plate_of(bag)
    return 0 if plate is None else int(plate.nbytes)


def _panorama_coded_bytes(ctx: StageContext, bag: Mapping[str, Any]) -> int | None:
    """What the plate really costs through `background.codec` (BP24).

    Returns ``None`` when there is no plate to send, so a caller can tell
    "nothing transmitted" from "transmitted for free". Before BP24 this path
    reported raw pixels and `background.codec` / `background.jpeg_quality`
    reached nothing at all — BP23 measured a 24.9 MB plate that way, 95% of
    `tier_fast`'s entire figure.
    """
    view = bag.get(ART_BACKGROUND_MODEL) or bag.get(STAGE_BACKGROUND)
    if isinstance(view, BackgroundModelView) and view.payload_bytes is not None:
        # The size the transmission actually produced. Preferred over any
        # re-encode: `view.plate` is already *decoded* pixels, so encoding it
        # again measures a second, easier compression of a cleaned-up image.
        return int(view.payload_bytes)
    plate = _plate_of(bag)
    if plate is None:
        return None
    from src.components.background.sidecar import build_sidecar

    sidecar = build_sidecar(
        ctx.config.background.codec,
        jpeg_quality=ctx.config.background.jpeg_quality,
    )
    return int(len(sidecar.encode(plate)))


def _measured_actor_bytes(bag: Mapping[str, Any]) -> int:
    payload = bag.get(ART_APPEARANCE_PAYLOAD)
    if isinstance(payload, Mapping) and "byte_count" in payload:
        return int(payload["byte_count"])
    return 0
