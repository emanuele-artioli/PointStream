"""Temporal policy driven by ``TemporalConfig``.

The three sparsity flags compose. The planner emits a ``PlannedSchedule`` that
must be serialized into the payload: encoder and reconstruction both honour
those bytes. Re-running this module on the decoder is the drift the contract
exists to prevent.

Pipeline sparsity's threshold is applied here. ``TemporalPolicy.plan`` in
contracts uses pipeline stride only — that is the gap this component fills,
and why perception indices travel next to the decisions rather than being
inferred from them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from src.components.transport.payload import PlannedSchedule
from src.contracts.config import TemporalConfig
from src.contracts.errors import ConfigValueError
from src.contracts.objectstream import (
    FrameAction,
    FrameDecision,
    Sparsity,
    TemporalPolicy,
    TemporalSchedule,
)


def policy_from_config(config: TemporalConfig) -> TemporalPolicy:
    """Map the YAML section onto the contract's three-level policy.

    A flag off is the dense ``Sparsity()``. A flag on uses ``keyframe_interval``
    as stride and ``delta_threshold`` as the motion bar. Generation sparsity
    counts keyframes, so it gets a stride and no threshold — matching
    ``TemporalPolicy._generated_flags``.
    """
    interval = config.keyframe_interval
    threshold = config.delta_threshold

    def metadata_or_pipeline(enabled: bool) -> Sparsity:
        if not enabled:
            return Sparsity()
        return Sparsity(stride=interval, threshold=threshold)

    generation = Sparsity() if not config.generation_sparsity else Sparsity(stride=interval)
    return TemporalPolicy(
        metadata=metadata_or_pipeline(config.metadata_sparsity),
        generation=generation,
        pipeline=metadata_or_pipeline(config.pipeline_sparsity),
        preroll_frames=config.preroll_frames,
    )


def adapted_threshold(config: TemporalConfig, motion: Sequence[float]) -> float:
    """The motion bar for this clip, in the caller's units.

    A hard-coded ``delta_threshold`` treats a slow rally and a fast exchange
    as the same clip. Scaling by the measured mean puts the bar in this
    scene's units; the two clips then get different bars, and accumulation
    against those bars yields different keyframe density.
    """
    if not motion:
        return float(config.delta_threshold)
    mean = sum(motion) / len(motion)
    if mean <= 0.0:
        return float(config.delta_threshold)
    return float(config.delta_threshold) * mean


class ConfigurableTemporalPolicy:
    """Named backend: three composing flags, driven by ``TemporalConfig``.

    Constructed with either a ``TemporalConfig`` or the individual fields
    ``Registry.build`` passes from spec defaults.
    """

    def __init__(
        self,
        temporal: TemporalConfig | None = None,
        *,
        metadata_sparsity: bool | None = None,
        generation_sparsity: bool | None = None,
        pipeline_sparsity: bool | None = None,
        delta_threshold: float | None = None,
        keyframe_interval: int | None = None,
        preroll_frames: int | None = None,
    ) -> None:
        base = temporal if temporal is not None else TemporalConfig()
        self.config = TemporalConfig(
            metadata_sparsity=(
                base.metadata_sparsity if metadata_sparsity is None else metadata_sparsity
            ),
            generation_sparsity=(
                base.generation_sparsity if generation_sparsity is None else generation_sparsity
            ),
            pipeline_sparsity=(
                base.pipeline_sparsity if pipeline_sparsity is None else pipeline_sparsity
            ),
            delta_threshold=base.delta_threshold if delta_threshold is None else delta_threshold,
            keyframe_interval=(
                base.keyframe_interval if keyframe_interval is None else keyframe_interval
            ),
            preroll_frames=base.preroll_frames if preroll_frames is None else preroll_frames,
        )
        self.policy = policy_from_config(self.config)

    def plan(
        self,
        *,
        frame_count: int,
        object_ids: Sequence[str] = ("object",),
        motion: Sequence[float] | None = None,
        discontinuities: Iterable[int] = (),
        path: str = "temporal-policy",
    ) -> PlannedSchedule:
        """Per-frame decisions and the perception mask, for one span."""
        if frame_count < 0:
            raise ValueError(f"plan needs a non-negative frame_count, got {frame_count}.")
        if not object_ids:
            raise ConfigValueError(path, "plan needs at least one object_id.")
        cuts = frozenset(int(cut) for cut in discontinuities)
        if self.policy.needs_measured_motion and motion is None:
            raise ConfigValueError(
                path,
                "a motion-adaptive sparsity threshold was configured, but no measured "
                "per-frame motion was supplied. The threshold is what makes keyframe "
                "density follow the content instead of a constant, so planning without "
                "it would quietly fall back to the fixed-stride behaviour it replaces.",
            )
        if motion is not None and len(motion) != frame_count:
            raise ConfigValueError(
                path,
                f"motion has {len(motion)} entries for {frame_count} frames.",
            )

        scene_motion = sum(motion) / len(motion) if motion else 0.0
        perceived = self._perception_flags(frame_count, motion, cuts, scene_motion)
        transmitted = self._transmitted_flags(
            frame_count, motion, cuts, scene_motion, perceived
        )
        generated = self._generated_flags(frame_count, cuts, transmitted)

        decisions: list[FrameDecision] = []
        perception: dict[str, tuple[int, ...]] = {}
        perception_indices = tuple(index for index, flag in enumerate(perceived) if flag)

        for object_id in object_ids:
            perception[str(object_id)] = perception_indices
            for index in range(frame_count):
                decisions.append(
                    self._decision(
                        index=index,
                        object_id=str(object_id),
                        transmitted=transmitted,
                        generated=generated,
                        cuts=cuts,
                    )
                )

        schedule = TemporalSchedule(decisions=tuple(decisions), discontinuities=cuts)
        adapted = None if motion is None else adapted_threshold(self.config, motion)
        planned = PlannedSchedule(
            schedule=schedule,
            perception=perception,
            scene_motion=scene_motion if motion is not None else None,
            adapted_threshold=adapted,
        )
        schedule.validate(path=f"{path}.schedule")
        return planned

    def adapted_threshold(self, motion: Sequence[float]) -> float:
        """See ``adapted_threshold``."""
        return adapted_threshold(self.config, motion)

    def _perception_flags(
        self,
        frame_count: int,
        motion: Sequence[float] | None,
        cuts: frozenset[int],
        scene_motion: float,
    ) -> list[bool]:
        """Which frames run detection, pose and segmentation.

        This is the encode-time saving. Metadata and generation sparsity never
        skip these stages; only pipeline sparsity does.
        """
        level = self.policy.pipeline
        if level.is_dense:
            return [True] * frame_count
        effective = _effective_threshold(level, scene_motion)
        flags = [False] * frame_count
        accumulated = 0.0
        span_start = 0
        for index in range(frame_count):
            if index == 0 or index in cuts:
                span_start = index
                flags[index] = True
                accumulated = 0.0
                continue
            offset = index - span_start
            if motion is not None:
                accumulated += motion[index]
            if _stride_or_threshold(level, offset, accumulated, effective):
                flags[index] = True
                accumulated = 0.0
        return flags

    def _transmitted_flags(
        self,
        frame_count: int,
        motion: Sequence[float] | None,
        cuts: frozenset[int],
        scene_motion: float,
        perceived: Sequence[bool],
    ) -> list[bool]:
        """Which frames put motion on the wire. Perception bounds this."""
        level = self.policy.metadata
        effective = _effective_threshold(level, scene_motion)
        flags = [False] * frame_count
        accumulated = 0.0
        span_start = 0
        for index in range(frame_count):
            if index == 0 or index in cuts:
                span_start = index
                flags[index] = True
                accumulated = 0.0
                continue
            offset = index - span_start
            if motion is not None:
                accumulated += motion[index]
            if not perceived[index]:
                continue
            if level.is_dense:
                send = True
            else:
                send = _stride_or_threshold(level, offset, accumulated, effective)
            flags[index] = send
            if send:
                accumulated = 0.0
        return flags

    def _generated_flags(
        self,
        frame_count: int,
        cuts: frozenset[int],
        transmitted: Sequence[bool],
    ) -> list[bool]:
        flags = [False] * frame_count
        span_start = 0
        keyframe_ordinal = 0
        stride = self.policy.generation.stride
        for index in range(frame_count):
            if index == 0 or index in cuts:
                span_start = index
                keyframe_ordinal = 0
            if not transmitted[index]:
                continue
            if index - span_start < self.policy.preroll_frames:
                continue
            ordinal = keyframe_ordinal
            keyframe_ordinal += 1
            flags[index] = stride == 1 or ordinal % stride == 0
        return flags

    @staticmethod
    def _decision(
        *,
        index: int,
        object_id: str,
        transmitted: Sequence[bool],
        generated: Sequence[bool],
        cuts: frozenset[int],
    ) -> FrameDecision:
        if transmitted[index]:
            action = FrameAction.FULL if generated[index] else FrameAction.TRANSMIT_ONLY
            return FrameDecision(index, object_id, action)
        anchor = max(position for position in range(index) if transmitted[position])
        target = next(
            (
                position
                for position in range(index + 1, len(transmitted))
                if transmitted[position] and not _cut_between(cuts, index, position)
            ),
            None,
        )
        if target is None:
            return FrameDecision(index, object_id, FrameAction.HOLD, anchor=anchor)
        return FrameDecision(
            index, object_id, FrameAction.INTERPOLATE, anchor=anchor, target=target
        )


def _effective_threshold(level: Sparsity, scene_motion: float) -> float | None:
    if level.threshold is None:
        return None
    if level.threshold_relative_to_scene_motion:
        return level.threshold * scene_motion
    return level.threshold


def _stride_or_threshold(
    level: Sparsity,
    offset: int,
    accumulated: float,
    effective: float | None,
) -> bool:
    """Either firing acts. Stride 1 without a threshold is 'every frame'."""
    stride_hit = level.stride > 1 and offset % level.stride == 0
    threshold_hit = effective is not None and accumulated >= effective
    if level.stride == 1 and effective is None:
        return True
    return stride_hit or threshold_hit


def _cut_between(cuts: frozenset[int], start: int, end: int) -> bool:
    return any(start < cut <= end for cut in cuts)


@dataclass(frozen=True)
class SceneCutMarks:
    """Discontinuities the policy accepts until B2 supplies scene spans.

    A mark is the first frame of a new span. Interpolation must not cross it.
    """

    frames: tuple[int, ...] = ()

    def as_discontinuities(self) -> frozenset[int]:
        return frozenset(int(frame) for frame in self.frames)
