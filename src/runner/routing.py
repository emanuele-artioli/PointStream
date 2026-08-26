"""Bind named backends to the types C1 and C2 inject.

C2 wants ``StageCallable``. C1 wants ``GeneratorRef`` and ``QualityEvaluator``.
This module is the only place that looks up a registry. Generation off never
constructs a generator. Perception backends are built lazily the first time
the stage actually needs them, so a test that injects objects never loads YOLO.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from src.contracts.config import LatticeConfig, PointstreamConfig
from src.contracts.conditioning import FrameGenerator, GenerationParams
from src.contracts.errors import ConfigValueError
from src.contracts.lattice import STAGE_GENERATION, StageLattice
from src.pipeline.dag.graph import StageCallable
from src.pipeline.reconstruction.dispatch import GeneratorRef, from_spec
from src.pipeline.reconstruction.quality import NumpyPsnrEvaluator, QualityEvaluator
from src.runner.stages import StageContext, default_backends


def lattice_config_from(corner: StageLattice) -> LatticeConfig:
    """The ``LatticeConfig`` that names ``corner``."""
    flags = {
        attribute: stage_name in corner.enabled
        for attribute, stage_name in LatticeConfig._STAGE_FIELDS.items()
    }
    return LatticeConfig(**flags)


def generation_params(config: PointstreamConfig) -> GenerationParams:
    gen = config.generator
    return GenerationParams(
        steps=gen.steps,
        strength=gen.strength,
        guidance_scale=gen.guidance,
        width=gen.width,
        height=gen.height,
    )


def bind_generator(
    config: PointstreamConfig,
    *,
    injected: GeneratorRef | None = None,
    factory: Callable[[], GeneratorRef] | None = None,
) -> GeneratorRef | None:
    """The run's one ``GeneratorRef``, or None when generation is off.

    ``factory`` exists so a test can prove all-off never constructs. The
    registry is consulted only when generation is on and nothing was injected.
    """
    if not config.stages.is_enabled(STAGE_GENERATION):
        return None
    if injected is not None:
        return injected
    if factory is not None:
        return factory()
    name = config.generator.resolved_name
    if name in ("", "none"):
        raise ConfigValueError(
            "generator.backend",
            "the generation stage is enabled but no generator is named, and none "
            "was injected. Name a backend or pass a GeneratorRef.",
        )
    from src.components.generation import REGISTRY

    spec = REGISTRY.spec(name)
    backend = REGISTRY.build(name)
    if not isinstance(backend, FrameGenerator):
        raise ConfigValueError(
            "generator.backend",
            f"backend {name!r} does not satisfy FrameGenerator.",
        )
    return from_spec(spec, backend)


def bind_evaluator(
    injected: QualityEvaluator | None = None,
    config: PointstreamConfig | None = None,
) -> QualityEvaluator:
    """The scorer this run uses.

    An injected evaluator wins. Otherwise the config's `evaluation.metrics`
    decides: PSNR alone stays on C1's numpy floor, anything richer binds the
    components-layer metric registry. Without a config the floor is used, which
    is what every existing caller that passes only chunks gets.

    This is the only place `evaluation.metrics` becomes something that runs. It
    was previously a field nothing read, so a config naming SSIM and VMAF
    produced PSNR and said nothing about it.
    """
    if injected is not None:
        return injected
    if config is None:
        return NumpyPsnrEvaluator()
    from src.runner.evaluation import evaluator_for

    return evaluator_for(config)


def bind_backends(
    ctx: StageContext,
    injected: Mapping[str, StageCallable] | None = None,
) -> dict[str, StageCallable]:
    """Defaults for every stage, overwritten by ``injected``.

    Extra keys for disabled stages stay in the mapping; C2 will not invoke them.
    """
    roster = default_backends(ctx)
    if injected:
        roster.update(injected)
    return roster


def _named(value: str | None) -> str | None:
    if value in (None, "", "none"):
        return None
    return value


def _build(ctx: StageContext, axis: str, name: str, **kwargs: Any) -> Any:
    from src.runner.perception import build_backend

    return build_backend(ctx, axis, name, **kwargs)


def ensure_detector(ctx: StageContext) -> Any:
    """The detector named by ``config.detector``, built once per run."""
    if ctx.detector is not None:
        return ctx.detector
    config = ctx.config
    if not config.lattice.detection:
        return None
    name = _named(config.detector.backend)
    if name is None:
        return None
    kwargs: dict[str, Any] = {}
    if config.detector.model:
        kwargs["model_name"] = config.detector.model
    if config.detector.prompt:
        kwargs["prompt"] = config.detector.prompt
    try:
        ctx.detector = _build(ctx, "detector", name, **kwargs)
    except TypeError:
        kwargs.pop("prompt", None)
        ctx.detector = _build(ctx, "detector", name, **kwargs)
    return ctx.detector


def ensure_pose(ctx: StageContext) -> Any:
    """The pose estimator named by ``config.pose``."""
    if ctx.pose_estimator is not None:
        return ctx.pose_estimator
    config = ctx.config
    if not config.lattice.pose:
        return None
    name = _named(config.pose.backend)
    if name is None:
        return None
    kwargs: dict[str, Any] = {}
    if config.pose.model:
        kwargs["model_name"] = config.pose.model
    ctx.pose_estimator = _build(ctx, "pose", name, **kwargs)
    return ctx.pose_estimator


def ensure_segmenter(ctx: StageContext) -> Any:
    """The segmenter named by ``config.segmenter``."""
    if ctx.segmenter is not None:
        return ctx.segmenter
    config = ctx.config
    if not config.lattice.segmentation:
        return None
    name = _named(config.segmenter.backend)
    if name is None:
        return None
    kwargs: dict[str, Any] = {}
    if config.segmenter.model:
        kwargs["model_name"] = config.segmenter.model
    ctx.segmenter = _build(ctx, "segmenter", name, **kwargs)
    return ctx.segmenter


def ensure_appearance(ctx: StageContext) -> Any:
    """The appearance encoder named by ``config.appearance.representation``."""
    if ctx.appearance_encoder is not None:
        return ctx.appearance_encoder
    config = ctx.config
    if not config.lattice.appearance:
        return None
    name = _named(config.appearance.representation)
    if name is None:
        return None
    kwargs: dict[str, Any] = {}
    if name == "compressed-image":
        kwargs["quality"] = config.appearance.jpeg_quality
        kwargs["downscale"] = config.appearance.downscale
    ctx.appearance_encoder = _build(ctx, "appearance", name, **kwargs)
    return ctx.appearance_encoder


def ensure_motion(ctx: StageContext) -> Any:
    """The motion encoder named by ``config.motion.representation``."""
    if ctx.motion_encoder is not None:
        return ctx.motion_encoder
    config = ctx.config
    if not config.lattice.motion:
        return None
    name = _named(config.motion.representation)
    if name is None:
        return None
    ctx.motion_encoder = _build(ctx, "motion", name)
    return ctx.motion_encoder


def ensure_temporal(ctx: StageContext) -> Any:
    """The temporal policy driven by ``config.temporal``.

    There is no ``temporal.backend`` field. The registry's ``config`` backend
    takes a ``TemporalConfig``; swapping flags or ``keyframe_interval`` is how
    this axis is named in the document.
    """
    if ctx.temporal_policy is not None:
        return ctx.temporal_policy
    config = ctx.config
    if not config.lattice.temporal_policy:
        return None
    ctx.temporal_policy = _build(ctx, "temporal", "config", temporal=config.temporal)
    return ctx.temporal_policy
