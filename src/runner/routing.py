"""Bind named backends to the types C1 and C2 inject.

C2 wants ``StageCallable``. C1 wants ``GeneratorRef`` and ``QualityEvaluator``.
This module is the only place that looks up a registry. Generation off never
constructs a generator.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

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


def bind_evaluator(injected: QualityEvaluator | None = None) -> QualityEvaluator:
    """C1's numpy PSNR floor, unless a richer evaluator was injected."""
    if injected is not None:
        return injected
    return NumpyPsnrEvaluator()


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
