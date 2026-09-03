"""A stage DAG built from the lattice enabled set.

The pipeline never knows which backend was chosen. Each enabled stage is a
callable the runner injected; disabled stages are absent from the graph, which
is what makes a reduced corner cheaper rather than nominally so. The all-off
corner is the same constructor as every other corner — three required stages,
nothing else.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

from src.contracts.config import LatticeConfig, PointstreamConfig
from src.contracts.errors import ConfigValueError
from src.contracts.lattice import STAGES, StageLattice, stage

#: A stage reads the artifact bag so far and returns the artifact it produces.
StageCallable: TypeAlias = Callable[[Mapping[str, Any]], Any]


def as_lattice(source: StageLattice | LatticeConfig | PointstreamConfig) -> StageLattice:
    """The corner a config names, or the lattice itself."""
    if isinstance(source, StageLattice):
        return source
    if isinstance(source, LatticeConfig):
        return source.to_lattice()
    if isinstance(source, PointstreamConfig):
        return source.stages
    raise TypeError(
        f"expected a StageLattice, LatticeConfig or PointstreamConfig, "
        f"got {type(source).__name__}"
    )


def enabled_stages(source: StageLattice | LatticeConfig | PointstreamConfig) -> frozenset[str]:
    """Stage names this corner will actually run."""
    return as_lattice(source).enabled


@dataclass(frozen=True)
class StageNode:
    """One enabled stage, bound to an injected callable."""

    name: str
    stage: StageCallable
    produces: frozenset[str]
    predecessors: tuple[str, ...]


@dataclass(frozen=True)
class StageDAG:
    """Enabled stages in an order that satisfies their dependencies.

    ``nodes`` contains exactly the enabled set. A callable injected for a
    disabled stage is not stored and is never invoked.
    """

    lattice: StageLattice
    nodes: tuple[StageNode, ...]

    @property
    def order(self) -> tuple[str, ...]:
        """Stage names in run order — the same tuple ``lattice.dag()`` returned."""
        return tuple(node.name for node in self.nodes)

    def run(
        self,
        source: Mapping[str, Any] | None = None,
        *,
        on_stage: Callable[[str, float], None] | None = None,
        heartbeat_interval: float | None = None,
    ) -> dict[str, Any]:
        """Execute every enabled stage once, in DAG order.

        ``source`` seeds the artifact bag (the runner puts the chunk here). Each
        stage's return value is stored under the stage name and under every
        artifact that stage produces. ``on_stage`` receives ``(name, seconds)``
        after each stage. ``heartbeat_interval`` prints a still-running line
        while a stage is blocked; ``None`` keeps unit tests quiet.
        """
        from src.pipeline.dag.heartbeat import Heartbeat

        bag: dict[str, Any] = dict(source or {})
        for node in self.nodes:
            started = time.perf_counter()
            if heartbeat_interval is not None and heartbeat_interval > 0:
                with Heartbeat(f"stage {node.name}", interval_s=heartbeat_interval):
                    output = node.stage(bag)
            else:
                output = node.stage(bag)
            elapsed = time.perf_counter() - started
            if on_stage is not None:
                on_stage(node.name, elapsed)
            bag[node.name] = output
            for artifact in node.produces:
                bag[artifact] = output
        return bag


def build_dag(
    lattice: StageLattice | LatticeConfig | PointstreamConfig,
    backends: Mapping[str, StageCallable],
    *,
    conditioning: Iterable[str] = (),
) -> StageDAG:
    """Bind injected callables onto the enabled stages of ``lattice``.

    Args:
        lattice: A corner, or a config that names one.
        backends: Stage name → callable. Extra keys (disabled stages, typos
            for stages that are off) are ignored, which is how a caller can
            inject a full roster and still pay only for what the lattice
            enables. Missing keys for *enabled* stages fail here, not mid-run.
        conditioning: Conditioning kinds the chosen generator declared. Passed
            through to ``assert_coherent`` so a pose-conditioned generator with
            pose off fails at build time.

    Raises:
        ConfigValueError: Incoherent enabled set, unsatisfied conditioning, or
            an enabled stage with no callable.
        TypeError: A bound backend is not callable.
    """
    corner = as_lattice(lattice)
    corner.assert_coherent(conditioning=conditioning)
    order = corner.dag()
    missing = [name for name in order if name not in backends]
    if missing:
        raise ConfigValueError(
            "pipeline.backends",
            f"enabled stage(s) {missing} have no injected backend. "
            f"The pipeline does not look up registries; bind a callable per "
            f"enabled stage.",
        )
    produced = corner.produced_artifacts()
    nodes: list[StageNode] = []
    for name in order:
        bound = backends[name]
        if not callable(bound):
            raise ConfigValueError(
                f"pipeline.backends.{name}",
                f"backend for stage {name!r} is not callable.",
            )
        spec = stage(name)
        nodes.append(
            StageNode(
                name=name,
                stage=bound,
                produces=spec.produces,
                predecessors=_predecessors(name, corner.enabled, produced),
            )
        )
    return StageDAG(lattice=corner, nodes=tuple(nodes))


def _predecessors(
    name: str, enabled: frozenset[str], produced: frozenset[str]
) -> tuple[str, ...]:
    """Enabled stages whose artifacts this stage consumes, catalogue order."""
    spec = STAGES[name]
    needed = spec.consumes | (spec.optional_inputs & produced)
    preds = [
        other
        for other in sorted(enabled, key=lambda item: STAGES[item].row)
        if other != name and STAGES[other].produces & needed
    ]
    return tuple(preds)
