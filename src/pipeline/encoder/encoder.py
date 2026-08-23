"""Encode one bag of inputs through a lattice corner.

The chunk loop, routing, accounting and evaluation belong to the runner.
This module stops at: take a corner, take injected stage callables, run them.
The all-off corner is not a special path — it is three nodes on the same DAG.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from src.contracts.config import LatticeConfig, PointstreamConfig
from src.contracts.lattice import StageLattice
from src.pipeline.dag.graph import StageCallable, StageDAG, as_lattice, build_dag

#: Convention for the runner: the source chunk/frames live under this key.
SOURCE = "source"


@dataclass(frozen=True)
class Encoder:
    """A bound DAG for one lattice corner.

    Built from the enabled stage set. Never reads a backend name, never
    branches on "baseline".
    """

    dag: StageDAG

    @property
    def lattice(self) -> StageLattice:
        return self.dag.lattice

    @property
    def stages(self) -> tuple[str, ...]:
        return self.dag.order

    def encode(self, source: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Run every enabled stage once. Disabled stages are not in the graph."""
        return self.dag.run(source)

    @classmethod
    def build(
        cls,
        lattice: StageLattice | LatticeConfig | PointstreamConfig,
        backends: Mapping[str, StageCallable],
        *,
        conditioning: Iterable[str] = (),
    ) -> Encoder:
        """Bind ``backends`` onto the enabled stages of ``lattice``."""
        return cls(build_dag(as_lattice(lattice), backends, conditioning=conditioning))
