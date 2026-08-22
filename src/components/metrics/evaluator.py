"""Build a metric set and score a reconstructed clip against its source.

Quality measurement is mandatory. An empty set is refused. Omitting PSNR is
not refused — it is added back and the result record says so, so a summary
can report the enforcement rather than hide it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from src.contracts.errors import ConfigValueError
from src.contracts.metrics import ALWAYS_ON, MetricSelection, resolve


class MetricBackend(Protocol):
    """One compute backend. Registry names match ``src.contracts.metrics``."""

    def score(self, reference: np.ndarray, predicted: np.ndarray) -> float: ...


@dataclass(frozen=True)
class EvaluationRecord:
    """One clip comparison. ``enforced`` is why PSNR is present if it was not asked."""

    scores: Mapping[str, float]
    enforced: tuple[str, ...]
    selection: MetricSelection
    n_frames: int

    def describe(self) -> str:
        return self.selection.describe()


class Evaluator:
    """Scores a reference/predicted clip pair on a resolved metric set."""

    def __init__(
        self,
        names: Sequence[str],
        *,
        backends: Mapping[str, MetricBackend] | None = None,
        registry: Any = None,
    ) -> None:
        requested = [str(name).strip().lower() for name in names if str(name).strip()]
        if not requested:
            raise ConfigValueError(
                "evaluation.metrics",
                "an empty metric set measures nothing. Quality measurement is "
                "mandatory in every configuration.",
            )
        self._selection = resolve(requested)
        table = registry
        if table is None:
            from src.components.metrics import REGISTRY as table
        self._backends = self._bind(self._selection, backends, table)

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        backends: Mapping[str, MetricBackend] | None = None,
        registry: Any = None,
    ) -> Evaluator:
        return cls(config.evaluation.metrics, backends=backends, registry=registry)

    @property
    def selection(self) -> MetricSelection:
        return self._selection

    def evaluate(self, reference: np.ndarray, predicted: np.ndarray) -> EvaluationRecord:
        from src.components.metrics.frames import paired

        ref, pred = paired(reference, predicted)
        scores: dict[str, float] = {}
        for spec in self._selection:
            scores[spec.name] = float(self._backends[spec.name].score(ref, pred))
        return EvaluationRecord(
            scores=scores,
            enforced=self._selection.enforced,
            selection=self._selection,
            n_frames=int(ref.shape[0]),
        )

    def _bind(
        self,
        selection: MetricSelection,
        backends: Mapping[str, MetricBackend] | None,
        registry: Any,
    ) -> dict[str, MetricBackend]:
        bound: dict[str, MetricBackend] = {}
        supplied = dict(backends or {})
        for spec in selection:
            if spec.name in supplied:
                bound[spec.name] = supplied[spec.name]
            else:
                bound[spec.name] = registry.build(spec.name)
        missing = sorted(ALWAYS_ON - set(bound))
        if missing:
            raise RuntimeError(f"Evaluator omitted mandatory metric(s): {', '.join(missing)}")
        return bound
