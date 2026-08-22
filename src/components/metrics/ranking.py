"""Rank scores by a metric's declared direction. Never by its name."""

from __future__ import annotations

from collections.abc import Mapping
from functools import cmp_to_key

from src.contracts.metrics import MetricSpec


def rank(scores: Mapping[str, float], spec: MetricSpec) -> tuple[str, ...]:
    """Candidate names, best first, using ``spec.is_better`` only."""

    def compare(left: str, right: str) -> int:
        if spec.is_better(scores[left], scores[right]):
            return -1
        if spec.is_better(scores[right], scores[left]):
            return 1
        return 0

    return tuple(sorted(scores, key=cmp_to_key(compare)))
