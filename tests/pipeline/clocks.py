"""Injected stage callables that record cost. Test helper, not a backend."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

from src.contracts.lattice import STAGES


class ClockedStage:
    """A stage callable that records calls, unit cost, and wall time.

    ``surcharge`` is added to ``cost`` on every call so a skip that still ran
    would dominate the measured total — the proof that a disabled stage is
    genuinely free, not nominally flagged.
    """

    def __init__(self, value: Any = "ok", *, surcharge: int = 0) -> None:
        self.value = value
        self.surcharge = surcharge
        self.calls = 0
        self.cost = 0
        self.elapsed_ns = 0

    def __call__(self, artifacts: Mapping[str, Any]) -> Any:
        start = time.perf_counter_ns()
        self.calls += 1
        self.cost += 1 + self.surcharge
        self.elapsed_ns += time.perf_counter_ns() - start
        return self.value


def full_roster(
    *,
    extra: Mapping[str, ClockedStage] | None = None,
    surcharge: Mapping[str, int] | None = None,
) -> dict[str, ClockedStage]:
    """A clock for every catalogue stage, enabled or not.

    Injecting the full roster is the measurement: a disabled stage's cost must
    stay zero even when a callable is sitting right there.
    """
    extras = dict(extra or {})
    surcharges = dict(surcharge or {})
    bound: dict[str, ClockedStage] = {}
    for name in STAGES:
        if name in extras:
            bound[name] = extras[name]
        else:
            bound[name] = ClockedStage(value=f"{name}:done", surcharge=surcharges.get(name, 0))
    return bound
