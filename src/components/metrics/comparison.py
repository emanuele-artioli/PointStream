"""Comparing two arms: the effect, its uncertainty, and whether it is readable.

**Why this exists.** A +0.98 dB difference between two engines was reported here
as a finding. Across 12 clips the per-clip standard deviation was ~2.0 dB, so the
standard error was ~0.58 dB and the effect was ~1.7 standard errors — about a one
in eight chance under the null. Nothing in the reporting path said so, because
nothing computed it.

This module makes that arithmetic unavoidable: a comparison carries its own
uncertainty, and ``describe()`` refuses to state a direction the sample does not
support.

No SciPy dependency: the normal approximation is enough to separate "clearly
real" from "inside the noise", and the small-sample warning covers the rest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

MIN_READABLE_N = 8
"""Below this, report the effect but never call a direction on it alone."""

CLEAR_SIGMA = 2.0
"""Roughly p < 0.05 two-sided under the normal approximation."""

SUGGESTIVE_SIGMA = 1.0


@dataclass(frozen=True)
class PairedComparison:
    """A paired arm-vs-arm comparison over the same items.

    Paired because the arms are run on the *same* clips: the per-item difference
    removes the clip-to-clip variance, which is the dominant term here and would
    otherwise swamp any effect.
    """

    name_a: str
    name_b: str
    n: int
    mean_difference: float
    standard_error: float
    higher_is_better: bool

    @property
    def sigmas(self) -> float:
        if self.standard_error == 0.0:
            return math.inf if self.mean_difference != 0.0 else 0.0
        return abs(self.mean_difference) / self.standard_error

    @property
    def verdict(self) -> str:
        """``clear``, ``suggestive``, ``inside-noise``, or ``underpowered``."""
        if self.n < MIN_READABLE_N:
            return "underpowered"
        if self.sigmas >= CLEAR_SIGMA:
            return "clear"
        if self.sigmas >= SUGGESTIVE_SIGMA:
            return "suggestive"
        return "inside-noise"

    @property
    def winner(self) -> str | None:
        """The better arm, or ``None`` when the sample cannot support a call."""
        if self.verdict in {"inside-noise", "underpowered"}:
            return None
        better_is_a = self.mean_difference > 0
        if not self.higher_is_better:
            better_is_a = not better_is_a
        return self.name_a if better_is_a else self.name_b

    def describe(self) -> str:
        delta = f"{self.mean_difference:+.3f} +/- {self.standard_error:.3f}"
        head = f"{self.name_a} - {self.name_b} = {delta} (n={self.n}, {self.sigmas:.1f}σ)"
        if self.verdict == "underpowered":
            return f"{head} — UNDERPOWERED: n<{MIN_READABLE_N}, no direction claimed."
        if self.verdict == "inside-noise":
            return f"{head} — INSIDE NOISE: do not report a winner."
        if self.verdict == "suggestive":
            return f"{head} — SUGGESTIVE only ({self.winner} ahead); not a result on its own."
        return f"{head} — CLEAR: {self.winner} ahead."


def compare_paired(
    name_a: str,
    scores_a: Sequence[float],
    name_b: str,
    scores_b: Sequence[float],
    *,
    higher_is_better: bool = True,
) -> PairedComparison:
    """Compare two arms scored on the same items, in order.

    Raises:
        ValueError: If the arms differ in length, or fewer than two items are
            supplied — a standard error is undefined for one item, and reporting
            a difference without one is the failure this module exists to stop.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"paired comparison needs the same items in both arms: "
            f"{len(scores_a)} vs {len(scores_b)}"
        )
    if len(scores_a) < 2:
        raise ValueError(
            "a paired comparison needs at least two items; with one there is no "
            "standard error, and a bare difference is exactly what this refuses."
        )
    differences = [float(a) - float(b) for a, b in zip(scores_a, scores_b)]
    n = len(differences)
    mean = sum(differences) / n
    variance = sum((d - mean) ** 2 for d in differences) / (n - 1)
    return PairedComparison(
        name_a=name_a,
        name_b=name_b,
        n=n,
        mean_difference=mean,
        standard_error=math.sqrt(variance / n),
        higher_is_better=higher_is_better,
    )
