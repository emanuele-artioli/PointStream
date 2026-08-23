"""Lattice corners derived from the contracts, not a hand-picked subset.

The required-behaviour suite asserts every one of these builds a runnable
pipeline. C3's ablation matrix should walk the same set.
"""

from __future__ import annotations

from src.contracts.lattice import (
    FULL,
    NAMED_CORNERS,
    OPTIONAL_STAGES,
    StageLattice,
)


def iter_lattice_corners() -> tuple[StageLattice, ...]:
    """Every corner the contracts name or derive.

    Includes ``all_on`` / ``all_off``, every named corner, and ``FULL.prune``
    of each optional stage. Deduplicated by enabled set, so ``prune(residual)``
    and ``generative-only`` appear once.
    """
    seen: set[frozenset[str]] = set()
    corners: list[StageLattice] = []
    candidates: tuple[StageLattice, ...] = (
        StageLattice.all_on(),
        StageLattice.all_off(),
        *NAMED_CORNERS.values(),
        *(FULL.prune(name) for name in OPTIONAL_STAGES),
    )
    for candidate in candidates:
        if candidate.enabled in seen:
            continue
        seen.add(candidate.enabled)
        corners.append(candidate)
    return tuple(corners)
