"""The staged low-rate search table. No runner, no encoder, no torch.

Kept separate from the sweep driver so the plan can be unit-tested without
paying the runner import tax.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.contracts import domain as domains


@dataclass(frozen=True)
class SweepPoint:
    """One configuration the search will actually run."""

    name: str
    stage: str
    stream_crf: int
    residual_qp: int | None
    residual_on: bool
    appearance_jpeg_quality: int
    appearance_downscale: int
    motion_max_points: int
    object_stream_on: bool
    background_method: str


#: Small staged walk, coarsest first. Not the full grid.
STAGES: tuple[tuple[str, tuple[SweepPoint, ...]], ...] = (
    (
        "background",
        (
            SweepPoint(
                "bg-crf51",
                "background",
                51,
                None,
                False,
                40,
                2,
                16,
                True,
                domains.BACKGROUND_PANORAMA_STREAM,
            ),
            SweepPoint(
                "bg-crf45",
                "background",
                45,
                None,
                False,
                40,
                2,
                16,
                True,
                domains.BACKGROUND_PANORAMA_STREAM,
            ),
            SweepPoint(
                "bg-crf38",
                "background",
                38,
                None,
                False,
                40,
                2,
                16,
                True,
                domains.BACKGROUND_PANORAMA_STREAM,
            ),
        ),
    ),
    (
        "correction",
        (
            SweepPoint(
                "corr-off",
                "correction",
                45,
                None,
                False,
                40,
                2,
                16,
                True,
                domains.BACKGROUND_PANORAMA_STREAM,
            ),
            SweepPoint(
                "corr-qp55",
                "correction",
                45,
                55,
                True,
                40,
                2,
                16,
                True,
                domains.BACKGROUND_PANORAMA_STREAM,
            ),
        ),
    ),
    (
        "appearance",
        (
            SweepPoint(
                "app-q30-ds2",
                "appearance",
                45,
                None,
                False,
                30,
                2,
                16,
                True,
                domains.BACKGROUND_PANORAMA_STREAM,
            ),
            SweepPoint(
                "app-q50-ds1",
                "appearance",
                45,
                None,
                False,
                50,
                1,
                16,
                True,
                domains.BACKGROUND_PANORAMA_STREAM,
            ),
        ),
    ),
    (
        "motion",
        (
            SweepPoint(
                "mot-8",
                "motion",
                45,
                None,
                False,
                40,
                2,
                8,
                True,
                domains.BACKGROUND_PANORAMA_STREAM,
            ),
            SweepPoint(
                "mot-32",
                "motion",
                45,
                None,
                False,
                40,
                2,
                32,
                True,
                domains.BACKGROUND_PANORAMA_STREAM,
            ),
        ),
    ),
    (
        "controls",
        (
            SweepPoint(
                "fallback-shape",
                "controls",
                45,
                None,
                False,
                40,
                2,
                16,
                False,
                domains.BACKGROUND_PANORAMA_STREAM,
            ),
            SweepPoint(
                "bg-full-crf45",
                "controls",
                45,
                None,
                False,
                40,
                2,
                16,
                True,
                domains.BACKGROUND_PANORAMA_FULL,
            ),
        ),
    ),
)


LEDGER_KEYS: tuple[str, ...] = (
    "panorama",
    "residual",
    "actor_reference",
    "metadata",
)


def stage_names() -> tuple[str, ...]:
    return tuple(name for name, _points in STAGES)


def points_for(stage: str) -> tuple[SweepPoint, ...]:
    for name, points in STAGES:
        if name == stage:
            return points
    raise ValueError(f"unknown stage {stage!r}; stages: {list(stage_names())}")


def all_points() -> tuple[SweepPoint, ...]:
    return tuple(point for _name, points in STAGES for point in points)


def named_point(name: str) -> SweepPoint:
    """One operating point by its unique name."""
    for point in all_points():
        if point.name == name:
            return point
    known = ", ".join(point.name for point in all_points())
    raise ValueError(f"unknown point {name!r}; known: {known}")


def select_work(
    *,
    stage: str | None = None,
    point: str | None = None,
) -> tuple[tuple[str, tuple[SweepPoint, ...]], ...]:
    """Which ``(stage, points)`` groups to run.

    ``point`` selects a single operating point. ``stage`` selects one family.
    Both unset runs the full staged walk. A point that does not belong to the
    named stage is refused rather than silently ignored.
    """
    if point is not None:
        chosen = named_point(point)
        if stage is not None and stage != chosen.stage:
            raise ValueError(
                f"point {point!r} belongs to stage {chosen.stage!r}, not {stage!r}"
            )
        return ((chosen.stage, (chosen,)),)
    if stage is not None:
        return ((stage, points_for(stage)),)
    return tuple((name, points) for name, points in STAGES)


def ledger_moved(rows: list[dict[str, Any]], *, key: str) -> bool:
    """Whether ``key`` takes more than one distinct byte count across rows."""
    values = {
        int((row.get("pointstream") or {}).get("parts", {}).get(key, -1))
        for row in rows
        if row.get("pointstream")
    }
    values.discard(-1)
    return len(values) > 1


def intended_category(stage: str) -> str:
    return {
        "background": "panorama",
        "correction": "residual",
        "appearance": "actor_reference",
        "motion": "metadata",
        "controls": "panorama",
    }[stage]
