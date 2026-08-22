"""Cross-engine probe harness. Coding-task triage only — nothing here is citable."""

from experiments.probe.bounds import (
    NOT_USING_APPEARANCE,
    STATIC_COPY_ALARM_HIGH_DB,
    STATIC_COPY_ALARM_LOW_DB,
    STATIC_COPY_EXPECTED_HIGH_DB,
    STATIC_COPY_EXPECTED_LOW_DB,
)
from experiments.probe.clips import DEFAULT_OFFSETS, HEADLINE_OFFSET
from experiments.probe.engines import PLANS, SEED, STATIC_COPY

__all__ = [
    "DEFAULT_OFFSETS",
    "HEADLINE_OFFSET",
    "NOT_USING_APPEARANCE",
    "PLANS",
    "SEED",
    "STATIC_COPY",
    "STATIC_COPY_ALARM_HIGH_DB",
    "STATIC_COPY_ALARM_LOW_DB",
    "STATIC_COPY_EXPECTED_HIGH_DB",
    "STATIC_COPY_EXPECTED_LOW_DB",
]
