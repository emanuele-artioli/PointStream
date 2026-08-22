"""Cross-engine probe harness. Triage only — nothing here is citable."""

from experiments.probe.bounds import (
    OBJECT_PSNR_ALARM_HIGH_DB,
    OBJECT_PSNR_ALARM_LOW_DB,
    OBJECT_PSNR_EXPECTED_HIGH_DB,
    OBJECT_PSNR_EXPECTED_LOW_DB,
)
from experiments.probe.engines import PLANS, SEED

__all__ = [
    "OBJECT_PSNR_ALARM_HIGH_DB",
    "OBJECT_PSNR_ALARM_LOW_DB",
    "OBJECT_PSNR_EXPECTED_HIGH_DB",
    "OBJECT_PSNR_EXPECTED_LOW_DB",
    "PLANS",
    "SEED",
]
