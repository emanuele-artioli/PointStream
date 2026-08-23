"""The residual coarseness spectrum, including absent.

This is a rate axis of its own: absent → progressively coarser → fine →
lossless. Lossless is a ceiling calibration, not an operating point. Absent
reports the unaided quality of the reconstruction — measurable before any
generator question is settled.

Lattice variants are ``lossy``, ``lossless``, ``none``. ``none`` is the
residual stage switched off, which is not the same as a lossless residual
of zeros: a residual with nothing to correct is still a coarseness knob,
and the all-off corner does not have that knob.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.contracts.codecs import RateControl
from src.contracts.config import ResidualConfig
from src.contracts.lattice import STAGE_RESIDUAL, StageLattice


class ResidualVariant(str, Enum):
    """The three catalogue variants on the residual row."""

    NONE = "none"
    LOSSY = "lossy"
    LOSSLESS = "lossless"


class Coarseness(str, Enum):
    """Rungs on the residual-coarseness curve, coarsest first after absent."""

    ABSENT = "absent"
    COARSE = "coarse"
    MEDIUM = "medium"
    FINE = "fine"
    LOSSLESS = "lossless"


@dataclass(frozen=True)
class ResidualPoint:
    """One operating point on the coarseness axis."""

    coarseness: Coarseness
    variant: ResidualVariant
    config: ResidualConfig | None
    """None when the residual stage is off — there is no encode to configure."""

    def describe(self) -> str:
        if self.variant is ResidualVariant.NONE or self.config is None:
            return "residual absent; quality is the unaided reconstruction"
        if self.variant is ResidualVariant.LOSSLESS:
            return "residual lossless (ceiling calibration, not an operating point)"
        cfg = self.config
        return (
            f"residual {self.coarseness.value}: {cfg.codec} {cfg.rate_control.value}="
            f"{cfg.rate}, block {cfg.block_size}/{cfg.block_threshold:g}, "
            f"bg downscale {cfg.background_downscale}"
        )


def variant_for(lattice: StageLattice, residual: ResidualConfig | None = None) -> ResidualVariant:
    """Catalogue variant implied by the lattice and residual config."""
    if STAGE_RESIDUAL not in lattice.enabled:
        return ResidualVariant.NONE
    if residual is not None and residual.rate_control is RateControl.LOSSLESS:
        return ResidualVariant.LOSSLESS
    return ResidualVariant.LOSSY


def point_for(
    lattice: StageLattice,
    residual: ResidualConfig | None = None,
    *,
    coarseness: Coarseness | None = None,
) -> ResidualPoint:
    """The spectrum point this lattice corner and residual config name."""
    variant = variant_for(lattice, residual)
    if variant is ResidualVariant.NONE:
        return ResidualPoint(Coarseness.ABSENT, ResidualVariant.NONE, None)
    if variant is ResidualVariant.LOSSLESS:
        return ResidualPoint(Coarseness.LOSSLESS, ResidualVariant.LOSSLESS, residual)
    rung = coarseness if coarseness is not None else infer_lossy_rung(residual)
    return ResidualPoint(rung, ResidualVariant.LOSSY, residual)


def infer_lossy_rung(residual: ResidualConfig | None) -> Coarseness:
    """Map block/downscale knobs onto the named lossy rungs.

    Fine is no block gate and no background downscale. Coarse is a high
    activity threshold or a large downscale. Medium is everything in between.
    Named so a sweep can say "coarse" without restating the numbers.
    """
    if residual is None:
        return Coarseness.FINE
    gated = residual.block_threshold > 0.0 and residual.block_size > 1
    downscaled = residual.background_downscale > 1
    if residual.block_threshold >= 8.0 or residual.background_downscale >= 4:
        return Coarseness.COARSE
    if gated or downscaled:
        return Coarseness.MEDIUM
    return Coarseness.FINE


def coarseness_ladder() -> tuple[ResidualPoint, ...]:
    """The residual-coarseness curve, absent through lossless.

    Lossy rungs keep the default residual codec so a sweep does not silently
    change the encoder. Lossless uses AVC: it is the rung that actually
    honours ``rate_control=lossless`` on this host. That is a ceiling
    calibration, not a recommended operating point.
    """
    return (
        ResidualPoint(Coarseness.ABSENT, ResidualVariant.NONE, None),
        ResidualPoint(
            Coarseness.COARSE,
            ResidualVariant.LOSSY,
            ResidualConfig(
                rate=63,
                block_size=16,
                block_threshold=16.0,
                background_downscale=4,
            ),
        ),
        ResidualPoint(
            Coarseness.MEDIUM,
            ResidualVariant.LOSSY,
            ResidualConfig(
                rate=45,
                block_size=8,
                block_threshold=4.0,
                background_downscale=2,
            ),
        ),
        ResidualPoint(
            Coarseness.FINE,
            ResidualVariant.LOSSY,
            ResidualConfig(
                rate=28,
                block_size=8,
                block_threshold=0.0,
                background_downscale=1,
            ),
        ),
        ResidualPoint(
            Coarseness.LOSSLESS,
            ResidualVariant.LOSSLESS,
            ResidualConfig(
                codec="avc",
                rate_control=RateControl.LOSSLESS,
                rate=0,
                block_size=1,
                block_threshold=0.0,
                background_downscale=1,
            ),
        ),
    )
