"""The size ledger must shrink when the residual coarsens.

`src/pipeline/residual/signal.py` says it plainly in its own docstring: a dense
array's `nbytes` does not change when the block gate zeroes a block, so a ledger
reading `payload.byte_count` for a lossy residual reports the same payload for a
coarse residual as for a fine one — and coarseness looks free. That is not a
cosmetic difference: the residual-coarseness curve (`PLAN.md` §7 P0 item 3) is a
plot of payload against quality, and a payload axis that does not move is a flat
line through every rung.

The stated `WireCost` carries the information content instead, and for a
lossless residual the two numbers are equal, so nothing about the lossless
ceiling moves.
"""

from __future__ import annotations

import numpy as np

from src.contracts.codecs import RateControl
from src.contracts.config import PointstreamConfig, ResidualConfig
from src.contracts.lattice import STAGE_BACKGROUND, STAGE_RESIDUAL, StageLattice
from src.runner import lattice_config_from, run


def _clip_with_a_faint_wash() -> np.ndarray:
    """Two frames whose difference is large in one corner and faint elsewhere.

    The faint part is what a block gate is supposed to drop and a dense byte
    count is supposed to keep counting.
    """
    clip = np.full((2, 64, 64, 3), 100, dtype=np.uint8)
    clip[1] = 102          # a one-to-two grey-level wash over the whole frame
    clip[1, :16, :16] = 240  # and one corner that no gate should ever drop
    return clip


def _residual_config(**overrides: object) -> PointstreamConfig:
    base = {
        "codec": "av1",
        "rate_control": RateControl.CRF,
        "rate": 45,
        "block_size": 8,
        "block_threshold": 0.0,
        "background_downscale": 1,
    }
    base.update(overrides)
    # Background on, so the reconstruction is the first frame repeated and the
    # residual is the frame-to-frame difference. On the whole-frame-residual
    # corner the reconstruction is zeros and the residual is the picture itself,
    # where no gate at any sane threshold drops anything — which would make this
    # test pass for the wrong reason.
    return PointstreamConfig(
        lattice=lattice_config_from(StageLattice.of(STAGE_BACKGROUND, STAGE_RESIDUAL)),
        residual=ResidualConfig(**base),  # type: ignore[arg-type]
    )


def test_gating_blocks_shrinks_the_reported_payload() -> None:
    """Bounds first: the fine rung keeps every changed pixel, so its payload is
    a large fraction of the clip; the gated rung keeps only the bright corner,
    which is 1/16 of the frame. A gated payload at or above the fine one means
    the ledger is reading a dense array size."""
    clip = _clip_with_a_faint_wash()
    fine = run(_residual_config(block_threshold=0.0), [clip])
    gated = run(_residual_config(block_threshold=8.0), [clip])

    assert fine.sizes.residual > 0
    assert gated.sizes.residual > 0
    assert gated.sizes.residual < fine.sizes.residual, (
        f"gated residual {gated.sizes.residual} is not smaller than fine "
        f"{fine.sizes.residual}; coarseness is being reported as free"
    )
    assert fine.sizes.residual <= clip.nbytes


def test_the_lossless_ceiling_is_still_the_dense_payload() -> None:
    """The fix must not move the calibration rung: a lossless residual is dense
    signed int16 and its cost is that array, exactly."""
    clip = _clip_with_a_faint_wash()
    config = _residual_config(
        codec="avc",
        rate_control=RateControl.LOSSLESS,
        rate=0,
        block_size=1,
    )
    result = run(config, [clip])
    assert result.sizes.residual == clip.size * 2
    assert result.delivered_quality.bit_identical


def test_a_residual_that_is_switched_off_costs_nothing() -> None:
    """Absent is not "lossless with nothing in it" — it has no coarseness knob
    at all, and it must not be handed a byte count by the fallback path."""
    clip = _clip_with_a_faint_wash()
    from src.contracts.lattice import SOURCE_PASSTHROUGH

    result = run(PointstreamConfig(lattice=lattice_config_from(SOURCE_PASSTHROUGH)), [clip])
    assert result.sizes.residual == 0
    assert result.sizes.transport_total == clip.nbytes
