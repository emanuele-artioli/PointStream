"""Residual coarseness spectrum, absorption, and lossless bit-identity."""

from __future__ import annotations

import numpy as np
import pytest

from src.contracts.codecs import RateControl
from src.contracts.config import ResidualConfig
from src.contracts.lattice import STAGE_BACKGROUND, STAGE_RESIDUAL, SOURCE_PASSTHROUGH, StageLattice
from src.pipeline.reconstruction import BackgroundModelView, ReconstructionRequest, bit_identical, reconstruct
from src.pipeline.residual import (
    Coarseness,
    ResidualVariant,
    apply_residual,
    coarseness_ladder,
    compute_residual,
    l1_energy,
    signed_residual,
    variant_for,
)


def _clip(value: int, *, frames: int = 1, size: int = 32) -> np.ndarray:
    return np.full((frames, size, size, 3), value, dtype=np.uint8)


def test_all_off_has_no_residual_variant() -> None:
    assert SOURCE_PASSTHROUGH.is_source_passthrough
    assert variant_for(SOURCE_PASSTHROUGH) is ResidualVariant.NONE


def test_absent_residual_leaves_the_reconstruction_unchanged() -> None:
    source = _clip(80)
    recon = _clip(10)
    result = compute_residual(source, recon, lattice=SOURCE_PASSTHROUGH)
    assert result.payload.is_absent
    assert result.payload.byte_count == 0
    assert result.payload.nonzero_bytes == 0
    assert bit_identical(recon, result.reconstructed)
    assert bit_identical(recon, apply_residual(recon, result.payload))


def test_absent_versus_lossless_actually_changes_the_payload() -> None:
    """Bounds: absent is 0 bytes. Lossless of a 1×32×32×3 uint8 pair that
    differs everywhere is 1*32*32*3*2 = 6144 bytes. Alarm if lossless is 0
    or absent is not 0 — the flag would be existing without working."""
    source = _clip(200, size=32)
    recon = _clip(10, size=32)
    absent = compute_residual(source, recon, lattice=SOURCE_PASSTHROUGH)
    lossless = compute_residual(
        source,
        recon,
        lattice=StageLattice.of(STAGE_RESIDUAL),
        residual=ResidualConfig(codec="avc", rate_control=RateControl.LOSSLESS, rate=0),
    )
    assert absent.payload.byte_count == 0
    assert lossless.payload.variant is ResidualVariant.LOSSLESS
    assert lossless.payload.byte_count == 32 * 32 * 3 * 2
    assert lossless.payload.byte_count != absent.payload.byte_count
    assert lossless.payload.l1_energy > 0.0


def test_lossless_apply_restores_the_source_bit_for_bit() -> None:
    source = _clip(17, size=16)
    source[0, 2, 3] = (1, 2, 3)
    recon = _clip(0, size=16)
    result = compute_residual(
        source,
        recon,
        lattice=StageLattice.of(STAGE_RESIDUAL),
        residual=ResidualConfig(
            codec="avc",
            rate_control=RateControl.LOSSLESS,
            rate=0,
        ),
    )
    assert bit_identical(source, result.reconstructed)
    assert bit_identical(source, apply_residual(recon, result.payload))


def test_coarser_lossy_transmits_less_information() -> None:
    """A 2-grey background residual is below the coarse threshold; a 40-grey
    object block is not. Bounds: coarse active_blocks < fine active_blocks.
    Alarm if they tie — coarseness would be a no-op."""
    recon = _clip(0, size=32)
    source = _clip(2, size=32)
    source[0, 0:8, 0:8] = 40
    fine = compute_residual(
        source,
        recon,
        lattice=StageLattice.of(STAGE_RESIDUAL),
        residual=ResidualConfig(block_size=8, block_threshold=0.0, background_downscale=1),
        coarseness=Coarseness.FINE,
    )
    coarse = compute_residual(
        source,
        recon,
        lattice=StageLattice.of(STAGE_RESIDUAL),
        residual=ResidualConfig(block_size=8, block_threshold=8.0, background_downscale=1),
        coarseness=Coarseness.COARSE,
    )
    assert coarse.payload.active_blocks < fine.payload.active_blocks
    assert coarse.payload.nonzero_bytes < fine.payload.nonzero_bytes
    assert coarse.payload.l1_energy < fine.payload.l1_energy


def test_disabled_background_grows_the_residual() -> None:
    source = _clip(40, size=32)
    source[0, 8:24, 8:24] = 200
    plate = np.full((32, 32, 3), 40, dtype=np.uint8)
    with_bg = reconstruct(
        ReconstructionRequest(
            lattice=StageLattice.of(STAGE_BACKGROUND, STAGE_RESIDUAL),
            source=source,
            background=BackgroundModelView(plate=plate),
        )
    )
    without_bg = reconstruct(
        ReconstructionRequest(
            lattice=StageLattice.of(STAGE_RESIDUAL),
            source=source,
            background=BackgroundModelView(plate=plate),
        )
    )
    residual_on = compute_residual(
        source, with_bg.frames, lattice=StageLattice.of(STAGE_RESIDUAL)
    )
    residual_off = compute_residual(
        source, without_bg.frames, lattice=StageLattice.of(STAGE_RESIDUAL)
    )
    assert residual_off.payload.l1_energy > residual_on.payload.l1_energy


def test_lossy_roundtrip_is_close_not_bit_identical_when_the_diff_clips() -> None:
    """A 200-grey difference does not fit in [-128, 127]. Lossy must clip.
    Bounds: restored is not the source; mean error is on the order of
    200-127 = 73 grey. Alarm if restored matches source (would mean lossless
    wearing a lossy name) or if error is ~0."""
    source = _clip(200, size=16)
    recon = _clip(0, size=16)
    result = compute_residual(
        source,
        recon,
        lattice=StageLattice.of(STAGE_RESIDUAL),
        residual=ResidualConfig(rate_control=RateControl.CRF, block_threshold=0.0, background_downscale=1),
    )
    assert not bit_identical(source, result.reconstructed)
    delta = np.abs(source.astype(np.int16) - result.reconstructed.astype(np.int16))
    assert float(delta.mean()) >= 50.0


def test_spectrum_includes_absent_and_lossless() -> None:
    ladder = coarseness_ladder()
    kinds = [point.coarseness for point in ladder]
    assert kinds[0] is Coarseness.ABSENT
    assert kinds[-1] is Coarseness.LOSSLESS
    assert Coarseness.FINE in kinds
    assert ladder[0].config is None
    assert ladder[-1].config is not None


def test_shape_mismatch_is_refused() -> None:
    with pytest.raises(ValueError, match="shape"):
        compute_residual(_clip(1, size=8), _clip(1, size=4), lattice=StageLattice.of(STAGE_RESIDUAL))


def test_l1_energy_of_identical_clips_is_zero() -> None:
    frames = _clip(9)
    assert l1_energy(signed_residual(frames, frames)) == 0.0


# ---------------------------------------------------------------------------
# `WireCost.exact` on the residual — the honesty pass (BP24)
#
# `exact` means "byte_count is what goes on the wire". Neither residual path
# produces that: both hand a pixel payload *to* a codec, and the bitstream that
# comes back is what is actually transmitted. Before this, both set exact=True
# on top of a basis describing an array, which was true only while nothing in
# the project ran an encoder.
#
# The consequence is not cosmetic. `__add__` conjoins the flag, so a stand-in
# left marked exact would be summed into a total that then reports a
# transport-to-source ratio — a compression ratio computed from an array size.
# ---------------------------------------------------------------------------


def test_a_pre_codec_residual_does_not_claim_to_be_a_bitstream() -> None:
    lattice = StageLattice.of(STAGE_RESIDUAL)
    source = _clip(200, frames=2)
    reconstruction = _clip(150, frames=2)

    lossless = compute_residual(
        source, reconstruction, lattice=lattice, coarseness=Coarseness.LOSSLESS
    )
    assert lossless.payload.cost.exact is False
    assert "not a bitstream" in lossless.payload.cost.basis

    lossy = compute_residual(
        source,
        reconstruction,
        lattice=lattice,
        residual=ResidualConfig(block_size=8, block_threshold=0.0),
        coarseness=Coarseness.MEDIUM,
    )
    assert lossy.payload.variant is ResidualVariant.LOSSY
    assert lossy.payload.cost.exact is False
    assert "not a bitstream" in lossy.payload.cost.basis


def test_an_absent_residual_is_an_exact_zero_not_a_stand_in() -> None:
    """Sending nothing is a measurement, not an unmeasured cost.

    The distinction matters because `exact=False` on a zero would make every
    residual-off corner refuse to report a ratio — the guard firing on the one
    corner with nothing to hide.

    Absence is a *lattice* property, not a coarseness value. `point_for` reads
    the variant from the lattice first and only then consults `coarseness`, so
    passing `Coarseness.ABSENT` to an enabled residual stage yields a LOSSY
    payload labelled absent. The corner is expressed by leaving the stage out.
    """
    absent = compute_residual(
        _clip(200), _clip(150), lattice=StageLattice.of(STAGE_BACKGROUND)
    )
    assert absent.payload.is_absent
    assert absent.payload.cost.byte_count == 0
    assert absent.payload.cost.exact is True


def test_summing_a_stand_in_with_a_real_cost_stays_a_stand_in() -> None:
    """The laundering case, at the `WireCost` level rather than the ledger's."""
    from src.contracts.objectstream import WireCost

    coded = WireCost(byte_count=1_000, exact=True, basis="measured bitstream")
    stand_in = compute_residual(
        _clip(200, frames=2), _clip(150, frames=2),
        lattice=StageLattice.of(STAGE_RESIDUAL),
        coarseness=Coarseness.LOSSLESS,
    ).payload.cost
    assert (coded + stand_in).exact is False
    assert (stand_in + coded).exact is False
