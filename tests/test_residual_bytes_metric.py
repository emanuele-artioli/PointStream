"""Tests for the residual-bytes ranking currency (Phase 0.2).

Under the Residual Guarantee a generator earns its place only by shrinking the
residual payload by more than the metadata it adds, so the campaign ranks on
bytes rather than on how good the output looks. These tests pin the two things
that make that ranking trustworthy: the measurement uses the *encoder's* exact
residual representation, and it prefers cheap-to-correct errors over
expensive-to-correct ones even when a pixel metric disagrees.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from scripts.eval_checkpoint import compute_residual_bytes
from scripts.train_campaign import PRIMARY_METRIC, rank_variants
from src.encoder.residual_calculator import apply_block_activity_gate, residual_to_encodable_uint8


# ---------------------------------------------------------------------------
# The shared residual representation
# ---------------------------------------------------------------------------


def test_residual_to_encodable_uint8_uses_the_encoder_offset() -> None:
    """Zero error must land at 128, and the range must clamp rather than wrap."""
    residual = torch.tensor([[[0.0, 10.0, -10.0, 400.0, -400.0]]])  # Shape: [1, 1, 5]

    encoded = residual_to_encodable_uint8(residual)

    assert encoded.dtype == torch.uint8
    assert encoded.flatten().tolist() == [128, 138, 118, 255, 0]


def test_block_activity_gate_is_a_noop_at_threshold_zero() -> None:
    """Threshold 0.0 is the measured best cell of the residual matrix, so it
    must pass the signal through untouched rather than silently gating."""
    residual = torch.randn(1, 3, 16, 16) * 5.0  # Shape: [1, 3, 16, 16]
    assert torch.equal(apply_block_activity_gate(residual, block_size=8, threshold=0.0), residual)


def test_block_activity_gate_drops_only_quiet_blocks() -> None:
    residual = torch.zeros(1, 3, 16, 16)  # Shape: [1, 3, 16, 16]
    residual[:, :, :8, :8] = 50.0  # one loud block

    gated = apply_block_activity_gate(residual, block_size=8, threshold=2.0)

    assert torch.all(gated[:, :, :8, :8] == 50.0), "loud block must survive"
    assert torch.all(gated[:, :, 8:, :] == 0.0)
    assert torch.all(gated[:, :, :, 8:] == 0.0)


# ---------------------------------------------------------------------------
# The measurement itself (needs ffmpeg)
# ---------------------------------------------------------------------------


_RESIDUAL_KWARGS = dict(
    codec="libx264", crf=28, preset="veryfast", pix_fmt="yuv444p", block_size=8, block_threshold=0.0
)


def _bytes_for(ground_truth: torch.Tensor, predicted: torch.Tensor) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        return compute_residual_bytes(
            ground_truth, predicted, fps=24.0, tmp_dir=Path(tmp), **_RESIDUAL_KWARGS
        )


@pytest.mark.integration
def test_perfect_prediction_is_the_cheapest_residual() -> None:
    """A perfect prediction leaves a constant (all-128) residual -- the floor."""
    torch.manual_seed(0)
    ground_truth = torch.rand(8, 3, 64, 64)  # Shape: [N, 3, H, W]

    perfect = _bytes_for(ground_truth, ground_truth.clone())
    noisy = _bytes_for(ground_truth, torch.rand(8, 3, 64, 64))

    assert perfect < noisy, "an exact prediction must cost fewer residual bytes than a random one"


@pytest.mark.integration
def test_bytes_discriminate_error_structure_not_just_magnitude() -> None:
    """The property that justifies ranking on bytes rather than on a pixel metric.

    Two predictions with **identical MSE** can cost very different payloads: a
    high-frequency error is expensive to code, a smooth low-frequency one is
    cheap. Any MSE-derived metric (PSNR included) scores these two the same, so
    it cannot express the question the Residual Guarantee actually asks.

    Measured at 16x256x256: white-noise error costs ~3.5x the bytes of a
    low-frequency error at the same magnitude. The assertion uses a loose 1.5x
    bound so it pins the direction without pinning a codec version's exact
    behaviour.

    Note on scale: at 8x64x64 both files are ~1.8 KB and indistinguishable --
    that is container overhead, not content. A byte-cost test below a few tens
    of KB measures nothing.
    """
    torch.manual_seed(0)
    ground_truth = torch.rand(16, 3, 256, 256) * 0.5 + 0.25  # Shape: [N, 3, H, W]

    white = torch.randn_like(ground_truth) * 0.1
    low_freq = torch.nn.functional.interpolate(
        torch.randn(16, 3, 16, 16) * 0.1, size=(256, 256), mode="bicubic", align_corners=False
    )
    low_freq = low_freq * (white.std() / low_freq.std())  # match magnitude exactly

    mse_white = float((white ** 2).mean())
    mse_low = float((low_freq ** 2).mean())
    assert abs(mse_white - mse_low) < 1e-4, "the two errors must have equal magnitude for this to mean anything"

    bytes_white = _bytes_for(ground_truth, ground_truth + white)
    bytes_low = _bytes_for(ground_truth, ground_truth + low_freq)

    assert bytes_white > 1.5 * bytes_low, (
        f"high-frequency error ({bytes_white} B) must cost materially more than an equal-magnitude "
        f"low-frequency one ({bytes_low} B); if this inverts, the case for bytes-as-currency needs re-examining"
    )


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def test_residual_bytes_outranks_the_perceptual_composite() -> None:
    """Fewest bytes wins even when another variant looks better on every metric.

    This is the whole point of Phase 0.2: the campaign previously averaged
    incommensurable perceptual metrics into a composite, which cannot express
    the project's actual objective.
    """
    aggregate = {
        "pretty": {"psnr_mean": 30.0, "ssim_mean": 0.95, "vmaf_mean": 80.0, PRIMARY_METRIC: 900_000},
        "cheap": {"psnr_mean": 20.0, "ssim_mean": 0.70, "vmaf_mean": 10.0, PRIMARY_METRIC: 100_000},
    }

    ranked, composite = rank_variants(aggregate)

    assert ranked[0] == "cheap"
    assert composite["pretty"] > composite["cheap"], "composite is still reported, just not decisive"


def test_ranking_falls_back_to_composite_when_bytes_are_missing() -> None:
    """--no-residual-bytes runs must still rank, on the weaker basis."""
    aggregate = {
        "a": {"psnr_mean": 30.0, "ssim_mean": 0.95},
        "b": {"psnr_mean": 20.0, "ssim_mean": 0.70},
    }

    ranked, _ = rank_variants(aggregate)

    assert ranked[0] == "a"


def test_partial_byte_coverage_does_not_rank_on_bytes() -> None:
    """If only some variants were priced, comparing them on bytes would rank a
    variant against a metric its rival never reported."""
    aggregate = {
        "priced": {"psnr_mean": 20.0, PRIMARY_METRIC: 100_000},
        "unpriced": {"psnr_mean": 30.0},
    }

    ranked, _ = rank_variants(aggregate)

    assert ranked[0] == "unpriced", "falls back to the composite, where 'unpriced' wins on PSNR"
