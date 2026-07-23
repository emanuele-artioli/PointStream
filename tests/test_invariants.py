"""Unit tests for the run-level invariant checks.

These run on synthetic summaries in the fast tier. The point of the checks is
the run that *looks* fine — a mock-source fallback, a null PSNR, payload
accounting that does not add up — so those are what is pinned here.
"""

from __future__ import annotations

import pytest

from src.shared.invariants import check_run


def good_summary(**overrides):
    """A run that satisfies every invariant, for tests to perturb."""
    summary = {
        "chunk_id": "chunk_0001",
        "source_uri": "assets/real_tennis.mp4",
        "num_frames": 60,
        "evaluation": {
            "psnr_mean": 31.5,
            "psnr_num_frames": 60,
            "sizes_bytes": {
                "source": 1_000_000,
                "metadata": 50_000,
                "actor_reference": 20_000,
                "residual": 300_000,
                "panorama": 30_000,
                "transport_total": 400_000,
                "transport_to_source_ratio": 0.4,
            },
        },
    }
    summary.update(overrides)
    return summary


def test_a_healthy_run_has_no_failures():
    assert check_run(good_summary()) == []


# --- did it run on anything real -----------------------------------------------


def test_a_mock_source_is_flagged():
    """Omitting --input falls back to a mock source and silently tests nothing.

    The run completes and produces numbers, which is exactly the problem.
    """
    summary = good_summary(source_uri="outputs/x/runtime_sources/mock_tennis.mp4")
    assert any("mock source" in f for f in check_run(summary))


def test_a_missing_source_is_flagged():
    summary = good_summary()
    del summary["source_uri"]
    assert check_run(summary)


def test_zero_frames_is_flagged():
    assert check_run(good_summary(num_frames=0))


# --- was quality actually measured ---------------------------------------------


def test_a_null_psnr_is_flagged():
    """A null PSNR is a failed evaluation, not a quality of zero."""
    summary = good_summary()
    summary["evaluation"]["psnr_mean"] = None

    assert any("psnr_mean is null" in f for f in check_run(summary))


def test_psnr_over_zero_frames_is_flagged():
    summary = good_summary()
    summary["evaluation"]["psnr_num_frames"] = 0
    assert any("zero frames" in f for f in check_run(summary))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0, 0.0])
def test_implausible_psnr_is_flagged(bad):
    summary = good_summary()
    summary["evaluation"]["psnr_mean"] = bad
    assert check_run(summary)


def test_a_missing_evaluation_block_is_flagged():
    assert check_run(good_summary(evaluation={}))


# --- payload accounting --------------------------------------------------------


def test_components_exceeding_the_total_are_flagged():
    """If the parts do not fit in the total, the size axis is measuring something else."""
    summary = good_summary()
    summary["evaluation"]["sizes_bytes"]["residual"] = 900_000

    assert any("more than the reported" in f for f in check_run(summary))


def test_small_overhead_is_tolerated():
    """Container overhead means the parts will not sum exactly; that is fine."""
    summary = good_summary()
    sizes = summary["evaluation"]["sizes_bytes"]
    sizes["residual"] = 305_000  # ~1% over the total once summed

    assert check_run(summary) == []


def test_a_missing_size_block_is_flagged():
    summary = good_summary()
    del summary["evaluation"]["sizes_bytes"]
    assert any("sizes_bytes is missing" in f for f in check_run(summary))


def test_a_zero_transport_total_is_flagged():
    summary = good_summary()
    summary["evaluation"]["sizes_bytes"]["transport_total"] = 0
    assert check_run(summary)


# --- the Residual Guarantee ----------------------------------------------------


def test_a_payload_larger_than_the_source_is_flagged():
    """The thesis: what we send must be smaller than what it replaces.

    A run above 1.0 has not disproved the approach — it has failed to
    demonstrate it, and must not be written up as a saving.
    """
    summary = good_summary()
    summary["evaluation"]["sizes_bytes"]["transport_to_source_ratio"] = 1.3

    assert any("shows no saving" in f for f in check_run(summary))


def test_a_payload_smaller_than_the_source_passes():
    summary = good_summary()
    summary["evaluation"]["sizes_bytes"]["transport_to_source_ratio"] = 0.85
    assert check_run(summary) == []


def test_a_missing_ratio_is_not_judged():
    """Unevaluable is not the same as failing."""
    summary = good_summary()
    del summary["evaluation"]["sizes_bytes"]["transport_to_source_ratio"]
    assert check_run(summary) == []
