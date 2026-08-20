"""The metrics contract: measurement is mandatory, and direction is declared.

The behaviour under test is architectural, not cosmetic. §2.3 of the plan makes
quality measurement a property of every configuration, because the residual
always quantizes and generative inference is statistical — so a run that
measured nothing has verified nothing. The arrangement being replaced accepted
`metrics: none` and returned an empty list.
"""

from __future__ import annotations

import pytest

from src.contracts.errors import ConfigValueError, UnknownBackendError
from src.contracts.metrics import (
    ALWAYS_ON,
    METRICS,
    MetricCost,
    MetricSelection,
    MetricTier,
    by_tier,
    metric,
    resolve,
    resolve_tiers,
)

# --------------------------------------------------------------------------
# The always-on rule
# --------------------------------------------------------------------------


def test_the_default_selection_is_the_floor_and_nothing_else() -> None:
    """Development runs pay for PSNR only, and still report a number."""
    assert resolve(None).names() == ("psnr",)


def test_psnr_is_added_to_any_selection_that_leaves_it_out() -> None:
    """And is reported as enforced, so a summary can say why it is there."""
    selection = resolve(["vmaf", "lpips"])

    assert "psnr" in selection
    assert selection.enforced == ("psnr",)
    assert "* required in every configuration" in selection.describe()


def test_disabling_measurement_entirely_is_rejected() -> None:
    """`none` was legal before, which is how a run could report no quality at all."""
    with pytest.raises(ConfigValueError, match="mandatory"):
        resolve("none")

    with pytest.raises(ConfigValueError, match="mandatory"):
        resolve([])


def test_a_selection_built_by_hand_without_psnr_is_refused() -> None:
    """The invariant lives on the type, not only on the function that builds it."""
    with pytest.raises(ValueError, match="mandatory"):
        MetricSelection(metrics=(METRICS["vmaf"],))


def test_always_on_names_are_all_registered() -> None:
    assert ALWAYS_ON <= set(METRICS)


# --------------------------------------------------------------------------
# Naming and lookup
# --------------------------------------------------------------------------


def test_an_unknown_metric_is_rejected_with_the_legal_set() -> None:
    with pytest.raises(UnknownBackendError) as caught:
        resolve(["psnr", "vmaff"])

    message = str(caught.value)
    assert "vmaff" in message
    assert "vmaf" in message  # the close-match suggestion
    assert "lpips" in message  # the full legal set


def test_metric_lookup_rejects_an_unknown_name() -> None:
    with pytest.raises(UnknownBackendError):
        metric("psnr_y")


def test_a_comma_separated_string_resolves_like_a_list() -> None:
    """Config files carry both spellings."""
    assert resolve("psnr, vmaf").names() == resolve(["psnr", "vmaf"]).names()


# --------------------------------------------------------------------------
# Tiers
# --------------------------------------------------------------------------


def test_the_tiers_are_the_ones_the_plan_specifies() -> None:
    assert {spec.name for spec in by_tier(MetricTier.FAST)} == {"psnr"}
    assert {spec.name for spec in by_tier(MetricTier.TRADITIONAL)} == {"ssim", "vmaf"}
    assert {spec.name for spec in by_tier(MetricTier.PERCEPTUAL)} == {"lpips"}
    assert {spec.name for spec in by_tier(MetricTier.TEMPORAL)} == {"fvmd"}
    # FVMD rather than FVD: the reviewer question is about temporal coherence
    # specifically, and the existing FVD wiring is prior art, not the default.
    assert "fvd" not in METRICS


def test_resolving_by_tier_takes_the_whole_tier() -> None:
    selection = resolve_tiers(["fast", MetricTier.TRADITIONAL])
    assert set(selection.names()) == {"psnr", "ssim", "vmaf"}


def test_an_unknown_tier_is_rejected() -> None:
    with pytest.raises(ConfigValueError, match="unknown metric tier"):
        resolve_tiers(["cheap"])


def test_selecting_no_tiers_is_rejected() -> None:
    with pytest.raises(ConfigValueError, match="mandatory"):
        resolve_tiers([])


def test_cheap_metrics_run_first() -> None:
    """A run killed partway through should still have its PSNR."""
    names = resolve(["fvmd", "lpips", "vmaf"]).names()
    assert names[0] == "psnr"
    assert names.index("vmaf") < names.index("lpips") < names.index("fvmd")


# --------------------------------------------------------------------------
# Direction, scope and bounds
# --------------------------------------------------------------------------


def test_ranking_follows_the_declared_direction() -> None:
    """So comparison code never has to remember that LPIPS is lower-better."""
    assert metric("psnr").is_better(35.0, 30.0)
    assert not metric("psnr").is_better(30.0, 35.0)
    assert metric("lpips").is_better(0.10, 0.22)
    assert not metric("lpips").is_better(0.22, 0.10)

    assert metric("psnr").best([30.0, 35.0, 31.0]) == 35.0
    assert metric("lpips").best([0.30, 0.10, 0.22]) == 0.10


def test_temporal_metrics_are_separated_from_per_frame_ones() -> None:
    """A temporal metric handed single frames measures nothing meaningful."""
    selection = resolve(["vmaf", "fvmd"])

    assert [spec.name for spec in selection.temporal] == ["fvmd"]
    assert "fvmd" not in [spec.name for spec in selection.per_frame]
    assert selection.max_cost is MetricCost.HEAVY


def test_out_of_range_scores_are_recognisable_as_alarms() -> None:
    """A VMAF of 118 is a misconfigured comparison, not a good result."""
    assert metric("vmaf").in_range(87.4)
    assert not metric("vmaf").in_range(118.0)
    assert not metric("ssim").in_range(-0.2)
    # FVMD declares no bounds, so nothing is out of range for it.
    assert metric("fvmd").in_range(9_999.0)
