"""The runner binding that turns `evaluation.metrics` into scores that exist.

Two silent wrong answers are what this file is for.

**The negation.** `Region.background(mask=m)` in the components layer scores
everything *except* `m`; the pipeline protocol's `background_mask` names the
pixels *to* score. Get that backwards and every run still reports an object and
a background number — swapped. Nothing downstream would notice, and the object
score would be the one the paper quotes.

**The rectangle.** VMAF, LPIPS and FVMD cannot score a frame with a
person-shaped hole in it. Substituting a bounding box would flatter a generated
player, which is the substitution `src/components/metrics/region.py` exists to
prevent. So those metrics score the whole frame and say so.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.contracts.config import EvaluationConfig, PointstreamConfig
from src.pipeline.reconstruction.quality import (
    ROLE_BACKGROUND,
    ROLE_FRAME,
    ROLE_OBJECT,
)
from src.runner.evaluation import ComponentMetricEvaluator, evaluator_for


def _config(*metrics: str) -> PointstreamConfig:
    return PointstreamConfig(evaluation=EvaluationConfig(metrics=tuple(metrics)))


def _split_clip():
    """Left half is the object, right half the background, in a 3-frame clip."""
    reference = np.zeros((3, 32, 64, 3), dtype=np.uint8)
    reference[:, :, :32] = 120
    reference[:, :, 32:] = 200
    mask = np.zeros((3, 32, 64), dtype=bool)
    mask[:, :, :32] = True
    return reference, mask


def test_every_config_gets_the_same_psnr_convention() -> None:
    """A PSNR-only config must not be scored by a different implementation.

    The pipeline floor pools the MSE across the clip; the components metric
    averages per-frame PSNRs. Both are defensible and they are not equal — on
    one 4K clip they read 47.63 dB and 48.28 dB on the same pixels. Binding the
    floor for cheap configs and the registry for rich ones would make a tier
    ladder measure its evaluator.
    """
    floor = evaluator_for(_config("psnr"))
    rich = evaluator_for(_config("psnr", "ssim"))
    assert isinstance(floor, ComponentMetricEvaluator)
    assert isinstance(rich, ComponentMetricEvaluator)

    reference, _mask = _split_clip()
    predicted = reference.copy()
    predicted[:, :, :32] = 60
    assert floor.evaluate(reference, predicted).whole_frame() == pytest.approx(
        rich.evaluate(reference, predicted).whole_frame()
    )
    # And it is not the pooled convention, which is reported separately. The
    # two only separate when the per-frame error varies, so the clip has to
    # vary — on a uniform one they agree and the check would pass for free.
    uneven = reference.copy()
    for index, level in enumerate((2, 40, 90)):
        uneven[index, :, :32] = level
    report = floor.evaluate(reference, uneven)
    assert report.whole_frame() != pytest.approx(report.closeness.psnr, rel=1e-6)


def test_a_richer_metric_set_binds_the_components_registry() -> None:
    evaluator = evaluator_for(_config("psnr", "ssim"))
    assert isinstance(evaluator, ComponentMetricEvaluator)
    assert set(evaluator.metric_names) == {"psnr", "ssim"}


def test_requested_metrics_actually_appear_in_the_report() -> None:
    reference, mask = _split_clip()
    predicted = reference.copy()
    predicted[:, :, :32] = 60
    report = evaluator_for(_config("psnr", "ssim")).evaluate(
        reference, predicted, object_mask=mask
    )
    metrics = {item.metric for item in report.scoped}
    assert metrics == {"psnr", "ssim"}


def test_object_and_background_regions_are_not_swapped() -> None:
    """Destroy only the object half. The object score must be the bad one.

    Anchors, so the numbers are readable rather than merely ordered: an
    untouched region scores infinite PSNR / SSIM 1.0, and a region driven from
    120 to 60 scores about 12.5 dB.
    """
    reference, mask = _split_clip()
    predicted = reference.copy()
    predicted[:, :, :32] = 60
    report = evaluator_for(_config("psnr", "ssim")).evaluate(
        reference, predicted, object_mask=mask
    )
    obj = {item.metric: item.value for item in report.for_role(ROLE_OBJECT)}
    background = {item.metric: item.value for item in report.for_role(ROLE_BACKGROUND)}
    assert obj and background
    assert math.isinf(background["psnr"])
    assert 10.0 <= obj["psnr"] <= 15.0
    assert background["ssim"] == pytest.approx(1.0, abs=1e-6)
    assert obj["ssim"] < background["ssim"]


def test_a_supplied_background_mask_names_the_pixels_to_score() -> None:
    """The other direction of the same negation, driven explicitly."""
    reference, mask = _split_clip()
    predicted = reference.copy()
    predicted[:, :, :32] = 60
    report = evaluator_for(_config("psnr", "ssim")).evaluate(
        reference,
        predicted,
        object_mask=mask,
        background_mask=np.logical_not(mask),
    )
    background = {item.metric: item.value for item in report.for_role(ROLE_BACKGROUND)}
    assert math.isinf(background["psnr"])


def test_a_rectangular_metric_scores_the_frame_and_says_it_skipped_the_regions() -> None:
    """No box substitution, and no silent omission either."""
    evaluator = ComponentMetricEvaluator(["psnr", "vmaf"])
    assert evaluator.skipped_on_regions == ("vmaf",)
    assert evaluator.region_metric_names == ("psnr",)


def test_bit_identity_survives_a_richer_metric_set() -> None:
    """Closeness is a pixel comparison, not an inference from a metric value.

    A `QualityReport` whose `bit_identical` came from "PSNR returned infinity"
    would be true for two clips that differ nowhere a metric looks.
    """
    reference, mask = _split_clip()
    report = evaluator_for(_config("psnr", "ssim")).evaluate(
        reference, reference.copy(), object_mask=mask
    )
    assert report.bit_identical
    assert math.isinf(report.whole_frame())
    assert report.for_role(ROLE_FRAME)
