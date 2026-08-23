"""Pipeline-path quality metrics: compute, rank, enforce, compare.

Not tested here: ffmpeg libvmaf internals, VGG layer indices, Lucas–Kanade
numerics, third-party SSIM window edge handling. Those are either integration
(marked) or someone else's code.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.components.metrics import REGISTRY
from src.components.metrics.evaluator import Evaluator, MetricBackend
from src.components.metrics.fvmd import FvmdMetric, frechet_distance
from src.components.metrics.identity import bit_identical, close
from src.components.metrics.lpips import LpipsMetric
from src.components.metrics.psnr import PsnrMetric
from src.components.metrics.ranking import rank
from src.components.metrics.ssim import SsimMetric
from src.components.metrics.vmaf import VmafMetric
from src.contracts.config import EvaluationConfig, PointstreamConfig, validate_backends
from src.contracts.errors import ConfigError, ConfigValueError
from src.contracts.metrics import METRICS as CONTRACT_METRICS
from src.contracts.metrics import metric as contract_metric
from src.contracts.registry import Registry


def _uniform_clip(value: float, *, frames: int = 2, size: int = 8) -> np.ndarray:
    return np.full((frames, size, size, 3), value, dtype=np.uint8)


def _mean_color_extractor(reference: np.ndarray, predicted: np.ndarray) -> float:
    """Stand-in for the calibrated network: mean-colour distance, no torch."""
    left = reference.mean(axis=(1, 2)) / 255.0
    right = predicted.mean(axis=(1, 2)) / 255.0
    return float(np.abs(left - right).mean())


def _static_tracker(clip: np.ndarray) -> np.ndarray:
    points = np.array([[1.0, 1.0], [4.0, 1.0], [1.0, 4.0], [4.0, 4.0]])
    return np.broadcast_to(points, (clip.shape[0], 4, 2)).copy()


def _slide_tracker(clip: np.ndarray) -> np.ndarray:
    points = np.array([[1.0, 1.0], [4.0, 1.0], [1.0, 4.0], [4.0, 4.0]])
    return np.stack([points + np.array([float(t), 0.0]) for t in range(clip.shape[0])])


def _mock_vmaf(reference: np.ndarray, predicted: np.ndarray) -> float:
    if np.array_equal(reference, predicted):
        return 100.0
    return 40.0


# --------------------------------------------------------------------------
# Tiers on a synthetic pair
# --------------------------------------------------------------------------


def test_psnr_of_uniform_patches_matches_the_closed_form() -> None:
    """Bound: identical → inf (best). 100-vs-0 → 10*log10(255^2/10000) ≈ 8.1 dB (worse).

    100 vs 50 is in between: MSE=2500, PSNR=10*log10(65025/2500) ≈ 14.151 dB.
    """
    ref = _uniform_clip(100)
    pred = _uniform_clip(50)
    value = PsnrMetric().score(ref, pred)
    expected = 10.0 * math.log10((255.0**2) / 2500.0)
    assert value == pytest.approx(expected, rel=1e-12)
    assert math.isinf(PsnrMetric().score(ref, ref))


def test_ssim_of_uniform_patches_matches_the_closed_form() -> None:
    """Global SSIM on constants: variances vanish, C2 cancels.

    Bound: identical → 1. 100 vs 50 → (2*100*50+C1)/(100^2+50^2+C1) with
    C1=(0.01*255)^2 = 6.5025, so 10006.5025/12506.5025 ≈ 0.8001.
    """
    ref = _uniform_clip(100)
    pred = _uniform_clip(50)
    c1 = (0.01 * 255.0) ** 2
    expected = (2.0 * 100.0 * 50.0 + c1) / (100.0**2 + 50.0**2 + c1)
    assert SsimMetric().score(ref, pred) == pytest.approx(expected, rel=1e-12)
    assert SsimMetric().score(ref, ref) == pytest.approx(1.0)


def test_every_tier_scores_a_synthetic_clip_on_the_pipeline_path() -> None:
    """LPIPS is scored here, on frames, not only inside checkpoint evaluation."""
    ref = _uniform_clip(120, frames=3)
    pred = _uniform_clip(80, frames=3)
    backends: dict[str, MetricBackend] = {
        "vmaf": VmafMetric(model=_mock_vmaf),
        "lpips": LpipsMetric(extractor=_mean_color_extractor),
        "fvmd": FvmdMetric(tracker=_slide_tracker),
    }
    record = Evaluator(
        ["psnr", "ssim", "vmaf", "lpips", "fvmd"],
        backends=backends,
    ).evaluate(ref, pred)

    assert set(record.scores) == {"psnr", "ssim", "vmaf", "lpips", "fvmd"}
    assert record.scores["psnr"] < PsnrMetric().score(ref, ref)
    assert 0.0 < record.scores["ssim"] < 1.0
    assert record.scores["vmaf"] == 40.0
    assert record.scores["lpips"] > 0.0
    assert record.scores["fvmd"] == pytest.approx(0.0, abs=1e-9)
    assert record.enforced == ()


def test_lpips_is_zero_for_identical_frames_and_grows_with_color_shift() -> None:
    """Bound: identical → 0. A 40/255 mean shift → 40/255 ≈ 0.157 under the
    injected mean-colour stand-in, which reports absolute difference."""
    ref = _uniform_clip(120)
    pred = _uniform_clip(80)
    metric = LpipsMetric(extractor=_mean_color_extractor)
    expected = 40.0 / 255.0
    assert metric.score(ref, ref) == pytest.approx(0.0, abs=1e-12)
    assert metric.score(ref, pred) == pytest.approx(expected, rel=1e-12)


def test_fvmd_is_zero_for_matching_motion_and_positive_when_it_differs() -> None:
    clip = _uniform_clip(32, frames=3, size=16)
    matching = FvmdMetric(tracker=_slide_tracker).score(clip, clip)
    calls = {"n": 0}

    def mixed(clip_in: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        return _static_tracker(clip_in) if calls["n"] == 2 else _slide_tracker(clip_in)

    gap = FvmdMetric(tracker=mixed).score(clip, clip)
    assert matching == pytest.approx(0.0, abs=1e-9)
    assert gap > 0.0


def test_frechet_distance_of_a_pure_mean_shift_is_the_euclidean_norm() -> None:
    """Bound: identical Gaussians → 0. A shift of 3 on 1-D, zero covariance → 3.

    Worst case is unbounded; this is the closed form, so 2.9 or 3.1 would be a bug.
    """
    mu = np.array([0.0])
    sigma = np.array([[0.0]])
    assert frechet_distance(mu, sigma, mu, sigma) == pytest.approx(0.0, abs=1e-12)
    assert frechet_distance(mu, sigma, mu + 3.0, sigma) == pytest.approx(3.0, abs=1e-12)


def test_fvmd_refuses_a_single_frame() -> None:
    frame = _uniform_clip(8, frames=1)
    with pytest.raises(ValueError, match="temporal"):
        FvmdMetric(tracker=_static_tracker).score(frame, frame)


# --------------------------------------------------------------------------
# Empty set / PSNR enforcement
# --------------------------------------------------------------------------


def test_empty_metrics_are_refused() -> None:
    with pytest.raises(ConfigValueError, match="empty metric set"):
        Evaluator([])


def test_psnr_enforcement_is_visible_on_the_result_record() -> None:
    ref = _uniform_clip(100)
    pred = _uniform_clip(90)
    record = Evaluator(["ssim"]).evaluate(ref, pred)
    assert record.enforced == ("psnr",)
    assert "psnr" in record.scores
    assert "ssim" in record.scores
    assert "*" in record.describe()
    assert "required in every configuration" in record.describe()


def test_registry_names_match_the_contract_and_do_not_include_fvd() -> None:
    assert set(REGISTRY.names()) == set(CONTRACT_METRICS)
    assert "fvmd" in REGISTRY
    assert "fvd" not in REGISTRY
    psnr = REGISTRY.build("psnr")
    assert isinstance(psnr, PsnrMetric)
    assert psnr.score(_uniform_clip(10), _uniform_clip(10)) == math.inf


def test_validate_backends_accepts_every_contract_metric() -> None:
    config = PointstreamConfig(
        evaluation=EvaluationConfig(metrics=("psnr", "ssim", "vmaf", "lpips", "fvmd")),
    )
    validate_backends(config, registries={"metric": REGISTRY})


def test_validate_backends_rejects_a_metric_missing_from_the_component_registry() -> None:
    config = PointstreamConfig(evaluation=EvaluationConfig(metrics=("psnr",)))
    with pytest.raises(ConfigError, match="psnr"):
        validate_backends(config, registries={"metric": Registry("metric")})


def test_evaluator_from_config_uses_the_evaluation_block() -> None:
    config = PointstreamConfig(evaluation=EvaluationConfig(metrics=("psnr",)))
    evaluator = Evaluator.from_config(config)
    record = evaluator.evaluate(_uniform_clip(8), _uniform_clip(8))
    assert record.enforced == ()
    assert set(record.scores) == {"psnr"}


# --------------------------------------------------------------------------
# Ranking uses direction, never a name
# --------------------------------------------------------------------------


def test_ranking_follows_declared_direction_with_no_per_name_branch() -> None:
    psnr = contract_metric("psnr")
    lpips = contract_metric("lpips")
    assert rank({"a": 30.0, "b": 40.0, "c": 35.0}, psnr) == ("b", "c", "a")
    assert rank({"a": 0.40, "b": 0.10, "c": 0.22}, lpips) == ("b", "c", "a")


# --------------------------------------------------------------------------
# Identity vs closeness
# --------------------------------------------------------------------------


def test_bit_identity_holds_for_a_lossless_copy_and_fails_for_one_grey_level() -> None:
    src = _uniform_clip(64)
    assert bit_identical(src, src.copy())
    dirty = src.copy()
    dirty[0, 0, 0, 0] = 65
    assert not bit_identical(src, dirty)


def test_generative_closeness_does_not_require_bit_identity() -> None:
    """A noisy sampler is close, not identical. Asserting identity would fail
    for reasons that have nothing to do with correctness.
    """
    clean = np.full((2, 8, 8, 3), 128, dtype=np.uint8)
    noisy = clean.copy()
    noisy[0, 0, 0, :] = 129
    assert not bit_identical(clean, noisy)
    assert close(clean, noisy, atol=1.0)


# --------------------------------------------------------------------------
# Plausible misuse
# --------------------------------------------------------------------------


def test_mismatched_shapes_are_refused_rather_than_broadcast() -> None:
    ref = _uniform_clip(8, size=8)
    pred = _uniform_clip(8, size=4)
    with pytest.raises(ValueError, match="shape"):
        PsnrMetric().score(ref, pred)


def test_vmaf_without_a_model_or_ffmpeg_does_not_invent_a_number(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FFMPEG_BIN", raising=False)
    monkeypatch.setattr("src.components.metrics.vmaf.shutil.which", lambda _name: None)
    with pytest.raises(RuntimeError, match="libvmaf|model"):
        VmafMetric().score(_uniform_clip(8), _uniform_clip(8))


def test_lpips_identical_frames_score_zero_through_the_injected_extractor() -> None:
    metric = LpipsMetric(extractor=_mean_color_extractor)
    clip = _uniform_clip(8)
    assert metric.score(clip, clip) == pytest.approx(0.0, abs=1e-9)
