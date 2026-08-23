"""Every metric is calibrated against known anchors, not merely non-zero.

Two metrics shipped here that passed "identical scores well, degraded scores
badly" while measuring nothing usable:

  * ``lpips`` computed an uncalibrated VGG feature MSE. An unrelated image
    scored 0.083 against 0.085 for a good reconstruction — no dynamic range.
    An engine ranking was published on it.
  * ``vmaf`` had ffmpeg's ``libvmaf`` inputs crossed. A blurred clip scored
    100.0 against 97.4 for an identical one — blur beating a perfect match.

Both smoke tests were of the form ``identical > degraded``, and both passed.
That assertion is satisfied by an instrument with almost no range and by one
that is monotonic in the wrong direction over part of its domain.

So this file asserts the shape of the response curve: a *severe* degradation
must be clearly separated from a *mild* one, in the right direction. It is
deliberately an invariant rather than a unit test — it guards the instruments
every other measurement in the project is read through.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.invariants

_SIZE = 96  # above libvmaf's 32px floor


def _textured(seed: int) -> np.ndarray:
    """Structured content — gradients plus shapes, **not white noise**.

    Two white-noise images are perceptually *similar* to a learned metric: real
    LPIPS scores an unrelated noise pair at 0.13, where an unrelated natural
    pair sits near 0.6. An earlier version of this file used noise and its
    published-range anchor therefore failed against a correct metric.
    """
    import cv2

    rs = np.random.RandomState(seed)
    grid_y, grid_x = np.mgrid[0:_SIZE, 0:_SIZE].astype(np.float32) / _SIZE
    image = np.stack(
        [
            (0.5 + 0.5 * np.sin(6 * grid_x + seed)) * 255,
            (0.5 + 0.5 * np.cos(5 * grid_y + seed)) * 255,
            (0.5 + 0.5 * np.sin(4 * (grid_x + grid_y) + seed)) * 255,
        ],
        axis=-1,
    )
    for _ in range(6):
        centre = (int(rs.randint(0, _SIZE)), int(rs.randint(0, _SIZE)))
        radius = int(rs.randint(6, _SIZE // 4))
        colour = tuple(int(v) for v in rs.randint(0, 255, 3))
        cv2.circle(image, centre, radius, colour, -1)
    frame = np.clip(image, 0, 255).astype(np.uint8)
    return np.stack([frame, frame])


def _perturb(clip: np.ndarray, amount: int, seed: int = 3) -> np.ndarray:
    rs = np.random.RandomState(seed)
    noise = rs.randint(-amount, amount + 1, clip.shape)
    return np.clip(clip.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _blur(clip: np.ndarray, kernel: int) -> np.ndarray:
    import cv2

    return np.stack([cv2.GaussianBlur(frame, (kernel, kernel), 0) for frame in clip])


def _anchors() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """identical, mild, severe, unrelated.

    **Severe is BLUR, deliberately.** The crossed-input VMAF defect was
    monotonic under additive noise and only inverted under blur — an earlier
    version of this file used noise for both rungs, passed against the
    reintroduced bug, and would have shipped as a guard that guards nothing.
    Smoothing and noise are different failure directions for a structural
    metric, and a calibration set needs both.
    """
    reference = _textured(0)
    return reference, _perturb(reference, 8), _blur(reference, 21), _textured(99)


@pytest.mark.parametrize(
    ("name", "higher_is_better"),
    [("psnr", True), ("ssim", True), ("lpips", False), ("vmaf", True)],
)
def test_metric_orders_identical_above_mild_above_severe(
    name: str, higher_is_better: bool
) -> None:
    """identical → mild → severe must move one way.

    Note what is deliberately **not** asserted: an ordering between *severe* and
    *unrelated*. Those are not comparable in general — heavily blurring textured
    content drives it toward flat grey, which can sit further from the reference
    in feature space than an unrelated image does. An earlier version of this
    file demanded that chain and failed on correct metrics.
    """
    from src.components.metrics import REGISTRY

    metric = REGISTRY.build(name)
    reference, mild, severe, _unrelated = _anchors()
    try:
        scores = [
            metric.score(reference, reference),
            metric.score(reference, mild),
            metric.score(reference, severe),
        ]
    except RuntimeError as exc:  # a missing external tool is not a calibration failure
        pytest.skip(f"{name} backend unavailable: {exc}")

    labels = ("identical", "mild", "severe")
    ordered = [-value for value in scores] if higher_is_better else scores
    for i in range(len(ordered) - 1):
        assert ordered[i] <= ordered[i + 1], (
            f"{name} is not ordered from {labels[i]} to {labels[i + 1]}. "
            f"Curve: {dict(zip(labels, scores))}"
        )
    assert scores[0] != scores[2], f"{name} scores identical and severe the same"


@pytest.mark.parametrize(
    ("name", "higher_is_better"),
    [("psnr", True), ("ssim", True), ("lpips", False), ("vmaf", True)],
)
def test_metric_ranks_a_mild_perturbation_above_an_unrelated_image(
    name: str, higher_is_better: bool
) -> None:
    """The assertion the broken lpips failed: an unrelated image scored 0.083
    against 0.085 for a good reconstruction, so a ranking on it was noise."""
    from src.components.metrics import REGISTRY

    metric = REGISTRY.build(name)
    reference, mild, _severe, unrelated = _anchors()
    try:
        near = metric.score(reference, mild)
        far = metric.score(reference, unrelated)
    except RuntimeError as exc:
        pytest.skip(f"{name} backend unavailable: {exc}")

    better = near > far if higher_is_better else near < far
    assert better, f"{name}: mild={near} does not beat unrelated={far}"


@pytest.mark.parametrize("name", ["psnr", "ssim", "lpips", "vmaf"])
def test_metric_separates_a_mild_perturbation_from_an_unrelated_image(name: str) -> None:
    """The defect both broken metrics shared: no usable range between
    'slightly wrong' and 'completely wrong'. A ranking taken on such an
    instrument is noise."""
    from src.components.metrics import REGISTRY

    metric = REGISTRY.build(name)
    reference, mild, _severe, unrelated = _anchors()
    try:
        near = metric.score(reference, mild)
        far = metric.score(reference, unrelated)
        perfect = metric.score(reference, reference)
    except RuntimeError as exc:
        pytest.skip(f"{name} backend unavailable: {exc}")

    if np.isinf(perfect):  # PSNR: identical is infinite, compare the finite pair
        assert near > far, f"{name}: mild={near} not better than unrelated={far}"
        return

    span = abs(perfect - far)
    gap = abs(near - far)
    assert span > 0, f"{name}: identical and unrelated score the same ({perfect})"
    assert gap > 0.15 * span, (
        f"{name} cannot separate mild from unrelated: identical={perfect}, "
        f"mild={near}, unrelated={far}. The gap is {gap:.4f} against a total "
        f"span of {span:.4f}."
    )


# Published absolute anchors. A metric can be perfectly ordered and still be
# uninterpretable if its scale is wrong: the broken lpips scored 0.000 /
# 0.009 / 0.032 / 0.083 across identical / mild / blur / unrelated. That curve
# is monotonic and separates mild from unrelated by 89% of its span, so every
# ordering test above passes on it. What made it useless was the *scale* —
# 0.085 read as an excellent score while sitting at unrelated-image level, and
# an engine ranking was published on that reading.
#
# So anchor to the published range of the metric itself. These bounds are wide
# on purpose; they catch "wrong instrument", not "slightly different build".
_ABSOLUTE_ANCHORS = {
    # metric: (unrelated_min, unrelated_max) on textured natural-ish content
    # Measured on the structured content above with correct backends:
    # lpips 0.570, vmaf 0.00. Bounds are wide enough to survive a different
    # build and narrow enough to reject an uncalibrated stand-in (which put an
    # unrelated pair at 0.083).
    "lpips": (0.30, 1.20),
    "vmaf": (0.0, 40.0),
}


@pytest.mark.parametrize("name", sorted(_ABSOLUTE_ANCHORS))
def test_metric_absolute_scale_matches_its_published_range(name: str) -> None:
    """An unrelated image must land where the published metric puts one."""
    from src.components.metrics import REGISTRY

    low, high = _ABSOLUTE_ANCHORS[name]
    metric = REGISTRY.build(name)
    reference, _mild, _severe, unrelated = _anchors()
    try:
        far = metric.score(reference, unrelated)
    except RuntimeError as exc:
        pytest.skip(f"{name} backend unavailable: {exc}")

    assert low <= far <= high, (
        f"{name} scores an unrelated image at {far}, outside the published "
        f"range [{low}, {high}]. The scale is wrong even if the ordering is "
        f"not — which is exactly how the uncalibrated VGG stand-in shipped."
    )
