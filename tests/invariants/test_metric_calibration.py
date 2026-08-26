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

from typing import cast

import numpy as np
import pytest

from src.components.metrics.evaluator import MetricBackend

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

    metric = cast(MetricBackend, REGISTRY.build(name))
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

    metric = cast(MetricBackend, REGISTRY.build(name))
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

    metric = cast(MetricBackend, REGISTRY.build(name))
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
    metric = cast(MetricBackend, REGISTRY.build(name))
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


# ---------------------------------------------------------------------------
# BP27: BP23 tier anchors on real 4K content.
#
# Numbers from outputs/bp23-tier/metric-calibration.json and metric-notes.md.
# Pin them; do not re-derive. Tolerances are wide enough for a different
# ffmpeg/torch build and narrow enough to catch crossed VMAF inputs or an
# uncalibrated LPIPS stand-in.
#
# Deliberately not covered here: PSNR/SSIM tier anchors (BP23 already pins
# ordering in calibrate.py); LPIPS mild-blur absolute at 960×540; VMAF mild-blur
# absolute scale; any metric on synthetic procedural textures (covered above).
# ---------------------------------------------------------------------------

#: BP23 measured on 2 frames of alcaraz_highlights/scene_000 at 3840×2160.
_BP23_VMAF_IDENTICAL = 97.54
_BP23_VMAF_FLOOR = 0.0
_BP23_VMAF_TOLERANCE = 2.0

_BP23_LPIPS_4K: dict[str, float] = {
    "identical": 0.0,
    "mild-blur": 0.0171,
    "severe-blur": 0.2982,
    "unrelated-clip": 0.5493,
}
_BP23_LPIPS_TOLERANCE = 0.08

#: At 960×540 severe blur scores worse than unrelated — inversion from 4K.
_BP23_LPIPS_960_SEVERE = 0.613
_BP23_LPIPS_960_UNRELATED = 0.522
_BP23_LPIPS_960_TOLERANCE = 0.12


def _tier_reference_4k(n_frames: int = 2) -> np.ndarray:
    """First ``n_frames`` of the BP23 calibration clip at native 4K."""
    from experiments.tier.clip import ClipUnusable, load_tier_clip

    try:
        return load_tier_clip(n_frames=n_frames).frames
    except ClipUnusable as exc:
        pytest.skip(str(exc))


def _tier_anchor_table(reference: np.ndarray) -> dict[str, np.ndarray]:
    from experiments.tier.calibrate import _unrelated_frames, anchors

    table = anchors(reference)
    if "unrelated-clip" not in table:
        unrelated_4k = _unrelated_frames((reference.shape[0], 2160, 3840, 3))
        if unrelated_4k is None:
            pytest.skip("unrelated anchor frames unavailable at native 4K")
        height, width = int(reference.shape[1]), int(reference.shape[2])
        table["unrelated-clip"] = _downscale_frames(unrelated_4k, width, height)
    return table


def _downscale_frames(frames: np.ndarray, width: int, height: int) -> np.ndarray:
    import cv2

    return np.stack(
        [cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA) for frame in frames]
    )


def test_vmaf_tier_anchor_ceiling_is_below_100_on_identical_4k() -> None:
    """VMAF does not reach 100 on identical tier frames at 4K — ceiling is ~97.5."""
    from src.components.metrics import REGISTRY

    reference = _tier_reference_4k()
    metric = cast(MetricBackend, REGISTRY.build("vmaf"))
    try:
        score = metric.score(reference, reference)
    except RuntimeError as exc:
        pytest.skip(f"vmaf backend unavailable: {exc}")

    assert abs(score - _BP23_VMAF_IDENTICAL) <= _BP23_VMAF_TOLERANCE, (
        f"VMAF identical tier anchor is {score:.2f}, expected ~{_BP23_VMAF_IDENTICAL} "
        f"(±{_BP23_VMAF_TOLERANCE}). A score near 100 means crossed libvmaf inputs."
    )
    assert score < 99.0, (
        f"VMAF identical tier anchor is {score:.2f} — above 99, so the ceiling "
        "reads as 'almost perfect' rather than 'at the top of this host's scale'."
    )


def test_vmaf_tier_anchor_floors_at_zero_for_severe_and_unrelated_4k() -> None:
    """Severe blur and an unrelated clip both hit VMAF's floor on tier content."""
    from src.components.metrics import REGISTRY

    reference = _tier_reference_4k()
    table = _tier_anchor_table(reference)
    metric = cast(MetricBackend, REGISTRY.build("vmaf"))
    try:
        severe = metric.score(reference, table["severe-blur"])
        unrelated = metric.score(reference, table["unrelated-clip"])
    except RuntimeError as exc:
        pytest.skip(f"vmaf backend unavailable: {exc}")

    assert severe <= _BP23_VMAF_FLOOR + 1.0, (
        f"VMAF severe-blur tier anchor is {severe:.2f}, above the measured floor "
        f"({_BP23_VMAF_FLOOR}). The instrument no longer saturates at the bottom."
    )
    assert unrelated <= _BP23_VMAF_FLOOR + 1.0, (
        f"VMAF unrelated-clip tier anchor is {unrelated:.2f}, above the measured "
        f"floor ({_BP23_VMAF_FLOOR}). Quote unrelated beside any VMAF score."
    )


def test_lpips_tier_anchor_absolute_scale_at_4k() -> None:
    """LPIPS absolute scale on tier anchors at 4K — ordering alone is not enough."""
    from src.components.metrics import REGISTRY

    reference = _tier_reference_4k()
    table = _tier_anchor_table(reference)
    metric = cast(MetricBackend, REGISTRY.build("lpips"))
    failures: list[str] = []
    for name, expected in _BP23_LPIPS_4K.items():
        try:
            score = metric.score(reference, table[name])
        except RuntimeError as exc:
            pytest.skip(f"lpips backend unavailable: {exc}")
        if abs(score - expected) > _BP23_LPIPS_TOLERANCE:
            failures.append(f"{name}: got {score:.4f}, expected {expected:.4f}")
    assert not failures, (
        "LPIPS tier anchors at 4K drifted from BP23 calibration: "
        + "; ".join(failures)
        + f" (tolerance ±{_BP23_LPIPS_TOLERANCE})."
    )


def test_lpips_anchor_ordering_inverts_when_downscaled_to_960x540() -> None:
    """LPIPS anchors measured at 4K do not transfer to 960×540."""
    from src.components.metrics import REGISTRY

    reference = _downscale_frames(_tier_reference_4k(), 960, 540)
    table = _tier_anchor_table(reference)
    metric = cast(MetricBackend, REGISTRY.build("lpips"))
    try:
        severe = metric.score(reference, table["severe-blur"])
        unrelated = metric.score(reference, table["unrelated-clip"])
    except RuntimeError as exc:
        pytest.skip(f"lpips backend unavailable: {exc}")

    assert severe > unrelated, (
        f"At 960×540 LPIPS should invert: severe blur ({severe:.4f}) worse than "
        f"unrelated ({unrelated:.4f}). A 4K calibration does not transfer."
    )
    assert abs(severe - _BP23_LPIPS_960_SEVERE) <= _BP23_LPIPS_960_TOLERANCE, (
        f"960×540 severe-blur LPIPS is {severe:.4f}, expected "
        f"~{_BP23_LPIPS_960_SEVERE} (±{_BP23_LPIPS_960_TOLERANCE})."
    )
    assert abs(unrelated - _BP23_LPIPS_960_UNRELATED) <= _BP23_LPIPS_960_TOLERANCE, (
        f"960×540 unrelated-clip LPIPS is {unrelated:.4f}, expected "
        f"~{_BP23_LPIPS_960_UNRELATED} (±{_BP23_LPIPS_960_TOLERANCE})."
    )


# ---------------------------------------------------------------------------
# BP18: the identity metric, calibrated on real players.
#
# These need `assets/dataset` and the ReID weights, so they are marked
# `integration` and deselected by default. The synthetic anchors above cannot
# stand in: the question is whether the embedding separates two people who
# share a court, a broadcast and a lighting setup, and no procedural texture
# poses that question.
#
# Measured 2026-08-23 on 27 labelled tracks across three videos, cosine
# similarity, OSNet x1_0 / MSMT17:
#
#   identical crop                       1.0000
#   same track, offset 8   (n=27)        0.8506 +/- 0.0185   <- ground truth
#   same track, offset 24  (n=27)        0.8063 +/- 0.0193
#   same player, other track (n=42)      0.7200 +/- 0.0150   <- inferred label
#   DIFFERENT PLAYER, same match (n=52)  0.5097 +/- 0.0130
#   different video        (n=60)        0.4167 +/- 0.0136
#   player vs official     (n=14)        0.3943 +/- 0.0242
#
# Bounds in `outputs/bp18-reid-bounds.txt` were written first. Three of them
# fired high, and the reason is recorded there and in PLAN.md: cosine
# similarity on person crops has no natural zero, so "unrelated" sits near 0.42
# rather than 0. The bands were wrong, not the metric. Assertions below are
# therefore on ORDERING and on the separation against the MEASURED floor.

_REID_ANCHOR_ORDER = (
    "identical",
    "same track, offset 8",
    "same player, different track",
    "different player, same match",
    "different video",
)

#: Below this the metric cannot tell two players in one match apart, and must
#: not be used to rank anything however well-ordered its output looks.
REID_MIN_SEPARATION = 0.25


@pytest.mark.integration
def test_reid_separates_two_people_on_ground_truth_pairs() -> None:
    """The gate that decides whether this instrument may be used at all.

    **Both sides are ground truth, with nobody's judgement in them.** Same
    person is one track at two frames — a track is one individual by
    construction of the tracker. Different people are two tracks visible in the
    *same source frame*, which cannot be the same individual, scored at that
    shared frame so the camera and the light are held fixed too.

    A metric that scores two different people as highly as one person twice
    still produces a perfectly ordered ranking. That is how the uncalibrated
    LPIPS shipped, so this asserts a *magnitude*, not an order.
    """
    from src.components.metrics.reid import ReidMetric

    same, different = _ground_truth_pair_scores(ReidMetric(device="cpu"))
    separation = same - different
    assert separation >= REID_MIN_SEPARATION, (
        f"reid separates same-person ({same:.4f}) from different-person "
        f"({different:.4f}) by only {separation:.4f}, below the "
        f"{REID_MIN_SEPARATION} gate. Do not rank anything on this metric "
        "until that is explained."
    )


#: Pairs per side. The full sweep is 106 same and 53 different across seven
#: videos and blew a 15-minute timeout on CPU — an invariant nobody can afford
#: to run is one that rots. At this cap the four tests here take **~4.5 minutes
#: on CPU**, which is why they carry the `integration` marker and are deselected
#: by default rather than being made faster still. The standard error is ~0.03
#: against a separation of ~0.33, so the gate cannot be decided by the sample
#: size. Full-scale figures: `outputs/bp18-reid-calibration.txt`.
MAX_PAIRS_PER_SIDE = 24


def _ground_truth_pair_scores(backend: MetricBackend) -> tuple[float, float]:
    """``(same person, different person)`` means, derived, never labelled.

    Capped and taken in a deterministic order, so two runs agree.
    """
    from pathlib import Path

    from PIL import Image

    from experiments.probe.player_labels import (
        cooccurring_pairs,
        crop_index_of,
        track_frame_ids,
    )

    root = Path("assets") / "dataset"
    if not root.is_dir():
        pytest.skip("assets/dataset is not present")
    videos = sorted(item.name for item in root.iterdir() if item.is_dir())
    cache: dict[tuple[str, str, int], np.ndarray] = {}

    def crop(video: str, key: str, index: int) -> np.ndarray | None:
        cache_key = (video, key, index)
        if cache_key not in cache:
            scene, track = key.split("/")
            found = sorted((root / video / "segmentations" / scene / track).glob("frame_*.png"))
            if not found:
                return None
            chosen = found[min(index, len(found) - 1)]
            cache[cache_key] = np.asarray(Image.open(chosen).convert("RGB"))[None, ...]
        return cache[cache_key]

    different: list[float] = []
    same: list[float] = []
    seen: set[tuple[str, str]] = set()
    for video in videos:
        if len(different) >= MAX_PAIRS_PER_SIDE and len(same) >= MAX_PAIRS_PER_SIDE:
            break
        for left, right, frame_id in cooccurring_pairs(video):
            if len(different) >= MAX_PAIRS_PER_SIDE and len(same) >= MAX_PAIRS_PER_SIDE:
                break
            left_index = crop_index_of(video, left, frame_id)
            right_index = crop_index_of(video, right, frame_id)
            if left_index is None or right_index is None:
                continue
            first, second = crop(video, left, left_index), crop(video, right, right_index)
            if first is None or second is None:
                continue
            if len(different) < MAX_PAIRS_PER_SIDE:
                try:
                    different.append(float(backend.score(first, second)))
                except FileNotFoundError as exc:  # weights absent
                    pytest.skip(str(exc))
            for key in (left, right):
                if (video, key) in seen or len(same) >= MAX_PAIRS_PER_SIDE:
                    continue
                seen.add((video, key))
                scene, track = key.split("/")
                if len(track_frame_ids(video, scene, track)) > 8:
                    start, later = crop(video, key, 0), crop(video, key, 8)
                    if start is not None and later is not None:
                        same.append(float(backend.score(start, later)))
    if len(different) < 8 or len(same) < 8:
        pytest.skip("too few ground-truth pairs to calibrate on")
    return sum(same) / len(same), sum(different) / len(different)


@pytest.mark.integration
def test_reid_anchors_are_ordered_from_identical_down_to_unrelated() -> None:
    scores = _reid_anchor_scores()
    values = [scores[name] for name in _REID_ANCHOR_ORDER]
    for first, second, left, right in zip(
        _REID_ANCHOR_ORDER, _REID_ANCHOR_ORDER[1:], values, values[1:]
    ):
        assert left > right, (
            f"ReID scores '{second}' ({right:.4f}) at least as high as '{first}' "
            f"({left:.4f}). Full curve: {scores}"
        )


@pytest.mark.integration
def test_reid_is_reported_against_a_measured_floor_not_against_zero() -> None:
    """Cosine similarity on person crops has no natural zero: every upright
    human in a tennis crop shares a large component. Quoting 0.51 as though 0
    were the floor overstates the distance by a factor of two."""
    scores = _reid_anchor_scores()
    floor = scores["different video"]
    assert 0.25 < floor < 0.60, (
        f"the unrelated-clip floor moved to {floor:.4f}; the numbers quoted in "
        "PLAN.md and in the metric summary are anchored on ~0.42 and need re-stating"
    )


def _reid_anchor_scores() -> dict[str, float]:
    """Mean similarity per anchor, computed from the hand labels."""
    from src.components.metrics.reid import ReidMetric

    return _anchor_scores(ReidMetric(device="cpu"))


def _anchor_scores(backend: MetricBackend) -> dict[str, float]:
    """Every anchor for one backend. Shared so the two are compared like for like."""
    from pathlib import Path

    from PIL import Image

    from experiments.probe.player_labels import (
        OFFICIAL,
        PLAYER_LABELS,
        different_player_pairs,
        labelled_tracks,
        same_player_pairs,
    )

    root = Path("assets") / "dataset"
    if not root.is_dir():
        pytest.skip("assets/dataset is not present")
    cache: dict[tuple[str, str, int], np.ndarray] = {}

    def frames(video: str, key: str) -> list[Path]:
        scene, track = key.split("/")
        return sorted((root / video / "segmentations" / scene / track).glob("frame_*.png"))

    def crop(video: str, key: str, index: int = 0) -> np.ndarray:
        cache_key = (video, key, index)
        if cache_key not in cache:
            found = frames(video, key)
            if not found:
                pytest.skip(f"no frames for {video}/{key}")
            chosen = found[min(index, len(found) - 1)]
            cache[cache_key] = np.asarray(Image.open(chosen).convert("RGB"))[None, ...]
        return cache[cache_key]

    def mean(values: list[float]) -> float:
        if not values:
            pytest.skip("an anchor had no usable pairs")
        return sum(values) / len(values)

    keys = [(video, key) for video in PLAYER_LABELS for key in labelled_tracks(video)]
    try:
        identical = mean([backend.score(crop(v, k), crop(v, k)) for v, k in keys])
    except FileNotFoundError as exc:  # weights absent
        pytest.skip(str(exc))
    offset = mean(
        [
            backend.score(crop(v, k, 0), crop(v, k, 8))
            for v, k in keys
            if len(frames(v, k)) > 8
        ]
    )
    same = mean(
        [
            backend.score(crop(v, a), crop(v, b))
            for v in PLAYER_LABELS
            for a, b in same_player_pairs(v)
        ]
    )
    different = mean(
        [
            backend.score(crop(v, a), crop(v, b))
            for v in PLAYER_LABELS
            for a, b in different_player_pairs(v)
        ]
    )
    across = mean(
        [
            backend.score(crop(*left), crop(*right))
            for index, left in enumerate(keys)
            for right in keys[index + 1 :]
            if left[0] != right[0]
        ][:60]
    )
    official = mean(
        [
            backend.score(crop(v, a), crop(v, b))
            for v in PLAYER_LABELS
            for a, left_label in labelled_tracks(v).items()
            for b, right_label in labelled_tracks(v).items()
            if left_label == OFFICIAL and right_label != OFFICIAL
        ]
        or [float("nan")]
    )
    return {
        "identical": identical,
        "same track, offset 8": offset,
        "same player, different track": same,
        "different player, same match": different,
        "player vs official": official,
        "different video": across,
    }


@pytest.mark.integration
def test_the_palette_companion_disagrees_where_it_should() -> None:
    """The reason `palette` is registered at all.

    Kit colour is most of what separates two players here, so `reid` is partly
    a colour detector and a learned metric with nothing to check it against is
    how the uncalibrated LPIPS shipped. The check is only worth keeping if the
    two can disagree, and they do, in the direction each is built for:

    * an **official** in a black tracksuit shares colour mass with a
      dark-shirted player, so `palette` scores that pair *higher* than two
      players — while `reid` scores it *lower*, because an umpire is not doing
      the thing a player is doing;
    * measured 2026-08-23, `reid` 0.394 vs 0.510 and `palette` 0.502 vs 0.364.

    If this ever stops holding, the two metrics have collapsed into one and the
    companion is no longer buying anything.
    """
    reid = _reid_anchor_scores()
    palette = _palette_anchor_scores()
    assert reid["player vs official"] < reid["different player, same match"], (
        "reid no longer ranks an official below two players: "
        f"{reid['player vs official']:.4f} vs {reid['different player, same match']:.4f}"
    )
    assert palette["player vs official"] > palette["different player, same match"], (
        "palette no longer confuses an official's kit with a player's; the two "
        "metrics may have collapsed into one measurement"
    )


def _palette_anchor_scores() -> dict[str, float]:
    from src.components.metrics.palette import PaletteMetric

    return _anchor_scores(PaletteMetric())
