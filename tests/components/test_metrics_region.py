"""Region-scoped scoring: the frame is not the object.

Behaviour
1. Perfect background + destroyed object → good frame PSNR, bad object PSNR.
2. Whole-frame is reported as well as scoped scores, never instead.
3. No region supplied → whole-frame score labelled as such.
4. Mask preferred over box; the record says which; they are not equal.
5. Region pixel count travels with the score.
6. triage() / Evaluator.triage() is PSNR-only.

Plausible misuse
7. Tiny region refused rather than reported as a result.
8. Mask spatial size mismatch refused (no silent resample).
9. Box that includes matching background flatters the object vs the mask.
10. VMAF/LPIPS/FVMD refuse a mask/background; they need a rectangular crop.

Deliberately not testing: libvmaf on crops, windowed-SSIM mask leakage, per-frame
box sequences. Those are either someone else's code or a later paper-path detail.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.components.metrics.evaluator import Evaluator, triage
from src.components.metrics.psnr import PsnrMetric
from src.components.metrics.region import MIN_REGION_PIXELS, Region, RegionKind, RegionRole
from src.components.metrics.vmaf import VmafMetric


# --------------------------------------------------------------------------
# Motivating fixture — bounds written before any score is read.
# --------------------------------------------------------------------------
#
# Frame 320x320, object 16x16 at (152, 152). Background 128, object 200,
# reconstruction zeros the object and leaves the background alone.
#
# Whole-frame PSNR. Best (alarm): inf — the object was ignored. Worst (alarm):
# ~2.1 dB — only the object was scored. Expected: MSE = (256/102400)*40000 = 100,
# PSNR = 10*log10(65025/100) ≈ 28.13 dB. Plausible band 20–40 dB.
#
# Object-scoped PSNR. Best (alarm): inf — the mask missed the damage. Worst:
# ~0 dB (255 vs 0). Expected: MSE = 40000, PSNR = 10*log10(65025/40000) ≈ 2.11 dB.
# Plausible band 0–10 dB.
#
# Background-scoped PSNR. Expected inf. Alarm if finite — the mask leaked.

_FRAME = 320
_OBJ = 16
_OBJ_XY = 152
_BG = 128
_FG = 200
_PEAK_SQ = 255.0**2


def _object_box() -> tuple[int, int, int, int]:
    return (_OBJ_XY, _OBJ_XY, _OBJ_XY + _OBJ, _OBJ_XY + _OBJ)


def _object_mask() -> np.ndarray:
    mask = np.zeros((_FRAME, _FRAME), dtype=bool)
    x1, y1, x2, y2 = _object_box()
    mask[y1:y2, x1:x2] = True
    return mask


def _reference(*, frames: int = 2) -> np.ndarray:
    clip = np.full((frames, _FRAME, _FRAME, 3), _BG, dtype=np.uint8)
    x1, y1, x2, y2 = _object_box()
    clip[:, y1:y2, x1:x2, :] = _FG
    return clip


def _destroyed(*, frames: int = 2) -> np.ndarray:
    clip = _reference(frames=frames)
    x1, y1, x2, y2 = _object_box()
    clip[:, y1:y2, x1:x2, :] = 0
    return clip


def _expected_frame_psnr() -> float:
    mse = (_OBJ * _OBJ / (_FRAME * _FRAME)) * float(_FG**2)
    return 10.0 * math.log10(_PEAK_SQ / mse)


def _expected_object_psnr() -> float:
    return 10.0 * math.log10(_PEAK_SQ / float(_FG**2))


def test_perfect_background_destroyed_object_is_good_frame_bad_object_psnr() -> None:
    """The point of this stream. Bounds: see module header, written first."""
    ref = _reference()
    pred = _destroyed()
    record = triage(ref, pred, regions=[Region.object(mask=_object_mask()), Region.background(mask=_object_mask())])

    frame = record.scores["psnr"]
    object_score = record.for_role("object")[0]
    background = record.for_role("background")[0]

    assert frame == pytest.approx(_expected_frame_psnr(), rel=1e-12)
    assert 20.0 < frame < 40.0
    assert object_score.value == pytest.approx(_expected_object_psnr(), rel=1e-12)
    assert 0.0 < object_score.value < 10.0
    assert frame > object_score.value + 10.0
    assert math.isinf(background.value)
    assert object_score.kind == "mask"
    assert background.kind == "mask"
    assert object_score.n_pixels == _OBJ * _OBJ
    assert background.n_pixels == _FRAME * _FRAME - _OBJ * _OBJ


def test_whole_frame_is_reported_as_well_never_instead() -> None:
    record = triage(
        _reference(),
        _destroyed(),
        regions=[Region.object(mask=_object_mask())],
    )
    roles = {item.role for item in record.scoped}
    assert roles == {"whole-frame", "object"}
    assert "psnr" in record.scores
    assert record.scores["psnr"] == record.for_role("whole-frame")[0].value
    assert record.for_role("whole-frame")[0].kind == "frame"


def test_no_region_is_labelled_whole_frame() -> None:
    record = triage(_reference(), _destroyed())
    assert len(record.scoped) == 1
    score = record.scoped[0]
    assert score.role == "whole-frame"
    assert score.kind == "frame"
    assert score.n_pixels == _FRAME * _FRAME
    assert score.metric == "psnr"
    assert score.value == pytest.approx(_expected_frame_psnr(), rel=1e-12)


def test_triage_path_is_psnr_only() -> None:
    evaluator = Evaluator.triage()
    assert evaluator.selection.names() == ("psnr",)
    record = triage(_reference(), _reference())
    assert set(record.scores) == {"psnr"}
    assert math.isinf(record.scores["psnr"])
    assert all(item.metric == "psnr" for item in record.scoped)


def test_mask_is_preferred_when_a_box_is_also_given() -> None:
    x1, y1, x2, y2 = _object_box()
    padded = (x1 - 4, y1 - 4, x2 + 4, y2 + 4)
    region = Region.object(mask=_object_mask(), box=padded)
    assert region.kind is RegionKind.MASK
    assert region.box is None
    record = triage(_reference(), _destroyed(), regions=[region])
    object_score = record.for_role("object")[0]
    assert object_score.kind == "mask"
    assert object_score.n_pixels == _OBJ * _OBJ


def test_box_includes_background_and_flatters_a_destroyed_object() -> None:
    """A box around a destroyed object contains matching background pixels.

    Mask PSNR ≈ 2.11 dB. Padded 24x24 box: MSE = (256/576)*40000 ≈ 17777.8,
    PSNR ≈ 5.63 dB. Frame ≈ 28.13 dB. Band: mask < box < frame, all finite.
    """
    x1, y1, x2, y2 = _object_box()
    padded = (x1 - 4, y1 - 4, x2 + 4, y2 + 4)
    record = triage(
        _reference(),
        _destroyed(),
        regions=[
            Region.object(mask=_object_mask()),
            Region.object(box=padded, name="padded-box"),
        ],
    )
    mask_score = next(item for item in record.for_role("object") if item.kind == "mask")
    box_score = next(item for item in record.for_role("object") if item.kind == "box")
    box_area = 24 * 24
    expected_box = 10.0 * math.log10(_PEAK_SQ / ((256 / box_area) * float(_FG**2)))
    assert mask_score.value == pytest.approx(_expected_object_psnr(), rel=1e-12)
    assert box_score.value == pytest.approx(expected_box, rel=1e-12)
    assert mask_score.value < box_score.value < record.scores["psnr"]
    assert box_score.n_pixels == box_area
    assert box_score.name == "padded-box"


def test_tiny_region_is_refused() -> None:
    mask = np.zeros((_FRAME, _FRAME), dtype=bool)
    mask[0:4, 0:4] = True
    assert int(mask.sum()) < MIN_REGION_PIXELS
    with pytest.raises(ValueError, match="small-sample artefact"):
        triage(_reference(), _destroyed(), regions=[Region.object(mask=mask)])


def test_mask_shape_mismatch_is_refused_not_resampled() -> None:
    small = np.ones((10, 10), dtype=bool)
    with pytest.raises(ValueError, match="resampling"):
        triage(_reference(), _destroyed(), regions=[Region.object(mask=small)])


def test_reference_prediction_shape_mismatch_is_still_a_bug() -> None:
    with pytest.raises(ValueError, match="shape"):
        triage(_reference(), np.zeros((2, 16, 16, 3), dtype=np.uint8))


def test_box_outside_the_frame_is_refused_not_clipped() -> None:
    with pytest.raises(ValueError, match="outside"):
        triage(
            _reference(),
            _destroyed(),
            regions=[Region.object(box=(0, 0, _FRAME + 1, 8))],
        )


def test_vmaf_refuses_a_mask_rather_than_scoring_a_flattering_crop() -> None:
    backends = {"vmaf": VmafMetric(model=lambda _ref, _pred: 100.0)}
    evaluator = Evaluator(["psnr", "vmaf"], backends=backends)
    with pytest.raises(ValueError, match="cannot score a mask"):
        evaluator.evaluate(
            _reference(),
            _destroyed(),
            regions=[Region.object(mask=_object_mask())],
        )


def test_object_box_is_reachable_for_a_rectangular_metric() -> None:
    seen: list[tuple[int, ...]] = []

    def capture(reference: np.ndarray, predicted: np.ndarray) -> float:
        seen.append(reference.shape)
        return 40.0

    backends = {"vmaf": VmafMetric(model=capture)}
    record = Evaluator(["psnr", "vmaf"], backends=backends).evaluate(
        _reference(),
        _destroyed(),
        regions=[Region.object(box=_object_box())],
    )
    vmaf_object = next(item for item in record.for_role("object") if item.metric == "vmaf")
    assert vmaf_object.kind == "box"
    assert vmaf_object.value == 40.0
    assert vmaf_object.n_pixels == _OBJ * _OBJ
    assert seen[-1] == (2, _OBJ, _OBJ, 3)


def test_region_role_is_object_or_background_not_a_bare_crop() -> None:
    region = Region.object(mask=_object_mask())
    assert region.role is RegionRole.OBJECT
    background = Region.background(mask=_object_mask())
    assert background.role is RegionRole.BACKGROUND
    mask = region.boolean_mask(2, _FRAME, _FRAME)
    complement = background.boolean_mask(2, _FRAME, _FRAME)
    assert bool(np.array_equal(mask, np.logical_not(complement)))


def test_masked_psnr_matches_whole_frame_when_the_mask_is_everything() -> None:
    full = np.ones((_FRAME, _FRAME), dtype=bool)
    ref = _reference()
    pred = _destroyed()
    whole = PsnrMetric().score(ref, pred)
    masked = PsnrMetric().score_masked(ref, pred, full)
    assert masked == pytest.approx(whole, rel=1e-12)
