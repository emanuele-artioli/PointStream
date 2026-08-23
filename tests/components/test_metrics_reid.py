"""The ReID identity metric: its contract, and the misuse that would go unnoticed.

No weights and no dataset here — the network is injected. What the real
backbone actually scores on real players is a calibration question and lives in
`tests/invariants/test_metric_calibration.py`, marked `integration`, because it
needs `assets/` which CI does not have.

Deliberately not tested: OSNet's own numerics (vendored third-party, copied
unmodified); exact cosine values as regression targets, which would pin a
backbone build rather than a behaviour; CPU/GPU agreement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.components.metrics.reid import ReidMetric, resolve_checkpoint
from src.contracts.metrics import Direction, metric


def _crop(height: int, width: int, value: int = 90) -> np.ndarray:
    frame = np.full((height, width, 3), value, dtype=np.uint8)
    return frame[None, ...]


class _Echo:
    """Stands in for the network. Returns the mean absolute difference of the
    two *resized* batches, so it can only work if the resize happened."""

    def __call__(self, reference: np.ndarray, predicted: np.ndarray) -> float:
        return float(np.abs(reference.astype(float) - predicted.astype(float)).mean())


def test_crops_of_different_sizes_are_compared_not_refused() -> None:
    """The defining property. Every other metric here demands pixel alignment;
    this one exists precisely to escape it, and a `paired()` call in the score
    path would silently re-impose it."""
    seen: dict[str, tuple[int, ...]] = {}

    def extractor(reference: np.ndarray, predicted: np.ndarray) -> float:
        seen["reference"] = reference.shape
        seen["predicted"] = predicted.shape
        return 0.5

    score = ReidMetric(extractor=extractor).score(_crop(400, 180), _crop(250, 300))
    assert score == 0.5
    assert seen["reference"][1:3] == (400, 180)
    assert seen["predicted"][1:3] == (250, 300)


def test_a_frame_count_mismatch_is_refused_with_both_counts() -> None:
    """Frame i is compared with frame i. Zipping unequal clips would score a
    shorter run against the wrong frames and return a plausible number."""
    left = np.zeros((3, 64, 32, 3), dtype=np.uint8)
    right = np.zeros((2, 64, 32, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="3 vs 2"):
        ReidMetric(extractor=lambda a, b: 0.0).score(left, right)


def test_missing_weights_name_the_file_and_refuse_to_download(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        resolve_checkpoint(tmp_path / "not-here.pth")
    message = str(excinfo.value)
    assert "not-here.pth" in message
    assert "kaiyangzhou/osnet" in message
    assert "downloads at runtime" in message


def test_reid_is_registered_as_higher_is_better() -> None:
    """Ranking code reads Direction rather than special-casing a name. Registering
    this one backwards would invert every identity comparison silently."""
    spec = metric("reid")
    assert spec.direction is Direction.HIGHER_IS_BETTER
    assert "floor" in spec.summary, "the summary must carry the measured floor"


def test_reid_refuses_a_mask_region_rather_than_scoring_a_hole() -> None:
    """A frame with a person-shaped hole is not something a ReID backbone has
    seen, and it would return a number anyway."""
    from src.components.metrics.evaluator import Evaluator
    from src.components.metrics.region import Region

    mask = np.zeros((64, 64), dtype=bool)
    mask[10:50, 10:50] = True
    clip = np.zeros((1, 64, 64, 3), dtype=np.uint8)
    evaluator = Evaluator(
        ["psnr", "reid"],
        backends={"psnr": _Psnr(), "reid": ReidMetric(extractor=lambda a, b: 0.9)},
    )
    with pytest.raises(ValueError, match="reid cannot score a mask"):
        evaluator.evaluate(clip, clip, regions=[Region.object(mask=mask, name="player")])


class _Psnr:
    def score(self, reference: np.ndarray, predicted: np.ndarray) -> float:
        del reference, predicted
        return 42.0


# ---------------------------------------------------------------------------
# The palette companion.


def test_palette_compares_crops_of_different_sizes() -> None:
    """A histogram is normalised by pixel count, so a crop and a rescaled copy
    of it are the same distribution. Demanding equal shapes would break that."""
    from src.components.metrics.palette import PaletteMetric

    small = np.zeros((40, 20, 3), dtype=np.uint8)
    small[:, :, 0] = 200
    large = np.zeros((400, 200, 3), dtype=np.uint8)
    large[:, :, 0] = 200
    assert PaletteMetric().score(small[None, ...], large[None, ...]) == pytest.approx(1.0)


def test_palette_ignores_letterbox_padding() -> None:
    """Padding is exactly black and is not part of the subject. A tall narrow
    crop is mostly pad on a square canvas, and counting it would make every
    letterboxed comparison look alike."""
    from src.components.metrics.palette import PaletteMetric

    subject = np.zeros((20, 20, 3), dtype=np.uint8)
    subject[:, :, 1] = 180
    padded = np.zeros((20, 60, 3), dtype=np.uint8)
    padded[:, 20:40, 1] = 180
    assert PaletteMetric().score(subject[None, ...], padded[None, ...]) == pytest.approx(1.0)
    assert PaletteMetric(mask_black=False).score(
        subject[None, ...], padded[None, ...]
    ) < 0.5


def test_palette_cannot_separate_two_subjects_in_the_same_colour() -> None:
    """Stated as a test because it is the companion's known blind spot, and a
    reader who forgets it will over-trust an agreement between the two."""
    from src.components.metrics.palette import PaletteMetric

    left = np.zeros((60, 30, 3), dtype=np.uint8)
    left[10:50, 5:25] = (200, 40, 90)
    right = np.zeros((60, 30, 3), dtype=np.uint8)
    right[5:55, 8:22] = (200, 40, 90)
    assert PaletteMetric().score(left[None, ...], right[None, ...]) > 0.9
