"""Heavy metric backends. Deselected unless ``pytest -m integration``."""

from __future__ import annotations

import numpy as np
import pytest

from src.components.metrics.lpips import LpipsMetric
from src.components.metrics.vmaf import VmafMetric

pytestmark = pytest.mark.integration


def _clip(value: int, *, frames: int = 3, size: int = 64) -> np.ndarray:
    return np.full((frames, size, size, 3), value, dtype=np.uint8)


def test_lpips_is_calibrated_and_has_real_dynamic_range() -> None:
    """The defect this replaced could not tell a good match from an unrelated
    image: it scored 0.083 for unrelated against 0.085 for a good
    reconstruction. Assert the published anchors instead of merely ">0"."""
    pytest.importorskip("lpips")
    metric = LpipsMetric()
    rs = np.random.RandomState(0)
    ref = rs.randint(0, 255, (2, 128, 128, 3), dtype=np.uint8)
    unrelated = rs.randint(0, 255, (2, 128, 128, 3), dtype=np.uint8)
    mild = np.clip(
        ref.astype(np.int16) + rs.randint(-15, 15, ref.shape), 0, 255
    ).astype(np.uint8)

    assert metric.score(ref, ref) == pytest.approx(0.0, abs=1e-5)
    near = metric.score(ref, mild)
    far = metric.score(ref, unrelated)
    assert 0.0 < near < far, f"no dynamic range: mild={near}, unrelated={far}"
    assert far > 4 * near, f"unrelated only {far / max(near, 1e-9):.1f}x a mild perturbation"


def test_vmaf_libvmaf_on_synthetic_frames() -> None:
    metric = VmafMetric()
    ref = _clip(120)
    try:
        identical = metric.score(ref, ref)
        degraded = metric.score(ref, _clip(40))
    except RuntimeError as exc:
        pytest.skip(str(exc))
    assert identical > degraded
    assert 0.0 <= degraded <= 100.0
    assert 0.0 <= identical <= 100.0
