"""Heavy metric backends. Deselected unless ``pytest -m integration``."""

from __future__ import annotations

import numpy as np
import pytest

from src.components.metrics.lpips import LpipsMetric, default_weights_path
from src.components.metrics.vmaf import VmafMetric

pytestmark = pytest.mark.integration


def _clip(value: int, *, frames: int = 3, size: int = 64) -> np.ndarray:
    return np.full((frames, size, size, 3), value, dtype=np.uint8)


def test_lpips_vgg_on_synthetic_frames() -> None:
    if not default_weights_path().is_file():
        pytest.skip(f"VGG weights missing at {default_weights_path()}")
    metric = LpipsMetric()
    ref = _clip(120)
    pred = _clip(40)
    assert metric.score(ref, ref) == pytest.approx(0.0, abs=1e-5)
    assert metric.score(ref, pred) > 0.0


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
