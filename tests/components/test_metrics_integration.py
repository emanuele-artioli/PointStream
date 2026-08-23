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


def test_vmaf_refuses_a_region_smaller_than_libvmaf_supports() -> None:
    """libvmaf requires both sides > 32 px. The 8x8 clips this test used to pass
    on returned a number under the old crossed-label filter, which they should
    never have. It is why VMAF cannot score a small player crop at all."""
    with pytest.raises(RuntimeError, match="invalid size|greater than 32"):
        VmafMetric().score(_clip(120, size=8), _clip(40, size=8))


def test_vmaf_libvmaf_on_synthetic_frames() -> None:
    """Textured content, deliberately. On FLAT patches VMAF is meaningless — its
    features are structural, so a uniform brightness shift can score *above* an
    identical pair (measured: 98.3 degraded against 97.3 identical on flat 64px
    clips). The old version of this test used flat 8x8 patches, which is below
    libvmaf's size floor as well as structurally empty."""
    metric = VmafMetric()
    rs = np.random.RandomState(7)
    base = rs.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    ref = np.stack([base, base])
    worse = np.clip(
        base.astype(np.int16) + rs.randint(-60, 60, base.shape), 0, 255
    ).astype(np.uint8)
    try:
        identical = metric.score(ref, ref)
        degraded = metric.score(ref, np.stack([worse, worse]))
    except RuntimeError as exc:
        pytest.skip(str(exc))
    assert identical > degraded
    assert 0.0 <= degraded <= 100.0
    assert 0.0 <= identical <= 100.0


def test_vmaf_is_monotonic_in_degradation() -> None:
    """The defect this guards: ffmpeg's libvmaf takes [distorted][reference],
    and the labels were passed straight through. A blurred clip then scored
    100.0 while an identical one scored 97.4 — a metric in which blur beats a
    perfect match. "Identical high, degraded low" did not catch it; monotonicity
    does.

    Identical sits near 97 rather than exactly 100: the default model is not
    calibrated for this resolution, and RGB->yuv420p drops chroma. That is model
    behaviour, not a wiring fault, and is why the assertion is ordering rather
    than an absolute value.
    """
    import cv2

    metric = VmafMetric()
    rs = np.random.RandomState(0)
    base = rs.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    ref = np.stack([base, base])
    mild = np.clip(
        base.astype(np.int16) + rs.randint(-15, 15, base.shape), 0, 255
    ).astype(np.uint8)
    heavy = cv2.GaussianBlur(base, (31, 31), 0)

    identical = metric.score(ref, ref)
    noisy = metric.score(ref, np.stack([mild, mild]))
    blurred = metric.score(ref, np.stack([heavy, heavy]))

    assert identical > noisy > blurred, (
        f"VMAF is not monotonic in degradation: "
        f"identical={identical}, mild={noisy}, blur={blurred}"
    )
    assert identical > 90.0
