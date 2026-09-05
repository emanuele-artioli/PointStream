"""Gate 2 controls: metric anchors, temporal null, object-stream-off, and fallback control.

Brief specifications:
- Run identical, mild, severe and unrelated metric anchors on the same working resolution.
- Required order: identical > mild > severe and mild > unrelated for VMAF/Y-PSNR/SSIM.
- VMAF identical must be in [95, 99] and unrelated in [0, 40].
- Shuffled-frame temporal null with full-frame scores reported.
- PointStream object-stream-off in the same session.
- Conventional fallback against matching anchor:
    rate ratio in [0.95, 1.05] and absolute VMAF difference <= 1.0.
- A control failure stops before curve ranking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tier.bp52_background_search import run_metric_controls
from experiments.tier.low_rate_measure import score_headlines


def verify_metric_anchors(reference: np.ndarray, destination: Path) -> dict[str, Any]:
    """Run native-resolution calibration fixtures and assert ordering & absolute scales."""
    result = run_metric_controls(reference, destination)
    if not result.get("valid", False):
        alarms = result.get("alarms", [])
        raise SystemExit(f"Gate 2 metric anchor controls failed: {alarms}")
    return result


def run_temporal_null(reference: np.ndarray, seed: int = 42) -> dict[str, Any]:
    """Shuffled-frame temporal null control.

    Permutes frame order along the time axis and scores against the original sequence.
    Demonstrates the floor for temporal coherence.
    """
    count = int(reference.shape[0])
    if count < 2:
        raise ValueError("temporal null requires at least 2 frames")

    rng = np.random.default_rng(seed)
    # Ensure permutation is not identity
    indices = rng.permutation(count)
    if np.array_equal(indices, np.arange(count)):
        indices[0], indices[1] = indices[1], indices[0]

    shuffled = reference[indices]
    scores = score_headlines(reference, shuffled)
    return {
        "control": "temporal_null_shuffled_frames",
        "frame_count": count,
        "seed": seed,
        "scores": scores,
    }


def verify_conventional_fallback(
    *,
    fallback_bytes: int,
    fallback_vmaf: float,
    anchor_bytes: int,
    anchor_vmaf: float,
) -> dict[str, Any]:
    """Check conventional fallback against matching anchor.

    Requires:
    - rate ratio [0.95, 1.05]
    - absolute VMAF difference <= 1.0
    """
    if anchor_bytes <= 0:
        raise ValueError("anchor_bytes must be positive")

    rate_ratio = float(fallback_bytes) / float(anchor_bytes)
    vmaf_diff = abs(float(fallback_vmaf) - float(anchor_vmaf))

    rate_ok = 0.95 <= rate_ratio <= 1.05
    vmaf_ok = vmaf_diff <= 1.0

    result = {
        "control": "conventional_fallback",
        "fallback_bytes": fallback_bytes,
        "anchor_bytes": anchor_bytes,
        "rate_ratio": round(rate_ratio, 4),
        "rate_ratio_in_bounds": rate_ok,
        "fallback_vmaf": round(fallback_vmaf, 3),
        "anchor_vmaf": round(anchor_vmaf, 3),
        "vmaf_difference": round(vmaf_diff, 3),
        "vmaf_difference_in_bounds": vmaf_ok,
        "passed": rate_ok and vmaf_ok,
    }

    if not result["passed"]:
        alarms: list[str] = []
        if not rate_ok:
            alarms.append(f"rate ratio {rate_ratio:.4f} outside [0.95, 1.05]")
        if not vmaf_ok:
            alarms.append(f"vmaf difference {vmaf_diff:.3f} > 1.0")
        raise SystemExit(f"Gate 2 conventional fallback control failed: {alarms}")

    return result


def run_gate_a_controls(
    reference: np.ndarray,
    destination: Path,
) -> dict[str, Any]:
    """Execute all pre-run controls and return combined control record."""
    destination.mkdir(parents=True, exist_ok=True)
    anchors_dest = destination / "metric-controls.json"

    anchors_result = verify_metric_anchors(reference[:2], anchors_dest)
    temporal_result = run_temporal_null(reference)

    combined = {
        "metric_anchors": anchors_result,
        "temporal_null": temporal_result,
        "valid": True,
    }
    (destination / "controls-summary.json").write_text(json.dumps(combined, indent=2) + "\n")
    return combined
