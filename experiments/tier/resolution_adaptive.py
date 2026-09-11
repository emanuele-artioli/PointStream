"""Resolution-adaptive codec anchors and quality curves (EVAL-ACT-07).

Implements:
- Native resolution and resolution-adaptive curves (1.0, 0.5, 0.25 linear dimensions)
  for AV1 (SvtAv1EncApp) and VVC (vvencapp).
- Display grid restoration: decodes back and restores to original display resolution
  using fixed interpolation, scoring all arms on identical pixel grids.
- Rescaling time accounting: measures downscaling and upscaling wall time and includes
  rescaling time in declared encoder/client budgets.
- Extrapolation prohibition: refuses BD-rate evaluation outside measured common quality support.
- Residual-on high-fidelity ladder specification coordinating with Worker A's interface.
"""

from __future__ import annotations

import sqlite3  # noqa: F401
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from experiments.tier.low_rate_measure import reference_request, score_headlines
from src.components.background.scale import HEADER_BYTES
from src.components.codec.frames import even_size
from src.components.codec.measure import timed_roundtrip
from src.components.metrics.bd_rate import (
    DegenerateCurveError,
    InsufficientOverlapError,
    RDCurve,
    compare_rd_curves,
)
from src.contracts.config import PointstreamConfig
from src.contracts.metrics import metric as metric_spec


RESOLUTION_SCALES: tuple[float, ...] = (1.0, 0.5, 0.25)
DEFAULT_DISPLAY_INTERPOLATION: int = cv2.INTER_LANCZOS4
DEFAULT_DOWNSCALE_INTERPOLATION: int = cv2.INTER_AREA


def rescale_frames(
    frames: np.ndarray,
    scale: float,
    *,
    interpolation: int = DEFAULT_DOWNSCALE_INTERPOLATION,
) -> tuple[np.ndarray, float]:
    """Downscale frames to target linear scale, ensuring even dimensions for YUV420p.

    Returns:
        (rescaled_frames, wall_seconds)
    """
    if scale >= 1.0:
        return np.asarray(frames), 0.0
    if scale <= 0.0:
        raise ValueError(f"scale must be positive, got {scale}")

    T, H, W, C = frames.shape
    target_H = max(2, int(round(H * scale)) & ~1)
    target_W = max(2, int(round(W * scale)) & ~1)

    start = time.perf_counter()
    scaled = np.stack(
        [cv2.resize(f, (target_W, target_H), interpolation=interpolation) for f in frames],
        axis=0,
    )
    elapsed = time.perf_counter() - start
    return scaled, elapsed


def restore_to_display_grid(
    frames: np.ndarray,
    display_shape: tuple[int, int],
    *,
    interpolation: int = DEFAULT_DISPLAY_INTERPOLATION,
) -> tuple[np.ndarray, float]:
    """Upscale decoded frames to exact original display resolution (H, W).

    Returns:
        (restored_frames, wall_seconds)
    """
    display_H, display_W = display_shape
    T, H, W, C = frames.shape
    if (H, W) == (display_H, display_W):
        return np.asarray(frames), 0.0

    start = time.perf_counter()
    restored = np.stack(
        [cv2.resize(f, (display_W, display_H), interpolation=interpolation) for f in frames],
        axis=0,
    )
    elapsed = time.perf_counter() - start
    return restored, elapsed


def encode_resolution_arm(
    frames: np.ndarray,
    *,
    codec: str,
    qp: int,
    preset: str,
    scale: float = 1.0,
    fps: float = 24.0,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    """Encode an arm at native or rescaled dimension, restoring to display grid for scoring.

    Timing tracks:
    - encode_seconds
    - decode_seconds
    - downscale_seconds (if scale < 1.0)
    - upscale_seconds (if scale < 1.0)
    - client_seconds = decode_seconds + upscale_seconds
    - encoder_seconds = encode_seconds + downscale_seconds
    """
    source = even_size(np.asarray(frames, dtype=np.uint8))
    T, orig_H, orig_W, _ = source.shape

    # 1. Downscale
    scaled_source, downscale_s = rescale_frames(source, scale)
    scaled_H, scaled_W = scaled_source.shape[1], scaled_source.shape[2]

    # 2. Coded roundtrip
    request = reference_request(codec, qp, preset)
    trip = timed_roundtrip(scaled_source, request=request, fps=fps, work_dir=work_dir)

    # 3. Restore to original display grid
    restored, upscale_s = restore_to_display_grid(trip.frames, (orig_H, orig_W))
    if restored.shape != source.shape:
        raise RuntimeError(
            f"Restored shape {restored.shape} does not match original {source.shape}"
        )

    # 4. Scoring on display grid
    scores = score_headlines(source, restored)
    usable = isinstance(scores.get("vmaf"), float)

    client_s = trip.decode_seconds + upscale_s
    encoder_s = trip.encode_seconds + downscale_s

    scale_label = "native" if scale == 1.0 else f"res_{int(round(scale * 100))}"
    scaling_bytes = HEADER_BYTES if scale < 1.0 else 0
    coded_bytes = int(trip.size_bytes)
    total_bytes = coded_bytes + scaling_bytes

    return {
        "codec": codec,
        "scale": float(scale),
        "scale_label": scale_label,
        "qp": int(qp),
        "preset": preset,
        "bytes": total_bytes,
        "coded_bytes": coded_bytes,
        "scaling_bytes": scaling_bytes,
        "n_frames": T,
        "original_resolution": f"{orig_W}x{orig_H}",
        "coded_resolution": f"{scaled_W}x{scaled_H}",
        "restored_resolution": f"{orig_W}x{orig_H}",
        "usable": usable,
        "scores": scores,
        "timing": {
            "encode_seconds": round(float(trip.encode_seconds), 4),
            "decode_seconds": round(float(trip.decode_seconds), 4),
            "downscale_seconds": round(float(downscale_s), 4),
            "upscale_seconds": round(float(upscale_s), 4),
            "rescaling_seconds": round(float(downscale_s + upscale_s), 4),
            "encoder_seconds": round(float(encoder_s), 4),
            "client_seconds": round(float(client_s), 4),
        },
        "tool_path": trip.tool_path,
        "tool_version": trip.tool_version,
    }


def build_nondominated_envelope(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Construct Pareto non-dominated envelope across all resolution arms.

    A point (bytes, vmaf) is non-dominated if no other point has both <= bytes AND >= vmaf.
    """
    usable = [p for p in points if p.get("usable") and isinstance((p.get("scores") or {}).get("vmaf"), float)]
    if not usable:
        return []

    # Sort by bytes ascending
    sorted_pts = sorted(usable, key=lambda p: (int(p["bytes"]), -float(p["scores"]["vmaf"])))
    envelope: list[dict[str, Any]] = []
    max_vmaf = -float("inf")

    for p in sorted_pts:
        vmaf = float(p["scores"]["vmaf"])
        if vmaf > max_vmaf:
            envelope.append(p)
            max_vmaf = vmaf

    return envelope


def compare_curves_no_extrapolation(
    candidate_rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, Any]],
    *,
    metric_name: str = "vmaf",
) -> dict[str, Any]:
    """Compare candidate against anchor strictly on common quality support.

    Prohibits BD-rate extrapolation:
    If the curves do not overlap, returns scorable=False and bd_rate_percent=None.
    """
    spec = metric_spec(metric_name)

    def _get_points(rows: list[dict[str, Any]]) -> list[tuple[float, float]]:
        res = []
        for r in rows:
            if not r.get("usable"):
                continue
            sc = (r.get("scores") or {}).get(metric_name)
            if isinstance(sc, (int, float)) and np.isfinite(sc):
                res.append((float(r["bytes"]), float(sc)))
        return res

    cand_pts = _get_points(candidate_rows)
    anc_pts = _get_points(anchor_rows)

    report: dict[str, Any] = {
        "quality_metric": metric_name,
        "bd_rate_percent": None,
        "bd_quality": None,
        "n_candidate": len(cand_pts),
        "n_anchor": len(anc_pts),
        "is_scorable": False,
        "extrapolation_prohibited": True,
        "overlap": None,
        "reason": None,
    }

    if len(cand_pts) < 2 or len(anc_pts) < 2:
        report["reason"] = f"need at least 2 usable points on both curves (cand={len(cand_pts)}, anc={len(anc_pts)})"
        return report

    cand_qualities = [p[1] for p in cand_pts]
    anc_qualities = [p[1] for p in anc_pts]

    # Native support intervals
    cand_min, cand_max = min(cand_qualities), max(cand_qualities)
    anc_min, anc_max = min(anc_qualities), max(anc_qualities)

    overlap_low = max(cand_min, anc_min)
    overlap_high = min(cand_max, anc_max)

    # Strictly prohibit non-overlapping comparison
    if overlap_high <= overlap_low:
        report["reason"] = (
            f"no common quality support: candidate [{cand_min:.2f}, {cand_max:.2f}] "
            f"vs anchor [{anc_min:.2f}, {anc_max:.2f}] (extrapolation prohibited)"
        )
        return report

    # Build RDCurve objects
    cand_curve = RDCurve(
        rates=tuple(p[0] for p in cand_pts),
        qualities=tuple(p[1] for p in cand_pts),
        label="candidate",
        quality_spec=spec,
    )
    anc_curve = RDCurve(
        rates=tuple(p[0] for p in anc_pts),
        qualities=tuple(p[1] for p in anc_pts),
        label="anchor",
        quality_spec=spec,
    )

    try:
        comparison = compare_rd_curves(anc_curve, cand_curve, quality_spec=spec)
    except (InsufficientOverlapError, DegenerateCurveError) as exc:
        report["reason"] = str(exc)
        report["overlap"] = list(getattr(exc, "overlap", (overlap_low, overlap_high)))
        return report

    report.update(
        {
            "is_scorable": True,
            "bd_rate_percent": round(float(comparison.bd_rate_percent), 3),
            "bd_quality": round(float(comparison.bd_quality), 3),
            "overlap": list(comparison.overlap),
            "overlap_fraction": round(float(comparison.overlap_fraction), 3),
            "reason": None,
        }
    )
    return report


def recompute_stored_bd_rate_arithmetic_check(
    candidate_rows: list[dict[str, Any]],
    anchor_rows: list[dict[str, Any]],
    stored_bd_rate: float,
    *,
    metric_name: str = "vmaf",
    tolerance: float = 0.1,
) -> dict[str, Any]:
    """Recompute stored BD-rate as an arithmetic check only, not a newly valid experiment.

    Historical reports remain immutable. This arithmetic check verifies whether
    the stored value matches mathematical recomputation on the recorded points,
    without certifying the underlying experiment as valid under modern protocol rules.
    """
    comp = compare_curves_no_extrapolation(candidate_rows, anchor_rows, metric_name=metric_name)
    if not comp["is_scorable"] or comp["bd_rate_percent"] is None:
        return {
            "arithmetic_check_passed": False,
            "stored_bd_rate": stored_bd_rate,
            "recomputed_bd_rate": None,
            "discrepancy": None,
            "is_scorable": False,
            "reason": comp.get("reason", "unscorable curve comparison"),
            "status": "arithmetic_check_unscorable",
        }
    recomputed = float(comp["bd_rate_percent"])
    diff = abs(recomputed - stored_bd_rate)
    passed = diff <= tolerance
    return {
        "arithmetic_check_passed": passed,
        "stored_bd_rate": stored_bd_rate,
        "recomputed_bd_rate": recomputed,
        "discrepancy": round(diff, 4),
        "is_scorable": True,
        "overlap_fraction": comp.get("overlap_fraction"),
        "status": "arithmetic_check_verified" if passed else "arithmetic_discrepancy_detected",
        "note": "Arithmetic check only; does not validate experiment protocol or gate passage.",
    }


@dataclass(frozen=True)
class ResidualLadderRungSpec:
    """High-fidelity residual-on ladder rung coordinating with Worker A's interface."""

    rung_id: str
    residual_qp: int
    residual_codec: str = "av1"
    residual_preset: str = "8"
    background_downscale: int = 1
    block_threshold: float = 0.0
    summary: str = ""


HIGH_FIDELITY_RESIDUAL_RUNGS: tuple[ResidualLadderRungSpec, ...] = (
    ResidualLadderRungSpec(
        rung_id="H0",
        residual_qp=42,
        summary="Entry high-fidelity rung; residual AV1 QP 42, 1:1 scale, zero gating",
    ),
    ResidualLadderRungSpec(
        rung_id="H1",
        residual_qp=35,
        summary="Balanced high-fidelity rung; residual AV1 QP 35, 1:1 scale, zero gating",
    ),
    ResidualLadderRungSpec(
        rung_id="H2",
        residual_qp=28,
        summary="High-quality residual rung; residual AV1 QP 28, 1:1 scale, zero gating",
    ),
    ResidualLadderRungSpec(
        rung_id="H3",
        residual_qp=20,
        summary="Near-transparent residual rung; residual AV1 QP 20, 1:1 scale, zero gating",
    ),
)

# Coarser than H0 (QP 42). Invalid Wave 2 H0–H3 sat at VMAF 87–96 while native
# VVC topped out near 87, so overlap needs PointStream residual down as well as
# VVC/AV1 up.
OVERLAP_RESIDUAL_RUNGS: tuple[ResidualLadderRungSpec, ...] = (
    ResidualLadderRungSpec(
        rung_id="R63",
        residual_qp=63,
        summary="Coarse residual; AV1 QP 63, 1:1 scale, zero gating",
    ),
    ResidualLadderRungSpec(
        rung_id="R55",
        residual_qp=55,
        summary="Coarse residual; AV1 QP 55, 1:1 scale, zero gating",
    ),
    ResidualLadderRungSpec(
        rung_id="R48",
        residual_qp=48,
        summary="Mid-coarse residual; AV1 QP 48, 1:1 scale, zero gating",
    ),
    *HIGH_FIDELITY_RESIDUAL_RUNGS,
)


def configure_high_fidelity_residual_rung(
    base: PointstreamConfig,
    rung: ResidualLadderRungSpec,
) -> PointstreamConfig:
    """Configure PointstreamConfig for residual-on high-fidelity evaluation."""
    from dataclasses import replace

    # Enable residual in lattice
    lattice = replace(base.lattice, residual=True)

    # Configure residual: disable downscaling and gating for high fidelity
    res_cfg = replace(
        base.residual,
        codec=rung.residual_codec,
        rate=rung.residual_qp,
        preset=rung.residual_preset,
        background_downscale=rung.background_downscale,
        block_threshold=rung.block_threshold,
    )

    return replace(base, lattice=lattice, residual=res_cfg)


__all__ = [
    "DEFAULT_DISPLAY_INTERPOLATION",
    "DEFAULT_DOWNSCALE_INTERPOLATION",
    "HIGH_FIDELITY_RESIDUAL_RUNGS",
    "OVERLAP_RESIDUAL_RUNGS",
    "RESOLUTION_SCALES",
    "ResidualLadderRungSpec",
    "build_nondominated_envelope",
    "compare_curves_no_extrapolation",
    "configure_high_fidelity_residual_rung",
    "encode_resolution_arm",
    "recompute_stored_bd_rate_arithmetic_check",
    "rescale_frames",
    "restore_to_display_grid",
]
