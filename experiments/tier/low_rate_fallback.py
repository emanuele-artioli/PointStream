"""Conventional-fallback control: the fallback codec must be the reference codec.

Turning the object stream off is a different control. This one encodes the same
frames through ``FallbackConfig.encode_request()`` and through the independent
reference request, then checks the bounds file's rate and VMAF bands.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from experiments.tier.low_rate_measure import (
    late_frame_report,
    reference_request,
    score_headlines,
    timed_roundtrip,
    timing_record,
)
from src.components.codec.frames import even_size
from src.contracts.codecs import EncodeRequest, RateControl
from src.contracts.config import FallbackConfig


def aligned_fallback_request(
    fallback: FallbackConfig,
    *,
    codec: str,
    qp: int,
    preset: str,
) -> EncodeRequest:
    """The fallback arm at the reference's QP and slowest preset, not CRF 35 / preset 8."""
    tuned = replace(
        fallback,
        codec=str(codec),
        rate_control=RateControl.QP,
        rate=int(qp),
        preset=str(preset),
        pix_fmt="yuv420p",
    )
    request = tuned.encode_request()
    request.validate()
    return request


def evaluate_fallback_equivalence(
    fallback_row: dict[str, Any],
    reference_row: dict[str, Any],
    *,
    rate_rel: tuple[float, float] = (0.95, 1.05),
    vmaf_abs: tuple[float, float] = (-1.0, 1.0),
) -> dict[str, Any]:
    """Compare one fallback point to one reference point. No BD-rate here."""
    fb_bytes = float(fallback_row["bytes"])
    ref_bytes = float(reference_row["bytes"])
    if ref_bytes <= 0:
        raise SystemExit("reference fallback control has zero bytes; not a codec result.")
    ratio = fb_bytes / ref_bytes
    fb_vmaf = fallback_row.get("scores", {}).get("vmaf")
    ref_vmaf = reference_row.get("scores", {}).get("vmaf")
    vmaf_delta: float | None
    if isinstance(fb_vmaf, (int, float)) and isinstance(ref_vmaf, (int, float)):
        vmaf_delta = float(fb_vmaf) - float(ref_vmaf)
        vmaf_ok = vmaf_abs[0] <= vmaf_delta <= vmaf_abs[1]
    else:
        vmaf_delta = None
        vmaf_ok = False
    rate_ok = rate_rel[0] <= ratio <= rate_rel[1]
    return {
        "control": "conventional-fallback-equivalence",
        "fallback_bytes": int(fb_bytes),
        "reference_bytes": int(ref_bytes),
        "rate_rel": round(ratio, 6),
        "vmaf_delta": vmaf_delta,
        "rate_rel_band": list(rate_rel),
        "vmaf_abs_band": list(vmaf_abs),
        "held": bool(rate_ok and vmaf_ok),
        "rate_ok": bool(rate_ok),
        "vmaf_ok": bool(vmaf_ok),
    }


def _score_trip(source: np.ndarray, trip: Any) -> dict[str, Any]:
    from experiments.tier.low_rate_validate import decode_rejections

    reasons = decode_rejections(
        bitstream_bytes=int(trip.size_bytes),
        source_shape=(
            int(source.shape[0]),
            int(source.shape[1]),
            int(source.shape[2]),
            int(source.shape[3]),
        ),
        decoded_shape=tuple(int(dim) for dim in trip.frames.shape),
    )
    scores: dict[str, float | str] = {}
    late: dict[str, Any] = {}
    usable = not reasons
    if usable:
        scores = score_headlines(source, trip.frames)
        late = late_frame_report(source, trip.frames)
        usable = isinstance(scores.get("vmaf"), float)
    return {
        "bytes": int(trip.size_bytes),
        "usable": usable,
        "rejections": reasons,
        "scores": scores,
        "late_frame": late,
        **timing_record(trip),
    }


def run_fallback_control(
    frames: np.ndarray,
    fallback: FallbackConfig,
    *,
    codec: str,
    qp: int,
    preset: str,
    fps: float,
    rate_rel: tuple[float, float] = (0.95, 1.05),
    vmaf_abs: tuple[float, float] = (-1.0, 1.0),
) -> dict[str, Any]:
    """Encode ``frames`` once as fallback and once as the independent reference."""
    source = even_size(np.asarray(frames, dtype=np.uint8))
    fb_request = aligned_fallback_request(fallback, codec=codec, qp=qp, preset=preset)
    ref_request = reference_request(codec, qp, preset)
    if (
        fb_request.codec_name != ref_request.codec_name
        or fb_request.rate != ref_request.rate
        or fb_request.preset != ref_request.preset
        or fb_request.rate_control != ref_request.rate_control
        or fb_request.pix_fmt != ref_request.pix_fmt
    ):
        raise SystemExit(
            "aligned fallback request does not equal the reference request. "
            "The control would not be testing the same codec."
        )
    fb_trip = timed_roundtrip(source, request=fb_request, fps=fps)
    ref_trip = timed_roundtrip(source, request=ref_request, fps=fps)
    fb_row = _score_trip(source, fb_trip)
    ref_row = _score_trip(source, ref_trip)
    comparison = evaluate_fallback_equivalence(
        fb_row, ref_row, rate_rel=rate_rel, vmaf_abs=vmaf_abs
    )
    return {
        "qp": int(qp),
        "preset": str(preset),
        "codec": str(codec),
        "fallback": fb_row,
        "reference": ref_row,
        "comparison": comparison,
    }


__all__ = [
    "aligned_fallback_request",
    "evaluate_fallback_equivalence",
    "run_fallback_control",
]
