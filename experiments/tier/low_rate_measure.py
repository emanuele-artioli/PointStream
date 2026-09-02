"""Shared scoring, preset selection, and late-frame helpers for the low-rate search.

No runner, no torch. Encoder binaries are used only by ``timed_roundtrip``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tier.low_rate_validate import PROBE_PATH
from src.components.codec.frames import rgb_to_luma
from src.components.codec.measure import PRESETS, TimedRoundtrip, timed_roundtrip
from src.contracts.codecs import EncodeRequest, RateControl

TIMING_KEYS: tuple[str, ...] = ("encode_seconds", "decode_seconds")


def y_psnr(reference: np.ndarray, predicted: np.ndarray) -> float:
    ref = rgb_to_luma(np.asarray(reference)).astype(np.float64)
    got = rgb_to_luma(np.asarray(predicted)).astype(np.float64)
    mse = float(np.mean((ref - got) ** 2))
    return float("inf") if mse == 0.0 else 10.0 * float(np.log10((255.0**2) / mse))


def per_frame_y_psnr(reference: np.ndarray, predicted: np.ndarray) -> list[float]:
    ref = np.asarray(reference)
    got = np.asarray(predicted)
    count = min(int(ref.shape[0]), int(got.shape[0]))
    return [y_psnr(ref[index], got[index]) for index in range(count)]


def last_minus_first(values: Sequence[float]) -> float:
    """Late-frame delta. Positive means the last frame scored higher than the first."""
    if len(values) < 2:
        raise ValueError("late-frame delta needs at least two frames")
    return float(values[-1] - values[0])


def score_headlines(reference: np.ndarray, predicted: np.ndarray) -> dict[str, float | str]:
    """VMAF (primary), Y-PSNR and SSIM (secondary). A missing VMAF binary is one point."""
    from src.components.metrics.ssim import SsimMetric
    from src.components.metrics.vmaf import VmafMetric

    scores: dict[str, float | str] = {
        "psnr_y": y_psnr(reference, predicted),
        "ssim": float(SsimMetric().score(reference, predicted)),
    }
    try:
        scores["vmaf"] = float(VmafMetric().score(reference, predicted))
    except (RuntimeError, FileNotFoundError) as exc:
        scores["vmaf_error"] = str(exc)
    return scores


def late_frame_report(reference: np.ndarray, predicted: np.ndarray) -> dict[str, Any]:
    """First vs last frame. The rot bound reads these, not the clip mean."""
    psnr_by_frame = per_frame_y_psnr(reference, predicted)
    report: dict[str, Any] = {
        "psnr_y_by_frame": psnr_by_frame,
        "psnr_y_first": psnr_by_frame[0],
        "psnr_y_last": psnr_by_frame[-1],
        "psnr_y_last_minus_first": last_minus_first(psnr_by_frame),
    }
    try:
        from src.components.metrics.vmaf import VmafMetric

        metric = VmafMetric()
        vmaf_first = float(metric.score(reference[:1], predicted[:1]))
        vmaf_last = float(metric.score(reference[-1:], predicted[-1:]))
        report.update(
            {
                "vmaf_first": vmaf_first,
                "vmaf_last": vmaf_last,
                "vmaf_last_minus_first": last_minus_first((vmaf_first, vmaf_last)),
            }
        )
    except (RuntimeError, FileNotFoundError) as exc:
        report["vmaf_error"] = str(exc)
    return report


def recorded_slowest_preset(codec: str, probe_path: Path | None = None) -> str:
    """The preset the M1 probe actually encoded with, not the convenience table.

    ``measure.PRESETS`` is ``av1=10`` / ``vvc=faster``. Those are accounting
    presets. The primary comparison must use the probe's ``selected_preset``.
    """
    path = probe_path or PROBE_PATH
    if not path.is_file():
        raise SystemExit(
            f"{path} does not exist. Probe AV1/VVC floors before the sweep "
            "(python -m experiments.tier.low_rate_probe)."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        return str(payload["tools"][codec]["selected_preset"])
    except (KeyError, TypeError) as exc:
        raise SystemExit(
            f"{path} has no tools.{codec}.selected_preset. Re-run the codec-floor probe."
        ) from exc


def primary_preset(
    codec: str,
    *,
    probe_path: Path | None = None,
    override: str | None = None,
) -> str:
    """Slowest valid preset for the primary comparison.

    An explicit ``override`` is allowed only as a labelled faster-preset arm
    after Gate A. The default path refuses the convenience table in
    ``measure.PRESETS``.
    """
    if override is not None:
        return str(override)
    selected = recorded_slowest_preset(codec, probe_path)
    convenience = PRESETS.get(codec)
    if convenience is not None and str(selected) == str(convenience):
        raise ValueError(
            f"{codec} probe selected {selected!r}, which is measure.PRESETS "
            f"({convenience!r}), not a slowest-preset primary comparison."
        )
    return selected


def reference_request(codec: str, qp: int, preset: str) -> EncodeRequest:
    """One independent-reference encode. The QP is the codec's, not residual QP."""
    request = EncodeRequest(
        codec_name=codec,
        rate_control=RateControl.QP,
        rate=int(qp),
        preset=str(preset),
        pix_fmt="yuv420p",
    )
    request.validate()
    return request


def timing_record(trip: TimedRoundtrip) -> dict[str, float]:
    return {
        "encode_seconds": round(float(trip.encode_seconds), 3),
        "decode_seconds": round(float(trip.decode_seconds), 3),
    }


__all__ = [
    "TIMING_KEYS",
    "last_minus_first",
    "late_frame_report",
    "per_frame_y_psnr",
    "primary_preset",
    "recorded_slowest_preset",
    "reference_request",
    "score_headlines",
    "timed_roundtrip",
    "timing_record",
    "y_psnr",
]
