"""FG and BG headroom measurements. Bounds are constants, written before any encode."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from src.components.background.sidecar import JpegSidecar
from src.components.metrics.bd_rate import InsufficientOverlapError, compare_rd_curves
from experiments.headroom.remove import plate_fill, player_fraction, prepare_fills

# Written down before any clip is encoded. FG is a fraction of conventional
# bitrate saved by not coding the players, at matched quality of each arm's
# own source. BG is how many times cheaper a plate+homographies is than
# coding the background sequence.
FG_STRONG = 0.25
FG_MODEST = 0.10
BG_ORDERS_OF_MAGNITUDE = 10.0

EncodeFn = Callable[..., dict[str, Any]]


def declared_bounds() -> dict[str, Any]:
    """The bars this measurement is judged against. Not derived from the run."""
    return {
        "fg_strong_saving": FG_STRONG,
        "fg_modest_saving": FG_MODEST,
        "fg_weak_below": FG_MODEST,
        "bg_orders_of_magnitude_ratio": BG_ORDERS_OF_MAGNITUDE,
        "written_before_measurement": True,
        "instrument_psnr_range": "finite PSNR in dB; identical frames are +inf; typical coded 20–50 dB",
        "instrument_rate_range": "positive payload bytes; a duplicate encode of the same clip must match within a few percent",
        # Per-codec FG saving bands, written before any 4K encode. Basis: the
        # withdrawn synthetic AVC figure was 12.2% at ~4.7% player area; real
        # 4K area is ~2.3% so a linear guess is half that, and a stronger
        # codec compresses the near-static court harder, so the players' share
        # of leftover bits should rise. AVC is the conservative rung.
        "fg_saving_band_avc": [0.03, 0.18],
        "fg_saving_band_hevc": [0.04, 0.22],
        "fg_saving_band_av1": [0.05, 0.28],
        "fg_saving_band_vvc": [0.05, 0.30],
        "fg_codec_ranking_prediction": "AV1 >= HEVC >= AVC on FG saving for the same clip",
        "fg_saving_below_zero_is_alarm": True,
        "fg_saving_above_0_40_is_alarm": True,
        "bg_intercoded_ratio_band": [1.5, 12.0],
        "player_area_band": [0.015, 0.035],
        "paste_back_mae_opaque_max": 2.0,
        "flat_fill_is_not_an_upper_bound": True,
        "empty_mask_saving_abs_max": 0.02,
        "duplicate_rate_ratio_band": [0.97, 1.03],
    }


def fg_verdict(saving: float) -> str:
    if saving >= FG_STRONG:
        return "strong"
    if saving >= FG_MODEST:
        return "modest"
    return "weak"


def _saving(anchor_curve: Any, candidate_curve: Any) -> dict[str, Any]:
    """Fraction of anchor rate saved by the candidate at matched quality."""
    try:
        comparison = compare_rd_curves(anchor_curve, candidate_curve)
    except InsufficientOverlapError as exc:
        return {
            "saving": None,
            "bd_rate": None,
            "overlap": list(exc.overlap),
            "overlap_fraction": exc.overlap_fraction,
            "error": str(exc),
        }
    saving = float(-comparison.bd_rate)
    return {
        "saving": saving,
        "bd_rate": comparison.bd_rate,
        "bd_rate_percent": comparison.bd_rate_percent,
        "bd_quality": comparison.bd_quality,
        "overlap": list(comparison.overlap),
        "overlap_fraction": comparison.overlap_fraction,
        "verdict": fg_verdict(saving),
    }


def fg_headroom(
    frames: np.ndarray,
    masks: np.ndarray,
    *,
    work_dir: Path,
    encode_curve: EncodeFn,
    qps: tuple[int, ...] = (32, 40, 48),
) -> dict[str, Any]:
    """Original vs plate vs flat vs court-median.

    Plate is the estimate. Flat is a bracket, not an upper bound: on the
    synthetic court it understated the prize (a grey hole in a green court).
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    fills = prepare_fills(frames, masks)
    original = encode_curve(
        frames, work_dir=work_dir / "original", qps=qps, masks=masks, label="original"
    )
    plate_arm = encode_curve(
        fills.plate, work_dir=work_dir / "plate", qps=qps, masks=masks, label="plate"
    )
    flat_arm = encode_curve(
        fills.flat, work_dir=work_dir / "flat", qps=qps, masks=masks, label="flat"
    )
    median_arm = encode_curve(
        fills.median,
        work_dir=work_dir / "median",
        qps=qps,
        masks=masks,
        label="median",
    )
    homog_bytes = int(np.asarray(fills.homographies, dtype=np.float32).nbytes)
    return {
        "player_area_fraction": player_fraction(masks),
        "n_frames": int(frames.shape[0]),
        "frame_hw": [int(frames.shape[1]), int(frames.shape[2])],
        "plate_vs_original": _saving(original["curve"], plate_arm["curve"]),
        "flat_vs_original": _saving(original["curve"], flat_arm["curve"]),
        "median_vs_original": _saving(original["curve"], median_arm["curve"]),
        "original_fg_psnr": original.get("fg_psnr"),
        "original_bg_psnr": original.get("bg_psnr"),
        "qps": list(qps),
        "tool": original.get("tool"),
        "homography_bytes": homog_bytes,
    }


def bg_headroom(
    frames: np.ndarray,
    masks: np.ndarray,
    *,
    jpeg_quality: int = 50,
    panorama_valid: bool = True,
) -> dict[str, Any]:
    """Plate once + homographies versus coding the background sequence as JPEGs.

    The conventional arm here is one JPEG per background frame (the plate-filled
    clip). That is a still-image baseline, not a video codec — it overstates the
    conventional cost relative to inter-predicted video, so a panorama that
    still wins by orders of magnitude is a stronger claim, not a weaker one.
    """
    filled, plate, homographies = plate_fill(frames, masks)
    sidecar = JpegSidecar(quality=jpeg_quality)
    plate_bytes = len(sidecar.encode(plate))
    lower_q = JpegSidecar(quality=max(1, jpeg_quality - 20))
    higher_q = JpegSidecar(quality=min(100, jpeg_quality + 20))
    lower_q_bytes = len(lower_q.encode(plate))
    higher_q_bytes = len(higher_q.encode(plate))
    # Higher JPEG quality must emit a larger payload. A no-op quality flag
    # would make these three sizes equal.
    if not (lower_q_bytes < plate_bytes < higher_q_bytes):
        raise RuntimeError(
            "JPEG quality did not move payload as claimed: "
            f"q={jpeg_quality - 20}:{lower_q_bytes} q={jpeg_quality}:{plate_bytes} "
            f"q={jpeg_quality + 20}:{higher_q_bytes}"
        )
    homog_bytes = int(np.asarray(homographies, dtype=np.float32).nbytes)
    panorama_bytes = plate_bytes + homog_bytes
    conventional = 0
    for frame in filled:
        bgr = frame[:, :, ::-1]
        conventional += len(sidecar.encode(bgr))
    ratio = (conventional / panorama_bytes) if panorama_bytes else None
    return {
        "panorama_valid": panorama_valid,
        "player_area_fraction": player_fraction(masks),
        "n_frames": int(frames.shape[0]),
        "plate_bytes": plate_bytes,
        "homography_bytes": homog_bytes,
        "panorama_bytes": panorama_bytes,
        "conventional_jpeg_bytes": conventional,
        "conventional_over_panorama": ratio,
        "orders_of_magnitude": (
            bool(ratio is not None and ratio >= BG_ORDERS_OF_MAGNITUDE)
            if panorama_valid
            else False
        ),
        "jpeg_quality_moves_size": True,
        "note": (
            None
            if panorama_valid
            else "free-moving camera: a panorama is not a valid background model"
        ),
    }


def duplicate_rate_ratio(
    frames: np.ndarray,
    *,
    work_dir: Path,
    encode_curve: EncodeFn,
    qp: int = 40,
) -> float:
    """Null: two encodes of the same clip. Must sit near 1."""
    a = encode_curve(frames, work_dir=work_dir / "dup_a", qps=(qp, qp + 8), label="dup_a")
    b = encode_curve(frames, work_dir=work_dir / "dup_b", qps=(qp, qp + 8), label="dup_b")
    rate_a = float(a["curve"].rates[0])
    rate_b = float(b["curve"].rates[0])
    return rate_b / rate_a if rate_a else float("inf")


def codec_ranking_alarm(savings: dict[str, float | None]) -> str | None:
    """AV1 >= HEVC >= AVC on FG saving. A reversal is an alarm, not a finding."""
    avc = savings.get("avc")
    hevc = savings.get("hevc")
    av1 = savings.get("av1")
    if avc is None or av1 is None:
        return None
    if av1 < avc:
        return (
            "ALARM: AV1 FG saving below AVC for the same clip; check encode "
            "settings and matched-quality pairing before reporting a ranking"
        )
    if hevc is not None and not (av1 >= hevc >= avc):
        return (
            f"ALARM: codec ranking AV1={av1:.4f} HEVC={hevc:.4f} AVC={avc:.4f} "
            "violates AV1 >= HEVC >= AVC"
        )
    return None


def bg_headroom_intercoded(
    *,
    conventional_curve: Any,
    plate_bgr: np.ndarray,
    homographies: Any,
    work_dir: Path,
    encode_curve: EncodeFn,
    qps: tuple[int, ...] = (32, 40, 48),
) -> dict[str, Any]:
    """Plate as a 1-frame encode + homographies, versus an intercoded filled clip.

    Conventional arm is the plate-filled sequence already encoded for FG.
    Panorama arm is the plate still encoded at the same QPs, plus homography
    bytes. Quality of the still is PSNR of the decoded plate against itself;
    if the quality ranges do not overlap, per-QP ratios are still reported.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    plate_rgb = np.asarray(plate_bgr)[:, :, ::-1]
    if plate_rgb.ndim == 3:
        plate_rgb = plate_rgb[np.newaxis, ...]
    plate_enc = encode_curve(
        plate_rgb, work_dir=work_dir / "plate_still", qps=qps, label="plate_still"
    )
    homog_bytes = int(np.asarray(homographies, dtype=np.float32).nbytes)
    conv_rates = tuple(float(r) for r in conventional_curve["curve"].rates)
    plate_rates = tuple(float(r) + homog_bytes for r in plate_enc["curve"].rates)
    ratios = [
        conv / plate if plate else None
        for conv, plate in zip(conv_rates, plate_rates)
    ]
    finite = [float(r) for r in ratios if r is not None]
    mean_ratio = float(sum(finite) / len(finite)) if finite else None
    from src.components.metrics.bd_rate import RDCurve

    plate_curve = plate_enc["curve"]
    adjusted = RDCurve(
        rates=tuple(float(r) + homog_bytes for r in plate_curve.rates),
        qualities=plate_curve.qualities,
        label="plate_still+homog",
    )
    bd = _saving(conventional_curve["curve"], adjusted)
    band = declared_bounds()["bg_intercoded_ratio_band"]
    alarms: list[str] = []
    if mean_ratio is not None and mean_ratio > 20:
        alarms.append("ALARM: intercoded BG ratio > 20; JPEG stills may have leaked in")
    if mean_ratio is not None and mean_ratio < 1:
        alarms.append("ALARM: plate+homographies cost more than intercoded video")
    if mean_ratio is not None and not (band[0] <= mean_ratio <= band[1]):
        alarms.append(
            f"intercoded BG ratio {mean_ratio:.3f} outside prewritten band {band}"
        )
    return {
        "conventional_rates": list(conv_rates),
        "plate_still_rates": list(plate_rates),
        "homography_bytes": homog_bytes,
        "ratio_at_qp": list(ratios),
        "mean_ratio": mean_ratio,
        "qps": list(qps),
        "bd_vs_conventional": bd,
        "alarms": alarms,
        "tool": plate_enc.get("tool"),
    }
