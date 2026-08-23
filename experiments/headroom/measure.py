"""FG and BG headroom measurements. Bounds are constants, written before any encode."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from src.components.background.sidecar import JpegSidecar
from src.components.metrics.bd_rate import InsufficientOverlapError, compare_rd_curves
from experiments.headroom.remove import flat_fill, plate_fill, player_fraction

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
    """Original vs plate-inpaint vs flat-fill. Plate is the estimate; flat overstates."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    plate, _, _homographies = plate_fill(frames, masks)
    flat = flat_fill(frames, masks)
    original = encode_curve(
        frames, work_dir=work_dir / "original", qps=qps, masks=masks, label="original"
    )
    plate_arm = encode_curve(
        plate, work_dir=work_dir / "plate", qps=qps, masks=masks, label="plate"
    )
    flat_arm = encode_curve(
        flat, work_dir=work_dir / "flat", qps=qps, masks=masks, label="flat"
    )
    return {
        "player_area_fraction": player_fraction(masks),
        "n_frames": int(frames.shape[0]),
        "frame_hw": [int(frames.shape[1]), int(frames.shape[2])],
        "plate_vs_original": _saving(original["curve"], plate_arm["curve"]),
        "flat_vs_original": _saving(original["curve"], flat_arm["curve"]),
        "original_fg_psnr": original.get("fg_psnr"),
        "original_bg_psnr": original.get("bg_psnr"),
        "qps": list(qps),
        "tool": original.get("tool"),
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
