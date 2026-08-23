"""Run the headroom measurement and write a JSON report."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys

from experiments.headroom.ladder import encode_luma_curve, encoders_available, resolved_tools
from experiments.headroom.measure import (
    bg_headroom,
    declared_bounds,
    duplicate_rate_ratio,
    fg_headroom,
)
from experiments.headroom.synthetic import handheld_clip, tennis_clip


def _mean_se(values: list[float]) -> dict[str, float | int | None]:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "se": None}
    mean = statistics.fmean(values)
    se = statistics.stdev(values) / (n**0.5) if n > 1 else 0.0
    return {"n": n, "mean": mean, "se": se}


def _energy_curve(frames, *, work_dir, qps, masks=None, label=""):
    """Stand-in encoder for hosts without a codec: rate tracks spatial energy."""
    from src.components.metrics.bd_rate import RDCurve

    del work_dir, masks
    luma = frames.astype("float64").mean(axis=-1)
    energy = float(((luma[1:] - luma[:-1]) ** 2).mean() + luma.std())
    rates = tuple(max(1.0, energy * 800.0 / qp) for qp in qps)
    qualities = tuple(48.0 - 0.2 * qp for qp in qps)
    return {"curve": RDCurve(rates=rates, qualities=qualities, label=label), "qps": qps}


def run(*, out_dir: Path, use_real_encoder: bool) -> dict:
    bounds = declared_bounds()
    encode_fn = encode_luma_curve if use_real_encoder else _energy_curve
    tools = None
    if use_real_encoder:
        tools = resolved_tools("avc")
    fg_plate: list[float] = []
    fg_flat: list[float] = []
    seeds = (0, 1, 2)
    last_fg: dict | None = None
    for seed in seeds:
        frames, masks = tennis_clip(n_frames=16, height=96, width=128, seed=seed)
        result = fg_headroom(
            frames,
            masks,
            work_dir=out_dir / f"fg_seed{seed}",
            encode_curve=encode_fn,
        )
        last_fg = result
        plate = result["plate_vs_original"].get("saving")
        flat = result["flat_vs_original"].get("saving")
        if plate is not None:
            fg_plate.append(float(plate))
        if flat is not None:
            fg_flat.append(float(flat))
    frames, masks = tennis_clip(n_frames=24, seed=0)
    bg_tennis = bg_headroom(frames, masks, panorama_valid=True)
    hand_frames, hand_masks = handheld_clip(n_frames=24)
    bg_general = bg_headroom(hand_frames, hand_masks, panorama_valid=False)
    empty = np_zeros_like(frames)
    empty_fg = fg_headroom(
        frames,
        empty,
        work_dir=out_dir / "fg_empty_mask",
        encode_curve=encode_fn,
        qps=(32, 40),
    )
    null_ratio = None
    if use_real_encoder:
        null_ratio = duplicate_rate_ratio(
            frames, work_dir=out_dir / "null", encode_curve=encode_fn
        )
    report = {
        "bounds_written_before_measurement": bounds,
        "not_a_pointstream_result": True,
        "encoder": "real" if use_real_encoder else "spatial-energy-standin",
        "tools": tools,
        "fg_tennis_plate_saving": _mean_se(fg_plate),
        "fg_tennis_flat_saving": _mean_se(fg_flat),
        "fg_last_seed": last_fg,
        "fg_empty_mask_plate_saving": empty_fg["plate_vs_original"].get("saving"),
        "fg_note": (
            "flat fill overstates the saving because a constant region is cheaper "
            "than a real background; plate inpaint is the estimate. "
            "This run uses a synthetic tennis court unless a later section names "
            "a real clip; it is a headroom argument, not a PointStream encode."
        ),
        "bg_tennis": bg_tennis,
        "bg_general": bg_general,
        "null_duplicate_rate_ratio": null_ratio,
        "instrument_range_psnr_dB": [20.0, 50.0],
        "skipped": [
            {
                "clip": "assets/real_tennis.mp4",
                "reason": "4K tennis clip is on disk (60 frames, 3840x2160) but there are no full-frame player masks; track crops are not a substitute. FG/BG headroom on that clip is skipped rather than faked.",
            },
            {
                "clip": "/home/itec/emanuele/Datasets/DAVIS",
                "reason": "DAVIS sequences are JPEG frames only on this host; annotation PNGs are missing, so general-domain FG is skipped. BG panorama is invalid for handheld DAVIS anyway.",
            },
        ],
        "synthetic_probe": {
            "domain": "tennis",
            "n_frames": 16,
            "frame_hw": [96, 128],
            "note": "Players occupy ~8–12% of pixels here versus ~2.1% in 4K broadcast. FG saving on this probe overstates the 4K area share; it is directional evidence, not the 4K number.",
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


def np_zeros_like(frames):
    import numpy as np

    return np.zeros(frames.shape[:3], dtype=bool)


def main() -> int:
    out = Path("outputs/bp13-headroom")
    out.mkdir(parents=True, exist_ok=True)
    real = encoders_available("avc")
    report = run(out_dir=out, use_real_encoder=real)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
