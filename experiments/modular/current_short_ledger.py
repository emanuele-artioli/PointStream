"""Measure the current PointStream 48-frame ladder and weighted anchors."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.modular.image_codec_probe import FRAMES, MASKS
from experiments.modular.measured_ladder import (
    beats,
    load_sequence,
    score_regions,
    measure_rungs,
    _yuv420_to_rgb,
)
from src.components.codec.y4m import read

OUT = Path(
    "/home/itec/emanuele/pointstream-data/outputs/modular/"
    "measured-tennis-codec/current-short.json"
)
ANCHOR_ROOT = Path(
    "/home/itec/emanuele/pointstream-data/outputs/modular/"
    "measured-tennis/anchors/short"
)


def _anchor(codec: str, qp: int, frames: np.ndarray, mask: np.ndarray) -> dict:
    decoded_path = ANCHOR_ROOT / codec / f"decoded_qp{qp}.y4m"
    decoded = read(decoded_path)
    if decoded.chroma is None:
        raise RuntimeError(f"{decoded_path} has no chroma")
    rgb = _yuv420_to_rgb(decoded.luma, decoded.chroma)
    overall, fg, bg, weighted = score_regions(
        frames,
        rgb,
        mask,
        fg_weight=0.70,
        bg_weight=0.30,
    )
    return {
        "bytes": (ANCHOR_ROOT / codec / f"{codec}_qp{qp}{'.vvc' if codec == 'vvc' else '.ivf'}").stat().st_size,
        "psnr_overall": overall,
        "psnr_fg": fg,
        "psnr_bg": bg,
        "psnr_weighted": weighted,
        "decoded_path": str(decoded_path),
    }


def main() -> None:
    frames, mask = load_sequence(FRAMES, MASKS, 48)
    anchors = {
        "vvc_qp46": _anchor("vvc", 46, frames, mask),
        "av1_qp54": _anchor("av1", 54, frames, mask),
    }
    rungs = measure_rungs(frames, mask)
    rows = []
    for rung in rungs:
        rows.append(
            {
                "rung_id": rung.rung_id,
                "bytes_background": rung.bytes_background,
                "bytes_appearance": rung.bytes_appearance,
                "bytes_metadata": rung.bytes_metadata,
                "bytes_residual": rung.bytes_residual,
                "total_bytes": rung.total_bytes,
                "psnr_overall": rung.psnr_overall,
                "psnr_fg": rung.psnr_fg,
                "psnr_bg": rung.psnr_bg,
                "psnr_weighted": rung.psnr_weighted,
                "beats_vvc_qp46_rate": beats(rung.total_bytes, anchors["vvc_qp46"]["bytes"]),
                "beats_av1_qp54_rate": beats(rung.total_bytes, anchors["av1_qp54"]["bytes"]),
                "beats_vvc_qp46_weighted_quality": (
                    rung.psnr_weighted is not None
                    and rung.psnr_weighted > anchors["vvc_qp46"]["psnr_weighted"]
                ),
                "beats_av1_qp54_weighted_quality": (
                    rung.psnr_weighted is not None
                    and rung.psnr_weighted > anchors["av1_qp54"]["psnr_weighted"]
                ),
            }
        )
        print(rows[-1], flush=True)
    result = {
        "source": str(FRAMES),
        "n_frames": 48,
        "plate_codec": "webp",
        "plate_quality": 40,
        "crop_codec": "av1",
        "crop_qp": 42,
        "residual_codec": "vvc",
        "residual_preset": "medium",
        "residual_qp": 40,
        "metric": {"foreground_weight": 0.70, "background_weight": 0.30},
        "anchors": anchors,
        "rungs": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
