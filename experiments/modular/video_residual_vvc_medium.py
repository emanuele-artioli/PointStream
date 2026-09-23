"""VVC residual points at preset medium.

libvvenc preset ``faster`` writes an empty bitstream for the 48-frame C1
error video and exits 0. Preset ``medium`` emits a file. Source anchors stay
on ``faster``; these rows are only the residual arm.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np

from experiments.modular.image_codec_probe import FRAMES, MASKS
from experiments.modular.measured_ladder import (
    _yuv420_to_rgb,
    load_sequence,
    measure_rungs,
)
from experiments.modular.video_codec_probe import OUT, WORK, _save
from src.components.codec.encode import BITSTREAM_SUFFIX, decode, encode
from src.components.codec.frames import even_size
from src.components.codec.y4m import Y4M, read, write
from src.components.metrics.psnr import PsnrMetric
from src.contracts.codecs import EncodeRequest, RateControl
from experiments.modular.measured_ladder import _rgb_to_yuv420

QPS = (32, 40, 46)


def _write_y4m(frames_rgb: np.ndarray, path: Path) -> np.ndarray:
    clip = even_size(frames_rgb)
    luma, chroma = _rgb_to_yuv420(clip)
    path.parent.mkdir(parents=True, exist_ok=True)
    write(
        path,
        Y4M(
            width=int(luma.shape[2]),
            height=int(luma.shape[1]),
            fps=25.0,
            luma=luma,
            chroma=chroma,
        ),
    )
    return clip


def main() -> None:
    frames, mask = load_sequence(FRAMES, MASKS, 48)
    c1 = measure_rungs(frames, mask, include_residuals=False)[1]
    error = np.clip(
        frames.astype(np.int16) - c1.reconstruction.astype(np.int16) + 128,
        0,
        255,
    ).astype(np.uint8)
    source = WORK / "residual_full" / "vvc" / "medium_input.y4m"
    _write_y4m(error, source)
    report = json.loads(OUT.read_text())
    for qp in QPS:
        row = _one(source, frames, c1.reconstruction, error, qp, "residual_full")
        report["rows"].append(row)
        print(row, flush=True)
        _save(report)


def _one(source, frames, base, error, qp, signal) -> dict:
    work = WORK / signal / "vvc" / f"qp{qp}_medium"
    work.mkdir(parents=True, exist_ok=True)
    request = EncodeRequest(
        codec_name="vvc",
        rate_control=RateControl.QP,
        rate=int(qp),
        preset="medium",
        pix_fmt="yuv420p",
    )
    dest = work / f"vvc_qp{qp}{BITSTREAM_SUFFIX['vvc']}"
    encode(source, dest, request, work_dir=work)
    decoded_path = work / "decoded.y4m"
    decode(dest, decoded_path, request)
    decoded = read(decoded_path)
    if decoded.chroma is None:
        raise RuntimeError(f"{dest} decoded without chroma")
    decoded_rgb = _yuv420_to_rgb(decoded.luma, decoded.chroma)
    height, width = frames.shape[1], frames.shape[2]
    decoded_rgb = decoded_rgb[:, :height, :width]
    signed = decoded_rgb.astype(np.int16) - 128
    corrected = np.clip(base.astype(np.int16) + signed, 0, 255).astype(np.uint8)
    reference = error[:, :height, :width]
    return {
        "signal": signal,
        "codec": "vvc",
        "qp": qp,
        "preset": "medium",
        "bytes": dest.stat().st_size,
        "psnr": round(float(PsnrMetric().score(reference, np.clip(decoded_rgb, 0, 255).astype(np.uint8))), 4),
        "psnr_corrected": round(float(PsnrMetric().score(frames, corrected)), 4),
    }


if __name__ == "__main__":
    main()
