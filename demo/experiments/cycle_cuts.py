"""Cut a repetitive take on the action cycle, not on a histogram change.

The HSV cutter stays quiet while the worker repeats the same motion, because
the background histogram does not move. This estimates the cycle from the
autocorrelation of a tiny grayscale proxy, then places a cut every period.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np

FFMPEG = os.environ.get("FFMPEG", "/opt/local/bin/ffmpeg")
SRC = Path(
    "/home/itec/emanuele/Datasets/Egocentric-10K/curated/clip_01_factory001_worker001_00001.mp4"
)
SW, SH = 16, 16
FPS = 30.0
# The long middle span the HSV cutter left as one scene.
SPAN = (5210, 29237)


def load_proxy() -> np.ndarray:
    cmd = [
        FFMPEG, "-v", "error", "-i", str(SRC),
        "-vf", f"scale={SW}:{SH},format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    n = len(raw) // (SW * SH)
    return np.frombuffer(raw, dtype=np.uint8).reshape(n, SH * SW).astype(np.float32)


def cycle_period(desc: np.ndarray, start: int, end: int) -> tuple[int, np.ndarray]:
    """Dominant lag, in frames, of the descriptor autocorrelation inside the span."""
    chunk = desc[start:end]
    chunk = chunk - chunk.mean(axis=0, keepdims=True)
    # Autocorrelation via FFT, averaged over pixels.
    n = chunk.shape[0]
    spec = np.fft.rfft(chunk, n=n * 2, axis=0)
    ac = np.fft.irfft(spec * np.conj(spec), axis=0)[:n].real.mean(axis=1)
    ac /= ac[0] + 1e-6
    # Ignore lags shorter than 1s or longer than 60s.
    lo, hi = int(FPS), min(int(60 * FPS), n // 2)
    window = ac[lo:hi]
    period = lo + int(np.argmax(window))
    return period, ac


def main() -> None:
    print("loading proxy", flush=True)
    desc = load_proxy()
    start, end = SPAN
    period, ac = cycle_period(desc, start, end)
    peak = float(ac[period])
    cuts = list(range(start, end - period, period))
    # How alike are two frames one cycle apart, on the proxy.
    gaps = []
    for cut in cuts[:40]:
        a = desc[cut]
        b = desc[cut + period]
        gaps.append(float(np.mean(np.abs(a - b))))
    report = {
        "span_frames": [start, end],
        "span_sec": round((end - start) / FPS, 1),
        "period_frames": period,
        "period_sec": round(period / FPS, 2),
        "autocorr_at_period": round(peak, 3),
        "n_cycles": len(cuts),
        "proxy_mad_one_cycle_apart": round(float(np.mean(gaps)), 2),
        "proxy_mad_scale": "0-255 on a 16x16 gray frame",
    }
    out = Path("/home/itec/emanuele/tmp/repetitive-plate/cycle_cuts.json")
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
