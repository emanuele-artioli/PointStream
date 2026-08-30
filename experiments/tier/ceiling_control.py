"""Is the paired ladder's anchor limited by the codec, or by its colour path?

BP29 §2 (wave 8, Stream C). Extending the av1 anchor *upward* to QP 5 showed
2.16x the bytes of QP 15 buying **0.21 dB** — 1,836,510 B at 44.23 dB against
851,572 B at 44.02 dB. A rung that costs and does not pay is the signature of a
ceiling somewhere other than the encoder, so this asks where.

`coded_roundtrip` carries the anchor's pixels as RGB, hands them to the encoder
at `request.pix_fmt` (yuv420p for every codec on the roster), and reads RGB
back. This module runs that colour path with **no lossy codec at all** — ffv1 is
mathematically lossless — so anything it loses is the pixel format, not the
encoder.

Three anchors rather than one, because a single number cannot separate "the
conversion costs something" from "the harness is broken":

* **RGB planar** — no chroma conversion. Must come back `inf`, or the harness
  itself is lossy and nothing below it means anything.
* **yuv444p** — 8-bit YUV conversion, full chroma. Isolates rounding.
* **yuv420p** — chroma subsampled. The suspect, and what the ladder uses.

Measured 2026-08-30 on `alcaraz_highlights/scene_000`, 8 frames of 4K:
**inf / 53.69 / 44.44 dB**. So the anchor arm cannot exceed ~44.44 dB on this
clip whatever it spends, and av1 at QP 5 is 0.21 dB under that — transparent,
and the measurement cannot see it.

**This applies to one arm only.** PointStream's delivered frames are assembled
in RGB and scored against the RGB source directly, so its curve is not capped
here — which is why it reads 46.55 dB, above a ceiling the anchor structurally
cannot cross. The bias therefore runs *toward* PointStream: at a given quality
the anchor appears to need more bytes than it does, so every paired BD-rate so
far understates PointStream's loss rather than overstating it.

    conda run -n pointstream --no-capture-output \
        python -m experiments.tier.ceiling_control
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tier.clip import load_tier_clip
from experiments.tier.ladder import pooled_psnr
from src.components.codec import tools
from src.components.codec.frames import even_size
from src.contracts import paths as ps_paths

OUT_DIR = ps_paths.outputs() / "bp29-low-rate"

#: `(label, ffmpeg pixel format)`, ordered identical -> mild -> suspect. The
#: first is the null: it must come back `inf`.
FORMATS: tuple[tuple[str, str], ...] = (
    ("rgb-control", "gbrp"),
    ("yuv444p", "yuv444p"),
    ("yuv420p", "yuv420p"),
)


def lossless_roundtrip(
    frames: np.ndarray, *, pix_fmt: str, ffmpeg_path: str, fps: float = 25.0
) -> np.ndarray:
    """``frames`` through ``pix_fmt`` and back, with a lossless codec throughout."""
    clip = np.ascontiguousarray(np.asarray(frames, dtype=np.uint8))
    count, height, width, _ = clip.shape
    with tempfile.TemporaryDirectory() as tmp:
        mid = Path(tmp) / "mid.mkv"
        subprocess.run(
            [ffmpeg_path, "-hide_banner", "-loglevel", "error", "-y",
             "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-framerate", str(fps),
             "-i", "-", "-pix_fmt", pix_fmt, "-c:v", "ffv1", str(mid)],
            input=clip.tobytes(), capture_output=True, check=True,
        )
        out = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-loglevel", "error",
             "-i", str(mid), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            capture_output=True, check=True,
        )
    got = np.frombuffer(out.stdout, dtype=np.uint8)
    return got[: count * height * width * 3].reshape(count, height, width, 3)


def main() -> int:
    clip = load_tier_clip(video="alcaraz_highlights", scene="scene_000", n_frames=8)
    reference = even_size(np.ascontiguousarray(np.asarray(clip.frames, dtype=np.uint8)))
    ffmpeg = tools.resolve_ffmpeg()
    print(f"clip {clip.video}/{clip.scene} x{reference.shape[0]}", flush=True)
    print(f"ffmpeg {ffmpeg.path} | {ffmpeg.version}", flush=True)

    results: dict[str, Any] = {}
    for label, pix_fmt in FORMATS:
        decoded = lossless_roundtrip(
            reference, pix_fmt=pix_fmt, ffmpeg_path=ffmpeg.path
        )
        luma = pooled_psnr(reference, decoded, luma=True)
        rgb = pooled_psnr(reference, decoded)
        results[label] = {"pix_fmt": pix_fmt, "luma_dB": luma, "rgb_dB": rgb}
        print(f"  {label:12s} -> Y-PSNR {luma:7.2f} dB   RGB-PSNR {rgb:7.2f} dB", flush=True)

    ceiling = results["yuv420p"]["luma_dB"]
    payload = {
        "brief": "BP29 §2 (wave 8, Stream C) — where the anchor's quality ceiling lives",
        "method": (
            "coded_roundtrip's colour path with a LOSSLESS codec (ffv1) in place "
            "of the rung's encoder, so anything lost is the pixel format"
        ),
        "clip": clip.describe(),
        "ffmpeg": {"path": ffmpeg.path, "version": ffmpeg.version},
        "results": results,
        "anchor_ceiling_dB": ceiling,
        "reading": (
            f"The anchor arm cannot exceed {ceiling:.2f} dB on this clip at any "
            "bitrate. PointStream's arm is scored in RGB against the RGB source "
            "and is not subject to it, so the two arms are not on the same "
            "interface at the high-quality end. The bias runs toward PointStream."
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = OUT_DIR / "ceiling-control.json"
    destination.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
