"""Coded size of a pixel payload — the thing that makes a byte count a rate.

`BP24`. Until this existed the runner's codec stage was an identity round-trip
and every byte count it reported was raw pixels (`PLAN.md` §2.16). A raw count
is not a rate and must never be divided by the source size and called a
compression ratio.

The encoder is resolved by **path and version** and both are returned, because
this host has carried two builds of the same encoder with different
capabilities: a size without provenance is not evidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

from src.components.codec.encode import BITSTREAM_SUFFIX, decode, encode, sweep_qp
from src.components.codec import tools
from src.components.codec.frames import even_size, rgb_to_luma
from src.components.codec.y4m import from_luma, read, write
from src.components.metrics.bd_rate import RDCurve
from src.contracts.codecs import EncodeRequest, RateControl

# Presets mirror `experiments/headroom/ladder.py`, so a coded size measured here
# is comparable with BP21's real-4K ladder rather than a second convention.
#
# They are NOT equal effort across codecs, and a cross-codec BD-rate from them
# understates the newer codec. Measured 2026-08-27 on four real 960x540 frames,
# HEVC over AVC came out at -4.2% BD-rate where the literature expects 30-50%:
# x264 `veryfast` against kvazaar `ultrafast` is not a fair fight. Fine for
# accounting one payload; for a codec-vs-codec claim, match effort first and say
# which presets were used.
PRESETS = {"avc": "veryfast", "hevc": "ultrafast", "av1": "10", "vvc": "faster"}


@dataclass(frozen=True)
class CodedSize:
    """A measured coded size, with the provenance that makes it citable."""

    byte_count: int
    raw_byte_count: int
    codec_name: str
    qp: int
    tool_path: str
    tool_version: str

    @property
    def ratio_to_raw(self) -> float:
        return self.byte_count / self.raw_byte_count if self.raw_byte_count else 0.0


def coded_size(
    frames: np.ndarray,
    *,
    codec_name: str,
    qp: int,
    fps: float = 25.0,
    work_dir: Path | None = None,
) -> CodedSize:
    """Encode ``frames`` and report what the bitstream actually cost.

    ``frames`` is ``(T, H, W, 3)`` RGB or ``(T, H, W)`` luma. A single still —
    a background plate — is just ``T == 1``.
    """
    clip = np.asarray(frames)
    if clip.ndim == 3:
        clip = clip[..., np.newaxis].repeat(3, axis=3)
    if clip.ndim != 4:
        raise ValueError(f"expected (T,H,W,3) or (T,H,W), got {tuple(clip.shape)}")
    if codec_name not in PRESETS:
        raise ValueError(f"no preset for codec {codec_name!r}; known: {sorted(PRESETS)}")

    raw = int(clip.nbytes)
    clip = even_size(clip)
    luma = rgb_to_luma(clip)

    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        root = Path(tmp)
        source = root / "payload.y4m"
        write(source, from_luma(luma, fps=fps))
        dest = root / f"payload{BITSTREAM_SUFFIX[codec_name]}"
        record = encode(
            source,
            dest,
            EncodeRequest(
                codec_name=codec_name,
                rate_control=RateControl.QP,
                rate=int(qp),
                preset=PRESETS[codec_name],
                pix_fmt="yuv420p",
            ),
        )
        return CodedSize(
            byte_count=int(record.size_bytes),
            raw_byte_count=raw,
            codec_name=codec_name,
            qp=int(qp),
            tool_path=record.tool_path,
            tool_version=record.tool_version,
        )


def _luma_psnr(reference: np.ndarray, decoded: np.ndarray) -> float:
    ref = reference.astype(np.float64)
    got = decoded.astype(np.float64)[: ref.shape[0]]
    mse = float(np.mean((ref - got) ** 2))
    return float("inf") if mse == 0 else 10.0 * float(np.log10((255.0**2) / mse))


def coded_curve(
    frames: np.ndarray,
    *,
    codec_name: str,
    qps: Sequence[int] = (32, 40, 46),
    fps: float = 25.0,
    work_dir: Path | None = None,
) -> RDCurve:
    """Sweep ``qps`` and return the rate-quality curve, not a single point.

    **Use this, not `coded_size`, for anything that compares.** A QP is an
    encoder knob, not a quality level: two codecs at the same QP land at
    different quality, so their bitrates are not comparable and a ratio between
    them means nothing. That confound is not hypothetical here — `BP21`'s VVC
    result collapsed on exactly it, and survived neither a common-QP nor a
    common-PSNR reading (`PLAN.md` §2.14).

    The honest comparison integrates over the overlapping quality range, which
    is what `src.components.metrics.bd_rate` does — and it refuses to return a
    number when that overlap is a sliver.
    """
    clip = np.asarray(frames)
    if clip.ndim == 3:
        clip = clip[..., np.newaxis].repeat(3, axis=3)
    if codec_name not in PRESETS:
        raise ValueError(f"no preset for codec {codec_name!r}; known: {sorted(PRESETS)}")
    clip = even_size(clip)
    luma = rgb_to_luma(clip)

    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        root = Path(tmp)
        source = root / "payload.y4m"
        write(source, from_luma(luma, fps=fps))
        request = EncodeRequest(
            codec_name=codec_name,
            rate_control=RateControl.QP,
            rate=int(qps[0]),
            preset=PRESETS[codec_name],
            pix_fmt="yuv420p",
        )
        records = sweep_qp(source, root, request, list(qps))
        rates: list[float] = []
        qualities: list[float] = []
        for record in records:
            decoded_path = root / f"decoded_qp{record.rate}.y4m"
            decode(record.output, decoded_path, request.replace_rate(int(record.rate or 0)))
            rates.append(float(record.size_bytes))
            qualities.append(_luma_psnr(luma, read(decoded_path).luma))
    return RDCurve(rates=tuple(rates), qualities=tuple(qualities), label=codec_name)


@dataclass(frozen=True)
class TimedRoundtrip:
    """One encode/decode of RGB frames, with the two halves timed separately.

    ``encode_seconds`` is the encoder's own wall clock from ``EncodeRecord``.
    ``decode_seconds`` is bitstream → RGB, including the ffmpeg pixel dump the
    quality metrics read. The lossless RGB wrap that feeds the encoder is not
    part of either, because it is not the codec under test.
    """

    size_bytes: int
    frames: np.ndarray
    encode_seconds: float
    decode_seconds: float
    tool_path: str
    tool_version: str
    preset: str | None
    qp: int | None


def timed_roundtrip(
    frames: np.ndarray,
    *,
    request: EncodeRequest,
    fps: float = 25.0,
    work_dir: Path | None = None,
) -> TimedRoundtrip:
    """Encode ``frames``, decode them back, and split encode vs decode time."""
    clip = np.ascontiguousarray(np.asarray(frames, dtype=np.uint8))
    if clip.ndim != 4 or clip.shape[3] != 3:
        raise ValueError(f"expected (T,H,W,3) uint8, got {tuple(clip.shape)}")
    clip = even_size(clip)
    count, height, width, _ = clip.shape
    ffmpeg = tools.resolve_ffmpeg()

    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        root = Path(tmp)
        lossless = root / "payload.mkv"
        _run_ffmpeg(
            [
                ffmpeg.path, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24",
                "-s", f"{width}x{height}", "-framerate", str(fps),
                "-i", "-", "-c:v", "ffv1", str(lossless),
            ],
            clip.tobytes(),
        )
        dest = root / f"payload{BITSTREAM_SUFFIX[request.codec_name]}"
        record = encode(lossless, dest, request)
        back = root / "decoded.mkv"
        decode_started = time.perf_counter()
        decode(dest, back, request)
        raw = _run_ffmpeg(
            [
                ffmpeg.path, "-hide_banner", "-loglevel", "error",
                "-i", str(back), "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
            ],
            None,
        )
        decode_seconds = time.perf_counter() - decode_started
    decoded = np.frombuffer(raw, dtype=np.uint8)
    usable = (decoded.size // (height * width * 3)) * height * width * 3
    decoded = decoded[:usable].reshape(-1, height, width, 3)
    if decoded.shape[0] < count:
        pad = np.repeat(decoded[-1:], count - decoded.shape[0], axis=0)
        decoded = np.concatenate([decoded, pad], axis=0)
    return TimedRoundtrip(
        size_bytes=int(record.size_bytes),
        frames=decoded[:count],
        encode_seconds=float(record.encode_seconds),
        decode_seconds=float(decode_seconds),
        tool_path=record.tool_path,
        tool_version=record.tool_version,
        preset=record.preset,
        qp=record.rate,
    )


def coded_roundtrip(
    frames: np.ndarray,
    *,
    request: EncodeRequest,
    fps: float = 25.0,
    work_dir: Path | None = None,
) -> tuple[int, np.ndarray]:
    """Encode ``frames``, decode them back, and report both cost and result.

    Returns ``(coded_bytes, decoded_frames)``. Both halves matter: a rate is
    only a rate-distortion point if the quality is measured on what the codec
    actually returned. Counting coded bytes while reconstructing from the
    pre-codec array puts the rate and the quality at different operating points
    — the mistake `BP24` made once on the background plate before catching it.

    **Colour-preserving.** The payload is written as lossless RGB first and the
    encoder converts to ``request.pix_fmt``. A residual carries a correction per
    channel, so the luma-only path used by `coded_size` would silently discard
    two thirds of it.
    """
    trip = timed_roundtrip(frames, request=request, fps=fps, work_dir=work_dir)
    return trip.size_bytes, trip.frames


def _run_ffmpeg(argv: list[str], stdin_bytes: bytes | None) -> bytes:
    result = subprocess.run(argv, input=stdin_bytes, capture_output=True)
    if result.returncode != 0:
        detail = (result.stderr or b"").decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffmpeg failed ({result.returncode}): {detail[:400]}")
    return result.stdout
