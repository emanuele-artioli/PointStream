"""Transport and bitstream codec for the corrective residual.

Preserves the actual encoded bitstream bytes, decoder settings, and allows
source-free fresh-process client reconstruction and ledger reconciliation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import time

import numpy as np

from src.components.codec import tools
from src.components.codec.encode import BITSTREAM_SUFFIX, decode, encode
from src.components.codec.frames import even_size
from src.components.codec.measure import _run_ffmpeg
from src.contracts.codecs import EncodeRequest
from src.pipeline.residual.lossy import decode_lossy


@dataclass(frozen=True)
class TransmittedResidual:
    """The serialized wire payload for residual correction.

    Carries the actual compressed bitstream bytes and all required side
    information for a standalone client to decode.
    """

    bitstream: bytes
    codec_name: str
    mode: str = "clipped"
    shape: tuple[int, int, int, int] = (0, 0, 0, 0)
    pix_fmt: str = "yuv420p"
    scale: float = 1.0
    offset: float = 128.0
    is_coded: bool = True
    raw_frames: np.ndarray | None = None
    fps: float = 25.0
    preset: str | None = None
    qp: int | None = None
    encode_seconds: float = 0.0
    decode_seconds: float = 0.0
    tool_path: str = ""
    tool_version: str = ""

    @property
    def byte_count(self) -> int:
        if self.is_coded:
            return len(self.bitstream)
        if self.raw_frames is not None:
            return int(self.raw_frames.nbytes)
        return len(self.bitstream)

    @property
    def is_absent(self) -> bool:
        return self.byte_count == 0 and not self.is_coded and self.raw_frames is None


def encode_residual_to_bitstream(
    frames: np.ndarray,
    request: EncodeRequest,
    *,
    mode: str = "clipped",
    scale: float = 1.0,
    offset: float = 128.0,
    fps: float = 25.0,
    work_dir: Path | None = None,
) -> tuple[TransmittedResidual, np.ndarray]:
    """Encode residual uint8 frames to bitstream bytes and decode them back.

    Returns:
        (transmitted_residual, decoded_frames)
    """
    clip = np.ascontiguousarray(np.asarray(frames, dtype=np.uint8))
    if clip.ndim != 4 or clip.shape[3] != 3:
        raise ValueError(f"expected (T,H,W,3) uint8, got {tuple(clip.shape)}")
    orig_shape = tuple(clip.shape)
    clip_even = even_size(clip)
    count, height, width, _ = clip_even.shape
    ffmpeg = tools.resolve_ffmpeg()

    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        root = Path(tmp)
        lossless = root / "payload.mkv"
        _run_ffmpeg(
            [
                ffmpeg.path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-framerate",
                str(fps),
                "-i",
                "-",
                "-c:v",
                "ffv1",
                str(lossless),
            ],
            clip_even.tobytes(),
        )
        suffix = BITSTREAM_SUFFIX.get(request.codec_name, ".mp4")
        dest = root / f"payload{suffix}"
        record = encode(lossless, dest, request)
        bitstream_bytes = dest.read_bytes()

        back = root / "decoded.mkv"
        decode_started = time.perf_counter()
        decode(dest, back, request)
        raw = _run_ffmpeg(
            [
                ffmpeg.path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(back),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
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

    decoded_cut = decoded[: orig_shape[0], : orig_shape[1], : orig_shape[2], :]
    transmitted = TransmittedResidual(
        bitstream=bitstream_bytes,
        codec_name=request.codec_name,
        mode=mode,
        shape=orig_shape,
        pix_fmt=request.pix_fmt,
        scale=scale,
        offset=offset,
        is_coded=True,
        fps=fps,
        preset=record.preset,
        qp=record.rate,
        encode_seconds=float(record.encode_seconds),
        decode_seconds=float(decode_seconds),
        tool_path=record.tool_path,
        tool_version=record.tool_version,
    )
    return transmitted, decoded_cut


def decode_residual_stream(
    transmitted: TransmittedResidual,
    *,
    work_dir: Path | None = None,
) -> np.ndarray:
    """Decode transmitted bitstream bytes to signed residual differences.

    Returns:
        (T, H, W, C) signed int16 differences.
    """
    if not transmitted.is_coded:
        if transmitted.raw_frames is not None:
            raw = transmitted.raw_frames
            if raw.dtype == np.int16:
                return raw
            return decode_lossy(
                raw,
                mode=transmitted.mode,
                scale=transmitted.scale,
                offset=transmitted.offset,
            )
        raise ValueError("TransmittedResidual has neither bitstream nor raw_frames")

    bitstream_bytes = transmitted.bitstream
    if not bitstream_bytes or len(bitstream_bytes) < 16:
        raise ValueError(
            f"Corrupted or truncated residual bitstream: got {len(bitstream_bytes)} bytes"
        )

    orig_shape = transmitted.shape
    if len(orig_shape) != 4 or orig_shape[3] != 3:
        raise ValueError(f"Invalid transmitted residual shape: {orig_shape}")

    count, orig_h, orig_w, _ = orig_shape
    # Even-pad dimensions that were used during encode
    even_h = orig_h + (orig_h % 2)
    even_w = orig_w + (orig_w % 2)

    ffmpeg = tools.resolve_ffmpeg()
    suffix = BITSTREAM_SUFFIX.get(transmitted.codec_name, ".mp4")

    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        root = Path(tmp)
        stream_path = root / f"stream{suffix}"
        stream_path.write_bytes(bitstream_bytes)

        raw_cmd = [
            ffmpeg.path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(stream_path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
        try:
            raw = _run_ffmpeg(raw_cmd, None)
        except Exception as e:
            raise ValueError(f"Failed to decode residual bitstream: {e}") from e

    decoded = np.frombuffer(raw, dtype=np.uint8)
    expected_even = count * even_h * even_w * 3
    if decoded.size < expected_even:
        # Check if decoded without padding
        expected_orig = count * orig_h * orig_w * 3
        if decoded.size >= expected_orig:
            decoded_frames = decoded[:expected_orig].reshape(count, orig_h, orig_w, 3)
            return decode_lossy(
                decoded_frames,
                mode=transmitted.mode,
                scale=transmitted.scale,
                offset=transmitted.offset,
            )
        raise ValueError(
            f"Residual bitstream decoded to {decoded.size} bytes; expected >= {expected_orig}"
        )

    decoded = decoded[:expected_even].reshape(-1, even_h, even_w, 3)
    decoded_cut = decoded[:count, :orig_h, :orig_w, :]
    return decode_lossy(
        decoded_cut,
        mode=transmitted.mode,
        scale=transmitted.scale,
        offset=transmitted.offset,
    )
