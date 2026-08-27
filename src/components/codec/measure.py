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

from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np

from src.components.codec.encode import BITSTREAM_SUFFIX, encode
from src.components.codec.frames import even_size, rgb_to_luma
from src.components.codec.y4m import from_luma, write
from src.contracts.codecs import EncodeRequest, RateControl

# Presets mirror `experiments/headroom/ladder.py`, so a coded size measured here
# is comparable with BP21's real-4K ladder rather than a second convention.
_PRESETS = {"avc": "veryfast", "hevc": "ultrafast", "av1": "10", "vvc": "faster"}


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
    if codec_name not in _PRESETS:
        raise ValueError(f"no preset for codec {codec_name!r}; known: {sorted(_PRESETS)}")

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
                preset=_PRESETS[codec_name],
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
