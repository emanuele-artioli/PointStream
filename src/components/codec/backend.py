"""Registry-facing codec backend. Construction stays in this module.

``CodecBackend`` is what ``BackendSpec.target`` points at. The encode path
lives in ``encode.py`` so importing the registry table does not import numpy
or reach for encoder binaries.
"""

from __future__ import annotations

from pathlib import Path

from src.contracts.codecs import EncodeRequest, RateControl
from src.components.codec.encode import EncodeRecord, RoiArm, decode, encode
from src.components.codec.roi import BlockRoiMap

_DEFAULT_QP = 32


class CodecBackend:
    """One rung of the codec ladder, selected by config name."""

    def __init__(self, codec_name: str) -> None:
        self.codec_name = codec_name

    def encode(
        self,
        source: Path,
        dest: Path,
        request: EncodeRequest | None = None,
        *,
        roi: BlockRoiMap | None = None,
        roi_arm: RoiArm = "auto",
        frames: int | None = None,
        work_dir: Path | None = None,
    ) -> EncodeRecord:
        """Encode ``source`` with ``request`` (or a default QP request for this rung)."""
        if request is None:
            request = EncodeRequest(
                codec_name=self.codec_name,
                rate_control=RateControl.QP,
                rate=_DEFAULT_QP,
            )
        if request.codec_name != self.codec_name:
            raise ValueError(
                f"backend {self.codec_name!r} was given a request for {request.codec_name!r}"
            )
        return encode(
            source,
            dest,
            request,
            roi=roi,
            roi_arm=roi_arm,
            frames=frames,
            work_dir=work_dir,
        )

    def decode(self, bitstream: Path, dest: Path, request: EncodeRequest) -> None:
        if request.codec_name != self.codec_name:
            raise ValueError(
                f"backend {self.codec_name!r} was given a request for {request.codec_name!r}"
            )
        decode(bitstream, dest, request)
