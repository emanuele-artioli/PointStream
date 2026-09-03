"""Sidecar codecs for a background plate: jpeg, png, roi-video, av1, vvc.

This is a separate axis from the transmission strategy. Splitting them is
what makes ``{method: panorama-delta, codec: roi-video}`` expressible. The
background registry keys on strategy (``background.method``); codec validity
is checked here.

**Why a modern intra codec belongs on this axis.** The plate is 88-91% of
PointStream's payload at every rung of every sweep (`plans/done/BP24-findings.md`
§13) and it was coded as a JPEG. Measured on one 4K plate at matched fidelity
near 38 dB (`outputs/bp24-ladder/plate-probe.json`): JPEG 283,431 B, av1-intra
79,726 B, vvc-intra 68,477 B — 3.6x to 4.1x on nearly all of the payload, for
no architectural change. A modern intra frame is what AVIF and HEIC already are.

It went unnoticed because ``background.jpeg_quality`` is a knob on the codec
that was *already chosen*: sweeping it explores quality within JPEG and never
questions JPEG (§16). `BP29` Stream B adds ``av1`` and ``vvc`` so the axis has
something worth sweeping.

**The codec is a parameter, never a constant.** `IntraCodecSidecar` takes its
codec name from ``background.codec``. When the paired ladder uses this, the
plate must sit on the *same codec as the anchor* in each pair, or the pairing
discipline that makes a BD-rate readable breaks (`plans/done/BP24-findings.md` §1).
Hardcoding av1 here would break it silently.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
from typing import Final, Protocol

import cv2
import numpy as np

from src.contracts.codecs import EncodeRequest, RateControl
from src.contracts.errors import ConfigValueError

SIDECAR_JPEG: Final = "jpeg"
SIDECAR_PNG: Final = "png"
SIDECAR_ROI_VIDEO: Final = "roi-video"
SIDECAR_AV1: Final = "av1"
SIDECAR_VVC: Final = "vvc"

#: Codec names that route to `IntraCodecSidecar`. These are the same registry
#: keys `src.contracts.codecs` uses, deliberately: ``{background.codec: av1,
#: residual.codec: av1}`` has to mean the same encoder on both, or a paired
#: arm is not paired.
INTRA_SIDECAR_CODECS: Final[frozenset[str]] = frozenset({SIDECAR_AV1, SIDECAR_VVC})

ALL_SIDECAR_CODECS: Final[frozenset[str]] = frozenset(
    {SIDECAR_JPEG, SIDECAR_PNG, SIDECAR_ROI_VIDEO} | INTRA_SIDECAR_CODECS
)

#: Preset per intra codec, matching `src.components.codec.measure.PRESETS`, so a
#: plate coded here and an anchor coded there ran at the same encoder effort.
#: These are **not** equal effort across codecs; that is why a pair must hold the
#: codec fixed rather than compare av1 against vvc.
INTRA_PRESETS: Final[dict[str, str]] = {SIDECAR_AV1: "10", SIDECAR_VVC: "faster"}

#: Default QP for an intra plate. Chosen to sit near the fidelity the previous
#: default (`jpeg_quality=50`, 40.04 dB on the BP24 plate) delivered: av1 qp45
#: measured 40.83 dB there for 143,925 B against JPEG's 345,558 B. It is a
#: starting point for a sweep, not a tuned operating point, and QP scales differ
#: between av1 and vvc so the same number is *not* the same quality on both.
DEFAULT_INTRA_QP: Final = 45

#: Resolved by path, not by name. This host has carried two ffmpeg builds.
FFMPEG_BIN: Final = Path("/opt/local/bin/ffmpeg")
FFMPEG_VERSION: Final = "n7.1.1"

# Fixed fractional (x, y, w, h, qoffset) regions for tennis broadcast frames.
# Not detected — a heuristic so roi-video has somewhere to steer bits. The
# same limitation the pre-rewrite encoder documented.
_DEFAULT_TENNIS_ROI: Final[tuple[tuple[float, float, float, float, float], ...]] = (
    (0.00, 0.00, 0.22, 0.12, -0.4),
    (0.42, 0.25, 0.16, 0.30, -0.4),
    (0.00, 0.80, 0.18, 0.20, -0.4),
    (0.82, 0.80, 0.18, 0.20, -0.4),
)


class SidecarCodec(Protocol):
    """Anything that turns a BGR plate into bytes on the wire."""

    name: str

    @property
    def codec_id(self) -> str:
        """Name plus settings. Two ids match only if they encode the same way."""

    def encode(self, image_bgr: np.ndarray) -> bytes:
        """Encode a BGR uint8 plate."""

    def decode(self, payload: bytes) -> np.ndarray:
        """Decode sidecar bytes back to BGR uint8."""


def normalize_sidecar(name: str) -> str:
    """Canonical sidecar name, or a config error listing what would work."""
    raw = name.strip().lower()
    aliases = {
        "jpg": SIDECAR_JPEG,
        "roi_video": SIDECAR_ROI_VIDEO,
        "av1-intra": SIDECAR_AV1,
        "vvc-intra": SIDECAR_VVC,
        "h266": SIDECAR_VVC,
    }
    canonical = aliases.get(raw, raw)
    if canonical not in ALL_SIDECAR_CODECS:
        known = ", ".join(sorted(ALL_SIDECAR_CODECS))
        raise ConfigValueError(
            "background.codec",
            f"{name!r} is not a sidecar codec. Known: {known}.",
        )
    return canonical


def _as_bgr(image_bgr: np.ndarray) -> np.ndarray:
    image = np.asarray(image_bgr, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected a BGR plate [H, W, 3], got {tuple(image.shape)}.")
    return image


class JpegSidecar:
    name = SIDECAR_JPEG

    def __init__(self, quality: int = 50) -> None:
        if not 1 <= quality <= 100:
            raise ConfigValueError(
                "background.jpeg_quality",
                f"JPEG quality must be 1-100, got {quality}.",
            )
        self._quality = int(quality)

    @property
    def codec_id(self) -> str:
        return f"jpeg:{self._quality}"

    def encode(self, image_bgr: np.ndarray) -> bytes:
        ok, buffer = cv2.imencode(
            ".jpg",
            _as_bgr(image_bgr),
            [int(cv2.IMWRITE_JPEG_QUALITY), self._quality],
        )
        if not ok:
            raise RuntimeError("Failed to encode JPEG background sidecar.")
        return bytes(buffer)

    def decode(self, payload: bytes) -> np.ndarray:
        decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None or decoded.size == 0:
            raise RuntimeError("Failed to decode JPEG background sidecar.")
        return np.asarray(decoded, dtype=np.uint8)


class PngSidecar:
    name = SIDECAR_PNG

    def __init__(self, compression: int = 3) -> None:
        if not 0 <= compression <= 9:
            raise ConfigValueError(
                "background.png_compression",
                f"PNG compression must be 0-9, got {compression}.",
            )
        self._compression = int(compression)

    @property
    def codec_id(self) -> str:
        return f"png:{self._compression}"

    def encode(self, image_bgr: np.ndarray) -> bytes:
        ok, buffer = cv2.imencode(
            ".png",
            _as_bgr(image_bgr),
            [int(cv2.IMWRITE_PNG_COMPRESSION), self._compression],
        )
        if not ok:
            raise RuntimeError("Failed to encode PNG background sidecar.")
        return bytes(buffer)

    def decode(self, payload: bytes) -> np.ndarray:
        decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None or decoded.size == 0:
            raise RuntimeError("Failed to decode PNG background sidecar.")
        return np.asarray(decoded, dtype=np.uint8)


def _ffmpeg() -> str:
    if FFMPEG_BIN.is_file():
        return str(FFMPEG_BIN)
    raise RuntimeError(
        f"roi-video sidecar needs ffmpeg at {FFMPEG_BIN} (version {FFMPEG_VERSION}); "
        f"that path is not a file on this host."
    )


class RoiVideoSidecar:
    """Single-frame libx264 encode with ``addroi`` bit steering.

    ``addroi`` is honoured by libx264/libx265, not by libsvtav1. This rung
    therefore uses libx264 regardless of the residual codec.
    """

    name = SIDECAR_ROI_VIDEO

    def __init__(
        self,
        crf: int = 30,
        preset: str = "veryfast",
        regions: tuple[tuple[float, float, float, float, float], ...] | None = None,
    ) -> None:
        if crf < 0:
            raise ConfigValueError("background.roi_crf", f"CRF must be >= 0, got {crf}.")
        self._crf = int(crf)
        self._preset = str(preset)
        self._regions = regions if regions is not None else _DEFAULT_TENNIS_ROI

    @property
    def codec_id(self) -> str:
        return (
            f"roi-video:libx264:crf{self._crf}:{self._preset}:regions{len(self._regions)}"
        )

    def _filters(self, width: int, height: int) -> str:
        parts = []
        for x_frac, y_frac, w_frac, h_frac, qoffset in self._regions:
            x = max(0, min(width - 1, int(round(x_frac * width))))
            y = max(0, min(height - 1, int(round(y_frac * height))))
            w = max(1, min(width - x, int(round(w_frac * width))))
            h = max(1, min(height - y, int(round(h_frac * height))))
            parts.append(f"addroi={x}:{y}:{w}:{h}:{qoffset}")
        return ",".join(parts)

    def encode(self, image_bgr: np.ndarray) -> bytes:
        image = _as_bgr(image_bgr)
        height, width = int(image.shape[0]), int(image.shape[1])
        even_h = height - (height % 2)
        even_w = width - (width % 2)
        if even_h != height or even_w != width:
            image = np.ascontiguousarray(image[:even_h, :even_w])
            height, width = even_h, even_w
        vf = self._filters(width, height)
        with tempfile.TemporaryDirectory(prefix="ps_b4_roi_") as tmp_dir:
            tmp = Path(tmp_dir)
            src = tmp / "src.png"
            out = tmp / "out.mp4"
            if not cv2.imwrite(str(src), image):
                raise RuntimeError("Failed to write intermediate PNG for roi-video sidecar.")
            cmd = [
                _ffmpeg(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-loop",
                "1",
                "-i",
                str(src),
                "-frames:v",
                "1",
                "-vf",
                f"{vf},format=yuv420p" if vf else "format=yuv420p",
                "-c:v",
                "libx264",
                "-crf",
                str(self._crf),
                "-preset",
                self._preset,
                "-y",
                str(out),
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return out.read_bytes()

    def decode(self, payload: bytes) -> np.ndarray:
        with tempfile.TemporaryDirectory(prefix="ps_b4_roi_dec_") as tmp_dir:
            tmp = Path(tmp_dir)
            src = tmp / "in.mp4"
            out = tmp / "frame0.png"
            src.write_bytes(payload)
            cmd = [
                _ffmpeg(),
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(src),
                "-vframes",
                "1",
                "-y",
                str(out),
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            decoded = cv2.imread(str(out), cv2.IMREAD_COLOR)
            if decoded is None or decoded.size == 0:
                raise RuntimeError("Failed to decode roi-video background sidecar.")
            return np.asarray(decoded, dtype=np.uint8)


def _even(image: np.ndarray) -> np.ndarray:
    """Crop to even dimensions. 4:2:0 has no odd row or column to give."""
    height, width = int(image.shape[0]), int(image.shape[1])
    even_h, even_w = height - (height % 2), width - (width % 2)
    if (even_h, even_w) == (height, width):
        return np.ascontiguousarray(image)
    return np.ascontiguousarray(image[:even_h, :even_w])


def _run(argv: list[str], stdin_bytes: bytes | None) -> None:
    result = subprocess.run(argv, input=stdin_bytes, capture_output=True)
    if result.returncode != 0:
        detail = (result.stderr or b"").decode("utf-8", "replace").strip()
        raise RuntimeError(f"ffmpeg failed ({result.returncode}): {detail[-600:]}")


class IntraCodecSidecar:
    """One still through a modern video codec: av1 (SVT-AV1) or vvc (libvvenc).

    A wrapper, not a new encoder path. The encode goes through
    `src.components.codec.encode.encode` with a normal `EncodeRequest`, so the
    plate sees exactly the argv an anchor of the same codec sees — same command
    builder, same tool resolution by path and version, same capability checks.
    That is the whole point: a plate coded by a *different* route than the
    anchor is not a paired arm.

    **The decode is the anchor's decode, not a lookalike.** It goes through
    `src.components.codec.encode.decode`, which names ``-c:v ffv1`` — ffmpeg
    without an explicit ``-c:v`` picks the muxer's default encoder, libx264 for
    Matroska, and `coded_roundtrip` spent a whole BP24 sweep handing back frames
    that had been through the rung's codec *and then* through x264, pinning every
    measured quality near x264's ceiling while the rate fell tenfold
    (`plans/done/BP24-findings.md` §14).

    Reusing it rather than writing a shorter one is not fussiness. The first
    version of this class decoded the bitstream straight to a PNG, which is one
    conversion instead of two and looks strictly better. On av1 it was
    bit-identical. On **vvc it lost 0.57 dB at qp25**, because libvvenc emits
    10-bit 4:2:0 and going straight from 10-bit YUV to 8-bit RGB is a different
    conversion from the anchor's 10-bit YUV to 8-bit YUV to 8-bit RGB. A plate
    decoded better or worse than the anchor it is paired against is not a paired
    arm, so the sidecar takes the anchor's path and reproduces
    `outputs/bp24-ladder/plate-probe.json` exactly on all eight rungs. Measured
    2026-08-30: the ffv1 intermediate rendered to a PNG is bit-identical to
    rendering it to rawvideo, which is what lets the sidecar drop the frame
    dimensions a raw decode would need.
    """

    name: str

    def __init__(
        self,
        codec_name: str,
        qp: int = DEFAULT_INTRA_QP,
        preset: str | None = None,
    ) -> None:
        canonical = normalize_sidecar(codec_name)
        if canonical not in INTRA_SIDECAR_CODECS:
            raise ConfigValueError(
                "background.codec",
                f"{codec_name!r} is not an intra sidecar codec. "
                f"Known: {', '.join(sorted(INTRA_SIDECAR_CODECS))}.",
            )
        # One source of truth for the QP range: the codec layer's own table.
        from src.components.codec.encode import QP_BOUNDS

        low, high = QP_BOUNDS[canonical]
        if not low <= int(qp) <= high:
            raise ConfigValueError(
                "background.intra_qp",
                f"QP for {canonical} must be {low}-{high}, got {qp}.",
            )
        self.name = canonical
        self._codec_name = canonical
        self._qp = int(qp)
        self._preset = str(preset) if preset is not None else INTRA_PRESETS[canonical]

    @property
    def codec_id(self) -> str:
        return f"{self._codec_name}:intra:qp{self._qp}:{self._preset}"

    @property
    def qp(self) -> int:
        return self._qp

    @property
    def preset(self) -> str:
        return self._preset

    def probe_encoder(self) -> tuple[str, str]:
        """``(path, version)`` of the binary that would run. Provenance, not decor.

        This host has carried two builds of the same encoder with different
        capabilities, so a size without the binary that produced it is not
        evidence. Raises `FileNotFoundError` when the encoder is absent, which
        is the honest answer to "is vvc available here?".
        """
        from src.components.codec import tools

        resolved = tools.resolve_encoder(self._codec_name)
        return resolved.path, resolved.version

    def _request(self) -> EncodeRequest:
        return EncodeRequest(
            codec_name=self._codec_name,
            rate_control=RateControl.QP,
            rate=self._qp,
            preset=self._preset,
            pix_fmt="yuv420p",
        )

    def encode(self, image_bgr: np.ndarray) -> bytes:
        from src.components.codec import tools
        from src.components.codec.encode import BITSTREAM_SUFFIX
        from src.components.codec.encode import encode as codec_encode

        image = _even(_as_bgr(image_bgr))
        height, width = int(image.shape[0]), int(image.shape[1])
        ffmpeg = tools.resolve_ffmpeg()
        request = self._request()
        with tempfile.TemporaryDirectory(prefix="ps_intra_enc_") as tmp_dir:
            root = Path(tmp_dir)
            # Lossless RGB first, so the encoder — not an intermediate — does
            # the only colour conversion in the chain.
            lossless = root / "plate.mkv"
            _run(
                [
                    ffmpeg.path, "-hide_banner", "-loglevel", "error", "-y",
                    "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{width}x{height}", "-framerate", "25",
                    "-i", "-", "-frames:v", "1", "-c:v", "ffv1", str(lossless),
                ],
                np.ascontiguousarray(image[:, :, ::-1]).tobytes(),
            )
            dest = root / f"plate{BITSTREAM_SUFFIX[self._codec_name]}"
            codec_encode(lossless, dest, request)
            return dest.read_bytes()

    def decode(self, payload: bytes) -> np.ndarray:
        from src.components.codec import tools
        from src.components.codec.encode import BITSTREAM_SUFFIX
        from src.components.codec.encode import decode as codec_decode

        if not payload:
            raise RuntimeError(f"Empty {self._codec_name} background sidecar payload.")
        ffmpeg = tools.resolve_ffmpeg()
        with tempfile.TemporaryDirectory(prefix="ps_intra_dec_") as tmp_dir:
            root = Path(tmp_dir)
            src = root / f"plate{BITSTREAM_SUFFIX[self._codec_name]}"
            src.write_bytes(payload)
            # Step one is the anchor's own decode: lossless ffv1 at the
            # request's pix_fmt, through the shared command builder.
            lossless = root / "plate.mkv"
            codec_decode(src, lossless, self._request(), ffmpeg=ffmpeg)
            # Step two only changes container. PNG is lossless and needs no
            # frame dimensions, which a sidecar decode does not have.
            out = root / "plate.png"
            _run(
                [
                    ffmpeg.path, "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(lossless), "-frames:v", "1",
                    "-c:v", "png", "-pix_fmt", "rgb24", str(out),
                ],
                None,
            )
            decoded = cv2.imread(str(out), cv2.IMREAD_COLOR)
        if decoded is None or decoded.size == 0:
            raise RuntimeError(
                f"Failed to decode {self._codec_name} background sidecar."
            )
        return np.asarray(decoded, dtype=np.uint8)


def build_sidecar(
    name: str,
    *,
    jpeg_quality: int = 50,
    png_compression: int = 3,
    roi_crf: int = 30,
    roi_preset: str = "veryfast",
    intra_qp: int = DEFAULT_INTRA_QP,
    intra_preset: str | None = None,
) -> SidecarCodec:
    """Construct the sidecar named by ``background.codec``.

    ``name`` carries the codec. A caller pairing the plate against an anchor
    passes the anchor's codec here rather than a constant.
    """
    canonical = normalize_sidecar(name)
    if canonical == SIDECAR_JPEG:
        return JpegSidecar(quality=jpeg_quality)
    if canonical == SIDECAR_PNG:
        return PngSidecar(compression=png_compression)
    if canonical in INTRA_SIDECAR_CODECS:
        return IntraCodecSidecar(canonical, qp=intra_qp, preset=intra_preset)
    return RoiVideoSidecar(crf=roi_crf, preset=roi_preset)
