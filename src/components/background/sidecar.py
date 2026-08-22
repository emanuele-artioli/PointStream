"""Sidecar codecs for a background plate: jpeg, png, roi-video.

This is a separate axis from the transmission strategy. Splitting them is
what makes ``{method: panorama-delta, codec: roi-video}`` expressible. The
background registry keys on strategy (``background.method``); codec validity
is checked here.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
from typing import Final, Protocol

import cv2
import numpy as np

from src.contracts.errors import ConfigValueError

SIDECAR_JPEG: Final = "jpeg"
SIDECAR_PNG: Final = "png"
SIDECAR_ROI_VIDEO: Final = "roi-video"

ALL_SIDECAR_CODECS: Final[frozenset[str]] = frozenset(
    {SIDECAR_JPEG, SIDECAR_PNG, SIDECAR_ROI_VIDEO}
)

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
    aliases = {"jpg": SIDECAR_JPEG, "roi_video": SIDECAR_ROI_VIDEO}
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


def build_sidecar(
    name: str,
    *,
    jpeg_quality: int = 50,
    png_compression: int = 3,
    roi_crf: int = 30,
    roi_preset: str = "veryfast",
) -> SidecarCodec:
    """Construct the sidecar named by ``background.codec``."""
    canonical = normalize_sidecar(name)
    if canonical == SIDECAR_JPEG:
        return JpegSidecar(quality=jpeg_quality)
    if canonical == SIDECAR_PNG:
        return PngSidecar(compression=png_compression)
    return RoiVideoSidecar(crf=roi_crf, preset=roi_preset)
