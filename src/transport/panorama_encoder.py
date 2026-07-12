from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import cv2
import numpy as np


class BasePanoramaEncoder(ABC):
    """Strategy interface for panorama sidecar image encoding."""

    name: str
    extension: str

    @property
    @abstractmethod
    def codec_id(self) -> str:
        """Identifier capturing codec name and quality/compression settings.

        Two encoders with the same `codec_id` are guaranteed to produce byte-identical
        output for the same input pixels; matching only `name` would miss a quality
        mismatch (e.g. jpeg q50 vs jpeg q90).
        """
        raise NotImplementedError

    @abstractmethod
    def encode_bytes(self, image_bgr: np.ndarray) -> bytes:
        """Encode a BGR panorama image to codec bytes, without touching disk."""
        raise NotImplementedError

    def decode_bytes(self, encoded: bytes) -> np.ndarray:
        """Decode codec bytes back to a BGR pixel array.

        Default implementation assumes a still-image codec (JPEG/PNG) decodable by
        OpenCV directly from an in-memory buffer. Codecs backed by a video container
        (e.g. `RoiVideoPanoramaEncoder`) override this.
        """
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None or decoded.size == 0:
            raise RuntimeError(f"Failed to decode round-tripped panorama bytes for codec '{self.name}'")
        return np.asarray(decoded, dtype=np.uint8)

    def encode(self, image_bgr: np.ndarray, output_stem: Path) -> Path:
        """Encode a BGR panorama image to disk and return the encoded path."""
        output_path = output_stem.with_suffix(self.extension)
        output_path.write_bytes(self.encode_bytes(image_bgr))
        return output_path


def _ensure_bgr_uint8(image_bgr: np.ndarray) -> np.ndarray:
    image = np.asarray(image_bgr, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "Invalid panorama image shape: "
            f"expected [H, W, 3], got {tuple(image.shape)}"
        )
    return image


class JpegPanoramaEncoder(BasePanoramaEncoder):
    name = "jpeg"
    extension = ".jpg"

    def __init__(self, quality: int = 90) -> None:
        self._quality = int(np.clip(int(quality), 1, 100))

    @property
    def codec_id(self) -> str:
        return f"jpeg:{self._quality}"

    def encode_bytes(self, image_bgr: np.ndarray) -> bytes:
        image = _ensure_bgr_uint8(image_bgr)
        ok, buffer = cv2.imencode(
            ".jpg",
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(self._quality)],
        )
        if not ok:
            raise RuntimeError("Failed to encode JPEG panorama bytes")
        return bytes(buffer)


class PngPanoramaEncoder(BasePanoramaEncoder):
    name = "png"
    extension = ".png"

    def __init__(self, compression: int = 3) -> None:
        self._compression = int(np.clip(int(compression), 0, 9))

    @property
    def codec_id(self) -> str:
        return f"png:{self._compression}"

    def encode_bytes(self, image_bgr: np.ndarray) -> bytes:
        image = _ensure_bgr_uint8(image_bgr)
        ok, buffer = cv2.imencode(
            ".png",
            image,
            [int(cv2.IMWRITE_PNG_COMPRESSION), int(self._compression)],
        )
        if not ok:
            raise RuntimeError("Failed to encode PNG panorama bytes")
        return bytes(buffer)


# Fixed fractional (x, y, w, h, qoffset) regions-of-interest for the "roi-video"
# background-layer rung (report 10 Phase 5.3). No court/net/scoreboard detector
# exists anywhere in src/ (verified: `grep -rniE "court|net_line|umpire|scoreboard"
# src/encoder/` turns up nothing) -- these are a fixed heuristic for a
# near-static, fixed-camera tennis broadcast frame, NOT derived from real
# detection. Coordinates are fractions of the panorama canvas so they scale
# with resolution; qoffset is ffmpeg's addroi convention (negative = better
# quality/lower QP in that region, range [-1, 1]).
_DEFAULT_TENNIS_ROI_REGIONS: tuple[tuple[float, float, float, float, float], ...] = (
    (0.00, 0.00, 0.22, 0.12, -0.4),  # scoreboard/broadcast-graphic corner (top-left)
    (0.42, 0.25, 0.16, 0.30, -0.4),  # umpire chair (near-net, camera-right heuristic)
    (0.00, 0.80, 0.18, 0.20, -0.4),  # ball-kid area, near baseline (left)
    (0.82, 0.80, 0.18, 0.20, -0.4),  # ball-kid area, near baseline (right)
)


def default_tennis_roi_regions() -> tuple[tuple[float, float, float, float, float], ...]:
    """Fixed heuristic ROI regions for `RoiVideoPanoramaEncoder` (see module-level
    comment above `_DEFAULT_TENNIS_ROI_REGIONS` for the documented limitation)."""
    return _DEFAULT_TENNIS_ROI_REGIONS


class RoiVideoPanoramaEncoder(BasePanoramaEncoder):
    """Background-layer rung 3 ("roi-video", report 10 Phase 5.3): encodes the
    panorama as a single-frame libx264 video with `addroi` region-of-interest
    side data steering bit allocation toward broadcast-relevant regions
    (scoreboard/umpire/ball-kid areas -- see `default_tennis_roi_regions`).

    `addroi` is only honored by libx264/libx265 (verified via `ffmpeg -h
    filter=addroi` and a synthetic round-trip test; libsvtav1, the project's
    default residual codec, has no equivalent mechanism) -- this rung
    necessarily uses libx264 regardless of the pipeline's configured
    `ffmpeg-codec`, same precedent as report 8's 2026-07-11 "libsvtav1 has no
    pixel-format flexibility" entry.
    """

    name = "roi-video"
    extension = ".mp4"

    def __init__(
        self,
        crf: int = 30,
        preset: str = "veryfast",
        roi_regions: tuple[tuple[float, float, float, float, float], ...] | None = None,
    ) -> None:
        self._crf = int(crf)
        self._preset = str(preset)
        self._regions = roi_regions if roi_regions is not None else default_tennis_roi_regions()

    @property
    def codec_id(self) -> str:
        return f"roi-video:libx264:crf{self._crf}:{self._preset}:regions{len(self._regions)}"

    def _region_filters(self, width: int, height: int) -> str:
        filters = []
        for x_frac, y_frac, w_frac, h_frac, qoffset in self._regions:
            x = max(0, min(width - 1, int(round(x_frac * width))))
            y = max(0, min(height - 1, int(round(y_frac * height))))
            w = max(1, min(width - x, int(round(w_frac * width))))
            h = max(1, min(height - y, int(round(h_frac * height))))
            filters.append(f"addroi={x}:{y}:{w}:{h}:{qoffset}")
        return ",".join(filters)

    def encode_bytes(self, image_bgr: np.ndarray) -> bytes:
        image = _ensure_bgr_uint8(image_bgr)
        height, width = image.shape[0], image.shape[1]
        # libx264 + yuv420p requires even width/height, but the stitched
        # panorama canvas (src.encoder.background_modeler's homography-based
        # canvas sizing) is not guaranteed even -- found via a real pipeline
        # run against assets/real_tennis.mp4 (3841x2190, odd width), which
        # failed with "width not divisible by 2". Truncate by <=1px/edge
        # instead of padding: this is deterministic from `image` alone, so
        # `decode_bytes` (which never sees the pre-encode shape -- the client
        # only has the transmitted bytes) reproduces the identical shape on
        # both sides without needing extra state (Residual Guarantee).
        even_height = height - (height % 2)
        even_width = width - (width % 2)
        if even_height != height or even_width != width:
            image = np.ascontiguousarray(image[:even_height, :even_width])
            height, width = even_height, even_width

        with tempfile.TemporaryDirectory(prefix="ps_roi_panorama_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            src_png = tmp_path / "src.png"
            out_mp4 = tmp_path / "out.mp4"
            ok = cv2.imwrite(str(src_png), image)
            if not ok:
                raise RuntimeError("Failed to write intermediate panorama PNG for roi-video encoding")

            vf = self._region_filters(width=width, height=height)
            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-loop", "1", "-i", str(src_png),
                "-frames:v", "1",
                "-vf", f"{vf},format=yuv420p" if vf else "format=yuv420p",
                "-c:v", "libx264", "-crf", str(self._crf), "-preset", self._preset,
                "-y", str(out_mp4),
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            return out_mp4.read_bytes()

    def decode_bytes(self, encoded: bytes) -> np.ndarray:
        with tempfile.TemporaryDirectory(prefix="ps_roi_panorama_decode_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            in_mp4 = tmp_path / "in.mp4"
            out_png = tmp_path / "frame0.png"
            in_mp4.write_bytes(encoded)

            cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-i", str(in_mp4), "-vframes", "1",
                "-y", str(out_png),
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            decoded = cv2.imread(str(out_png), cv2.IMREAD_COLOR)
            if decoded is None or decoded.size == 0:
                raise RuntimeError("Failed to decode roi-video panorama bytes")
            return np.asarray(decoded, dtype=np.uint8)


def round_trip_panorama(image_bgr: np.ndarray, encoder: BasePanoramaEncoder) -> tuple[bytes, np.ndarray]:
    """Encode then decode `image_bgr` through `encoder`, returning the sidecar bytes and resulting pixels.

    The encoder pipeline calls this so residuals are computed against the same lossy
    reconstruction the client will decode from the transmitted panorama sidecar, instead
    of against pre-codec pixels the client never sees (Residual Guarantee, CLAUDE.md).
    """
    encoded_bytes = encoder.encode_bytes(image_bgr)
    decoded = encoder.decode_bytes(encoded_bytes)
    return encoded_bytes, decoded


def compute_panorama_delta(current_bgr: np.ndarray, previous_bgr: np.ndarray) -> np.ndarray:
    """Signed pixel diff of `current` against `previous`, shifted into uint8 range.

    Shared by encoder (to build the diff image before compressing it) and by
    `apply_panorama_delta` (its exact inverse), so both sides of the Residual
    Guarantee use the identical arithmetic.

    Known limitation: the diff is carried in a uint8 image (so it can go
    through an ordinary still-image codec), which only represents per-channel
    deltas in [-128, 127] exactly; a larger jump between `previous` and
    `current` clips and is *not* perfectly invertible. This is an accepted
    trade-off for background-layer rung 2 ("panorama+delta", report 10 Phase
    5.3): a scene's background is expected to change only slightly between
    consecutive sub-chunks of the same point (scoreboard digits, minor crowd
    motion), not arbitrarily -- see reports/8_residual_guarantee_benchmarks_report.md.
    """
    current = _ensure_bgr_uint8(current_bgr).astype(np.int16)
    previous = _ensure_bgr_uint8(previous_bgr).astype(np.int16)
    if current.shape != previous.shape:
        raise ValueError(
            f"Panorama delta shape mismatch: current={current.shape}, previous={previous.shape}"
        )
    diff = np.clip(current - previous + 128, 0, 255)
    return diff.astype(np.uint8)


def apply_panorama_delta(previous_bgr: np.ndarray, diff_bgr: np.ndarray) -> np.ndarray:
    """Exact inverse of `compute_panorama_delta`: reconstructs `current` from
    `previous` and a (possibly lossily re-encoded) diff image."""
    previous = _ensure_bgr_uint8(previous_bgr).astype(np.int16)
    diff = _ensure_bgr_uint8(diff_bgr).astype(np.int16)
    if previous.shape != diff.shape:
        raise ValueError(
            f"Panorama delta shape mismatch: previous={previous.shape}, diff={diff.shape}"
        )
    current = np.clip(previous + diff - 128, 0, 255)
    return current.astype(np.uint8)


def round_trip_panorama_delta(
    current_bgr: np.ndarray,
    previous_bgr: np.ndarray,
    encoder: BasePanoramaEncoder,
) -> tuple[bytes, np.ndarray]:
    """Delta counterpart of `round_trip_panorama` (background-layer rung 2,
    "panorama+delta"): encodes `current` as a diff against `previous` through
    `encoder`, then reconstructs the exact pixels the client will derive
    (`previous + decode(encoded_diff)`) so residuals are computed against
    what the client actually ends up seeing -- not an idealized `current` it
    never gets bit-identically (Residual Guarantee; see CLAUDE.md and report
    8's "Panorama JPEG quality does not affect the residual" entry for the
    class of bug this guards against).
    """
    diff_image = compute_panorama_delta(current_bgr=current_bgr, previous_bgr=previous_bgr)
    encoded_bytes, decoded_diff = round_trip_panorama(diff_image, encoder)
    reconstructed_current = apply_panorama_delta(previous_bgr=previous_bgr, diff_bgr=decoded_diff)
    return encoded_bytes, reconstructed_current


def read_panorama_pixels_from_path(path: str | Path) -> np.ndarray:
    """Decode a materialized panorama sidecar file to BGR pixels, dispatching on
    file extension since some codecs (roi-video) use a video container that
    `cv2.imread` cannot open directly."""
    resolved = Path(path)
    if resolved.suffix.lower() in {".mp4", ".mkv", ".mov"}:
        return RoiVideoPanoramaEncoder().decode_bytes(resolved.read_bytes())

    decoded = cv2.imread(str(resolved), cv2.IMREAD_COLOR)
    if decoded is None or decoded.size == 0:
        raise ValueError(f"Failed to decode panorama image from {resolved}")
    return np.asarray(decoded, dtype=np.uint8)


def build_panorama_encoder(panorama_encoder: str | BasePanoramaEncoder | None = None, config: Any = None) -> BasePanoramaEncoder:
    if isinstance(panorama_encoder, BasePanoramaEncoder):
        return panorama_encoder

    codec = panorama_encoder or (config.panorama_codec if config else "jpeg")
    normalized = str(codec).strip().lower()

    if normalized in {"jpeg", "jpg"}:
        quality = config.panorama_jpeg_quality if config and config.panorama_jpeg_quality else 90
        return JpegPanoramaEncoder(quality=int(quality))

    if normalized == "png":
        compression = config.panorama_png_compression if config and config.panorama_png_compression is not None else 3
        return PngPanoramaEncoder(compression=int(compression))

    if normalized in {"roi-video", "roi_video"}:
        crf = config.panorama_roi_crf if config and config.panorama_roi_crf is not None else 30
        preset = config.panorama_roi_preset if config and config.panorama_roi_preset else "veryfast"
        return RoiVideoPanoramaEncoder(crf=int(crf), preset=str(preset))

    raise ValueError(
        f"Unsupported panorama encoder '{codec}'. "
        "Supported values: jpeg, png, roi-video."
    )
