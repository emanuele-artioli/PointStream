"""Encode a clip and score it. No constant bitrate table and no grey stand-in frames.

Callers that want a rate comparison pass a measured anchor byte count into
``beats``. A missing anchor withholds the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

import cv2
import numpy as np

from experiments.headroom.ladder import resolved_tools
from src.components.background.plate import build_plate
from src.components.background.sidecar import build_sidecar
from src.components.codec.encode import BITSTREAM_SUFFIX, decode, encode
from src.components.codec.frames import even_size
from src.components.codec.tools import resolve_vvenc
from src.components.codec.y4m import Y4M, parse_header, read, write
from src.components.metrics.psnr import PsnrMetric, masked_psnr
from src.components.metrics.visual_inspection import create_comparison_strip, save_montage_image
from src.contracts.codecs import EncodeRequest, RateControl
from src.pipeline.residual.steered_residual import ActorMaskProcessor


class MeasuredInputError(ValueError):
    """The clip or its mask cannot be scored."""


@dataclass(frozen=True)
class MeasuredRung:
    """One encoded rung. ``pose_oks`` stays None until a pose backend is supplied.

    ``reconstruction`` is RGB uint8, shaped like the source, and is not written
    into the JSON report.
    """

    rung_id: str
    total_bytes: int
    bytes_background: int
    bytes_appearance: int
    bytes_metadata: int
    bytes_residual: int
    bytes_container: int
    psnr_weighted: float | None
    psnr_overall: float
    psnr_fg: float
    psnr_bg: float
    pose_oks: float | None
    reconstruction: np.ndarray

    def beats_anchor(self, anchor_bytes: int | None) -> bool:
        """True only when ``anchor_bytes`` was measured and this rung is smaller."""
        return beats(self.total_bytes, anchor_bytes)


@dataclass(frozen=True)
class MeasuredAnchor:
    """A native encode of the same frames. Paths and versions are the binaries that ran."""

    codec: str
    qp: int
    total_bytes: int
    psnr_overall: float
    encoder_path: str
    encoder_version: str
    psnr_fg: float | None = None
    psnr_bg: float | None = None
    psnr_weighted: float | None = None
    ffmpeg_path: str | None = None
    ffmpeg_version: str | None = None


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _as_clip_rgb(frames_rgb: np.ndarray) -> np.ndarray:
    array = np.asarray(frames_rgb)
    if array.ndim != 4 or array.shape[-1] != 3 or array.shape[0] < 1:
        raise MeasuredInputError(
            f"expected non-empty RGB clip (T, H, W, 3), got {tuple(array.shape)}"
        )
    return np.asarray(array, dtype=np.uint8)


def _as_mask(mask: np.ndarray, frames: int, height: int, width: int) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 2:
        if array.shape != (height, width):
            raise MeasuredInputError(
                f"mask shape {array.shape} does not match frame {(height, width)}"
            )
        array = np.broadcast_to(array, (frames, height, width))
    elif array.shape != (frames, height, width):
        raise MeasuredInputError(
            f"mask shape {array.shape} does not match clip {(frames, height, width)}"
        )
    return np.asarray(array, dtype=bool)


def _rgb_to_bgr(frames_rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(frames_rgb[..., ::-1])


def _bgr_to_rgb(frames_bgr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(frames_bgr[..., ::-1])


def _direct_vvc_encode(
    source: Path,
    dest: Path,
    request: EncodeRequest,
) -> tuple[str, str]:
    """Encode with vvencapp when FFmpeg's libvvenc wrapper emits no bytes.

    The installed vvencapp 1.11.0 accepts the same Y4M and QP, and produces a
    valid VVC stream at QPs where FFmpeg/libvvenc exits 0 after writing zero
    bytes. This is a measured fallback, not a silent substitution: the caller
    records vvencapp's path and version.
    """
    if request.rate_control is not RateControl.QP or request.rate is None:
        raise MeasuredInputError("direct vvenc fallback requires an explicit QP")
    width, height, fps = parse_header(Path(source))
    vvenc = resolve_vvenc()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    command = [
        vvenc.path,
        "--input",
        str(source),
        "--size",
        f"{width}x{height}",
        "--framerate",
        str(max(1, int(round(fps)))),
        "--format",
        "yuv420",
        "--preset",
        str(request.preset or "medium"),
        "--qp",
        str(int(request.rate)),
        "--qpa",
        "0",
        "--output",
        str(dest),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not dest.is_file() or dest.stat().st_size == 0:
        detail = (result.stderr or result.stdout or "").strip()[-1000:]
        raise RuntimeError(
            f"vvencapp failed ({result.returncode}) or wrote an empty bitstream: "
            f"{' '.join(command)}\n{detail}"
        )
    return vvenc.path, vvenc.version


# The full-resolution sweep found a VVC intra plate point, but the PointStream
# plate remains WebP for now. Replacing it is a deferred background-model
# experiment, not something justified by the old project's Presley budget.
PLATE_CODEC = "webp"
PLATE_QP = 40
CROP_CODEC = "av1"
CROP_QP = 42
# libvvenc preset "faster" writes an empty bitstream for the 48-frame C1
# error video. Preset "medium" emits one. QP 40 is the knee: 279514 B,
# corrected frame 31.24 dB, against AV1 preset 10 QP 54 at 581561 B and
# 32.25 dB. VVC medium QP 32 dominates AV1 QP 46 and QP 50 on that residual.
RESIDUAL_CODEC = "vvc"
RESIDUAL_QP = 40
RESIDUAL_PRESET = "medium"


def _intra_roundtrip(image_bgr: np.ndarray, codec: str, qp: int) -> tuple[bytes, np.ndarray]:
    """Encode and decode one BGR still through the anchor's intra sidecar.

    SVT-AV1 rejects crops below 64 pixels on a side. Those are padded with
    black for the encoder and cropped back after decode. Real actor crops and
    the plate are already larger, so they are not padded.
    """
    height, width = int(image_bgr.shape[0]), int(image_bgr.shape[1])
    coded_h = max(64, height + (-height % 8))
    coded_w = max(64, width + (-width % 8))
    coded = image_bgr
    if (coded_h, coded_w) != (height, width):
        coded = np.zeros((coded_h, coded_w, image_bgr.shape[2]), dtype=image_bgr.dtype)
        coded[:height, :width] = image_bgr
    coder = build_sidecar(codec, intra_qp=int(qp))
    payload = coder.encode(coded)
    decoded = coder.decode(payload)
    if decoded.shape[0] < height or decoded.shape[1] < width:
        canvas = np.zeros_like(image_bgr)
        canvas[: decoded.shape[0], : decoded.shape[1]] = decoded[:height, :width]
        return payload, canvas
    return payload, np.ascontiguousarray(decoded[:height, :width])


def _webp_roundtrip(image_bgr: np.ndarray, quality: int) -> tuple[bytes, np.ndarray]:
    """Encode and decode a BGR still as WebP."""
    ok, buffer = cv2.imencode(
        ".webp",
        image_bgr,
        [int(cv2.IMWRITE_WEBP_QUALITY), int(quality)],
    )
    if not ok:
        raise MeasuredInputError("WebP encode failed")
    payload = buffer.tobytes()
    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise MeasuredInputError("WebP decode failed")
    return payload, decoded


def _plate_roundtrip(image_bgr: np.ndarray) -> tuple[bytes, np.ndarray]:
    """Use the current PointStream plate codec."""
    if PLATE_CODEC == "webp":
        return _webp_roundtrip(image_bgr, PLATE_QP)
    return _intra_roundtrip(image_bgr, PLATE_CODEC, PLATE_QP)


def _pack_metadata(entries: list[tuple[int, int, int, int, int]]) -> bytes:
    if not entries:
        return b""
    return np.asarray(entries, dtype="<i4").reshape(-1).tobytes()


def _paste_crop(
    canvas_bgr: np.ndarray,
    crop_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    fg_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Paste ``crop_bgr`` into ``bbox`` on a copy of ``canvas_bgr``.

    When ``fg_mask`` is set, only foreground pixels of that mask are written so
    padded crop margins do not overwrite background on later frames.
    """
    y1, y2, x1, x2 = bbox
    out = canvas_bgr.copy()
    h, w = y2 - y1, x2 - x1
    patch = crop_bgr
    if patch.shape[0] != h or patch.shape[1] != w:
        patch = cv2.resize(patch, (w, h), interpolation=cv2.INTER_LINEAR)
    if fg_mask is None:
        out[y1:y2, x1:x2] = patch
        return out
    region = out[y1:y2, x1:x2]
    local = np.asarray(fg_mask[y1:y2, x1:x2], dtype=bool)
    region[local] = patch[local]
    out[y1:y2, x1:x2] = region
    return out


def _encode_appearance_crop(
    frame_bgr: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    codec: str = CROP_CODEC,
    qp: int = CROP_QP,
) -> tuple[bytes, np.ndarray]:
    y1, y2, x1, x2 = bbox
    crop = frame_bgr[y1:y2, x1:x2]
    return _intra_roundtrip(crop, codec, qp)


def _fg_mse_vs_keyframe(
    frame_bgr: np.ndarray,
    keyframe_on_plate: np.ndarray,
    fg_mask: np.ndarray,
) -> float:
    if not np.any(fg_mask):
        return 0.0
    diff = frame_bgr.astype(np.int16) - keyframe_on_plate.astype(np.int16)
    return float(np.mean(np.square(diff[fg_mask])))


def _make_rung(
    rung_id: str,
    *,
    bytes_background: int,
    bytes_appearance: int,
    bytes_metadata: int,
    bytes_residual: int,
    reconstruction_bgr: np.ndarray,
    frames_rgb: np.ndarray,
    mask: np.ndarray,
    fg_weight: float,
    bg_weight: float,
) -> MeasuredRung:
    reconstruction_rgb = _bgr_to_rgb(reconstruction_bgr)
    psnr_overall, psnr_fg, psnr_bg, psnr_weighted = score_regions(
        frames_rgb,
        reconstruction_rgb,
        mask,
        fg_weight=fg_weight,
        bg_weight=bg_weight,
    )
    bytes_container = 0
    total_bytes = (
        bytes_background
        + bytes_appearance
        + bytes_metadata
        + bytes_residual
        + bytes_container
    )
    return MeasuredRung(
        rung_id=rung_id,
        total_bytes=total_bytes,
        bytes_background=bytes_background,
        bytes_appearance=bytes_appearance,
        bytes_metadata=bytes_metadata,
        bytes_residual=bytes_residual,
        bytes_container=bytes_container,
        psnr_weighted=psnr_weighted,
        psnr_overall=psnr_overall,
        psnr_fg=psnr_fg,
        psnr_bg=psnr_bg,
        pose_oks=None,
        reconstruction=reconstruction_rgb,
    )


def load_sequence(
    frames_dir: Path,
    mask_path: Path,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load ``n_frames`` RGB images and a boolean foreground mask.

    Images are read in sorted filename order from ``frames_dir`` and returned
    as uint8 ``(T, H, W, 3)`` RGB. The mask file is a ``.npy`` boolean array,
    or a ``.npz`` with a ``masks`` array, shaped ``(T, H, W)`` or ``(H, W)``.
    A single-frame mask is repeated across time.

    Raises:
        FileNotFoundError: the directory, an image, or the mask is missing.
        MeasuredInputError: fewer than ``n_frames`` images, ``n_frames < 1``,
            images disagree in shape, or the mask does not match ``(T, H, W)``.

    The caller relies on every returned frame having the same shape as the mask.
    """
    frames_dir = Path(frames_dir)
    mask_path = Path(mask_path)
    if n_frames < 1:
        raise MeasuredInputError(f"n_frames must be >= 1, got {n_frames}")
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"frames directory not found: {frames_dir}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"mask file not found: {mask_path}")

    image_paths = sorted(
        p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    )
    if len(image_paths) < n_frames:
        raise MeasuredInputError(
            f"need {n_frames} images in {frames_dir}, found {len(image_paths)}"
        )

    frames: list[np.ndarray] = []
    for path in image_paths[:n_frames]:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"failed to read image: {path}")
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

    stack = np.stack(frames, axis=0)
    shapes = {tuple(frame.shape) for frame in frames}
    if len(shapes) != 1:
        raise MeasuredInputError(f"images disagree in shape: {sorted(shapes)}")

    mask = np.load(mask_path)
    if isinstance(mask, np.lib.npyio.NpzFile):
        if "masks" not in mask.files:
            raise MeasuredInputError(f"{mask_path} has no 'masks' array")
        mask = mask["masks"]
    height, width = int(stack.shape[1]), int(stack.shape[2])
    mask_bool = _as_mask(mask, n_frames, height, width)
    return np.asarray(stack, dtype=np.uint8), mask_bool


def score_regions(
    reference_rgb: np.ndarray,
    reconstruction_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    fg_weight: float,
    bg_weight: float,
) -> tuple[float, float, float, float | None]:
    """Return overall, foreground, background, and weighted PSNR in dB.

    Overall, foreground, and background use ``PsnrMetric`` and ``masked_psnr``.
    Foreground is where ``mask`` is True. An identical region scores infinite
    PSNR. Weighted PSNR is ``fg_weight * psnr_fg + bg_weight * psnr_bg`` when
    both regional scores are finite, the weights are non-negative, and they sum
    to 1. Otherwise the weighted score is None. Infinite PSNR is never substituted
    with a finite stand-in.

    Raises:
        MeasuredInputError: empty clip, shape mismatch, or weights that are
            negative or do not sum to 1 within 1e-6.

    A caller relies on a one-pixel, all-channel difference of 255 on an
    otherwise identical 8×8 frame scoring overall PSNR ``10 * log10(64)``.
    """
    reference = _as_clip_rgb(reference_rgb)
    reconstruction = np.asarray(reconstruction_rgb, dtype=np.uint8)
    if reconstruction.shape != reference.shape:
        raise MeasuredInputError(
            f"reconstruction shape {reconstruction.shape} != reference {reference.shape}"
        )
    if fg_weight < 0 or bg_weight < 0:
        raise MeasuredInputError(
            f"weights must be non-negative, got fg={fg_weight}, bg={bg_weight}"
        )
    if abs(float(fg_weight) + float(bg_weight) - 1.0) > 1e-6:
        raise MeasuredInputError(
            f"weights must sum to 1 within 1e-6, got fg={fg_weight}, bg={bg_weight}"
        )

    frames, height, width, _ = reference.shape
    mask_bool = _as_mask(mask, frames, height, width)

    overall = float(PsnrMetric().score(reference, reconstruction))
    psnr_fg = float(masked_psnr(reference, reconstruction, mask_bool))
    psnr_bg = float(masked_psnr(reference, reconstruction, ~mask_bool))

    if np.isfinite(psnr_fg) and np.isfinite(psnr_bg):
        weighted: float | None = float(fg_weight) * psnr_fg + float(bg_weight) * psnr_bg
    else:
        weighted = None
    return overall, psnr_fg, psnr_bg, weighted


def measure_rungs(
    frames_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    fg_weight: float = 0.70,
    bg_weight: float = 0.30,
    include_residuals: bool = True,
) -> list[MeasuredRung]:
    """Encode rungs C0–C3 from the pixels and score each reconstruction.

    The plate is ``build_plate(..., register=False)`` on BGR frames, then
    ``PLATE_CODEC`` at ``PLATE_QP`` (currently WebP q40; the full-resolution
    VVC intra replacement is deferred). Crops are ``CROP_CODEC``
    intra at ``CROP_QP`` (AV1 QP 42). The crop box is the tight even-aligned
    bbox of that frame's mask, from ``ActorMaskProcessor.compute_tight_bbox``.
    C0 sends the frame-0 crop and pastes its decode into that box on every
    frame. C1 sends a new crop whenever the foreground MSE against the last
    keyframe exceeds 50, and pastes each decode from its keyframe until the
    next one. C2 encodes the C1 error, offset by 128, as one ``RESIDUAL_CODEC``
    video (VVC, preset medium, QP 40) and adds the decode to every pixel. C3
    is the mask ablation of that same video: foreground-only and
    background-only, each an alternative to C2 rather than a stack on top of
    it. libvvenc preset faster emits an empty file on this residual, which is
    why the preset is medium. The still-image decision for the plate and the
    crops is in ``outputs/image-codec-probe/federer007.json``. The video
    comparison is in ``outputs/video-codec-probe/federer007.json``.

    Metadata for a rung is the little-endian int32 sequence of
    ``(frame_index, y0, x0, y1, x1)`` for every crop or residual box that rung
    transmits. Container bytes are 0. ``pose_oks`` is None. Byte fields are
    ``len`` of the payloads just produced, and ``total_bytes`` is their sum.
    Regional PSNR comes from ``score_regions`` on that rung's reconstruction.

    Raises:
        MeasuredInputError: empty clip or a mask that does not match the frames.

    A caller relies on two clips that differ inside the mask not producing
    identical ``(bytes_appearance, psnr_fg)`` pairs, and on ``total_bytes``
    equalling the five component fields.
    """
    frames = _as_clip_rgb(frames_rgb)
    t_count, height, width, _ = frames.shape
    mask_bool = _as_mask(mask, t_count, height, width)
    frames_bgr = _rgb_to_bgr(frames)

    plate, _maps = build_plate(
        frames_bgr,
        masks=mask_bool.astype(np.uint8),
        register=False,
    )
    plate = np.asarray(plate, dtype=np.uint8)
    if plate.shape[0] != height or plate.shape[1] != width:
        plate = plate[:height, :width]

    bg_payload, plate_dec = _plate_roundtrip(plate)
    if plate_dec.shape[0] != height or plate_dec.shape[1] != width:
        plate_dec = plate_dec[:height, :width]
    bytes_background = len(bg_payload)

    mask_processor = ActorMaskProcessor()

    bboxes: list[tuple[int, int, int, int] | None] = [
        mask_processor.compute_tight_bbox(mask_bool[index], pad=8, even_align=True)
        for index in range(t_count)
    ]

    if bboxes[0] is None:
        raise MeasuredInputError("frame 0 has an empty foreground mask; no crop to send")

    crop0_payload, crop0_decoded = _encode_appearance_crop(frames_bgr[0], bboxes[0])
    y1_0, y2_0, x1_0, x2_0 = bboxes[0]
    crop_meta_c0: list[tuple[int, int, int, int, int]] = [
        (0, y1_0, x1_0, y2_0, x2_0)
    ]
    meta_c0 = _pack_metadata(crop_meta_c0)
    bytes_appearance_c0 = len(crop0_payload)

    recon_c0 = np.stack(
        [
            _paste_crop(plate_dec, crop0_decoded, bboxes[0], fg_mask=mask_bool[0])
            for _ in range(t_count)
        ],
        axis=0,
    )

    # C1 adaptive keyframes
    keyframe_indices = [0]
    keyframe_crops: dict[
        int, tuple[tuple[int, int, int, int], np.ndarray, bytes, np.ndarray]
    ] = {
        0: (bboxes[0], crop0_decoded, crop0_payload, mask_bool[0])
    }
    last_paste = _paste_crop(
        plate_dec, crop0_decoded, bboxes[0], fg_mask=mask_bool[0]
    )
    appearance_payloads_c1 = [crop0_payload]
    crop_meta_c1 = list(crop_meta_c0)

    for index in range(1, t_count):
        mse = _fg_mse_vs_keyframe(frames_bgr[index], last_paste, mask_bool[index])
        if mse <= 50.0:
            continue
        bbox = bboxes[index]
        if bbox is None:
            continue
        payload, decoded = _encode_appearance_crop(frames_bgr[index], bbox)
        keyframe_indices.append(index)
        keyframe_crops[index] = (bbox, decoded, payload, mask_bool[index])
        appearance_payloads_c1.append(payload)
        y1, y2, x1, x2 = bbox
        crop_meta_c1.append((index, y1, x1, y2, x2))
        last_paste = _paste_crop(plate_dec, decoded, bbox, fg_mask=mask_bool[index])

    bytes_appearance_c1 = sum(len(item) for item in appearance_payloads_c1)
    meta_c1 = _pack_metadata(crop_meta_c1)

    recon_c1_frames: list[np.ndarray] = []
    for index in range(t_count):
        active = max(k for k in keyframe_indices if k <= index)
        bbox, decoded, _payload, kf_mask = keyframe_crops[active]
        recon_c1_frames.append(
            _paste_crop(plate_dec, decoded, bbox, fg_mask=kf_mask)
        )
    recon_c1 = np.stack(recon_c1_frames, axis=0)

    c0 = _make_rung(
        "C0_compact_baseline",
        bytes_background=bytes_background,
        bytes_appearance=bytes_appearance_c0,
        bytes_metadata=len(meta_c0),
        bytes_residual=0,
        reconstruction_bgr=recon_c0,
        frames_rgb=frames,
        mask=mask_bool,
        fg_weight=fg_weight,
        bg_weight=bg_weight,
    )
    c1 = _make_rung(
        "C1_adaptive_keyframes",
        bytes_background=bytes_background,
        bytes_appearance=bytes_appearance_c1,
        bytes_metadata=len(meta_c1),
        bytes_residual=0,
        reconstruction_bgr=recon_c1,
        frames_rgb=frames,
        mask=mask_bool,
        fg_weight=fg_weight,
        bg_weight=bg_weight,
    )
    if not include_residuals:
        return [c0, c1]

    full = _video_residual_rung(
        "C2_unified_video_residual",
        base_bgr=recon_c1,
        frames_bgr=frames_bgr,
        frames_rgb=frames,
        mask=mask_bool,
        mode="full",
        bytes_background=bytes_background,
        bytes_appearance=bytes_appearance_c1,
        metadata=meta_c1,
        fg_weight=fg_weight,
        bg_weight=bg_weight,
    )
    foreground = _video_residual_rung(
        "C3_foreground_video_residual",
        base_bgr=recon_c1,
        frames_bgr=frames_bgr,
        frames_rgb=frames,
        mask=mask_bool,
        mode="foreground",
        bytes_background=bytes_background,
        bytes_appearance=bytes_appearance_c1,
        metadata=meta_c1,
        fg_weight=fg_weight,
        bg_weight=bg_weight,
    )
    background = _video_residual_rung(
        "C3_background_video_residual",
        base_bgr=recon_c1,
        frames_bgr=frames_bgr,
        frames_rgb=frames,
        mask=mask_bool,
        mode="background",
        bytes_background=bytes_background,
        bytes_appearance=bytes_appearance_c1,
        metadata=meta_c1,
        fg_weight=fg_weight,
        bg_weight=bg_weight,
    )
    return [c0, c1, full, foreground, background]


def _video_residual_rung(
    rung_id: str,
    *,
    base_bgr: np.ndarray,
    frames_bgr: np.ndarray,
    frames_rgb: np.ndarray,
    mask: np.ndarray,
    mode: str,
    bytes_background: int,
    bytes_appearance: int,
    metadata: bytes,
    fg_weight: float,
    bg_weight: float,
) -> MeasuredRung:
    """Encode one offset error video and add its decode back onto ``base_bgr``."""
    error = np.clip(frames_bgr.astype(np.int16) - base_bgr.astype(np.int16) + 128, 0, 255)
    error = np.ascontiguousarray(error.astype(np.uint8))
    apply = np.ones(mask.shape, dtype=bool)
    if mode == "foreground":
        error[~mask] = 128
        apply = mask
    elif mode == "background":
        error[mask] = 128
        apply = ~mask
    elif mode != "full":
        raise MeasuredInputError(f"unknown residual mode {mode!r}")
    _payload, decoded_bgr = _roundtrip_clip(
        _bgr_to_rgb(error),
        codec=RESIDUAL_CODEC,
        qp=RESIDUAL_QP,
        preset=RESIDUAL_PRESET,
    )
    if decoded_bgr.shape[1] != base_bgr.shape[1] or decoded_bgr.shape[2] != base_bgr.shape[2]:
        decoded_bgr = decoded_bgr[:, : base_bgr.shape[1], : base_bgr.shape[2]]
    signed = decoded_bgr.astype(np.int16) - 128
    corrected = base_bgr.astype(np.int16).copy()
    corrected[apply] = np.clip(corrected[apply] + signed[apply], 0, 255)
    return _make_rung(
        rung_id,
        bytes_background=bytes_background,
        bytes_appearance=bytes_appearance,
        bytes_metadata=len(metadata),
        bytes_residual=len(_payload),
        reconstruction_bgr=corrected.astype(np.uint8),
        frames_rgb=frames_rgb,
        mask=mask,
        fg_weight=fg_weight,
        bg_weight=bg_weight,
    )


def _roundtrip_clip(
    frames_rgb: np.ndarray,
    *,
    codec: str,
    qp: int,
    preset: str,
) -> tuple[bytes, np.ndarray]:
    """Encode an RGB clip and return the bitstream plus the decoded BGR frames."""
    import tempfile

    frames = _as_clip_rgb(frames_rgb)
    with tempfile.TemporaryDirectory(prefix="ps_residual_") as tmp:
        root = Path(tmp)
        clip = even_size(frames)
        luma, chroma = _rgb_to_yuv420(clip)
        source = root / "input.y4m"
        write(
            source,
            Y4M(
                width=int(luma.shape[2]),
                height=int(luma.shape[1]),
                fps=25.0,
                luma=luma,
                chroma=chroma,
            ),
        )
        request = EncodeRequest(
            codec_name=codec,
            rate_control=RateControl.QP,
            rate=int(qp),
            preset=preset,
            pix_fmt="yuv420p",
        )
        bitstream = root / f"{codec}_qp{int(qp)}{BITSTREAM_SUFFIX[codec]}"
        try:
            encode(source, bitstream, request, work_dir=root)
        except RuntimeError:
            if codec != "vvc":
                raise
            _direct_vvc_encode(source, bitstream, request)
        payload = bitstream.read_bytes()
        decoded_path = root / "decoded.y4m"
        decode(bitstream, decoded_path, request)
        decoded = read(decoded_path)
        if decoded.chroma is None:
            raise MeasuredInputError(f"{decoded_path} decoded without chroma")
        decoded_rgb = _yuv420_to_rgb(decoded.luma, decoded.chroma)
        decoded_bgr = _rgb_to_bgr(decoded_rgb)
    if decoded_bgr.shape[0] != frames.shape[0]:
        decoded_bgr = decoded_bgr[: frames.shape[0]]
    if decoded_bgr.shape[1] != frames.shape[1] or decoded_bgr.shape[2] != frames.shape[2]:
        canvas = np.zeros(frames.shape, dtype=np.uint8)
        height = min(decoded_bgr.shape[1], frames.shape[1])
        width = min(decoded_bgr.shape[2], frames.shape[2])
        canvas[:, :height, :width] = decoded_bgr[:, :height, :width]
        decoded_bgr = canvas
    return payload, decoded_bgr


def _rgb_to_yuv420(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """BT.601 4:2:0. Chroma is the mean of each 2×2 block. Shape ``(T, 2, H/2, W/2)``."""
    clip = np.asarray(frames, dtype=np.float64)
    red, green, blue = clip[..., 0], clip[..., 1], clip[..., 2]
    luma = np.clip(np.round(0.299 * red + 0.587 * green + 0.114 * blue), 0, 255).astype(np.uint8)
    cb = -0.168736 * red - 0.331264 * green + 0.5 * blue + 128.0
    cr = 0.5 * red - 0.418688 * green - 0.081312 * blue + 128.0
    frames_n, height, width = cb.shape
    cb_sub = cb.reshape(frames_n, height // 2, 2, width // 2, 2).mean(axis=(2, 4))
    cr_sub = cr.reshape(frames_n, height // 2, 2, width // 2, 2).mean(axis=(2, 4))
    chroma = np.stack(
        [np.clip(np.round(cb_sub), 0, 255), np.clip(np.round(cr_sub), 0, 255)],
        axis=1,
    ).astype(np.uint8)
    return luma, chroma


def _yuv420_to_rgb(luma: np.ndarray, chroma: np.ndarray) -> np.ndarray:
    """Inverse of ``_rgb_to_yuv420`` with nearest chroma upsampling."""
    y = luma.astype(np.float64)
    cb = np.repeat(np.repeat(chroma[:, 0].astype(np.float64), 2, axis=1), 2, axis=2) - 128.0
    cr = np.repeat(np.repeat(chroma[:, 1].astype(np.float64), 2, axis=1), 2, axis=2) - 128.0
    red = y + 1.402 * cr
    green = y - 0.344136 * cb - 0.714136 * cr
    blue = y + 1.772 * cb
    return np.clip(np.round(np.stack([red, green, blue], axis=-1)), 0, 255).astype(np.uint8)


def beats(measured_bytes: int, anchor_bytes: int | None) -> bool:
    """True only when a measured anchor exists and ``measured_bytes`` is strictly smaller.

    ``anchor_bytes is None`` and equal byte counts are both False. Negative
    inputs raise MeasuredInputError.
    """
    if measured_bytes < 0:
        raise MeasuredInputError(f"measured_bytes must be non-negative, got {measured_bytes}")
    if anchor_bytes is not None and anchor_bytes < 0:
        raise MeasuredInputError(f"anchor_bytes must be non-negative, got {anchor_bytes}")
    if anchor_bytes is None:
        return False
    return int(measured_bytes) < int(anchor_bytes)


def measure_native_anchor(
    frames_rgb: np.ndarray,
    *,
    codec: str,
    qp: int,
    work_dir: Path,
    mask: np.ndarray | None = None,
    fg_weight: float = 0.70,
    bg_weight: float = 0.30,
) -> MeasuredAnchor:
    """Encode ``frames_rgb`` with the resolved native encoder and score the decode.

    Uses ``experiments.headroom.ladder`` tool resolution and
    ``src.components.codec.encode``. Records that binary's path and version.
    ``total_bytes`` is the bitstream length. ``psnr_overall`` is the decode
    against ``frames_rgb``. When ``mask`` is supplied, foreground, background,
    and saliency-weighted PSNR are scored from the same decoded frames. This
    keeps anchor quality comparable with the PointStream regional ledger.

    Raises:
        FileNotFoundError: ffmpeg or the named encoder is not on the machine.
        MeasuredInputError: ``qp`` is outside 0..63 or the clip is empty.

    A caller relies on ``total_bytes`` matching the file size of the bitstream
    written under ``work_dir``.
    """
    frames = _as_clip_rgb(frames_rgb)
    if not 0 <= int(qp) <= 63:
        raise MeasuredInputError(f"qp must be in 0..63, got {qp}")

    tools = resolved_tools(codec)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    clip = even_size(frames)
    luma, chroma = _rgb_to_yuv420(clip)
    source = work_dir / "input.y4m"
    write(
        source,
        Y4M(
            width=int(luma.shape[2]),
            height=int(luma.shape[1]),
            fps=25.0,
            luma=luma,
            chroma=chroma,
        ),
    )

    presets = {"avc": "veryfast", "hevc": "ultrafast", "av1": "10", "vvc": "faster"}
    if codec not in presets:
        raise MeasuredInputError(f"unsupported codec {codec!r}")
    if codec not in BITSTREAM_SUFFIX:
        raise MeasuredInputError(f"no bitstream suffix for codec {codec!r}")

    request = EncodeRequest(
        codec_name=codec,
        rate_control=RateControl.QP,
        rate=int(qp),
        preset=presets[codec],
        pix_fmt="yuv420p",
    )
    bitstream = work_dir / f"{codec}_qp{int(qp)}{BITSTREAM_SUFFIX[codec]}"
    encoder_path = tools["encoder_path"]
    encoder_version = tools["encoder_version"]
    try:
        record = encode(source, bitstream, request, work_dir=work_dir)
    except RuntimeError:
        if codec != "vvc":
            raise
        encoder_path, encoder_version = _direct_vvc_encode(source, bitstream, request)
        record = None
    total_bytes = int(bitstream.stat().st_size)
    if record is not None and total_bytes != int(record.size_bytes):
        total_bytes = int(bitstream.stat().st_size)

    decoded_path = work_dir / f"decoded_qp{int(qp)}.y4m"
    decode(bitstream, decoded_path, request)
    decoded = read(decoded_path)
    if decoded.chroma is None:
        raise MeasuredInputError(f"{decoded_path} decoded without chroma")
    decoded_rgb = _yuv420_to_rgb(decoded.luma, decoded.chroma)
    psnr_overall = float(PsnrMetric().score(clip, decoded_rgb))
    psnr_fg: float | None = None
    psnr_bg: float | None = None
    psnr_weighted: float | None = None
    if mask is not None:
        mask_bool = _as_mask(mask, clip.shape[0], clip.shape[1], clip.shape[2])
        _overall, psnr_fg, psnr_bg, psnr_weighted = score_regions(
            clip,
            decoded_rgb,
            mask_bool,
            fg_weight=fg_weight,
            bg_weight=bg_weight,
        )

    return MeasuredAnchor(
        codec=codec,
        qp=int(qp),
        total_bytes=total_bytes,
        psnr_overall=psnr_overall,
        encoder_path=encoder_path,
        encoder_version=encoder_version,
        psnr_fg=psnr_fg,
        psnr_bg=psnr_bg,
        psnr_weighted=psnr_weighted,
        ffmpeg_path=tools["ffmpeg_path"],
        ffmpeg_version=tools["ffmpeg_version"],
    )


def write_comparison_strip(
    frames_rgb: np.ndarray,
    reconstruction_rgb: np.ndarray,
    output_path: Path,
    *,
    summary: str,
) -> Path:
    """Write a four-panel strip whose first panel is ``frames_rgb[0]``.

    The conditioning panel is the reconstruction of frame 0. The prediction
    panel is the same reconstruction. The error panel is the amplified
    difference from ``create_comparison_strip``. The image is not a constant fill.

    Raises:
        MeasuredInputError: the clip is empty or the two arrays differ in shape.
    """
    frames = _as_clip_rgb(frames_rgb)
    reconstruction = np.asarray(reconstruction_rgb, dtype=np.uint8)
    if reconstruction.shape != frames.shape:
        raise MeasuredInputError(
            f"reconstruction shape {reconstruction.shape} != frames {frames.shape}"
        )
    strip = create_comparison_strip(
        frames[0],
        reconstruction[0],
        conditioning=reconstruction[0],
        metrics_summary=summary,
    )
    return save_montage_image(strip, Path(output_path))
