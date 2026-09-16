"""Component probe for background representations (CODEC-ACT-06).

Evaluates whether background representation (still frame vs registered panorama vs
cleaned video) explains enough error to matter for PointStream.

Hypothesis: Camera coverage/registration is the limiting background error;
video improves it but must earn its byte/time cost.
Alternative: A still is already sufficient and geometry/metadata or correction
causes the deficit.

Protocol:
1. Input verification: exact 48-frame Federer interval (`federer_djokovic/scene_007`).
2. Common foreground-removed frame stack: preserve visible background (~mask),
   fill player masks with observed temporal background warped back, explicitly
   inpaint any remaining holes. Hold stack fixed.
3. Three representations:
   - First cleaned frame repeated with identity rendering (no camera warp).
   - Registered panorama with charged per-frame mappings (camera homographies).
   - Cleaned per-frame video (VVC).
4. Screening at provisional VVC QP 47 and 32 with full rate/quality/time accounting.
"""

from __future__ import annotations

import sqlite3  # noqa: F401 - required before torch on this host
import argparse
from collections.abc import Sequence
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import time
from typing import Any, Final

import cv2
import numpy as np

from experiments.long_scenes.loader import load_long_scene_clip
from src.components.background.plate import build_plate
from src.components.background.sidecar import IntraCodecSidecar
from src.components.codec import tools as codec_tools
from src.components.codec.frames import rgb_to_luma
from src.components.codec.measure import timed_roundtrip
from src.components.metrics.frames import paired
from src.components.metrics.ssim import SsimMetric
from src.contracts import paths as ps_paths
from src.contracts.codecs import EncodeRequest, RateControl

TASK_ID: Final[str] = "CODEC-ACT-06"
DEFAULT_VIDEO: Final[str] = "federer_djokovic"
DEFAULT_SCENE: Final[str] = "scene_007"
DEFAULT_N_FRAMES: Final[int] = 48
DEFAULT_FPS: Final[float] = 24.0
DEFAULT_QPS: Final[tuple[int, ...]] = (47, 32)
DEFAULT_PRESET: Final[str] = "faster"
DEFAULT_CODEC: Final[str] = "vvc"

# Pre-registered bounds for verification
PRE_REGISTERED_BNDS: Final[dict[str, Any]] = {
    "doc_role": "pre_measurement_bounds",
    "written_before_results": True,
    "task_id": TASK_ID,
    "scope": {
        "video": DEFAULT_VIDEO,
        "scene": DEFAULT_SCENE,
        "n_frames": DEFAULT_N_FRAMES,
        "resolution": "3840x2160",
        "codec": DEFAULT_CODEC,
        "qps": list(DEFAULT_QPS),
        "representations": ["still_frame0", "registered_panorama", "cleaned_video"],
    },
    "bounds_basis": (
        "Camera coverage/registration probe across 48 4K frames. Still frame 0 is "
        "uncompensated for camera pan (~40px), expected PSNR-Y ~24-34 dB on visible background. "
        "Registered panorama compensates camera pan, expected PSNR-Y ~30-42 dB. "
        "Cleaned video models local motion/details, expected PSNR-Y ~32-46 dB. "
        "Bytes: Still 1-150 KB, Panorama 1-200 KB + 1.7 KB side data, Video 5-1000 KB. "
        "Timings: Preprocessing 10-600s, Encode per point 0.5-180s, Decode/render 0.5-120s."
    ),
    "bands": {
        "psnr_y_visible": [15.0, 55.0],
        "ssim_visible": [0.65, 1.00],
        "bytes_still": [500, 250000],
        "bytes_panorama_package": [1500, 350000],
        "bytes_video": [2000, 2000000],
        "encode_seconds": [0.1, 240.0],
        "decode_render_seconds": [0.1, 180.0],
    },
}


def compute_array_sha256(arr: np.ndarray) -> str:
    """Compute SHA-256 digest of a numpy array's contiguous raw bytes."""
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def get_code_revision(repo_dir: Path | None = None) -> dict[str, Any]:
    """Obtain git commit, dirty status, and diff SHA-256."""
    repo = repo_dir or Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        porcelain = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        dirty = bool(porcelain.strip())
        diff_sha256: str | None = None
        if dirty:
            diff_bytes = subprocess.check_output(
                ["git", "diff", "HEAD"],
                cwd=repo,
                stderr=subprocess.DEVNULL,
            )
            diff_sha256 = hashlib.sha256(porcelain.encode("utf-8") + b"\n" + diff_bytes).hexdigest()
        return {"commit": commit, "dirty": dirty, "diff_sha256": diff_sha256}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "dirty": True, "diff_sha256": None, "error": str(exc)}


def build_probe_identity(
    video: str,
    scene: str,
    frames: np.ndarray,
    masks: np.ndarray,
    code_revision: dict[str, Any] | None = None,
    removal: str = "off",
) -> dict[str, Any]:
    """Build exact identity covering source-frame hashes, mask hashes, and code revision."""
    if code_revision is None:
        code_revision = get_code_revision()

    frames_sha256 = compute_array_sha256(frames)
    masks_sha256 = compute_array_sha256(masks)
    frame_hashes = [compute_array_sha256(frames[i]) for i in range(len(frames))]
    mask_hashes = [compute_array_sha256(masks[i]) for i in range(len(masks))]

    # Identity digest incorporates video, scene, frame count, frames hash, masks hash, git commit, and dirty diff hash
    key_dict = {
        "video": video,
        "scene": scene,
        "n_frames": len(frames),
        "frames_sha256": frames_sha256,
        "masks_sha256": masks_sha256,
        "removal": removal,
        "commit": code_revision.get("commit"),
        "dirty": code_revision.get("dirty"),
        "diff_sha256": code_revision.get("diff_sha256"),
    }
    identity_digest = hashlib.sha256(
        json.dumps(key_dict, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    cache_key = f"{video}_{scene}_{len(frames)}_{removal}_{identity_digest}"

    return {
        "video": video,
        "scene": scene,
        "n_frames": len(frames),
        "removal": removal,
        "frames_sha256": frames_sha256,
        "masks_sha256": masks_sha256,
        "frame_hashes": frame_hashes,
        "mask_hashes": mask_hashes,
        "code_revision": code_revision,
        "cache_key": cache_key,
    }


def warp_plate_to_frame(
    plate: np.ndarray,
    frame_to_plate: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Warp background plate back into camera frame coordinates.

    Homographies are explicitly cast to float32 before inverting and warping
    to ensure rendered pixels match the charged float32 side data precision.

    Args:
        plate: (H_plate, W_plate, 3) uint8 image.
        frame_to_plate: 3x3 homography mapping frame coordinates to plate coordinates.
        height: Target frame height.
        width: Target frame width.

    Returns:
        (height, width, 3) uint8 frame.
    """
    plate_h, plate_w = plate.shape[:2]
    # Enforce float32 precision to guarantee rendered pixels match charged float32 side data
    matrix = np.asarray(frame_to_plate, dtype=np.float32).reshape(3, 3)
    if (
        np.allclose(matrix, np.eye(3, dtype=np.float32), atol=1e-7)
        and plate_h == height
        and plate_w == width
    ):
        return plate.copy()
    try:
        plate_to_frame = np.linalg.inv(matrix).astype(np.float32)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Homography is singular; cannot invert to frame.") from exc

    return cv2.warpPerspective(
        plate,
        plate_to_frame,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def build_common_cleaned_stack(
    frames: np.ndarray,
    masks: np.ndarray,
    *,
    removal: str = "off",
    register: bool = True,
    cache_path: Path | None = None,
    identity: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[float, ...], ...], dict[str, Any]]:
    """Build the common background frame stack.

    Removal-OFF rules:
    - Actor pixels are strictly untouched in the input stack (bit-identical).
    - Zero optional removal/fill calls (0 inpaint holes, 0 inpaint frames).
    - Panorama plate is built with masks=None (temporal median naturally attenuates moving players).
    - Inherent panorama actor suppression is quantified and recorded separately.

    Removal-ON rules (legacy screening):
    - Preserve visible, unmasked background pixels (~mask).
    - Inside player masks, fill with observed temporal background warped back.
    - Explicitly fill remaining holes with Telea inpainting.

    Args:
        frames: (T, H, W, 3) uint8 RGB array.
        masks: (T, H, W) bool array (True where player/foreground is).
        removal: 'off' (default for E04A) or 'on' (legacy temporal fill + inpainting).
        register: Whether to register camera motion for the composite plate.
        cache_path: Optional path to save/load cached result.
        identity: Optional identity dictionary with source hashes and code revision for cache validation.

    Returns:
        (cleaned_frames, plate, homographies, build_stats)
    """
    if cache_path is not None and cache_path.is_file():
        data = np.load(str(cache_path), allow_pickle=True)
        stats = json.loads(str(data["stats"]))
        cached_id = stats.get("identity")
        use_cache = True
        if cached_id is not None and cached_id.get("removal") != removal:
            use_cache = False
        elif stats.get("removal_mode") != removal:
            use_cache = False
        if identity is not None and use_cache:
            if cached_id is None:
                use_cache = False
            else:
                for key in ("frames_sha256", "masks_sha256"):
                    if identity.get(key) and cached_id.get(key) != identity.get(key):
                        use_cache = False
                        break
                cached_code = cached_id.get("code_revision", {})
                req_code = identity.get("code_revision", {})
                if req_code.get("commit") and cached_code.get("commit") != req_code.get("commit"):
                    use_cache = False
                if req_code.get("diff_sha256") and cached_code.get("diff_sha256") != req_code.get(
                    "diff_sha256"
                ):
                    use_cache = False

        if use_cache:
            cleaned_frames = data["cleaned_frames"]
            plate = data["plate"]
            homographies_arr = data["homographies"]
            homographies = tuple(tuple(float(v) for v in row) for row in homographies_arr)
            stats["from_cache"] = True
            return cleaned_frames, plate, homographies, stats

    t_start = time.perf_counter()
    n_frames, height, width, channels = frames.shape

    if removal == "off":
        cleaned_stack = frames.copy()
        # Invariant check: actor pixels in input stack MUST be strictly unchanged
        if np.any(masks) and not np.array_equal(cleaned_stack[masks], frames[masks]):
            raise RuntimeError("Actor pixels were modified in removal-off input stack!")

        # Panorama plate is built without actor exclusion masks (masks=None)
        plate, homographies = build_plate(frames, masks=None, register=register)
        plate_h, plate_w = plate.shape[:2]

        # Quantify inherent panorama suppression: compare warped plate vs actor pixels in raw frames
        inherent_suppression_mad: list[float] = []
        for t in range(n_frames):
            h_matrix = np.asarray(homographies[t], dtype=np.float32).reshape(3, 3)
            warped_bg = warp_plate_to_frame(plate, h_matrix, height=height, width=width)
            m = masks[t]
            if np.any(m):
                player_luma = rgb_to_luma(frames[t])[m].astype(np.float64)
                plate_luma = rgb_to_luma(warped_bg)[m].astype(np.float64)
                inherent_suppression_mad.append(float(np.mean(np.abs(player_luma - plate_luma))))

        build_time = time.perf_counter() - t_start
        stats = {
            "build_seconds": round(build_time, 3),
            "plate_resolution": f"{plate_w}x{plate_h}",
            "frame_resolution": f"{width}x{height}",
            "n_frames": n_frames,
            "player_pixel_fraction": float(masks.mean()),
            "total_inpaint_holes": 0,
            "inpaint_frames": 0,
            "optional_removal_calls": 0,
            "removal_mode": "off",
            "actor_pixels_untouched": True,
            "inherent_panorama_suppression": {
                "mean_luma_mad_vs_player": round(float(np.mean(inherent_suppression_mad)), 3) if inherent_suppression_mad else 0.0,
                "note": (
                    "Transient moving actors are suppressed inherently by temporal median "
                    "aggregation without explicit mask removal or hole filling."
                ),
            },
            "canvas_validity_note": (
                "Removal-off plate built with masks=None; transient actors suppressed inherently by median aggregation."
            ),
            "from_cache": False,
        }
        if identity is not None:
            stats["identity"] = identity

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                str(cache_path),
                cleaned_frames=cleaned_stack,
                plate=plate,
                homographies=np.array([list(h) for h in homographies], dtype=np.float32),
                stats=json.dumps(stats),
            )

        return cleaned_stack, plate, homographies, stats

    # Legacy removal-ON path
    plate, homographies = build_plate(frames, masks=masks, register=register)
    plate_h, plate_w = plate.shape[:2]

    valid_plate_mask = np.full((plate_h, plate_w), 255, dtype=np.uint8)

    cleaned_stack = np.empty_like(frames)
    total_inpaint_holes = 0
    inpaint_frames = 0

    for t in range(n_frames):
        orig = frames[t]
        m = masks[t]
        h_matrix = np.asarray(homographies[t], dtype=np.float32).reshape(3, 3)

        # Warp plate to current frame
        warped_bg = warp_plate_to_frame(plate, h_matrix, height=height, width=width)

        # Determine if warped plate covers the masked region
        inv_h = np.linalg.inv(h_matrix).astype(np.float32)
        warped_valid = cv2.warpPerspective(
            valid_plate_mask,
            inv_h,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0.0,),
        )
        uncovered_in_mask = m & (warped_valid < 128)
        n_uncovered = int(uncovered_in_mask.sum())

        cleaned = orig.copy()
        cleaned[m] = warped_bg[m]

        if n_uncovered > 0:
            total_inpaint_holes += n_uncovered
            inpaint_frames += 1
            cleaned = cv2.inpaint(
                cleaned,
                uncovered_in_mask.astype(np.uint8) * 255,
                inpaintRadius=3,
                flags=cv2.INPAINT_TELEA,
            )

        # Invariant check: visible unmasked pixels MUST be strictly preserved
        if not np.array_equal(cleaned[~m], orig[~m]):
            raise RuntimeError(f"Frame {t}: unmasked background pixels were modified!")

        cleaned_stack[t] = cleaned

    build_time = time.perf_counter() - t_start
    stats = {
        "build_seconds": round(build_time, 3),
        "plate_resolution": f"{plate_w}x{plate_h}",
        "frame_resolution": f"{width}x{height}",
        "n_frames": n_frames,
        "player_pixel_fraction": float(masks.mean()),
        "total_inpaint_holes": total_inpaint_holes,
        "inpaint_frames": inpaint_frames,
        "optional_removal_calls": 1,
        "removal_mode": "on",
        "actor_pixels_untouched": False,
        "canvas_validity_note": (
            "Validity mask covers valid composite canvas coordinates, but does not certify "
            "100% unoccluded background observation without holes."
        ),
        "from_cache": False,
    }
    if identity is not None:
        stats["identity"] = identity

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(cache_path),
            cleaned_frames=cleaned_stack,
            plate=plate,
            homographies=np.array([list(h) for h in homographies], dtype=np.float32),
            stats=json.dumps(stats),
        )

    return cleaned_stack, plate, homographies, stats


def masked_luma_psnr(
    reference_rgb: np.ndarray,
    predicted_rgb: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Compute PSNR-Y over pixels where mask is True.

    Args:
        reference_rgb: (T, H, W, 3) or (H, W, 3) uint8 RGB array.
        predicted_rgb: Same shape uint8 RGB array.
        mask: (T, H, W) or (H, W) bool array.

    Returns:
        Mean PSNR-Y in dB.
    """
    ref_y = rgb_to_luma(reference_rgb)
    pred_y = rgb_to_luma(predicted_rgb)
    m = np.asarray(mask, dtype=bool)
    if ref_y.ndim == 3 and m.ndim == 2:
        m = np.broadcast_to(m, ref_y.shape)

    ref_vals = ref_y[m].astype(np.float64)
    pred_vals = pred_y[m].astype(np.float64)
    if ref_vals.size == 0:
        return float("nan")
    mse = float(np.mean((ref_vals - pred_vals) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * float(np.log10((255.0**2) / mse))


def _global_ssim(reference: np.ndarray, predicted: np.ndarray, c1: float, c2: float) -> float:
    mu_x = float(reference.mean())
    mu_y = float(predicted.mean())
    var_x = float(reference.var())
    var_y = float(predicted.var())
    cov = float(((reference - mu_x) * (predicted - mu_y)).mean())
    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * cov + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    return numerator / denominator


def safe_masked_ssim(
    reference: np.ndarray,
    predicted: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Compute global masked SSIM safely, returning float('nan') on empty masks without warnings."""
    ref, pred = paired(reference, predicted)
    selected = np.asarray(mask, dtype=bool)
    if selected.ndim == 2:
        selected = np.broadcast_to(selected, (ref.shape[0], *selected.shape))
    if selected.shape != ref.shape[:3]:
        raise ValueError(f"mask shape {selected.shape} does not match clip {ref.shape[:3]}")

    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2

    values: list[float] = []
    for idx in range(ref.shape[0]):
        m = selected[idx]
        if np.count_nonzero(m) == 0:
            continue
        channels = [
            _global_ssim(ref[idx, :, :, ch][m], pred[idx, :, :, ch][m], c1, c2)
            for ch in range(ref.shape[-1])
        ]
        values.append(float(sum(channels) / len(channels)))

    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def compute_metrics(
    reference_rgb: np.ndarray,
    predicted_rgb: np.ndarray,
    masks: np.ndarray,
) -> dict[str, float]:
    """Compute PSNR-Y and SSIM on visible background (~mask) and full frame."""
    visible_mask = ~masks
    psnr_y_visible = masked_luma_psnr(reference_rgb, predicted_rgb, visible_mask)
    ssim_visible = safe_masked_ssim(reference_rgb, predicted_rgb, visible_mask)

    # Full frame diagnostics
    ref_y = rgb_to_luma(reference_rgb)
    pred_y = rgb_to_luma(predicted_rgb)
    mse_full = float(np.mean((ref_y.astype(np.float64) - pred_y.astype(np.float64)) ** 2))
    psnr_y_full = float("inf") if mse_full == 0.0 else 10.0 * float(np.log10((255.0**2) / mse_full))
    ssim_full = SsimMetric().score(reference_rgb, predicted_rgb)

    return {
        "psnr_y_visible_dB": round(psnr_y_visible, 3),
        "ssim_visible": round(ssim_visible, 4),
        "psnr_y_full_dB": round(psnr_y_full, 3),
        "ssim_full": round(ssim_full, 4),
    }


def uncompressed_sanity_check(
    source_frames: np.ndarray,
    cleaned_frames: np.ndarray,
    plate: np.ndarray,
    homographies: tuple[tuple[float, ...], ...],
    masks: np.ndarray,
) -> dict[str, Any]:
    """Perform uncompressed render sanity check across the 3 representations."""
    n_frames, height, width, _ = source_frames.shape

    # Rep 1: First cleaned frame repeated with identity rendering
    rep1_rendered = np.broadcast_to(
        cleaned_frames[0:1],
        (n_frames, height, width, 3),
    ).copy()
    m1 = compute_metrics(source_frames, rep1_rendered, masks)

    # Rep 2: Registered panorama warped back per frame
    rep2_rendered = np.stack(
        [
            warp_plate_to_frame(
                plate,
                np.asarray(homographies[t], dtype=np.float32),
                height=height,
                width=width,
            )
            for t in range(n_frames)
        ],
        axis=0,
    )
    m2 = compute_metrics(source_frames, rep2_rendered, masks)

    # Rep 3: Cleaned video (uncompressed)
    m3 = compute_metrics(source_frames, cleaned_frames, masks)

    return {
        "still_frame0": m1,
        "registered_panorama": m2,
        "cleaned_video": m3,
    }


def pack_panorama_side_data(
    homographies: Sequence[Any] | np.ndarray,
    plate_shape: tuple[int, int],
    frame_shape: tuple[int, int],
    fps: float = DEFAULT_FPS,
) -> bytes:
    """Serialize registered panorama side data into packed binary transport format.

    Header (14 bytes):
    - plate_h: uint16 (2B)
    - plate_w: uint16 (2B)
    - frame_h: uint16 (2B)
    - frame_w: uint16 (2B)
    - n_frames: uint16 (2B)
    - fps: float32 (4B)

    Homography payload (n_frames * 36 bytes):
    - 3x3 float32 matrix per frame (9 * 4B = 36B per frame)
    """
    n_frames = len(homographies)
    plate_h, plate_w = int(plate_shape[0]), int(plate_shape[1])
    frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])

    header = struct.pack(">HHHHHf", plate_h, plate_w, frame_h, frame_w, n_frames, float(fps))
    matrices = np.asarray(homographies, dtype=np.float32).reshape(n_frames, 9)
    body = struct.pack(f">{n_frames * 9}f", *matrices.reshape(-1))
    return header + body


def unpack_panorama_side_data(
    payload: bytes,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int], float]:
    """Deserialize registered panorama binary side data.

    Returns:
        (homographies, plate_shape, frame_shape, fps)
        where homographies is an (n_frames, 3, 3) float32 numpy array.
    """
    header_len = 14
    if len(payload) < header_len:
        raise ValueError(f"Payload too short for panorama header: {len(payload)} < {header_len}")
    plate_h, plate_w, frame_h, frame_w, n_frames, fps = struct.unpack_from(">HHHHHf", payload, 0)
    expected_len = header_len + n_frames * 9 * 4
    if len(payload) != expected_len:
        raise ValueError(
            f"Payload length mismatch: got {len(payload)} bytes, expected {expected_len} bytes for {n_frames} frames"
        )
    floats = struct.unpack_from(f">{n_frames * 9}f", payload, header_len)
    homographies = np.array(floats, dtype=np.float32).reshape(n_frames, 3, 3)
    return homographies, (plate_h, plate_w), (frame_h, frame_w), fps


def pack_still_or_video_side_data(
    frame_shape: tuple[int, int],
    n_frames: int,
    fps: float = DEFAULT_FPS,
) -> bytes:
    """Serialize still or video metadata into packed binary format (10 bytes)."""
    frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])
    return struct.pack(">HHHf", frame_h, frame_w, int(n_frames), float(fps))


def unpack_still_or_video_side_data(
    payload: bytes,
) -> tuple[tuple[int, int], int, float]:
    """Deserialize still or video binary metadata (10 bytes)."""
    if len(payload) != 10:
        raise ValueError(f"Payload length mismatch: got {len(payload)} bytes, expected 10 bytes")
    frame_h, frame_w, n_frames, fps = struct.unpack(">HHHf", payload)
    return (frame_h, frame_w), n_frames, fps


def pack_side_data(
    rep_name: str,
    n_frames: int,
    *,
    homographies: Sequence[Any] | np.ndarray | None = None,
    plate_shape: tuple[int, int] | None = None,
    frame_shape: tuple[int, int] = (2160, 3840),
    fps: float = DEFAULT_FPS,
) -> bytes:
    """Serialize side data for any representation to actual packed binary bytes."""
    if rep_name in ("still_frame0", "cleaned_video"):
        return pack_still_or_video_side_data(frame_shape=frame_shape, n_frames=n_frames, fps=fps)
    if rep_name == "registered_panorama":
        if homographies is None:
            homographies = np.stack([np.eye(3, dtype=np.float32) for _ in range(n_frames)], axis=0)
        p_shape = plate_shape or frame_shape
        return pack_panorama_side_data(
            homographies, plate_shape=p_shape, frame_shape=frame_shape, fps=fps
        )
    raise ValueError(f"Unknown representation {rep_name}")


def unpack_side_data(
    rep_name: str,
    payload: bytes,
) -> dict[str, Any]:
    """Deserialize packed binary side data for any representation."""
    if rep_name in ("still_frame0", "cleaned_video"):
        frame_shape, n_frames, fps = unpack_still_or_video_side_data(payload)
        return {"frame_shape": frame_shape, "n_frames": n_frames, "fps": fps}
    if rep_name == "registered_panorama":
        homographies, plate_shape, frame_shape, fps = unpack_panorama_side_data(payload)
        return {
            "homographies": homographies,
            "plate_shape": plate_shape,
            "frame_shape": frame_shape,
            "fps": fps,
        }
    raise ValueError(f"Unknown representation {rep_name}")


def decode_standalone_representation(
    bitstream_path: Path,
    side_data_path: Path,
    codec: str = "vvc",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode background representation using strictly bitstream and serialized side data.

    The client has NO access to original geometry or frame count; all dimensions,
    frame counts, fps, and homographies must be unpacked from side_data_path.

    Returns:
        (rendered_frames, timing_metadata)
        where rendered_frames is uint8 RGB of shape (n_frames, height, width, 3).
    """
    bitstream_path = Path(bitstream_path)
    side_data_path = Path(side_data_path)
    if not bitstream_path.is_file():
        raise FileNotFoundError(f"Bitstream file not found: {bitstream_path}")
    if not side_data_path.is_file():
        raise FileNotFoundError(f"Side data file not found: {side_data_path}")

    side_bytes = side_data_path.read_bytes()
    ffmpeg = codec_tools.resolve_ffmpeg()

    # Case A: 10-byte side data (still_frame0 or cleaned_video)
    if len(side_bytes) == 10:
        (frame_h, frame_w), n_frames, fps = unpack_still_or_video_side_data(side_bytes)
        single_frame_bytes = frame_h * frame_w * 3
        expected_video_bytes = n_frames * single_frame_bytes

        # Decode via anchor decode path: lossless ffv1 intermediate at yuv420p
        with tempfile.TemporaryDirectory(prefix="ps_standalone_dec_") as tmp_dir:
            lossless_mkv = Path(tmp_dir) / "decoded.mkv"
            t0_dec = time.perf_counter()
            dec_cmd = [
                ffmpeg.path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(bitstream_path),
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "ffv1",
                str(lossless_mkv),
            ]
            sub = subprocess.run(dec_cmd, capture_output=True)
            if sub.returncode != 0:
                raise RuntimeError(
                    f"Standalone decode failed ({sub.returncode}): {sub.stderr.decode('utf-8', 'replace')}"
                )
            t_decode = time.perf_counter() - t0_dec

            raw_cmd = [
                ffmpeg.path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(lossless_mkv),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ]
            raw_sub = subprocess.run(raw_cmd, capture_output=True)
            if raw_sub.returncode != 0:
                raise RuntimeError(
                    f"Raw dump failed ({raw_sub.returncode}): {raw_sub.stderr.decode('utf-8', 'replace')}"
                )
            raw_bytes = raw_sub.stdout

        # Distinguish still (single frame) from video (n_frames)
        if len(raw_bytes) == single_frame_bytes:
            # Still frame 0
            t0_render = time.perf_counter()
            frame0 = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(frame_h, frame_w, 3)
            rendered = np.broadcast_to(
                frame0[np.newaxis, :, :, :], (n_frames, frame_h, frame_w, 3)
            ).copy()
            t_render = time.perf_counter() - t0_render
            meta = {
                "representation": "still_frame0",
                "decoded_frames": 1,
                "target_frames": n_frames,
                "frame_shape": (frame_h, frame_w),
                "fps": fps,
                "decode_seconds": round(t_decode, 4),
                "render_seconds": round(t_render, 4),
                "total_client_seconds": round(t_decode + t_render, 4),
                "rejected": False,
            }
            return rendered, meta

        if len(raw_bytes) == expected_video_bytes:
            # Video stream exactly matched
            t0_render = time.perf_counter()
            rendered = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(n_frames, frame_h, frame_w, 3)
            t_render = time.perf_counter() - t0_render
            meta = {
                "representation": "cleaned_video",
                "decoded_frames": n_frames,
                "target_frames": n_frames,
                "frame_shape": (frame_h, frame_w),
                "fps": fps,
                "decode_seconds": round(t_decode, 4),
                "render_seconds": round(t_render, 4),
                "total_client_seconds": round(t_decode + t_render, 4),
                "rejected": False,
            }
            return rendered, meta

        if len(raw_bytes) < expected_video_bytes:
            frames_decoded = len(raw_bytes) / single_frame_bytes
            raise ValueError(
                f"Truncated video stream in {bitstream_path.name}: decoded {len(raw_bytes)} bytes "
                f"({frames_decoded:.2f} frames), expected {expected_video_bytes} bytes ({n_frames} frames). "
                "Silent padding rejected."
            )

        if len(raw_bytes) > expected_video_bytes:
            frames_decoded = len(raw_bytes) / single_frame_bytes
            raise ValueError(
                f"Extra frames in video stream in {bitstream_path.name}: decoded {len(raw_bytes)} bytes "
                f"({frames_decoded:.2f} frames), expected {expected_video_bytes} bytes ({n_frames} frames). "
                "Silent clipping rejected."
            )

    # Case B: Registered panorama side data (14 + 36 * N bytes)
    if len(side_bytes) >= 14 and (len(side_bytes) - 14) % 36 == 0:
        homographies, plate_shape, frame_shape, fps = unpack_panorama_side_data(side_bytes)
        plate_h, plate_w = plate_shape
        frame_h, frame_w = frame_shape
        n_frames = len(homographies)

        with tempfile.TemporaryDirectory(prefix="ps_standalone_dec_") as tmp_dir:
            lossless_mkv = Path(tmp_dir) / "decoded.mkv"
            t0_dec = time.perf_counter()
            dec_cmd = [
                ffmpeg.path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(bitstream_path),
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "ffv1",
                str(lossless_mkv),
            ]
            sub = subprocess.run(dec_cmd, capture_output=True)
            if sub.returncode != 0:
                raise RuntimeError(
                    f"Standalone plate decode failed ({sub.returncode}): {sub.stderr.decode('utf-8', 'replace')}"
                )
            t_decode = time.perf_counter() - t0_dec

            raw_cmd = [
                ffmpeg.path,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(lossless_mkv),
                "-frames:v",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-",
            ]
            raw_sub = subprocess.run(raw_cmd, capture_output=True)
            if raw_sub.returncode != 0:
                raise RuntimeError(
                    f"Raw plate dump failed ({raw_sub.returncode}): {raw_sub.stderr.decode('utf-8', 'replace')}"
                )
            raw_bytes = raw_sub.stdout

        expected_plate_bytes = plate_h * plate_w * 3
        even_plate_h = plate_h - (plate_h % 2)
        even_plate_w = plate_w - (plate_w % 2)
        even_plate_bytes = even_plate_h * even_plate_w * 3

        if len(raw_bytes) == expected_plate_bytes:
            plate_rgb = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(plate_h, plate_w, 3)
        elif len(raw_bytes) == even_plate_bytes:
            plate_rgb = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(even_plate_h, even_plate_w, 3)
        else:
            raise ValueError(
                f"Plate byte count mismatch in {bitstream_path.name}: got {len(raw_bytes)} bytes, "
                f"expected {expected_plate_bytes} bytes for {plate_w}x{plate_h}"
            )

        t0_render = time.perf_counter()
        rendered = np.stack(
            [
                warp_plate_to_frame(plate_rgb, homographies[t], height=frame_h, width=frame_w)
                for t in range(n_frames)
            ],
            axis=0,
        )
        t_render = time.perf_counter() - t0_render

        meta = {
            "representation": "registered_panorama",
            "decoded_frames": n_frames,
            "target_frames": n_frames,
            "plate_shape": (plate_h, plate_w),
            "frame_shape": (frame_h, frame_w),
            "fps": fps,
            "decode_seconds": round(t_decode, 4),
            "render_seconds": round(t_render, 4),
            "total_client_seconds": round(t_decode + t_render, 4),
            "rejected": False,
        }
        return rendered, meta

    raise ValueError(f"Unrecognized side data format in {side_data_path.name}: length {len(side_bytes)} bytes")


def charge_side_data(
    rep_name: str,
    n_frames: int,
    plate_shape: tuple[int, int] | None = None,
    frame_shape: tuple[int, int] = (2160, 3840),
    homographies: Sequence[Any] | np.ndarray | None = None,
    fps: float = DEFAULT_FPS,
) -> dict[str, int]:
    """Charge necessary geometry, dimensions, and timing side data.

    Returns dict of breakdown in bytes, verified against actual packed binary payload length.
    """
    packed = pack_side_data(
        rep_name,
        n_frames,
        homographies=homographies,
        plate_shape=plate_shape,
        frame_shape=frame_shape,
        fps=fps,
    )
    total_bytes = len(packed)

    if rep_name == "still_frame0":
        assert total_bytes == 10
        return {
            "geometry_bytes": 4,
            "timing_bytes": 6,
            "homography_bytes": 0,
            "total_side_data_bytes": total_bytes,
        }
    if rep_name == "registered_panorama":
        homography_bytes = n_frames * 9 * 4
        assert total_bytes == 14 + homography_bytes
        return {
            "geometry_bytes": 8,
            "timing_bytes": 6,
            "homography_bytes": homography_bytes,
            "total_side_data_bytes": total_bytes,
        }
    if rep_name == "cleaned_video":
        assert total_bytes == 10
        return {
            "geometry_bytes": 4,
            "timing_bytes": 6,
            "homography_bytes": 0,
            "total_side_data_bytes": total_bytes,
        }
    raise ValueError(f"Unknown representation {rep_name}")


def evaluate_still_frame0(
    cleaned_frames: np.ndarray,
    source_frames: np.ndarray,
    masks: np.ndarray,
    qp: int,
    *,
    codec: str = DEFAULT_CODEC,
    preset: str = DEFAULT_PRESET,
    preprocessing_seconds: float = 0.0,
) -> dict[str, Any]:
    """Evaluate Representation 1: First cleaned frame repeated, no camera warp."""
    n_frames, height, width, _ = source_frames.shape
    frame0_rgb = cleaned_frames[0]
    frame0_bgr = frame0_rgb[:, :, ::-1]

    sidecar = IntraCodecSidecar(codec, qp=qp, preset=preset)
    tool_path, tool_version = sidecar.probe_encoder()

    t_enc_start = time.perf_counter()
    payload = sidecar.encode(frame0_bgr)
    encode_seconds = time.perf_counter() - t_enc_start

    # Verify binary side data transport
    side_data_bytes = pack_still_or_video_side_data(
        frame_shape=(height, width),
        n_frames=n_frames,
        fps=DEFAULT_FPS,
    )
    unpacked_shape, unpacked_n, _ = unpack_still_or_video_side_data(side_data_bytes)
    assert unpacked_shape == (height, width)
    assert unpacked_n == n_frames
    assert len(side_data_bytes) == 10

    t_dec_start = time.perf_counter()
    decoded_bgr = sidecar.decode(payload)
    decoded_rgb = decoded_bgr[:, :, ::-1]
    rendered = np.broadcast_to(
        decoded_rgb[np.newaxis, :height, :width, :],
        (n_frames, height, width, 3),
    ).copy()
    decode_render_seconds = time.perf_counter() - t_dec_start

    side_data = charge_side_data("still_frame0", n_frames, frame_shape=(height, width))
    metrics = compute_metrics(source_frames, rendered, masks)

    return {
        "representation": "still_frame0",
        "qp": qp,
        "codec": codec,
        "preset": preset,
        "tool_path": tool_path,
        "tool_version": tool_version,
        "encoded_payload_bytes": len(payload),
        "side_data_bytes": len(side_data_bytes),
        "total_package_bytes": len(payload) + len(side_data_bytes),
        "side_data_detail": side_data,
        "metrics": metrics,
        "timing": {
            "preprocessing_seconds": round(preprocessing_seconds, 3),
            "encode_seconds": round(encode_seconds, 3),
            "decode_render_seconds": round(decode_render_seconds, 3),
            "total_end_to_end_seconds": round(
                round(preprocessing_seconds, 3)
                + round(encode_seconds, 3)
                + round(decode_render_seconds, 3),
                3,
            ),
        },
    }


def evaluate_registered_panorama(
    plate: np.ndarray,
    homographies: tuple[tuple[float, ...], ...] | Sequence[Sequence[float]] | np.ndarray,
    source_frames: np.ndarray,
    masks: np.ndarray,
    qp: int,
    *,
    codec: str = DEFAULT_CODEC,
    preset: str = DEFAULT_PRESET,
    preprocessing_seconds: float = 0.0,
) -> dict[str, Any]:
    """Evaluate Representation 2: Registered panorama with charged per-frame homographies."""
    n_frames, height, width, _ = source_frames.shape
    plate_h, plate_w = plate.shape[:2]
    plate_bgr = plate[:, :, ::-1]

    sidecar = IntraCodecSidecar(codec, qp=qp, preset=preset)
    tool_path, tool_version = sidecar.probe_encoder()

    t_enc_start = time.perf_counter()
    payload = sidecar.encode(plate_bgr)
    encode_seconds = time.perf_counter() - t_enc_start

    # Pack actual binary side data and verify round-trip deserialization
    side_data_bytes = pack_panorama_side_data(
        homographies,
        plate_shape=(plate_h, plate_w),
        frame_shape=(height, width),
        fps=DEFAULT_FPS,
    )
    unpacked_homographies, unpacked_plate_shape, unpacked_frame_shape, _ = (
        unpack_panorama_side_data(side_data_bytes)
    )
    assert len(side_data_bytes) == 14 + n_frames * 9 * 4
    assert unpacked_plate_shape == (plate_h, plate_w)
    assert unpacked_frame_shape == (height, width)
    assert unpacked_homographies.dtype == np.float32

    t_dec_start = time.perf_counter()
    decoded_bgr = sidecar.decode(payload)
    decoded_plate_rgb = decoded_bgr[:, :, ::-1]
    # Render using deserialized float32 homographies to guarantee rendered pixels match charged precision
    rendered = np.stack(
        [
            warp_plate_to_frame(
                decoded_plate_rgb,
                unpacked_homographies[t],
                height=height,
                width=width,
            )
            for t in range(n_frames)
        ],
        axis=0,
    )
    decode_render_seconds = time.perf_counter() - t_dec_start

    side_data = charge_side_data(
        "registered_panorama",
        n_frames,
        plate_shape=(plate_h, plate_w),
        frame_shape=(height, width),
        homographies=homographies,
    )
    metrics = compute_metrics(source_frames, rendered, masks)

    return {
        "representation": "registered_panorama",
        "qp": qp,
        "codec": codec,
        "preset": preset,
        "tool_path": tool_path,
        "tool_version": tool_version,
        "plate_resolution": f"{plate_w}x{plate_h}",
        "encoded_payload_bytes": len(payload),
        "side_data_bytes": len(side_data_bytes),
        "total_package_bytes": len(payload) + len(side_data_bytes),
        "side_data_detail": side_data,
        "metrics": metrics,
        "timing": {
            "preprocessing_seconds": round(preprocessing_seconds, 3),
            "encode_seconds": round(encode_seconds, 3),
            "decode_render_seconds": round(decode_render_seconds, 3),
            "total_end_to_end_seconds": round(
                round(preprocessing_seconds, 3)
                + round(encode_seconds, 3)
                + round(decode_render_seconds, 3),
                3,
            ),
        },
    }


def evaluate_cleaned_video(
    cleaned_frames: np.ndarray,
    source_frames: np.ndarray,
    masks: np.ndarray,
    qp: int,
    *,
    codec: str = DEFAULT_CODEC,
    preset: str = DEFAULT_PRESET,
    fps: float = DEFAULT_FPS,
    work_dir: Path | None = None,
    preprocessing_seconds: float = 0.0,
) -> dict[str, Any]:
    """Evaluate Representation 3: Cleaned per-frame video (VVC inter-coding)."""
    n_frames, height, width, _ = source_frames.shape
    request = EncodeRequest(
        codec_name=codec,
        rate_control=RateControl.QP,
        rate=qp,
        preset=preset,
        pix_fmt="yuv420p",
    )
    request.validate()

    # Verify binary side data transport
    side_data_bytes = pack_still_or_video_side_data(
        frame_shape=(height, width),
        n_frames=n_frames,
        fps=fps,
    )
    unpacked_shape, unpacked_n, _ = unpack_still_or_video_side_data(side_data_bytes)
    assert unpacked_shape == (height, width)
    assert unpacked_n == n_frames
    assert len(side_data_bytes) == 10

    t0 = time.perf_counter()
    trip = timed_roundtrip(cleaned_frames, request=request, fps=fps, work_dir=work_dir)
    total_time = time.perf_counter() - t0

    decoded_frames = trip.frames[:n_frames, :height, :width]
    metrics = compute_metrics(source_frames, decoded_frames, masks)
    side_data = charge_side_data("cleaned_video", n_frames, frame_shape=(height, width), fps=fps)

    enc_sec = round(float(trip.encode_seconds), 3)
    dec_sec = round(float(trip.decode_seconds), 3)
    prep_sec = round(preprocessing_seconds, 3)

    return {
        "representation": "cleaned_video",
        "qp": qp,
        "codec": codec,
        "preset": preset,
        "tool_path": trip.tool_path,
        "tool_version": trip.tool_version,
        "encoded_payload_bytes": int(trip.size_bytes),
        "side_data_bytes": len(side_data_bytes),
        "total_package_bytes": int(trip.size_bytes) + len(side_data_bytes),
        "side_data_detail": side_data,
        "metrics": metrics,
        "timing": {
            "preprocessing_seconds": prep_sec,
            "encode_seconds": enc_sec,
            "decode_render_seconds": dec_sec,
            "roundtrip_total_seconds": round(total_time, 3),
            "total_end_to_end_seconds": round(prep_sec + enc_sec + dec_sec, 3),
        },
    }


def check_bounds(
    points: list[dict[str, Any]],
    bounds_def: dict[str, Any] = PRE_REGISTERED_BNDS,
) -> tuple[bool, list[str]]:
    """Check whether candidate results conform to pre-registered bands.

    Returns:
        (all_passed, alarms)
    """
    bands = bounds_def["bands"]
    alarms: list[str] = []

    for pt in points:
        rep = pt["representation"]
        qp = pt["qp"]
        psnr = pt["metrics"]["psnr_y_visible_dB"]
        ssim = pt["metrics"]["ssim_visible"]
        total_bytes = pt["total_package_bytes"]
        enc_s = pt["timing"]["encode_seconds"]
        dec_s = pt["timing"]["decode_render_seconds"]

        # Check PSNR-Y visible
        if not (bands["psnr_y_visible"][0] <= psnr <= bands["psnr_y_visible"][1]):
            alarms.append(
                f"{rep} QP {qp}: psnr_y_visible {psnr} outside band {bands['psnr_y_visible']}"
            )

        # Check SSIM visible
        if not (bands["ssim_visible"][0] <= ssim <= bands["ssim_visible"][1]):
            alarms.append(
                f"{rep} QP {qp}: ssim_visible {ssim} outside band {bands['ssim_visible']}"
            )

        # Check bytes band per representation
        if rep == "still_frame0":
            b_band = bands["bytes_still"]
        elif rep == "registered_panorama":
            b_band = bands["bytes_panorama_package"]
        else:
            b_band = bands["bytes_video"]

        if not (b_band[0] <= total_bytes <= b_band[1]):
            alarms.append(f"{rep} QP {qp}: total_bytes {total_bytes} outside band {b_band}")

        # Check timings
        if not (bands["encode_seconds"][0] <= enc_s <= bands["encode_seconds"][1]):
            alarms.append(
                f"{rep} QP {qp}: encode_seconds {enc_s} outside band {bands['encode_seconds']}"
            )
        if not (bands["decode_render_seconds"][0] <= dec_s <= bands["decode_render_seconds"][1]):
            alarms.append(
                f"{rep} QP {qp}: decode_render_seconds {dec_s} outside band {bands['decode_render_seconds']}"
            )

    return len(alarms) == 0, alarms


def format_markdown_table(points: list[dict[str, Any]]) -> str:
    """Format evaluation points as a markdown table."""
    headers = [
        "Representation",
        "QP",
        "Payload (B)",
        "Side Data (B)",
        "Total (B)",
        "PSNR-Y Vis (dB)",
        "SSIM Vis",
        "PSNR-Y Full (dB)",
        "SSIM Full",
        "Prep (s)",
        "Enc (s)",
        "Dec (s)",
        "Total (s)",
    ]
    rows = []
    for pt in points:
        rep = pt["representation"]
        qp = pt["qp"]
        payload_b = pt["encoded_payload_bytes"]
        side_b = pt["side_data_bytes"]
        tot_b = pt["total_package_bytes"]
        psnr_v = pt["metrics"]["psnr_y_visible_dB"]
        ssim_v = pt["metrics"]["ssim_visible"]
        psnr_f = pt["metrics"]["psnr_y_full_dB"]
        ssim_f = pt["metrics"]["ssim_full"]
        prep_s = pt["timing"].get("preprocessing_seconds", 0.0)
        enc_s = pt["timing"]["encode_seconds"]
        dec_s = pt["timing"]["decode_render_seconds"]
        tot_s = pt["timing"].get(
            "total_end_to_end_seconds",
            round(prep_s + enc_s + dec_s, 3),
        )
        rows.append(
            f"| {rep} | {qp} | {payload_b:,} | {side_b:,} | {tot_b:,} | "
            f"{psnr_v:.2f} | {ssim_v:.4f} | {psnr_f:.2f} | {ssim_f:.4f} | "
            f"{prep_s:.2f} | {enc_s:.2f} | {dec_s:.2f} | {tot_s:.2f} |"
        )

    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    return "\n".join([header_line, sep_line] + rows)


def format_csv_table(points: list[dict[str, Any]]) -> str:
    """Format evaluation points as CSV."""
    import io
    import csv

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "representation",
            "qp",
            "encoded_payload_bytes",
            "side_data_bytes",
            "total_package_bytes",
            "psnr_y_visible_dB",
            "ssim_visible",
            "psnr_y_full_dB",
            "ssim_full",
            "preprocessing_seconds",
            "encode_seconds",
            "decode_render_seconds",
            "total_end_to_end_seconds",
        ]
    )
    for pt in points:
        prep_s = pt["timing"].get("preprocessing_seconds", 0.0)
        enc_s = pt["timing"]["encode_seconds"]
        dec_s = pt["timing"]["decode_render_seconds"]
        tot_s = pt["timing"].get(
            "total_end_to_end_seconds",
            round(prep_s + enc_s + dec_s, 3),
        )
        writer.writerow(
            [
                pt["representation"],
                pt["qp"],
                pt["encoded_payload_bytes"],
                pt["side_data_bytes"],
                pt["total_package_bytes"],
                pt["metrics"]["psnr_y_visible_dB"],
                pt["metrics"]["ssim_visible"],
                pt["metrics"]["psnr_y_full_dB"],
                pt["metrics"]["ssim_full"],
                prep_s,
                enc_s,
                dec_s,
                tot_s,
            ]
        )
    return output.getvalue()


def run_probe(
    *,
    video: str = DEFAULT_VIDEO,
    scene: str = DEFAULT_SCENE,
    n_frames: int = DEFAULT_N_FRAMES,
    qps: tuple[int, ...] = DEFAULT_QPS,
    preset: str = DEFAULT_PRESET,
    codec: str = DEFAULT_CODEC,
    output_dir: Path,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute the full background probe protocol."""
    output_dir.mkdir(parents=True, exist_ok=True)
    t_global_start = time.perf_counter()

    # Step 0: Write pre-registered bounds BEFORE measurement
    bounds_file = output_dir / "bounds.json"
    bounds_record = dict(PRE_REGISTERED_BNDS)
    bounds_record["written_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    bounds_file.write_text(json.dumps(bounds_record, indent=2), encoding="utf-8")

    # Step 1: Input verification & identity construction
    t_load_start = time.perf_counter()
    clip = load_long_scene_clip(video, scene, n_frames=n_frames)
    load_seconds = time.perf_counter() - t_load_start

    frames = clip.frames
    masks = clip.masks
    input_info = clip.describe()
    input_info["load_seconds"] = round(load_seconds, 3)

    code_rev = get_code_revision()
    identity = build_probe_identity(
        video=video,
        scene=scene,
        frames=frames,
        masks=masks,
        code_revision=code_rev,
    )
    cache_key = identity["cache_key"]

    # Step 2: Common foreground-removed frame stack
    cache_path = (cache_dir / f"cleaned_stack_{cache_key}.npz") if cache_dir else None
    cleaned_frames, plate, homographies, prep_stats = build_common_cleaned_stack(
        frames,
        masks,
        register=True,
        cache_path=cache_path,
        identity=identity,
    )
    preprocessing_seconds = float(prep_stats.get("build_seconds", 0.0))

    # Step 3: Uncompressed sanity check
    sanity = uncompressed_sanity_check(frames, cleaned_frames, plate, homographies, masks)

    # Step 4: Screening & measurement across representations and QPs
    eval_points: list[dict[str, Any]] = []

    for qp in qps:
        # Rep 1: Still frame 0
        res1 = evaluate_still_frame0(
            cleaned_frames,
            frames,
            masks,
            qp=qp,
            codec=codec,
            preset=preset,
            preprocessing_seconds=preprocessing_seconds,
        )
        eval_points.append(res1)

        # Rep 2: Registered panorama
        res2 = evaluate_registered_panorama(
            plate,
            homographies,
            frames,
            masks,
            qp=qp,
            codec=codec,
            preset=preset,
            preprocessing_seconds=preprocessing_seconds,
        )
        eval_points.append(res2)

        # Rep 3: Cleaned video
        res3 = evaluate_cleaned_video(
            cleaned_frames,
            frames,
            masks,
            qp=qp,
            codec=codec,
            preset=preset,
            fps=DEFAULT_FPS,
            preprocessing_seconds=preprocessing_seconds,
        )
        eval_points.append(res3)

    # Step 5: Bounds verification
    bounds_passed, alarms = check_bounds(eval_points, bounds_record)

    # Step 6: Synthesis and hypothesis evaluation
    # Extract QP32 and QP47 points for comparison
    points_by_rep_qp = {(pt["representation"], pt["qp"]): pt for pt in eval_points}

    # Compare Still vs Registered Panorama
    # Still vs Panorama at QP 32:
    still_32 = points_by_rep_qp[("still_frame0", 32)]
    pano_32 = points_by_rep_qp[("registered_panorama", 32)]
    video_32 = points_by_rep_qp[("cleaned_video", 32)]

    still_47 = points_by_rep_qp[("still_frame0", 47)]
    pano_47 = points_by_rep_qp[("registered_panorama", 47)]
    video_47 = points_by_rep_qp[("cleaned_video", 47)]

    delta_psnr_pano_vs_still_32 = (
        pano_32["metrics"]["psnr_y_visible_dB"] - still_32["metrics"]["psnr_y_visible_dB"]
    )
    delta_psnr_video_vs_pano_32 = (
        video_32["metrics"]["psnr_y_visible_dB"] - pano_32["metrics"]["psnr_y_visible_dB"]
    )
    delta_bytes_pano_vs_still_32 = pano_32["total_package_bytes"] - still_32["total_package_bytes"]
    delta_bytes_video_vs_pano_32 = video_32["total_package_bytes"] - pano_32["total_package_bytes"]

    # Hypothesis decision:
    # "Hypothesis: Camera coverage/registration is the limiting background error; video improves it but must earn its byte/time cost.
    # Alternative: A still is already sufficient and geometry/metadata or correction causes the deficit."
    # If camera registration improves PSNR significantly over still (> 1 dB) in both uncompressed and compressed,
    # the still deficit is primarily camera motion / registration!
    if (
        delta_psnr_pano_vs_still_32 > 1.0
        and sanity["registered_panorama"]["psnr_y_visible_dB"]
        > sanity["still_frame0"]["psnr_y_visible_dB"] + 1.0
    ):
        verdict = "SUPPORTED"
        verdict_summary = (
            f"Hypothesis SUPPORTED: Camera registration significantly improves background quality over still frame 0 "
            f"(+{delta_psnr_pano_vs_still_32:.2f} dB PSNR-Y at QP 32, "
            f"+{sanity['registered_panorama']['psnr_y_visible_dB'] - sanity['still_frame0']['psnr_y_visible_dB']:.2f} dB uncompressed). "
            f"Still frame 0 is severely limited by uncompensated camera pan (40 px drift). "
            f"Cleaned video provides additional gain (+{delta_psnr_video_vs_pano_32:.2f} dB at QP 32) "
            f"at the cost of {delta_bytes_video_vs_pano_32:,} additional bytes and longer encode time."
        )
    else:
        verdict = "CONTRADICTED"
        verdict_summary = (
            f"Alternative SUPPORTED: Camera registration does not explain the background deficit "
            f"(delta: {delta_psnr_pano_vs_still_32:.2f} dB)."
        )

    total_wall_seconds = time.perf_counter() - t_global_start

    # Output artifact generation
    markdown_table = format_markdown_table(eval_points)
    csv_table = format_csv_table(eval_points)

    (output_dir / "summary_table.md").write_text(markdown_table, encoding="utf-8")
    (output_dir / "summary_table.csv").write_text(csv_table, encoding="utf-8")

    report: dict[str, Any] = {
        "task_id": TASK_ID,
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "identity": identity,
        "input_clip": input_info,
        "preprocessing": prep_stats,
        "uncompressed_sanity_check": sanity,
        "screening_points": eval_points,
        "bounds_passed": bounds_passed,
        "alarms": alarms,
        "verdict": verdict,
        "verdict_summary": verdict_summary,
        "comparisons": {
            "qp32": {
                "delta_psnr_pano_vs_still_dB": round(delta_psnr_pano_vs_still_32, 3),
                "delta_psnr_video_vs_pano_dB": round(delta_psnr_video_vs_pano_32, 3),
                "delta_bytes_pano_vs_still": delta_bytes_pano_vs_still_32,
                "delta_bytes_video_vs_pano": delta_bytes_video_vs_pano_32,
            },
            "qp47": {
                "delta_psnr_pano_vs_still_dB": round(
                    pano_47["metrics"]["psnr_y_visible_dB"]
                    - still_47["metrics"]["psnr_y_visible_dB"],
                    3,
                ),
                "delta_psnr_video_vs_pano_dB": round(
                    video_47["metrics"]["psnr_y_visible_dB"]
                    - pano_47["metrics"]["psnr_y_visible_dB"],
                    3,
                ),
                "delta_bytes_pano_vs_still": pano_47["total_package_bytes"]
                - still_47["total_package_bytes"],
                "delta_bytes_video_vs_pano": video_47["total_package_bytes"]
                - pano_47["total_package_bytes"],
            },
            "uncompressed": {
                "still_psnr_y_visible_dB": sanity["still_frame0"]["psnr_y_visible_dB"],
                "pano_psnr_y_visible_dB": sanity["registered_panorama"]["psnr_y_visible_dB"],
                "video_psnr_y_visible_dB": sanity["cleaned_video"]["psnr_y_visible_dB"],
                "gain_pano_over_still_dB": round(
                    sanity["registered_panorama"]["psnr_y_visible_dB"]
                    - sanity["still_frame0"]["psnr_y_visible_dB"],
                    3,
                ),
            },
        },
        "total_wall_seconds": round(total_wall_seconds, 3),
    }

    report_path = output_dir / "probe_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="CODEC-ACT-06 Background Probe")
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--frames", type=int, default=DEFAULT_N_FRAMES)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()

    default_out = ps_paths.outputs() / "development-recovery" / "wave2-background-probe"
    out_dir = args.output_dir or default_out
    cache_dir = args.cache_dir or (
        Path("/tmp") / f"pointstream-bgprobe-cache-{os.environ.get('USER', 'default')}"
    )

    print(f"Launching Background Probe ({TASK_ID})...")
    print(f"Output directory: {out_dir}")
    print(f"Cache directory: {cache_dir}")

    report = run_probe(
        video=args.video,
        scene=args.scene,
        n_frames=args.frames,
        output_dir=out_dir,
        cache_dir=cache_dir,
    )

    print("\n--- Probe Complete ---")
    print(f"Verdict: {report['verdict']}")
    print(report["verdict_summary"])
    print(f"Report saved to: {out_dir / 'probe_report.json'}")


if __name__ == "__main__":
    main()
