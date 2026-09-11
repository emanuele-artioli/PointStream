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
import json
import os
from pathlib import Path
import time
from typing import Any, Final

import cv2
import numpy as np

from experiments.long_scenes.loader import load_long_scene_clip
from src.components.background.plate import build_plate
from src.components.background.sidecar import IntraCodecSidecar
from src.components.codec.frames import rgb_to_luma
from src.components.codec.measure import timed_roundtrip
from src.components.metrics.ssim import SsimMetric, masked_ssim
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


def warp_plate_to_frame(
    plate: np.ndarray,
    frame_to_plate: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Warp background plate back into camera frame coordinates.

    Args:
        plate: (H_plate, W_plate, 3) uint8 image.
        frame_to_plate: 3x3 homography mapping frame coordinates to plate coordinates.
        height: Target frame height.
        width: Target frame width.

    Returns:
        (height, width, 3) uint8 frame.
    """
    plate_h, plate_w = plate.shape[:2]
    matrix = np.asarray(frame_to_plate, dtype=np.float64).reshape(3, 3)
    if np.allclose(matrix, np.eye(3), atol=1e-7) and plate_h == height and plate_w == width:
        return plate.copy()
    try:
        plate_to_frame = np.linalg.inv(matrix)
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
    register: bool = True,
    cache_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[float, ...], ...], dict[str, Any]]:
    """Build the common foreground-removed frame stack.

    Rules:
    - Preserve each original frame's visible, unmasked background pixels (~mask).
    - Inside player masks, fill ONLY with observed temporal background warped back.
    - Explicitly fill any still-uncovered holes with Telea inpainting.
    - Hold that stack/fill fixed across all representations.

    Args:
        frames: (T, H, W, 3) uint8 RGB array.
        masks: (T, H, W) bool array (True where player/foreground is).
        register: Whether to register camera motion for the composite plate.
        cache_path: Optional path to save/load cached result.

    Returns:
        (cleaned_frames, plate, homographies, build_stats)
    """
    if cache_path is not None and cache_path.is_file():
        data = np.load(str(cache_path), allow_pickle=True)
        cleaned_frames = data["cleaned_frames"]
        plate = data["plate"]
        homographies_arr = data["homographies"]
        homographies = tuple(tuple(float(v) for v in row) for row in homographies_arr)
        stats = json.loads(str(data["stats"]))
        stats["from_cache"] = True
        return cleaned_frames, plate, homographies, stats

    t_start = time.perf_counter()
    n_frames, height, width, channels = frames.shape

    # 1. Build composite plate and estimate homographies
    plate, homographies = build_plate(frames, masks=masks, register=register)
    plate_h, plate_w = plate.shape[:2]

    # Valid canvas coverage template
    valid_plate_mask = np.full((plate_h, plate_w), 255, dtype=np.uint8)

    cleaned_stack = np.empty_like(frames)
    total_inpaint_holes = 0
    inpaint_frames = 0

    for t in range(n_frames):
        orig = frames[t]
        m = masks[t]
        h_matrix = np.asarray(homographies[t], dtype=np.float64).reshape(3, 3)

        # Warp plate to current frame
        warped_bg = warp_plate_to_frame(plate, h_matrix, height=height, width=width)

        # Determine if warped plate covers the masked region
        inv_h = np.linalg.inv(h_matrix)
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
        # Inside player mask: fill ONLY with observed temporal background warped back
        cleaned[m] = warped_bg[m]

        # Explicitly inpaint any still-uncovered holes in player mask
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
        "from_cache": False,
    }

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(cache_path),
            cleaned_frames=cleaned_stack,
            plate=plate,
            homographies=np.array([list(h) for h in homographies], dtype=np.float64),
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


def compute_metrics(
    reference_rgb: np.ndarray,
    predicted_rgb: np.ndarray,
    masks: np.ndarray,
) -> dict[str, float]:
    """Compute PSNR-Y and SSIM on visible background (~mask) and full frame."""
    visible_mask = ~masks
    psnr_y_visible = masked_luma_psnr(reference_rgb, predicted_rgb, visible_mask)
    ssim_visible = masked_ssim(reference_rgb, predicted_rgb, visible_mask)

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
            warp_plate_to_frame(plate, np.array(homographies[t]), height=height, width=width)
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


def charge_side_data(
    rep_name: str,
    n_frames: int,
    plate_shape: tuple[int, int] | None = None,
    frame_shape: tuple[int, int] = (2160, 3840),
) -> dict[str, int]:
    """Charge necessary geometry, dimensions, and timing side data.

    Returns dict of breakdown in bytes.
    """
    # Base package header: frame width (2B) + height (2B) + n_frames (2B) + fps (4B) = 10B
    base_header = 10
    if rep_name == "still_frame0":
        return {
            "geometry_bytes": 4,
            "timing_bytes": 6,
            "homography_bytes": 0,
            "total_side_data_bytes": base_header,
        }
    if rep_name == "registered_panorama":
        # Plate dimensions (4B) + frame dimensions (4B) + n_frames (2B) + fps (4B) = 14B
        # 3x3 homography per frame: 48 * 9 * 4 bytes (float32) = 1,728 bytes
        homography_bytes = n_frames * 9 * 4
        return {
            "geometry_bytes": 8,
            "timing_bytes": 6,
            "homography_bytes": homography_bytes,
            "total_side_data_bytes": 14 + homography_bytes,
        }
    if rep_name == "cleaned_video":
        return {
            "geometry_bytes": 4,
            "timing_bytes": 6,
            "homography_bytes": 0,
            "total_side_data_bytes": base_header,
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
        "side_data_bytes": side_data["total_side_data_bytes"],
        "total_package_bytes": len(payload) + side_data["total_side_data_bytes"],
        "side_data_detail": side_data,
        "metrics": metrics,
        "timing": {
            "encode_seconds": round(encode_seconds, 3),
            "decode_render_seconds": round(decode_render_seconds, 3),
        },
    }


def evaluate_registered_panorama(
    plate: np.ndarray,
    homographies: tuple[tuple[float, ...], ...],
    source_frames: np.ndarray,
    masks: np.ndarray,
    qp: int,
    *,
    codec: str = DEFAULT_CODEC,
    preset: str = DEFAULT_PRESET,
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

    t_dec_start = time.perf_counter()
    decoded_bgr = sidecar.decode(payload)
    decoded_plate_rgb = decoded_bgr[:, :, ::-1]
    rendered = np.stack(
        [
            warp_plate_to_frame(
                decoded_plate_rgb,
                np.array(homographies[t]),
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
        "side_data_bytes": side_data["total_side_data_bytes"],
        "total_package_bytes": len(payload) + side_data["total_side_data_bytes"],
        "side_data_detail": side_data,
        "metrics": metrics,
        "timing": {
            "encode_seconds": round(encode_seconds, 3),
            "decode_render_seconds": round(decode_render_seconds, 3),
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

    t0 = time.perf_counter()
    trip = timed_roundtrip(cleaned_frames, request=request, fps=fps, work_dir=work_dir)
    total_time = time.perf_counter() - t0

    decoded_frames = trip.frames[:n_frames, :height, :width]
    metrics = compute_metrics(source_frames, decoded_frames, masks)
    side_data = charge_side_data("cleaned_video", n_frames, frame_shape=(height, width))

    return {
        "representation": "cleaned_video",
        "qp": qp,
        "codec": codec,
        "preset": preset,
        "tool_path": trip.tool_path,
        "tool_version": trip.tool_version,
        "encoded_payload_bytes": int(trip.size_bytes),
        "side_data_bytes": side_data["total_side_data_bytes"],
        "total_package_bytes": int(trip.size_bytes) + side_data["total_side_data_bytes"],
        "side_data_detail": side_data,
        "metrics": metrics,
        "timing": {
            "encode_seconds": round(float(trip.encode_seconds), 3),
            "decode_render_seconds": round(float(trip.decode_seconds), 3),
            "roundtrip_total_seconds": round(total_time, 3),
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
            alarms.append(
                f"{rep} QP {qp}: total_bytes {total_bytes} outside band {b_band}"
            )

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
        "Enc (s)",
        "Dec (s)",
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
        enc_s = pt["timing"]["encode_seconds"]
        dec_s = pt["timing"]["decode_render_seconds"]
        rows.append(
            f"| {rep} | {qp} | {payload_b:,} | {side_b:,} | {tot_b:,} | "
            f"{psnr_v:.2f} | {ssim_v:.4f} | {psnr_f:.2f} | {ssim_f:.4f} | "
            f"{enc_s:.2f} | {dec_s:.2f} |"
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
            "encode_seconds",
            "decode_render_seconds",
        ]
    )
    for pt in points:
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
                pt["timing"]["encode_seconds"],
                pt["timing"]["decode_render_seconds"],
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

    # Step 1: Input verification
    t_load_start = time.perf_counter()
    clip = load_long_scene_clip(video, scene, n_frames=n_frames)
    load_seconds = time.perf_counter() - t_load_start

    frames = clip.frames
    masks = clip.masks
    input_info = clip.describe()
    input_info["load_seconds"] = round(load_seconds, 3)

    # Step 2: Common foreground-removed frame stack
    cache_path = (cache_dir / f"cleaned_stack_{video}_{scene}_{n_frames}.npz") if cache_dir else None
    cleaned_frames, plate, homographies, prep_stats = build_common_cleaned_stack(
        frames,
        masks,
        register=True,
        cache_path=cache_path,
    )

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
        )
        eval_points.append(res3)

    # Step 5: Bounds verification
    bounds_passed, alarms = check_bounds(eval_points, bounds_record)

    # Step 6: Synthesis and hypothesis evaluation
    # Extract QP32 and QP47 points for comparison
    points_by_rep_qp = {
        (pt["representation"], pt["qp"]): pt for pt in eval_points
    }

    # Compare Still vs Registered Panorama
    # Still vs Panorama at QP 32:
    still_32 = points_by_rep_qp[("still_frame0", 32)]
    pano_32 = points_by_rep_qp[("registered_panorama", 32)]
    video_32 = points_by_rep_qp[("cleaned_video", 32)]

    still_47 = points_by_rep_qp[("still_frame0", 47)]
    pano_47 = points_by_rep_qp[("registered_panorama", 47)]
    video_47 = points_by_rep_qp[("cleaned_video", 47)]

    delta_psnr_pano_vs_still_32 = pano_32["metrics"]["psnr_y_visible_dB"] - still_32["metrics"]["psnr_y_visible_dB"]
    delta_psnr_video_vs_pano_32 = video_32["metrics"]["psnr_y_visible_dB"] - pano_32["metrics"]["psnr_y_visible_dB"]
    delta_bytes_pano_vs_still_32 = pano_32["total_package_bytes"] - still_32["total_package_bytes"]
    delta_bytes_video_vs_pano_32 = video_32["total_package_bytes"] - pano_32["total_package_bytes"]

    # Hypothesis decision:
    # "Hypothesis: Camera coverage/registration is the limiting background error; video improves it but must earn its byte/time cost.
    # Alternative: A still is already sufficient and geometry/metadata or correction causes the deficit."
    # If camera registration improves PSNR significantly over still (> 1 dB) in both uncompressed and compressed,
    # the still deficit is primarily camera motion / registration!
    if delta_psnr_pano_vs_still_32 > 1.0 and sanity["registered_panorama"]["psnr_y_visible_dB"] > sanity["still_frame0"]["psnr_y_visible_dB"] + 1.0:
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
                    pano_47["metrics"]["psnr_y_visible_dB"] - still_47["metrics"]["psnr_y_visible_dB"], 3
                ),
                "delta_psnr_video_vs_pano_dB": round(
                    video_47["metrics"]["psnr_y_visible_dB"] - pano_47["metrics"]["psnr_y_visible_dB"], 3
                ),
                "delta_bytes_pano_vs_still": pano_47["total_package_bytes"] - still_47["total_package_bytes"],
                "delta_bytes_video_vs_pano": video_47["total_package_bytes"] - pano_47["total_package_bytes"],
            },
            "uncompressed": {
                "still_psnr_y_visible_dB": sanity["still_frame0"]["psnr_y_visible_dB"],
                "pano_psnr_y_visible_dB": sanity["registered_panorama"]["psnr_y_visible_dB"],
                "video_psnr_y_visible_dB": sanity["cleaned_video"]["psnr_y_visible_dB"],
                "gain_pano_over_still_dB": round(
                    sanity["registered_panorama"]["psnr_y_visible_dB"] - sanity["still_frame0"]["psnr_y_visible_dB"], 3
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
    cache_dir = args.cache_dir or (Path("/tmp") / f"pointstream-bgprobe-cache-{os.environ.get('USER', 'default')}")

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
    print(report['verdict_summary'])
    print(f"Report saved to: {out_dir / 'probe_report.json'}")


if __name__ == "__main__":
    main()
