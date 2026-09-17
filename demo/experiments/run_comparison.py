"""End-to-end benchmark comparison: PointStream vs AV1 on Egocentric-10K."""

from __future__ import annotations

import sqlite3  # noqa: F401
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch

from demo.evaluation.encode_av1_ladder import encode_ladder
from demo.evaluation.evaluate_quality import QualityEvaluator
from demo.evaluation.evaluate_robotics_teleop import evaluate_teleop_utility
from demo.evaluation.latency_profiler import profile_pipeline_latency
from demo.models.dataset import build_curated_samples, to_torch_tensor
from demo.models.unet_generator import HandPix2PixUNet, HandSPADEUNet
from demo.pipeline.background_codec import (
    BackgroundCodec,
    BackgroundPlateCodec,
    read_video_frames_robust,
)
from demo.pipeline.foreground_segmenter import create_box_feather_mask, letterbox_crop, unletterbox_crop
from demo.pipeline.hand_keypoints import (
    FrameHandPose,
    extract_video_hand_poses,
    render_skeleton_on_canvas,
)
from demo.pipeline.keypoint_compressor import KeypointCompressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CURATED_DIR = Path("/home/itec/emanuele/Datasets/Egocentric-10K/curated")
DEFAULT_OUTPUT_DIR = Path("demo/outputs/results")
DEFAULT_CHECKPOINT = Path("demo/outputs/models/overfit_generator.pt")

# PointStream multi-tier background ladder.
# Each tier: (label, scale_resolution or None for native 1080p, target_bg_kbps, SVT-AV1 preset)
POINTSTREAM_TIERS = [
    ("PS Extreme Starve (240p bg, 30k)", (426, 240), 30, 7),
    ("PS Heavy Starve (360p bg, 70k)", (640, 360), 70, 7),
    ("PS Low Teleop (540p bg, 140k)", (960, 540), 140, 7),
    ("PS Standard (540p bg, 250k)", (960, 540), 250, 7),
    ("PS Standard 1080p (native bg, 300k)", None, 300, 10),
]


def reconstruct_pointstream_video(
    ref_frames: list[np.ndarray],
    poses: list[FrameHandPose],
    model: torch.nn.Module,
    anchors: dict[str, np.ndarray],
    bg_frames: list[np.ndarray],
    output_mp4: Path,
    device: torch.device,
    fps: float = 30.0,
    image_size: int = 256,
) -> tuple[Path, int]:
    h, w = ref_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_mp4), fourcc, fps, (w, h))

    model.eval()
    n_frames = min(len(ref_frames), len(bg_frames), len(poses))

    for idx in range(n_frames):
        bg = bg_frames[idx].copy()
        frame_pose = poses[idx]

        for hand in frame_pose.hands:
            side = hand.handedness
            if side not in anchors:
                continue

            app_anchor = anchors[side]
            single_hand_pose = FrameHandPose(frame_idx=idx, hands=[hand])
            skel_crop = render_skeleton_on_canvas(
                single_hand_pose,
                width=image_size,
                height=image_size,
                crop_bbox=hand.bbox,
            )

            app_t = to_torch_tensor(app_anchor).unsqueeze(0).to(device)
            skel_t = to_torch_tensor(skel_crop).unsqueeze(0).to(device)
            inp = torch.cat([app_t, skel_t], dim=1)

            with torch.no_grad():
                pred = model(inp)

            # Convert prediction [1, 3, H, W] in [-1, 1] to uint8 BGR
            pred_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
            pred_uint8 = np.clip((pred_np + 1.0) * 127.5, 0, 255).astype(np.uint8)
            pred_bgr = cv2.cvtColor(pred_uint8, cv2.COLOR_RGB2BGR)

            # Composite back into the decoded background
            _, meta = letterbox_crop(ref_frames[idx], hand.bbox, target_size=image_size)
            restored_crop, (x1, y1, x2, y2) = unletterbox_crop(pred_bgr, meta, w, h)

            # Alpha blend using box-margin feathering (preserves wrist context, removes square borders)
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(x1 + 1, min(w, x2))
            y2 = max(y1 + 1, min(h, y2))
            crop_h, crop_w = y2 - y1, x2 - x1
            if crop_h >= 8 and crop_w >= 8:
                fitted_hand = cv2.resize(restored_crop, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
                alpha = create_box_feather_mask(crop_h, crop_w, margin_fraction=0.08)

                bg_roi = bg[y1:y2, x1:x2].astype(np.float32)
                hand_roi = fitted_hand.astype(np.float32)
                blended = hand_roi * alpha + bg_roi * (1.0 - alpha)
                bg[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

        writer.write(bg)

    writer.release()
    file_size = output_mp4.stat().st_size
    return output_mp4, file_size


def run_single_clip_comparison(
    clip_path: Path,
    clip_id: int,
    output_dir: Path,
    model: HandPix2PixUNet,
    saved_anchors: dict[str, Any],
    evaluator: QualityEvaluator,
    device: torch.device,
    n_frames: int = 300,
    total_session_sec: float = 30.0,
    shared_anchor_bytes: int = 0,
) -> dict[str, Any]:
    """Benchmark a single clip across the full PointStream and AV1 ladders.

    Args:
        total_session_sec: Total duration across all clips (for shared anchor amortization).
        shared_anchor_bytes: Total WebP anchor bytes shared across all clips (same worker).
    """
    clip_stem = clip_path.stem
    clip_out_dir = output_dir / clip_stem
    clip_out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load reference frames
    cap = cv2.VideoCapture(str(clip_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    ref_frames: list[np.ndarray] = []
    frame_idx = 0
    while cap.isOpened() and frame_idx < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        ref_frames.append(frame)
        frame_idx += 1
    cap.release()

    h, w = ref_frames[0].shape[:2]
    duration_sec = len(ref_frames) / fps

    # Save clean reference trimmed mp4
    ref_trimmed_mp4 = clip_out_dir / "reference_trimmed.mp4"
    writer = cv2.VideoWriter(str(ref_trimmed_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in ref_frames:
        writer.write(f)
    writer.release()

    # 2. Extract hand poses
    logger.info(f"[{clip_stem}] Extracting reference hand poses...")
    poses = extract_video_hand_poses(ref_trimmed_mp4, max_frames=n_frames)

    # 3. Compress keypoints
    keypoint_packets = [KeypointCompressor.compress_frame(p, w, h) for p in poses]
    keypoint_bytes = sum(len(pkt) for pkt in keypoint_packets)
    keypoint_kbps = (keypoint_bytes * 8) / (duration_sec * 1000.0)

    # 4. Extract appearance anchors (WebP-compressed, shared across clips)
    _, clip_anchors, clip_anchor_bytes = build_curated_samples(
        ref_trimmed_mp4, poses, image_size=256, max_frames=n_frames, clip_id=clip_id,
    )
    per_clip_anchor_bytes = sum(clip_anchor_bytes.values())

    # Anchor cost is amortized across the full session (shared worker identity).
    anchor_kbps_amortized = (shared_anchor_bytes * 8) / (total_session_sec * 1000.0)

    # ====================================================================
    # 5. PointStream Multi-Tier Ladder
    # ====================================================================
    pointstream_variants: list[dict[str, Any]] = []

    for tier_name, scale_res, bg_target_kbps, preset in POINTSTREAM_TIERS:
        tag = tier_name.split("(")[0].strip().lower().replace(" ", "_")
        bg_mp4 = clip_out_dir / f"ps_bg_{tag}.mp4"

        logger.info(f"[{clip_stem}] Encoding PointStream background: {tier_name}...")

        if scale_res is not None:
            bg_codec = BackgroundCodec(
                scale_resolution=scale_res,
                target_bitrate_kbps=bg_target_kbps,
                preset=preset,
            )
        else:
            # Native 1080p — no downscaling
            bg_codec = BackgroundCodec(
                downscale_factor=1.0,
                target_bitrate_kbps=bg_target_kbps,
                preset=preset,
            )

        _, bg_bytes = bg_codec.prepare_background_video(
            ref_trimmed_mp4, poses, bg_mp4, max_frames=n_frames,
        )
        bg_frames = bg_codec.decode_background_frames(bg_mp4, w, h)
        bg_kbps = (bg_bytes * 8) / (duration_sec * 1000.0)

        # Verify all background frames are at native resolution for compositing
        assert bg_frames[0].shape[:2] == (h, w), (
            f"Background frame shape {bg_frames[0].shape[:2]} != native ({h}, {w})"
        )

        # Reconstruct
        ps_rec_mp4 = clip_out_dir / f"ps_rec_{tag}.mp4"
        logger.info(f"[{clip_stem}] Synthesizing {tier_name} frames...")
        _, _ = reconstruct_pointstream_video(
            ref_frames, poses, model, clip_anchors, bg_frames,
            ps_rec_mp4, device, fps=fps, image_size=256,
        )

        total_bytes = bg_bytes + keypoint_bytes + shared_anchor_bytes
        total_kbps = bg_kbps + keypoint_kbps + anchor_kbps_amortized

        ps_rec_frames = read_video_frames_robust(ps_rec_mp4, max_frames=n_frames)
        ps_quality = evaluator.evaluate_frames(ref_frames, ps_rec_frames, poses=poses)
        ps_teleop = evaluate_teleop_utility(poses, ps_rec_mp4, max_frames=n_frames)

        record: dict[str, Any] = {
            "name": tier_name,
            "total_bytes": total_bytes,
            "bitrate_kbps": round(total_kbps, 1),
            "background_kbps": round(bg_kbps, 1),
            "keypoint_kbps": round(keypoint_kbps, 1),
            "anchor_kbps": round(anchor_kbps_amortized, 1),
            "metrics": ps_quality,
            "teleop_utility": ps_teleop,
            "video_path": str(ps_rec_mp4),
        }
        pointstream_variants.append(record)

    # 6. Periodic Infilled Keyframe Plate (every 60 frames = 2 seconds)
    logger.info(f"[{clip_stem}] Encoding periodic background keyframe plates (every 2s)...")
    plate_codec = BackgroundPlateCodec(plate_interval_frames=60, quality=75)
    plate_dir = clip_out_dir / "pointstream_plates"
    plate_paths, plate_bytes = plate_codec.prepare_background_plates(
        ref_trimmed_mp4, poses, plate_dir, max_frames=n_frames,
    )
    bg_frames_plate = plate_codec.decode_background_frames(plate_paths, len(ref_frames), w, h)
    plate_kbps = (plate_bytes * 8) / (duration_sec * 1000.0)

    ps_rec_plate_mp4 = clip_out_dir / "pointstream_reconstructed_plate.mp4"
    logger.info(f"[{clip_stem}] Synthesizing PointStream (Plate 2s) frames from keypoints...")
    _, _ = reconstruct_pointstream_video(
        ref_frames, poses, model, clip_anchors, bg_frames_plate,
        ps_rec_plate_mp4, device, fps=fps, image_size=256,
    )
    total_ps_plate_bytes = plate_bytes + keypoint_bytes + shared_anchor_bytes
    total_ps_plate_kbps = plate_kbps + keypoint_kbps + anchor_kbps_amortized
    ps_rec_plate_frames = read_video_frames_robust(ps_rec_plate_mp4, max_frames=n_frames)
    ps_quality_plate = evaluator.evaluate_frames(ref_frames, ps_rec_plate_frames, poses=poses)
    ps_teleop_plate = evaluate_teleop_utility(poses, ps_rec_plate_mp4, max_frames=n_frames)

    pointstream_variants.append({
        "name": "PointStream (Plate 2s)",
        "total_bytes": total_ps_plate_bytes,
        "bitrate_kbps": round(total_ps_plate_kbps, 1),
        "background_kbps": round(plate_kbps, 1),
        "keypoint_kbps": round(keypoint_kbps, 1),
        "anchor_kbps": round(anchor_kbps_amortized, 1),
        "metrics": ps_quality_plate,
        "teleop_utility": ps_teleop_plate,
        "video_path": str(ps_rec_plate_mp4),
    })

    # 7. Encode Fair AV1 Ladder (spanning 180p–1080p)
    logger.info(f"[{clip_stem}] Running fair AV1 comparison ladder...")
    ladder_dir = clip_out_dir / "av1_ladder"
    av1_results = encode_ladder(ref_trimmed_mp4, ladder_dir, max_frames=n_frames)

    # Evaluate AV1 streams
    evaluated_av1: list[dict[str, Any]] = []
    for item in av1_results:
        av1_vid = Path(item["output_path"])
        av1_frames = read_video_frames_robust(av1_vid, max_frames=n_frames)

        av1_qual = evaluator.evaluate_frames(ref_frames, av1_frames, poses=poses)
        av1_tel = evaluate_teleop_utility(poses, av1_vid, max_frames=n_frames)
        evaluated_av1.append({
            "name": item["name"],
            "target_kbps": item["bitrate_target_kbps"],
            "actual_kbps": round(item["actual_bitrate_kbps"], 1),
            "scale": item.get("scale"),
            "preset": item.get("preset"),
            "ms_per_frame": item.get("ms_per_frame"),
            "encode_fps": item.get("encode_fps"),
            "size_bytes": item["size_bytes"],
            "deblocked": item["deblocked"],
            "metrics": av1_qual,
            "teleop_utility": av1_tel,
            "video_path": str(av1_vid),
        })

    ref_total_hands = sum(len(p.hands) for p in poses)
    ref_frames_with_hands = sum(1 for p in poses if len(p.hands) > 0)
    ref_detection_ceiling = float(ref_frames_with_hands / max(1, len(ref_frames)))

    # The "primary" pointstream record is the Standard 540p variant (backward compat)
    primary_ps = pointstream_variants[3] if len(pointstream_variants) > 3 else pointstream_variants[0]

    return {
        "clip_id": clip_id,
        "clip_name": clip_stem,
        "duration_sec": duration_sec,
        "frames": len(ref_frames),
        "resolution": f"{w}x{h}",
        "reference_oracle": {
            "total_hands": ref_total_hands,
            "frames_with_hands": ref_frames_with_hands,
            "total_frames": len(ref_frames),
            "detection_ceiling": round(ref_detection_ceiling, 3),
        },
        "pointstream": primary_ps,
        "pointstream_variants": pointstream_variants,
        "av1_arms": evaluated_av1,
        "per_clip_anchor_bytes": per_clip_anchor_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PointStream vs AV1 Egocentric Video Benchmark")
    parser.add_argument("--curated-dir", type=Path, default=DEFAULT_CURATED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--frames", type=int, default=300, help="Frames per clip to benchmark")
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.curated_dir / "manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model_state_dict"]
    model_type = ckpt.get("model_type", "spade" if "enc1.0.weight" in state_dict else "unet")

    if model_type == "spade" or "enc1.0.weight" in state_dict:
        model = HandSPADEUNet(in_channels=6, out_channels=3).to(device)
        logger.info("Instantiated HandSPADEUNet generator.")
    else:
        model = HandPix2PixUNet(in_channels=6, out_channels=3).to(device)
        logger.info("Instantiated HandPix2PixUNet generator.")

    model.load_state_dict(state_dict)
    model.eval()

    evaluator = QualityEvaluator(device_str=args.device)

    # --- Two-pass approach for shared anchor amortization ---
    # Pass 1: Extract anchor bytes from each clip (lightweight — just pose extraction + anchor selection).
    # Pass 2: Run the full benchmark with the total shared anchor bytes.
    clip_paths = [Path(item["path"]) for item in manifest[:3]]
    clip_durations: list[float] = []
    per_clip_anchor_bytes_list: list[int] = []

    logger.info("Pass 1: Pre-extracting shared worker anchor bytes across all clips...")
    for idx, clip_path in enumerate(clip_paths):
        cap = cv2.VideoCapture(str(clip_path))
        fps_val = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        n_total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), args.frames)
        cap.release()
        clip_durations.append(n_total / fps_val)

        poses = extract_video_hand_poses(clip_path, max_frames=args.frames)
        _, _, anchor_bytes_dict = build_curated_samples(
            clip_path, poses, image_size=256, max_frames=args.frames, clip_id=idx,
        )
        per_clip_anchor_bytes_list.append(sum(anchor_bytes_dict.values()))

    # Same worker across all 3 clips: take the max anchor set (superset of sides).
    # In practice all clips have both hands, so this is the same as any single clip.
    shared_anchor_bytes = max(per_clip_anchor_bytes_list) if per_clip_anchor_bytes_list else 0
    total_session_sec = sum(clip_durations) if clip_durations else 30.0
    anchor_kbps = (shared_anchor_bytes * 8) / (total_session_sec * 1000.0)
    logger.info(
        f"Shared WebP anchor: {shared_anchor_bytes} bytes, "
        f"amortized over {total_session_sec:.1f}s = {anchor_kbps:.2f} kbps"
    )

    # Pass 2: Full benchmark
    all_clip_results = []
    for idx, clip_path in enumerate(clip_paths):
        logger.info("\n==========================================")
        logger.info(f"BENCHMARKING CLIP {idx + 1}/3: {clip_path.name}")
        logger.info("==========================================")
        res = run_single_clip_comparison(
            clip_path=clip_path,
            clip_id=idx,
            output_dir=args.output_dir,
            model=model,
            saved_anchors=ckpt.get("anchors", {}),
            evaluator=evaluator,
            device=device,
            n_frames=args.frames,
            total_session_sec=total_session_sec,
            shared_anchor_bytes=shared_anchor_bytes,
        )
        all_clip_results.append(res)

    # Measure latency profile
    logger.info("Profiling per-frame latency...")
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    latency_profile = profile_pipeline_latency(model, device, dummy_frame)

    final_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "shared_anchor_bytes": shared_anchor_bytes,
        "shared_anchor_kbps": round(anchor_kbps, 2),
        "total_session_sec": round(total_session_sec, 1),
        "latency_profile": latency_profile,
        "clips": all_clip_results,
    }

    out_json = args.output_dir / "comparison_results.json"
    with open(out_json, "w") as f:
        json.dump(final_payload, f, indent=2)
    logger.info(f"Full benchmark results written to {out_json}")


if __name__ == "__main__":
    main()
