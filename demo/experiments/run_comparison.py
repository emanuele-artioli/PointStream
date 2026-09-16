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
            if crop_h > 0 and crop_w > 0:
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
    target_bg_kbps: int = 250,
) -> dict[str, Any]:
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

    # 4. Background encode & decode (Standard 250k bg, preset 7)
    logger.info(f"[{clip_stem}] Encoding standard background stream (target {target_bg_kbps} kbps, preset 7)...")
    bg_codec = BackgroundCodec(downscale_factor=0.5, target_bitrate_kbps=target_bg_kbps, preset=7)
    bg_encoded_mp4 = clip_out_dir / "pointstream_background.mp4"
    _, bg_bytes = bg_codec.prepare_background_video(
        ref_trimmed_mp4, poses, bg_encoded_mp4, max_frames=n_frames
    )
    bg_frames = bg_codec.decode_background_frames(bg_encoded_mp4, w, h)
    bg_kbps = (bg_bytes * 8) / (duration_sec * 1000.0)

    # 4b. Background encode & decode (Ultra-low 90k bg, preset 7)
    logger.info(f"[{clip_stem}] Encoding ultra-low background stream (target 90 kbps, preset 7)...")
    bg_codec_low = BackgroundCodec(downscale_factor=0.5, target_bitrate_kbps=90, preset=7)
    bg_encoded_low_mp4 = clip_out_dir / "pointstream_bg_90k.mp4"
    _, bg_bytes_low = bg_codec_low.prepare_background_video(
        ref_trimmed_mp4, poses, bg_encoded_low_mp4, max_frames=n_frames
    )
    bg_frames_low = bg_codec_low.decode_background_frames(bg_encoded_low_mp4, w, h)
    bg_low_kbps = (bg_bytes_low * 8) / (duration_sec * 1000.0)

    # 4c. Periodic Infilled Keyframe Plate (every 60 frames = 2 seconds)
    logger.info(f"[{clip_stem}] Encoding periodic background keyframe plates (every 2s)...")
    plate_codec = BackgroundPlateCodec(plate_interval_frames=60, quality=75)
    plate_dir = clip_out_dir / "pointstream_plates"
    plate_paths, plate_bytes = plate_codec.prepare_background_plates(
        ref_trimmed_mp4, poses, plate_dir, max_frames=n_frames
    )
    bg_frames_plate = plate_codec.decode_background_frames(plate_paths, len(ref_frames), w, h)
    plate_kbps = (plate_bytes * 8) / (duration_sec * 1000.0)

    # 5. Extract appearance anchors for this clip
    _, clip_anchors = build_curated_samples(ref_trimmed_mp4, poses, image_size=256, max_frames=n_frames, clip_id=clip_id)
    anchor_bytes = sum(cv2.imencode(".jpg", anchor)[1].nbytes for anchor in clip_anchors.values())
    anchor_kbps = (anchor_bytes * 8) / (duration_sec * 1000.0)

    # 6. Reconstruct PointStream Video (Standard)
    ps_rec_mp4 = clip_out_dir / "pointstream_reconstructed.mp4"
    logger.info(f"[{clip_stem}] Synthesizing PointStream (Standard) frames from keypoints...")
    _, _ = reconstruct_pointstream_video(
        ref_frames, poses, model, clip_anchors, bg_frames, ps_rec_mp4, device, fps=fps, image_size=256
    )
    total_ps_bytes = bg_bytes + keypoint_bytes + anchor_bytes
    total_ps_kbps = (total_ps_bytes * 8) / (duration_sec * 1000.0)
    ps_rec_frames = read_video_frames_robust(ps_rec_mp4, max_frames=n_frames)
    ps_quality = evaluator.evaluate_frames(ref_frames, ps_rec_frames, poses=poses)
    ps_teleop = evaluate_teleop_utility(poses, ps_rec_mp4, max_frames=n_frames)

    pointstream_record = {
        "name": "PointStream (Standard, 250k bg)",
        "total_bytes": total_ps_bytes,
        "bitrate_kbps": round(total_ps_kbps, 1),
        "background_kbps": round(bg_kbps, 1),
        "keypoint_kbps": round(keypoint_kbps, 1),
        "anchor_kbps": round(anchor_kbps, 1),
        "metrics": ps_quality,
        "teleop_utility": ps_teleop,
        "video_path": str(ps_rec_mp4),
    }

    # 6b. Reconstruct PointStream Video (Ultra-Low Rate)
    ps_rec_low_mp4 = clip_out_dir / "pointstream_reconstructed_low.mp4"
    logger.info(f"[{clip_stem}] Synthesizing PointStream (Ultra-Low) frames from keypoints...")
    _, _ = reconstruct_pointstream_video(
        ref_frames, poses, model, clip_anchors, bg_frames_low, ps_rec_low_mp4, device, fps=fps, image_size=256
    )
    total_ps_low_bytes = bg_bytes_low + keypoint_bytes + anchor_bytes
    total_ps_low_kbps = (total_ps_low_bytes * 8) / (duration_sec * 1000.0)
    ps_rec_low_frames = read_video_frames_robust(ps_rec_low_mp4, max_frames=n_frames)
    ps_quality_low = evaluator.evaluate_frames(ref_frames, ps_rec_low_frames, poses=poses)
    ps_teleop_low = evaluate_teleop_utility(poses, ps_rec_low_mp4, max_frames=n_frames)

    pointstream_low_record = {
        "name": "PointStream (Ultra-Low, 90k bg)",
        "total_bytes": total_ps_low_bytes,
        "bitrate_kbps": round(total_ps_low_kbps, 1),
        "background_kbps": round(bg_low_kbps, 1),
        "keypoint_kbps": round(keypoint_kbps, 1),
        "anchor_kbps": round(anchor_kbps, 1),
        "metrics": ps_quality_low,
        "teleop_utility": ps_teleop_low,
        "video_path": str(ps_rec_low_mp4),
    }

    # 6c. Reconstruct PointStream Video (Periodic Plate)
    ps_rec_plate_mp4 = clip_out_dir / "pointstream_reconstructed_plate.mp4"
    logger.info(f"[{clip_stem}] Synthesizing PointStream (Plate 2s) frames from keypoints...")
    _, _ = reconstruct_pointstream_video(
        ref_frames, poses, model, clip_anchors, bg_frames_plate, ps_rec_plate_mp4, device, fps=fps, image_size=256
    )
    total_ps_plate_bytes = plate_bytes + keypoint_bytes + anchor_bytes
    total_ps_plate_kbps = (total_ps_plate_bytes * 8) / (duration_sec * 1000.0)
    ps_rec_plate_frames = read_video_frames_robust(ps_rec_plate_mp4, max_frames=n_frames)
    ps_quality_plate = evaluator.evaluate_frames(ref_frames, ps_rec_plate_frames, poses=poses)
    ps_teleop_plate = evaluate_teleop_utility(poses, ps_rec_plate_mp4, max_frames=n_frames)

    pointstream_plate_record = {
        "name": "PointStream (Plate 2s)",
        "total_bytes": total_ps_plate_bytes,
        "bitrate_kbps": round(total_ps_plate_kbps, 1),
        "background_kbps": round(plate_kbps, 1),
        "keypoint_kbps": round(keypoint_kbps, 1),
        "anchor_kbps": round(anchor_kbps, 1),
        "metrics": ps_quality_plate,
        "teleop_utility": ps_teleop_plate,
        "video_path": str(ps_rec_plate_mp4),
    }

    # 8. Encode Fair AV1 Ladder (spanning 540p, 720p, and 1080p tiers)
    logger.info(f"[{clip_stem}] Running fair AV1 comparison ladder...")
    ladder_dir = clip_out_dir / "av1_ladder"
    av1_results = encode_ladder(ref_trimmed_mp4, ladder_dir, max_frames=n_frames)

    # Evaluate AV1 streams
    evaluated_av1 = []
    for item in av1_results:
        av1_vid = Path(item["output_path"])
        av1_frames = read_video_frames_robust(av1_vid, max_frames=n_frames)

        av1_qual = evaluator.evaluate_frames(ref_frames, av1_frames, poses=poses)
        av1_tel = evaluate_teleop_utility(poses, av1_vid, max_frames=n_frames)
        evaluated_av1.append(
            {
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
            }
        )

    ref_total_hands = sum(len(p.hands) for p in poses)
    ref_frames_with_hands = sum(1 for p in poses if len(p.hands) > 0)
    ref_detection_ceiling = float(ref_frames_with_hands / max(1, len(ref_frames)))

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
        "pointstream": pointstream_record,
        "pointstream_variants": [pointstream_record, pointstream_low_record, pointstream_plate_record],
        "av1_arms": evaluated_av1,
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

    all_clip_results = []
    for idx, item in enumerate(manifest[:3]):
        clip_path = Path(item["path"])
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
        )
        all_clip_results.append(res)

    # Measure latency profile
    logger.info("Profiling per-frame latency...")
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    latency_profile = profile_pipeline_latency(model, device, dummy_frame)

    final_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "latency_profile": latency_profile,
        "clips": all_clip_results,
    }

    out_json = args.output_dir / "comparison_results.json"
    with open(out_json, "w") as f:
        json.dump(final_payload, f, indent=2)
    logger.info(f"Full benchmark results written to {out_json}")


if __name__ == "__main__":
    main()

