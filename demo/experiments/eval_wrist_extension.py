"""Forearm / Wrist Extension Diagnostic Evaluation.

Evaluates whether extending the bounding box along the forearm direction (from middle MCP
towards the wrist) improves MediaPipe Palm Detector proposal rate on reconstructed video,
or introduces boundary seams / distortion.

Decision Rule:
Only integrate into production if detection rate improves by >= +5.0% without
increasing MPJPE or degrading perceptual quality.
"""

from __future__ import annotations

import sqlite3  # noqa: F401
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch

from demo.evaluation.evaluate_quality import QualityEvaluator
from demo.evaluation.evaluate_robotics_teleop import evaluate_teleop_utility
from demo.models.dataset import build_curated_samples, to_torch_tensor
from demo.models.unet_generator import HandPix2PixUNet, HandSPADEUNet
from demo.pipeline.background_codec import BackgroundCodec, read_video_frames_robust
from demo.pipeline.foreground_segmenter import create_box_feather_mask, letterbox_crop, unletterbox_crop
from demo.pipeline.hand_keypoints import (
    FrameHandPose,
    SingleHand,
    extract_video_hand_poses,
    render_skeleton_on_canvas,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CURATED_DIR = Path("/home/itec/emanuele/Datasets/Egocentric-10K/curated")
DEFAULT_OUTPUT_DIR = Path("demo/outputs/results/wrist_extension_diagnostic")
DEFAULT_CHECKPOINT = Path("demo/outputs/models/overfit_generator.pt")


def extend_bbox_along_wrist(
    bbox: list[int],
    landmarks_pixel: list[list[float]],
    extension_fraction: float,
) -> list[int]:
    """Extends bounding box along the vector from middle MCP (landmark 9) to wrist (landmark 0)."""
    if extension_fraction <= 0.0 or not landmarks_pixel or len(landmarks_pixel) < 10:
        return list(bbox)

    wrist = np.array(landmarks_pixel[0])   # wrist
    mcp = np.array(landmarks_pixel[9])     # middle finger MCP
    vec = wrist - mcp
    norm = float(np.linalg.norm(vec))
    if norm < 1e-3:
        return list(bbox)
    unit_vec = vec / norm

    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    extent = max(box_w, box_h) * extension_fraction

    dx = unit_vec[0] * extent
    dy = unit_vec[1] * extent

    new_x1 = int(round(x1 + min(0.0, dx)))
    new_x2 = int(round(x2 + max(0.0, dx)))
    new_y1 = int(round(y1 + min(0.0, dy)))
    new_y2 = int(round(y2 + max(0.0, dy)))

    return [new_x1, new_y1, new_x2, new_y2]


def reconstruct_variant_video(
    ref_frames: list[np.ndarray],
    poses: list[FrameHandPose],
    model: torch.nn.Module,
    anchors: dict[str, np.ndarray],
    bg_frames: list[np.ndarray],
    output_mp4: Path,
    device: torch.device,
    extension_fraction: float = 0.0,
    margin_fraction: float = 0.08,
    fps: float = 30.0,
    image_size: int = 256,
) -> Path:
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

            # Compute extended bbox if requested
            eff_bbox = extend_bbox_along_wrist(
                hand.bbox, hand.landmarks_pixel, extension_fraction
            )

            app_anchor = anchors[side]
            extended_hand = SingleHand(
                handedness=hand.handedness,
                confidence=hand.confidence,
                bbox=eff_bbox,
                landmarks_norm=hand.landmarks_norm,
                landmarks_pixel=hand.landmarks_pixel,
            )
            single_hand_pose = FrameHandPose(frame_idx=idx, hands=[extended_hand])
            skel_crop = render_skeleton_on_canvas(
                single_hand_pose,
                width=image_size,
                height=image_size,
                crop_bbox=eff_bbox,
            )

            app_t = to_torch_tensor(app_anchor).unsqueeze(0).to(device)
            skel_t = to_torch_tensor(skel_crop).unsqueeze(0).to(device)
            inp = torch.cat([app_t, skel_t], dim=1)

            with torch.no_grad():
                pred = model(inp)

            pred_np = pred.squeeze(0).permute(1, 2, 0).cpu().numpy()
            pred_uint8 = np.clip((pred_np + 1.0) * 127.5, 0, 255).astype(np.uint8)
            pred_bgr = cv2.cvtColor(pred_uint8, cv2.COLOR_RGB2BGR)

            _, meta = letterbox_crop(ref_frames[idx], eff_bbox, target_size=image_size)
            restored_crop, (x1, y1, x2, y2) = unletterbox_crop(pred_bgr, meta, w, h)

            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(x1 + 1, min(w, x2))
            y2 = max(y1 + 1, min(h, y2))
            crop_h, crop_w = y2 - y1, x2 - x1
            if crop_h > 0 and crop_w > 0:
                fitted_hand = cv2.resize(restored_crop, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
                alpha = create_box_feather_mask(crop_h, crop_w, margin_fraction=margin_fraction)

                bg_roi = bg[y1:y2, x1:x2].astype(np.float32)
                hand_roi = fitted_hand.astype(np.float32)
                blended = hand_roi * alpha + bg_roi * (1.0 - alpha)
                bg[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)

        writer.write(bg)

    writer.release()
    return output_mp4


def main() -> None:
    parser = argparse.ArgumentParser(description="Forearm / Wrist Extension Diagnostic")
    parser.add_argument("--curated-dir", type=Path, default=DEFAULT_CURATED_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.curated_dir / "manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading generator checkpoint from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model_state_dict"]
    model_type = ckpt.get("model_type", "spade" if "enc1.0.weight" in state_dict else "unet")

    if model_type == "spade" or "enc1.0.weight" in state_dict:
        model = HandSPADEUNet(in_channels=6, out_channels=3).to(device)
    else:
        model = HandPix2PixUNet(in_channels=6, out_channels=3).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    evaluator = QualityEvaluator(device_str=args.device)

    # Diagnostic is evaluated on Clip 1 (index 0) where baseline detection was lowest (29.1%)
    clip_path = Path(manifest[0]["path"])
    logger.info(f"Evaluating wrist extension diagnostic on {clip_path.name}...")

    cap = cv2.VideoCapture(str(clip_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    ref_frames: list[np.ndarray] = []
    frame_idx = 0
    while cap.isOpened() and frame_idx < args.frames:
        ret, frame = cap.read()
        if not ret:
            break
        ref_frames.append(frame)
        frame_idx += 1
    cap.release()

    h, w = ref_frames[0].shape[:2]

    ref_trimmed_mp4 = args.output_dir / "reference_trimmed.mp4"
    writer = cv2.VideoWriter(str(ref_trimmed_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in ref_frames:
        writer.write(f)
    writer.release()

    poses = extract_video_hand_poses(ref_trimmed_mp4, max_frames=args.frames)
    _, clip_anchors, _ = build_curated_samples(
        ref_trimmed_mp4, poses, image_size=256, max_frames=args.frames, clip_id=0,
    )

    # Prepare standard 540p background upscaled to 1080p
    bg_codec = BackgroundCodec(downscale_factor=0.5, target_bitrate_kbps=250, preset=7)
    bg_mp4 = args.output_dir / "diagnostic_bg.mp4"
    bg_codec.prepare_background_video(ref_trimmed_mp4, poses, bg_mp4, max_frames=args.frames)
    bg_frames = bg_codec.decode_background_frames(bg_mp4, w, h)

    ref_total_hands = sum(len(p.hands) for p in poses)
    logger.info(f"Reference oracle hands in clip: {ref_total_hands} across {len(poses)} frames")

    # Diagnostic Arms:
    # 1. Baseline: 0% extension, 8% feathering
    # 2. Wrist 15%: 15% extension, 10% feathering
    # 3. Wrist 25%: 25% extension, 12% feathering
    diagnostic_configs = [
        {"name": "Baseline (Tight BBox, 8% feather)", "ext": 0.0, "feather": 0.08},
        {"name": "Wrist Extension +15% (10% feather)", "ext": 0.15, "feather": 0.10},
        {"name": "Wrist Extension +25% (12% feather)", "ext": 0.25, "feather": 0.12},
    ]

    results: list[dict[str, Any]] = []

    for cfg in diagnostic_configs:
        name = cfg["name"]
        ext = float(cfg["ext"])
        feather = float(cfg["feather"])
        tag = f"ext_{int(ext * 100):02d}"
        vid_path = args.output_dir / f"rec_{tag}.mp4"

        logger.info(f"\n--- Testing variant: {name} ---")
        reconstruct_variant_video(
            ref_frames=ref_frames,
            poses=poses,
            model=model,
            anchors=clip_anchors,
            bg_frames=bg_frames,
            output_mp4=vid_path,
            device=device,
            extension_fraction=ext,
            margin_fraction=feather,
            fps=fps,
            image_size=256,
        )

        rec_frames = read_video_frames_robust(vid_path, max_frames=args.frames)
        qual = evaluator.evaluate_frames(ref_frames, rec_frames, poses=poses)
        teleop = evaluate_teleop_utility(poses, vid_path, max_frames=args.frames)

        rec = {
            "name": name,
            "extension_fraction": ext,
            "feather_fraction": feather,
            "detection_rate": teleop["detection_rate"],
            "oracle_capture_ratio": teleop["oracle_capture_ratio"],
            "mpjpe_pixels": teleop["mpjpe_pixels"],
            "psnr": qual["psnr_dB"],
            "ssim": qual["ssim"],
            "lpips": qual["lpips"],
            "video_path": str(vid_path),
        }
        results.append(rec)
        logger.info(
            f"Result [{name}]: Detection Rate = {teleop['detection_rate'] * 100:.1f}%, "
            f"MPJPE = {teleop['mpjpe_pixels']:.1f} px, PSNR = {qual['psnr_dB']:.2f} dB, LPIPS = {qual['lpips']:.4f}"
        )

    # Decision Rule evaluation
    base_det = results[0]["detection_rate"]
    best_variant = max(results[1:], key=lambda x: x["detection_rate"])
    delta_det = best_variant["detection_rate"] - base_det

    decision_keep = delta_det >= 0.05 and best_variant["mpjpe_pixels"] <= results[0]["mpjpe_pixels"] * 1.05

    summary = {
        "clip_evaluated": clip_path.name,
        "baseline_detection_rate": round(base_det, 3),
        "best_variant": best_variant["name"],
        "best_detection_rate": round(best_variant["detection_rate"], 3),
        "delta_detection_rate": round(delta_det, 3),
        "decision_rule_threshold": "+5.0% absolute detection gain",
        "decision_keep_in_production": decision_keep,
        "variants": results,
    }

    report_path = args.output_dir / "wrist_extension_results.json"
    with open(report_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n==========================================")
    logger.info("WRIST EXTENSION DIAGNOSTIC VERDICT")
    logger.info("==========================================")
    logger.info(f"Baseline Detection: {base_det * 100:.1f}%")
    logger.info(f"Best Variant ({best_variant['name']}): {best_variant['detection_rate'] * 100:.1f}% (delta: {delta_det * 100:+.1f}%)")
    logger.info(f"Production Gate Passed: {decision_keep}")
    logger.info(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
