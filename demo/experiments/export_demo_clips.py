"""Slim 3-clip demo rebuild: PS ladder + matched AV1 + web H.264 + RTMPose numbers."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3  # noqa: F401
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch

from demo.evaluation.encode_av1_ladder import encode_av1
from demo.evaluation.evaluate_robotics_teleop import score_pose_tracks
from demo.evaluation.pose_backends import BACKENDS
from demo.experiments.run_comparison import POINTSTREAM_TIERS, reconstruct_pointstream_video
from demo.models.dataset import build_curated_samples
from demo.models.unet_generator import HandPix2PixUNet, HandSPADEUNet
from demo.pipeline.background_codec import BackgroundCodec, read_video_frames_robust
from demo.pipeline.hand_keypoints import serialize_poses_to_json
from demo.pipeline.keypoint_compressor import KeypointCompressor
from demo.pitch.export_av1_web import transcode as web_transcode

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

AV1_EXPORTS = {
    "av1_180": ("320:180", 25),
    "av1_240": ("426:240", 40),
    "av1_360": ("640:360", 80),
    "av1_540": ("960:540", 250),
    "av1_720": ("1280:720", 300),
    "av1_1080": (None, 500),
}
PS_KEYS = ["ps_starve", "ps_heavy", "ps_low", "ps_std", "ps_1080"]


def _load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"]
    if ckpt.get("model_type") == "spade" or "enc1.0.weight" in state:
        model = HandSPADEUNet(in_channels=6, out_channels=3).to(device)
    else:
        model = HandPix2PixUNet(in_channels=6, out_channels=3).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def _ffmpeg() -> str:
    return os.environ.get("FFMPEG", "/opt/local/bin/ffmpeg")


def process_clip(clip_path: Path, short: str, work: Path, pitch: Path, model, device, frames: int) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    pitch.mkdir(parents=True, exist_ok=True)
    extractor = BACKENDS["rtm_hand"]
    ref_frames = read_video_frames_robust(clip_path, max_frames=frames)
    h, w = ref_frames[0].shape[:2]
    fps = 30.0
    duration = len(ref_frames) / fps
    ref_mp4 = work / "reference_trimmed.mp4"
    writer = cv2.VideoWriter(str(ref_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in ref_frames:
        writer.write(f)
    writer.release()
    poses = extractor(ref_mp4, frames)
    gt = poses
    packets = [KeypointCompressor.compress_frame(p, w, h) for p in poses]
    kp_bytes = sum(len(p) for p in packets)
    kp_kbps = (kp_bytes * 8) / (duration * 1000.0)
    _, anchors, _ = build_curated_samples(ref_mp4, poses, image_size=256, max_frames=frames, clip_id=0)
    streams = {}
    for (tier_name, scale_res, bg_kbps, preset), key in zip(POINTSTREAM_TIERS, PS_KEYS):
        tag = tier_name.split("(")[0].strip().lower().replace(" ", "_")
        bg_mp4 = work / f"ps_bg_{tag}.mp4"
        if scale_res is not None:
            codec = BackgroundCodec(scale_resolution=scale_res, target_bitrate_kbps=bg_kbps, preset=preset)
        else:
            codec = BackgroundCodec(downscale_factor=1.0, target_bitrate_kbps=bg_kbps, preset=preset)
        _, bg_bytes = codec.prepare_background_video(ref_mp4, poses, bg_mp4, max_frames=frames)
        bg_frames = codec.decode_background_frames(bg_mp4, w, h)
        ps_mp4 = work / f"ps_rec_{tag}.mp4"
        reconstruct_pointstream_video(ref_frames, poses, model, anchors, bg_frames, ps_mp4, device, fps=fps)
        pred = extractor(ps_mp4, frames)
        scored = score_pose_tracks(gt, pred)
        total_kbps = (bg_bytes * 8) / (duration * 1000.0) + kp_kbps
        streams[key] = {
            "kind": "ps",
            "res": {0: "240p", 1: "360p", 2: "540p", 3: "540p", 4: "1080p"}[PS_KEYS.index(key)],
            "kbps": round(total_kbps, 1),
            "mpjpe": round(scored["mpjpe_pixels"], 1),
            "det": round(scored["detection_rate"] * 100, 1),
            "kp": round(kp_kbps, 1),
        }
        web_transcode(ps_mp4, pitch / f"web_{short}_{key}.mp4", _ffmpeg())
    for key, (scale, kbps) in AV1_EXPORTS.items():
        av_mp4 = work / f"{key}.mp4"
        encode_av1(ref_mp4, av_mp4, target_bitrate_kbps=kbps, max_frames=frames, scale=scale, preset=7)
        pred = extractor(av_mp4, frames)
        scored = score_pose_tracks(gt, pred)
        actual = (av_mp4.stat().st_size * 8) / (duration * 1000.0)
        streams[key] = {
            "kind": "av1",
            "res": {"av1_180": "180p", "av1_240": "240p", "av1_360": "360p", "av1_540": "540p", "av1_720": "720p", "av1_1080": "1080p"}[key],
            "kbps": round(actual, 1),
            "mpjpe": round(scored["mpjpe_pixels"], 1),
            "det": round(scored["detection_rate"] * 100, 1),
            "kp": 0,
        }
        web_transcode(av_mp4, pitch / f"web_{short}_{key}.mp4", _ffmpeg())
    web_transcode(ref_mp4, pitch / f"web_{short}_ref.mp4", _ffmpeg())
    serialize_poses_to_json(poses, pitch / f"keypoints_{short}.json")
    streams["ref"] = {"kind": "ref", "res": "1080p", "kbps": 4200, "mpjpe": None, "det": 100, "kp": 0}
    return {"name": short, "clip_path": str(clip_path), "streams": streams}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curated-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("demo/outputs/models/overfit_generator.pt"))
    parser.add_argument("--work", type=Path, default=Path("demo/outputs/results/demo_rebuild"))
    parser.add_argument("--pitch", type=Path, default=Path("demo/outputs/pitch"))
    parser.add_argument("--out", type=Path, default=Path("demo/outputs/results/demo_streams.json"))
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _ = _load_model(args.checkpoint, device)
    manifest = json.loads((args.curated_dir / "manifest.json").read_text())
    report = {"clips": {}}
    shorts = ["clip_01", "clip_02", "clip_03"]
    for item, short in zip(manifest[:3], shorts):
        logger.info("demo rebuild %s", item["filename"])
        report["clips"][short] = process_clip(Path(item["path"]), short, args.work / short, args.pitch, model, device, args.frames)
    args.out.write_text(json.dumps(report, indent=2))
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
