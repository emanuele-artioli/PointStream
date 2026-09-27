"""Slim clip-1 demo rebuild: PS CRF ladder + AV1 CRF 63 + RTMPose numbers.

AV1 files are served as AV1 (no H.264 CRF 23 wrapper). PS composites stay H.264
for broad canvas playback.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sqlite3  # noqa: F401
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import torch

from demo.evaluation.evaluate_robotics_teleop import score_pose_tracks
from demo.evaluation.pose_backends import BACKENDS
from demo.experiments.run_comparison import POINTSTREAM_TIERS, reconstruct_pointstream_video
from demo.models.dataset import build_curated_samples
from demo.models.hand_objective import composite_hand_metrics, selection_min
from demo.models.matte import dwb2_roundtrip, interpolate_hand_alphas, read_hand_alphas
from demo.models.unet_generator import HandPix2PixUNet, HandSPADEUNet
from demo.pipeline.background_codec import BackgroundCodec, read_video_frames_robust
from demo.pipeline.hand_keypoints import serialize_poses_to_json
from demo.pipeline.keypoint_compressor import KeypointCompressor
from demo.pipeline.maps.av1_crf import AV1_LADDER, encode_av1_crf
from demo.pitch.export_av1_web import transcode as web_transcode

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

AV1_KEYS = {
    "180p": "av1_180",
    "240p": "av1_240",
    "360p": "av1_360",
    "540p": "av1_540",
    "720p": "av1_720",
    "1080p": "av1_1080",
}
PS_KEYS = ["ps_starve", "ps_heavy", "ps_low", "ps_std", "ps_1080"]


def _load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"]
    out_channels = int(ckpt.get("out_channels") or 3)
    if ckpt.get("model_type") == "spade" or "enc1.0.weight" in state:
        model = HandSPADEUNet(in_channels=6, out_channels=out_channels).to(device)
    else:
        model = HandPix2PixUNet(in_channels=6, out_channels=out_channels).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, ckpt


def _ffmpeg() -> str:
    return os.environ.get("FFMPEG", "/opt/local/bin/ffmpeg")


def process_clip(
    clip_path: Path,
    short: str,
    work: Path,
    pitch: Path,
    model,
    device,
    frames: int,
    *,
    mask_video: Path | None = None,
    pose_backend: str = "dwpose_hands",
    skip_av1: bool = False,
    background_mp4s: dict[str, Path] | None = None,
    baseline_hand: dict[str, float] | None = None,
) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    pitch.mkdir(parents=True, exist_ok=True)
    judge = BACKENDS["rtm_hand"]
    driver = BACKENDS[pose_backend]
    ref_frames = read_video_frames_robust(clip_path, max_frames=frames)
    h, w = ref_frames[0].shape[:2]
    fps = 30.0
    duration = len(ref_frames) / fps
    ref_mp4 = work / "reference_trimmed.mp4"
    writer = cv2.VideoWriter(str(ref_mp4), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in ref_frames:
        writer.write(f)
    writer.release()
    driven = driver(ref_mp4, frames)
    poses, kp_bytes = dwb2_roundtrip(driven, w, h, stabilize=True)
    if kp_bytes == 0:
        packets = [KeypointCompressor.compress_frame(p, w, h) for p in poses]
        kp_bytes = sum(len(p) for p in packets)
    kp_kbps = (kp_bytes * 8) / (duration * 1000.0)
    frame_alphas = None
    if mask_video is not None:
        frame_alphas = interpolate_hand_alphas(read_hand_alphas(mask_video, w, h, frames))
    _, anchors, _ = build_curated_samples(
        ref_mp4, poses, image_size=256, max_frames=frames, clip_id=0, frame_alphas=frame_alphas
    )
    gt = judge(ref_mp4, frames)
    streams = {}
    for (tier_name, scale_res, _bg_kbps, preset), key in zip(POINTSTREAM_TIERS, PS_KEYS):
        tag = tier_name.split("(")[0].strip().lower().replace(" ", "_")
        bg_mp4 = work / f"ps_bg_{tag}.mp4"
        if background_mp4s and key in background_mp4s:
            bg_src = background_mp4s[key]
            shutil.copy2(bg_src, bg_mp4)
            bg_bytes = bg_mp4.stat().st_size
            codec = BackgroundCodec(scale_resolution=scale_res or (w, h), preset=preset)
            bg_frames = codec.decode_background_frames(bg_mp4, w, h)
        else:
            if scale_res is not None:
                codec = BackgroundCodec(scale_resolution=scale_res, preset=preset)
            else:
                codec = BackgroundCodec(downscale_factor=1.0, preset=preset)
            _, bg_bytes = codec.prepare_background_video(ref_mp4, poses, bg_mp4, max_frames=frames)
            bg_frames = codec.decode_background_frames(bg_mp4, w, h)
        ps_mp4 = work / f"ps_rec_{tag}.mp4"
        reconstruct_pointstream_video(ref_frames, poses, model, anchors, bg_frames, ps_mp4, device, fps=fps)
        pred = judge(ps_mp4, frames)
        scored = score_pose_tracks(gt, pred)
        total_kbps = (bg_bytes * 8) / (duration * 1000.0) + kp_kbps
        res_label = {0: "240p", 1: "360p", 2: "540p", 3: "540p", 4: "1080p"}[PS_KEYS.index(key)]
        streams[key] = {
            "kind": "ps",
            "res": res_label,
            "kbps": round(total_kbps, 1),
            "mpjpe": round(scored["mpjpe_pixels"], 1),
            "det": round(scored["detection_rate"] * 100, 1),
            "kp": round(kp_kbps, 1),
            "bg_bytes": int(bg_bytes),
        }
        if frame_alphas is not None:
            rec_frames = read_video_frames_robust(ps_mp4, max_frames=frames)
            hand = composite_hand_metrics(ref_frames, rec_frames, bg_frames, frame_alphas)
            streams[key]["hand"] = {k: round(v, 6) for k, v in hand.items()}
            if baseline_hand is not None:
                score, bottleneck = selection_min(hand, baseline_hand)
                streams[key]["hand_min"] = round(score, 4)
                streams[key]["hand_bottleneck"] = bottleneck
                streams[key]["hand_ship"] = bool(score >= 1.0)
        # PS reconstruct is mp4v; wrap to H.264 for the canvas.
        web_transcode(ps_mp4, pitch / f"web_{short}_{key}.mp4", _ffmpeg())
    if skip_av1:
        return {"name": short, "clip_path": str(clip_path), "streams": streams, "kp_bytes": kp_bytes}
    for rung_name, scale in AV1_LADDER:
        key = AV1_KEYS[rung_name]
        av_mp4 = work / f"{key}_crf63.mp4"
        encode_av1_crf(ref_mp4, av_mp4, scale=scale, ffmpeg=_ffmpeg(), max_frames=frames)
        pred = judge(av_mp4, frames)
        scored = score_pose_tracks(gt, pred)
        actual = (av_mp4.stat().st_size * 8) / (duration * 1000.0)
        streams[key] = {
            "kind": "av1",
            "res": rung_name,
            "kbps": round(actual, 1),
            "mpjpe": round(scored["mpjpe_pixels"], 1),
            "det": round(scored["detection_rate"] * 100, 1),
            "kp": 0,
            "bytes": av_mp4.stat().st_size,
        }
        # Serve the AV1 file directly — no H.264 CRF 23 wrapper.
        dest = pitch / f"web_{short}_{key}.mp4"
        shutil.copy2(av_mp4, dest)
    # Reference: keep a playable H.264 for the inspector.
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
    parser.add_argument("--mask-video", type=Path, default=None, help="SAM mask for clip 1")
    parser.add_argument("--pose-backend", default="dwpose_hands")
    parser.add_argument("--skip-av1", action="store_true")
    parser.add_argument("--clip-only", type=str, default=None, help="e.g. clip_01")
    parser.add_argument("--bg-dir", type=Path, default=None, help="Optional winning background encodes per PS key")
    parser.add_argument(
        "--baseline-hand",
        type=Path,
        default=None,
        help="JSON with appearance, matte, jitter from the shipped clip (lower is better)",
    )
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _ = _load_model(args.checkpoint, device)
    manifest = json.loads((args.curated_dir / "manifest.json").read_text())
    baseline_hand = None
    if args.baseline_hand is not None:
        baseline_hand = json.loads(args.baseline_hand.read_text())
    report = {"clips": {}}
    shorts = ["clip_01", "clip_02", "clip_03"]
    background_mp4s = None
    if args.bg_dir is not None:
        background_mp4s = {k: args.bg_dir / f"{k}.mp4" for k in PS_KEYS if (args.bg_dir / f"{k}.mp4").is_file()}
    for item, short in zip(manifest[:3], shorts):
        if args.clip_only and short != args.clip_only:
            continue
        mask = args.mask_video if short == "clip_01" else None
        logger.info("demo rebuild %s", item["filename"])
        report["clips"][short] = process_clip(
            Path(item["path"]),
            short,
            args.work / short,
            args.pitch,
            model,
            device,
            args.frames,
            mask_video=mask,
            pose_backend=args.pose_backend,
            skip_av1=args.skip_av1,
            background_mp4s=background_mp4s,
            baseline_hand=baseline_hand,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
