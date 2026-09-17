"""Score candidate windows at starve: PointStream vs AV1 240p, pick the 3 best."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from demo.evaluation.encode_av1_ladder import encode_av1
from demo.evaluation.evaluate_robotics_teleop import score_pose_tracks
from demo.evaluation.pose_backends import BACKENDS
from demo.experiments.run_comparison import reconstruct_pointstream_video
from demo.models.dataset import build_curated_samples
from demo.models.unet_generator import HandPix2PixUNet, HandSPADEUNet
from demo.pipeline.background_codec import BackgroundCodec, read_video_frames_robust
from demo.pipeline.hand_keypoints import extract_video_hand_poses

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def probe(clip_path: Path, work: Path, model, device, frames: int) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    ref_frames = read_video_frames_robust(clip_path, max_frames=frames)
    h, w = ref_frames[0].shape[:2]
    poses = extract_video_hand_poses(clip_path, max_frames=frames)
    gt = BACKENDS["rtm_hand"](clip_path, frames)
    _, anchors, _ = build_curated_samples(clip_path, poses, image_size=256, max_frames=frames, clip_id=0)
    bg_mp4 = work / "ps_bg_240.mp4"
    codec = BackgroundCodec(scale_resolution=(426, 240), target_bitrate_kbps=30, preset=7)
    _, bg_bytes = codec.prepare_background_video(clip_path, poses, bg_mp4, max_frames=frames)
    bg_frames = codec.decode_background_frames(bg_mp4, w, h)
    ps_mp4 = work / "ps_rec_240.mp4"
    reconstruct_pointstream_video(ref_frames, poses, model, anchors, bg_frames, ps_mp4, device, fps=30.0)
    av1_mp4 = work / "av1_240.mp4"
    av1 = encode_av1(clip_path, av1_mp4, target_bitrate_kbps=40, max_frames=frames, scale="426:240", preset=7)
    ps_pred = BACKENDS["rtm_hand"](ps_mp4, frames)
    av_pred = BACKENDS["rtm_hand"](av1_mp4, frames)
    ps = score_pose_tracks(gt, ps_pred)
    av = score_pose_tracks(gt, av_pred)
    duration = len(ref_frames) / 30.0
    ps_kbps = (bg_bytes * 8) / (duration * 1000.0)
    av_kbps = (Path(av1["output_path"]).stat().st_size * 8) / (duration * 1000.0)
    det_gap = ps["detection_rate"] - av["detection_rate"]
    mpjpe_gap = av["mpjpe_pixels"] - ps["mpjpe_pixels"] if ps["mpjpe_pixels"] and av["mpjpe_pixels"] else 0.0
    rate_ratio = av_kbps / max(ps_kbps, 1e-3)
    score = det_gap * 100.0 + 0.05 * mpjpe_gap + 8.0 * max(0.0, rate_ratio - 1.0)
    return {
        "path": str(clip_path),
        "ps": ps,
        "av1": av,
        "ps_bg_kbps": round(ps_kbps, 1),
        "av1_kbps": round(av_kbps, 1),
        "det_gap": round(det_gap, 4),
        "score": round(score, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("demo/outputs/models/overfit_generator.pt"))
    parser.add_argument("--work", type=Path, default=Path("demo/outputs/results/clip_probe"))
    parser.add_argument("--out", type=Path, default=Path("demo/outputs/results/clip_probe.json"))
    parser.add_argument("--curated-dir", type=Path, default=Path("/home/itec/emanuele/Datasets/Egocentric-10K/curated"))
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    cands = json.loads(args.candidates.read_text())
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model_state_dict"]
    model = HandSPADEUNet(in_channels=6, out_channels=3).to(device) if ckpt.get("model_type") == "spade" or "enc1.0.weight" in state else HandPix2PixUNet(in_channels=6, out_channels=3).to(device)
    model.load_state_dict(state)
    model.eval()

    rows = []
    for i, item in enumerate(cands):
        path = Path(item["path"])
        logger.info("probe %s", path.name)
        row = probe(path, args.work / path.stem, model, device, args.frames)
        row["candidate"] = item
        rows.append(row)
        logger.info(
            "  score=%.3f det PS %.1f AV1 %.1f  kbps PS %.1f AV1 %.1f",
            row["score"],
            row["ps"]["detection_rate"] * 100,
            row["av1"]["detection_rate"] * 100,
            row["ps_bg_kbps"],
            row["av1_kbps"],
        )
    rows.sort(key=lambda r: r["score"], reverse=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"ranked": rows}, indent=2))

    args.curated_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for rank, row in enumerate(rows[: args.top_k], start=1):
        src = Path(row["path"])
        dest = args.curated_dir / f"clip_{rank:02d}_{src.stem}.mp4"
        shutil.copy2(src, dest)
        manifest.append({"rank": rank, "filename": dest.name, "path": str(dest), **row["candidate"], "probe_score": row["score"]})
        logger.info("selected clip %s <- %s (score %.3f)", dest.name, src.name, row["score"])
    (args.curated_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
