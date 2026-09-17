"""Re-synthesize PointStream using RTMPose as the live encoder (wire keypoints).

Reuses existing ``ps_bg_*.mp4`` backgrounds so AV1 is not re-encoded. Appearance
anchors still come from MediaPipe crops (the UNet was trained that way); only
the skeleton stream is swapped to RTMPose.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from demo.evaluation.evaluate_robotics_teleop import score_pose_tracks
from demo.evaluation.pose_backends import BACKENDS
from demo.experiments.eval_split_teleop import cached_extract, wire_roundtrip
from demo.experiments.run_comparison import reconstruct_pointstream_video
from demo.models.dataset import build_curated_samples
from demo.models.unet_generator import HandPix2PixUNet, HandSPADEUNet
from demo.pipeline.background_codec import BackgroundCodec, read_video_frames_robust
from demo.pipeline.hand_keypoints import FrameHandPose, SingleHand

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TIERS = [
    ("PS Extreme Starve (240p bg, 30k)", "ps_extreme_starve"),
    ("PS Heavy Starve (360p bg, 70k)", "ps_heavy_starve"),
    ("PS Standard (540p bg, 250k)", "ps_standard"),
    ("PS Standard 1080p (native bg, 300k)", "ps_standard_1080p"),
]


def assign_handedness(rtm: list[FrameHandPose], mp: list[FrameHandPose]) -> list[FrameHandPose]:
    """Copy Left/Right from the nearest MediaPipe hand, else x-order."""
    out: list[FrameHandPose] = []
    n = min(len(rtm), len(mp))
    for i in range(n):
        mp_hands = mp[i].hands
        new_hands: list[SingleHand] = []
        used: set[int] = set()
        for h in rtm[i].hands:
            pts = np.array(h.landmarks_pixel, dtype=np.float64)
            c = np.mean(pts, axis=0)
            best_j, best_d = -1, 1e9
            for j, mh in enumerate(mp_hands):
                if j in used:
                    continue
                mc = np.mean(np.array(mh.landmarks_pixel, dtype=np.float64), axis=0)
                d = float(np.linalg.norm(c - mc))
                if d < best_d:
                    best_d, best_j = d, j
            if best_j >= 0 and best_d < 350:
                used.add(best_j)
                side = mp_hands[best_j].handedness
            else:
                side = "Left" if c[0] < 960 else "Right"
            new_hands.append(
                SingleHand(
                    handedness=side,
                    confidence=h.confidence,
                    bbox=h.bbox,
                    landmarks_norm=h.landmarks_norm,
                    landmarks_pixel=h.landmarks_pixel,
                )
            )
        out.append(FrameHandPose(frame_idx=rtm[i].frame_idx, hands=new_hands))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-json", type=Path, default=Path("demo/outputs/results/comparison_results.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("demo/outputs/results/offline_gt"))
    parser.add_argument("--checkpoint", type=Path, default=Path("demo/outputs/models/overfit_generator.pt"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--out", type=Path, default=Path("demo/outputs/results/rtm_encoder_report.json"))
    args = parser.parse_args()

    results = json.loads(args.results_json.read_text())
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt["model_state_dict"]
    if ckpt.get("model_type") == "spade" or "enc1.0.weight" in state:
        model = HandSPADEUNet(in_channels=6, out_channels=3).to(device)
    else:
        model = HandPix2PixUNet(in_channels=6, out_channels=3).to(device)
    model.load_state_dict(state)
    model.eval()

    report = {"clips": []}
    for clip in results["clips"]:
        name = clip["clip_name"]
        clip_dir = Path(clip["pointstream_variants"][0]["video_path"]).parent
        ref = clip_dir / "reference_trimmed.mp4"
        n = int(clip.get("frames") or args.frames)
        logger.info("RTM encoder reconstruct %s", name)
        gt = cached_extract(args.cache_dir / name, "ref_gt", ref, "rtm_hand", n)
        mp = cached_extract(args.cache_dir / name, "ref_encoder", ref, "mp_live", n)
        rtm = assign_handedness(gt, mp)
        wire = wire_roundtrip(rtm)
        control = score_pose_tracks(gt, wire)
        ref_frames = read_video_frames_robust(ref, max_frames=n)
        _, anchors, _ = build_curated_samples(ref, mp, image_size=256, max_frames=n, clip_id=clip["clip_id"])
        bg_codec = BackgroundCodec(downscale_factor=1.0, target_bitrate_kbps=30, preset=7)
        variants = []
        for tier_name, tag in TIERS:
            bg_mp4 = clip_dir / f"ps_bg_{tag}.mp4"
            if not bg_mp4.exists():
                logger.warning("missing %s", bg_mp4)
                continue
            bg_frames = bg_codec.decode_background_frames(bg_mp4, ref_frames[0].shape[1], ref_frames[0].shape[0])
            out_mp4 = clip_dir / f"ps_rec_{tag}_rtm_encoder.mp4"
            reconstruct_pointstream_video(
                ref_frames, wire, model, anchors, bg_frames, out_mp4, device, fps=30.0, image_size=256,
            )
            pred = BACKENDS["rtm_hand"](out_mp4, n)
            display = score_pose_tracks(gt, pred)
            logger.info(
                "  %s control_det=%.1f display_det=%.1f mpjpe=%.1f pck_all=%.1f",
                tag,
                control["detection_rate"] * 100,
                display["detection_rate"] * 100,
                display["mpjpe_pixels"],
                display["pck50_all_gt"] * 100,
            )
            variants.append({"name": tier_name, "video_path": str(out_mp4), "display": display, "control_wire": control})
        report["clips"].append({"clip_name": name, "control_wire": control, "variants": variants})
    args.out.write_text(json.dumps(report, indent=2))
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
