"""Split control-path vs display-path teleop eval against a slow offline GT.

Control path
------------
PointStream: MediaPipe-live encoder keypoints after 47-byte quantize/dequantize.
AV1: the chosen pose backend run on the decoded AV1 video.

Display path
------------
Both codecs: the same pose backend run on the reconstructed RGB.

GT backends (slow, reference video only): ``mp_offline_gt`` (MediaPipe complexity 2,
same 21-joint topology — MPJPE is valid) and optional ``rtm_hand`` (detection /
PCK only vs MediaPipe encoder; MPJPE valid when both sides are RTM).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.evaluation.evaluate_robotics_teleop import score_pose_tracks
from demo.evaluation.pose_backends import BACKENDS
from demo.pipeline.hand_keypoints import FrameHandPose, serialize_poses_to_json
from demo.pipeline.keypoint_compressor import KeypointCompressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MATCHED_AV1 = {
    "PS Extreme Starve (240p bg, 30k)": "AV1 240p (40k, p7 - Starved)",
    "PS Heavy Starve (360p bg, 70k)": "AV1 360p (80k, p7 - Heavy Starve)",
    "PS Low Teleop (540p bg, 140k)": "AV1 540p (140k, p7 - Low)",
    "PS Standard (540p bg, 250k)": "AV1 540p (250k, p7 - Matched Rate & Latency)",
    "PS Standard 1080p (native bg, 300k)": "AV1 1080p (500k, p6 - Matched Quality)",
}


def load_poses_json(path: Path) -> list[FrameHandPose]:
    from demo.pipeline.hand_keypoints import SingleHand

    raw = json.loads(path.read_text())
    poses = []
    for row in raw:
        hands = [SingleHand(**h) for h in row["hands"]]
        poses.append(FrameHandPose(frame_idx=row["frame_idx"], hands=hands))
    return poses


def cached_extract(cache_dir: Path, key: str, video: Path, backend: str, max_frames: int) -> list[FrameHandPose]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{key}_{backend}.json"
    if out.exists():
        logger.info("Reusing cached poses %s", out.name)
        return load_poses_json(out)
    extractor = BACKENDS[backend]
    poses = extractor(video, max_frames)
    serialize_poses_to_json(poses, out)
    return poses


def wire_roundtrip(poses: list[FrameHandPose], w: int = 1920, h: int = 1080) -> list[FrameHandPose]:
    out: list[FrameHandPose] = []
    for pose in poses:
        pkt = KeypointCompressor.compress_frame(pose, w, h)
        hands = KeypointCompressor.decompress_frame(pkt, w, h)
        out.append(FrameHandPose(frame_idx=pose.frame_idx, hands=hands))
    return out


def _resolve(repo: Path, rel: str) -> Path:
    p = Path(rel)
    return p if p.is_absolute() else repo / p


def eval_existing(args: argparse.Namespace) -> dict[str, Any]:
    repo = REPO_ROOT
    results = json.loads((repo / args.results_json).read_text())
    cache = Path(args.cache_dir)
    report: dict[str, Any] = {"gt_backend": args.gt_backend, "eval_backend": args.eval_backend, "clips": []}

    for clip in results["clips"]:
        name = clip["clip_name"]
        logger.info("=== %s ===", name)
        ref = _resolve(repo, str(Path(clip["pointstream_variants"][0]["video_path"]).parent / "reference_trimmed.mp4"))
        if not ref.exists():
            logger.warning("Missing reference %s", ref)
            continue
        n_frames = int(clip.get("frames") or args.frames)
        gt = cached_extract(cache / name, "ref_gt", ref, args.gt_backend, n_frames)
        encoder = cached_extract(cache / name, "ref_encoder", ref, "mp_live", n_frames)
        wire = wire_roundtrip(encoder)
        control_ps = score_pose_tracks(gt, wire)

        clip_row: dict[str, Any] = {
            "clip_name": name,
            "gt_hands": control_ps["gt_hands"],
            "encoder_hands": control_ps["pred_hands"],
            "control_ps_wire": control_ps,
            "variants": [],
        }

        for ps in clip["pointstream_variants"]:
            av1_name = MATCHED_AV1.get(ps["name"])
            if av1_name is None:
                continue
            av1 = next((a for a in clip["av1_arms"] if a["name"] == av1_name), None)
            row: dict[str, Any] = {
                "ps_name": ps["name"],
                "ps_kbps": ps.get("bitrate_kbps"),
                "av1_name": av1["name"] if av1 else None,
                "av1_kbps": av1.get("actual_kbps") if av1 else None,
            }
            ps_vid = _resolve(repo, ps["video_path"])
            if ps_vid.exists():
                ps_disp = cached_extract(cache / name, Path(ps_vid).stem, ps_vid, args.eval_backend, n_frames)
                row["display_ps"] = score_pose_tracks(gt, ps_disp)
            if av1:
                av1_vid = _resolve(repo, av1["video_path"])
                if av1_vid.exists():
                    av1_pred = cached_extract(cache / name, Path(av1_vid).stem, av1_vid, args.eval_backend, n_frames)
                    scored = score_pose_tracks(gt, av1_pred)
                    row["control_av1"] = scored
                    row["display_av1"] = scored
            clip_row["variants"].append(row)
        report["clips"].append(clip_row)
    return report


def eval_scan(args: argparse.Namespace) -> dict[str, Any]:
    """Mine curated clips: encoder control-path vs AV1 240p on the GT backend."""
    from demo.evaluation.encode_av1_ladder import encode_av1

    manifest = json.loads(Path(args.curated_dir).joinpath("manifest.json").read_text())
    cache = Path(args.cache_dir)
    out_clips = []
    for item in manifest[: args.max_clips]:
        clip_path = Path(item["path"])
        stem = clip_path.stem
        logger.info("scan %s", stem)
        n = args.frames
        gt = cached_extract(cache / stem, "ref_gt", clip_path, args.gt_backend, n)
        encoder = cached_extract(cache / stem, "ref_encoder", clip_path, "mp_live", n)
        wire = wire_roundtrip(encoder)
        control_ps = score_pose_tracks(gt, wire)

        av1_path = cache / stem / "av1_240p.mp4"
        encode_av1(clip_path, av1_path, target_bitrate_kbps=40, max_frames=n, scale="426:240", preset=7)
        av1_pred = cached_extract(cache / stem, "av1_240", av1_path, args.eval_backend, n)
        control_av1 = score_pose_tracks(gt, av1_pred)
        out_clips.append({
            "clip_name": stem,
            "path": str(clip_path),
            "control_ps_wire": control_ps,
            "control_av1_240p": control_av1,
            "ps_wins_det": control_ps["detection_rate"] > control_av1["detection_rate"] + 0.005,
            "ps_wins_pck": control_ps["pck50_all_gt"] > control_av1["pck50_all_gt"] + 0.005,
            "ps_wins_mpjpe": (
                control_ps["mpjpe_pixels"] > 0
                and control_av1["mpjpe_pixels"] > 0
                and control_ps["mpjpe_pixels"] < control_av1["mpjpe_pixels"] - 0.5
            ),
        })
    return {"gt_backend": args.gt_backend, "eval_backend": args.eval_backend, "clips": out_clips}


def summarize(report: dict[str, Any]) -> None:
    for clip in report["clips"]:
        logger.info("--- %s ---", clip.get("clip_name"))
        if "control_ps_wire" in clip and "variants" in clip:
            ps = clip["control_ps_wire"]
            logger.info(
                "  control PS wire: det=%.1f%%  mpjpe=%.1f  pck50_all=%.1f%%  (gt_hands=%.0f)",
                ps["detection_rate"] * 100,
                ps["mpjpe_pixels"],
                ps["pck50_all_gt"] * 100,
                ps["gt_hands"],
            )
            for v in clip["variants"]:
                dps = v.get("display_ps") or {}
                av = v.get("display_av1") or {}
                logger.info(
                    "  %s vs %s | display det PS %.1f / AV1 %.1f | pck_all PS %.1f / AV1 %.1f | mpjpe PS %.1f / AV1 %.1f",
                    v["ps_name"],
                    v.get("av1_name"),
                    (dps.get("detection_rate") or 0) * 100,
                    (av.get("detection_rate") or 0) * 100,
                    (dps.get("pck50_all_gt") or 0) * 100,
                    (av.get("pck50_all_gt") or 0) * 100,
                    dps.get("mpjpe_pixels") or 0,
                    av.get("mpjpe_pixels") or 0,
                )
        elif "control_av1_240p" in clip:
            ps, av = clip["control_ps_wire"], clip["control_av1_240p"]
            logger.info(
                "  PS wire det=%.1f pck=%.1f mpjpe=%.1f | AV1 240p det=%.1f pck=%.1f mpjpe=%.1f | wins det=%s pck=%s mpjpe=%s",
                ps["detection_rate"] * 100,
                ps["pck50_all_gt"] * 100,
                ps["mpjpe_pixels"],
                av["detection_rate"] * 100,
                av["pck50_all_gt"] * 100,
                av["mpjpe_pixels"],
                clip["ps_wins_det"],
                clip["ps_wins_pck"],
                clip["ps_wins_mpjpe"],
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["existing", "scan"], default="existing")
    parser.add_argument("--results-json", type=Path, default=Path("demo/outputs/results/comparison_results.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("demo/outputs/results/offline_gt"))
    parser.add_argument("--gt-backend", default="rtm_hand", choices=list(BACKENDS))
    parser.add_argument("--eval-backend", default="rtm_hand", choices=list(BACKENDS))
    parser.add_argument("--curated-dir", type=Path, default=Path("/home/itec/emanuele/Datasets/Egocentric-10K/curated"))
    parser.add_argument("--max-clips", type=int, default=12)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--out", type=Path, default=Path("demo/outputs/results/offline_gt_report.json"))
    args = parser.parse_args()
    report = eval_existing(args) if args.mode == "existing" else eval_scan(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    summarize(report)
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
