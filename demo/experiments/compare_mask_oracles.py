"""Oracle screen: metadata plus a measured background, against AV1.

Each foreground channel keeps reference pixels inside its decoded support and
uses the cheaper background outside. Rate is payload bytes plus that
background, not the original picture. LPIPS, segmentation IoU, and RTMPose-m
pose agreement are all reported. A worse LPIPS at a lower rate stays in the
report. Canny and DW-Pose are not candidates.

The RTMPose-m channel is the encoder and the judge, so its pose score is
marked partly circular. YOLOE is marked partly circular when it is also the
segmenter.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from demo.pipeline.maps.encode import union_mask_from_frame, unpack_rle_stream

logger = logging.getLogger(__name__)

CANDIDATES = ("yoloe", "sam31", "rtmpose", "dino")
FLAT_FILL = (114, 114, 114)


def composite_oracle(
    reference: np.ndarray,
    background: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    """Keep reference pixels on the support and background pixels elsewhere."""
    if reference.shape != background.shape:
        raise ValueError(f"shape {reference.shape} != background {background.shape}")
    mask = np.asarray(support) > 0
    if mask.shape[:2] != reference.shape[:2]:
        raise ValueError(f"support {mask.shape} != frame {reference.shape[:2]}")
    out = background.copy()
    out[mask] = reference[mask]
    return out


def flat_fill(frame: np.ndarray, support: np.ndarray, color: tuple[int, int, int] = FLAT_FILL) -> np.ndarray:
    """Replace the support with a constant color. Empty support leaves the frame unchanged."""
    out = frame.copy()
    mask = np.asarray(support) > 0
    if mask.shape[:2] != frame.shape[:2]:
        raise ValueError(f"support {mask.shape} != frame {frame.shape[:2]}")
    out[mask] = np.array(color, dtype=np.uint8)
    return out


def select_background(clean_bytes: int, hole_bytes: int) -> tuple[str, int]:
    """Keep the hole-fill only when it is strictly smaller than the clean encode."""
    if hole_bytes < clean_bytes:
        return "hole", int(hole_bytes)
    return "clean", int(clean_bytes)


def total_kbps(metadata_bytes: int, background_bytes: int, duration_s: float) -> float:
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    return (int(metadata_bytes) + int(background_bytes)) * 8.0 / (duration_s * 1000.0)


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = int(np.logical_or(left > 0, right > 0).sum())
    if union == 0:
        return 1.0
    inter = int(np.logical_and(left > 0, right > 0).sum())
    return inter / union


def mean_iou(reference_masks: list[np.ndarray], other_masks: list[np.ndarray]) -> float | None:
    n = min(len(reference_masks), len(other_masks))
    if n == 0:
        return None
    return float(np.mean([mask_iou(reference_masks[i], other_masks[i]) for i in range(n)]))


def dilate_support(support: np.ndarray, radius: int) -> np.ndarray:
    binary = (np.asarray(support) > 0).astype(np.uint8)
    if radius <= 0 or not binary.any():
        return binary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.dilate(binary, kernel)


def dino_support(features: np.ndarray, height: int, width: int, percentile: float = 75.0) -> np.ndarray:
    """Coarse patch mask from int8 feature magnitude. A proxy, not a segmentation."""
    magnitude = np.abs(np.asarray(features).astype(np.float32)).mean(axis=-1)
    threshold = float(np.percentile(magnitude, percentile))
    small = (magnitude >= threshold).astype(np.uint8)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


def rle_supports(payload: bytes) -> list[np.ndarray]:
    doc = unpack_rle_stream(payload)
    height, width = int(doc["height"]), int(doc["width"])
    return [union_mask_from_frame(frame, height, width) for frame in doc.get("frames") or []]


def build_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep every row. A worse LPIPS at a lower rate is still a candidate."""
    kept = []
    for row in rows:
        item = dict(row)
        item["kept"] = True
        lpips = item.get("lpips")
        av1_lpips = item.get("av1_lpips")
        rate = item.get("kbps")
        av1_rate = item.get("av1_kbps")
        item["worse_lpips_lower_rate"] = bool(
            lpips is not None
            and av1_lpips is not None
            and rate is not None
            and av1_rate is not None
            and float(lpips) > float(av1_lpips)
            and float(rate) < float(av1_rate)
        )
        kept.append(item)
    _attach_av1_reference(kept)
    return {
        "schema": "pointstream.mask_oracle_screen.v1",
        "dropped": [],
        "rows": kept,
        "recommendation": _recommend(kept),
    }


def _attach_av1_reference(rows: list[dict[str, Any]]) -> None:
    """Point each oracle at the same-clip AV1 rung whose published rate is closest."""
    by_clip: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if str(row.get("candidate") or "").startswith("av1_") and row.get("lpips") is not None:
            by_clip.setdefault(row["clip"], []).append(row)
    for row in rows:
        if row.get("candidate") not in CANDIDATES or row.get("kbps") is None:
            continue
        choices = by_clip.get(row.get("clip"), [])
        if not choices:
            continue
        match = min(choices, key=lambda item: abs(float(item["kbps"]) - float(row["kbps"])))
        row["av1_kbps"] = match["kbps"]
        row["av1_lpips"] = match.get("lpips")
        row["av1_pose_pck"] = match.get("pose_pck")
        row["av1_seg_iou"] = match.get("seg_iou")
        row["worse_lpips_lower_rate"] = bool(
            row.get("lpips") is not None
            and match.get("lpips") is not None
            and float(row["lpips"]) > float(match["lpips"])
            and float(row["kbps"]) < float(match["kbps"])
        )


def _recommend(rows: list[dict[str, Any]]) -> str:
    scored = [row for row in rows if row.get("kbps") is not None and row.get("candidate") in CANDIDATES]
    if not scored:
        return "No candidate had a measured rate. Nothing was dropped."
    def key(row: dict[str, Any]) -> tuple[float, float]:
        pose = row.get("pose_pck")
        pose_term = 0.0 if pose is None else float(pose)
        return (-pose_term, float(row["kbps"]))
    by_pose = sorted(scored, key=key)[0]
    by_rate = min(scored, key=lambda row: float(row["kbps"]))
    circular = " Pose on the RTMPose-m channel is partly circular." if any(row.get("candidate") == "rtmpose" for row in scored) else ""
    sam = " SAM 3.1 has no native payload on disk, so it was not scored." if any(row.get("error") == "no native SAM payload on disk" for row in rows) else ""
    return (
        f"Highest pose agreement is {by_pose['candidate']} at {by_pose['kbps']:.1f} kbps "
        f"(LPIPS {by_pose.get('lpips')}). Lowest rate is {by_rate['candidate']} at {by_rate['kbps']:.1f} kbps "
        f"(LPIPS {by_rate.get('lpips')}, pose {by_rate.get('pose_pck')}). "
        f"DINOv3 rate is the SVT-AV1 CRF 63 PCA preview, not the int8 feature file. "
        f"This is an oracle paste, not a trained generator. "
        f"Every measured map is listed; choose what to train.{circular}{sam}"
    )


def skeleton_support(frame_shape: tuple[int, int], pose: Any, radius: int = 6) -> np.ndarray:
    """Rasterize one frame of hand landmarks and dilate. pose is a FrameHandPose."""
    height, width = frame_shape
    canvas = np.zeros((height, width), dtype=np.uint8)
    for hand in getattr(pose, "hands", []) or []:
        points = getattr(hand, "landmarks_pixel", None) or []
        for point in points:
            x, y = int(round(point[0])), int(round(point[1]))
            if 0 <= x < width and 0 <= y < height:
                canvas[y, x] = 1
        for index in range(len(points) - 1):
            x1, y1 = int(round(points[index][0])), int(round(points[index][1]))
            x2, y2 = int(round(points[index + 1][0])), int(round(points[index + 1][1]))
            cv2.line(canvas, (x1, y1), (x2, y2), 1, 1)
    return dilate_support(canvas, radius)


def _write_mp4(frames: list[np.ndarray], path: Path, fps: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()


def encode_pair(
    reference: list[np.ndarray],
    supports: list[np.ndarray],
    dest_dir: Path,
    *,
    fps: float,
    scale: str = "426:240",
) -> dict[str, Any]:
    """Encode a clean downscale and a flat hole-fill. Keep the smaller file."""
    from demo.pipeline.maps.av1_crf import encode_av1_crf

    dest_dir.mkdir(parents=True, exist_ok=True)
    clean_src = dest_dir / "clean_src.mp4"
    hole_src = dest_dir / "hole_src.mp4"
    _write_mp4(reference, clean_src, fps)
    filled = [flat_fill(frame, support) for frame, support in zip(reference, supports)]
    _write_mp4(filled, hole_src, fps)
    clean_out = dest_dir / "clean.mp4"
    hole_out = dest_dir / "hole.mp4"
    encode_av1_crf(clean_src, clean_out, scale=scale)
    encode_av1_crf(hole_src, hole_out, scale=scale)
    choice, nbytes = select_background(clean_out.stat().st_size, hole_out.stat().st_size)
    return {
        "choice": choice,
        "background_bytes": nbytes,
        "clean_bytes": clean_out.stat().st_size,
        "hole_bytes": hole_out.stat().st_size,
        "path": str(clean_out if choice == "clean" else hole_out),
    }


def _resize_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    if mask.shape[:2] == (height, width):
        return (mask > 0).astype(np.uint8)
    return cv2.resize((mask > 0).astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)


def _find_payload(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


PUBLISHED_KBPS = {
    "clip_01": {
        "ps_180": 46.5, "ps_starve": 56.1, "ps_heavy": 91.3, "ps_low": 163.1, "ps_720": 216.1, "ps_1080": 368.8,
        "av1_180": 34.3, "av1_240": 49.1, "av1_360": 83.1, "av1_540": 141.4, "av1_720": 204.3, "av1_1080": 358.1,
    },
    "clip_02": {
        "ps_180": 50.1, "ps_starve": 66.2, "ps_heavy": 109.3, "ps_low": 203.8, "ps_720": 305.0, "ps_1080": 518.7,
        "av1_180": 45.6, "av1_240": 66.3, "av1_360": 112.7, "av1_540": 199.2, "av1_720": 293.0, "av1_1080": 516.3,
    },
    "clip_03": {
        "ps_180": 60.5, "ps_starve": 75.2, "ps_heavy": 124.6, "ps_low": 230.7, "ps_720": 354.8, "ps_1080": 597.9,
        "av1_180": 51.4, "av1_240": 76.6, "av1_360": 135.1, "av1_540": 239.2, "av1_720": 351.2, "av1_1080": 609.9,
    },
}


DEMO_SOURCES = {
    "clip_01": "clip_01_factory001_worker001_00001.mp4",
    "clip_02": "clip_02_factory001_worker001_00002.mp4",
    "clip_03": "clip_03_factory001_worker001_00000.mp4",
}


def _bytes_for_frames(path: Path, n_scored: int) -> int:
    """Bill the scored prefix of a whole-clip payload, using the sidecar frame count."""
    total = path.stat().st_size
    n_payload = 0
    sidecar = path.parent / "sidecar.json"
    if sidecar.is_file():
        try:
            n_payload = int(json.loads(sidecar.read_text(encoding="utf-8")).get("n_frames") or 0)
        except (OSError, ValueError, TypeError):
            n_payload = 0
    if n_payload > n_scored > 0:
        return int(round(total * n_scored / n_payload))
    return total


def _dino_pca_rate(ladder_root: Path | None, clip_id: str) -> dict[str, float] | None:
    """File rates of the SVT-AV1 CRF 63 PCA previews. Not the int8 feature blob."""
    if ladder_root is None:
        return None
    report = Path(ladder_root) / "report.json"
    if not report.is_file():
        return None
    clip = json.loads(report.read_text(encoding="utf-8")).get(clip_id) or {}
    rates = {
        name: float(item["kbps"])
        for name, item in clip.items()
        if isinstance(item, dict) and "kbps" in item
    }
    return rates or None


def av1_mask_supports(path: Path, n_frames: int, height: int, width: int) -> list[np.ndarray]:
    """Decode a native AV1 mask video. Non-black pixels are the support."""
    from demo.pipeline.background_codec import read_video_frames_robust

    frames = read_video_frames_robust(path, target_w=width, target_h=height, max_frames=n_frames)
    supports = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        supports.append((gray > 8).astype(np.uint8))
    return supports


def load_candidate_supports(
    maps_root: Path | None,
    clip_id: str,
    n_frames: int,
    height: int,
    width: int,
) -> dict[str, tuple[list[np.ndarray], int, bool]]:
    """Return candidate -> (supports, payload_bytes, circular_segmentation)."""
    found: dict[str, tuple[list[np.ndarray], int, bool]] = {}
    if maps_root is None:
        return found
    clip_dir = maps_root / clip_id
    yolo = _find_payload(clip_dir / "yoloe_masks", ("masks.rle.zst", "payload.bin", "masks.av1.mp4"))
    if yolo is not None:
        if yolo.suffix == ".mp4":
            supports = av1_mask_supports(yolo, n_frames, height, width)
        else:
            supports = [_resize_mask(mask, height, width) for mask in rle_supports(yolo.read_bytes())]
        found["yoloe"] = (supports[:n_frames], _bytes_for_frames(yolo, n_frames), False)
    sam = _find_payload(clip_dir / "sam31_masks", ("masks.rle.zst", "payload.bin", "masks.av1.mp4"))
    if sam is None:
        sam = _find_payload(clip_dir / "sam31", ("masks.rle.zst", "payload.bin", "masks.av1.mp4"))
    if sam is not None:
        if sam.suffix == ".mp4":
            supports = av1_mask_supports(sam, n_frames, height, width)
        else:
            supports = [_resize_mask(mask, height, width) for mask in rle_supports(sam.read_bytes())]
        found["sam31"] = (supports[:n_frames], _bytes_for_frames(sam, n_frames), False)
    dino = _find_payload(clip_dir / "dino_feat", ("feat_32x32_int8.bin",))
    if dino is not None:
        from demo.pipeline.maps.dinov3_features import unpack_int8_features

        features, _scale = unpack_int8_features(dino.read_bytes())
        supports = [dino_support(frame, height, width) for frame in features[:n_frames]]
        found["dino"] = (supports, 0, False)
    return found


def score_pictures(
    reference: list[np.ndarray],
    pictures: dict[str, list[np.ndarray]],
    *,
    gt_poses: list[Any],
    judge: Any,
    quality: Any,
    segmenter: Any | None,
) -> dict[str, dict[str, Any]]:
    """LPIPS, segmentation IoU, and pose agreement of each picture against the reference."""
    from demo.evaluation.evaluate_robotics_teleop import score_pose_tracks

    ref_masks = segmenter(reference) if segmenter is not None else []
    scored: dict[str, dict[str, Any]] = {}
    for name, frames in pictures.items():
        lpips = None
        if quality is not None:
            metrics = quality.evaluate_frames(reference, frames)
            lpips = metrics.get("lpips")
        poses = judge(frames)
        pose = score_pose_tracks(gt_poses, poses)
        iou = mean_iou(ref_masks, segmenter(frames)) if segmenter is not None else None
        scored[name] = {
            "lpips": None if lpips is None else round(float(lpips), 4),
            "seg_iou": None if iou is None else round(float(iou), 4),
            "pose_detection": round(float(pose["detection_rate"]), 4),
            "pose_pck": round(float(pose["pck50_all_gt"]), 4),
            "mpjpe": round(float(pose["mpjpe_pixels"]), 2),
        }
    return scored


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rows", type=Path, default=None, help="Existing row JSON to wrap without scoring")
    parser.add_argument("--clips", type=Path, default=None)
    parser.add_argument("--maps-root", type=Path, default=None)
    parser.add_argument("--work", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--only", default="", help="Comma-separated short clip ids, e.g. clip_01")
    parser.add_argument("--shipped-only", action="store_true", help="Score shipped videos and skip oracle encodes")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--av1-dir", type=Path, default=None, help="Directory of web_{clip}_av1_{180,240,360}.mp4")
    parser.add_argument("--pitch-dir", type=Path, default=None, help="Shipped web_*.mp4 files to score for the legend")
    parser.add_argument("--segmenter-weights", type=Path, default=None)
    parser.add_argument(
        "--dino-ladder",
        type=Path,
        default=None,
        help="Directory with report.json from encode_dino_av1_ladder.py",
    )
    args = parser.parse_args(argv)
    if args.rows is not None:
        rows = json.loads(args.rows.read_text(encoding="utf-8"))
    else:
        if args.clips is None or args.work is None:
            raise SystemExit("scoring requires --clips and --work")
        rows = _score_clips(args)
    report = build_report(rows)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("wrote %s", args.report)
    return 0


def _score_clips(args: argparse.Namespace) -> list[dict[str, Any]]:
    import torch

    from demo.evaluation.evaluate_quality import QualityEvaluator
    from demo.evaluation.pose_backends import BACKENDS
    from demo.pipeline.background_codec import read_video_frames_robust
    from demo.pipeline.keypoint_compressor import KeypointCompressor

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    quality = QualityEvaluator(device_str=str(device))
    judge_video = BACKENDS["rtm_hand"]
    only = {item.strip() for item in str(getattr(args, "only", "") or "").split(",") if item.strip()}
    clips = []
    for clip_id, filename in DEMO_SOURCES.items():
        if only and clip_id not in only:
            continue
        path = args.clips / filename
        if path.is_file():
            clips.append((clip_id, path))
    if not clips:
        raise SystemExit(f"demo clips missing under {args.clips}")
    rows: list[dict[str, Any]] = []
    segmenter = _segmenter(args.segmenter_weights)
    for clip_id, clip in clips:
        frames = read_video_frames_robust(clip, max_frames=args.frames)
        if not frames:
            rows.append({"clip": clip_id, "candidate": None, "error": "no frames", "kbps": None})
            continue
        height, width = frames[0].shape[:2]
        duration = len(frames) / 30.0
        ref_path = args.work / clip_id / "reference.mp4"
        _write_mp4(frames, ref_path, 30.0)
        gt_poses = judge_video(ref_path, len(frames))
        packets = [KeypointCompressor.compress_frame(pose, width, height) for pose in gt_poses]
        rtm_supports = [skeleton_support((height, width), pose) for pose in gt_poses]
        supports = {} if args.shipped_only else load_candidate_supports(args.maps_root, clip_id, len(frames), height, width)
        if not args.shipped_only:
            supports["rtmpose"] = (rtm_supports, sum(len(packet) for packet in packets), False)
        if not args.shipped_only and "sam31" not in supports:
            rows.append({
                "clip": clip_id,
                "candidate": "sam31",
                "kbps": None,
                "error": "no native SAM payload on disk",
                "frames": len(frames),
            })
        for name, (masks, payload_bytes, circular_seg) in supports.items():
            aligned = masks[: len(frames)]
            if len(aligned) < len(frames):
                aligned = aligned + [np.zeros((height, width), dtype=np.uint8)] * (len(frames) - len(aligned))
            encoded = encode_pair(frames, aligned, args.work / clip_id / name, fps=30.0)
            from demo.pipeline.background_codec import read_video_frames_robust as read_frames

            background = read_frames(Path(encoded["path"]), target_w=width, target_h=height, max_frames=len(frames))
            composite = [
                composite_oracle(frame, bg, support)
                for frame, bg, support in zip(frames, background, aligned)
            ]
            kbps = total_kbps(payload_bytes, encoded["background_bytes"], duration)
            dino_rate = _dino_pca_rate(getattr(args, "dino_ladder", None), clip_id) if name == "dino" else None
            if dino_rate is not None:
                kbps = dino_rate["240p"]
            picture_path = args.work / clip_id / f"oracle_{name}.mp4"
            _write_mp4(composite, picture_path, 30.0)
            measured = _measure_picture(
                frames,
                composite,
                picture_path,
                gt_poses,
                judge_video,
                quality,
                segmenter,
            )
            rows.append(
                {
                    "clip": clip_id,
                    "candidate": name,
                    "kbps": round(kbps, 2),
                    "metadata_bytes": int(payload_bytes),
                    "background_bytes": encoded["background_bytes"],
                    "background_choice": encoded["choice"],
                    "clean_bytes": encoded["clean_bytes"],
                    "hole_bytes": encoded["hole_bytes"],
                    "circular_pose": name == "rtmpose",
                    "circular_segmentation": circular_seg,
                    "frames": len(frames),
                    "rate_basis": "pca_av1_crf63_240p" if dino_rate else "metadata_plus_background",
                    "dino_ladder_kbps": dino_rate,
                    **measured,
                }
            )
        rows.extend(
            _score_shipped(args.pitch_dir, clip_id, frames, gt_poses, judge_video, quality, segmenter)
        )
    return rows


def _score_shipped(pitch_dir, clip_id, reference, gt_poses, judge_video, quality, segmenter):
    if pitch_dir is None:
        return []
    from demo.pipeline.background_codec import read_video_frames_robust

    rows = []
    keys = (
        "av1_180", "av1_240", "av1_360", "av1_540", "av1_720", "av1_1080",
        "ps_180", "ps_starve", "ps_heavy", "ps_low", "ps_720", "ps_1080",
    )
    for key in keys:
        path = pitch_dir / f"web_{clip_id}_{key}.mp4"
        if not path.is_file():
            continue
        decoded = read_video_frames_robust(path, max_frames=len(reference))
        if not decoded:
            continue
        measured = _measure_picture(reference, decoded, path, gt_poses, judge_video, quality, segmenter)
        rows.append(
            {
                "clip": clip_id,
                "candidate": key,
                "kbps": PUBLISHED_KBPS.get(clip_id, {}).get(key),
                "shipped": True,
                "frames": len(decoded),
                **measured,
            }
        )
    return rows


def _measure_picture(reference, picture, picture_path, gt_poses, judge_video, quality, segmenter):
    from demo.evaluation.evaluate_robotics_teleop import score_pose_tracks

    metrics = quality.evaluate_frames(reference, picture)
    pose = score_pose_tracks(gt_poses, judge_video(picture_path, len(picture)))
    iou = None
    if segmenter is not None:
        reference_masks = segmenter(reference)
        if any(np.asarray(mask).any() for mask in reference_masks):
            iou = mean_iou(reference_masks, segmenter(picture))
    return {
        "lpips": round(float(metrics.get("lpips") or 0.0), 4),
        "seg_iou": None if iou is None else round(float(iou), 4),
        "pose_detection": round(float(pose["detection_rate"]), 4),
        "pose_pck": round(float(pose["pck50_all_gt"]), 4),
        "mpjpe": round(float(pose["mpjpe_pixels"]), 2),
    }


def _nearest_av1(av1_dir, clip_id, kbps, reference, gt_poses, judge_video, quality, segmenter=None):
    empty = {"av1_kbps": None, "av1_lpips": None, "av1_pose_pck": None, "av1_seg_iou": None}
    if av1_dir is None:
        return empty
    best = None
    for height in (180, 240, 360):
        path = av1_dir / f"web_{clip_id}_av1_{height}.mp4"
        if not path.is_file():
            continue
        from demo.pipeline.background_codec import read_video_frames_robust

        decoded = read_video_frames_robust(path, max_frames=len(reference))
        if not decoded:
            continue
        file_kbps = path.stat().st_size * 8.0 / ((len(decoded) / 30.0) * 1000.0)
        if best is None or abs(file_kbps - kbps) < abs(best[0] - kbps):
            best = (file_kbps, path, decoded)
    if best is None:
        return empty
    file_kbps, path, decoded = best
    measured = _measure_picture(reference, decoded, path, gt_poses, judge_video, quality, segmenter)
    return {
        "av1_kbps": round(file_kbps, 2),
        "av1_lpips": measured["lpips"],
        "av1_pose_pck": measured["pose_pck"],
        "av1_seg_iou": measured["seg_iou"],
    }


def _segmenter(weights: Path | None):
    if weights is None or not weights.is_file():
        return None
    from ultralytics import YOLO

    model = YOLO(str(weights))

    def run(frames: list[np.ndarray]) -> list[np.ndarray]:
        masks = []
        for frame in frames:
            height, width = frame.shape[:2]
            result = model.predict(frame[:, :, ::-1], verbose=False)[0]
            canvas = np.zeros((height, width), dtype=np.uint8)
            data = getattr(getattr(result, "masks", None), "data", None)
            if data is not None:
                if hasattr(data, "detach"):
                    data = data.detach().cpu().numpy()
                for mask in np.asarray(data):
                    binary = (mask > 0.5).astype(np.uint8)
                    if binary.shape != (height, width):
                        binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
                    canvas |= binary
            masks.append(canvas)
        return masks

    return run


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    raise SystemExit(main())
