"""Background bake-off on clip 1: untouched / cleaned / sprite+residual at CRF 63."""

from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from demo.models.matte import interpolate_hand_alphas, read_hand_alphas
from demo.pipeline.background_codec import read_video_frames_robust
from demo.pipeline.maps.av1_crf import AV1_LADDER, encode_av1_crf
from src.components.background.plate import build_plate

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _ffmpeg() -> str:
    return os.environ.get("FFMPEG", "/opt/local/bin/ffmpeg")


def _write_raw_mp4(frames: list[np.ndarray], path: Path, fps: float = 30.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return path


def _kbps(nbytes: int, n_frames: int, fps: float = 30.0) -> float:
    duration = max(n_frames / fps, 1e-6)
    return (nbytes * 8) / (duration * 1000.0)


def temporal_median_fill(
    frames: list[np.ndarray],
    alphas: list[np.ndarray],
    *,
    window: int = 21,
) -> list[np.ndarray]:
    """Replace hand pixels with a temporal median of non-hand pixels from nearby frames."""
    n = len(frames)
    half = window // 2
    stacked = np.stack(frames, axis=0)  # N,H,W,3 uint8
    masks = []
    for a in alphas:
        if a.shape[:2] != frames[0].shape[:2]:
            a = cv2.resize(a, (frames[0].shape[1], frames[0].shape[0]), interpolation=cv2.INTER_NEAREST)
        masks.append(a > 127)
    masks_arr = np.stack(masks, axis=0)  # N,H,W

    # Coarse global plate as fallback when a pixel is hand in every window frame.
    small = [
        cv2.resize(f, (frames[0].shape[1] // 4, frames[0].shape[0] // 4), interpolation=cv2.INTER_AREA)
        for f in frames
    ]
    small_m = [
        cv2.resize(m.astype(np.uint8), (small[0].shape[1], small[0].shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        for m in masks
    ]
    small_stack = np.stack(small, axis=0).astype(np.float32)
    small_stack[np.stack(small_m, axis=0)] = np.nan
    with np.errstate(all="ignore"):
        plate_s = np.nanmedian(small_stack, axis=0)
    plate_s = np.nan_to_num(plate_s, nan=128.0)
    h, w = frames[0].shape[:2]
    plate = cv2.resize(plate_s.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR)

    out: list[np.ndarray] = []
    for i in range(n):
        frame = frames[i].copy()
        hand = masks_arr[i]
        if not hand.any():
            out.append(frame)
            continue
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        neigh = stacked[lo:hi][:, hand].astype(np.float32)  # Wwin,P,3
        valid = ~masks_arr[lo:hi][:, hand]  # Wwin,P
        masked = neigh.copy()
        masked[~valid] = np.nan
        with np.errstate(all="ignore"):
            med = np.nanmedian(masked, axis=0)  # P,3
        need = np.isnan(med).any(axis=1)
        if need.any():
            med[need] = plate[hand][need]
        frame[hand] = np.clip(np.nan_to_num(med, nan=128.0), 0, 255).astype(np.uint8)
        out.append(frame)
        if i % 50 == 0:
            logger.info("cleaned frame %d/%d (hand_px=%d)", i, n, int(hand.sum()))
    return out


def sprite_residual(
    frames: list[np.ndarray],
    alphas: list[np.ndarray],
    work: Path,
) -> tuple[Path, Path, list[np.ndarray], list[np.ndarray]]:
    """Registered panorama + per-frame residual of what the warp misses."""
    stack = np.stack(frames, axis=0)
    masks = []
    for a in alphas:
        if a.shape[:2] != frames[0].shape[:2]:
            a = cv2.resize(a, (frames[0].shape[1], frames[0].shape[0]), interpolation=cv2.INTER_NEAREST)
        masks.append(a)
    plate, packed = build_plate(stack, masks=masks, register=True)
    plate_path = work / "sprite_plate.png"
    cv2.imwrite(str(plate_path), plate)
    # packed are 9-tuples mapping frame -> plate canvas.
    h, w = frames[0].shape[:2]
    residuals: list[np.ndarray] = []
    warps: list[np.ndarray] = []
    for i, H in enumerate(packed):
        mat = np.asarray(H, dtype=np.float64).reshape(3, 3)
        # Inverse: plate -> frame
        try:
            inv = np.linalg.inv(mat)
        except np.linalg.LinAlgError:
            inv = np.eye(3)
        warped = cv2.warpPerspective(plate, inv, (w, h), flags=cv2.INTER_LINEAR)
        warps.append(warped)
        # Residual only outside the hand (hand is synthesized separately).
        hand = masks[i] > 127
        residual = cv2.absdiff(frames[i], warped)
        residual[hand] = 0
        residuals.append(residual)
    warp_bytes = work / "sprite_warps.bin"
    with warp_bytes.open("wb") as f:
        f.write(struct.pack("<I", len(packed)))
        for H in packed:
            f.write(struct.pack("<9d", *[float(x) for x in H]))
    return plate_path, warp_bytes, warps, residuals


def encode_arm(
    frames: list[np.ndarray],
    arm_name: str,
    work: Path,
    fps: float = 30.0,
) -> dict[str, dict]:
    raw = _write_raw_mp4(frames, work / f"{arm_name}_raw.mp4", fps=fps)
    results = {}
    for rung, scale in AV1_LADDER:
        dest = work / f"{arm_name}_{rung}_crf63.mp4"
        encode_av1_crf(raw, dest, scale=scale, ffmpeg=_ffmpeg())
        nbytes = dest.stat().st_size
        results[rung] = {
            "path": str(dest),
            "bytes": nbytes,
            "kbps": round(_kbps(nbytes, len(frames), fps), 1),
        }
        logger.info("%s %s: %d bytes (%.1f kbps)", arm_name, rung, nbytes, results[rung]["kbps"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=Path("demo/outputs/bg_bakeoff"))
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--out", type=Path, default=Path("demo/outputs/bg_bakeoff/scorecard.json"))
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)

    frames = read_video_frames_robust(args.clip, max_frames=args.frames)
    h, w = frames[0].shape[:2]
    alphas = interpolate_hand_alphas(read_hand_alphas(args.mask, w, h, args.frames))
    while len(alphas) < len(frames):
        alphas.append(np.zeros((h, w), dtype=np.uint8))
    alphas = alphas[: len(frames)]

    scorecard: dict = {"frames": len(frames), "arms": {}}

    logger.info("Arm: untouched")
    scorecard["arms"]["untouched"] = encode_arm(frames, "untouched", args.work)

    logger.info("Arm: cleaned (temporal median fill)")
    cleaned = temporal_median_fill(frames, alphas)
    scorecard["arms"]["cleaned"] = encode_arm(cleaned, "cleaned", args.work)
    _write_raw_mp4(cleaned, args.work / "cleaned_preview.mp4")

    logger.info("Arm: sprite + residual")
    plate_path, warp_path, _warps, residuals = sprite_residual(frames, alphas, args.work)
    plate_bytes = plate_path.stat().st_size
    warp_bytes = warp_path.stat().st_size
    residual_rungs = encode_arm(residuals, "residual", args.work)
    sprite_arm = {}
    for rung, info in residual_rungs.items():
        total = plate_bytes + warp_bytes + info["bytes"]
        sprite_arm[rung] = {
            "plate_bytes": plate_bytes,
            "warp_bytes": warp_bytes,
            "residual_bytes": info["bytes"],
            "bytes": total,
            "kbps": round(_kbps(total, len(frames)), 1),
            "residual_path": info["path"],
            "plate_path": str(plate_path),
            "warp_path": str(warp_path),
        }
        logger.info("sprite %s: total %d bytes (%.1f kbps)", rung, total, sprite_arm[rung]["kbps"])
    scorecard["arms"]["sprite"] = sprite_arm

    # Pick winner per rung: smallest bytes. Prefer cleaned/untouched over sprite if within 5%.
    winners = {}
    for rung, _ in AV1_LADDER:
        cands = {
            "untouched": scorecard["arms"]["untouched"][rung]["bytes"],
            "cleaned": scorecard["arms"]["cleaned"][rung]["bytes"],
            "sprite": scorecard["arms"]["sprite"][rung]["bytes"],
        }
        winner = min(cands, key=cands.get)
        winners[rung] = {"arm": winner, "bytes": cands[winner], "all": cands}
    scorecard["winners"] = winners

    # Overall: majority of mid rungs (240/360/540), break ties toward cleaned then untouched.
    mid = ["240p", "360p", "540p"]
    votes: dict[str, int] = {}
    for r in mid:
        arm = winners[r]["arm"]
        votes[arm] = votes.get(arm, 0) + 1
    overall = max(votes, key=lambda a: (votes[a], {"cleaned": 2, "untouched": 1, "sprite": 0}[a]))
    scorecard["overall_winner"] = overall
    scorecard["overall_reason"] = (
        f"{overall} won {votes.get(overall, 0)}/{len(mid)} mid rungs "
        f"(bytes at 240p: untouched={winners['240p']['all']['untouched']}, "
        f"cleaned={winners['240p']['all']['cleaned']}, sprite={winners['240p']['all']['sprite']})"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scorecard, indent=2))
    logger.info("winner=%s reason=%s", overall, scorecard["overall_reason"])
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
