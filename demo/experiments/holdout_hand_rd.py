"""Hand masks, RTMW-l poses, and an AV1 crop curve on the last-10-second hold-outs.

The three hold-out videos were cut and never segmented. This scores the same
hand crops two ways: SVT-AV1 CRF 63 at the shared resolution ladder, and the
keypoint packet that a hand generator would send. Segment length is the same
on both sides. Speed is frames per second after the models are loaded. 24 fps
is the real-time line.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HOLDOUTS = Path("/home/itec/emanuele/Datasets/pointstream-demo/holdouts")
CLIPS = (
    "clip_01_factory001_worker001_00001_last10s.mp4",
    "clip_03_factory001_worker001_00000_last10s.mp4",
    "factory002_worker001_00000_last10s.mp4",
)
FPS = 30.0
REALTIME_FPS = 24.0
CROP = 256


def steady_fps(n_frames: int, elapsed_s: float) -> float:
    if n_frames <= 0 or elapsed_s <= 0:
        return 0.0
    return n_frames / elapsed_s


def segment_bounds(n_frames: int, length: int) -> list[tuple[int, int]]:
    return [(start, min(start + length, n_frames)) for start in range(0, n_frames, length)]


def kbps(nbytes: int, n_frames: int, fps: float = FPS) -> float:
    if n_frames <= 0:
        return 0.0
    return (nbytes * 8) / ((n_frames / fps) * 1000.0)


def _extract_frames(clip: Path, frames_dir: Path, ffmpeg: str, max_frames: int | None) -> int:
    frames_dir.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-y", "-i", str(clip), "-q:v", "2", "-start_number", "0"]
    if max_frames is not None:
        cmd.extend(["-frames:v", str(max_frames)])
    cmd.append(str(frames_dir / "%05d.jpg"))
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frames = sorted(frames_dir.glob("*.jpg"))
    if res.returncode != 0 or not frames:
        tail = res.stderr.decode("utf-8", errors="replace")[-1500:]
        raise RuntimeError(f"frame extract failed for {clip}: {tail}")
    return len(frames)


def _sam_masks(frames: Path, masks: Path, dest: Path) -> dict:
    """SAM 3.1 egocentric foreground (arm + hand) as 0/255 PNGs, plus its timing."""
    from PIL import Image

    from src.segmentation import build, load_domain

    clip = build("sam31").segment(frames, load_domain("egocentric"))
    clip.save(dest / "sam")
    masks.mkdir(parents=True, exist_ok=True)
    nonempty = 0
    for index in range(len(clip)):
        plane = clip.foreground(index).astype(np.uint8) * 255
        nonempty += bool(plane.any())
        Image.fromarray(plane).save(masks / f"{index:05d}.png")
    timing = clip.meta["timing"]
    return {
        "n_frames": len(clip),
        "model_load_s": timing["model_load_s"],
        "propagate_s": timing["total_s"],
        "fps_excluding_load": timing["fps"],
        "n_nonempty": nonempty,
        "gpu": ((clip.meta.get("worker_runtime") or {}).get("gpu") or {}).get("name"),
        "prompts": clip.meta.get("prompts"),
    }


def _pose_clip(frames_dir: Path, masks_dir: Path) -> tuple[list[list[dict]], dict]:
    import cv2
    from rtmlib import RTMPose

    from demo.experiments.compare_pose_on_sam_crops import (
        MIN_AREA,
        RTMW_L,
        choose_hand,
        components,
        expand_box,
        parse_people,
    )
    from demo.pipeline.foreground_segmenter import letterbox_crop

    model = RTMPose(str(RTMW_L), model_input_size=(288, 384), to_openpose=False, backend="onnxruntime", device="cuda")
    paths = sorted(frames_dir.glob("*.jpg"))
    warmup = cv2.imread(str(paths[0]))
    model(warmup, [[0, 0, warmup.shape[1], warmup.shape[0]]])
    rows: list[list[dict]] = []
    crops: dict[str, list[np.ndarray | None]] = {"Left": [], "Right": []}
    started = time.perf_counter()
    calls = 0
    for path in paths:
        image = cv2.imread(str(path))
        mask = cv2.imread(str(masks_dir / f"{path.stem}.png"), cv2.IMREAD_GRAYSCALE)
        hands = []
        present = {"Left": None, "Right": None}
        if image is not None and mask is not None:
            for comp in components(mask, MIN_AREA):
                box = expand_box(comp, image.shape[1], image.shape[0])
                calls += 1
                keypoints, scores = model(image, [box])
                chosen = choose_hand(parse_people(keypoints, scores), mask)
                if chosen is None:
                    continue
                side = chosen["side"] if chosen["side"] in crops else "Left"
                hand = {
                    "side": side,
                    "confidence": chosen["confidence"],
                    "box": box,
                    "landmarks_pixel": chosen["points"],
                    "selected": True,
                }
                hands.append(hand)
                alpha = mask
                target, _ = letterbox_crop(image, [int(v) for v in box], target_size=CROP)
                gray = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)
                canvas, _ = letterbox_crop(gray, [int(v) for v in box], target_size=CROP)
                keep = canvas[:, :, 0][:, :, None] > 8
                present[side] = np.where(keep, target, 0)
        for side in ("Left", "Right"):
            crops[side].append(present[side])
        rows.append(hands)
    elapsed = time.perf_counter() - started
    timing = {
        "n_frames": len(paths),
        "pose_calls": calls,
        "pose_s": round(elapsed, 3),
        "fps_excluding_load": round(steady_fps(len(paths), elapsed), 3),
        "ms_per_frame": round(1000.0 * elapsed / max(1, len(paths)), 2),
    }
    return rows, {"timing": timing, "crops": crops}


def _write_raw(frames: list[np.ndarray], dest: Path, ffmpeg: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{CROP}x{CROP}",
        "-r", f"{FPS:.6f}", "-i", "-", "-c:v", "ffv1", str(dest),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    for frame in frames:
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    proc.stdin.close()
    code = proc.wait()
    if code != 0 or not dest.is_file():
        tail = proc.stderr.read().decode("utf-8", errors="replace")[-1500:] if proc.stderr else ""
        raise RuntimeError(f"crop video failed: {tail}")


_LPIPS = None


def _lpips_mean(refs: list[np.ndarray], recs: list[np.ndarray]) -> float | None:
    global _LPIPS
    try:
        import lpips
        import torch
    except ImportError:
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if _LPIPS is None:
        _LPIPS = lpips.LPIPS(net="alex", verbose=False).to(device)
        _LPIPS.eval()
    fn = _LPIPS
    scores = []
    with torch.no_grad():
        for ref, rec in zip(refs, recs):
            if rec.shape[:2] != ref.shape[:2]:
                import cv2
                rec = cv2.resize(rec, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
            pair = []
            for image in (ref, rec):
                import cv2
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 127.5 - 1.0
                pair.append(torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device))
            scores.append(float(fn(pair[0], pair[1])))
    return float(np.mean(scores)) if scores else None


def _scaled_size(scale: str | None) -> tuple[int, int]:
    if scale is None:
        return CROP, CROP
    w, h = scale.split(":")
    return int(w), int(h)


def av1_curve(crops: dict[str, list[np.ndarray | None]], dest: Path, ffmpeg: str, lengths: tuple[int, ...]) -> list[dict]:
    from demo.pipeline.maps.av1_crf import AV1_CRF, AV1_LADDER, encode_av1_crf
    n_frames = len(next(iter(crops.values())))
    rows = []
    blank = np.zeros((CROP, CROP, 3), dtype=np.uint8)
    for length in lengths:
        for start, stop in segment_bounds(n_frames, length):
            span = stop - start
            for side, series in crops.items():
                window = series[start:stop]
                if not any(frame is not None for frame in window):
                    continue
                filled = [frame if frame is not None else blank for frame in window]
                raw = dest / f"seg{length}_{start}_{side}.ffv1.mkv"
                _write_raw(filled, raw, ffmpeg)
                for name, scale in AV1_LADDER:
                    encoded = dest / f"seg{length}_{start}_{side}_{name}_crf{AV1_CRF}.mp4"
                    encode_av1_crf(raw, encoded, scale=scale, ffmpeg=ffmpeg)
                    refs = [frame for frame in window if frame is not None]
                    decoded = _frames_from_av1(encoded, ffmpeg, _scaled_size(scale), span)
                    recs = [decoded[i] for i, frame in enumerate(window) if frame is not None]
                    rows.append({
                        "segment_frames": length,
                        "start": start,
                        "side": side,
                        "rung": name,
                        "bytes": encoded.stat().st_size,
                        "kbps": kbps(encoded.stat().st_size, span),
                        "lpips": _lpips_mean(refs, recs),
                        "hands": len(refs),
                    })
                raw.unlink(missing_ok=True)
    return rows


def _frames_from_av1(path: Path, ffmpeg: str, size: tuple[int, int], n_frames: int) -> list[np.ndarray]:
    width, height = size
    cmd = [ffmpeg, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.decode("utf-8", errors="replace")[-800:])
    frame_bytes = width * height * 3
    blob = res.stdout
    frames = []
    for index in range(n_frames):
        chunk = blob[index * frame_bytes : (index + 1) * frame_bytes]
        if len(chunk) < frame_bytes:
            break
        frames.append(np.frombuffer(chunk, dtype=np.uint8).reshape(height, width, 3).copy())
    return frames


def _collapse(rows: list[dict]) -> list[dict]:
    """Add the bytes of every hand that shares a segment, and average LPIPS.

    Two hands in the same second are one duration. Their AV1 files both count.
    """
    grouped: dict[tuple, list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["segment_frames"], row["rung"]), []).append(row)
    out = []
    for (length, rung), items in sorted(grouped.items()):
        span_frames: dict[int, int] = {}
        for item in items:
            span_frames[item["start"]] = int(item.get("span_frames", length))
        total_frames = sum(span_frames.values())
        total = sum(item["bytes"] for item in items)
        lpips_values = [item["lpips"] for item in items if item["lpips"] is not None]
        out.append({
            "segment_frames": length,
            "rung": rung,
            "bytes": total,
            "kbps": kbps(total, total_frames),
            "lpips": float(np.mean(lpips_values)) if lpips_values else None,
        })
    return out


def run_clip(clip: Path, dest: Path, ffmpeg: str, max_frames: int | None, lengths: tuple[int, ...]) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    frames = dest / "frames"
    n_frames = _extract_frames(clip, frames, ffmpeg, max_frames)
    masks = dest / "masks"
    sam = _sam_masks(frames, masks, dest)
    poses, pose_pack = _pose_clip(frames, masks)
    (dest / "poses.json").write_text(json.dumps(poses) + "\n")
    from demo.experiments.hand_packet_rate import score_tracks
    packets = score_tracks(poses, segment_lengths=lengths, fps=FPS)
    curve = av1_curve(pose_pack["crops"], dest / "av1", ffmpeg, lengths)
    for row in curve:
        row["span_frames"] = min(row["segment_frames"], n_frames - row["start"])
    pose_timing = pose_pack["timing"]
    combined_s = float(sam["propagate_s"]) + float(pose_timing["pose_s"])
    return {
        "clip": clip.name,
        "frames": n_frames,
        "sam": sam,
        "pose": pose_timing,
        "combined_fps_excluding_load": round(steady_fps(n_frames, combined_s), 3),
        "realtime_24fps": steady_fps(n_frames, combined_s) >= REALTIME_FPS,
        "packets": packets,
        "av1": _collapse(curve),
    }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="/opt/local/bin/ffmpeg")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--segments", default="30,300")
    parser.add_argument("--clips", nargs="*", default=list(CLIPS))
    parser.add_argument("--smoke-first", action="store_true")
    ns = parser.parse_args(args)
    lengths = tuple(int(part) for part in ns.segments.split(",") if part)
    ns.out.mkdir(parents=True, exist_ok=True)
    if ns.smoke_first:
        smoke_clip = Path(ns.clips[0]) if Path(ns.clips[0]).is_file() else HOLDOUTS / ns.clips[0]
        print("SMOKE", smoke_clip, flush=True)
        smoke = run_clip(smoke_clip, ns.out / "smoke", ns.ffmpeg, 8, (8,))
        (ns.out / "smoke_report.json").write_text(json.dumps(smoke, indent=2) + "\n")
        if smoke["frames"] < 1 or "fps_excluding_load" not in smoke["sam"]:
            raise SystemExit("hold-out smoke did not time a SAM pass")
        print("SMOKE fps", smoke["combined_fps_excluding_load"], flush=True)
    clips = []
    for name in ns.clips:
        path = Path(name) if Path(name).is_file() else HOLDOUTS / name
        print("CLIP", path, flush=True)
        clips.append(run_clip(path, ns.out / path.stem, ns.ffmpeg, ns.max_frames, lengths))
        print("DONE", path.name, "fps", clips[-1]["combined_fps_excluding_load"], flush=True)
    report = {"schema": "pointstream.holdout_hand_rd.v1", "realtime_fps": REALTIME_FPS, "clips": clips}
    (ns.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("WROTE", ns.out / "report.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
