"""PLAN step K4: what broadcast tennis is made of, and whether body keypoints could carry the players.

    python -m experiments.tennis.k4 run --windows JSON --select all|ID,ID --frames N --source NAME PATH ... \\
        --sam-checkpoint PT --pose NAME PT ... --inspect BIN
    python -m experiments.tennis.k4 camera --video MP4 --fps F --limit-seconds S --pose PT
    python -m experiments.tennis.k4 validate
    python -m experiments.tennis.k4 report --published TAR ... --camera JSON --windows JSON --out DIR

``run``, per window (docs/experiments.md, 2026-10-13 K4):

1. *Window.* ffmpeg to 8-bit 4:2:0 (`experiments.tennis.windows`); every analysis reads that file.
2. *Encodes.* SVT-AV1 preset 4 at `CRFS`, one keyframe (B2's settings), started at once on the CPU while the GPU works.
3. *Masks.* SAM 3.1 text prompts (`PROMPTS`) on 960x540 JPEGs, back to the window's size. TrackNet's ball is its
   label: the SAM ball mask that reaches the labelled point, else a disk of `BALL_DISK_PX`.
4. *Parts.* Each pixel background, player, racket or ball (later wins).
5. *Pose.* Each YOLO pose model on every frame (image size 1280); each SAM player instance takes the detection with
   at least half its box inside the mask's box (H1's matching). Fit, keypoints inside the mask, stability, and the
   quantized keypoints' first-difference symbols (as histograms, pooled by ``report``).
6. *Bits.* libaom ``inspect`` with bit accounting on each stream (H1's method), split by the parts, and by the
   ball's neighbourhood.
7. *Camera.* Frames 0.5 s apart: still, moving or cut.
8. *Review.* Content-blind frames with the parts and both models' skeletons.

``camera`` measures the whole broadcast match every 0.5 s (still, moving, cut; wide or close by YOLO's people).

Writes ``k4.json`` (or ``k4-camera.json``) to ``PS_STAGE_DIR``; each window to ``publish/windows/<id>.json`` with its
review frames, checkpointed as it finishes and restored on a declared resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tarfile
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import windows as tw
from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json

#: The decision rule, fixed before any run (docs/experiments.md, K4 entry).
DECISION: dict[str, Any] = {
    "a_min_foreground_bit_share": 0.10, "a_crfs": [48, 62],
    "b_min_fitted_share": 0.90, "b_min_keypoints_inside_median": 0.90,
    "c_max_rate_share": 0.10, "c_crf": 62,
    "k6_max_rate_share": 0.50, "k6_dataset": "broadcast",
}
PARTS = ("background", "player", "racket", "ball")
FOREGROUND = PARTS[1:]
NBHD_PARTS = ("other", "ball_neighbourhood")
PROMPTS = {"player": "tennis player", "racket": "tennis racket", "ball": "tennis ball"}
CRFS = (41, 48, 55, 62)
PRESET = 4
SAM_SIZE = (960, 540)
BALL_DISK_PX = 4.0
BALL_REACH_PX = 3
BALL_NBHD_PX = 16
POSE_IMGSZ = 1280
POSE_CONF = 0.25
KP_CONF = 0.5
KP_DILATE_1080 = 8
QUANT_PX = (0.25, 1.0)
VALUE_BOUND_BITS = 16
REVIEW_SEED = "pointstream-k4-review"
REVIEW_FRAMES = 2
COLOURS = {"player": (60, 120, 250), "racket": (250, 200, 40), "ball": (250, 40, 200)}
SKELETON = ((5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (0, 1), (0, 2), (1, 3), (2, 4))


def safe(window_id: str) -> str:
    return window_id.replace("/", "__")


def review_frames(window_id: str, frames: int) -> list[int]:
    out: list[int] = []
    n = 0
    while len(out) < min(REVIEW_FRAMES, frames):
        t = int(hashlib.sha256(f"{REVIEW_SEED}:{window_id}:{n}".encode()).hexdigest(), 16) % frames
        if t not in out:
            out.append(t)
        n += 1
    return out


# ----------------------------------------------------------------- encodes (CPU, beside the GPU work)


def encode(encoder: str, source: Path, work: Path, window: dict[str, Any], frames: int, crf: int, cpus: list[int]) -> dict[str, Any]:
    from src.codecs import svtav1

    width, height = window["size"]
    work.mkdir(parents=True, exist_ok=True)
    stream = work / f"crf{crf}.ivf"
    fps = Fraction(window["fps"]).limit_denominator(1001)
    command = svtav1.encode_command(encoder, source, stream, width=width, height=height, fps=fps, frames=frames, crf=crf,
                                    preset=PRESET, full_range=False)
    timing = svtav1.run(command, 7200, cpus)
    data = stream.read_bytes()
    sizes = svtav1.ivf_frames(data)
    return {"crf": crf, "stream": str(stream), "stream_sha256": hashlib.sha256(data).hexdigest(), "payload_bytes": sum(sizes),
            "temporal_units": len(sizes), "kbps": sum(sizes) * 8 / (frames / float(window["fps"])) / 1000.0,
            "command": command, "encode_seconds": round(timing["wall"], 2), "cpus": cpus}


# ----------------------------------------------------------------- masks


def write_jpegs(rgb: np.ndarray, out: Path) -> Path:
    import cv2

    chunk = out / "chunk_000000"
    chunk.mkdir(parents=True, exist_ok=True)
    for t in range(rgb.shape[0]):
        small = cv2.resize(np.asarray(rgb[t]), SAM_SIZE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(chunk / f"{t:05d}.jpg"), small[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
    return chunk


def upsample(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import cv2

    width, height = size
    if mask.shape == (height, width):
        return mask.astype(bool)
    return cv2.resize(mask.astype(np.uint8) * 255, (width, height), interpolation=cv2.INTER_LINEAR) > 127


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    import cv2

    if px <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def instances(clip: Any) -> list[dict[str, list[tuple[int, np.ndarray, float]]]]:
    """Per frame, per class: (track, mask at SAM's size, score); `upsample` brings a mask to the window's size."""
    out = []
    for frame in clip.frames:
        by: dict[str, list[tuple[int, np.ndarray, float]]] = {c: [] for c in PROMPTS}
        for inst in frame:
            m = inst.mask().astype(bool)
            if m.any():
                by[inst.class_name].append((int(inst.track_id), m, float(inst.score)))
        out.append(by)
    return out


def ball_masks(window: dict[str, Any], found: list[dict[str, Any]], frames: int) -> tuple[list[np.ndarray], dict[str, Any]]:
    """The ball part per frame: TrackNet's label (the SAM ball mask reaching it, else a disk); SAM's masks elsewhere."""
    width, height = window["size"]
    out: list[np.ndarray] = []
    stats: Counter[str] = Counter()
    rows = tw.labels(window, frames) if window["dataset"] == "tracknet" else None
    for t in range(frames):
        sam = [upsample(m, (width, height)) for _, m, _ in found[t]["ball"]]
        if rows is None:
            mask = np.zeros((height, width), bool)
            for m in sam:
                mask |= m
            stats["sam_frames" if sam else "empty_frames"] += 1
            out.append(mask)
            continue
        vis, x, y = rows[t]
        if vis not in tw.VISIBLE:
            out.append(np.zeros((height, width), bool))
            stats["not_visible"] += 1
            continue
        near = tw.disk((height, width), x, y, BALL_REACH_PX)
        reached = [m for m in sam if (m & near).any()]
        if reached:
            out.append(reached[0])
            stats["label_sam"] += 1
        else:
            out.append(tw.disk((height, width), x, y, BALL_DISK_PX))
            stats["label_disk"] += 1
        stats["sam_other_balls"] += len(sam) - len(reached)
    return out, dict(stats)


# ----------------------------------------------------------------- pose


class Poses:
    """Ultralytics YOLO pose models, on the claimed GPU."""

    def __init__(self, models: list[tuple[str, str]]):
        import torch
        from ultralytics import YOLO

        if not torch.cuda.is_available():
            raise RuntimeError("YOLO pose needs the claimed GPU")
        self.models = {name: YOLO(path) for name, path in models}
        self.paths = dict(models)
        self.device = "cuda:0"

    def record(self) -> dict[str, Any]:
        import torch
        import ultralytics

        return {"ultralytics": ultralytics.__version__, "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
                "models": {n: {"path": p, "sha256": file_sha256(Path(p))} for n, p in self.paths.items()},
                "imgsz": POSE_IMGSZ, "conf": POSE_CONF, "on_cuda": {n: str(next(m.model.parameters()).device)
                                                                     for n, m in self.models.items()}}

    def run(self, name: str, rgb: np.ndarray, batch: int = 16, imgsz: int = POSE_IMGSZ) -> list[dict[str, np.ndarray]]:
        out = []
        model = self.models[name]
        for s in range(0, rgb.shape[0], batch):
            frames = [np.ascontiguousarray(np.asarray(rgb[t])[:, :, ::-1]) for t in range(s, min(s + batch, rgb.shape[0]))]
            for r in model.predict(frames, imgsz=imgsz, conf=POSE_CONF, classes=[0], device=self.device, verbose=False):
                boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
                kps = r.keypoints.data.cpu().numpy() if r.keypoints is not None else np.zeros((0, 17, 3))
                out.append({"boxes": boxes.astype(np.float32), "kps": kps.astype(np.float32)})
        return out


def pose_analysis(name: str, dets: list[dict[str, np.ndarray]], players: list[list[tuple[int, np.ndarray, float]]],
                  size: tuple[int, int], fps: float) -> dict[str, Any]:
    """Fit, keypoints inside the mask, stability and the first-difference symbols of one model on one window."""
    from experiments.visor.h1 import box_of, match_detections

    width, height = size
    grow = max(1, int(round(KP_DILATE_1080 * height / 1080)))
    rows = []
    tracks: dict[int, list[tuple[int, np.ndarray, float]]] = {}
    for t, (small_players, det) in enumerate(zip(players, dets)):
        frame_players = [(track, upsample(m, size), s) for track, m, s in small_players]
        masks = {str(track): m for track, m, _ in frame_players}
        pairs = match_detections(masks, det["boxes"]) if len(det["boxes"]) else {}
        for track, m, _ in frame_players:
            box = box_of(m)
            row: dict[str, Any] = {"t": t, "track": track, "mask_px": int(m.sum()),
                                   "mask_height": (box[3] - box[1]) / height if box else None, "fitted": str(track) in pairs}
            if row["fitted"]:
                kp = det["kps"][pairs[str(track)]]
                confident = kp[:, 2] >= KP_CONF
                grown = dilate(m, grow)
                xs = np.clip(np.round(kp[:, 0]).astype(int), 0, width - 1)
                ys = np.clip(np.round(kp[:, 1]).astype(int), 0, height - 1)
                inside = grown[ys, xs] & confident
                row.update(confident=int(confident.sum()), inside=float(inside.sum() / confident.sum()) if confident.any() else None,
                           box_height=float(det["boxes"][pairs[str(track)]][3] - det["boxes"][pairs[str(track)]][1]))
                tracks.setdefault(track, []).append((t, kp, row["box_height"]))
            rows.append(row)
    symbols: dict[str, Counter[int]] = {f"{q:g}": Counter() for q in QUANT_PX}
    visibility: Counter[int] = Counter()
    starts = 0
    accel = []
    for track, seq in tracks.items():
        for i, (t, kp, box_h) in enumerate(seq):
            visibility.update((kp[:, 2] >= KP_CONF).astype(int).tolist())
            continuous = i > 0 and seq[i - 1][0] == t - 1
            if not continuous:
                starts += 1
            else:
                prev = seq[i - 1][1]
                for q in QUANT_PX:
                    d = np.round(kp[:, :2] / q).astype(int) - np.round(prev[:, :2] / q).astype(int)
                    symbols[f"{q:g}"].update(d.ravel().tolist())
            if i >= 2 and seq[i - 2][0] == t - 2 and continuous:
                a, b = seq[i - 2][1], seq[i - 1][1]
                ok = (a[:, 2] >= KP_CONF) & (b[:, 2] >= KP_CONF) & (kp[:, 2] >= KP_CONF)
                if ok.any():
                    acc = np.linalg.norm(kp[ok, :2] - 2 * b[ok, :2] + a[ok, :2], axis=1)
                    accel.append(float(np.median(acc)) / max(box_h, 1.0))
    fitted = [r for r in rows if r["fitted"]]
    inside = [r["inside"] for r in fitted if r["inside"] is not None]
    return {
        "model": name, "player_frames": len(rows), "fitted": len(fitted),
        "fitted_share": len(fitted) / len(rows) if rows else None,
        "inside_median": float(np.median(inside)) if inside else None, "inside_mean": float(np.mean(inside)) if inside else None,
        "accel_box_per_frame2_median": float(np.median(accel)) if accel else None,
        "symbols": {q: {str(k): v for k, v in c.items()} for q, c in symbols.items()},
        "visibility_symbols": {str(k): v for k, v in visibility.items()}, "track_starts": starts,
        "fitted_frames": len(fitted), "fps": fps, "rows": rows,
    }


# ----------------------------------------------------------------- review


def draw_review(rgb: np.ndarray, labels_map: np.ndarray, poses: dict[str, dict[str, np.ndarray]], out: Path, title: str) -> None:
    import cv2

    canvas = rgb.copy()
    for p, name in enumerate(PARTS):
        if name in COLOURS:
            sel = labels_map == p
            canvas[sel] = (0.45 * canvas[sel] + 0.55 * np.array(COLOURS[name])).astype(np.uint8)
    for colour, (_, det) in zip(((255, 255, 255), (0, 255, 0)), sorted(poses.items())):
        for kp in det["kps"]:
            for a, b in SKELETON:
                if kp[a, 2] >= KP_CONF and kp[b, 2] >= KP_CONF:
                    cv2.line(canvas, (int(kp[a, 0]), int(kp[a, 1])), (int(kp[b, 0]), int(kp[b, 1])), colour, 2)
    cv2.putText(canvas, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas[:, :, ::-1])


# ----------------------------------------------------------------- per window


def process_window(args: argparse.Namespace, window: dict[str, Any], sources: dict[str, str], segmenter: Any, poses: Poses,
                   pool: ThreadPoolExecutor, cpu_sets: list[list[int]], scratch: Path, publish: Path) -> dict[str, Any]:
    import cv2

    began = time.time()
    width, height = window["size"]
    frames = min(int(args.frames), int(window["frames"]))
    work = scratch / "work" / safe(window["id"])
    work.mkdir(parents=True, exist_ok=True)
    source = Path(sources[window["source"]["staged"]])
    yuv = work / "source.yuv"
    converted = tw.convert(args.ffmpeg, window, source, yuv, frames)
    encodes: list[Future[dict[str, Any]]] = [pool.submit(encode, args.encoder, yuv, work / "streams", window, frames, crf, cpu_sets[i % len(cpu_sets)])
                                             for i, crf in enumerate(CRFS)]
    rgb = tw.rgb_frames(yuv, width, height, frames, work / "rgb.npy")
    import torch

    t0 = time.time()
    clip = segmenter.segment_frames(write_jpegs(rgb, work / "jpeg"), dict(PROMPTS), policy="offline_bidirectional", fps=float(window["fps"]))
    sam_seconds = time.time() - t0
    found = instances(clip)
    if len(found) != frames:
        raise RuntimeError(f"{window['id']}: SAM returned {len(found)} of {frames} frames")
    balls, ball_stats = ball_masks(window, found, frames)
    labels_maps = np.zeros((frames, height, width), np.uint8)
    nbhd = np.zeros((frames, height, width), np.uint8)
    for t in range(frames):
        lab = labels_maps[t]
        for _, m, _ in found[t]["player"]:
            lab[upsample(m, (width, height))] = 1
        for _, m, _ in found[t]["racket"]:
            lab[upsample(m, (width, height))] = 2
        lab[balls[t]] = 3
        if balls[t].any():
            nbhd[t][dilate(balls[t], BALL_NBHD_PX)] = 1
    shares = (np.stack([(labels_maps == p).mean((1, 2)) for p in range(len(PARTS))], 1)).mean(0)
    counts = {"parts": np.stack([tw.mi_counts(labels_maps[t], len(PARTS)) for t in range(frames)]),
              "nbhd": np.stack([tw.mi_counts(nbhd[t], len(NBHD_PARTS)) for t in range(frames)])}
    t0 = time.time()
    dets = {name: poses.run(name, rgb) for name in poses.models}
    pose_seconds = time.time() - t0
    pose = {name: pose_analysis(name, d, [f["player"] for f in found], (width, height), float(window["fps"])) for name, d in dets.items()}
    grey = []
    step = max(1, int(round(tw.CAMERA["step_s"] * float(window["fps"]))))
    for t in range(0, frames, step):
        small = cv2.resize(np.asarray(rgb[t]), tw.CAMERA["analysis"], interpolation=cv2.INTER_AREA)
        grey.append(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY))
    camera_pairs = tw.camera_series(grey, 1080.0 / tw.CAMERA["analysis"][1])
    for n, t in enumerate(review_frames(window["id"], frames)):
        draw_review(np.asarray(rgb[t]), labels_maps[t], {name: d[t] for name, d in dets.items()},
                    publish / safe(window["id"]) / f"review-{n}-t{t:03d}.png", f"{window['id']} t={t}")
    coded = [f.result() for f in encodes]
    seconds_of_video = frames / float(window["fps"])
    bits: dict[str, Any] = {}
    inspect_futures = {c["crf"]: pool.submit(tw.stream_bits, args.inspect, Path(c["stream"]), counts, frames) for c in coded}
    for c in coded:
        acc = inspect_futures[c["crf"]].result()
        parts_bits = dict(zip(PARTS, acc["bits"]["parts"]))
        nb = dict(zip(NBHD_PARTS, acc["bits"]["nbhd"]))
        bits[str(c["crf"])] = {
            **{k: c[k] for k in ("stream_sha256", "payload_bytes", "temporal_units", "kbps", "command", "encode_seconds")},
            "accounted_bits": acc["accounted_bits"], "outside_window_bits": acc["outside_window_bits"],
            "display_frames": acc["display_frames"], "displayed_twice": acc["displayed_twice"], "frames_decoded": acc["frames_decoded"],
            "bits_by_part": parts_bits, "kbps_by_part": {k: v / seconds_of_video / 1000.0 for k, v in parts_bits.items()},
            "ball_neighbourhood_kbps": nb["ball_neighbourhood"] / seconds_of_video / 1000.0,
            "player_bits_by_frame": [round(r[1], 1) for r in acc["bits_by_frame"]["parts"]],
            "inspect_seconds": acc["seconds"],
        }
    per_frame_counts = {c: [len(found[t][c]) for t in range(frames)] for c in PROMPTS}
    row = {
        "id": window["id"], "dataset": window["dataset"], "size": [width, height], "fps": window["fps"], "frames": frames,
        "convert": converted, "pixel_share": dict(zip(PARTS, (float(v) for v in shares))),
        "sam": {"seconds": round(sam_seconds, 2), "frames_per_s": round(frames / max(sam_seconds, 1e-9), 3),
                "instances_per_frame": {c: {"mean": float(np.mean(v)), "max": int(max(v)), "frames_with": int(sum(x > 0 for x in v))}
                                        for c, v in per_frame_counts.items()},
                "player_tracks": sorted({tr for f in found for tr, _, _ in f["player"]})},
        "ball": ball_stats, "pose": pose, "pose_seconds": round(pose_seconds, 2),
        "camera": {"pairs": camera_pairs, "summary": tw.camera_summary(camera_pairs, step / float(window["fps"]))},
        "bits": bits, "seconds": round(time.time() - began, 2), "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    if window["dataset"] == "tracknet":
        row["trajectory_symbols"] = tw.trajectory_symbols(tw.labels(window, frames))
    shutil.rmtree(work, ignore_errors=True)
    return row


def command_run(args: argparse.Namespace) -> int:
    spec = json.loads(Path(args.windows).read_text())
    by_id = {w["id"]: w for w in spec["windows"]}
    wanted = [w["id"] for w in spec["windows"]] if args.select == "all" else args.select.split(",")
    missing = [w for w in wanted if w not in by_id]
    if missing:
        raise SystemExit(f"unknown windows: {missing}")
    sources = dict(args.source)
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(scratch / "ultralytics"))
    publish = scratch / "publish" / "windows"
    publish.mkdir(parents=True, exist_ok=True)
    # The staging cache keeps files read-only and drops the execute bit: run a copy (as H1).
    inspect = scratch / "bin" / Path(args.inspect).name
    inspect.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.inspect, inspect)
    inspect.chmod(0o755)
    args.inspect_source, args.inspect = args.inspect, str(inspect)
    restored = []
    checkpoints = os.environ.get("PS_CHECKPOINT_DIR")
    if checkpoints and Path(checkpoints).is_dir():
        for path in sorted(Path(checkpoints).glob("window--*.tar")):
            with tarfile.open(path) as tar:
                tar.extractall(publish, filter="data")
            restored.append(path.name[len("window--"):-4])
    todo = [w for w in wanted if safe(w) not in restored]
    import torch

    from src.segmentation.sam31 import Sam31SequenceSegmenter
    from src.segmentation.sam31_tracker import runtime

    began = time.time()
    segmenter = Sam31SequenceSegmenter(checkpoint_path=args.sam_checkpoint)
    provenance = segmenter.provenance("offline_bidirectional")
    sam = {"load_seconds": round(time.time() - began, 2), "runtime": runtime(), "sdpa_backend_policy": segmenter.sdpa_backend_policy,
           "model_revision": provenance.model_revision, "checkpoint_sha256": provenance.checkpoint_sha256,
           "config_sha256": provenance.config_sha256, "prob_threshold": segmenter.prob_threshold, "prompts": PROMPTS}
    poses = Poses([(n, p) for n, p in args.pose])
    threads = int(os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 16)
    cpus = sorted(os.sched_getaffinity(0))[:threads]  # type: ignore[attr-defined,unused-ignore]
    per = max(1, len(cpus) // len(CRFS))
    cpu_sets = [cpus[i * per:(i + 1) * per] for i in range(len(CRFS))]
    done = len(wanted) - len(todo)
    progress(done)
    with ThreadPoolExecutor(max_workers=len(CRFS)) as pool:
        for window_id in todo:
            torch.cuda.reset_peak_memory_stats()
            row = process_window(args, by_id[window_id], sources, segmenter, poses, pool, cpu_sets, scratch, publish)
            body = json.dumps(row, sort_keys=True, default=str).encode()
            (publish / f"{safe(window_id)}.json").write_bytes(body)
            if checkpoints:
                from experiments.jobs.monitor import save_checkpoint

                buffer = scratch / f"{safe(window_id)}.tar"
                with tarfile.open(buffer, "w") as tar:
                    tar.add(publish / f"{safe(window_id)}.json", arcname=f"{safe(window_id)}.json")
                    if (publish / safe(window_id)).is_dir():
                        tar.add(publish / safe(window_id), arcname=safe(window_id))
                save_checkpoint(f"window--{safe(window_id)}.tar", buffer.read_bytes())
                buffer.unlink()
            done += 1
            progress(done)
            print(f"{window_id}: {row['seconds']} s (SAM {row['sam']['seconds']} s)", flush=True)
    from src.codecs import svtav1

    write_json(stage_dir() / "k4.json", {
        "stage": os.environ.get("PS_STAGE", "local"), "windows": wanted, "restored": restored,
        "windows_json": {"path": args.windows, "sha256": file_sha256(Path(args.windows))},
        "decision": DECISION, "sam": sam, "pose": poses.record(),
        "tools": {"encoder": svtav1.tool(args.encoder), "ffmpeg": svtav1.tool(args.ffmpeg),
                  "inspect": {"path": args.inspect_source, "sha256": file_sha256(Path(args.inspect))}},
        "settings": {"crfs": CRFS, "preset": PRESET, "frames": int(args.frames), "prompts": PROMPTS, "sam_size": SAM_SIZE,
                     "ball_disk_px": BALL_DISK_PX, "ball_reach_px": BALL_REACH_PX, "ball_neighbourhood_px": BALL_NBHD_PX,
                     "pose_imgsz": POSE_IMGSZ, "pose_conf": POSE_CONF, "kp_conf": KP_CONF, "kp_dilate_1080": KP_DILATE_1080,
                     "quant_px": QUANT_PX, "camera": tw.CAMERA, "cpu_sets": cpu_sets},
        "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    })
    return 0


# ----------------------------------------------------------------- camera on the whole match


def command_camera(args: argparse.Namespace) -> int:
    import cv2
    import torch

    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(scratch / "ultralytics"))
    began = time.time()
    width, height = tw.CAMERA["analysis"]
    rate = 1.0 / tw.CAMERA["step_s"]
    command = [args.ffmpeg, "-v", "error", "-nostdin", "-i", args.video, *(["-t", str(args.limit_seconds)] if float(args.limit_seconds) > 0 else []),
               "-vf", f"fps={rate:g},scale={width}:{height}:flags=area", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE)
    assert proc.stdout is not None
    poses = Poses([("yolo26x-pose", args.pose)])
    sift = cv2.SIFT.create(nfeatures=tw.CAMERA["sift_features"])
    frame_bytes = width * height * 3
    pairs: list[dict[str, Any]] = []
    wide: list[bool] = []
    tallest: list[float] = []
    prev = None
    batch: list[np.ndarray] = []

    def flush() -> None:
        if not batch:
            return
        dets = poses.run("yolo26x-pose", np.stack(batch), batch=len(batch), imgsz=960)
        for d in dets:
            h = float((d["boxes"][:, 3] - d["boxes"][:, 1]).max() / height) if len(d["boxes"]) else 0.0
            tallest.append(round(h, 4))
            wide.append(h <= tw.CAMERA["wide_person_height"])
        batch.clear()

    while True:
        data = proc.stdout.read(frame_bytes)
        if len(data) < frame_bytes:
            break
        rgb = np.frombuffer(data, np.uint8).reshape(height, width, 3)
        grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if prev is not None:
            pairs.append(tw.camera_pair(prev, grey, sift, 1080.0 / height))
        prev = grey
        batch.append(rgb.copy())
        if len(batch) == 32:
            flush()
            progress(len(wide))
    flush()
    if proc.wait():
        raise RuntimeError(f"ffmpeg failed ({proc.returncode})")
    for i, p in enumerate(pairs):
        p["wide"] = bool(wide[i] and wide[i + 1])
    summary = tw.camera_summary(pairs, tw.CAMERA["step_s"])
    n = max(len(pairs), 1)
    summary.update(still_and_wide=sum(p["kind"] == "still" and p["wide"] for p in pairs) / n,
                   wide=sum(p["wide"] for p in pairs) / n,
                   still_given_wide=(sum(p["kind"] == "still" and p["wide"] for p in pairs) / max(sum(p["wide"] for p in pairs), 1)),
                   frames=len(wide))
    publish = scratch / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    write_json(publish / "camera-pairs.json", {"pairs": pairs, "tallest_person": tallest})
    from src.codecs import svtav1

    write_json(stage_dir() / "k4-camera.json", {
        "stage": os.environ.get("PS_STAGE", "local"), "video": {"path": args.video, "sha256": file_sha256(Path(args.video))},
        "fps": args.fps, "limit_seconds": float(args.limit_seconds), "command": command, "summary": summary,
        "camera": tw.CAMERA, "pose": poses.record(), "ffmpeg": svtav1.tool(args.ffmpeg),
        "seconds": round(time.time() - began, 2), "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    })
    return 0


# ----------------------------------------------------------------- validate


def rows_from(directory: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))]


def rows_from_tar(path: Path) -> list[dict[str, Any]]:
    out = []
    with tarfile.open(path) as tar:
        for member in sorted((m for m in tar.getmembers() if m.name.startswith("publish/windows/") and m.name.endswith(".json")
                              and m.name.count("/") == 2), key=lambda m: m.name):
            handle = tar.extractfile(member)
            assert handle is not None
            out.append(json.loads(handle.read()))
    return out


def validate_rows(rows: list[dict[str, Any]], record: dict[str, Any]) -> dict[str, bool]:
    sam = record["sam"]
    frames_ok = all(sorted(b["display_frames"]) == list(range(r["frames"])) and not b["displayed_twice"]
                    for r in rows for b in r["bits"].values())
    return {
        "every_window_published": sorted(r["id"] for r in rows) == sorted(record["windows"]),
        "converted_every_frame": all(r["convert"]["frames"] == r["frames"] for r in rows),
        "sam_on_ada_or_a6000": any(n in sam["runtime"]["gpu"] for n in ("RTX 6000 Ada", "RTX A6000")),
        "sam_pinned_checkpoint": sam["checkpoint_sha256"] == "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6",
        "sam_native_attention": sam["sdpa_backend_policy"] != "efficient_then_math_fallback",
        "pose_models_on_cuda": all(v.startswith("cuda") for v in record["pose"]["on_cuda"].values()),
        "four_crfs_per_window": all(sorted(int(c) for c in r["bits"]) == list(CRFS) for r in rows),
        "every_frame_displayed_once": frames_ok,
        "accounting_covers_85pct_of_payload": all(b["accounted_bits"] >= 0.85 * b["payload_bytes"] * 8 for r in rows for b in r["bits"].values()),
        "parts_cover_every_pixel": all(abs(sum(r["pixel_share"].values()) - 1.0) < 1e-6 for r in rows),
        "players_found_on_half_the_frames": all(r["sam"]["instances_per_frame"]["player"]["frames_with"] >= 0.5 * r["frames"] for r in rows),
        "tracknet_ball_on_every_visible_frame": all(r["ball"].get("label_sam", 0) + r["ball"].get("label_disk", 0)
                                                    == sum(r["trajectory_symbols"]["visibility"]) for r in rows if r["dataset"] == "tracknet"),
        "bits_split_sum_to_accounted": all(abs(sum(b["bits_by_part"].values()) - (b["accounted_bits"] - b["outside_window_bits"]))
                                           <= 0.01 * b["accounted_bits"] + 1 for r in rows for b in r["bits"].values()),
        "pose_rows_cover_player_frames": all(p["player_frames"] == sum(1 for _ in p["rows"]) for r in rows for p in r["pose"].values()),
    }


def validate_camera(record: dict[str, Any]) -> dict[str, bool]:
    s = record["summary"]
    expected = record["limit_seconds"] * 2 if record["limit_seconds"] > 0 else None
    return {
        "pairs_measured": s["pairs"] > 0 and (expected is None or abs(s["frames"] - expected) <= 2),
        "kinds_sum_to_one": abs(s["still"] + s["moving"] + s["cut"] - 1.0) < 1e-6,
        "pose_on_cuda": all(v.startswith("cuda") for v in record["pose"]["on_cuda"].values()),
    }


def command_validate(args: argparse.Namespace) -> int:
    stage = stage_dir()
    if (stage / "k4-camera.json").exists():
        checks = validate_camera(json.loads((stage / "k4-camera.json").read_text()))
    else:
        record = json.loads((stage / "k4.json").read_text())
        publish = Path(os.environ.get("PS_SCRATCH_DIR") or stage / "scratch") / "publish" / "windows"
        rows = rows_from(publish) if publish.is_dir() else rows_from_tar(stage / "published.tar")
        checks = validate_rows(rows, record)
    report = {"passed": all(checks.values()), "checks": checks}
    target = os.environ.get("PS_VALIDATION_PATH")
    if target:
        write_json(Path(target), report)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


# ----------------------------------------------------------------- report


def histogram_entropy(counter: Counter[str]) -> float:
    n = sum(counter.values())
    return -sum(c / n * math.log2(c / n) for c in counter.values() if c) if n else 0.0


def pose_rate(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    """Pooled keypoint rate of one model over a dataset's windows: first-difference entropy per value, the visibility
    entropy, run starts at the 16-bit bound; kbps from the bits per fitted player-frame, the players per frame and
    the frame rate of each window."""
    out: dict[str, Any] = {}
    vis: Counter[str] = Counter()
    for r in rows:
        vis.update(r["pose"][model]["visibility_symbols"])
    vis_bits = histogram_entropy(vis)
    for q in (f"{v:g}" for v in QUANT_PX):
        pooled: Counter[str] = Counter()
        for r in rows:
            pooled.update(r["pose"][model]["symbols"][q])
        h = histogram_entropy(pooled)
        kbps = []
        for r in rows:
            p = r["pose"][model]
            if not p["fitted_frames"]:
                kbps.append(0.0)
                continue
            diffs = sum(p["symbols"][q].values())
            bits = h * diffs + vis_bits * 17 * p["fitted_frames"] + p["track_starts"] * 34 * VALUE_BOUND_BITS
            kbps.append(bits / (r["frames"] / float(r["fps"])) / 1000.0)
        out[q] = {"bits_per_value": h, "kbps_per_window": kbps, "kbps_mean": float(np.mean(kbps)) if kbps else None,
                  "bound_kbps_mean": float(np.mean([r["pose"][model]["fitted_frames"] * (34 * VALUE_BOUND_BITS + 17)
                                                    / (r["frames"] / float(r["fps"])) / 1000.0 for r in rows]))}
    out["visibility_bits_per_keypoint"] = vis_bits
    return out


def summarize(rows: list[dict[str, Any]], model: str) -> dict[str, Any]:
    def mean(values: list[float | None]) -> float | None:
        vals = [v for v in values if v is not None]
        return float(np.mean(vals)) if vals else None

    crfs = [str(c) for c in CRFS]
    fg_share = {c: {p: mean([r["bits"][c]["bits_by_part"][p] / max(sum(r["bits"][c]["bits_by_part"][q] for q in FOREGROUND), 1e-9)
                             for r in rows]) for p in FOREGROUND} for c in crfs}
    all_share = {c: {p: mean([r["bits"][c]["bits_by_part"][p] / max(sum(r["bits"][c]["bits_by_part"].values()), 1e-9) for r in rows])
                     for p in PARTS} for c in crfs}
    kbps = {c: {p: mean([r["bits"][c]["kbps_by_part"][p] for r in rows]) for p in PARTS} for c in crfs}
    total = {c: mean([r["bits"][c]["kbps"] for r in rows]) for c in crfs}
    nbhd = {c: mean([r["bits"][c]["ball_neighbourhood_kbps"] for r in rows]) for c in crfs}
    poses: dict[str, Any] = {m: {"fitted_share": mean([r["pose"][m]["fitted_share"] for r in rows]),
                 "inside_median": mean([r["pose"][m]["inside_median"] for r in rows]),
                 "accel": mean([r["pose"][m]["accel_box_per_frame2_median"] for r in rows]),
                 "rate": pose_rate(rows, m)} for m in rows[0]["pose"]}
    rate = poses[model]["rate"]["0.25"]["kbps_mean"]
    player_62 = kbps[str(DECISION["c_crf"])]["player"]
    return {"windows": len(rows), "pixel_share": {p: mean([r["pixel_share"][p] for r in rows]) for p in PARTS},
            "foreground_bit_share": fg_share, "bit_share": all_share, "kbps_by_part": kbps, "kbps_total": total,
            "ball_neighbourhood_kbps": nbhd, "pose": poses, "model": model,
            "rate_share_crf62": rate / player_62 if rate is not None and player_62 else None,
            "camera_still": mean([r["camera"]["summary"]["still"] for r in rows]),
            "instances_per_frame": {c: mean([r["sam"]["instances_per_frame"][c]["mean"] for r in rows]) for c in PROMPTS}}


def decide(summary: dict[str, Any]) -> dict[str, Any]:
    m = summary["model"]
    pose = summary["pose"][m]
    a = all((summary["foreground_bit_share"][str(c)]["player"] or 0) >= DECISION["a_min_foreground_bit_share"] for c in DECISION["a_crfs"])
    b = (pose["fitted_share"] or 0) >= DECISION["b_min_fitted_share"] and (pose["inside_median"] or 0) >= DECISION["b_min_keypoints_inside_median"]
    share = summary["rate_share_crf62"]
    c = share is not None and share <= DECISION["c_max_rate_share"]
    return {"a_bits": a, "b_fit": b, "c_rate": c, "passes_h1_screen": a and b and c, "rate_share_crf62": share,
            "k6_gate": a and b and share is not None and share <= DECISION["k6_max_rate_share"]}


def command_report(args: argparse.Namespace) -> int:
    rows = [r for p in args.published for r in rows_from_tar(Path(p))]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("a window appears twice")
    pilot = [r for r in rows if r["id"] in set(args.pilot_windows.split(","))] if args.pilot_windows else rows
    fit = {m: float(np.mean([r["pose"][m]["fitted_share"] or 0 for r in pilot])) for m in rows[0]["pose"]}
    inside = {m: float(np.mean([r["pose"][m]["inside_median"] or 0 for r in pilot])) for m in rows[0]["pose"]}
    model = max(fit, key=lambda m: (fit[m], inside[m]))
    by_dataset = {d: [r for r in rows if r["dataset"] == d] for d in sorted({r["dataset"] for r in rows})}
    summaries = {d: summarize(rs, model) for d, rs in by_dataset.items()}
    decisions = {d: decide(s) for d, s in summaries.items()}
    tn = [r for r in rows if r["dataset"] == "tracknet"]
    trajectory = tw.trajectory_rate([r["trajectory_symbols"] for r in tn], 30.0) if tn else None
    camera = json.loads(Path(args.camera).read_text()) if args.camera else None
    k6 = decisions.get(DECISION["k6_dataset"], {}).get("k6_gate")
    out = {"decision_rule": DECISION, "published": {p: file_sha256(Path(p)) for p in args.published},
           "windows": sorted(ids), "model": model, "model_choice": {"fitted_share": fit, "inside_median": inside},
           "summaries": summaries, "decisions": decisions, "ball_trajectory": trajectory,
           "camera_match": camera["summary"] if camera else None, "k6_runs": k6}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "k4-report.json", out)
    print(json.dumps({k: out[k] for k in ("model", "decisions", "ball_trajectory", "camera_match", "k6_runs")}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--windows", required=True)
    run.add_argument("--select", required=True)
    run.add_argument("--frames", required=True)
    run.add_argument("--source", nargs=2, action="append", required=True, metavar=("NAME", "PATH"))
    run.add_argument("--sam-checkpoint", required=True)
    run.add_argument("--pose", nargs=2, action="append", required=True, metavar=("NAME", "PT"))
    run.add_argument("--inspect", required=True)
    run.add_argument("--encoder", default="SvtAv1EncApp")
    run.add_argument("--ffmpeg", default="ffmpeg")
    cam = sub.add_parser("camera")
    cam.add_argument("--video", required=True)
    cam.add_argument("--fps", required=True, type=float)
    cam.add_argument("--limit-seconds", default="0")
    cam.add_argument("--pose", required=True)
    cam.add_argument("--ffmpeg", default="ffmpeg")
    sub.add_parser("validate")
    rep = sub.add_parser("report")
    rep.add_argument("--published", action="append", required=True)
    rep.add_argument("--camera")
    rep.add_argument("--pilot-windows", default="")
    rep.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return {"run": command_run, "camera": command_camera, "validate": command_validate, "report": command_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
