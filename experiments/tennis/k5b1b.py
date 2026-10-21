"""PLAN step K5b 1b: the per-frame ball (positions refined by the encoder, an appearance per frame, every byte billed),
from three position sources, judged by BallTrack's detection and DISTS on a crop around the ball.

    python -m experiments.tennis.k5b1b run --windows JSON --select ID,ID --frames N --source NAME PATH ... \\
        --image-ckpt PTH --image-sha256 H --video-ckpt PTH --video-sha256 H \\
        --wasb-code DIR --wasb-weights DIR --balltrack PT --balltrack-code DIR --dists-vgg PTH --lpips-alex PTH
    python -m experiments.tennis.k5b1b validate
    python -m experiments.tennis.k5b1b report --published TAR ... --out DIR

``run``, per TrackNet window (docs/experiments.md, 2026-10-21 K5b 1b): K5's coding (`k5.code_window`); the ball found
on the source by BallTrack and by WASB (step 1); per position source (`SOURCES`), renderer and decode the encoder's
per-frame fit (`src.pointstream.ball.fit_frames`), its payload (`encode_frames`) decoded back and drawn as the
decoder would, with the encoder's last check (off when the frame error rises); every variant judged on the GPU:
BallTrack's detections against the labels, DISTS and LPIPS on a `CROP` crop at the label, ball- and streak-region
PSNR, frame PSNR. ``report`` applies `DECISION`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import k5, k5b
from experiments.tennis import k5b_trackers as trk
from experiments.tennis import windows as tw
from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json
from src.pointstream import ball

#: The decision rule, fixed before any run (docs/experiments.md, K5b 1b entry).
DECISION: dict[str, Any] = {
    "decides_on": "balltrack", "min_windows": 6, "of_windows": 8,
    "detection_share": 0.8, "max_rate_ratio": 0.5, "max_dists_bd_rate": -0.25, "max_frame_bd_rate": 0.01,
    "overlap": {"dists": 0.01, "lpips": 0.01, "ball_psnr": 0.5, "streak_psnr": 0.5, "frame_psnr": 0.5},
}
SOURCES = ("labels", "balltrack", "wasb")
RENDERERS = k5b.RENDERERS
DRAWN = tuple(f"{r}@{s}" for s in SOURCES for r in RENDERERS)
VARIANTS = ("decode", "oracle", *DRAWN)
#: Judges whose quality rises with the number (DISTS and LPIPS enter as 1 - value).
JUDGES = ("dists", "lpips", "ball_psnr", "streak_psnr", "frame_psnr")
CROP = 64
SBS_POINTS = (("dcvc-uf-htl", 42), ("svt-av1", 55))
SBS_SOURCE = "balltrack"
JUDGE_BATCH = 64
SQ_CHUNK = 40


# ----------------------------------------------------------------- labels and tracks


def label_kinds(rows: list[tuple[int, int, int]]) -> list[tuple[int, float, float]]:
    """TrackNet rows as the judge reads them: 1 visible (1, 2), 0 no ball (0), -1 excluded (3, occluded)."""
    return [(1 if v in tw.VISIBLE else 0 if v == 0 else -1, float(x), float(y)) for v, x, y in rows]


def track_of(preds: list[trk.Prediction], shape: tuple[int, int]) -> ball.Track:
    h, w = shape
    return [(min(w - 1, max(0, int(round(x)))), min(h - 1, max(0, int(round(y))))) if v else None for v, x, y in preds]


def placement(track: ball.Track, kinds: list[tuple[int, float, float]]) -> dict[str, Any]:
    """How a track places the ball against the labels: shares of visible frames within 5 and 10 px, the median
    distance where both exist, and frames sent where the label has no ball."""
    d = [math.hypot(p[0] - x, p[1] - y) for p, (k, x, y) in zip(track, kinds) if k == 1 and p is not None]
    pos = sum(k == 1 for k, _, _ in kinds)
    return {"within_10px": sum(v <= 10 for v in d) / pos if pos else None, "within_5px": sum(v <= 5 for v in d) / pos if pos else None,
            "median_px": float(np.median(d)) if d else None, "sent": sum(p is not None for p in track),
            "sent_without_ball": sum(p is not None and k == 0 for p, (k, _, _) in zip(track, kinds))}


# ----------------------------------------------------------------- judges on the GPU


class Judges:
    """BallTrack, DISTS and LPIPS on the claimed GPU, and the label regions of one window."""

    def __init__(self, args: argparse.Namespace):
        import torch

        from src.codecs import quality

        self.device = "cuda"
        self.balltrack = trk.BallTrack(Path(args.balltrack_code), Path(args.balltrack))
        self.dists = quality.load_dists(Path(args.dists_vgg), self.device)
        self.lpips = quality.load_lpips(Path(args.lpips_alex), self.device)
        self.on_cuda = {"balltrack": str(next(self.balltrack.net.parameters()).device),
                        "dists": str(next(self.dists.parameters()).device), "lpips": str(next(self.lpips.parameters()).device)}
        self.torch = torch

    def prepare(self, source: Any, rows: list[tuple[int, int, int]]) -> None:
        """Per window: the source on the GPU, the label regions (ball and streak within `k5.REGION_PX`), the crops."""
        torch = self.torch
        self.kinds = label_kinds(rows)
        self.source = torch.from_numpy(np.ascontiguousarray(source)).to(self.device)
        n, h, w = self.source.shape[:3]
        track: ball.Track = [(x, y) if v in tw.VISIBLE else None for v, x, y in rows]
        vel = ball.velocities(track)
        self.visible = [t for t, p in enumerate(track) if p is not None]
        yy = torch.arange(h, device=self.device, dtype=torch.float32)[:, None]
        xx = torch.arange(w, device=self.device, dtype=torch.float32)[None]
        ball_m = torch.zeros((n, h, w), dtype=torch.bool, device=self.device)
        streak_m = torch.zeros_like(ball_m)
        for t in self.visible:
            x, y = track[t]  # type: ignore[misc]
            vx, vy = float(vel[t][0]), float(vel[t][1])
            dx, dy = xx - x, yy - y
            ball_m[t] = dx * dx + dy * dy <= k5.REGION_PX**2
            vv = vx * vx + vy * vy
            s = ((dx * vx + dy * vy) / vv).clamp(-0.5, 0.5) if vv > 0 else torch.zeros_like(dx * dy)
            streak_m[t] = torch.hypot(dx - s * vx, dy - s * vy) <= k5.REGION_PX
        self.ball_mask, self.streak_mask = ball_m, streak_m
        self.crop_origin = [(int(np.clip(track[t][0] - CROP // 2, 0, w - CROP)), int(np.clip(track[t][1] - CROP // 2, 0, h - CROP)))  # type: ignore[index]
                            for t in self.visible]
        self.source_crops = self.crops(self.source)
        self.paste = torch.zeros((n, h, w), dtype=torch.bool, device=self.device)
        for t in self.visible:
            x, y = track[t]  # type: ignore[misc]
            self.paste[t] = (xx - x) ** 2 + (yy - y) ** 2 <= k5.PASTE_PX**2

    def crops(self, frames: Any) -> Any:
        """(visible, 3, CROP, CROP) float in [0, 1] at each visible frame's label."""
        return self.torch.stack([frames[t, y:y + CROP, x:x + CROP] for t, (x, y) in zip(self.visible, self.crop_origin)]
                                ).permute(0, 3, 1, 2).float() / 255.0

    def score(self, frames: Any) -> dict[str, Any]:
        """Every judge of one variant's frames (on the GPU, uint8 RGB)."""
        torch = self.torch
        from src.codecs import quality

        n = frames.shape[0]
        frame_se = torch.zeros(n, dtype=torch.float64, device=self.device)
        region = {"ball_psnr": 0.0, "streak_psnr": 0.0}
        for b in range(0, n, SQ_CHUNK):
            diff = frames[b:b + SQ_CHUNK].to(torch.int32) - self.source[b:b + SQ_CHUNK].to(torch.int32)
            sq = (diff * diff).sum(-1).to(torch.float64)  # (chunk, h, w)
            frame_se[b:b + SQ_CHUNK] = sq.flatten(1).sum(1)
            region["ball_psnr"] += float((sq * self.ball_mask[b:b + SQ_CHUNK]).sum())
            region["streak_psnr"] += float((sq * self.streak_mask[b:b + SQ_CHUNK]).sum())
            del diff, sq
        frame_mse = frame_se / (frames[0].numel())
        out: dict[str, Any] = {"frame_psnr": k5.mse_to_psnr(float(frame_mse.mean())), "frame_mse": frame_mse.cpu().numpy()}
        for name, m in (("ball_psnr", self.ball_mask), ("streak_psnr", self.streak_mask)):
            count = float(m.sum()) * 3
            out[name] = k5.mse_to_psnr(region[name] / count) if count else None
        comp = self.crops(frames)
        d, lp = [], []
        with torch.inference_mode():
            for b in range(0, len(comp), JUDGE_BATCH):
                x, y = comp[b:b + JUDGE_BATCH], self.source_crops[b:b + JUDGE_BATCH]
                d.append(quality.masked_dists(self.dists, x, y, torch.ones_like(x[:, :1])))
                lp.append(self.lpips(x * 2 - 1, y * 2 - 1)[:, 0].flatten(1).mean(1))
        out["dists"] = 1.0 - float(torch.cat(d).mean()) if d else None
        out["lpips"] = 1.0 - float(torch.cat(lp).mean()) if lp else None
        preds = self.balltrack.run_gpu(frames)
        c = trk.counts(preds, self.kinds)
        out["detection"] = {"counts": c, **trk.rates(c)}
        return out


def composite(judges: Judges, decode: Any, decode_cpu: Any, track: ball.Track, appearances: list[ball.Appearance | None]) -> Any:
    """The decoder's frames on the GPU: the decode with the per-frame ball drawn (on the CPU, from the decoded
    payload), pasted region by region."""
    out = decode.clone()
    vel = ball.velocities(track)
    for t, (p, a) in enumerate(zip(track, appearances)):
        if p is None or a is None:
            continue
        got = ball.patch(decode_cpu[t], p, vel[t], a)
        if got is not None:
            (y0, y1, x0, x1), pixels = got
            out[t, y0:y1, x0:x1] = judges.torch.from_numpy(pixels).to(out.device)
    return out


# ----------------------------------------------------------------- per window


def fit_source(name: str, start: ball.Track, source: Any, decodes: list[Any], shape: tuple[int, int]) -> dict[str, Any]:
    began = time.time()
    crops = ball.FrameCrops.build(start, source, decodes)
    fits = {r: ball.fit_frames(crops, ball.grid(r), shape) for r in RENDERERS}
    return {"source": name, "fits": fits, "seconds": round(time.time() - began, 2)}


def score_window(args: argparse.Namespace, coded: dict[str, Any], trajectory: dict[str, Any], judges: Judges, zoo: trk.Zoo,
                 publish: Path) -> dict[str, Any]:
    began = time.time()
    torch = judges.torch
    window, frames, work = coded["window"], coded["frames"], coded["work"]
    width, height = window["size"]
    fps = float(window["fps"])
    seconds = frames / fps
    rows = tw.labels(window, frames)
    kinds = label_kinds(rows)
    source = tw.rgb_frames(coded["yuv"], width, height, frames, work / "source.npy")
    judges.prepare(source, rows)
    # Trackers on the source; BallTrack also through its CPU path, for the GPU path's agreement.
    t0 = time.time()
    found = {"balltrack": judges.balltrack.run_gpu(judges.source), "wasb": zoo.run("wasb-step1", np.asarray(source))}
    cpu_path = judges.balltrack.run(np.asarray(source))
    tracker_seconds = round(time.time() - t0, 2)
    agree = [a[0] == b[0] and (not a[0] or math.hypot(a[1] - b[1], a[2] - b[2]) <= 2.5) for a, b in zip(found["balltrack"], cpu_path)]
    starts: dict[str, ball.Track] = {"labels": [(x, y) if v in tw.VISIBLE else None for v, x, y in rows],
                                     **{k: track_of(v, (height, width)) for k, v in found.items()}}
    trackers = {k: {"counts": trk.counts(found[k], kinds), **trk.rates(trk.counts(found[k], kinds))} for k in found}
    trackers["balltrack_cpu_path"] = {**trk.rates(trk.counts(cpu_path, kinds)), "agreement": sum(agree) / len(agree)}
    # Decodes to RGB on disk, then the fits per source (threads; numpy releases the GIL in its kernels).
    paths = []
    for rec in coded["points"]:
        npy = work / f"decoded-{rec['codec']}-{rec['param']}.npy"
        tw.rgb_frames(Path(rec["decoded"]), width, height, frames, npy)
        Path(rec.pop("decoded")).unlink()
        paths.append(npy)
    decodes = [np.load(p, mmap_mode="r") for p in paths]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(SOURCES)) as pool:
        fitted = {s: f for s, f in zip(SOURCES, pool.map(lambda s: fit_source(s, starts[s], source, decodes, (height, width)), SOURCES))}
    fit_seconds = round(time.time() - t0, 2)
    reference = judges.score(judges.source)
    points = []
    sbs_frames: dict[tuple[str, int], dict[str, Any]] = {}
    for i, rec in enumerate(coded["points"]):
        dec = torch.from_numpy(np.ascontiguousarray(decodes[i])).to(judges.device)
        base = judges.score(dec)
        variants: dict[str, Any] = {"decode": {**base, "kbps": rec["kbps"]}}
        oracle = torch.where(judges.paste[..., None], judges.source, dec)
        variants["oracle"] = {**judges.score(oracle), "kbps": rec["kbps"] + trajectory["kbps"]}
        del oracle
        for src in SOURCES:
            for r in RENDERERS:
                fit = fitted[src]["fits"][r][i]
                payload = ball.encode_frames(fit["track"], fit["appearances"], width, height)
                track, apps = ball.decode_frames(payload, frames, width, height)
                roundtrip = (track, apps) == (fit["track"], fit["appearances"])
                comp = composite(judges, dec, decodes[i], track, apps)
                q = judges.score(comp)
                on = any(p is not None for p in track)
                if on and q["frame_psnr"] < base["frame_psnr"]:
                    # The encoder's last check, on the composite it would send: off when the frame error rises.
                    payload = ball.encode_frames([None] * frames, [None] * frames, width, height)
                    track, apps, on = [None] * frames, [None] * frames, False
                    comp, q = dec, dict(base)
                nbytes = k5b.segment_bytes(payload)
                q.update({"kbps": rec["kbps"] + nbytes * 8 / seconds / 1000.0, "ball_bytes": nbytes, "ball_kbps": nbytes * 8 / seconds / 1000.0,
                          "on": on, "roundtrip": roundtrip, "fit_delta_se": fit["delta_se"], "placement": placement(track, kinds),
                          "settings": _settings_summary(apps)})
                variants[f"{r}@{src}"] = q
                if (rec["codec"], rec["param"]) in SBS_POINTS and src == SBS_SOURCE:
                    sbs_frames.setdefault((rec["codec"], rec["param"]), {"decode": dec.cpu().numpy()})[r] = comp.cpu().numpy()
                del comp
        for v in variants.values():
            v.pop("frame_mse", None)
        rec["variants"] = variants
        points.append(rec)
        del dec
        torch.cuda.empty_cache()
    out_dir = publish / k5.safe(window["id"])
    sbs = {}
    for (codec, param), arrays in sbs_frames.items():
        rec = next(p for p in points if (p["codec"], p["param"]) == (codec, param))
        v = rec["variants"]
        name, setting = ("DCVC-UF", f"QP {param}") if codec == "dcvc-uf-htl" else ("SVT-AV1", f"CRF {param}")
        titles = ["source", f"{name} {setting}  {v['decode']['kbps']:.1f} kbps",
                  f"+ disk  {v[f'disk@{SBS_SOURCE}']['kbps']:.1f} kbps", f"+ streak  {v[f'streak@{SBS_SOURCE}']['kbps']:.1f} kbps"]
        tag = f"{'dcvc' if codec == 'dcvc-uf-htl' else 'svt'}{param}"
        sbs[tag] = side_by_side(args, source, [arrays["decode"], arrays["disk"], arrays["streak"]], starts["labels"], titles, fps,
                                out_dir, tag, full=(codec, param) == SBS_POINTS[0])
    del sbs_frames
    row = {"id": window["id"], "frames": frames, "fps": window["fps"], "size": [width, height], "convert": coded["convert"],
           "visible": len(judges.visible), "source_judged": {k: v for k, v in reference.items() if k != "frame_mse"},
           "trackers": trackers, "placement": {s: placement(starts[s], kinds) for s in SOURCES},
           "refined_placement": {f"{r}@{s}": [placement(f["track"], kinds) for f in fitted[s]["fits"][r]] for s in SOURCES for r in RENDERERS},
           "points": sorted(points, key=lambda p: (p["codec"], p["kbps"])), "trajectory": trajectory, "sbs": sbs,
           "coding_seconds": coded["coding_seconds"], "tracker_seconds": tracker_seconds, "fit_seconds": fit_seconds,
           "fit_seconds_per_source": {s: fitted[s]["seconds"] for s in SOURCES},
           "seconds": round(coded["coding_seconds"] + time.time() - began, 2)}
    judges.source = None
    decodes.clear()
    shutil.rmtree(work, ignore_errors=True)
    return row


def _settings_summary(apps: list[ball.Appearance | None]) -> dict[str, Any] | None:
    shown = [a for a in apps if a is not None]
    if not shown:
        return None
    return {"frames": len(shown), "median_radius": float(np.median([ball.RADII[a.radius] for a in shown])),
            "median_exposure": float(np.median([ball.EXPOSURES[a.exposure] for a in shown])),
            "median_colour": [int(np.median([a.colour[c] for a in shown])) for c in range(3)]}


def side_by_side(args: argparse.Namespace, source: Any, versions: list[np.ndarray], track: ball.Track, titles: list[str], fps: float,
                 out_dir: Path, tag: str, full: bool) -> dict[str, Any]:
    """Source | decode | + disk | + streak as a 2 x 2 grid, following the labelled ball (K5b 1's layout)."""
    import cv2

    h, w = source.shape[1:3]
    origins = k5b.follow_centres(track, (h, w))
    fw, fh = k5b.FOLLOW
    z = k5b.FOLLOW_ZOOM

    def panels(t: int) -> list[np.ndarray]:
        return [np.asarray(source[t]), *(v[t] for v in versions)]

    def follow() -> Any:
        for t in range(source.shape[0]):
            x0, y0 = origins[t]
            tiles = [k5b.label(cv2.resize(np.ascontiguousarray(p[y0:y0 + fh, x0:x0 + fw]), (fw * z, fh * z), interpolation=cv2.INTER_NEAREST), s)
                     for p, s in zip(panels(t), titles)]
            yield np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0)

    clips = {"follow": k5b.write_clip(args.ffmpeg, follow(), (2 * fw * z, 2 * fh * z), fps, k5b.X264_CRF["follow"], out_dir / f"sbs-{tag}-follow.mp4")}
    if full:
        def whole() -> Any:
            for t in range(source.shape[0]):
                tiles = [k5b.label(p, s) for p, s in zip(panels(t), titles)]
                yield np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0)

        clips["full"] = k5b.write_clip(args.ffmpeg, whole(), (2 * w, 2 * h), fps, k5b.X264_CRF["full"], out_dir / f"sbs-{tag}-full.mp4")
    return {"titles": titles, **clips}


def command_run(args: argparse.Namespace) -> int:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("K5b 1b needs the claimed GPU")
    spec = json.loads(Path(args.windows).read_text())
    by_id = {w["id"]: w for w in spec["windows"]}
    wanted = args.select.split(",")
    if any(w not in by_id or by_id[w]["dataset"] != "tracknet" for w in wanted):
        raise SystemExit("K5b 1b runs on TrackNet windows of windows.json")
    all_tn = [w for w in spec["windows"] if w["dataset"] == "tracknet"]
    trajectory = tw.trajectory_rate([tw.trajectory_symbols(tw.labels(w, int(w["frames"]))) for w in all_tn], 30.0)
    trajectory["windows"] = [w["id"] for w in all_tn]
    sources = dict(args.source)
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish" / "windows"
    publish.mkdir(parents=True, exist_ok=True)
    restored = []
    checkpoints = os.environ.get("PS_CHECKPOINT_DIR")
    if checkpoints and Path(checkpoints).is_dir():
        for path in sorted(Path(checkpoints).glob("window--*.tar")):
            with tarfile.open(path) as tar:
                tar.extractall(publish, filter="data")
            restored.append(path.name[len("window--"):-4])
    todo = [w for w in wanted if k5.safe(w) not in restored]
    judges = Judges(args)
    zoo = trk.Zoo(Path(args.wasb_code), Path(args.wasb_weights), names=("wasb-step1",))
    on_cuda = {**judges.on_cuda, "wasb": str(next(zoo.settings["wasb-step1"]["net"].parameters()).device)}
    threads = int(os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 16)
    cpus = sorted(os.sched_getaffinity(0))[:threads]  # type: ignore[attr-defined,unused-ignore]
    per = max(1, len(cpus) // len(k5.CRFS))
    cpu_sets = [cpus[i * per:(i + 1) * per] for i in range(len(k5.CRFS))]
    done = len(wanted) - len(todo)
    progress(done)
    with ThreadPoolExecutor(max_workers=len(k5.CRFS)) as pool:
        for w in todo:
            torch.cuda.reset_peak_memory_stats()
            coded = k5.code_window(args, by_id[w], sources, pool, cpu_sets, scratch)
            row = score_window(args, coded, trajectory, judges, zoo, publish)
            row["gpu_peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
            (publish / f"{k5.safe(w)}.json").write_bytes(json.dumps(row, sort_keys=True, default=str).encode())
            if checkpoints:
                from experiments.jobs.monitor import save_checkpoint

                buffer = scratch / f"{k5.safe(w)}.tar"
                with tarfile.open(buffer, "w") as tar:
                    tar.add(publish / f"{k5.safe(w)}.json", arcname=f"{k5.safe(w)}.json")
                    if (publish / k5.safe(w)).is_dir():
                        tar.add(publish / k5.safe(w), arcname=k5.safe(w))
                save_checkpoint(f"window--{k5.safe(w)}.tar", buffer.read_bytes())
                buffer.unlink()
            done += 1
            progress(done)
            print(f"{w}: {row['seconds']} s (trackers {row['tracker_seconds']} s, fit {row['fit_seconds']} s)", flush=True)
    from src.codecs import quality, svtav1

    write_json(stage_dir() / "k5b1b.json", {
        "stage": os.environ.get("PS_STAGE", "local"), "windows": wanted, "restored": restored,
        "windows_json": {"path": args.windows, "sha256": file_sha256(Path(args.windows))}, "decision": DECISION, "trajectory": trajectory,
        "tools": {"encoder": svtav1.tool(args.encoder), "decoder": svtav1.tool(args.decoder), "ffmpeg": svtav1.tool(args.ffmpeg),
                  "dcvc": {"tree": str(Path(sys.prefix) / "opt" / "DCVC"), "image_sha256": args.image_sha256, "video_sha256": args.video_sha256}},
        "judges": {"dists": quality.dists_record(Path(args.dists_vgg)), "lpips": quality.lpips_record(Path(args.lpips_alex)),
                   "balltrack": {"path": args.balltrack, "sha256": file_sha256(Path(args.balltrack)), "meta": judges.balltrack.meta}},
        "wasb": {"code": args.wasb_code, "weights": file_sha256(Path(args.wasb_weights) / trk.ZOO["wasb-step1"]["weights"]),
                 "setting": trk.ZOO["wasb-step1"]},
        "on_cuda": on_cuda, "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
        "settings": {"crfs": k5.CRFS, "preset": k5.PRESET, "qps": k5.QPS, "paste_px": k5.PASTE_PX, "region_px": k5.REGION_PX, "crop": CROP,
                     "hit_px": trk.HIT_PX, "shift": ball.SHIFT, "rounds": ball.ROUNDS, "min_gain": ball.MIN_GAIN, "radii": ball.RADII,
                     "exposures": ball.EXPOSURES, "phases": ball.PHASES, "samples": ball.SAMPLES, "sbs_points": SBS_POINTS,
                     "sbs_source": SBS_SOURCE, "frames": int(args.frames), "cpu_sets": cpu_sets},
    })
    return 0


# ----------------------------------------------------------------- validate


def validate_rows(rows: list[dict[str, Any]], record: dict[str, Any]) -> dict[str, bool]:
    pts = [p for r in rows for p in r["points"]]
    dcvc = [p for p in pts if p["codec"] == "dcvc-uf-htl"]
    drawn = [(p, p["variants"][k]) for p in pts for k in DRAWN]
    return {
        "every_window_published": sorted(r["id"] for r in rows) == sorted(record["windows"]),
        "converted_every_frame": all(r["convert"]["frames"] == r["frames"] for r in rows),
        "every_point_coded": all(sorted((p["codec"], p["param"]) for p in r["points"])
                                 == sorted([("dcvc-uf-htl", q) for q in k5.QPS] + [("svt-av1", c) for c in k5.CRFS]) for r in rows),
        "dcvc_decodes_deterministic": all(p["deterministic"] and p["decoder_matches_encoder_intra"] for p in dcvc),
        "dcvc_decodes_are_pictures": all(p["variants"]["decode"]["frame_psnr"] >= k5.SANE_FRAME_PSNR for p in dcvc),
        "models_on_cuda": all(v.startswith("cuda") for v in record["on_cuda"].values()) and set(record["on_cuda"]) == {"balltrack", "dists", "lpips", "wasb"},
        "every_variant_judged": all(set(p["variants"]) == set(VARIANTS) and all(
            p["variants"][k]["dists"] is not None and p["variants"][k]["detection"]["detection_rate"] is not None for k in VARIANTS) for p in pts),
        "judge_finds_the_source_ball": all(r["source_judged"]["detection"]["detection_rate"] >= 0.8 for r in rows),
        "source_judged_against_itself": all(abs(r["source_judged"]["dists"] - 1.0) < 1e-4 and r["source_judged"]["frame_psnr"] > 90 for r in rows),
        "balltrack_gpu_path_agrees": all(r["trackers"]["balltrack_cpu_path"]["agreement"] >= 0.97 for r in rows),
        "ball_payload_round_trips": all(v["roundtrip"] for _, v in drawn),
        "ball_bytes_billed": all(abs(v["kbps"] - v["ball_kbps"] - p["kbps"]) < 1e-9 and v["ball_bytes"] >= 3 for p, v in drawn),
        "frame_error_never_rises": all(v["frame_psnr"] >= p["variants"]["decode"]["frame_psnr"] - 1e-9 for p, v in drawn),
        "disk_is_sharp": all(v["settings"] is None or v["settings"]["median_exposure"] == 0 for p in pts for k, v in p["variants"].items()
                             if k.startswith("disk@")),
        "side_by_side_clips_written": all(set(r["sbs"]) == {"dcvc42", "svt55"} and all(c["follow"]["bytes"] > 0 for c in r["sbs"].values())
                                          for r in rows),
    }


def command_validate(args: argparse.Namespace) -> int:
    stage = stage_dir()
    record = json.loads((stage / "k5b1b.json").read_text())
    publish = Path(os.environ.get("PS_SCRATCH_DIR") or stage / "scratch") / "publish" / "windows"
    rows = k5.rows_from(publish) if publish.is_dir() else k5.rows_from_tar(stage / "published.tar")
    checks = validate_rows(rows, record)
    report = {"passed": all(checks.values()), "checks": checks}
    target = os.environ.get("PS_VALIDATION_PATH")
    if target:
        write_json(Path(target), report)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


# ----------------------------------------------------------------- report


def detection(v: dict[str, Any]) -> float:
    return float(v["detection"]["detection_rate"])


def hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """The upper envelope of (rate, quality): points no cheaper point beats, then the upper concave hull in log rate."""
    pts = sorted(points)
    front: list[tuple[float, float]] = []
    for r, q in pts:
        if not front or q > front[-1][1]:
            front.append((r, q))
    out: list[tuple[float, float]] = []
    for p in front:
        while len(out) >= 2:
            (r0, q0), (r1, q1) = out[-2], out[-1]
            if (q1 - q0) * (math.log(p[0]) - math.log(r0)) <= (p[1] - q0) * (math.log(r1) - math.log(r0)):
                out.pop()
            else:
                break
        out.append(p)
    return out


def rate_to_reach(points: list[tuple[float, float]], target: float) -> float | None:
    """The rate at which a curve (sorted by rate) first reaches ``target``, linear in log rate between points; the
    lowest point's rate when it already does; None when it never does."""
    pts = sorted(points)
    if not pts:
        return None
    if pts[0][1] >= target:
        return pts[0][0]
    for (r0, q0), (r1, q1) in zip(pts, pts[1:]):
        if q0 < target <= q1:
            w = (target - q0) / (q1 - q0)
            return math.exp(math.log(r0) + w * (math.log(r1) - math.log(r0)))
    return None


def detection_verdict(test: list[tuple[float, float]], anchor: list[tuple[float, float]], target: float) -> dict[str, Any]:
    r_test, r_anchor = rate_to_reach(test, target), rate_to_reach(hull(anchor), target)
    if r_test is None:
        return {"rate": None, "anchor_rate": r_anchor, "ratio": None, "passes": False}
    if r_anchor is None:
        return {"rate": r_test, "anchor_rate": None, "ratio": 0.0, "passes": True}
    ratio = r_test / r_anchor
    return {"rate": r_test, "anchor_rate": r_anchor, "ratio": ratio, "passes": ratio <= DECISION["max_rate_ratio"]}


def curve(points: list[dict[str, Any]], codec: str, variant: str, judge: str) -> list[tuple[float, float]]:
    return [(p["variants"][variant]["kbps"], detection(p["variants"][variant]) if judge == "detection" else p["variants"][variant][judge])
            for p in points if p["codec"] == codec]


def window_report(r: dict[str, Any]) -> dict[str, Any]:
    pts = r["points"]
    coders = sorted({p["codec"] for p in pts})
    target = DECISION["detection_share"] * r["source_judged"]["detection"]["detection_rate"]
    pooled_det = [pt for c in coders for pt in curve(pts, c, "decode", "detection")]
    out: dict[str, Any] = {"detection_target": target, "detection": {}, "passes": {}}
    for judge in JUDGES:
        anchors = {c: curve(pts, c, "decode", judge) for c in coders}
        anchors["pooled"] = [pt for c in coders for pt in anchors[c]]
        out[judge] = {f"{k}-{c}": {a: k5.verdict(curve(pts, c, k, judge), anchors[a], DECISION["overlap"][judge]) for a in (c, "pooled")}
                      for k in ("oracle", *DRAWN) for c in coders}
    for k in ("oracle", *DRAWN):
        for c in coders:
            key = f"{k}-{c}"
            det = detection_verdict(curve(pts, c, k, "detection"), pooled_det, target)
            out["detection"][key] = det
            dists_bd = out["dists"][key]["pooled"]["bd_rate"]
            frame_bd = out["frame_psnr"][key][c]["bd_rate"]
            out["passes"][key] = {"detection": det["passes"],
                                  "dists": dists_bd is not None and dists_bd <= DECISION["max_dists_bd_rate"],
                                  "frame": frame_bd is not None and frame_bd <= DECISION["max_frame_bd_rate"]}
    return out


def decide(per_window: dict[str, dict[str, Any]]) -> dict[str, Any]:
    keys = next(iter(per_window.values()))["passes"].keys()
    n = len(per_window)
    out = {}
    for key in keys:
        counts = {j: sum(w["passes"][key][j] for w in per_window.values()) for j in ("detection", "dists", "frame")}
        both = sum(all(w["passes"][key].values()) for w in per_window.values())
        out[key] = {**{f"{j}_windows": v for j, v in counts.items()}, "all_three_windows": both,
                    "passes": n >= DECISION["of_windows"] and both >= DECISION["min_windows"]}
    src = DECISION["decides_on"]
    adopted = {r: any(v["passes"] for k, v in out.items() if k.startswith(f"{r}@{src}-")) for r in RENDERERS}
    bound = {r: any(v["passes"] for k, v in out.items() if k.startswith(f"{r}@labels-")) for r in RENDERERS}
    return {"per_variant_coder": out, "adopted": adopted, "labels_pass": bound, "windows": n}


def command_report(args: argparse.Namespace) -> int:
    rows = [r for p in args.published for r in k5.rows_from_tar(Path(p))]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("a window appears twice")
    per_window = {r["id"]: window_report(r) for r in rows}
    decision = decide(per_window)
    means: dict[str, Any] = {}
    for codec, param in sorted({(p["codec"], p["param"]) for r in rows for p in r["points"]}):
        sel = [p for r in rows for p in r["points"] if (p["codec"], p["param"]) == (codec, param)]
        means[f"{codec}@{param}"] = {k: {**{j: float(np.mean([p["variants"][k][j] for p in sel])) for j in (*JUDGES, "kbps")},
                                         "detection": float(np.mean([detection(p["variants"][k]) for p in sel]))} for k in VARIANTS}
    placement_means = {k: {m: float(np.mean([r["placement"][k][m] for r in rows])) for m in ("within_10px", "within_5px")} for k in SOURCES}
    report = {"decision_rule": DECISION, "published": {p: file_sha256(Path(p)) for p in args.published}, "windows": sorted(ids),
              "decision": decision, "means": means, "placement": placement_means, "per_window": per_window}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "k5b1b-report.json", report)
    print(json.dumps({"adopted": decision["adopted"], "labels_pass": decision["labels_pass"]}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    for name in ("--windows", "--select", "--frames", "--image-ckpt", "--image-sha256", "--video-ckpt", "--video-sha256",
                 "--wasb-code", "--wasb-weights", "--balltrack", "--balltrack-code", "--dists-vgg", "--lpips-alex"):
        run.add_argument(name, required=True)
    run.add_argument("--source", nargs=2, action="append", required=True, metavar=("NAME", "PATH"))
    run.add_argument("--encoder", default="SvtAv1EncApp")
    run.add_argument("--decoder", default="dav1d")
    run.add_argument("--ffmpeg", default="ffmpeg")
    sub.add_parser("validate")
    rep = sub.add_parser("report")
    rep.add_argument("--published", action="append", required=True)
    rep.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return {"run": command_run, "validate": command_validate, "report": command_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
