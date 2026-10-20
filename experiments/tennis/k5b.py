"""PLAN step K5b (1): the ball renderer's oracle, a drawn ball (sharp disk or exposure streak) at the coded
positions over the pixel coders' decodes, every byte of the ball segment billed.

    python -m experiments.tennis.k5b run --windows JSON --select ID,ID --frames N --source NAME PATH ... \\
        --image-ckpt PTH --image-sha256 H --video-ckpt PTH --video-sha256 H
    python -m experiments.tennis.k5b validate
    python -m experiments.tennis.k5b report --published TAR ... --out DIR

``run``, per TrackNet window (docs/experiments.md, 2026-10-20 K5b 1): K5's coding (`k5.code_window`: SVT-AV1 at
`k5.CRFS`, DCVC-UF HT-L at `k5.QPS`, decoded from bytes); per decode the encoder's fit of each renderer
(`src.pointstream.ball.fit`), the composites (decode, K5's pixel oracle, disk, streak) scored on the visible
frames by ball-region PSNR (within `k5.REGION_PX` of the label), streak-region PSNR (within `REGION_PX` of the
label's displacement over one frame) and frame PSNR; the ball segment's bytes added to the rate. At `SBS_POINTS`
the side-by-side clips for the user's preference. ``report`` applies `DECISION`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import k5
from experiments.tennis import windows as tw
from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json
from src.pointstream import ball
from src.pointstream.bitstream import leb128

#: The decision rule, fixed before any run (docs/experiments.md, K5b 1 entry).
DECISION: dict[str, Any] = {
    "max_bd_rate": -0.25, "min_windows": 6, "of_windows": 8, "max_frame_bd_rate": 0.01,
    "overlap": {"ball_psnr": 0.5, "streak_psnr": 0.5, "frame_psnr": 0.5},
}
RENDERERS = ("disk", "streak")
VARIANTS = ("decode", "oracle", *RENDERERS)
JUDGES = ("ball_psnr", "streak_psnr", "frame_psnr")
#: Coder points with side-by-side clips (ball-following crop); the first also as a full-frame clip.
SBS_POINTS = (("dcvc-uf-htl", 21), ("dcvc-uf-htl", 42))
FOLLOW = (256, 144)
FOLLOW_ZOOM = 3
FOLLOW_SMOOTH = 7
X264_CRF = {"follow": 10, "full": 14}
REVIEW_CROP = 48


def segment_bytes(payload: bytes) -> int:
    """The BALL segment's size in the container: tag, LEB128 length, payload."""
    return 1 + len(leb128(len(payload))) + len(payload)


# ----------------------------------------------------------------- scoring on crops


def judge_masks(crops: ball.Crops) -> list[dict[str, np.ndarray]]:
    """Per crop group, the ball region (within REGION_PX of the label), the streak region (within REGION_PX of the
    segment the label's velocity sweeps over one frame, centred on it) and the pixel oracle's paste disk."""
    out = []
    for g in crops.groups:
        dx, dy, v = g["dx"][None], g["dy"][None], g["velocity"]
        vx, vy = v[:, 0, None, None], v[:, 1, None, None]
        vv = vx * vx + vy * vy
        s = np.clip(np.where(vv > 0, (dx * vx + dy * vy) / np.maximum(vv, 1e-12), 0.0), -0.5, 0.5)
        seg = np.hypot(dx - s * vx, dy - s * vy)
        r2 = dx * dx + dy * dy
        valid = g["valid"] > 0
        out.append({"ball": (r2 <= k5.REGION_PX**2) & valid, "streak": (seg <= k5.REGION_PX) & valid,
                    "paste": np.broadcast_to(r2 <= k5.PASTE_PX**2, valid.shape) & valid})
    return out


def variant_crops(kind: str, g: dict[str, Any], dec: np.ndarray, masks: dict[str, np.ndarray],
                  appearance: ball.Appearance | None) -> np.ndarray:
    """The variant's pixels in one crop group (float32 of uint8 values), as the decoder would show them."""
    if kind == "decode" or (kind in RENDERERS and appearance is None):
        return dec
    if kind == "oracle":
        return np.where(masks["paste"][..., None], g["source"], dec)
    assert appearance is not None
    radius, exposure, phase = appearance.values
    a = ball.coverage(g["dx"], g["dy"], g["velocity"], radius, exposure, phase)[..., None]
    return np.clip(np.round(dec * (1 - a) + np.asarray(appearance.colour, np.float32) * a), 0, 255)


def score_variant(crops: ball.Crops, masks: list[dict[str, np.ndarray]], dec: list[np.ndarray], frame_mse: np.ndarray,
                  kind: str, appearance: ball.Appearance | None, shape: tuple[int, int]) -> dict[str, Any]:
    h, w = shape
    se = {"ball": 0.0, "streak": 0.0}
    n = {"ball": 0.0, "streak": 0.0}
    mse = frame_mse.astype(np.float64).copy()
    for g, m, d in zip(crops.groups, masks, dec):
        comp = variant_crops(kind, g, d, m, appearance)
        err = (comp - g["source"]) ** 2
        base = (d - g["source"]) ** 2
        valid = g["valid"][..., None]
        delta = ((err - base) * valid).sum((1, 2, 3))
        mse[g["frames"]] += delta / (h * w * 3)
        for key in se:
            se[key] += float(err[m[key]].sum())
            n[key] += float(m[key].sum() * 3)
    return {"ball_psnr": k5.mse_to_psnr(se["ball"] / n["ball"]) if n["ball"] else None,
            "streak_psnr": k5.mse_to_psnr(se["streak"] / n["streak"]) if n["streak"] else None,
            "frame_psnr": k5.mse_to_psnr(float(mse.mean()))}


def frame_mse(decoded: Any, source: Any) -> np.ndarray:
    return np.array([float(((np.asarray(decoded[t]).astype(np.float32) - np.asarray(source[t]).astype(np.float32)) ** 2).mean())
                     for t in range(source.shape[0])])


# ----------------------------------------------------------------- side by side


def follow_centres(track: ball.Track, shape: tuple[int, int]) -> np.ndarray:
    """A smoothed ball-following crop origin per frame: positions interpolated over gaps, held at the ends,
    a centred moving average, the crop kept inside the frame."""
    h, w = shape
    seen = [t for t, p in enumerate(track) if p is not None]
    t_all = np.arange(len(track))
    if not seen:
        xs, ys = np.full(len(track), w / 2), np.full(len(track), h / 2)
    else:
        xs = np.interp(t_all, seen, [track[t][0] for t in seen])  # type: ignore[index]
        ys = np.interp(t_all, seen, [track[t][1] for t in seen])  # type: ignore[index]
    k = np.ones(2 * FOLLOW_SMOOTH + 1) / (2 * FOLLOW_SMOOTH + 1)
    xs = np.convolve(np.pad(xs, FOLLOW_SMOOTH, mode="edge"), k, "valid")
    ys = np.convolve(np.pad(ys, FOLLOW_SMOOTH, mode="edge"), k, "valid")
    fw, fh = FOLLOW
    x0 = np.clip(np.round(xs - fw / 2), 0, w - fw).astype(int)
    y0 = np.clip(np.round(ys - fh / 2), 0, h - fh).astype(int)
    return np.stack([x0, y0], 1)


def label(panel: np.ndarray, text: str) -> np.ndarray:
    import cv2

    out = np.array(panel, np.uint8, copy=True)  # frames may be read-only memmaps
    scale = max(0.5, panel.shape[0] / 720)
    (tw_, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(out, (0, 0), (tw_ + 12, th + base + 12), (0, 0, 0), -1)
    cv2.putText(out, text, (6, th + 6), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def write_clip(ffmpeg: str, frames: Any, size: tuple[int, int], fps: float, crf: int, out: Path) -> dict[str, Any]:
    width, height = size
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, "-v", "error", "-nostdin", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps:g}",
               "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               "-y", str(out)]
    proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    count = 0
    for frame in frames:
        proc.stdin.write(np.ascontiguousarray(frame, np.uint8).tobytes())
        count += 1
    proc.stdin.close()
    err = proc.stderr.read().decode() if proc.stderr else ""
    if proc.wait():
        raise RuntimeError(f"ffmpeg clip failed: {err[-2000:]}")
    return {"path": out.name, "frames": count, "bytes": out.stat().st_size, "crf": crf, "command": command[:-1] + [out.name]}


def side_by_side(args: argparse.Namespace, source: Any, decoded: Any, track: ball.Track, velocity: np.ndarray,
                 fits: dict[str, ball.Appearance | None], titles: list[str], fps: float, out_dir: Path, tag: str,
                 full: bool) -> dict[str, Any]:
    """Clips of source | decode | + disk | + streak, as a 2 x 2 grid: a ball-following crop enlarged
    `FOLLOW_ZOOM` times, and (``full``) the whole frames."""
    h, w = source.shape[1:3]

    def panels(t: int) -> list[np.ndarray]:
        d = np.asarray(decoded[t])
        out = [np.asarray(source[t]), d]
        for kind in RENDERERS:
            app, p = fits[kind], track[t]
            out.append(ball.draw(d, p, velocity[t], app) if app is not None and p is not None else d)
        return out

    origins = follow_centres(track, (h, w))
    fw, fh = FOLLOW

    def follow_frames() -> Any:
        import cv2

        for t in range(source.shape[0]):
            x0, y0 = origins[t]
            tiles = [label(cv2.resize(p[y0:y0 + fh, x0:x0 + fw], (fw * FOLLOW_ZOOM, fh * FOLLOW_ZOOM), interpolation=cv2.INTER_NEAREST), s)
                     for p, s in zip(panels(t), titles)]
            yield np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0)

    clips = {"follow": write_clip(args.ffmpeg, follow_frames(), (2 * fw * FOLLOW_ZOOM, 2 * fh * FOLLOW_ZOOM), fps,
                                  X264_CRF["follow"], out_dir / f"sbs-{tag}-follow.mp4")}
    if full:
        def full_frames() -> Any:
            for t in range(source.shape[0]):
                tiles = [label(p, s) for p, s in zip(panels(t), titles)]
                yield np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0)

        clips["full"] = write_clip(args.ffmpeg, full_frames(), (2 * w, 2 * h), fps, X264_CRF["full"], out_dir / f"sbs-{tag}-full.mp4")
    return {"titles": titles, **clips}


def review_strip(source: Any, decoded: Any, track: ball.Track, velocity: np.ndarray, fits: dict[str, ball.Appearance | None],
                 t: int, out: Path) -> dict[str, Any]:
    """Crops around the ball on frame ``t``: source, decode, K5's pixel oracle, disk, streak (4x, nearest)."""
    import cv2

    h, w = source.shape[1:3]
    p = track[t]
    assert p is not None
    s, d = np.asarray(source[t]), np.asarray(decoded[t])
    orc = d.copy()
    m = tw.disk((h, w), p[0], p[1], k5.PASTE_PX)
    orc[m] = s[m]
    versions = [s, d, orc] + [ball.draw(d, p, velocity[t], fits[k]) if fits[k] is not None else d for k in RENDERERS]  # type: ignore[arg-type]
    x0, y0 = int(np.clip(p[0] - REVIEW_CROP // 2, 0, w - REVIEW_CROP)), int(np.clip(p[1] - REVIEW_CROP // 2, 0, h - REVIEW_CROP))
    strip = np.concatenate([cv2.resize(v[y0:y0 + REVIEW_CROP, x0:x0 + REVIEW_CROP], (REVIEW_CROP * 4, REVIEW_CROP * 4),
                                       interpolation=cv2.INTER_NEAREST) for v in versions], 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), strip[:, :, ::-1])
    return {"t": t, "speed_px_per_frame": float(np.hypot(*velocity[t])), "order": ["source", "decode", "oracle", *RENDERERS], "path": out.name}


# ----------------------------------------------------------------- per window


def score_window(args: argparse.Namespace, coded: dict[str, Any], trajectory: dict[str, Any], publish: Path) -> dict[str, Any]:
    began = time.time()
    window, frames, work = coded["window"], coded["frames"], coded["work"]
    width, height = window["size"]
    fps = float(window["fps"])
    seconds = frames / fps
    rows = tw.labels(window, frames)
    track: ball.Track = [(x, y) if v in tw.VISIBLE else None for v, x, y in rows]
    velocity = ball.velocities(track)
    source = tw.rgb_frames(coded["yuv"], width, height, frames, work / "source.npy")
    crops = ball.Crops.build(track, source, velocity)
    masks = judge_masks(crops)
    decs, mses, kept = [], [], {}
    for rec in coded["points"]:
        npy = work / f"decoded-{rec['codec']}-{rec['param']}.npy"
        decoded = tw.rgb_frames(Path(rec["decoded"]), width, height, frames, npy)
        Path(rec.pop("decoded")).unlink()
        mses.append(frame_mse(decoded, source))
        decs.append([ball.crops_of(decoded, g["frames"], g["centres"], g["half"]) for g in crops.groups])
        if (rec["codec"], rec["param"]) in SBS_POINTS:
            kept[(rec["codec"], rec["param"])] = npy
        else:
            del decoded
            npy.unlink()
    fit_began = time.time()
    fits = {kind: ball.fit(crops, decs, ball.grid(kind)) for kind in RENDERERS}
    fit_seconds = round(time.time() - fit_began, 2)
    points = []
    for i, rec in enumerate(coded["points"]):
        variants: dict[str, Any] = {}
        for kind in VARIANTS:
            app = fits[kind][i]["appearance"] if kind in RENDERERS else None
            q = score_variant(crops, masks, decs[i], mses[i], kind, app, (height, width))
            if app is not None and q["frame_psnr"] < variants["decode"]["frame_psnr"]:
                # The encoder's last check, on the rounded composite it would send: off when it does not help.
                app = fits[kind][i]["appearance"] = None
                q = score_variant(crops, masks, decs[i], mses[i], kind, app, (height, width))
            if kind == "decode":
                q["kbps"] = rec["kbps"]
            elif kind == "oracle":
                q["kbps"] = rec["kbps"] + trajectory["kbps"]
            else:
                payload = ball.encode(track, app, width, height)
                back_track, back_app = ball.decode(payload, frames, width, height)
                nbytes = segment_bytes(payload)
                q.update({"kbps": rec["kbps"] + nbytes * 8 / seconds / 1000.0, "ball_bytes": nbytes,
                          "ball_kbps": nbytes * 8 / seconds / 1000.0, "on": app is not None,
                          "appearance": None if app is None else {"colour": list(app.colour), "radius": ball.RADII[app.radius],
                                                                  "exposure": ball.EXPOSURES[app.exposure], "phase": ball.PHASES[app.phase],
                                                                  "indices": [app.radius, app.exposure, app.phase]},
                          "delta_se": fits[kind][i]["delta_se"], "base_se": fits[kind][i]["base_se"],
                          "roundtrip": back_app == app and (app is None or back_track == track)})
            variants[kind] = q
        rec["variants"] = variants
        points.append(rec)
    out_dir = publish / k5.safe(window["id"])
    sbs = {}
    seen = [t for t, p in enumerate(track) if p is not None]
    fastest = max(seen, key=lambda t: float(np.hypot(*velocity[t]))) if seen else None
    review = None
    for (codec, param), npy in kept.items():
        i = next(j for j, r in enumerate(points) if (r["codec"], r["param"]) == (codec, param))
        rec = points[i]
        app = {kind: fits[kind][i]["appearance"] for kind in RENDERERS}
        decoded = np.load(npy, mmap_mode="r")
        v = rec["variants"]
        name = "DCVC-UF" if codec == "dcvc-uf-htl" else "SVT-AV1"
        setting = f"QP {param}" if codec == "dcvc-uf-htl" else f"CRF {param}"
        titles = ["source", f"{name} {setting}  {v['decode']['kbps']:.1f} kbps",
                  f"+ sharp disk  {v['disk']['kbps']:.1f} kbps", f"+ streak  {v['streak']['kbps']:.1f} kbps"]
        tag = f"{'dcvc' if codec == 'dcvc-uf-htl' else 'svt'}{param}"
        sbs[tag] = side_by_side(args, source, decoded, track, velocity, app, titles, fps, out_dir, tag, (codec, param) == SBS_POINTS[0])
        if (codec, param) == SBS_POINTS[0] and fastest is not None:
            review = review_strip(source, decoded, track, velocity, app, fastest, out_dir / f"review-fastest-{tag}.png")
        del decoded
        npy.unlink()
    import shutil

    row = {"id": window["id"], "frames": frames, "fps": window["fps"], "size": [width, height], "convert": coded["convert"],
           "visible": len(seen), "crop_groups": [{"half": g["half"], "frames": len(g["frames"])} for g in crops.groups],
           "speed_px_per_frame": {"median": float(np.median([np.hypot(*velocity[t]) for t in seen])) if seen else None,
                                  "max": float(np.hypot(*velocity[fastest])) if fastest is not None else None},
           "points": sorted(points, key=lambda p: (p["codec"], p["kbps"])), "trajectory": trajectory, "sbs": sbs,
           "review": review, "coding_seconds": coded["coding_seconds"], "fit_seconds": fit_seconds,
           "seconds": round(coded["coding_seconds"] + time.time() - began, 2)}
    del source
    shutil.rmtree(work, ignore_errors=True)
    return row


def command_run(args: argparse.Namespace) -> int:
    spec = json.loads(Path(args.windows).read_text())
    by_id = {w["id"]: w for w in spec["windows"]}
    wanted = args.select.split(",")
    if any(w not in by_id or by_id[w]["dataset"] != "tracknet" for w in wanted):
        raise SystemExit("K5b runs on TrackNet windows of windows.json")
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
    import torch

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
            row = score_window(args, coded, trajectory, publish)
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
            print(f"{w}: {row['seconds']} s (fit {row['fit_seconds']} s)", flush=True)
    from src.codecs import svtav1

    write_json(stage_dir() / "k5b.json", {
        "stage": os.environ.get("PS_STAGE", "local"), "windows": wanted, "restored": restored,
        "windows_json": {"path": args.windows, "sha256": file_sha256(Path(args.windows))}, "decision": DECISION,
        "trajectory": trajectory,
        "tools": {"encoder": svtav1.tool(args.encoder), "decoder": svtav1.tool(args.decoder), "ffmpeg": svtav1.tool(args.ffmpeg),
                  "dcvc": {"tree": str(Path(sys.prefix) / "opt" / "DCVC"), "image_sha256": args.image_sha256, "video_sha256": args.video_sha256}},
        "settings": {"crfs": k5.CRFS, "preset": k5.PRESET, "qps": k5.QPS, "paste_px": k5.PASTE_PX, "region_px": k5.REGION_PX,
                     "radii": ball.RADII, "exposures": ball.EXPOSURES, "phases": ball.PHASES, "samples": ball.SAMPLES,
                     "sbs_points": SBS_POINTS, "frames": int(args.frames), "cpu_sets": cpu_sets},
        "device": torch.cuda.get_device_name(0),
    })
    return 0


# ----------------------------------------------------------------- validate


def validate_rows(rows: list[dict[str, Any]], record: dict[str, Any]) -> dict[str, bool]:
    pts = [p for r in rows for p in r["points"]]
    dcvc = [p for p in pts if p["codec"] == "dcvc-uf-htl"]
    rend = [p["variants"][k] for p in pts for k in RENDERERS]
    return {
        "every_window_published": sorted(r["id"] for r in rows) == sorted(record["windows"]),
        "converted_every_frame": all(r["convert"]["frames"] == r["frames"] for r in rows),
        "every_point_coded": all(sorted((p["codec"], p["param"]) for p in r["points"])
                                 == sorted([("dcvc-uf-htl", q) for q in k5.QPS] + [("svt-av1", c) for c in k5.CRFS]) for r in rows),
        "dcvc_decodes_deterministic": all(p["deterministic"] and p["decoder_matches_encoder_intra"] for p in dcvc),
        "dcvc_decodes_are_pictures": all(p["variants"]["decode"]["frame_psnr"] >= k5.SANE_FRAME_PSNR for p in dcvc),
        "every_variant_scored": all(set(p["variants"]) == set(VARIANTS) and all(p["variants"][k]["ball_psnr"] is not None for k in VARIANTS) for p in pts),
        "ball_payload_round_trips": all(v["roundtrip"] for v in rend),
        "ball_bytes_billed": all(abs(v["kbps"] - v["ball_kbps"] - p["kbps"]) < 1e-9 and v["ball_bytes"] >= 3
                                 for p in pts for v in (p["variants"][k] for k in RENDERERS)),
        "fit_never_raises_the_frame_error": all(p["variants"][k]["frame_psnr"] >= p["variants"]["decode"]["frame_psnr"] - 1e-9
                                                for p in pts for k in RENDERERS),
        "disk_is_sharp": all(p["variants"]["disk"]["appearance"] is None or p["variants"]["disk"]["appearance"]["exposure"] == 0
                             for p in pts),
        "oracle_not_below_decode_on_ball": all(p["variants"]["oracle"]["ball_psnr"] >= p["variants"]["decode"]["ball_psnr"] - 1e-6 for p in pts),
        "side_by_side_clips_written": all(set(r["sbs"]) == {f"dcvc{q}" for _, q in SBS_POINTS} and all(c["follow"]["bytes"] > 0 for c in r["sbs"].values())
                                          for r in rows),
    }


def command_validate(args: argparse.Namespace) -> int:
    stage = stage_dir()
    record = json.loads((stage / "k5b.json").read_text())
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


def curve(points: list[dict[str, Any]], codec: str, variant: str, judge: str) -> list[tuple[float, float]]:
    return [(p["variants"][variant]["kbps"], p["variants"][variant][judge]) for p in points if p["codec"] == codec]


def passes(v: dict[str, Any], bar: float) -> bool:
    return v["bd_rate"] is not None and v["bd_rate"] <= bar


def window_report(r: dict[str, Any]) -> dict[str, Any]:
    pts = r["points"]
    coders = sorted({p["codec"] for p in pts})
    out: dict[str, Any] = {}
    for judge in JUDGES:
        overlap = DECISION["overlap"][judge]
        anchors = {c: curve(pts, c, "decode", judge) for c in coders}
        anchors["pooled"] = [pt for c in coders for pt in anchors[c]]
        out[judge] = {f"{kind}-{c}": {**{a: k5.verdict(curve(pts, c, kind, judge), anchors[a], overlap) for a in (c, "pooled")},
                                      **({"vs_oracle": k5.verdict(curve(pts, c, kind, judge), curve(pts, c, "oracle", judge), overlap)}
                                         if kind in RENDERERS else {})}
                      for kind in ("oracle", *RENDERERS) for c in coders}
    out["passes"] = {f"{kind}-{c}": {"ball": passes(out["ball_psnr"][f"{kind}-{c}"]["pooled"], DECISION["max_bd_rate"]),
                                     "frame": passes(out["frame_psnr"][f"{kind}-{c}"][c], DECISION["max_frame_bd_rate"])}
                     for kind in ("oracle", *RENDERERS) for c in coders}
    out["on"] = {f"{kind}-{p['codec']}@{p['param']}": p["variants"][kind]["on"] for kind in RENDERERS for p in pts}
    out["appearance"] = {f"{kind}-{p['codec']}@{p['param']}": p["variants"][kind]["appearance"] for kind in RENDERERS for p in pts}
    out["ball_kbps"] = {f"{kind}-{p['codec']}@{p['param']}": p["variants"][kind]["ball_kbps"] for kind in RENDERERS for p in pts}
    return out


def decide(per_window: dict[str, dict[str, Any]]) -> dict[str, Any]:
    keys = next(iter(per_window.values()))["passes"].keys()
    n = len(per_window)
    out = {}
    for key in keys:
        ball_n = sum(w["passes"][key]["ball"] for w in per_window.values())
        frame_n = sum(w["passes"][key]["frame"] for w in per_window.values())
        out[key] = {"ball_windows": ball_n, "frame_windows": frame_n,
                    "adopted": n >= DECISION["of_windows"] and ball_n >= DECISION["min_windows"] and frame_n >= DECISION["min_windows"]}
    adopted = {kind: any(v["adopted"] for k, v in out.items() if k.startswith(f"{kind}-")) for kind in RENDERERS}
    return {"per_renderer_coder": out, "adopted": adopted}


def command_report(args: argparse.Namespace) -> int:
    rows = [r for p in args.published for r in k5.rows_from_tar(Path(p))]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("a window appears twice")
    per_window = {r["id"]: window_report(r) for r in rows}
    decision = decide(per_window)
    summary: dict[str, Any] = {}
    for judge in JUDGES:
        for key in per_window[ids[0]][judge]:
            for anchor in per_window[ids[0]][judge][key]:
                vals = [w[judge][key][anchor]["bd_rate"] for w in per_window.values()]
                finite = [v for v in vals if v is not None and math.isfinite(v)]
                summary.setdefault(judge, {}).setdefault(key, {})[anchor] = {
                    "median": float(np.median(finite)) if finite else None, "min": min(finite) if finite else None,
                    "max": max(finite) if finite else None, "better_every_rate": sum(v == -math.inf for v in vals if v is not None),
                    "worse_every_rate": sum(v == math.inf for v in vals if v is not None), "unclear": sum(v is None for v in vals)}
    means: dict[str, Any] = {}
    for codec, param in sorted({(p["codec"], p["param"]) for r in rows for p in r["points"]}):
        sel = [p for r in rows for p in r["points"] if (p["codec"], p["param"]) == (codec, param)]
        means[f"{codec}@{param}"] = {kind: {j: float(np.mean([p["variants"][kind][j] for p in sel])) for j in (*JUDGES, "kbps")}
                                     for kind in VARIANTS}
    report = {"decision_rule": DECISION, "published": {p: file_sha256(Path(p)) for p in args.published}, "windows": sorted(ids),
              "decision": decision, "summary": summary, "means": means, "per_window": per_window}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "k5b-report.json", report)
    print(json.dumps(decision, indent=1, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--windows", required=True)
    run.add_argument("--select", required=True)
    run.add_argument("--frames", required=True)
    run.add_argument("--source", nargs=2, action="append", required=True, metavar=("NAME", "PATH"))
    run.add_argument("--image-ckpt", required=True)
    run.add_argument("--image-sha256", required=True)
    run.add_argument("--video-ckpt", required=True)
    run.add_argument("--video-sha256", required=True)
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
