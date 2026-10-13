"""PLAN step K5: the ball oracle, the true ball drawn at its coded position over the pixel coders' decodes.

    python -m experiments.tennis.k5 run --windows JSON --select ID,ID --frames N --source NAME PATH ... \\
        --image-ckpt PTH --image-sha256 H --video-ckpt PTH --video-sha256 H --detector NAME PT ...
    python -m experiments.tennis.k5 validate
    python -m experiments.tennis.k5 report --published TAR ... --pilot-windows ID,ID --out DIR

``run``, per TrackNet window (docs/experiments.md, 2026-10-13 K5): the window as K4 converts it; SVT-AV1 preset 4 at
`CRFS` (encoded and decoded on the CPU while the GPU works) and DCVC-UF HT-L at `QPS` (B2's worker, decoded from the
bytes alone); per decode the oracle composite (the source's pixels within `PASTE_PX` of the labelled position on
every visible frame); scored on the visible frames by ball-region PSNR (within `REGION_PX`), frame PSNR and each
detector's recall (COCO "sports ball", hit within `HIT_PX`). The oracle's rate adds the ball trajectory's rate,
pooled over every TrackNet window of ``windows.json`` (`experiments.tennis.windows.trajectory_rate`).
``report`` applies `DECISION`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import windows as tw
from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json

#: The decision rule, fixed before any run (docs/experiments.md, K5 entry).
DECISION: dict[str, Any] = {
    "max_bd_rate": -0.25, "min_windows": 6, "of_windows": 8, "detector_min_source_recall": 0.5,
    "overlap": {"ball_psnr": 0.5, "recall": 0.05}, "collapse_share": 0.90,
}
CRFS = (35, 41, 48, 55, 62)
PRESET = 4
QPS = (0, 21, 42, 63)
PASTE_PX = 4.0
REGION_PX = 8.0
HIT_PX = (10.0, 5.0)
DET_IMGSZ = 1280
DET_CONF = 0.1
SPORTS_BALL = 32
REVIEW_CROP = 48
#: The environment as the fleet started the stage (its GPU claim a UUID), before anything this process imports can
#: change it: ultralytics rewrites CUDA_VISIBLE_DEVICES when it selects a device. DCVC-UF's worker gets this one.
STAGE_ENV = dict(os.environ)
#: A DCVC-UF decode below this frame PSNR is not a picture of the source (the K5 smoke, 2026-10-13: 8-14 dB).
SANE_FRAME_PSNR = 20.0


def safe(window_id: str) -> str:
    return window_id.replace("/", "__")


def mse_to_psnr(mse: float) -> float:
    return 100.0 if mse <= 0 else min(100.0, 10.0 * math.log10(255.0**2 / mse))


# ----------------------------------------------------------------- coders


def code_svt(args: argparse.Namespace, yuv: Path, work: Path, window: dict[str, Any], frames: int, crf: int, cpus: list[int]) -> dict[str, Any]:
    from src.codecs import svtav1

    width, height = window["size"]
    work.mkdir(parents=True, exist_ok=True)
    fps = Fraction(window["fps"]).limit_denominator(1001)
    rec = svtav1.code(yuv, work, width=width, height=height, fps=fps, frames=frames, crf=crf, preset=PRESET,
                      threads=len(cpus), full_range=False, encoder=args.encoder, decoder=args.decoder, cpus=cpus)
    if rec["decoded_frames"] != frames:
        raise RuntimeError(f"SVT-AV1 CRF {crf}: decoded {rec['decoded_frames']} of {frames} frames")
    return {"codec": "svt-av1", "param": crf, "kbps": rec["payload_bytes"] * 8 / (frames / float(window["fps"])) / 1000.0,
            "bytes": rec["payload_bytes"], "stream_sha256": rec["stream_sha256"], "decoded": rec["decoded"],
            "command": rec["encode_command"], "decode_command": rec["decode_command"]}


def code_dcvc(args: argparse.Namespace, yuv: Path, work: Path, window: dict[str, Any], frames: int, qp: int) -> dict[str, Any]:
    from src.codecs.dcvc_uf_worker import dcvc_command

    width, height = window["size"]
    work.mkdir(parents=True, exist_ok=True)
    plan = {"structure": "htl", "qp": qp, "frame_count": frames, "height": height, "width": width, "src_type": "yuv420",
            "frames_file": str(yuv), "container": str(work / "stream.psdc"), "out_file": str(work / "decoded.yuv"),
            "image_ckpt": args.image_ckpt, "image_sha256": args.image_sha256, "video_ckpt": args.video_ckpt,
            "video_sha256": args.video_sha256, "profile": False}
    write_json(work / "plan.json", plan)
    reports = {}
    for action in ("encode", "decode"):
        command, env, cwd = dcvc_command(Path(sys.prefix), action, work / "plan.json", work / f"{action}.json")
        done = subprocess.run(command, env={**STAGE_ENV, **env}, cwd=cwd, capture_output=True, text=True, timeout=3600)
        if done.returncode:
            raise RuntimeError(f"DCVC-UF {action} failed ({done.returncode}):\n{done.stderr[-4000:]}")
        reports[action] = json.loads((work / f"{action}.json").read_text())
    data = (work / "stream.psdc").read_bytes()
    enc, dec = reports["encode"], reports["decode"]
    got = (work / "decoded.yuv").stat().st_size // (width * height * 3 // 2)
    if got != frames:
        raise RuntimeError(f"DCVC-UF QP {qp}: decoded {got} of {frames} frames")
    return {"codec": "dcvc-uf-htl", "param": qp, "kbps": len(data) * 8 / (frames / float(window["fps"])) / 1000.0, "bytes": len(data),
            "stream_sha256": hashlib.sha256(data).hexdigest(), "decoded": str(work / "decoded.yuv"),
            "deterministic": dec["deterministic"], "decoder_matches_encoder_intra": dec["passes"][0]["frame_sha256"][0] == enc["i_recon_sha256"],
            "kernels": enc.get("kernels"), "peak_allocated_mib": enc.get("peak_allocated_mib"),
            "extension": (enc.get("environment") or {}).get("extension")}


# ----------------------------------------------------------------- scoring


class Detectors:
    def __init__(self, models: list[tuple[str, str]]):
        import torch
        from ultralytics import YOLO

        if not torch.cuda.is_available():
            raise RuntimeError("the detectors need the claimed GPU")
        self.models = {n: YOLO(p) for n, p in models}
        self.paths = dict(models)

    def record(self) -> dict[str, Any]:
        import torch
        import ultralytics

        return {"ultralytics": ultralytics.__version__, "device": torch.cuda.get_device_name(0), "imgsz": DET_IMGSZ, "conf": DET_CONF,
                "class": SPORTS_BALL, "models": {n: {"path": p, "sha256": file_sha256(Path(p))} for n, p in self.paths.items()},
                "on_cuda": {n: str(next(m.model.parameters()).device) for n, m in self.models.items()}}

    def centres(self, name: str, rgb: Any, batch: int = 16) -> list[np.ndarray]:
        """Per frame, (n, 3) array of detection centre x, y and confidence."""
        out = []
        for s in range(0, rgb.shape[0], batch):
            frames = [np.ascontiguousarray(np.asarray(rgb[t])[:, :, ::-1]) for t in range(s, min(s + batch, rgb.shape[0]))]
            for r in self.models[name].predict(frames, imgsz=DET_IMGSZ, conf=DET_CONF, classes=[SPORTS_BALL], device="cuda:0", verbose=False):
                b = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
                c = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
                out.append(np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, c], 1) if len(b) else np.zeros((0, 3)))
        return out


def recall(dets: list[np.ndarray], rows: list[tuple[int, int, int]]) -> dict[str, Any]:
    visible = [t for t, (v, _, _) in enumerate(rows) if v in tw.VISIBLE]
    out: dict[str, Any] = {"visible": len(visible)}
    for px in HIT_PX:
        hits = 0
        for t in visible:
            _, x, y = rows[t]
            d = dets[t]
            hits += bool(len(d)) and bool((np.hypot(d[:, 0] - x, d[:, 1] - y) <= px).any())
        out[f"recall_{px:g}px"] = hits / len(visible) if visible else None
    false = 0
    for t, d in enumerate(dets):
        v, x, y = rows[t]
        if not len(d):
            continue
        far = np.hypot(d[:, 0] - x, d[:, 1] - y) > HIT_PX[0] if v in tw.VISIBLE else np.ones(len(d), bool)
        false += int(far.sum())
    out["false_per_frame"] = false / max(len(dets), 1)
    return out


def oracle(decoded: np.ndarray, source: np.ndarray, rows: list[tuple[int, int, int]], out: Path) -> np.ndarray:
    """The decode with the source's pixels within `PASTE_PX` of the labelled (coded) position on visible frames."""
    frames, height, width, _ = decoded.shape
    comp = np.lib.format.open_memmap(out, mode="w+", dtype=np.uint8, shape=decoded.shape)
    for t in range(frames):
        frame = np.asarray(decoded[t]).copy()
        v, x, y = rows[t]
        if v in tw.VISIBLE:
            m = tw.disk((height, width), x, y, PASTE_PX)
            frame[m] = np.asarray(source[t])[m]
        comp[t] = frame
    comp.flush()
    return np.load(out, mmap_mode="r")


def quality(decoded: Any, source: Any, rows: list[tuple[int, int, int]]) -> dict[str, Any]:
    frames, height, width, _ = source.shape
    se_ball = n_ball = 0.0
    se_frame = 0.0
    for t in range(frames):
        d = np.asarray(decoded[t]).astype(np.float64)
        s = np.asarray(source[t]).astype(np.float64)
        err = (d - s) ** 2
        se_frame += float(err.mean())
        v, x, y = rows[t]
        if v in tw.VISIBLE:
            m = tw.disk((height, width), x, y, REGION_PX)
            se_ball += float(err[m].sum())
            n_ball += float(m.sum() * 3)
    return {"ball_psnr": mse_to_psnr(se_ball / n_ball) if n_ball else None, "frame_psnr": mse_to_psnr(se_frame / frames)}


def fastest(rows: list[tuple[int, int, int]]) -> tuple[float, int] | None:
    """The visible frame where the ball moves most since the previous one (speed, frame)."""
    speeds = []
    for t in range(1, len(rows)):
        (v0, x0, y0), (v1, x1, y1) = rows[t - 1], rows[t]
        if v0 in tw.VISIBLE and v1 in tw.VISIBLE:
            speeds.append((math.hypot(x1 - x0, y1 - y0), t))
    return max(speeds) if speeds else None


def review_strip(source: np.ndarray, versions: dict[str, np.ndarray], row: tuple[int, int, int], at: tuple[float, int],
                 out: Path) -> dict[str, Any]:
    """Crops around the ball on one frame: the source, then each named version's frame."""
    import cv2

    speed, t = at
    _, x, y = row
    height, width = source.shape[:2]
    x0, y0 = int(np.clip(x - REVIEW_CROP // 2, 0, width - REVIEW_CROP)), int(np.clip(y - REVIEW_CROP // 2, 0, height - REVIEW_CROP))
    crops = [source[y0:y0 + REVIEW_CROP, x0:x0 + REVIEW_CROP]]
    crops += [v[y0:y0 + REVIEW_CROP, x0:x0 + REVIEW_CROP] for v in versions.values()]
    strip = np.concatenate([cv2.resize(c, (REVIEW_CROP * 4, REVIEW_CROP * 4), interpolation=cv2.INTER_NEAREST) for c in crops], 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), strip[:, :, ::-1])
    return {"t": t, "speed_px_per_frame": speed, "order": ["source", *versions]}


# ----------------------------------------------------------------- per window


def code_window(args: argparse.Namespace, window: dict[str, Any], sources: dict[str, str], pool: ThreadPoolExecutor,
                cpu_sets: list[list[int]], scratch: Path) -> dict[str, Any]:
    """Phase 1, before any detector is loaded: the window converted, DCVC-UF at every QP on the GPU (as H3c ran its
    worker, from a process without ultralytics) while SVT-AV1 codes every CRF on the CPU. Decodes stay on disk."""
    began = time.time()
    frames = min(int(args.frames), int(window["frames"]))
    work = scratch / "work" / safe(window["id"])
    work.mkdir(parents=True, exist_ok=True)
    yuv = work / "source.yuv"
    converted = tw.convert(args.ffmpeg, window, Path(sources[window["source"]["staged"]]), yuv, frames)
    svt: list[Future[dict[str, Any]]] = [pool.submit(code_svt, args, yuv, work / f"svt-{crf}", window, frames, crf, cpu_sets[i % len(cpu_sets)])
                                         for i, crf in enumerate(CRFS)]
    points = [code_dcvc(args, yuv, work / f"dcvc-{qp}", window, frames, qp) for qp in QPS]
    points += [f.result() for f in svt]
    return {"window": window, "frames": frames, "work": work, "yuv": yuv, "convert": converted, "points": points,
            "coding_seconds": round(time.time() - began, 2)}


def score_window(coded: dict[str, Any], detectors: Detectors, trajectory: dict[str, Any], publish: Path) -> dict[str, Any]:
    """Phase 2: every decode and its oracle composite scored, the review strip drawn, the decodes removed."""
    began = time.time()
    window, frames, work = coded["window"], coded["frames"], coded["work"]
    width, height = window["size"]
    rows = tw.labels(window, frames)
    source = tw.rgb_frames(coded["yuv"], width, height, frames, work / "source.npy")
    det_source = {n: recall(detectors.centres(n, source), rows) for n in detectors.models}
    points = []
    low: dict[str, Any] = {}
    at = fastest(rows)
    for rec in coded["points"]:
        decoded = tw.rgb_frames(Path(rec["decoded"]), width, height, frames, work / "decoded.npy")
        comp = oracle(decoded, source, rows, work / "oracle.npy")
        rec["decode"] = {**quality(decoded, source, rows), "detectors": {n: recall(detectors.centres(n, decoded), rows) for n in detectors.models}}
        rec["oracle"] = {**quality(comp, source, rows), "detectors": {n: recall(detectors.centres(n, comp), rows) for n in detectors.models},
                         "kbps": rec["kbps"] + trajectory["kbps"], "raw_kbps": rec["kbps"] + trajectory["raw_kbps"]}
        lowest = low.get(rec["codec"])
        if at is not None and (lowest is None or rec["kbps"] < lowest["kbps"]):
            low[rec["codec"]] = {"kbps": rec["kbps"], "decode": np.array(decoded[at[1]]), "oracle": np.array(comp[at[1]]),
                                 "label": f"{rec['codec']}@{rec['param']}"}
        del decoded, comp
        Path(rec.pop("decoded")).unlink()
        points.append(rec)
    versions = {}
    for codec, info in sorted(low.items()):
        versions[f"{info['label']} decode"] = info["decode"]
        versions[f"{info['label']} oracle"] = info["oracle"]
    review = review_strip(np.asarray(source[at[1]]), versions, rows[at[1]], at, publish / safe(window["id"]) / "review-fastest.png") \
        if at is not None else None
    import torch

    row = {"id": window["id"], "frames": frames, "fps": window["fps"], "size": [width, height], "convert": coded["convert"],
           "source_detectors": det_source, "points": sorted(points, key=lambda p: (p["codec"], p["kbps"])),
           "trajectory": trajectory, "review": review, "coding_seconds": coded["coding_seconds"],
           "seconds": round(coded["coding_seconds"] + time.time() - began, 2),
           "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    del source
    shutil.rmtree(work, ignore_errors=True)
    return row


def command_run(args: argparse.Namespace) -> int:
    spec = json.loads(Path(args.windows).read_text())
    by_id = {w["id"]: w for w in spec["windows"]}
    wanted = args.select.split(",")
    if any(w not in by_id or by_id[w]["dataset"] != "tracknet" for w in wanted):
        raise SystemExit("K5 runs on TrackNet windows of windows.json")
    all_tn = [w for w in spec["windows"] if w["dataset"] == "tracknet"]
    trajectory = tw.trajectory_rate([tw.trajectory_symbols(tw.labels(w, int(w["frames"]))) for w in all_tn], 30.0)
    trajectory["windows"] = [w["id"] for w in all_tn]
    sources = dict(args.source)
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(scratch / "ultralytics"))
    publish = scratch / "publish" / "windows"
    publish.mkdir(parents=True, exist_ok=True)
    restored = []
    checkpoints = os.environ.get("PS_CHECKPOINT_DIR")
    if checkpoints and Path(checkpoints).is_dir():
        for path in sorted(Path(checkpoints).glob("window--*.tar")):
            with tarfile.open(path) as tar:
                tar.extractall(publish, filter="data")
            restored.append(path.name[len("window--"):-4])
    todo = [w for w in wanted if safe(w) not in restored]
    import torch

    threads = int(os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 16)
    cpus = sorted(os.sched_getaffinity(0))[:threads]  # type: ignore[attr-defined,unused-ignore]
    per = max(1, len(cpus) // len(CRFS))
    cpu_sets = [cpus[i * per:(i + 1) * per] for i in range(len(CRFS))]
    done = len(wanted) - len(todo)
    progress(done)
    with ThreadPoolExecutor(max_workers=len(CRFS)) as pool:
        coded = [code_window(args, by_id[w], sources, pool, cpu_sets, scratch) for w in todo]
    detectors = Detectors([(n, p) for n, p in args.detector])
    for c in coded:
        window_id = c["window"]["id"]
        torch.cuda.reset_peak_memory_stats()
        row = score_window(c, detectors, trajectory, publish)
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
        print(f"{window_id}: {row['seconds']} s", flush=True)
    from src.codecs import svtav1

    write_json(stage_dir() / "k5.json", {
        "stage": os.environ.get("PS_STAGE", "local"), "windows": wanted, "restored": restored,
        "windows_json": {"path": args.windows, "sha256": file_sha256(Path(args.windows))}, "decision": DECISION,
        "detectors": detectors.record(), "trajectory": trajectory,
        "tools": {"encoder": svtav1.tool(args.encoder), "decoder": svtav1.tool(args.decoder), "ffmpeg": svtav1.tool(args.ffmpeg),
                  "dcvc": {"tree": str(Path(sys.prefix) / "opt" / "DCVC"), "image_sha256": args.image_sha256, "video_sha256": args.video_sha256}},
        "settings": {"crfs": CRFS, "preset": PRESET, "qps": QPS, "paste_px": PASTE_PX, "region_px": REGION_PX, "hit_px": HIT_PX,
                     "det_imgsz": DET_IMGSZ, "det_conf": DET_CONF, "frames": int(args.frames), "cpu_sets": cpu_sets},
        "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
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
    dcvc = [p for r in rows for p in r["points"] if p["codec"] == "dcvc-uf-htl"]
    return {
        "every_window_published": sorted(r["id"] for r in rows) == sorted(record["windows"]),
        "converted_every_frame": all(r["convert"]["frames"] == r["frames"] for r in rows),
        "every_point_coded": all(sorted((p["codec"], p["param"]) for p in r["points"])
                                 == sorted([("dcvc-uf-htl", q) for q in QPS] + [("svt-av1", c) for c in CRFS]) for r in rows),
        "dcvc_decodes_deterministic": all(p["deterministic"] and p["decoder_matches_encoder_intra"] for p in dcvc),
        "dcvc_decodes_are_pictures": all(p["decode"]["frame_psnr"] >= SANE_FRAME_PSNR for p in dcvc),
        "detectors_on_cuda": all(v.startswith("cuda") for v in record["detectors"]["on_cuda"].values()),
        "oracle_not_below_decode_on_ball": all((p["oracle"]["ball_psnr"] or 0) >= (p["decode"]["ball_psnr"] or 0) - 1e-6 for r in rows for p in r["points"]),
        "oracle_adds_the_trajectory": all(abs(p["oracle"]["kbps"] - p["kbps"] - record["trajectory"]["kbps"]) < 1e-9 for r in rows for p in r["points"]),
        "visible_frames_scored": all(v["visible"] > 0 for r in rows for v in r["source_detectors"].values()),
        "rates_increase_with_quality_setting": all(
            all(a["kbps"] > b["kbps"] for a, b in zip(sorted((p for p in r["points"] if p["codec"] == "svt-av1"), key=lambda p: p["param"]),
                                                     sorted((p for p in r["points"] if p["codec"] == "svt-av1"), key=lambda p: p["param"])[1:]))
            for r in rows),
    }


def command_validate(args: argparse.Namespace) -> int:
    stage = stage_dir()
    record = json.loads((stage / "k5.json").read_text())
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


def bd_rate(test: list[tuple[float, float]], anchor: list[tuple[float, float]], overlap: float) -> float | None:
    """Bjøntegaard rate difference (fraction) of ``test`` against the anchor's envelope, PCHIP log-rate over the
    overlapping quality range; None under ``overlap``."""
    from scipy.interpolate import PchipInterpolator

    from experiments.background.g5 import envelope, pareto

    t, a = pareto(test), envelope(anchor)
    if len(t) < 2 or len(a) < 2:
        return None
    lo, hi = max(t[0][1], a[0][1]), min(t[-1][1], a[-1][1])
    if hi - lo < overlap:
        return None

    def area(points: list[tuple[float, float]]) -> float:
        f = PchipInterpolator([q for _, q in points], [math.log(r) for r, _ in points])
        return float(f.integrate(lo, hi))

    return math.exp((area(t) - area(a)) / (hi - lo)) - 1.0


def verdict(test: list[tuple[float, float]], anchor: list[tuple[float, float]], overlap: float) -> dict[str, Any]:
    from experiments.background.g5 import anchor_quality

    bd = bd_rate(test, anchor, overlap)
    if bd is not None:
        return {"bd_rate": bd, "basis": "bd"}
    above = [None if (q := anchor_quality(anchor, r)) is None else quality > q for r, quality in test]
    known = [a for a in above if a is not None]
    if known and all(known) and len(known) == len(above):
        return {"bd_rate": -math.inf, "basis": "better at every rate"}
    if known and not any(known) and len(known) == len(above):
        return {"bd_rate": math.inf, "basis": "worse at every rate"}
    return {"bd_rate": None, "basis": "unclear", "above": above}


def collapse_rate(points: list[dict[str, Any]], key: str, source_recall: float) -> float | None:
    """The rate below which a coder's recall falls under `DECISION['collapse_share']` of the source's (log-linear
    between points; None when every point is above it, 0 when none is)."""
    target = DECISION["collapse_share"] * source_recall
    pts = sorted((p["kbps"], p["decode"]["detectors"][key]["recall_10px"]) for p in points)
    if all(r >= target for _, r in pts):
        return None
    for (r0, q0), (r1, q1) in zip(pts, pts[1:]):
        if q0 < target <= q1:
            w = (target - q0) / (q1 - q0)
            return math.exp(math.log(r0) + w * (math.log(r1) - math.log(r0)))
    return math.inf if pts[-1][1] < target else 0.0


def command_report(args: argparse.Namespace) -> int:
    rows = [r for p in args.published for r in rows_from_tar(Path(p))]
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("a window appears twice")
    pilot = [r for r in rows if r["id"] in set(args.pilot_windows.split(","))]
    names = sorted(rows[0]["source_detectors"])
    src = {n: float(np.mean([r["source_detectors"][n]["recall_10px"] for r in pilot])) for n in names}
    detector = max(src, key=lambda n: src[n])
    valid = src[detector] >= DECISION["detector_min_source_recall"]
    judges = {"ball_psnr": lambda d: d["ball_psnr"], "recall": lambda d: d["detectors"][detector]["recall_10px"],
              "frame_psnr": lambda d: d["frame_psnr"]}
    deciding = ["ball_psnr"] + (["recall"] if valid else [])
    per_window = {}
    passes = 0
    for r in rows:
        out: dict[str, Any] = {}
        coders = sorted({p["codec"] for p in r["points"]})
        for judge, get in judges.items():
            overlap = DECISION["overlap"].get(judge, 0.5)
            anchors = {c: [(p["kbps"], get(p["decode"])) for p in r["points"] if p["codec"] == c] for c in coders}
            anchors["pooled"] = [pt for c in coders for pt in anchors[c]]
            tests = {c: [(p["oracle"]["kbps"], get(p["oracle"])) for p in r["points"] if p["codec"] == c] for c in coders}
            out[judge] = {f"oracle-{c}": {a: verdict(tests[c], pts, overlap) for a, pts in anchors.items() if a in (c, "pooled")}
                          for c in coders}
        winners = [c for c in coders if all((out[j][f"oracle-{c}"]["pooled"]["bd_rate"] is not None
                                             and out[j][f"oracle-{c}"]["pooled"]["bd_rate"] <= DECISION["max_bd_rate"]) for j in deciding)]
        out["passes_with"] = winners
        passes += bool(winners)
        sr = r["source_detectors"][detector]["recall_10px"]
        out["collapse_kbps"] = {c: collapse_rate([p for p in r["points"] if p["codec"] == c], detector, sr) for c in coders}
        out["source_recall"] = sr
        per_window[r["id"]] = out
    build = passes >= DECISION["min_windows"] and len(rows) >= DECISION["of_windows"]
    out_all = {"decision_rule": DECISION, "published": {p: file_sha256(Path(p)) for p in args.published}, "windows": sorted(ids),
               "detector": detector, "detector_source_recall_pilot": src, "detector_valid": valid, "deciding_judges": deciding,
               "per_window": per_window, "windows_passing": passes, "build_ball_mode": build,
               "trajectory": rows[0]["trajectory"]}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "k5-report.json", out_all)
    print(json.dumps({k: out_all[k] for k in ("detector", "detector_source_recall_pilot", "detector_valid", "windows_passing",
                                              "build_ball_mode")}, indent=1, default=str))
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
    run.add_argument("--detector", nargs=2, action="append", required=True, metavar=("NAME", "PT"))
    run.add_argument("--encoder", default="SvtAv1EncApp")
    run.add_argument("--decoder", default="dav1d")
    run.add_argument("--ffmpeg", default="ffmpeg")
    sub.add_parser("validate")
    rep = sub.add_parser("report")
    rep.add_argument("--published", action="append", required=True)
    rep.add_argument("--pilot-windows", required=True)
    rep.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return {"run": command_run, "validate": command_validate, "report": command_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
