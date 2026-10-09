"""PLAN step G1d: how much of G1's VISOR background residual does a depth-aware warp remove?

    python -m experiments.background.g1d prepare --clips CLIPS.JSON --g1 DIR --select all|pilot|ID,ID \\
        --limit-seconds S --source NAME PATH ... --visor-archive DIR --visor-fill DIR --hand-objects DIR
    python -m experiments.background.g1d validate-prepare
    python -m experiments.background.g1d run --prepared DIR --select all|pilot|rest|ID,ID --limit-frames N \\
        [--methods rot,h1,...]
    python -m experiments.background.g1d validate
    python -m experiments.background.g1d report --result g1d.json ... --published DIR ... --out DIR

``prepare`` decodes G1's analysis frames of the selected clips (`select_clips`)
once and stores them losslessly (FFV1) with G1's dilated foreground, checking
each frame's foreground share against G1's record. ``--g1`` holds G1's
published ``clips/<id>/result.json`` and, for stretches, ``masks.rle``.

``run`` re-runs G1's registration with G1's lens, so that every ``direct``
frame is warped from its G1 reference by every method (`METHODS`), and
measures each warp with G1's `camera.residual`. Per clip it writes
``publish/clips/<id>/result.json`` (per-frame records, summary) and an overlay
sheet; ``g1d.json`` in ``PS_STAGE_DIR`` holds the summaries. Finished clips are
checkpointed (`g1.save_clip`) and restored on a declared resume.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import multiprocessing
import os
import pickle
import shutil
import time
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from experiments.background import camera, depth, g1
from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json

#: The decision rule, fixed before any run (docs/experiments.md, G1d entry).
DECISION: dict[str, Any] = {
    "explained_px": camera.EXPLAINED_PX,
    "meaningful_median_clip_share": 0.50,
    "groups": ["visor-window", "visor-long"],
    "step1_measures": ["epi", "tri"],  # (c) and (d) run unless neither reaches a meaningful share anywhere
    "realizable": ["planes2", "planes3", "planes4", "tri", "tri_raw", "da3", "da3_tri", "kf"],
    "hole_share_for_3dgs": 0.05,
    "max_hole_share": 0.10,  # a frame counts as explained only if the method renders 90% of h1's region
}
SEED = "pointstream-g1d"
STRETCHES = 10
PILOT = {"visor-window": 2, "visor-long": 1}
METHODS_AB = ("rot", "h1", "planes2", "planes3", "planes4", "epi", "tri", "tri_raw")  # steps (a) and (b)
METHODS = METHODS_AB + ("da3", "da3_tri", "kf")  # (c) needs --da3; (d) renders the --kf-depth source
COMPANION_S = (0.5, 1.0)
COMPANION_RANGE_S = (0.3, 1.5)
LENS_PAIR_S = 0.5
LENS_PAIR_STARTS = 40
NEAR_FG_PX = 16  # analysis px (32 at 1080p) around the foreground for the mask-leak measure
REPRODUCE = {"reference_share": 0.99, "p90_tolerance_px": 0.05, "p90_share": 0.95}
CHUNK_FRAMES = 40
FFV1_OPTIONS = {"level": "3", "slicecrc": "1", "slices": "16"}
SELF_TEST: dict[str, Any] = {"yaw_deg": 1.5, "pitch_deg": 0.5, "translation": [0.06, 0.01, 0.03], "max_flow_p90_px": 0.5,
             "max_angle_error_deg": 0.1}


# ----------------------------------------------------------------- selection


def rank(group: str, video: str) -> str:
    return hashlib.sha256(f"{SEED}:{group}:{video}".encode()).hexdigest()


def select_clips(spec: dict[str, Any], which: str) -> list[dict[str, Any]]:
    """All 34 windows and the ``STRETCHES`` stretches of smallest rank; ``pilot`` is the first of each by rank."""
    windows = sorted((c for c in spec["clips"] if c["group"] == "visor-window"), key=lambda c: rank(c["group"], c["video"]))
    stretches = sorted((c for c in spec["clips"] if c["group"] == "visor-long"),
                       key=lambda c: rank(c["group"], c["video"]))[:STRETCHES]
    chosen = windows + stretches
    pilot = windows[:PILOT["visor-window"]] + stretches[:PILOT["visor-long"]]
    if which == "all":
        return chosen
    if which in ("pilot", "smoke"):
        return pilot
    if which == "rest":
        return [c for c in chosen if c not in pilot]
    by_id = {c["id"]: c for c in chosen}
    missing = [w for w in which.split(",") if w not in by_id]
    if missing:
        raise SystemExit(f"not in G1d's selection: {missing}")
    return [by_id[w] for w in which.split(",")]


# ----------------------------------------------------------------- frames archive (FFV1)


def encode_ffv1(frames: Any, path: Path, threads: int) -> str:
    """Write RGB frames losslessly; returns the sha256 of the raw frame bytes."""
    import av

    digest = hashlib.sha256()
    with av.open(str(path), "w", format="matroska") as out:
        stream = out.add_stream("ffv1", rate=10)
        stream.width, stream.height = camera.ANALYSIS_SIZE
        stream.pix_fmt = "bgr0"
        stream.options = dict(FFV1_OPTIONS)
        stream.thread_count = max(1, threads)
        for rgb in frames:
            rgb = np.ascontiguousarray(rgb)
            digest.update(rgb.tobytes())
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    return digest.hexdigest()


def decode_ffv1(path: Path, threads: int = 4) -> Any:
    """RGB frames of an FFV1 archive, in order."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = max(1, threads)
        for frame in container.decode(stream):
            yield frame.to_ndarray(format="rgb24")


def save_foreground(path: Path, fg: np.ndarray) -> None:
    np.savez_compressed(path, bits=np.packbits(np.asarray(fg, bool).reshape(len(fg), -1), axis=1), shape=np.array(fg.shape))


def load_foreground(path: Path) -> np.ndarray:
    with np.load(path) as data:
        shape = tuple(int(x) for x in data["shape"])
        return np.unpackbits(data["bits"], axis=1, count=shape[1] * shape[2]).reshape(shape).astype(bool)


# ----------------------------------------------------------------- prepare (spawned processes)


def prepare_clip(task: dict[str, Any]) -> dict[str, Any]:
    import cv2

    cv2.setNumThreads(1)
    began = time.time()
    clip = task["plan"]
    work, publish = Path(task["work"]), Path(task["publish"])
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    publish.mkdir(parents=True, exist_ok=True)
    decode_task = {"plan": clip, "out": str(work), "path": task["path"], "threads": task["threads"], "jpeg_dir": None}
    if task.get("jpeg_gate") is not None:
        decode_task["jpeg_gate"] = task["jpeg_gate"]
    decoded = g1.decode_clip(decode_task)
    g1_dir = Path(task["g1"]) / "clips" / g1.safe(clip["id"])
    g1_result = json.loads((g1_dir / "result.json").read_text())
    n = len(clip["indices"])
    base = None
    if clip["masks"]["kind"] == "sam_text":
        from src.segmentation.masks import ClipMasks

        masks = ClipMasks.load(g1_dir / "masks.rle")
        base = [g1.to_analysis(g1.union(f, (masks.height, masks.width))) for f in masks.frames[:n]]
    record = g1.foreground(clip, work, base, task["visor_fill"], task["hand_objects"])
    if clip["masks"]["kind"] == "sam_text":
        record.update(tier=g1.SAM_TIER, g1_masks_rle_sha256=file_sha256(g1_dir / "masks.rle"))
    fg = np.load(work / "foreground.npy", mmap_mode="r")
    g1_share = [r["fg_share"] for r in g1_result["frames"]][:n]
    share = [round(float(np.asarray(m).mean()), 4) for m in fg]
    differ = [i for i, (a, b) in enumerate(zip(share, g1_share)) if abs(a - b) > 1e-4]
    frames = np.load(work / "frames.npy", mmap_mode="r")
    sha = encode_ffv1(frames, publish / "frames.mkv", task["threads"])
    back = hashlib.sha256()
    count = 0
    for rgb in decode_ffv1(publish / "frames.mkv", task["threads"]):
        back.update(np.ascontiguousarray(rgb).tobytes())
        count += 1
    save_foreground(publish / "foreground.npz", np.asarray(fg))
    shutil.copyfile(g1_dir / "result.json", publish / "g1_result.json")
    meta = {
        "id": clip["id"], "group": clip["group"], "video": clip["video"], "frames": n,
        "indices": clip["indices"], "times": clip["times"], "limit_seconds": clip["limit_seconds"],
        "decode": decoded, "masks": record, "fg_share": share,
        "fg_share_matches_g1": {"frames_compared": min(n, len(g1_share)), "differ": differ[:20], "count_differ": len(differ)},
        "frames_sha256": sha, "roundtrip_sha256": back.hexdigest(), "roundtrip_frames": count,
        "lossless": back.hexdigest() == sha and count == n,
        "ffv1_bytes": (publish / "frames.mkv").stat().st_size, "g1_result_sha256": file_sha256(g1_dir / "result.json"),
        "g1_lens": g1_result["lens"], "seconds": round(time.time() - began, 2),
    }
    write_json(publish / "meta.json", meta)
    shutil.rmtree(work, ignore_errors=True)
    return {k: v for k, v in meta.items() if k not in ("indices", "times", "fg_share")}


def command_prepare(args: argparse.Namespace) -> int:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    spec_path = Path(args.clips)
    spec = json.loads(spec_path.read_text())
    clips = [g1.plan(c, args.limit_seconds) for c in select_clips(spec, args.select)]
    sources = {name: path for name, path in args.source}
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    allowance = max(2, int(os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 2))
    threads = 4
    pool = ProcessPoolExecutor(max_workers=max(1, allowance // threads), mp_context=multiprocessing.get_context("spawn"))
    checkpoints = Path(os.environ["PS_CHECKPOINT_DIR"]) if os.environ.get("PS_CHECKPOINT_DIR") else None
    restored = g1.restore_clips(checkpoints, publish) if checkpoints else {}
    futures: dict[str, Future[Any]] = {}
    for clip_id, saved in restored.items():
        future: Future[Any] = Future()
        future.set_result({**saved["result"], "restored_from_checkpoint": True})
        futures[clip_id] = future

    def keep(clip_id: str, future: Future[Any]) -> None:
        if checkpoints is not None and future.exception() is None:
            g1.save_clip(checkpoints, publish / "clips" / g1.safe(clip_id), clip_id, future.result(), None)

    for clip in clips:
        if clip["id"] in restored:
            continue
        task = {"plan": clip, "work": str(scratch / "work" / g1.safe(clip["id"])),
                "publish": str(publish / "clips" / g1.safe(clip["id"])), "path": sources[clip["source"]["name"]],
                "threads": threads, "g1": args.g1, "visor_fill": args.visor_fill, "hand_objects": args.hand_objects}
        if clip["group"] == "visor-window":
            task["jpeg_gate"] = g1.jpeg_gate(clip, Path(args.visor_archive))
        futures[clip["id"]] = pool.submit(prepare_clip, task)
        futures[clip["id"]].add_done_callback(functools.partial(keep, clip["id"]))
    results = []
    for done, clip in enumerate(clips, 1):
        results.append(futures[clip["id"]].result())
        progress(done)
    pool.shutdown()
    write_json(stage_dir() / "prepare.json", {
        "clips_file": {"path": str(spec_path), "sha256": file_sha256(spec_path)}, "select": args.select,
        "limit_seconds": args.limit_seconds, "rule": {"seed": SEED, "stretches": STRETCHES, "pilot": PILOT},
        "g1": args.g1, "restored_clips": sorted(restored), "results": results,
    })
    return 0


def validate_prepare(stage: Path) -> dict[str, bool]:
    import tarfile

    result = json.loads((stage / "prepare.json").read_text())
    rows = result["results"]
    with tarfile.open(stage / "published.tar") as tar:
        names = set(tar.getnames())
    windows = [r for r in rows if r["group"] == "visor-window"]
    return {
        "clips_prepared": bool(rows),
        "every_clip_published": all(all(f"publish/clips/{g1.safe(r['id'])}/{name}" in names for name in
                                        ("frames.mkv", "foreground.npz", "meta.json", "g1_result.json")) for r in rows),
        "frames_lossless": all(r["lossless"] for r in rows),
        "decoded_every_analysis_frame": all(r["decode"]["frames"] == r["frames"] == r["roundtrip_frames"] for r in rows),
        "foreground_equals_g1": all(r["fg_share_matches_g1"]["count_differ"] == 0
                                    and r["fg_share_matches_g1"]["frames_compared"] == r["frames"] for r in rows),
        "window_jpegs_match_decoded_frames": all(all(g["holds"] for g in r["decode"]["jpeg_gate"]) for r in windows),
        "windows_use_the_fill": all(r["masks"]["tier"] == "visor_dense_sam_fill" for r in windows),
        "stretches_use_g1_sam_masks": all(r["masks"]["tier"] == g1.SAM_TIER and r["masks"]["detector"]["boxes"] > 0
                                          for r in rows if r["group"] == "visor-long"),
    }


# ----------------------------------------------------------------- run, phase A: registration (spawned)


def calibration_pairs(feats: list[camera.Features], delta: int) -> list[tuple[np.ndarray, np.ndarray]]:
    import cv2

    n = len(feats)
    pairs = []
    for t in np.unique(np.linspace(0, max(0, n - 1 - delta), min(LENS_PAIR_STARTS, max(1, n - delta))).astype(int)):
        if t + delta >= n:
            continue
        pa, pb = camera.match(feats[t], feats[t + delta])
        if len(pa) < 2 * camera.MIN_INLIERS:
            continue
        try:
            F, keep = cv2.findFundamentalMat(pa, pb, cv2.USAC_MAGSAC, 3.0, 0.999, 5000)
        except cv2.error:
            continue
        if F is not None and keep is not None:
            keep = keep.ravel().astype(bool)
            pairs.append((pa[keep], pb[keep]))
    return pairs


def choose_companions(lens: camera.Lens, k: int, feats: list[camera.Features], poses: list[camera.Pose],
                      times: np.ndarray) -> list[dict[str, Any]]:
    """Frames nearest ``COMPANION_S`` after the keyframe (before, at a clip's end) that match it."""
    chosen: list[dict[str, Any]] = []
    lo, hi = COMPANION_RANGE_S
    candidates = [j for j in range(len(poses)) if j != k and poses[j].status != "lost"
                  and poses[j].segment == poses[k].segment and lo <= abs(times[j] - times[k]) <= hi]
    for target in COMPANION_S:
        order = sorted(candidates, key=lambda j: (abs(abs(times[j] - times[k]) - target) + (0.0 if times[j] > times[k] else 0.25), j))
        for j in order:
            if any(c["index"] == j for c in chosen):
                continue
            pa, pb = camera.match(feats[k], feats[j])
            H, keep = depth.homography_u(lens, pa, pb)
            if H is not None and keep.sum() >= camera.MIN_INLIERS:
                chosen.append({"index": int(j), "pa": pa, "pb": pb, "dt": round(float(times[j] - times[k]), 3),
                               "h_inliers": int(keep.sum())})
                break
    return chosen


def register_clip(task: dict[str, Any]) -> dict[str, Any]:
    """Decode, re-run G1's registration with G1's lens, calibrate the lens, choose companions; pickle the state."""
    import cv2

    cv2.setNumThreads(1)
    began = time.time()
    clip_dir, work = Path(task["clip_dir"]), Path(task["work"])
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    meta = json.loads((clip_dir / "meta.json").read_text())
    g1_result = json.loads((clip_dir / "g1_result.json").read_text())
    n = meta["frames"] if not task["limit_frames"] else min(meta["frames"], task["limit_frames"])
    width, height = camera.ANALYSIS_SIZE
    gray = np.lib.format.open_memmap(work / "gray.npy", mode="w+", dtype=np.uint8, shape=(n, height, width))
    digest = hashlib.sha256()
    keep_rgb: dict[int, np.ndarray] = {}
    count = 0
    for i, rgb in enumerate(decode_ffv1(clip_dir / "frames.mkv", 2)):
        digest.update(np.ascontiguousarray(rgb).tobytes())
        count += 1
        if i < n:
            gray[i] = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            if i == n // 2:
                keep_rgb[i] = rgb
    gray.flush()
    frames_ok = digest.hexdigest() == meta["frames_sha256"] and count == meta["frames"]
    fg = load_foreground(clip_dir / "foreground.npz")[:n]
    np.save(work / "fg.npy", fg)
    bg = ~fg
    times = np.asarray(meta["times"][:n], float)
    feats = [camera.features(gray[t], bg[t]) for t in range(n)]
    f2f: list[tuple[np.ndarray, np.ndarray] | None] = [None]
    for t in range(1, n):
        pa, pb = camera.match(feats[t - 1], feats[t])
        H, keep = camera.homography(pa, pb)
        f2f.append((pa[keep], pb[keep]) if H is not None else None)
    g1_lens = camera.Lens(g1_result["lens"]["f"], g1_result["lens"]["k1"])
    reg = camera.register(g1_lens, feats, f2f)
    g1_frames = g1_result["frames"]
    direct = [t for t in range(1, n) if reg.poses[t].status == "direct"]
    g1_direct = [t for t in range(1, n) if g1_frames[t]["status"] == "direct"]
    same_ref = [t for t in direct if g1_frames[t]["status"] == "direct" and g1_frames[t]["reference"] == reg.poses[t].reference]
    rate = (n - 1) / max(times[-1], 1e-9) if n > 1 else 1.0
    # Depth methods use the published lens; the self-calibration is only reported (it drifts to long focal lengths).
    _, cal_report = depth.calibrate(calibration_pairs(feats, max(1, int(round(LENS_PAIR_S * rate)))))
    cal_lens = depth.epic_fields_lens()
    referenced = sorted({k for t in direct if (k := reg.poses[t].reference) is not None})
    companions = {k: choose_companions(cal_lens, k, feats, reg.poses, times) for k in referenced}
    speeds: dict[int, float] = {}
    for t in range(1, n):
        a, b = reg.poses[t - 1], reg.poses[t]
        if a.status != "lost" and b.status != "lost" and a.segment == b.segment:
            speeds[t] = camera.angle_deg(b.R @ a.R.T) / max(times[t] - times[t - 1], 1e-9)
    state = {"n": n, "times": times, "g1_lens": g1_lens, "cal_lens": cal_lens, "poses": reg.poses,
             "keyframes": reg.keyframes, "companions": companions, "speeds": speeds}
    with (work / "state.pkl").open("wb") as handle:
        pickle.dump(state, handle)
    summary = {
        "frames": n, "frames_ok": frames_ok, "direct": len(direct), "g1_direct": len(g1_direct),
        "same_reference": len(same_ref),
        "reference_share": round(len(same_ref) / max(1, len(g1_direct)), 4),
        "g1_lens": {"f": g1_lens.f, "k1": g1_lens.k1}, "lens": {"f": cal_lens.f, "k1": cal_lens.k1, "source": "EPIC Fields"},
        "calibration": cal_report,
        "keyframes_referenced": len(referenced),
        "keyframes_without_companion": sum(1 for k in referenced if not companions[k]),
        "companions_per_keyframe_median": float(np.median([len(c) for c in companions.values()])) if companions else 0.0,
        "seconds": round(time.time() - began, 2),
    }
    if task.get("self_test"):
        mid = n // 2
        summary["self_test"] = self_test(gray[mid], bg[mid], cal_lens)
    return summary


def self_test(gray: np.ndarray, bg: np.ndarray, lens: camera.Lens) -> dict[str, Any]:
    """A real frame re-rendered through a known depth (tilted plane plus a bump) and pose must come back."""
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    truth = 2.0 + 0.8 * (xx / w) + 0.6 * (yy / h) - 0.7 * np.exp(-(((xx - 0.3 * w) / (0.12 * w)) ** 2 + ((yy - 0.6 * h) / (0.15 * h)) ** 2))
    R = camera.rotation(np.radians([SELF_TEST["pitch_deg"], SELF_TEST["yaw_deg"], 0.0]))
    t = np.asarray(SELF_TEST["translation"], float)
    everywhere = np.ones_like(bg)
    warp, hit = depth.warp_depth(lens, truth, everywhere, R, t)
    import cv2

    moved = cv2.remap(gray, warp.map_x, warp.map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    bg_t = cv2.erode((warp.valid & hit).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    pa, pb = camera.match(camera.features(gray, everywhere), camera.features(moved, bg_t))
    pose = depth.pnp(lens, truth, everywhere, pa, pb)
    if pose is None:
        return {"passed": False, "reason": "no pose"}
    angle = camera.angle_deg(pose[0] @ R.T)
    back, _ = depth.warp_depth(lens, truth, everywhere, pose[0], pose[1])
    res = camera.residual(lens, moved, gray, back, bg_t, everywhere)
    passed = bool(res.get("measured") and res["flow_p90_px"] <= SELF_TEST["max_flow_p90_px"]
                  and angle <= SELF_TEST["max_angle_error_deg"])
    return {"passed": passed, "angle_error_deg": round(angle, 5), "translation_error": round(float(np.linalg.norm(pose[1] - t)), 5),
            "flow_p90_px": res.get("flow_p90_px"), "pnp_inliers": pose[2], "settings": SELF_TEST}


# ----------------------------------------------------------------- run, phase B: evaluation (spawned)


class Products:
    """Per-keyframe products (label maps, depth), computed once per process and chunk."""

    def __init__(self, state: dict[str, Any], gray: Any, bg: np.ndarray, da3: Path | None = None) -> None:
        self.state, self.gray, self.bg, self.da3 = state, gray, bg, da3
        self.labels: dict[tuple[int, int], tuple[np.ndarray | None, dict[str, Any]]] = {}
        self.depths: dict[int, tuple[depth.KeyDepth | None, dict[str, Any]]] = {}
        self.da3_depths: dict[tuple[int, bool], tuple[depth.KeyDepth | None, dict[str, Any]]] = {}
        self._da3_npz: Any = None

    def da3_depth(self, k: int, aligned: bool) -> tuple[depth.KeyDepth | None, dict[str, Any]]:
        """DA3's depth of keyframe ``k`` at the analysis size; ``aligned``: its inverse depth fitted to the
        triangulated pixels of `key_depth` (DA3's shape, triangulation's scale and offset)."""
        if (k, aligned) not in self.da3_depths:
            began = time.time()
            if self._da3_npz is None:
                assert self.da3 is not None, "da3 methods need --da3"
                self._da3_npz = dict(np.load(self.da3 / "depth.npz"))
            info: dict[str, Any] = {}
            kd = None
            if f"depth_{k}" in self._da3_npz:
                width, height = camera.ANALYSIS_SIZE
                d = depth.resize_depth(self._da3_npz[f"depth_{k}"], width, height)
                if aligned:
                    tri, _ = self.key_depth(k)
                    if tri is not None:
                        d, info = depth.align_inverse_depth(d, tri.depth, tri.measured & np.isfinite(d))
                        kd = depth.sent_depth(d, self.bg[k], info)
                else:
                    kd = depth.sent_depth(d, self.bg[k], info)
            info["seconds"] = round(time.time() - began, 3)
            if kd is not None:
                info.update(png_bytes=kd.png_bytes, png16_bytes=kd.png16_bytes)
            self.da3_depths[(k, aligned)] = (kd, info)
        return self.da3_depths[(k, aligned)]

    def source_depth(self, k: int, source: str) -> tuple[depth.KeyDepth | None, dict[str, Any]]:
        return self.key_depth(k) if source == "tri" else self.da3_depth(k, source == "da3_tri")

    def companions(self, k: int) -> list[dict[str, Any]]:
        return [{**c, "gray": np.asarray(self.gray[c["index"]]), "bg": self.bg[c["index"]]}
                for c in self.state["companions"].get(k, [])]

    def label_map(self, k: int, count: int) -> tuple[np.ndarray | None, dict[str, Any]]:
        import cv2

        if (k, count) not in self.labels:
            began = time.time()
            comps = self.companions(k)
            labels = None
            if comps:
                c = comps[0]
                labels = depth.plane_labels(self.state["cal_lens"], np.asarray(self.gray[k]), c["gray"], self.bg[k], c["bg"],
                                            c["pa"], c["pb"], count)
            info: dict[str, Any] = {"seconds": round(time.time() - began, 3)}
            if labels is not None:
                info["png_bytes"] = len(cv2.imencode(".png", (labels + 1).astype(np.uint8), [cv2.IMWRITE_PNG_COMPRESSION, 9])[1])
                info["planes"] = int(labels.max()) + 1
            self.labels[(k, count)] = (labels, info)
        return self.labels[(k, count)]

    def key_depth(self, k: int) -> tuple[depth.KeyDepth | None, dict[str, Any]]:
        if k not in self.depths:
            began = time.time()
            kd = depth.keyframe_depth(self.state["cal_lens"], np.asarray(self.gray[k]), self.bg[k], self.companions(k))
            info: dict[str, Any] = {"seconds": round(time.time() - began, 3)}
            if kd is not None:
                info.update(png_bytes=kd.png_bytes, png16_bytes=kd.png16_bytes, **kd.info)
            self.depths[k] = (kd, info)
        return self.depths[k]


def measure(lens: camera.Lens, gray_t: np.ndarray, gray_k: np.ndarray, warp: camera.Warp, bg_t: np.ndarray,
            bg_k: np.ndarray, F: np.ndarray | None, near: np.ndarray) -> tuple[dict[str, Any], dict[str, Any] | None]:
    res = camera.residual(lens, gray_t, gray_k, depth.complete(warp, (gray_k.shape[0], gray_k.shape[1])), bg_t, bg_k, F=F, attribute=True,
                          keep=True)
    images = res.pop("_images", None)
    out = g1.strip(res) or {}
    if images is not None and out.get("measured"):
        big = images["region"] & (images["flow_mag"] > camera.EXPLAINED_PX)
        out["misaligned_near_fg_share"] = round(float((big & near).sum() / max(1, big.sum())), 4)
    out["rendered_share"] = round(float((warp.valid & bg_t).sum() / max(1, bg_t.sum())), 4)
    return out, images


def evaluate_chunk(task: dict[str, Any]) -> list[dict[str, Any]]:
    import cv2

    cv2.setNumThreads(1)
    work, publish = Path(task["work"]), Path(task["publish"])
    with (work / "state.pkl").open("rb") as handle:
        state = pickle.load(handle)
    gray = np.load(work / "gray.npy", mmap_mode="r")
    fg = np.load(work / "fg.npy", mmap_mode="r")
    bg = ~np.asarray(fg)
    methods = task["methods"]
    g1_frames = task["g1_frames"]
    products = Products(state, gray, bg, Path(task["da3"]) if task.get("da3") else None)
    g1_lens, cal = state["g1_lens"], state["cal_lens"]
    poses, times = state["poses"], state["times"]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * NEAR_FG_PX + 1, 2 * NEAR_FG_PX + 1))
    records = []
    for t in range(task["lo"], task["hi"]):
        pose = poses[t]
        rec: dict[str, Any] = {"i": t, "time": round(float(times[t]), 4), "status": pose.status, "reference": pose.reference}
        g1_rec = g1_frames[t]
        rec["g1"] = {"status": g1_rec["status"], "reference": g1_rec.get("reference"),
                     "flow_p90_px": (g1_rec.get("f2r_rot") or {}).get("flow_p90_px"),
                     "measured": bool((g1_rec.get("f2r_rot") or {}).get("measured"))}
        if pose.status != "direct" or pose.reference is None or pose.matches is None:
            records.append(rec)
            continue
        k = pose.reference
        rec["dt_reference_s"] = round(float(times[t] - times[k]), 4)
        rec["speed_deg_s"] = None if t not in state["speeds"] else round(state["speeds"][t], 3)
        companions = state["companions"].get(k, [])
        rec["companion_of_reference"] = any(c["index"] == t for c in companions)
        gt, gk = np.asarray(gray[t]), np.asarray(gray[k])
        near = (cv2.dilate(fg[t].astype(np.uint8), kernel) > 0) & ~fg[t]
        pa, pb = pose.matches
        out: dict[str, Any] = {}
        overlay: dict[str, Any] = {}
        if "rot" in methods:
            F = pose.pair["F"] if pose.pair and pose.pair["prefers_f"] else None
            began = time.perf_counter()
            warp = camera.warp_rotation(g1_lens, pose, poses[k])
            seconds = time.perf_counter() - began
            res, images = measure(g1_lens, gt, gk, warp, bg[t], bg[k], F, near)
            out["rot"] = {**res, "render_s": round(seconds, 4), "frame_bytes": 16}
            # G1's own measure (holes left black), only to check that G1 is reproduced.
            rec["rot_g1"] = g1.strip(camera.residual(g1_lens, gt, gk, warp, bg[t], bg[k]))
            overlay["rot"] = images
        pair = camera.translation_pair(cal, pa, pb)
        F_cal = pair["F"] if pair else None
        F_attr = F_cal if pair and pair["prefers_f"] else None
        rec["pair"] = g1.strip(pair)
        H, h_keep = depth.homography_u(cal, pa, pb)
        h1_region = None
        if H is not None:
            began = time.perf_counter()
            warp = depth.warp_h(cal, H)
            seconds = time.perf_counter() - began
            warped_bg = cv2.remap(bg[k].astype(np.uint8), warp.map_x, warp.map_y, cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT) > 0
            h1_region = warp.valid & bg[t] & warped_bg
            if "h1" in methods:
                res, images = measure(cal, gt, gk, warp, bg[t], bg[k], F_attr, near)
                out["h1"] = {**res, "render_s": round(seconds, 4), "frame_bytes": 32}
                overlay["h1"] = images
        samples = None
        for count in (2, 3, 4):
            name = f"planes{count}"
            if name not in methods or H is None:
                continue
            labels, info = products.label_map(k, count)
            if labels is None:
                out[name] = {"measured": False, "reason": "no label map"}
                continue
            if samples is None:
                # The encoder fits each plane's homography to the dense flow from the keyframe to the frame.
                pos, ok = depth.dense_correspondence(cal, gk, gt, np.linalg.inv(H), bg[k], bg[t])
                samples = depth.dense_samples(gk, pos, ok)
            sa, sb = samples
            at = labels[sa[:, 1].astype(int), sa[:, 0].astype(int)]
            homographies, fallback = [], 0
            for i in range(int(labels.max()) + 1):
                sel = at == i
                Hi = depth.homography_u(cal, sa[sel], sb[sel])[0] if sel.sum() >= camera.MIN_INLIERS else None
                if Hi is None:
                    Hi, fallback = H, fallback + 1
                homographies.append(Hi)
            began = time.perf_counter()
            warp, hit = depth.warp_planes(cal, labels, homographies)
            seconds = time.perf_counter() - began
            res, images = measure(cal, gt, gk, warp, bg[t], bg[k], F_attr, near)
            out[name] = {**res, "render_s": round(seconds, 4), "frame_bytes": 32 * len(homographies), "planes": len(homographies),
                         "fallback_planes": fallback, "keyframe": k, "keyframe_bytes": info.get("png_bytes")}
            overlay[name] = images
        if "epi" in methods and H is not None and F_cal is not None:
            began = time.perf_counter()
            warp = depth.warp_epipolar(cal, gt, gk, H, F_cal, bg[t], bg[k])
            seconds = time.perf_counter() - began
            res, images = measure(cal, gt, gk, warp, bg[t], bg[k], F_attr, near)
            out["epi"] = {**res, "render_s": round(seconds, 4)}
            overlay["epi"] = images
        for name in ("tri", "tri_raw", "da3", "da3_tri"):
            if name not in methods:
                continue
            kd, info = products.key_depth(k) if name.startswith("tri") else products.da3_depth(k, name == "da3_tri")
            pose_t = depth.pnp(cal, kd.depth, kd.measured, pa, pb) if kd is not None else None
            if kd is None or pose_t is None:
                out[name] = {"measured": False, "reason": "no keyframe depth" if kd is None else "no PnP pose"}
                continue
            # tri renders the filled depth; tri_raw only the triangulated pixels (its holes count against it).
            source_depth = np.where(kd.measured, kd.depth, np.nan) if name == "tri_raw" else kd.depth
            began = time.perf_counter()
            warp, hit = depth.warp_depth(cal, source_depth, bg[k], pose_t[0], pose_t[1])
            seconds = time.perf_counter() - began
            res, images = measure(cal, gt, gk, warp, bg[t], bg[k], F_attr, near)
            out[name] = {**res, "render_s": round(seconds, 4), "frame_bytes": 24, "pnp_inliers": pose_t[2],
                         "keyframe": k, "keyframe_bytes": info.get("png_bytes"), "keyframe_bytes16": info.get("png16_bytes")}
            overlay[name] = images
        if "kf" in methods:
            out["kf"], overlay["kf"] = render_kf(task, products, state, t, k, gray, bg, pa, pb, near)
        for name, images in overlay.items():
            entry = out.get(name) or {}
            if images is None or h1_region is None or not entry.get("measured"):
                continue
            # Holes: background both views see (h1's region) that the method does not render.
            hole = float((h1_region & ~images["region"]).sum() / max(1, h1_region.sum()))
            entry["hole_share"] = round(hole, 4)
            entry["covered_explained"] = bool(entry["explained"] and hole <= DECISION["max_hole_share"])
        rec["methods"] = out
        if t in task["overlay_frames"]:
            write_overlay(publish, t, gt, overlay)
        records.append(rec)
    keyframes = {"labels": {f"{k}:{c}": info for (k, c), (_, info) in products.labels.items()},
                 "depth": {str(k): info for k, (_, info) in products.depths.items()}}
    return [{"keyframes": keyframes}] + records


def render_kf(task: dict[str, Any], products: Products, state: dict[str, Any], t: int, k: int, gray: Any,
              bg: np.ndarray, pa: np.ndarray, pb: np.ndarray, near: np.ndarray) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """(d) Depth-augmented keyframes: the reference rendered by its depth, its holes filled from the nearest earlier
    keyframe with depth (causal: a decoder already holds it). The two are measured as one source image, side by
    side, so the attribution has no epipolar split (one fundamental matrix cannot describe two sources)."""
    cal, times, source = state["cal_lens"], state["times"], task["kf_depth"]
    gt = np.asarray(gray[t])
    began = time.perf_counter()
    kd, info = products.source_depth(k, source)
    pose_t = depth.pnp(cal, kd.depth, kd.measured, pa, pb) if kd is not None else None
    if kd is None or pose_t is None:
        return {"measured": False, "reason": "no keyframe depth" if kd is None else "no PnP pose"}, None
    warp, _ = depth.warp_depth(cal, kd.depth, bg[k], pose_t[0], pose_t[1])
    earlier = [j for j in state["companions"] if j != k and times[j] <= times[t]]
    second = max(earlier, key=lambda j: times[j]) if earlier else None
    used = [k]
    gk = np.asarray(gray[k])
    width = gk.shape[1]
    source_gray, source_bg = gk, bg[k]
    map_x, map_y, valid = warp.map_x, warp.map_y, warp.valid
    frame_bytes = 24
    if second is not None:
        kd2, _ = products.source_depth(second, source)
        g2 = np.asarray(gray[second])
        pa2, pb2 = camera.match(camera.features(g2, bg[second]), camera.features(gt, bg[t]))
        pose2 = depth.pnp(cal, kd2.depth, kd2.measured, pa2, pb2) if kd2 is not None else None
        if kd2 is not None and pose2 is not None:
            warp2, _ = depth.warp_depth(cal, kd2.depth, bg[second], pose2[0], pose2[1])
            fill = ~valid & warp2.valid
            map_x = np.where(fill, warp2.map_x + width, map_x).astype(np.float32)
            map_y = np.where(fill, warp2.map_y, map_y).astype(np.float32)
            valid = valid | warp2.valid
            source_gray, source_bg = np.hstack([gk, g2]), np.hstack([bg[k], bg[second]])
            used.append(second)
            frame_bytes = 48
    seconds = time.perf_counter() - began
    res, images = measure(cal, gt, source_gray, camera.Warp(map_x, map_y, valid), bg[t], source_bg, None, near)
    colour = (task.get("colour_bytes") or {}).get(str(k))
    return {**res, "render_s": round(seconds, 4), "frame_bytes": frame_bytes, "keyframes_used": used, "keyframe": k,
            "keyframe_bytes": info.get("png_bytes"), "colour_bytes": colour, "depth_source": source}, images


def write_overlay(publish: Path, t: int, gray: np.ndarray, images: dict[str, Any]) -> None:
    """One sheet: the frame, then the residual flow (0-8 px at 1080p) left by each method."""
    import cv2

    w, h = camera.ANALYSIS_SIZE[0] // 2, camera.ANALYSIS_SIZE[1] // 2
    panels = [cv2.resize(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), (w, h), interpolation=cv2.INTER_AREA)]
    labels = [f"frame {t}"]
    for name, image in images.items():
        if image is None:
            continue
        mag = np.clip(image["flow_mag"] / 8.0, 0, 1)
        heat = cv2.applyColorMap((mag * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)[:, :, ::-1]
        heat[~image["region"]] = (40, 40, 40)
        panels.append(cv2.resize(heat, (w, h), interpolation=cv2.INTER_AREA))
        labels.append(f"{name}: residual flow 0-8 px")
    for panel, label in zip(panels, labels):
        cv2.putText(panel, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    while len(panels) % 4:
        panels.append(np.zeros_like(panels[0]))
    rows = [np.hstack(panels[i:i + 4]) for i in range(0, len(panels), 4)]
    publish.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(publish / f"overlay_{t:05d}.jpg"), np.vstack(rows)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])


# ----------------------------------------------------------------- summaries


def summarize(records: list[dict[str, Any]], keyframes: dict[str, Any], methods: list[str]) -> dict[str, Any]:
    """Per method, over G1's measured frames that are not companions of their own reference."""
    from scipy.stats import spearmanr

    frames = [r for r in records if r["g1"]["measured"] and r.get("methods") is not None and not r.get("companion_of_reference")]
    excluded = sum(1 for r in records if r.get("companion_of_reference"))
    out: dict[str, Any] = {"frames": len(frames), "excluded_companions": excluded}
    for name in methods:
        rows = [r["methods"].get(name) or {"measured": False} for r in frames]
        measured = [m for m in rows if m.get("measured")]

        def med(key: str, rows_: list[dict[str, Any]] = measured) -> float | None:
            values = [m[key] for m in rows_ if m.get(key) is not None]
            return round(float(np.median(values)), 4) if values else None

        entry: dict[str, Any] = {
            "measured": len(measured),
            "explained_share": round(sum(bool(m.get("covered_explained")) for m in rows) / max(1, len(rows)), 4),
            "explained_share_ignoring_holes": round(sum(bool(m.get("explained")) for m in rows) / max(1, len(rows)), 4),
            "explained_share_of_measured": round(float(np.mean([m["explained"] for m in measured])), 4) if measured else None,
            "flow_p90_px_median": med("flow_p90_px"), "flow_p50_px_median": med("flow_p50_px"),
            **{f"within_{x:g}px_median": med(f"flow_within_{x:g}px") for x in camera.FLOW_THRESHOLDS},
            "psnr_median": med("psnr"), "psnr_gain_median": med("psnr_gain"), "psnr_flow_median": med("psnr_flow"),
            "rendered_share_median": med("rendered_share"), "hole_share_median": med("hole_share"),
            "misaligned_near_fg_share_median": med("misaligned_near_fg_share"), "render_s_median": med("render_s"),
            "energy_share_mean": {k: round(float(np.mean([m["energy_share"][k] for m in measured])), 4)
                                  for k in ("exposure", "parallax", "independent", "remainder")} if measured else None,
        }
        frame_bytes: list[float] = [m["frame_bytes"] for m in measured if m.get("frame_bytes") is not None]
        used = {m["keyframe"]: m.get("keyframe_bytes") for m in measured if m.get("keyframe") is not None}
        key_bytes: list[float] = [b for b in used.values() if b is not None]
        if frame_bytes:
            entry["rate"] = {"frame_bytes_mean": round(float(np.mean(frame_bytes)), 2), "keyframes": len(used),
                             "keyframe_bytes_mean": round(float(np.mean(key_bytes)), 1) if key_bytes else 0.0,
                             "bytes_per_frame": round(float(np.mean(frame_bytes)) + float(sum(key_bytes)) / max(1, len(measured)), 2)}
            b16: dict[int, float] = {m["keyframe"]: m["keyframe_bytes16"] for m in measured if m.get("keyframe_bytes16") is not None}
            if b16:
                entry["rate"]["keyframe_bytes16_mean"] = round(float(np.mean(list(b16.values()))), 1)
        for against in ("speed_deg_s", "dt_reference_s"):
            pairs = [(r[against], (r["methods"].get(name) or {}).get("flow_p90_px")) for r in frames
                     if r.get(against) is not None and (r["methods"].get(name) or {}).get("measured")]
            if len(pairs) >= 10:
                rho = spearmanr([p[0] for p in pairs], [p[1] for p in pairs]).statistic
                entry[f"spearman_p90_vs_{against}"] = round(float(rho), 4) if np.isfinite(rho) else None
        out[name] = entry
    rot = [r for r in records if r["g1"]["measured"] and (r.get("rot_g1") or {}).get("measured")]
    close = [abs(r["rot_g1"]["flow_p90_px"] - r["g1"]["flow_p90_px"]) <= REPRODUCE["p90_tolerance_px"] for r in rot]
    out["reproduction"] = {"frames": len(rot), "p90_within_tolerance_share": round(float(np.mean(close)), 4) if close else None}
    out["keyframe_products"] = {
        "label_seconds_median": round(float(np.median([v["seconds"] for v in keyframes["labels"].values()])), 3) if keyframes["labels"] else None,
        "depth_seconds_median": round(float(np.median([v["seconds"] for v in keyframes["depth"].values()])), 3) if keyframes["depth"] else None,
        "depth_measured_share_median": round(float(np.median([v["measured_share"] for v in keyframes["depth"].values()
                                                               if "measured_share" in v])), 4)
        if any("measured_share" in v for v in keyframes["depth"].values()) else None,
        "depth_missing": sum(1 for v in keyframes["depth"].values() if "png_bytes" not in v),
    }
    return out


# ----------------------------------------------------------------- run


def command_run(args: argparse.Namespace) -> int:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    import cv2

    prepared = Path(args.prepared)
    methods = [m for m in args.methods.split(",") if m]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"unknown methods: {unknown}")
    available = {d.name: d for d in (prepared / "publish" / "clips").iterdir() if (d / "meta.json").is_file()}
    metas = {json.loads((d / "meta.json").read_text())["id"]: d for d in available.values()}
    spec = {"clips": [{"id": i, "group": i.split("/")[0], "video": i.split("/")[1]} for i in metas]}
    clips = [c["id"] for c in select_clips(spec, args.select)]
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    allowance = max(2, int(os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 2))
    pool = ProcessPoolExecutor(max_workers=max(1, allowance - 1), mp_context=multiprocessing.get_context("spawn"))
    checkpoints = Path(os.environ["PS_CHECKPOINT_DIR"]) if os.environ.get("PS_CHECKPOINT_DIR") else None
    restored = g1.restore_clips(checkpoints, publish) if checkpoints else {}
    todo = [c for c in clips if c not in restored]
    da3_results: dict[str, Any] = {}
    for clip_id in todo if args.da3 else []:
        record = Path(args.da3) / "publish" / "da3" / g1.safe(clip_id) / "result.json"
        if record.is_file():
            da3_results[clip_id] = json.loads(record.read_text())
    phase_a = {c: pool.submit(register_clip, {"clip_dir": str(metas[c]), "work": str(scratch / "work" / g1.safe(c)),
                                              "limit_frames": args.limit_frames, "self_test": i == 0})
               for i, c in enumerate(todo)}
    chunks: dict[str, list[Future[Any]]] = {}
    registration: dict[str, dict[str, Any]] = {}
    done = 0
    finished: dict[str, dict[str, Any]] = {c: {**restored[c]["result"], "restored_from_checkpoint": True} for c in restored}

    def finish(clip_id: str) -> None:
        nonlocal done
        parts = [f.result() for f in chunks[clip_id]]
        keyframes: dict[str, Any] = {"labels": {}, "depth": {}}
        records: list[dict[str, Any]] = []
        for part in parts:
            for key in ("labels", "depth"):
                keyframes[key].update(part[0]["keyframes"][key])
            records.extend(part[1:])
        records.sort(key=lambda r: r["i"])
        summary = summarize(records, keyframes, methods)
        target = publish / "clips" / g1.safe(clip_id)
        meta = json.loads((metas[clip_id] / "meta.json").read_text())
        result = {"id": clip_id, "group": clip_id.split("/")[0], "methods": methods, "registration": registration[clip_id],
                  "summary": summary, "keyframe_products": keyframes, "frames": records,
                  "inputs": {"frames_sha256": meta["frames_sha256"], "g1_result_sha256": meta["g1_result_sha256"]},
                  "images": sorted(p.name for p in target.glob("overlay_*.jpg"))}
        write_json(target / "result.json", result)
        light = {k: v for k, v in result.items() if k not in ("frames", "keyframe_products")}
        if checkpoints is not None:
            g1.save_clip(checkpoints, target, clip_id, light, None)
        finished[clip_id] = light
        shutil.rmtree(scratch / "work" / g1.safe(clip_id), ignore_errors=True)
        done += 1
        progress(done)

    for clip_id in todo:
        registration[clip_id] = phase_a[clip_id].result()
        n = registration[clip_id]["frames"]
        g1_frames = json.loads((metas[clip_id] / "g1_result.json").read_text())["frames"][:n]
        direct = [r["i"] for r in g1_frames if r["status"] == "direct"]
        overlay = {direct[len(direct) // 2]} if direct else set()
        da3_dir = Path(args.da3) / "publish" / "da3" / g1.safe(clip_id) if args.da3 else None
        colour = da3_results.get(clip_id, {}).get("jpeg_q90_bytes")
        chunks[clip_id] = [pool.submit(evaluate_chunk, {
            "work": str(scratch / "work" / g1.safe(clip_id)), "publish": str(publish / "clips" / g1.safe(clip_id)),
            "lo": lo, "hi": min(n, lo + CHUNK_FRAMES), "methods": methods, "overlay_frames": overlay,
            "g1_frames": g1_frames, "da3": str(da3_dir) if da3_dir else None, "kf_depth": args.kf_depth,
            "colour_bytes": colour}) for lo in range(0, n, CHUNK_FRAMES)]
        for ready in [c for c in list(chunks) if c not in finished and all(f.done() for f in chunks[c])]:
            finish(ready)
    for clip_id in todo:
        if clip_id not in finished:
            finish(clip_id)
    pool.shutdown()
    test = next((finished[c]["registration"].get("self_test") for c in todo if finished[c]["registration"].get("self_test")), None)
    write_json(stage_dir() / "g1d.json", {
        "prepared": str(prepared), "select": args.select, "limit_frames": args.limit_frames, "methods": methods,
        "da3": args.da3, "kf_depth": args.kf_depth,
        "decision": DECISION, "settings": {
            "COMPANION_S": COMPANION_S, "COMPANION_RANGE_S": COMPANION_RANGE_S, "NEAR_FG_PX": NEAR_FG_PX,
            "MIN_ANGLE_DEG": depth.MIN_ANGLE_DEG, "TRI_REPROJECTION_PX": depth.TRI_REPROJECTION_PX, "PNP_PX": depth.PNP_PX,
            "MODE_FILTER": depth.MODE_FILTER, "DEPTH_LEVELS": depth.DEPTH_LEVELS, "CAL_HFOV_DEG": depth.CAL_HFOV_DEG,
            "CAL_K1": depth.CAL_K1, "REPRODUCE": REPRODUCE},
        "self_test": test, "restored_clips": sorted(restored), "opencv": cv2.__version__,
        "results": [finished[c] for c in clips],
    })
    return 0


# ----------------------------------------------------------------- validate


def unit(value: Any) -> bool:
    return value is None or (isinstance(value, (int, float)) and 0.0 <= value <= 1.0)


def validate_stage(stage: Path) -> dict[str, bool]:
    import tarfile

    result = json.loads((stage / "g1d.json").read_text())
    rows = result["results"]
    with tarfile.open(stage / "published.tar") as tar:
        names = set(tar.getnames())
    fresh = [r for r in rows if not r.get("restored_from_checkpoint")]
    checks = {
        "clips_evaluated": bool(rows),
        "every_clip_has_result_and_overlay": all(f"publish/clips/{g1.safe(r['id'])}/result.json" in names and r["images"]
                                                 and all(f"publish/clips/{g1.safe(r['id'])}/{i}" in names for i in r["images"])
                                                 for r in rows),
        "frames_match_prepared_archive": all(r["registration"]["frames_ok"] for r in rows),
        "g1_references_reproduced": all(r["registration"]["reference_share"] >= REPRODUCE["reference_share"] for r in rows),
        "g1_rotation_residual_reproduced": all((r["summary"]["reproduction"]["p90_within_tolerance_share"] or 0)
                                               >= REPRODUCE["p90_share"] for r in rows if "rot" in result["methods"]),
        "self_test_recovers_known_depth_and_pose": bool(result["self_test"] and result["self_test"]["passed"]) or not fresh,
        "lens_calibrated": all(r["registration"]["calibration"].get("observable") is not None for r in rows),
        "shares_in_unit_interval": all(unit(r["summary"][m]["explained_share"]) and unit(r["summary"][m]["rendered_share_median"])
                                       for r in rows for m in result["methods"]),
        "every_method_measured": all(any(r["summary"][m]["measured"] > 0 for r in rows) for m in result["methods"]),
        "rates_positive": all((r["summary"][m].get("rate") or {"bytes_per_frame": 1})["bytes_per_frame"] > 0
                              for r in rows for m in result["methods"]),
    }
    return checks


def command_validate(args: argparse.Namespace) -> int:
    stage = stage_dir()
    checks = validate_prepare(stage) if args.command == "validate-prepare" else validate_stage(stage)
    report = {"passed": all(checks.values()), "checks": checks}
    target = os.environ.get("PS_VALIDATION_PATH")
    if target:
        write_json(Path(target), report)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


# ----------------------------------------------------------------- report


def group_table(rows: list[dict[str, Any]], methods: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group in DECISION["groups"]:
        clips = [r for r in rows if r["group"] == group]
        if not clips:
            continue
        table: dict[str, Any] = {"clips": len(clips)}
        for m in methods:
            s = [r["summary"][m] for r in clips if m in r["summary"]]

            def med(key: str, sub: str | None = None) -> float | None:
                raw = [(x.get(sub) or {}).get(key) if sub else x.get(key) for x in s]
                values: list[float] = [v for v in raw if v is not None]
                return round(float(np.median(values)), 4) if values else None

            shares = [x["explained_share"] for x in s]
            table[m] = {
                "explained_share_median_clip": med("explained_share"),
                "clips_at_least_half": sum(v >= DECISION["meaningful_median_clip_share"] for v in shares),
                "clips_holding_g1_rule": sum(v >= 0.90 for v in shares),
                "flow_p90_px_median": med("flow_p90_px_median"), "psnr_flow_median": med("psnr_flow_median"),
                "rendered_share_median": med("rendered_share_median"), "hole_share_median": med("hole_share_median"),
                "near_fg_share_median": med("misaligned_near_fg_share_median"), "render_s_median": med("render_s_median"),
                "bytes_per_frame_median": med("bytes_per_frame", "rate"),
                "keyframe_bytes_median": med("keyframe_bytes_mean", "rate"),
                "energy_share_mean": {k: round(float(np.mean([x["energy_share_mean"][k] for x in s if x.get("energy_share_mean")])), 4)
                                      for k in ("exposure", "parallax", "independent", "remainder")}
                if any(x.get("energy_share_mean") for x in s) else None,
                "spearman_p90_vs_speed_median": med("spearman_p90_vs_speed_deg_s"),
                "spearman_p90_vs_dt_median": med("spearman_p90_vs_dt_reference_s"),
            }
        out[group] = table
    return out


def decide(table: dict[str, Any]) -> dict[str, Any]:
    bar = DECISION["meaningful_median_clip_share"]

    def meaningful(method: str, group: str) -> bool:
        cell = table.get(group, {}).get(method)
        return bool(cell and cell["explained_share_median_clip"] is not None and cell["explained_share_median_clip"] >= bar)

    groups = [g for g in DECISION["groups"] if g in table]
    ceiling = {g: any(meaningful(m, g) for m in DECISION["step1_measures"]) for g in groups}
    out: dict[str, Any] = {"ceiling_meaningful": ceiling, "complete": len(groups) == len(DECISION["groups"])}
    if not any(ceiling.values()):
        out.update(step=1, verdict="no static-scene warp at hand brings VISOR under 2 px on a meaningful share; (c) and (d) are not "
                   "run; egocentric goes to G5")
        return out
    qualifying = [m for m in DECISION["realizable"] if all(meaningful(m, g) for g in groups)]
    rates = {m: min((table[g][m]["bytes_per_frame_median"] or math.inf) for g in groups) for m in qualifying}
    out.update(step=2, qualifying=qualifying)
    if qualifying:
        out["chosen"] = min(qualifying, key=lambda m: rates[m])
    else:
        out["verdict"] = "the ceiling reaches a meaningful share but no realizable method does in both groups so far"
    return out


AGE_BINS_S = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)


def reference_ages(paths: list[Path], methods: list[str]) -> dict[str, Any]:
    """Explained share and median p90 by the age of the reference (seconds), per group, from the per-frame
    records in runs' ``published.tar`` (the same frames as the report: G1-measured, companions excluded).
    Bins mix clips, so a bin's share also depends on which clips have frames at that age."""
    import tarfile

    cells: dict[tuple[str, int], dict[str, list[tuple[bool, float | None]]]] = {}
    for path in paths:
        with tarfile.open(path) as tar:
            for member in tar.getmembers():
                if not member.name.endswith("/result.json"):
                    continue
                handle = tar.extractfile(member)
                assert handle is not None
                result = json.load(handle)
                for f in result["frames"]:
                    if not f["g1"]["measured"] or f.get("methods") is None or f.get("companion_of_reference"):
                        continue
                    b = max(i for i, lo in enumerate(AGE_BINS_S[:-1]) if f["dt_reference_s"] >= lo)
                    cell = cells.setdefault((result["group"], b), {m: [] for m in methods})
                    for m in methods:
                        x = f["methods"].get(m) or {}
                        cell[m].append((bool(x.get("covered_explained")), x.get("flow_p90_px")))
    out: dict[str, Any] = {"bins_s": list(AGE_BINS_S), "groups": {}}
    for (group, b), cell in sorted(cells.items()):
        row: dict[str, Any] = {"from_s": AGE_BINS_S[b], "to_s": AGE_BINS_S[b + 1], "frames": len(cell[methods[0]])}
        for m in methods:
            p90 = [q for _, q in cell[m] if q is not None]
            row[m] = {"explained_share": round(sum(e for e, _ in cell[m]) / max(1, len(cell[m])), 4),
                      "flow_p90_px_median": round(float(np.median(p90)), 3) if p90 else None}
        out["groups"].setdefault(group, []).append(row)
    return out


def command_ages(args: argparse.Namespace) -> int:
    methods = args.methods.split(",")
    ages = reference_ages([Path(p) for p in args.published], methods)
    ages["published"] = [{"path": p, "sha256": file_sha256(Path(p))} for p in args.published]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "g1d-ages.json", ages)
    for group, rows in ages["groups"].items():
        print(group)
        for row in rows:
            cells = " | ".join(f"{m} {100 * row[m]['explained_share']:3.0f}% p90 {row[m]['flow_p90_px_median']}" for m in methods)
            print(f"  {row['from_s']}-{row['to_s']} s  n={row['frames']}  {cells}")
    return 0


def command_report(args: argparse.Namespace) -> int:
    rows: dict[str, dict[str, Any]] = {}
    methods: list[str] = []
    runs = []
    for path in args.result:
        result = json.loads(Path(path).read_text())
        runs.append({"path": path, "sha256": file_sha256(Path(path)), "select": result["select"],
                     "limit_frames": result["limit_frames"], "clips": len(result["results"])})
        methods = methods or result["methods"]
        for r in result["results"]:
            rows[r["id"]] = r
    table = group_table(list(rows.values()), methods)
    report = {"runs": runs, "decision_rule": DECISION, "groups": table, "decision": decide(table),
              "clips": {i: {"group": r["group"], "frames": r["summary"]["frames"],
                            **{m: r["summary"][m]["explained_share"] for m in methods},
                            "p90": {m: r["summary"][m]["flow_p90_px_median"] for m in methods},
                            "calibration": {k: r["registration"]["calibration"].get(k) for k in ("hfov_deg", "k1", "error_px", "error_at_epic_fields_px")}}
                        for i, r in sorted(rows.items())}}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "g1d-report.json", report)
    lines = ["| Group | Method | median clip explained | clips ≥ 50% | p90 px | holes | bytes/frame | render s | parallax / independent / remainder |",
             "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for group, cells in table.items():
        for m in methods:
            c = cells[m]
            e = c["energy_share_mean"] or {}
            lines.append(f"| {group} ({cells['clips']}) | {m} | {c['explained_share_median_clip']} | {c['clips_at_least_half']} | "
                         f"{c['flow_p90_px_median']} | {c['hole_share_median']} | {c['bytes_per_frame_median']} | "
                         f"{c['render_s_median']} | {e.get('parallax')} / {e.get('independent')} / {e.get('remainder')} |")
    (out / "g1d-report.md").write_text("\n".join(lines) + "\n\n" + json.dumps(report["decision"], indent=1) + "\n")
    print("\n".join(lines))
    print(json.dumps(report["decision"], indent=1))
    return 0


# ----------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--clips", required=True, help="G1's clips.json")
    prep.add_argument("--g1", required=True, help="G1's published clips/<id>/{result.json,masks.rle}")
    prep.add_argument("--select", required=True)
    prep.add_argument("--limit-seconds", type=float, default=0.0)
    prep.add_argument("--source", nargs=2, action="append", default=[], metavar=("NAME", "PATH"))
    prep.add_argument("--visor-archive", required=True)
    prep.add_argument("--visor-fill", required=True)
    prep.add_argument("--hand-objects", required=True)
    sub.add_parser("validate-prepare")
    run = sub.add_parser("run")
    run.add_argument("--prepared", required=True, help="extracted published.tar of a prepare job")
    run.add_argument("--select", required=True)
    run.add_argument("--limit-frames", type=int, default=0)
    run.add_argument("--methods", default=",".join(METHODS_AB))
    run.add_argument("--da3", help="extracted published.tar of an experiments.background.da3 job")
    run.add_argument("--kf-depth", default="tri", choices=("tri", "da3", "da3_tri"), help="depth source of kf")
    sub.add_parser("validate")
    report = sub.add_parser("report")
    report.add_argument("--result", action="append", required=True)
    report.add_argument("--out", required=True)
    ages = sub.add_parser("ages", help="explained share by the age of the reference, from runs' published.tar")
    ages.add_argument("--published", action="append", required=True)
    ages.add_argument("--methods", default="rot,planes4,epi,tri")
    ages.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    handlers = {"prepare": command_prepare, "validate-prepare": command_validate, "run": command_run,
                "validate": command_validate, "report": command_report, "ages": command_ages}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
