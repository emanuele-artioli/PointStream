"""PLAN step G1e: can a VISOR background reference refreshed often enough reach the 2 px bar?

    python -m experiments.background.g1e prepare --clips CLIPS.JSON --g1 DIR --select all|pilot|ID,ID \\
        --limit-seconds S --source NAME PATH ...
    python -m experiments.background.g1e validate-prepare
    python -m experiments.background.g1e run --prepared DIR --ott DIR --clips CLIPS.JSON --visor-fill DIR \\
        --select all|pilot|rest|rest:I/N|ID,ID [--ages frame,0.1,1] [--limit-targets N]
    python -m experiments.background.g1e validate
    python -m experiments.background.g1e report --result g1e.json ... --out DIR

``prepare`` stores the OpenTTGames clips of G1 (the measurement floor) the way
G1d stored VISOR's (`g1d.prepare_clip`). ``run`` scores fixed targets
(`targets`) against references forced to each age (`pairs`), with G1d's warps
and measure, on G1d's prepared VISOR archive and on that OpenTTGames archive.
Each pair has its own reference, so the reference's products (label map,
triangulated depth) are computed per pair. The oracle is the per-pair best of
`ORACLE_OF`. On the stretch frames inside their evaluation window, every pair
is also scored with the window's dense masks as the foreground (the mask
check). Per clip it writes ``publish/clips/<id>/result.json``; ``g1e.json`` in
``PS_STAGE_DIR`` holds the summaries. Finished clips are checkpointed
(`g1.save_clip`) and restored on a declared resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import shutil
import time
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from experiments.background import camera, depth, g1, g1d
from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json

#: The decision rule, fixed before any run (docs/experiments.md, G1e entry).
DECISION: dict[str, Any] = {
    "explained_px": camera.EXPLAINED_PX,
    "max_hole_share": g1d.DECISION["max_hole_share"],
    "meaningful_median_clip_share": 0.50,
    "groups": ["visor-window", "visor-long"],
    "gate_min_age_s": 0.1,  # step 1: the oracle must be meaningful at >= 0.1 s on the windows
    "sendable": ["planes4", "tri"],
    "mask_check_max_gap": 0.10,  # step 2: dense and sam_text explained shares may differ by <= 10 points
    "refine_age_s": 0.3,  # step 4
    "floor_px": camera.EXPLAINED_PX,  # step 5: OpenTTGames h1 p90 at 1 s and VISOR's forward-backward p90
    "floor_measure": "epi",  # the warp whose forward-backward p90 the floor reads
}
SEED = "pointstream-g1e"
AGES = ("frame", 0.1, 1.0)
VISOR_METHODS = ("h1", "planes4", "epi", "tri")
OTT_METHODS = ("h1",)
ORACLE_OF = ("epi", "planes4", "tri")
FIRST_TARGET_S = 1.0
WINDOW_STRIDE = 10  # frames between window targets (0.2 s at 50 fps)
TARGET_EVERY_S = 1.0  # stretches and OpenTTGames
PILOT_OTT = 1
CHUNK_PAIRS = 6


# ----------------------------------------------------------------- selection


def rank(group: str, video: str) -> str:
    return hashlib.sha256(f"{SEED}:{group}:{video}".encode()).hexdigest()


def select_ott(spec: dict[str, Any], which: str) -> list[dict[str, Any]]:
    """The OpenTTGames clips of G1; ``pilot`` is the first by rank."""
    ott = sorted((c for c in spec["clips"] if c["group"] == "ott"), key=lambda c: rank(c["group"], c["video"]))
    if which == "all":
        return ott
    if which in ("pilot", "smoke"):
        return ott[:PILOT_OTT]
    by_id = {c["id"]: c for c in ott}
    missing = [w for w in which.split(",") if w not in by_id]
    if missing:
        raise SystemExit(f"not OpenTTGames clips of G1: {missing}")
    return [by_id[w] for w in which.split(",")]


def select_clips(spec: dict[str, Any], which: str) -> list[dict[str, Any]]:
    """G1d's 44 VISOR clips and the OpenTTGames clips. ``pilot`` and ``smoke``: G1d's pilot and the first
    OpenTTGames clip. ``rest``: the others; ``rest:I/N``: the I-th of N shards of them (1-based)."""
    visor = g1d.select_clips(spec, "all")
    ott = select_ott(spec, "all")
    pilot = g1d.select_clips(spec, "pilot") + ott[:PILOT_OTT]
    chosen = visor + ott
    if which == "all":
        return chosen
    if which in ("pilot", "smoke"):
        return pilot
    if which == "rest" or which.startswith("rest:"):
        rest = [c for c in chosen if c not in pilot]
        if which == "rest":
            return rest
        part, parts = (int(x) for x in which.split(":")[1].split("/"))
        if not 1 <= part <= parts:
            raise SystemExit(f"bad shard {which}")
        return rest[part - 1::parts]
    by_id = {c["id"]: c for c in chosen}
    missing = [w for w in which.split(",") if w not in by_id]
    if missing:
        raise SystemExit(f"not in G1e's selection: {missing}")
    return [by_id[w] for w in which.split(",")]


# ----------------------------------------------------------------- targets and pairs


def targets(times: np.ndarray, group: str) -> list[int]:
    """Fixed targets: windows every `WINDOW_STRIDE` frames from `FIRST_TARGET_S`; other clips the frame nearest
    each whole second from `FIRST_TARGET_S`."""
    times = np.asarray(times, float)
    if group == "visor-window":
        first = int(np.searchsorted(times, FIRST_TARGET_S - 1e-6))
        return list(range(first, len(times), WINDOW_STRIDE))
    out: list[int] = []
    at = FIRST_TARGET_S
    while at <= times[-1] + 1e-6:
        i = int(np.argmin(np.abs(times - at)))
        if not out or i != out[-1]:
            out.append(i)
        at += TARGET_EVERY_S
    return out


def gaps(times: np.ndarray, ages: list[Any]) -> dict[int, list[Any]]:
    """Reference gap in frames for each age (``"frame"`` is one frame); ages that land on the same gap share it."""
    step = float(np.median(np.diff(np.asarray(times, float))))
    out: dict[int, list[Any]] = {}
    for age in ages:
        gap = 1 if age == "frame" else max(1, int(round(float(age) / step)))
        out.setdefault(gap, []).append(age)
    return out


def parse_ages(text: str) -> list[Any]:
    return [a if a == "frame" else float(a) for a in text.split(",") if a]


# ----------------------------------------------------------------- prepare (OpenTTGames)


def command_prepare(args: argparse.Namespace) -> int:
    args.visor_archive = args.visor_fill = args.hand_objects = None
    return g1d.command_prepare(args, select_ott)


def validate_prepare(stage: Path) -> dict[str, bool]:
    checks = g1d.validate_prepare(stage)
    rows = json.loads((stage / "prepare.json").read_text())["results"]
    checks["only_openttgames"] = all(r["group"] == "ott" for r in rows)
    checks["openttgames_use_g1_sam_masks"] = all(r["masks"].get("tier") == g1.SAM_TIER for r in rows)
    return checks


# ----------------------------------------------------------------- floor: forward-backward consistency


def forward_backward_p90(target: np.ndarray, warped: np.ndarray, region: np.ndarray) -> float | None:
    """p90 at 1080p of |f(x) + b(x + f(x))| on the textured region, f the DIS flow from ``target`` to ``warped``
    and b the flow back: how far the measure's own flow disagrees with itself on this pair."""
    import cv2

    tex = region & camera.textured(target)
    if tex.sum() < 100:
        return None
    f = camera.flow(target, warped)
    b = camera.flow(warped, target)
    h, w = target.shape
    grid = camera.pixel_grid(w, h).astype(np.float32)
    bx = cv2.remap(b[..., 0], grid[..., 0] + f[..., 0], grid[..., 1] + f[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    by = cv2.remap(b[..., 1], grid[..., 0] + f[..., 0], grid[..., 1] + f[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    err = np.hypot(f[..., 0] + bx, f[..., 1] + by) * camera.TO_1080
    return round(float(np.percentile(err[tex], 90)), 3)


# ----------------------------------------------------------------- per clip: frames to scratch (spawned)


def prepare_work(task: dict[str, Any]) -> dict[str, Any]:
    """Decode a prepared clip to grey frames, its foreground and (stretches) the window's dense masks."""
    import cv2

    cv2.setNumThreads(1)
    began = time.time()
    clip_dir, work = Path(task["clip_dir"]), Path(task["work"])
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    meta = json.loads((clip_dir / "meta.json").read_text())
    n = meta["frames"]
    width, height = camera.ANALYSIS_SIZE
    gray = np.lib.format.open_memmap(work / "gray.npy", mode="w+", dtype=np.uint8, shape=(n, height, width))
    digest = hashlib.sha256()
    count = 0
    for i, rgb in enumerate(g1d.decode_ffv1(clip_dir / "frames.mkv", 2)):
        digest.update(np.ascontiguousarray(rgb).tobytes())
        gray[i] = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        count += 1
    gray.flush()
    np.save(work / "fg.npy", g1d.load_foreground(clip_dir / "foreground.npz"))
    dense_frames = 0
    if task.get("spec_clip") and task["spec_clip"]["masks"].get("window_item"):
        raw, _ = g1.window_dense(g1.plan(task["spec_clip"], meta["limit_seconds"]), Path(task["visor_fill"]))
        np.savez(work / "dense.npz", **{str(i): g1.dilate(d) for i, d in raw.items()})
        dense_frames = len(raw)
    return {"frames": n, "frames_ok": digest.hexdigest() == meta["frames_sha256"] and count == n,
            "dense_frames": dense_frames, "seconds": round(time.time() - began, 2)}


# ----------------------------------------------------------------- per pair (spawned)


def companions_for(lens: camera.Lens, r: int, gray: Any, bg: np.ndarray, times: np.ndarray,
                   feats: dict[int, camera.Features]) -> list[dict[str, Any]]:
    """G1d's companions of a keyframe (`g1d.choose_companions`) without G1's registration: the frames nearest
    0.5 s and 1.0 s after ``r`` (before, at a clip's end) that a homography with enough inliers matches, trying
    the six nearest candidates for each."""
    lo, hi = g1d.COMPANION_RANGE_S

    def feat(j: int) -> camera.Features:
        if j not in feats:
            feats[j] = camera.features(np.asarray(gray[j]), bg[j])
        return feats[j]

    candidates = [j for j in range(len(times)) if j != r and lo <= abs(times[j] - times[r]) <= hi]
    chosen: list[dict[str, Any]] = []
    for target in g1d.COMPANION_S:
        order = sorted(candidates, key=lambda j: (abs(abs(times[j] - times[r]) - target) + (0.0 if times[j] > times[r] else 0.25), j))
        for j in order[:6]:
            if any(c["index"] == j for c in chosen):
                continue
            pa, pb = camera.match(feat(r), feat(j))
            H, keep = depth.homography_u(lens, pa, pb)
            if H is not None and keep.sum() >= camera.MIN_INLIERS:
                chosen.append({"index": int(j), "pa": pa, "pb": pb, "dt": round(float(times[j] - times[r]), 3),
                               "h_inliers": int(keep.sum())})
                break
    return chosen


def oracle(methods: dict[str, Any]) -> dict[str, Any]:
    """Per-pair best of `ORACLE_OF`: explained if any explains the pair; its p90 is the least of theirs among
    those within the hole bar."""
    parts = {m: methods.get(m) or {} for m in ORACLE_OF}
    explained = [m for m, x in parts.items() if x.get("covered_explained")]
    covered = {m: x["flow_p90_px"] for m, x in parts.items()
               if x.get("measured") and x.get("hole_share") is not None and x["hole_share"] <= DECISION["max_hole_share"]}
    best = min(covered, key=lambda m: covered[m]) if covered else None
    return {"covered_explained": bool(explained), "explained_by": explained, "best": best,
            "flow_p90_px": covered.get(best) if best else None, "measured": bool(covered),
            "epi_matches": bool(parts["epi"].get("covered_explained")) == bool(explained)}


def evaluate_pair(lens: camera.Lens, gray: Any, fg: Any, times: np.ndarray, t: int, r: int, methods: tuple[str, ...],
                  products: g1d.Products, feats: dict[int, camera.Features], near_kernel: np.ndarray) -> dict[str, Any]:
    import cv2

    bg = products.bg
    gt, gr = np.asarray(gray[t]), np.asarray(gray[r])
    rec: dict[str, Any] = {"t": t, "r": r, "age_s": round(float(times[t] - times[r]), 4)}
    for j in (t, r):
        if j not in feats:
            feats[j] = camera.features(np.asarray(gray[j]), bg[j])
    pa, pb = camera.match(feats[r], feats[t])
    H, keep = depth.homography_u(lens, pa, pb)
    rec["inliers"] = int(keep.sum()) if H is not None else 0
    if H is None:
        rec["matched"] = False
        return rec
    rec["matched"] = True
    near = (cv2.dilate(np.asarray(fg[t]).astype(np.uint8), near_kernel) > 0) & ~np.asarray(fg[t])
    pair = camera.translation_pair(lens, pa, pb)
    F_cal = pair["F"] if pair else None
    F_attr = F_cal if pair and pair["prefers_f"] else None
    out: dict[str, Any] = {}
    images: dict[str, Any] = {}

    def keep_result(name: str, res: dict[str, Any], image: Any, seconds: float, **extra: Any) -> None:
        if image is not None and res.get("measured"):
            res["fb_p90_px"] = forward_backward_p90(gt, image["warped"], image["region"])
        out[name] = {**res, "render_s": round(seconds, 4), **extra}
        images[name] = image

    began = time.perf_counter()
    warp = depth.warp_h(lens, H)
    seconds = time.perf_counter() - began
    warped_bg = cv2.remap(bg[r].astype(np.uint8), warp.map_x, warp.map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT) > 0
    h1_region = warp.valid & bg[t] & warped_bg
    if "h1" in methods:
        res, image = g1d.measure(lens, gt, gr, warp, bg[t], bg[r], F_attr, near)
        keep_result("h1", res, image, seconds, frame_bytes=32)
    if "planes4" in methods:
        labels, info = products.label_map(r, 4)
        if labels is None:
            out["planes4"] = {"measured": False, "reason": "no label map"}
        else:
            pos, ok = depth.dense_correspondence(lens, gr, gt, np.linalg.inv(H), bg[r], bg[t])
            sa, sb = depth.dense_samples(gr, pos, ok)
            at = labels[sa[:, 1].astype(int), sa[:, 0].astype(int)]
            homographies, fallback = [], 0
            for i in range(int(labels.max()) + 1):
                sel = at == i
                Hi = depth.homography_u(lens, sa[sel], sb[sel])[0] if sel.sum() >= camera.MIN_INLIERS else None
                if Hi is None:
                    Hi, fallback = H, fallback + 1
                homographies.append(Hi)
            began = time.perf_counter()
            warp_p, _ = depth.warp_planes(lens, labels, homographies)
            seconds = time.perf_counter() - began
            res, image = g1d.measure(lens, gt, gr, warp_p, bg[t], bg[r], F_attr, near)
            keep_result("planes4", res, image, seconds, frame_bytes=32 * len(homographies), fallback_planes=fallback,
                        reference_bytes=info.get("png_bytes"))
    if "epi" in methods and F_cal is not None:
        began = time.perf_counter()
        warp_e = depth.warp_epipolar(lens, gt, gr, H, F_cal, bg[t], bg[r])
        seconds = time.perf_counter() - began
        res, image = g1d.measure(lens, gt, gr, warp_e, bg[t], bg[r], F_attr, near)
        keep_result("epi", res, image, seconds)
    if "tri" in methods:
        kd, info = products.key_depth(r)
        pose_t = depth.pnp(lens, kd.depth, kd.measured, pa, pb) if kd is not None else None
        if kd is None or pose_t is None:
            out["tri"] = {"measured": False, "reason": "no reference depth" if kd is None else "no PnP pose"}
        else:
            began = time.perf_counter()
            warp_d, _ = depth.warp_depth(lens, kd.depth, bg[r], pose_t[0], pose_t[1])
            seconds = time.perf_counter() - began
            res, image = g1d.measure(lens, gt, gr, warp_d, bg[t], bg[r], F_attr, near)
            keep_result("tri", res, image, seconds, frame_bytes=24, pnp_inliers=pose_t[2], reference_bytes=info.get("png_bytes"))
    for name, image in images.items():
        entry = out[name]
        if image is None or not entry.get("measured"):
            continue
        hole = float((h1_region & ~image["region"]).sum() / max(1, h1_region.sum()))
        entry["hole_share"] = round(hole, 4)
        entry["covered_explained"] = bool(entry["explained"] and hole <= DECISION["max_hole_share"])
    if all(m in methods for m in ORACLE_OF):
        out["oracle"] = oracle(out)
    rec["methods"] = out
    return rec


def evaluate_chunk(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Pairs of one clip, with one foreground (``mask``: the clip's own, or ``dense`` on window frames)."""
    import cv2

    cv2.setNumThreads(1)
    work = Path(task["work"])
    gray = np.load(work / "gray.npy", mmap_mode="r")
    times = np.asarray(task["times"], float)
    if task["mask"] == "dense":
        dense = np.load(work / "dense.npz")
        fg = np.zeros(gray.shape, bool)
        for key in dense.files:
            fg[int(key)] = dense[key]
    else:
        fg = np.asarray(np.load(work / "fg.npy", mmap_mode="r"))
    bg = ~fg
    lens = camera.Lens(*task["lens"])
    feats: dict[int, camera.Features] = {}
    state: dict[str, Any] = {"cal_lens": lens, "companions": {}}
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g1d.NEAR_FG_PX + 1, 2 * g1d.NEAR_FG_PX + 1))
    records = []
    methods = tuple(task["methods"])
    for t, r, ages, roles in task["pairs"]:
        began = time.perf_counter()
        products = g1d.Products(state, gray, bg)
        if methods != OTT_METHODS:
            state["companions"][r] = companions_for(lens, r, gray, bg, times, feats)
        rec = evaluate_pair(lens, gray, fg, times, t, r, methods, products, feats, kernel)
        rec.update(ages=ages, roles=roles, mask=task["mask"], companions=len(state["companions"].get(r, [])),
                   reference_products={"labels": products.labels.get((r, 4), (None, {}))[1],
                                       "depth": products.depths.get(r, (None, {}))[1]},
                   pair_seconds=round(time.perf_counter() - began, 3))
        records.append(rec)
        state["companions"].pop(r, None)
        for j in [j for j in feats if j != t]:  # the next pair usually shares the target
            feats.pop(j)
    return records


# ----------------------------------------------------------------- summaries


#: Summaries per clip: the fixed targets with the clip's own foreground (the result), and the mask check's pairs
#: with the window's dense masks and with the own foreground.
VIEWS = {"own": ("own", "target"), "dense": ("dense", "mask_check"), "own_on_dense": ("own", "mask_check")}


def summarize(records: list[dict[str, Any]], methods: list[str], ages: list[Any]) -> dict[str, Any]:
    """Per view (`VIEWS`), age (as given) and method: shares over the view's pairs (an unmatched pair is not
    explained)."""
    out: dict[str, Any] = {}
    for view, (mask, role) in VIEWS.items():
        per_mask: dict[str, Any] = {}
        for age in ages:
            rows = [r for r in records if r["mask"] == mask and role in r["roles"] and age in r["ages"]]
            if not rows:
                continue
            entry: dict[str, Any] = {"pairs": len(rows), "matched_share": round(float(np.mean([r["matched"] for r in rows])), 4),
                                     "age_s_median": round(float(np.median([r["age_s"] for r in rows])), 4)}
            names = list(methods) + (["oracle"] if all(m in methods for m in ORACLE_OF) else [])
            for name in names:
                cells = [(r.get("methods") or {}).get(name) or {} for r in rows]
                measured = [c for c in cells if c.get("measured")]

                def med(key: str, cells_: list[dict[str, Any]] = measured) -> float | None:
                    values = [c[key] for c in cells_ if c.get(key) is not None]
                    return round(float(np.median(values)), 4) if values else None

                cell: dict[str, Any] = {
                    "measured": len(measured),
                    "explained_share": round(sum(bool(c.get("covered_explained")) for c in cells) / len(rows), 4),
                    "flow_p90_px_median": med("flow_p90_px"),
                }
                if name == "oracle":
                    cell["epi_matches_share"] = round(float(np.mean([c["epi_matches"] for c in measured])), 4) if measured else None
                    cell["best"] = {m: sum(c.get("best") == m for c in measured) for m in ORACLE_OF}
                else:
                    cell.update(hole_share_median=med("hole_share"), fb_p90_px_median=med("fb_p90_px"),
                                psnr_flow_median=med("psnr_flow"), render_s_median=med("render_s"),
                                near_fg_share_median=med("misaligned_near_fg_share"))
                    energy = [c["energy_share"] for c in measured if c.get("energy_share")]
                    if energy:
                        cell["energy_share_mean"] = {k: round(float(np.mean([e[k] for e in energy])), 4)
                                                     for k in ("exposure", "parallax", "independent", "remainder")}
                    frame_bytes = [c["frame_bytes"] for c in measured if c.get("frame_bytes") is not None]
                    ref_bytes = [c["reference_bytes"] for c in measured if c.get("reference_bytes") is not None]
                    if frame_bytes:
                        cell["rate"] = {"frame_bytes_mean": round(float(np.mean(frame_bytes)), 2),
                                        "reference_bytes_mean": round(float(np.mean(ref_bytes)), 1) if ref_bytes else 0.0}
                entry[name] = cell
            per_mask[str(age)] = entry
        if per_mask:
            out[view] = per_mask
    seconds = [r["pair_seconds"] for r in records]
    out["pair_seconds_median"] = round(float(np.median(seconds)), 3) if seconds else None
    out["pair_seconds_total"] = round(float(np.sum(seconds)), 1)
    return out


# ----------------------------------------------------------------- run


def command_run(args: argparse.Namespace) -> int:
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    import cv2

    ages = parse_ages(args.ages)
    spec = json.loads(Path(args.clips).read_text())
    by_id = {c["id"]: c for c in spec["clips"]}
    dirs: dict[str, Path] = {}
    for root in (Path(args.prepared), Path(args.ott)):
        for d in (root / "publish" / "clips").iterdir():
            if (d / "meta.json").is_file():
                dirs[json.loads((d / "meta.json").read_text())["id"]] = d
    clips = [c["id"] for c in select_clips(spec, args.select)]
    missing = [c for c in clips if c not in dirs]
    if missing:
        raise SystemExit(f"not in the prepared archives: {missing}")
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    allowance = max(2, int(os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 2))
    pool = ProcessPoolExecutor(max_workers=max(1, allowance - 1), mp_context=multiprocessing.get_context("spawn"))
    checkpoints = Path(os.environ["PS_CHECKPOINT_DIR"]) if os.environ.get("PS_CHECKPOINT_DIR") else None
    restored = g1.restore_clips(checkpoints, publish) if checkpoints else {}
    todo = [c for c in clips if c not in restored]
    work = {c: scratch / "work" / g1.safe(c) for c in todo}
    phase_a = {c: pool.submit(prepare_work, {"clip_dir": str(dirs[c]), "work": str(work[c]),
                                             "spec_clip": by_id[c] if by_id[c]["group"] == "visor-long" else None,
                                             "visor_fill": args.visor_fill}) for c in todo}
    finished: dict[str, dict[str, Any]] = {c: {**restored[c]["result"], "restored_from_checkpoint": True} for c in restored}
    chunks: dict[str, list[Future[Any]]] = {}
    plans: dict[str, dict[str, Any]] = {}
    done = 0

    def finish(clip_id: str) -> None:
        nonlocal done
        records = sorted((r for f in chunks[clip_id] for r in f.result()), key=lambda r: (r["mask"], r["t"], r["r"]))
        plan = plans[clip_id]
        summary = summarize(records, plan["methods"], ages)
        target = publish / "clips" / g1.safe(clip_id)
        result = {"id": clip_id, "group": by_id[clip_id]["group"], "ages": ages, "methods": plan["methods"],
                  "lens": plan["lens"], "work": plan["work"], "targets": plan["targets"], "gaps": plan["gaps"],
                  "dense_targets": plan["dense_targets"], "summary": summary, "pairs": records,
                  "inputs": {k: plan["meta"][k] for k in ("frames_sha256", "g1_result_sha256")}}
        write_json(target / "result.json", result)
        light = {k: v for k, v in result.items() if k != "pairs"}
        if checkpoints is not None:
            g1.save_clip(checkpoints, target, clip_id, light, None)
        finished[clip_id] = light
        shutil.rmtree(work[clip_id], ignore_errors=True)
        done += 1
        progress(done)

    for clip_id in todo:
        prepared = phase_a[clip_id].result()
        meta = json.loads((dirs[clip_id] / "meta.json").read_text())
        group = by_id[clip_id]["group"]
        times = np.asarray(meta["times"], float)
        if group == "ott":
            g1_lens = json.loads((dirs[clip_id] / "g1_result.json").read_text())["lens"]
            lens, methods = (float(g1_lens["f"]), float(g1_lens["k1"])), list(OTT_METHODS)
        else:
            cal = depth.epic_fields_lens()
            lens, methods = (cal.f, cal.k1), list(VISOR_METHODS)
        chosen = targets(times, group)
        if args.limit_targets:
            chosen = chosen[:args.limit_targets]
        by_gap = gaps(times, ages)
        pairs = [(t, t - g, a) for t in chosen for g, a in sorted(by_gap.items()) if t - g >= 0]
        dense_keys: set[int] = set()
        if prepared["dense_frames"]:
            with np.load(work[clip_id] / "dense.npz") as dense:
                dense_keys = {int(k) for k in dense.files}
        # Mask check: every stretch frame from FIRST_TARGET_S whose reference also has a dense mask.
        dense_targets = sorted(t for t in dense_keys if times[t] >= FIRST_TARGET_S - 1e-6)
        if args.limit_targets:
            dense_targets = dense_targets[:args.limit_targets]
        dense_pairs = [(t, t - g, a) for t in dense_targets for g, a in sorted(by_gap.items()) if t - g in dense_keys]
        plans[clip_id] = {"methods": methods, "lens": {"f": lens[0], "k1": lens[1]}, "work": prepared, "meta": meta,
                          "targets": chosen, "gaps": {str(g): a for g, a in by_gap.items()}, "dense_targets": dense_targets}
        base = {"work": str(work[clip_id]), "times": meta["times"], "lens": lens, "methods": methods}
        # The mask check scores its pairs with both foregrounds; a pair that is also a target is scored once per mask.
        roles: dict[tuple[int, int], tuple[list[Any], list[str]]] = {}
        for t, r, a in pairs:
            roles.setdefault((t, r), (a, []))[1].append("target")
        for t, r, a in dense_pairs:
            roles.setdefault((t, r), (a, []))[1].append("mask_check")
        chunks[clip_id] = []
        for mask in ("own", "dense"):
            mine = [(t, r, a, rs) for (t, r), (a, rs) in sorted(roles.items()) if mask == "own" or "mask_check" in rs]
            for lo in range(0, len(mine), CHUNK_PAIRS):
                chunks[clip_id].append(pool.submit(evaluate_chunk, {**base, "mask": mask, "pairs": mine[lo:lo + CHUNK_PAIRS]}))
        for ready in [c for c in list(chunks) if c not in finished and all(f.done() for f in chunks[c])]:
            finish(ready)
    for clip_id in todo:
        if clip_id not in finished:
            finish(clip_id)
    pool.shutdown()
    write_json(stage_dir() / "g1e.json", {
        "prepared": args.prepared, "ott": args.ott, "select": args.select, "ages": ages, "limit_targets": args.limit_targets,
        "decision": DECISION, "settings": {
            "VISOR_METHODS": VISOR_METHODS, "OTT_METHODS": OTT_METHODS, "ORACLE_OF": ORACLE_OF,
            "FIRST_TARGET_S": FIRST_TARGET_S, "WINDOW_STRIDE": WINDOW_STRIDE, "TARGET_EVERY_S": TARGET_EVERY_S,
            "COMPANION_S": g1d.COMPANION_S, "COMPANION_RANGE_S": g1d.COMPANION_RANGE_S},
        "restored_clips": sorted(restored), "opencv": cv2.__version__, "results": [finished[c] for c in clips],
    })
    return 0


# ----------------------------------------------------------------- validate


def unit(value: Any) -> bool:
    return value is None or (isinstance(value, (int, float)) and 0.0 <= value <= 1.0)


def validate_stage(stage: Path) -> dict[str, bool]:
    import tarfile

    result = json.loads((stage / "g1e.json").read_text())
    rows = result["results"]
    with tarfile.open(stage / "published.tar") as tar:
        names = set(tar.getnames())
    pairs: dict[str, list[dict[str, Any]]] = {}
    with tarfile.open(stage / "published.tar") as tar:
        for r in rows:
            member = f"publish/clips/{g1.safe(r['id'])}/result.json"
            if member in names:
                handle = tar.extractfile(member)
                assert handle is not None
                pairs[r["id"]] = json.load(handle)["pairs"]
    own = {c: [p for p in ps if p["mask"] == "own" and "target" in p["roles"]] for c, ps in pairs.items()}
    visor = [p for r in rows if r["group"] != "ott" for p in own.get(r["id"], []) if p.get("matched")]
    stretches = [r for r in rows if r["group"] == "visor-long"]

    def same_targets(r: dict[str, Any]) -> bool:
        # Every age is scored on the clip's targets (those far enough in for the age's reference).
        return all({p["t"] for p in own[r["id"]] if str(g) in r["gaps"] and p["t"] - p["r"] == int(g)}
                   == {t for t in r["targets"] if t - int(g) >= 0} for g in r["gaps"])

    return {
        "clips_evaluated": bool(rows),
        "every_clip_has_result": all(r["id"] in pairs for r in rows),
        "frames_match_prepared_archive": all(r["work"]["frames_ok"] for r in rows if not r.get("restored_from_checkpoint")),
        "every_age_on_the_same_targets": all(same_targets(r) for r in rows if r["id"] in own),
        "pairs_have_the_forced_age": all(p["ages"] == r["gaps"][str(p["t"] - p["r"])] for r in rows for p in pairs.get(r["id"], [])),
        "pairs_matched": bool(visor) and float(np.mean([p["matched"] for ps in own.values() for p in ps])) > 0.5,
        "oracle_dominates_its_parts": all(
            p["methods"]["oracle"]["covered_explained"] >= bool(p["methods"].get(m, {}).get("covered_explained"))
            for p in visor for m in ORACLE_OF if "oracle" in p["methods"]),
        "forward_backward_measured": any(p["methods"].get("epi", {}).get("fb_p90_px") is not None for p in visor),
        "shares_in_unit_interval": all(unit(cell.get("explained_share")) for r in rows for mask in VIEWS
                                       for entry in (r["summary"].get(mask) or {}).values() for cell in entry.values()
                                       if isinstance(cell, dict)),
        "mask_check_scored": all(r["summary"].get("dense") for r in stretches if r["dense_targets"]) and
                             any(r["dense_targets"] for r in stretches) if stretches else True,
    }


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


def group_table(rows: list[dict[str, Any]], ages: list[Any], mask: str = "own") -> dict[str, Any]:
    """Per group, age and method: the median clip's explained share and medians of the clips' medians."""
    out: dict[str, Any] = {}
    for group in sorted({r["group"] for r in rows}):
        clips = [r for r in rows if r["group"] == group]
        table: dict[str, Any] = {"clips": len(clips)}
        for age in ages:
            entries: list[dict[str, Any]] = [e for r in clips if (e := (r["summary"].get(mask) or {}).get(str(age)))]
            if not entries:
                continue
            row: dict[str, Any] = {"clips": len(entries), "pairs": sum(e["pairs"] for e in entries),
                                   "matched_share_median": round(float(np.median([e["matched_share"] for e in entries])), 4)}
            for name in [k for k in entries[0] if isinstance(entries[0][k], dict)]:
                cells = [e[name] for e in entries if name in e]

                def med(key: str, cells_: list[dict[str, Any]] = cells) -> float | None:
                    values = [c[key] for c in cells_ if c.get(key) is not None]
                    return round(float(np.median(values)), 4) if values else None

                shares = [c["explained_share"] for c in cells]
                row[name] = {"explained_share_median_clip": med("explained_share"),
                             "clips_at_least_half": sum(v >= DECISION["meaningful_median_clip_share"] for v in shares),
                             "flow_p90_px_median": med("flow_p90_px_median"), "fb_p90_px_median": med("fb_p90_px_median"),
                             "hole_share_median": med("hole_share_median"), "epi_matches_share_median": med("epi_matches_share"),
                             "frame_bytes_median": round(float(np.median([c["rate"]["frame_bytes_mean"] for c in cells if c.get("rate")])), 1)
                             if any(c.get("rate") for c in cells) else None,
                             "reference_bytes_median": round(float(np.median([c["rate"]["reference_bytes_mean"] for c in cells if c.get("rate")])), 1)
                             if any(c.get("rate") for c in cells) else None}
            table[str(age)] = row
        out[group] = table
    return out


def age_seconds(age: Any) -> float:
    return 0.0 if age == "frame" else float(age)


def decide(table: dict[str, Any], dense: dict[str, Any], ages: list[Any]) -> dict[str, Any]:
    """The rule of the G1e entry. ``table``: `group_table` with each clip's own foreground; ``dense``: the same on
    the mask-check pairs, with the window's dense masks and with the clip's own (``dense['own']``)."""
    bar = DECISION["meaningful_median_clip_share"]

    def share(tab: dict[str, Any], group: str, age: Any, method: str) -> float | None:
        cell = tab.get(group, {}).get(str(age), {}).get(method)
        return cell["explained_share_median_clip"] if cell else None

    def refresh_age(tab: dict[str, Any], group: str, method: str) -> Any:
        passing = [a for a in ages if (share(tab, group, a, method) or 0.0) >= bar]
        return max(passing, key=age_seconds) if passing else None

    out: dict[str, Any] = {"refresh_age": {g: {m: refresh_age(table, g, m) for m in ("oracle", *DECISION["sendable"])}
                                           for g in DECISION["groups"] if g in table}}
    # Step 5: the floor, per age.
    floor: dict[str, Any] = {}
    for age in ages:
        ott = table.get("ott", {}).get(str(age), {}).get("h1", {}).get("flow_p90_px_median")
        fb = [table.get(g, {}).get(str(age), {}).get(DECISION["floor_measure"], {}).get("fb_p90_px_median") for g in DECISION["groups"]]
        floor[str(age)] = {"ott_h1_p90": ott, "visor_fb_p90": fb,
                           "measurable": all(v is not None and v <= DECISION["floor_px"] for v in fb)}
    out["floor"] = floor
    out["ott_holds_at_1s"] = (floor.get("1.0") or {}).get("ott_h1_p90") is not None and floor["1.0"]["ott_h1_p90"] <= DECISION["floor_px"]
    # Mask check per age and method on the stretches.
    gaps_: dict[str, Any] = {}
    for age in ages:
        for m in ("oracle", *DECISION["sendable"]):
            a, b = share(dense.get("dense", {}), "visor-long", age, m), share(dense.get("own", {}), "visor-long", age, m)
            if a is not None and b is not None:
                gaps_[f"{age}:{m}"] = round(a - b, 4)
    out["mask_gap"] = gaps_
    mask_ok = all(abs(v) <= DECISION["mask_check_max_gap"] for v in gaps_.values())
    out["mask_check_holds"] = mask_ok
    stretch_table = table if mask_ok else {"visor-long": dense.get("dense", {}).get("visor-long", {})}
    gate = out["refresh_age"].get("visor-window", {}).get("oracle")
    if gate is None or age_seconds(gate) < DECISION["gate_min_age_s"]:
        out.update(step=1, verdict="the oracle needs references younger than 0.1 s on the windows: refreshing is no cheaper "
                   "than coding every frame; egocentric goes to G5")
        return out
    qualifying = {}
    for m in DECISION["sendable"]:
        w = refresh_age(table, "visor-window", m)
        s = refresh_age(stretch_table, "visor-long", m)
        if w is not None and s is not None and min(age_seconds(w), age_seconds(s)) >= DECISION["gate_min_age_s"]:
            qualifying[m] = min(w, s, key=age_seconds)
    out["qualifying"] = qualifying
    out["stretch_verdict_from"] = "own masks" if mask_ok else "dense masks on the window frames (fewer pairs)"
    refine = [m for m in ("oracle", "planes4") if (share(table, "visor-window", 0.1, m) or 0) >= bar
              and (share(table, "visor-window", 1.0, m) or 0) < bar]
    out["refine_at_0.3s"] = bool(refine) and DECISION["refine_age_s"] not in ages
    if qualifying:
        def rate(m: str) -> float:
            cell = table["visor-window"][str(qualifying[m])][m]
            return (cell["frame_bytes_median"] or 0) + (cell["reference_bytes_median"] or 0)

        out.update(step=2, chosen=min(qualifying, key=rate), verdict="G2 prices the chosen warp refreshed at its refresh age")
    else:
        out.update(step=3, verdict="the oracle qualifies but no sendable warp does in both groups; a warp between them "
                   "(DA3 or depth-augmented keyframes) at the oracle's refresh age gets its own entry")
    return out


def command_report(args: argparse.Namespace) -> int:
    rows: dict[str, dict[str, Any]] = {}
    runs, ages = [], []
    for path in args.result:
        result = json.loads(Path(path).read_text())
        runs.append({"path": path, "sha256": file_sha256(Path(path)), "select": result["select"], "ages": result["ages"],
                     "clips": len(result["results"])})
        for a in result["ages"]:
            if a not in ages:
                ages.append(a)
        for r in result["results"]:
            rows.setdefault(r["id"], {**r, "summary": {}})
            for mask, per_age in r["summary"].items():
                if isinstance(per_age, dict):
                    rows[r["id"]]["summary"].setdefault(mask, {}).update(per_age)
    ages.sort(key=age_seconds)
    table = group_table(list(rows.values()), ages)
    # The mask check compares the two foregrounds on the same pairs.
    checked = [r for r in rows.values() if r["group"] == "visor-long" and r["dense_targets"]]
    dense_tables = {"dense": group_table(checked, ages, "dense"), "own": group_table(checked, ages, "own_on_dense")}
    report = {"runs": runs, "decision_rule": DECISION, "ages": ages, "groups": table, "mask_check": dense_tables,
              "decision": decide(table, dense_tables, ages)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "g1e-report.json", report)
    lines = ["| Group | Age | Method | median clip explained | clips ≥ 50% | p90 px | forward-backward p90 | holes |",
             "|---|---|---|---:|---:|---:|---:|---:|"]
    for group, cells in table.items():
        for age in ages:
            row = cells.get(str(age))
            if not row:
                continue
            for name, c in row.items():
                if isinstance(c, dict):
                    lines.append(f"| {group} ({cells['clips']}) | {age} | {name} | {c['explained_share_median_clip']} | "
                                 f"{c['clips_at_least_half']} | {c['flow_p90_px_median']} | {c['fb_p90_px_median']} | {c['hole_share_median']} |")
    (out / "g1e-report.md").write_text("\n".join(lines) + "\n\n" + json.dumps(report["decision"], indent=1) + "\n")
    print("\n".join(lines))
    print(json.dumps(report["decision"], indent=1))
    return 0


# ----------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="store G1's OpenTTGames clips as G1d stored VISOR's")
    prep.add_argument("--clips", required=True, help="G1's clips.json")
    prep.add_argument("--g1", required=True, help="G1's published clips/<id>/{result.json,masks.rle}")
    prep.add_argument("--select", required=True)
    prep.add_argument("--limit-seconds", type=float, default=0.0)
    prep.add_argument("--source", nargs=2, action="append", default=[], metavar=("NAME", "PATH"))
    sub.add_parser("validate-prepare")
    run = sub.add_parser("run")
    run.add_argument("--prepared", required=True, help="extracted published.tar of G1d's prepare job")
    run.add_argument("--ott", required=True, help="extracted published.tar of G1e's prepare job")
    run.add_argument("--clips", required=True, help="G1's clips.json (the stretches' window items)")
    run.add_argument("--visor-fill", required=True, help="B1b's fill (the windows' dense masks)")
    run.add_argument("--select", required=True)
    run.add_argument("--ages", default=",".join(str(a) for a in AGES))
    run.add_argument("--limit-targets", type=int, default=0)
    sub.add_parser("validate")
    report = sub.add_parser("report")
    report.add_argument("--result", action="append", required=True)
    report.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    handlers = {"prepare": command_prepare, "validate-prepare": command_validate, "run": command_run,
                "validate": command_validate, "report": command_report}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
