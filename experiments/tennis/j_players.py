"""J on players: K6's `ref` re-judged on the experience track (docs/experiments.md, 2026-10-22 J).

    python -m experiments.tennis.j_players run --inputs DIR --pose PT --inspect BIN --inception PTH --k6-report JSON \\
        --frames N --max-windows N --windows ID ... --sources MP4 ...
    python -m experiments.tennis.j_players validate [--directory DIR --summary JSON]
    python -m experiments.tennis.j_players report STAGE_DIR ... --k6-report JSON --out DIR

Per window, K6's pipeline unchanged (``experiments.tennis.k6``): K4's conversion, the adopted players, their pose,
K4b.3's chosen code (q 4 velocity, every 4th frame) and `ref` drawn as K6 draws it. The anchors are SVT-AV1's four
low-resolution points (CRF 63 at 1/8, 1/4 and 1/2, CRF 55 at 1/4), which span every window's keypoint rate. They are
a subset of K6's envelope, so they decide only where `ref` fails (`experiments.visor.j`).

Judges on K6's judged frames (scored, t mod 3 = 0), for the source, `ref` and each anchor:

* same content: YOLO11x-pose by K6's step 3 (K4b.2's x1 crop on the player's mask box under 15% of the frame height,
  a whole-frame detection above), run on that output; the error is the mean distance over the keypoints confident
  in the source's re-run, over the source fit's box height. A player not fitted on the output counts as 1 (a box).
  The coded keypoints' own error is the same distance from K6's coded keypoints;
* same over time: on R at t against t - 1 (both scored), with DIS flow on the source (H3e's construction):
  ``h3b.flicker``, and each residual's energy (mean absolute motion-compensated luma change, ``g5c.flicker``'s);
* realism: Inception features of each player's square crop (``h3b``'s), kept for KID against a held-out set.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import k4, k6
from experiments.tennis.k4b_features import convert
from experiments.tennis.k4b_pose import Crops, crop_box, fit, fit_whole, frame_dets, magnify
from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json
from src.segmentation.masks import ClipMasks, decompress

ANCHORS: tuple[tuple[str, int, int], ...] = (("svt", 63, 8), ("svt", 63, 4), ("svt", 55, 4), ("svt", 63, 2))
METHOD = f"ref@{k6.code_name(k6.CHOSEN)}"
LUMA = np.array([0.299, 0.587, 0.114], np.float32)
MISSED = 1.0  # content error of a player the output's estimator does not fit (one box)
RATE_TOLERANCE = 0.001  # relative: the SVT-AV1 points must reproduce K6's recorded kbps on R


def anchor_names() -> list[str]:
    return [k6.anchor_name(a) for a in ANCHORS]


def pose_error(pred: np.ndarray | None, ref: np.ndarray, height: float) -> float | None:
    """Mean distance over ``ref``'s confident keypoints, over ``height``; `MISSED` when ``pred`` is None."""
    visible = ref[:, 2] >= k4.KP_CONF
    if not visible.any():
        return None
    if pred is None:
        return MISSED
    return float(np.linalg.norm(pred[visible, :2] - ref[visible, :2], axis=1).mean() / max(height, 1.0))


def residual_energies(out_t: np.ndarray, out_p: np.ndarray, truth_t: np.ndarray, truth_p: np.ndarray, region: np.ndarray,
                      region_p: np.ndarray) -> tuple[float, float, float] | None:
    """(h3b.flicker, the output's residual energy, the source's), on ``region`` at t, flow from the source (as H3e)."""
    import cv2

    from experiments.visor.h3 import dis_flow, flow_box, warp
    from experiments.visor.h3b import flicker

    if not region.any() or not region_p.any():
        return None
    fb = flow_box(region, region_p)
    crop = (slice(fb[1], fb[3]), slice(fb[0], fb[2]))
    tt, tp = truth_t[crop], truth_p[crop]
    flow = dis_flow(cv2.cvtColor(tt, cv2.COLOR_RGB2GRAY), cv2.cvtColor(tp, cv2.COLOR_RGB2GRAY))
    ow = warp(np.ascontiguousarray(out_p[crop]), flow)
    tw = warp(np.ascontiguousarray(tp), flow)
    m = region[crop]
    f = flicker(out_t[crop], ow, tt, tw, m)
    e_out = float(np.abs((out_t[crop].astype(np.float32) - ow.astype(np.float32)) @ LUMA)[m].mean())
    e_src = float(np.abs((tt.astype(np.float32) - tw.astype(np.float32)) @ LUMA)[m].mean())
    return f, e_out, e_src


class Judge:
    """K6's step-3 estimator applied to any output frame (crop1 players on their source mask box, whole above)."""

    def __init__(self, poses: Any, crops: Crops):
        self.poses, self.crops = poses, crops

    def run(self, frames: dict[tuple[str, int], np.ndarray], bodies: list[dict[str, np.ndarray]], setting: dict[str, str],
            players: list[str]) -> dict[tuple[str, int, str], np.ndarray | None]:
        """``frames``: (variant, t) -> image. Returns (variant, t, player) -> keypoints (17, 3) or None (not fitted)."""
        from experiments.visor.h1 import box_of

        out: dict[tuple[str, int, str], np.ndarray | None] = {}
        whole_needed = [(v, t) for (v, t) in frames if any(setting[p] == "whole" and p in bodies[t] for p in players)]
        whole: dict[tuple[str, int], Any] = {}
        if whole_needed:
            dets = self.poses.run("yolo11x-pose", np.stack([frames[k] for k in whole_needed]), imgsz=k4.POSE_IMGSZ)
            whole = dict(zip(whole_needed, dets))
        queued = False
        for (v, t), image in frames.items():
            for p in players:
                m = bodies[t].get(p)
                if m is None or not m.any():
                    continue
                if setting[p] == "whole":
                    frame = {(t, q): x for q, x in bodies[t].items()}
                    r = fit_whole(m, frame, t, p, *frame_dets(whole[(v, t)]))
                    out[(v, t, p)] = np.asarray(r["kps"], np.float64) if r["fitted"] else None
                else:
                    x0, y0, x1, y1 = crop_box(box_of(m))  # type: ignore[arg-type]
                    crop, factors = magnify(np.ascontiguousarray(image[y0:y1, x0:x1]), 1)
                    self.crops.add(crop, factors, (x0, y0), (v, t, p))
                    queued = True
        if queued:
            for (v, t, p), boxes, kps in self.crops.run("judge"):
                r = fit(bodies[t][p], boxes, kps)
                out[(v, t, p)] = np.asarray(r["kps"], np.float64) if r["fitted"] else None
        return out


def process_window(args: argparse.Namespace, entry: dict[str, Any], source: Path, nets: dict[str, Any], judge: Judge,
                   scratch: Path, publish: Path, cpus: list[int], recorded: dict[str, Any]) -> dict[str, Any]:
    import cv2

    from experiments.visor.h3 import rgb_memmap
    from experiments.visor.h3b import CROP_GROW, INCEPTION_SIZE, inception_features, square_crop

    began = time.time()
    w = entry["window"]
    wid = w["id"]
    frames = min(int(args.frames), int(w["frames"]))
    fps = Fraction(w["fps"]).limit_denominator(1001)
    work = scratch / "work" / k4.safe(wid)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    root = Path(args.inputs) / "windows" / k4.safe(wid)
    for name, digest in entry["files"].items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"Changed input: {root / name}")
    if file_sha256(source) != w["sha256"]:
        raise ValueError(f"{wid}: original clip identity mismatch")
    yuv = work / "source.yuv"
    converted = convert(args.ffmpeg, {**w, "frames": frames}, source, yuv)
    truth = rgb_memmap(yuv, work / "truth.npy", frames, k6.FRAME[0], k6.FRAME[1], False, "cuda")

    # K6's players, regions, pose, codes and references, unchanged
    features = json.loads((root / "features.json").read_text())
    clip = ClipMasks.load(root / "sam-masks.rle")
    whole = json.loads(decompress((root / "pose.json.zst").read_bytes()))
    fits = {int(f): np.asarray(h, np.float64) for f, h in entry["fits"].items()}
    players = k6.window_players(features, clip, int(w["start"]), fits)["players"]
    scored_all = [t for t in range(frames) if int(w["start"]) + t in fits]
    per_frame = [k6.frame_regions(clip.frames[t], features["aliases"], players) for t in range(frames)]
    regions = [f["regions"] for f in per_frame]
    bodies = [f["bodies"] for f in per_frame]
    median_height = k6.heights(bodies, players)
    poses = k6.pose_window(judge.crops, truth, frames, bodies, whole, players, median_height)
    union = np.stack([np.logical_or.reduce([*f["regions"].values(), np.zeros((k6.FRAME[1], k6.FRAME[0]), bool)]) for f in per_frame])
    scored = [t for t in scored_all if union[t].any()]
    scored_r = np.zeros_like(union)
    scored_r[scored] = union[scored]
    decoded, _ = k6.coded_keypoints(k6.code_tracks(poses, scored))
    coded = decoded[k6.code_name(k6.CHOSEN)]
    region_height = {(p, t): float(np.ptp(np.nonzero(r[p].any(1))[0]) + 1) if p in r and r[p].any() else 1.
                     for t, r in enumerate(regions) for p in players}
    order = sorted(players, key=lambda p: median_height[p])
    gap = max(1, round(k6.REF_GAP_S * float(fps)))
    setting = {p: "whole" if median_height[p] >= k6.WHOLE_FRAME_ABOVE * k6.FRAME[1] else "crop1" for p in players}
    timing = {"prepare": round(time.time() - began, 1)}

    # SVT-AV1's low points (K6's code_svt: the same streams and bit accounting)
    anchors: dict[str, dict[str, Any]] = {}
    decodes: dict[str, Any] = {}
    for a in ANCHORS:
        name = k6.anchor_name(a)
        rec = k6.code_svt(args, yuv, frames, fps, a[1], a[2], scored_r, None, work / name, cpus)
        sw, sh = rec["size"]
        decodes[name] = rgb_memmap(Path(rec["decoded"]), work / f"{name}-rgb.npy", frames, sw, sh, False, "cuda")
        Path(rec["decoded"]).unlink()
        seconds = len(scored) / float(fps)
        anchors[name] = {**{k: v for k, v in rec.items() if k != "decoded"}, "kbps_r": rec["region_bits"] / seconds / 1000}
    timing["anchors"] = round(time.time() - began, 1)

    judged = [t for t in scored if t % k6.JUDGE_EVERY == 0]
    pairs = [t for t in judged if t - 1 in set(scored)]
    references: dict[str, int | None] = {}

    def ref_frame(t: int) -> np.ndarray:
        truth_t = np.asarray(truth[t])
        canvas = k6.telea_fill(truth_t, union[t])
        drawn = np.zeros_like(union[t])
        for p in order:
            pose = poses[p][t]
            if p not in regions[t] or pose is None or not pose.get("fitted"):
                continue
            target = coded.get((p, t))
            if target is None:
                continue
            r = k6.choose_reference(t, p, target, poses, scored, region_height, gap)
            references[f"{p}@{t}"] = r
            if r is None:
                continue
            k6.draw(canvas, drawn, np.asarray(truth[r]), regions[r][p], np.asarray(poses[p][r]["kps"], np.float64),  # type: ignore[index]
                    target, k6.SIGMA_OF_HEIGHT * region_height[(p, t)])
        return canvas

    def anchor_frame(name: str, t: int) -> np.ndarray:
        truth_t = np.asarray(truth[t])
        up = cv2.resize(np.asarray(decodes[name][t]), k6.FRAME, interpolation=cv2.INTER_CUBIC)
        out = truth_t.copy()
        out[union[t]] = up[union[t]]
        return out

    variants = ["source", METHOD, *anchor_names()]
    pose_err = np.full((len(variants), len(judged), len(players)), np.nan)
    kp_err = np.full((len(judged), len(players)), np.nan)
    rerun_vs_k6 = np.full((len(judged), len(players)), np.nan)
    flick = np.full((len(variants), len(pairs)), np.nan)
    energy = np.full((len(variants), len(pairs)), np.nan)
    energy_src = np.full(len(pairs), np.nan)
    feats: list[list[np.ndarray]] = [[] for _ in variants]  # Inception features, one block per chunk
    feature_rows: list[tuple[int, int]] = []
    looks: dict[str, np.ndarray] = {}
    look_t = judged[len(judged) // 2] if judged else None
    for chunk_start in range(0, len(judged), 8):
        chunk = judged[chunk_start:chunk_start + 8]
        times = sorted(set(chunk) | {t - 1 for t in chunk if t in pairs})
        images: dict[tuple[str, int], np.ndarray] = {}
        for t in times:
            images[("source", t)] = np.asarray(truth[t])
            images[(METHOD, t)] = ref_frame(t)
            for name in anchor_names():
                images[(name, t)] = anchor_frame(name, t)
        kps = judge.run({k: v for k, v in images.items() if k[1] in chunk}, bodies, setting, players)
        crops: list[list[np.ndarray]] = [[] for _ in variants]
        for t in chunk:
            i = judged.index(t)
            for j, p in enumerate(players):
                src = kps.get(("source", t, p))
                pose = poses[p][t]
                if src is None or pose is None or not pose.get("fitted"):
                    continue
                height = float(np.asarray(pose["box"])[3] - np.asarray(pose["box"])[1])
                rerun_vs_k6[i, j] = float(np.abs(src[:, :2] - np.asarray(pose["kps"])[:, :2]).max())
                for v_i, v in enumerate(variants):
                    e = pose_error(kps.get((v, t, p)), src, height)
                    pose_err[v_i, i, j] = np.nan if e is None else e
                if (p, t) in coded:
                    e = pose_error(coded[(p, t)], src, height)
                    kp_err[i, j] = np.nan if e is None else e
                box = np.asarray(pose["box"], np.float64)
                for v_i, v in enumerate(variants):
                    crops[v_i].append(square_crop(images[(v, t)], box, CROP_GROW, INCEPTION_SIZE))
                feature_rows.append((t, j))
            if t in pairs:
                k = pairs.index(t)
                for v_i, v in enumerate(variants):
                    got = residual_energies(images[(v, t)], images[(v, t - 1)], images[("source", t)], images[("source", t - 1)],
                                            union[t], union[t - 1])
                    if got is not None:
                        flick[v_i, k], energy[v_i, k], energy_src[k] = got
            if t == look_t:
                looks = {v: images[(v, t)] for v in variants}
        for v_i in range(len(variants)):
            if crops[v_i]:
                feats[v_i].append(inception_features(nets["inception"], crops[v_i], "cuda"))
        print(f"phase=judge window={wid} frames={min(chunk_start + 8, len(judged))}/{len(judged)}", flush=True)
    timing["judge"] = round(time.time() - began, 1)

    out_dir = publish / k4.safe(wid)
    out_dir.mkdir(parents=True, exist_ok=True)
    features_arr = np.stack([np.concatenate(f) if f else np.zeros((0, 2048), np.float16) for f in feats]).astype(np.float16)
    np.savez_compressed(out_dir / "judges.npz", pose_err=pose_err, kp_err=kp_err, rerun_vs_k6=rerun_vs_k6, flicker=flick,
                        energy=energy, energy_src=energy_src, features=features_arr,
                        feature_rows=np.asarray(feature_rows, np.int64).reshape(-1, 2), judged=np.asarray(judged, np.int64),
                        pairs=np.asarray(pairs, np.int64), scored=np.asarray(scored, np.int64))
    if looks:
        strip = np.concatenate([looks[v][::2, ::2] for v in variants], 1)
        cv2.imwrite(str(out_dir / f"look-t{look_t:03d}-{'-'.join(v.replace('@', '_') for v in variants)}.jpg"),
                    cv2.cvtColor(strip, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90])
    rec_w = recorded["windows"].get(wid, {})
    row = {"id": wid, "match": wid.split("/")[0], "frames": frames, "players": players, "setting": setting,
           "variants": variants, "judged": len(judged), "pairs": len(pairs), "convert": converted,
           "anchors": anchors, "recorded_kbps": {n: rec_w.get("anchors", {}).get(n, {}).get("kbps") for n in anchor_names()},
           "method_kbps": rec_w.get("keypoint_kbps", {}).get(k6.code_name(k6.CHOSEN)), "references": references,
           "timing": timing, "seconds": round(time.time() - began, 2)}
    row["files"] = {p.name: file_sha256(p) for p in sorted(out_dir.iterdir()) if p.is_file()}
    write_json(out_dir / "window.json", row)
    shutil.rmtree(work, ignore_errors=True)
    return row


def command_run(args: argparse.Namespace) -> int:
    import torch

    from experiments.visor.h3b import load_inception
    from src.codecs import svtav1

    began = time.time()
    inputs = json.loads((Path(args.inputs) / "inputs.json").read_text())
    recorded = json.loads(Path(args.k6_report).read_text())
    if len(args.windows) != len(args.sources) or len(set(args.windows)) != len(args.windows):
        raise ValueError("Give one source per window, without repeated windows")
    chosen = args.windows[:args.max_windows or None]
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    publish = scratch / "publish"
    publish.mkdir(exist_ok=True)
    inspect = scratch / "bin" / Path(args.inspect).name  # the staging cache drops the execute bit (as K6)
    inspect.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.inspect, inspect)
    inspect.chmod(0o755)
    args.inspect = str(inspect)
    os.environ["YOLO_CONFIG_DIR"] = str(scratch / "ultralytics")
    poses = k4.Poses([("yolo11x-pose", args.pose)])
    judge = Judge(poses, Crops(poses))
    nets = {"inception": load_inception(Path(args.inception), "cuda")}
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 2))
    rows: list[dict[str, Any]] = []
    for n, wid in enumerate(chosen):
        rows.append(process_window(args, inputs["windows"][wid], Path(args.sources[args.windows.index(wid)]), nets, judge,
                                   scratch, publish, cpus, recorded))
        summary = {"stage": os.environ.get("PS_STAGE"), "requested": args.windows, "windows": rows, "complete": n + 1 == len(chosen),
                   "anchors": [list(a) for a in ANCHORS], "method": METHOD, "frames_argument": int(args.frames),
                   "inputs_sha256": file_sha256(Path(args.inputs) / "inputs.json"), "k6_report_sha256": file_sha256(Path(args.k6_report)),
                   "tools": {"svt-av1": svtav1.tool(args.encoder), "dav1d": svtav1.tool(args.decoder), "ffmpeg": svtav1.tool(args.ffmpeg),
                             "inspect_sha256": file_sha256(Path(args.inspect))},
                   "weights": {"pose_sha256": file_sha256(Path(args.pose)), "inception_sha256": file_sha256(Path(args.inception))},
                   "pose": poses.record(), "on_cuda": {"inception": next(nets["inception"].parameters()).is_cuda},
                   "gpu": torch.cuda.get_device_name(0), "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                   "seconds": round(time.time() - began, 2)}
        with tarfile.open(stage_dir() / "published.tar", "w") as tar:
            tar.add(publish, arcname="publish")
        write_json(stage_dir() / "j_players.json", summary)
        print(f"phase=window window={wid} done={n + 1}/{len(chosen)} seconds={rows[-1]['seconds']}", flush=True)
        progress(n + 1)
    return 0


# ----------------------------------------------------------------- validation


def validate_window(root: Path, row: dict[str, Any], full: bool) -> dict[str, bool]:
    a = np.load(root / "judges.npz")
    v = len(row["variants"])
    checks = {
        "files": all(file_sha256(root / n) == d for n, d in row["files"].items()),
        "shapes": a["pose_err"].shape[0] == v and a["flicker"].shape[0] == v and a["features"].shape[0] == v,
        "judged": row["judged"] > 0 and a["pose_err"].shape[1] == row["judged"],
        "source_scores_itself": bool(np.nanmax(np.nan_to_num(a["pose_err"][0], nan=0.0)) == 0.0),
        "source_has_no_flicker": bool(np.nanmax(np.nan_to_num(a["flicker"][0], nan=0.0)) < 1e-4),
        "method_scored": bool(np.isfinite(a["pose_err"][1]).any() and np.isfinite(a["flicker"][1]).any()),
        "features_finite": bool(np.isfinite(a["features"].astype(np.float32)).all()) and a["features"].shape[1] > 0,
        # the judge's re-run on the source is K6's pose path: the same keypoints (median within 1 px; tall players'
        # K6 poses come from K4b.1's whole-frame detections, made in another job)
        "rerun_is_k6_pose": bool(np.isfinite(a["rerun_vs_k6"]).any() and np.nanmedian(a["rerun_vs_k6"]) <= 1.0),
        "references_drawn": sum(r is not None for r in row["references"].values()) > 0,
    }
    if full:  # a full window's SVT-AV1 points reproduce K6's recorded rates on R
        checks["rates_reproduce_k6"] = all(row["recorded_kbps"][n] is not None
                                           and abs(row["anchors"][n]["kbps_r"] / row["recorded_kbps"][n] - 1) <= RATE_TOLERANCE
                                           for n in anchor_names())
    return checks


def validate(directory: Path, summary: dict[str, Any]) -> dict[str, Any]:
    full = summary["frames_argument"] >= 240
    checks: dict[str, bool] = {"some_window": bool(summary["windows"]), "complete": bool(summary["complete"]),
                               "on_cuda": all(summary["on_cuda"].values())}
    for row in summary["windows"]:
        for k, ok in validate_window(directory / k4.safe(row["id"]), row, full).items():
            checks[f"{row['id']}:{k}"] = ok
    return {"passed": all(checks.values()), "checks": checks}


def command_validate(args: argparse.Namespace) -> int:
    stage = Path(os.environ.get("PS_STAGE_DIR") or ".")
    summary_path = args.summary or stage / "j_players.json"
    summary = json.loads(Path(summary_path).read_text())
    directory = args.directory
    if directory is None:
        directory = Path(os.environ.get("PS_SCRATCH_DIR") or stage) / "validate"
        shutil.rmtree(directory, ignore_errors=True)
        with tarfile.open(stage / "published.tar") as tar:
            tar.extractall(directory, filter="data")
        directory = directory / "publish"
    result = validate(Path(directory), summary)
    write_json(stage / "validation.json", result)
    print(json.dumps(result, indent=1))
    return 0 if result["passed"] else 1


# ----------------------------------------------------------------- report (CPU)


def window_report(row: dict[str, Any], a: dict[str, np.ndarray]) -> dict[str, Any]:
    from experiments.visor import j

    names = row["variants"]
    mean = {v: float(np.nanmean(a["pose_err"][i])) if np.isfinite(a["pose_err"][i]).any() else None for i, v in enumerate(names)}
    flick = {v: float(np.nanmean(a["flicker"][i])) if np.isfinite(a["flicker"][i]).any() else None for i, v in enumerate(names)}
    energy = {v: float(np.nanmean(a["energy"][i])) if np.isfinite(a["energy"][i]).any() else None for i, v in enumerate(names)}
    src_energy = float(np.nanmean(a["energy_src"])) if np.isfinite(a["energy_src"]).any() else None
    kp = float(np.nanmean(a["kp_err"])) if np.isfinite(a["kp_err"]).any() else None
    rate = float(row["method_kbps"])
    pose_anchor = j.read([(row["anchors"][n]["kbps_r"], mean[n]) for n in anchor_names()], rate)
    flick_anchor = j.read([(row["anchors"][n]["kbps_r"], flick[n]) for n in anchor_names()], rate)
    checks = {"content": j.content(mean[METHOD], pose_anchor, kp, complete=False),
              "time": j.over_time(flick[METHOD], flick_anchor, energy[METHOD], src_energy, complete=False),
              "realism": None}
    method_err = a["pose_err"][names.index(METHOD)]
    missed = float(np.mean(method_err[np.isfinite(method_err)] == MISSED)) if np.isfinite(method_err).any() else None
    return {"kbps": rate, "pose_error": mean, "keypoints_error": kp, "pose_anchor": pose_anchor, "flicker": flick,
            "energy": energy, "source_energy": src_energy, "flicker_anchor": flick_anchor, "checks": checks,
            "status": j.item_status(checks), "method_missed_share": missed, "judged": row["judged"], "pairs": row["pairs"],
            "match": row["match"]}


def command_report(args: argparse.Namespace) -> int:
    from experiments.visor import j

    rows: dict[str, dict[str, Any]] = {}
    sources = []
    for stage in args.stages:
        summary = json.loads((stage / "j_players.json").read_text())
        sources.append({"path": str(stage), "j_players_sha256": file_sha256(stage / "j_players.json"),
                        "published_sha256": file_sha256(stage / "published.tar") if (stage / "published.tar").exists() else None})
        root = stage / "publish"
        for row in summary["windows"]:
            a = dict(np.load(root / k4.safe(row["id"]) / "judges.npz"))
            rows[row["id"]] = window_report(row, a)
    statuses = {k: v["status"] for k, v in rows.items()}
    report = {"result": f"K6 {METHOD} on players", "decision": j.decide(statuses), "rule": j.DECISION,
              "checks": j.check_rates({k: v["checks"] for k, v in rows.items()}),
              "anchor": "SVT-AV1's four low points: a subset of K6's envelope, conclusive where it fails",
              "windows": rows, "stages": sources}
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "j-players.json", report)
    print(json.dumps({"decision": report["decision"], "checks": report["checks"],
                      "windows": {k: {"status": v["status"], **v["checks"], "kbps": round(v["kbps"], 2)} for k, v in rows.items()}}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    for name in ("inputs", "pose", "inspect", "inception", "k6-report"):
        run.add_argument("--" + name, required=True)
    run.add_argument("--frames", type=int, required=True)
    run.add_argument("--max-windows", type=int, required=True, help="0 runs every listed window")
    run.add_argument("--windows", nargs="+", required=True)
    run.add_argument("--sources", nargs="+", required=True)
    run.add_argument("--encoder", default="SvtAv1EncApp")
    run.add_argument("--decoder", default="dav1d")
    run.add_argument("--ffmpeg", default="ffmpeg")
    val = sub.add_parser("validate")
    val.add_argument("--directory", type=Path)
    val.add_argument("--summary", type=Path)
    rep = sub.add_parser("report")
    rep.add_argument("stages", nargs="+", type=Path, help="stage directories: j_players.json and publish/")
    rep.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return {"run": command_run, "validate": command_validate, "report": command_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
