"""J2: the coded keypoints' temporal jitter (docs/experiments.md, 2026-10-22 "J2").

    python -m experiments.visor.j2 measure --k1f-stage TAR ... --out DIR          (CPU)
    python -m experiments.visor.j2 run --eval-set JSON --archive TAR --masks TAR --mask-record JSON --video MP4 \\
        --items IDS --frames N --k1f-stage TAR ... --mano-left PKL --mano-right PKL --wilor-checkpoint CKPT
    python -m experiments.visor.j2 validate
    python -m experiments.visor.j2 report --stage DIR ... --j-stage DIR ... --out DIR

``warp`` (K1f's bound) moves the true pixels by d_t = coded_t - uncoded_t. ``measure`` splits how d changes between
frames (coding jitter) from the source's motion and the estimator's own noise. ``run`` re-draws ``warp`` with
decoder-side smoothing (``warp@S1``, ``@S2``, ``@S4``: uncoded -> S_sigma(coded)) and the no-coding diagnostic
``warp@Su`` (uncoded -> S_2(uncoded)), judged as J judges (``j_hands``); ``report`` reads J's saved SVT-AV1 hulls.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.visor import j, j_hands, k1c, k1f
from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json

SIGMAS = (1, 2, 4)  # frames
NOISE_SIGMA = 2
VARIANTS = ("source", "warp", *(f"warp@S{s}" for s in SIGMAS), "warp@Su")
CHECK_TOLERANCE = 1e-3  # the re-drawn `warp` against J's saved judges (px, luma)


def tracks(t: np.ndarray, right: np.ndarray) -> list[np.ndarray]:
    """Row indices of each hand track: one side's rows in frame order, split where a frame is skipped."""
    out: list[np.ndarray] = []
    for side in (False, True):
        idx = np.nonzero(right == side)[0]
        idx = idx[np.argsort(t[idx], kind="stable")]
        if not len(idx):
            continue
        cut = np.nonzero(np.diff(t[idx]) != 1)[0] + 1
        out.extend(p for p in np.split(idx, cut) if len(p))
    return out


def smooth(points: np.ndarray, sigma: float) -> np.ndarray:
    """A centred Gaussian over the first axis (frames), renormalized at the ends; sigma 0 returns a copy."""
    n = len(points)
    if sigma <= 0 or n < 2:
        return points.copy()
    k = np.arange(n)
    w = np.exp(-0.5 * ((k[:, None] - k[None, :]) / sigma) ** 2)
    w /= w.sum(1, keepdims=True)
    return np.tensordot(w, points, axes=(1, 0))


def smoothed_rows(values: np.ndarray, t: np.ndarray, right: np.ndarray, sigma: float) -> np.ndarray:
    out = values.copy()
    for tr in tracks(t, right):
        out[tr] = smooth(values[tr], sigma)
    return out


def item_measure(a: dict[str, np.ndarray]) -> dict[str, float | None]:
    """Medians over each item's consecutive-frame pairs of per-joint means, over the track's box."""
    t, right = np.asarray(a["row_t"]).astype(int), np.asarray(a["row_right"]).astype(bool)
    u, c = np.asarray(a["uncoded_kp2d"], float), np.asarray(a["coded_kp2d"], float)
    jitter, motion, noise, sjitter = [], [], [], []
    for tr in tracks(t, right):
        if len(tr) < 2:
            continue
        box = np.maximum(np.ptp(u[tr], axis=1).max(-1), 1.0)
        d = c[tr] - u[tr]
        jitter += list(np.linalg.norm(np.diff(d, axis=0), axis=-1).mean(-1) / box[1:])
        motion += list(np.linalg.norm(np.diff(u[tr], axis=0), axis=-1).mean(-1) / box[1:])
        noise += list(np.linalg.norm(u[tr] - smooth(u[tr], NOISE_SIGMA), axis=-1).mean(-1) / box)
        ds = smooth(c[tr], 2) - u[tr]
        sjitter += list(np.linalg.norm(np.diff(ds, axis=0), axis=-1).mean(-1) / box[1:])

    def med(x: list[float]) -> float | None:
        return float(np.median(x)) if x else None

    return {"coding_jitter": med(jitter), "source_motion": med(motion), "estimator_noise": med(noise),
            "jitter_after_S2": med(sjitter), "pairs": len(jitter)}


def command_measure(args: argparse.Namespace) -> int:
    from experiments.visor.h2 import load_units

    rows = {m["item"]: item_measure(a) for m, a in load_units(args.k1f_stage, "k1c") if "coded_kp2d" in a}
    keys = ("coding_jitter", "source_motion", "estimator_noise", "jitter_after_S2")
    summary = {k: float(np.median([r[k] for r in rows.values() if r[k] is not None])) for k in keys}
    out = {"items": rows, "median_over_items": summary, "noise_sigma": NOISE_SIGMA,
           "inputs": [{"path": p, "sha256": file_sha256(Path(p))} for p in args.k1f_stage]}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    write_json(Path(args.out) / "j2-measure.json", out)
    print(json.dumps({"items": len(rows), **summary}, indent=1))
    return 0


# ----------------------------------------------------------------- the oracle (GPU: WiLoR)


def process_item(args: argparse.Namespace, item: dict[str, Any], decoded: dict[str, Any], nets: dict[str, Any],
                 unit: tuple[dict[str, Any], dict[str, np.ndarray]], device: str, publish: Path, scratch: Path) -> None:
    from experiments.visor.b2 import load_mask_sets
    from experiments.visor.h2 import save_unit
    from experiments.visor.h3 import FILL, HANDS, rgb_memmap

    began = time.time()
    frames = int(args.frames)
    width, height = decoded["width"], decoded["height"]
    work = scratch / "work" / item["id"]
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    clips, mask_info = load_mask_sets(item, frames, {FILL: args.masks}, {FILL: args.mask_record}, Path(args.archive))
    clip = clips[FILL]
    regions = [clip.class_mask(t, HANDS[0]) | clip.class_mask(t, HANDS[1]) for t in range(frames)]
    truth = rgb_memmap(Path(decoded["path"]), work / "truth.npy", frames, width, height, decoded["full_range"], device)
    meta_c, a_c = unit
    t_all, right_all = np.asarray(a_c["row_t"]).astype(int), np.asarray(a_c["row_right"]).astype(bool)
    u2, c2 = np.asarray(a_c["uncoded_kp2d"], float), np.asarray(a_c["coded_kp2d"], float)
    ones = np.ones((len(t_all), 21, 1))
    targets = {"warp": c2, **{f"warp@S{s}": smoothed_rows(c2, t_all, right_all, s) for s in SIGMAS},
               "warp@Su": smoothed_rows(u2, t_all, right_all, NOISE_SIGMA)}
    size = np.ptp(c2[:, :, 1], axis=1) + np.ptp(c2[:, :, 0], axis=1)  # draw order as k6bh (from the coded keypoints)
    rows_at: dict[int, list[int]] = {}
    for i, t in enumerate(t_all):
        if t < frames:
            rows_at.setdefault(int(t), []).append(i)

    def frame(v: str, t: int) -> np.ndarray:
        truth_t = np.asarray(truth[t])
        if v == "source" or not regions[t].any():
            return truth_t
        sides = {s: clip.class_mask(t, s) for s in HANDS}
        objects = clip.foreground(t) & ~regions[t]
        here = sorted(rows_at.get(t, []), key=lambda i: size[i])
        hands = [(bool(right_all[i]), np.concatenate([u2[i], ones[i]], -1), np.concatenate([targets[v][i], ones[i]], -1),
                  -float(size[i])) for i in here]
        return np.asarray(k1c.compose(truth_t, sides, objects, hands)[0])

    judged = [t for t in range(frames) if t % j_hands.JUDGE_EVERY == 0 and regions[t].any() and rows_at.get(t)]
    pairs = [t for t in judged if t >= 1 and regions[t - 1].any()]
    judge_rows = [i for t in judged for i in rows_at[t]]
    boxes = {i: j_hands.row_box(u2[i]) for i in judge_rows}
    kp2d, flick, energy = [], [], []
    energy_src = np.full(len(pairs), np.nan)
    for v in VARIANTS:
        items = [nets["wilor"].item(frame(v, int(t_all[i])), boxes[i], bool(right_all[i])) for i in judge_rows]
        kp2d.append(nets["wilor"].forward(items)["kp2d"].astype(np.float32))
        f_row, e_row = np.full(len(pairs), np.nan), np.full(len(pairs), np.nan)
        for k, t in enumerate(pairs):
            got = j.residual_energies(frame(v, t), frame(v, t - 1), np.asarray(truth[t]), np.asarray(truth[t - 1]),
                                      regions[t], regions[t - 1])
            if got is not None:
                f_row[k], e_row[k], energy_src[k] = got
        flick.append(f_row)
        energy.append(e_row)
        k1f.release()
    arrays = {"judge_row": np.asarray(judge_rows, np.int64), "kp2d": np.stack(kp2d), "flicker": np.stack(flick),
              "energy": np.stack(energy), "energy_src": energy_src, "pairs": np.asarray(pairs, np.int64)}
    meta = {"unit": item["id"], "item": item["id"], "variants": list(VARIANTS), "frames": frames, "judged_rows": len(judge_rows),
            "pairs": len(pairs), "sigmas": list(SIGMAS), "noise_sigma": NOISE_SIGMA, "mask_set": mask_info,
            "seconds": round(time.time() - began, 2)}
    save_unit(publish, "j2", item["id"], arrays, meta)
    os.remove(decoded["path"])
    shutil.rmtree(work, ignore_errors=True)


def command_run(args: argparse.Namespace) -> int:
    import torch

    from experiments.audit.env_smoke import device_record, model_on_cuda
    from experiments.visor.b2 import select_items
    from experiments.visor.h2 import Regressor, load_units, mano_dir, restore_units
    from experiments.visor.h3c import decode_windows

    items = select_items(json.loads(Path(args.eval_set).read_text()), args.items)
    units = {m["item"]: (m, a) for m, a in load_units(args.k1f_stage, "k1c")}
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    restored = restore_units(publish, "j2")
    windows = decode_windows(args, [it for it in items if it["id"] not in restored], int(args.frames), scratch)
    if not torch.cuda.is_available():
        raise SystemExit("j2 run needs a GPU (WiLoR)")
    args.mano_dir = mano_dir(args, scratch)
    nets = {"wilor": Regressor("wilor", args)}
    began = time.time()
    done = len(restored)
    for item, decoded in windows:
        process_item(args, item, decoded, nets, units[item["id"]], "cuda", publish, scratch)
        done += 1
        progress(done)
    with tarfile.open(stage_dir() / "published.tar", "w") as tar:
        tar.add(publish, arcname="publish")
    write_json(stage_dir() / "j2.json", {
        "command": "run", "device": device_record(torch.cuda.get_device_name(0)), "on_cuda": {"wilor": model_on_cuda(nets["wilor"].model)},
        "seconds": round(time.time() - began, 2), "wilor": nets["wilor"].load,
        "inputs": {"k1f_stages": [file_sha256(Path(t)) for t in args.k1f_stage], "wilor": file_sha256(Path(args.wilor_checkpoint))},
        "settings": {"frames": int(args.frames), "items": [it["id"] for it in items], "variants": list(VARIANTS)}})
    return 0


def validate(stage: Path, result: dict[str, Any]) -> dict[str, bool]:
    from experiments.visor.h2 import load_units

    units = load_units([str(stage / "published.tar")], "j2")
    checks: dict[str, bool] = {"every_item": sorted(m["item"] for m, _ in units) == sorted(result["settings"]["items"]),
                               "on_cuda": all(result["on_cuda"].values())}
    for m, a in units:
        k = m["item"]
        checks[f"{k}:shapes"] = a["kp2d"].shape[0] == len(VARIANTS) == a["flicker"].shape[0] and m["judged_rows"] > 0
        checks[f"{k}:finite"] = bool(np.isfinite(a["kp2d"]).all())
        checks[f"{k}:source_flicker_zero"] = bool(np.nanmax(np.nan_to_num(a["flicker"][0], nan=0.0)) < 1e-4)
        # smoothing changes the drawing: every smoothed variant differs from warp somewhere
        checks[f"{k}:smoothing_draws"] = all(bool(np.nanmax(np.abs(np.nan_to_num(a["flicker"][i] - a["flicker"][1]))) > 0)
                                             for i in range(2, len(VARIANTS)))
    return checks


def command_validate(args: argparse.Namespace) -> int:
    stage = Path(args.stage or os.environ.get("PS_STAGE_DIR") or ".")
    result = json.loads((stage / "j2.json").read_text())
    checks = validate(stage, result)
    out = {"passed": all(checks.values()), "checks": checks}
    write_json(stage / "validation.json", out)
    if target := os.environ.get("PS_VALIDATION_PATH"):
        write_json(Path(target), out)
    print(json.dumps(out, indent=1))
    return 0 if out["passed"] else 1


# ----------------------------------------------------------------- report (CPU)


def command_report(args: argparse.Namespace) -> int:
    from experiments.visor.h2 import load_units

    j_units = {m["item"]: (m, a) for s in args.j_stage for m, a in load_units([str(Path(s) / "published.tar")], "j_hands")}
    items: dict[str, Any] = {}
    stages = []
    for s in args.stage:
        stage = Path(s)
        stages.append({"path": str(stage), "published_sha256": file_sha256(stage / "published.tar")})
        for m, a in load_units([str(stage / "published.tar")], "j2"):
            mj, aj = j_units[m["item"]]
            base = j_hands.item_report(mj, aj)  # J's anchors (SVT-AV1 hulls), keypoint error and source energy
            if not np.array_equal(a["judge_row"], aj["judge_row"]):
                raise RuntimeError(f"{m['item']}: J2 judged other rows than J")
            side = aj["box_side"]
            src = aj["source_kp2d"]
            row: dict[str, Any] = {"kbps": base["kbps"], "pose_anchor": base["pose_anchor"], "flicker_anchor": base["flicker_anchor"],
                                   "keypoints_error": base["keypoints_error"], "source_energy": float(np.nanmean(a["energy_src"])),
                                   "j_warp": {"pose": base["pose_error"]["warp"], "flicker": base["flicker"]["warp"]}, "variants": {}}
            for i, v in enumerate(m["variants"]):
                if v == "source":
                    continue
                pose = float((np.linalg.norm(a["kp2d"][i] - src, axis=-1).mean(-1) / side).mean())
                f, e = float(np.nanmean(a["flicker"][i])), float(np.nanmean(a["energy"][i]))
                checks = {"content": j.content(pose, base["pose_anchor"], base["keypoints_error"], complete=False),
                          "time": j.over_time(f, base["flicker_anchor"], e, row["source_energy"], complete=False)}
                row["variants"][v] = {"pose": pose, "flicker": f, "energy": e, "energy_ratio": e / row["source_energy"], "checks": checks}
            row["warp_reproduces_j"] = (abs(row["variants"]["warp"]["pose"] - row["j_warp"]["pose"]) < CHECK_TOLERANCE
                                        and abs(row["variants"]["warp"]["flicker"] - row["j_warp"]["flicker"]) < CHECK_TOLERANCE)
            items[m["item"]] = row

    def passes(v: str, check: str) -> list[bool | None]:
        return [r["variants"][v]["checks"][check] for r in items.values()]

    fixed = [s for s in SIGMAS if all(x is True for x in passes(f"warp@S{s}", "time"))
             and all(x is not False for x in passes(f"warp@S{s}", "content"))]
    if fixed:
        decision = f"smoothing fixes it (sigma {fixed})"
    elif all(x is False for x in passes("warp@Su", "time")):
        decision = "the estimator is the cause"
    else:
        decision = "unclear"
    report = {"decision": decision, "items": items, "stages": stages,
              "j_stages": [{"path": s, "sha256": file_sha256(Path(s) / "published.tar")} for s in args.j_stage]}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    write_json(Path(args.out) / "j2-report.json", report)
    print(json.dumps({"decision": decision, "items": {k: {"warp_reproduces_j": r["warp_reproduces_j"], "source_energy": round(r["source_energy"], 2),
                      "flicker_anchor": round(r["flicker_anchor"]["value"], 2), "pose_anchor": round(100 * r["pose_anchor"]["value"], 2),
                      "kp": round(100 * r["keypoints_error"], 2),
                      **{v: (round(100 * x["pose"], 2), round(x["flicker"], 2), round(x["energy_ratio"], 2), x["checks"]) for v, x in r["variants"].items()}}
                      for k, r in items.items()}}, indent=1, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments.visor.j2", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    mea = sub.add_parser("measure")
    mea.add_argument("--k1f-stage", action="append", required=True)
    mea.add_argument("--out", required=True)
    run = sub.add_parser("run")
    for flag in ("--eval-set", "--archive", "--masks", "--mask-record", "--items", "--mano-left", "--mano-right", "--wilor-checkpoint"):
        run.add_argument(flag, required=True)
    run.add_argument("--video", action="append", default=[])
    run.add_argument("--k1f-stage", action="append", required=True)
    run.add_argument("--frames", default="240")
    val = sub.add_parser("validate")
    val.add_argument("--stage", default="")
    rep = sub.add_parser("report")
    rep.add_argument("--stage", action="append", required=True)
    rep.add_argument("--j-stage", action="append", required=True, help="J's j_hands stage directories (published.tar)")
    rep.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return {"measure": command_measure, "run": command_run, "validate": command_validate, "report": command_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
