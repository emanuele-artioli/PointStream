"""J on hands: FoundHand `gen` (K6b hands rung 2) re-run and re-judged on the experience track (docs/experiments.md,
2026-10-22 J).

    python -m experiments.visor.j_hands run --eval-set JSON --archive TAR --masks TAR --mask-record JSON --video MP4 ... \\
        --items IDS --frames N --k1f-stage TAR ... --foundhand-src TGZ --foundhand-ckpt CKPT --vae CKPT --steps 50 \\
        --mano-left PKL --mano-right PKL --wilor-checkpoint CKPT --inception PTH \\
        --inspect BIN
    python -m experiments.visor.j_hands validate
    python -m experiments.visor.j_hands report --stage DIR ... --out DIR

Per K1f item, rung 2's rendering unchanged (``k6bh.render_item``: its crops, reference, 50 steps and item-keyed seeds;
the composites as ``k6bh.process_item`` builds them): ``gen``, with ``ref`` and ``warp`` beside it. The anchors are
SVT-AV1's seven low points as K1f codes them (``h3d.code_point``), a subset of the rule's SVT-AV1 + DCVC-UF + G hull,
so they decide only where a check fails (``experiments.visor.j``).

Judges, on every output (the source, the three variants and each anchor composited inside R):

* same content: WiLoR re-run on the output with a box from the row's uncoded keypoints (their box grown by
  `BOX_GROW`, the same box for every output), H3c2's error: the mean 2D joint distance to WiLoR on the source, over
  the box's longer side. Rows at judged frames (t mod 3 = 0). The coded keypoints' own error is the same distance;
* same over time: on R at t against t - 1 for judged t, flow from the source (``j.residual_energies``);
* realism: Inception features of ``h3b``'s square crop around the same box, kept for KID against a held-out set.
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

from experiments.visor import j, k1c, k1f, k6bh
from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json

JUDGE_EVERY = 3
BOX_GROW = 1.2  # the WiLoR box: the uncoded keypoints' box grown about its centre
METHOD = "gen"
BESIDE = ("ref", "warp")
RATE_TOLERANCE = 0.001  # relative: SVT-AV1's points must reproduce K1f's recorded kbps on R


def anchor_names() -> list[str]:
    from experiments.visor import h3d

    return [h3d.point_name("svt", crf, scale) for crf, scale in k1f.SVT_POINTS]


def row_box(points: np.ndarray) -> np.ndarray:
    """The keypoints' box grown by `BOX_GROW` about its centre (x0, y0, x1, y1)."""
    x0, y0 = points[:, 0].min(), points[:, 1].min()
    x1, y1 = points[:, 0].max(), points[:, 1].max()
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = BOX_GROW * max(x1 - x0, 2.0) / 2, BOX_GROW * max(y1 - y0, 2.0) / 2
    return np.array([cx - hw, cy - hh, cx + hw, cy + hh], np.float32)


def compose(rows: dict[str, np.ndarray], rendered: dict[str, Any], truth: Any, clip: Any, regions: list[np.ndarray],
            size: np.ndarray, rows_at: dict[int, list[int]], t: int, v: str) -> np.ndarray:
    """One frame of ``v`` as ``k6bh.process_item`` composes it (``warp``: K1c's compose; else Telea fill, drawn on top)."""
    from experiments.tennis.k6 import draw, telea_fill
    from experiments.visor.h3 import HANDS

    truth_t = np.asarray(truth[t])
    if not regions[t].any():
        return truth_t
    sides = {s: clip.class_mask(t, s) for s in HANDS}
    objects = clip.foreground(t) & ~regions[t]
    here = sorted(rows_at.get(t, []), key=lambda i: size[i])
    if v == "warp":
        hands = [(bool(rows["right"][i]), rows["uncoded"][i], rows["coded"][i], -float(size[i])) for i in here]
        return np.asarray(k1c.compose(truth_t, sides, objects, hands)[0])
    canvas = telea_fill(truth_t, regions[t])
    drawn = np.zeros(regions[t].shape, bool)
    for i in here:
        side = HANDS[1] if rows["right"][i] else HANDS[0]
        mask = sides[side]
        if not mask.any() or side not in rendered["refs"]:
            continue
        if v == "ref":
            ref = rendered["refs"][side]
            ys = np.nonzero(ref["mask"].any(1))[0]
            draw(canvas, drawn, ref["image"], ref["mask"], rows["uncoded"][ref["row"]], rows["coded"][i],
                 k1c.SIGMA_OF_HEIGHT * float(ys.max() - ys.min() + 1), k1c.HAND_SKELETON)
        elif i in rendered[v]:
            drawn |= k6bh.paste(canvas, rendered[v][i], rendered["boxes"][i], mask)
    over = drawn & objects
    canvas[over] = truth_t[over]
    return canvas


def process_item(args: argparse.Namespace, item: dict[str, Any], decoded: dict[str, Any], nets: dict[str, Any], fh: Any,
                 k1f_units: tuple[Any, Any], device: str, publish: Path, scratch: Path) -> None:
    import cv2

    from experiments.visor import h3d
    from experiments.visor.b2 import load_mask_sets
    from experiments.visor.h2 import save_unit
    from experiments.visor.h3 import FILL, HANDS, rgb_memmap
    from experiments.visor.h3b import CROP_GROW, INCEPTION_SIZE, inception_features, square_crop
    from experiments.visor.h3c import load_window

    began = time.time()
    frames = int(args.frames)
    width, height = decoded["width"], decoded["height"]
    fps = Fraction(item["fps"]).limit_denominator(1001)
    work = scratch / "work" / item["id"]
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    planes, regions_r, mask_info = load_window(args, item, frames, decoded)
    clips, _ = load_mask_sets(item, frames, {FILL: args.masks}, {FILL: args.mask_record}, Path(args.archive))
    clip = clips[FILL]
    regions = [clip.class_mask(t, HANDS[0]) | clip.class_mask(t, HANDS[1]) for t in range(frames)]
    if any(not np.array_equal(a, b) for a, b in zip(regions, regions_r)):
        raise RuntimeError(f"{item['id']}: k6bh's R and K1f's R differ")
    truth = rgb_memmap(Path(decoded["path"]), work / "truth.npy", frames, width, height, decoded["full_range"], device)
    (meta_c, a_c), meta_f = k1f_units
    ones = np.ones((len(a_c["row_t"]), 21, 1))
    rows = {"t": np.asarray(a_c["row_t"]).astype(int), "right": np.asarray(a_c["row_right"]).astype(bool),
            "uncoded": np.concatenate([a_c["uncoded_kp2d"], ones], -1), "coded": np.concatenate([a_c["coded_kp2d"], ones], -1)}
    size = np.ptp(rows["coded"][:, :, 1], axis=1) + np.ptp(rows["coded"][:, :, 0], axis=1)
    rows_at: dict[int, list[int]] = {}
    for i, t in enumerate(rows["t"]):
        if t < frames:
            rows_at.setdefault(int(t), []).append(i)

    rendered = k6bh.render_item(fh, rows, truth, clip, frames, k6bh.item_seed(item["id"]), (METHOD,))
    k1f.release()
    timing = {"render": round(time.time() - began, 1)}

    judged = [t for t in range(frames) if t % JUDGE_EVERY == 0 and regions[t].any() and rows_at.get(t)]
    pairs = [t for t in judged if t >= 1 and regions[t - 1].any()]
    judge_rows = [i for t in judged for i in rows_at[t]]
    boxes = {i: row_box(rows["uncoded"][i][:, :2]) for i in judge_rows}
    box_side = np.array([max(boxes[i][2] - boxes[i][0], boxes[i][3] - boxes[i][1]) for i in judge_rows], float)

    variants: list[str] = []
    kp2d: list[np.ndarray] = []
    feats: list[np.ndarray] = []
    flick: list[np.ndarray] = []
    energy: list[np.ndarray] = []
    energy_src = np.full(len(pairs), np.nan)
    looks: dict[str, np.ndarray] = {}
    look_t = judged[len(judged) // 2] if judged else 0

    def judge(name: str, frame_of: Any) -> None:
        cache: dict[int, np.ndarray] = {}

        def get(t: int) -> np.ndarray:
            if t not in cache:
                cache[t] = frame_of(t)
            return cache[t]

        items = [nets["wilor"].item(get(int(rows["t"][i])), boxes[i], bool(rows["right"][i])) for i in judge_rows]
        kp2d.append(nets["wilor"].forward(items)["kp2d"].astype(np.float32) if items else np.zeros((0, 21, 2), np.float32))
        crops = [square_crop(get(int(rows["t"][i])), boxes[i], CROP_GROW, INCEPTION_SIZE) for i in judge_rows]
        feats.append(inception_features(nets["inception"], crops, device) if crops else np.zeros((0, 2048), np.float16))
        f_row, e_row = np.full(len(pairs), np.nan), np.full(len(pairs), np.nan)
        for k, t in enumerate(pairs):
            got = j.residual_energies(get(t), get(t - 1), np.asarray(truth[t]), np.asarray(truth[t - 1]), regions[t], regions[t - 1])
            if got is not None:
                f_row[k], e_row[k], energy_src[k] = got
        flick.append(f_row)
        energy.append(e_row)
        looks[name] = get(look_t)
        variants.append(name)
        k1f.release()

    judge("source", lambda t: np.asarray(truth[t]))
    for v in (METHOD, *BESIDE):
        judge(v, lambda t, v=v: compose(rows, rendered, truth, clip, regions, size, rows_at, t, v))
    timing["variants"] = round(time.time() - began, 1)
    anchors: dict[str, dict[str, Any]] = {}
    for crf, scale in k1f.SVT_POINTS:
        rgb, rec = h3d.code_point(args, "svt", crf, scale, decoded, planes, regions, frames, fps, work, device)
        name = h3d.point_name("svt", crf, scale)

        def anchor(t: int, rgb: Any = rgb, scale: int = scale) -> np.ndarray:
            frame = np.asarray(rgb[t])
            if scale > 1:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
            out = np.asarray(truth[t]).copy()
            out[regions[t]] = frame[regions[t]]
            return out

        judge(name, anchor)
        anchors[name] = {"kbps_r": float(rec["kbps_r"]), "kbps": float(rec["kbps"]),
                         "record": {k: v for k, v in rec.items() if k != "stream"}}
        del rgb
        for f in work.glob("low-*"):
            shutil.rmtree(f) if f.is_dir() else f.unlink()
    timing["anchors"] = round(time.time() - began, 1)

    source_kp = kp2d[0]
    coded_kp = rows["coded"][judge_rows][:, :, :2].astype(np.float32)
    uncoded_kp = rows["uncoded"][judge_rows][:, :, :2].astype(np.float32)
    (publish / "frames").mkdir(parents=True, exist_ok=True)
    if judged:
        ys, xs = np.nonzero(regions[look_t])
        box = (slice(max(0, ys.min() - 64), min(height, ys.max() + 65)), slice(max(0, xs.min() - 64), min(width, xs.max() + 65)))
        pick = ["source", METHOD, *BESIDE, "svt@63/4", "svt@63/8"]
        strip = np.concatenate([looks[v][box] for v in pick if v in looks], 1)
        cv2.imwrite(str(publish / "frames" / f"{item['id']}-f{look_t:03d}-{'-'.join(p.replace('@', '_').replace('/', 's') for p in pick)}.jpg"),
                    cv2.cvtColor(strip, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        crops_look = [rendered[METHOD][i] for i in rows_at.get(look_t, []) if i in rendered[METHOD]]
        if crops_look:
            cv2.imwrite(str(publish / "frames" / f"{item['id']}-f{look_t:03d}-gen-crops.jpg"),
                        cv2.cvtColor(np.concatenate(crops_look, 1), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    arrays = {"judge_row": np.asarray(judge_rows, np.int64), "judge_t": rows["t"][judge_rows].astype(np.int64),
              "box_side": box_side, "kp2d": np.stack(kp2d), "source_kp2d": source_kp, "coded_kp2d": coded_kp,
              "uncoded_kp2d": uncoded_kp, "features": np.stack(feats).astype(np.float16), "flicker": np.stack(flick),
              "energy": np.stack(energy), "energy_src": energy_src, "pairs": np.asarray(pairs, np.int64)}
    recorded = {p["name"]: float(p["kbps"]) for p in meta_f["points"]} if meta_f else {}
    meta = {"unit": item["id"], "item": item["id"], "video": item["video"], "frames": frames, "variants": variants,
            "judged_frames": len(judged), "judged_rows": len(judge_rows), "pairs": len(pairs), "anchors": anchors,
            "recorded_kbps": {n: recorded.get(n) for n in anchor_names()},
            "method_kbps": float(meta_c["cover"]["kbps"]) if meta_c.get("cover") else k1c.POSE_KBPS,
            "references": {s: {"t": r["t"], "box": list(r["box"])} for s, r in rendered["refs"].items()},
            "render_seconds": rendered["seconds"], "g_overlap": item["video"] in k1f.G_VIDEOS, "mask_set": mask_info,
            "timing": timing, "seconds": round(time.time() - began, 2)}
    save_unit(publish, "j_hands", item["id"], arrays, meta)
    os.remove(decoded["path"])
    shutil.rmtree(work, ignore_errors=True)


def command_run(args: argparse.Namespace) -> int:
    import torch

    from experiments.audit.env_smoke import device_record, model_on_cuda
    from experiments.visor.b2 import select_items
    from experiments.visor.h2 import Regressor, load_units, mano_dir, restore_units
    from experiments.visor.h3b import _tool, load_inception
    from experiments.visor.h3c import decode_windows

    items = select_items(json.loads(Path(args.eval_set).read_text()), args.items)
    k1c_units = {m["item"]: (m, a) for m, a in load_units(args.k1f_stage, "k1c")}
    k1f_meta = {m["item"]: m for m, _ in load_units(args.k1f_stage, "k1f")}
    if missing := [it["id"] for it in items if it["id"] not in k1c_units]:
        raise SystemExit(f"no K1f rows for {missing}")
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    restored = restore_units(publish, "j_hands")
    todo = [it for it in items if it["id"] not in restored]
    copy = scratch / "bin" / Path(args.inspect).name  # the staging cache drops the execute bit (H1)
    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.inspect, copy)
    copy.chmod(0o755)
    args.inspect_source, args.inspect = args.inspect, str(copy)
    windows = decode_windows(args, todo, int(args.frames), scratch)
    if not torch.cuda.is_available():
        raise SystemExit("j_hands run needs a GPU")
    device = "cuda"
    args.mano_dir = mano_dir(args, scratch)
    nets: dict[str, Any] = {"wilor": Regressor("wilor", args), "inception": load_inception(Path(args.inception), device)}
    fh = k6bh.FoundHand(Path(args.foundhand_src), Path(args.foundhand_ckpt), Path(args.vae), int(args.steps), device, scratch)
    on_cuda = {"wilor": model_on_cuda(nets["wilor"].model), "inception": next(nets["inception"].parameters()).is_cuda,
               "foundhand": model_on_cuda(fh.model), "vae": model_on_cuda(fh.vae)}
    began = time.time()
    done = len(restored)
    for item, decoded in windows:
        process_item(args, item, decoded, nets, fh, (k1c_units[item["id"]], k1f_meta.get(item["id"])), device, publish, scratch)
        done += 1
        progress(done)
    with tarfile.open(stage_dir() / "published.tar", "w") as tar:
        tar.add(publish, arcname="publish")
    write_json(stage_dir() / "j_hands.json", {
        "command": "run", "device": device_record(torch.cuda.get_device_name(0)), "on_cuda": on_cuda,
        "seconds": round(time.time() - began, 2), "foundhand": fh.record, "wilor": nets["wilor"].load,
        "inputs": {"foundhand_src": file_sha256(Path(args.foundhand_src)), "foundhand_ckpt": file_sha256(Path(args.foundhand_ckpt)),
                   "vae": file_sha256(Path(args.vae)), "k1f_stages": [file_sha256(Path(t)) for t in args.k1f_stage],
                   "inception": file_sha256(Path(args.inception)), "wilor": file_sha256(Path(args.wilor_checkpoint)),
                   "inspect": {"path": args.inspect_source, "sha256": file_sha256(Path(args.inspect))}},
        "svt": {"decoder": _tool(args.decoder), "encoder": _tool(args.encoder), "preset": args.preset},
        "settings": {"frames": int(args.frames), "items": [it["id"] for it in items], "steps": int(args.steps), "method": METHOD,
                     "beside": list(BESIDE), "anchors": anchor_names(), "judge_every": JUDGE_EVERY, "box_grow": BOX_GROW,
                     "rule": j.DECISION},
        "restored_units": sorted(restored)})
    return 0


# ----------------------------------------------------------------- validation (CPU)


def validate(stage: Path, result: dict[str, Any]) -> dict[str, bool]:
    from experiments.visor.h2 import load_units

    units = load_units([str(stage / "published.tar")], "j_hands")
    full = int(result["settings"]["frames"]) >= 240
    checks: dict[str, bool] = {"every_item": sorted(m["item"] for m, _ in units) == sorted(result["settings"]["items"]),
                               "on_cuda": all(result["on_cuda"].values())}
    for m, a in units:
        k = m["item"]
        n_v = len(m["variants"])
        checks[f"{k}:variants"] = m["variants"] == ["source", METHOD, *BESIDE, *anchor_names()]
        checks[f"{k}:shapes"] = a["kp2d"].shape[0] == n_v and a["features"].shape[0] == n_v and a["flicker"].shape[0] == n_v
        checks[f"{k}:judged"] = m["judged_rows"] > 0 and a["kp2d"].shape[1] == m["judged_rows"]
        checks[f"{k}:finite"] = bool(np.isfinite(a["kp2d"]).all() and np.isfinite(a["features"].astype(np.float32)).all())
        checks[f"{k}:source_flicker_zero"] = bool(np.nanmax(np.nan_to_num(a["flicker"][0], nan=0.0)) < 1e-4)
        checks[f"{k}:method_differs"] = bool(np.abs(a["kp2d"][1] - a["kp2d"][0]).max() > 0)
        # WiLoR on the source lands near the rows' own keypoints (K1f's WiLoR rows and hand-box fits): median < 15% of the box
        err = np.linalg.norm(a["source_kp2d"] - a["uncoded_kp2d"], axis=-1).mean(-1) / a["box_side"]
        checks[f"{k}:source_rerun_near_rows"] = bool(np.median(err) < 0.15)
        if full:
            checks[f"{k}:rates_reproduce_k1f"] = all(m["recorded_kbps"][n] is not None and
                                                     abs(m["anchors"][n]["kbps_r"] / m["recorded_kbps"][n] - 1) <= RATE_TOLERANCE
                                                     for n in anchor_names())
    return checks


def command_validate(args: argparse.Namespace) -> int:
    stage = Path(args.stage or os.environ.get("PS_STAGE_DIR") or ".")
    result = json.loads((stage / "j_hands.json").read_text())
    checks = validate(stage, result)
    out = {"passed": all(checks.values()), "checks": checks}
    write_json(stage / "validation.json", out)
    if target := os.environ.get("PS_VALIDATION_PATH"):  # where the fleet reads the smoke gate
        write_json(Path(target), out)
    print(json.dumps(out, indent=1))
    return 0 if out["passed"] else 1


# ----------------------------------------------------------------- report (CPU)


def item_report(m: dict[str, Any], a: dict[str, np.ndarray]) -> dict[str, Any]:
    names = m["variants"]
    side = a["box_side"]

    def err(i: int) -> float:
        return float((np.linalg.norm(a["kp2d"][i] - a["source_kp2d"], axis=-1).mean(-1) / side).mean())

    pose = {v: err(i) for i, v in enumerate(names)}
    kp = float((np.linalg.norm(a["coded_kp2d"] - a["source_kp2d"], axis=-1).mean(-1) / side).mean())
    flick = {v: float(np.nanmean(a["flicker"][i])) if np.isfinite(a["flicker"][i]).any() else None for i, v in enumerate(names)}
    energy = {v: float(np.nanmean(a["energy"][i])) if np.isfinite(a["energy"][i]).any() else None for i, v in enumerate(names)}
    src_energy = float(np.nanmean(a["energy_src"])) if np.isfinite(a["energy_src"]).any() else None
    rate = float(m["method_kbps"])
    pose_anchor = j.read([(m["anchors"][n]["kbps_r"], pose[n]) for n in anchor_names()], rate)
    flick_anchor = j.read([(m["anchors"][n]["kbps_r"], flick[n]) for n in anchor_names()], rate)
    out: dict[str, Any] = {"kbps": rate, "pose_error": pose, "keypoints_error": kp, "pose_anchor": pose_anchor,
                           "flicker": flick, "energy": energy, "source_energy": src_energy, "flicker_anchor": flick_anchor,
                           "judged_rows": m["judged_rows"], "pairs": m["pairs"], "g_overlap": m["g_overlap"], "variants": {}}
    for v in (METHOD, *BESIDE):
        checks = {"content": j.content(pose[v], pose_anchor, kp, complete=False),
                  "time": j.over_time(flick[v], flick_anchor, energy[v], src_energy, complete=False), "realism": None}
        out["variants"][v] = {"checks": checks, "status": j.item_status(checks)}
    return out


def command_report(args: argparse.Namespace) -> int:
    from experiments.visor.h2 import load_units

    items: dict[str, dict[str, Any]] = {}
    stages = []
    for s in args.stage:
        stage = Path(s)
        stages.append({"path": str(stage), "published_sha256": file_sha256(stage / "published.tar"),
                       "j_hands_sha256": file_sha256(stage / "j_hands.json")})
        for m, a in load_units([str(stage / "published.tar")], "j_hands"):
            items[m["item"]] = item_report(m, a)
    report: dict[str, Any] = {"result": "FoundHand gen (K6b hands rung 2)", "rule": j.DECISION, "stages": stages, "items": items,
                              "anchor": "SVT-AV1's seven low points (K1f's): a subset of SVT-AV1 + DCVC-UF + G, conclusive where it fails"}
    for v in (METHOD, *BESIDE):
        statuses = {k: r["variants"][v]["status"] for k, r in items.items()}
        report[v] = {"decision": j.decide(statuses), "checks": j.check_rates({k: r["variants"][v]["checks"] for k, r in items.items()})}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "j-hands.json", report)
    print(json.dumps({v: report[v] for v in (METHOD, *BESIDE)}, indent=1))
    print(json.dumps({k: {"kbps": round(r["kbps"], 2), "pose": {v: round(100 * x, 2) for v, x in r["pose_error"].items()},
                          "kp": round(100 * r["keypoints_error"], 2), "pose_anchor": r["pose_anchor"],
                          "flicker": r["flicker"], "energy": r["energy"], "src_energy": r["source_energy"],
                          "checks": {v: r["variants"][v]["checks"] for v in r["variants"]}} for k, r in items.items()}, indent=1, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments.visor.j_hands", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    for flag in ("--eval-set", "--archive", "--masks", "--mask-record", "--items", "--foundhand-src", "--foundhand-ckpt", "--vae",
                 "--mano-left", "--mano-right", "--wilor-checkpoint", "--inception", "--inspect"):
        run.add_argument(flag, required=True)
    run.add_argument("--video", action="append", default=[])
    run.add_argument("--k1f-stage", action="append", required=True, help="K1f's published.tar (rows, its points' rates)")
    run.add_argument("--frames", default="240")
    run.add_argument("--steps", default="50", help="FoundHand's respaced sampling steps (rung 2: 50)")
    run.add_argument("--encoder", default="SvtAv1EncApp")
    run.add_argument("--decoder", default="dav1d")
    run.add_argument("--preset", default="4")
    val = sub.add_parser("validate")
    val.add_argument("--stage", default="")
    rep = sub.add_parser("report")
    rep.add_argument("--stage", action="append", required=True)
    rep.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return {"run": command_run, "validate": command_validate, "report": command_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
