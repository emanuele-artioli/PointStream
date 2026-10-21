"""K1f: the hand bound's scope on held-out items (docs/experiments.md, 2026-10-21 K1f).

    python -m experiments.visor.k1f run <k1c run's arguments> --image-ckpt PTH --video-ckpt PTH --generic TAR ... \\
        --inspect BIN [--cover --hand-box --wilor-checkpoint PTH --mano-left PKL]
    python -m experiments.visor.k1f validate
    python -m experiments.visor.k1f report --stage DIR ... --pilot-report JSON --out JSON

Per held-out item (B2's other 30, in parts): the pixel envelope near the keypoint rate on R, coded as the pilot's was
(``h3d.code_point``: H3b2's SVT-AV1 low points, DCVC-UF HT-L off the shelf, and G's network on DCVC-UF's ½ and ¼ points,
``h3d.apply``), and K1c's bound (``k1c.process_item``) on the same decoded window. Points above 20 kbps are left out:
on the pilot the hull with and without them is identical at every rate up to 5 kbps.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.visor import k1c
from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json

SVT_POINTS = ((63.0, 1), (63.0, 2), (63.0, 4), (63.0, 8), (63.0, 16), (55.0, 4), (55.0, 8))  # H3b2's low points
DCVC_POINTS = ((0, 1), (0, 2), (21, 2), (42, 2), (0, 4), (21, 4), (42, 4))
G_POINTS = ((0, 2), (21, 2), (42, 2), (0, 4), (21, 4), (42, 4))  # H3d's G-L networks
#: G's training footage (h3dc-footage-G.json): items from these videos overlap G's training (reported apart).
G_VIDEOS = ("P06_107", "P06_14", "P07_110", "P11_103", "P22_107", "P33_109")
COVERAGE_THRESHOLD = 0.8  # H2 rows / boxable hand side-frames; pilot: winners 0.99 and 0.97, losers 0.51 and 0.54
CRITERION = 0.75  # the user's: mean over {LPIPS, DISTS} on R of the share of items below the hull
DECISION: dict[str, Any] = {
    "scope": f"H2's coverage (rows / boxable hand side-frames) >= {COVERAGE_THRESHOLD}, measured at the encoder",
    "criterion": "mean over {LPIPS on R, DISTS on R} of the share of items whose bound is below the hull at its rate >= 0.75",
    "headline": "the scope holds if in-scope items meet the criterion; reported also for out-of-scope items, all items, and "
                "the items clean of G's training videos",
    "envelope": "per item, g5.envelope of SVT-AV1's low points, DCVC-UF off the shelf and DCVC-UF + G, on R, up to 20 kbps",
}


def release() -> None:
    """Return PyTorch's cached GPU memory between points (the fleet charges the job's processes; job
    20261021T211220Z-3d431acd overran 36,000 MiB at 38,200 with the cache held)."""
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()


def point_record(name: str, rec: dict[str, Any], scored: dict[str, np.ndarray], hand: np.ndarray) -> dict[str, Any]:
    return {"name": name, "kbps": float(rec["kbps_r"]), "lpips": float(np.nanmean(scored["lpips"][hand])),
            "dists": float(np.nanmean(scored["dists"][hand])), "record": {k: v for k, v in rec.items() if k != "stream"}}


def process_item(args: argparse.Namespace, item: dict[str, Any], decoded: dict[str, Any], nets: dict[str, Any], device: str,
                 publish: Path, scratch: Path, first: bool) -> None:
    from experiments.visor import h3d
    from experiments.visor.h2 import save_unit
    from experiments.visor.h3 import rgb_memmap
    from experiments.visor.h3c import Scorer, load_window

    began = time.time()
    frames = int(args.frames)
    width, height = decoded["width"], decoded["height"]
    fps = Fraction(item["fps"]).limit_denominator(1001)
    work = scratch / "work" / item["id"]
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    planes, regions, mask_info = load_window(args, item, frames, decoded)
    truth = rgb_memmap(Path(decoded["path"]), work / "truth.npy", frames, width, height, decoded["full_range"], device)
    score = Scorer(nets, truth, regions, device)
    hand = np.array(score.hand, int)
    points: list[dict[str, Any]] = []
    lp_rows, dists_rows = [], []
    for crf, scale in SVT_POINTS:
        rgb, rec = h3d.code_point(args, "svt", crf, scale, decoded, planes, regions, frames, fps, work, device)
        s = score(rgb, upscale=scale > 1)
        points.append(point_record(h3d.point_name("svt", crf, scale), rec, s, hand))
        lp_rows.append(s["lpips"])
        dists_rows.append(s["dists"])
        del rgb
        release()
        for f in work.glob("low-*"):
            shutil.rmtree(f) if f.is_dir() else f.unlink()
    for qp, scale in args.dcvc_points:
        rgb, rec = h3d.code_point(args, "dcvc", qp, scale, decoded, planes, regions, frames, fps, work, device)
        upscaled = h3d.upscale_all(rgb, frames, width, height) if scale > 1 else np.asarray(rgb)
        del rgb
        s = score(upscaled, upscale=False)
        name = h3d.point_name("dcvc", qp, scale)
        points.append(point_record(name, rec, s, hand))
        lp_rows.append(s["lpips"])
        dists_rows.append(s["dists"])
        if (qp, scale) in G_POINTS:
            out = h3d.apply(nets["g"][name], upscaled, regions, device)
            s = score(out, upscale=False)
            points.append(point_record(f"G-L:{name}", rec, s, hand))
            lp_rows.append(s["lpips"])
            dists_rows.append(s["dists"])
            del out
        del upscaled
        release()
        (work / "dcvc-rgb.npy").unlink(missing_ok=True)
        shutil.rmtree(work / f"dcvc-{scale}-{qp:g}", ignore_errors=True)
    for f in work.glob("source-*.yuv"):
        f.unlink()
    for f in work.glob("region-*.npy"):
        f.unlink()
    del truth, planes
    arrays = {"pixels_r": np.array([int(r.sum()) for r in regions], np.int64), "lpips": np.stack(lp_rows), "dists": np.stack(dists_rows)}
    meta = {"unit": item["id"], "item": item["id"], "video": item["video"], "frames": frames, "hand_frames": len(score.hand),
            "points": points, "g_overlap": item["video"] in G_VIDEOS, "mask_set": mask_info, "seconds": round(time.time() - began, 2)}
    save_unit(publish, "k1f", item["id"], arrays, meta)
    shutil.rmtree(work, ignore_errors=True)
    release()
    k1c.process_item(args, item, decoded, nets, device, publish, scratch, first)  # the bound; removes the decoded window


def command_run(args: argparse.Namespace) -> int:
    import torch

    from experiments.audit.env_smoke import device_record
    from experiments.visor import h3d, mano_np
    from experiments.visor import pose_coding as pc
    from experiments.visor.b2 import select_items
    from experiments.visor.h2 import load_units, restore_units
    from experiments.visor.h3b import load_dists
    from experiments.visor.h3c import combo_at, decode_windows
    from experiments.visor.h3d_component import generic_network
    from src.codecs.quality import load_lpips, lpips_record

    items = select_items(json.loads(Path(args.eval_set).read_text()), args.items)
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    done_k1f, done_k1c = restore_units(publish, "k1f"), restore_units(publish, "k1c")
    todo = [it for it in items if it["id"] not in done_k1f or it["id"] not in done_k1c]
    copy = scratch / "bin" / Path(args.inspect).name  # the staging cache drops the execute bit (H1)
    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.inspect, copy)
    copy.chmod(0o755)
    args.inspect_source, args.inspect = args.inspect, str(copy)
    args.image_sha256 = file_sha256(Path(args.image_ckpt))
    args.video_sha256 = file_sha256(Path(args.video_ckpt))
    windows = decode_windows(args, todo, int(args.frames), scratch)
    if not torch.cuda.is_available():
        raise SystemExit("k1f run needs a GPU (DCVC-UF)")
    device = "cuda"
    code = json.loads(Path(args.pose_code).read_text())
    combo = combo_at(code, "wilor", k1c.BUDGET)
    args.combos = {"wilor": combo, "hamer": combo}
    args.mano = mano_np.Mano.load(args.mano_right)
    args.basis = pc.Basis(args.mano.hands_mean, args.mano.hands_components)
    args.pose_units = {meta["unit"]: (meta, a) for meta, a in load_units(args.poses, "visor")}
    if missing := [it["id"] for it in items if it["id"] not in args.pose_units]:
        raise SystemExit(f"no H2 poses for {missing}")
    nets: dict[str, Any] = {"lpips": load_lpips(Path(args.lpips_backbone), device), "dists": load_dists(Path(args.dists_backbone), device)}
    g_info = {}
    nets["g"] = {}
    args.dcvc_points = [tuple(int(v) for v in p.split("/")) for p in args.dcvc.split(",")]
    if not set(args.dcvc_points) <= set(DCVC_POINTS):
        raise SystemExit(f"--dcvc: points outside {DCVC_POINTS}")
    for qp, scale in G_POINTS:
        if (qp, scale) not in args.dcvc_points:
            continue
        name = h3d.point_name("dcvc", qp, scale)
        nets["g"][name], info = generic_network(args.generic, name, device)
        g_info[name] = {"sha256": info["sha256"], "data": info["data"]}
    on_cuda = {k: next(v.parameters()).is_cuda for k, v in nets.items() if k != "g"}
    if args.cover:
        from experiments.audit.env_smoke import model_on_cuda
        from experiments.visor.h2 import Regressor, mano_dir

        args.mano_dir = mano_dir(args, scratch)
        nets["wilor"] = Regressor("wilor", args)
        on_cuda["wilor"] = model_on_cuda(nets["wilor"].model)
    began = time.time()
    done = len(items) - len(todo)
    for number, (item, decoded) in enumerate(windows):
        process_item(args, item, decoded, nets, device, publish, scratch, first=number == 0)
        done += 1
        progress(done)
    import torchmetrics

    from experiments.visor.h3b import _tool

    settings = {"frames": int(args.frames), "items": [it["id"] for it in items], "skeleton": k1c.HAND_SKELETON,
                "sigma_of_height": k1c.SIGMA_OF_HEIGHT, "cover": bool(args.cover), "hand_box": bool(args.hand_box),
                "decision": DECISION, "svt_points": SVT_POINTS, "dcvc_points": args.dcvc_points,
                "g_points": [p for p in G_POINTS if p in args.dcvc_points]}
    record = {"device": device_record(torch.cuda.get_device_name(0)), "on_cuda": on_cuda, "seconds": round(time.time() - began, 2),
              "lpips": lpips_record(Path(args.lpips_backbone)),
              "dists": {"torchmetrics": torchmetrics.__version__, "backbone_sha256": file_sha256(Path(args.dists_backbone))},
              "pose": {"budget_ms": k1c.BUDGET, "combo": pc.combo_key(combo), "visor_kbps": float(code["choices"]["wilor"][k1c.BUDGET]["visor_kbps"]),
                       "code_sha256": file_sha256(Path(args.pose_code)), "poses_sha256": [file_sha256(Path(x)) for x in args.poses],
                       "mano_sha256": file_sha256(Path(args.mano_right))},
              "dcvc": {"image_sha256": args.image_sha256, "video_sha256": args.video_sha256}, "g": g_info,
              "svt": {"decoder": _tool(args.decoder), "encoder": _tool(args.encoder), "preset": args.preset,
                      "inspect": {"path": args.inspect_source, "sha256": file_sha256(Path(args.inspect))}},
              "settings": settings, "restored": sorted(set(done_k1f) & set(done_k1c))}
    write_json(stage_dir() / "k1f.json", {"command": "run", **record})
    write_json(stage_dir() / "k1c.json", {"command": "run", **{k: v for k, v in record.items() if k != "g"},
                                          "settings": {**settings, "decision": k1c.DECISION}})
    return 0


# ----------------------------------------------------------------- validation (CPU)


def validate(stage: Path, result: dict[str, Any]) -> dict[str, bool]:
    from experiments.visor.h2 import load_units

    checks = k1c.validate(stage, json.loads((stage / "k1c.json").read_text()))
    units = {m["item"]: (m, a) for m, a in load_units([str(stage / "published.tar")], "k1f")}
    checks["k1f:every_item"] = sorted(units) == sorted(result["settings"]["items"])
    for item, (m, a) in units.items():
        hand = a["pixels_r"] > 0
        names = [p["name"] for p in m["points"]]
        expected = len(SVT_POINTS) + len(result["settings"]["dcvc_points"]) + len(result["settings"]["g_points"])
        checks[f"{item}:k1f_points"] = len(names) == expected == len(a["lpips"])
        checks[f"{item}:k1f_finite"] = bool(np.isfinite(a["lpips"][:, hand]).all() and np.isfinite(a["dists"][:, hand]).all())
        rate = {p["name"]: p["kbps"] for p in m["points"]}
        lp = {p["name"]: p["lpips"] for p in m["points"]}
        checks[f"{item}:k1f_svt_coarsens"] = lp["svt@63/16"] > lp["svt@63/4"] > lp["svt@63"] and rate["svt@63/16"] < rate["svt@63"]
        checks[f"{item}:k1f_dcvc_coarsens"] = lp["dcvc@0/4"] > lp["dcvc@0/1"] and rate["dcvc@0/4"] < rate["dcvc@0/1"]
        checks[f"{item}:k1f_g_shares_rate"] = all(rate[f"G-L:{n}"] == rate[n] for n in names if not n.startswith("G-L:") and f"G-L:{n}" in rate)
    return checks


def command_validate(args: argparse.Namespace) -> int:
    stage = Path(args.stage) if args.stage else stage_dir()
    result = json.loads((stage / "k1f.json").read_text())
    checks = {k: bool(v) for k, v in validate(stage, result).items()}
    report = {"passed": all(checks.values()), "checks": checks}
    if target := os.environ.get("PS_VALIDATION_PATH"):
        write_json(Path(target), report)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


# ----------------------------------------------------------------- report (CPU)


def criterion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"items": 0}
    lp = float(np.mean([r["lpips_below_hull"] for r in rows]))
    di = float(np.mean([r["dists_below_hull"] for r in rows]))
    return {"items": len(rows), "lpips_share": lp, "dists_share": di, "mean": (lp + di) / 2, "passes": (lp + di) / 2 >= CRITERION,
            "median_lpips_saving": float(np.median([r["lpips_saving"] for r in rows if r["lpips_saving"] is not None]))
            if any(r["lpips_saving"] is not None for r in rows) else None}


def command_report(args: argparse.Namespace) -> int:
    from experiments.background.g5 import envelope
    from experiments.visor.h2 import load_units
    from experiments.visor.k1 import at_rate, rate_for

    rows: list[dict[str, Any]] = []
    sources = []
    for stage in map(Path, args.stage):
        result = json.loads((stage / "k1f.json").read_text())
        checks = validate(stage, result)
        checks["k1f:all_points"] = len(result["settings"]["dcvc_points"]) == len(DCVC_POINTS)  # a full stage, not a smoke
        if not all(checks.values()):
            raise SystemExit(f"{stage} fails validation: {[k for k, v in checks.items() if not v]}")
        sources.append({"stage": str(stage), "sha256": file_sha256(stage / "published.tar")})
        bound = {m["item"]: m for m, _ in load_units([str(stage / "published.tar")], "k1c")}
        for m, _ in load_units([str(stage / "published.tar")], "k1f"):
            b = bound[m["item"]]
            kbps = float(b["cover"]["kbps"]) if b.get("cover") else k1c.POSE_KBPS
            row: dict[str, Any] = {"item": m["item"], "video": m["video"], "g_overlap": m["g_overlap"], "kbps": kbps,
                                   "coverage": b["h2_rows_in_window"] / max(1, b["masked_side_frames"]),
                                   "hand_frames": m["hand_frames"]}
            for metric in ("lpips", "dists"):
                hull = envelope([(p["kbps"], -p[metric]) for p in m["points"]])
                q = at_rate(hull, kbps)
                value = b["summary"]["warp"][f"{metric}_r"]
                row[f"{metric}_bound"], row[f"{metric}_hull"] = value, None if q is None else -q
                row[f"{metric}_below_hull"] = bool(q is not None and value < -q) if q is not None else bool(value < -hull[0][1])
                need = rate_for(hull, -value)
                row[f"{metric}_saving"] = None if need is None else 1.0 - kbps / need
            row["in_scope"] = row["coverage"] >= COVERAGE_THRESHOLD
            rows.append(row)
    groups = {"all": rows, "in_scope": [r for r in rows if r["in_scope"]], "out_of_scope": [r for r in rows if not r["in_scope"]],
              "clean_of_g": [r for r in rows if not r["g_overlap"]],
              "in_scope_clean_of_g": [r for r in rows if r["in_scope"] and not r["g_overlap"]]}
    out = {"inputs": {"stages": sources, "code_sha256": file_sha256(Path(__file__))}, "decision_rule": DECISION,
           "rows": rows, "groups": {k: criterion(v) for k, v in groups.items()},
           "scope_share_of_items": len(groups["in_scope"]) / max(1, len(rows)),
           "scope_share_of_hand_frames": sum(r["hand_frames"] for r in groups["in_scope"]) / max(1, sum(r["hand_frames"] for r in rows))}
    write_json(Path(args.out), out)
    for r in sorted(rows, key=lambda r: -r["coverage"]):
        print(r["item"], f"cov {r['coverage']:.2f}", f"LPIPS {r['lpips_bound']:.3f} vs {r['lpips_hull']}", r["lpips_below_hull"],
              f"DISTS {r['dists_bound']:.3f} vs {r['dists_hull']}", r["dists_below_hull"], "G-overlap" if r["g_overlap"] else "")
    print(json.dumps(out["groups"], indent=1), out["scope_share_of_items"], out["scope_share_of_hand_frames"])
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    for flag in ("--eval-set", "--archive", "--masks", "--mask-record", "--items", "--lpips-backbone", "--dists-backbone",
                 "--pose-code", "--mano-right", "--image-ckpt", "--video-ckpt", "--inspect"):
        run.add_argument(flag, required=True)
    run.add_argument("--video", action="append", default=[])
    run.add_argument("--poses", action="append", required=True)
    run.add_argument("--generic", action="append", required=True, help="H3d's generic jobs' published.tar (G's networks)")
    run.add_argument("--frames", default="240")
    run.add_argument("--dcvc", default=",".join(f"{q}/{s}" for q, s in DCVC_POINTS), help="DCVC-UF points QP/scale (smoke: fewer)")
    run.add_argument("--decoder", default="dav1d")
    run.add_argument("--encoder", default="SvtAv1EncApp")
    run.add_argument("--preset", default="4")
    run.add_argument("--cover", action="store_true")
    run.add_argument("--hand-box", action="store_true")
    run.add_argument("--mano-left", default="")
    run.add_argument("--wilor-checkpoint", default="")
    val = sub.add_parser("validate")
    val.add_argument("--stage", default="")
    rep = sub.add_parser("report")
    rep.add_argument("--stage", action="append", required=True)
    rep.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return {"run": command_run, "validate": command_validate, "report": command_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
