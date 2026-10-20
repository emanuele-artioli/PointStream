"""PLAN step K3: keypoint-conditioned pixel coding, the oracle (docs/experiments.md, 2026-10-16).

    python -m experiments.visor.k3 code --eval-set JSON --archive DIR --masks DIR --mask-record JSON --video V.MP4 ... \\
        --items ID --frames N --image-ckpt PTH --video-ckpt PTH --lpips-backbone PTH --dists-backbone PTH \\
        [--qps 0,21,42] [--scales 1,2,4] [--arms A,B,C,D,E]
    python -m experiments.visor.k3 validate
    python -m experiments.visor.k3 report --code TAR ... --h3c TAR ... --h1-report JSON --pose-code JSON --out DIR

``code`` codes each window at H3c's points with DCVC-UF HT-L off the shelf, once per arm of `k3_ref.ARMS` (its
reference replaced at every NAL by a frame showing the source's hand where the keypoints would put it, and the
controls), through DCVC's PyTorch path; each NAL's estimated bits are calibrated by the unmodified codec's real bytes
for that NAL (B2's worker, encode only). Every arm's decode is scored by H3c's `Scorer` on the hand region R. Each
window is published as ``code/<item>@<label>`` and checkpointed. ``report`` applies `DECISION`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json
from experiments.visor.h2 import load_units, restore_units, save_unit
from experiments.visor.h3 import RESAMPLES, bootstrap_mean, hand_kbps, rgb_memmap, seed_of
from experiments.visor.h3b import DISTS_PAD, dists_crops, load_dists, scaled_size
from experiments.visor.h3c import (QPS, SCALES, Scorer, decode_windows, downscale_mask, downscale_planes, load_window,
                                   point_summary, run_tree_script)

ARMS = ("A", "B", "C", "D", "E", "F")  # `k3_ref.ARMS`
BILLED = ("C", "D")  # the arms a keypoint warp would replace: they carry H2's coded keypoints
POSE_BUDGET = "100"  # H2's WiLoR choice named for H3 (H3c2's keypoints)
KEYPOINT_GRID = (0.0, 0.25, 0.5, 1.0)  # kbps, beside H2's rate: where C/D would pass if the keypoints cost less
PASTED = ("C", "D", "F")  # the arms whose reference changes inside the hand region only
FLOW_TOLERANCE = 0.25  # luma levels: the validator's allowance for resampling noise where the hand stands still
FLOW_KEYPOINTS = "gauss:33.3|10Hz-linear|pca12|q8|previous"  # H2's best WiLoR coding at about 1 kbps (+11% 2D on HOT3D)
METRICS = ("lpips", "dists", "psnr")
SEED = "pointstream-k3"

#: The decision rule, fixed before any fleet run (docs/experiments.md, K3 entry).
DECISION: dict[str, Any] = {
    "quality": "lpips_r (H3c's Scorer: the decode inside R, the source outside; LPIPS averaged over R, mean over hand frames)",
    "rate": "kbps on R: DCVC's per-cell estimate on R, each NAL calibrated by the unmodified codec's real bytes for it",
    "keypoints": f"H2's coded WiLoR stream at {POSE_BUDGET} ms (its VISOR rate in h2-code.json) added to C and D",
    "gate": "the better of C and D (by mean BD), billed, against A on LPIPS_R; each arm's upper concave hull against the "
            "anchor's (amended before any result: a front against a hull charged A itself +13% in a dry run)",
    "bd": "experiments.visor.h3d.compare: PCHIP of log rate over the shared LPIPS range (>= 0.005), else better/worse "
          "at every rate",
    "verdict": "experiments.visor.h3d.arm_verdict: passes at mean BD <= -10% with >= 3 of 4 items below 0 and none worse "
               "at every rate; fails at mean >= 0 or <= 1 item below 0; otherwise unclear",
    "then": {"passes": "build the keypoint warp (H3b's MANO correspondence) as the next stage, with its own rule",
             "fails": "K3 closed: neither the warp nor a trained conditional coder is pursued",
             "unclear": "to the user with the numbers"},
    "beside": ["B vs A (the memory's worth)", "C, D vs B unbilled (the hand pixels' value on a fixed path)",
               "E vs B (the most any reference saves: the motion-bit share's counterpart)", "C, D billed vs the pixel envelope",
               "C, D vs A unbilled and at lower keypoint rates"],
}


#: K3b's rule, fixed before its run (docs/experiments.md, 2026-10-17).
FLOW_DECISION: dict[str, Any] = {
    "gate": "F (x̂(t0-1) warped to t0 inside R(t0-1) | R(t0) by DIS flow on the source) unbilled against A on LPIPS_R, "
            "hull against hull, h3d.arm_verdict",
    "then": {"fails": "hands closed for keypoint conditioning at every keypoint rate; move to tennis players",
             "passes": f"F billed at H2's {FLOW_KEYPOINTS} by the same verdict: passes -> the MANO warp at that point is next; "
                       "otherwise hands close with the keypoint rate named",
             "unclear": "to the user"},
    "beside": ["F vs B", "F billed at about 1.01 and 1.93 kbps vs A", "K3's C vs F (appearance beyond motion)",
               "the flow's own warp error inside R(t0)"],
}


# ----------------------------------------------------------------- code


def calibrated_bits(real: list[dict[str, Any]], anchor: list[dict[str, Any]], arm: list[dict[str, Any]],
                    container_bytes: int) -> dict[str, float]:
    """``arm``'s estimated bits per NAL scaled by the unmodified codec's real bits over its estimate for the same NAL
    (``real``: the worker's NAL records, ``anchor``: arm A's estimate); the container header by the arm's estimated share
    on R.
    For arm A itself this is `h3c.nal_region_bits`."""
    if not len(real) == len(anchor) == len(arm):
        raise RuntimeError(f"{len(real)} NALs coded, {len(anchor)} and {len(arm)} estimated")
    region = total = actual = 0.0
    for nal, a, x in zip(real, anchor, arm):
        if a["frames"] != x["frames"]:
            raise RuntimeError(f"NAL frames differ: {a['frames']} and {x['frames']}")
        bits = 8.0 * (nal["nal_bytes"] + nal["sps_bytes"])
        factor = bits / a["estimated_bits"] if a["estimated_bits"] > 0 else 0.0
        region += x["estimated_region_bits"] * factor
        total += x["estimated_bits"] * factor
        actual += bits
    header = 8.0 * container_bytes - actual
    est_region, est = float(sum(x["estimated_region_bits"] for x in arm)), float(sum(x["estimated_bits"] for x in arm))
    share = est_region / est if est > 0 else 0.0  # h3c.nal_region_bits' overall share
    return {"region_bits": region + header * share, "bits": total + header, "estimated_region_bits": est_region, "estimated_bits": est,
            "estimate_over_actual": float(sum(a["estimated_bits"] for a in anchor)) / actual if actual else math.nan}


def encode_real(args: argparse.Namespace, source: Path, size: tuple[int, int], frames: int, qp: int, work: Path) -> dict[str, Any]:
    """The unmodified codec's real stream (B2's worker, encode only): its NAL bytes calibrate the estimate."""
    from src.codecs.dcvc_uf_worker import dcvc_command

    work.mkdir(parents=True, exist_ok=True)
    width, height = size
    plan = {"structure": "htl", "qp": qp, "frame_count": frames, "height": height, "width": width, "src_type": "yuv420",
            "frames_file": str(source), "container": str(work / "stream.psdc"), "out_file": str(work / "decoded.yuv"),
            "image_ckpt": args.image_ckpt, "image_sha256": args.image_sha256, "video_ckpt": args.video_ckpt,
            "video_sha256": args.video_sha256, "profile": False}
    write_json(work / "plan.json", plan)
    command, env, cwd = dcvc_command(Path(sys.prefix), "encode", work / "plan.json", work / "encode.json")
    done = subprocess.run(command, env={**os.environ, **env}, cwd=cwd, capture_output=True, text=True, timeout=3600)
    if done.returncode:
        raise RuntimeError(f"DCVC-UF encode failed ({done.returncode}):\n{done.stderr[-4000:]}")
    report = json.loads((work / "encode.json").read_text())
    data = (work / "stream.psdc").read_bytes()
    return {"nals": report["nals"], "bytes": len(data), "stream_sha256": file_sha256(work / "stream.psdc"),
            "encode_seconds": report["encode_seconds"], "device": report["environment"]["device_name"]}


def process_item(args: argparse.Namespace, item: dict[str, Any], decoded: dict[str, Any], nets: dict[str, Any], device: str,
                 publish: Path, scratch: Path, label: str, first: bool) -> None:
    import cv2
    import torch

    began = time.time()
    frames = int(args.frames)
    width, height, full_range = decoded["width"], decoded["height"], decoded["full_range"]
    seconds = frames / float(Fraction(item["fps"]).limit_denominator(1001))
    work = scratch / "work" / item["id"]
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    planes, regions, mask_info = load_window(args, item, frames, decoded)
    truth = rgb_memmap(Path(decoded["path"]), work / "truth.npy", frames, width, height, full_range, device)
    score = Scorer(nets, truth, regions, device)
    look = score.hand[len(score.hand) // 2] if score.hand else 0
    ys, xs = np.nonzero(regions[look]) if score.hand else (np.array([0]), np.array([0]))
    look_box = (slice(max(0, ys.min() - DISTS_PAD), min(height, ys.max() + 1 + DISTS_PAD)),
                slice(max(0, xs.min() - DISTS_PAD), min(width, xs.max() + 1 + DISTS_PAD)))
    frames_dir = publish / "frames" / item["id"]
    frames_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(frames_dir / f"source-f{look:03d}.png"), cv2.cvtColor(np.asarray(truth[look])[look_box], cv2.COLOR_RGB2BGR))
    points: list[dict[str, Any]] = []
    scores: list[dict[str, np.ndarray]] = []
    for scale in args.scales:
        sw, sh = scaled_size(width, height, scale)
        source = Path(decoded["path"])
        if scale > 1:
            source = work / f"source-{scale}.yuv"
            with source.open("wb") as handle:
                for t in range(frames):
                    handle.write(downscale_planes(np.asarray(planes[t]), width, height, sw, sh).tobytes())
        region_file = work / f"region-{scale}.npy"
        np.save(region_file, np.stack([downscale_mask(r, sw, sh) for r in regions]))
        for qp in args.qps:
            point_work = work / f"{scale}-{qp}"
            torch.cuda.empty_cache()  # the worker and the tree script share the GPU with the scorer's cache
            real = encode_real(args, source, (sw, sh), frames, qp, point_work / "real")
            coded = run_tree_script(Path(__file__).resolve().parent / "k3_ref.py",
                                    {"frames_file": str(source), "region_file": str(region_file), "width": sw, "height": sh,
                                     "frame_count": frames, "qp": qp, "arms": list(args.arms), "out_dir": str(point_work / "arms"),
                                     "image_ckpt": args.image_ckpt, "image_sha256": args.image_sha256,
                                     "video_ckpt": args.video_ckpt, "video_sha256": args.video_sha256}, point_work, "k3-ref")
            by_arm = {a["arm"]: a for a in coded["arms"]}
            for arm in args.arms:
                rec = by_arm[arm]
                bits = calibrated_bits(real["nals"], by_arm["A"]["nals"], rec["nals"], real["bytes"])
                rgb = rgb_memmap(Path(rec["decoded"]), point_work / "rgb.npy", frames, sw, sh, full_range, device)
                name = f"{arm}@{qp}/{scale}"
                scores.append(score(rgb, upscale=scale > 1))
                frame = np.asarray(rgb[look])
                if scale > 1:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_CUBIC)
                out = np.asarray(truth[look]).copy()
                out[regions[look]] = frame[regions[look]]
                cv2.imwrite(str(frames_dir / f"{arm}-{qp}_{scale}-f{look:03d}.png"), cv2.cvtColor(out[look_box], cv2.COLOR_RGB2BGR))
                points.append({"name": name, "arm": arm, "qp": qp, "scale": scale, "kbps_r": bits["region_bits"] / seconds / 1000,
                               "kbps": bits["bits"] / seconds / 1000,
                               "kbps_r_estimated": bits["estimated_region_bits"] / seconds / 1000,
                               "estimate_over_actual": bits["estimate_over_actual"], "nals": len(rec["nals"]),
                               "decoded_sha256": file_sha256(Path(rec["decoded"])), "pasted_share": rec["pasted_share"], "flow_error": rec.get("flow_error"),
                               "real_stream_sha256": real["stream_sha256"], "real_bytes": real["bytes"]})
                os.remove(point_work / "rgb.npy")
            points[-1]["ref_seconds"] = coded["seconds"]
            points[-1]["ref_peak_allocated_mib"] = coded["peak_allocated_mib"]
            points[-1]["ref_gpu"] = coded["gpu"]
            shutil.rmtree(point_work)
        if scale > 1:
            source.unlink()
        region_file.unlink()
    checks: dict[str, Any] = {}
    if first:
        from src.codecs.quality import load_lpips

        t = score.hand[0]
        out = np.asarray(truth[t]).copy()
        out[regions[t]] = 0
        cpu = load_lpips(Path(args.lpips_backbone), "cpu")

        def lp(net: Any, dev: str) -> float:
            a = torch.from_numpy(out).to(dev).permute(2, 0, 1)[None].float() / 127.5 - 1.0
            b = torch.from_numpy(np.asarray(truth[t])).to(dev).permute(2, 0, 1)[None].float() / 127.5 - 1.0
            with torch.inference_mode():
                return float((net(a, b)[0, 0] * torch.from_numpy(regions[t]).to(dev)).sum()) / int(regions[t].sum())

        checks["lpips_cpu_vs_device"] = abs(lp(cpu, "cpu") - lp(nets["lpips"], device))
        checks["dists_identity"] = dists_crops(nets["dists"], [np.asarray(truth[t])], np.asarray(truth[t]), regions[t], device)[0]
    os.remove(decoded["path"])
    shutil.rmtree(work)
    arrays = {"pixels_r": np.array([int(r.sum()) for r in regions], np.int64),
              "lpips": np.stack([s["lpips"] for s in scores]), "sse": np.stack([s["sse"] for s in scores]),
              "dists": np.stack([s["dists"] for s in scores])}
    meta = {"unit": f"{item['id']}@{label}", "item": item["id"], "label": label, "video": item["video"], "fps": float(item["fps"]),
            "frames": frames, "seconds_coded": seconds, "hand_frames": len(score.hand), "points": points, "look_frame": look,
            "decode": {k: v for k, v in decoded.items() if k != "path"}, "mask_set": mask_info, "checks": checks,
            "seconds": round(time.time() - began, 2)}
    save_unit(publish, "code", f"{item['id']}@{label}", arrays, meta)


def command_code(args: argparse.Namespace) -> int:
    import torch
    import torchmetrics
    from torchmetrics.functional.image import dists as dists_module

    from experiments.audit.env_smoke import device_record
    from experiments.visor.b2 import select_items
    from src.codecs.quality import load_lpips, lpips_record

    eval_set = json.loads(Path(args.eval_set).read_text())
    items = select_items(eval_set, args.items)
    args.arms = tuple(args.arms.split(","))
    if "A" not in args.arms or set(args.arms) - set(ARMS):
        raise SystemExit(f"arms must include A (the calibration) and come from {ARMS}")
    args.image_sha256 = file_sha256(Path(args.image_ckpt))
    args.video_sha256 = file_sha256(Path(args.video_ckpt))
    args.qps = [int(q) for q in args.qps.split(",")]
    args.scales = [int(s) for s in args.scales.split(",")]
    label = "k3-s" + "-".join(map(str, args.scales))
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    restored = restore_units(publish, "code")
    todo = [it for it in items if f"{it['id']}@{label}" not in restored]
    windows = decode_windows(args, todo, int(args.frames), scratch)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("k3 code needs a GPU (DCVC-UF)")
    nets = {"lpips": load_lpips(Path(args.lpips_backbone), device), "dists": load_dists(Path(args.dists_backbone), device)}
    on_cuda = {k: next(v.parameters()).is_cuda for k, v in nets.items()}
    done = len(restored)
    began = time.time()
    for number, (item, decoded) in enumerate(windows):
        process_item(args, item, decoded, nets, device, publish, scratch, label, first=number == 0)
        done += 1
        progress(done)
    write_json(stage_dir() / "k3.json", {
        "command": "code", "label": label, "device": device_record(torch.cuda.get_device_name(0)), "on_cuda": on_cuda,
        "seconds": round(time.time() - began, 2), "lpips": lpips_record(Path(args.lpips_backbone)),
        "dists": {"torchmetrics": torchmetrics.__version__,
                  "weights_sha256": file_sha256(Path(dists_module.__file__).parent / "dists_models" / "weights.pt"),
                  "backbone": args.dists_backbone, "backbone_sha256": file_sha256(Path(args.dists_backbone))},
        "dcvc": {"tree": str(Path(sys.prefix) / "opt" / "DCVC"), "image_sha256": args.image_sha256, "video_sha256": args.video_sha256},
        "settings": {"frames": int(args.frames), "items": args.items, "qps": args.qps, "scales": args.scales,
                     "arms": list(args.arms), "decision": DECISION, "dists_pad": DISTS_PAD},
        "items": [it["id"] for it in items], "restored_units": sorted(restored)})
    return 0


# ----------------------------------------------------------------- validation


def validate_code(stage: Path, result: dict[str, Any]) -> dict[str, bool]:
    units = load_units([str(stage / "published.tar")], "code")
    points = [p for m, _ in units for p in m["points"]]
    first = [m for m, _ in units if "lpips_cpu_vs_device" in m["checks"]]
    hand = [(m, a, a["pixels_r"] > 0) for m, a in units]

    def at(m: dict[str, Any], arm: str) -> dict[tuple[int, int], dict[str, Any]]:
        return {(p["qp"], p["scale"]): p for p in m["points"] if p["arm"] == arm}

    arms = result["settings"]["arms"]
    return {
        "every_item_present": bool(units) and len(units) == len(result["items"]) and all(m["npz_hash_ok"] for m, _ in units),
        "every_net_on_cuda": all(result["on_cuda"].values()),
        "sparse_jpegs_match_decoded_frames": all(g["holds"] for m, _ in units for g in m["decode"]["jpeg_gate"]),
        "mask_set_matches_record": all(m["mask_set"]["masks_rle_sha256"] == m["mask_set"]["record_masks_rle_sha256"] for m, _ in units),
        "every_arm_at_every_point": all(len(m["points"]) == len(arms) * len(result["settings"]["qps"]) * len(result["settings"]["scales"])
                                        for m, _ in units),
        "every_point_scored": all(a["lpips"].shape[0] == len(m["points"]) for m, a in units),
        "scores_finite_on_hand_frames": all(np.isfinite(a["lpips"][:, h]).all() and np.isfinite(a["dists"][:, h]).all() for _, a, h in hand),
        "lpips_cpu_matches_device_1e-3": bool(first) and all(m["checks"]["lpips_cpu_vs_device"] < 1e-3 for m in first),
        "dists_of_identical_is_zero": all(abs(m["checks"]["dists_identity"]) < 1e-4 for m in first),
        # The PyTorch path's estimate of the unmodified codec should be near its real bytes (H3c: 0.78 to 1.27).
        "estimate_near_actual_bits": all(0.5 < p["estimate_over_actual"] < 2.0 for p in points),
        "region_rate_inside_stream_rate": all(0.0 < p["kbps_r"] < p["kbps"] for p in points),
        # Calibrated by its own real bytes, arm A's whole-stream rate is the real stream's.
        "a_rate_is_the_real_stream": all(math.isclose(p["kbps"] * m["seconds_coded"] * 1000, 8 * p["real_bytes"], rel_tol=1e-9)
                                         for m, _ in units for p in m["points"] if p["arm"] == "A"),
        # Every substituted reference changes the decode; C and D paste some hand at every point with hands.
        "arms_differ_from_a": all(p["decoded_sha256"] != at(m, "A")[(p["qp"], p["scale"])]["decoded_sha256"]
                                  for m, _ in units for p in m["points"] if p["arm"] != "A"),
        "hand_arms_paste_hand_pixels": all(p["pasted_share"] and p["pasted_share"] > 0 for m, _ in units for p in m["points"]
                                           if p["arm"] in PASTED and m["hand_frames"] > 0),
        # The flow warp brings the source's previous frame no further from its next one inside R(t0) than no warp does,
        # up to a quarter luma level (a still hand leaves only resampling noise; a broken flow costs many levels).
        "flow_warp_does_not_add_error": all(p["flow_error"]["warped"] <= p["flow_error"]["unwarped"] + FLOW_TOLERANCE
                                            for m, _ in units for p in m["points"] if p["arm"] == "F" and p.get("flow_error")),
        "rate_rises_with_qp": all(a["kbps"] < b["kbps"] for m, _ in units for a in m["points"] for b in m["points"]
                                  if a["arm"] == b["arm"] and a["scale"] == b["scale"] and a["qp"] < b["qp"]),
    }


def command_validate(args: argparse.Namespace) -> int:
    stage = stage_dir()
    result = json.loads((stage / "k3.json").read_text())
    checks = {k: bool(v) for k, v in validate_code(stage, result).items()}  # numpy booleans do not serialize
    report = {"passed": all(checks.values()), "checks": checks}
    target = os.environ.get("PS_VALIDATION_PATH")
    if target:
        write_json(Path(target), report)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


# ----------------------------------------------------------------- report (CPU)


def item_rows(units: list[tuple[dict[str, Any], dict[str, np.ndarray]]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """item -> arm -> scored points (rate on R, quality), over every unit of the item (the scale groups)."""
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for meta, a in units:
        hand = a["pixels_r"] > 0
        for j, p in enumerate(meta["points"]):
            row = {"name": p["name"], "qp": p["qp"], "scale": p["scale"], "kbps": float(p["kbps_r"]),
                   "kbps_estimated": float(p["kbps_r_estimated"]), **point_summary(a, j, hand),
                   "lpips_frames": a["lpips"][j][hand], "real_stream_sha256": p["real_stream_sha256"],
                   "flow_error": p.get("flow_error")}
            out.setdefault(meta["item"], {}).setdefault(p["arm"], []).append(row)
    return out


def billed(rows: list[dict[str, Any]], kbps: float) -> list[dict[str, Any]]:
    return [{**r, "kbps": r["kbps"] + kbps} for r in rows]


def hull(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    """The rows on ``rows``' upper concave hull in (kbps, quality) on ``metric`` (`g5.envelope`): the test's achievable
    envelope, as the anchor's is. A Pareto front against the anchor's hull would charge the test for the hull's
    interpolation (in a dry run, arm A against its own hull)."""
    from experiments.background.g5 import envelope
    from experiments.visor.h3d import quality

    vertices = set(envelope(quality(rows, metric)))
    return [r for r, q in zip(rows, quality(rows, metric)) if q in vertices]


def versus(tests: dict[str, list[dict[str, Any]]], anchors: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Per item, `h3d.compare` of the test's hull against the anchor's on every metric, and `h3d.arm_verdict` on LPIPS."""
    from experiments.visor.h3d import arm_verdict, compare

    per_item = {item: {m: compare(hull(test, m), anchors[item], m) for m in METRICS}
                for item, test in sorted(tests.items()) if item in anchors}
    return {"items": per_item, "verdict": arm_verdict(per_item)}


def decide(gate: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The gate: the better of C and D (by mean BD on LPIPS_R, billed, against A)."""
    verdicts = {arm: g["verdict"] for arm, g in gate.items()}
    if any(v["verdict"] == "incomplete" for v in verdicts.values()) or set(verdicts) != set(BILLED):
        return {"k3": "incomplete", "arms": verdicts}
    best = min(verdicts, key=lambda arm: verdicts[arm].get("mean_bd", math.inf))
    return {"k3": verdicts[best]["verdict"], "better_arm": best, "arms": verdicts, "then": DECISION["then"][verdicts[best]["verdict"]]}


def decide_flow(unbilled: dict[str, Any], billed_low: dict[str, Any]) -> dict[str, Any]:
    """K3b: F unbilled against A; if it passes, F billed at H2's ~1 kbps point."""
    gate = unbilled["verdict"]
    if gate == "incomplete":
        return {"k3b": "incomplete"}
    if gate != "passes":
        return {"k3b": gate, "then": FLOW_DECISION["then"][gate]}
    low = billed_low["verdict"]
    return {"k3b": "passes", "billed_at_low_rate": low,
            "then": "build the MANO warp at that keypoint rate (its own rule)" if low == "passes"
            else "hands close; the keypoint rate is the obstacle"}


def flow_report(items: dict[str, dict[str, list[dict[str, Any]]]], earlier: dict[str, dict[str, list[dict[str, Any]]]],
                rates: dict[str, float]) -> dict[str, Any]:
    """K3b's section: F against A unbilled (the gate), billed at H2's low and chosen points, against B, and K3's C against F."""
    def arm(name: str, kbps: float = 0.0, source: dict[str, dict[str, list[dict[str, Any]]]] = items) -> dict[str, list[dict[str, Any]]]:
        return {item: billed(by_arm[name], kbps) for item, by_arm in source.items() if name in by_arm}

    a = arm("A")
    out: dict[str, Any] = {"decision_rule": FLOW_DECISION, "keypoint_kbps": rates,
                           "F_unbilled_vs_A": versus(arm("F"), a), "F_vs_B": versus(arm("F"), arm("B")),
                           **{f"F_billed_{name}_vs_A": versus(arm("F", kbps), a) for name, kbps in rates.items()}}
    if earlier:
        out["K3_C_unbilled_vs_F"] = versus(arm("C", source=earlier), arm("F"))
    errors = [p["flow_error"] for by_arm in items.values() for p in by_arm.get("F", []) if p.get("flow_error")]
    if errors:
        out["flow_error"] = {k: float(np.mean([e[k] for e in errors])) for k in ("warped", "unwarped")}
    out["decision"] = decide_flow(out["F_unbilled_vs_A"]["verdict"], out["F_billed_low_vs_A"]["verdict"])
    return out


def command_report(args: argparse.Namespace) -> int:
    items = item_rows(load_units(args.code, "code"))
    if any("F" in by_arm for by_arm in items.values()):
        code = json.loads(Path(args.pose_code).read_text())
        low = [r for r in code["rows"]["visor"]["wilor"] if r["key"] == FLOW_KEYPOINTS]
        rates = {"low": float(low[0]["kbps"]), "chosen": float(code["choices"]["wilor"][POSE_BUDGET]["visor_kbps"])}
        earlier = item_rows(load_units(args.k3, "code")) if args.k3 else {}
        section = flow_report(items, earlier, rates)
        for name, entry in section.items():
            if isinstance(entry, dict) and "verdict" in entry:
                numbers = [e["lpips"]["bd"] for e in entry["items"].values() if e["lpips"]["bd"] is not None]
                if numbers:
                    entry["verdict"]["mean_bd_ci"] = bootstrap_mean(np.array(numbers), np.random.default_rng(seed_of(SEED)), RESAMPLES)
        out = Path(args.out or stage_dir())
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "k3b-report.json", {**section, "runs": [{"path": p, "sha256": file_sha256(Path(p))} for p in [*args.code, *args.k3]]})
        print(json.dumps({k: (v["verdict"] if isinstance(v, dict) and "verdict" in v else v) for k, v in section.items()
                          if k != "decision_rule"}, indent=1, default=str))
        return 0
    h1 = {row["id"]: row for row in json.loads(Path(args.h1_report).read_text())["per_item"]}
    keypoint_kbps = float(json.loads(Path(args.pose_code).read_text())["choices"]["wilor"][POSE_BUDGET]["visor_kbps"])
    bar: dict[str, list[dict[str, Any]]] = {}
    real: dict[tuple[str, str], dict[str, Any]] = {}
    for meta, a in load_units(args.h3c, "code"):
        hand = a["pixels_r"] > 0
        rates = hand_kbps(h1[meta["item"]])
        for j, p in enumerate(meta["points"]):
            if p["arm"] not in ("svtav1", "reference"):
                continue
            kbps = p["kbps_r"] if "kbps_r" in p else rates[f"{p['crf']:g}"]
            row = {"name": p["name"], "kbps": float(kbps), **point_summary(a, j, hand)}
            bar.setdefault(meta["item"], []).append(row)
            if p["arm"] == "reference":
                real[(meta["item"], f"A@{p['qp']}/{p['scale']}")] = {**row, "stream_sha256": p["stream_sha256"],
                                                                     "lpips_frames": a["lpips"][j][hand]}
    # The PyTorch path against H3c's real codec, point by point (arm A): rate within 5%, LPIPS within 0.005 (H3d's tolerance).
    cross: dict[str, dict[str, Any]] = {}
    for item, by_arm in items.items():
        for row in by_arm.get("A", []):
            ref = real.get((item, row["name"]))
            if ref is None:
                continue
            entry = {"rate_ratio": row["kbps"] / ref["kbps"], "lpips_mean_gap": row["lpips"] - ref["lpips"],
                     "lpips_max_frame_gap": float(np.max(np.abs(row["lpips_frames"] - ref["lpips_frames"]))),
                     "psnr_gap": row["psnr"] - ref["psnr"]}
            entry["stream_matches"] = row["real_stream_sha256"] == ref["stream_sha256"]
            # Reported, not required: DCVC-UF's encoder is bound to the GPU (docs/fleet.md), as H3d found for its base points.
            entry["holds"] = abs(entry["rate_ratio"] - 1) <= 0.05 and abs(entry["lpips_mean_gap"]) <= 0.005
            cross.setdefault(item, {})[row["name"]] = entry

    def arm(name: str, kbps: float = 0.0) -> dict[str, list[dict[str, Any]]]:
        return {item: billed(by_arm[name], kbps) for item, by_arm in items.items() if name in by_arm}

    anchor_a, anchor_b = arm("A"), arm("B")
    gate = {name: versus(arm(name, keypoint_kbps), anchor_a) for name in BILLED if any(name in b for b in items.values())}
    beside: dict[str, Any] = {
        "B_vs_A": versus(anchor_b, anchor_a),
        **{f"{n}_unbilled_vs_A": versus(arm(n), anchor_a) for n in BILLED},
        **{f"{n}_unbilled_vs_B": versus(arm(n), anchor_b) for n in BILLED},
        "E_vs_B": versus(arm("E"), anchor_b),
        "E_vs_A": versus(arm("E"), anchor_a),
        **{f"{n}_billed_vs_pixel_envelope": versus(arm(n, keypoint_kbps), bar) for n in BILLED},
        "A_vs_pixel_envelope": versus(anchor_a, bar),
        "keypoint_rate": {f"{n}@{k:g}": versus(arm(n, k), anchor_a)["verdict"] for n in BILLED for k in KEYPOINT_GRID},
    }
    # Per point (same QP and scale), each arm's rate on R and LPIPS_R against A's and B's: where the bits go.
    ratios: dict[str, dict[str, Any]] = {}
    for item, by_arm in items.items():
        base = {r["name"].split("@")[1]: r for r in by_arm.get("A", [])}
        ctrl = {r["name"].split("@")[1]: r for r in by_arm.get("B", [])}
        for name, rows in by_arm.items():
            for r in rows:
                key = r["name"].split("@")[1]
                ratios.setdefault(item, {})[r["name"]] = {
                    "rate_vs_A": r["kbps"] / base[key]["kbps"] if key in base else None,
                    "lpips_minus_A": r["lpips"] - base[key]["lpips"] if key in base else None,
                    "rate_vs_B": r["kbps"] / ctrl[key]["kbps"] if key in ctrl else None,
                    "lpips_minus_B": r["lpips"] - ctrl[key]["lpips"] if key in ctrl else None}
    for entry in [*gate.values(), *(v for k, v in beside.items() if k != "keypoint_rate")]:
        numbers = [e["lpips"]["bd"] for e in entry["items"].values() if e["lpips"]["bd"] is not None]
        if numbers:
            entry["verdict"]["mean_bd_ci"] = bootstrap_mean(np.array(numbers), np.random.default_rng(seed_of(SEED)), RESAMPLES)
    report = {"decision_rule": DECISION, "keypoint_kbps": keypoint_kbps, "decision": decide(gate), "gate": gate, "beside": beside,
              "per_point": ratios, "cross_check_a_vs_h3c": cross,
              "fronts": {item: {name: [{k: v for k, v in r.items() if k != "lpips_frames"} for r in rows] for name, rows in by_arm.items()}
                         for item, by_arm in items.items()},
              "runs": [{"path": p, "sha256": file_sha256(Path(p))} for p in [*args.code, *args.h3c]]}
    out = Path(args.out or stage_dir())
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "k3-report.json", report)
    print(json.dumps({"decision": report["decision"],
                      "beside": {k: (v["verdict"] if "verdict" in v else v) for k, v in beside.items()},
                      "cross_check_holds": {i: all(e["holds"] for e in c.values()) for i, c in cross.items()}}, indent=1, default=str))
    return 0


# ----------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments.visor.k3")
    commands = parser.add_subparsers(dest="command", required=True)
    code = commands.add_parser("code")
    code.add_argument("--eval-set", required=True)
    code.add_argument("--archive", required=True)
    code.add_argument("--masks", required=True)
    code.add_argument("--mask-record", required=True)
    code.add_argument("--video", action="append", default=[])
    code.add_argument("--items", required=True)
    code.add_argument("--frames", default="240")
    code.add_argument("--image-ckpt", required=True)
    code.add_argument("--video-ckpt", required=True)
    code.add_argument("--lpips-backbone", required=True)
    code.add_argument("--dists-backbone", required=True)
    code.add_argument("--qps", default=",".join(map(str, QPS)))
    code.add_argument("--scales", default=",".join(map(str, SCALES)))
    code.add_argument("--arms", default=",".join(ARMS))
    commands.add_parser("validate")
    report = commands.add_parser("report")
    report.add_argument("--code", action="append", required=True, help="K3 coding jobs' published.tar")
    report.add_argument("--h3c", action="append", required=True, help="H3c's anchor+reference published.tar (the pixel envelope)")
    report.add_argument("--h1-report", required=True)
    report.add_argument("--pose-code", required=True, help="H2's code.json (the keypoints' rate)")
    report.add_argument("--k3", action="append", default=[], help="K3's published.tar, for K3b's C against F")
    report.add_argument("--out", default="")
    args = parser.parse_args(argv)
    return {"code": command_code, "validate": command_validate, "report": command_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
