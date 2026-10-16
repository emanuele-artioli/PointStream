"""PLAN step H4c: handled objects coded as pixels with H3d's recipe (docs/experiments.md, 2026-10-16, "H4c").

    python -m experiments.visor.h4c run --eval-set JSON --archive DIR --masks DIR --mask-record JSON --video V.MP4 \\
        --items ID --frames N --points NAME,... --steps N --image-ckpt PTH --video-ckpt PTH --lpips-backbone PTH \\
        --dists-backbone PTH --inspect BIN --svt-streams DIR --svt-result JSON ... --generic TAR ...
    python -m experiments.visor.h4c validate
    python -m experiments.visor.h4c report --run TAR ... --h1-report JSON --h3d-report JSON --out DIR

``run`` scores each named point of one item's window on two regions: O, H1's handled-object part (`object_regions`),
and R, H3c's hand region. Points are B2's SVT-AV1 streams (``svt@41`` …), H3b2's low points (``svt@63/4`` …) and
DCVC-UF HT-L off the shelf (``dcvc@QP/SCALE``), each coded and its rate split onto O (and R) as H3c does. At DCVC-UF's
¼ and ½ points it also applies G, H3d's generic network for that point (arm 1), and fits the oracle: G5d's enhancer on
the window's own O, its bits free. Each point is published as ``point/<item>@<name>`` and checkpointed. ``report``
builds the bar on O (SVT-AV1 and DCVC-UF off the shelf; B2's points at H1's ``handled_object`` rate) and applies
`DECISION` to arm 1 and the oracle.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json
from experiments.visor.h2 import load_units, restore_units, save_unit
from experiments.visor.h3 import FILL, HANDS, POINTS, psnr_of, rgb_memmap
from experiments.visor.h3b import LOW_POINTS, low_name, scaled_size
from experiments.visor.h3c import QPS, SCALES, decode_windows, downscale_mask, downscale_planes
from experiments.visor import h3d

SEED = "pointstream-h4c"
REGIONS = ("o", "r")  # handled objects (decides), hands (H3d's G reproduced; G_HO's cost on hands)
#: Every point of the bar, by name; DCVC-UF's ¼ and ½ points also carry G and the oracle.
BAR = [f"svt@{p:g}" for p in POINTS] + [low_name(c, s) for c, s in LOW_POINTS] + \
      [f"dcvc@{q}/{s}" for s in SCALES for q in QPS]
FITTED = {f"dcvc@{q:g}/{s}" for q, s in h3d.BASES["dcvc"]}

#: The decision rule, fixed before any fleet run (docs/experiments.md, H4c entry).
DECISION: dict[str, Any] = {
    "quality": "lpips_o: the output inside O (H1's handled-object part of the fill set), the source outside; full-frame "
               "LPIPS averaged over O, mean over frames with O",
    "bar": "per item, the upper concave hull (g5.envelope) on (kbps on O, -LPIPS_O) of SVT-AV1 (B2's four CRFs at H1's "
           "handled_object rate, H3b2's seven low points) and DCVC-UF HT-L off the shelf (nine points)",
    "stands": "h3d.DECISION against the region's bar: mean BD-rate on LPIPS <= -10% with at least 3 of 4 items below 0; "
              "fails at mean >= 0 or at most 1 item below 0; otherwise unclear",
    "stage1": "oracle fails on O: arms 2 and 3 are not built; objects by DCVC-UF + G if arm 1 stands on O, else DCVC-UF off "
              "the shelf (generative completion reported). Oracle passes or unclear: stage 2 (G_O, G_HO)",
    "stage2": "object network: first standing arm on O in the order G, G_HO, G_O unless a later standing arm beats it by 10% "
              "(mean BD-rate against it on LPIPS_O <= -10%); hands keep G unless the object network is G_HO, it stands on "
              "R and its mean BD-rate against G on LPIPS_R is below +10%; no arm stands on O: DCVC-UF off the shelf",
    "beats_by": -0.10,
    "keeps_hands_below": 0.10,
}


# ----------------------------------------------------------------- regions


def object_regions(clip: Any, frames: int, handled: set[str]) -> list[np.ndarray]:
    """H1's ``handled_object`` part per frame (`experiments.visor.h1`, its labelling of the fill set): each object
    label's union of instances, painted in instance order onto pixels no earlier object took, kept where the label is
    handled; then the hand masks painted over, so O never overlaps R."""
    from src.segmentation import visor

    out = []
    for t in range(frames):
        objects: dict[str, np.ndarray] = {}
        for inst in clip.frames[t]:
            if inst.class_name in visor.HANDS:
                continue
            key = visor.object_key(inst.class_name, inst.label)
            objects[key] = objects.get(key, np.zeros(visor.FRAME_SIZE, bool)) | inst.mask()
        taken = np.zeros(visor.FRAME_SIZE, bool)
        region = np.zeros(visor.FRAME_SIZE, bool)
        for key, mask in objects.items():
            free = mask & ~taken
            if key in handled:
                region |= free
            taken |= free
        region &= ~(clip.class_mask(t, HANDS[0]) | clip.class_mask(t, HANDS[1]))
        out.append(region)
    return out


def load_regions(args: argparse.Namespace, item: dict[str, Any], frames: int,
                 decoded: dict[str, Any]) -> tuple[np.ndarray, dict[str, list[np.ndarray]], dict[str, Any], dict[str, Any]]:
    """The window's 4:2:0 planes (memmap), O and R per frame, the mask set's record and H1's contacts."""
    from experiments.visor.b1b import load_inputs
    from experiments.visor.b2 import load_mask_sets
    from experiments.visor.h1 import contact_objects

    clips, info = load_mask_sets(item, frames, {FILL: args.masks}, {FILL: args.mask_record}, Path(args.archive))
    clip = clips[FILL]
    _, sparse, mapping = load_inputs(item, Path(args.archive))
    contact = contact_objects(item, sparse, mapping)
    regions = {"o": object_regions(clip, frames, set(contact["handled"])),
               "r": [clip.class_mask(t, HANDS[0]) | clip.class_mask(t, HANDS[1]) for t in range(frames)]}
    width, height = decoded["width"], decoded["height"]
    planes = np.memmap(decoded["path"], np.uint8, "r", shape=(frames, height * 3 // 2, width))
    return planes, regions, info[FILL], contact


def parse_point(name: str) -> tuple[str, float, int]:
    """(``b2`` | ``low`` | ``dcvc``, CRF or QP, scale) of a point name in `BAR`."""
    if name not in BAR:
        raise SystemExit(f"{name} is not one of H4c's points")
    codec, rest = name.split("@")
    value, _, scale = rest.partition("/")
    if codec == "dcvc":
        return "dcvc", float(value), int(scale)
    if not scale and float(value) in POINTS:
        return "b2", float(value), 1
    return "low", float(value), int(scale or 1)


# ----------------------------------------------------------------- coding one point


def code_point(args: argparse.Namespace, name: str, item: dict[str, Any], decoded: dict[str, Any], planes: Any,
               regions: dict[str, list[np.ndarray]], frames: int, fps: Fraction, work: Path,
               device: str) -> tuple[Any, bool, dict[str, Any]]:
    """The point's decode (RGB at its coded size; whether it needs upscaling) and its record: kbps on O (and on R for
    DCVC-UF) where the point is coded here; B2's streams take theirs from H1's report."""
    from experiments.visor.h1 import stream_paths
    from experiments.visor.h3b import encode_low
    from experiments.visor.h3c import code_dcvc
    from src.codecs import svtav1

    width, height, full_range = decoded["width"], decoded["height"], decoded["full_range"]
    seconds = frames / float(fps)
    kind, value, scale = parse_point(name)
    if kind == "b2":
        streams, info = stream_paths(args, item["id"])
        key = f"{value:g}"
        yuv = work / f"crf{key}.yuv"
        allowance = max(1, int(os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 1))
        svtav1.run(svtav1.decode_command(args.decoder, streams[key], yuv, threads=allowance), 1800)
        rgb = rgb_memmap(yuv, work / "rgb.npy", frames, width, height, full_range, device)
        yuv.unlink()
        return rgb, False, {"codec": "svtav1", "crf": value, "scale": 1, "stream": info[key], "kbps": info[key]["kbps"]}
    if kind == "low":
        rgb, record = encode_low(args, Path(decoded["path"]), frames, (width, height, full_range), value, scale, regions["o"],
                                 fps, work, device)
        return rgb, scale > 1, {"codec": "svtav1", "crf": value, "scale": scale, "kbps_o": record["hand_bits"] / seconds / 1000,
                                "kbps": 8 * record["payload_bytes"] / seconds / 1000, "stream": record}
    sw, sh = scaled_size(width, height, scale)
    source = Path(decoded["path"])
    if scale > 1:
        source = work / f"source-{scale}.yuv"
        with source.open("wb") as handle:
            for t in range(frames):
                handle.write(downscale_planes(np.asarray(planes[t]), width, height, sw, sh).tobytes())
    files = {}
    for region in REGIONS:
        files[region] = work / f"region-{region}-{scale}.npy"
        np.save(files[region], np.stack([downscale_mask(m, sw, sh) for m in regions[region]]))
    rec = code_dcvc(args, source, files["o"], (sw, sh), frames, int(value), args.video_ckpt, args.video_sha256,
                    work / f"dcvc-{scale}-{value:g}", extra_regions={"r": files["r"]})
    if scale > 1:
        source.unlink()
    for path in files.values():
        path.unlink()
    if not rec["deterministic"] or not rec["decoder_matches_encoder_intra"]:
        raise RuntimeError(f"DCVC-UF QP {value:g} scale {scale}: decode is not the encoder's (two decodes agree: "
                           f"{rec['deterministic']}, first differing frames {rec['decode_passes_differ_at']}; intra matches: "
                           f"{rec['decoder_matches_encoder_intra']})")
    count = Path(rec["decoded"]).stat().st_size // (sw * sh * 3 // 2)
    if count < frames:
        raise RuntimeError(f"DCVC-UF QP {value:g} scale {scale}: decoded {count} of {frames} frames")
    rgb = rgb_memmap(Path(rec["decoded"]), work / "rgb.npy", frames, sw, sh, full_range, device)
    return rgb, scale > 1, {"codec": "dcvc-uf-htl", "qp": int(value), "scale": scale, "kbps_o": rec["region_bits"] / seconds / 1000,
                            "kbps_r": rec["extra_region_bits"]["r"] / seconds / 1000, "kbps": rec["bits"] / seconds / 1000,
                            "video_sha256": args.video_sha256, **{k: v for k, v in rec.items() if k != "decoded"}}


def process_item(args: argparse.Namespace, item: dict[str, Any], decoded: dict[str, Any], nets: dict[str, Any], device: str,
                 publish: Path, scratch: Path, done: set[str]) -> int:
    import cv2

    from experiments.visor.h3c import Scorer
    from experiments.visor.h3d_component import generic_network

    frames = int(args.frames)
    width, height, full_range = decoded["width"], decoded["height"], decoded["full_range"]
    fps = Fraction(item["fps"]).limit_denominator(1001)
    work = scratch / "work" / item["id"]
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    planes, regions, mask_info, contact = load_regions(args, item, frames, decoded)
    if not any(m.any() for m in regions["o"]):
        raise SystemExit(f"{item['id']}: no handled-object pixels in the window ({contact['handled']})")
    truth = rgb_memmap(Path(decoded["path"]), work / "truth.npy", frames, width, height, full_range, device)
    scorers = {r: Scorer(nets, truth, regions[r], device) for r in REGIONS}
    looks = [scorers["o"].hand[len(scorers["o"].hand) * k // 4] for k in (1, 2, 3)]  # frames for the visual check
    crops = publish / "frames" / item["id"]

    def keep_look(label: str, frames_rgb: Any) -> None:
        crops.mkdir(parents=True, exist_ok=True)
        for t in looks:
            both = regions["o"][t] | regions["r"][t]
            ys, xs = np.nonzero(both)
            box = (slice(max(0, ys.min() - 64), ys.max() + 65), slice(max(0, xs.min() - 64), xs.max() + 65))
            out = np.asarray(truth[t]).copy()
            if frames_rgb is not None:
                out[regions["o"][t]] = frames_rgb[t][regions["o"][t]]
            cv2.imwrite(str(crops / f"{label.replace('/', '_').replace('@', '-')}-f{t:03d}.png"),
                        cv2.cvtColor(out[box], cv2.COLOR_RGB2BGR))

    if not done:
        keep_look("source", None)
    finished = 0
    for name in args.points:
        unit = f"{item['id']}@{name.replace('/', '_')}"
        if unit in done:
            continue
        began = time.time()
        rgb, small, record = code_point(args, name, item, decoded, planes, regions, frames, fps, work, device)
        upscaled = h3d.upscale_all(rgb, frames, width, height) if small else np.array(rgb)  # a copy: its file goes next
        del rgb
        (work / "rgb.npy").unlink(missing_ok=True)
        for leftover in work.glob("low-*"):
            shutil.rmtree(leftover, ignore_errors=True)
        for leftover in work.glob("dcvc-*"):
            shutil.rmtree(leftover, ignore_errors=True)
        rows: list[str] = []
        fits: dict[str, Any] = {}
        checks: dict[str, Any] = {}
        scores: dict[str, list[dict[str, np.ndarray]]] = {r: [] for r in REGIONS}
        flicker: dict[str, list[np.ndarray]] = {r: [] for r in REGIONS}

        def add(row: str, outputs: dict[str, Any]) -> None:
            """Score one row on each region it has an output for (the oracle: O only), keep its crops, drop it."""
            rows.append(row)
            for region in REGIONS:
                out = outputs.get(region)
                if out is None:
                    scores[region].append({"lpips": np.full(frames, np.nan), "dists": np.full(frames, np.nan),
                                           "sse": np.zeros(frames, np.int64)})
                    flicker[region].append(np.full(frames, np.nan))
                    continue
                scores[region].append(scorers[region](out, upscale=False))
                flicker[region].append(h3d.flicker_flow_free(h3d.composites(out, truth, regions[region]), truth, regions[region]))
            keep_look(f"{name}-{row}", outputs["o"])

        add("base", {r: upscaled for r in REGIONS})
        if name in FITTED:
            g_net, g_info = generic_network(args.generic, name, device)
            fits["G"] = {"network_sha256": g_info["sha256"], "fit": g_info["fit"], "data": g_info["data"]}
            add("G", {r: h3d.apply(g_net, upscaled, regions[r], device) for r in REGIONS})
            del g_net
            net, info = h3d.fit(upscaled, truth, regions["o"], int(args.steps), nets["lpips"], device)
            fits["oracle"] = info
            add("oracle", {"o": h3d.apply(net, upscaled, regions["o"], device)})
            del net
            if not done and not any(n in FITTED for n in args.points[:args.points.index(name)]):
                import torch

                from experiments.background.g5d import enhancer

                # The network at step 0 is the identity: on the job's first fitted point it must reproduce the base.
                torch.manual_seed(h3d.TRAIN["seed"])
                identity = enhancer(h3d.TRAIN["blocks"], h3d.TRAIN["channels"]).to(device).eval()
                checks["identity_reproduces_base"] = bool(np.array_equal(h3d.apply(identity, upscaled, regions["o"], device), upscaled))
        arrays: dict[str, np.ndarray] = {}
        for region in REGIONS:
            arrays.update({f"pixels_{region}": np.array([int(m.sum()) for m in regions[region]], np.int64),
                           f"lpips_{region}": np.stack([s["lpips"] for s in scores[region]]),
                           f"sse_{region}": np.stack([s["sse"] for s in scores[region]]),
                           f"dists_{region}": np.stack([s["dists"] for s in scores[region]]),
                           f"flicker_{region}": np.stack(flicker[region])})
        meta = {"unit": unit, "item": item["id"], "name": name, "video": item["video"], "fps": float(item["fps"]),
                "frames": frames, "seconds_coded": frames / float(fps), "rows": rows, "point": record, "fits": fits,
                "checks": checks, "contact": contact, "object_frames": int(sum(m.any() for m in regions["o"])),
                "decode": {k: v for k, v in decoded.items() if k != "path"}, "mask_set": mask_info, "train": h3d.TRAIN,
                "apply_margin": h3d.APPLY_MARGIN, "seconds": round(time.time() - began, 2)}
        save_unit(publish, "point", unit, arrays, meta)
        del upscaled
        finished += 1
        progress(len(done) + finished)
    os.remove(decoded["path"])
    shutil.rmtree(work)
    return finished


def command_run(args: argparse.Namespace) -> int:
    import torch

    from experiments.audit.env_smoke import device_record
    from experiments.visor.b2 import select_items
    from experiments.visor.h3b import _tool, load_dists
    from src.codecs.quality import load_lpips, lpips_record

    eval_set = json.loads(Path(args.eval_set).read_text())
    items = select_items(eval_set, args.items)
    if len(items) != 1:
        raise SystemExit("run takes one item")
    args.points = args.points.split(",")
    for name in args.points:
        parse_point(name)
    if any(name in FITTED for name in args.points) and not args.generic:
        raise SystemExit("DCVC-UF's ¼ and ½ points need --generic (G's networks)")
    args.image_sha256 = file_sha256(Path(args.image_ckpt))
    args.video_sha256 = file_sha256(Path(args.video_ckpt))
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    restored = restore_units(publish, "point")
    copy = scratch / "bin" / Path(args.inspect).name  # the staging cache drops the execute bit (H1)
    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.inspect, copy)
    copy.chmod(0o755)
    args.inspect_source, args.inspect = args.inspect, str(copy)
    windows = decode_windows(args, items, int(args.frames), scratch)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("h4c run needs a GPU (DCVC-UF)")
    nets = {"lpips": load_lpips(Path(args.lpips_backbone), device), "dists": load_dists(Path(args.dists_backbone), device)}
    for p in nets["lpips"].parameters():  # LPIPS is a loss in the oracle's fit: gradients flow through it, not into it
        p.requires_grad_(False)
    on_cuda = {k: next(v.parameters()).is_cuda for k, v in nets.items()}
    began = time.time()
    count = len(restored)
    progress(count)
    for item, decoded in windows:
        count += process_item(args, item, decoded, nets, device, publish, scratch, set(restored))
    write_json(stage_dir() / "h4c.json", {
        "command": "run", "device": device_record(torch.cuda.get_device_name(0)),
        "gpu_capability": list(torch.cuda.get_device_capability(0)), "on_cuda": on_cuda, "seconds": round(time.time() - began, 2),
        "lpips": lpips_record(Path(args.lpips_backbone)), "dists_backbone_sha256": file_sha256(Path(args.dists_backbone)),
        "dcvc": {"tree": str(Path(sys.prefix) / "opt" / "DCVC"), "image_sha256": args.image_sha256, "video_sha256": args.video_sha256},
        "svt": {"decoder": _tool(args.decoder), "encoder": _tool(args.encoder), "preset": args.preset,
                "inspect": {"path": args.inspect_source, "sha256": file_sha256(Path(args.inspect))}},
        "generic": [{"path": p, "sha256": file_sha256(Path(p))} for p in args.generic],
        "settings": {"frames": int(args.frames), "items": args.items, "steps": int(args.steps), "points": args.points,
                     "train": h3d.TRAIN, "decision": DECISION},
        "items": [it["id"] for it in items], "restored_units": sorted(restored)})
    return 0


# ----------------------------------------------------------------- validation


def validate(stage: Path, result: dict[str, Any]) -> dict[str, bool]:
    units = load_units([str(stage / "published.tar")], "point")
    expected = len(result["items"]) * len(result["settings"]["points"])
    present = [(m, a, a["pixels_o"] > 0) for m, a in units]
    dcvc = [m["point"] for m, _ in units if m["point"]["codec"] == "dcvc-uf-htl"]
    first = [m for m, _ in units if "identity_reproduces_base" in m["checks"]]
    fitted = [(m, a, o) for m, a, o in present if "oracle" in m["rows"]]
    oracle = [(m["rows"].index("oracle"), a, o) for m, a, o in fitted]
    return {
        "every_point_present": len(units) == expected and all(m["npz_hash_ok"] for m, _ in units),
        "every_net_on_cuda": all(result["on_cuda"].values()),
        "intended_gpu": tuple(result["gpu_capability"]) >= (8, 6),
        "sparse_jpegs_match_decoded_frames": all(g["holds"] for m, _ in units for g in m["decode"]["jpeg_gate"]),
        "mask_set_matches_record": all(m["mask_set"]["masks_rle_sha256"] == m["mask_set"]["record_masks_rle_sha256"] for m, _ in units),
        "objects_handled_in_window": all(m["contact"]["handled"] and m["object_frames"] > 0 for m, _ in units),
        "regions_disjoint_and_scored": all(np.isfinite(a["lpips_o"][0][o]).all() and np.isfinite(a["dists_o"][0][o]).all()
                                           and np.isfinite(a["lpips_r"][0][a["pixels_r"] > 0]).all() for _, a, o in present),
        "every_row_scored_on_o": all(np.isfinite(a["lpips_o"][:, o]).all() for _, a, o in present),
        "identity_reproduces_base": not fitted or (bool(first) and all(m["checks"]["identity_reproduces_base"] for m in first)),
        "fit_ran_every_step": all(m["fits"]["oracle"]["curve"][-1]["step"] == m["fits"]["oracle"]["steps"] for m, _, _ in fitted),
        # Fitted on these very frames with LPIPS in the loss, the oracle should not raise LPIPS_O.
        "oracle_lpips_not_worse": all(np.nanmean(a["lpips_o"][j][o]) <= np.nanmean(a["lpips_o"][0][o]) + 1e-3 for j, a, o in oracle),
        "dcvc_decodes_deterministic": all(p["deterministic"] and p["decoder_matches_encoder_intra"] for p in dcvc),
        "region_rate_inside_stream_rate": all(0.0 <= p["kbps_o"] < p["kbps"] and 0.0 < p["kbps_r"] < p["kbps"] for p in dcvc),
    }


def command_validate(args: argparse.Namespace) -> int:
    stage = stage_dir()
    result = json.loads((stage / "h4c.json").read_text())
    checks = {k: bool(v) for k, v in validate(stage, result).items()}  # numpy booleans do not serialize
    report = {"passed": all(checks.values()), "checks": checks}
    target = os.environ.get("PS_VALIDATION_PATH")
    if target:
        write_json(Path(target), report)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


# ----------------------------------------------------------------- report (CPU)


def summary(a: dict[str, np.ndarray], j: int, region: str) -> dict[str, float]:
    """LPIPS, DISTS and PSNR of row ``j`` on ``region``, means over the frames where the region has pixels."""
    pixels = a[f"pixels_{region}"]
    has = pixels > 0
    return {"lpips": float(np.nanmean(a[f"lpips_{region}"][j][has])), "dists": float(np.nanmean(a[f"dists_{region}"][j][has])),
            "psnr": float(np.mean([psnr_of(s, p) for s, p in zip(a[f"sse_{region}"][j][has], pixels[has])])),
            "flicker": float(np.nanmean(a[f"flicker_{region}"][j]))}


def object_kbps(h1_item: dict[str, Any]) -> dict[str, float]:
    """SVT-AV1's kbps on H1's handled-object part (fill set) per CRF."""
    return {p: float(h1_item["bits"][p][FILL]["kbps"]["handled_object"]) for p in h1_item["bits"]}


def rows_by_arm(units: list[tuple[dict[str, Any], dict[str, np.ndarray]]], h1: dict[str, dict[str, Any]],
                region: str) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Points per arm (``bar``, ``G``, ``oracle``) and item on ``region``; the bar is every base point."""
    out: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for meta, a in units:
        p = meta["point"]
        if region == "o":
            kbps = object_kbps(h1[meta["item"]])[f"{p['crf']:g}"] if "kbps_o" not in p else p["kbps_o"]
        elif "kbps_r" in p:
            kbps = p["kbps_r"]
        else:
            continue  # SVT-AV1's rate on R is H3c's; the hands' bar is H3d's
        for j, row in enumerate(meta["rows"]):
            q = summary(a, j, region)
            if not math.isfinite(q["lpips"]):
                continue
            arm = "bar" if row == "base" else row
            if arm != "bar" and meta["name"] not in FITTED:
                continue
            out.setdefault(arm, {}).setdefault(meta["item"], []).append({"name": meta["name"], "kbps": float(kbps), **q})
    return out


def verdicts(arms: dict[str, dict[str, list[dict[str, Any]]]], bar: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out = {}
    for arm, per in sorted(arms.items()):
        items = {}
        for item, test in sorted(per.items()):
            if item in bar:
                items[item] = {**{m: h3d.compare(test, bar[item], m) for m in h3d.METRICS}, "front": test}
        out[arm] = {"items": items, "verdict": h3d.arm_verdict(items)}
    return out


def stage1(o: dict[str, Any]) -> dict[str, Any]:
    oracle = o.get("oracle", {}).get("verdict", {}).get("verdict", "incomplete")
    g = o.get("G", {}).get("verdict", {}).get("verdict", "incomplete")
    if "incomplete" in (oracle, g):
        return {"oracle": oracle, "G": g, "next": "incomplete"}
    if oracle == "fails":
        coder = "DCVC-UF + G" if g == "passes" else "DCVC-UF off the shelf (generative completion reported, not built)"
        return {"oracle": oracle, "G": g, "next": f"arms 2 and 3 not built; objects coded by {coder}"}
    return {"oracle": oracle, "G": g, "next": "stage 2: G_O and G_HO"}


def command_report(args: argparse.Namespace) -> int:
    units = load_units(args.run, "point")
    h1 = {row["id"]: row for row in json.loads(Path(args.h1_report).read_text())["per_item"]}
    h3d_report = json.loads(Path(args.h3d_report).read_text())
    by_region = {r: rows_by_arm(units, h1, r) for r in REGIONS}
    o_arms = {a: p for a, p in by_region["o"].items() if a != "bar"}
    report_o = verdicts(o_arms, by_region["o"].get("bar", {}))
    # On R, G and the oracle's base points against H3d's bar on R (its twenty points, H3c's units), and the base points
    # and G checked against H3d's numbers: the same code path reproduces the hands' result.
    bar_r = {item: [{k: q[k] for k in ("name", "kbps", "lpips", "dists", "psnr")} for q in rows] for item, rows in h3d_report["bar"].items()}
    report_r = verdicts({"G": by_region["r"].get("G", {})}, bar_r)
    cross: dict[str, Any] = {}
    for item, rows in by_region["r"].get("bar", {}).items():
        ref = {q["name"].replace("reference@", "dcvc@"): q for q in bar_r.get(item, [])}
        g_ref = {q["name"]: q for q in h3d_report["component"]["arms"]["G-L"]["items"].get(item, {}).get("front", [])}
        for q in rows:
            if q["name"] in ref:
                cross.setdefault(item, {})[q["name"]] = {"lpips_gap": q["lpips"] - ref[q["name"]]["lpips"],
                                                         "rate_ratio": q["kbps"] / ref[q["name"]]["kbps"]}
        for q in by_region["r"].get("G", {}).get(item, []):
            if q["name"] in g_ref:
                cross.setdefault(item, {})[f"G:{q['name']}"] = {"lpips_gap": q["lpips"] - g_ref[q["name"]]["lpips"]}
    holds = all(abs(e["lpips_gap"]) <= 0.005 and abs(e.get("rate_ratio", 1.0) - 1) <= 0.05 for c in cross.values() for e in c.values())
    pixels = {meta["item"]: {"object_frames": meta["object_frames"], "object_pixels_mean": float(np.mean(a["pixels_o"][a["pixels_o"] > 0])),
                             "handled": meta["contact"]["handled"]} for meta, a in units}
    report = {"decision_rule": DECISION, "stage1": stage1(report_o), "o": report_o, "r": report_r,
              "bar_o": by_region["o"].get("bar", {}), "cross_check_r": cross, "cross_check_holds": holds, "objects": pixels,
              "runs": [{"path": p, "sha256": file_sha256(Path(p))} for p in args.run],
              "h1_report_sha256": file_sha256(Path(args.h1_report)), "h3d_report_sha256": file_sha256(Path(args.h3d_report))}
    out = Path(args.out or stage_dir())
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "h4c-report.json", report)
    print(json.dumps({"stage1": report["stage1"], "o": {a: s["verdict"] for a, s in report_o.items()},
                      "r": {a: s["verdict"] for a, s in report_r.items()}, "cross_check_holds": holds}, indent=1, default=str))
    return 0


# ----------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments.visor.h4c")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--eval-set", required=True)
    run.add_argument("--archive", required=True)
    run.add_argument("--masks", required=True)
    run.add_argument("--mask-record", required=True)
    run.add_argument("--video", action="append", default=[])
    run.add_argument("--items", required=True)
    run.add_argument("--frames", default="240")
    run.add_argument("--points", required=True, help="comma list of names in BAR")
    run.add_argument("--steps", default="2000")
    run.add_argument("--image-ckpt", required=True)
    run.add_argument("--video-ckpt", required=True)
    run.add_argument("--lpips-backbone", required=True)
    run.add_argument("--dists-backbone", required=True)
    run.add_argument("--inspect", required=True)
    run.add_argument("--svt-streams", action="append", default=[])
    run.add_argument("--svt-result", action="append", default=[])
    run.add_argument("--generic", action="append", default=[], help="H3d's generic jobs' published.tar (G's networks)")
    run.add_argument("--decoder", default="dav1d")
    run.add_argument("--encoder", default="SvtAv1EncApp")
    run.add_argument("--preset", default="4")
    commands.add_parser("validate")
    report = commands.add_parser("report")
    report.add_argument("--run", action="append", required=True)
    report.add_argument("--h1-report", required=True)
    report.add_argument("--h3d-report", required=True)
    report.add_argument("--out", default="")
    args = parser.parse_args(argv)
    return {"run": command_run, "validate": command_validate, "report": command_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
