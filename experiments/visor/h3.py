"""PLAN step H3: hand and forearm rendering. H3a, the texture-transfer oracle.

    python -m experiments.visor.h3 oracle --eval-set JSON --archive DIR --masks DIR --mask-record JSON \\
        --svt-streams DIR --svt-result JSON [--svt-streams DIR --svt-result JSON] --lpips-backbone PTH \\
        --video V.MP4 ... --items all|ID,ID --frames N
    python -m experiments.visor.h3 validate
    python -m experiments.visor.h3 report --oracle TAR ... --h1-report JSON --code JSON --out DIR

``oracle`` scores, on every frame of an evaluation-set window with a hand, the union of VISOR's hand
masks (forearm included) in two ways (docs/experiments.md, 2026-10-10 H3a):

* SVT-AV1: B2's decoded stream at each CRF inside the region, the source outside it.
* The oracle, per CRF ``c`` and margin ``δ``: a bank of reference frames (SVT-AV1's decoded pixels at
  ``c``) each warped to the frame by DIS flow computed between the source frames (the target's own
  motion); the warp with the least squared error is kept unless its LPIPS exceeds SVT-AV1's by more
  than ``δ``, in which case the frame becomes a reference and shows SVT-AV1's pixels.

LPIPS is B2's (AlexNet spatial map at full resolution) averaged inside the region, on the full frame.
Each finished window is published as ``oracle/<item>.npz`` with a ``.json`` and checkpointed.
``report`` adds rates (H1's SVT-AV1 bits on the same pixels, H2's pose stream) and applies
``ORACLE_DECISION``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json
from experiments.visor.h2 import load_units, restore_units, save_unit

FILL = "visor_dense_sam_fill"
HANDS = ("left hand", "right hand")
POINTS = (41.0, 48.0, 55.0, 62.0)  # B2's SVT-AV1 CRFs that H1 accounted by part
MARGINS = (0.02, 0.05, 0.10)
BANK = 4  # most recent references tried per frame
FLOW_PAD = 64  # px around the two regions' boxes
FLOW_SCALE = 0.5  # DIS on half-resolution crops (as H1), the flow upsampled
LPIPS_BATCH = 8
COST_FACTORS = (1.0, 3.0, 5.0)
RESAMPLES = 1000
SEED = "pointstream-h3"

ORACLE_DECISION = {
    "region": "union of VISOR's left- and right-hand masks (fill set), forearm included",
    "metric": "LPIPS (B2's AlexNet spatial map at full resolution) averaged inside the region; PSNR beside it",
    "svt": "B2's streams at CRF 41, 48, 55, 62 composited over the source outside the region; rate = H1's bits "
           "on hand + forearm + hand without fit at that CRF (fill set)",
    "oracle": f"bank of SVT-AV1-decoded references at the same CRF, the {BANK} most recent tried, each warped by "
              "DIS flow between the source frames; a frame becomes a reference when no warp is within the margin "
              "of SVT-AV1's LPIPS on it",
    "rate": "references × SVT-AV1's hand bits per hand-frame at that CRF + H2's pose stream (WiLoR, 100 ms budget)",
    "comparison": "per item, SVT-AV1's LPIPS at the oracle's rate (linear in log-rate, held at the end points); "
                  "difference oracle − SVT-AV1 averaged over items, 95% paired bootstrap by item",
    "pass": "mean difference below zero at some (CRF, margin)",
    "margins": MARGINS, "points": POINTS, "bank": BANK, "cost_factors": COST_FACTORS,
}


# ----------------------------------------------------------------- geometry and warps

def box_of(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def flow_box(a: np.ndarray, b: np.ndarray, pad: int = FLOW_PAD) -> tuple[int, int, int, int]:
    """The union of two masks' boxes grown by ``pad``, inside the frame, with even sides."""
    boxes = [x for x in (box_of(a), box_of(b)) if x is not None]
    if not boxes:
        raise ValueError("both masks are empty")
    h, w = a.shape
    x0 = max(0, min(x[0] for x in boxes) - pad)
    y0 = max(0, min(x[1] for x in boxes) - pad)
    x1 = min(w, max(x[2] for x in boxes) + pad)
    y1 = min(h, max(x[3] for x in boxes) + pad)
    if (x1 - x0) % 2:
        x1 = x1 + 1 if x1 < w else x1 - 1
    if (y1 - y0) % 2:
        y1 = y1 + 1 if y1 < h else y1 - 1
    return x0, y0, x1, y1


def dis_flow(target: np.ndarray, source: np.ndarray, scale: float = FLOW_SCALE) -> np.ndarray:
    """Backward flow (h, w, 2): ``target[y, x] ≈ source[y + f_y, x + f_x]``, grayscale uint8 inputs."""
    import cv2

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)  # type: ignore[attr-defined]
    target, source = np.ascontiguousarray(target), np.ascontiguousarray(source)  # crops are views
    h, w = target.shape
    if scale != 1.0:
        size = (int(round(w * scale)), int(round(h * scale)))
        small = dis.calc(cv2.resize(target, size, interpolation=cv2.INTER_AREA),
                         cv2.resize(source, size, interpolation=cv2.INTER_AREA), None)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR) / scale
    return dis.calc(target, source, None)


def warp(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """``image`` sampled at each pixel plus its flow (bilinear, edges replicated)."""
    import cv2

    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = np.asarray(gx + flow[..., 0], np.float32)
    map_y = np.asarray(gy + flow[..., 1], np.float32)
    return np.asarray(cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))


def region_sse(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> int:
    d = a[mask].astype(np.int32) - b[mask].astype(np.int32)
    return int((d * d).sum())


def psnr_of(sse: float, pixels: float) -> float:
    from src.codecs.quality import psnr

    return psnr(sse / (3 * pixels))


# ----------------------------------------------------------------- the oracle's state per (CRF, margin)

class Run:
    """One operating point's bank and its per-frame record."""

    def __init__(self, point: float, margin: float, frames: int) -> None:
        self.point, self.margin = point, margin
        self.bank: list[int] = []
        self.lpips = np.full(frames, np.nan)
        self.sse = np.zeros(frames, np.int64)
        self.reference = np.zeros(frames, bool)
        self.source = np.full(frames, -1, int)

    def candidates(self) -> list[int]:
        return self.bank[-BANK:]

    def settle(self, t: int, warp_lpips: float | None, warp_sse: int, chosen: int, svt_lpips: float, svt_sse: int) -> None:
        """Frame ``t`` shows the warp unless it is worse than SVT-AV1 by more than the margin (or there is none)."""
        if warp_lpips is None or warp_lpips - svt_lpips > self.margin:
            self.reference[t] = True
            self.lpips[t], self.sse[t], self.source[t] = svt_lpips, svt_sse, t
            self.bank.append(t)
        else:
            self.lpips[t], self.sse[t], self.source[t] = warp_lpips, warp_sse, chosen


# ----------------------------------------------------------------- per item

def rgb_memmap(yuv: Path, out: Path, frames: int, width: int, height: int, full_range: bool, device: str) -> np.ndarray:
    """B2's RGB view of a 4:2:0 file, as a (frames, h, w, 3) uint8 memmap."""
    import torch

    from src.codecs.quality import read_frames, yuv420_to_rgb

    array = np.lib.format.open_memmap(out, mode="w+", dtype=np.uint8, shape=(frames, height, width, 3))
    for first in range(0, frames, 8):
        count = min(8, frames - first)
        planes = torch.from_numpy(read_frames(yuv, width, height, first, count)).to(device)
        rgb = yuv420_to_rgb(planes, width, height, full_range=full_range)
        array[first:first + count] = rgb.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
    array.flush()
    return np.load(out, mmap_mode="r")


def region_lpips(net: Any, frames: list[np.ndarray], truth: np.ndarray, mask: np.ndarray, device: str) -> list[float]:
    """LPIPS of each full frame against ``truth``, averaged inside ``mask``."""
    import torch

    out: list[float] = []
    ref = torch.from_numpy(np.ascontiguousarray(truth)).to(device).permute(2, 0, 1)[None].float() / 127.5 - 1.0
    m = torch.from_numpy(mask).to(device).float()
    n = float(mask.sum())
    for first in range(0, len(frames), LPIPS_BATCH):
        batch = torch.from_numpy(np.stack(frames[first:first + LPIPS_BATCH])).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        with torch.inference_mode():
            lp = net(batch, ref.expand(len(batch), -1, -1, -1))[:, 0]
        out.extend(((lp * m).flatten(1).sum(1) / n).double().cpu().tolist())
    return out


def composite(truth: np.ndarray, inside: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """``truth`` with ``inside``'s pixels in ``mask``; ``inside`` is a crop at ``box`` when given."""
    out = truth.copy()
    if box is None:
        out[mask] = inside[mask]
    else:
        x0, y0, x1, y1 = box
        crop_mask = mask[y0:y1, x0:x1]
        out[y0:y1, x0:x1][crop_mask] = inside[crop_mask]
    return out


def process_item(args: argparse.Namespace, item: dict[str, Any], net: Any, device: str, publish: Path,
                 decoded_source: dict[str, Any], scratch: Path, cross_device: bool) -> None:
    import cv2

    from experiments.visor.b2 import load_mask_sets
    from experiments.visor.h1 import stream_paths
    from src.codecs import svtav1

    began = time.time()
    frames = int(args.frames)
    work = scratch / "work" / item["id"]
    work.mkdir(parents=True, exist_ok=True)
    width, height, full_range = decoded_source["width"], decoded_source["height"], decoded_source["full_range"]
    clips, mask_info = load_mask_sets(item, frames, {FILL: args.masks}, {FILL: args.mask_record}, Path(args.archive))
    clip = clips[FILL]
    regions = [clip.class_mask(t, HANDS[0]) | clip.class_mask(t, HANDS[1]) for t in range(frames)]
    hand_frames = [t for t in range(frames) if regions[t].any()]
    truth = rgb_memmap(Path(decoded_source["path"]), work / "source-rgb.npy", frames, width, height, full_range, device)
    os.remove(decoded_source["path"])
    gray = np.stack([cv2.cvtColor(np.asarray(truth[t]), cv2.COLOR_RGB2GRAY) for t in range(frames)])
    streams, stream_info = stream_paths(args, item["id"])
    names = [f"{p:g}" for p in POINTS]
    if set(names) - set(streams):
        raise RuntimeError(f"{item['id']}: streams missing for CRF {sorted(set(names) - set(streams))}")
    allowance = max(1, int(os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 1))
    decoded: dict[str, np.ndarray] = {}
    for name in names:
        yuv = work / f"crf{name}.yuv"
        command = svtav1.decode_command(args.decoder, streams[name], yuv, threads=allowance)
        timing = svtav1.run(command, 1800)
        count = yuv.stat().st_size // (width * height * 3 // 2)
        stream_info[name].update({"decode_command": command, "decode_seconds": round(timing["wall"], 2), "decoded_frames": int(count)})
        if count < frames:
            raise RuntimeError(f"{item['id']}: CRF {name} decoded {count} frames, need {frames}")
        decoded[name] = rgb_memmap(yuv, work / f"crf{name}-rgb.npy", frames, width, height, full_range, device)
        yuv.unlink()

    runs = [Run(p, m, frames) for p in POINTS for m in MARGINS]
    svt_lpips = np.full((len(POINTS), frames), np.nan)
    svt_sse = np.zeros((len(POINTS), frames), np.int64)
    pixels = np.array([int(r.sum()) for r in regions])
    flows_computed = 0
    checks: dict[str, Any] = {}
    cv2.setNumThreads(1)
    pool = ThreadPoolExecutor(max_workers=allowance)
    for t in hand_frames:
        mask = regions[t]
        truth_t = np.asarray(truth[t])
        svt_frames = [composite(truth_t, np.asarray(decoded[n][t]), mask) for n in names]
        lp = region_lpips(net, svt_frames, truth_t, mask, device)
        if cross_device and "lpips_cpu_vs_device" not in checks:
            cpu = region_lpips(_cpu_net(args), svt_frames[-1:], truth_t, mask, "cpu")
            checks["lpips_cpu_vs_device"] = abs(cpu[0] - lp[-1])
        for i, n in enumerate(names):
            svt_lpips[i, t] = lp[i]
            svt_sse[i, t] = region_sse(svt_frames[i], truth_t, mask)
        wanted = sorted({r for run in runs for r in run.candidates()})
        boxes = {r: flow_box(mask, regions[r]) for r in wanted}

        def flow_of(r: int) -> np.ndarray:
            x0, y0, x1, y1 = boxes[r]
            return dis_flow(gray[t, y0:y1, x0:x1], gray[r, y0:y1, x0:x1])

        flows = dict(zip(wanted, pool.map(flow_of, wanted)))
        flows_computed += len(wanted)
        best: list[tuple[int, int, np.ndarray | None]] = []
        for run in runs:
            i = names.index(f"{run.point:g}")
            r_best, sse_best, crop_best = -1, 0, None
            for r in run.candidates():
                x0, y0, x1, y1 = boxes[r]
                warped = warp(np.asarray(decoded[names[i]][r, y0:y1, x0:x1]), flows[r])
                sse = region_sse(warped, truth_t[y0:y1, x0:x1], mask[y0:y1, x0:x1])
                if crop_best is None or sse < sse_best:
                    r_best, sse_best, crop_best = r, sse, warped
            best.append((r_best, sse_best, None if crop_best is None else composite(truth_t, crop_best, mask, boxes[r_best])))
        shown = [c[2] for c in best if c[2] is not None]
        warp_lp = iter(region_lpips(net, shown, truth_t, mask, device)) if shown else iter(())
        for run, (r, sse, frame) in zip(runs, best):
            i = names.index(f"{run.point:g}")
            run.settle(t, next(warp_lp) if frame is not None else None, sse, r, svt_lpips[i, t], int(svt_sse[i, t]))
    pool.shutdown()
    shutil.rmtree(work)
    arrays: dict[str, np.ndarray] = {
        "pixels": pixels, "svt_lpips": svt_lpips, "svt_sse": svt_sse,
        "oracle_lpips": np.stack([r.lpips for r in runs]), "oracle_sse": np.stack([r.sse for r in runs]),
        "oracle_reference": np.stack([r.reference for r in runs]), "oracle_source": np.stack([r.source for r in runs]),
        "run_point": np.array([r.point for r in runs]), "run_margin": np.array([r.margin for r in runs]),
        "points": np.array(POINTS),
    }
    meta = {"unit": item["id"], "video": item["video"], "fps": float(item["fps"]), "frames": frames,
            "hand_frames": len(hand_frames), "decode": {k: v for k, v in decoded_source.items() if k != "path"},
            "mask_set": mask_info[FILL], "streams": stream_info, "flows": flows_computed, "checks": checks,
            "seconds": round(time.time() - began, 2)}
    save_unit(publish, "oracle", item["id"], arrays, meta)


_CPU_NET: dict[str, Any] = {}


def _cpu_net(args: argparse.Namespace) -> Any:
    from src.codecs.quality import load_lpips

    if "net" not in _CPU_NET:
        _CPU_NET["net"] = load_lpips(Path(args.lpips_backbone), "cpu")
    return _CPU_NET["net"]


def command_oracle(args: argparse.Namespace) -> int:
    from experiments.visor.b2 import decode_window, select_items

    eval_set = json.loads(Path(args.eval_set).read_text())
    items = select_items(eval_set, args.items)
    videos = {Path(v).stem: Path(v) for v in args.video}
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    restored = restore_units(publish, "oracle")
    todo = [it for it in items if it["id"] not in restored]
    allowance = max(1, int(os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 1))
    frames = int(args.frames)
    # PyAV decodes in fresh processes, before torch loads (B2).
    pool = ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"))

    def submit(item: dict[str, Any]) -> Any:
        target = scratch / "source" / f"{item['id']}.yuv"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        return pool.submit(decode_window, videos[item["video"]], item, frames, Path(args.archive), target, max(1, allowance // 2))

    pending = {i: submit(todo[i]) for i in range(min(2, len(todo)))}
    import torch

    from src.codecs.quality import load_lpips, lpips_record

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(allowance)
    net = load_lpips(Path(args.lpips_backbone), device)
    record: dict[str, Any] = {"device": device}
    if device == "cuda":
        from experiments.audit.env_smoke import device_record

        record = device_record(torch.cuda.get_device_name(0))
    done = len(restored)
    began = time.time()
    for number, item in enumerate(todo):
        source = pending.pop(number).result()
        if number + 2 < len(todo):
            pending[number + 2] = submit(todo[number + 2])
        process_item(args, item, net, device, publish, source, scratch, cross_device=number == 0)
        done += 1
        progress(done)
    pool.shutdown()
    write_json(stage_dir() / "h3.json", {
        "command": "oracle", "device": record, "seconds": round(time.time() - began, 2),
        "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1) if device == "cuda" else None,
        "lpips": lpips_record(Path(args.lpips_backbone)),
        "decoder": _tool(args.decoder),
        "inputs": {"eval_set_sha256": file_sha256(Path(args.eval_set)), "mask_record_sha256": file_sha256(Path(args.mask_record)),
                   "svt_result_sha256": [file_sha256(Path(p)) for p in args.svt_result]},
        "settings": {"frames": frames, "items": args.items, "decision": ORACLE_DECISION, "flow_pad": FLOW_PAD,
                     "flow_scale": FLOW_SCALE, "flow": "OpenCV DIS, PRESET_MEDIUM, grayscale"},
        "items": [it["id"] for it in items], "restored_units": sorted(restored),
    })
    return 0


def _tool(name: str) -> dict[str, Any]:
    from src.codecs import svtav1

    try:
        return svtav1.tool(name)
    except FileNotFoundError:
        return {"path": name, "missing": True}


# ----------------------------------------------------------------- validate

def validate_result(stage: Path) -> dict[str, bool]:
    if (stage / "h3a.json").exists():
        out = json.loads((stage / "h3a.json").read_text())
        return {"every_operating_point_scored": len(out["points"]) == len(POINTS) * len(MARGINS),
                "items_scored": out["items"] > 0, "decision_made": out["decision"]["passes"] in (True, False)}
    result = json.loads((stage / "h3.json").read_text())
    units = load_units([str(stage / "published.tar")], "oracle")
    first = [m for m, _ in units if "lpips_cpu_vs_device" in m["checks"]]
    lp = [a["svt_lpips"] for _, a in units]
    shares = [a["oracle_reference"].sum(1) / max(1, m["hand_frames"]) for m, a in units]
    by_margin = np.mean([[s[[j for j in range(len(s)) if a["run_margin"][j] == margin]].mean() for margin in MARGINS]
                         for s, (_, a) in zip(shares, units)], 0) if units else np.zeros(len(MARGINS))
    return {
        "every_item_present": bool(units) and len(units) == len(result["items"]) and all(m["npz_hash_ok"] for m, _ in units),
        "sparse_jpegs_match_decoded_frames": all(g["holds"] for m, _ in units for g in m["decode"]["jpeg_gate"]),
        "mask_set_matches_record": all(m["mask_set"]["masks_rle_sha256"] == m["mask_set"]["record_masks_rle_sha256"] for m, _ in units),
        "streams_match_records": all(s["sha256"] == s["recorded_sha256"] for m, _ in units for s in m["streams"].values()),
        "streams_decode_whole_window": all(s["decoded_frames"] >= m["frames"] for m, _ in units for s in m["streams"].values()),
        "lpips_cpu_matches_device_1e-3": bool(first) and all(m["checks"]["lpips_cpu_vs_device"] < 1e-3 for m in first),
        "scores_finite_on_hand_frames": all(np.isfinite(a["svt_lpips"][:, a["pixels"] > 0]).all()
                                            and np.isfinite(a["oracle_lpips"][:, a["pixels"] > 0]).all() for _, a in units),
        # Coarser CRFs look worse on the hands, on average.
        "svt_lpips_rises_with_crf": bool(lp) and float(np.nanmean(np.concatenate([x[0] for x in lp]))) < float(np.nanmean(np.concatenate([x[-1] for x in lp]))),
        "first_hand_frame_is_a_reference": all(a["oracle_reference"][:, int(np.argmax(a["pixels"] > 0))].all() for _, a in units if (a["pixels"] > 0).any()),
        # A larger margin sends fewer references, on average.
        "references_fall_with_margin": bool(units) and bool(np.all(np.diff(by_margin) <= 1e-9)),
    }


def command_validate(args: argparse.Namespace) -> int:
    checks = validate_result(stage_dir())
    report = {"passed": all(checks.values()), "checks": checks}
    target = os.environ.get("PS_VALIDATION_PATH")
    if target:
        write_json(Path(target), report)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


# ----------------------------------------------------------------- report (CPU)

def at_rate(rates: np.ndarray, values: np.ndarray, rate: float) -> float:
    """``values`` at ``rate``, linear in log-rate between the points, held at the end points."""
    order = np.argsort(rates)
    return float(np.interp(math.log(rate), np.log(rates[order]), values[order]))


def hand_kbps(h1_item: dict[str, Any]) -> dict[str, float]:
    """SVT-AV1's kbps on the hand masks' pixels (hand, forearm, hand without fit; fill set) per CRF."""
    return {p: sum(h1_item["bits"][p][FILL]["kbps"][k] for k in ("hand", "forearm", "hand_no_fit")) for p in h1_item["bits"]}


def item_curves(meta: dict[str, Any], a: dict[str, np.ndarray], kbps: dict[str, float], pose_kbps: float,
                factor: float = 1.0) -> dict[str, Any]:
    """The item's SVT-AV1 points and each oracle operating point's rate, LPIPS and PSNR, and both compared."""
    hand = a["pixels"] > 0
    n = int(hand.sum())
    names = [f"{p:g}" for p in a["points"]]
    svt_rate = np.array([kbps[k] for k in names])
    svt_lp = np.nanmean(a["svt_lpips"][:, hand], 1)
    svt_ps = np.array([np.mean([psnr_of(s, p) for s, p in zip(row[hand], a["pixels"][hand])]) for row in a["svt_sse"]])
    out = []
    for j, (point, margin) in enumerate(zip(a["run_point"], a["run_margin"])):
        refs = int(a["oracle_reference"][j].sum())
        rate = factor * kbps[f"{point:g}"] * refs / n + pose_kbps
        lp = float(np.nanmean(a["oracle_lpips"][j][hand]))
        ps = float(np.mean([psnr_of(s, p) for s, p in zip(a["oracle_sse"][j][hand], a["pixels"][hand])]))
        out.append({"point": float(point), "margin": float(margin), "reference_share": refs / n, "kbps": rate,
                    "lpips": lp, "psnr": ps, "svt_lpips_at_rate": at_rate(svt_rate, svt_lp, rate),
                    "svt_psnr_at_rate": at_rate(svt_rate, svt_ps, rate)})
    return {"item": meta["unit"], "hand_frames": n, "svt": [{"point": float(p), "kbps": float(r), "lpips": float(lv), "psnr": float(pv)}
                                                         for p, r, lv, pv in zip(a["points"], svt_rate, svt_lp, svt_ps)],
            "oracle": out}


def seed_of(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:16], 16)


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator, resamples: int = RESAMPLES) -> dict[str, float]:
    idx = rng.integers(0, len(values), (resamples, len(values)))
    means = values[idx].mean(1)
    return {"mean": float(values.mean()), "low": float(np.percentile(means, 2.5)), "high": float(np.percentile(means, 97.5))}


def summarize(curves: list[dict[str, Any]], rng: np.random.Generator) -> list[dict[str, Any]]:
    out = []
    for j, first in enumerate(curves[0]["oracle"]):
        rows = [c["oracle"][j] for c in curves]
        d_lp = np.array([r["lpips"] - r["svt_lpips_at_rate"] for r in rows])
        d_ps = np.array([r["psnr"] - r["svt_psnr_at_rate"] for r in rows])
        out.append({"point": first["point"], "margin": first["margin"],
                    "kbps": float(np.mean([r["kbps"] for r in rows])), "lpips": float(np.mean([r["lpips"] for r in rows])),
                    "psnr": float(np.mean([r["psnr"] for r in rows])),
                    "reference_share": float(np.mean([r["reference_share"] for r in rows])),
                    "lpips_minus_svt": bootstrap_mean(d_lp, rng), "psnr_minus_svt": bootstrap_mean(d_ps, rng),
                    "items_better_lpips": int((d_lp < 0).sum())})
    return out


def command_report(args: argparse.Namespace) -> int:
    units = load_units(args.oracle, "oracle")
    h1 = {row["id"]: row for row in json.loads(Path(args.h1_report).read_text())["per_item"]}
    code = json.loads(Path(args.code).read_text())
    pose_kbps = float(code["choices"]["wilor"]["100"]["visor_kbps"])
    by_factor = {}
    curves_1 = []
    for factor in COST_FACTORS:
        curves = [item_curves(m, a, hand_kbps(h1[m["unit"]]), pose_kbps, factor) for m, a in units if (a["pixels"] > 0).any()]
        by_factor[f"{factor:g}"] = summarize(curves, np.random.default_rng(seed_of(f"{SEED}:{factor:g}")))
        if factor == 1.0:
            curves_1 = curves
    points = by_factor["1"]
    best = min(points, key=lambda r: r["lpips_minus_svt"]["mean"])
    svt_mean = [{"point": float(p), "kbps": float(np.mean([c["svt"][i]["kbps"] for c in curves_1])),
                 "lpips": float(np.mean([c["svt"][i]["lpips"] for c in curves_1])),
                 "psnr": float(np.mean([c["svt"][i]["psnr"] for c in curves_1]))} for i, p in enumerate(POINTS)]
    out = {
        "rule": ORACLE_DECISION, "items": len(curves_1), "pose_kbps": pose_kbps, "svt": svt_mean, "points": points,
        "cost_sensitivity": {k: v for k, v in by_factor.items() if k != "1"},
        "decision": {"passes": bool(best["lpips_minus_svt"]["mean"] < 0), "best": best},
        "per_item": curves_1,
        "inputs": {"oracle": [{"path": p, "sha256": file_sha256(Path(p))} for p in args.oracle],
                   "h1_report_sha256": file_sha256(Path(args.h1_report)), "code_sha256": file_sha256(Path(args.code))},
        "units_hash_ok": all(m["npz_hash_ok"] for m, _ in units),
    }
    target = Path(args.out) if args.out else stage_dir()
    write_json(target / "h3a.json", out)
    print(json.dumps({"decision": out["decision"], "svt": svt_mean,
                      "points": [{k: r[k] for k in ("point", "margin", "kbps", "lpips", "reference_share")}
                                 | {"d": r["lpips_minus_svt"]} for r in points]}, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments.visor.h3")
    commands = parser.add_subparsers(dest="command", required=True)
    oracle = commands.add_parser("oracle")
    oracle.add_argument("--eval-set", required=True)
    oracle.add_argument("--archive", required=True)
    oracle.add_argument("--masks", required=True)
    oracle.add_argument("--mask-record", required=True)
    oracle.add_argument("--svt-streams", action="append", required=True)
    oracle.add_argument("--svt-result", action="append", required=True)
    oracle.add_argument("--lpips-backbone", required=True)
    oracle.add_argument("--video", action="append", default=[])
    oracle.add_argument("--decoder", default="dav1d")
    oracle.add_argument("--items", default="all")
    oracle.add_argument("--frames", default="240")
    commands.add_parser("validate")
    report = commands.add_parser("report")
    report.add_argument("--oracle", action="append", required=True)
    report.add_argument("--h1-report", required=True)
    report.add_argument("--code", required=True)
    report.add_argument("--out", default="", help="default: the stage directory")
    args = parser.parse_args(argv)
    return {"oracle": command_oracle, "validate": command_validate, "report": command_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
