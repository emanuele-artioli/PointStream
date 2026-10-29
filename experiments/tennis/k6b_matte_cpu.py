"""CPU-only saved-crop matte packets, coverage and bounded real neural-judge pilot."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import resource
import struct
import tarfile
import time
from typing import Any

import numpy as np

from experiments.tennis.k6b_rate_streams import (
    decode_keypoints, decode_masks, encode_masks, seal, unseal,
)
from experiments.tennis.k6b_rate_transport import array_sha, sha, write
from src.pointstream import mask as mk
from src.pointstream.composite import OBJECT

SCALES = (8, 16, 32)
PILOT_WINDOWS = ("alcaraz_hurkacz/031@129", "medvedev_struff/014@194")


def ordered_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = (*PILOT_WINDOWS, "alcaraz_hurkacz/016@273")
    indexed = {r["window"]: r for r in rows}
    return [indexed[w] for w in order] + [r for r in rows if r["window"] not in order]


def cpu() -> float:
    return sum(u.ru_utime + u.ru_stime for u in (
        resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)))


def encode(planes: np.ndarray, scale: int, names: list[str], mode: str) -> bytes:
    meta = {"scale": scale, "shape": list(planes.shape), "units": names,
            "mode": mode, "output_size": [1920, 1080],
            "upsample": "pointstream.mask.upsample-bilinear-half", "frame_ticks": [0, 240]}
    if mode == "h5":
        streams = [mk.encode(p.astype(np.uint8) * OBJECT, scale) for p in planes.transpose(1, 0, 2, 3)]
        body = b"".join(struct.pack(">I", len(s)) + s for s in streams)
    elif mode == "rle":
        body = encode_masks(planes, "rle", scale // 2)
    else:
        raise ValueError(mode)
    return seal(b"MTQ1", meta, body)


def decode(blob: bytes) -> tuple[dict[str, Any], np.ndarray]:
    meta, body = unseal(blob, b"MTQ1")
    if meta["mode"] == "rle":
        _, out = decode_masks(body)
    else:
        parts = []
        pos = 0
        for _ in meta["units"]:
            n = struct.unpack_from(">I", body, pos)[0]
            classes, scale = mk.decode(body[pos + 4:pos + 4 + n])
            if scale != meta["scale"]:
                raise ValueError("Scale disagreement")
            parts.append(classes > 0)
            pos += 4 + n
        if pos != len(body):
            raise ValueError("Trailing matte data")
        out = np.stack(parts, 1)
    if list(out.shape) != meta["shape"]:
        raise ValueError("Shape disagreement")
    return meta, out


def box_maps(units: list[dict[str, Any]]) -> list[dict[int, np.ndarray]]:
    from experiments.tennis.k6b import run_boxes, runs
    out = []
    for u in units:
        boxes: dict[int, np.ndarray] = {}
        index = {int(t): i for i, t in enumerate(u["t"])}
        for ts in runs(u["t"].tolist()):
            kps = [u["kp"][index[t]] for t in ts]
            boxes.update(zip(ts, run_boxes(kps)))
        out.append(boxes)
    return out


def boundary(mask: np.ndarray) -> np.ndarray:
    import cv2
    return mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def coverage(mask: np.ndarray, box: np.ndarray | None) -> int:
    from experiments.tennis.k6b import int_box
    if box is None:
        return int(mask.sum())
    x, y, w, h = int_box(box)
    return int(mask.sum()) - int(mask[max(y, 0):min(y+h, 1080), max(x, 0):min(x+w, 1920)].sum()) if x+w > 0 and y+h > 0 and x < 1920 and y < 1080 else int(mask.sum())


def rates(row: dict[str, Any], packets: Path, target: Path, native: dict[str, Any]) -> dict[str, Any]:
    import cv2
    from experiments.tennis.k4 import safe
    root = packets / safe(row["window"])
    kp_meta, units = decode_keypoints((root / "keypoints.kps").read_bytes())
    _, masks = decode_masks((root / "exact-rle.matte").read_bytes())
    if masks.shape != (240, 2, 540, 960):
        raise ValueError("Frozen source mask dimensions changed")
    boxes = box_maps(units)
    names = [u["name"] for u in units]
    if kp_meta["window"] != row["window"] or native["frames"] != 240:
        raise ValueError("Window identity changed")
    result: dict[str, Any] = {"window": row["window"], "source_mask_sha256": array_sha(masks),
        "source_kps_sha256": sha((root / "keypoints.kps").read_bytes()), "settings": {},
        "frames": 240, "players": names, "judged": native["judged"]}
    for scale in SCALES:
        small = np.stack([mk.reduce(p.astype(np.uint8) * OBJECT, scale // 2) > 0
                          for p in masks.transpose(1, 0, 2, 3)], 1)
        candidates = []
        for mode in ("h5", "rle"):
            blob = encode(small, scale, names, mode)
            meta, restored = decode(blob)
            if not np.array_equal(small, restored) or meta["units"] != names:
                raise ValueError("Reduced-grid round trip failed")
            name = f"{safe(row['window'])}-s{scale}-{mode}.mtq"
            (target / name).write_bytes(blob)
            candidates.append({"mode": mode, "file": name, "bytes": len(blob), "sha256": sha(blob)})
        totals = {k: 0 for k in ("intersection", "union", "false_positive", "false_negative", "boundary_source", "boundary_decoded", "boundary_hit_source", "boundary_hit_decoded", "matte_pixels", "outside_crop", "outside_crop_judged", "new_outside_crop_judged")}
        for t in range(240):
            for p in range(2):
                source = cv2.resize(masks[t, p].astype(np.uint8), (1920, 1080), interpolation=cv2.INTER_NEAREST).astype(bool)
                restored = mk.upsample(small[t:t+1, p].astype(np.uint8) * OBJECT, (1920, 1080))[0] > 0
                totals["intersection"] += int((source & restored).sum())
                totals["union"] += int((source | restored).sum())
                totals["false_positive"] += int((restored & ~source).sum())
                totals["false_negative"] += int((source & ~restored).sum())
                bs, bd = boundary(source), boundary(restored)
                kernel = np.ones((5, 5), np.uint8)
                totals["boundary_source"] += int(bs.sum())
                totals["boundary_decoded"] += int(bd.sum())
                totals["boundary_hit_source"] += int((bs & cv2.dilate(bd.astype(np.uint8), kernel).astype(bool)).sum())
                totals["boundary_hit_decoded"] += int((bd & cv2.dilate(bs.astype(np.uint8), kernel).astype(bool)).sum())
                totals["matte_pixels"] += int(restored.sum())
                missing = coverage(restored, boxes[p].get(t))
                totals["outside_crop"] += missing
                if t in native["judged"]:
                    totals["outside_crop_judged"] += missing
                    totals["new_outside_crop_judged"] += coverage(restored & ~source, boxes[p].get(t))
        precision = totals["boundary_hit_decoded"] / max(totals["boundary_decoded"], 1)
        recall = totals["boundary_hit_source"] / max(totals["boundary_source"], 1)
        best = min(candidates, key=lambda c: c["bytes"])
        result["settings"][str(scale)] = {"candidates": candidates, "selected": best,
            "kbps": best["bytes"] * 8 / row["registered_seconds"] / 1000,
            "whole_window_kbps": best["bytes"] * 8 / row["whole_window_seconds"] / 1000,
            "iou": totals["intersection"] / max(totals["union"], 1),
            "boundary_f_2px_chebyshev": 2 * precision * recall / max(precision + recall, 1e-12),
            "counts": totals, "reduced_sha256": array_sha(small)}
    return result


def metric_pilot(row: dict[str, Any], packets: Path, anchor: Path, generated: dict[Any, Any], source: Path,
                 source_record: dict[str, Any], control: Path, backbones: list[Path], scratch: Path, target: Path) -> dict[str, Any]:
    from fractions import Fraction
    import cv2
    import torch
    from experiments.tennis import k6b, k6
    from experiments.tennis.k4 import safe
    from experiments.tennis.k4b_features import convert
    from experiments.visor.h3 import rgb_memmap
    from experiments.visor.h3b import load_dists, lpips_reduce, dists_crops
    from src.codecs.quality import load_lpips, lpips_record
    from src.segmentation.masks import decompress
    torch.set_num_threads(4)
    w = k6b.k6_window(anchor)
    native = w["row"]
    kp_meta, units = decode_keypoints((packets / safe(row["window"]) / "keypoints.kps").read_bytes())
    maps = box_maps(units)
    _, native_masks = decode_masks((packets / safe(row["window"]) / "exact-rle.matte").read_bytes())
    if sha(source.read_bytes()) != source_record["sha256"]:
        raise ValueError("Clip identity changed")
    work = scratch / safe(row["window"])
    work.mkdir()
    converted = convert("ffmpeg", {"id": row["window"], "start": native["start"], "frames": 240}, source, work / "source.yuv")
    truth = rgb_memmap(work / "source.yuv", work / "truth.npy", 240, 1920, 1080, False, "cpu")
    regions = [{p: cv2.resize(native_masks[t, i].astype(np.uint8), (1920, 1080), interpolation=cv2.INTER_NEAREST).astype(bool)
                for i, p in enumerate(native["players"]) if native_masks[t, i].any()} for t in range(240)]
    heights = {(p, t): float(np.ptp(np.nonzero(r[p].any(1))[0]) + 1) if p in r and r[p].any() else 1.0
               for t, r in enumerate(regions) for p in native["players"]}
    decoded_all = {key: {(n.rsplit("@", 1)[0], int(n.rsplit("@", 1)[1])): np.asarray(v, np.float64)
                         for n, v in d.items()} for key, d in json.loads(decompress((anchor / "keypoints.json.zst").read_bytes()))["decoded"].items()}
    order = kp_meta["compositing_order"]
    t = native["judged"][0]
    exact_r = np.logical_or.reduce(list(regions[t].values()))
    fill = k6.telea_fill(np.asarray(truth[t]), exact_r)
    _, drawn = k6b.k6_oracle(t, w, k6.variant_names()[-9:], truth, regions, fill, order, native["scored"], heights,
                            decoded_all, max(1, round(k6.REF_GAP_S * float(Fraction(native["fps"])))) )
    fixed_s = exact_r.copy()
    for m in drawn.values():
        fixed_s |= m
    hashes = {"r": sha(np.packbits(exact_r).tobytes()), "s": sha(np.packbits(fixed_s).tobytes())}
    if hashes != native["scoring_masks"][str(t)]:
        raise ValueError("Original judge R/S changed")
    canvases = []
    written_masks = []
    for scale in (0, *SCALES):
        canvas = fill.copy()
        written = np.zeros_like(exact_r)
        for player in order:
            i = native["players"].index(player)
            u = units[i]
            g = generated[(u["name"], "match5")]
            times = list(g["t"])
            if t not in times:
                continue
            if scale:
                reduced = mk.reduce(native_masks[t:t+1, i].astype(np.uint8) * OBJECT, scale // 2)
                m = mk.upsample(reduced, (1920, 1080))[0] > 0
            else:
                m = regions[t].get(player, np.zeros_like(exact_r))
            written |= k6b.paste(canvas, g["crops"][times.index(t)], maps[i][t], m)
        canvases.append(canvas)
        written_masks.append(written)
    lp = load_lpips(backbones[0], "cpu")
    di = load_dists(backbones[1], "cpu")
    if any(next(n.parameters()).is_cuda for n in (lp, di)):
        raise ValueError("CPU-only judge violated")
    ys, xs = np.nonzero(fixed_s)
    box = (slice(max(0, ys.min()-k6.DISTS_PAD), min(1080, ys.max()+1+k6.DISTS_PAD)),
           slice(max(0, xs.min()-k6.DISTS_PAD), min(1920, xs.max()+1+k6.DISTS_PAD)))
    began = cpu()
    lps = lpips_reduce(lp, canvases, np.asarray(truth[t]), [[fixed_s, written] for written in written_masks], "cpu")
    dis = dists_crops(di, [im[box] for im in canvases], np.asarray(truth[t])[box], fixed_s[box], "cpu")
    seconds = cpu()-began
    old_row = json.loads((control / "window.json").read_text())
    idx = old_row["variants"].index("gen-match5")
    with np.load(control / "scores.npz") as old:
        old_control = {"lpips": float(old["lpips"][idx, t]), "dists": float(old["dists"][idx, t])}
    differences = {"lpips": abs(lps[0][0]-old_control["lpips"]), "dists": abs(dis[0]-old_control["dists"])}
    if any(d > 1e-4 for d in differences.values()):
        raise ValueError(f"CPU original composite control mismatch: {differences}")
    rows = [{"scale": scale, "lpips": lps[i][0], "dists": dis[i], "composite_sha256": array_sha(canvases[i])}
            for i, scale in enumerate((0, *SCALES))]
    np.savez_compressed(target / (safe(row["window"])+"-pilot.npz"), scales=np.array((0, *SCALES)),
                        composites=np.stack(canvases), truth=np.asarray(truth[t]), r=exact_r, s=fixed_s)
    return {"window": row["window"], "frame": t, "scoring_masks": hashes, "rows": rows,
            "control_differences": differences, "control_gpu_scores": old_control, "metric_cpu_seconds": seconds, "projected_956_frames_cpu_seconds": seconds*956,
            "lpips": lpips_record(backbones[0]), "dists_backbone_sha256": sha(backbones[1].read_bytes()),
            "converted": converted, "source_sha256": source_record["sha256"], "torch": torch.__version__,
            "all_models_cpu": True, "background": "original exact-R Telea, unchanged between variants"}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnostic", type=Path)
    p.add_argument("--transport", type=Path)
    p.add_argument("--anchor", type=Path, action="append", default=[])
    p.add_argument("--generated", type=Path, action="append", default=[])
    p.add_argument("--source", type=Path, action="append", default=[])
    p.add_argument("--manifest", type=Path)
    p.add_argument("--control", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--backbone", type=Path, action="append", default=[])
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--validate", action="store_true")
    a = p.parse_args()
    stage = Path(os.environ["PS_STAGE_DIR"])
    if a.validate:
        result = json.loads((stage / "matte.json").read_text())
        checks = {"cpu_only": result["cuda_visible"] == "", "windows": len(result["rows"]) == result["expected_windows"], "two_actual_metric_pilots": len(result["pilots"]) == 2}
        with tarfile.open(stage / "published.tar") as tar:
            for row in result["rows"]:
                for scale, setting in row["settings"].items():
                    for candidate in setting["candidates"]:
                        f = tar.extractfile("publish/"+candidate["file"])
                        if f is None:
                            raise ValueError("Missing packet")
                        blob = f.read()
                        meta, planes = decode(blob)
                        checks[candidate["file"]] = sha(blob) == candidate["sha256"] and array_sha(planes) == setting["reduced_sha256"] and meta["units"] == row["players"] and planes.shape[0] == 240 and int(scale) == meta["scale"]
        for pilot in result["pilots"]:
            checks[pilot["window"]+":finite_actual_metrics"] = all(np.isfinite(r[m]) for r in pilot["rows"] for m in ("lpips", "dists")) and pilot["all_models_cpu"]
        write(Path(os.environ["PS_VALIDATION_PATH"]), {"passed": all(checks.values()), "checks": checks})
        return 0 if all(checks.values()) else 1
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Explicit CPU-only visibility required")
    import cv2
    cv2.setNumThreads(4)
    began = time.monotonic()
    resource.setrlimit(resource.RLIMIT_CPU, (1600, 1650))
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    publish = scratch / "publish"
    publish.mkdir()
    with tarfile.open(a.transport) as tar:
        tar.extractall(scratch / "transport", filter="data")
    packets = scratch / "transport/packets"
    anchors = {}
    for i, archive in enumerate(a.anchor):
        root = scratch / f"anchor{i}"
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter="data")
        for q in root.glob("publish/*/window.json"):
            native = json.loads(q.read_text())
            anchors[native["id"]] = q.parent
    control_root = scratch / "controls"
    with tarfile.open(a.control) as tar:
        tar.extractall(control_root, filter="data")
    from experiments.tennis.k6b import load_generated
    generated = load_generated([str(t) for t in a.generated], scratch)
    from experiments.tennis.k6b_rate_budget import DIAGNOSTIC_SHA
    if sha(a.diagnostic.read_bytes()) != DIAGNOSTIC_SHA:
        raise ValueError("Diagnostic changed")
    diagnostic = json.loads(a.diagnostic.read_text())
    chosen = ordered_rows(diagnostic["rows"])[:a.max_windows or None]
    manifest = json.loads(a.manifest.read_text())
    source_records = {c["id"]: c["staged"] for c in manifest["clips"]}
    sources_by_sha = {sha(path.read_bytes()): path for path in a.source}
    result = {"rows": [], "pilots": [], "expected_windows": len(chosen), "cuda_visible": os.environ["CUDA_VISIBLE_DEVICES"], "gpu_hours": 0,
        "input_sha256": {str(q): sha(q.read_bytes()) for q in [a.diagnostic, a.transport, a.manifest, a.control, *a.anchor, *a.generated]},
        "code_sha256": sha(Path(__file__).read_bytes()), "numpy": np.__version__, "opencv": cv2.__version__}
    if a.resume:
        with tarfile.open(a.resume) as tar:
            tar.extractall(scratch, filter="data")
        prior = json.loads((publish / "matte.json").read_text())
        for row in prior["rows"]:
            for setting in row["settings"].values():
                for candidate in setting["candidates"]:
                    blob = (publish / candidate["file"]).read_bytes()
                    _, planes = decode(blob)
                    if sha(blob) != candidate["sha256"] or array_sha(planes) != setting["reduced_sha256"]:
                        raise ValueError("Resume packet differs from preserved checkpoint")
        result["rows"] = prior["rows"]
        result["pilots"] = prior["pilots"]
        result["resumed_archive_sha256"] = sha(a.resume.read_bytes())
        result["prior_checkpoint_cpu_seconds"] = prior["cpu_seconds_self_and_children"]
    def checkpoint() -> None:
        result["cpu_seconds_self_and_children"] = cpu()
        result["wall_seconds"] = time.monotonic()-began
        write(stage / "matte.json", result)
        write(publish / "matte.json", result)
        with tarfile.open(stage / "published.tar", "w") as tar:
            tar.add(publish, arcname="publish")
    for i, row in enumerate(chosen):
        if any(r["window"] == row["window"] for r in result["rows"]):
            continue
        native = json.loads((anchors[row["window"]] / "window.json").read_text())
        result["rows"].append(rates(row, packets, publish, native))
        if row["window"] in PILOT_WINDOWS:
            source_record = source_records[row["window"].split("@")[0]]
            result["pilots"].append(metric_pilot(row, packets, anchors[row["window"]], generated, sources_by_sha[source_record["sha256"]],
                source_records[row["window"].split("@")[0]], control_root / "publish" / row["window"].replace("/", "__"), a.backbone, scratch, publish))
        checkpoint()
        from experiments.jobs.monitor import publish_progress
        publish_progress(str(stage), i+1)
        print(json.dumps({"completed_windows": i+1, "cpu_seconds": cpu()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
