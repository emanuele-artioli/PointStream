"""Bounded GPU judging of frozen saved crops with decoder-verified reduced mattes."""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
from typing import Any

import numpy as np

from experiments.tennis.k4 import safe
from experiments.tennis.k6b_matte_cpu import SCALES, box_maps, decode
from experiments.tennis.k6b_rate_budget import DIAGNOSTIC_SHA, comparison
from experiments.tennis.k6b_rate_streams import decode_keypoints, decode_masks
from experiments.tennis.k6b_rate_transport import array_sha, sha, write
from src.pointstream import mask as mk
from src.pointstream.composite import OBJECT

VARIANTS = (0, *SCALES)


def extract(archives: list[Path], parent: Path) -> dict[str, Path]:
    found = {}
    for i, archive in enumerate(archives):
        root = parent / str(i)
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter="data")
        for q in root.glob("publish/*/window.json"):
            row = json.loads(q.read_text())
            if row["id"] in found:
                raise ValueError("Duplicate source window")
            for name, digest in row["files"].items():
                if sha((q.parent / name).read_bytes()) != digest:
                    raise ValueError("Source member hash mismatch")
            found[row["id"]] = q.parent
    return found


def control_error(current: np.ndarray, expected: np.ndarray) -> float:
    if current.shape != expected.shape or not np.isfinite(current).all() or not np.isfinite(expected).all():
        raise ValueError("Missing/nonfinite original control scores")
    difference = float(np.max(np.abs(current-expected)))
    if difference > 1e-4:
        raise ValueError(f"Original control exceeds frozen tolerance: {difference}")
    return difference


def judge_window(wid: str, anchor: Path, control: Path, packets: Path, matte: Path,
                 matte_row: dict[str, Any], generated: dict[Any, Any], source: Path, source_sha: str,
                 nets: dict[str, Any], scratch: Path, publish: Path, max_judged: int) -> dict[str, Any]:
    import cv2
    from experiments.tennis import k6, k6b
    from experiments.tennis.k4b_features import convert
    from experiments.visor.h3 import rgb_memmap
    from experiments.visor.h3b import lpips_reduce, dists_crops
    from src.segmentation.masks import decompress

    started = time.monotonic()
    w = k6b.k6_window(anchor)
    native = w["row"]
    requested = native["judged"][:max_judged or None]
    kp_meta, units = decode_keypoints((packets / safe(wid) / "keypoints.kps").read_bytes())
    if kp_meta["window"] != wid or len(units) != len(native["players"]):
        raise ValueError("Conditioning window/players changed")
    boxes = box_maps(units)
    _, source_masks = decode_masks((packets / safe(wid) / "exact-rle.matte").read_bytes())
    if source_masks.shape != (240, len(units), 540, 960):
        raise ValueError("Original mask frame/player/grid shape changed")
    reduced = {}
    selected_packets = {}
    for scale in SCALES:
        selected = matte_row["settings"][str(scale)]["selected"]
        blob = (matte / selected["file"]).read_bytes()
        meta, small = decode(blob)
        if sha(blob) != selected["sha256"] or len(blob) != selected["bytes"] or meta["units"] != [u["name"] for u in units] or meta["scale"] != scale or array_sha(small) != matte_row["settings"][str(scale)]["reduced_sha256"]:
            raise ValueError("Actual decoded matte differs from frozen CPU packet")
        if small.shape != (240, len(units), -(-1080 // scale), -(-1920 // scale)):
            raise ValueError("Decoded matte frame/player/grid shape changed")
        reduced[scale] = small
        selected_packets[str(scale)] = selected
    if sha(source.read_bytes()) != source_sha:
        raise ValueError("Original clip identity mismatch")
    crops = {}
    for i, unit in enumerate(units):
        g = generated[(unit["name"], "match5")]
        expected_times = sorted(set(unit["t"].tolist()) & set(native["judged"]))
        if g["t"].tolist() != expected_times or g["crops"].shape != (len(expected_times), 784, 512, 3):
            raise ValueError("Saved generated RGB coverage changed")
        crops[i] = dict(zip(g["t"].tolist(), g["crops"]))
    work = scratch / safe(wid)
    work.mkdir()
    converted = convert("ffmpeg", {"id": wid, "start": native["start"], "frames": 240}, source, work / "source.yuv")
    truth = rgb_memmap(work / "source.yuv", work / "truth.npy", 240, 1920, 1080, False, "cuda")
    regions = [{p: cv2.resize(source_masks[t, i].astype(np.uint8), (1920, 1080), interpolation=cv2.INTER_NEAREST).astype(bool)
                for i, p in enumerate(native["players"]) if source_masks[t, i].any()} for t in range(240)]
    heights = {(p, t): float(np.ptp(np.nonzero(r[p].any(1))[0]) + 1) if p in r and r[p].any() else 1.0
               for t, r in enumerate(regions) for p in native["players"]}
    decoded_all = {key: {(n.rsplit("@", 1)[0], int(n.rsplit("@", 1)[1])): np.asarray(v, np.float64)
                         for n, v in d.items()} for key, d in json.loads(decompress((anchor / "keypoints.json.zst").read_bytes()))["decoded"].items()}
    old_row = json.loads((control / "window.json").read_text())
    old_index = old_row["variants"].index("gen-match5")
    with np.load(control / "scores.npz") as a:
        expected_control = np.stack([a[m][old_index] for m in ("lpips", "dists")])
    dest = publish / safe(wid)
    dest.mkdir()
    arrays = {m: np.full((4, 240), np.nan) for m in ("lpips", "dists", "outside_r_sse", "sse")}
    hashes = {}
    composite_hashes = {}
    row: dict[str, Any] = {"id": wid, "scales": list(VARIANTS), "expected_judged": requested,
        "original_judged": native["judged"], "judged": [], "scored": native["scored"],
        "selected_packets": selected_packets, "source_sha256": source_sha, "converted": converted,
        "complete_window": max_judged == 0, "condition_boxes_sha256": [array_sha(np.stack(list(b.values()))) for b in boxes]}
    for n, t in enumerate(requested):
        truth_t = np.array(truth[t])
        r = np.logical_or.reduce([*regions[t].values(), np.zeros((1080, 1920), bool)])
        fill = k6.telea_fill(truth_t, r)
        _, drawn = k6b.k6_oracle(t, w, k6.variant_names()[-9:], truth, regions, fill,
            kp_meta["compositing_order"], native["scored"], heights, decoded_all,
            max(1, round(k6.REF_GAP_S * float(Fraction(native["fps"])))))
        s = r.copy()
        for m in drawn.values():
            s |= m
        hashes[str(t)] = {"r": sha(np.packbits(r).tobytes()), "s": sha(np.packbits(s).tobytes())}
        if hashes[str(t)] != native["scoring_masks"][str(t)] or hashes[str(t)] != old_row["scoring_masks"][str(t)]:
            raise ValueError("Frozen R/S judging regions changed")
        frames, supports = [], []
        for scale in VARIANTS:
            canvas, written = fill.copy(), np.zeros_like(r)
            for player in kp_meta["compositing_order"]:
                i = native["players"].index(player)
                if t not in crops[i]:
                    continue  # identical native missing-pose/crop fallback
                mask = regions[t].get(player, np.zeros_like(r)) if not scale else mk.upsample(reduced[scale][t:t+1, i].astype(np.uint8)*OBJECT, (1920, 1080))[0] > 0
                written |= k6b.paste(canvas, crops[i][t], boxes[i][t], mask)
            frames.append(canvas)
            supports.append(written)
        lp = lpips_reduce(nets["lpips"], frames, truth_t, [[s, m] for m in supports], "cuda")
        ys, xs = np.nonzero(s)
        crop = (slice(max(0, ys.min()-k6.DISTS_PAD), min(1080, ys.max()+1+k6.DISTS_PAD)),
                slice(max(0, xs.min()-k6.DISTS_PAD), min(1920, xs.max()+1+k6.DISTS_PAD)))
        di = dists_crops(nets["dists"], [im[crop] for im in frames], truth_t[crop], s[crop], "cuda")
        values = np.array([[v[0] for v in lp], di])
        control_error(values[:, 0], expected_control[:, t])
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite actual neural scores")
        arrays["lpips"][:, t], arrays["dists"][:, t] = values
        for i, image in enumerate(frames):
            error = (image.astype(np.float64)-truth_t.astype(np.float64))**2
            arrays["sse"][i, t] = error[s].sum()
            arrays["outside_r_sse"][i, t] = error[~r].sum()
        composite_hashes[str(t)] = [array_sha(im) for im in frames]
        row["judged"].append(t)
        if n % 20 == 0 or n+1 == len(requested):
            np.savez_compressed(dest / "scores.npz", **arrays, judged=np.array(row["judged"]))
            write(dest / "partial.json", {"id": wid, "finished_frames": row["judged"], "expected_frames": requested})
    row["control_max_abs_difference"] = control_error(np.stack([arrays[m][0, requested] for m in ("lpips", "dists")]), expected_control[:, requested])
    row["means"] = {str(scale): {m: float(arrays[m][i, requested].mean()) for m in ("lpips", "dists")} for i, scale in enumerate(VARIANTS)}
    row["scoring_masks"] = hashes
    row["composite_hashes"] = composite_hashes
    row["seconds"] = time.monotonic()-started
    row["files"] = {q.name: sha(q.read_bytes()) for q in dest.iterdir() if q.is_file()}
    write(dest / "window.json", row)
    del truth
    shutil.rmtree(work)
    return row


def assess(diagnostic: dict[str, Any], transport: dict[str, Any], matte: dict[str, Any], measured: list[dict[str, Any]]) -> dict[str, Any]:
    base = {r["window"]: r for r in diagnostic["rows"]}
    wire = {r["window"]: r for r in transport["rows"]}
    masks = {r["window"]: r for r in matte["rows"]}
    banks = {match: sum(r["references"]["png_bytes"] for r in wire.values() if r["match"] == match) for match in ("alcaraz_hurkacz", "medvedev_struff")}
    settings = {}
    for scale in SCALES:
        cells = []
        for row in measured:
            if row["judged"] != row["original_judged"]:
                raise ValueError("Partial smoke cannot decide cohort quality")
            original = base[row["id"]]
            kp = wire[row["id"]]["keypoints"]["bytes"]*8/original["registered_seconds"]/1000
            mask = masks[row["id"]]["settings"][str(scale)]["selected"]["bytes"]*8/original["registered_seconds"]/1000
            rate = kp + mask + (banks[original["match"]]+original["adapter_guider_bytes"])*8/7200/1000
            actual = {**original, "method": row["means"][str(scale)]}
            cells.append({"window": row["id"], "match": original["match"], "kbps": rate,
                "keypoint_kbps": kp, "matte_kbps": mask,
                "quality": actual["method"], "primary": {m: comparison(actual, m, rate) for m in ("lpips", "dists")},
                "base_charged": {m: comparison(actual, m, rate+original["base_checkpoint_bytes"]*8/7200/1000) for m in ("lpips", "dists")}})
        wins = sum(c["primary"][m]["won"] is True for c in cells for m in ("lpips", "dists"))
        unresolved = sum(c["primary"][m]["won"] is None for c in cells for m in ("lpips", "dists"))
        bound = wins+unresolved+2*(12-len(cells))
        settings[str(scale)] = {"rows": cells, "known_wins": wins, "unresolved": unresolved, "optimistic_final_wins_bound": bound, "can_reach_18": bound >= 18, "passes": wins >= 18}
    return {"measured_windows": len(measured), "unmeasured_windows": 12-len(measured), "scales": settings,
            "decision_made": any(s["passes"] for s in settings.values()) or all(not s["can_reach_18"] for s in settings.values())}


def validate(archive: Path, stage: str) -> dict[str, Any]:
    import tempfile
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="matte-judge-validate-") as d:
        root = Path(d)
        with tarfile.open(archive) as tar:
            tar.extractall(root, filter="data")
        data = json.loads((root / "publish/judge.json").read_text())
        checks = {"window_set": [r["id"] for r in data["rows"]] == data["requested"], "cuda": all(data["on_cuda"].values()),
                  "class": data["device"]["class_matches"], "memory": data["gpu_peak_gib"]*1024 <= data["gpu_memory_limit_mib"]}
        for row in data["rows"]:
            path = root / "publish" / safe(row["id"])
            checks[row["id"]+":hashes"] = all(sha((path/n).read_bytes()) == h for n, h in row["files"].items())
            checks[row["id"]+":frames"] = row["judged"] == row["expected_judged"] and (stage != "full" or row["judged"] == row["original_judged"])
            with np.load(path / "scores.npz") as a:
                checks[row["id"]+":scores"] = all(a[m].shape == (4, 240) and np.isfinite(a[m][:, row["judged"]]).all() and np.allclose(a[m][:, row["judged"]].mean(1), [row["means"][str(s)][m] for s in VARIANTS], atol=1e-12, rtol=0) for m in ("lpips", "dists"))
            checks[row["id"]+":control"] = row["control_max_abs_difference"] <= 1e-4
        return {"passed": all(checks.values()), "checks": checks}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("diagnostic", "transport", "matte", "manifest"):
        p.add_argument("--"+name, type=Path)
    for name in ("anchor", "control", "generated", "source", "backbone"):
        p.add_argument("--"+name, type=Path, action="append", default=[])
    p.add_argument("--windows", nargs="+", default=[])
    p.add_argument("--max-windows", type=int, default=0)
    p.add_argument("--max-judged", type=int, default=0)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--check-archive", type=Path)
    a = p.parse_args()
    stage = Path(os.environ.get("PS_STAGE_DIR", "/tmp"))
    if a.validate or a.check_archive:
        result = validate(a.check_archive or stage / "published.tar", os.environ.get("PS_STAGE", "full"))
        if a.check_archive:
            print(json.dumps(result))
        else:
            write(Path(os.environ["PS_VALIDATION_PATH"]), result)
        return 0 if result["passed"] else 1
    import cv2
    import torch
    from experiments.audit.env_smoke import device_record
    from experiments.tennis.k6b import load_generated
    from experiments.visor.h3b import load_dists
    from src.codecs.quality import load_lpips, lpips_record
    torch.set_num_threads(8)
    cv2.setNumThreads(8)
    if not torch.cuda.is_available():
        raise ValueError("GPU judging cannot fall back to CPU")
    device = device_record("RTX 6000 Ada")
    if not device["class_matches"]:
        raise ValueError("Unexpected GPU class")
    limit = int(os.environ["PS_GPU_MEMORY_LIMIT_MIB"])
    torch.cuda.set_per_process_memory_fraction(min(1., limit*2**20/torch.cuda.get_device_properties(0).total_memory))
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    publish = scratch / "publish"
    publish.mkdir()
    anchors = extract(a.anchor, scratch / "anchors")
    controls = extract(a.control, scratch / "controls")
    for name in ("transport", "matte"):
        with tarfile.open(getattr(a, name)) as tar:
            tar.extractall(scratch / name, filter="data")
    packet_root = scratch / "transport"
    if (packet_root / "publish").is_dir():
        packet_root /= "publish"
    matte_root = scratch / "matte/publish"
    matte = json.loads((matte_root / "matte.json").read_text())
    transport = json.loads((packet_root / "transport.json").read_text())
    mask_rows = {r["window"]: r for r in matte["rows"]}
    if sha(a.diagnostic.read_bytes()) != DIAGNOSTIC_SHA:
        raise ValueError("Frozen diagnostic changed")
    manifest = json.loads(a.manifest.read_text())
    sources = {sha(q.read_bytes()): q for q in a.source}
    expected_sources = {c["id"]: c["staged"]["sha256"] for c in manifest["clips"]}
    generated = load_generated([str(q) for q in a.generated], scratch)
    nets = {"lpips": load_lpips(a.backbone[0], "cuda"), "dists": load_dists(a.backbone[1], "cuda")}
    on_cuda = {n: next(net.parameters()).is_cuda for n, net in nets.items()}
    if not all(on_cuda.values()):
        raise ValueError("Metric network is not on CUDA")
    requested = a.windows[:a.max_windows or None]
    result = {"requested": requested, "rows": [], "device": device, "on_cuda": on_cuda,
        "gpu_memory_limit_mib": limit, "code_sha256": sha(Path(__file__).read_bytes()), "torch": torch.__version__,
        "lpips": lpips_record(a.backbone[0]), "dists_backbone_sha256": sha(a.backbone[1].read_bytes()),
        "inputs": {str(q): sha(q.read_bytes()) for q in [a.diagnostic, a.transport, a.matte, a.manifest, *a.anchor, *a.control, *a.generated, *a.source]}}
    started = time.monotonic()
    for i, wid in enumerate(requested):
        source_sha = expected_sources[wid.split("@")[0]]
        result["rows"].append(judge_window(wid, anchors[wid], controls[wid], packet_root / "packets", matte_root, mask_rows[wid], generated, sources[source_sha], source_sha, nets, scratch, publish, a.max_judged))
        result["gpu_peak_gib"] = torch.cuda.max_memory_allocated()/2**30
        result["seconds"] = time.monotonic()-started
        write(publish / "judge.json", result)
        write(stage / "judge.json", result)
        with tarfile.open(stage / "published.tar", "w") as tar:
            tar.add(publish, arcname="publish")
        from experiments.jobs.monitor import publish_progress
        publish_progress(str(stage), i+1)
        print(json.dumps({"completed_windows": i+1, "seconds": result["seconds"], "peak_gib": result["gpu_peak_gib"]}), flush=True)
    # Same accounting callable is available to the local CPU report; no smoke can decide the cohort.
    _ = transport
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
