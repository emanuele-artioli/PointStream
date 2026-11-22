"""Score reviewed native-resolution frames with the registered regional metrics."""
import argparse
import hashlib
import json
import math
import os
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

from experiments.codec_ladder.__main__ import standard_lpips_scalar
from experiments.visor.h5 import dists_regions
from src.codecs import quality
from src.pointstream.evaluation import RegionalFrame, weighted_region

SOURCE_SHA = "4c891777b33b10ba92fd94fa1a59ba360aacf175e7bfd67c382206058d59550a"
MASK_SHA = "16b4d488b41cd1f53434a258d37ea9baf5ba00e1e1c52465585393658fce7d11"
KEY_SHA = "c5c79d2b01ea4e2bfc70d53bef1b050f46fe1838cd9af27cad0bcda0cc4f8040"
ALEX_SHA = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
VGG_SHA = "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"
NAMES = ("full", "player1", "player2", "body", "nonbody", "body_disk4", "rest_disk4", "body_disk8", "rest_disk8")
ARMS = ("plain", "roi", "pointstream")


def file_sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_frame(archive, digest):
    data = archive.read(digest + ".png")
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError("Frame content does not match its name")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image.shape != (1080, 1920, 3):
        raise ValueError("Frame geometry differs")
    return image[:, :, ::-1].copy()


def run(args):
    import torch
    import torch.nn.functional as F

    started = time.monotonic()
    if torch.cuda.device_count() != 1 or len(os.sched_getaffinity(0)) != 1:
        raise ValueError("Expected one GPU and one CPU")
    torch.cuda.reset_peak_memory_stats()
    cv2.setNumThreads(1)
    supports = args.supports
    if (file_sha(args.frames) != SOURCE_SHA or file_sha(supports / "supports.npz") != MASK_SHA
            or file_sha(supports / "key.json") != KEY_SHA or file_sha(args.alex) != ALEX_SHA
            or file_sha(args.vgg) != VGG_SHA):
        raise ValueError("Input identity differs from the frozen sample")
    inputs = json.loads((supports / "key.json").read_text())
    index = {(arm, tick): digest for digest, (arm, tick) in inputs.items()}
    regions = np.load(supports / "supports.npz", allow_pickle=False)
    ticks, bodies, centers = regions["ticks"], regions["bodies"], regions["centers"]
    if (len(inputs) != 48 or bodies.shape != (12, 2, 1080, 1920)
            or centers.shape != (12, 2) or not 1 <= args.count <= 12):
        raise ValueError("Unexpected source supports")
    lpips = quality.load_lpips(args.alex, "cuda")
    dists = quality.load_dists(args.vgg, "cuda")
    if any(p.device.type != "cuda" for net in (lpips, dists) for p in net.parameters()):
        raise ValueError("A metric model left CUDA")
    fixed_seconds = time.monotonic() - started
    rows = []
    control = None
    score_start = time.monotonic()
    with zipfile.ZipFile(args.frames) as archive, torch.inference_mode():
        for i in range(args.count):
            tick = int(ticks[i])
            original = read_frame(archive, index["source", tick])
            x = torch.from_numpy(original).to("cuda").permute(2, 0, 1).unsqueeze(0).float() / 255
            body = bodies[i, 0] | bodies[i, 1]
            yy, xx = np.ogrid[:1080, :1920]
            cx, cy = centers[i]
            disk4 = (xx - cx) ** 2 + (yy - cy) ** 2 <= 16
            disk8 = (xx - cx) ** 2 + (yy - cy) ** 2 <= 64
            fg4, fg8 = body | disk4, body | disk8
            masks = [np.ones_like(body), bodies[i, 0], bodies[i, 1], body, ~body,
                     fg4, ~fg4, fg8, ~fg8]
            if any(not mask.any() for mask in masks):
                raise ValueError("An evaluation region is empty")
            tmasks = [torch.from_numpy(mask).to("cuda") for mask in masks]
            if control is None:
                self_dists = float(dists_regions(dists, x, x, [tmasks[0][None, None]])[0][0])
                self_lpips = float(lpips(x * 2 - 1, x * 2 - 1).mean().item())
                control = {"dists": self_dists, "lpips_spatial": self_lpips}
                if any(not math.isfinite(v) or abs(v) > 1e-5 for v in control.values()):
                    raise ValueError("Metric self-check failed")
            for arm in ARMS:
                recon = read_frame(archive, index[arm, tick])
                y = torch.from_numpy(recon).to("cuda").permute(2, 0, 1).unsqueeze(0).float() / 255
                lpips_standard = standard_lpips_scalar(lpips, x * 2 - 1, y * 2 - 1)
                spatial = lpips(x * 2 - 1, y * 2 - 1)
                spatial = F.interpolate(spatial, size=(1080, 1920), mode="bilinear", align_corners=False)[0, 0]
                lvalues = {name: float(spatial[mask].mean().item()) for name, mask in zip(NAMES, tmasks)}
                dvalues = dists_regions(dists, x, y, [mask[None, None] for mask in tmasks])
                values = {"lpips_standard": lpips_standard,
                          "lpips": lvalues,
                          "dists": {name: float(value[0]) for name, value in zip(NAMES, dvalues)}}
                sizes = {name: int(mask.sum()) for name, mask in zip(NAMES, masks)}
                for metric in ("dists", "lpips"):
                    values[metric]["weighted_body"] = weighted_region(
                        RegionalFrame(values[metric]["body"], values[metric]["nonbody"], sizes["body"], sizes["nonbody"]))
                    for radius in (4, 8):
                        fg, bg = f"body_disk{radius}", f"rest_disk{radius}"
                        values[metric][f"weighted_disk{radius}"] = weighted_region(
                            RegionalFrame(values[metric][fg], values[metric][bg], sizes[fg], sizes[bg]))
                flat = [lpips_standard, *values["lpips"].values(), *values["dists"].values()]
                if any(not math.isfinite(v) or v < 0 for v in flat):
                    raise ValueError("Metric is missing, negative or nonfinite")
                rows.append({"tick": tick, "arm": arm, "frame_sha256": index[arm, tick],
                             "source_sha256": index["source", tick], "region_pixels": sizes, **values})
    score_seconds = time.monotonic() - score_start
    means = {}
    for arm in ARMS:
        subset = [r for r in rows if r["arm"] == arm]
        means[arm] = {metric: {name: float(np.mean([r[metric][name] for r in subset]))
                               for name in subset[0][metric]} for metric in ("lpips", "dists")}
        means[arm]["lpips_standard"] = float(np.mean([r["lpips_standard"] for r in subset]))
    report = {"count": args.count, "ticks": [int(x) for x in ticks[:args.count]], "rows": rows, "means": means,
              "fixed_seconds": fixed_seconds, "score_seconds": score_seconds,
              "elapsed_seconds": time.monotonic() - started, "self_check": control,
              "cuda_device": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0)),
              "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
              "lpips_model": quality.lpips_record(args.alex), "dists_model": quality.dists_record(args.vgg),
              "frames_sha256": SOURCE_SHA, "supports_sha256": MASK_SHA, "key_sha256": KEY_SHA}
    publish = Path(os.environ["PS_SCRATCH_DIR"]) / "publish"
    publish.mkdir()
    data = json.dumps(report, indent=2, allow_nan=False) + "\n"
    (publish / "scores.json").write_text(data)
    (Path(os.environ["PS_STAGE_DIR"]) / "scores.json").write_text(data)


def validate():
    report = json.loads((Path(os.environ["PS_STAGE_DIR"]) / "scores.json").read_text())
    count = 1 if os.environ["PS_STAGE"] == "smoke" else 12
    checks = {"count": report["count"] == count, "rows": len(report["rows"]) == 3 * count,
              "gpu": report["capability"] == [8, 9], "models": report["lpips_model"]["backbone_sha256"] == ALEX_SHA and report["dists_model"]["backbone_sha256"] == VGG_SHA,
              "inputs": report["frames_sha256"] == SOURCE_SHA and report["supports_sha256"] == MASK_SHA and report["key_sha256"] == KEY_SHA,
              "self": all(abs(v) <= 1e-5 for v in report["self_check"].values()),
              "all_regions": all(set(row["dists"]) == set(NAMES) | {"weighted_body", "weighted_disk4", "weighted_disk8"} for row in report["rows"])}
    if count == 1:
        checks["forecast"] = report["fixed_seconds"] + 1.5 * report["score_seconds"] * 12 + 90 < 850
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps({"passed": all(checks.values()), "checks": checks}) + "\n")
    if not all(checks.values()):
        raise ValueError("Scoring validation failed: " + str(checks))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("frames", "supports", "alex", "vgg"):
        p.add_argument("--" + key, type=Path)
    p.add_argument("--count", type=int)
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    validate() if args.validate else run(args)
