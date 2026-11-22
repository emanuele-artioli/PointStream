"""Re-score registered, price-matched receiver outputs on identical source frames and supports."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import sys
import time

import cv2
import numpy as np
import torch
from torchmetrics.functional.image import multiscale_structural_similarity_index_measure

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "shanghai-joint-review-20261122"
sys.path.insert(0, str(PRIOR))
from score_receivers import (  # noqa: E402
    EXPECTED,
    WEIGHTS,
    automatic_regions,
    crop,
    load_rgb,
    quality,
    read_review,
    sha,
    tensor,
)

CASES = {
    "shanghai": {
        "directory": "shanghai-joint-review-20261122",
        "joint": "joint-crf50",
        "arms": {
            "joint-crf50": "joint-50",
            "ordinary-640-warm": "ordinary-640-warm",
            "roi-640-warm": "roi-640-warm",
            "dcvc-480-q34": "dcvc-q34",
        },
        "scope": "The four fixed physical Shanghai CRF50 comparison arms, at the original twelve prespecified instants",
    },
    "sydney": {
        "directory": "sydney-joint-foreground-20261122",
        "joint": "joint-crf70",
        "arms": {
            "joint-crf70": "joint-70",
            "ordinary-320-warm": "ordinary-320-warm",
            "dcvc-480-q26": "dcvc-480-q26",
        },
        "scope": "Three physical Sydney receivers at the cached 12,890-byte ceiling on the twelve fixed source ticks. The smallest measured effective ROI stream exceeds the ceiling.",
    },
}
ALEX_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"


def image_psnr(source, decoded, mask=None):
    delta = source.astype(np.float32) - decoded.astype(np.float32)
    mse = np.mean(np.square(delta if mask is None else delta[mask]), dtype=np.float64)
    return 10 * math.log10(255**2 / mse) if mse else math.inf


def ssim_maps(x, y):
    def gaussian(value):
        return cv2.GaussianBlur(value, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT_101)

    mx, my = gaussian(x), gaussian(y)
    xx = np.maximum(gaussian(x**2) - mx**2, 0)
    yy = np.maximum(gaussian(y**2) - my**2, 0)
    xy = gaussian(x * y) - mx * my
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    contrast = (2 * xy + c2) / (xx + yy + c2)
    return ((2 * mx * my + c1) / (mx**2 + my**2 + c1)) * contrast, contrast


def luma_ssim(source, decoded, mask=None):
    # The local SSIM window is evaluated before restricting its center to the source mask.
    x, y = [cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float64) for rgb in (source, decoded)]
    similarity, _ = ssim_maps(x, y)
    return float(np.mean(similarity if mask is None else similarity[mask], dtype=np.float64))


def rgb_ms_ssim(source, decoded):
    def downsample(image):
        h, w = image.shape[:2]
        return cv2.resize(
            image[: h - h % 2, : w - w % 2], (w // 2, h // 2), interpolation=cv2.INTER_AREA
        )

    x, y = source.astype(np.float64), decoded.astype(np.float64)
    weights = (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)
    scores = []
    for index, weight in enumerate(weights):
        similarity, contrast = ssim_maps(x, y)
        measure = similarity if index == len(weights) - 1 else contrast[5:-5, 5:-5]
        scores.append(max(0.0, float(np.mean(measure, dtype=np.float64))) ** weight)
        x, y = downsample(x), downsample(y)
    return math.prod(scores)


@torch.inference_mode()
def region_scores(original, display, mask, dists, lpips):
    source, decoded = tensor(original), tensor(display)
    support = torch.from_numpy(np.ascontiguousarray(mask))[None, None]
    lp = lpips(decoded * 2 - 1, source * 2 - 1)
    lp = torch.nn.functional.interpolate(lp, size=mask.shape, mode="bilinear", align_corners=False)
    return {
        "dists": float(quality.masked_dists(dists, decoded, source, support).item()),
        "lpips": float(lp[support].mean().item()),
        "luma_ssim": luma_ssim(original, display, mask),
        "rgb_psnr": image_psnr(original, display, mask),
    }


@torch.inference_mode()
def whole_scores(original, display):
    return {
        "luma_ssim": luma_ssim(original, display),
        "rgb_psnr": image_psnr(original, display),
        "rgb_ms_ssim": rgb_ms_ssim(original, display),
    }


def aggregate(rows):
    return {k: mean(r[k] for r in rows) for k in rows[0]}


def load_arm(root, tag, review, cached_path):
    path = root / tag / "arm.json"
    record = json.loads(path.read_text())
    cached = json.loads(cached_path.read_text())
    assert record["source_sha256"] == review["source_video_sha256"] == cached["source_sha256"]
    assert record["fps"] == review["source_video_fps"] and record["source_ticks"] == 240
    assert (
        record["bytes"] == cached["bytes"]
        and record["bitstream_sha256"] == cached["bitstream_sha256"]
    )
    recorded = {(r["tick"], r["region"]): r["dists"] for r in cached["scores"]}
    rows = {r["tick"]: r for r in record["frames"]}
    assert sorted(rows) == [r["frame_index"] for r in review["frames"]]
    return path, record, rows, recorded, cached["kbps"]


def sha_array(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def run(args):
    started = time.process_time()
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    review = read_review(args.review)
    assert sha(args.ball) == review["official_ball_sha256"]
    assert sha(WEIGHTS) == EXPECTED and sha(args.alex) == ALEX_SHA256
    with args.ball.open(encoding="utf-8-sig") as f:
        ball_rows = {int(row["Frame"]): row for row in csv.DictReader(f)}
    case = CASES[args.case]
    receipt_dir = args.receipts or HERE.parent / case["directory"]
    arms = {
        tag: load_arm(args.arms, tag, review, receipt_dir / f"automatic-{receipt}.json")
        for tag, receipt in case["arms"].items()
    }
    dists, lpips = quality.load_dists(WEIGHTS, "cpu"), quality.load_lpips(args.alex, "cpu")
    setup_seconds = time.process_time() - started
    results = {tag: [] for tag in arms}
    self_control = None
    validation_seconds = 0.0
    for i, source in enumerate(review["frames"][: args.ticks]):
        tick = source["frame_index"]
        original_path = args.review / source["source"]
        assert sha(original_path) == source["source_sha256"]
        original = load_rgb(original_path)
        supports = list(automatic_regions(args.review, source, ball_rows[tick]))
        assert len(supports) == 3 and supports[-1][0] == "ball"
        if i == 0:
            _, rect, mask = supports[-1]
            target = crop(original, rect)
            self_control = region_scores(target, target, mask, dists, lpips)
            assert abs(self_control["dists"]) < 1e-5 and abs(self_control["lpips"]) < 1e-9
            assert abs(self_control["luma_ssim"] - 1) < 1e-7 and math.isinf(
                self_control["rgb_psnr"]
            )
            self_control["rgb_ms_ssim"] = whole_scores(original, original)["rgb_ms_ssim"]
            assert abs(self_control["rgb_ms_ssim"] - 1) < 1e-5, self_control
            self_control["rgb_psnr"] = "infinite"
        for tag, (arm_path, arm, frames, established, _) in arms.items():
            frame = frames[tick]
            path = arm_path.parent / frame["path"]
            assert sha(path) == frame["png_sha256"]
            display = load_rgb(path)
            assert display.shape == original.shape == (1080, 1920, 3)
            assert sha_array(display) == frame["rgb_sha256"]
            values = {}
            for name, rect, mask in supports:
                pair = region_scores(crop(original, rect), crop(display, rect), mask, dists, lpips)
                assert all(math.isfinite(n) for n in pair.values())
                assert abs(pair["dists"] - established[tick, name]) < 0.00015
                values[name] = pair
            players = aggregate([r for name, r in values.items() if name != "ball"])
            whole = whole_scores(original, display)
            assert all(math.isfinite(x) for x in whole.values())
            if i == 0 and tag == case["joint"]:
                before = time.process_time()
                with torch.inference_mode():
                    reference = float(
                        multiscale_structural_similarity_index_measure(
                            tensor(display), tensor(original), data_range=1
                        ).item()
                    )
                difference = abs(reference - whole["rgb_ms_ssim"])
                assert difference < 0.00015, (reference, whole["rgb_ms_ssim"])
                assert self_control is not None
                self_control["ms_ssim_implementation_difference"] = difference
                validation_seconds = time.process_time() - before
            results[tag].append(
                {
                    "tick": tick,
                    "source_png_sha256": source["source_sha256"],
                    "decoded_rgb_sha256": frame["rgb_sha256"],
                    "regions": values,
                    "foreground": aggregate([players, values["ball"]]),
                    "player": players,
                    "ball": values["ball"],
                    "whole": whole,
                }
            )
        consumed = time.process_time() - started
        if i == 0:
            forecast = (
                setup_seconds
                + validation_seconds
                + 1.5 * (consumed - setup_seconds - validation_seconds) * 12
                + 20
            )
            assert forecast < 300, f"CPU forecast {forecast:.1f} s exceeds cap"
        assert consumed < 300
    domains = ("player", "ball", "foreground", "whole")
    record = {
        "scope": case["scope"],
        "source_sha256": review["source_video_sha256"],
        "review_sha256": review["review_sha256"],
        "official_ball_sha256": review["official_ball_sha256"],
        "source_ticks": 240,
        "sampled_ticks": args.ticks,
        "source_fps": review["source_video_fps"],
        "lpips_model": quality.lpips_record(args.alex)
        | {"pooling": "crop spatial map, bilinear to native crop, mean over source mask"},
        "dists_backbone_sha256": EXPECTED,
        "runner_sha256": sha(Path(__file__)),
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "self_control": self_control,
        "forecast_seconds": forecast,
        "cpu_seconds": time.process_time() - started,
        "arms": {
            tag: {
                "bytes": arms[tag][1]["bytes"],
                "kbps": arms[tag][4],
                "bitstream_sha256": arms[tag][1]["bitstream_sha256"],
                "rescore_manifest_sha256": sha(arms[tag][0]),
                "means": {k: aggregate([r[k] for r in rows]) for k in domains},
                "rows": rows,
            }
            for tag, rows in results.items()
        },
    }
    with args.out.open("x") as file:
        json.dump(record, file, indent=2, allow_nan=False)
        file.write("\n")
    for tag, arm in record["arms"].items():
        print(tag, arm["means"]["foreground"], "whole", arm["means"]["whole"])
    print(f"CPU forecast {forecast:.1f}s; consumed {record['cpu_seconds']:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("review", "ball", "arms", "alex", "out"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--case", choices=tuple(CASES), default="shanghai")
    parser.add_argument("--receipts", type=Path)
    parser.add_argument("--ticks", type=int, choices=(1, 12), required=True)
    run(parser.parse_args())
