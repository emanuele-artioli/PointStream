"""Score the sealed Sydney review samples without downloading a full native replay."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean
import time

import cv2
import numpy as np
import torch

from score import (
    ALEX_SHA256,
    EXPECTED,
    WEIGHTS,
    aggregate,
    automatic_regions,
    crop,
    image_psnr,
    load_rgb,
    quality,
    read_review,
    region_scores,
    sha,
)

from experiments.tennis.med_mixed_rung import (
    SYDNEY_REVIEW_TICKS,
    read_review_samples,
    review_receipt,
)

SOURCE_SHA = "8ebca8b693202d8bf7afaee515dde4d723a9fb83250115e6eff080038b0bbf50"
CACHE_SHA = "1f191ca3d078c258c6252003d746b6077c268bb98cff7260c4c960cab5a8d569"


def bind_arm(directory, expected_receipt_sha):
    path = directory / "rung.json"
    if sha(path) != expected_receipt_sha:
        raise ValueError("rung receipt differs from the validated fleet digest")
    rung = json.loads(path.read_text())
    proof = rung["isolation"]
    if (
        rung["ticks"] != list(range(240))
        or [row["tick"] for row in rung["rows"]] != rung["ticks"]
        or proof["interfaces"] != ["lo"]
        or not proof["mutation_denied"]
        or len(proof["denials"]) < 4
        or rung["wire"]["cache_sha256"] != CACHE_SHA
    ):
        raise ValueError("expected a sealed source-denied Sydney display")
    report = review_receipt(directory, rung)
    if tuple(row["tick"] for row in report["rows"]) != SYDNEY_REVIEW_TICKS:
        raise ValueError("review samples differ from the frozen Sydney ticks")
    return rung, report


def run(args):
    start = time.process_time()
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    review = read_review(args.review)
    if (
        sha(args.ball) != review["official_ball_sha256"]
        or review["source_video_sha256"] != SOURCE_SHA
        or review["source_video_fps"] != [60, 1]
        or tuple(row["frame_index"] for row in review["frames"]) != SYDNEY_REVIEW_TICKS
    ):
        raise ValueError("review inputs differ from the selected Sydney rally")
    if sha(WEIGHTS) != EXPECTED or sha(args.alex) != ALEX_SHA256:
        raise ValueError("metric backbone digest differs from the main comparison")
    with args.ball.open(encoding="utf-8-sig") as f:
        balls = {int(row["Frame"]): row for row in csv.DictReader(f)}
    if not 1 <= len(args.rung) <= 3:
        raise ValueError("score one to three Sydney replays")
    arms = [(Path(directory), expected) for directory, expected in args.rung]
    manifests = [bind_arm(directory, expected) for directory, expected in arms]
    streams = [
        read_review_samples(directory, report)
        for (directory, _), (_, report) in zip(arms, manifests, strict=True)
    ]
    dists = quality.load_dists(WEIGHTS, "cpu")
    lpips = quality.load_lpips(args.alex, "cpu")
    setup = time.process_time() - start
    scores: list[list[dict]] = [[] for _ in arms]
    for i, source in enumerate(review["frames"]):
        tick = source["frame_index"]
        original_path = args.review / source["source"]
        if sha(original_path) != source["source_sha256"]:
            raise ValueError("frozen source pixels changed")
        original = load_rgb(original_path)
        supports = list(automatic_regions(args.review, source, balls[tick]))
        if len(supports) != 3 or supports[-1][0] != "ball":
            raise ValueError("expected the two registered player regions and visible ball")
        foreground = np.zeros(original.shape[:2], dtype=bool)
        for _, (x0, y0, x1, y1), mask in supports:
            foreground[y0:y1, x0:x1] |= mask
        for stream, rows in zip(streams, scores, strict=True):
            sampled_tick, display = next(stream)
            if sampled_tick != tick or display.shape != original.shape:
                raise ValueError("sample and source ticks or dimensions differ")
            values = {
                name: region_scores(crop(original, rect), crop(display, rect), mask, dists, lpips)
                for name, rect, mask in supports
            }
            players = aggregate([r for name, r in values.items() if name != "ball"])
            rows.append(
                {
                    "tick": tick,
                    "source_png_sha256": source["source_sha256"],
                    "regions": values,
                    "player": players,
                    "ball": values["ball"],
                    "foreground": aggregate([players, values["ball"]]),
                    "background_psnr": image_psnr(original, display, ~foreground),
                    "whole_psnr": image_psnr(original, display),
                }
            )
            assert all(math.isfinite(n) for r in values.values() for n in r.values())
        elapsed = time.process_time() - start
        if i == 0 and not args.smoke:
            forecast = setup + (elapsed - setup) * 18 + 20
            if forecast > 300:
                raise ValueError(f"Sydney spatial score forecast {forecast:.1f}s exceeds 300s")
        if elapsed > 300:
            raise ValueError("Sydney spatial scoring exceeded the 300s CPU budget")
        if args.smoke:
            break
    if not args.smoke and any(next(stream, None) is not None for stream in streams):
        raise ValueError("unexpected frames after the frozen review")
    result = {
        "source_sha256": SOURCE_SHA,
        "review_sha256": review["review_sha256"],
        "official_ball_sha256": review["official_ball_sha256"],
        "lpips": quality.lpips_record(args.alex),
        "dists_backbone_sha256": EXPECTED,
        "sampled_ticks": list(SYDNEY_REVIEW_TICKS[:1] if args.smoke else SYDNEY_REVIEW_TICKS),
        "cpu_seconds": time.process_time() - start,
        "arms": [
            {
                "rung_sha256": report["rung_sha256"],
                "full_frames_sha256": report["frames_sha256"],
                "review_samples_sha256": report["sample_sha256"],
                "joint_sha256": rung["wire"]["joint_sha256"],
                "coded_size": rung["coded_size"],
                "warm_kbps": rung["wire"]["joint_bytes"] * 0.002,
                "cold_kbps": (rung["wire"]["joint_bytes"] + rung["wire"]["cache_bytes"]) * 0.002,
                "means": {
                    **{
                        k: aggregate([row[k] for row in rows])
                        for k in ("foreground", "player", "ball")
                    },
                    "background_psnr": mean(row["background_psnr"] for row in rows),
                    "whole_psnr": mean(row["whole_psnr"] for row in rows),
                },
                "rows": rows,
            }
            for (rung, report), rows in zip(manifests, scores, strict=True)
        ],
    }
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review", required=True, type=Path)
    p.add_argument("--ball", required=True, type=Path)
    p.add_argument("--alex", required=True, type=Path)
    p.add_argument(
        "--rung", action="append", nargs=2, required=True, metavar=("DIR", "VALIDATED_RUNG_SHA256")
    )
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())
