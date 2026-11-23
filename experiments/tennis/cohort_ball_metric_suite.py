"""Re-score the frozen twelve-event matched pair on aligned native-display ball crops."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean
import time

import cv2
import numpy as np
import torch
from torchmetrics.functional.image.ssim import structural_similarity_index_measure

from experiments.tennis import cohort_ball_score as previous
from experiments.tennis.k5d_encoder import crop64
from experiments.visor.h3b import dists_crops
from src.codecs import quality, svtav1
from src.pointstream import ball

ALEX = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"


def tensor(rgb):
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].float() / 255


@torch.inference_mode()
def compare(rgb, target, net):
    x, y = tensor(rgb), tensor(target)
    lpips = float(net(x * 2 - 1, y * 2 - 1).mean().item())
    luma = [torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))[None, None].float()
            for img in (rgb, target)]
    ssim = float(structural_similarity_index_measure(*luma, data_range=255.0).item())
    mse = float(torch.mean((x - y) ** 2).item())
    return dict(lpips=lpips, luma_ssim=ssim, rgb_mse=mse)


def run(args):
    start = time.monotonic()
    stage = os.environ["PS_STAGE"]
    assert args.frames == {"smoke": 24, "full": 240}[stage]
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and len(os.sched_getaffinity(0)) == 1
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    meta = json.loads(args.metadata.read_text())
    primary = json.loads(args.score.read_text())
    rates, streams, paid = previous.load_bank(args.rate, meta)
    assert previous.sha(args.rate) == primary["rate_archive_sha256"]
    source, fps = previous.source(args.source, meta, 240)
    assert previous.sha(args.source) == primary["source_sha256"]
    assert previous.digest(source) == primary["source_rgb_sha256"]
    assert paid.fps == fps
    marks = [r for r in previous.marks(args.labels, primary["labels_sha256"], 240)
             if r["visibility"] == 1]
    total_labels = len(marks)
    assert total_labels == primary["visible"] > 0
    if stage == "smoke":
        marks = [marks[i] for i in np.linspace(0, total_labels - 1, min(6, total_labels), dtype=int)]
    metric_time = 0.0
    assert previous.sha(args.backbone) == previous.BACKBONE and previous.sha(args.lpips) == ALEX
    lpips = quality.load_lpips(args.lpips, "cpu")
    dists = quality.load_dists(args.backbone, "cpu")
    work = Path(os.environ["PS_SCRATCH_DIR"]) / "matched"
    work.mkdir()
    decoder = svtav1.tool("dav1d")
    assert decoder["sha256"] == previous.TOOLS["dav1d"]
    output = {}
    pos, look = ball.decode_frames(paid.ball_payload, 240, 960, 544)
    motion = ball.velocities(pos)
    smoke_hashes = {}
    for mode in ("candidate", "ordinary"):
        rate = rates[mode]
        data = paid.native if mode == "candidate" else streams[rate["native_sha256"]]
        packet_sha = rate["ball_sha256"] if mode == "candidate" else rate["native_sha256"]
        nbytes = rate["ball_bytes"] if mode == "candidate" else rate["bytes"]
        assert packet_sha == primary[mode + "_sha256"] and nbytes == primary[mode + "_bytes"]
        video = previous.native(data, rate, work, decoder["path"])
        expected = {r["tick"]: r["dists"] for r in primary[mode]["rows"]}
        dig = hashlib.sha256()
        sse = 0
        rows = []
        marked = {r["tick"]: r for r in marks}
        dimensions = rate["width"], rate["height"]
        assert 0 <= primary["gap_fraction"] <= 0.005
        for tick, frame in enumerate(video):
            rgb = previous.display(frame, *dimensions)
            if mode == "candidate":
                rgb = ball.render(rgb, tick, pos, motion, look)
            dig.update(rgb)
            if stage == "full":
                delta = rgb.astype(np.int16) - source[tick]
                sse += int(np.einsum("ijk,ijk->", delta, delta, dtype=np.int64))
            label = marked.get(tick)
            if label is None:
                continue
            center = label["x"] / 2, label["y"] * 544 / 1080
            metric_start = time.monotonic()
            target = crop64(source[tick], *center)
            crop = crop64(rgb, *center)
            if not rows:
                identity = compare(target, target, lpips)
                assert abs(identity["lpips"]) < 1e-9, identity
                # Float32 Gaussian convolution can lose precision on flat luma patches.
                assert abs(identity["luma_ssim"] - 1) < 2e-5, identity
            score = float(dists_crops(dists, [crop], target, np.ones((64, 64), bool), "cpu")[0])
            assert abs(score - expected[tick]) < 2e-5
            values = compare(crop, target, lpips)
            assert all(math.isfinite(x) for x in values.values())
            rows.append(dict(tick=tick, dists=score, **values))
            metric_time += time.monotonic() - metric_start
        rgb_sha = dig.hexdigest()
        assert rgb_sha == primary[mode]["rgb_sha256"]
        smoke_hashes[mode] = rgb_sha
        whole = 10 * math.log10(255**2 * source.size / sse) if sse else None
        if stage == "full":
            assert abs(whole - primary[mode]["whole_rgb_psnr"]) < 1e-8
        output[mode] = dict(
            bytes=nbytes, kbps=primary[mode + "_kbps"], packet_sha256=packet_sha,
            rgb_sha256=rgb_sha, width=dimensions[0], height=dimensions[1], rows=rows,
            means={k: mean(r[k] for r in rows) for k in ("dists", "lpips", "luma_ssim", "rgb_mse")},
            whole_rgb_psnr=whole,
        )
    assert len({r["width"] for r in output.values()}) == 1
    elapsed = time.monotonic() - start
    extra_label_time = metric_time * (total_labels / len(marks) - 1)
    forecast = 1.5 * (elapsed + extra_label_time) + 25
    result = dict(
        match=meta["match"], source_sha256=primary["source_sha256"],
        source_rgb_sha256=primary["source_rgb_sha256"], labels_sha256=primary["labels_sha256"],
        rate_archive_sha256=primary["rate_archive_sha256"], primary_sha256=previous.sha(args.score),
        source_ticks=240, selected_visible_ticks=[r["tick"] for r in marks],
        total_visible=total_labels, matched_rate_gap=primary["gap_fraction"], arms=output,
        checks={"original_native_pixels": len(smoke_hashes) == 2, "common_visible_ticks": True,
                "dists_parity": True, "identical_crop_control": True, "forecast": forecast < 315},
        tools={"opencv": cv2.__version__, "torch": torch.__version__, "dav1d": decoder,
               "lpips": quality.lpips_record(args.lpips), "dists": quality.dists_record(args.backbone)},
        elapsed=elapsed, full_forecast_seconds=forecast,
    )
    (Path(os.environ["PS_STAGE_DIR"]) / "ball-metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


def validate():
    result = json.loads((Path(os.environ["PS_STAGE_DIR"]) / "ball-metrics.json").read_text())
    n = min(6, result["total_visible"])
    checks = dict(result["checks"], paired_ticks=all(
        [row["tick"] for row in arm["rows"]] == result["selected_visible_ticks"]
        and len(arm["rows"]) == n for arm in result["arms"].values()
    ))
    passed = all(checks.values())
    validation = {"passed": passed, "checks": checks, "forecast": result["full_forecast_seconds"]}
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps(validation) + "\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "metadata", "rate", "labels", "score", "backbone", "lpips"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    validate() if args.validate else run(args)
