"""Compare the two fixed affordable physical displays using four metrics on identical source supports."""

import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import mean
import sys
import tarfile
import time

import cv2
import numpy as np
import torch

from src.pointstream.tracked_patch import yuv_to_rgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "al028-neural-display-20261122"))
from score_bodies import REVIEW, sha, load_rgb  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "fixed_native_metrics", HERE.parent / "joint-metric-crosscheck-20261123/score.py"
)
multi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multi)


def main(ordinary, samples, output):
    start = time.process_time()
    groups = ["dists", "lpips", "luma_ssim", "rgb_psnr"]
    labels = Path(
        "/Users/manu/Datasets/pointstream-submission-20261118/al028-author-marks-7bb50c395742/annotations.json"
    )
    prior = json.loads((HERE / "al028-strict-scores.json").read_text())
    receiver = json.loads((HERE / "al028-strict-display.json").read_text())
    assert (
        sha(labels) == prior["source_labels_sha256"] and sha(samples) == receiver["samples_sha256"]
    )
    assert sha(ordinary / "stream.ivf") == prior["anchors"]["ordinary"]["wire_sha256"]
    raw = ordinary / "decoded.yuv"
    assert sha(raw) == "5506f33f8543a07cdf1bf75305209320f6a6bf8cd5d46b8b08da5fe8d675719e"
    base = np.memmap(raw, "u1", "r", shape=(240, 576, 640))
    source = json.loads((REVIEW / "review.json").read_text())
    accepted = json.loads(labels.read_text())
    alex = multi.WEIGHTS.parent / "alexnet-owt-7be5be79.pth"
    assert sha(alex) == multi.ALEX_SHA256 and sha(multi.WEIGHTS) == multi.EXPECTED
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    dists = multi.quality.load_dists(multi.WEIGHTS, "cpu")
    lpips = multi.quality.load_lpips(alex, "cpu")
    rows = []
    with tarfile.open(samples) as archive:
        t0 = time.process_time()
        for i, (frame, mark) in enumerate(zip(source["frames"], accepted["frames"], strict=True)):
            tick = mark["tick"]
            ref = REVIEW / frame["source"]
            assert (
                tick == frame["frame_index"]
                and sha(ref) == frame["source_sha256"] == mark["source_png_sha256"]
            )
            original = load_rgb(ref)
            content = archive.extractfile(f"{tick:03d}.png").read()
            assert (
                hashlib.sha256(content).hexdigest()
                == receiver["private_review_png_sha256"][f"{tick:03d}/joint-54"]
            )
            joint = cv2.cvtColor(
                cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
            )
            av = cv2.resize(
                yuv_to_rgb(base[tick], 640, 384, False), (1920, 1080), interpolation=cv2.INTER_CUBIC
            )
            supports = []
            for p, v in zip(frame["players"], mark["players"], strict=True):
                path = REVIEW / p["proposal"]
                assert (
                    v["status"] == "accept"
                    and p["id"] == v["id"]
                    and sha(path) == p["sha256"] == v["proposal_sha256"]
                )
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
                yy, xx = np.nonzero(mask)
                y = slice(max(0, yy.min() - 32), min(1080, yy.max() + 33))
                x = slice(max(0, xx.min() - 32), min(1920, xx.max() + 33))
                supports.append((p["id"], y, x, mask[y, x]))
            if mark["ball"]["status"] == "visible":
                x, y = mark["ball"]["center_xy"]
                supports.append(
                    ("ball", slice(y - 32, y + 32), slice(x - 32, x + 32), np.ones((64, 64), bool))
                )
            for arm, display in (("joint", joint), ("av1", av)):
                for region, ys, xs, mask in supports:
                    values = multi.region_scores(
                        original[ys, xs], display[ys, xs], mask, dists, lpips
                    )
                    assert all(np.isfinite(v) for v in values.values())
                    rows.append({"arm": arm, "tick": tick, "region": region, **values})
            if i == 0:
                forecast = (
                    (time.process_time() - t0) * len(source["frames"]) * 1.5 + (t0 - start) + 20
                )
                assert forecast < 280, forecast
    means = {}
    visible = {r["tick"] for r in rows if r["region"] == "ball"}
    assert len(visible) == 11 and len(rows) == 70
    for method in ("joint", "av1"):
        group = [r for r in rows if r["arm"] == method and r["tick"] in visible]
        player = {
            metric: mean(r[metric] for r in group if r["region"] != "ball") for metric in groups
        }
        ball = {
            metric: mean(r[metric] for r in group if r["region"] == "ball") for metric in groups
        }
        means[method] = {
            "players": player,
            "ball": ball,
            "equal_class": {k: (player[k] + ball[k]) / 2 for k in groups},
        }
    expected = {"joint": prior["foreground"], "av1": prior["anchors"]["ordinary"]["foreground"]}
    delta = max(abs(means[name]["equal_class"]["dists"] - val) for name, val in expected.items())
    assert delta < 1e-5, delta
    payload = dict(
        means=means,
        rows=rows,
        common_ticks=sorted(visible),
        source_labels_sha256=sha(labels),
        cpu_seconds=time.process_time() - start,
        forecast_seconds=forecast,
        av1_bytes=(ordinary / "stream.ivf").stat().st_size,
        av1_wire_sha256=sha(ordinary / "stream.ivf"),
        joint_wire_sha256=prior["wire_sha256"],
        native_cpu_job="20261122T222252Z-34de1411",
        metric_code_sha256=sha(Path(multi.__file__)),
        script_sha256=sha(Path(__file__)),
        weights=dict(vgg=sha(multi.WEIGHTS), alex=sha(alex)),
        max_previous_dists_delta=delta,
        aggregation="Same two selected physical streams for all metrics. Mean per-region dB for RGB PSNR, mask-weighted spatial AlexNet LPIPS, source-mask-centred luma SSIM, and original masked DISTS. Equal player/ball class averages on 11 common native ticks.",
    )
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(means, indent=2))


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:]))
