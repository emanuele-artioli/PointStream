"""Measure the same frozen source-derived player regions on each native display."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import cv2
import numpy as np
import torch

from src.codecs import quality
from src.pointstream.tracked_patch import yuv_to_rgb

HERE = Path(__file__).resolve().parent
REVIEW = Path("/Users/manu/Documents/PointStream-review/al028-source-annotations-20261122")
WEIGHTS = Path(
    "/Users/manu/.codex/visualizations/2026/11/01/01a196fe-ab4d-70f0-a4b4-6843ca289667/atlas-metric-models/checkpoints/vgg16-397923af.pth"
)
EXPECTED = "397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0"


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def load_rgb(path):
    return cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def tensor(rgb):
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)[None].float() / 255


def run(args):
    t0 = time.process_time()
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert sha(WEIGHTS) == EXPECTED
    net = quality.load_dists(WEIGHTS, "cpu")
    meta = json.loads((REVIEW / "review.json").read_text())
    canonical = meta.pop("review_sha256")
    assert (
        canonical
        == hashlib.sha256(
            json.dumps(meta, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )
    assert canonical == "fb41c943575b35e6ecec36916b8cfab41b8023744614a98681b7cce884681eee"
    assert [r["frame_index"] for r in meta["frames"]] == list(range(10, 240, 20))
    pairs = json.loads((HERE / "common-budget.json").read_text())["selected"]
    neural = json.loads((HERE / "verified.json").read_text())
    decodes = {}
    if args.joint_directory:
        assert (
            args.reconstruction_sha
            and sha(args.joint_directory / "result.json") == args.reconstruction_sha
        )
        reconstruction = json.loads((args.joint_directory / "result.json").read_text())
    else:
        for arm, path, w, h, expected in (
            (
                "ordinary",
                Path(pairs["ordinary"]["path"]) / "decoded.yuv",
                320,
                192,
                pairs["ordinary"]["raw_sha256"],
            ),
            ("neural", args.dcvc, 960, 544, neural["raw_sha256"]),
        ):
            assert sha(path) == expected and path.stat().st_size == 240 * w * h * 3 // 2
            decodes[arm] = (np.memmap(path, "uint8", "r", shape=(240, h * 3 // 2, w)), w, h)
    scores = []
    for i, row in enumerate(meta["frames"]):
        t = row["frame_index"]
        assert sha(REVIEW / row["source"]) == row["source_sha256"]
        ref = load_rgb(REVIEW / row["source"])
        displays = {}
        if args.joint_directory:
            for arm in ("joint-20", "source", "av1_320_q52_75", "dcvc_q24"):
                key = f"{t:03d}/{arm}"
                path = args.joint_directory / "private-review" / (key + ".png")
                assert sha(path) == reconstruction["private_review_png_sha256"][key]
                pix = load_rgb(path)
                if arm == "source":
                    assert np.array_equal(pix, ref)
                else:
                    displays[
                        {"joint-20": "joint", "av1_320_q52_75": "ordinary", "dcvc_q24": "neural"}[
                            arm
                        ]
                    ] = pix
        else:
            for name, (video, w, h) in decodes.items():
                displays[name] = cv2.resize(
                    yuv_to_rgb(video[t], w, h, False), (1920, 1080), interpolation=cv2.INTER_CUBIC
                )
        if i == 0:
            displays["self"] = ref
            smoke = time.process_time()
        assert ref.shape == (1080, 1920, 3) and len(row["players"]) == 2
        for p in row["players"]:
            name = p["id"]
            path = REVIEW / p["proposal"]
            assert sha(path) == p["sha256"]
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
            yy, xx = np.nonzero(mask)
            assert len(yy) == p["pixels"]
            y = slice(max(0, yy.min() - 32), min(1080, yy.max() + 33))
            x = slice(max(0, xx.min() - 32), min(1920, xx.max() + 33))
            m = torch.from_numpy(mask[y, x].copy())[None, None]
            target = tensor(ref[y, x])
            for arm, img in displays.items():
                val = quality.masked_dists(net, tensor(img[y, x]), target, m).item()
                assert np.isfinite(val) and (arm != "self" or abs(val) < 1e-5)
                if arm != "self":
                    scores.append(
                        dict(tick=t, person=name, arm=arm, dists=val, mask_pixels=len(xx))
                    )
        if i == 0:
            forecast = (time.process_time() - smoke) * 12 * 1.5 + 60
            assert forecast < 600
    means = {
        arm: statistics.mean(r["dists"] for r in scores if r["arm"] == arm)
        for arm in displays
        if arm != "self"
    }
    cpu = time.process_time() - t0
    assert cpu < 600 and len(scores) == 24 * len(means)
    result = dict(
        scope="Source-model mask proposals, no human labels. Each person and tick is equally weighted.",
        review_sha256=canonical,
        mean_dists=means,
        rows=scores,
        forecast_cpu=forecast,
        cpu=cpu,
        native_rates={
            "ordinary": pairs["ordinary"]["kbps"],
            "neural": neural["kbps"],
            "joint": pairs["joint"]["kbps"],
        },
        weights_sha256=EXPECTED,
        metric=quality.dists_record(WEIGHTS),
        reconstruction_sha256=args.reconstruction_sha,
    )
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(means, cpu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--dcvc", type=Path, default=Path("/private/tmp/ps-nov22-al028-q24-full/full.yuv")
    )
    parser.add_argument("--joint-directory", type=Path)
    parser.add_argument("--reconstruction-sha")
    run(parser.parse_args())
