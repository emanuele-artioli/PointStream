"""Score sealed joint PNGs from additional Al028 receiver settings."""

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from score_bodies import EXPECTED, REVIEW, WEIGHTS, check_source_pair, load_rgb, sha, tensor
from src.codecs import quality


def run(a):
    torch.set_num_interop_threads(1)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    assert sha(WEIGHTS) == EXPECTED and sha(a.directory / "result.json") == a.receipt_sha
    report = json.loads((a.directory / "result.json").read_text())
    meta = json.loads((REVIEW / "review.json").read_text())
    assert (
        meta["review_sha256"] == "fb41c943575b35e6ecec36916b8cfab41b8023744614a98681b7cce884681eee"
    )
    assert [r["frame_index"] for r in meta["frames"]] == list(range(10, 240, 20))
    net = quality.load_dists(WEIGHTS, "cpu")
    start = time.process_time()
    scores = []
    for i, frame in enumerate(meta["frames"]):
        t = frame["frame_index"]
        source = REVIEW / frame["source"]
        assert sha(source) == frame["source_sha256"]
        ref = load_rgb(source)

        def image(name):
            k = f"{t:03d}/{name}"
            p = a.directory / "private-review" / (k + ".png")
            assert sha(p) == report["private_review_png_sha256"][k]
            return load_rgb(p)

        image("source")
        check_source_pair(t, source, a.directory / "private-review" / f"{t:03d}/source.png")
        for arm in a.arms:
            rendered = image(arm)
            assert rendered.shape == ref.shape == (1080, 1920, 3)
            delta = rendered.astype(np.int32) - ref.astype(np.int32)
            sse = int(np.sum(delta * delta, dtype=np.int64))
            regional = []
            for person in frame["players"]:
                mask_path = REVIEW / person["proposal"]
                assert sha(mask_path) == person["sha256"]
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 0
                yy, xx = np.nonzero(mask)
                assert len(xx) == person["pixels"]
                region = (
                    slice(max(0, yy.min() - 32), min(1080, yy.max() + 33)),
                    slice(max(0, xx.min() - 32), min(1920, xx.max() + 33)),
                )
                support = torch.from_numpy(mask[region].copy())[None, None]
                target = tensor(ref[region])
                val = quality.masked_dists(net, tensor(rendered[region]), target, support).item()
                assert np.isfinite(val)
                if i == 0:
                    assert abs(quality.masked_dists(net, target, target, support).item()) < 1e-5
                regional.append(dict(person=person["id"], pixels=len(xx), dists=val))
            scores.append(dict(tick=t, arm=arm, body=regional, rgb_sse=sse))
        if i == 0:
            forecast = (time.process_time() - start) * 12 * 1.5 + 25
            assert forecast < 600
    assert len(scores) == len(a.arms) * 12 and time.process_time() - start < 600
    with a.output.open("x") as f:
        json.dump(
            dict(
                reconstruction_sha256=a.receipt_sha,
                source_review_sha256=meta["review_sha256"],
                backbone_sha256=EXPECTED,
                forecast_cpu_seconds=forecast,
                cpu_seconds=time.process_time() - start,
                rows=scores,
            ),
            f,
            indent=2,
        )
        f.write("\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("receipt_sha")
    p.add_argument("output", type=Path)
    p.add_argument("arms", nargs="+")
    run(p.parse_args())
