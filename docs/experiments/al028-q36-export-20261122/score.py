"""Score source-reviewed Al028 proposal regions on two authentic neural displays."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from src.codecs import quality
from src.pointstream.tracked_patch import yuv_to_rgb

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "prior_body", HERE.parent / "al028-neural-display-20261122/score_bodies.py"
)
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
PRIVATE = Path("/private/tmp/ps-al028-q36-review12-20261122")
Q24 = Path("/private/tmp/ps-nov22-al028-q24-full/full.yuv")
TICKS = list(range(10, 240, 20))


def main():
    began = time.process_time()
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if prior.sha(prior.WEIGHTS) != prior.EXPECTED:
        raise ValueError("metric weights")
    net = quality.load_dists(prior.WEIGHTS, "cpu")
    review = json.loads((prior.REVIEW / "review.json").read_text())
    canonical = review.pop("review_sha256")
    if (
        canonical != "fb41c943575b35e6ecec36916b8cfab41b8023744614a98681b7cce884681eee"
        or hashlib.sha256(
            json.dumps(review, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        != canonical
    ):
        raise ValueError("review identity")
    meta = json.loads((PRIVATE / "review12.json").read_text())
    if (
        meta["ticks"] != TICKS
        or meta["frames_verified"] != 240
        or prior.sha(PRIVATE / "review12.yuv") != meta["review12_sha256"]
    ):
        raise ValueError("Q36 frame identity")
    if prior.sha(Q24) != "bac9b77792126c748a7640eee2ce80649f54c2509d5973623d06e1ed35b931bb":
        raise ValueError("Q24 native identity")
    q36 = np.memmap(PRIVATE / "review12.yuv", np.uint8, "r", shape=(12, 816, 960))
    q24 = np.memmap(Q24, np.uint8, "r", shape=(240, 816, 960))
    old = json.loads(
        (HERE.parent / "al028-neural-display-20261122/player-controls.json").read_text()
    )
    expected = {(r["tick"], r["person"]): r["dists"] for r in old["rows"] if r["arm"] == "neural"}
    rows = []
    sse = {"q24": 0, "q36": 0}
    counts = 0
    for i, row in enumerate(review["frames"]):
        t = row["frame_index"]
        if (
            t != TICKS[i]
            or prior.sha(prior.REVIEW / row["source"]) != row["source_sha256"]
            or hashlib.sha256(q36[i]).hexdigest() != meta["frame_sha256"][i]
        ):
            raise ValueError("source/tick/Q36 digest")
        ref = prior.load_rgb(prior.REVIEW / row["source"])
        images = {
            name: cv2.resize(
                yuv_to_rgb(raw, 960, 544, False), (1920, 1080), interpolation=cv2.INTER_CUBIC
            )
            for name, raw in [("q24", q24[t]), ("q36", q36[i])]
        }
        for name, img in images.items():
            delta = ref.astype(np.int32) - img.astype(np.int32)
            sse[name] += int(np.sum(delta * delta, dtype=np.int64))
        counts += ref.size
        before = time.process_time()
        for person in row["players"]:
            maskpath = prior.REVIEW / person["proposal"]
            if prior.sha(maskpath) != person["sha256"]:
                raise ValueError("proposal identity")
            mask = cv2.imread(str(maskpath), cv2.IMREAD_GRAYSCALE) > 0
            ys, xs = np.nonzero(mask)
            if len(xs) != person["pixels"]:
                raise ValueError("proposal size")
            y = slice(max(0, ys.min() - 32), min(1080, ys.max() + 33))
            x = slice(max(0, xs.min() - 32), min(1920, xs.max() + 33))
            m = torch.from_numpy(mask[y, x].copy())[None, None]
            target = prior.tensor(ref[y, x])
            for name, img in images.items():
                score = quality.masked_dists(net, prior.tensor(img[y, x]), target, m).item()
                if not math.isfinite(score):
                    raise ValueError("nonfinite score")
                rows.append(
                    {
                        "tick": t,
                        "person": person["id"],
                        "arm": name,
                        "dists": score,
                        "mask_pixels": len(xs),
                    }
                )
        if i == 0 and (time.process_time() - before) * 12 * 1.5 + 60 >= 600:
            raise ValueError("CPU forecast")
        if time.process_time() - began >= 600:
            raise ValueError("CPU limit")
    errors = [
        abs(r["dists"] - expected[(r["tick"], r["person"])]) for r in rows if r["arm"] == "q24"
    ]
    if len(errors) != 24 or max(errors) > 1e-4:
        raise ValueError("Q24 original-score mismatch")
    means = {
        name: float(np.mean([r["dists"] for r in rows if r["arm"] == name])) for name in images
    }
    result = {
        "source_review_sha256": canonical,
        "q36_review12_sha256": meta["review12_sha256"],
        "q24_yuv_sha256": prior.sha(Q24),
        "ticks": TICKS,
        "rows": rows,
        "mean_dists": means,
        "q24_control_max_abs_error": max(errors),
        "full_frame_rgb_psnr": {k: 10 * math.log10(255**2 * counts / v) for k, v in sse.items()},
        "rates_kbps": {"q24": 20.411666666666665, "q36": 39.770833333333336},
        "scope": "12 source ticks, two source-model proposal masks per tick; 32-source-pixel crop padding; no human body labels",
        "cpu_seconds": time.process_time() - began,
    }
    p = HERE / "body-scores.json"
    with p.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(
        json.dumps(
            {
                "mean_dists": means,
                "psnr": result["full_frame_rgb_psnr"],
                "q24_error": max(errors),
                "cpu": result["cpu_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
