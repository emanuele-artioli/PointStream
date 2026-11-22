"""Measure authentic native displays on the frozen twelve Al028 body-mask pairs."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch

from score_bodies import EXPECTED, REVIEW, WEIGHTS, load_rgb, sha, tensor
from src.codecs import quality
from src.pointstream.tracked_patch import yuv_to_rgb


def run(args):
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert sha(WEIGHTS) == EXPECTED
    net = quality.load_dists(WEIGHTS, "cpu")
    meta = json.loads((REVIEW / "review.json").read_text())
    ident = meta.pop("review_sha256")
    assert (
        ident
        == hashlib.sha256(
            json.dumps(meta, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        == "fb41c943575b35e6ecec36916b8cfab41b8023744614a98681b7cce884681eee"
    )
    configs = {}
    for receipt in args.inputs:
        d = json.loads(receipt.read_text())
        for row in d["rows"]:
            configs[row["sha256"]] = row
    rows = sorted(configs.values(), key=lambda r: (r["width"], r["bytes"]))[args.start : args.end]
    assert 0 < len(rows) <= 12
    outputs = []
    controls = json.loads((Path(__file__).parent / "player-controls.json").read_text())
    ref_control = {
        (r["tick"], r["person"]): r["dists"] for r in controls["rows"] if r["arm"] == "ordinary"
    }
    sources = {}
    for frame in meta["frames"]:
        t = frame["frame_index"]
        assert sha(REVIEW / frame["source"]) == frame["source_sha256"]
        image = load_rgb(REVIEW / frame["source"])
        masks = []
        for person in frame["players"]:
            path = REVIEW / person["proposal"]
            assert sha(path) == person["sha256"]
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 0
            yy, xx = np.nonzero(mask)
            assert len(xx) == person["pixels"]
            roi = (
                slice(max(0, min(yy) - 32), min(1080, max(yy) + 33)),
                slice(max(0, min(xx) - 32), min(1920, max(xx) + 33)),
            )
            masks.append(
                (
                    person["id"],
                    roi,
                    tensor(image[roi]),
                    torch.from_numpy(mask[roi].copy())[None, None],
                )
            )
        sources[t] = (image, masks)
    assert sorted(sources) == list(range(10, 240, 20))
    start = time.process_time()
    for i, r in enumerate(rows):
        w, h = r["width"], r["height"]
        raw = Path(r["path"]) / "decoded.yuv"
        stream = Path(r["path"]) / "stream.ivf"
        assert (
            sha(raw) == r["raw_sha256"]
            and sha(stream) == r["sha256"]
            and stream.stat().st_size == r["bytes"]
        )
        video = np.memmap(raw, np.uint8, "r", shape=(240, h * 3 // 2, w))
        results = []
        sse = 0
        for t, (ref, masks) in sources.items():
            display = cv2.resize(
                yuv_to_rgb(video[t], w, h, False), (1920, 1080), interpolation=cv2.INTER_CUBIC
            )
            diff = display.astype(np.int16) - ref.astype(np.int16)
            sse += int(np.einsum("ijk,ijk->", diff, diff, dtype=np.int64))
            for name, roi, target, mask in masks:
                val = quality.masked_dists(net, tensor(display[roi]), target, mask).item()
                assert np.isfinite(val)
                if (
                    r["raw_sha256"]
                    == "c60215a513435b726bf9c3dd398b651b1c9add176245a354eb1076ee21445bf1"
                ):
                    assert abs(val - ref_control[t, name]) < 1e-7
                if i == 0 and t == 10:
                    assert abs(quality.masked_dists(net, target, target, mask).item()) < 1e-5
                results.append({"tick": t, "person": name, "dists": val})
        assert len(results) == 24
        outputs.append(
            dict(
                stream_sha256=r["sha256"],
                raw_sha256=r["raw_sha256"],
                width=w,
                height=h,
                bytes=r["bytes"],
                crf=r["crf"],
                kbps=r["kbps"],
                rows=results,
                player_dists=float(np.mean([m["dists"] for m in results])),
                whole_rgb_psnr=float(10 * np.log10(255**2 * 1920 * 1080 * 3 * 12 / sse)),
            )
        )
        if i == 0:
            forecast = (time.process_time() - start) * len(rows) * 1.5 + 15
            if forecast > 600:
                raise ValueError(f"Projected cost outside CPU stage: {forecast}s")
    assert time.process_time() - start < 600
    with args.output.open("x") as f:
        json.dump(
            dict(
                proposal_sha256=ident,
                backbone_sha256=EXPECTED,
                receipts=[sha(p) for p in args.inputs],
                points=outputs,
                forecast_cpu_seconds=forecast,
                cpu_seconds=time.process_time() - start,
            ),
            f,
            indent=2,
        )
        f.write("\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    cli.add_argument("inputs", nargs="+", type=Path)
    cli.add_argument("--start", type=int, default=0)
    cli.add_argument("--end", type=int, required=True)
    run(cli.parse_args())
