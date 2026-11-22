"""Measure full-picture RGB PSNR on the same twelve source frames as the object scores."""

import argparse
import json
import math
from pathlib import Path
import time
import zipfile

import cv2
import numpy as np
import motion


def main(controls, out):
    cv2.setNumThreads(1)
    started = time.process_time()
    c = motion.crops
    assert (
        c.file_sha(c.BUNDLE) == "5c402652ea7e2b471d870fb211f263ba1f7975afaf4b9b8a55be814d25ae45eb"
    )
    regions = json.loads((motion.HERE / "control-scores.json").read_text())["points"]
    arms = []
    for cap in (20, 33):
        for width in (160, 240, 480):
            data = json.loads((controls / f"{cap}-{width}/result.json").read_text())
            w, h = data["geometry"]
            for row in data["selected"].values():
                identity = row["decoded_sha256"]
                match = [
                    r
                    for r in regions
                    if (r["cap"], r["width"], r["mode"], r["decoded_sha256"])
                    == (cap, width, row["mode"], identity)
                ]
                assert len(match) == 1 and c.file_sha(Path(row["decoded"])) == identity
                yuv = np.memmap(row["decoded"], np.uint8, "r", shape=(240, h * 3 // 2, w))
                arms.append((match[0], w, h, yuv))
    sse = [0] * len(arms)
    labels = json.loads(c.LABELS.read_text())["frames"]
    assert [f["tick"] for f in labels] == c.TICKS
    with zipfile.ZipFile(c.BUNDLE) as z:
        for idx, label in enumerate(labels):
            t = label["tick"]
            encoded = z.read(f"frame-{t:03d}/source.png")
            assert motion.trial.sha(encoded) == label["source_png_sha256"]
            rgb = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)[
                :, :, ::-1
            ].astype(np.int32)
            assert rgb.shape == (1080, 1920, 3)
            for i, (_, w, h, yuv) in enumerate(arms):
                diff = rgb - motion.view(yuv[t], w, h)
                sse[i] += int(np.sum(diff * diff, dtype=np.int64))
            if idx == 0:
                assert (time.process_time() - started) * 12 * 1.5 < 180
    n = 12 * 1920 * 1080 * 3
    result = [
        dict(**r, sse=error, full_rgb_psnr=10 * math.log10(255**2 * n / error))
        for (r, _, _, _), error in zip(arms, sse, strict=True)
    ]
    joint = json.loads((motion.HERE / "scores.json").read_text())["points"]
    record = dict(
        ordinary_and_roi=result,
        joint=[
            {
                k: r[k]
                for k in ("bytes", "kbps", "cold_kbps", "players", "ball", "reviewed_full_rgb_psnr")
            }
            for r in joint
        ],
        ticks=c.TICKS,
        source_bundle_sha256=c.file_sha(c.BUNDLE),
        script_sha256=c.file_sha(Path(__file__)),
        denominator=n,
        cpu_seconds=time.process_time() - started,
    )
    with out.open("x") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    for r in result:
        print(r["cap"], r["width"], r["mode"], r["bytes"], round(r["full_rgb_psnr"], 3))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("controls", type=Path)
    cli.add_argument("out", type=Path)
    args = cli.parse_args()
    main(args.controls, args.out)
