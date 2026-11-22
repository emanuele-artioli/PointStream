"""Audit foreground-removal coverage using existing source masks after freezing a delivery."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from build import sha


def main(a):
    receipt = json.loads((a.build / "result.json").read_text())
    assert receipt["frames"] == 240
    footprint = a.build / "sender-fill-mask.npz"
    assert sha(footprint) == receipt["donor"]["sender_fill_mask_sha256"]
    with np.load(footprint, allow_pickle=False) as f:
        times = f["ticks"]
        masks = np.unpackbits(f["packed"], axis=2)
    start, stop = receipt["active"]
    assert np.array_equal(times, np.arange(start, stop)) and masks.shape == (120, 544, 960)
    human = json.loads(a.supports.read_text())
    raw = np.memmap(a.build / "source.yuv", "uint8", "r", shape=(240, 816, 960))
    repl = np.memmap(a.build / "mixed.yuv", "uint8", "r", shape=raw.shape)
    rows = []
    for r in human["rows"]:
        t = r["tick"]
        if not start <= t < stop or not r["area"]:
            continue
        x0, y0, x1, y1 = r["xyxy"]
        p = a.supports.parent / r["mask"]
        assert sha(p) == r["sha256"]
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0
        fill = cv2.resize(masks[t - start], (1920, 1080), interpolation=cv2.INTER_NEAREST)[
            y0:y1, x0:x1
        ].astype(bool)
        lit = (raw[t, :544] != repl[t, :544]).astype("uint8")
        difference = cv2.resize(lit, (1920, 1080), interpolation=cv2.INTER_NEAREST)[
            y0:y1, x0:x1
        ].astype(bool)
        rows.append(
            dict(
                tick=t,
                object_index=r["object_index"],
                source_pixels=int(m.sum()),
                filled_pixels=int((m & fill).sum()),
                luma_changed_pixels=int((m & difference).sum()),
            )
        )
    assert rows and all(x["luma_changed_pixels"] <= x["filled_pixels"] for x in rows)
    summed = {
        k: sum(r[k] for r in rows)
        for k in ["source_pixels", "filled_pixels", "luma_changed_pixels"]
    }
    out = dict(
        scope="Mask coverage and luma replacement are geometric checks against the sender-side archived W7 human-body model. They do not establish true occluded-court fidelity or racket removal.",
        source_supports_sha256=sha(a.supports),
        source_sha256=receipt["source_sha256"],
        fill_mask_sha256=sha(footprint),
        per_object=rows,
        total=summed,
        filled_fraction=summed["filled_pixels"] / summed["source_pixels"],
        luma_change_fraction=summed["luma_changed_pixels"] / summed["source_pixels"],
    )
    a.output.write_text(json.dumps(out, indent=2) + "\n")
    print(out["filled_fraction"], out["luma_change_fraction"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("build", "supports", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    main(p.parse_args())
