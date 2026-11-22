"""Measure source-flow temporal residuals for the paired reference ablation."""

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path
import time

import cv2
import numpy as np
from experiments.visor.h3 import dis_flow, warp
from experiments.visor.h3b import luma
from experiments.tennis.k6b import int_box
import score


def main(a):
    cv2.setNumThreads(1)
    start = time.process_time()
    people, frames, _, masks, boxes, renders, control, raw_sha, _ = score.read_inputs(a)
    assert max(abs(v["observed"] - v["previous"]) for v in control.values()) < 1e-5
    modes = list(score.MODES)
    if a.with_jpeg:
        receipts = json.loads((score.HERE / "jpeg-result.json").read_text())["init"]
        for mode, expected in receipts.items():
            archive = a.scratch / (mode + ".tar")
            assert score.digest(archive) == expected["archive_sha256"]
            with tarfile.open(archive) as tar:
                for person in people:
                    record = expected["units"][person]
                    blob = tar.extractfile("publish/" + record["unit"] + "@tennis_only.npz").read()
                    assert hashlib.sha256(blob).hexdigest() == record["output_sha256"]
                    with np.load(io.BytesIO(blob), allow_pickle=False) as z:
                        assert np.array_equal(z["box"], boxes[person])
                        assert np.array_equal(z["t"], np.arange(24))
                        renders[mode, person] = z["crops"]
            modes.append(mode)
    rows = []
    yy, xx = np.mgrid[: score.H, : score.W]
    for t in range(1, 24):
        flow = dis_flow(
            cv2.cvtColor(frames[t], cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(frames[t - 1], cv2.COLOR_RGB2GRAY),
        )
        qx, qy = xx + flow[:, :, 0], yy + flow[:, :, 1]
        raw_delta = luma(frames[t]) - luma(warp(frames[t - 1], flow))
        for person in people:
            x, y, w, h = int_box(boxes[person][t])
            px, py, pw, ph = int_box(boxes[person][t - 1])
            sx, sy = slice(x, x + w), slice(y, y + h)
            # Both endpoints must be inside the corresponding decoded patch and body mask.
            eligible = masks[t, person][sy, sx] & (
                (qx[sy, sx] >= px)
                & (qx[sy, sx] < px + pw - 1)
                & (qy[sy, sx] >= py)
                & (qy[sy, sx] < py + ph - 1)
            )
            mapped_previous = warp(masks[t - 1, person].astype(np.float32), flow)[sy, sx]
            eligible &= mapped_previous > 0.999
            assert eligible.any()
            for mode in modes:
                prev = np.zeros((score.H, score.W, 3), np.uint8)
                prev[py : py + ph, px : px + pw] = cv2.resize(
                    renders[mode, person][t - 1], (pw, ph), interpolation=cv2.INTER_LANCZOS4
                )
                cur = cv2.resize(renders[mode, person][t], (w, h), interpolation=cv2.INTER_LANCZOS4)
                delta = luma(cur) - luma(warp(prev, flow)[sy, sx])
                error = np.mean(np.abs(raw_delta[sy, sx][eligible] - delta[eligible]))
                rows.append(
                    dict(
                        tick=t,
                        person=person,
                        mode=mode,
                        pixels=int(eligible.sum()),
                        luma_error=float(error),
                    )
                )
        if time.process_time() - start > 300:
            raise ValueError("Temporal diagnostic exceeds the reserved five CPU minutes")
    summary = {m: float(np.mean([r["luma_error"] for r in rows if r["mode"] == m])) for m in modes}
    with a.output.open("x") as out:
        json.dump(
            dict(
                source_yuv_sha256=raw_sha,
                script_sha256=score.digest(Path(__file__)),
                cpu_seconds=time.process_time() - start,
                meaning="Absolute motion-compensated luma residual error on the intersection of adjacent source-model bodies, both endpoints inside native patches; equal player/transition weighting.",
                rows=rows,
                summary=summary,
            ),
            out,
            indent=2,
            allow_nan=False,
        )
        out.write("\n")
    print(summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("scratch", "labels", "weights", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--yuv", type=Path)
    p.add_argument("--with-jpeg", action="store_true")
    main(p.parse_args())
