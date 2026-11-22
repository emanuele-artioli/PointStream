"""Score paid and baseline motion over every matched source-supported native transition."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np

from experiments.tennis.player_requalification import array_sha
from experiments.visor.h3 import dis_flow, warp
from experiments.visor.h3b import luma
from src.pointstream.tracked_patch import yuv_to_rgb
from score_native import FPS, NT, StreamingYUV, get_support, prep, sha


def source_mask(crop):
    (x0, y0, x1, y1), m = crop
    mask = np.zeros((1080, 1920), np.uint8)
    mask[y0:y1, x0:x1] = m
    return cv2.resize(mask, (960, 544), interpolation=cv2.INTER_NEAREST).astype(np.float32)


def main(a):
    cv2.setNumThreads(1)
    arms = json.loads(a.arms.read_text())
    assert len(arms) == len({x["name"] for x in arms})
    supports, support_hash = get_support(a.features)
    yy, xx = np.indices((544, 960), dtype=np.float32)
    cap = cv2.VideoCapture(str(getattr(a, "source_pixels", a.source)), cv2.CAP_FFMPEG)
    assert abs(cap.get(cv2.CAP_PROP_FPS) - float(FPS)) < 0.001
    rows = {r["name"]: [] for r in arms}
    prev = {}
    prev_gray = None
    prev_masks = []
    cpu0 = time.process_time()
    with ExitStack() as scope:
        images = {}
        for r in arms:
            work = Path(scope.enter_context(tempfile.TemporaryDirectory(dir=a.scratch)))
            images[r["name"]] = prep(a, r, work)[0]
            if isinstance(images[r["name"]], StreamingYUV):
                scope.callback(images[r["name"]].close)
        setup_seconds = time.process_time() - cpu0
        scoring_start = time.process_time()
        for t in range(a.frames):
            ok, bgr = cap.read()
            assert ok
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if getattr(a, "source_hashes", None) is not None:
                assert array_sha(rgb) == a.source_hashes[t]
            image = cv2.resize(rgb, (960, 544), interpolation=cv2.INTER_AREA)
            g = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            values = {"source": luma(image)}
            masks = [source_mask(v) for v in supports[t]]
            for r in arms:
                name = r["name"]
                fr = images[name][t]
                if r["kind"] not in ("rgb_npy", "rgb_zseq"):
                    fr = cv2.resize(
                        yuv_to_rgb(fr, r["width"], r["height"], False),
                        (1920, 1080),
                        interpolation=cv2.INTER_CUBIC,
                    )
                values[name] = luma(cv2.resize(fr, (960, 544), interpolation=cv2.INTER_AREA))
            if t:
                assert prev_gray is not None
                flow = dis_flow(g, prev_gray)
                x, y = xx + flow[:, :, 0], yy + flow[:, :, 1]
                inside = (x >= 0) & (y >= 0) & (x < 959) & (y < 543)
                delta = {n: im - warp(prev[n], flow) for n, im in values.items()}
                for role, index in (("near", 0), ("far", 1)):
                    region = inside & (masks[index] > 0) & (warp(prev_masks[index], flow) >= 0.5)
                    if not region.any():
                        continue
                    src = delta["source"][region]
                    for r in arms:
                        name = r["name"]
                        d = delta[name][region]
                        rows[name].append(
                            dict(
                                tick=t,
                                role=role,
                                pixels=int(region.sum()),
                                error=float(np.abs(d - src).mean()),
                                source_delta=float(np.abs(src).mean()),
                                decoded_delta=float(np.abs(d).mean()),
                            )
                        )
                if t == 4:
                    forecast = 1.5 * (
                        setup_seconds + (time.process_time() - scoring_start) * NT / 5
                    )
                    if forecast > 600:
                        raise ValueError(
                            f"temporal CPU stage forecast {forecast:.1f} s exceeds cap"
                        )
            prev, prev_gray, prev_masks = values, g, masks
    cap.release()
    summary = {}
    for n, seq in rows.items():
        summary[n] = dict(
            n=len(seq),
            **{
                key: float(np.mean([r[key] for r in seq]))
                for key in ("error", "source_delta", "decoded_delta")
            },
        )
    result = dict(
        source_sha256=sha(a.source),
        source_mask_rle_sha256=support_hash,
        grid=[960, 544],
        fps=str(FPS),
        recorded_ticks=a.frames,
        arms=[{"name": r["name"], "bytes": r["bytes"]} for r in arms],
        tool=f"OpenCV {cv2.__version__}, source-only backward DIS medium",
        summary=summary,
        rows=rows,
        cpu_seconds=time.process_time() - cpu0,
    )
    a.out.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("source", "features", "arms", "scratch", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--frames", type=int, choices=(5, NT), default=NT)
    main(p.parse_args())
