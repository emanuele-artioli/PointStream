"""K5b (1) diagnostic, not evidence (docs/experiments.md, K5b 1 reading): how far a drawn ball gets over the true
background (the window's temporal median; TrackNet's camera is still), on the source frames alone.

    python -m experiments.tennis.k5b_diagnostic WINDOWS_JSON FRAMES_ROOT OUT_JSON

FRAMES_ROOT holds K4's TrackNet archive (``tracknet-k4.tar``) extracted. Per window, ball-region PSNR (within 8 px of
the label) of: the median alone; K5's paste (the source within 4 px); one appearance per window (the renderer, with a
saturation gain of 1 or 3); that renderer shifted per frame by its best offset within `SHIFT` px (a position oracle);
a free setting and colour per frame at the label (an appearance oracle).
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any

import numpy as np

from experiments.tennis import windows as tw
from src.pointstream import ball

SETTINGS = [(g, r, e, p) for g in (1, 3) for r in range(len(ball.RADII)) for e in range(len(ball.EXPOSURES)) for p in (2, 3)
            if e or p == 3]
SHIFT = 4


def alpha(g: dict[str, Any], gain: int, ri: int, ei: int, pi: int) -> np.ndarray:
    return np.minimum(1.0, gain * ball.coverage(g["dx"], g["dy"], g["velocity"], ball.RADII[ri], ball.EXPOSURES[ei], ball.PHASES[pi]))


def window(w: dict[str, Any], root: str) -> dict[str, Any]:
    import cv2

    n = min(240, int(w["frames"]))
    src = np.stack([cv2.imread(f"{root}/{w['source']['member_dir']}/{t:04d}.jpg")[:, :, ::-1] for t in range(n)])
    bg = np.median(src[::2], 0).astype(np.uint8)
    track: ball.Track = [(x, y) if v in tw.VISIBLE else None for v, x, y in tw.labels(w, n)]
    groups = ball.Crops.build(track, src, ball.velocities(track)).groups
    dec = [ball.crops_of(np.broadcast_to(bg, src.shape), g["frames"], g["centres"], g["half"]) for g in groups]
    region = [((g["dx"] ** 2 + g["dy"] ** 2 <= 64)[None] & (g["valid"] > 0)) for g in groups]
    paste = [((g["dx"] ** 2 + g["dy"] ** 2 <= 16)[None] & (g["valid"] > 0)) for g in groups]
    count = sum(float(m.sum() * 3) for m in region)

    def psnr(se: float) -> float:
        return float(10 * np.log10(255**2 / (se / count)))

    def se_region(comps: list[np.ndarray]) -> float:
        return sum(float(((c - g["source"]) ** 2)[m].sum()) for g, c, m in zip(groups, comps, region))

    res: dict[str, Any] = {"visible": sum(len(g["frames"]) for g in groups), "background": psnr(se_region(dec)),
                           "pixel_oracle": psnr(se_region([np.where(p[..., None], g["source"], d) for g, d, p in zip(groups, dec, paste)]))}
    best_frame = np.full(res["visible"], np.inf)
    best: tuple[float, Any] = (np.inf, None)
    for gain, ri, ei, pi in SETTINGS:
        aa, aw, ww = [], [], []
        for g, d, m in zip(groups, dec, region):
            a = alpha(g, gain, ri, ei, pi) * m
            wv = (g["source"] - d * (1 - a[..., None])) * m[..., None]
            aa.append((a * a).sum((1, 2)))
            aw.append((a[..., None] * wv).sum((1, 2)))
            ww.append((wv * wv).sum((1, 2)))
        A, AW, WW = np.concatenate(aa), np.concatenate(aw), np.concatenate(ww)
        per_frame = np.clip(np.where(A[:, None] > 0, AW / np.maximum(A[:, None], 1e-9), 0), 0, 255)
        best_frame = np.minimum(best_frame, (per_frame**2 * A[:, None] - 2 * per_frame * AW + WW).sum(1))
        col = np.clip(np.round(AW.sum(0) / max(float(A.sum()), 1e-9)), 0, 255)
        se = float((col**2 * A.sum() - 2 * col * AW.sum(0) + WW.sum(0)).sum())
        if se < best[0]:
            best = (se, (gain, ri, ei, pi, col))
    gain, ri, ei, pi, col = best[1]
    res["window_renderer"] = {"ball_psnr": psnr(best[0]), "gain": gain, "radius": ball.RADII[ri], "exposure": round(ball.EXPOSURES[ei], 2),
                              "phase": round(ball.PHASES[pi], 2), "colour": col.astype(int).tolist()}
    res["per_frame_appearance"] = psnr(float(best_frame.sum()))
    se_shift = 0.0
    shifts = []
    for g, d, m in zip(groups, dec, region):
        a = alpha(g, gain, ri, ei, pi)
        low = np.full(len(g["frames"]), np.inf)
        arg = np.zeros((len(g["frames"]), 2), int)
        for sy in range(-SHIFT, SHIFT + 1):
            for sx in range(-SHIFT, SHIFT + 1):
                moved = np.roll(a, (sy, sx), axis=(1, 2))[..., None]
                e = ((((d * (1 - moved) + col * moved) - g["source"]) ** 2) * m[..., None]).sum((1, 2, 3))
                better = e < low
                low[better] = e[better]
                arg[better] = (sx, sy)
        se_shift += float(low.sum())
        shifts.append(arg)
    s = np.concatenate(shifts)
    res["position_oracle"] = {"ball_psnr": psnr(se_shift), "median_abs_shift_px": float(np.median(np.hypot(s[:, 0], s[:, 1]))),
                              "mean_shift": s.mean(0).round(2).tolist()}
    return res


def main(argv: list[str]) -> int:
    windows_json, root, out_json = argv
    out = {}
    for w in (w for w in json.load(open(windows_json))["windows"] if w["dataset"] == "tracknet"):
        t0 = time.time()
        out[w["id"]] = window(w, root)
        print(w["id"], f"{time.time() - t0:.0f}s", json.dumps(out[w["id"]]), flush=True)
    with open(out_json, "w") as f:
        json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
