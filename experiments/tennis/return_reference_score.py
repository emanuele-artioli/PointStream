"""Score the two archived native return-shot renders against the original body silhouettes."""

from __future__ import annotations

import argparse
import hashlib
from importlib import import_module
import json
import math
import os
from pathlib import Path
import tarfile
import time

import cv2
import numpy as np

from experiments.tennis import k6b, k6b_rate_streams as rates
from experiments.tennis.player_requalification import make_plan, sha256
from experiments.visor.h3 import dis_flow, warp
from experiments.visor.h3b import luma
from src.codecs.quality import load_dists

SOURCE = "f0551443eaff4e4b3539f21cd0c6ac66f55601d480ad099a571e8626db735e4a"
TARGET = "nadal_kyrgios__002@120"
ARMS = ("fresh", "reuse")
HASHES = {
    "fresh": "7d118605424da592cb040d58b4d2ca2fd0a4f6dc2a2c4bcae075ab737027dbe2",
    "reuse": "6f4c5d5bf1bfacf57c2d5cacb9fff505e71ffd79bad8b75909deaa06e5626c21",
}
CARD = Path(__file__).resolve().parents[2] / "docs/experiments/cross-shot-reference-20261122"
PACKET = "d80fac2ab880743e100a90a0dbb46af6173bc2479389a6d8f75663578b9db017"


def registered(a, frames):
    review = import_module("docs.experiments.cross-shot-reference-20261122.build_review")
    a.base = a.base / "publish/tennis-base@fit-100.pt"
    arrays, plans, identities = {}, {}, {}
    configs = {}
    for arm in ARMS:
        for name in ("full", "smoke"):
            key = arm + "_" + name
            setattr(a, key, getattr(a, key) / "publish")
        setattr(a, arm + "_hash", HASHES[arm])
        config, receipt = review.checked(a, arm)
        plans[arm], arrays[arm] = make_plan(config, HASHES[arm], 2)
        configs[arm] = config
        identities[arm] = dict(
            model=receipt["model_record"],
            fit=receipt["fit_sha256"],
            device=receipt["device"],
            outputs=[o["output_sha256"] for o in receipt["outputs"]],
        )
    assert (
        identities["fresh"]["model"] == identities["reuse"]["model"]
        and identities["fresh"]["fit"] == identities["reuse"]["fit"]
    )
    assert (
        plans["fresh"]["settings"] == plans["reuse"]["settings"]
        and plans["fresh"]["source"] == plans["reuse"]["source"]
    )
    renders, boxes = {}, {}
    for i, unit in enumerate(review.UNITS):
        f, r = (plans[arm]["units"][i] for arm in ARMS)
        assert all(f[k] == r[k] for k in ("unit", "run", "t", "seed", "pose_sha256", "box_sha256"))
        for key in ("pose", "t", "box"):
            assert np.array_equal(arrays["fresh"][i][key], arrays["reuse"][i][key])
        assert not np.array_equal(arrays["fresh"][i]["reference"], arrays["reuse"][i]["reference"])
        with np.load(configs["fresh"]["units"][i]["path"], allow_pickle=False) as cond:
            source_ref = k6b.crop_image(frames[0], cond["reference_box"])
        assert np.array_equal(source_ref, arrays["fresh"][i]["reference"])
        boxes[i] = arrays["fresh"][i]["box"]
        for arm in ARMS:
            path = getattr(a, arm + "_full") / (unit + "@tennis_only.npz")
            assert sha256(path) == identities[arm]["outputs"][i]
            with np.load(path, allow_pickle=False) as z:
                assert np.array_equal(z["t"], np.arange(24)) and np.array_equal(z["box"], boxes[i])
                assert z["crops"].shape == (24, 784, 512, 3)
                renders[arm, i] = z["crops"]
    return renders, boxes, identities


def visible_box(box):
    x, y, w, h = k6b.int_box(box)
    left, top = max(0, x), max(0, y)
    right, bottom = min(1920, x + w), min(1080, y + h)
    if left >= right or top >= bottom:
        raise ValueError("No visible native source pixels in paid player envelope")
    return left, top, right - left, bottom - top


def display_crop(rgb, source_box, visible):
    x, y, w, h = k6b.int_box(source_box)
    left, top, dw, dh = visible
    resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return resized[top - y : top - y + dh, left - x : left - x + dw]


def run(a):
    import torch

    score = import_module("docs.experiments.reference-body-rd-20261122.score")
    start = time.monotonic()
    assert not torch.cuda.is_available() and a.frames in (2, 24)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    assert (
        sha256(a.source) == SOURCE
        and sha256(a.transport) == PACKET
        and sha256(a.backbone) == score.WEIGHT_SHA
    )
    src, rawsha, _ = score.source_rgb(a.source)
    arrays, original_boxes, producers = registered(a, src)
    boxes = {
        i: np.asarray([visible_box(box) for box in track]) for i, track in original_boxes.items()
    }
    with tarfile.open(a.transport) as z:
        member = z.extractfile("publish/packets/" + TARGET + "/exact-rle.matte").read()
    meta, labels = rates.decode_masks(member)
    assert labels.ndim == 4 and labels.shape[0] == 240 and labels.shape[2:] == (540, 960)
    ids = {}
    for i, b in boxes.items():
        track = []
        for t in range(a.frames):
            x, y, w, h = k6b.int_box(b[t])
            assert x >= 0 and y >= 0 and x + w <= 1920 and y + h <= 1080
            count = [
                labels[t, k, y // 2 : (y + h + 1) // 2, x // 2 : (x + w + 1) // 2].sum()
                for k in range(labels.shape[1])
            ]
            ids_valid = [k for k, c in enumerate(count) if c > 0]
            assert len(ids_valid) == 1
            track.append(ids_valid[0])
        assert len(set(track)) == 1
        ids[i] = track[0]
    assert len(set(ids.values())) == 2
    net = load_dists(a.backbone, "cpu")
    yy, xx = np.mgrid[:1080, :1920]
    prev, prior_masks = {}, {}
    rows, temporal = [], []
    core = time.monotonic()
    with torch.inference_mode():
        for t in range(a.frames):
            if t:
                flow = dis_flow(
                    cv2.cvtColor(src[t], cv2.COLOR_RGB2GRAY),
                    cv2.cvtColor(src[t - 1], cv2.COLOR_RGB2GRAY),
                )
                source_delta = luma(src[t]) - luma(warp(src[t - 1], flow))
                qx, qy = xx + flow[:, :, 0], yy + flow[:, :, 1]
            for i, arr in boxes.items():
                x, y, w, h = k6b.int_box(arr[t])
                mask = cv2.resize(
                    labels[t, ids[i]].astype(np.uint8),
                    (1920, 1080),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                assert mask.any()
                outputs = {
                    m: display_crop(arrays[m, i][t], original_boxes[i][t], (x, y, w, h))
                    for m in ARMS
                }
                if t == 0 and i == 0:
                    ident = score.compare(
                        net, src[t], {"ident": src[t][y : y + h, x : x + w]}, mask, arr[t]
                    )
                    assert (
                        ident["scores"]["ident"]["sum_squared_rgb"] == 0
                        and abs(ident["scores"]["ident"]["dists"]) < 1e-5
                    )
                r = score.compare(net, src[t], outputs, mask, arr[t])
                rows.append(
                    dict(
                        tick=t,
                        person=i,
                        visible_box=arr[t].tolist(),
                        original_box=list(k6b.int_box(original_boxes[i][t])),
                        **r,
                    )
                )
                if t:
                    px, py, pw, ph = k6b.int_box(arr[t - 1])
                    sy, sx = slice(y, y + h), slice(x, x + w)
                    support = (
                        mask[sy, sx]
                        & (qx[sy, sx] >= px)
                        & (qx[sy, sx] < px + pw - 1)
                        & (qy[sy, sx] >= py)
                        & (qy[sy, sx] < py + ph - 1)
                    )
                    support &= warp(prior_masks[i].astype(np.float32), flow)[sy, sx] > 0.999
                    assert support.any()
                    errors = {}
                    for m in ARMS:
                        canvas = np.zeros_like(src[t])
                        canvas[py : py + ph, px : px + pw] = prev[m, i]
                        delta = luma(outputs[m]) - luma(warp(canvas, flow)[sy, sx])
                        errors[m] = float(
                            np.abs(source_delta[sy, sx][support] - delta[support]).mean()
                        )
                    temporal.append(
                        dict(tick=t, person=i, pixels=int(support.sum()), errors=errors)
                    )
                for m in ARMS:
                    prev[m, i] = outputs[m]
                prior_masks[i] = mask
    summary = {}
    for arm in ARMS:
        per = {}
        for i in boxes:
            vals = [r["scores"][arm] for r in rows if r["person"] == i]
            per[i] = dict(
                dists=float(np.mean([v["dists"] for v in vals])),
                body_psnr=10
                * math.log10(
                    255**2
                    * sum(v["samples"] for v in vals)
                    / sum(v["sum_squared_rgb"] for v in vals)
                ),
                motion_error=float(
                    np.mean([r["errors"][arm] for r in temporal if r["person"] == i])
                ),
            )
        summary[arm] = dict(
            per_player=per,
            **{
                k: float(np.mean([v[k] for v in per.values()]))
                for k in ("dists", "body_psnr", "motion_error")
            },
        )
    body_end = time.monotonic()
    bill = json.loads((CARD / "channel-webp-audit.json").read_text())
    assert (
        bill["passed"]
        and bill["return_reuse_bytes"] == 354
        and bill["return_refresh_bytes"] == 215667
    )
    out = dict(
        stage=os.environ["PS_STAGE"],
        scope="native body quality within transmitted boxes of two anonymous target tracks",
        summary=summary,
        frames=a.frames,
        body_rows=rows,
        temporal_rows=temporal,
        mask_indices=ids,
        matte_meta=meta,
        source_yuv_sha256=rawsha,
        source_sha256=SOURCE,
        source_mask_sha256=hashlib.sha256(member).hexdigest(),
        source_mask_carrier_sha256=PACKET,
        source_ref_max_rgb_delta=[0, 0],
        model=producers,
        stream_cost=bill,
        native_crop_sha256=producers["fresh"]["outputs"] + producers["reuse"]["outputs"],
        vgg_sha256=score.WEIGHT_SHA,
        scoring_revision=sha256(Path(__file__)),
        scoring_lib_sha256=sha256(Path(score.__file__)),
        cv2=cv2.__version__,
        torch=str(torch.__version__),
        compute_seconds=body_end - core,
        total_seconds=body_end - start,
    )
    (Path(os.environ["PS_STAGE_DIR"]) / "return-quality.json").write_text(
        json.dumps(out, indent=2, allow_nan=False) + "\n"
    )


def validate():
    stage = Path(os.environ["PS_STAGE_DIR"])
    r = json.loads((stage / "return-quality.json").read_text())
    n = r["frames"]
    assert n in (2, 24) and len(r["body_rows"]) == 2 * n and len(r["temporal_rows"]) == 2 * (n - 1)
    assert all(v["coverage"] > 0.5 and v["pixels"] > 0 for v in r["body_rows"])
    assert all(
        math.isfinite(r["summary"][m][k])
        for m in ARMS
        for k in ("body_psnr", "dists", "motion_error")
    )
    estimate = (
        1.5 * (r["total_seconds"] - r["compute_seconds"] + r["compute_seconds"] * 24 / n) + 30
    )
    if n == 2 and estimate >= 540:
        raise ValueError(f"Estimated {estimate:.1f} seconds exceeds the full-stage allowance")
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(
            dict(
                passed=True,
                checks=dict(
                    source_reference_control=True,
                    source_mask_mapping=r["mask_indices"],
                    frames=n,
                    paired_rows=len(r["body_rows"]),
                    expected_seconds=estimate,
                ),
            )
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--validate", action="store_true")
    for name in (
        "source",
        "transport",
        "backbone",
        "base",
        *(f"{arm}-{name}" for arm in ARMS for name in ("smoke", "full", "config", "conditioning")),
    ):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--frames", type=int)
    args = p.parse_args()
    if args.validate:
        validate()
    else:
        run(args)
