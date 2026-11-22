"""Recompose two paid Al028 base rates using the same sealed player crops and masks.

Reads a source video solely for whole-frame RGB PSNR, never for reconstruction.
No human labels or sender-track accuracy are read or inferred.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import cv2
import numpy as np

from experiments.tennis import common_player_al028_finish as finish
from experiments.tennis import player_requalification as replay
from src.pointstream import ball
from src.pointstream.tracked_patch import yuv_to_rgb

ASSET_ROOT = Path(os.environ.get("AL028_ASSETS", "/private/tmp"))

UNITS = (
    "alcaraz_hurkacz__028_at120__person_000000000004",
    "alcaraz_hurkacz__028_at120__person_000000000006",
)
CARRIER = ASSET_ROOT / "ps-al028-portable-20261122/players.pscp"
SOURCE = ASSET_ROOT / "ps-al028-review-source-20261122.mp4"
BALL = ASSET_ROOT / "ps-al028-ball-sender-full.payload"
BASES = {
    11: (
        (ASSET_ROOT / "ps-al028-native25-pinned.yuv"),
        160,
        96,
        "f2b9dd6a2370b3a6267a0379339360307bfe7900f58e7a503b8b2a2b1756bafb",
    ),
    20: (
        (ASSET_ROOT / "ps-nov22-al028-exact-budget/joint-53.5/decoded.yuv"),
        320,
        192,
        "0831a58cfe5b5b491235486f52bf220886f2ad9e3f4ff3914e30b90d8a9f278d",
    ),
    21: (
        (ASSET_ROOT / "ps-al028-native320-pinned.yuv"),
        320,
        192,
        "0474cc789810859513295a1bc41c357fb286ad60414864ba6e31b27d1f22e965",
    ),
}
BASELINES = {
    "av1_320_q52_75": (
        (ASSET_ROOT / "ps-nov22-al028-exact-budget/ordinary-52.75/decoded.yuv"),
        320,
        192,
        20.360833333333332,
        "c60215a513435b726bf9c3dd398b651b1c9add176245a354eb1076ee21445bf1",
    ),
    "av1_320_q52": (
        (ASSET_ROOT / "ps-al028-av1-target-20261122/320-crf52/decoded.yuv"),
        320,
        192,
        21.008333333333333,
        "dec4ea4740491d2eec30b6b94e67a154485bf8dc8bb390d0a1abb310e548194e",
    ),
    "av1_640_q62": (
        (ASSET_ROOT / "ps-al028-av1-refine-20261122/640-crf62/decoded.yuv"),
        640,
        384,
        20.598333333333333,
        "c3545779ac5588b9bf65d2cc6429e36180b06c741fd4d62a556e30efa58b4cc9",
    ),
}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(a):
    cv2.setNumThreads(1)
    if (
        sha(SOURCE) != "c51dace59349914da401356984de4e83da658c878011db3f54ff233077152934"
        or sha(CARRIER) != "731cc34e1fcb7c0874cd5722ab63241a123d31fbcd699e199cee4163cdcbc0dc"
        or sha(BALL) != "16214ff3ac718e6b28ae3cb21778c8440057c1e68f030e5ac904f20528f18883"
    ):
        raise ValueError("paid input or source identity")
    paid_joint = {
        (
            ASSET_ROOT / "ps-al028-joint25-packets-20261122/rally.psj"
        ): "fb69663af77820efc3d84c65ec314a15948805ab1f3bc70571fcab0ee2d1e1bd",
        (
            ASSET_ROOT / "ps-nov22-al028-exact-budget/joint-53.5/rally.psj"
        ): "ab12294fec2bba5bb5b21bf373a3a31c620b80cf9463cd92ab0383647d1a8aa9",
        (
            ASSET_ROOT / "ps-al028-joint25-crf52-20261122/rally.psj"
        ): "4a53f55b07b8a2371ee54e8d57c4834888e1f46761b8ea53a9871cab3adcec9b",
        (
            ASSET_ROOT / "ps-nov22-al028-exact-budget/joint-53.5/stream.ivf"
        ): "8d96056a21002279b05e9817e411d960e96b76610da6e932acb12a55093db845",
        (
            ASSET_ROOT / "ps-nov22-al028-exact-budget/ordinary-52.75/stream.ivf"
        ): "32e9379693742bf3ddd2324c14e31dfafdf8f7c7fd52230ca8b317d197cf76bf",
    }
    if any(sha(path) != digest for path, digest in paid_joint.items()):
        raise ValueError("paid joint or ordinary stream differs")
    if sha(a.finish / "finish.json") != a.finish_sha:
        raise ValueError("finisher receipt identity")
    report = json.loads((a.finish / "finish.json").read_text())
    if (
        report["ticks"] != list(range(240))
        or report["order"] != list(UNITS)
        or report["carrier_sha256"] != sha(CARRIER)
    ):
        raise ValueError("finisher binding")
    profile, _, geom = finish.geometry(CARRIER.read_bytes(), sha(CARRIER))
    crops = {}
    masks = {}
    for unit, archive in zip(UNITS, (a.gen4, a.gen6), strict=True):
        row = next(x for x in report["rows"][0]["status"] if x == unit)
        if row != unit:
            raise ValueError("unit mismatch")
        path = archive / (unit + ".npz")
        seal = archive / "generation.json"
        recorded = report["generated"][unit]["sha256"]
        recorded_seal = report["generation_receipts"][unit]["sha256"]
        if sha(path) != recorded or sha(seal) != recorded_seal:
            raise ValueError("generation seal")
        with np.load(path, allow_pickle=False) as z:
            paid_ticks, paid_keypoints, local_boxes = geom[unit]
            sealed_boxes = z["box"]
            if not np.array_equal(z["t"], np.arange(240)) or not np.allclose(
                sealed_boxes, local_boxes, rtol=0, atol=1e-10
            ):
                raise ValueError("paid geometry")
            # Render using the exact Linux boxes bound by the sealed generation;
            # cross-platform floating-point reconstruction can differ by ~1e-13.
            geom[unit] = (paid_ticks, paid_keypoints, sealed_boxes)
            crops[unit] = z["crops"]
        mask = a.finish / (unit + "-masks.npz")
        if sha(mask) != report["mask_files"][unit]:
            raise ValueError("mask seal")
        with np.load(mask, allow_pickle=False) as z:
            if not np.array_equal(z["t"], np.arange(240)):
                raise ValueError("mask ticks")
            masks[unit] = z["masks"]
    bases = {}
    for mode, (path, w, h, digest) in BASES.items():
        if sha(path) != digest or path.stat().st_size != 240 * w * h * 3 // 2:
            raise ValueError("pinned base identity")
        bases[mode] = (np.memmap(path, np.uint8, "r", shape=(240, h * 3 // 2, w)), w, h)
    if (a.dcvc is None) != (a.dcvc_sha is None):
        raise ValueError("DCVC path and expected native YUV digest must be supplied together")
    baselines = dict(BASELINES)
    if a.dcvc is not None:
        baselines["dcvc_q24"] = (a.dcvc, 960, 544, 20.411666666666665, a.dcvc_sha)
    controls = {}
    for name, (path, w, h, rate, digest) in baselines.items():
        if sha(path) != digest or path.stat().st_size != 240 * w * h * 3 // 2:
            raise ValueError("baseline native decode identity")
        controls[name] = (np.memmap(path, np.uint8, "r", shape=(240, h * 3 // 2, w)), w, h, rate)
    positions, looks = ball.decode_frames(BALL.read_bytes(), 240, 960, 544)
    velocity = ball.velocities(positions)
    a.out.mkdir(exist_ok=False)
    dest = {
        k: np.lib.format.open_memmap(
            a.out / f"frames-{k}.npy", mode="w+", dtype=np.uint8, shape=(a.frames, 1080, 1920, 3)
        )
        for k in a.save_rates
    }
    cap = cv2.VideoCapture(str(SOURCE), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise ValueError("source decoder")
    sse = {k: 0 for k in [*BASES, *baselines]}
    hashes = {k: hashlib.sha256() for k in BASES}
    start = time.monotonic()
    review = a.out / "private-review"
    review.mkdir()
    review_hashes = {}
    review_ticks = set(range(10, 231, 20))

    def save_image(tick, name, image):
        folder = review / f"{tick:03d}"
        folder.mkdir(exist_ok=True)
        path = folder / (name + ".png")
        pixels = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if image.ndim == 3 else image
        if not cv2.imwrite(str(path), pixels):
            raise ValueError("private frame write failed")
        review_hashes[f"{tick:03d}/{name}"] = sha(path)

    try:
        for t in range(a.frames):
            ok, bgr = cap.read()
            if not ok:
                raise ValueError("source ended")
            truth = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if t in review_ticks:
                save_image(t, "source", truth)
            active = {u: (crops[u][t], geom[u][2][t]) for u in UNITS}
            mm = {u: masks[u][t] for u in UNITS}
            if t in review_ticks:
                for i, u in enumerate(UNITS):
                    projected = finish.sam.project(mm[u], geom[u][2][t])
                    save_image(t, f"receiver-mask-{i + 1}", projected.astype(np.uint8) * 255)
            for k, (yuv, w, h) in bases.items():
                rgb = yuv_to_rgb(yuv[t], w, h, False)
                if k == 11:
                    plain = cv2.resize(rgb, (1920, 1080), interpolation=cv2.INTER_CUBIC)
                    witness, _ = finish.compose_tick(plain, list(UNITS), active, mm)
                    if replay.array_sha(witness) != report["rows"][t]["frame_sha256"]:
                        raise ValueError("recomposed source-denied witness differs")
                scene = cv2.resize(rgb, (960, 544), interpolation=cv2.INTER_CUBIC)
                scene = ball.render(scene, t, positions, velocity, looks)
                display = cv2.resize(scene, (1920, 1080), interpolation=cv2.INTER_CUBIC)
                output, _ = finish.compose_tick(display, list(UNITS), active, mm)
                if k in dest:
                    dest[k][t] = output
                if t in review_ticks:
                    save_image(t, f"joint-{k}", output)
                hashes[k].update(output)
                delta = truth.astype(np.int32) - output.astype(np.int32)
                sse[k] += int(np.sum(delta * delta, dtype=np.int64))
            for name, (yuv, w, h, _) in controls.items():
                rgb = cv2.resize(
                    yuv_to_rgb(yuv[t], w, h, False), (1920, 1080), interpolation=cv2.INTER_CUBIC
                )
                if t in review_ticks:
                    save_image(t, name, rgb)
                delta = truth.astype(np.int32) - rgb.astype(np.int32)
                sse[name] += int(np.sum(delta * delta, dtype=np.int64))
            if t == 23 and a.frames == 240 and 1.5 * (time.monotonic() - start) * 10 + 300 > 1800:
                raise ValueError("24-frame full-stage forecast exceeds 1200 seconds")
    finally:
        cap.release()
    for out in dest.values():
        out.flush()
    samples = a.frames * 1080 * 1920 * 3
    result = {
        "scope": "Single Al028 shot, native 25fps; original 11-kbps source-denied player generation with paired 20- and 21-kbps base recompositions. Full-frame RGB PSNR is descriptive. No reviewed foreground masks or ball accuracy used.",
        "frames": a.frames,
        "sse": sse,
        "rgb_psnr": {str(k): 10 * math.log10(255**2 * samples / v) for k, v in sse.items()},
        "rgb_stream_sha256": {str(k): v.hexdigest() for k, v in hashes.items()},
        "baseline_kbps": {k: v[3] for k, v in controls.items()},
        "joint_warm_kbps": {"11": 11.133333333333333, "20": 20.4075, "21": 21.298333333333332},
        "saved_full_rates": a.save_rates,
        "finish_receipt_sha256": a.finish_sha,
        "paid_stream_sha256": {str(path): digest for path, digest in paid_joint.items()},
        "private_review_ticks": sorted(t for t in review_ticks if t < a.frames),
        "private_review_png_sha256": review_hashes,
        "elapsed_seconds": time.monotonic() - start,
        "order": "decoded base, fitted ball, far player, near player",
    }
    (a.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--finish", type=Path, required=True)
    p.add_argument("--finish-sha", required=True)
    p.add_argument("--gen4", type=Path, required=True)
    p.add_argument("--gen6", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--frames", type=int, choices=(24, 240), default=240)
    p.add_argument("--dcvc", type=Path)
    p.add_argument("--dcvc-sha")
    p.add_argument("--save-rates", nargs="+", type=int, choices=(11, 20, 21), default=[11, 20, 21])
    main(p.parse_args())
