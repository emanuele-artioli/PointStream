"""Recompose two paid Al028 base rates using the same sealed player crops and masks.

Reads a source video solely for whole-frame RGB PSNR, never for reconstruction.
No human labels or sender-track accuracy are read or inferred.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from experiments.tennis import common_player_al028_finish as finish
from experiments.tennis import player_requalification as replay
from src.pointstream import ball
from src.pointstream.tracked_patch import yuv_to_rgb

UNITS = (
    "alcaraz_hurkacz__028_at120__person_000000000004",
    "alcaraz_hurkacz__028_at120__person_000000000006",
)
CARRIER = Path("/private/tmp/ps-al028-portable-20261122/players.pscp")
SOURCE = Path("/private/tmp/ps-al028-review-source-20261122.mp4")
BALL = Path("/private/tmp/ps-al028-ball-sender-full.payload")
BASES = {
    11: (
        Path("/private/tmp/ps-al028-native25-pinned.yuv"),
        160,
        96,
        "f2b9dd6a2370b3a6267a0379339360307bfe7900f58e7a503b8b2a2b1756bafb",
    ),
    21: (
        Path("/private/tmp/ps-al028-native320-pinned.yuv"),
        320,
        192,
        "0474cc789810859513295a1bc41c357fb286ad60414864ba6e31b27d1f22e965",
    ),
}
BASELINES = {
    "av1_320_q52": (
        Path("/private/tmp/ps-al028-av1-target-20261122/320-crf52/decoded.yuv"),
        320,
        192,
        21.008333333333333,
        "dec4ea4740491d2eec30b6b94e67a154485bf8dc8bb390d0a1abb310e548194e",
    ),
    "av1_640_q62": (
        Path("/private/tmp/ps-al028-av1-refine-20261122/640-crf62/decoded.yuv"),
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
            if not np.array_equal(z["t"], np.arange(240)) or not np.array_equal(
                z["box"], geom[unit][2]
            ):
                raise ValueError("paid geometry")
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
    controls = {}
    for name, (path, w, h, rate, digest) in BASELINES.items():
        if sha(path) != digest or path.stat().st_size != 240 * w * h * 3 // 2:
            raise ValueError("baseline native decode identity")
        controls[name] = (np.memmap(path, np.uint8, "r", shape=(240, h * 3 // 2, w)), w, h, rate)
    positions, looks = ball.decode_frames(BALL.read_bytes(), 240, 960, 544)
    velocity = ball.velocities(positions)
    a.out.mkdir(exist_ok=False)
    dest = {
        k: np.lib.format.open_memmap(
            a.out / f"frames-{k}.npy", mode="w+", dtype=np.uint8, shape=(240, 1080, 1920, 3)
        )
        for k in BASES
    }
    cap = cv2.VideoCapture(str(SOURCE), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise ValueError("source decoder")
    sse = {k: 0 for k in [*BASES, *BASELINES]}
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
        for t in range(240):
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
            if t == 23 and 1.5 * (time.monotonic() - start) * 10 + 60 > 1200:
                raise ValueError("24-frame full-stage forecast exceeds 1200 seconds")
    finally:
        cap.release()
    for out in dest.values():
        out.flush()
    samples = 240 * 1080 * 1920 * 3
    result = {
        "scope": "Single Al028 shot, native 25fps; original 11-kbps source-denied player generation with paired 21-kbps base recomposition. Full-frame RGB PSNR is descriptive. No reviewed foreground masks or ball accuracy used.",
        "frames": 240,
        "sse": sse,
        "rgb_psnr": {str(k): 10 * math.log10(255**2 * samples / v) for k, v in sse.items()},
        "rgb_stream_sha256": {str(k): v.hexdigest() for k, v in hashes.items()},
        "baseline_kbps": {k: v[3] for k, v in controls.items()},
        "joint_warm_kbps": {"11": 11.133333333333333, "21": 21.298333333333332},
        "finish_receipt_sha256": a.finish_sha,
        "private_review_ticks": sorted(review_ticks),
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
    main(p.parse_args())
