"""Frozen WASB on the strongest affordable 640p and native AV1 arms."""

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tarfile
import time
import zlib

import cv2
import numpy as np

from experiments.tennis import ball_confirmation_score as scorer
from experiments.tennis import k5b_trackers as trackers
from experiments.tennis import k5d_source_oracle as oracle
from experiments.tennis.ball_starved_scaled import render_scaled, unpack_scaled
from src.codecs import svtav1
from src.pointstream import tracked_patch

WEIGHT = "9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb"
CONFIG = "6dbcad60e207bf684d0094f59aa922b107be1c1ce83998dc5b167ed59b0d1e0d"


def contents(path, name):
    with tarfile.open(path) as archive:
        member = archive.extractfile(name)
        if member is None:
            raise ValueError(f"missing {name}")
        return member.read()


def wire(data):
    compressed = zlib.compress(data, 6)
    return b"\x01" + compressed if len(compressed) < len(data) else b"\x00" + data


def run(args):
    import torch

    start = time.monotonic()
    if torch.cuda.device_count() != 1 or len(os.sched_getaffinity(0)) != 1:
        raise ValueError("GPU/CPU isolation")
    torch.set_num_threads(1)
    oracle.bind_config(args.wasb_code, CONFIG)
    if oracle.sha(args.wasb_weights / "wasb_tennis_best.pth.tar") != WEIGHT:
        raise ValueError("checkpoint identity")
    sources = [(640, args.scaled), (1280, args.native)]
    selected = []
    for width, path in sources:
        report = json.loads(contents(path, "publish/result.json"))
        if report["source"] != args.match + "_000":
            raise ValueError("source mismatch")
        candidates = report["selected"] if width == 640 else report["results"]
        for mode in ("plain", "roi"):
            choices = []
            for row in candidates:
                if row["mode"] != mode:
                    continue
                data = contents(path, "publish/" + row["file"])
                raw = hashlib.sha256(data).hexdigest()
                if raw != row["raw_sha256" if width == 640 else "sha256"]:
                    raise ValueError("IVF identity")
                encoded = wire(data)
                if width == 640 and (
                    hashlib.sha256(encoded).hexdigest() != row["wire_sha256"]
                    or len(encoded) != row["bytes"]
                ):
                    raise ValueError("640 wire identity")
                rate = len(encoded) * 8 * report["fps"] / report["frames"] / 1000
                score = row.get("mean_dists")
                if score is None:
                    score = sum(r["dists"] for r in row["crop"]) / len(row["crop"])
                if rate <= args.cap + 1e-9:
                    choices.append((score, row, data, rate, report))
            if choices:
                score, row, data, rate, report = min(choices, key=lambda item: item[0])
                selected.append(
                    {
                        "width": width,
                        "mode": mode,
                        "row": row,
                        "data": data,
                        "wire_kbps": rate,
                        "ball_dists": score,
                        "fps": report["fps"],
                        "frames": report["frames"],
                    }
                )
    if not selected:
        raise ValueError("no affordable anchors")
    tile_report = json.loads(contents(args.tile, "publish/result.json"))
    (tile_row,) = [
        r for r in tile_report["results"] if r["mode"] == "tracked" and r["sha256"] == args.tile_sha
    ]
    tile_data = contents(args.tile, "publish/" + tile_row["file"])
    if hashlib.sha256(tile_data).hexdigest() != args.tile_sha:
        raise ValueError("tile packet identity")
    tile_rate = len(wire(tile_data)) * 8 * tile_report["fps"] / tile_report["frames"] / 1000
    if abs(tile_rate - args.cap) > 1e-7:
        raise ValueError("tiled cap identity")
    manifest, raw = scorer.authenticated_manifest(args.labels)
    if (
        hashlib.sha256(raw).hexdigest()
        != "df0dabf48b61ed3a096a9a6b4e54d823ca3b279c86d6a09e69ef0cd9bf2ceeb1"
    ):
        raise ValueError("label manifest")
    (item,) = [v for v in manifest["sources"] if v["id"] == args.match + "_000"]
    labels = [r for r in item["official_window_rows"] if r["native_frame_index"] < args.frames]
    root = Path(os.environ["PS_SCRATCH_DIR"])
    work = root / "anchor-judge"
    work.mkdir()
    decoder = svtav1.tool("dav1d")
    if decoder["sha256"] != "e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6":
        raise ValueError("decoder identity")
    model_start = time.monotonic()
    zoo = trackers.Zoo(args.wasb_code, args.wasb_weights, names=("wasb-step1",))
    model_seconds = time.monotonic() - model_start
    settings = zoo.settings["wasb-step1"]
    if next(settings["net"].parameters()).device.type != "cuda":
        raise ValueError("WASB CUDA")
    results = []
    for item in selected:
        width = item["width"]
        height = 360 if width == 640 else 720
        n = item["frames"]
        folder = work / f"{width}-{item['mode']}"
        folder.mkdir()
        rawpath = folder / "decoded.yuv"
        metadata = {"width": width, "height": height, "fps": [item["fps"], 1], "frames": n}
        rec = tracked_patch.decode_native(item["data"], rawpath, metadata, decoder["path"])
        if rec["decoded_yuv_sha256"] != item["row"]["decoded_sha256"]:
            raise ValueError("decoded hash")
        frames = np.memmap(rawpath, np.uint8, "r", shape=(n, height * 3 // 2, width))
        images = []
        for f in frames[: args.frames]:
            rgb = tracked_patch.yuv_to_rgb(f, width, height, False)
            images.append(
                cv2.resize(rgb, (1280, 720), interpolation=cv2.INTER_CUBIC)
                if width != 1280
                else rgb
            )
        images = np.stack(images)
        rgb_hash = hashlib.sha256(images.tobytes()).hexdigest()
        infer = time.monotonic()
        pred = zoo.run("wasb-step1", images)
        infer = time.monotonic() - infer
        rows = []
        for label in labels:
            tick = label["native_frame_index"]
            present, x, y = pred[tick]
            error = (
                math.hypot(1.5 * x - label["native_X"], 1.5 * y - label["native_Y"])
                if present and label["visibility"] == 1
                else None
            )
            rows.append(
                {
                    "tick": tick,
                    "visibility": label["visibility"],
                    "detected": bool(present),
                    "native_error": error,
                    "hit_10px": error is not None and error <= 10,
                }
            )
        results.append(
            {
                "resolution": width,
                "mode": item["mode"],
                "wire_kbps": item["wire_kbps"],
                "ball_dists": item["ball_dists"],
                "stream_sha256": hashlib.sha256(item["data"]).hexdigest(),
                "decoded_sha256": rec["decoded_yuv_sha256"],
                "rgb_sha256": rgb_hash,
                "rows": rows,
                "seconds": infer,
            }
        )
    folder = work / "tile"
    folder.mkdir()
    if args.tile_rgb:
        saved = json.loads(contents(args.tile_rgb, "publish/result.json"))
        blob = contents(args.tile_rgb, "publish/frames.npz")
        if (
            hashlib.sha256(blob).hexdigest() != saved["npz_sha256"]
            or saved["packet_sha256"] != args.tile_sha
        ):
            raise ValueError("origin RGB archive identity")
        with np.load(io.BytesIO(blob), allow_pickle=False) as stored:
            images = stored["frames"]
        if images.shape != (tile_report["frames"], 720, 1280, 3):
            raise ValueError("origin RGB frame geometry")
    else:
        images = render_scaled(tile_data, folder, decoder["path"])
    rgb_hash = hashlib.sha256(images.tobytes()).hexdigest()
    if rgb_hash != tile_row["rendered_sha256"]:
        raise ValueError("tile reconstructed hash")
    meta, _, _ = unpack_scaled(tile_data)
    positions = meta["positions"]
    infer = time.monotonic()
    pred = zoo.run("wasb-step1", images[: args.frames])
    infer = time.monotonic() - infer
    rows = []
    for label in labels:
        tick = label["native_frame_index"]
        present, x, y = pred[tick]
        pos = positions[tick]
        error = (
            math.hypot(1.5 * x - label["native_X"], 1.5 * y - label["native_Y"])
            if present and label["visibility"] == 1
            else None
        )
        direct = (
            math.hypot(
                1.5 * (pos[0] + 16) - label["native_X"], 1.5 * (pos[1] + 16) - label["native_Y"]
            )
            if pos is not None and label["visibility"] == 1
            else None
        )
        rows.append(
            {
                "tick": tick,
                "visibility": label["visibility"],
                "detected": bool(present),
                "native_error": error,
                "hit_10px": error is not None and error <= 10,
                "direct_present": pos is not None,
                "direct_hit_10px": direct is not None and direct <= 10,
            }
        )
    results.append(
        {
            "resolution": 320,
            "mode": "tracked",
            "wire_kbps": tile_rate,
            "ball_dists": tile_row["mean_dists"],
            "stream_sha256": args.tile_sha,
            "rgb_sha256": rgb_hash,
            "rows": rows,
            "seconds": infer,
        }
    )
    report = {
        "match": args.match,
        "frames": args.frames,
        "source_frames": selected[0]["frames"],
        "fps": selected[0]["fps"],
        "phase": os.environ["PS_STAGE"],
        "cap_kbps": args.cap,
        "checkpoint_sha256": WEIGHT,
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "settings": {
            "frames_in": settings["cfg"]["model"]["frames_in"],
            "frames_out": settings["cfg"]["model"]["frames_out"],
            "step": settings["cfg"]["detector"]["step"],
            "threshold": settings["cfg"]["detector"]["postprocessor"]["score_threshold"],
        },
        "model_seconds": model_seconds,
        "elapsed_seconds": time.monotonic() - start,
        "results": results,
    }
    publish = root / "publish"
    publish.mkdir()
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    (publish / "result.json").write_text(payload)
    (Path(os.environ["PS_STAGE_DIR"]) / "result.json").write_text(payload)


def validate(args):
    r = json.loads((Path(os.environ["PS_STAGE_DIR"]) / "result.json").read_text())
    checks = {
        "source": r["match"] == args.match,
        "checkpoint": r["checkpoint_sha256"] == WEIGHT,
        "cuda": r["capability"] == [8, 9],
        "settings": r["settings"] == {"frames_in": 3, "frames_out": 3, "step": 1, "threshold": 0.5},
        "affordable": all(x["wire_kbps"] <= args.cap + 1e-9 for x in r["results"]),
        "labels": all(len(x["rows"]) > 0 for x in r["results"]),
        "arms": len(r["results"]) >= 1,
    }
    if r["phase"] == "smoke":
        checks["forecast"] = (
            r["model_seconds"]
            + 1.5 * sum(x["seconds"] for x in r["results"]) * (r["source_frames"] / r["frames"])
            + 180
            < args.full_limit
        )
    report = {"passed": all(checks.values()), "checks": checks}
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps(report) + "\n")
    if not report["passed"]:
        raise SystemExit(str(checks))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("scaled", "native", "tile", "labels", "wasb-code", "wasb-weights"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--match", required=True)
    p.add_argument("--cap", type=float, required=True)
    p.add_argument("--tile-sha")
    p.add_argument("--tile-rgb", type=Path)
    p.add_argument("--frames", type=int)
    p.add_argument("--full-limit", type=int, required=True)
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    validate(args) if args.validate else run(args)


if __name__ == "__main__":
    main()
