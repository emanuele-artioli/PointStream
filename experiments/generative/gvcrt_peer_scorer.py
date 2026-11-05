"""Prepare pinned K5d source and score GVC reconstructions in a separate process.

Run as a direct script with PYTHONPATH pointing at the historical K5d snapshot.
The score process is deliberately separate from the codec receiver.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

TRACKNET_SHA256 = "bb45372d011ef4092d78173169c8ad6280d7dbdb9998efaf2b1c72a63ca1dd18"
YUV_SHA256 = "c82d2e12e18e19f0453cb4d1b497581841c423149b153a1fc4ba2de6ff4353f4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoded_rgb(array: np.ndarray, count: int) -> np.ndarray:
    if count not in (4, 16) or array.shape != (count, 3, 720, 1280):
        raise ValueError("decoded array must be NCHW with 4 or 16 1280x720 frames")
    if not np.issubdtype(array.dtype, np.floating) or not np.isfinite(array).all():
        raise ValueError("decoded array must contain finite floats")
    return np.moveaxis(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8), 1, -1)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(args: argparse.Namespace) -> None:
    from PIL import Image

    from experiments.tennis import k5b_trackers as trk
    from experiments.tennis import k5d_encoder, windows

    if sha256(args.archive) != TRACKNET_SHA256:
        raise ValueError("TrackNet archive SHA256 mismatch")
    parts = args.clip.split("/")
    if args.clip != "tracknet/game10/Clip3":
        raise ValueError("clip must be tracknet/game10/Clip3")
    found = [c for c in trk.tracknet_clips(args.archive, [int(parts[1][4:])]) if c["id"] == args.clip]
    if len(found) != 1 or len(found[0]["rows"]) < 16:
        raise ValueError("clip missing, ambiguous, or shorter than 16 frames")
    clip = {**found[0], "rows": found[0]["rows"][:16]}
    members = []
    with zipfile.ZipFile(args.archive) as archive:
        for row in clip['rows']:
            name = f"{clip['dir']}/{row[0]}"
            raw = archive.read(name)
            members.append({'path': name, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)})
    if len({x['path'] for x in members}) != 16:
        raise ValueError('duplicate source JPEG member')
    args.output.mkdir(parents=True, exist_ok=False)
    source = trk.tracknet_frames(args.archive, clip)
    if source.shape != (16, 720, 1280, 3) or source.dtype != np.uint8:
        raise ValueError("source frame geometry mismatch")
    np.save(args.output / "source_rgb.npy", source)
    conversion = k5d_encoder.convert_rgb(args.ffmpeg, source, args.output / "source.yuv", Fraction(30, 1))
    if sha256(args.output / "source.yuv") != YUV_SHA256:
        raise ValueError("converted 16-frame YUV SHA256 mismatch")
    converted = windows.rgb_frames(args.output / "source.yuv", 1280, 720, 16, args.output / "converted_rgb.npy")
    frame_dir = args.output / "frames"
    frame_dir.mkdir()
    frames = []
    for index, frame in enumerate(converted):
        path = frame_dir / f"{index:06d}.png"
        Image.fromarray(frame).save(path)
        frames.append({"path": str(path.resolve()), "sha256": sha256(path)})
    write_json(args.output / "frames.json", frames)
    write_json(args.output / "labels.json", {"clip": args.clip, "labels": trk.tracknet_labels(clip)})
    write_json(args.output / "prepare_receipt.json", {
        "clip": args.clip, "frames": 16, "fps": 30, "size": [1280, 720],
        "tracknet_sha256": TRACKNET_SHA256, "source_yuv_sha256": YUV_SHA256,
        "source_rgb_sha256": sha256(args.output / "source_rgb.npy"),
        "converted_rgb_sha256": sha256(args.output / "converted_rgb.npy"),
        "labels_sha256": sha256(args.output / "labels.json"),
        "frames_manifest_sha256": sha256(args.output / "frames.json"), "source_members": members,
        "conversion_floor_psnr": k5d_encoder.pooled_psnr(source, converted), "conversion_command": conversion,
    })


def score(args: argparse.Namespace) -> None:
    import torch

    from experiments.tennis import k5b_trackers as trk
    from experiments.tennis import k5d_encoder
    from src.codecs import quality

    start = time.monotonic()
    config = json.loads(args.config.read_text())
    source = np.load(args.source, allow_pickle=False, mmap_mode="r")
    if source.dtype != np.uint8 or source.shape != (16, 720, 1280, 3):
        raise ValueError("source must be the prepared 16-frame RGB array")
    source = source[:args.count]
    images = decoded_rgb(np.load(args.decoded, allow_pickle=False, mmap_mode="r"), args.count)
    label_doc = json.loads(args.labels.read_text())
    labels = [tuple(row) for row in label_doc["labels"][:args.count]]
    if len(labels) != args.count:
        raise ValueError("label count mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    nets = (quality.load_lpips(Path(config["lpips_backbone"]), "cuda"),
            quality.load_dists(Path(config["dists_backbone"]), "cuda"))
    tracker = trk.BallTrack(Path(config["balltrack_code"]), Path(config["balltrack_weight"]))
    zoo = trk.Zoo(Path(config["wasb_code"]), Path(config["wasb_weights"]), names=("wasb-step1",))
    models = (*nets, tracker.net, zoo.settings["wasb-step1"]["net"])
    if any(next(model.parameters()).device.type != "cuda" for model in models):
        raise ValueError("metric or judge model is not on CUDA")
    loaded = time.monotonic()
    metrics = k5d_encoder.metrics(source, images, labels, nets)
    metric_done = time.monotonic()
    wasb = k5d_encoder.prediction_rows(labels, zoo.run("wasb-step1", images))
    wasb_done = time.monotonic()
    balltrack = k5d_encoder.prediction_rows(labels, tracker.run(images))
    done = time.monotonic()
    write_json(args.output, {"clip": label_doc["clip"], "frames": args.count, "metrics": metrics,
              "wasb": wasb, "balltrack": balltrack,
              "inputs_sha256": {"source": sha256(args.source), "decoded": sha256(args.decoded),
                                "labels": sha256(args.labels), "config": sha256(args.config)},
              "metric_models": {"lpips": quality.lpips_record(Path(config["lpips_backbone"])),
                                "dists": quality.dists_record(Path(config["dists_backbone"]))},
              "device": {"name": torch.cuda.get_device_name(0), "count": torch.cuda.device_count(),
                         "visible": __import__("os").environ.get("CUDA_VISIBLE_DEVICES")},
              "timings_seconds": {"load": loaded - start, "metrics": metric_done - loaded,
                                  "wasb": wasb_done - metric_done, "balltrack": done - wasb_done,
                                  "total": done - start}})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--archive", type=Path, required=True)
    prep.add_argument("--clip", required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--ffmpeg", default="ffmpeg")
    sc = sub.add_parser("score")
    for name in ("source", "decoded", "labels", "config", "output"):
        sc.add_argument(f"--{name}", type=Path, required=True)
    sc.add_argument("--count", type=int, choices=(4, 16), required=True)
    args = parser.parse_args()
    {"prepare": prepare, "score": score}[args.mode](args)


if __name__ == "__main__":
    main()
