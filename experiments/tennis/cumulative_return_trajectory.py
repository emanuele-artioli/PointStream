"""Fit an actual paid target trajectory on the previously selected sender positions."""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from src.codecs import svtav1
from src.pointstream import ball, tracked_patch

SOURCE_SHA = "f0551443eaff4e4b3539f21cd0c6ac66f55601d480ad099a571e8626db735e4a"
TRACK_SHA = "43a816146a72daebfab62db5a87dcb651d431aa408feb859f7df9b7dc60f0859"
BASE_SHA = "2fa1f8d752b527707a5e0c1e1be6e070ef34efb91a37aa8ae20ca66215ec2ab9"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def run(args):
    assert len(os.sched_getaffinity(0)) == 1 and os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert sha(args.source) == SOURCE_SHA and sha(args.track) == TRACK_SHA
    n = args.frames
    assert n == (24 if os.environ["PS_STAGE"] == "smoke" else 240)
    import tarfile

    with tarfile.open(args.base) as tar:
        data = tar.extractfile("publish/clean-crf60.ivf").read()
    assert hashlib.sha256(data).hexdigest() == BASE_SHA
    work = Path(os.environ["PS_SCRATCH_DIR"]) / "fit-trajectory"
    work.mkdir()
    started = time.monotonic()
    ff = svtav1.tool("ffmpeg")
    dec = svtav1.tool("dav1d")
    assert ff["sha256"] == "7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2"
    assert dec["sha256"] == "e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6"
    source = work / "source.yuv"
    subprocess.run(
        [
            ff["path"],
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(args.source),
            "-frames:v",
            str(n),
            "-vf",
            "scale=960:544:flags=area:out_range=tv,format=yuv420p",
            "-filter_threads",
            "1",
            "-fps_mode",
            "passthrough",
            "-threads",
            "1",
            "-f",
            "rawvideo",
            str(source),
        ],
        check=True,
        timeout=180,
    )
    assert source.stat().st_size == n * 960 * 816
    native = work / "base.yuv"
    tracked_patch.decode_native(
        data, native, {"width": 240, "height": 136, "fps": [50, 1], "frames": 240}, dec["real_path"]
    )
    assert sha(native) == "ab8d5890ae3d0365e5dab07e9ee8f2eba934feb165b4cc1a20e35b20be2a5b5c"
    source_yuv = np.memmap(source, np.uint8, "r", shape=(n, 816, 960))
    base_yuv = np.memmap(native, np.uint8, "r", shape=(240, 204, 240))
    source_rgb = np.memmap(work / "source.rgb", np.uint8, "w+", shape=(n, 544, 960, 3))
    base_rgb = np.memmap(work / "base.rgb", np.uint8, "w+", shape=(n, 544, 960, 3))
    for t in range(n):
        source_rgb[t] = tracked_patch.yuv_to_rgb(source_yuv[t], 960, 544, False)
        base_rgb[t] = cv2.resize(
            tracked_patch.yuv_to_rgb(base_yuv[t], 240, 136, False),
            (960, 544),
            interpolation=cv2.INTER_CUBIC,
        )
    raw = json.loads(args.track.read_text())["centers"][:n]
    track = [
        None if p is None else (min(959, round(p[0] * 0.5)), min(543, round(p[1] * 544 / 1080)))
        for p in raw
    ]
    detections = sum(p is not None for p in track)
    assert detections > 0
    fit_start = time.monotonic()
    crops = ball.FrameCrops.build(track, source_rgb, [base_rgb])
    (fitted,) = ball.fit_frames(crops, ball.grid("streak"), (544, 960))
    fitted_track, appearances = fitted["track"], fitted["appearances"]
    elapsed_fit = time.monotonic() - fit_start
    payload = ball.encode_frames(fitted_track, appearances, 960, 544)
    decoded_track, decoded_appearance = ball.decode_frames(payload, n, 960, 544)
    assert decoded_track == fitted_track and decoded_appearance == appearances
    stage = Path(os.environ["PS_STAGE_DIR"])
    (stage / "trajectory.bin").write_bytes(payload)
    report = {
        "frames": n,
        "sender_positions": detections,
        "fitted_positions": sum(p is not None for p in fitted_track),
        "payload_bytes": len(payload),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "fit_seconds": elapsed_fit,
        "elapsed_seconds": time.monotonic() - started,
        "source_sha256": SOURCE_SHA,
        "track_sha256": TRACK_SHA,
        "base_sha256": BASE_SHA,
        "canvas": [960, 544],
        "fps": [50, 1],
        "method": "source-fitted streak; source-defined sender positions, no ball labels",
    }
    (stage / "trajectory-report.json").write_text(json.dumps(report, indent=2) + "\n")


def validate():
    stage = Path(os.environ["PS_STAGE_DIR"])
    j = json.loads((stage / "trajectory-report.json").read_text())
    data = (stage / "trajectory.bin").read_bytes()
    smoke = os.environ["PS_STAGE"] == "smoke"
    n = 24 if smoke else 240
    tracks, appearances = ball.decode_frames(data, n, 960, 544)
    forecast = 1.5 * j["fit_seconds"] * 231 / j["sender_positions"] + 120 if smoke else None
    checks = {
        "frames": j["frames"] == n and len(tracks) == len(appearances) == n,
        "inputs": (j["source_sha256"], j["track_sha256"], j["base_sha256"])
        == (SOURCE_SHA, TRACK_SHA, BASE_SHA),
        "payload": hashlib.sha256(data).hexdigest() == j["payload_sha256"]
        and len(data) == j["payload_bytes"],
        "forecast": not smoke or forecast <= 900,
    }
    result = {"passed": all(checks.values()), "checks": checks, "forecast_seconds": forecast}
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps(result, indent=2))
    assert result["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path)
    parser.add_argument("--track", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    validate() if args.validate else run(args)
