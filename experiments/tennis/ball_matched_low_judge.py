"""WASB scores complete matched-ball packets using decoded pictures only."""
import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

import numpy as np

from experiments.tennis import k5b_trackers as trackers, k5d_source_oracle as oracle
from experiments.tennis.windows import yuv_to_rgb
from src.codecs import svtav1
from src.pointstream import ball, ball_stream

WIDTH, HEIGHT, TOTAL = 1280, 720, 292
WEIGHT = "9d391239ab10c733f8e5bfadf16ab72838e7a8ebc88e8ae2038501c03d42b4bb"
CONFIG = "6dbcad60e207bf684d0094f59aa922b107be1c1ce83998dc5b167ed59b0d1e0d"
DECODER = "e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6"


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def read_member(tar, name):
    handle = tar.extractfile(name)
    if handle is None:
        raise ValueError("Missing member " + name)
    return handle.read()


def make_frames(packed, count, root, decoder):
    stream = ball_stream.unpack(packed)
    if stream.frames != TOTAL or (stream.width, stream.height) != (WIDTH, HEIGHT):
        raise ValueError("Invalid saved packet geometry")
    ivf, yuv = root / "native.ivf", root / "native.yuv"
    ivf.write_bytes(stream.native)
    command = svtav1.decode_command(decoder, ivf, yuv, threads=1)
    subprocess.run(command, check=True, capture_output=True, timeout=100)
    if yuv.stat().st_size != TOTAL * WIDTH * HEIGHT * 3 // 2:
        raise ValueError("Native decode incomplete")
    native = np.memmap(yuv, dtype=np.uint8, mode="r",
                       shape=(TOTAL, HEIGHT * 3 // 2, WIDTH))
    rgb = np.lib.format.open_memmap(root / "judge-input.npy", mode="w+", dtype=np.uint8,
                                    shape=(count, HEIGHT, WIDTH, 3))
    track, apps = ball.decode_frames(stream.ball_payload, TOTAL, WIDTH, HEIGHT)
    velocity = ball.velocities(track)
    rgb_hash = hashlib.sha256()
    for t in range(count):
        frame = yuv_to_rgb(native[t], WIDTH, HEIGHT)
        rgb[t] = ball.render(frame, t, track, velocity, apps)
        rgb_hash.update(rgb[t].tobytes())
    rgb.flush()
    return rgb, rgb_hash.hexdigest()


def run(args):
    import torch

    started = time.monotonic()
    if torch.cuda.device_count() != 1 or len(os.sched_getaffinity(0)) != 1:
        raise ValueError("Expected one claimed GPU and one CPU")
    oracle.bind_config(args.wasb_code, CONFIG)
    if oracle.sha(args.wasb_weights / "wasb_tennis_best.pth.tar") != WEIGHT:
        raise ValueError("WASB checkpoint mismatch")
    decoder = svtav1.tool("dav1d")
    if decoder["sha256"] != DECODER:
        raise ValueError("Native decoder mismatch")
    with tarfile.open(args.packets) as archive:
        report = json.loads(read_member(archive, "publish/report.json"))
        if not report["within_half_percent"] or report["frames"] != TOTAL:
            raise ValueError("CPU rate matching did not succeed")
        index = next(i for i, row in enumerate(report["trials"])
                     if row["packet_sha256"] == report["best"]["packet_sha256"])
        packets = {
            "baseline": read_member(archive, "publish/historical-baseline.psbl"),
            "candidate": read_member(archive, f"publish/candidate-{index}.psbl"),
        }
    if digest(packets["baseline"]) != report["baseline_sha256"] or digest(packets["candidate"]) != report["best"]["packet_sha256"]:
        raise ValueError("Paid packet hash differs from CPU receipt")
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    model_start = time.monotonic()
    zoo = trackers.Zoo(args.wasb_code, args.wasb_weights, names=("wasb-step1",))
    if next(zoo.settings["wasb-step1"]["net"].parameters()).device.type != "cuda":
        raise ValueError("WASB model did not load on CUDA")
    model_seconds = time.monotonic() - model_start
    result = {}
    for name, packet in packets.items():
        root = scratch / name
        root.mkdir()
        frames, frame_hash = make_frames(packet, args.frames, root, decoder["path"])
        infer_start = time.monotonic()
        predictions = zoo.run("wasb-step1", frames)
        infer_seconds = time.monotonic() - infer_start
        if len(predictions) != args.frames:
            raise ValueError("Detector frame count mismatch")
        rows = [{"frame": i, "detected": bool(hit), "x": float(x), "y": float(y)}
                for i, (hit, x, y) in enumerate(predictions)]
        result[name] = {"packet_bytes": len(packet), "packet_sha256": digest(packet),
                        "decoded_rgb_sha256": frame_hash, "rows": rows, "seconds": infer_seconds}
    summary = {"stage": os.environ["PS_STAGE"], "frames": args.frames, "results": result,
               "detector": "wasb-step1", "judge_checkpoint_sha256": WEIGHT, "config_sha256": CONFIG,
               "decoder": decoder, "gpu": torch.cuda.get_device_name(0),
               "capability": list(torch.cuda.get_device_capability(0)),
               "model_seconds": model_seconds, "elapsed_seconds": time.monotonic() - started,
               "cpu_packet_receipt": report}
    publish = scratch / "publish"
    publish.mkdir()
    data = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    (publish / "judge.json").write_text(data)
    (Path(os.environ["PS_STAGE_DIR"]) / "judge.json").write_text(data)


def validate():
    stage = Path(os.environ["PS_STAGE_DIR"])
    data = json.loads((stage / "judge.json").read_text())
    expected = 24 if os.environ["PS_STAGE"] == "smoke" else TOTAL
    checks = {"frame_count": data["frames"] == expected,
              "checkpoint": data["judge_checkpoint_sha256"] == WEIGHT,
              "config": data["config_sha256"] == CONFIG,
              "cuda": data["capability"] == [8, 9],
              "both_arms": set(data["results"]) == {"baseline", "candidate"},
              "rate": data["cpu_packet_receipt"]["within_half_percent"]}
    for name in ("baseline", "candidate"):
        rows = data["results"][name]["rows"]
        checks[name] = len(rows) == expected and [r["frame"] for r in rows] == list(range(expected))
    if expected == 24:
        seconds = max(x["seconds"] for x in data["results"].values())
        checks["forecast"] = data["model_seconds"] + 1.5 * seconds * (TOTAL / 24) * 2 + 120 < 850
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps({"passed": all(checks.values()), "checks": checks}) + "\n")
    if not all(checks.values()):
        raise ValueError("Detector validation failed: " + str(checks))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--packets", type=Path)
    parser.add_argument("--wasb-code", type=Path)
    parser.add_argument("--wasb-weights", type=Path)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    validate() if args.validate else run(args)
