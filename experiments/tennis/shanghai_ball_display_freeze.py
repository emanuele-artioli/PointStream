"""Hash complete source and selected PSBL/ordinary displays without reading labels."""

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import time
from pathlib import Path

import cv2
import numpy as np

from src.codecs import svtav1
from src.pointstream import ball, ball_stream, tracked_patch

SOURCE = "0afb0fb191881def1235285b58fe54814556c83fed7815cd99a00b92d26bc86c"


def digest(b):
    return hashlib.sha256(b).hexdigest()


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def member(path, name):
    with tarfile.open(path) as t:
        f = t.extractfile("publish/" + name)
        if f is None:
            raise ValueError("missing published member " + name)
        return f.read()


def run(a):
    start = time.monotonic()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "" and len(os.sched_getaffinity(0)) == 1
    cv2.setNumThreads(1)
    assert (
        sha(a.source) == SOURCE
        and sha(a.ordinary) == a.ordinary_sha256
        and sha(a.candidate) == a.candidate_sha256
    )
    ordinary = json.loads(member(a.ordinary, "report.json"))
    candidate = json.loads(member(a.candidate, "report.json"))
    assert (
        ordinary["frames"] == candidate["frames"] == 240
        and ordinary["source_sha256"] == candidate["source_sha256"] == SOURCE
        and ordinary["arm"] == "ordinary"
        and candidate["arm"] == "ball"
    )
    target = candidate["target_bytes"]
    assert target == 24480
    if a.decision == "primary":
        assert ordinary["target_bytes"] == target
    pairs = []
    for width in (320, 640):
        oo = [x["selected"] for x in ordinary["groups"] if x["width"] == width]
        bb = [x["selected"] for x in candidate["groups"] if x["width"] == width]
        if not oo or not bb:
            continue
        o, b = oo[0], bb[0]
        if a.decision == "secondary":
            assert ordinary["target_bytes"] == b["bytes"]
        if abs(o["bytes"] - b["bytes"]) / o["bytes"] <= 0.005 and (
            a.decision == "secondary" or abs(b["bytes"] - target) / target <= 0.005
        ):
            pairs.append((abs(b["bytes"] - target), -width, o, b))
    if not pairs:
        result = {
            "stage": os.environ["PS_STAGE"],
            "frames": a.frames,
            "matched": False,
            "decision": a.decision,
            "ordinary_archive_sha256": sha(a.ordinary),
            "candidate_archive_sha256": sha(a.candidate),
            "pairs": [],
        }
        pub = Path(os.environ["PS_SCRATCH_DIR"]) / "publish"
        pub.mkdir()
        (pub / "report.json").write_text(json.dumps(result, indent=2))
        (Path(os.environ["PS_STAGE_DIR"]) / "report.json").write_text(json.dumps(result, indent=2))
        return
    _, neg, o, b = min(pairs, key=lambda x: (x[0], x[1]))
    width = -neg
    height = {320: 192, 640: 384}[width]
    native = member(a.ordinary, o["selected_file"])
    packed = member(a.candidate, b["selected_file"])
    assert digest(native) == o["sha256"] and digest(packed) == b["sha256"]
    packet = ball_stream.unpack(packed)
    assert packet.frames == 240 and (packet.width, packet.height) == (960, 544)
    decoder = svtav1.tool("dav1d")
    assert decoder["sha256"] == "e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6"
    work = Path(os.environ["PS_SCRATCH_DIR"]) / "shanghai-display"
    work.mkdir()
    arrays = []
    for name, data, row in [("ordinary", native, o), ("candidate", packet.native, b)]:
        stream = work / (name + ".ivf")
        stream.write_bytes(data)
        yuv = work / (name + ".yuv")
        subprocess.run(
            svtav1.decode_command(decoder["path"], stream, yuv, threads=1),
            check=True,
            capture_output=True,
            timeout=120,
        )
        assert (
            sha(yuv) == row["decoded_yuv_sha256"]
            and yuv.stat().st_size == 240 * width * height * 3 // 2
        )
        arrays.append(np.memmap(yuv, np.uint8, "r", shape=(240, height * 3 // 2, width)))
    positions, looks = ball.decode_frames(packet.ball_payload, 240, 960, 544)
    velocity = ball.velocities(positions)
    cap = cv2.VideoCapture(str(a.source), cv2.CAP_FFMPEG)
    assert cap.isOpened()
    hs = {k: hashlib.sha256() for k in ("source", "ordinary", "candidate")}
    prefix = {k: hashlib.sha256() for k in hs}
    for t in range(a.frames):
        ok, bgr = cap.read()
        assert ok and bgr.shape == (1080, 1920, 3)
        source = cv2.resize(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (960, 544), interpolation=cv2.INTER_AREA
        )
        displays = [
            cv2.resize(
                tracked_patch.yuv_to_rgb(row[t], width, height, False),
                (960, 544),
                interpolation=cv2.INTER_CUBIC,
            )
            for row in arrays
        ]
        displays[1] = ball.render(displays[1], t, positions, velocity, looks)
        for name, img in zip(("source", "ordinary", "candidate"), [source, *displays], strict=True):
            hs[name].update(img.tobytes())
            if t < 24:
                prefix[name].update(img.tobytes())
    cap.release()
    result = {
        "stage": os.environ["PS_STAGE"],
        "frames": a.frames,
        "matched": True,
        "decision": a.decision,
        "width": width,
        "height": height,
        "logical_canvas": [960, 544],
        "source_sha256": SOURCE,
        "ordinary_archive_sha256": sha(a.ordinary),
        "candidate_archive_sha256": sha(a.candidate),
        "ordinary": o,
        "candidate": b,
        "payload_sha256": digest(packet.ball_payload),
        "rgb_sha256": {k: v.hexdigest() for k, v in hs.items()},
        "prefix_rgb_sha256": {k: v.hexdigest() for k, v in prefix.items()},
        "positions_present": sum(p is not None for p in positions),
        "decoder_sha256": decoder["sha256"],
        "opencv_version": cv2.__version__,
        "elapsed_seconds": time.monotonic() - start,
    }
    pub = Path(os.environ["PS_SCRATCH_DIR"]) / "publish"
    pub.mkdir()
    (pub / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    (Path(os.environ["PS_STAGE_DIR"]) / "report.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )


def validate():
    r = json.loads((Path(os.environ["PS_STAGE_DIR"]) / "report.json").read_text())
    n = 24 if os.environ["PS_STAGE"] == "smoke" else 240
    checks = {"frames": r["frames"] == n, "decision": isinstance(r["matched"], bool)}
    if r["matched"]:
        checks["hashes"] = len(r["rgb_sha256"]) == 3 and all(
            len(v) == 64 for v in r["rgb_sha256"].values()
        )
        if n == 24:
            checks["forecast"] = r["elapsed_seconds"] * 10 * 1.5 + 120 < 1200
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps({"passed": all(checks.values()), "checks": checks}) + "\n"
    )
    assert all(checks.values())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path)
    p.add_argument("--ordinary", type=Path)
    p.add_argument("--candidate", type=Path)
    p.add_argument("--ordinary-sha256")
    p.add_argument("--candidate-sha256")
    p.add_argument("--frames", type=int)
    p.add_argument("--decision", choices=("primary", "secondary"))
    p.add_argument("--validate", action="store_true")
    a = p.parse_args()
    validate() if a.validate else run(a)
