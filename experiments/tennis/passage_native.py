"""Price and decode continuous AV1 on the authenticated 226.1-second broadcast."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import tarfile
import tempfile
import time
from fractions import Fraction

import cv2
import numpy as np

from experiments.tennis import fallback_gap_oracle as gap
from src.codecs import svtav1

FIRST = gap.START - 240
COUNT = gap.COUNT + 480
SUPPORT = (480, 272)


def convert(ffmpeg, source, output, size, frames, *, start=None):
    w, h = size
    command = [ffmpeg, "-v", "error", "-nostdin"]
    if start is not None:
        command += ["-ss", f"{(start + 1) / gap.FPS:.6f}"]
    command += [
        "-i",
        str(source),
        "-frames:v",
        str(frames),
        "-vf",
        f"scale={w}:{h}:flags=area:out_range=tv,format=yuv420p",
        "-filter_threads",
        "1",
        "-fps_mode",
        "passthrough",
        "-threads",
        "1",
        "-f",
        "rawvideo",
        str(output),
    ]
    subprocess.run(command, check=True, timeout=1500)
    if output.stat().st_size != frames * w * h * 3 // 2:
        raise ValueError("Incomplete native frame conversion")
    return command


def sections(raw, n):
    assert raw[:4] == b"DKIF"
    pos = 32
    out = []
    for t in range(n):
        size, tick = struct.unpack_from("<IQ", raw, pos)
        assert tick == t and pos + 12 + size <= len(raw)
        pos += 12 + size
        out.append(size + 12 + (32 if t == 0 else 0))
    assert pos == len(raw) and sum(out) == pos
    return out


def run(a):
    assert len(os.sched_getaffinity(0)) == 1 and os.environ["CUDA_VISIBLE_DEVICES"] == ""
    assert (gap.sha(a.source), gap.sha(a.prior), gap.sha(a.next)) == (
        gap.SOURCE_SHA,
        gap.PRIOR_SHA,
        gap.NEXT_SHA,
    )
    tools = {name: svtav1.tool(name) for name in gap.TOOLS}
    assert all(tools[k]["sha256"] == h for k, h in gap.TOOLS.items())
    cv2.setNumThreads(1)
    n = a.frames
    w, h = (240, 136) if a.target == 20 else SUPPORT
    root = Path(os.environ["PS_SCRATCH_DIR"])
    pub = root / "publish"
    pub.mkdir()
    clock = time.monotonic()
    ff = tools["ffmpeg"]["path"]
    truth = root / "truth.yuv"
    commands = [convert(ff, a.source, truth, SUPPORT, n, start=FIRST)]
    ref = np.memmap(truth, np.uint8, "r", shape=(n, 408, 480))
    for start, clip, tag in ((0, a.prior, "initial"), (COUNT - 240, a.next, "return")):
        if start >= n:
            continue
        path = root / f"{tag}.yuv"
        convert(ff, clip, path, SUPPORT, 240)
        for t in (0, 119, 239):
            assert np.array_equal(
                ref[start + t], np.memmap(path, np.uint8, "r", shape=(240, 408, 480))[t]
            )
    source = truth
    if (w, h) != SUPPORT:
        source = root / "small.yuv"
        commands.append(convert(ff, a.source, source, (w, h), n, start=FIRST))
    encoded = pub / "anchor.ivf"
    decoded = root / "received.yuv"
    encoder = svtav1.encode_command(
        tools["SvtAv1EncApp"]["path"],
        source,
        encoded,
        width=w,
        height=h,
        fps=Fraction(50),
        frames=n,
        crf=45,
        preset=4,
        full_range=False,
    )
    encoder[encoder.index("--rc") + 1] = "1"
    idx = encoder.index("--crf")
    encoder[idx : idx + 2] = ["--tbr", str(round(a.target - 4.8))]
    encoder[encoder.index("--keyint") + 1] = "250"
    subprocess.run(encoder, check=True, capture_output=True, timeout=1900)
    commands.append(encoder)
    decoder = svtav1.decode_command(tools["dav1d"]["path"], encoded, decoded, threads=1)
    subprocess.run(decoder, check=True, capture_output=True, timeout=450)
    commands.append(decoder)
    assert decoded.stat().st_size == n * w * h * 3 // 2
    frames = np.memmap(decoded, np.uint8, "r", shape=(n, h * 3 // 2, w))
    raw = encoded.read_bytes()
    assert struct.unpack_from("<HHIIII", raw, 12) == (w, h, 50, 1, n, 0)
    costs = sections(raw, n)
    sse = []
    hashes = {}
    for t in range(n):
        y = frames[t, :h]
        if (w, h) != SUPPORT:
            y = cv2.resize(y, SUPPORT, interpolation=cv2.INTER_CUBIC)
        delta = np.asarray(ref[t, :272], np.int32) - y
        sse.append(int(np.sum(np.square(delta), dtype=np.int64)))
        if t in (0, 119, 239, COUNT - 240, COUNT - 1):
            hashes[t] = hashlib.sha256(frames[t].tobytes()).hexdigest()
    metrics = {}
    for name, lo, hi in (
        ("full", 0, n),
        ("initial", 0, 240),
        ("interlude", 240, COUNT - 240),
        ("return", COUNT - 240, COUNT),
    ):
        if hi > n or lo == hi:
            continue
        total = sum(costs[lo:hi])
        metrics[name] = dict(
            frames=hi - lo,
            bytes=total,
            kbps=total * 0.4 / (hi - lo),
            luma_psnr=10 * math.log10(255**2 * (hi - lo) * 480 * 272 / sum(sse[lo:hi])),
            sse=sum(sse[lo:hi]),
        )
    r = dict(
        target=a.target,
        geometry=[w, h],
        source_sha256=gap.SOURCE_SHA,
        raw_source_sha256=gap.sha(truth),
        stream_sha256=gap.sha(encoded),
        decoded_sha256=gap.sha(decoded),
        frame_hashes=hashes,
        metrics=metrics,
        tools=tools,
        commands=commands,
        seconds=time.monotonic() - clock,
    )
    (pub / "result.json").write_text(json.dumps(r, indent=2) + "\n")


def validate():
    stage = Path(os.environ["PS_STAGE_DIR"])
    with tarfile.open(stage / "published.tar") as archive:
        r = json.load(archive.extractfile("publish/result.json"))
        data = archive.extractfile("publish/anchor.ivf").read()
    smoke = os.environ["PS_STAGE"] == "smoke"
    n = 240 if smoke else COUNT
    m = r["metrics"]
    sizes = sections(data, n)
    valid = (
        hashlib.sha256(data).hexdigest() == r["stream_sha256"] and sum(sizes) == m["full"]["bytes"]
    )
    tests = {
        "packet": valid,
        "frames": m["full"]["frames"] == n,
        "source": r["source_sha256"] == gap.SOURCE_SHA,
        "ranges": all(0 < p["luma_psnr"] < 100 for p in m.values()),
    }
    forecast = None
    if smoke:
        folder = Path(os.environ["PS_SCRATCH_DIR"])
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as name:
            p = Path(name)
            (p / "anchor.ivf").write_bytes(data)
            subprocess.run(
                svtav1.decode_command(
                    r["tools"]["dav1d"]["path"], p / "anchor.ivf", p / "out.yuv", threads=1
                ),
                check=True,
                timeout=60,
            )
            tests["independent_replay"] = gap.sha(p / "out.yuv") == r["decoded_sha256"]
        forecast = r["seconds"] * COUNT / n * 1.5 + 120
        tests["cost"] = forecast < 2250
    else:
        tests["parts"] = set(m) == {"full", "initial", "interlude", "return"} and sum(
            m[k]["bytes"] for k in ("initial", "interlude", "return")
        ) == len(data)
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(dict(passed=all(tests.values()), checks=tests, forecast_seconds=forecast))
    )
    if not all(tests.values()):
        raise ValueError(tests)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "prior", "next"):
        parser.add_argument("--" + key, type=Path)
    parser.add_argument("--target", type=int, choices=(20, 65))
    parser.add_argument("--frames", type=int, choices=(240, COUNT))
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    validate() if args.validate else run(args)
