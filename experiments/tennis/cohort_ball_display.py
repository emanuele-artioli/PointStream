"""Reproduce each scored source and receiver pixel on its original CPU host, then store lossless displays for transport."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time
import zlib

import cv2
import numpy as np
from experiments.tennis import cohort_ball_score as cohort
from src.codecs import svtav1
from src.pointstream import ball

NAMES = ("source", "ordinary", "candidate")


def reconstruct(a, meta, receipt, work, n):
    rates, streams, paid = cohort.load_bank(a.rate, meta)
    frames, _ = cohort.source(a.source, meta, n)
    assert cohort.digest(frames) == receipt["source_rgb_sha256"]
    pos, looks = ball.decode_frames(paid.ball_payload, 240, 960, 544)
    velocity = ball.velocities(pos)
    decoder = svtav1.tool("dav1d")
    assert decoder["sha256"] == cohort.TOOLS["dav1d"]
    display = {"source": frames}
    for name in ("ordinary", "candidate"):
        r = rates[name]
        native = paid.native if name == "candidate" else streams[r["native_sha256"]]
        raw = cohort.native(native, r, work, decoder["path"])
        out = np.lib.format.open_memmap(
            work / (name + ".npy"), mode="w+", dtype=np.uint8, shape=frames.shape
        )
        for t in range(n):
            img = cohort.display(raw[t], r["width"], r["height"])
            out[t] = ball.render(img, t, pos, velocity, looks) if name == "candidate" else img
        assert hashlib.sha256(out).hexdigest() == receipt[name]["rgb_sha256"]
        display[name] = out
    return display


def restore(archive, expected_sha, receipt, n, work):
    assert cohort.sha(archive) == expected_sha
    displays = {}
    with tarfile.open(archive) as tar:
        report = json.load(tar.extractfile("publish/display.json"))
        assert report["frames"] == 240 and report["match"] == receipt["match"]
        assert (
            report["rate_archive_sha256"] == receipt["rate_archive_sha256"]
            and report["source_sha256"] == receipt["source_sha256"]
        )
        framebytes = 544 * 960 * 3
        for name in NAMES:
            stream = tar.extractfile("publish/" + name + ".rgb.z")
            out = work / (name + "-frozen.rgb")
            decomp = zlib.decompressobj()
            length = 0
            raw_sha = hashlib.sha256()
            with out.open("wb") as f:
                while block := stream.read(32768):
                    pixels = decomp.decompress(block)
                    length += len(pixels)
                    raw_sha.update(pixels)
                    f.write(pixels)
                rest = decomp.flush()
                length += len(rest)
                raw_sha.update(rest)
                f.write(rest)
            assert (
                decomp.eof
                and not decomp.unused_data
                and length == 240 * framebytes
                and raw_sha.hexdigest() == report["rgb_sha256"][name]
            )
            arr = np.memmap(out, np.uint8, "r", shape=(240, 544, 960, 3))[:n]
            expected = (
                receipt["source_rgb_sha256"] if name == "source" else receipt[name]["rgb_sha256"]
            )
            assert hashlib.sha256(arr).hexdigest() == expected
            displays[name] = arr
    return displays


def run(a):
    start = time.monotonic()
    n = a.frames
    assert n == {"smoke": 24, "full": 240}[os.environ["PS_STAGE"]]
    assert len(os.sched_getaffinity(0)) == 1 and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    cv2.setNumThreads(1)
    meta = json.loads(a.metadata.read_text())
    receipt = json.loads((a.score if n == 240 else a.smoke_score).read_text())
    assert cohort.sha(a.source) == meta["source"]["sha256"] == receipt["source_sha256"]
    assert cohort.sha(a.rate) == a.rate_sha == receipt["rate_archive_sha256"]
    assert receipt["frames"] == n
    work = Path(os.environ["PS_SCRATCH_DIR"]) / "frozen"
    work.mkdir()
    publish = Path(os.environ["PS_SCRATCH_DIR"]) / "publish"
    publish.mkdir()
    display = reconstruct(a, meta, receipt, work, n)
    hashes = {}
    sizes = {}
    for name, arr in display.items():
        raw_sha = hashlib.sha256()
        z = zlib.compressobj(1)
        path = publish / (name + ".rgb.z")
        with path.open("wb") as f:
            for frame in arr:
                raw_sha.update(frame)
                block = z.compress(frame)
                f.write(block)
            tail = z.flush()
            f.write(tail)
        hashes[name] = raw_sha.hexdigest()
        sizes[name] = path.stat().st_size
        expect = receipt["source_rgb_sha256"] if name == "source" else receipt[name]["rgb_sha256"]
        assert hashes[name] == expect
    r = dict(
        match=meta["match"],
        frames=n,
        source_sha256=receipt["source_sha256"],
        rate_archive_sha256=a.rate_sha,
        rgb_sha256=hashes,
        bytes=sizes,
        elapsed=time.monotonic() - start,
    )
    for path in (publish / "display.json", Path(os.environ["PS_STAGE_DIR"]) / "display.json"):
        path.write_text(json.dumps(r, indent=2, allow_nan=False) + "\n")


def validate():
    stage = Path(os.environ["PS_STAGE_DIR"])
    r = json.loads((stage / "display.json").read_text())
    with tarfile.open(stage / "published.tar") as tar:
        assert json.load(tar.extractfile("publish/display.json")) == r
        for name in NAMES:
            src = tar.extractfile("publish/" + name + ".rgb.z")
            z = zlib.decompressobj()
            h = hashlib.sha256()
            size = 0
            while block := src.read(32768):
                raw = z.decompress(block)
                size += len(raw)
                h.update(raw)
            raw = z.flush()
            size += len(raw)
            h.update(raw)
            assert (
                z.eof
                and not z.unused_data
                and size == r["frames"] * 544 * 960 * 3
                and h.hexdigest() == r["rgb_sha256"][name]
            )
    checks = dict(
        native=True,
        lossless=True,
        independent_hashes=len(r["rgb_sha256"]) == 3,
        quota=sum(r["bytes"].values()) <= r["frames"] * 544 * 960 * 9 + 4096,
    )
    if r["frames"] == 24:
        checks["forecast"] = 1.5 * r["elapsed"] * 10 + 60 < 1400
    ok = all(checks.values())
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(dict(passed=ok, checks=checks)) + "\n"
    )
    if not ok:
        raise ValueError(checks)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("source", "metadata", "rate", "score", "smoke-score"):
        p.add_argument("--" + k, type=Path)
    p.add_argument("--rate-sha")
    p.add_argument("--frames", type=int)
    p.add_argument("--validate", action="store_true")
    a = p.parse_args()
    validate() if a.validate else run(a)
