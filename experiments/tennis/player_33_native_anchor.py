"""Encode one registered W7 ordinary AV1 point and reproduce its historical control."""

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

from experiments.tennis.k4b_features import convert_command
from experiments.tennis.k6 import downscaled_source
from src.codecs import svtav1


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def run(args):
    start = time.monotonic()
    n = args.frames
    if n != (24 if os.environ["PS_STAGE"] == "smoke" else 240) or len(os.sched_getaffinity(0)) != 1:
        raise ValueError("Unexpected frame or CPU allocation")
    if digest(args.source) != args.source_sha:
        raise ValueError("Authenticated W7 source differs")
    root = Path(os.environ["PS_SCRATCH_DIR"])
    pub = root / "publish"
    pub.mkdir()
    ff = svtav1.tool("ffmpeg")
    native = root / "source.yuv"
    cmd = convert_command(ff["path"], args.source, native, 0, n)
    subprocess.run(cmd, check=True, timeout=240)
    original = digest(native)
    if n == 240 and original != args.original_yuv:
        raise ValueError("Original W7 native source conversion changed")
    reduced = downscaled_source(native, n, 4, root / "480.yuv")
    encoder, decoder = svtav1.tool("SvtAv1EncApp"), svtav1.tool("dav1d")
    rows = []
    for quality in (63, args.crf):
        work = root / f"q{quality}"
        work.mkdir()
        r = svtav1.code(
            reduced,
            work,
            width=480,
            height=272,
            fps=Fraction(50),
            frames=n,
            crf=quality,
            preset=4,
            threads=1,
            full_range=False,
            encoder=encoder["path"],
            decoder=decoder["path"],
            timeout=180,
        )
        stream = Path(r["stream"])
        raw = Path(r["decoded"])
        row = dict(
            crf=quality,
            bytes=stream.stat().st_size,
            stream_sha256=digest(stream),
            decoded_sha256=digest(raw),
            frames=r["decoded_frames"],
            kbps=stream.stat().st_size * 8 * 50 / 1000 / n,
            encoder_command=r["encode_command"],
            decoder_command=r["decode_command"],
        )
        if r["decoded_frames"] != n or len(svtav1.ivf_frames(stream.read_bytes())) != n:
            raise ValueError("Incomplete native encode/decode")
        if (
            quality == 63
            and n == 240
            and (row["stream_sha256"], row["decoded_sha256"])
            != (args.control_stream, args.control_yuv)
        ):
            raise ValueError("Historical CRF63 control did not reproduce")
        if quality == args.crf:
            shutil.copyfile(stream, pub / "candidate.ivf")
        rows.append(row)
    result = dict(
        **rows[-1],
        stage=os.environ["PS_STAGE"],
        window=args.window,
        width=480,
        height=272,
        fps=50,
        source_sha256=args.source_sha,
        source_yuv_sha256=original,
        reference=rows[0],
        scale=4,
        offset=0,
        encoder=encoder,
        decoder=decoder,
        source_converter=ff,
        source_command=cmd,
        elapsed_seconds=time.monotonic() - start,
    )
    (pub / "result.json").write_text(json.dumps(result, indent=2) + "\n")


def validate(args):
    stage = Path(os.environ["PS_STAGE_DIR"])
    with tarfile.open(stage / "published.tar") as t:
        r = json.load(t.extractfile("publish/result.json"))
        data = t.extractfile("publish/candidate.ivf").read()
    smoke = os.environ["PS_STAGE"] == "smoke"
    n = 24 if smoke else 240
    checks = dict(
        frames=r["frames"] == r["reference"]["frames"] == len(svtav1.ivf_frames(data)) == n,
        rate=r["kbps"] == len(data) * 8 * 50 / 1000 / n,
        source=r["source_sha256"] == args.source_sha,
        settings=(r["width"], r["height"], r["fps"], r["crf"]) == (480, 272, 50, args.crf),
        transport=len(data) == r["bytes"]
        and hashlib.sha256(data).hexdigest() == r["stream_sha256"],
        original=smoke
        or (
            r["reference"]["stream_sha256"],
            r["reference"]["decoded_sha256"],
            r["source_yuv_sha256"],
        )
        == (args.control_stream, args.control_yuv, args.original_yuv),
    )
    forecast = r["elapsed_seconds"] * 10 * 1.5 + 45 if smoke else None
    checks["budget"] = not smoke or forecast < 600
    ok = all(checks.values())
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(dict(passed=ok, checks=checks, forecast=forecast)) + "\n"
    )
    if not ok:
        raise ValueError(str(checks))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path)
    for option in ("window", "source-sha", "control-stream", "control-yuv", "original-yuv"):
        p.add_argument("--" + option, required=True)
    p.add_argument("--frames", type=int)
    p.add_argument("--crf", type=int, required=True)
    p.add_argument("--validate", action="store_true")
    opts = p.parse_args()
    validate(opts) if opts.validate else run(opts)
