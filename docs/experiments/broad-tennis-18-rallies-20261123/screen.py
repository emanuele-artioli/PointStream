"""Run the frozen court-view screen on one authenticated native rally interval."""

import argparse
from collections import Counter
from fractions import Fraction
from hashlib import file_digest
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import time

import cv2

from experiments.tennis import court, k8, k8b


def hash_file(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("expected_sha256")
    parser.add_argument("rally_id")
    parser.add_argument("start", type=int)
    parser.add_argument("stop", type=int)
    parser.add_argument("out", type=Path)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if hash_file(args.source) != args.expected_sha256 or not 0 <= args.start < args.stop:
        raise ValueError("source identity or frame range differs from enrollment")
    meta = k8b.probe(args.source, "ffprobe")
    if args.stop > meta["frames"]:
        raise ValueError("rally lies beyond the authenticated original source")
    start = args.start
    stop = min(start + 12, args.stop) if args.pilot else args.stop
    begin = time.monotonic()
    rows = k8b.register_chunk(str(args.source), start, stop, "ffmpeg")
    if [r["frame"] for r in rows] != list(range(start, stop)):
        raise ValueError("incomplete source screen")
    fps = Fraction(meta["r_frame_rate"])
    counts = Counter(bool(r["fits"]) for r in rows)
    shots = None
    if not args.pilot:
        if args.start != 0 or args.stop != meta["frames"]:
            raise ValueError("full screen requires a source-verified lossless native rally clip")
        with TemporaryDirectory() as scratch:
            scenes, _ = k8.cuts(args.source, float(fps), args.stop, "ffmpeg", Path(scratch))
        cuts = sorted({s["frame"] for s in scenes if s["score"] >= k8b.DECISION["scene_threshold"]})
        shots = [
            {"start": lo, "stop": hi, **k8b.prediction(rows[lo:hi])}
            for lo, hi in k8b.clip_shots(rows, cuts, float(fps))
        ]
        assert sum(s["stop"] - s["start"] for s in shots) == args.stop
    result = {
        "rally_id": args.rally_id,
        "source_sha256": args.expected_sha256,
        "native_range": [args.start, args.stop],
        "evaluated_range": [start, stop],
        "native_fps": [fps.numerator, fps.denominator],
        "native_dimensions": [meta["width"], meta["height"]],
        "fits": counts[True],
        "shots": shots,
        "opencv_version": cv2.__version__,
        "rows": rows,
        "selector_sha256": hash_file(Path(court.__file__)),
        "screen_sha256": hash_file(Path(__file__)),
        "ffmpeg_version": subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[
            0
        ],
        "wall_seconds": time.monotonic() - begin,
    }
    with args.out.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(
        args.rally_id,
        counts[True],
        stop - start,
        round(result["wall_seconds"], 2),
        result["shots"],
    )


if __name__ == "__main__":
    main()
