"""Measure a fixed 1080p spatial rendition of a frozen 4K prefix at original cadence."""

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from experiments.tennis import court, k8b
from experiments.visor.b1 import file_sha256


def measure(a, frames, directory):
    begin = time.monotonic()
    meta = k8b.probe(a.source, "ffprobe")
    assert (meta["width"], meta["height"], meta["frames"]) == (3840, 2160, a.length)
    assert Fraction(meta["r_frame_rate"]) == Fraction(a.fps) and file_sha256(a.source) == a.sha
    command = k8b.decode_command("ffmpeg", a.source, 0, frames)
    chosen = set(int(k) for k in np.linspace(0, frames - 1, 6))
    gallery = []
    rows = []
    prev = None
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    for i, rgb in enumerate(k8b.frames(command, frames)):
        assert rgb.shape == (1080, 1920, 3) and rgb.dtype == np.uint8
        row = court.register_v2(rgb, prev)
        prev = np.array(row["homography"]) if row["homography"] is not None else None
        rows.append(dict(tick=i, rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(), **row))
        if i in chosen:
            gallery.append(cv2.resize(rgb, (448, 252), interpolation=cv2.INTER_AREA)[:, :, ::-1])
    assert len(rows) == frames and len(gallery) == 6
    preview = directory / "source.jpg"
    assert cv2.imwrite(str(preview), np.vstack([np.hstack(gallery[:3]), np.hstack(gallery[3:])]))
    fits = sum(bool(r["fits"]) for r in rows)
    res = dict(
        event=a.event,
        sha256=a.sha,
        source=meta,
        output_raster=[1920, 1080],
        fps=str(Fraction(a.fps)),
        frames=frames,
        eligible=fits / frames >= 0.8,
        fit_count=fits,
        rows=rows,
        preview_ticks=sorted(chosen),
        preview_sha256=file_sha256(preview),
        command=command,
        cv2=cv2.__version__,
        ffmpeg=subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0],
        wall_seconds=time.monotonic() - begin,
    )
    (directory / "result.json").write_text(json.dumps(res, indent=2, allow_nan=False) + "\n")
    return res


def validate(r, stage, base, limit):
    n = {"smoke": 12, "full": 240}[stage]
    assert r["frames"] == n and len(r["rows"]) == n
    checks = dict(
        native_ticks=[v["tick"] for v in r["rows"]] == list(range(n)),
        raster=r["output_raster"] == [1920, 1080]
        and [r["source"]["width"], r["source"]["height"]] == [3840, 2160],
        native_cadence=Fraction(r["fps"]) == Fraction(r["source"]["r_frame_rate"]),
        fit_count=sum(bool(v["fits"]) for v in r["rows"]) == r["fit_count"],
        pass_rule=r["eligible"] == (r["fit_count"] / n >= 0.8),
        pixels=all(len(v["rgb_sha256"]) == 64 for v in r["rows"]),
        preview=file_sha256(base / stage / "source.jpg") == r["preview_sha256"],
    )
    if stage == "smoke":
        checks["forecast"] = r["wall_seconds"] * 20 * 1.5 + 100 < limit
    else:
        smoke = json.loads((base / "smoke/result.json").read_text())
        checks["same_twelve"] = smoke["rows"] == r["rows"][:12]
    return dict(passed=all(checks.values()), checks=checks)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path)
    p.add_argument("--event")
    p.add_argument("--sha")
    p.add_argument("--fps")
    p.add_argument("--length", type=int)
    p.add_argument("--frames", type=int)
    p.add_argument("--validate", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--full-seconds", type=int, default=1800)
    a = p.parse_args()
    if a.output:
        a.output.mkdir(exist_ok=False)
        for stage, frames in [("smoke", 12), ("full", 240)]:
            target = a.output / stage
            target.mkdir()
            r = measure(a, frames, target)
            v = validate(r, stage, a.output, a.full_seconds)
            (target / "validation.json").write_text(json.dumps(v) + "\n")
            assert v["passed"], v
        print(a.event, r["fit_count"], r["eligible"], round(r["wall_seconds"], 2))
    elif a.validate:
        folder = Path(os.environ["PS_STAGE_DIR"])
        r = json.loads((folder / "result.json").read_text())
        v = validate(r, os.environ["PS_STAGE"], folder.parent, a.full_seconds)
        Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps(v) + "\n")
        assert v["passed"], v
    else:
        assert a.frames == {"smoke": 12, "full": 240}[os.environ["PS_STAGE"]]
        measure(a, a.frames, Path(os.environ["PS_STAGE_DIR"]))


if __name__ == "__main__":
    main()
