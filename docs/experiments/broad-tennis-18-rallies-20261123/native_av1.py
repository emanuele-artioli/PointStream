"""Calibrate and verify ordinary AV1 at native timing on a lossless rally clip."""

import argparse
from fractions import Fraction
from hashlib import file_digest
import json
from pathlib import Path
import subprocess
import time

from src.codecs import svtav1


PRICES = (20400, 33000)
SCALES = ((320, 180), (640, 360))
CRFS = (45, 58, 67, 70)


def sha(path):
    with path.open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def command(args, timeout=120):
    process = subprocess.run(args, check=False, capture_output=True, text=True, timeout=timeout)
    if process.returncode:
        raise RuntimeError(process.stderr[-4000:])
    return process.stdout


def codec_meta(path, ffprobe):
    result = command(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ]
    )
    return json.loads(result)["streams"][0]


def run(args):
    if sha(args.source) != args.sha256:
        raise ValueError("lossless source clip differs from extraction receipt")
    began = time.monotonic()
    args.out.mkdir(exist_ok=False)
    tools = {n: svtav1.tool(n) for n in ("ffmpeg", "ffprobe", "SvtAv1EncApp")}
    enc = tools["SvtAv1EncApp"]["path"]
    ffmpeg = tools["ffmpeg"]["path"]
    ffprobe = tools["ffprobe"]["path"]
    meta = codec_meta(args.source, ffprobe)
    fps = Fraction(meta["r_frame_rate"])
    frames = int(meta["nb_read_frames"])
    assert fps == Fraction(meta["avg_frame_rate"]) == Fraction(args.fps) and frames == args.frames
    assert (meta["width"], meta["height"]) == (1280, 720)
    results = []
    for width, height in SCALES:
        raw = args.out / f"input-{width}.yuv"
        command(
            [
                ffmpeg,
                "-nostdin",
                "-v",
                "error",
                "-threads",
                "1",
                "-i",
                str(args.source),
                "-an",
                "-vf",
                f"scale={width}:{height}:flags=area:out_range=tv,format=yuv420p",
                "-f",
                "rawvideo",
                str(raw),
            ]
        )
        assert raw.stat().st_size == frames * width * height * 3 // 2
        encodes = []
        for crf in CRFS:
            target = args.out / f"{width}-crf{crf}.ivf"
            line = svtav1.encode_command(
                enc,
                raw,
                target,
                width=width,
                height=height,
                fps=fps,
                frames=frames,
                crf=crf,
                preset=4,
                full_range=False,
            )
            command(line)
            encodes.append(
                dict(crf=crf, bytes=target.stat().st_size, file=target.name, sha256=sha(target))
            )
        selected = {}
        for price in PRICES:
            ceiling = int(Fraction(price * frames, 8) / fps)
            affordable = [item for item in encodes if item["bytes"] <= ceiling]
            if not affordable:
                selected[str(price)] = dict(ceiling=ceiling, chosen=None)
                continue
            chosen = max(affordable, key=lambda item: item["bytes"])
            video = args.out / chosen["file"]
            probe = codec_meta(video, ffprobe)
            assert Fraction(probe["r_frame_rate"]) == fps and int(probe["nb_read_frames"]) == frames
            decoded = command(
                [
                    ffmpeg,
                    "-nostdin",
                    "-v",
                    "error",
                    "-threads",
                    "1",
                    "-i",
                    str(video),
                    "-f",
                    "framehash",
                    "-",
                ]
            )
            rows = [ln for ln in decoded.splitlines() if ln and not ln.startswith("#")]
            assert len(rows) == frames
            assert [int(ln.split(",")[2]) for ln in rows] == list(range(frames))
            selected[str(price)] = dict(
                ceiling=ceiling,
                chosen=chosen["file"],
                bytes=chosen["bytes"],
                decoded_frame_sha256=[ln.split(",")[-1].strip() for ln in rows],
            )
        results.append(dict(width=width, height=height, available=encodes, selected=selected))
    record = dict(
        source_sha256=args.sha256,
        native=meta,
        num_frames=frames,
        tools=tools,
        arms=results,
        worker_sha256=sha(Path(__file__)),
        wall_seconds=time.monotonic() - began,
    )
    (args.out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        args.source.stem,
        [(r["width"], [(k, v.get("bytes")) for k, v in r["selected"].items()]) for r in results],
        round(record["wall_seconds"], 1),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("sha256")
    parser.add_argument("frames", type=int)
    parser.add_argument("out", type=Path)
    parser.add_argument("--fps", default="25/1")
    run(parser.parse_args())
