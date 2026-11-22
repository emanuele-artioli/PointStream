"""Encode W7's first 240 source frames with plain SVT-AV1 at true 25 fps.

Run from the repository root. Output is a fresh directory under the configured
dataset or an approved local scratch location. No GPU is used.
"""
import argparse
import hashlib
import json
import math
import platform
import resource
import struct
import subprocess
import time
from fractions import Fraction
from pathlib import Path

from src.codecs import svtav1

SOURCE_SHA256 = "c51dace59349914da401356984de4e83da658c878011db3f54ff233077152934"
SOURCE_WIDTH, SOURCE_HEIGHT = 1920, 1080
SOURCE_FRAMES, SOURCE_FPS = 492, 25
WINDOW_START, WINDOW_FRAMES = 0, 240
TARGETS = (20.0, 50.0, 125.0)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def brackets(rates, targets=TARGETS):
    """Describe measured rate brackets without extrapolating."""
    ordered = sorted(set(float(rate) for rate in rates))
    if any(not math.isfinite(rate) or rate <= 0 for rate in ordered):
        raise ValueError("Rates must be positive and finite")
    result = []
    for target in targets:
        below = [rate for rate in ordered if rate <= target]
        above = [rate for rate in ordered if rate >= target]
        result.append({
            "target_kbps": target,
            "lower_kbps": below[-1] if below else None,
            "upper_kbps": above[0] if above else None,
            "bracketed": bool(below and above),
        })
    return result


def verify_ivf(path, width, height, count):
    data = path.read_bytes()
    sizes = svtav1.ivf_frames(data)
    if struct.unpack_from("<HHIIII", data, 12) != (width, height, SOURCE_FPS, 1, count, 0):
        raise ValueError("Unexpected IVF dimensions, cadence, frame count or header")
    offset = 32
    for tick, size in enumerate(sizes):
        if struct.unpack_from("<IQ", data, offset) != (size, tick):
            raise ValueError(f"Invalid IVF timestamp at frame {tick}")
        offset += 12 + size
    if len(sizes) != count or offset != len(data):
        raise ValueError("Incomplete IVF or byte count")
    return sum(sizes)


def cpu():
    return sum(r.ru_utime + r.ru_stime for r in (
        resource.getrusage(resource.RUSAGE_SELF),
        resource.getrusage(resource.RUSAGE_CHILDREN),
    ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=int, choices=(24, WINDOW_FRAMES))
    parser.add_argument("--scale", required=True, type=int, choices=(2, 4))
    parser.add_argument("--crfs", required=True, type=int, nargs="+")
    args = parser.parse_args()
    if not 1 <= len(args.crfs) <= 3 or len(set(args.crfs)) != len(args.crfs):
        parser.error("Supply one to three distinct CRF values")
    if any(value < 0 or value > 63 for value in args.crfs):
        parser.error("CRF values must be between 0 and 63")
    if args.frames == 24 and len(args.crfs) != 1:
        parser.error("The smoke runs exactly one CRF")
    if sha(args.source) != SOURCE_SHA256:
        raise ValueError("Input is not the authenticated W7 al028 source")
    tools = {name: svtav1.tool(name) for name in ("ffmpeg", "ffprobe", "SvtAv1EncApp", "dav1d")}
    probe = json.loads(subprocess.check_output([
        tools["ffprobe"]["path"], "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames",
        "-of", "json", str(args.source),
    ], timeout=30))
    stream, = probe["streams"]
    if (stream["width"], stream["height"], stream["avg_frame_rate"], int(stream["nb_frames"])) != (
        SOURCE_WIDTH, SOURCE_HEIGHT, "25/1", SOURCE_FRAMES
    ):
        raise ValueError("Source dimensions, frame count or cadence do not match W7 selection")
    width = SOURCE_WIDTH // args.scale
    height = 272 if args.scale == 4 else 544
    args.output.mkdir(parents=True, exist_ok=False)
    start, cpu_start = time.monotonic(), cpu()
    raw = args.output / "source.yuv"
    # W7 selection is frames 0–239, where @120 refers to the window row.
    conversion = [
        tools["ffmpeg"]["path"], "-v", "error", "-nostdin", "-threads", "2",
        "-i", str(args.source), "-vf",
        f"select=between(n\\,{WINDOW_START}\\,{WINDOW_START + args.frames - 1}),"
        f"scale={width}:{height}:flags=area:out_range=tv,format=yuv420p",
        "-frames:v", str(args.frames), "-fps_mode", "passthrough", "-f", "rawvideo", str(raw),
    ]
    subprocess.run(conversion, check=True, timeout=120)
    if raw.stat().st_size != args.frames * width * height * 3 // 2:
        raise ValueError("Source conversion has incomplete frames")
    rows = []
    for crf in args.crfs:
        work = args.output / f"plain-{crf}"
        work.mkdir()
        row = svtav1.code(
            raw, work, width=width, height=height, fps=Fraction(SOURCE_FPS),
            frames=args.frames, crf=crf, preset=4, threads=2, full_range=False,
            encoder=tools["SvtAv1EncApp"]["path"], decoder=tools["dav1d"]["path"],
            timeout=150,
        )
        payload = verify_ivf(Path(row["stream"]), width, height, args.frames)
        if row["decoded_frames"] != args.frames or row["payload_bytes"] != payload:
            raise ValueError("Incomplete decoded output or IVF payload mismatch")
        if row["file_bytes"] != payload + 32 + 12 * args.frames:
            raise ValueError("Incomplete IVF byte accounting")
        row.update(
            crf=crf, mode="plain", decoded_sha256=sha(Path(row["decoded"])),
            complete_kbps=row["file_bytes"] * 8 * SOURCE_FPS / args.frames / 1000,
        )
        rows.append(row)
    report = {
        "kind": "smoke" if args.frames == 24 else "full",
        "source_sha256": SOURCE_SHA256, "source_probe": stream, "window_start": WINDOW_START,
        "window_frames": args.frames, "source_fps": SOURCE_FPS,
        "converted_dimensions": [width, height], "scaled_yuv_sha256": sha(raw),
        "tools": tools, "conversion": conversion, "rows": rows,
        "target_brackets": brackets([row["complete_kbps"] for row in rows]),
        "script_sha256": sha(Path(__file__)), "host": platform.platform(), "gpu": None,
        "elapsed_seconds": time.monotonic() - start, "cpu_seconds": cpu() - cpu_start,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"kind": report["kind"], "rates": [r["complete_kbps"] for r in rows],
                      "target_brackets": report["target_brackets"]}))


if __name__ == "__main__":
    main()
