"""Extract author-marked points with source timestamps; verify every native YUV and RGB frame."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def decode_hashes(source_args, count, pixel_format):
    data = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "2",
            *source_args,
            "-map",
            "0:v:0",
            "-an",
            "-frames:v",
            str(count),
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            pixel_format,
            "-f",
            "framehash",
            "-hash",
            "sha256",
            "-",
        ]
    ).decode()
    return [line.split(",")[-1].strip() for line in data.splitlines() if not line.startswith("#")]


def extract(source, rows, directory):
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_packets",
                "-show_streams",
                "-show_entries",
                "packet=pts:stream=width,height,pix_fmt,avg_frame_rate,time_base,start_pts,nb_frames",
                "-of",
                "json",
                str(source),
            ]
        )
    )
    meta = probe["streams"][0]
    fps = Fraction(meta["avg_frame_rate"])
    step = 1 / (fps * Fraction(meta["time_base"]))
    if step.denominator != 1 or meta["start_pts"] != 0 or meta["pix_fmt"] != "yuv420p":
        raise ValueError("Source timestamp grid or pixel format needs investigation")
    grid = sorted(int(p["pts"]) for p in probe["packets"])
    total = int(meta["nb_frames"])
    if grid != list(range(0, total * int(step), int(step))):
        raise ValueError("Missing or repeated native display timestamp")
    output = {
        "source_sha256": sha(source),
        "source_bytes": source.stat().st_size,
        "media": meta,
        "rallies": [],
    }
    for r in rows:
        start, stop = r["native_range"]
        if not 0 <= start < stop <= total:
            raise ValueError("Invalid original interval")
        target = directory / (r["rally_id"] + ".mkv")
        count = stop - start
        # Decode from a separate earlier seek, then select by absolute source packet timestamp.
        early = str(float(max(start - 125, 0) / fps))
        filt = f"select='between(pts,{start * int(step)},{(stop - 1) * int(step)})',setpts=PTS-STARTPTS"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-n",
                "-copyts",
                "-ss",
                early,
                "-threads",
                "2",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-an",
                "-vf",
                filt,
                "-frames:v",
                str(count),
                "-fps_mode",
                "passthrough",
                "-c:v",
                "ffv1",
                "-threads",
                "2",
                "-level",
                "3",
                "-pix_fmt",
                "yuv420p",
                str(target),
            ],
            check=True,
        )
        proof = {}
        for fmt in ("yuv420p", "rgb24"):
            expected = decode_hashes(
                ["-ss", str(float(start / fps)), "-i", str(source)], count, fmt
            )
            actual = decode_hashes(["-i", str(target)], count + 1, fmt)
            if expected != actual or len(actual) != count:
                raise ValueError(f"{r['rally_id']}: mismatching {fmt} frames")
            proof[fmt] = {
                "count": count,
                "first": actual[0],
                "last": actual[-1],
                "ordered_hash_sha256": hashlib.sha256("\n".join(actual).encode()).hexdigest(),
            }
        output["rallies"].append(
            {
                "rally_id": r["rally_id"],
                "native_range": [start, stop],
                "clip_sha256": sha(target),
                "filename": target.name,
                "proof": proof,
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event_id")
    parser.add_argument("source", type=Path)
    parser.add_argument("directory", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    events = json.loads((base / "manifest.json").read_text())["events"]
    rows = next(e for e in events if e["event_id"] == args.event_id)["rallies"]
    args.directory.mkdir(parents=True, exist_ok=True)
    receipt = extract(args.source, rows, args.directory)
    receipt["ffmpeg"] = subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0]
    with args.receipt.open("x") as f:
        json.dump(receipt, f, indent=2)
        f.write("\n")
    print(
        args.event_id,
        receipt["source_sha256"],
        [r["proof"]["rgb24"]["count"] for r in receipt["rallies"]],
    )


if __name__ == "__main__":
    main()
