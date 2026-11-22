"""Generate a frozen source-only SVT-AV1 rate ladder on the 50-fps Medvedev–Struff rally."""

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import resource
import subprocess

from src.codecs import svtav1

from build import SOURCES as MED_SOURCES

SOURCES = {
    **MED_SOURCES,
    "al031": "054e4535638d8dfa3dee69258f45c231aaeab2241b91309d449be7b0934cef76",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu():
    return sum(
        resource.getrusage(k).ru_utime + resource.getrusage(k).ru_stime
        for k in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)
    )


def main(a):
    if sha(a.source) != SOURCES[a.clip]:
        raise ValueError("source clip identity")
    fps = 25 if a.clip == "al031" else 50
    rate_divisor = 240 * 1000 / (fps * 8)
    OUT = a.out
    OUT.mkdir(exist_ok=False)
    source = OUT / "source.yuv"
    start = cpu()
    tools = {k: svtav1.tool(k) for k in ("ffmpeg", "SvtAv1EncApp", "dav1d")}
    subprocess.run(
        [
            tools["ffmpeg"]["path"],
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            str(a.source),
            "-vf",
            "scale=960:544:flags=area,format=yuv420p",
            "-fps_mode",
            "passthrough",
            "-frames:v",
            "240",
            "-f",
            "rawvideo",
            "-n",
            str(source),
        ],
        check=True,
        timeout=180,
    )
    if source.stat().st_size != 240 * 960 * 544 * 3 // 2:
        raise ValueError("source frame count")
    inputs = {}
    for width, height in ((160, 96), (320, 192), (640, 384)):
        raw = OUT / f"{width}.yuv"
        subprocess.run(
            [
                tools["ffmpeg"]["path"],
                "-v",
                "error",
                "-threads",
                "1",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "yuv420p",
                "-s",
                "960x544",
                "-r",
                str(fps),
                "-i",
                str(source),
                "-vf",
                f"scale={width}:{height}:flags=area",
                "-frames:v",
                "240",
                "-f",
                "rawvideo",
                str(raw),
            ],
            check=True,
            timeout=120,
        )
        if raw.stat().st_size != 240 * width * height * 3 // 2:
            raise ValueError("source frame count")
        inputs[width] = (height, raw)

    def encode(width, q, n, name):
        height, raw = inputs[width]
        dest = OUT / name
        dest.mkdir()
        r = svtav1.code(
            raw,
            dest,
            width=width,
            height=height,
            fps=Fraction(fps),
            frames=n,
            crf=q,
            preset=4,
            threads=1,
            full_range=False,
            encoder=tools["SvtAv1EncApp"]["path"],
            decoder=tools["dav1d"]["path"],
            timeout=180,
        )
        if r["decoded_frames"] != n or r["temporal_units"] != n:
            raise ValueError("baseline decode coverage")
        return r

    before = cpu()
    encode(640, 50, 24, "smoke")
    smoke = cpu() - before
    forecast = (cpu() - start) + smoke * 10 * 12 * 1.5 + 60
    if forecast > 1200:
        raise ValueError(f"baseline projected CPU seconds {forecast} exceed cap")
    rows = []
    for width in (160, 320, 640):
        for q in (40, 50, 60, 70):
            r = encode(width, q, 240, f"{width}-crf{q}")
            rows.append(
                {
                    "width": width,
                    "height": inputs[width][0],
                    "crf": q,
                    "bytes": r["file_bytes"],
                    "kbps": r["file_bytes"] / rate_divisor,
                    "stream_sha256": sha(Path(r["stream"])),
                    "decoded_sha256": sha(Path(r["decoded"])),
                }
            )
    result = {
        "source_yuv_sha256": sha(source),
        "ticks": 240,
        "fps": fps,
        "tools": tools,
        "original_source_sha256": SOURCES[a.clip],
        "grid": rows,
        "smoke_cpu": smoke,
        "forecast_cpu": forecast,
        "total_cpu": cpu() - start,
        "scope": "Native ordinary SVT-AV1 bytes only; quality and selection at final PointStream rate remain unmeasured.",
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "count": len(rows),
                "min_kbps": min(r["kbps"] for r in rows),
                "max_kbps": max(r["kbps"] for r in rows),
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--clip", choices=SOURCES, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
