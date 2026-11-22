"""Generate a frozen source-only SVT-AV1 rate ladder for the Al028 rally."""

import hashlib
import json
from fractions import Fraction
from pathlib import Path
import resource
import subprocess

from src.codecs import svtav1

ROOT = Path("/private/tmp/ps-al028-donor-base-20261122")
OUT = Path("/private/tmp/ps-al028-av1-ladder-20261122")
SOURCE_SHA = "4f1185782292ebcfbf4952dae111acf867578c8fc83a7e57778b1bfca022c186"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cpu():
    return sum(
        resource.getrusage(k).ru_utime + resource.getrusage(k).ru_stime
        for k in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)
    )


def main():
    source = ROOT / "source.yuv"
    if sha(source) != SOURCE_SHA or source.stat().st_size != 240 * 960 * 544 * 3 // 2:
        raise ValueError("source input identity")
    start = cpu()
    OUT.mkdir(exist_ok=False)
    tools = {k: svtav1.tool(k) for k in ("ffmpeg", "SvtAv1EncApp", "dav1d")}
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
                "25",
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
            fps=Fraction(25),
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
    forecast = (cpu() - start) + smoke * 10 * 9 * 1.5 + 60
    if forecast > 1200:
        raise ValueError(f"baseline projected CPU seconds {forecast} exceed cap")
    rows = []
    for width in (160, 320, 640):
        for q in (40, 50, 60):
            r = encode(width, q, 240, f"{width}-crf{q}")
            rows.append(
                {
                    "width": width,
                    "height": inputs[width][0],
                    "crf": q,
                    "bytes": r["file_bytes"],
                    "kbps": r["file_bytes"] / 1200,
                    "stream_sha256": sha(Path(r["stream"])),
                    "decoded_sha256": sha(Path(r["decoded"])),
                }
            )
    result = {
        "source_yuv_sha256": SOURCE_SHA,
        "ticks": 240,
        "fps": 25,
        "tools": tools,
        "grid": rows,
        "smoke_cpu": smoke,
        "forecast_cpu": forecast,
        "total_cpu": cpu() - start,
        "scope": "Native ordinary SVT-AV1 bytes only; quality and selection at final PointStream rate remain unmeasured.",
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
