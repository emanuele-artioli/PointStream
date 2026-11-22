"""Rate-only refinement of the frozen 25fps Al028 ordinary ladder."""

import hashlib
import json
from fractions import Fraction
from pathlib import Path
import resource
from src.codecs import svtav1

ROOT = Path("/private/tmp/ps-al028-av1-ladder-20261122")
OUT = Path("/private/tmp/ps-al028-av1-refine-20261122")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    start = (
        resource.getrusage(resource.RUSAGE_SELF).ru_utime
        + resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
    )
    orig = json.loads((ROOT / "result.json").read_text())
    if orig["ticks"] != 240 or orig["fps"] != 25:
        raise ValueError("base ladder mismatch")
    OUT.mkdir(exist_ok=False)
    tools = {k: svtav1.tool(k) for k in ("SvtAv1EncApp", "dav1d")}
    if any(tools[k]["sha256"] != orig["tools"][k]["sha256"] for k in tools):
        raise ValueError("native tools differ")
    rows = []
    for width, q in ((320, 51), (320, 53), (640, 62), (640, 65)):
        h = 192 if width == 320 else 384
        raw = ROOT / f"{width}.yuv"
        if raw.stat().st_size != 240 * width * h * 3 // 2:
            raise ValueError("source count")
        dest = OUT / f"{width}-crf{q}"
        dest.mkdir()
        r = svtav1.code(
            raw,
            dest,
            width=width,
            height=h,
            fps=Fraction(25),
            frames=240,
            crf=q,
            preset=4,
            threads=1,
            full_range=False,
            encoder=tools["SvtAv1EncApp"]["path"],
            decoder=tools["dav1d"]["path"],
            timeout=180,
        )
        if r["decoded_frames"] != 240 or r["temporal_units"] != 240:
            raise ValueError("decode coverage")
        rows.append(
            {
                "width": width,
                "height": h,
                "crf": q,
                "bytes": r["file_bytes"],
                "kbps": r["file_bytes"] / 1200,
                "stream_sha256": sha(Path(r["stream"])),
                "decoded_sha256": sha(Path(r["decoded"])),
            }
        )
    end = (
        resource.getrusage(resource.RUSAGE_SELF).ru_utime
        + resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime
    )
    (OUT / "result.json").write_text(
        json.dumps(
            {
                "parent_sha256": sha(ROOT / "result.json"),
                "grid": rows,
                "cpu_user_seconds": end - start,
                "scope": "Rate-only baseline refinement; no quality selection.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
