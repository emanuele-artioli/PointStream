"""Source-only 320×192 donor-base size bracket for the 22 kbps joint budget."""

import hashlib
import json
from fractions import Fraction
from pathlib import Path
import subprocess
from src.codecs import svtav1

SOURCE = Path("/private/tmp/ps-al028-donor-base-20261122/clean.yuv")
OUT = Path("/private/tmp/ps-al028-donor-rate-20261122")


def main():
    if (
        hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        != "5895cb05c85ee72f57d919eb15278ad6043d842eb1ca30cd4844ad1486ada7af"
    ):
        raise ValueError("clean source identity")
    OUT.mkdir(exist_ok=False)
    tools = {k: svtav1.tool(k) for k in ("ffmpeg", "SvtAv1EncApp", "dav1d")}
    raw = OUT / "input.yuv"
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
            str(SOURCE),
            "-vf",
            "scale=320:192:flags=area",
            "-frames:v",
            "240",
            "-f",
            "rawvideo",
            str(raw),
        ],
        check=True,
        timeout=90,
    )
    if raw.stat().st_size != 240 * 320 * 192 * 3 // 2:
        raise ValueError("input frames")
    rows = []
    for q in (52, 55, 58):
        dest = OUT / f"crf{q}"
        dest.mkdir()
        r = svtav1.code(
            raw,
            dest,
            width=320,
            height=192,
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
            raise ValueError("decode count")
        rows.append(
            {
                "crf": q,
                "base_bytes": r["file_bytes"],
                "base_sha256": hashlib.sha256(Path(r["stream"]).read_bytes()).hexdigest(),
                "estimated_warm_kbps": (r["file_bytes"] - 8858 + 13360) / 1200,
            }
        )
    (OUT / "result.json").write_text(
        json.dumps(
            {
                "rows": rows,
                "tools": tools,
                "scope": "Source-only rate bracket; no visual-quality selection.",
            },
            indent=2,
        )
        + "\n"
    )
    print(rows)


if __name__ == "__main__":
    main()
