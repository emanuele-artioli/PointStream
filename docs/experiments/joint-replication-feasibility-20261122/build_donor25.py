"""Build the Al028 25fps carrier on its authenticated donor-filled base."""

import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path

from src.codecs import svtav1

import build_common25 as builder

ROOT = Path("/private/tmp/ps-al028-donor-base-20261122")
OUT = Path("/private/tmp/ps-al028-donor25-packet-20261122")
EXPECTED = "5895cb05c85ee72f57d919eb15278ad6043d842eb1ca30cd4844ad1486ada7af"


def main():
    source = ROOT / "clean.yuv"
    if (
        hashlib.sha256(source.read_bytes()).hexdigest() != EXPECTED
        or source.stat().st_size != 240 * 960 * 544 * 3 // 2
    ):
        raise ValueError("donor input identity")
    tmp = Path("/private/tmp/ps-al028-donor25-encode-20261122")
    tmp.mkdir(exist_ok=False)
    tools = {x: svtav1.tool(x) for x in ("ffmpeg", "SvtAv1EncApp", "dav1d")}
    small = tmp / "small.yuv"
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
            "scale=160:96:flags=area",
            "-frames:v",
            "240",
            "-f",
            "rawvideo",
            str(small),
        ],
        check=True,
        timeout=90,
    )
    native = tmp / "native"
    native.mkdir()
    row = svtav1.code(
        small,
        native,
        width=160,
        height=96,
        fps=Fraction(25),
        frames=240,
        crf=70,
        preset=4,
        threads=1,
        full_range=False,
        encoder=tools["SvtAv1EncApp"]["path"],
        decoder=tools["dav1d"]["path"],
        timeout=120,
    )
    if row["decoded_frames"] != 240 or row["temporal_units"] != 240:
        raise ValueError("native encode/decode mismatch")
    builder.BASE = Path(row["stream"])
    builder.OUT = OUT
    builder.main()
    receipt = json.loads((OUT / "receipt.json").read_text())
    receipt.update(
        donor_yuv_sha256=EXPECTED,
        ball_payload_sha256="16214ff3ac718e6b28ae3cb21778c8440057c1e68f030e5ac904f20528f18883",
        ball_payload_bytes=996,
        base_crf=70,
        tools=tools,
        native=row,
        scope="Authenticated donor-filled base and 25fps carrier only; no common-model replay, human quality, or total delivered stream yet.",
    )
    (OUT / "donor-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
