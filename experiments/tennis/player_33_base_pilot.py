"""Fixed al034 one-window base/anchor rate pilot using original W7 conversion."""

import argparse
import hashlib
import json
import os
from fractions import Fraction
from pathlib import Path
import subprocess
import tarfile
import time

from experiments.tennis.k4b_features import convert_command
from experiments.tennis.k6 import downscaled_source
from src.codecs import svtav1

SOURCE = "7db0c143cd416ad17b4bda4eaa191d51da197eccc1143e9eb402f8bb2a7f0ee6"
SOURCE_YUV = "3734897c6b329649ff886c053a9c6fe0bb215a30bec5629934fdc5fd10b0d317"
ORIGINAL = "bb9bb58d6101b947cdad32c371f9755d3c635b18ee29e0fccf498861b2be7479"
ORIGINAL_DECODE = "d764dd1252019456026211c856ffa501312c9dcfe3ad149369adb3d4029695dd"
SIDE = 4.347078333333333 + 4.5 * 164079 * 8 / 7_200_000
WINDOWS = {
    "al028": (
        "c51dace59349914da401356984de4e83da658c878011db3f54ff233077152934",
        "5788a0f1b0b2ce3c8d57ea842807a43dda1f9fa3cc06cd90c3aa4b27ff31d22a",
        "b206613e503f525ec28f6e277d176ea71eff294f31eb898adecac658433da843",
        "bebf7d41b70351b0df09ab658fdcf5e68b60fe8e197ae57f10bfd7826e970bb6",
        5.119,
    ),
    "al043": (
        "847cb8272134bd31a5447ff3cce02b28a906fae44559b3e0a58661d8171b6c16",
        "0d90ad9784493401561e4390ca83d31c9dc40ac1752eb9ee40ef75190d0837d6",
        "9b58ceabe7cf8f04426da3451b1afeadcd15f8a29a517950e6b99a5e91be3a09",
        "b0c69928f40e3dd7198ea7d9fe8e9622023aa3425dc1dff04e7c0e54e6d07abd",
        5.04804,
    ),
}


def configure(window):
    global SOURCE, SOURCE_YUV, ORIGINAL, ORIGINAL_DECODE, SIDE
    if window in WINDOWS:
        SOURCE, SOURCE_YUV, ORIGINAL, ORIGINAL_DECODE, SIDE = WINDOWS[window]


def digest(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def run(a):
    start = time.monotonic()
    stage = os.environ["PS_STAGE"]
    n = 24 if stage == "smoke" else 240
    if a.frames != n or digest(a.source) != SOURCE or len(os.sched_getaffinity(0)) != 1:
        raise ValueError("source or stage mismatch")
    root = Path(os.environ["PS_SCRATCH_DIR"])
    publish = root / "publish"
    publish.mkdir()
    source = root / "source.yuv"
    subprocess.run(convert_command("ffmpeg", a.source, source, 0, n), check=True, timeout=300)
    if n == 240 and digest(source) != SOURCE_YUV:
        raise ValueError("original source conversion changed")
    small = downscaled_source(source, n, 2, root / "small.yuv")
    encoder, decoder = svtav1.tool("SvtAv1EncApp"), svtav1.tool("dav1d")
    rows = []
    for q in [63, 61] if n == 24 else [63, 62, 61, 60]:
        folder = root / f"q{q}"
        folder.mkdir()
        record = svtav1.code(
            small,
            folder,
            width=960,
            height=544,
            fps=Fraction(25),
            frames=n,
            crf=q,
            preset=4,
            threads=1,
            full_range=False,
            encoder=encoder["path"],
            decoder=decoder["path"],
            timeout=600,
        )
        stream, decoded = Path(record["stream"]), Path(record["decoded"])
        stream_hash, decoded_hash = digest(stream), digest(decoded)
        if n == 240 and q == 63 and (stream_hash != ORIGINAL or decoded_hash != ORIGINAL_DECODE):
            raise ValueError("original W7 encoder/decoder control failed")
        data = stream.read_bytes()
        target = publish / f"q{q}.ivf"
        target.write_bytes(data)
        rate = len(data) * 8 * 25 / n / 1000
        rows.append(
            {
                "crf": q,
                "base_kbps": rate,
                "modeled_player_kbps": rate + SIDE,
                "bytes": len(data),
                "stream_sha256": stream_hash,
                "decoded_sha256": decoded_hash,
                "frames": record["decoded_frames"],
            }
        )
    result = {
        "stage": stage,
        "frames": n,
        "source_sha256": SOURCE,
        "source_yuv_sha256": digest(source),
        "corrected_side_kbps": SIDE,
        "rows": rows,
        "elapsed_seconds": time.monotonic() - start,
        "encoder": encoder,
        "decoder": decoder,
    }
    text = json.dumps(result, indent=2) + "\n"
    (publish / "result.json").write_text(text)
    (Path(os.environ["PS_STAGE_DIR"]) / "result.json").write_text(text)


def validate(a):
    root = Path(os.environ["PS_STAGE_DIR"])
    result = json.loads((root / "result.json").read_text())
    n = 24 if result["stage"] == "smoke" else 240
    checks = {
        "stage": result["frames"] == n,
        "grid": len(result["rows"]) == (2 if n == 24 else 4),
        "frames": all(x["frames"] == n for x in result["rows"]),
    }
    if n == 24:
        checks["forecast"] = result["elapsed_seconds"] * 10 * 2 * 1.5 + 120 < a.full_limit
    else:
        checks["original"] = (
            next(x for x in result["rows"] if x["crf"] == 63)["stream_sha256"] == ORIGINAL
        )
    with tarfile.open(root / "published.tar") as archive:
        for row in result["rows"]:
            data = archive.extractfile(f"publish/q{row['crf']}.ivf").read()
            checks[str(row["crf"])] = (
                hashlib.sha256(data).hexdigest() == row["stream_sha256"]
                and len(svtav1.ivf_frames(data)) == n
            )
    report = {"passed": all(checks.values()), "checks": checks}
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(json.dumps(report) + "\n")
    if not report["passed"]:
        raise SystemExit(str(checks))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path)
    p.add_argument("--window", choices=("al028", "al034", "al043"), default="al034")
    p.add_argument("--frames", type=int)
    p.add_argument("--full-limit", type=int, required=True)
    p.add_argument("--validate", action="store_true")
    a = p.parse_args()
    configure(a.window)
    validate(a) if a.validate else run(a)


if __name__ == "__main__":
    main()
