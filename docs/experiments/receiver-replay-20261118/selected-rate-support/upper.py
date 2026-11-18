"""One CRF18 pair on the byte-verified 24- or 240-frame matched-placement anchor inputs."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import resource
import struct
import time

import numpy as np

from src.codecs import svtav1


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cpu():
    return sum(
        r.ru_stime + r.ru_utime
        for r in [
            resource.getrusage(resource.RUSAGE_SELF),
            resource.getrusage(resource.RUSAGE_CHILDREN),
        ]
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ["prior", "archive", "output", "rates"]:
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--smoke-result", type=Path)
    a = p.parse_args()
    t = time.monotonic()
    c = cpu()
    old = json.loads(a.prior.read_text())
    n = old["frames"]
    assert n in (24, 240)
    rates = json.loads(a.rates.read_text())
    target = max(r["joint_kbps"] for r in rates["rows"])
    assert target > min(x["complete_kbps"] for x in old["rows"]) or n == 24
    smoke_wall = smoke_cpu = 0.0
    if n == 240:
        s = json.loads(a.smoke_result.read_text())
        smoke_wall = s["wall_seconds"]
        smoke_cpu = s["cpu_seconds"]
        assert s["passed"] and smoke_wall * 21 < 180 and smoke_cpu * 21 < 600
    a.output.mkdir(exist_ok=False)
    source = a.archive / "source.yuv"
    region = a.archive / "roi.txt"
    assert sha(source) == old["scaled_yuv_sha256"] and sha(region) == old["map_sha256"]
    for v in old["tools"].values():
        assert sha(Path(v["path"])) == v["sha256"]
    frames = np.memmap(source, np.uint8, mode="r", shape=(n, 816, 960))
    rows = []
    for mode in ["plain", "roi"]:
        work = a.output / mode
        work.mkdir()
        allowed = 180 - smoke_wall - (time.monotonic() - t)
        assert allowed > 0
        row = svtav1.code(
            source,
            work,
            width=960,
            height=544,
            fps=Fraction(50),
            frames=n,
            crf=18,
            preset=4,
            threads=2,
            full_range=False,
            encoder=old["tools"]["SvtAv1EncApp"]["path"],
            decoder=old["tools"]["dav1d"]["path"],
            timeout=min(120, allowed),
            extra=["--roi-map-file", str(region)] if mode == "roi" else [],
        )
        prior = next(r for r in old["rows"] if r["mode"] == mode)
        before = prior["encode_command"].copy()
        after = row["encode_command"].copy()
        for option in ["-i", "-b", "--crf"] + (["--roi-map-file"] if mode == "roi" else []):
            before[before.index(option) + 1] = after[after.index(option) + 1]
        assert before == after and row["decoded_frames"] == n
        blob = Path(row["stream"]).read_bytes()
        lengths = svtav1.ivf_frames(blob)
        assert (
            struct.unpack_from("<HHIIII", blob, 12) == (960, 544, 50, 1, n, 0) and len(lengths) == n
        )
        off = 32
        for tick, sz in enumerate(lengths):
            assert struct.unpack_from("<IQ", blob, off) == (sz, tick)
            off += 12 + sz
        assert (
            off == len(blob)
            and row["file_bytes"] == len(blob)
            and Path(row["decoded"]).stat().st_size == n * 960 * 816
        )
        ys = np.memmap(row["decoded"], np.uint8, mode="r", shape=(n, 816, 960))
        se = 0.0
        for i in range(n):
            se += np.square(np.subtract(ys[i, :544], frames[i, :544], dtype=np.float32)).sum(
                dtype=np.float64
            )
        rows.append(
            {
                **row,
                "mode": mode,
                "crf": 18,
                "complete_kbps": len(blob) * 8 * 50 / (n * 1000),
                "whole_luma_psnr": 10 * math.log10(255**2 / (se / (n * 960 * 544))),
                "decoded_sha256": sha(Path(row["decoded"])),
            }
        )
    assert (
        rows[1]["file_bytes"] < rows[0]["file_bytes"]
        and rows[1]["decoded_sha256"] != rows[0]["decoded_sha256"]
    )
    elapsed = time.monotonic() - t
    used = cpu() - c
    assert smoke_wall + elapsed < 180 and smoke_cpu + used < 600
    result = {
        "passed": True,
        "same_source": old["source_sha256"],
        "source_yuv_sha256": sha(source),
        "matched_map_sha256": sha(region),
        "track_sha256": old["ball_track_sha256"],
        "prior_record_sha256": sha(a.prior),
        "script_sha256": sha(Path(__file__)),
        "tools": old["tools"],
        "gpu": "none",
        "frames": n,
        "wall_seconds": elapsed,
        "cpu_seconds": used,
        "max_joint_target_kbps": target,
        "rows": rows,
    }
    (a.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(n, elapsed, used, [(r["mode"], r["complete_kbps"]) for r in rows])


if __name__ == "__main__":
    main()
