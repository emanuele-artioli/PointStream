"""Calibrate honest affordable ordinary streams at the separately costed Shanghai joint ceilings."""

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
import zlib

from experiments.tennis.shanghai_ball_rate_crf import TOOLS, sha
from src.codecs import svtav1

CEILINGS = {"warm": 17794, "cold": 155274}
GEOMETRY = {320: 192, 480: 272, 640: 384, 960: 544, 1920: 1080}
FPS = Fraction(30000, 1001)


def run(a):
    begin = time.monotonic()
    n = a.frames
    w = a.width
    h = GEOMETRY[w]
    assert n == {"smoke": 24, "full": 240}[os.environ["PS_STAGE"]]
    assert len(os.sched_getaffinity(0)) == 1 and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    meta = json.loads(a.metadata.read_text())
    assert (
        meta["match"] == "match102"
        and sha(a.source) == meta["source"]["sha256"]
        and Fraction(meta["video"]["avg_frame_rate"]) == FPS
    )
    assert sha(a.earlier) == a.earlier_sha
    tools = {k: svtav1.tool(k) for k in TOOLS}
    assert all(v["sha256"] == TOOLS[k] for k, v in tools.items())
    work = Path(os.environ["PS_SCRATCH_DIR"]) / "native"
    work.mkdir()
    pub = Path(os.environ["PS_SCRATCH_DIR"]) / "publish"
    pub.mkdir()
    native = work / "source.yuv"
    cmd = [
        tools["ffmpeg"]["path"],
        "-v",
        "error",
        "-threads",
        "1",
        "-nostdin",
        "-i",
        str(a.source),
        "-an",
        "-frames:v",
        str(n),
        "-vf",
        f"scale={w}:{h}:flags=area,format=yuv420p",
        "-fps_mode",
        "passthrough",
        "-f",
        "rawvideo",
        str(native),
    ]
    subprocess.run(cmd, check=True, capture_output=True, timeout=270)
    assert native.stat().st_size == n * w * h * 3 // 2
    rows = {}
    with tarfile.open(a.earlier) as t:
        previous = json.load(t.extractfile("publish/rates.json"))
        assert (
            previous["frames"] == 240
            and previous["match"] == "match102"
            and previous["source_sha256"] == meta["source"]["sha256"]
        )
        for old in previous["rows"] if n == 240 else []:
            if old["width"] != w:
                continue
            data = t.extractfile("publish/" + old["native_file"]).read()
            assert (
                hashlib.sha256(data).hexdigest() == old["native_sha256"]
                and len(data) == old["bytes"]
                and len(svtav1.ivf_frames(data)) == n
            )
            q = round(old["crf"] * 4)
            path = pub / f"q{q}.ivf"
            path.write_bytes(data)
            rows[q] = dict(
                quarter_crf=q,
                bytes=len(data),
                file=path.name,
                sha256=old["native_sha256"],
                origin="frozen bank",
            )

    def encode(q):
        if q not in rows:
            path = pub / f"q{q}.ivf"
            started = time.monotonic()
            c = svtav1.encode_command(
                tools["SvtAv1EncApp"]["path"],
                native,
                path,
                width=w,
                height=h,
                fps=FPS,
                frames=n,
                crf=q / 4,
                preset=4,
                threads=1,
                full_range=False,
            )
            subprocess.run(c, check=True, capture_output=True, timeout=500)
            data = path.read_bytes()
            assert len(svtav1.ivf_frames(data)) == n
            rows[q] = dict(
                quarter_crf=q,
                bytes=len(data),
                file=path.name,
                sha256=hashlib.sha256(data).hexdigest(),
                origin="new",
                elapsed=time.monotonic() - started,
            )
        return rows[q]["bytes"]

    # A 24-frame smoke uses representative working limits; selection for the paper uses only full files.
    selected = {}
    floor = encode(280)
    for label, cap in CEILINGS.items():
        limit = round(Fraction(n, 240) * cap)
        if floor <= limit:
            encode(64)
            for step in range(10):
                under = [q for q, r in rows.items() if r["bytes"] <= limit]
                over = [q for q, r in rows.items() if r["bytes"] > limit and q < min(under)]
                if not over:
                    break
                high = min(under)
                low = max(over)
                middle = (low + high) // 2
                if middle == low or middle == high:
                    break
                encode(middle)
            best = max(
                (r for r in rows.values() if r["bytes"] <= limit),
                key=lambda r: (r["bytes"], -r["quarter_crf"]),
            )
            # Nearby fractional settings protect against a minor reversal in the file-size curve.
            for q in (best["quarter_crf"] - 1, best["quarter_crf"] + 1):
                if 64 <= q <= 280:
                    encode(q)
    for label, cap in CEILINGS.items():
        limit = round(Fraction(n, 240) * cap)
        affordable = [r for r in rows.values() if r["bytes"] <= limit]
        if not affordable:
            selected[label] = dict(
                ceiling=limit, affordable=False, floor_bytes=min(r["bytes"] for r in rows.values())
            )
            continue
        best = max(affordable, key=lambda r: (r["bytes"], -r["quarter_crf"]))
        path = pub / best["file"]
        dec = work / (label + ".yuv")
        subprocess.run(
            svtav1.decode_command(tools["dav1d"]["path"], path, dec, threads=1),
            check=True,
            capture_output=True,
            timeout=250,
        )
        assert dec.stat().st_size == n * w * h * 3 // 2
        decoded_sha = sha(dec)
        out = pub / (label + ".yuv.z")
        z = zlib.compressobj(1)
        with dec.open("rb") as src, out.open("wb") as dst:
            while block := src.read(1048576):
                dst.write(z.compress(block))
            dst.write(z.flush())
        selected[label] = dict(
            ceiling=limit,
            affordable=True,
            **best,
            decoded_sha256=decoded_sha,
            decoded_file=out.name,
            shortfall=1 - best["bytes"] / limit,
        )
    r = dict(
        match=meta["match"],
        source_sha256=meta["source"]["sha256"],
        earlier_sha256=a.earlier_sha,
        frames=n,
        fps=str(FPS),
        width=w,
        height=h,
        tools=tools,
        convert_command=cmd,
        source_yuv_sha256=sha(native),
        selected=selected,
        bank=[rows[q] for q in sorted(rows)],
        elapsed=time.monotonic() - begin,
    )
    for path in (pub / "anchors.json", Path(os.environ["PS_STAGE_DIR"]) / "anchors.json"):
        path.write_text(json.dumps(r, indent=2, allow_nan=False) + "\n")


def validate():
    stage = Path(os.environ["PS_STAGE_DIR"])
    r = json.loads((stage / "anchors.json").read_text())
    n = r["frames"]
    with tarfile.open(stage / "published.tar") as tar:
        assert json.load(tar.extractfile("publish/anchors.json")) == r
        for row in r["bank"]:
            b = tar.extractfile("publish/" + row["file"]).read()
            assert (
                len(b) == row["bytes"]
                and hashlib.sha256(b).hexdigest() == row["sha256"]
                and len(svtav1.ivf_frames(b)) == n
            )
        for key, x in r["selected"].items():
            assert x["ceiling"] == round(Fraction(n, 240) * CEILINGS[key])
            if x["affordable"]:
                assert x["bytes"] <= x["ceiling"] and x["bytes"] == max(
                    v["bytes"] for v in r["bank"] if v["bytes"] <= x["ceiling"]
                )
                src = tar.extractfile("publish/" + x["decoded_file"])
                dec = zlib.decompressobj()
                digest = hashlib.sha256()
                length = 0
                while buf := src.read(65536):
                    raw = dec.decompress(buf)
                    digest.update(raw)
                    length += len(raw)
                tail = dec.flush()
                digest.update(tail)
                length += len(tail)
                assert (
                    dec.eof
                    and not dec.unused_data
                    and digest.hexdigest() == x["decoded_sha256"]
                    and length == n * r["width"] * r["height"] * 3 // 2
                )
            else:
                assert all(v["bytes"] > x["ceiling"] for v in r["bank"])
    # Full encodes may reuse the earlier bank. Charge all measured smoke probes conservatively.
    forecast = 1.5 * r["elapsed"] * 240 / n + 45
    checks = dict(
        source=True, native=True, delivery=True, lossless=True, forecast=n == 240 or forecast < 2100
    )
    passed = all(checks.values())
    Path(os.environ["PS_VALIDATION_PATH"]).write_text(
        json.dumps(dict(passed=passed, checks=checks, forecast=forecast)) + "\n"
    )
    if not passed:
        raise ValueError(checks)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("source", "metadata", "earlier"):
        p.add_argument("--" + k, type=Path)
    p.add_argument("--earlier-sha")
    p.add_argument("--frames", type=int)
    p.add_argument("--width", type=int, choices=GEOMETRY)
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    validate() if args.validate else run(args)
