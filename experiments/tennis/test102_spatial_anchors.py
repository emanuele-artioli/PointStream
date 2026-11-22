"""Calibrate native conventional streams at the measured joint receiver byte ceilings."""

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
from experiments.tennis import test102_roi_map as roi

CEILINGS = {"warm": 17794, "cold": 155274}
GEOMETRY = {320: 192, 480: 272, 640: 384, 960: 544, 1920: 1080}
FPS = Fraction(30000, 1001)


def operating_points(a):
    if not a.ceiling:
        if a.match != "match102":
            raise ValueError("A new joint receiver requires measured byte ceilings")
        return CEILINGS
    pairs = [item.split("=") for item in a.ceiling]
    ceilings = {label: int(size) for label, size in pairs}
    if len(ceilings) != len(pairs) or any(
        not name.isalnum() or size <= 0 for name, size in ceilings.items()
    ):
        raise ValueError("Distinct alphanumeric names and positive byte ceilings are required")
    return ceilings


def run(a):
    begin = time.monotonic()
    n = a.frames
    w = a.width
    h = GEOMETRY[w]
    fps = a.fps
    assert n == {"smoke": 24, "full": 240}[os.environ["PS_STAGE"]]
    assert len(os.sched_getaffinity(0)) == 1 and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    meta = json.loads(a.metadata.read_text())
    assert (
        meta["match"] == a.match
        and sha(a.source) == meta["source"]["sha256"]
        and Fraction(meta["video"]["avg_frame_rate"]) == fps
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
    roi_path = None
    regions = None
    budgets = operating_points(a)
    if a.roi_features:
        assert a.ball_payload
        roi_path, regions = roi.region_map(a.roi_features, a.ball_payload, n, w, h, work, a.match)
        budgets = {k: budgets[k] for k in (a.roi_ceiling or ["warm"])}
    rows = {}
    with tarfile.open(a.earlier) as t:
        previous = json.load(t.extractfile("publish/rates.json"))
        assert (
            previous["frames"] == 240
            and previous["match"] == a.match
            and previous["source_sha256"] == meta["source"]["sha256"]
        )
        for old in previous["rows"] if n == 240 and not roi_path else []:
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
                fps=fps,
                frames=n,
                crf=q / 4,
                preset=4,
                full_range=False,
                extra=["--roi-map-file", str(roi_path)] if roi_path else None,
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
    decoded_by_q = {}
    for label, cap in budgets.items():
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
    for label, cap in budgets.items():
        limit = round(Fraction(n, 240) * cap)
        affordable = [r for r in rows.values() if r["bytes"] <= limit]
        if not affordable:
            selected[label] = dict(
                ceiling=limit, affordable=False, floor_bytes=min(r["bytes"] for r in rows.values())
            )
            continue
        best = max(affordable, key=lambda r: (r["bytes"], -r["quarter_crf"]))
        q = best["quarter_crf"]
        if q in decoded_by_q:
            selected[label] = dict(
                decoded_by_q[q], ceiling=limit, shortfall=1 - best["bytes"] / limit
            )
            continue
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
        twin = (
            roi.compare_same_q(native, dec, best, regions, w, h, n, fps, tools, work, pub)
            if roi_path
            else None
        )
        selected[label] = dict(
            ceiling=limit,
            affordable=True,
            identical_qp_control=twin,
            **best,
            decoded_sha256=decoded_sha,
            decoded_file=out.name,
            shortfall=1 - best["bytes"] / limit,
        )
        decoded_by_q[q] = selected[label]
    r = dict(
        match=meta["match"],
        source_sha256=meta["source"]["sha256"],
        earlier_sha256=a.earlier_sha,
        mode="foreground-roi" if roi_path else "ordinary",
        roi_map_sha256=sha(roi_path) if roi_path else None,
        source_masks_sha256=roi.SOURCES[a.match][0] if roi_path else None,
        ball_sha256=roi.SOURCES[a.match][1] if roi_path else None,
        frames=n,
        fps=str(fps),
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


def validate(a):
    stage = Path(os.environ["PS_STAGE_DIR"])
    r = json.loads((stage / "anchors.json").read_text())
    n = r["frames"]
    caps = operating_points(a)
    assert r["match"] == a.match and Fraction(r["fps"]) == a.fps
    expected = set(a.roi_ceiling or ["warm"]) if r["mode"] == "foreground-roi" else set(caps)
    assert set(r["selected"]) == expected
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
            assert x["ceiling"] == round(Fraction(n, 240) * caps[key])
            if x["affordable"]:
                assert x["bytes"] <= x["ceiling"] and x["bytes"] == max(
                    v["bytes"] for v in r["bank"] if v["bytes"] <= x["ceiling"]
                )
                if r.get("mode") == "foreground-roi":
                    t = x["identical_qp_control"]
                    twin = tar.extractfile("publish/" + t["twin_file"]).read()
                    assert (
                        len(twin) == t["twin_bytes"]
                        and hashlib.sha256(twin).hexdigest() == t["twin_sha256"]
                        and len(svtav1.ivf_frames(twin)) == n
                    )
                    assert t["support_pixels"] + t["outside_pixels"] == r["width"] * r["height"] * n
                    assert t["non_inert"] == (
                        t["changed_luma_pixels"] > 0
                        and t["twin_decoded_sha256"] != x["decoded_sha256"]
                    )
                    assert (
                        r["roi_map_sha256"]
                        and r["source_masks_sha256"] == roi.SOURCES[a.match][0]
                        and r["ball_sha256"] == roi.SOURCES[a.match][1]
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
    for k in ("source", "metadata", "earlier", "roi-features", "ball-payload"):
        p.add_argument("--" + k, type=Path)
    p.add_argument("--earlier-sha")
    p.add_argument("--frames", type=int)
    p.add_argument("--match", choices=roi.SOURCES, default="match102")
    p.add_argument("--fps", type=Fraction, default=FPS)
    p.add_argument("--ceiling", action="append", metavar="NAME=BYTES")
    p.add_argument("--roi-ceiling", action="append", metavar="NAME")
    p.add_argument("--width", type=int, choices=GEOMETRY)
    p.add_argument("--validate", action="store_true")
    args = p.parse_args()
    validate(args) if args.validate else run(args)
