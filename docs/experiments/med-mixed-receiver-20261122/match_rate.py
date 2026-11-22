"""Select a physical ordinary encode at each width using bytes only."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess

from experiments.tennis import common_player_packet as packet

from build import sha
from src.codecs import svtav1


def main(a):
    grid = json.loads((a.original / "result.json").read_text())
    assert grid["fps"] == 50 and grid["ticks"] == 240 and grid["original_source_sha256"]
    a.out.mkdir(exist_ok=False)
    tools = grid["tools"]
    cap = int(a.kbps * 600 + 1e-6)
    arms = []
    tried = []
    for w, h in [(160, 96), (320, 192), (480, 272), (640, 384)]:
        raw = a.original / f"{w}.yuv"
        if w == 480:
            raw = a.out / "480.yuv"
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
                    "50",
                    "-i",
                    str(a.original / "source.yuv"),
                    "-vf",
                    "scale=480:272:flags=area",
                    "-frames:v",
                    "240",
                    "-f",
                    "rawvideo",
                    "-n",
                    str(raw),
                ],
                check=True,
                timeout=120,
            )
        assert raw.stat().st_size == 240 * w * h * 3 // 2
        known = {
            int(r["crf"] * 4): {**r, "path": str(a.original / f"{w}-crf{r['crf']}" / "decoded.yuv")}
            for r in grid["grid"]
            if r["width"] == w
        }

        def measurement(q4):
            if q4 not in known:
                if a.prior:
                    old = a.prior / f"w{w}-q4{q4}"
                    stream, decoded = old / "stream.ivf", old / "decoded.yuv"
                    if stream.is_file() and decoded.is_file():
                        packet._ivf(stream.read_bytes(), w, h, (50, 1))
                        assert decoded.stat().st_size == 240 * w * h * 3 // 2
                        b = stream.stat().st_size
                        known[q4] = dict(
                            width=w,
                            height=h,
                            crf=q4 / 4,
                            bytes=b,
                            kbps=b / 600,
                            stream_sha256=sha(stream),
                            decoded_sha256=sha(decoded),
                            path=str(decoded),
                        )
                        return b
                d = a.out / f"w{w}-q4{q4}"
                d.mkdir()
                r = svtav1.code(
                    raw,
                    d,
                    width=w,
                    height=h,
                    fps=Fraction(50),
                    frames=240,
                    crf=q4 / 4,
                    preset=4,
                    threads=1,
                    full_range=False,
                    encoder=tools["SvtAv1EncApp"]["path"],
                    decoder=tools["dav1d"]["path"],
                    timeout=180,
                )
                assert r["decoded_frames"] == r["temporal_units"] == 240
                known[q4] = dict(
                    width=w,
                    height=h,
                    crf=q4 / 4,
                    bytes=r["file_bytes"],
                    kbps=r["file_bytes"] / 600,
                    stream_sha256=sha(Path(r["stream"])),
                    decoded_sha256=sha(Path(r["decoded"])),
                    path=r["decoded"],
                )
            return known[q4]["bytes"]

        lo, hi = 40 * 4, 70 * 4
        assert measurement(lo) >= cap
        if measurement(hi) > cap:
            tried.extend(known.values())
            continue
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if measurement(mid) <= cap:
                hi = mid
            else:
                lo = mid
        # Nonmonotone local QP response: retain every tried encode and check +/- one nominal CRF.
        for q in range(max(4 * 40, hi - 4), min(70 * 4, hi + 4) + 1):
            measurement(q)
        admissible = [r for r in known.values() if r["bytes"] <= cap]
        pick = max(admissible, key=lambda r: r["bytes"])
        arms.append(
            dict(
                name=f"ordinary-w{w}-near-{cap}B",
                kind="native_yuv420p",
                width=w,
                height=h,
                sha256=pick["decoded_sha256"],
                path=pick["path"],
                bytes=pick["bytes"],
                kbps=pick["kbps"],
                crf=pick["crf"],
                stream_sha256=pick["stream_sha256"],
                price_cap_bytes=cap,
            )
        )
        tried.extend(known.values())
    assert arms and all(arm["bytes"] <= cap for arm in arms)
    (a.out / "arms.json").write_text(json.dumps(arms, indent=2) + "\n")
    (a.out / "tried.json").write_text(json.dumps(tried, indent=2) + "\n")
    print([(r["width"], r["kbps"]) for r in arms])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--original", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--kbps", type=float, required=True)
    p.add_argument("--prior", type=Path)
    main(p.parse_args())
