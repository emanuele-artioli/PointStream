"""Price one achievable ordinary AV1 encode at the cold mixed-stream ceiling."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import shutil
import tempfile
from build import sha
from av1 import cpu
from src.codecs import svtav1


def main(a):
    receipt = json.loads((a.joint / "result.json").read_text())
    chosen = next(r for r in receipt["rates"] if r["crf"] == receipt["selected_crf"])
    target = chosen["joint_bytes"] + chosen["init_bytes"]
    ordinary = json.loads((a.ordinary / "result.json").read_text())
    src = a.ordinary / "source.yuv"
    fps = ordinary["fps"]
    assert ordinary["ticks"] == 240 and fps in (25, 50)
    if fps == 25:
        assert receipt["source_sha256"] == ordinary["original_source_sha256"]
        assert str(fps) == receipt["native_fps"]
        sender_source = sha(a.joint / "source.yuv")
    else:
        sender_source = receipt["donor"]["source_yuv_sha256"]
    assert sha(src) == ordinary["source_yuv_sha256"] == sender_source
    tools = ordinary["tools"]
    a.out.mkdir(exist_ok=False)
    rows = {}
    clock = cpu()

    def run(q4, n):
        with tempfile.TemporaryDirectory(dir=a.out) as local:
            d = Path(local)
            r = svtav1.code(
                src,
                d,
                width=960,
                height=544,
                fps=Fraction(fps),
                frames=n,
                crf=q4 / 4,
                preset=4,
                threads=1,
                full_range=False,
                encoder=tools["SvtAv1EncApp"]["path"],
                decoder=tools["dav1d"]["path"],
                timeout=240,
            )
            assert r["decoded_frames"] == r["temporal_units"] == n
            stream = a.out / f"n{n}q4{q4}.ivf"
            shutil.copy(r["stream"], stream)
            return dict(
                q4=q4,
                crf=q4 / 4,
                bytes=r["file_bytes"],
                kbps=r["file_bytes"] * 8 * fps / (n * 1000),
                stream=str(stream),
                stream_sha256=sha(stream),
                decoded_sha256=sha(Path(r["decoded"])),
            )

    t = cpu()
    run(42 * 4, 24)
    smoke = cpu() - t
    if 1.5 * smoke * 10 * 11 + 45 > 1200:
        raise RuntimeError("24 tick cold-price CPU forecast exceeds 20 minutes")
    before = cpu()
    for x in [28, 42, 56]:
        rows[x * 4] = run(x * 4, 240)
    average = (cpu() - before) / 3
    if cpu() - clock + 1.5 * average * 8 + 35 > 1200:
        raise RuntimeError("full cold-price CPU forecast exceeds 20 minutes")
    lo_candidates = [q for q, r in rows.items() if r["bytes"] >= target]
    hi_candidates = [q for q, r in rows.items() if r["bytes"] <= target]
    if not lo_candidates:
        rows[12 * 4] = run(12 * 4, 240)
        lo_candidates = [12 * 4]
    if not hi_candidates:
        rows[70 * 4] = run(70 * 4, 240)
        hi_candidates = [70 * 4]
    lo = max(lo_candidates)
    hi = min(hi_candidates)
    assert rows[lo]["bytes"] >= target >= rows[hi]["bytes"] and lo <= hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        rows[mid] = run(mid, 240)
        if rows[mid]["bytes"] <= target:
            hi = mid
        else:
            lo = mid
    pick = max((x for x in rows.values() if x["bytes"] <= target), key=lambda x: x["bytes"])
    path = a.out / "selected.yuv"
    import subprocess

    subprocess.run(
        [
            tools["dav1d"]["path"],
            "-i",
            pick["stream"],
            "-o",
            str(path),
            "--muxer",
            "yuv",
            "--threads",
            "1",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    assert sha(path) == pick["decoded_sha256"] and path.stat().st_size == 240 * 960 * 544 * 3 // 2
    arm = dict(
        name="ordinary-960-cold",
        kind="native_yuv420p",
        width=960,
        height=544,
        sha256=sha(path),
        path=str(path),
        stream_sha256=pick["stream_sha256"],
        bytes=pick["bytes"],
        kbps=pick["kbps"],
        crf=pick["crf"],
        cold_ceiling_bytes=target,
    )
    (a.out / "arms.json").write_text(json.dumps([arm], indent=2) + "\n")
    (a.out / "search.json").write_text(
        json.dumps(
            dict(
                target=target,
                grid=list(rows.values()),
                cpu_seconds=cpu() - clock,
                smoke_cpu=smoke,
                source_sha256=ordinary["source_yuv_sha256"],
                tools=tools,
            ),
            indent=2,
        )
        + "\n"
    )
    print(arm)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("joint", "ordinary", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    main(p.parse_args())
