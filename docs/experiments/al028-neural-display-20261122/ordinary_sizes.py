"""Choose physical ordinary AV1 streams by bytes at three Al028 budgets."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import resource
from src.codecs import svtav1

TMP = Path("/private/tmp")
BUDGETS = (13360, 24494, 47725)
HEIGHT = {160: 96, 320: 192, 640: 384}


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def cpu():
    return sum(
        x.ru_utime + x.ru_stime
        for x in (
            resource.getrusage(resource.RUSAGE_SELF),
            resource.getrusage(resource.RUSAGE_CHILDREN),
        )
    )


def run(width, out, budgets=BUDGETS):
    out.mkdir(exist_ok=False)
    start = cpu()
    ladder = TMP / "ps-al028-av1-ladder-20261122"
    src = ladder / f"{width}.yuv"
    height = HEIGHT[width]
    source_provenance = json.loads((ladder / "result.json").read_text())
    tools = {k: svtav1.tool(k) for k in ("SvtAv1EncApp", "dav1d")}
    if any(v["sha256"] != source_provenance["tools"][k]["sha256"] for k, v in tools.items()):
        raise ValueError("Unqualified encoder/decoder")
    rows = {}

    def add(crf, root, size, native, yuv):
        stream, raw = root / "stream.ivf", root / "decoded.yuv"
        if (stream.stat().st_size, sha(stream), sha(raw)) != (size, native, yuv):
            raise ValueError("Existing native identity")
        if raw.stat().st_size != 240 * width * height * 3 // 2:
            raise ValueError("Incomplete native raw")
        rows[crf] = dict(
            crf=crf,
            width=width,
            height=height,
            bytes=size,
            kbps=size / 1200,
            sha256=native,
            raw_sha256=yuv,
            path=str(root),
        )

    for dirname in [
        ladder,
        TMP / "ps-al028-av1-refine-20261122",
        TMP / "ps-al028-av1-target-20261122",
    ]:
        for old in json.loads((dirname / "result.json").read_text())["grid"]:
            q = old["crf"]
            if old["width"] == width:
                add(
                    q,
                    dirname / f"{width}-crf{q:g}",
                    old["bytes"],
                    old["stream_sha256"],
                    old["decoded_sha256"],
                )
    if width == 320:
        for name in ["ps-nov22-al028-exact-budget", "ps-nov22-al028-matched-39771"]:
            for x in json.loads((TMP / name / "result.json").read_text())["rows"]["ordinary"]:
                add(x["crf"], Path(x["path"]), x["bytes"], x["ivf_sha256"], x["raw_sha256"])

    def probe(q):
        root = out / f"q{q:g}"
        root.mkdir()
        r = svtav1.code(
            src,
            root,
            width=width,
            height=height,
            fps=Fraction(25),
            frames=240,
            crf=q,
            preset=4,
            threads=1,
            full_range=False,
            encoder=tools["SvtAv1EncApp"]["path"],
            decoder=tools["dav1d"]["path"],
            timeout=90,
        )
        assert r["decoded_frames"] == r["temporal_units"] == 240
        stream = Path(r["stream"])
        add(q, root, stream.stat().st_size, sha(stream), sha(Path(r["decoded"])))

    old = rows[50].copy()
    t = cpu()
    probe(50)
    cost = cpu() - t
    assert rows[50]["sha256"] == old["sha256"] and rows[50]["raw_sha256"] == old["raw_sha256"]
    forecast = 1.5 * cost * 8 * len(budgets) + 30
    if forecast >= 600:
        raise ValueError(f"Full forecast outside ten CPU minutes: {forecast}")
    selected = {}
    for budget in budgets:
        for _ in range(8):
            closest = min(rows.values(), key=lambda x: (abs(x["bytes"] - budget), x["bytes"]))
            below = [r for r in rows.values() if r["bytes"] <= budget]
            above = [r for r in rows.values() if r["bytes"] > budget]
            if (
                below
                and abs(closest["bytes"] / budget - 1) <= 0.005
                and max(r["bytes"] for r in below) / budget >= 0.995
            ):
                break
            if not below:
                q = 70
            elif not above:
                q = 20
            else:
                lower = max(below, key=lambda x: x["bytes"])
                upper = min(above, key=lambda x: x["bytes"])
                lo, hi = sorted((int(lower["crf"] * 4), int(upper["crf"] * 4)))
                candidates = [i / 4 for i in range(lo + 1, hi) if i / 4 not in rows]
                if not candidates:
                    break
                q = min(candidates, key=lambda x: abs(x - (lo + hi) / 8))
            if q in rows:
                break
            if cpu() - start + 1.5 * cost + 15 > 600:
                raise ValueError("Reached CPU cap")
            probe(q)
        near = min(rows.values(), key=lambda x: (abs(x["bytes"] - budget), x["bytes"]))
        affordable = [r for r in rows.values() if r["bytes"] <= budget]
        selected[str(budget)] = {
            "nearest": near,
            "affordable": max(affordable, key=lambda x: x["bytes"]) if affordable else None,
        }
    (out / "result.json").write_text(
        json.dumps(
            dict(
                width=width,
                source=sha(src),
                tools=tools,
                selection=selected,
                rows=sorted(rows.values(), key=lambda x: x["bytes"]),
                cpu_seconds=cpu() - start,
                forecast_seconds=forecast,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("width", choices=tuple(HEIGHT), type=int)
    cli.add_argument("out", type=Path)
    cli.add_argument("--budgets", type=int, nargs="+", choices=BUDGETS, default=BUDGETS)
    a = cli.parse_args()
    run(a.width, a.out, a.budgets)
