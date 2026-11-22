"""Select ordinary controls under the strict 20 kbps cap, then measure their fidelity."""

import argparse
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import time
import zipfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("joint_score", HERE / "score.py")
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)
q = score.quarter
import motion  # noqa: E402
from src.codecs import svtav1  # noqa: E402


def choose(args):
    eligible = []
    summary = []
    start = time.monotonic()
    for w, h in ((160, 96), (240, 136), (480, 272)):
        old = args.controls / f"20-{w}"
        d = json.loads((old / "result.json").read_text())
        for mode in ("plain", "roi"):
            records = {r["q"]: r for r in d["all_encodes"] if r["n"] == 240 and r["mode"] == mode}
            assert max(records) == 70
            if min(r["file_bytes"] for r in records.values()) > 12000:
                summary.append(
                    dict(
                        width=w,
                        mode=mode,
                        minimum_tested_bytes=min(r["file_bytes"] for r in records.values()),
                    )
                )
                continue
            source = old / f"{w}.yuv"
            assert q.sha(source) == d["scaled_source_sha256"]
            tools = d["tools"]
            assert all(q.sha(Path(t["path"])) == t["sha256"] for t in tools.values())
            while True:
                ordered = sorted(records)
                pairs = [
                    (lo, hi)
                    for lo, hi in zip(ordered, ordered[1:])
                    if records[lo]["file_bytes"] > 12000 >= records[hi]["file_bytes"]
                    and hi - lo >= 0.5
                ]
                if (
                    not pairs
                    or min(
                        12000 - r["file_bytes"]
                        for r in records.values()
                        if r["file_bytes"] <= 12000
                    )
                    <= 60
                ):
                    break
                lo, hi = pairs[0]
                value = round(2 * (lo + hi)) / 4
                out = args.output / f"{mode}-{w}-{value}"
                out.mkdir()
                r = svtav1.code(
                    source,
                    out,
                    width=w,
                    height=h,
                    fps=Fraction(50),
                    frames=240,
                    crf=value,
                    preset=4,
                    threads=2,
                    full_range=False,
                    encoder=tools["SvtAv1EncApp"]["path"],
                    decoder=tools["dav1d"]["path"],
                    timeout=60,
                    extra=["--roi-map-file", str(old / "roi.txt")] if mode == "roi" else None,
                )
                assert r["decoded_frames"] == r["temporal_units"] == 240
                r.update(q=value, decoded_sha256=q.sha(Path(r["decoded"])))
                records[value] = r
                assert time.monotonic() - start < 300
            win = max(
                (r for r in records.values() if r["file_bytes"] <= 12000),
                key=lambda r: r["file_bytes"],
            )
            assert (
                q.sha(Path(win["decoded"])) == win["decoded_sha256"]
                and q.sha(Path(win["stream"])) == win["stream_sha256"]
            )
            eligible.append((w, h, mode, win))
            summary.append(
                dict(
                    width=w,
                    mode=mode,
                    trials={str(k): v["file_bytes"] for k, v in sorted(records.items())},
                    selected_q=win["q"],
                )
            )
    return eligible, summary


def main(args):
    args.output.mkdir()
    q.cv2.setNumThreads(1)
    q.torch.set_num_threads(2)
    q.torch.set_num_interop_threads(1)
    eligible, selections = choose(args)
    assert q.sha(q.WEIGHTS) == q.EXPECTED
    net = q.quality.load_dists(q.WEIGHTS, "cpu")
    c = motion.crops
    assert q.sha(c.LABELS) == "9918c87fa6a309052bbac8ef0be21429885e6579fd632754ed9aeb343c912d8a"
    assert q.sha(c.BUNDLE) == "5c402652ea7e2b471d870fb211f263ba1f7975afaf4b9b8a55be814d25ae45eb"
    labels = json.loads(c.LABELS.read_text())["frames"]
    assert [r["tick"] for r in labels] == c.TICKS
    arrays = {
        f"{m}{w}": (w, h, q.np.memmap(r["decoded"], q.np.uint8, "r", shape=(240, h * 3 // 2, w)))
        for w, h, m, r in eligible
    }
    with zipfile.ZipFile(c.BUNDLE) as refs, zipfile.ZipFile(args.output / "crops.zip", "x") as z:
        for item in labels:
            t = item["tick"]
            data = refs.read(f"frame-{t:03d}/source.png")
            assert motion.trial.sha(data) == item["source_png_sha256"]
            rgb = q.cv2.imdecode(q.np.frombuffer(data, q.np.uint8), q.cv2.IMREAD_COLOR)[:, :, ::-1]
            frames = {k: motion.view(arr[t], w, h) for k, (w, h, arr) in arrays.items()}
            for region, y, x, mask in c.supports(item):
                p = f"{t:03d}/{region}/"
                z.writestr(p + "source.png", c.png(rgb[y, x, ::-1]))
                z.writestr(p + "mask.png", c.png(mask))
                for k, frame in frames.items():
                    z.writestr(p + k + ".png", c.png(frame[y, x, ::-1]))
    with zipfile.ZipFile(args.output / "crops.zip") as z:
        start = time.process_time()
        q.scorer.score_tick(net, z, 10, ["source", *arrays])
        assert (time.process_time() - start) * 12 * 1.5 + 15 < 300
        rows = [r for t in c.TICKS for r in q.scorer.score_tick(net, z, t, list(arrays))]
    points = []
    for w, _, mode, r in eligible:
        quality = {
            reg: sum(s["dists"] for s in rows if s["region"] == reg and s["arm"] == f"{mode}{w}")
            / 12
            for reg in ("p1", "p2", "ball")
        }
        points.append(
            dict(
                width=w,
                mode=mode,
                crf=r["q"],
                kbps=r["file_bytes"] / 600,
                bytes=r["file_bytes"],
                stream_sha256=r["stream_sha256"],
                decoded_sha256=r["decoded_sha256"],
                players=(quality["p1"] + quality["p2"]) / 2,
                **quality,
            )
        )
    result = dict(
        points=points,
        selections=selections,
        rows=rows,
        script_sha256=q.sha(Path(__file__)),
        metric=q.quality.dists_record(q.WEIGHTS),
        label_sha256=q.sha(c.LABELS),
        source_sha256=q.sha(c.BUNDLE),
    )
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(points)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("controls", type=Path)
    cli.add_argument("output", type=Path)
    main(cli.parse_args())
