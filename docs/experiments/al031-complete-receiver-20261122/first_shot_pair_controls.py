"""Supply the two omitted conventional price comparators on the first of the returning shots."""

import argparse
import hashlib
from importlib import import_module
import json
from pathlib import Path
from statistics import mean
import subprocess
import sys

import cv2
import numpy as np
import torch

from src.codecs import quality, svtav1
from src.pointstream.tracked_patch import yuv_to_rgb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "med-mixed-receiver-20261122"))
cold = import_module("cold")
cpu, sha, streamed_trial = cold.cpu, cold.sha, cold.streamed_trial

sys.path.insert(0, str(HERE.parent / "al028-neural-display-20261122"))
bodies = import_module("score_bodies")
EXPECTED, REVIEW, WEIGHTS, tensor = bodies.EXPECTED, bodies.REVIEW, bodies.WEIGHTS, bodies.tensor


def read(p):
    return json.loads(p.read_text())


def measure(row, net, meta):
    digest = hashlib.sha256()
    stages = {x["frame_index"]: x for x in meta["frames"]}
    values = []
    with subprocess.Popen(row["decode_command"], stdout=subprocess.PIPE) as proc:
        assert proc.stdout is not None
        for t in range(240):
            b = proc.stdout.read(640 * 384 * 3 // 2)
            assert len(b) == 640 * 384 * 3 // 2
            digest.update(b)
            if t not in stages:
                continue
            frame = stages[t]
            assert sha(REVIEW / frame["source"]) == frame["source_sha256"]
            src = cv2.cvtColor(cv2.imread(str(REVIEW / frame["source"])), cv2.COLOR_BGR2RGB)
            rgb = yuv_to_rgb(np.frombuffer(b, np.uint8).reshape(576, 640), 640, 384, False)
            disp = cv2.resize(rgb, (1920, 1080), interpolation=cv2.INTER_CUBIC)
            for person in frame["players"]:
                f = REVIEW / person["proposal"]
                assert sha(f) == person["sha256"]
                mask = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE) > 0
                y, x = np.nonzero(mask)
                assert len(x) == person["pixels"]
                roi = (
                    slice(max(0, min(y) - 32), min(1080, max(y) + 33)),
                    slice(max(0, min(x) - 32), min(1920, max(x) + 33)),
                )
                a, bp, m = (
                    tensor(src[roi]),
                    tensor(disp[roi]),
                    torch.from_numpy(mask[roi].copy())[None, None],
                )
                val = quality.masked_dists(net, bp, a, m).item()
                assert np.isfinite(val)
                values.append(dict(tick=t, person=person["id"], dists=val))
        assert proc.stdout.read(1) == b"" and proc.wait(timeout=120) == 0
    assert len(values) == 24 and digest.hexdigest() == row["decoded_sha256"]
    return dict(**row, rows=values, player_dists=mean(x["dists"] for x in values))


def main(a):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    cv2.setNumThreads(1)
    fixed = read(HERE / "two-shot-price.json")
    returns = read(HERE / "return-fixed12-ordinary-arms.json")
    previous = read(HERE.parent / "al028-neural-display-20261122/ordinary-three-size-curves.json")
    early = read(HERE.parent / "al028-neural-display-20261122/player-controls.json")
    reverse = read(HERE / "ordinary-fresh-cold.json")["arm"]["bytes"]
    budgets = {}
    for k in ["reuse", "send_new"]:
        total = fixed[k]["total_bytes"]
        second = (
            reverse
            if k == "send_new"
            else max(r["bytes"] for r in returns if r["bytes"] <= total / 2)
        )
        budgets[k] = dict(total=total, second=second, first=total - second)
    tools = {k: svtav1.tool(k) for k in ("SvtAv1EncApp", "dav1d")}
    assert all(
        tools[k]["sha256"] == read(a.ladder / "result.json")["tools"][k]["sha256"] for k in tools
    )
    src = a.ladder / "640.yuv"
    source_hash = sha(src)
    assert source_hash == next(
        v["input_sha256"] for v in previous["selection_controls"] if v["width"] == 640
    )
    meta = read(REVIEW / "review.json")
    assert meta["review_sha256"] == early["review_sha256"] and sha(WEIGHTS) == EXPECTED
    a.out.mkdir(exist_ok=False)
    start = cpu()
    bins = {}

    def trial(q, n):
        return streamed_trial(src, a.out, tools, 25, q, n, 640, 384)

    trial(40 * 4, 24)
    smoke = cpu() - start
    assert 1.5 * smoke * 10 * 15 + 40 < 1200
    for q in [28 * 4, 40 * 4, 52 * 4]:
        bins[q] = trial(q, 240)
    avg = (cpu() - start - smoke) / 3
    assert cpu() - start + 1.5 * avg * 12 + 40 < 1200
    net = quality.load_dists(WEIGHTS, "cpu")
    calibration = measure(bins[160], net, meta)
    known = next(
        r for r in previous["points"] if r["stream_sha256"] == calibration["stream_sha256"]
    )
    assert all(
        abs(x["dists"] - y["dists"]) < 1e-6
        for x, y in zip(known["rows"], calibration["rows"], strict=True)
    )
    measured = []
    for key, budget in budgets.items():
        target = budget["first"]
        lo = max(q for q, r in bins.items() if r["bytes"] >= target)
        hi = min(q for q, r in bins.items() if r["bytes"] <= target)
        while hi - lo > 1:
            q = (hi + lo) // 2
            if q not in bins:
                bins[q] = trial(q, 240)
            if bins[q]["bytes"] <= target:
                hi = q
            else:
                lo = q
            assert cpu() - start < 1200
        chosen = max((x for x in bins.values() if x["bytes"] <= target), key=lambda r: r["bytes"])
        measured.append(dict(selection=key, budget=budget, **measure(chosen, net, meta)))
    assert cpu() - start < 1200
    result = dict(
        points=measured,
        cpu_seconds=cpu() - start,
        smoke_cpu_seconds=smoke,
        smoke_forecast_seconds=1.5 * smoke * 10 * 15 + 40,
        source_yuv_sha256=source_hash,
        calibration_sha256=calibration["stream_sha256"],
        backbone=EXPECTED,
        supports=meta["review_sha256"],
        tools=tools,
        grid=list(bins.values()),
    )
    a.result.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for key in ["ladder", "out", "result"]:
        parser.add_argument("--" + key, required=True, type=Path)
    main(parser.parse_args())
