"""Nine authenticated W7 windows: real paired samples, no extrapolated curves."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "experiments/player-rd-historical-20261122"
W9 = ROOT / "experiments/w9-inputs/w7-repaired-2h.json"
W9_HASH = "5245d83df6f8ee16ade28174f6f3c46a94b42ae4d32520c2165fa602e83e9d86"
REFS = {
    "al028": 155683,
    "al034": 164079,
    "al043": 152664,
    "med000": 201292,
    "med019": 195067,
    "med024": 189263,
    "nad000": 366161,
    "nad001": 262185,
    "nad002": 319207,
}
ORDER = ["al028", "al034", "al043", "med019", "med024", "med000", "nad000", "nad001", "nad002"]
HASHES = json.loads((DATA / "numeric-hashes.json").read_text())


def read(name):
    data = (DATA / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != HASHES[name]:
        raise ValueError("numeric receipt identity: " + name)
    return data


raw = W9.read_bytes()
assert hashlib.sha256(raw).hexdigest() == W9_HASH
prior = json.loads(raw)
windows = {r["window"]: r for r in prior["tier1"]["windows"] + prior["tier2"]["windows"]}
fig, axes = plt.subplots(3, 3, figsize=(10.5, 7.3), constrained_layout=True, sharey=True)
out = []
intervals = []
for ax, name in zip(axes.flat, ORDER):
    window = json.loads(read(name + "-window.json"))
    score = json.loads(read(name + "-svt.json"))
    read(name + "-scores.npz")
    with np.load(DATA / (name + "-scores.npz"), allow_pickle=False) as arrays:
        assert np.array_equal(arrays["ticks"], np.arange(0, 240, 3))
        assert np.allclose(
            arrays["foreground_dists"].mean(1), window["means"]["foreground_dists"], atol=1e-12
        )
    assert score["ticks"] == window["ticks"] == list(range(0, 240, 3))
    assert [r["scale"] for r in window["rates"]] == [8, 4, 2]
    correction = 4.5 * REFS[name] * 8 / 7_200_000
    player = [
        (r["total_kbps"] + correction, window["means"]["foreground_dists"][2 * i + 1])
        for i, r in enumerate(window["rates"])
    ]
    assert abs(player[1][0] - windows[name]["rate_kbps"]) < 1e-9
    ax.plot(*zip(*player), color="#16735c", marker="o", ms=4, lw=1.5)
    av1 = {}
    for policy, style in (("s2_o0", "--"), ("s4_o0", ":")):
        rows = score["analysis"]["by_policy"][policy]
        points = sorted((v["rate_kbps"], v["foreground_dists"]) for v in rows)
        av1[policy] = points
        ax.plot(*zip(*points), color="#326a99", marker="x", ms=3.5, lw=1, ls=style)
    dcvc = sorted((v[1], v[5]) for v in windows[name]["dcvc_points"])
    if dcvc:
        ax.plot(*zip(*dcvc), color="#bd542e", marker="s", ms=3.5, lw=1.1)
        av = av1["s2_o0"] + av1["s4_o0"]
        intervals.append(
            (
                max(min(x for x, _ in player), min(x for x, _ in dcvc), min(x for x, _ in av)),
                min(max(x for x, _ in player), max(x for x, _ in dcvc), max(x for x, _ in av)),
            )
        )
    else:
        ax.text(
            0.98, 0.96, "DCVC unavailable", transform=ax.transAxes, ha="right", va="top", fontsize=8
        )
    ax.set_title(name + ("  ·  unpaired" if not dcvc else ""), fontsize=10)
    ax.grid(alpha=0.18)
    ax.tick_params(labelsize=8)
    out.append(
        {"window": name, "paired_dcvc": bool(dcvc), "pointstream": player, "av1": av1, "dcvc": dcvc}
    )
assert len(intervals) == 8
common = [max(x[0] for x in intervals), min(x[1] for x in intervals)]
assert common[0] > common[1]
fig.supxlabel("Rate (kbps); PointStream asset cost assumes two-hour reuse", fontsize=10)
fig.supylabel("Player-region DISTS (lower is better)", fontsize=10)
fig.legend(
    handles=[
        Line2D([], [], color="#16735c", marker="o", label="PointStream"),
        Line2D([], [], color="#326a99", marker="x", ls="--", label="AV1 half scale"),
        Line2D([], [], color="#326a99", marker="x", ls=":", label="AV1 quarter scale"),
        Line2D([], [], color="#bd542e", marker="s", label="DCVC-UF"),
    ],
    loc="outside upper center",
    ncol=4,
    frameon=False,
    fontsize=9,
)
fig.savefig(HERE / "player-rd-nine.pdf")
fig.savefig(HERE / "player-rd-nine.png", dpi=180)
(HERE / "nine-summary.json").write_text(
    json.dumps(
        {
            "windows": out,
            "paired_common_lower_kbps": common[0],
            "paired_common_upper_kbps": common[1],
        },
        indent=2,
    )
    + "\n"
)
