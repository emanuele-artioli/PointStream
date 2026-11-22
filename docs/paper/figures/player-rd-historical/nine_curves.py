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
    recompose = None
    if name in ("al028", "al034", "al043"):
        extra = (DATA / f"{name}-recompose-result.json").read_bytes()
        expected = {
            "al043": "783272146e069a2cd9c0a57ed1674917654dfafccd961e52285a52848e72dfd9",
            "al028": "bf981c9409039accec159fab42fdfce3675dfe285a0975db15d233008dec537c",
            "al034": "ff2dac7379f9c3855d65c38a27b2ba1a99a15f1a96475bd7e5375da2c38eba74",
        }
        assert hashlib.sha256(extra).hexdigest() == expected[name]
        recompose = json.loads(extra)
        assert len(recompose["records"]) == 80
        player.append(
            (
                recompose["candidate"]["modeled_player_kbps"],
                np.mean([x["player_dists"] for x in recompose["records"]]),
            )
        )
    ax.plot(*zip(*player), color="#16735c", marker="o", ms=4, lw=1.5)
    av1 = {}
    for policy, style in (("s2_o0", "--"), ("s4_o0", ":")):
        rows = score["analysis"]["by_policy"][policy]
        points = sorted((v["rate_kbps"], v["foreground_dists"]) for v in rows)
        if recompose is not None and policy == "s2_o0":
            points.append(
                (
                    recompose["candidate"]["base_kbps"],
                    np.mean([x["plain_dists"] for x in recompose["records"]]),
                )
            )
            points.sort()
        additional = {
            "al034": ("s2_o0", "83f9d4a7c09b2aaf816b00d15dab8a9b292bb18df446d1093620a93c141c0f80"),
            "nad001": ("s4_o0", "4019841409fe433ee84a3e0fdf4c9f28e81f6fca5f51a215bcf9eb7aeec38741"),
            "nad002": ("s4_o0", "f1b535641584c5e8be020bd0c9ba025b3da1c96d60831cce2719fcbbc926ca4e"),
            "nad000": ("s4_o0", "58a401f8b9ae41dd4f18f18f6e1fed08892d4a798afc0ebf5e1dc672e6557c09"),
            "med024": ("s4_o0", "2a64a1c38c4ff229b26104b1de885502e1e107cc66b61abb78addb1f7120ee33"),
            "med019": ("s4_o0", "c32499fcf4141f892e6b3eb7854e6dc2e70ce45acf9567b255044a3ffca06ede"),
        }
        if name in additional and policy == additional[name][0]:
            raw = (DATA / f"{name}-av1-score-result.json").read_bytes()
            assert hashlib.sha256(raw).hexdigest() == additional[name][1]
            item = json.loads(raw)
            assert len(item["ticks"]) == 80 and item["control_max_error"] < 0.001
            candidate = item["candidate"]
            points.append((candidate.get("kbps", candidate.get("base_kbps")), item["mean"]))
            points.sort()
        av1[policy] = points
        ax.plot(*zip(*points), color="#326a99", marker="x", ms=3.5, lw=1, ls=style)
    dcvc = sorted((v[1], v[5]) for v in windows[name]["dcvc_points"])
    if name == "al034":
        extra_raw = (DATA / "al034-dcvc33-result.json").read_bytes()
        assert (
            hashlib.sha256(extra_raw).hexdigest()
            == "bfd4d5c052198ead5351d674dd829ee296ac6714c4af28d8d59cf482b259d759"
        )
        extra = json.loads(extra_raw)["windows"][0]["dcvc"][0]
        assert extra["accepted"] and extra["frames"] == 240 and extra["identical_frame_hashes"]
        dcvc.append((extra["kbps"], extra["mean"]["foreground_dists"]))
        dcvc.sort()
    if name == "med019":
        extra_raw = (DATA / "med019-dcvc33-result.json").read_bytes()
        assert (
            hashlib.sha256(extra_raw).hexdigest()
            == "b1ec2aafb0c5d84d7161f6d14e2b3cdcf75ccb5add4264d131ec011693ff937d"
        )
        extra = json.loads(extra_raw)["windows"][0]["dcvc"][0]
        assert extra["accepted"] and extra["frames"] == 240 and extra["identical_frame_hashes"]
        dcvc.append((extra["kbps"], extra["mean"]["foreground_dists"]))
        dcvc.sort()
    if name == "nad000":
        extra_raw = (DATA / "nad000-dcvc33-result.json").read_bytes()
        assert (
            hashlib.sha256(extra_raw).hexdigest()
            == "47be107f690bbe44ce5e566a1972c3225f57a45ae0d122c36490a9cdf61bd942"
        )
        extra = json.loads(extra_raw)["windows"][0]["dcvc"][0]
        assert extra["accepted"] and extra["frames"] == 240 and extra["identical_frame_hashes"]
        dcvc.append((extra["kbps"], extra["mean"]["foreground_dists"]))
        dcvc.sort()
    if name in ("al028", "al043", "med024", "nad002"):
        expected = {
            "al028": "c08c8fe717d50607648025b64b1685e4b4493c6fc3d0fa4f80ddc6a36636ce4c",
            "al043": "4ae75729210ab1c830c97ebe8f25fe2267a6fb4cda4f60d38f56f57b84ed2555",
            "med024": "b2667c8353c8b1a44b12d30b8bb859a667f65d42eef6baeb41be12a6d5703d8b",
            "nad002": "fb1a854ce1fc1ebe6ed7efc03464202ea403da3bcfd42acb27a75dadb36b6839",
        }
        extra_raw = (DATA / f"{name}-dcvc33-result.json").read_bytes()
        assert hashlib.sha256(extra_raw).hexdigest() == expected[name]
        extra = json.loads(extra_raw)["windows"][0]["dcvc"][0]
        assert extra["accepted"] and extra["frames"] == 240 and extra["identical_frame_hashes"]
        dcvc.append((extra["kbps"], extra["mean"]["foreground_dists"]))
        dcvc.sort()
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
    match = {"al": "Alcaraz–Hurkacz", "med": "Medvedev–Struff", "nad": "Nadal–Kyrgios"}[name[:-3]]
    ax.set_title(f"{match} · rally {int(name[-3:])}", fontsize=10)
    ax.grid(alpha=0.18)
    ax.tick_params(labelsize=8)
    out.append(
        {"window": name, "paired_dcvc": bool(dcvc), "pointstream": player, "av1": av1, "dcvc": dcvc}
    )
assert len(intervals) == 8
common = [max(x[0] for x in intervals), min(x[1] for x in intervals)]
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
