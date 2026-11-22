"""Plot achieved mean prices and qualities of physically encoded same-match shot choices."""

import argparse
from itertools import chain
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from replicate import controls, load


ROOT = Path(__file__).resolve().parent
SHOTS = ("000", "019", "024")
MODES = ("PointStream", "SVT-AV1", "DCVC-UF")


def plot(outer=False):
    suffix = "-expanded-neural" if outer else ""
    candidates = {}
    for shot in SHOTS:
        one = controls(shot, outer=outer)
        candidates[shot] = {
            "PointStream": [load(f"{shot}-joint-q{q}.json") for q in (70, 60, 50)],
            "SVT-AV1": list(
                chain.from_iterable(v for k, v in one.items() if k.startswith("ordinary"))
            ),
            "DCVC-UF": list(
                chain.from_iterable(v for k, v in one.items() if k.startswith("neural"))
            ),
        }
    data = {}
    for mode in MODES:
        prices = sorted(
            {
                r["arm"]["bytes"]
                for s in SHOTS
                for r in candidates[s][mode]
                if r["arm"]["kbps"] <= 32
            }
        )
        points = []
        for price in prices:
            eligible = [
                [r for r in candidates[s][mode] if r["arm"]["bytes"] <= price] for s in SHOTS
            ]
            if not all(eligible):
                continue
            chosen = [
                min(one, key=lambda r: (r["mean_dists"], r["arm"]["bytes"])) for one in eligible
            ]
            arms = [
                dict(shot=s, arm=r["arm"]["name"], bytes=r["arm"]["bytes"], dists=r["mean_dists"])
                for s, r in zip(SHOTS, chosen, strict=True)
            ]
            if points and arms == points[-1]["shots"]:
                continue
            points.append(
                dict(
                    per_shot_cap_bytes=price,
                    achieved_mean_kbps=mean(r["arm"]["kbps"] for r in chosen),
                    mean_dists=mean(r["mean_dists"] for r in chosen),
                    shots=arms,
                )
            )
        data[mode] = points
    assert all(len(points) >= 2 for points in data.values())
    (ROOT / f"three-shots-rate-plot-data{suffix}.json").write_text(
        json.dumps(data, indent=2) + "\n"
    )
    plt.rcParams.update(
        {"pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10}
    )
    fig, ax = plt.subplots(figsize=(6.4, 3.8), layout="constrained")
    for name, color, marker, style in [
        ("SVT-AV1", "#235c82", "o", "--"),
        ("DCVC-UF", "#753969", "s", "-."),
        ("PointStream", "#ab4317", "D", "-"),
    ]:
        pts = data[name]
        ax.plot(
            [p["achieved_mean_kbps"] for p in pts],
            [p["mean_dists"] for p in pts],
            color=color,
            linestyle=style,
            marker=marker,
            markersize=4.5,
            lw=1.6,
            label=name,
        )
    ax.set_xlabel("Mean delivered bitrate across three shots (kbps)")
    ax.set_ylabel("Mean player DISTS (lower is better)")
    ax.grid(axis="y", color="#c9cdd1", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / f"three-shots-player-rates{suffix}.{ext}", dpi=190)
    plt.close(fig)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--outer", action="store_true")
    plot(cli.parse_args().outer)
