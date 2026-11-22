"""Plot the same complete returning-shot bitstreams against body and ball quality separately."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


from experiments.tennis.k6b_side_feasibility import hull


plt.rcParams["pdf.fonttype"] = 42


def draw(a):
    bank = json.loads(a.summary.read_text())["results"]
    rates = {"SVT-AV1": "ordinary", "DCVC-UF": "dcvc", "PointStream": "fresh-joint"}
    style = {
        "SVT-AV1": ("#17617e", "o", "--"),
        "DCVC-UF": ("#954e03", "s", "-."),
        "PointStream": ("#663598", "D", "-"),
    }
    for k, label, filename in [
        ("mean_dists", "Player-region DISTS", "players"),
        ("ball_dists", "Ball-crop DISTS", "ball"),
    ]:
        fig, ax = plt.subplots(figsize=(5.6, 3.25), layout="constrained")
        plotted = {}
        for family, prefix in rates.items():
            rows = [
                r
                for name, r in bank.items()
                if name.startswith(prefix) and k in r and r["arm"]["kbps"] <= 18
            ]
            assert len(rows) >= 3 and len({r["source"] for r in rows}) == 1
            if k == "ball_dists":
                assert all(r["ball_present"] == 235 for r in rows)
            raw = [(r["arm"]["kbps"], -r[k]) for r in rows]
            frontier = hull(raw)
            color, marker, line = style[family]
            ax.scatter(
                [r["arm"]["kbps"] for r in rows],
                [r[k] for r in rows],
                facecolors="none",
                edgecolors=color,
                marker=marker,
                s=13,
                linewidths=0.75,
                alpha=0.58,
            )
            ax.plot(
                [x for x, _ in frontier],
                [-y for _, y in frontier],
                color=color,
                marker=marker,
                ls=line,
                lw=1.65,
                markersize=4.4,
                label="PointStream (warm)" if family == "PointStream" else family,
            )
            plotted[family + " measured"] = [
                dict(kbps=r["arm"]["kbps"], dists=r[k], arm=r["arm"]["name"]) for r in rows
            ]
            plotted[family] = [
                {
                    "kbps": x,
                    "dists": -y,
                    "arm": next(
                        r["arm"]["name"] for r in rows if r["arm"]["kbps"] == x and -r[k] == y
                    ),
                }
                for x, y in frontier
            ]
        prior = bank.get("prior-joint-q50")
        if prior is not None:
            x, y = prior["arm"]["kbps"], prior[k]
            ax.scatter(
                x,
                y,
                zorder=5,
                color="#181818",
                marker="*",
                s=80,
                label="PointStream, observed cache hit",
            )
            plotted["observed_cache_hit"] = {"kbps": x, "dists": y}
        ax.set_xlabel("Delivered video bitrate on the returning shot (kbps)")
        ax.set_ylabel(label + " (lower is better)")
        ax.set_xlim(7, 18)
        ax.set_xticks([8, 10, 12, 14, 16, 18])
        ax.grid(axis="y", lw=0.55, color="#cfd4d7")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="best", frameon=False, fontsize=8.4)
        for ext in ("pdf", "png"):
            fig.savefig(a.out / f"{a.prefix}-{filename}.{ext}", dpi=190)
        plt.close(fig)
        (a.out / f"{a.prefix}-{filename}.json").write_text(json.dumps(plotted, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--prefix", required=True)
    draw(p.parse_args())
