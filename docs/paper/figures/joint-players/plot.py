"""Body fidelity from two complete native-rate two-player-and-ball receivers."""

import json
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[2] / "experiments"


def load(path):
    return json.loads(path.read_text())


def main():
    first = EXP / "joint-trajectory-20261122"
    second = EXP / "al028-neural-display-20261122"
    joint = next(r for r in load(first / "scores.json")["points"] if r["cap"] == 20)
    ordinary = next(
        r
        for r in load(first / "control-scores.json")["points"]
        if r["mode"] == "plain" and abs(r["kbps"] - 21.623333333333335) < 1e-6
    )
    neural = load(first / "dcvc-q6-human-regions.json")
    second_scores = load(second / "player-joint.json")
    controls = load(second / "player-controls.json")
    assert second_scores["reconstruction_sha256"] and len(second_scores["rows"]) == 72
    # The private Linux displays and independently reconstructed workstation RGB
    # have slightly different color conversion. Both use the same source ticks.
    def key(r):
        return (r["tick"], r["person"], r["mask_pixels"])

    for arm in ("ordinary", "neural"):
        assert [key(r) for r in controls["rows"] if r["arm"] == arm] == [
            key(r) for r in second_scores["rows"] if r["arm"] == arm
        ]
    ordinary_grid = load(second / "ordinary-three-size-curves.json")["points"]
    cap = second_scores["native_rates"]["neural"]
    best_ordinary = min(
        (r for r in ordinary_grid if r["kbps"] <= cap), key=lambda r: r["player_dists"]
    )
    modes = [
        ("joint", "Joint PointStream", "#00788a", "o"),
        ("ordinary", "Ordinary AV1", "#b25110", "s"),
        ("neural", "DCVC-UF", "#654594", "D"),
    ]
    rows = [
        {
            "match": "Nadal–Kyrgios",
            "source": "12 human-reviewed instants",
            "fps": 50,
            "cold_kbps": joint["cold_kbps"],
            "modes": {
                arm: {"rate": r["kbps"], "dists": r["players"]}
                for arm, r in zip(
                    ("joint", "ordinary", "neural"), (joint, ordinary, neural), strict=True
                )
            },
        },
        {
            "match": "Alcaraz–Hurkacz",
            "source": "12 source-model mask instants",
            "fps": 25,
            "cold_kbps": second_scores["native_rates"]["joint"] + 157517 / 1200,
            "modes": {
                arm: {
                    "rate": second_scores["native_rates"][arm],
                    "dists": mean(r["dists"] for r in second_scores["rows"] if r["arm"] == arm),
                }
                for arm, *_ in modes
            },
        },
    ]
    rows[1]["modes"]["ordinary"] = {
        "rate": best_ordinary["kbps"],
        "dists": best_ordinary["player_dists"],
    }
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    for i, r in enumerate(rows):
        center = 1 - i
        ax.text(
            -0.025,
            center + 0.02,
            r["match"],
            ha="right",
            va="center",
            transform=ax.get_yaxis_transform(),
            fontweight="bold",
            fontsize=10,
        )
        ax.text(
            -0.025,
            center - 0.19,
            r["source"],
            ha="right",
            va="center",
            transform=ax.get_yaxis_transform(),
            fontsize=8,
            color="#42525b",
        )
        for j, (key, name, color, marker) in enumerate(modes):
            d = r["modes"][key]
            y = center + 0.25 - j * 0.22
            ax.scatter(
                d["dists"], y, marker=marker, color=color, s=48, label=name if not i else None
            )
            ax.annotate(
                f"{d['dists']:.3f}   ({d['rate']:.2f})",
                (d["dists"], y),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=color,
            )
    points = [p["dists"] for r in rows for p in r["modes"].values()]
    ax.set(
        xlim=(min(points) - 0.03, max(points) + 0.105),
        ylim=(-0.42, 1.52),
        yticks=[],
        xlabel="Player-region DISTS   ·   lower is better",
    )
    ax.grid(axis="x", alpha=0.18)
    ax.spines[["top", "left", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.275, right=0.98, bottom=0.26, top=0.79)
    fig.text(0.035, 0.929, "Player fidelity at about 21 kbps", fontsize=16, fontweight="bold")
    fig.text(
        0.035,
        0.86,
        "Both players and the ball are decoded. Parentheses give actual kbps with valid cached appearances.",
        fontsize=9,
    )
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=3, bbox_to_anchor=(1.0, 1.13))
    fig.text(
        0.035,
        0.103,
        "Each row uses 24 common body regions. The second row includes the best affordable AV1 resolution.",
        fontsize=8.7,
    )
    fig.text(
        0.035,
        0.038,
        f"Charging both references to a single rally raises PointStream to {rows[0]['cold_kbps']:.2f} and {rows[1]['cold_kbps']:.2f} kbps, respectively.",
        fontsize=8.7,
    )
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"joint-players-two-matches.{ext}", dpi=175)
    plt.close(fig)
    (HERE / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
