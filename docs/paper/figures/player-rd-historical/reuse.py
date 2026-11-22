"""Price the actual paid per-window components as the modeled asset lifetime varies."""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE.parents[2] / "experiments/player-rd-historical-20261122"
RATE = 33
TRAINED = {"al": 1488252, "med": 1495592, "nad": 1489876}


def measure():
    curves = json.loads((HERE / "nine-summary.json").read_text())["windows"]
    rows = []
    for window in curves:
        if not window["paired_dcvc"]:
            continue
        name = window["window"]
        original = json.loads((RUN / (name + "-window.json")).read_text())["rates"]
        points = window["pointstream"]
        assert len(original) == 3
        for index, r in enumerate(original):
            seconds = r["base_ivf_bytes"] * 8 / (1000 * r["base_kbps"])
            reference_pair = round(r["bank_worst_case_kbps"] * seconds * 1000 / 8)
            bank = reference_pair * 6
            assets = bank + TRAINED[name[:-3]]
            ordinary_bytes = r["base_ivf_bytes"] + r["kps_bytes"] + r["bootstrap_bytes"]
            running = ordinary_bytes * 8 / (1000 * seconds)
            if not math.isclose(running + assets * 8 / 7200000, points[index][0], abs_tol=1e-8):
                raise ValueError("Historical accounting does not reproduce its two-hour rate")
        scored = []
        for modeled, dists in points:
            running = modeled - assets * 8 / 7200000
            duration = assets * 8 / (1000 * (RATE - running)) if running < RATE else None
            scored.append(
                dict(
                    two_hour_kbps=modeled,
                    running_kbps=running,
                    dists=dists,
                    seconds_to_fit_33=duration,
                )
            )
        cheapest = min(p["seconds_to_fit_33"] for p in scored if p["seconds_to_fit_33"] is not None)
        rows.append(
            dict(
                window=name,
                reference_pair_bytes=reference_pair,
                modeled_12_reference_bank_bytes=bank,
                adapter_and_guider_bytes=TRAINED[name[:-3]],
                total_asset_bytes=assets,
                points=scored,
                minimum_seconds_to_fit=cheapest,
            )
        )
    assert len(rows) == 8
    return rows


def main():
    rows = measure()
    data = {
        "budget_kbps": RATE,
        "rows": rows,
        "assumption": "The fixed historical 12-crop bank and same-match trained adapter/guider are used over the stated active duration. Each running rate charges actual frame payload, motion and covers its tested operating point. The exercise assumes continuing content at that rate.",
    }
    (HERE / "reuse-33.json").write_text(json.dumps(data, indent=2) + "\n")
    colors = {"al": "#26619a", "med": "#ca661c", "nad": "#26846b"}
    labels = {"al": "Alcaraz–Hurkacz", "med": "Medvedev–Struff", "nad": "Nadal–Kyrgios"}
    fig, ax = plt.subplots(figsize=(8.3, 4.8))
    ys = np.arange(len(rows))
    for y, r in zip(ys, rows, strict=True):
        minutes = r["minimum_seconds_to_fit"] / 60
        color = colors[r["window"][:-3]]
        ax.hlines(y, 0, minutes, color=color, alpha=0.45)
        ax.scatter(minutes, y, marker="o", s=41, color=color, zorder=3)
        ax.text(minutes + 2, y, f"{minutes:.1f}", va="center", fontsize=8.5)
    ax.axvline(120, color="#444", ls=":", lw=1.2)
    ax.set(
        yticks=ys,
        yticklabels=[labels[r["window"][:-3]] + "  " + r["window"][-3:] for r in rows],
        xlabel="Active minutes using the same bank and adapter",
        xlim=(0, 130),
        xticks=[0, 15, 30, 45, 60, 90, 120],
    )
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.15)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=0.28, right=0.97, top=0.79, bottom=0.24)
    fig.suptitle(
        "When does the historical player stream fit 33 kbps?",
        x=0.04,
        y=0.97,
        ha="left",
        fontsize=15.5,
        weight="bold",
    )
    fig.text(
        0.04,
        0.88,
        "Earliest affordable measured player setting per clip. The vertical line marks the two-hour costing assumption.",
        fontsize=8.8,
    )
    fig.text(
        0.04,
        0.065,
        "The historical model prices a 12-image appearance bank and same-match adaptation at 2.40–3.69 MB.",
        fontsize=8.8,
    )
    fig.text(
        0.04,
        0.025,
        "These durations price reuse. Actual cache validity was tested separately on a two-visit broadcast.",
        fontsize=8.8,
    )
    for ext in ("pdf", "png"):
        fig.savefig(HERE / ("reuse-33." + ext), dpi=175)
    plt.close(fig)


if __name__ == "__main__":
    main()
