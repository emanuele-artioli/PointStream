"""Pooled physically affordable fidelity and event-paired uncertainty on a frozen price grid."""

import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments.tennis.w3_analysis import compare, rate_dominance

HERE = Path(__file__).resolve().parent


def analysis(grid, data):
    plots = []
    per_event = {}
    curve_inputs = {}
    for m in grid["members"]:
        r = json.loads((data / (m["match"] + "-curve.json")).read_text())
        assert (
            r["source_sha256"] == m["source_sha256"]
            and r["frames"] == 240
            and r["rate_archive_sha256"] == m["rate_archive_sha256"]
        )
        assert len(r["ticks"]) == m["visible"]
        bank = r["bank"]
        curve_inputs[m["match"]] = bank
        family = [
            [(i["kbps"], -i["dists"]) for i in bank if i["arm"] == arm]
            for arm in ("candidate", "ordinary")
        ]
        bd = compare(*family)
        bd["dominance"] = rate_dominance(*family)
        per_event[m["match"]] = dict(visible=m["visible"], bd_rate=bd, points={})
    fixed_budget = []
    scopes = [(price, list(per_event), plots) for price in grid["grid_kbps"]]
    scopes.extend(
        (r["budget_kbps"], r["matches"], fixed_budget) for r in grid["fixed_budget_support"]
    )
    for budget, matches, output in scopes:
        n = len(matches)
        if n == 0:
            output.append(dict(budget_kbps=budget, count=0, matches=matches))
            continue
        draws = np.random.default_rng(20261122).integers(n, size=(10000, n))
        vals = []
        actuals = []
        for m in matches:
            chosen = {}
            for arm in ("ordinary", "candidate"):
                options = [
                    row
                    for row in curve_inputs[m]
                    if row["arm"] == arm and row["kbps"] <= budget + 1e-9
                ]
                if not options:
                    raise ValueError("unsupported operating rate")
                v = min(options, key=lambda r: (r["dists"], r["bytes"], r["sha256"]))
                chosen[arm] = {
                    k: v[k] for k in ("bytes", "kbps", "dists", "width", "crf", "sha256")
                }
            vals.append([chosen["ordinary"]["dists"], chosen["candidate"]["dists"]])
            actuals.append([chosen["ordinary"]["kbps"], chosen["candidate"]["kbps"]])
            per_event[m]["points"][str(budget)] = chosen
        x = np.array(vals)
        mean = x.mean(axis=0)
        ranges = np.array(actuals)
        bootstrap = x[draws].mean(axis=1)
        per_shot_oracle = np.min(x, axis=1)
        oracle_gain = np.min(bootstrap, axis=1) - per_shot_oracle[draws].mean(axis=1)
        output.append(
            dict(
                budget_kbps=budget,
                matches=matches,
                count=n,
                ordinary=mean[0],
                candidate=mean[1],
                delta=mean[1] - mean[0],
                per_shot_oracle=float(per_shot_oracle.mean()),
                oracle_decrease_from_best_fixed=float(mean.min() - per_shot_oracle.mean()),
                oracle_decrease95=np.quantile(oracle_gain, [0.025, 0.975]).tolist(),
                ordinary95=np.quantile(bootstrap[:, 0], [0.025, 0.975]).tolist(),
                candidate95=np.quantile(bootstrap[:, 1], [0.025, 0.975]).tolist(),
                delta95=np.quantile(bootstrap[:, 1] - bootstrap[:, 0], [0.025, 0.975]).tolist(),
                wins=int((x[:, 1] < x[:, 0]).sum()),
                actual_ordinary_range=[ranges[:, 0].min(), ranges[:, 0].max()],
                actual_candidate_range=[ranges[:, 1].min(), ranges[:, 1].max()],
            )
        )
    return dict(grid=grid, pooled=plots, fixed_budget=fixed_budget, per_event=per_event)


def write_table(result, destination):
    n_total = len(result["grid"]["members"]) + len(result["grid"]["excluded"])
    rows = [
        "# Ball fidelity under a fixed available bitrate",
        "",
        "Each family selects its lowest measured crop DISTS among its affordable encoded files at 320 and 640 pixels wide. The same original set of matches contributes every point of the common-support curve. The two lower-budget columns include only matches for which both measured banks span that budget. A paired bootstrap resamples matches. The final column describes the potential gain from choosing the lower-loss family separately on each source with knowledge of the marks.",
        "",
        "| Scope | Budget, kbps | Coverage | AV1 DISTS | PS DISTS | PS − AV1 (95% interval) | PS wins | Oracle gain vs best fixed family |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("pooled", "Common support"), ("fixed_budget", "Low-rate eligible")):
        for r in result[key]:
            count = r["count"]
            if not count:
                rows.append(f"| {label} | {r['budget_kbps']:g} | 0/{n_total} | — | — | — | — | — |")
                continue
            low, high = r["delta95"]
            rows.append(
                f"| {label} | {r['budget_kbps']:g} | {count}/{n_total} | {r['ordinary']:.4f} | {r['candidate']:.4f} | {r['delta']:+.4f} ({low:+.4f}, {high:+.4f}) | {r['wins']}/{count} | {r['oracle_decrease_from_best_fixed']:.4f} |"
            )
    rows.extend(
        [
            "",
            "## Actual selected deliveries",
            "",
            "| Available kbps | Matches | Range of selected AV1 kbps | Range of selected PointStream kbps |",
            "|---:|---|---:|---:|",
        ]
    )
    for r in result["pooled"] + result["fixed_budget"]:
        if r["count"]:
            a0, a1 = r["actual_ordinary_range"]
            b0, b1 = r["actual_candidate_range"]
            rows.append(
                f"| {r['budget_kbps']:g} | {', '.join(r['matches'])} | {a0:.2f}–{a1:.2f} | {b0:.2f}–{b1:.2f} |"
            )
    rows.extend(
        [
            "",
            "## Rate change at equal ball-crop DISTS",
            "",
            "| Original match | Common quality range, DISTS | PointStream BD-rate vs AV1 | Rate-domain comparison |",
            "|---|---:|---:|---|",
        ]
    )
    for m, r in result["per_event"].items():
        bd = r["bd_rate"]
        val = bd["bd_rate_percent"]
        lo, hi = bd["quality_overlap_negative_distance"]
        interval = f"{-hi:.4f}–{-lo:.4f}" if hi > lo else "none"
        rate = f"{val:+.2f}%" if val is not None else "undefined"
        rows.append(
            f"| {m} | {interval} | {rate} | {bd['dominance']['verdict'] or 'no common rate'} |"
        )
    rows += [
        "",
        "Per-match BD-rate uses monotone piecewise-cubic interpolation over common attainable quality. The anchor is its achievable concave envelope. A negative rate change means fewer delivered bits at equal quality. Full packet identities, selected rates and every original registered mark are in the numerical companion.",
        "",
    ]
    destination.write_text("\n".join(rows))


def plot(result, destination):
    plt.rcParams.update(
        {"font.size": 8, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False}
    )
    p = result["pooled"]
    x = [r["budget_kbps"] for r in p]
    fig, ax = plt.subplots(figsize=(4.6, 2.65), layout="constrained")
    for name, label, marker, color, ls in [
        ("ordinary", "Strongest affordable AV1", "s", "#bc5b24", "--"),
        ("candidate", "PointStream ball stream", "o", "#165976", "-"),
    ]:
        y = np.array([r[name] for r in p])
        bounds = np.array([r[name + "95"] for r in p])
        ax.fill_between(x, bounds[:, 0], bounds[:, 1], color=color, alpha=0.10, linewidth=0)
        ax.plot(x, y, marker=marker, label=label, color=color, ls=ls, ms=4, lw=1.4)
    ax.set_xscale("log")
    ax.set_xticks(x, [f"{t:g}" for t in x])
    ax.minorticks_off()
    ax.set_xlabel("Available total bitrate (kbps)")
    ax.set_ylabel("Ball-crop DISTS (lower is better)")
    ax.text(
        0.99,
        0.02,
        f"{p[0]['count']} matches at every budget",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
    )
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, fontsize=7, loc="best")
    for ext in ("pdf", "png"):
        fig.savefig(
            destination.with_suffix("." + ext), dpi=190, pad_inches=0.04, bbox_inches="tight"
        )
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", type=Path, required=True)
    p.add_argument("--data", type=Path, default=HERE / "test-physical")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--figure", type=Path, required=True)
    p.add_argument("--table", type=Path)
    a = p.parse_args()
    result = analysis(json.loads(a.grid.read_text()), a.data)
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    plot(result, a.figure)
    if a.table:
        write_table(result, a.table)
