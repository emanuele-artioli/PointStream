"""Compare complete codec populations at their realized delivered rates under each common budget."""

from argparse import ArgumentParser
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_neural(data, match, q_indices):
    def native(q):
        tag = "-neuroscore" if q == 24 else f"-neuroscore-q{q}"
        file = data / (match + tag + ".json")
        if q == 24 and not file.exists():
            file = data / (match + "-neuroscore-fixed.json")
        return json.loads(file.read_text())

    receipts = [native(q) for q in q_indices]
    assert [r["qp"] for r in receipts] == q_indices
    assert (
        len(
            {
                (
                    r["source_sha256"],
                    r["labels_sha256"],
                    r["source_rgb_sha256"],
                    tuple(row["tick"] for row in r["rows"]),
                )
                for r in receipts
            }
        )
        == 1
    )
    return [
        dict(
            kbps=r["kbps"],
            bytes=r["bytes"],
            dists=r["ball_dists"],
            sha256=r["packet_sha256"],
            qp=r["qp"],
        )
        for r in receipts
    ]


def results(curves, data, low_indices, all_indices):
    policies = {}
    low = set(curves["starved_cohort"][0]["matches"])
    for name in curves["per_event"]:
        indices = sorted(set(all_indices + (low_indices if name in low else [])))
        policies[name] = read_neural(data, name, indices)
    out = {}
    for section in ("pooled", "starved_cohort"):
        points = []
        allowed_q = set(all_indices) | (set(low_indices) if section == "starved_cohort" else set())
        expected = curves[section][0]["matches"]
        for point in curves[section]:
            matches = point["matches"]
            assert matches == expected
            cap = point["budget_kbps"]
            by_family = {arm: [] for arm in ("ordinary", "candidate", "neural")}
            missing = []
            for match in matches:
                match_points = {
                    float(k): v for k, v in curves["per_event"][match]["points"].items()
                }
                for arm in ("ordinary", "candidate"):
                    by_family[arm].append(dict(match=match, **match_points[cap][arm]))
                affordable = [
                    r for r in policies[match] if r["kbps"] <= cap + 1e-9 and r["qp"] in allowed_q
                ]
                if affordable:
                    chosen = min(affordable, key=lambda r: (r["dists"], r["bytes"], r["sha256"]))
                    by_family["neural"].append(dict(match=match, **chosen))
                else:
                    missing.append(match)
            row = dict(
                available_kbps=cap, fixed_population=matches, missing_neural=missing, families={}
            )
            for arm, items in by_family.items():
                if len(items) != len(matches):
                    row["families"][arm] = dict(complete=False, count=len(items), selections=items)
                    continue
                q = np.array([r["dists"] for r in items])
                rate = np.array([r["kbps"] for r in items])
                draws = np.random.default_rng(20261122).integers(len(q), size=(10000, len(q)))
                assert np.all(rate <= cap + 1e-9)
                row["families"][arm] = dict(
                    complete=True,
                    count=len(items),
                    mean_kbps=float(rate.mean()),
                    mean_dists=float(q.mean()),
                    dists95=np.quantile(q[draws].mean(axis=1), [0.025, 0.975]).tolist(),
                    selections=items,
                )
            row["paired"] = {}
            a = np.array([v["dists"] for v in by_family["candidate"]])
            for anchor in ("ordinary", "neural"):
                if not row["families"][anchor]["complete"]:
                    continue
                b = np.array([v["dists"] for v in by_family[anchor]])
                assert [v["match"] for v in by_family[anchor]] == matches
                draws = np.random.default_rng(20261122).integers(len(a), size=(10000, len(a)))
                delta = a - b
                row["paired"][anchor] = dict(
                    delta=float(delta.mean()),
                    decrease_percent=float(100 * (1 - a.mean() / b.mean())),
                    difference95=np.quantile(delta[draws].mean(axis=1), [0.025, 0.975]).tolist(),
                    wins=int((delta < 0).sum()),
                )
            points.append(row)
        out[section] = points
    return dict(results=out, native_neural=policies)


def table(result, path):
    lines = [
        "# Three codec families under the same per-match bitrate ceiling",
        "",
        "The available bitrate constrains every match. Rate and DISTS are then averaged with equal weight per match. Every selected point is an actual native stream. The twelve-match table uses the neural indices measured on all twelve. The seven-match table can also use its registered lower-price neural anchor.",
        "",
        "| Matches | Available kbps | AV1 delivered / DISTS | PointStream delivered / DISTS | DCVC-UF delivered / DISTS | PS − DCVC DISTS, 95% interval |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for series in result["results"].values():
        for row in series:
            cells = []
            for arm in ("ordinary", "candidate", "neural"):
                r = row["families"][arm]
                cells.append(
                    f"{r['mean_kbps']:.2f} / {r['mean_dists']:.4f}"
                    if r["complete"]
                    else f"affordable on {r['count']}/{len(row['fixed_population'])}"
                )
            delta = row["paired"].get("neural")
            effect = (
                f"{delta['delta']:+.4f} ({delta['difference95'][0]:+.4f}, {delta['difference95'][1]:+.4f})"
                if delta
                else "—"
            )
            lines.append(
                f"| {len(row['fixed_population'])} | {row['available_kbps']:g} | {' | '.join(cells)} | {effect} |"
            )
    lines += [
        "",
        "The paired interval resamples whole matches. The table reports a finite tested neural ladder. The companion JSON records the selected quantizer, native byte count, and packet hash for every underlying event and budget.",
        "",
    ]
    path.write_text("\n".join(lines))


def figure(series, path):
    plt.rcParams.update(
        {"pdf.fonttype": 42, "font.size": 8, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, ax = plt.subplots(figsize=(4.6, 2.72), layout="constrained")
    delivered = []
    for key, name, color, marker, style in (
        ("ordinary", "AV1", "#b25020", "s", "--"),
        ("candidate", "PointStream", "#176882", "o", "-"),
        ("neural", "DCVC-UF", "#623c79", "^", "-."),
    ):
        x = []
        y = []
        identities = set()
        for row in series:
            v = row["families"][key]
            if not v["complete"]:
                continue
            fingerprint = tuple(r["sha256"] for r in v["selections"])
            if fingerprint in identities:
                continue
            identities.add(fingerprint)
            x.append(v["mean_kbps"])
            y.append(v["mean_dists"])
        if x:
            delivered.extend(x)
            ax.plot(x, y, color=color, marker=marker, ls=style, ms=4.5, lw=1.6, label=name)
    if delivered:
        ax.set_xscale("log")
        lo, hi = ax.get_xlim()
        ticks = [v for v in (5, 10, 15, 20, 35, 60, 100, 150, 200) if lo <= v <= hi]
        if len(ticks) >= 2:
            ax.set_xticks(ticks, [str(v) for v in ticks])
            ax.minorticks_off()
    ax.set_xlabel("Mean delivered bitrate per match (kbps)")
    ax.set_ylabel("Mean ball-crop DISTS (lower is better)")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, fontsize=7)
    ax.text(
        1,
        0.0,
        f"{len(series[0]['fixed_population'])} matches per plotted point",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
    )
    for kind in ("pdf", "png"):
        fig.savefig(path.with_suffix("." + kind), dpi=190, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    p = ArgumentParser(description=__doc__)
    p.add_argument("curves", type=Path)
    p.add_argument("data", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("figures", type=Path)
    p.add_argument("--all-q", nargs="+", type=int, required=True)
    p.add_argument("--low-q", nargs="*", type=int, default=[])
    p.add_argument("--table", type=Path)
    a = p.parse_args()
    result = results(json.loads(a.curves.read_text()), a.data, a.low_q, a.all_q)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if a.table:
        table(result, a.table)
    for section, rows in result["results"].items():
        suffix = "-starved" if section == "starved_cohort" else ""
        figure(rows, a.figures / ("official-test-three-codecs" + suffix))
