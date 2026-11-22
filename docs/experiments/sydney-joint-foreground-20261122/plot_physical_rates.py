"""Plot the attainable warm-cache operating points on the frozen Sydney support."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
STYLE = {
    "Ordinary AV1": ("#2669A0", "o", "-"),
    "AV1 with ROI": ("#AD7020", "s", "--"),
    "DCVC-UF": ("#8A4495", "D", "-."),
    "PointStream": ("#008566", "^", "-"),
}


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    by_family = {name: [] for name in STYLE}
    common = None
    for path in sorted(HERE.glob("automatic-*.json")):
        row = json.loads(path.read_text())
        support = (
            row["source_sha256"],
            row["review_sha256"],
            row["official_ball_sha256"],
            row["player_support"],
            row["backbone_sha256"],
            tuple(t["tick"] for t in row["paired_ticks"]),
        )
        if common is None:
            common = support
        assert common == support and len(support[-1]) == 12 and not row["qa_only"]
        assert abs(row["kbps"] - row["bytes"] * 60 / 30000) < 1e-7
        tag = path.stem.removeprefix("automatic-")
        if tag.startswith("joint-"):
            family = "PointStream"
        elif tag.startswith("dcvc-"):
            family = "DCVC-UF"
        elif tag.startswith("roi"):
            family = "AV1 with ROI"
        else:
            assert tag.startswith("ordinary-")
            family = "Ordinary AV1"
        by_family[family].append(
            {
                "receipt": path.name,
                "kbps": row["kbps"],
                "bytes": row["bytes"],
                "sha256": row["bitstream_sha256"],
                **row["mean_paired"],
            }
        )
    assert all(by_family.values())
    price_ledger = json.loads(
        (
            HERE.parent
            / "test-joint-extension-20261122/native/full-confirmation-20261123/physical-prices.json"
        ).read_text()
    )
    scenarios = {}
    for row in by_family["PointStream"]:
        paid = next(r for r in price_ledger["physical_rates"] if r["joint_sha256"] == row["sha256"])
        scenarios[row["receipt"]] = {}
        for horizon, rate in paid["modeled_kbps_by_eligible_seconds"].items():
            competitors = {}
            for name, choices in by_family.items():
                if name == "PointStream":
                    continue
                affordable = [item for item in choices if item["kbps"] <= rate]
                if affordable:
                    competitors[name] = min(
                        affordable, key=lambda item: (item["foreground"], item["bytes"])
                    )["receipt"]
            scenarios[row["receipt"]][horizon] = {
                "kbps": rate,
                "lowest_foreground_dists_by_baseline": competitors,
            }
    sheet = {}
    for endpoint, label in (("players", "Players"), ("ball", "Ball"), ("foreground", "Foreground")):
        fig, ax = plt.subplots(figsize=(6.3, 3.65), layout="constrained")
        selected = {}
        for name, records in by_family.items():
            feasible = sorted(
                (r for r in records if r["kbps"] <= 42),
                key=lambda r: (r["bytes"], r[endpoint]),
            )
            by_cost = {}
            for row in feasible:
                by_cost.setdefault(row["bytes"], row)
            frontier = []
            best = float("inf")
            for row in by_cost.values():
                if row[endpoint] < best:
                    frontier.append(row)
                    best = row[endpoint]
            assert frontier
            selected[name] = [r["receipt"] for r in frontier]
            color, marker, pattern = STYLE[name]
            prices = [r["kbps"] for r in frontier]
            fidelity = [r[endpoint] for r in frontier]
            ax.scatter(
                [r["kbps"] for r in feasible],
                [r[endpoint] for r in feasible],
                marker=marker,
                facecolors="none",
                edgecolors=color,
                linewidths=0.9,
                alpha=0.45,
                s=26,
            )
            ax.step(
                [*prices, 42],
                [*fidelity, fidelity[-1]],
                where="post",
                color=color,
                linestyle=pattern,
                linewidth=2.1,
                label=name,
            )
            ax.scatter(prices, fidelity, marker=marker, c=color, s=35, zorder=4)
        ax.set(xlim=(20, 42), xlabel="Available bitrate (kbps)", ylabel=f"{label} DISTS (lower is better)")
        ax.set_title("Sydney · Shapovalov–Safiullin", loc="left", fontsize=12)
        ax.grid(axis="y", color="#DDE1E5", linewidth=0.7)
        ax.spines[["right", "top"]].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
        for ext in ("pdf", "png"):
            fig.savefig(output / (endpoint + "." + ext), dpi=200, bbox_inches="tight")
        plt.close(fig)
        sheet[endpoint] = selected
    (output / "plot-data.json").write_text(
        json.dumps(
            {
                "case": "Sydney, one 4-second shot",
                "price": "physical complete packets, cached player appearance",
                "interpretation": "Minimum measured DISTS attainable at or below each budget. Hollow markers show all tested choices. Solid markers and steps show achievable best values below the budget. Curves use the two original automatic player masks and official visible ball on twelve fixed source ticks.",
                "by_family": by_family,
                "reuse_assumption": price_ledger["scope"],
                "by_eligible_reference_seconds": scenarios,
                "plotted_frontiers": sheet,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    main(p.parse_args().output)
