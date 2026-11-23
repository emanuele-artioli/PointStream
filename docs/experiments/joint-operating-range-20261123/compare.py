"""Compare the physically attainable tennis ladders on each rally's fixed supports."""

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.pointstream.evaluation import anchor_sensitivity


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
SYDNEY = EXPERIMENTS / "sydney-joint-foreground-20261122"
SHANGHAI = EXPERIMENTS / "shanghai-joint-low-anchors-20261122"
STYLE = {
    "pointstream": ("PointStream", "#007757", "^", "-"),
    "ordinary": ("AV1", "#2669A0", "o", "--"),
    "roi": ("AV1 ROI", "#96601D", "s", ":"),
    "neural": ("DCVC-UF", "#8A4495", "D", "-."),
}
ENDPOINTS = ("foreground", "players", "ball")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def add_rungs(bank, path, fps):
    """Bind actual complete native replay scores to the same sampled source."""
    if path is None:
        return []
    metrics = load(path)
    assert metrics["source_sha256"] == bank["source"]
    assert metrics["sampled_ticks"] == bank["ticks"]
    assert metrics["review_sha256"] and metrics["official_ball_sha256"]
    if fps == 60:
        original = load(SYDNEY / "automatic-joint-70.json")
    else:
        original = load(EXPERIMENTS / "shanghai-joint-review-20261122/automatic-joint-70.json")
    assert metrics["review_sha256"] == original["review_sha256"]
    assert metrics["official_ball_sha256"] == original["official_ball_sha256"]
    assert metrics["dists_backbone_sha256"] == original["backbone_sha256"]
    rows = []
    for arm in metrics["arms"]:
        kbps = arm["warm_kbps"]
        nbytes = round(kbps * 240 * 1000 / (8 * float(fps)))
        assert math.isclose(kbps, nbytes * float(fps) * 8 / 240000, abs_tol=1e-7)
        rows.append(
            {
                "arm": "PointStream " + "x".join(map(str, arm["coded_size"])),
                "bytes": nbytes,
                "kbps": kbps,
                "cold_kbps": arm["cold_kbps"],
                "sha256": arm["joint_sha256"],
                "receipt_sha256": digest(path),
                "scores": {
                    endpoint: arm["means"][region]["dists"]
                    for endpoint, region in (("foreground", "foreground"), ("players", "player"), ("ball", "ball"))
                },
            }
        )
    return rows


def add_controls(bank, inputs):
    """Require original support and exact bitstream identities for extra native controls."""
    result = {k: list(rows) for k, rows in bank["controls"].items()}
    original = load(EXPERIMENTS / "shanghai-joint-review-20261122/automatic-joint-70.json")
    for family, path in inputs:
        score = load(path)
        assert score["source_sha256"] == bank["source"]
        assert score["review_sha256"] == original["review_sha256"]
        assert score["official_ball_sha256"] == original["official_ball_sha256"]
        assert score["backbone_sha256"] == original["backbone_sha256"]
        assert score["player_support"] == bank["player_support"]
        assert [item["tick"] for item in score["paired_ticks"]] == bank["ticks"]
        assert not score["qa_only"]
        result[family].append(
            {
                "arm": score["arm"],
                "bytes": score["bytes"],
                "kbps": score["kbps"],
                "sha256": score["bitstream_sha256"],
                "receipt_sha256": digest(path),
                "scores": score["mean_paired"],
            }
        )
    return result


def frontier(rows, endpoint):
    selected = []
    for row in sorted(rows, key=lambda r: (r["bytes"], r["scores"][endpoint])):
        if not selected or row["scores"][endpoint] < selected[-1]["scores"][endpoint]:
            selected.append(row)
    return selected


def summarize(bank_path, fps, wider=None, controls=()):
    bank = load(bank_path)
    pointstream = [
        dict(rung["joint"], cold_kbps=rung["cases"]["cold"]["kbps"])
        for rung in bank["measured_rungs"].values()
    ]
    pointstream.extend(add_rungs(bank, wider, fps))
    baselines = add_controls(bank, controls) if controls else bank["controls"]
    groups = dict(baselines, pointstream=pointstream)
    by_packet = {}
    for family, rows in groups.items():
        by_packet[family] = {}
        for row in rows:
            assert math.isclose(row["kbps"], row["bytes"] * float(fps) * 8 / 240000, abs_tol=1e-7)
            assert all(math.isfinite(row["scores"][k]) for k in ENDPOINTS) and row["bytes"] > 0
            assert math.isclose(
                row["scores"]["foreground"], (row["scores"]["players"] + row["scores"]["ball"]) / 2
            )
            old = by_packet[family].setdefault(row["sha256"], row)
            assert old["bytes"] == row["bytes"]
            assert all(abs(old["scores"][k] - row["scores"][k]) < 1e-7 for k in ENDPOINTS)
    frontiers = {
        endpoint: {family: frontier(rows.values(), endpoint) for family, rows in by_packet.items()}
        for endpoint in ENDPOINTS
    }
    return {
        "source_sha256": bank["source"],
        "baseline_receipt_sha256": digest(bank_path),
        "wider_receipt_sha256": digest(wider) if wider is not None else None,
        "source_ticks": 240,
        "scored_ticks": bank["ticks"],
        "frontiers": frontiers,
        "bd_rate": {
            endpoint: {
                family: anchor_sensitivity(
                    [(r["kbps"], -r["scores"][endpoint]) for r in families["pointstream"]],
                    [(r["kbps"], -r["scores"][endpoint]) for r in families[family]],
                )
                for family in baselines
            }
            for endpoint, families in frontiers.items()
        },
        "measured_changes": {
            endpoint: order_changes(families, endpoint) for endpoint, families in frontiers.items()
        },
    }


def order_changes(families, endpoint):
    changes = []
    bank = sorted({r["bytes"] for rows in families.values() for r in rows})
    previous = None
    previous_family = None
    for limit in bank:
        affordable = {
            family: min(
                (r for r in rows if r["bytes"] <= limit),
                key=lambda r: r["scores"][endpoint],
                default=None,
            )
            for family, rows in families.items()
        }
        if not affordable["pointstream"] or all(
            row is None for family, row in affordable.items() if family != "pointstream"
        ):
            continue
        family, winner = min(
            ((family, row) for family, row in affordable.items() if row is not None),
            key=lambda pair: (pair[1]["scores"][endpoint], pair[1]["bytes"]),
        )
        current = {
            "ceiling_bytes": limit,
            "winner": family,
            "winning_rate": winner["kbps"],
            "winning_dists": winner["scores"][endpoint],
        }
        if family != previous_family:
            changes.append({"from": previous, "to": current})
        previous_family, previous = family, current
    return changes


def plot(result, output, endpoint):
    figure, axes = plt.subplots(1, 2, sharey=True, figsize=(7.3, 2.88))
    for ax, (match, data) in zip(axes, result.items(), strict=True):
        ax.set_title(match, fontweight="semibold", fontsize=9.4)
        ax.set_xscale("log")
        ax.set_xlim(10, 235)
        ax.set_xticks([10, 20, 40, 80, 160], [10, 20, 40, 80, 160])
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.grid(axis="y", alpha=0.24)
        ax.spines[["right", "top"]].set_visible(False)
        for family, (label, color, marker, line) in STYLE.items():
            knots = [
                r for r in data["frontiers"][endpoint][family] if 10 <= r["kbps"] <= 235
            ]
            if not knots:
                continue
            x, y = [r["kbps"] for r in knots], [r["scores"][endpoint] for r in knots]
            ax.step(x, y, where="post", color=color, linestyle=line, lw=1.15)
            ax.plot(x, y, linestyle="none", color=color, marker=marker, ms=3.7, label=label)
    endpoint_label = {"players": "Player", "ball": "Ball", "foreground": "Foreground"}[endpoint]
    axes[0].set_ylabel(endpoint_label + " DISTS ↓", fontsize=9)
    figure.supxlabel("Delivered bitrate (kbps, appearance cached)", fontsize=9, y=0.105)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
                  ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.007))
    figure.subplots_adjust(bottom=0.26, left=0.09, right=0.99, top=0.87, wspace=0.10)
    figure.savefig(output, metadata={"CreationDate": None})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--shanghai-wider", type=Path)
    for family in ("ordinary", "neural", "roi"):
        parser.add_argument("--" + family, type=Path, nargs="*", default=[])
    args = parser.parse_args()
    result = {
        "Sydney · Safiullin–Shapovalov": summarize(
            SYDNEY / "comparison-all-physical.json", Fraction(60), SYDNEY / "wider-native-metrics.json"
        ),
        "Shanghai · Sinner–Shelton": summarize(
            SHANGHAI / "comparison-final-neural.json",
            Fraction(30000, 1001),
            args.shanghai_wider,
            [(family, path) for family in ("ordinary", "roi", "neural") for path in getattr(args, family)],
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for endpoint in ENDPOINTS:
        plot(result, args.output / (endpoint + ".pdf"), endpoint)
    (args.output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
