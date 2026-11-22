"""Combine compatible source-supported foreground measurements with equal rally weights."""

import json
from pathlib import Path
from statistics import mean

from src.pointstream.evaluation import anchor_sensitivity

HERE = Path(__file__).resolve().parent
FIGURES = HERE.parents[1] / "paper/figures"
MODES = ("PointStream", "AV1", "DCVC-UF")


def read(path):
    return json.loads(path.read_text())


def best(points, cap):
    return {
        mode: min(
            (
                r
                for r in points
                if r["mode"] == mode and r["kbps"] <= cap and not r.get("archived_only", False)
            ),
            key=lambda r: (r["foreground"], r["kbps"]),
        )
        for mode in MODES
    }


def brief(point):
    return {
        k: v
        for k, v in point.items()
        if k
        in {
            "mode",
            "kbps",
            "source",
            "bytes",
            "stream_sha256",
            "setting",
            "crf",
            "width",
            "height",
            "paired_player",
            "paired_ball",
            "body_all_24",
            "foreground",
            "archived_only",
        }
    }


def main():
    al = [
        r
        for name in ("00", "10", "20", "30", "40")
        for r in read(HERE / f"ordinary-{name}.json")["points"]
    ]
    assert len(al) == len({r["stream_sha256"] for r in al}) == 52
    ordinary = list(al)
    al.extend(read(HERE / "joint-neural-native.json")["points"])
    frozen = read(HERE / "matched-three.json")
    assert len(frozen["jointly_scorable_ticks"]) == 11
    first = []
    for r in read(FIGURES / "starved-choice/points.json"):
        mode = {
            "Joint PointStream · valid-cache scenario": "PointStream",
            "Ordinary AV1": "AV1",
            "AV1 with ROI": "AV1",
            "DCVC-UF": "DCVC-UF",
        }.get(r["mode"])
        if mode:
            first.append(
                dict(
                    mode=mode,
                    kbps=r["kbps"],
                    paired_player=r["players"],
                    paired_ball=r["ball"],
                    foreground=(r["players"] + r["ball"]) / 2,
                    source=r["source"],
                )
            )
    first_budget = max(21.545, 21.623333333333335, 20.45)
    second_budget = frozen["actual_kbps"]["neural"]
    points = {
        "Nadal–Kyrgios": best(first, first_budget),
        "Alcaraz–Hurkacz": best(al, second_budget),
    }
    assert (
        points["Alcaraz–Hurkacz"]["PointStream"]["foreground"]
        == frozen["arms"]["joint"]["paired_foreground_dists"]
    )
    pooled = {
        k: {
            name: mean(r[k][name] for r in points.values())
            for name in ("paired_player", "paired_ball", "foreground")
        }
        for k in MODES
    }
    gains = {
        k: 100 * (1 - pooled["PointStream"]["foreground"] / pooled[k]["foreground"])
        for k in MODES[1:]
    }
    gold = [
        (r["kbps"], -r["foreground"])
        for r in al
        if r["mode"] in MODES[1:] and not r.get("archived_only")
    ]
    warm = [(r["kbps"], -r["foreground"]) for r in al if r["mode"] == "PointStream"]
    init = read(HERE.parent / "al028-neural-display-20261122/common-budget.json")["cache_bytes"]
    audit = [r for a in ordinary for r in a.get("cross_host_control", [])]
    support = [[r["kbps"], r["foreground"]] for r in ordinary]
    summary = dict(
        registered_aggregation="Both bodies averaged per instant, then player and ball classes weighted equally on jointly visible ticks; each rally weighted equally.",
        per_shot={name: {k: brief(r) for k, r in row.items()} for name, row in points.items()},
        caps_kbps={"Nadal–Kyrgios": first_budget, "Alcaraz–Hurkacz": second_budget},
        source_ticks={"Nadal–Kyrgios": 12, "Alcaraz–Hurkacz": 12},
        visible_joint_ticks={"Nadal–Kyrgios": 12, "Alcaraz–Hurkacz": 11},
        pooled=pooled,
        pooled_foreground_relative_decrease_percent=gains,
        upper_alcaraz={k: brief(r) for k, r in best(al, 39.770833333333336).items()},
        limited_rate_foreground_bd_vs_envelope={
            k: anchor_sensitivity(curve, gold, minimum_quality_span=0.005)
            for k, curve in [("cached", warm), ("cold", [(r + init / 1200, q) for r, q in warm])]
        },
        duplicate_320_upscaler={
            "max_crop_channel_difference": max(r["max_channel_difference"] for r in audit),
            "max_crop_dists_difference": max(
                abs(r["linux_dists"] - r["native_dists"]) for r in audit
            ),
        },
        alcaraz_points={mode: [brief(r) for r in al if r["mode"] == mode] for mode in MODES},
    )
    assert len(support) == 52 and len(audit) == 11
    out = HERE / "figure-data.json"
    with out.open("x") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
