"""Summarize complete native mixed timelines, keeping the three shots as dependent repeats."""

import argparse
import json
import math
from pathlib import Path
from statistics import mean

from rate_summary import summarize
from scipy.interpolate import PchipInterpolator

from experiments.tennis.k6b_side_feasibility import hull

ROOT = Path(__file__).resolve().parent


def load(name):
    return json.loads((ROOT / name).read_text())


def compact(r):
    return {k: r[k] for k in ["arm", "mean_dists", "active_dists", "fallback_dists", "rgb_psnr"]}


def controls(c, *, outer=False):
    if c == "000":
        anchors = load("000-control-scores.json")
    else:
        anchors = {"ordinary_grid": load(f"{c}-ordinary-source-grid.json")}
        for q in [70, 60, 50]:
            anchors[f"ordinary_at_joint_{q}"] = load(f"{c}-ordinary-at-joint-q{q}.json")
        for q in [24, 26] if c == "019" else [24, 27]:
            anchors[f"neural_q{q}"] = [compact(load(f"{c}-dcvc-q{q}.json"))]
    if not outer:
        return anchors
    for q in [22, 29] if c != "024" else [23, 31]:
        path = ROOT / f"{c}-dcvc-q{q}.json"
        anchors[f"neural_q{q}"] = [compact(json.loads(path.read_text()))]
    return anchors


def reuse_threshold(primary, rows, init_bytes):
    """Find a modeled horizon only inside the measured anchor quality interval."""
    anchors = hull([(r["arm"]["kbps"], -r["mean_dists"]) for r in rows])
    target = -primary["mean_dists"]
    result = dict(anchor_hull=anchors, target_dists=-target)
    if len(anchors) < 2 or not anchors[0][1] <= target <= anchors[-1][1]:
        return dict(**result, state="outside measured anchor quality range")
    at_quality = math.exp(
        float(PchipInterpolator([q for _, q in anchors], [math.log(r) for r, _ in anchors])(target))
    )
    allowance = at_quality - primary["arm"]["kbps"]
    if allowance <= 0:
        return dict(**result, state="no positive warm bitrate advantage", anchor_kbps=at_quality)
    return dict(
        **result,
        state="modeled equal-quality threshold",
        anchor_kbps=at_quality,
        seconds=8 * init_bytes / (1000 * allowance),
        init_bytes=init_bytes,
    )


def core(c, *, outer=False):
    joint = [compact(load(f"{c}-joint-q{q}.json")) for q in [70, 60, 50]]
    anchors = controls(c, outer=outer)
    primary = joint[1]
    cap = primary["arm"]["bytes"]
    best = {}
    for mode in ["ordinary", "neural"]:
        rows = [r for k, v in anchors.items() if k.startswith(mode) for r in v]
        eligible = [r for r in rows if r["arm"]["bytes"] <= cap]
        assert eligible
        best[mode] = compact(min(eligible, key=lambda r: r["mean_dists"]))
        near = min(rows, key=lambda r: (abs(r["arm"]["bytes"] - cap), r["arm"]["bytes"]))
        best[f"{mode}_nearest"] = compact(near)
    cold = compact(load(f"{c}-ordinary-cold-score.json"))
    assert cold["arm"]["kbps"] <= primary["arm"]["cold_kbps"]
    wire = next(x for x in load("three-native-wire-costs.json") if x["shot"] == c)
    assert wire["joint_bytes"] == cap
    horizon = {}
    for mode in ["ordinary", "neural"]:
        measured = [r for k, v in anchors.items() if k.startswith(mode) for r in v]
        if mode == "ordinary":
            measured.append(cold)
        horizon[mode] = reuse_threshold(primary, measured, wire["init_bytes"])
    return dict(
        modeled_reuse=horizon,
        clip=c,
        joint=joint,
        primary=primary,
        anchors=best,
        bd=summarize(anchors, joint),
        cold=dict(
            pointstream_kbps=primary["arm"]["cold_kbps"],
            ordinary=cold,
            player_dists_difference=primary["mean_dists"] - cold["mean_dists"],
        ),
    )


def run(outer=False):
    cases = [core(c, outer=outer) for c in ["000", "019", "024"]]
    result = dict(
        denominator="One development match, three prespecified shots, each 240 native 50-fps frames with earliest 120-tick jointly supported interval and 120 conventional fallback ticks",
        cases=cases,
        within_match={},
    )
    for key in ["ordinary", "neural", "neural_nearest"]:
        result["within_match"][key] = {}
        for field in ["mean_dists", "active_dists", "fallback_dists", "rgb_psnr"]:
            j = [c["primary"][field] for c in cases]
            a = [c["anchors"][key][field] for c in cases]
            result["within_match"][key][field] = dict(
                pointstream_mean=mean(j),
                anchor_mean=mean(a),
                difference=mean(j) - mean(a),
                relative_change=(mean(j) / mean(a) - 1) if field != "rgb_psnr" else None,
                by_shot_differences=[x - y for x, y in zip(j, a, strict=True)],
            )
    stem = "three-native-shots-expanded-neural" if outer else "three-native-shots"
    (ROOT / (stem + ".json")).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--outer", action="store_true")
    run(cli.parse_args().outer)
