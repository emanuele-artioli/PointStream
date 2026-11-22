"""Summarize complete native mixed timelines, keeping the three shots as dependent repeats."""

import json
from pathlib import Path
from statistics import mean

from rate_summary import summarize

ROOT = Path(__file__).resolve().parent


def load(name):
    return json.loads((ROOT / name).read_text())


def compact(r):
    return {k: r[k] for k in ["arm", "mean_dists", "active_dists", "fallback_dists", "rgb_psnr"]}


def core(c):
    joint = [compact(load(f"{c}-joint-q{q}.json")) for q in [70, 60, 50]]
    if c == "000":
        anchors = load("000-control-scores.json")
    else:
        anchors = {"ordinary_grid": load(f"{c}-ordinary-source-grid.json")}
        for q in [70, 60, 50]:
            anchors[f"ordinary_at_joint_{q}"] = load(f"{c}-ordinary-at-joint-q{q}.json")
        for q in [24, 26] if c == "019" else [24, 27]:
            anchors[f"neural_q{q}"] = [compact(load(f"{c}-dcvc-q{q}.json"))]
    for q in [22, 29] if c != "024" else [23, 31]:
        path = ROOT / f"{c}-dcvc-q{q}.json"
        if path.exists():
            anchors[f"neural_q{q}"] = [compact(json.loads(path.read_text()))]
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
    return dict(clip=c, joint=joint, primary=primary, anchors=best, bd=summarize(anchors, joint))


def run():
    cases = [core(c) for c in ["000", "019", "024"]]
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
    (ROOT / "three-native-shots.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    run()
