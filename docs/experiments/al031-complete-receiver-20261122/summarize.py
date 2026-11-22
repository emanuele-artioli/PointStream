"""Recover fixed-support fresh/hit comparisons and the exact affordable control envelopes."""

import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent


def read(p):
    return json.loads(p.read_text())


def summarize(r):
    rows = r["body_rows"]
    assert len(rows) == 480 and {(q["tick"], q["player_index"]) for q in rows} == {
        (t, p) for t in range(240) for p in range(2)
    }
    ticks = set(range(10, 240, 20))
    roles = {}
    for i, n in enumerate(["far", "near"]):
        one = [x for x in rows if x["player_index"] == i]
        roles[n] = dict(
            all240=mean(x["dists"] for x in one),
            fixed12=mean(x["dists"] for x in one if x["tick"] in ticks),
        )
    result = dict(
        arm=r["arm"],
        support=r["supports_sha256"],
        source=r["source_sha256"],
        mean_dists=r["mean_dists"],
        fixed12=mean(x["dists"] for x in rows if x["tick"] in ticks),
        roles=roles,
        rgb_psnr=r["rgb_psnr"],
    )
    if "ball" in r:
        result.update(
            ball_dists=r["ball"]["dists"],
            ball_present=r["ball"]["present"],
            ball_source_payload=r["ball"]["payload_sha256"],
        )
    return result


def run(a):
    warm = read(ROOT / "two-shot-price.json")["reuse"]["return_kbps"]
    cold = read(ROOT / "two-shot-price.json")["send_new"]["return_kbps"]
    bank = {}
    for p in sorted(ROOT.glob("*joint-q*-scores.json")):
        q = read(p)
        bank[q["arm"]["name"]] = summarize(q)
    for p in sorted(ROOT.glob("ordinary-retained-*-scores.json")):
        q = read(p)
        bank[q["arm"]["name"]] = summarize(q)
    for p in sorted(ROOT.glob("neural-*-body-and-ball.json")):
        q = read(p)
        bank[q["arm"]["name"]] = summarize(q)
    for f in [a.grid, a.matched, a.cold]:
        for p in sorted(f.glob("*.json")):
            if p.name in ["arms.json", "summary.json"]:
                continue
            try:
                q = read(p)
            except ValueError:
                continue
            if "body_rows" not in q:
                continue
            bank[q["arm"]["name"]] = summarize(q)
    low_grid = ROOT / "original-ordinary-low-grid-body-and-ball.json"
    if low_grid.exists():
        for q in read(low_grid):
            name = q["arm"]["name"]
            assert q["arm"]["sha256"] == bank[name]["arm"]["sha256"]
            assert abs(q["mean_dists"] - bank[name]["mean_dists"]) < 1e-7
            bank[name] = summarize(q)
    comparisons = {}
    for name in ["fresh-joint-q70", "fresh-joint-q60", "fresh-joint-q50", "prior-joint-q50"]:
        if name not in bank:
            continue
        j = bank[name]
        ceilings = (
            {"warm": j["arm"]["kbps"]} if name.startswith("fresh") else {"observed_hit": warm}
        )
        if name == "fresh-joint-q50":
            ceilings["cold"] = cold
        for state, cap in ceilings.items():
            pair = {}
            for family in ["ordinary", "dcvc"]:
                options = [
                    v
                    for k, v in bank.items()
                    if k.startswith(family) and v["arm"]["kbps"] <= cap + 1e-10
                ]
                if not options:
                    pair[family] = dict(
                        compared=False, reason="No affordable native encode at this rate."
                    )
                    continue
                best = min(options, key=lambda q: q["mean_dists"])
                if (
                    state == "cold"
                    and family == "dcvc"
                    and max(item["arm"]["kbps"] for item in options) < 0.98 * cap
                ):
                    pair[family] = dict(
                        compared=False,
                        reason="Measured neural packets do not cover the cold-start bitrate.",
                        maximum_measured_kbps=max(item["arm"]["kbps"] for item in options),
                    )
                    continue
                pairs = dict(
                    mean_dists=best["mean_dists"],
                    player_control=best["arm"]["name"],
                    player_control_kbps=best["arm"]["kbps"],
                    player_percent_lower=100 * (1 - j["mean_dists"] / best["mean_dists"]),
                    rgb_psnr_change=j["rgb_psnr"] - best["rgb_psnr"],
                )
                with_ball = [v for v in options if "ball_dists" in v]
                if with_ball:
                    b = min(with_ball, key=lambda q: q["ball_dists"])
                    pairs.update(
                        ball_control=b["arm"]["name"],
                        ball_control_kbps=b["arm"]["kbps"],
                        ball_percent_lower=100 * (1 - j["ball_dists"] / b["ball_dists"]),
                    )
                pair[family] = pairs
            comparisons[name + "-" + state] = dict(price=cap, controls=pair)
    result = dict(
        frames=240,
        players_per_frame=2,
        fixed12=list(range(10, 240, 20)),
        source_support="source-SAM model, no human contours",
        results=bank,
        comparisons=comparisons,
    )
    a.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    for k in ["grid", "matched", "cold", "out"]:
        p.add_argument("--" + k, type=Path, required=True)
    run(p.parse_args())
