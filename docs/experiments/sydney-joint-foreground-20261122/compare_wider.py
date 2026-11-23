"""Compare the new native Sydney measurements with the frozen tested codec bank."""

import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ranked(bank, ceiling):
    available = [
        min(
            (r for r in rows if r["bytes"] <= ceiling),
            key=lambda row: (row["scores"]["foreground"], row["bytes"]),
            default=None,
        )
        for rows in bank.values()
    ]
    return min(
        (row for row in available if row is not None),
        key=lambda row: (row["scores"]["foreground"], row["bytes"]),
        default=None,
    )


def compare(metrics_path, calibration_paths):
    baseline_path = HERE / "comparison-all-physical.json"
    bank = read(baseline_path)
    metrics = read(metrics_path)
    calibrated = {
        r["files"]["rally.psj"]: r for path in calibration_paths for r in read(path)["rates"]
    }
    assert metrics["source_sha256"] == bank["source"] and metrics["sampled_ticks"] == bank["ticks"]
    pointstream = [r["joint"] for r in bank["measured_rungs"].values()]
    new = []
    for arm in metrics["arms"]:
        q = calibrated[arm["joint_sha256"]]
        n, init = q["joint_bytes"], q["initial_bytes"]
        assert arm["warm_kbps"] == 0.002 * n and arm["cold_kbps"] == 0.002 * (n + init)
        row = {
            "arm": f"PointStream {arm['coded_size'][0]} wide CRF {q['crf']}",
            "bytes": n,
            "sha256": arm["joint_sha256"],
            "scores": {
                "foreground": arm["means"]["foreground"]["dists"],
                "players": arm["means"]["player"]["dists"],
                "ball": arm["means"]["ball"]["dists"],
            },
        }
        pointstream.append(row)
        new.append(
            {
                "crf": q["crf"],
                "pointstream": row,
                "cold_bytes": n + init,
                "cached_anchor": ranked(bank["controls"], n),
                "cold_anchor": ranked(bank["controls"], n + init),
                "metrics": arm["means"],
            }
        )
    budgets = {}
    for key in ("42500", "68000", "96000"):
        cap = int(key)
        budgets[key] = {
            "best_pointstream": ranked({"pointstream": pointstream}, cap),
            "best_anchor": ranked(bank["controls"], cap),
        }
    tested_prices = sorted(
        {row["bytes"] for group in [*bank["controls"].values(), pointstream] for row in group}
    )
    measured_order = []
    for ceiling in tested_prices:
        p = ranked({"pointstream": pointstream}, ceiling)
        a = ranked(bank["controls"], ceiling)
        if p and a:
            winner = (
                "PointStream" if p["scores"]["foreground"] < a["scores"]["foreground"] else a["arm"]
            )
            measured_order.append(
                {"ceiling_bytes": ceiling, "winner": winner, "pointstream": p, "anchor": a}
            )
    return {
        "scope": "One Sydney rally. DISTS on twelve frozen source instants. Available measured packets only.",
        "metrics_sha256": digest(metrics_path),
        "original_bank_sha256": digest(baseline_path),
        "calibration_sha256": {path.name: digest(path) for path in calibration_paths},
        "wider": new,
        "budget_comparisons": budgets,
        "measured_order": measured_order,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("metrics", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--calibration", action="append", type=Path, required=True)
    args = p.parse_args()
    with args.out.open("x") as handle:
        json.dump(compare(args.metrics, args.calibration), handle, indent=2)
        handle.write("\n")
