"""Summarize native mixed-mode rates on fixed paired source player supports."""

import argparse
import json
from pathlib import Path

from experiments.tennis.k5 import bd_rate
from experiments.tennis.k6b_side_feasibility import hull


def summarize(controls, trials):
    def prices(rows):
        return hull([(r["arm"]["kbps"], -r["mean_dists"]) for r in rows])

    test = prices(trials)
    results = {}
    for key in ("ordinary", "neural"):
        rows = [r for k, choices in controls.items() if k.startswith(key) for r in choices]
        anchor = prices(rows)
        lo, hi = max(anchor[0][1], test[0][1]), min(anchor[-1][1], test[-1][1])
        score = bd_rate(test, anchor, 1e-9) if hi > lo else None
        results[key] = dict(
            common_dists=[-hi, -lo] if hi > lo else [],
            bd_rate_percent=None if score is None else score * 100,
            anchor_hull=anchor,
            test_hull=test,
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for k in ("controls", "joint", "out"):
        parser.add_argument("--" + k, type=Path, required=True)
    a = parser.parse_args()
    r = summarize(json.loads(a.controls.read_text()), json.loads(a.joint.read_text()))
    a.out.write_text(json.dumps(r, indent=2) + "\n")
