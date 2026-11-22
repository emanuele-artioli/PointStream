"""Apply the observed court route to the measured same-quality per-match rate ratios."""

import json
from pathlib import Path

import numpy as np


def calculate(routed, curves):
    rows = routed["rows"]
    assert sorted(r["match"] for r in rows) == sorted(curves["per_event"])
    ratios = {}
    for row in rows:
        match = row["match"]
        rate_change = curves["per_event"][match]["bd_rate"]["bd_rate_percent"]
        ratios[match] = 1 + rate_change / 100 if row["ball_selected"] else 1.0
    log_ratios = np.log(list(ratios.values()))
    draws = np.random.default_rng(20261122).integers(len(rows), size=(10000, len(rows)))
    return dict(
        rates=ratios,
        saving_percent=100 * (1 - np.exp(np.mean(log_ratios))),
        saving_percent_95=np.quantile(
            100 * (1 - np.exp(log_ratios[draws].mean(axis=1))), [0.025, 0.975]
        ).tolist(),
        matches=len(rows),
        conventional=sum(not row["ball_selected"] for row in rows),
    )


if __name__ == "__main__":
    card = Path(__file__).resolve().parent
    curves = card.parent / "cohort-expansion-20261122" / "curve-summary.json"
    result = calculate(
        json.loads((card / "result.json").read_text()), json.loads(curves.read_text())
    )
    print(json.dumps(result))
    (card / "routed-rate.json").write_text(json.dumps(result, indent=2) + "\n")
