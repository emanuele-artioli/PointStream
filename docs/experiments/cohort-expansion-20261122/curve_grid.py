"""Freeze the physical rate interval without reading any secondary quality scores."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
GRID = [10, 15, 20.4, 35, 60, 100, 150]


def build(data):
    staged = json.loads((HERE / "staged.json").read_text())["selection"]
    members = []
    excluded = []
    for row in staged:
        m = row["match"]
        rate = data / (m + "-rate.json")
        score = data / (m + "-score.json")
        if not rate.exists() or not score.exists():
            raise ValueError("Incomplete registered primary cohort: " + m)
        r = json.loads(rate.read_text())
        s = json.loads(score.read_text())
        assert (
            r["match"] == s["match"] == m
            and r["source_sha256"] == s["source_sha256"] == row["source"]["sha256"]
        )
        if s["visible"] == 0:
            excluded.append(dict(match=m, reason="no independently visible mark"))
            continue
        fps = float(Fraction(r["fps"]))
        ranges = {}
        for arm, field in [("ordinary", "bytes"), ("candidate", "ball_bytes")]:
            prices = [v[field] * 8 * fps / 240000 for v in r["rows"]]
            ranges[arm] = [min(prices), max(prices)]
        members.append(
            dict(
                match=m,
                visible=s["visible"],
                ranges=ranges,
                source_sha256=r["source_sha256"],
                rate_archive_sha256=s["rate_archive_sha256"],
                rate_report_sha256=hashlib.sha256(rate.read_bytes()).hexdigest(),
            )
        )
    lower = max(v[0] for m in members for v in m["ranges"].values())
    upper = min(v[1] for m in members for v in m["ranges"].values())
    if lower >= upper:
        raise ValueError("Empty common physical price interval; report each source separately.")
    grid = [v for v in GRID if lower <= v <= upper]
    if len(grid) < 3:
        grid = sorted(set(grid + np.geomspace(lower, upper, 3).tolist()))
    return dict(
        members=members,
        excluded=excluded,
        price_interval_kbps=[lower, upper],
        grid_kbps=grid,
        registered_prices_kbps=GRID,
        estimand="Per complete event and available ceiling, select lowest metric loss from physical packets no dearer than the ceiling; equal weight events. Supported on the intersection of both families’ physically observed prices.",
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=HERE / "test-physical")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = build(a.data)
    with a.out.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
