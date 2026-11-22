"""Choose between real deliveries using each fixed match's source-side court result."""

import json
from fractions import Fraction
from pathlib import Path

import numpy as np


def evaluate(data):
    rows = data["rows"]
    assert len(rows) == len({r["match"] for r in rows}) == 12
    assert all(r["court_ticks"] in (0, 240) for r in rows)
    records = []
    for r in rows:
        take_ball = r["court_ticks"] == 240
        duration = Fraction(240) / Fraction(r["fps"])

        def byte_count(kbps):
            estimate = kbps * 125 * float(duration)
            assert abs(estimate - round(estimate)) < 1e-6
            return round(estimate)

        records.append(
            dict(
                match=r["match"],
                ball_selected=take_ball,
                seconds=str(duration),
                selected_bytes=byte_count(r["candidate_kbps"] if take_ball else r["ordinary_kbps"]),
                ordinary_bytes=byte_count(r["ordinary_kbps"]),
                selected_dists=r["candidate_dists"] if take_ball else r["ordinary_dists"],
                ordinary_dists=r["ordinary_dists"],
                visible=r["direct"]["visible"],
                hit10=r["direct"]["hit10"] if take_ball else 0,
                absent=r["direct"]["marked_absent"],
                emitted_absent=r["direct"]["emitted_absent"] if take_ball else 0,
            )
        )
    candidate = np.array([r["selected_dists"] for r in records])
    ordinary = np.array([r["ordinary_dists"] for r in records])
    bootstrap = np.random.default_rng(20261122).integers(0, len(rows), size=(10000, len(rows)))
    gains = 100 * (1 - candidate[bootstrap].mean(1) / ordinary[bootstrap].mean(1))
    return dict(
        rows=records,
        seconds=float(sum(Fraction(r["seconds"]) for r in records)),
        routes_to_ball=sum(r["ball_selected"] for r in records),
        total={
            k: sum(r[k] for r in records)
            for k in (
                "selected_bytes",
                "ordinary_bytes",
                "hit10",
                "visible",
                "emitted_absent",
                "absent",
            )
        },
        mean_selected_dists=float(candidate.mean()),
        mean_ordinary_dists=float(ordinary.mean()),
        gain_percent=100 * (1 - float(candidate.mean() / ordinary.mean())),
        gain_percent_95=np.quantile(gains, [0.025, 0.975]).tolist(),
    )


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    source = root.parent / "cohort-expansion-20261122" / "cohort-results.json"
    result = evaluate(json.loads(source.read_text()))
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))
