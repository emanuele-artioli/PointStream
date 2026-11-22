"""Compare measured rate-quality envelopes on one source and one label set."""

import json
from pathlib import Path

from src.pointstream.evaluation import anchor_sensitivity

HERE = Path(__file__).resolve().parent
ANCHORS = {"Ordinary AV1", "AV1 with ROI", "DCVC-UF"}


def analyze(rows):
    result = {}
    for region in ("players", "ball"):
        anchors = [r for r in rows if r["mode"] in ANCHORS]
        modes = sorted({r["mode"] for r in rows} - ANCHORS)
        measurements = {}
        for mode in modes:
            test = [r for r in rows if r["mode"] == mode]
            measurements[mode] = {}
            for name, lo, hi in (("full", 0, float("inf")), ("low", 19, 126)):
                curves = [
                    [(r["kbps"], -r[region]) for r in group if lo <= r["kbps"] <= hi]
                    for group in (test, anchors)
                ]
                if all(curves):
                    measurements[mode][name] = anchor_sensitivity(
                        *curves, minimum_quality_span=0.005
                    )
        winners = {}
        for budget in (20, 22, 33, 50, 65, 125):
            winners[budget] = {}
            for kind, group in (
                ("measured_anchors", anchors),
                ("actual_deliveries", [r for r in rows if "valid-cache scenario" not in r["mode"]]),
                ("including_valid_cache_scenario", rows),
            ):
                eligible = [r for r in group if r["kbps"] <= budget]
                winners[budget][kind] = (
                    min(eligible, key=lambda r: (r[region], r["kbps"])) if eligible else None
                )
        result[region] = {"comparisons": measurements, "observed_winners_under_caps": winners}
    return result


if __name__ == "__main__":
    data = json.loads((HERE / "points.json").read_text())
    (HERE / "envelopes.json").write_text(json.dumps(analyze(data), indent=2) + "\n")
