"""Account for full London routes and source-reviewed ball emissions."""
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUTS = [HERE / "tenniset-ball-summary.json", HERE / "london-reviewed-scores.json",
          HERE / "route-summary.json"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    delivery = json.loads(INPUTS[0].read_text())
    scores = json.loads(INPUTS[1].read_text())
    routes = json.loads(INPUTS[2].read_text())
    by_key = {}
    for row in scores["rows"]:
        by_key.setdefault((row["rally"], row["tick"]), []).append(row)
    assert len(by_key) == 48 and all(len(v) == 2 for v in by_key.values())
    groups = {}
    for (rally, tick), paired in by_key.items():
        first, second = paired
        for field in ("mark", "assisted", "route", "emitted", "error_px"):
            assert first[field] == second[field], (rally, tick, field)
        key = first["mark"] + ("_assisted" if first["assisted"] else "_independent")
        counter = groups.setdefault(rally, Counter())
        counter[key] += 1
        counter[key + "_emitted"] += int(first["emitted"])
        if first["mark"] == "visible":
            counter[key + "_" + first["route"]] += 1
            counter[key + "_within_10px"] += int(first["error_px"] is not None and first["error_px"] <= 10)
    by_rally = {row["rally_id"]: row for row in delivery["rows"]}
    route_by = {row["rally_id"]: row for row in routes["rows"]}
    output = []
    for rally, marks in groups.items():
        case, route = by_rally[rally], route_by[rally]
        assert route["evaluated_range"] == [0, case["frames"]]
        output.append({"rally": rally, "native_frames": case["frames"],
            "court_frames": case["frames"] - case["fallback_ticks"],
            "fallback_frames": case["fallback_ticks"], "delivered_positions": case["positions"],
            "reviewed": dict(marks)})
    totals = Counter()
    for row in output:
        totals.update(row["reviewed"])
        for name in ("native_frames", "court_frames", "fallback_frames", "delivered_positions"):
            totals[name] += row[name]
    result = {"scope": "Six London points; two matches at one event; source-reviewed mark availability depends on tracker suggestions",
              "inputs": {p.name: sha(p) for p in INPUTS}, "script_sha256": sha(Path(__file__)),
              "totals": dict(totals), "rallies": output}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    run()
