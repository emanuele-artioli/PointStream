"""Find all changes in the best affordable delivery as the weight on bodies changes."""

from itertools import combinations, pairwise
import json
from pathlib import Path
from statistics import mean

CARD = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def banks():
    data = read(CARD / "figure-data.json")
    first = []
    for r in read(CARD.parents[1] / "paper/figures/starved-choice/points.json"):
        name = {
            "Joint PointStream · valid-cache scenario": "PointStream",
            "Ordinary AV1": "AV1",
            "AV1 with ROI": "AV1",
            "DCVC-UF": "DCVC-UF",
        }.get(r["mode"])
        if name:
            first.append((name, r["kbps"], r["players"], r["ball"], r["source"]))
    second = [
        (m, r["kbps"], r["paired_player"], r["paired_ball"], r["stream_sha256"])
        for m, series in data["alcaraz_points"].items()
        for r in series
        if not r.get("archived_only")
    ]
    for row in read(CARD.parent / "two-rally-roi-20261122/foreground-scores.json")["al028"]["arms"]:
        second.append(
            ("AV1", row["bytes"] / 1200, row["players"], row["ball"], row["stream_sha256"])
        )
    low, high = data["caps_kbps"].values()
    return {
        "Nadal–Kyrgios": [r for r in first if r[1] <= low],
        "Alcaraz–Hurkacz": [r for r in second if r[1] <= high],
    }, [r for r in second if r[1] <= data["upper_alcaraz"]["DCVC-UF"]["kbps"]]


def score(r, w):
    return w * r[2] + (1 - w) * r[3]


def crossings(rows):
    ticks = set()
    for a, b in combinations(rows, 2):
        d = (a[2] - a[3]) - (b[2] - b[3])
        if d:
            w = (b[3] - a[3]) / d
            if 0 < w < 1:
                ticks.add(w)
    return ticks


def measure(groups):
    boundaries = sorted(set.union({0.0, 1.0}, *(crossings(rows) for rows in groups.values())))
    parts = []
    for left, right in pairwise(boundaries):
        selections = []
        for match, rows in groups.items():
            actual = min(rows, key=lambda r: (score(r, (left + right) / 2), r[1], r[-1]))
            selections.append(dict(match=match, mode=actual[0], kbps=actual[1], receipt=actual[-1]))
        if parts and parts[-1]["selections"] == selections:
            parts[-1]["stop"] = right
        else:
            parts.append(dict(start=left, stop=right, selections=selections))
    fixed = {}
    for w in (0, 0.25, 0.5, 0.75, 1):
        fixed[str(w)] = {
            mode: mean(min(score(r, w) for r in rows if r[0] == mode) for rows in groups.values())
            for mode in ("PointStream", "AV1", "DCVC-UF")
        }
    return dict(intervals=parts, pooled=fixed)


if __name__ == "__main__":
    low, upper = banks()
    data = dict(starved=measure(low), upper=measure({"Alcaraz–Hurkacz": upper}))
    output = CARD / "object-weights.json"
    output.write_text(json.dumps(data, indent=2) + "\n")
    for key, d in data.items():
        print(
            key,
            [
                (round(r["start"], 4), round(r["stop"], 4), [v["mode"] for v in r["selections"]])
                for r in d["intervals"]
            ],
        )
