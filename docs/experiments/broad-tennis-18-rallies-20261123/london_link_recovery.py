"""Retry bounded gaps on saved London person detections."""
import hashlib
import json
from pathlib import Path

import math


def distance(first, second):
    ac = ((first[0] + first[2]) / 2, (first[1] + first[3]) / 2)
    bc = ((second[0] + second[2]) / 2, (second[1] + second[3]) / 2)
    return math.dist(ac, bc) / math.dist(first[:2], first[2:])

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "london-source-detection-106.json"
PROPOSALS = Path("/private/tmp/pointstream-source-proposals/review/person-proposals.js")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    data = json.loads(SOURCE.read_text())
    assert data["frames"] == 106 and data["review_sha256"] == sha(PROPOSALS)
    proposals = json.loads(PROPOSALS.read_text().split("=", 1)[1].strip().removesuffix(";"))
    reviewed = {int(k.split(":")[1]): v for k, v in proposals.items() if k.startswith("v006_p0014:")}
    tracked = {6: [p["box"] for p in reviewed[6]]}
    for indices in (range(7, 106), range(5, -1, -1)):
        previous = tracked[6].copy()
        gaps = [0, 0]
        for tick in indices:
            candidates = data["detections"][tick]["boxes"]
            used, selected = set(), [None, None]
            for role, old in enumerate(previous):
                if gaps[role] > 3:
                    continue
                options = sorted((distance(old, box), i, box) for i, box in enumerate(candidates)
                                 if i not in used)
                if options and options[0][0] <= 1:
                    _, index, box = options[0]
                    used.add(index)
                    selected[role] = box
                    previous[role] = box
                    gaps[role] = 0
                else:
                    gaps[role] += 1
            tracked[tick] = selected
    comparisons = []
    for tick, expected in sorted(reviewed.items()):
        actual = tracked[tick]
        errors = [distance(p["box"], found) if found is not None else None
                  for p, found in zip(expected, actual, strict=True)]
        comparisons.append({"tick": tick, "normalized_center_error": errors,
                            "matched": all(v is not None and v <= 0.5 for v in errors)})
    result = {"scope": "exploratory seeded candidate association, no inferred boxes or pose",
              "source_receipt_sha256": sha(SOURCE), "script_sha256": sha(Path(__file__)),
              "review_sha256": sha(PROPOSALS), "gap_limit_ticks": 3,
              "complete_ticks": sum(all(box is not None for box in v) for v in tracked.values()),
              "missing_ticks": sorted(t for t, v in tracked.items() if any(box is None for box in v)),
              "review_comparisons": comparisons}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    run()
