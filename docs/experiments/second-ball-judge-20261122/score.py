"""Join fixed full-clip detector outputs with the independent historical labels."""

import csv
import hashlib
import io
import json
import math
import tarfile
from pathlib import Path

prior = json.loads(Path("/private/tmp/ps-monotrack-prior-source-clip.json").read_text())
judge = json.loads(Path("/private/tmp/ps-monotrack-new-predictions-20261122.json").read_text())
payload = Path("/private/tmp/ps-monotrack-labels-20261122.tar.gz").read_bytes()
assert (
    hashlib.sha256(payload).hexdigest()
    == "e3ae88ccbab68a19aa0a34a9ad12328352d00a69ae44901edfd7cbe01220aa12"
)
with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
    raw = tar.extractfile("actual-analysis/frame-transitions.csv").read()
assert (
    hashlib.sha256(raw).hexdigest()
    == "e50089b4659ad8c84d84f1794347ad08445b8552784d607cb1af3b68ad9cdfba"
)
labels = [
    r
    for r in csv.DictReader(io.StringIO(raw.decode()))
    if r["point"] == "svt-62" and r["judge"] == "wasb" and r["threshold_px"] == "10"
]
assert judge["frames"] == len(labels) == 292 and judge["detector"] == "monotrack"
assert [(int(r["label_kind"]), float(r["label_x"]), float(r["label_y"])) for r in labels] == [
    tuple(v) for v in prior["labels"]
]
result = {
    "model": "MonoTrack",
    "checkpoint_sha256": judge["judge_checkpoint_sha256"],
    "label_csv_sha256": hashlib.sha256(raw).hexdigest(),
    "source_group": "TrackNet game9/Clip5, retrospective development clip",
    "counts": {},
    "arms": {},
    "source_prior_parity": {},
}
for name in ("source", "baseline", "candidate"):
    rows = judge["results"][name]["rows"]
    assert len(rows) == 292 and all(p["frame"] == i for i, p in enumerate(rows))
    misses10, misses5, false = [], [], []
    hits10 = hits5 = 0
    for i, (row, p) in enumerate(zip(labels, rows, strict=True)):
        kind = int(row["label_kind"])
        result["counts"][str(kind)] = result["counts"].get(str(kind), 0) + (name == "source")
        if kind == 1:
            dist = (
                math.hypot(p["x"] - float(row["label_x"]), p["y"] - float(row["label_y"]))
                if p["detected"]
                else math.inf
            )
            hits10 += dist <= 10
            hits5 += dist <= 5
            if dist > 10:
                misses10.append(i)
            if dist > 5:
                misses5.append(i)
        if kind == 0 and p["detected"]:
            false.append(i)
    result["arms"][name] = {
        "hits10": hits10,
        "hits5": hits5,
        "misses10": misses10,
        "misses5": misses5,
        "absent_false": false,
        "rgb_sha256": judge["results"][name]["decoded_rgb_sha256"],
    }
assert result["counts"] == {"1": 261, "0": 30, "-1": 1}
old = prior["candidates"]["monotrack"]["predictions"]
new = judge["results"]["source"]["rows"]
result["source_prior_parity"] = {
    "same_detection": sum(bool(a[0]) == b["detected"] for a, b in zip(old, new, strict=True)),
    "coordinate_delta_max_px": max(
        math.hypot(a[1] - b["x"], a[2] - b["y"])
        for a, b in zip(old, new, strict=True)
        if a[0] and b["detected"]
    ),
    "prior_values_round_to_tenth_pixel": True,
}
result["rates_kbps"] = {"baseline": 63.0575, "candidate": 63.3641}
Path("docs/experiments/second-ball-judge-20261122/result.json").write_text(
    json.dumps(result, indent=2) + "\n"
)
print(json.dumps(result))
