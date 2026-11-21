"""Score transmitted ball locations against the authenticated source labels."""
import argparse
import csv
import hashlib
import io
import json
import math
import tarfile
from pathlib import Path

ARCHIVE = "e3ae88ccbab68a19aa0a34a9ad12328352d00a69ae44901edfd7cbe01220aa12"
CSV = "e50089b4659ad8c84d84f1794347ad08445b8552784d607cb1af3b68ad9cdfba"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--historical", type=Path, required=True)
    p.add_argument("--positions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    archive_bytes = args.historical.read_bytes()
    if hashlib.sha256(archive_bytes).hexdigest() != ARCHIVE:
        raise ValueError("Historical archive mismatch")
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
        raw = archive.extractfile("actual-analysis/frame-transitions.csv").read()
    if hashlib.sha256(raw).hexdigest() != CSV:
        raise ValueError("Label CSV mismatch")
    labels = [r for r in csv.DictReader(io.StringIO(raw.decode())) if
              r["point"] == "svt-62" and r["judge"] == "wasb" and r["threshold_px"] == "10"]
    if len(labels) != 292:
        raise ValueError("Expected exactly one row per frame")
    paid = json.loads(args.positions.read_text())
    result = {}
    for name, stream in paid.items():
        if len(stream["positions"]) != 292:
            raise ValueError("Position frame count differs")
        counts = {"visible": 0, "absent": 0, "uncertain": 0, "visible_hit10": 0,
                  "visible_hit5": 0, "absent_false": 0, "positions_present": 0}
        for i, (label, point) in enumerate(zip(labels, stream["positions"])):
            if int(label["frame"]) != i:
                raise ValueError("Frame order differs")
            kind = int(label["label_kind"])
            counts[{1: "visible", 0: "absent", -1: "uncertain"}[kind]] += 1
            if point is not None:
                counts["positions_present"] += 1
                if kind == 1:
                    distance = math.hypot(point[0] - float(label["label_x"]),
                                          point[1] - float(label["label_y"]))
                    counts["visible_hit10"] += distance <= 10
                    counts["visible_hit5"] += distance <= 5
                if kind == 0:
                    counts["absent_false"] += 1
        if (counts["visible"], counts["absent"], counts["uncertain"]) != (261, 30, 1):
            raise ValueError("Label coverage differs")
        result[name] = {**counts, "packet_bytes": stream["packet_bytes"],
                        "packet_sha256": stream["packet_sha256"]}
    output = {"scope": "positions decoded directly from the already-paid fitted-streak payload",
              "historical_archive_sha256": ARCHIVE, "label_csv_sha256": CSV, "arms": result}
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
