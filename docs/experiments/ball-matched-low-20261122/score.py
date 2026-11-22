"""Join independent RGB detector output to the authenticated saved source labels."""
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


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main(args):
    original = args.historical.read_bytes()
    if digest(original) != ARCHIVE:
        raise ValueError("Historical source archive differs")
    with tarfile.open(fileobj=io.BytesIO(original), mode="r:gz") as tar:
        data = tar.extractfile("actual-analysis/frame-transitions.csv").read()
    if digest(data) != CSV:
        raise ValueError("Historical source labels differ")
    records = [r for r in csv.DictReader(io.StringIO(data.decode())) if
               r["point"] == "svt-62" and r["judge"] == "wasb" and r["threshold_px"] == "10"]
    if len(records) != 292:
        raise ValueError("Historical record count differs")
    judge = json.loads(args.predictions.read_text())
    if judge["frames"] != 292 or judge["detector"] != "wasb-step1":
        raise ValueError("Need complete WASB detections")
    counts = {"visible": 0, "absent": 0, "uncertain": 0}
    detections = {name: {"10": 0, "5": 0, "false": 0} for name in ("baseline", "candidate")}
    parity = {"equal_detection": 0, "equal_coordinate_when_detected": 0, "detected_both": 0}
    labels = []
    for i, row in enumerate(records):
        if int(row["frame"]) != i:
            raise ValueError("Historical labels are out of sequence")
        kind = int(row["label_kind"])
        x, y = float(row["label_x"]), float(row["label_y"])
        labels.append({"frame": i, "kind": kind, "x": x, "y": y})
        counts[{1: "visible", 0: "absent", -1: "uncertain"}[kind]] += 1
        for name in detections:
            pred = judge["results"][name]["rows"][i]
            if pred["frame"] != i:
                raise ValueError("Detector labels are out of sequence")
            if kind == 1 and pred["detected"]:
                distance = math.hypot(pred["x"] - x, pred["y"] - y)
                for threshold in (5, 10):
                    detections[name][str(threshold)] += distance <= threshold
            elif kind == 0 and pred["detected"]:
                detections[name]["false"] += 1
        old = row["baseline_detected"] == "True"
        new = judge["results"]["baseline"]["rows"][i]
        parity["equal_detection"] += old == new["detected"]
        if old and new["detected"]:
            parity["detected_both"] += 1
            parity["equal_coordinate_when_detected"] += (
                abs(float(row["baseline_x"]) - new["x"]) < 1e-6
                and abs(float(row["baseline_y"]) - new["y"]) < 1e-6
            )
    assert counts == {"visible": 261, "absent": 30, "uncertain": 1}
    result = {"counts": counts, "detections": detections, "baseline_historical_parity": parity,
              "baseline_bytes": judge["results"]["baseline"]["packet_bytes"],
              "candidate_bytes": judge["results"]["candidate"]["packet_bytes"],
              "judged_rgb_hashes": {k: v["decoded_rgb_sha256"] for k,v in judge["results"].items()},
              "source_archive_sha256": ARCHIVE, "source_label_csv_sha256": CSV,
              "checkpoint_sha256": judge["judge_checkpoint_sha256"],
              "scope": ("one retrospective game9/Clip5; refitted sender symbols" if "refit_receipt" in judge
                        else "one retrospective game9/Clip5; fixed original sender symbols")}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args())
