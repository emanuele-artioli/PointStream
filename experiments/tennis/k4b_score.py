"""Score frozen K4b rules against the returned hand-selected players.

Each marked frame is scored on its own window's features (K4b.1 amendment 1).
Accept only features with a preserved local full-stage validation receipt and
job revision from ready.json. Never applies a rule to K4's original windows.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from experiments.tennis import k4, k4b_rules as rules
from experiments.tennis.k4b import iou, numbered
from experiments.visor.b1 import file_sha256, write_json
from src.segmentation.masks import ClipMasks


def load_windows(inputs: list[dict[str, Any]]) -> dict[str, tuple[dict[str, Any], ClipMasks]]:
    """Window features from feature jobs whose full stage passed the validator locally."""
    windows: dict[str, tuple[dict[str, Any], ClipMasks]] = {}
    for source in inputs:
        root = Path(source["directory"])
        for name, digest in source["files"].items():
            if file_sha256(root / name) != digest:
                raise ValueError(f"Changed feature artifact: {root / name}")
        validation = json.loads((root / "local-validation.json").read_text())
        ready = json.loads((root / "ready.json").read_text())
        summary = json.loads((root / "k4b-features.json").read_text())
        if (validation.get("passed") is not True or not validation.get("checks")
                or not all(validation["checks"].values()) or summary["stage"] != "full"
                or summary["configured_windows"] != 0):
            raise ValueError("Only locally validated full-stage window features may be scored")
        if ready["git_head"] != source["git_head"] or validation["windows"] != [w["id"] for w in summary["windows"]]:
            raise ValueError("Job revision/window provenance mismatch")
        for w in summary["windows"]:
            directory = root / "publish" / k4.safe(w["id"])
            for name in ("features.json", "target-masks.rle"):
                if file_sha256(directory / name) != w["files"][name]:
                    raise ValueError(f"Changed window artifact: {directory / name}")
            feature = json.loads((directory / "features.json").read_text())
            if feature["frame_count"] != w["frames"] or feature["target_frames"] != [w["target"]]:
                raise ValueError("Scoring requires the complete window and its marked frame")
            if feature["rules"] != rules.RULES or feature["picks"] != rules.choose(feature["features"]):
                raise ValueError("Feature selections do not use the frozen rules")
            if w["id"] in windows:
                raise ValueError(f"Duplicate window: {w['id']}")
            windows[w["id"]] = (feature, ClipMasks.load(directory / "target-masks.rle"))
    return windows


def score(marks: dict[str, Any], review: dict[str, Any], review_path: Path,
          review_dir: Path, inputs: list[dict[str, Any]], partial: bool = False) -> dict[str, Any]:
    labels = rules.validate_marks(marks, review, review_path)
    windows = load_windows(inputs)
    rows = []
    for frame in review["frames"]:
        clip_id = f"{frame['match']}/{frame['clip']}"
        window_id = f"{clip_id}@{frame['frame']}"
        if window_id not in windows:
            if partial:
                continue
            raise ValueError(f"Missing window features: {window_id}")
        feature, masks = windows[window_id]
        by_track = {inst.track_id: k4.upsample(inst.mask(), (1920, 1080)) for inst in masks.frames[0]}
        path = review_dir / frame["masks"]
        if file_sha256(path) != frame["masks_sha256"]:
            raise ValueError("Hand-marked review mask identity changed")
        by_id = {r["id"]: k4.upsample(r["mask"], (1920, 1080)) for r in numbered(ClipMasks.load(path))}
        label = labels[frame["id"]]
        truth = [by_id[n] for n in label["selected"]]
        oracle = (not label["missing_instance"] if label["visible"] == 0
                  else rules.correct(list(by_track.values()), truth, label["visible"], label["missing_instance"]))
        comparisons = {}
        for candidate, chosen in feature["picks"].items():
            present = [track for track in chosen if feature["track_ids"][track] in by_track]
            predicted = [by_track[feature["track_ids"][track]] for track in present]
            comparisons[candidate] = {"correct": rules.correct(predicted, truth, label["visible"], label["missing_instance"]),
                                      "window_picks": chosen, "present": present,
                                      "iou": [[iou(a, b) for b in truth] for a in predicted]}
        rows.append({"id": frame["id"], "clip": clip_id, "window": window_id, "match": frame["match"], "visible": label["visible"],
                     "selected": label["selected"], "proposal_oracle": oracle, "candidates": comparisons})
    if not rows:
        raise ValueError("No scored review frames")
    total, done = len(labels), len(rows)
    complete = done == total
    table = []
    for candidate in rules.RULES["tie_order"]:
        hits = sum(r["candidates"][candidate]["correct"] for r in rows)
        table.append({"candidate": candidate, "correct": hits, "scored": done,
                      "share": hits / done, "maximum_final_share": (hits + total - done) / total,
                      "reaches_bar": complete and hits / total >= .95})
    highest = max(r["correct"] for r in table)
    winner = next((r["candidate"] for r in table if r["correct"] == highest and r["reaches_bar"]), None) if complete else None
    return {"rules": rules.RULES, "frames": total, "scored": done, "complete": complete,
            "table": table, "selected_by_frozen_rule": winner, "applied_to_k4": False,
            "next": ("review K4 amendment before application" if winner else "user decides; court candidate deferred to adopted K8") if complete else "collect remaining window features",
            "visibility": dict(Counter(r["visible"] for r in rows)),
            "proposal_oracle": {"correct": sum(r["proposal_oracle"] for r in rows), "scored": done},
            "rows": rows, "jobs": [{k: v for k, v in source.items() if k != "files"} for source in inputs]}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--marks", type=Path, required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--features-index", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--partial", action="store_true")
    args = p.parse_args()
    review = args.review / "instances.json"
    result = score(json.loads(args.marks.read_text()), json.loads(review.read_text()), review, args.review,
                   json.loads(args.features_index.read_text())["jobs"], args.partial)
    result["inputs"] = {str(p): file_sha256(p) for p in (args.marks, review, args.features_index)}
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "k4b-report.json", result)
    lines = ["| Candidate | Correct / scored | Share | Maximum over all frames |", "|---|---:|---:|---:|"]
    lines += [f"| {r['candidate']} | {r['correct']} / {r['scored']} | {r['share']:.4f} | {r['maximum_final_share']:.4f} |" for r in result["table"]]
    (args.out / "table.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k: result[k] for k in ("frames", "scored", "complete", "table", "selected_by_frozen_rule", "next")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
