"""K4b.1's court candidate (d) and its combinations, from K8b's per-frame homographies.

Defined in the K4b.1 entry before any (d) score was computed. Reads the same
windows, canonical tracks and SAM masks as the frozen v1 candidates (whose rules
are unchanged) and K8b's registration reports; CPU only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import k4, k4b_rules as rules
from experiments.tennis.k4b import numbered
from experiments.tennis.k4b_features import key
from experiments.tennis.k4b_score import load_windows
from experiments.visor.b1 import file_sha256, write_json
from src.segmentation.masks import ClipMasks

WIDTH, LENGTH = 10.97, 23.77
RULES_D: dict[str, Any] = {
    "version": 1, "registered": "K8b fits == true (error <= 6 px) on clip frame start + t",
    "foot": "lowest mask row at the centroid's x, 1080p", "side_margin_m": 1.0, "end_margin_m": 4.0,
    "d": "share of the track's registered frames inside; 0 without one", "inside_most": .5,
    "rank": {"d": ["inside_most only", "-d", "key"], "c+d": ["-inside_most", "-racket_most", "-d", "-c", "key"],
             "a+d": ["-inside_most", "-a", "key"]},
    "tie_order": ["a", "b", "c", "d", "e", "a+c", "c+d", "a+d"],
}


def registrations(reports: list[Path]) -> dict[str, dict[str, Any]]:
    """Clip id -> {sha256, fits: {frame: homography}} over K8b's reports (one clip in one report)."""
    out: dict[str, dict[str, Any]] = {}
    for path in reports:
        for clip in json.loads(path.read_text())["clips"]:
            if clip["id"] in out:
                raise ValueError(f"Clip in two reports: {clip['id']}")
            out[clip["id"]] = {"sha256": clip["sha256"], "fits": {r["frame"]: np.asarray(r["homography"], np.float64)
                                                                  for r in clip["registration"] if r["fits"]}}
    return out


def foot(mask: np.ndarray) -> tuple[float, float]:
    """The mask's lowest row at its centroid's x, in 1080p pixels (masks are at SAM's 960x540)."""
    ys, xs = np.nonzero(mask)
    scale = 1080 / mask.shape[0]
    return float(xs.mean()) * scale, float(ys.max() + 1) * scale


def inside(point: tuple[float, float], homography: np.ndarray) -> bool:
    """Whether the image point lies on the court plane within the doubles court plus the margins."""
    x, y, w = np.linalg.inv(homography) @ np.array([point[0], point[1], 1.])
    if w <= 0:
        return False
    cx, cy = x / w, y / w
    side, end = RULES_D["side_margin_m"], RULES_D["end_margin_m"]
    return bool(-side <= cx <= WIDTH + side and -end <= cy <= LENGTH + end)


def court_features(clip: ClipMasks, aliases: dict[str, str], start: int, fits: dict[int, np.ndarray]) -> dict[str, dict[str, Any]]:
    """Per canonical track: registered frames, inside frames and d (person mask first, as K4b.1's features)."""
    counts: dict[str, list[int]] = {}
    for t, frame in enumerate(clip.frames):
        seen: set[str] = set()
        for inst in sorted(frame, key=lambda i: (i.class_name, i.track_id)):
            if inst.class_name not in ("person", "player"):
                continue
            track = aliases[key(inst.class_name, inst.track_id)]
            if track in seen:
                continue
            seen.add(track)
            c = counts.setdefault(track, [0, 0])
            if start + t in fits:
                c[0] += 1
                c[1] += inside(foot(inst.mask()), fits[start + t])
    return {k: {"registered": r, "inside": i, "d": i / r if r else 0., "inside_most": (i / r if r else 0.) > RULES_D["inside_most"]}
            for k, (r, i) in counts.items()}


def choose_d(tracks: dict[str, dict[str, Any]], court: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    """(d), (c)+(d) and (a)+(d) picks over the eligible tracks (presence >= 0.5)."""
    eligible = sorted(k for k, v in tracks.items() if v["presence"] >= .5)
    d = {k: court.get(k, {"d": 0., "inside_most": False}) for k in eligible}
    return {
        "d": sorted([k for k in eligible if d[k]["inside_most"]], key=lambda k: (-d[k]["d"], k))[:2],
        "c+d": sorted(eligible, key=lambda k: (-int(d[k]["inside_most"]), -int(tracks[k]["racket_most"]), -d[k]["d"],
                                               -tracks[k]["c"], k))[:2],
        "a+d": sorted(eligible, key=lambda k: (-int(d[k]["inside_most"]), -tracks[k]["a"], k))[:2],
    }


def score(marks: dict[str, Any], review: dict[str, Any], review_path: Path, review_dir: Path,
          inputs: list[dict[str, Any]], regs: dict[str, dict[str, Any]], clip_sha: dict[str, str]) -> dict[str, Any]:
    labels = rules.validate_marks(marks, review, review_path)
    windows = load_windows(inputs)
    directories = {}
    for job in inputs:
        root = Path(job["directory"])
        for w in json.loads((root / "k4b-features.json").read_text())["windows"]:
            directories[w["id"]] = (root / "publish" / k4.safe(w["id"]), w)
    rows = []
    for frame in review["frames"]:
        clip_id = f"{frame['match']}/{frame['clip']}"
        wid = f"{clip_id}@{frame['frame']}"
        feature, masks = windows[wid]
        directory, w = directories[wid]
        if file_sha256(directory / "sam-masks.rle") != w["files"]["sam-masks.rle"]:
            raise ValueError(f"Changed window artifact: {directory}")
        reg = regs[clip_id]
        if reg["sha256"] != clip_sha[clip_id]:
            raise ValueError(f"K8b registered a different file for {clip_id}")
        court = court_features(ClipMasks.load(directory / "sam-masks.rle"), feature["aliases"], w["start"], reg["fits"])
        picks = {**feature["picks"], **choose_d(feature["features"], court)}
        by_track = {inst.track_id: k4.upsample(inst.mask(), (1920, 1080)) for inst in masks.frames[0]}
        if file_sha256(review_dir / frame["masks"]) != frame["masks_sha256"]:
            raise ValueError("Hand-marked review mask identity changed")
        by_id = {r["id"]: k4.upsample(r["mask"], (1920, 1080)) for r in numbered(ClipMasks.load(review_dir / frame["masks"]))}
        label = labels[frame["id"]]
        truth = [by_id[n] for n in label["selected"]]
        comparisons = {}
        for candidate in RULES_D["tie_order"]:
            chosen = picks[candidate]
            predicted = [by_track[feature["track_ids"][t]] for t in chosen if feature["track_ids"][t] in by_track]
            comparisons[candidate] = {"correct": rules.correct(predicted, truth, label["visible"], label["missing_instance"]),
                                      "picks": chosen}
        rows.append({"id": frame["id"], "window": wid, "match": frame["match"], "visible": label["visible"],
                     "court_view": frame["frame"] in reg["fits"], "court": court, "candidates": comparisons})
    tables = {}
    for name, subset in (("court_view", [r for r in rows if r["court_view"]]), ("all", rows)):
        table = []
        for candidate in RULES_D["tie_order"]:
            hits = sum(r["candidates"][candidate]["correct"] for r in subset)
            table.append({"candidate": candidate, "correct": hits, "frames": len(subset), "share": hits / len(subset),
                          "reaches_bar": hits / len(subset) >= rules.RULES["threshold"]})
        best = max(r["correct"] for r in table)
        tables[name] = {"table": table, "visibility": dict(Counter(r["visible"] for r in subset)),
                        "selected": next((r["candidate"] for r in table if r["correct"] == best and r["reaches_bar"]), None)}
    return {"rules_v1": rules.RULES, "rules_d": RULES_D, "tables": tables, "decides": "court_view", "rows": rows}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--marks", type=Path, required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--features-index", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True, help="K4b.1's clip manifest, for the clip identities")
    p.add_argument("--registration", type=Path, nargs="+", required=True, help="K8b reports covering the five matches")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    review = args.review / "instances.json"
    clip_sha = {c["id"]: c["sha256"] for c in json.loads(args.manifest.read_text())["clips"]}
    result = score(json.loads(args.marks.read_text()), json.loads(review.read_text()), review, args.review,
                   json.loads(args.features_index.read_text())["jobs"], registrations(args.registration), clip_sha)
    result["inputs"] = {str(p): file_sha256(p) for p in (args.marks, review, args.features_index, args.manifest, *args.registration)}
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "k4b-court-report.json", result)
    lines = []
    for name, t in result["tables"].items():
        lines += [f"{name} ({t['table'][0]['frames']} frames)", "| Candidate | Correct | Share |", "|---|---:|---:|"]
        lines += [f"| {r['candidate']} | {r['correct']} / {r['frames']} | {r['share']:.4f} |" for r in t["table"]] + [""]
    (args.out / "table.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print({k: v["selected"] for k, v in result["tables"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
