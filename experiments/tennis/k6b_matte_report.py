"""CPU report of validated complete GPU judging windows, with conservative stopping."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import tarfile
from typing import Any

import numpy as np

from experiments.tennis.k4 import safe
from experiments.tennis.k6b_matte_judge import assess, validate, control_error
from experiments.tennis.k6b_rate_budget import DIAGNOSTIC_SHA
from experiments.tennis.k6b_rate_transport import sha, write


def report(diagnostic: Path, transport: Path, matte: Path, archives: list[Path], references: list[Path]) -> dict[str, Any]:
    if sha(diagnostic.read_bytes()) != DIAGNOSTIC_SHA:
        raise ValueError("Original frozen report changed")
    reference_controls = {}
    for reference in references:
        digest = sha(reference.read_bytes())
        with tarfile.open(reference) as tar:
            for member in tar:
                if not member.name.endswith("/window.json"):
                    continue
                f = tar.extractfile(member)
                if f is None:
                    raise ValueError("Missing frozen original control")
                row = json.load(f)
                idx = row["variants"].index("gen-match5")
                f = tar.extractfile(member.name.replace("window.json", "scores.npz"))
                if f is None:
                    raise ValueError("Missing frozen original score arrays")
                with np.load(io.BytesIO(f.read())) as scores:
                    reference_controls[row["id"]] = (row, np.stack([scores[m][idx] for m in ("lpips", "dists")]), digest)
    measured = []
    inputs = []
    for archive in archives:
        validation = validate(archive, "full")
        if not validation["passed"]:
            raise ValueError("Incomplete/invalid GPU artifact")
        with tarfile.open(archive) as tar:
            f = tar.extractfile("publish/judge.json")
            if f is None:
                raise ValueError("Missing native GPU report")
            stage = json.load(f)
        reference_checks = {}
        with tarfile.open(archive) as tar:
            for row in stage["rows"]:
                old, expected_scores, digest = reference_controls[row["id"]]
                if digest not in stage["inputs"].values() or old["judged"] != row["judged"] or old["scoring_masks"] != row["scoring_masks"]:
                    raise ValueError("Independent original frame/mask/source identities disagree")
                f = tar.extractfile("publish/"+safe(row["id"])+"/scores.npz")
                if f is None:
                    raise ValueError("Missing new score arrays")
                with np.load(io.BytesIO(f.read())) as scores:
                    current = np.stack([scores[m][0, row["judged"]] for m in ("lpips", "dists")])
                reference_checks[row["id"]] = control_error(current, expected_scores[:, row["judged"]])
        measured.extend(stage["rows"])
        inputs.append({"path": str(archive), "sha256": sha(archive.read_bytes()),
                       "checks": validation["checks"], "independent_reference_max_abs_difference": reference_checks, "source_sha256": stage["code_sha256"],
                       "device": stage["device"], "input_sha256": stage["inputs"]})
    if len({r["id"] for r in measured}) != len(measured):
        raise ValueError("Duplicate window cannot increase the win count")
    original = json.loads(diagnostic.read_text())
    result = assess(original, json.loads(transport.read_text()), json.loads(matte.read_text()), measured)
    from experiments.background.g5 import envelope, pareto
    from experiments.tennis.k5 import bd_rate
    references_by_window = {r["window"]: r for r in original["rows"]}
    curves: dict[str, Any] = {}
    for row in measured:
        wid = row["id"]
        curves[wid] = {}
        for metric in ("lpips", "dists"):
            points = [(next(r for r in data["rows"] if r["window"] == wid)["kbps"], -row["means"][scale][metric]) for scale, data in result["scales"].items()]
            anchor = [(r["kbps"], -r[metric]) for r in references_by_window[wid]["anchor_points"].values()]
            front, hull = pareto(points), envelope(anchor)
            lo, hi = max(front[0][1], hull[0][1]), min(front[-1][1], hull[-1][1])
            curves[wid][metric] = {"bd_rate_fraction": bd_rate(points, anchor, 1e-12),
                                  "quality_overlap": [lo, hi] if hi > lo else None,
                                  "method_points": points, "anchor_envelope": hull,
                                  "scope": "three matte settings on this measured window only; PCHIP log-rate, no quality extrapolation"}
    result["prefix_bd_rate"] = curves
    return {**result, "inputs": inputs,
            "quality_status": "actual complete-window fixed-region LPIPS/DISTS; unmeasured windows are not scores",
            "accounting": "actual keypoints + all240-frame matte +7200s all-PNG match bank +7200s adapter/guider, shipped base primary",
            "limitations": ["same two familiar broadcasts; source/replay independence unresolved",
                            "exact-R target-derived Telea background still granted, native crop clipping/fallback",
                            "no new generations; JPEG/reference compression quality unmeasured",
                            "historical step5 failure and exploratory115px status unchanged"],
            "report_code_sha256": sha(Path(__file__).read_bytes())}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("diagnostic", "transport", "matte", "out"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--archive", type=Path, action="append", required=True)
    p.add_argument("--reference", type=Path, action="append", required=True)
    a = p.parse_args()
    result = report(a.diagnostic, a.transport, a.matte, a.archive, a.reference)
    write(a.out, result)
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["scale", "window", "match", "total_kbps", "metric", "actual_error", "primary_won", "base_charged_won"])
    for scale, data in result["scales"].items():
        for row in data["rows"]:
            for metric in ("lpips", "dists"):
                writer.writerow([scale, row["window"], row["match"], row["kbps"], metric, row["quality"][metric], row["primary"][metric]["won"], row["base_charged"][metric]["won"]])
    a.out.with_suffix(".csv").write_text(stream.getvalue())
    print(json.dumps({"windows": result["measured_windows"], "decision_made": result["decision_made"],
                      "scales": {s: {k: d[k] for k in ("known_wins", "unresolved", "optimistic_final_wins_bound", "can_reach_18", "passes")} for s, d in result["scales"].items()}, "report": str(a.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
