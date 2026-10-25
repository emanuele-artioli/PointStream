"""CPU report of validated complete GPU judging windows, with conservative stopping."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
import tarfile
from typing import Any

from experiments.tennis.k6b_matte_judge import assess, validate
from experiments.tennis.k6b_rate_budget import DIAGNOSTIC_SHA
from experiments.tennis.k6b_rate_transport import sha, write


def report(diagnostic: Path, transport: Path, matte: Path, archives: list[Path]) -> dict[str, Any]:
    if sha(diagnostic.read_bytes()) != DIAGNOSTIC_SHA:
        raise ValueError("Original frozen report changed")
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
        measured.extend(stage["rows"])
        inputs.append({"path": str(archive), "sha256": sha(archive.read_bytes()),
                       "checks": validation["checks"], "source_sha256": stage["code_sha256"],
                       "device": stage["device"], "input_sha256": stage["inputs"]})
    if len({r["id"] for r in measured}) != len(measured):
        raise ValueError("Duplicate window cannot increase the win count")
    result = assess(json.loads(diagnostic.read_text()), json.loads(transport.read_text()), json.loads(matte.read_text()), measured)
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
    a = p.parse_args()
    result = report(a.diagnostic, a.transport, a.matte, a.archive)
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
