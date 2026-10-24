"""Frozen clip-local K6b diagnostic: input gate and feature-stage validation.

No quality selection, fitting or source recovery. Later stages must also pass
native K6/generator/judge validators; this gate cannot certify independence.
"""
from __future__ import annotations

import argparse
import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

FROZEN = (('alcaraz_hurkacz/031', 9, 25, '054e4535638d8dfa3dee69258f45c231aaeab2241b91309d449be7b0934cef76'), ('alcaraz_hurkacz/013', 93, 25, '1af169200b28659c503499979863b6219b0d331f532962ac69fb505d13c4976c'), ('alcaraz_hurkacz/023', 108, 25, '88345a9deac75a4b6180daa56a2ec7cda2afea25babb5b549b937105bca45d1a'), ('alcaraz_hurkacz/016', 153, 25, '305a369d74882a628ff7efd6b49cf3f75c56a985d78ffd798863542df20551bd'), ('alcaraz_hurkacz/008', 141, 25, '355e4b0a3408e4c8f049baadf749c937e52930f57df5018867e5baa0cdd49e54'), ('alcaraz_hurkacz/047', 3, 25, '7bb3a53e3b416d3b84c0344c99b29d1374f8c49b54f970034f4f29a95609e99b'), ('medvedev_struff/014', 74, 50, 'a2898b5cdfb463f59b5084fddfe23a2442cf2ba5feaaddb9d5414873e52dd0be'), ('medvedev_struff/037', 229, 50, '97372d45ba53068078fdc85300a550221b2c4bd31a3074958b52081326e3feeb'), ('medvedev_struff/002', 218, 50, '49cc09d072d740bd4d5ed02fdd24f2b3edbdc5fb16a660f66f1e8148ff8be14a'), ('medvedev_struff/017', 86, 50, '64cc4bc0a0712b5daa6bc00c5ed592b31a0f8333cb36a4caf5b05d24f9e39873'), ('medvedev_struff/029', 18, 50, '013979d403ec9ae3e48ce3dc49bbf261b37fb6a3dd78af71d5b8d81419612e8f'), ('medvedev_struff/021', 126, 50, 'fa2da44fdc64d6d95c9a111953dd6d16a088cb534dc7336a98d3cb2557d30b4a'))
PROVENANCE = {"clip_audit": "complete", "original_source_gate": "unresolved",
              "claim_scope": "clip_local_diagnostic"}


def gate(rows: list[Any]) -> list[int]:
    """Relative frame indices passing the frozen K8b fits/finite-H rule."""
    if len(rows) != 240:
        raise ValueError("Registration must have exactly 240 rows, including missing rows")
    valid = []
    for t, r in enumerate(rows):
        if not r or r.get("fits") is not True:
            continue
        try:
            h = np.asarray(r.get("homography"), dtype=float)
            if h.shape == (3, 3) and np.isfinite(h).all() and np.linalg.matrix_rank(h) == 3:
                valid.append(t)
        except (ValueError, TypeError, np.linalg.LinAlgError):
            continue
    return valid


def check_manifest(manifest: dict[str, Any]) -> dict[str, list[int]]:
    """Reject changed cohort, overlap claims and mismatched per-clip placement."""
    from experiments.tennis.k4b_features import plan_windows

    if manifest.get("provenance") != PROVENANCE or manifest.get("budget_gpu_hours") != 8:
        raise ValueError("Wrong diagnostic scope or budget")
    clips = manifest["clips"]
    if [c["id"] for c in clips] != [r[0] for r in FROZEN]:
        raise ValueError("Frozen cohort changed or duplicated")
    old = manifest["old_clips"]
    if len(old) != 49 or set(old) & {r[0] for r in FROZEN}:
        raise ValueError("Old-manifest clip intersection or incomplete audit")
    plan = plan_windows(manifest)
    gates = {}
    for c, (cid, start, fps, digest) in zip(clips, FROZEN):
        wid = f"{cid}@{start + 120}"
        w = plan.get(wid, {})
        if (w.get("start"), w.get("frames"), w.get("target"), w.get("fps"), w.get("sha256")) != (start, 240, 120, fps, digest):
            raise ValueError(f"Changed source/span/fps: {cid}")
        if c["review_frames"] != [start + 120] or (c["frames"] - 240) // 2 != start:
            raise ValueError(f"Wrong centered frame placement: {cid}")
        if c["staged"]["sha256"] != digest:
            raise ValueError(f"Staged source mismatch: {cid}")
        gates[wid] = gate(c["registration"])
        # All 12 frozen windows passed in the original pre-quality audit.
        if len(gates[wid]) < 228:
            raise ValueError(f"Changed registration eligibility: {cid}")
    if len(plan) != 12:
        raise ValueError("Extra windows")
    return gates


def check_feature_rows(manifest: dict[str, Any], summary: dict[str, Any]) -> None:
    from experiments.tennis.k4b_features import plan_windows

    check_manifest(manifest)
    plan = plan_windows(manifest)
    requested = summary["requested"][:summary["configured_windows"] or None]
    if not requested or len(set(requested)) != len(requested) or any(w not in plan for w in requested):
        raise ValueError("Invalid requested windows")
    if [r["id"] for r in summary["windows"]] != requested:
        raise ValueError("Missing, duplicated or additional feature output")
    for row in summary["windows"]:
        if any(row.get(k) != v for k, v in plan[row["id"]].items()):
            raise ValueError("Feature/source span mismatch")
        if row["conversion"]["frames"] != 240:
            raise ValueError("Incomplete source decode")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("features", "validate-features", "preflight"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--sam-checkpoint")
    parser.add_argument("--pose")
    parser.add_argument("--max-windows", type=int, default=0)
    parser.add_argument("--windows", nargs="+")
    parser.add_argument("--sources", nargs="+")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    gates = check_manifest(manifest)
    if args.command == "preflight":
        print(json.dumps({"windows": len(gates), "fits": {k: len(v) for k, v in gates.items()}, **PROVENANCE}))
        return 0
    from experiments.tennis import k4b_features
    from experiments.visor.b1 import stage_dir, write_json

    if args.command == "features":
        if not args.windows or not args.sources or not args.sam_checkpoint or not args.pose:
            raise ValueError("Missing feature arguments")
        if len(set(args.windows)) != len(args.windows) or any(w not in gates for w in args.windows):
            raise ValueError("Unknown/duplicate window before GPU loading")
        return k4b_features.run(args)
    stage = stage_dir()
    summary = json.loads((stage / "k4b-features.json").read_text())
    check_feature_rows(manifest, summary)
    with tempfile.TemporaryDirectory(prefix="k6b-confirm-validate-", dir="/tmp") as temp:
        with tarfile.open(stage / "published.tar") as archive:
            archive.extractall(temp, filter="data")
        report = k4b_features.validate(Path(temp) / "publish", summary, args.manifest)
    report["checks"]["frozen_cohort_and_geometry"] = True
    report["provenance"] = PROVENANCE
    if target := os.environ.get("PS_VALIDATION_PATH"):
        write_json(Path(target), report)
    print(json.dumps(report))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
