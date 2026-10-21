"""Freeze K4b window feature jobs (K4b.1 amendment 1) without exposing human selections.

Reads Codex's frozen clip manifest (``features-v1/manifest.json``, no marks) and one
of its specs for the pinned environment and weights. Writes the window plan, the
pilot's specification and the remaining windows' part specifications. Parts are
filled longest window first into the part with the fewest frames (ties by id),
so the split reads only window lengths and ids.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.tennis.k4b_features import WINDOW, plan_windows
from tools.datasets.k4_inputs import GPU_MODELS, HOSTS, SHARING, file_sha256

#: One 240-frame window per source format: 2160p60, 2160p50, 1080p25 (amendment 1).
PILOT = ["djokovic_zverev/004@381", "djokovic_federer/069@305", "alcaraz_hurkacz/006@387"]
DATA_ROOT = "/home/itec/emanuele/Datasets/pointstream-data/tennis/k4b-2026-10-16"


def split(windows: list[dict[str, Any]], parts: int) -> list[list[dict[str, Any]]]:
    out: list[list[dict[str, Any]]] = [[] for _ in range(parts)]
    for w in sorted(windows, key=lambda w: (-w["frames"], w["id"])):
        min(out, key=lambda p: (sum(x["frames"] for x in p), out.index(p))).append(w)
    return out


def spec(windows: list[dict[str, Any]], clips: dict[str, dict[str, Any]], v1: dict[str, Any], manifest: str,
         manifest_sha: str, deadline: str, full_seconds: int) -> dict[str, Any]:
    """One part: smoke is its first (longest) window whole; full is every window."""
    names: dict[str, str] = {}
    for w in windows:
        names.setdefault(w["clip"], f"c{len(names)}")
    weights = {r["name"]: r for r in v1["staged_inputs"] if r["name"] in ("sam", "yolo11x_pose")}
    staged = [{"name": "manifest", "path": manifest, "sha256": manifest_sha}, weights["sam"], weights["yolo11x_pose"]]
    staged += [{"name": n, "path": clips[c]["staged"]["path"], "sha256": clips[c]["sha256"]} for c, n in names.items()]
    frames = sum(w["frames"] for w in windows)
    return {"schema": 1, "entrypoint": ["-m", "experiments.tennis.k4b_features"], "environment": v1["environment"],
            "inputs": [{"path": manifest, "sha256": manifest_sha}], "staged_inputs": staged,
            "arguments": ["run", "--manifest", "{staged:manifest}", "--sam-checkpoint", "{staged:sam}",
                          "--pose", "{staged:yolo11x_pose}", "--max-windows", "{windows}",
                          "--windows", *[w["id"] for w in windows],
                          "--sources", *[f"{{staged:{names[w['clip']]}}}" for w in windows]],
            "scale": {"windows": {"smoke": 1, "full": 0}},
            "smoke": {"seconds": 600, "representative_basis": (
                f"The part's first window whole ({windows[0]['frames']} frames, {windows[0]['id']}) through the full "
                "path: K4's conversion, YOLO11x-pose, native SAM 3.1 (three prompts, no offload, default grounding "
                f"batch) and the frozen features, so the smoke measures the real per-window peak. Full: {len(windows)} "
                f"windows, {frames} frames.")},
            "full": {"seconds": full_seconds},
            "validator": ["{python}", "-m", "experiments.tennis.k4b_features", "validate"], "validator_seconds": 300,
            "budget_seconds": 600 + full_seconds + 2 * 300 + 600, "deadline": deadline, "stall_seconds": 1200,
            "device": "gpu", "hosts": HOSTS, "gpu_models": GPU_MODELS, "gpu_memory_mib": 32000,
            "cpu_threads": 8, "local_storage_gib": 40, "required_commands": [],
            "gpu_sharing": {**SHARING, "margin_mib": 4096}}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True, help="local copy of features-v1/manifest.json")
    p.add_argument("--v1-spec", type=Path, required=True, help="one features-v1 spec, for environment and weights")
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--deadline", required=True)
    p.add_argument("--parts", type=int, default=0, help="split the non-pilot windows into this many parts")
    p.add_argument("--full-seconds", type=int, default=2700)
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text())
    plan = plan_windows(manifest)
    clips = {c["id"]: c for c in manifest["clips"]}
    v1 = json.loads(args.v1_spec.read_text())
    remote = f"{DATA_ROOT}/features-v1/manifest.json"
    sha = file_sha256(args.manifest)
    args.dest.mkdir(parents=True, exist_ok=True)
    (args.dest / "windows.json").write_text(json.dumps({"schema": 1, "length": WINDOW, "manifest": remote,
                                                         "manifest_sha256": sha, "pilot": PILOT,
                                                         "windows": list(plan.values())}, indent=1) + "\n")
    jobs = {"pilot": [plan[w] for w in PILOT]}
    rest = [w for k, w in plan.items() if k not in PILOT]
    if args.parts:
        jobs.update({f"part{n + 1}": part for n, part in enumerate(split(rest, args.parts))})
    for name, windows in jobs.items():
        windows = sorted(windows, key=lambda w: (-w["frames"], w["id"]))
        out = spec(windows, clips, v1, remote, sha, args.deadline, args.full_seconds)
        (args.dest / f"{name}.json").write_text(json.dumps(out, indent=1) + "\n")
        print(name, len(windows), sum(w["frames"] for w in windows), file_sha256(args.dest / f"{name}.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
