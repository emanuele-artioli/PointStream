"""Freeze K6's inputs (docs/experiments.md, 2026-10-21 K6).

``features``: K4's nine broadcast windows (the outro card left out, as K4) as a K4b.1 feature manifest, each clip
listed with frame 120 as its only review frame, which ``k4b_features.window_span`` places at frames 0-239 (K4's
window); and the feature job's specification, K4b.1's in everything but the windows. No labels are involved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments.tennis.k4b_features import WINDOW, plan_windows
from tools.datasets.k4_inputs import file_sha256

DATA_ROOT = "/home/itec/emanuele/Datasets/pointstream-data/tennis/k6-2026-10-21"
#: K4's outro card (`k4 report --exclude`): no tennis, no person.
K4_EXCLUDED = ("broadcast/medvedev_struff/042",)
#: The frame whose `window_span` is K4's window (frames 0-239) in a clip of at least 240 frames.
K4_FRAME = WINDOW // 2


def k4_manifest(windows: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    clips = []
    for w in windows["windows"]:
        if w["dataset"] != "broadcast" or w["id"] in K4_EXCLUDED:
            continue
        staged = inputs["files"][w["source"]["staged"]]
        match, clip = w["id"].split("/")[1:]
        clips.append({"match": match, "clip": clip, "id": f"{match}/{clip}", "path": staged["path"], "width": w["size"][0],
                      "height": w["size"][1], "fps": w["fps"], "frames": w["source_frames"], "sha256": staged["sha256"],
                      "staged": {"name": "clip", "path": staged["path"], "sha256": staged["sha256"]},
                      "review_frames": [K4_FRAME], "k4_window": w["id"]})
    manifest = {"schema": 1, "contains_human_marks": False, "purpose": "K6: K4's broadcast windows through K4b.1's features",
                "window_rule": "frame 120 of each clip, so window_span gives K4's first 240 frames", "clips": clips}
    for w in plan_windows(manifest).values():
        if (w["start"], w["frames"]) != (0, WINDOW):
            raise ValueError(f"{w['id']} is not K4's window")
    return manifest


def feature_spec(manifest: dict[str, Any], manifest_sha: str, base: dict[str, Any], deadline: str) -> dict[str, Any]:
    """K4b.1's feature job (`tools.datasets.k4b_feature_inputs.spec`) on K4's windows, one part; the manifest's
    identity is checked when the full stage is validated locally (`k4b_features validate --manifest`)."""
    from tools.datasets.k4b_feature_inputs import spec

    remote = f"{DATA_ROOT}/features/manifest.json"
    windows = sorted(plan_windows(manifest).values(), key=lambda w: (-w["frames"], w["id"]))
    clips = {c["id"]: c for c in manifest["clips"]}
    return spec(windows, clips, base, remote, manifest_sha, deadline, 2700)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("features")
    f.add_argument("--k4-windows", type=Path, required=True, help="K4's windows.json")
    f.add_argument("--k4-inputs", type=Path, required=True, help="K4's inputs-*.json")
    f.add_argument("--base-spec", type=Path, required=True, help="one K4b.1 feature spec (features-v2), for environment and weights")
    f.add_argument("--dest", type=Path, required=True)
    f.add_argument("--deadline", required=True)
    args = p.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    manifest = k4_manifest(json.loads(args.k4_windows.read_text()), json.loads(args.k4_inputs.read_text()))
    path = args.dest / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1) + "\n")
    sha = file_sha256(path)
    spec = feature_spec(manifest, sha, json.loads(args.base_spec.read_text()), args.deadline)
    (args.dest / "features-job.json").write_text(json.dumps(spec, indent=1) + "\n")
    print("manifest", sha, len(manifest["clips"]), "clips;", "spec", file_sha256(args.dest / "features-job.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
