"""Freeze full-clip K4b feature inputs/specs without exposing human selections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.datasets.k4_inputs import GPU_MODELS, HOSTS, SHARING, entry, file_sha256, link


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--k4-spec", type=Path, required=True)
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--deadline", required=True)
    args = p.parse_args()
    args.dest.mkdir(parents=True, exist_ok=False)
    old = json.loads(args.selection.read_text())
    k4 = json.loads(args.k4_spec.read_text())
    weights = {r["name"]: r for r in k4["staged_inputs"]}
    clips = []
    for source in old["sources"]:
        match, clip = source["match"], source["clip"]
        if file_sha256(Path(source["path"])) != source["sha256"]:
            raise ValueError(f"Archived source changed: {source['path']}")
        path = link(Path(source["path"]), args.dest / "inputs" / f"{match}__{clip}.mp4")
        clips.append({**source, "id": f"{match}/{clip}", "staged": entry("clip", path, source["sha256"]),
                      "review_frames": sorted(r["frame"] for r in old["frames"] if r["match"] == match and r["clip"] == clip)})
    clips.sort(key=lambda c: (c["frames"], c["id"]))
    manifest = {"schema": 1, "selection_sha256": file_sha256(args.selection), "clips": clips,
                "contains_human_marks": False, "total_frames": sum(c["frames"] for c in clips),
                "pilot": [clips[i]["id"] for i in (0, len(clips)//2, len(clips)-1)]}
    target = args.dest / "manifest.json"
    target.write_text(json.dumps(manifest, indent=1) + "\n")
    specs = []
    for c in clips:
        full = min(2700, max(300, round(c["frames"] * .8 + 120)))
        spec = {"schema": 1, "entrypoint": ["-m", "experiments.tennis.k4b_features"], "environment": k4["environment"],
                "inputs": [{"path": str(target), "sha256": file_sha256(target)}],
                "staged_inputs": [entry("manifest", target), c["staged"], weights["sam"], weights["yolo11x_pose"]],
                "arguments": ["run", "--manifest", "{staged:manifest}", "--clip", c["id"], "--source", "{staged:clip}",
                              "--sam-checkpoint", "{staged:sam}", "--pose", "{staged:yolo11x_pose}", "--frames", "{frames}"],
                "scale": {"frames": {"smoke": min(48, c["frames"]), "full": 0}},
                "smoke": {"seconds": 600, "representative_basis": "First <=48 native frames of this same clip through K4 conversion, YOLO11x-pose and all three SAM prompts with native grounding batch 16 and CPU video offload; tracker state stays on GPU. A separate first-12-frame native batch-16 control is compared to video-offloaded batch 16 with IoU >=.999. Full-clip scalar features and target masks are exported; full uses all native frames."},
                "full": {"seconds": full}, "validator": ["{python}", "-m", "experiments.tennis.k4b_features", "validate"],
                "validator_seconds": 300, "budget_seconds": full + 900, "deadline": args.deadline, "stall_seconds": 1800,
                "device": "gpu", "hosts": HOSTS, "gpu_models": GPU_MODELS, "gpu_memory_mib": 24000,
                "cpu_threads": 8, "local_storage_gib": 40, "required_commands": [],
                "gpu_sharing": {**SHARING, "margin_mib": 4096}}
        name = c["id"].replace("/", "__") + ".json"
        (args.dest / name).write_text(json.dumps(spec, indent=1) + "\n")
        specs.append({"clip": c["id"], "spec": str(args.dest / name), "frames": c["frames"], "budget_seconds": spec["budget_seconds"]})
    print(json.dumps({"manifest_sha256": file_sha256(target), "pilot": manifest["pilot"], "specs": specs}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
