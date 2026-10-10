"""Staged inputs and fleet specifications for PLAN step H2 (hand-pose estimators and pose coding).

    python tools/datasets/h2_inputs.py --dest DIR --h1-dir DIR --h1-job JOBDIR ... --hint ZIP \\
        --audit-inputs DIR --hot3d DIR --environment TAR SHA256 --deadline ISO

Under ``DEST/inputs`` it hard-links (same NFS filesystem, no bytes copied) HInt's archive, HaMeR's
weights from the environment audit's links, WiLoR's and MANO's from H1's inputs, and the chosen
HOT3D clips; it writes ``h1-boxes.json`` (each VISOR hand-frame H1 fitted, with WiLoR's detector box,
read from H1's recorded job archives) and ``inputs.json`` (sha256 of everything). Then the six specs
``h2-{hint,hot3d,visor}-{pilot,full}.json`` beside them; the VISOR inputs (evaluation set, masks,
videos) are H1's staged inputs, read from H1's specs.

HOT3D clips: train_aria, ``--per-participant`` per participant in content-blind order (lowest
sha256 of "pointstream-h2:hot3d:<clip>"), listed first one per participant (the pilot), then the rest.
Self-contained (standard library only), so it runs from a single copied file on a host that sees the data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from pathlib import Path
from typing import Any


PILOT_ITEMS = ("P01_107", "P09_106", "P02_02", "P03_10")  # B2's pilot videos
GPU_MODELS = ["RTX 6000 Ada", "RTX A6000", "Quadro RTX 8000", "Quadro GV100"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def link(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if os.path.samefile(source, target):
            return target
        raise SystemExit(f"{target} exists and is not {source}")
    os.link(source, target)
    return target


def hot3d_clips(root: Path, per_participant: int) -> list[dict[str, Any]]:
    definitions = json.loads((root / "clip_definitions.json").read_text())
    by_participant: dict[str, list[str]] = {}
    for name in sorted(os.listdir(root / "train_aria")):
        if not name.endswith(".tar"):
            continue
        clip_id = str(int(name[5:11]))
        participant = definitions[clip_id]["sequence_id"].split("_")[0]
        by_participant.setdefault(participant, []).append(name)
    chosen = {p: sorted(names, key=lambda n: hashlib.sha256(f"pointstream-h2:hot3d:{n}".encode()).hexdigest())[:per_participant]
              for p, names in sorted(by_participant.items())}
    order = [names[0] for names in chosen.values()] + [n for names in chosen.values() for n in names[1:]]
    participant_of = {n: p for p, names in chosen.items() for n in names}
    return [{"name": n, "participant": participant_of[n], "path": root / "train_aria" / n} for n in order]


def h1_boxes(jobs: list[Path]) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    items: dict[str, list[dict[str, Any]]] = {}
    sources = []
    for job in jobs:
        for stage_tar in (job / "full" / "published.tar", job / "full" / "partial.tar"):
            if stage_tar.exists():
                break
        else:
            raise SystemExit(f"{job}: no published.tar or partial.tar")
        sources.append({"job": job.name, "archive": str(stage_tar), "sha256": file_sha256(stage_tar)})
        with tarfile.open(stage_tar) as tar:
            for member in tar.getmembers():
                if not (member.name.startswith("publish/items/") and member.name.endswith(".json")):
                    continue
                handle = tar.extractfile(member)
                assert handle is not None
                row = json.loads(handle.read())
                if row["id"] in items:
                    continue
                items[row["id"]] = [{"t": h["t"], "side": h["side"], "box": h["box"]} for h in row["hands"] if h["fitted"]]
    return items, sources


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", required=True)
    parser.add_argument("--h1-dir", required=True, help="pointstream-data/visor/h1-2026-10-08 (inputs/ and the specs)")
    parser.add_argument("--h1-job", action="append", required=True, help="an H1 job directory")
    parser.add_argument("--hint", required=True)
    parser.add_argument("--audit-inputs", required=True, help="pointstream-data/audit/env-2026-10-06/inputs")
    parser.add_argument("--hot3d", required=True)
    parser.add_argument("--per-participant", type=int, default=6)
    parser.add_argument("--environment", nargs=2, required=True, metavar=("TAR", "SHA256"))
    parser.add_argument("--deadline", required=True)
    args = parser.parse_args()
    dest = Path(args.dest)
    inputs = dest / "inputs"
    h1_dir = Path(args.h1_dir)
    audit = Path(args.audit_inputs)
    files: dict[str, Path] = {
        "hint_zip": link(Path(args.hint), inputs / "hint" / Path(args.hint).name),
        "hamer_checkpoint": link(audit / "hamer_checkpoint" / "hamer.ckpt", inputs / "hamer" / "hamer.ckpt"),
        "hamer_config": link(audit / "hamer_config" / "model_config.yaml", inputs / "hamer" / "model_config.yaml"),
        "hamer_mean_params": link(audit / "hamer_mean_params" / "mano_mean_params.npz", inputs / "hamer" / "mano_mean_params.npz"),
        "wilor_checkpoint": link(h1_dir / "inputs" / "wilor" / "wilor_final.ckpt", inputs / "wilor" / "wilor_final.ckpt"),
        "wilor_detector": link(h1_dir / "inputs" / "wilor" / "detector.pt", inputs / "wilor" / "detector.pt"),
        "mano_left": link(h1_dir / "inputs" / "mano" / "MANO_LEFT.pkl", inputs / "mano" / "MANO_LEFT.pkl"),
        "mano_right": link(h1_dir / "inputs" / "mano" / "MANO_RIGHT.pkl", inputs / "mano" / "MANO_RIGHT.pkl"),
    }
    clips = hot3d_clips(Path(args.hot3d), args.per_participant)
    for clip in clips:
        files[f"hot3d_{clip['name'][:-4]}"] = link(clip["path"], inputs / "hot3d" / clip["name"])
    boxes, sources = h1_boxes([Path(j) for j in args.h1_job])
    boxes_path = inputs / "h1-boxes.json"
    boxes_path.write_text(json.dumps({"sources": sources, "items": boxes}, indent=1, sort_keys=True) + "\n")
    files["h1_boxes"] = boxes_path
    staged = {name: {"name": name, "path": str(path), "sha256": file_sha256(path)} for name, path in files.items()}
    # VISOR inputs: exactly what H1 staged.
    visor_staged: dict[str, dict[str, Any]] = {}
    for spec_name in ("h1-pilot.json", "h1-full-a.json", "h1-full-b.json"):
        spec = json.loads((h1_dir / spec_name).read_text())
        for entry in spec["staged_inputs"]:
            if entry["name"] in ("eval_set", "dense_archive", "fill_masks", "fill_record") or entry["name"].startswith("video_"):
                visor_staged[entry["name"]] = entry
    eval_set = json.loads(Path(visor_staged["eval_set"]["path"]).read_text())
    ids = [it["id"] for it in eval_set["items"]]
    missing = [i for i in ids if i not in boxes]
    if missing:
        raise SystemExit(f"H1 boxes missing for {missing}")
    pilot_ids = [i for i in ids if any(i.startswith(v + "_0") for v in PILOT_ITEMS)]
    rest_ids = [i for i in ids if i not in pilot_ids]
    manifest = {"files": {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in staged.items()},
                "visor_from_h1": {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in visor_staged.items()},
                "hot3d_clips": [{"name": c["name"], "participant": c["participant"],
                                 "sha256": staged[f"hot3d_{c['name'][:-4]}"]["sha256"]} for c in clips],
                "pilot_items": pilot_ids, "h1_boxes_sources": sources}
    manifest_path = dest / "inputs.json"
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    environment = {"path": args.environment[0], "sha256": args.environment[1]}
    models = ["--hamer-checkpoint", "{staged:hamer_checkpoint}", "--hamer-config", "{staged:hamer_config}",
              "--hamer-mean-params", "{staged:hamer_mean_params}", "--wilor-checkpoint", "{staged:wilor_checkpoint}",
              "--wilor-detector", "{staged:wilor_detector}", "--mano-left", "{staged:mano_left}", "--mano-right", "{staged:mano_right}"]
    model_staged = [staged[k] for k in ("hamer_checkpoint", "hamer_config", "hamer_mean_params", "wilor_checkpoint",
                                        "wilor_detector", "mano_left", "mano_right")]

    def spec(part: str, stage: str, arguments: list[str], staged_inputs: list[dict[str, Any]], scale: dict[str, Any],
             basis: str, full_seconds: int, gpu_models: list[str], storage: int) -> dict[str, Any]:
        return {
            "schema": 1, "entrypoint": ["-m", "experiments.visor.h2"], "environment": environment,
            "inputs": [{"path": str(manifest_path), "sha256": file_sha256(manifest_path)}],
            "arguments": ["run", "--part", part, *models, *arguments], "staged_inputs": model_staged + staged_inputs,
            "scale": scale, "smoke": {"seconds": 600, "representative_basis": basis}, "full": {"seconds": full_seconds},
            "validator": ["{python}", "-m", "experiments.visor.h2", "validate"], "validator_seconds": 300,
            "required_commands": [], "budget_seconds": full_seconds + 1500, "deadline": args.deadline,
            "stall_seconds": 1800, "cpu_threads": 16, "device": "gpu", "hosts": ["gpu1", "gpu2", "gpu3", "gpu4", "gpu5", "gpu6"],
            "gpu_models": gpu_models, "gpu_memory_mib": 10000, "local_storage_gib": storage,
            "contention": {"policy": "pause", "pause_seconds": 900, "resume_attempts": 2},
        }

    specs = {}
    hint_args = ["--hint-zip", "{staged:hint_zip}", "--limit", "{limit}", "--speed"]
    specs["hint-pilot"] = spec("hint", "pilot", hint_args, [staged["hint_zip"]], {"limit": {"smoke": 10, "full": 50}},
                               "HInt: the first 10 hands per test split in content-blind order through both regressors, distances, and the speed runs; the stage runs 50 per split.",
                               1800, ["RTX A6000"], 30)
    specs["hint-full"] = spec("hint", "full", hint_args, [staged["hint_zip"]], {"limit": {"smoke": 10, "full": 0}},
                              "HInt: the first 10 hands per test split in content-blind order through both regressors, distances, and the speed runs; the stage runs every test hand.",
                              3600, ["RTX A6000"], 30)
    clip_args: list[str] = []
    clip_staged = []
    for clip in clips:
        name = f"hot3d_{clip['name'][:-4]}"
        entry = dict(staged[name], name=f"hot3d_{clip['name'][5:11]}")  # staged names are identifiers
        clip_args += ["--hot3d-clip", f"{{staged:{entry['name']}}}"]
        clip_staged.append(entry)
    participants = len({c["participant"] for c in clips})
    specs["hot3d-pilot"] = spec("hot3d", "pilot", [*clip_args, "--clips", "{clips}", "--frames", "{frames}"], clip_staged,
                                {"clips": {"smoke": 1, "full": participants}, "frames": {"smoke": 30, "full": 150}},
                                "HOT3D: the first clip's first 30 frames (crops warped from the fisheye stream, ground-truth MANO, both regressors, frame changes checked against numpy MANO); the stage runs one clip per participant at 150 frames.",
                                3600, GPU_MODELS, 40)
    specs["hot3d-full"] = spec("hot3d", "full", [*clip_args, "--clips", "{clips}", "--frames", "{frames}"], clip_staged,
                               {"clips": {"smoke": 1, "full": 0}, "frames": {"smoke": 30, "full": 150}},
                               "HOT3D: the first clip's first 30 frames through the full path; the stage runs every chosen clip at 150 frames.",
                               7200, GPU_MODELS, 40)
    videos = sorted(k for k in visor_staged if k.startswith("video_"))
    visor_args = ["--eval-set", "{staged:eval_set}", "--archive", "{staged:dense_archive}", "--masks", "{staged:fill_masks}",
                  "--mask-record", "{staged:fill_record}", "--h1-boxes", "{staged:h1_boxes}",
                  *[a for v in videos for a in ("--video", f"{{staged:{v}}}")], "--items", "{items}", "--frames", "{frames}"]
    visor_inputs = [visor_staged[k] for k in ("eval_set", "dense_archive", "fill_masks", "fill_record")] + \
        [staged["h1_boxes"]] + [visor_staged[v] for v in videos]
    specs["visor-pilot"] = spec("visor", "pilot", visor_args, visor_inputs,
                                {"items": {"smoke": pilot_ids[0], "full": ",".join(pilot_ids)}, "frames": {"smoke": 48, "full": 240}},
                                f"VISOR: {pilot_ids[0]}, its first 48 window frames, through the full path (window decode with the sparse-JPEG gate, fill masks checked against their record, H1's boxes, both regressors, silhouettes and wrist split, numpy MANO check); the stage runs B2's four pilot items at 240 frames.",
                                3600, GPU_MODELS, 60)
    specs["visor-full"] = spec("visor", "full", visor_args, visor_inputs,
                               {"items": {"smoke": rest_ids[0], "full": ",".join(rest_ids)}, "frames": {"smoke": 48, "full": 240}},
                               f"VISOR: {rest_ids[0]}, its first 48 window frames, through the full path; the stage runs the other 30 items at 240 frames.",
                               7200, GPU_MODELS, 60)
    for name, body in specs.items():
        (dest / f"h2-{name}.json").write_text(json.dumps(body, indent=1) + "\n")
    print(json.dumps({"inputs": str(manifest_path), "specs": sorted(specs), "hot3d_clips": len(clips),
                      "participants": participants, "pilot_items": pilot_ids, "rest_items": len(rest_ids)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
