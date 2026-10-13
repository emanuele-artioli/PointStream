"""Staged inputs and the fleet specification for H4a (grip pose on HOT3D; docs/experiments.md, 2026-10-13 H4a).

    python tools/datasets/h4a_inputs.py --dest DIR --hot3d DIR --environment TAR SHA256 --deadline ISO

HOT3D-Clips ``train_aria``, ``--per-participant`` clips per participant in content-blind order (lowest sha256 of
"pointstream-h4:hot3d:<clip>"), listed first one per participant (so a smoke of N clips spans N participants), then
the rest; hard-linked under ``DEST/inputs/hot3d`` (same NFS filesystem, no bytes copied). ``object_models_eval`` is
packed into ``DEST/inputs/object_models_eval.tar`` (sorted, zero mtimes, so its sha256 is reproducible). Writes the
content-addressed manifest ``DEST/inputs-<sha12>.json`` and the spec ``DEST/h4a.json``. Standard library only, so it
runs from a single copied file on a host that sees the data.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path
from typing import Any

HOSTS = ["gpu1", "gpu2", "gpu3", "gpu4", "gpu5", "gpu6"]
SEED = "pointstream-h4:hot3d"


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


def choose_clips(root: Path, per_participant: int) -> list[dict[str, Any]]:
    definitions = json.loads((root / "clip_definitions.json").read_text())
    by_participant: dict[str, list[str]] = {}
    for name in sorted(os.listdir(root / "train_aria")):
        if name.endswith(".tar"):
            participant = definitions[str(int(name[5:11]))]["sequence_id"].split("_")[0]
            by_participant.setdefault(participant, []).append(name)
    chosen = {p: sorted(names, key=lambda n: hashlib.sha256(f"{SEED}:{n}".encode()).hexdigest())[:per_participant]
              for p, names in sorted(by_participant.items())}
    order = [names[0] for names in chosen.values()] + [n for names in chosen.values() for n in names[1:]]
    participant_of = {n: p for p, names in chosen.items() for n in names}
    return [{"name": n, "participant": participant_of[n], "path": root / "train_aria" / n} for n in order]


def pack_models(source: Path, target: Path) -> Path:
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, "w", format=tarfile.USTAR_FORMAT) as tar:
        for path in sorted(source.iterdir()):
            data = path.read_bytes()
            info = tarfile.TarInfo(f"object_models_eval/{path.name}")
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", required=True)
    parser.add_argument("--hot3d", required=True)
    parser.add_argument("--per-participant", type=int, default=12)
    parser.add_argument("--environment", nargs=2, required=True, metavar=("TAR", "SHA256"))
    parser.add_argument("--deadline", required=True)
    args = parser.parse_args()
    dest = Path(args.dest)
    root = Path(args.hot3d)
    clips = choose_clips(root, args.per_participant)
    staged = []
    for clip in clips:
        path = link(clip["path"], dest / "inputs" / "hot3d" / clip["name"])
        staged.append({"name": f"hot3d_{clip['name'][5:11]}", "path": str(path), "sha256": file_sha256(path)})
    models = pack_models(root / "object_models_eval", dest / "inputs" / "object_models_eval.tar")
    models_entry = {"name": "object_models", "path": str(models), "sha256": file_sha256(models)}
    manifest = {"clips": [{"name": c["name"], "participant": c["participant"], "sha256": s["sha256"]}
                          for c, s in zip(clips, staged)],
                "object_models": models_entry, "seed": SEED, "per_participant": args.per_participant,
                "hot3d_root": str(root)}
    body = json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    manifest_path = dest / f"inputs-{hashlib.sha256(body.encode()).hexdigest()[:12]}.json"
    manifest_path.write_text(body)
    participants = len({c["participant"] for c in clips})
    spec = {
        "schema": 1, "entrypoint": ["-m", "experiments.visor.h4a"],
        "environment": {"path": args.environment[0], "sha256": args.environment[1]},
        "inputs": [{"path": str(manifest_path), "sha256": file_sha256(manifest_path)}],
        "arguments": ["run", "--models", "{staged:object_models}",
                      *[a for s in staged for a in ("--clip", f"{{staged:{s['name']}}}")],
                      "--clips", "{clips}", "--workers", "14"],
        "staged_inputs": [models_entry, *staged],
        "scale": {"clips": {"smoke": 2, "full": 0}},
        "smoke": {"seconds": 600, "representative_basis":
                  f"The first two clips (two of {participants} participants) through the full path: the tar read in one "
                  "pass, the eval meshes against models_info, the toolkit's fisheye projection, every coder at every "
                  "threshold, the grip spread and the noise floor; the stage runs all "
                  f"{len(clips)} clips ({args.per_participant} per participant)."},
        "full": {"seconds": 2700},
        "validator": ["{python}", "-m", "experiments.visor.h4a", "validate"], "validator_seconds": 300,
        "required_commands": [], "budget_seconds": 3600, "deadline": args.deadline, "stall_seconds": 900,
        "cpu_threads": 16, "device": "cpu", "hosts": HOSTS, "gpu_models": [], "gpu_memory_mib": 0,
        "local_storage_gib": 20,
    }
    (dest / "h4a.json").write_text(json.dumps(spec, indent=1) + "\n")
    print(json.dumps({"clips": len(clips), "participants": participants, "manifest": str(manifest_path),
                      "spec": str(dest / "h4a.json")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
