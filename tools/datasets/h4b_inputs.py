"""Staged inputs and fleet specifications for H4b (a bank of reference views; docs/experiments.md, 2026-10-13 H4b).

    python tools/datasets/h4b_inputs.py --dest DIR --h1-dir DIR --h1-job JOBDIR ... --deadline ISO [--parts N]

The VISOR inputs (evaluation set, the dense archive, the fill masks and their record, the videos) and the
environment are exactly H1's, read from H1's specs in ``--h1-dir``, so ``single`` can reproduce H1's chains.
``h1-chains.json`` collects H1's recorded chain summaries per item and track from the H1 jobs' published archives
(``published.tar``, or ``partial.tar`` where only that exists). Writes the content-addressed manifest
``DEST/inputs-<sha12>.json`` and the specs ``h4b-pilot.json`` (B2's four pilot items; smoke: the first, 48 frames)
and, with ``--parts N``, ``h4b-full-<k>.json`` (the other 30 items in N parts). Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

HOSTS = ["gpu1", "gpu2", "gpu3", "gpu4", "gpu5", "gpu6"]
PILOT = ("P01_107_0000003049", "P09_106_0000006303", "P02_02_0000006080", "P03_10_0000000616")  # H1's pilot items
VISOR = ("eval_set", "dense_archive", "fill_masks", "fill_record")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def h1_chains(jobs: list[Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    items: dict[str, Any] = {}
    sources = []
    for job in jobs:
        tar_path = job / "full" / "published.tar"
        if not tar_path.exists():
            tar_path = job / "full" / "partial.tar"
        sources.append({"job": job.name, "archive": str(tar_path), "sha256": file_sha256(tar_path)})
        with tarfile.open(tar_path) as tar:
            for member in tar.getmembers():
                if member.name.startswith("publish/items/") and member.name.endswith(".json"):
                    handle = tar.extractfile(member)
                    assert handle is not None
                    row = json.loads(handle.read())
                    if row["id"] in items:
                        raise SystemExit(f"{row['id']} appears in two H1 archives")
                    items[row["id"]] = {key: {th: {"held": c["held"], "references": c["references"], "frames": c["frames"]}
                                              for th, c in info["chains"].items()}
                                        for key, info in row["tracks"].items()
                                        if info["kind"] in ("handled_object", "other_object")}
    return items, sources


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", required=True)
    parser.add_argument("--h1-dir", required=True)
    parser.add_argument("--h1-job", action="append", required=True)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--parts", type=int, default=0)
    args = parser.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    h1_dir = Path(args.h1_dir)
    staged: dict[str, dict[str, Any]] = {}
    environment = None
    for spec_name in ("h1-pilot.json", "h1-full-a.json", "h1-full-b.json"):
        spec = json.loads((h1_dir / spec_name).read_text())
        environment = environment or spec["environment"]
        if spec["environment"] != environment:
            raise SystemExit("H1's specs name different environments")
        for entry in spec["staged_inputs"]:
            if entry["name"] in VISOR or entry["name"].startswith("video_"):
                staged[entry["name"]] = entry
    eval_set = json.loads(Path(staged["eval_set"]["path"]).read_text())
    ids = [it["id"] for it in eval_set["items"]]
    if not set(PILOT) <= set(ids):
        raise SystemExit("pilot items missing from the evaluation set")
    chains, sources = h1_chains([Path(j) for j in args.h1_job])
    missing = [i for i in ids if i not in chains]
    if missing:
        raise SystemExit(f"H1 chains missing for {missing}")
    chains_path = dest / "h1-chains.json"
    chains_path.write_text(json.dumps({"sources": sources, "items": chains}, indent=1, sort_keys=True) + "\n")
    chains_entry = {"name": "h1_chains", "path": str(chains_path), "sha256": file_sha256(chains_path)}
    manifest = {"visor_from_h1": {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in sorted(staged.items())},
                "h1_chains": chains_entry, "h1_sources": sources, "environment": environment, "pilot": list(PILOT)}
    body = json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    manifest_path = dest / f"inputs-{hashlib.sha256(body.encode()).hexdigest()[:12]}.json"
    manifest_path.write_text(body)
    by_id = {it["id"]: it for it in eval_set["items"]}

    def spec(items: list[str], smoke_item: str, full_seconds: int, basis: str) -> dict[str, Any]:
        videos = sorted({by_id[i]["video"] for i in items})
        return {
            "schema": 1, "entrypoint": ["-m", "experiments.visor.h4b"], "environment": environment,
            "inputs": [{"path": str(manifest_path), "sha256": file_sha256(manifest_path)}],
            "arguments": ["run", "--eval-set", "{staged:eval_set}", "--archive", "{staged:dense_archive}",
                          "--masks", "{staged:fill_masks}", "--mask-record", "{staged:fill_record}",
                          "--h1-chains", "{staged:h1_chains}",
                          *[a for v in videos for a in ("--video", f"{{staged:video_{v}}}")],
                          "--items", "{items}", "--frames", "{frames}", "--workers", "8"],
            "staged_inputs": [staged[k] for k in VISOR] + [chains_entry] + [staged[f"video_{v}"] for v in videos],
            "scale": {"items": {"smoke": smoke_item, "full": ",".join(items)}, "frames": {"smoke": 48, "full": 240}},
            "smoke": {"seconds": 600, "representative_basis": basis}, "full": {"seconds": full_seconds},
            "validator": ["{python}", "-m", "experiments.visor.h4b", "validate"],
            "validator_seconds": 300, "required_commands": [], "budget_seconds": full_seconds + 900,
            "deadline": args.deadline, "stall_seconds": 1200, "cpu_threads": 16, "device": "cpu", "hosts": HOSTS,
            "gpu_models": [], "gpu_memory_mib": 0, "local_storage_gib": 60,
        }

    basis = ("{item}: its first 48 window frames through the full path (window decode with the sparse-JPEG gate, fill "
             "masks checked against their record, H1's contacts and object tracks, flow and homography fits, all four "
             "coders at both thresholds with SIFT registration); the stage runs {what} at 240 frames, where single is "
             "checked against H1's recorded chains.")
    specs = {"h4b-pilot": spec(list(PILOT), PILOT[0], 2700, basis.format(item=PILOT[0], what="B2's four pilot items"))}
    rest = [i for i in ids if i not in PILOT]
    for k in range(args.parts):
        part = rest[k::args.parts]
        specs[f"h4b-full-{k + 1}"] = spec(part, part[0], 2700, basis.format(item=part[0], what=f"part {k + 1} ({len(part)} items)"))
    for name, body_spec in specs.items():
        (dest / f"{name}.json").write_text(json.dumps(body_spec, indent=1) + "\n")
    print(json.dumps({"manifest": str(manifest_path), "specs": sorted(specs), "h1_items": len(chains)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
