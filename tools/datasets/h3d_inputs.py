"""Fleet specifications for PLAN step H3d's oracle (a content-aware super-resolution of the hand pixels).

    python tools/datasets/h3d_inputs.py --dest DIR --h3c-dir DIR --environment TAR SHA256 --deadline ISO [--checked]

Every staged input is H3c's (its ``h3c-anchor-<video>.json`` specs: evaluation set, dense archive, fill masks and
record, DCVC-UF's image and HT-L checkpoints, LPIPS, DISTS, H1's libaom inspect binary, the pilot videos). Writes
``h3d-oracle-<base>-<video>.json`` for each pilot video and base (dcvc, svt), each staging only its own video, and a
content-addressed ``inputs-<sha12>.json``. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PILOT = ("P01_107", "P02_02", "P03_10", "P09_106")  # B2's pilot videos
GPU_MODELS = ["RTX 6000 Ada", "RTX A6000"]  # DCVC-UF, LPIPS and DISTS verified there (docs/fleet.md)
HOSTS = ["gpu1", "gpu2", "gpu3", "gpu4", "gpu5", "gpu6"]
SHARING = {"mode": "shared", "basis": "DCVC-UF and SVT-AV1 streams, decodes, the seeded enhancer fits and their rate and "
           "LPIPS/PSNR/DISTS on the hand region do not depend on wall time; recorded seconds are not evidence"}
POINTS = {"dcvc": ["dcvc@0/4", "dcvc@21/4", "dcvc@42/4", "dcvc@0/2", "dcvc@21/2", "dcvc@42/2"],
          "svt": ["svt@55/4", "svt@63/4", "svt@55/8", "svt@63/8"]}  # experiments.visor.h3d.BASES
SMOKE_POINT = {"dcvc": "dcvc@42/4", "svt": "svt@63/8"}
SHARED = ("eval_set", "dense_archive", "fill_masks", "fill_record", "dcvc_image", "dcvc_video", "lpips", "dists", "inspect")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", required=True)
    parser.add_argument("--h3c-dir", required=True, help="pointstream-data/visor/h3c-2026-10-11")
    parser.add_argument("--environment", nargs=2, required=True, metavar=("TAR", "SHA256"))
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--steps", default="2000")
    parser.add_argument("--checked", action="store_true", help="H3c's inputs were re-hashed by an earlier run (skip it)")
    args = parser.parse_args()
    dest = Path(args.dest).resolve()
    specs = {v: json.loads((Path(args.h3c_dir) / f"h3c-anchor-{v}.json").read_text()) for v in PILOT}
    staged: dict[str, dict[str, Any]] = {}
    for spec in specs.values():
        for e in spec["staged_inputs"]:
            if e["name"] in SHARED or e["name"] in {f"video_{v}" for v in PILOT}:
                if e["name"] in staged and staged[e["name"]] != e:
                    raise SystemExit(f"H3c's anchor specs disagree on {e['name']}")
                staged[e["name"]] = e
    if missing := [k for k in [*SHARED, *(f"video_{v}" for v in PILOT)] if k not in staged]:
        raise SystemExit(f"not in H3c's anchor specs: {missing}")
    for e in ([] if args.checked else staged.values()):
        if file_sha256(Path(e["path"])) != e["sha256"]:
            raise SystemExit(f"{e['name']}: {e['path']} changed since H3c")
    items = {v: specs[v]["scale"]["items"]["full"] for v in PILOT}
    dest.mkdir(parents=True, exist_ok=True)
    # Content-addressed, so a later run never changes the manifest an earlier spec (maybe running) names.
    text = json.dumps({"files": {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in sorted(staged.items())},
                       "pilot_items": items}, indent=1, sort_keys=True) + "\n"
    inputs = dest / f"inputs-{hashlib.sha256(text.encode()).hexdigest()[:12]}.json"
    if not inputs.exists():
        inputs.write_text(text)
    written = []
    for base in POINTS:
        for video, item in items.items():
            arguments = ["oracle", "--eval-set", "{staged:eval_set}", "--archive", "{staged:dense_archive}",
                         "--masks", "{staged:fill_masks}", "--mask-record", "{staged:fill_record}", "--video", f"{{staged:video_{video}}}",
                         "--items", "{items}", "--frames", "{frames}", "--base", base, "--only", "{points}", "--steps", "{steps}",
                         "--image-ckpt", "{staged:dcvc_image}", "--video-ckpt", "{staged:dcvc_video}",
                         "--lpips-backbone", "{staged:lpips}", "--dists-backbone", "{staged:dists}", "--inspect", "{staged:inspect}"]
            basis = (f"H3d oracle ({base}): {item}, its first 48 window frames through the full path (window decode with the "
                     f"sparse-JPEG gate, fill masks checked against their record, {SMOKE_POINT[base]} coded with H3c's code and rate "
                     "split onto R, bicubic upscale, G5d's enhancer checked to be the identity at step 0 and fitted 50 steps with "
                     "MSE plus LPIPS on R, LPIPS/DISTS/PSNR and flicker before and after); the stage runs 240 frames, "
                     f"{args.steps} steps, at {', '.join(POINTS[base])}.")
            body = {
                "schema": 1, "entrypoint": ["-m", "experiments.visor.h3d"],
                "environment": {"path": args.environment[0], "sha256": args.environment[1]},
                "inputs": [{"path": str(inputs), "sha256": file_sha256(inputs)}], "arguments": arguments,
                "staged_inputs": [staged[k] for k in (*SHARED, f"video_{video}")],
                "scale": {"items": {"smoke": item, "full": item}, "frames": {"smoke": 48, "full": 240},
                          "points": {"smoke": SMOKE_POINT[base], "full": ",".join(POINTS[base])},
                          "steps": {"smoke": 50, "full": int(args.steps)}},
                "smoke": {"seconds": 600, "representative_basis": basis}, "full": {"seconds": 2700},
                "validator": ["{python}", "-m", "experiments.visor.h3d", "validate"], "validator_seconds": 300,
                "required_commands": [], "budget_seconds": 4500, "deadline": args.deadline, "stall_seconds": 1800,
                "cpu_threads": 16, "device": "gpu", "hosts": HOSTS, "gpu_models": GPU_MODELS, "gpu_memory_mib": 32000,
                "local_storage_gib": 60, "gpu_sharing": SHARING,
            }
            name = f"h3d-oracle-{base}-{video}.json"
            (dest / name).write_text(json.dumps(body, indent=1) + "\n")
            written.append(name)
    print(json.dumps({"inputs": str(inputs), "specs": written, "pilot_items": items}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
