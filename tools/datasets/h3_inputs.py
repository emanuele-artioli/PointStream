"""Staged inputs and fleet specifications for PLAN step H3a (the texture-transfer oracle).

    python tools/datasets/h3_inputs.py --dest DIR --h1-dir DIR --b2-inputs DIR --environment TAR SHA256 \\
        --deadline ISO [--parts N]

Under ``DEST/inputs`` it hard-links B2's LPIPS backbone, and writes ``inputs.json`` (sha256 of
everything). The VISOR inputs (evaluation set, masks, videos) and B2's SVT-AV1 streams with their
records are H1's staged inputs, read from H1's specs. Then the specs beside them:
``h3-oracle-pilot.json`` (B2's four pilot items) and ``h3-oracle-full-<k>.json`` (the other 30 items in
``--parts`` shares, each a job of at most 45 minutes). Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

PILOT_ITEMS = ("P01_107", "P09_106", "P02_02", "P03_10")  # B2's pilot videos
GPU_MODELS = ["RTX 6000 Ada", "RTX A6000"]  # where B2 ran LPIPS on the GPU


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", required=True)
    parser.add_argument("--h1-dir", required=True, help="pointstream-data/visor/h1-2026-10-08 (the specs)")
    parser.add_argument("--b2-inputs", required=True, help="pointstream-data/visor/b2-2026-10-07/inputs")
    parser.add_argument("--environment", nargs=2, required=True, metavar=("TAR", "SHA256"))
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--parts", type=int, default=3)
    args = parser.parse_args()
    dest = Path(args.dest)
    lpips = link(Path(args.b2_inputs) / "lpips" / "alexnet-owt-7be5be79.pth", dest / "inputs" / "lpips" / "alexnet-owt-7be5be79.pth")
    staged: dict[str, dict[str, Any]] = {"lpips": {"name": "lpips", "path": str(lpips), "sha256": file_sha256(lpips)}}
    wanted = ("eval_set", "dense_archive", "fill_masks", "fill_record", "svt_streams_a", "svt_result_a", "svt_streams_b", "svt_result_b")
    for spec_name in ("h1-pilot.json", "h1-full-a.json", "h1-full-b.json"):
        for entry in json.loads((Path(args.h1_dir) / spec_name).read_text())["staged_inputs"]:
            if entry["name"] in wanted or entry["name"].startswith("video_"):
                staged.setdefault(entry["name"], entry)
    if missing := [k for k in wanted if k not in staged]:
        raise SystemExit(f"not in H1's specs: {missing}")
    eval_set = json.loads(Path(staged["eval_set"]["path"]).read_text())
    ids = [it["id"] for it in eval_set["items"]]
    pilot_ids = [i for i in ids if any(i.startswith(v + "_0") for v in PILOT_ITEMS)]
    rest_ids = [i for i in ids if i not in pilot_ids]
    manifest = {"files": {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in sorted(staged.items())},
                "pilot_items": pilot_ids, "rest_items": rest_ids}
    manifest_path = dest / "inputs.json"
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    videos = sorted(k for k in staged if k.startswith("video_"))
    arguments = ["oracle", "--eval-set", "{staged:eval_set}", "--archive", "{staged:dense_archive}",
                 "--masks", "{staged:fill_masks}", "--mask-record", "{staged:fill_record}",
                 "--svt-streams", "{staged:svt_streams_a}", "--svt-result", "{staged:svt_result_a}",
                 "--svt-streams", "{staged:svt_streams_b}", "--svt-result", "{staged:svt_result_b}",
                 "--lpips-backbone", "{staged:lpips}", *[a for v in videos for a in ("--video", f"{{staged:{v}}}")],
                 "--items", "{items}", "--frames", "{frames}"]
    inputs = [staged[k] for k in ("lpips", *wanted)] + [staged[v] for v in videos]

    def spec(items: list[str], basis: str, full_seconds: int) -> dict[str, Any]:
        return {
            "schema": 1, "entrypoint": ["-m", "experiments.visor.h3"],
            "environment": {"path": args.environment[0], "sha256": args.environment[1]},
            "inputs": [{"path": str(manifest_path), "sha256": file_sha256(manifest_path)}],
            "arguments": arguments, "staged_inputs": inputs,
            "scale": {"items": {"smoke": items[0], "full": ",".join(items)}, "frames": {"smoke": 48, "full": 240}},
            "smoke": {"seconds": 600, "representative_basis": basis}, "full": {"seconds": full_seconds},
            "validator": ["{python}", "-m", "experiments.visor.h3", "validate"], "validator_seconds": 300,
            "required_commands": [], "budget_seconds": full_seconds + 1500, "deadline": args.deadline,
            "stall_seconds": 1800, "cpu_threads": 16, "device": "gpu", "hosts": ["gpu1", "gpu2", "gpu3", "gpu4", "gpu5", "gpu6"],
            "gpu_models": GPU_MODELS, "gpu_memory_mib": 8000, "local_storage_gib": 40,
            "contention": {"policy": "pause", "pause_seconds": 900, "resume_attempts": 2},
        }

    path_basis = ("its first 48 window frames through the full path (window decode with the sparse-JPEG gate, fill masks "
                  "checked against their record, B2's four SVT-AV1 streams checked against their records and decoded by "
                  "dav1d, LPIPS on CPU against the GPU, DIS flows, warps and every operating point)")
    specs = {"pilot": spec(pilot_ids, f"H3a: {pilot_ids[0]}, {path_basis}; the stage runs B2's four pilot items at 240 frames.", 2700)}
    shares = [rest_ids[k::args.parts] for k in range(args.parts)]
    for k, share in enumerate(shares):
        specs[f"full-{chr(97 + k)}"] = spec(share, f"H3a: {share[0]}, {path_basis}; the stage runs {len(share)} items at 240 frames.", 2700)
    for name, body in specs.items():
        (dest / f"h3-oracle-{name}.json").write_text(json.dumps(body, indent=1) + "\n")
    print(json.dumps({"inputs": str(manifest_path), "specs": sorted(specs), "pilot_items": pilot_ids,
                      "shares": [len(s) for s in shares]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
