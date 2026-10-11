"""Staged inputs and fleet specifications for PLAN step H3b's oracle (perfect appearance at the coded pose).

    python tools/datasets/h3b_inputs.py --dest DIR --h3a-dir DIR --h2-dir DIR --poses TAR TAR --models DIR \\
        --environment TAR SHA256 --deadline ISO [--parts N]

Under ``DEST/inputs`` it hard-links the DISTS and Inception backbones from ``--models``, H2's VISOR pose
archives and H2's ``code.json``, and writes ``inputs.json`` (sha256 of everything). The VISOR inputs, B2's
SVT-AV1 streams and LPIPS are H3a's staged inputs, and MANO and WiLoR are H2's, read from their specs.
Then the specs beside them: ``h3b-oracle-pilot.json`` (B2's four pilot items) and
``h3b-oracle-full-<k>.json`` (the other 30 items in ``--parts`` shares, each a job of at most 45 minutes).
Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

PILOT_ITEMS = ("P01_107", "P09_106", "P02_02", "P03_10")  # B2's pilot videos
GPU_MODELS = ["RTX 6000 Ada", "RTX A6000"]  # LPIPS, DISTS and WiLoR verified there (docs/fleet.md)


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


def entry(name: str, path: Path) -> dict[str, Any]:
    return {"name": name, "path": str(path), "sha256": file_sha256(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", required=True)
    parser.add_argument("--h3a-dir", required=True, help="pointstream-data/visor/h3-2026-10-10 (the specs)")
    parser.add_argument("--h2-dir", required=True, help="pointstream-data/visor/h2-2026-10-09 (specs and report-997432d)")
    parser.add_argument("--poses", nargs=2, required=True, help="H2's VISOR published.tar, pilot and full")
    parser.add_argument("--models", required=True, help="the Models directory")
    parser.add_argument("--environment", nargs=2, required=True, metavar=("TAR", "SHA256"))
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--parts", type=int, default=6)
    args = parser.parse_args()
    dest = Path(args.dest)
    inputs = dest / "inputs"
    staged: dict[str, dict[str, Any]] = {}
    for spec_name in sorted(p.name for p in Path(args.h3a_dir).glob("h3-oracle-*.json")):
        for e in json.loads((Path(args.h3a_dir) / spec_name).read_text())["staged_inputs"]:
            staged.setdefault(e["name"], e)
    for e in json.loads((Path(args.h2_dir) / "h2-visor-full.json").read_text())["staged_inputs"]:
        if e["name"] in ("mano_left", "mano_right", "wilor_checkpoint"):
            staged.setdefault(e["name"], e)
    models = Path(args.models)
    staged["dists"] = entry("dists", link(models / "DISTS" / "vgg16-397923af.pth", inputs / "dists" / "vgg16-397923af.pth"))
    staged["inception"] = entry("inception", link(models / "Inception" / "inception_v3_google-0cc3c7bd.pth",
                                                  inputs / "inception" / "inception_v3_google-0cc3c7bd.pth"))
    for k, tar in enumerate(args.poses):
        staged[f"poses_{k}"] = entry(f"poses_{k}", link(Path(tar), inputs / "poses" / f"{Path(tar).parent.parent.name}.tar"))
    staged["code"] = entry("code", link(Path(args.h2_dir) / "report-997432d" / "code.json", inputs / "h2-code.json"))
    wanted = ("lpips", "eval_set", "dense_archive", "fill_masks", "fill_record", "svt_streams_a", "svt_result_a",
              "svt_streams_b", "svt_result_b", "mano_left", "mano_right", "wilor_checkpoint")
    if missing := [k for k in wanted if k not in staged]:
        raise SystemExit(f"not in H3a's or H2's specs: {missing}")
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
                 "--lpips-backbone", "{staged:lpips}", "--dists-backbone", "{staged:dists}", "--inception", "{staged:inception}",
                 "--poses", "{staged:poses_0}", "--poses", "{staged:poses_1}", "--code", "{staged:code}",
                 "--mano-left", "{staged:mano_left}", "--mano-right", "{staged:mano_right}",
                 "--wilor-checkpoint", "{staged:wilor_checkpoint}",
                 *[a for v in videos for a in ("--video", f"{{staged:{v}}}")],
                 "--items", "{items}", "--frames", "{frames}"]
    staged_list = [staged[k] for k in (*wanted, "dists", "inception", "poses_0", "poses_1", "code")] + [staged[v] for v in videos]

    def spec(items: list[str], basis: str, full_seconds: int) -> dict[str, Any]:
        return {
            "schema": 1, "entrypoint": ["-m", "experiments.visor.h3b"],
            "environment": {"path": args.environment[0], "sha256": args.environment[1]},
            "inputs": [{"path": str(manifest_path), "sha256": file_sha256(manifest_path)}],
            "arguments": arguments, "staged_inputs": staged_list,
            "scale": {"items": {"smoke": items[0], "full": ",".join(items)}, "frames": {"smoke": 48, "full": 240}},
            "smoke": {"seconds": 600, "representative_basis": basis}, "full": {"seconds": full_seconds},
            "validator": ["{python}", "-m", "experiments.visor.h3b", "validate"], "validator_seconds": 300,
            "required_commands": [], "budget_seconds": full_seconds + 1500, "deadline": args.deadline,
            "stall_seconds": 1800, "cpu_threads": 16, "device": "gpu", "hosts": ["gpu1", "gpu2", "gpu3", "gpu4", "gpu5", "gpu6"],
            "gpu_models": GPU_MODELS, "gpu_memory_mib": 16000, "local_storage_gib": 40,
            "contention": {"policy": "pause", "pause_seconds": 900, "resume_attempts": 2},
        }

    path_basis = ("its first 48 window frames through the full path (window decode with the sparse-JPEG gate, fill masks "
                  "checked against their record, B2's four SVT-AV1 streams checked and decoded by dav1d, H2's poses coded "
                  "and rendered, every composite, LPIPS on CPU against the GPU, DISTS, flicker, WiLoR and Inception on "
                  "judge frames)")
    specs = {"pilot": spec(pilot_ids, f"H3b oracle: {pilot_ids[0]}, {path_basis}; the stage runs B2's four pilot items at 240 frames.", 2700)}
    shares = [rest_ids[k::args.parts] for k in range(args.parts)]
    for k, share in enumerate(shares):
        specs[f"full-{chr(97 + k)}"] = spec(share, f"H3b oracle: {share[0]}, {path_basis}; the stage runs {len(share)} items at 240 frames.", 2700)
    for name, body in specs.items():
        (dest / f"h3b-oracle-{name}.json").write_text(json.dumps(body, indent=1) + "\n")
    print(json.dumps({"inputs": str(manifest_path), "specs": sorted(specs), "pilot_items": pilot_ids,
                      "shares": [len(s) for s in shares]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
