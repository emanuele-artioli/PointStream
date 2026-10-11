"""Fleet specifications for PLAN step H3c (a perceptually tuned DCVC-UF for the hand pixels).

    python tools/datasets/h3c_inputs.py --dest DIR --h3b-dir DIR --dcvc-dir DIR --inspect BIN \\
        --environment TAR SHA256 --deadline ISO [--train FIT NAME WEIGHT STEPS] [--code LABEL NAME=CKPT ...]

Every staged input is H3b's (``--h3b-dir``'s pilot spec, checked against its ``inputs.json``: evaluation set, dense archive, fill masks and record,
B2's SVT-AV1 streams and results, LPIPS, DISTS, the videos), plus DCVC-UF's image and HT-L checkpoints from
``--dcvc-dir`` and H1's libaom inspect binary. Always written: ``h3c-anchor-<video>.json`` (SVT-AV1's points and
the reference DCVC-UF on one pilot item each) and ``h3c-calibrate.json`` (40 steps of the upper bound at β = 0).
``--train`` adds ``h3c-train-<FIT>-<NAME>.json``; ``--code`` adds ``h3c-code-<LABEL>-<video>.json``, one per pilot
item, for fine-tuned checkpoints (hard-linked under ``DEST/inputs/models``). Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

PILOT = ("P01_107", "P02_02", "P03_10", "P09_106")  # B2's pilot videos
EXCLUDED = ("P01", "P02", "P03", "P09")  # their participants
GPU_MODELS = ["RTX 6000 Ada", "RTX A6000"]  # DCVC-UF, LPIPS and DISTS verified there (docs/fleet.md)
TRAIN_GPU_MODELS = ["RTX 6000 Ada"]  # HT-L training ran only on Ada (G5b): an A6000 first run would be exclusive
CODE_SHARING = {"mode": "shared", "basis": "DCVC-UF and SVT-AV1 streams, decodes and their rate and LPIPS/PSNR/DISTS on the "
                "hand region are deterministic and do not depend on wall time; recorded encode and decode seconds are not evidence"}
TRAIN_SHARING = {"mode": "shared", "margin_mib": 1024,
                 "basis": "the fine-tuned weights and monitor losses depend on the seed and the steps, not on wall time; "
                          "step times are not evidence (the step count comes from G5b's exclusive Ada measurement)"}
HOSTS = ["gpu1", "gpu2", "gpu3", "gpu4", "gpu5", "gpu6"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def entry(name: str, path: Path) -> dict[str, Any]:
    return {"name": name, "path": str(path), "sha256": file_sha256(path)}


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
    parser.add_argument("--h3b-dir", required=True, help="pointstream-data/visor/h3b-2026-10-11")
    parser.add_argument("--dcvc-dir", required=True, help="B2's staged DCVC-UF checkpoints")
    parser.add_argument("--inspect", required=True, help="H1's libaom inspect binary")
    parser.add_argument("--environment", nargs=2, required=True, metavar=("TAR", "SHA256"))
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--train", nargs=4, metavar=("FIT", "NAME", "WEIGHT", "STEPS"))
    parser.add_argument("--code", nargs="+", metavar="LABEL NAME=CKPT")
    args = parser.parse_args()
    dest = Path(args.dest)
    manifest = json.loads((Path(args.h3b_dir) / "inputs.json").read_text())
    files = manifest["files"]
    # H3b's spec entries, not its manifest's: they carry ``extract`` for the archives.
    pilot_spec = json.loads((Path(args.h3b_dir) / "h3b-oracle-pilot.json").read_text())
    staged = {e["name"]: e for e in pilot_spec["staged_inputs"]
              if e["name"] in ("eval_set", "dense_archive", "fill_masks", "fill_record", "lpips", "dists", "svt_streams_a",
                               "svt_result_a", "svt_streams_b", "svt_result_b") or e["name"].startswith("video_")}
    if mismatch := [k for k, e in staged.items() if files[k]["sha256"] != e["sha256"]]:
        raise SystemExit(f"H3b's spec and manifest disagree on {mismatch}")
    staged["dcvc_image"] = entry("dcvc_image", Path(args.dcvc_dir) / "cvpr2026_image.pth.tar")
    staged["dcvc_video"] = entry("dcvc_video", Path(args.dcvc_dir) / "cvpr2026_video_htl.pth.tar")
    staged["inspect"] = entry("inspect", Path(args.inspect))
    pilot = [i for i in manifest["pilot_items"] + manifest["rest_items"] if any(i.startswith(v + "_0") for v in PILOT)]
    component = [i for i in manifest["rest_items"] if i.split("_")[0] not in EXCLUDED]
    videos = sorted(k for k in staged if k.startswith("video_"))
    dest.mkdir(parents=True, exist_ok=True)
    inputs = dest / "inputs.json"
    inputs.write_text(json.dumps({"files": {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in sorted(staged.items())},
                                  "pilot_items": pilot, "component_items": component}, indent=1, sort_keys=True) + "\n")
    windows = ["--eval-set", "{staged:eval_set}", "--archive", "{staged:dense_archive}", "--masks", "{staged:fill_masks}",
               "--mask-record", "{staged:fill_record}", "--image-ckpt", "{staged:dcvc_image}", "--lpips-backbone", "{staged:lpips}",
               *[a for v in videos for a in ("--video", f"{{staged:{v}}}")], "--items", "{items}", "--frames", "{frames}"]
    common = [staged[k] for k in ("eval_set", "dense_archive", "fill_masks", "fill_record", "dcvc_image", "lpips")] + \
        [staged[v] for v in videos]

    def spec(arguments: list[str], staged_list: list[dict[str, Any]], scale: dict[str, Any], basis: str, memory: int,
             sharing: dict[str, Any], gpu_models: list[str]) -> dict[str, Any]:
        return {
            "schema": 1, "entrypoint": ["-m", "experiments.visor.h3c"],
            "environment": {"path": args.environment[0], "sha256": args.environment[1]},
            "inputs": [{"path": str(inputs), "sha256": file_sha256(inputs)}],
            "arguments": arguments, "staged_inputs": staged_list, "scale": scale,
            "smoke": {"seconds": 600, "representative_basis": basis}, "full": {"seconds": 2700},
            "validator": ["{python}", "-m", "experiments.visor.h3c", "validate"], "validator_seconds": 300,
            "required_commands": [], "budget_seconds": 4500, "deadline": args.deadline, "stall_seconds": 1800,
            "cpu_threads": 16, "device": "gpu", "hosts": HOSTS, "gpu_models": gpu_models, "gpu_memory_mib": memory,
            "local_storage_gib": 60, "gpu_sharing": sharing,
        }

    written = []

    def code_spec(name: str, item: str, arms: list[tuple[str, dict[str, Any]]], anchor: bool) -> None:
        arguments = ["code", *windows, "--dists-backbone", "{staged:dists}", "--qps", "{qps}", "--scales", "{scales}",
                     *[a for n, e in arms for a in ("--arm", n, f"{{staged:{e['name']}}}")]]
        extra = [staged["dists"]] + [e for _, e in arms]
        if anchor:
            arguments += ["--anchor", "--svt-streams", "{staged:svt_streams_a}", "--svt-result", "{staged:svt_result_a}",
                          "--svt-streams", "{staged:svt_streams_b}", "--svt-result", "{staged:svt_result_b}", "--inspect", "{staged:inspect}"]
            extra += [staged[k] for k in ("svt_streams_a", "svt_result_a", "svt_streams_b", "svt_result_b", "inspect")]
        scale = {"items": {"smoke": item, "full": item}, "frames": {"smoke": 48, "full": 240},
                 "qps": {"smoke": "0", "full": "0,21,42"}, "scales": {"smoke": "1,4", "full": "1,2,4"}}
        basis = (f"H3c code: {item}, its first 48 window frames through the full path (window decode with the sparse-JPEG gate, "
                 f"fill masks checked against their record, {'every SVT-AV1 point (B2 streams checked and decoded by dav1d, H3b2 low points re-encoded and accounted by libaom inspect), ' if anchor else ''}"
                 "DCVC-UF HT-L encode and decode by B2's worker at QP 0 and scales 1 and 4 with the per-NAL bit split by the model's "
                 "own estimate, LPIPS on CPU against the GPU, DISTS); the stage runs 240 frames at QP 0, 21, 42 and scales 1, 2, 4.")
        # 24000 MiB was overrun (26,098 MiB, job 20261011T220701Z-f13c88e2) before the scorer's cache was released.
        body = spec(arguments, common + extra, scale, basis, 32000, CODE_SHARING, GPU_MODELS)
        path = dest / f"h3c-{name}-{next(v for v in PILOT if item.startswith(v + '_0'))}.json"
        path.write_text(json.dumps(body, indent=1) + "\n")
        written.append(path.name)

    for item in pilot:
        code_spec("anchor", item, [("reference", staged["dcvc_video"])], anchor=True)

    def train_spec(name: str, fit: str, weight: str, steps: dict[str, int]) -> None:
        items = pilot if fit == "upper" else component
        arguments = ["train", *windows, "--fit", fit, "--segments", "all" if fit == "upper" else "2", "--steps", "{steps}",
                     "--lpips-weight", weight, "--video-ckpt", "{staged:dcvc_video}"]
        scale = {"items": {"smoke": items[0], "full": ",".join(items)}, "frames": {"smoke": 48, "full": 240}, "steps": steps}
        basis = (f"H3c train: {items[0]}, its first 48 window frames as one 48-frame segment per scale (1, 1/2, 1/4), "
                 f"hand-centred 512x512 crops with R as the loss mask, {steps['smoke']} steps of G5b's trainer at the stage's settings "
                 "(512x512, 4 groups, batch 4: the stage's memory), monitor curve, state save, checkpoint published; "
                 f"the stage builds {len(items)} windows and runs {steps['full']} steps.")
        # G5b's run peaked at 42.7 GiB allocated: nearly a whole 48 GB card, admitted only when it is near empty.
        body = spec(arguments, common + [staged["dcvc_video"]], scale, basis, 47000, TRAIN_SHARING, TRAIN_GPU_MODELS)
        path = dest / f"h3c-{name}.json"
        path.write_text(json.dumps(body, indent=1) + "\n")
        written.append(path.name)

    train_spec("calibrate", "upper", "0", {"smoke": 3, "full": 40})
    if args.train:
        fit, name, weight, steps = args.train
        train_spec(f"train-{fit}-{name}", fit, weight, {"smoke": 3, "full": int(steps)})
    if args.code:
        label, *arms = args.code
        staged_arms = []
        for spec_arm in arms:
            arm, _, ckpt = spec_arm.partition("=")
            target = link(Path(ckpt), dest / "inputs" / "models" / f"{arm}.pth.tar")
            staged_arms.append((arm, entry(f"ckpt_{arm.replace('-', '_')}", target)))
        for item in pilot:
            code_spec(f"code-{label}", item, staged_arms, anchor=False)
    print(json.dumps({"inputs": str(inputs), "specs": written, "pilot_items": pilot, "component_items": len(component)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
