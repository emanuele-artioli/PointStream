"""Fleet specifications for PLAN step H3e (a pretrained generator renders the hand from the coded keypoints).

    python tools/datasets/h3e_inputs.py --dest DIR --h3c-dir DIR --wan-dir DIR --dit FILE --jcv-dir DIR \\
        --environment TAR SHA256 --jcv-environment TAR SHA256 --deadline ISO \\
        [--prepared TAR] [--generate LABEL ITEMS ORDERS STEPS] [--judge NAME=TAR ORDER]

Staged inputs come from H3c2's spec (``--h3c-dir``: windows, masks, videos, DCVC-UF, H2's poses and coding, MANO,
WiLoR), plus H3b's Inception weights, and the generator's weights: Wan2.1-I2V-14B-480P's T5, VAE, CLIP and
tokenizer (``--wan-dir``), its DiT cast to bf16 (``--dit``, tools/models/wan_dit_bf16.py) and JointControlVideo's
LoRA and hand module (``--jcv-dir``), hard-linked under ``DEST/inputs/models`` (the tokenizer as a tar).

Always written: ``h3e-prepare.json`` (CPU). ``--prepared`` (the prepare job's published.tar) enables
``--generate``: ``h3e-generate-<LABEL>.json`` for ITEMS (comma list; ``pilot`` for B2's four) with joint ORDERS
(comma list) and sampling STEPS. ``--judge`` adds ``h3e-judge-<video>.json`` per pilot item, for the named generate
job's published.tar and ORDER. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from pathlib import Path
from typing import Any

PILOT = ("P01_107", "P02_02", "P03_10", "P09_106")  # B2's pilot videos
DEV_CANDIDATES = ("P04_06_0000038498", "P22_107_0000012402", "P30_07_0000005463")  # outside the pilot participants
HOSTS = ["gpu1", "gpu2", "gpu3", "gpu4", "gpu5", "gpu6"]
GEN_GPU_MODELS = ["RTX 6000 Ada"]  # the model's first class; the smoke asserts device and kernels
JUDGE_GPU_MODELS = ["RTX 6000 Ada", "RTX A6000"]  # DCVC-UF, WiLoR and Inception verified there (docs/fleet.md)
JUDGE_SHARING = {"mode": "shared", "basis": "DCVC-UF streams, decodes, the composites and their WiLoR, Inception and flicker "
                 "values are deterministic and do not depend on wall time; recorded seconds are not evidence"}
GEN_SHARING = {"mode": "shared", "margin_mib": 1024,
               "basis": "the generated frames depend on the seed, steps and inputs, not on wall time; generation seconds "
                        "are not evidence (the step count comes from the exclusive smoke on Ada)"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def entry(name: str, path: Path, extract: bool = False) -> dict[str, Any]:
    return {"name": name, "path": str(path), "sha256": file_sha256(path), **({"extract": True} if extract else {})}


def link(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if os.path.samefile(source, target):
            return target
        raise SystemExit(f"{target} exists and is not {source}")
    os.link(source, target)
    return target


def tar_dir(source: Path, target: Path) -> Path:
    """The directory's files at the archive's top level (an extracted archive is the directory)."""
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(target, "w") as t:
            for f in sorted(source.rglob("*")):
                if f.is_file():
                    t.add(f, arcname=str(f.relative_to(source)))
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", required=True)
    parser.add_argument("--h3c-dir", required=True, help="pointstream-data/visor/h3c-2026-10-11")
    parser.add_argument("--h3b-dir", required=True, help="pointstream-data/visor/h3b-2026-10-11 (Inception)")
    parser.add_argument("--wan-dir", help="Models/Wan/Wan2.1-I2V-14B-480P")
    parser.add_argument("--dit", help="the DiT cast to bf16")
    parser.add_argument("--jcv-dir", help="Models/JointControlVideo")
    parser.add_argument("--environment", nargs=2, required=True, metavar=("TAR", "SHA256"))
    parser.add_argument("--jcv-environment", nargs=2, metavar=("TAR", "SHA256"))
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--prepared", help="the prepare job's published.tar")
    parser.add_argument("--generate", nargs=4, action="append", default=[], metavar=("LABEL", "ITEMS", "ORDERS", "STEPS"))
    parser.add_argument("--checked", action="store_true", help="H3c2's inputs were re-hashed by an earlier run (skip it)")
    parser.add_argument("--judge", nargs="+", action="append", default=[], metavar="NAME=TAR ORDER [ITEMS]",
                        help="judge specs for a generate job's published.tar, joint ORDER, ITEMS (comma list; default pilot)")
    args = parser.parse_args()
    dest = Path(args.dest).resolve()
    h3c_spec = json.loads((Path(args.h3c_dir) / "h3c2-P01_107.json").read_text())
    staged = {e["name"]: e for e in h3c_spec["staged_inputs"]}
    inception = next(e for e in json.loads((Path(args.h3b_dir) / "h3b-oracle-pilot.json").read_text())["staged_inputs"]
                     if e["name"] == "inception")
    staged["inception"] = inception
    for e in ([] if args.checked else staged.values()):  # H3c2's inputs are unchanged since its jobs
        if file_sha256(Path(e["path"])) != e["sha256"]:
            raise SystemExit(f"{e['name']}: {e['path']} changed since H3c2")
    models = dest / "inputs" / "models"
    if args.generate:  # the generator's weights, needed only by generate jobs
        if not (args.wan_dir and args.dit and args.jcv_dir and args.jcv_environment):
            raise SystemExit("--generate needs --wan-dir, --dit, --jcv-dir and --jcv-environment")
        wan, jcv = Path(args.wan_dir), Path(args.jcv_dir)
        staged["t5"] = entry("t5", link(wan / "models_t5_umt5-xxl-enc-bf16.pth", models / "wan-t5-umt5-xxl-enc-bf16.pth"))
        staged["vae"] = entry("vae", link(wan / "Wan2.1_VAE.pth", models / "wan-vae.pth"))
        staged["clip"] = entry("clip", link(wan / "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", models / "wan-clip.pth"))
        staged["tokenizer"] = entry("tokenizer", tar_dir(wan / "google" / "umt5-xxl", models / "wan-umt5-xxl-tokenizer.tar"), extract=True)
        staged["dit"] = entry("dit", link(Path(args.dit), models / "wan-dit-bf16.safetensors"))
        staged["lora"] = entry("lora", link(jcv / "dit.safetensors", models / "jcv-dit-lora.safetensors"))
        staged["hand_controller"] = entry("hand_controller", link(jcv / "hand-controller.safetensors", models / "jcv-hand-controller.safetensors"))
    pilot = [json.loads((Path(args.h3c_dir) / f"h3c2-{v}.json").read_text())["scale"]["items"]["full"] for v in PILOT]
    videos = sorted(k for k in staged if k.startswith("video_"))
    dest.mkdir(parents=True, exist_ok=True)
    inputs = dest / "inputs.json"
    inputs.write_text(json.dumps({"files": {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in sorted(staged.items())},
                                  "pilot_items": pilot, "dev_candidates": list(DEV_CANDIDATES)}, indent=1, sort_keys=True) + "\n")
    poses = ["--poses", "{staged:poses_0}", "--poses", "{staged:poses_1}", "--pose-code", "{staged:code}",
             "--mano-left", "{staged:mano_left}", "--mano-right", "{staged:mano_right}"]
    window = ["--eval-set", "{staged:eval_set}", "--archive", "{staged:dense_archive}",
              *[a for v in videos for a in ("--video", f"{{staged:{v}}}")], "--items", "{items}", "--frames", "{frames}", *poses]
    window_staged = [staged[k] for k in ("eval_set", "dense_archive", "poses_0", "poses_1", "code", "mano_left", "mano_right")] + \
        [staged[v] for v in videos]

    def spec(module: list[str], validator: list[str], environment: list[str], arguments: list[str], staged_list: list[dict[str, Any]],
             scale: dict[str, Any], basis: str, device: str, memory: int, sharing: dict[str, Any] | None,
             gpu_models: list[str], storage: int, full_seconds: int = 2700) -> dict[str, Any]:
        body = {
            "schema": 1, "entrypoint": module, "environment": {"path": environment[0], "sha256": environment[1]},
            "inputs": [{"path": str(inputs), "sha256": file_sha256(inputs)}], "arguments": arguments,
            "staged_inputs": staged_list, "scale": scale, "smoke": {"seconds": 600, "representative_basis": basis},
            "full": {"seconds": full_seconds}, "validator": validator, "validator_seconds": 300, "required_commands": [],
            "budget_seconds": full_seconds + 1800, "deadline": args.deadline, "stall_seconds": 1800, "cpu_threads": 16,
            "device": device, "hosts": HOSTS, "gpu_models": gpu_models, "gpu_memory_mib": memory, "local_storage_gib": storage,
        }
        if sharing:
            body["gpu_sharing"] = sharing
        return body

    written = []
    prepare_items = ",".join([*pilot, *DEV_CANDIDATES])
    body = spec(["-m", "experiments.visor.h3e"], ["{python}", "-m", "experiments.visor.h3e", "validate"], args.environment,
                ["prepare", *window], window_staged,
                {"items": {"smoke": DEV_CANDIDATES[0], "full": prepare_items}, "frames": {"smoke": 240, "full": 240}},
                f"H3e prepare: {DEV_CANDIDATES[0]} through the full path (window decode with the sparse-JPEG gate, H2's WiLoR rows "
                "coded at 100 ms and decoded, joints and intrinsics on the 832x480 canvas); the stage runs the four pilot items "
                "and three development candidates.", "cpu", 0, None, [], 60, full_seconds=1800)
    (dest / "h3e-prepare.json").write_text(json.dumps(body, indent=1) + "\n")
    written.append("h3e-prepare.json")

    if args.generate and not args.prepared:
        raise SystemExit("--generate needs --prepared")
    if args.prepared:
        staged["prepared"] = entry("prepared", Path(args.prepared), extract=True)
    weights = ["dit", "t5", "vae", "clip", "tokenizer", "hand_controller", "lora"]
    for label, items, orders, steps in args.generate:
        chosen = ",".join(pilot) if items == "pilot" else items
        first = chosen.split(",")[0]
        arguments = ["generate", "--inputs", "{staged:prepared}", "--items", "{items}", "--frames", "{frames}", "--steps", "{steps}",
                     "--joint-order", *orders.split(","), "--hash-weights",
                     *[a for w in weights for a in (f"--{w.replace('_', '-')}", f"{{staged:{w}}}")]]
        scale = {"items": {"smoke": first, "full": chosen}, "frames": {"smoke": 9, "full": 81},
                 "steps": {"smoke": 4, "full": int(steps)}}
        basis = (f"H3e generate: {first}, JointControlVideo on Wan2.1-I2V-14B-480P through the full path (T5 on the CPU, DiT, "
                 "CLIP, VAE and the hand module on the GPU, expanded patch embedding and LoRA, the coded joints and canvas "
                 f"intrinsics, joint orders {orders}) for 9 frames at 4 steps; asserts the device, bf16 and the attention kernel "
                 f"(first run on Ada) and records the peak memory and seconds per step; the stage runs 81 frames at {steps} steps.")
        # The first run on the class is exclusive (AGENTS.md); its peak sets the shared declaration afterwards.
        exclusive = label.startswith("smoke")
        body = spec(["experiments/visor/h3e_generate.py"], ["{python}", "experiments/visor/h3e_generate.py", "validate"],
                    args.jcv_environment, arguments, [staged[k] for k in ["prepared", *weights]], scale, basis, "gpu",
                    # The exclusive smoke (20261012T123958Z-7b531e66) peaked at 39,654 MiB reserved on Ada.
                    40000 if exclusive else 42000, None if exclusive else GEN_SHARING, GEN_GPU_MODELS, 80)
        (dest / f"h3e-generate-{label}.json").write_text(json.dumps(body, indent=1) + "\n")
        written.append(f"h3e-generate-{label}.json")

    for judge_args in args.judge:
        (name, _, tar), order = judge_args[0].partition("="), judge_args[1]
        items = judge_args[2].split(",") if len(judge_args) > 2 else pilot
        development = len(judge_args) > 2
        staged[f"generated_{name}"] = entry(f"generated_{name}", Path(tar), extract=True)
        for item in items:
            arguments = ["judge", *window, "--masks", "{staged:fill_masks}", "--mask-record", "{staged:fill_record}",
                         "--generated", f"{{staged:generated_{name}}}", "--order", order, "--image-ckpt", "{staged:dcvc_image}",
                         "--video-ckpt", "{staged:dcvc_video}", "--lpips-backbone", "{staged:lpips}", "--inception", "{staged:inception}",
                         "--wilor-checkpoint", "{staged:wilor_checkpoint}", "--qps", "{qps}", "--scales", "{scales}"]
            extra = [staged[k] for k in ("fill_masks", "fill_record", f"generated_{name}", "dcvc_image", "dcvc_video", "lpips",
                                         "inception", "wilor_checkpoint")]
            full_points = ("42", "4") if development else ("0,21,42", "1,2,4")  # the order check needs the generator's frames only
            scale = {"items": {"smoke": item, "full": item}, "frames": {"smoke": 240, "full": 240},
                     "qps": {"smoke": "42", "full": full_points[0]}, "scales": {"smoke": "4", "full": full_points[1]}}
            basis = (f"H3e judge: {item} through the full path (window decode with the sparse-JPEG gate, the generated frames "
                     "back to the window's resolution, V1's paste masks from the coded silhouettes, DCVC-UF HT-L at QP 42 and scale 4 "
                     "with the split onto R, V1's and V2's billing regions, WiLoR, Inception and flicker on the source, the "
                     f"generator, V2, the DCVC-UF composite and V1); the stage runs QP {full_points[0]} at scales {full_points[1]}.")
            body = spec(["-m", "experiments.visor.h3e"], ["{python}", "-m", "experiments.visor.h3e", "validate"], args.environment,
                        arguments, window_staged + extra, scale, basis, "gpu", 36000, JUDGE_SHARING, JUDGE_GPU_MODELS, 60)
            video = item if development else next(v for v in PILOT if item.startswith(v + "_0"))
            (dest / f"h3e-judge-{name}-{video}.json").write_text(json.dumps(body, indent=1) + "\n")
            written.append(f"h3e-judge-{name}-{video}.json")
    print(json.dumps({"inputs": str(inputs), "specs": written, "pilot_items": pilot}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
