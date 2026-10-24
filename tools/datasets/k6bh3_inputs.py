"""Fleet specifications for K6b (hands) 3, the overfit oracle (docs/experiments.md, 2026-10-22 K6b (hands) 3).

    python tools/datasets/k6bh3_inputs.py --mode pilot|fit|transfer --base-spec K6BH-SPEC.json --staged STAGED.json --items ID \\
        --k1f-stages NAME,NAME --deadline ISO --dest DIR --name NAME [--fit-seconds S] [--candidates NAME]

The environment, gate inputs and hosts are a rung-2 spec's; ``--staged`` maps names to staged entries (the window,
masks, LPIPS, DISTS, FoundHand's source, checkpoint and VAE, the named K1f stages and the item's video; in
``transfer`` mode also 2c's stage archive with the item's candidates). One item per job. Standard library only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COMMON = ("eval_set", "dense_archive", "fill_masks", "fill_record", "lpips", "dists", "foundhand_src", "foundhand_ckpt", "vae")


def main() -> int:
    ap = argparse.ArgumentParser()
    for flag in ("--mode", "--base-spec", "--staged", "--items", "--k1f-stages", "--deadline", "--dest", "--name"):
        ap.add_argument(flag, required=True)
    ap.add_argument("--fit-seconds", type=int, default=1800)
    ap.add_argument("--full-seconds", type=int, default=2700)
    ap.add_argument("--candidates", default="", help="transfer: the staged name of 2c's archive with the item's candidates")
    args = ap.parse_args()
    base = json.loads(Path(args.base_spec).read_text())
    staged = json.loads(Path(args.staged).read_text())
    item = args.items
    video = f"video_{item.rsplit('_', 1)[0]}"
    stages = args.k1f_stages.split(",")
    transfer = args.mode == "transfer"
    if transfer and not args.candidates:
        raise SystemExit("--mode transfer needs --candidates")
    entries = [staged[k] for k in (*COMMON, *stages, video, *([args.candidates] if transfer else []))]
    pilot = args.mode == "pilot"
    arguments = ["run", "--mode", args.mode, "--eval-set", "{staged:eval_set}", "--archive", "{staged:dense_archive}",
                 "--masks", "{staged:fill_masks}", "--mask-record", "{staged:fill_record}", "--lpips-backbone", "{staged:lpips}",
                 "--dists-backbone", "{staged:dists}", "--foundhand-src", "{staged:foundhand_src}",
                 "--foundhand-ckpt", "{staged:foundhand_ckpt}", "--vae", "{staged:vae}",
                 *[a for s in stages for a in ("--k1f-stage", f"{{staged:{s}}}")], "--video", f"{{staged:{video}}}",
                 "--items", "{items}", "--frames", "{frames}", "--fit-seconds", "{fit_seconds}", "--eval-frames", "{eval_frames}",
                 *(["--candidates", f"{{staged:{args.candidates}}}", "--fit-frames", "{fit_frames}"] if transfer else [])]
    scale = {"items": {"smoke": item, "full": item},
             "frames": {"smoke": 240 if pilot else 24, "full": 240},
             "fit_seconds": {"smoke": 120, "full": args.fit_seconds},
             "eval_frames": {"smoke": 8 if pilot else 24, "full": 30 if pilot else 240}}
    if transfer:
        scale["fit_frames"] = {"smoke": 60, "full": 240}
    source = ("60 of 2c's candidate frames outside the window decoded and their kept hands cropped and encoded by the VAE"
              if transfer else "the window's crops encoded by the VAE")
    basis = (f"K6b (hands) {'4' if transfer else '3'}, {args.mode}: {item} through the full path (FoundHand's EMA weights loaded, "
             f"{source}, 120 s of fitting with its training_losses, the fixed-probe loss and a training-pair sample "
             "at each mark, rung 2's gen rendered with both hands in lockstep, composited and scored on R"
             + (", on 8 held-out frames from 180" if pilot else ", on 24 frames, with K1f's warp") + "); the stage fits "
             f"{args.fit_seconds} s and judges {scale['eval_frames']['full']} frames.")
    determinism = ("the fit, the samples and LPIPS and DISTS on R depend on the seeds and the step count reached in the fit "
                   "budget; seconds are not evidence, and the steps reached are recorded")
    spec = {
        "schema": 1, "entrypoint": ["-m", "experiments.visor.k6bh3"], "environment": base["environment"], "inputs": base["inputs"],
        "arguments": arguments, "staged_inputs": entries, "scale": scale,
        "smoke": {"seconds": 600, "representative_basis": basis}, "full": {"seconds": args.full_seconds},
        "validator": ["{python}", "-m", "experiments.visor.k6bh3", "validate"], "validator_seconds": 300, "required_commands": [],
        "budget_seconds": args.full_seconds + 900, "deadline": args.deadline, "stall_seconds": args.full_seconds - 300,
        "cpu_threads": 16, "device": "gpu", "hosts": base["hosts"], "gpu_models": ["RTX 6000 Ada"], "gpu_memory_mib": 30000,
        "local_storage_gib": 60, "gpu_sharing": {"mode": "shared", "margin_mib": 1024, "basis": determinism},
    }
    out = Path(args.dest) / f"k6bh3-{args.mode}-{args.name}.json"
    out.write_text(json.dumps(spec, indent=1) + "\n")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
