"""Fleet specifications for K6b (hands) 2, the zero-shot oracle (docs/experiments.md, 2026-10-22 K6b (hands) 2).

    python tools/datasets/k6bh_inputs.py --k1f-spec K1F-PART0-SPEC.json --catalog CATALOG.json --items IDS \\
        --steps N --deadline ISO --dest DIR --name NAME [--smoke-frames 24] [--full-frames 240]

The window, masks, LPIPS and DISTS entries are K1f's (identities unchanged, from one K1f part's spec). The catalog
(JSON list of {name, path, sha256}) adds FoundHand's source archive and weights (``foundhand_src``,
``foundhand_ckpt``, ``vae``), the items' videos (``video_<VIDEO>``) and K1f's published stages (``k1f_<N>``), all
under the data root. Standard library only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COMMON = ("eval_set", "dense_archive", "fill_masks", "fill_record", "lpips", "dists")


def main() -> int:
    ap = argparse.ArgumentParser()
    for flag in ("--k1f-spec", "--catalog", "--items", "--steps", "--deadline", "--dest", "--name"):
        ap.add_argument(flag, required=True)
    ap.add_argument("--smoke-frames", type=int, default=24)
    ap.add_argument("--full-frames", type=int, default=240)
    ap.add_argument("--variants", default="gen,gen@frame,ref,warp")
    ap.add_argument("--full-seconds", type=int, default=2700)
    args = ap.parse_args()
    k1f = json.loads(Path(args.k1f_spec).read_text())
    staged = {e["name"]: e for e in k1f["staged_inputs"]}
    catalog = {e["name"]: e for e in json.loads(Path(args.catalog).read_text())}
    items = args.items.split(",")
    videos = sorted({f"video_{i.rsplit('_', 1)[0]}" for i in items})
    stages = sorted(n for n in catalog if n.startswith("k1f_"))
    entries = [staged[k] for k in COMMON] + [catalog[k] for k in ("foundhand_src", "foundhand_ckpt", "vae", *stages, *videos)]
    arguments = ["run", "--eval-set", "{staged:eval_set}", "--archive", "{staged:dense_archive}", "--masks", "{staged:fill_masks}",
                 "--mask-record", "{staged:fill_record}", "--lpips-backbone", "{staged:lpips}", "--dists-backbone", "{staged:dists}",
                 "--foundhand-src", "{staged:foundhand_src}", "--foundhand-ckpt", "{staged:foundhand_ckpt}", "--vae", "{staged:vae}",
                 *[a for s in stages for a in ("--k1f-stage", f"{{staged:{s}}}")],
                 *[a for v in videos for a in ("--video", f"{{staged:{v}}}")],
                 "--items", "{items}", "--frames", "{frames}", "--steps", args.steps, "--variants", args.variants]
    basis = (f"K6b (hands) 2: {items[0]}'s first {args.smoke_frames} window frames through the full path (window decode, fill masks, "
             "K1f's rows, FoundHand loaded from the pinned source with the EMA weights, one guided step profiled for kernels, "
             f"the video and per-frame procedures at {args.steps} steps, the reference warp, K1f's warp, LPIPS and DISTS on R); "
             f"the stage runs {len(items)} item(s) at {args.full_frames} frames.")
    spec = {
        "schema": 1, "entrypoint": ["-m", "experiments.visor.k6bh"], "environment": k1f["environment"], "inputs": k1f["inputs"],
        "arguments": arguments, "staged_inputs": entries,
        "scale": {"items": {"smoke": items[0], "full": ",".join(items)}, "frames": {"smoke": args.smoke_frames, "full": args.full_frames}},
        "smoke": {"seconds": 600, "representative_basis": basis}, "full": {"seconds": args.full_seconds},
        "validator": ["{python}", "-m", "experiments.visor.k6bh", "validate"], "validator_seconds": 300, "required_commands": [],
        "budget_seconds": args.full_seconds + 900, "deadline": args.deadline, "stall_seconds": 1500, "cpu_threads": 16, "device": "gpu",
        "hosts": k1f["hosts"], "gpu_models": ["RTX 6000 Ada"], "gpu_memory_mib": 16000, "local_storage_gib": 60,
        "gpu_sharing": {"mode": "shared", "margin_mib": 4096,
                        "basis": "FoundHand's samples depend on the seeds, steps and inputs, and LPIPS and DISTS on R are deterministic; "
                                 "generation seconds are not evidence (the step count comes from the smoke)"},
    }
    out = Path(args.dest) / f"k6bh-{args.name}.json"
    out.write_text(json.dumps(spec, indent=1) + "\n")
    print(out, items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
