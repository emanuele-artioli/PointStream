"""Fleet specifications for K6b (hands) 2c, the buildable bank (docs/experiments.md, 2026-10-22 K6b (hands) 2c).

    python tools/datasets/k6bh2c_inputs.py --base-spec K6BH-SPEC.json --staged STAGED.json --items IDS \\
        --deadline ISO --dest DIR --name NAME

The environment, gate inputs and hosts are a rung-2b spec's (``--base-spec``). ``--staged`` maps names to staged
entries ({name, path, sha256[, extract]}, all under the data root, from earlier K1f and K6b jobs): the window, masks,
LPIPS and DISTS entries, WiLoR's checkpoint and MANO, K1f's published stages (``k1f_<N>``) and the items' videos
(``video_<VIDEO>``). One item per job. Standard library only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COMMON = ("eval_set", "dense_archive", "fill_masks", "fill_record", "lpips", "dists", "wilor_checkpoint", "mano_right", "mano_left")


def main() -> int:
    ap = argparse.ArgumentParser()
    for flag in ("--base-spec", "--staged", "--items", "--deadline", "--dest", "--name"):
        ap.add_argument(flag, required=True)
    ap.add_argument("--smoke-frames", type=int, default=24)
    ap.add_argument("--full-frames", type=int, default=240)
    ap.add_argument("--smoke-stride", type=int, default=200)
    ap.add_argument("--full-stride", type=int, default=5)
    ap.add_argument("--full-seconds", type=int, default=2700)
    args = ap.parse_args()
    base = json.loads(Path(args.base_spec).read_text())
    staged = json.loads(Path(args.staged).read_text())
    items = args.items.split(",")
    videos = sorted({f"video_{i.rsplit('_', 1)[0]}" for i in items})
    stages = sorted(n for n in staged if n.startswith("k1f_"))
    entries = [staged[k] for k in (*COMMON, *stages, *videos)]
    arguments = ["run", "--eval-set", "{staged:eval_set}", "--archive", "{staged:dense_archive}", "--masks", "{staged:fill_masks}",
                 "--mask-record", "{staged:fill_record}", "--lpips-backbone", "{staged:lpips}", "--dists-backbone", "{staged:dists}",
                 "--wilor-checkpoint", "{staged:wilor_checkpoint}", "--mano-right", "{staged:mano_right}", "--mano-left", "{staged:mano_left}",
                 *[a for s in stages for a in ("--k1f-stage", f"{{staged:{s}}}")],
                 *[a for v in videos for a in ("--video", f"{{staged:{v}}}")],
                 "--items", "{items}", "--frames", "{frames}", "--stride", "{stride}"]
    basis = (f"K6b (hands) 2c: {items[0]}'s first {args.smoke_frames} window frames and every {args.smoke_stride}th VISOR frame "
             "outside the window through the full path (bank frames decoded with PyAV in a fresh process, dense masks, WiLoR's "
             "two-pass hand-box fit, greedy medoids, SVT-AV1 stills and dav1d, polygon masks, the window decode, fill masks, "
             "K1f's rows, the coded and free banks and K1f's warp, LPIPS and DISTS on R); the stage runs "
             f"{len(items)} item(s) at {args.full_frames} frames and stride {args.full_stride}.")
    determinism = ("WiLoR's fits, the medoids, SVT-AV1's stills and dav1d's decodes, the warps and LPIPS and DISTS on R do not "
                   "depend on wall time; recorded seconds are not evidence")
    spec = {
        "schema": 1, "entrypoint": ["-m", "experiments.visor.k6bh2c"], "environment": base["environment"], "inputs": base["inputs"],
        "arguments": arguments, "staged_inputs": entries,
        "scale": {"items": {"smoke": items[0], "full": ",".join(items)}, "frames": {"smoke": args.smoke_frames, "full": args.full_frames},
                  "stride": {"smoke": args.smoke_stride, "full": args.full_stride}},
        "smoke": {"seconds": 600, "representative_basis": basis}, "full": {"seconds": args.full_seconds},
        "validator": ["{python}", "-m", "experiments.visor.k6bh2c", "validate"], "validator_seconds": 300, "required_commands": [],
        "budget_seconds": args.full_seconds + 900, "deadline": args.deadline, "stall_seconds": args.full_seconds - 300,
        "cpu_threads": 16, "device": "gpu", "hosts": base["hosts"], "gpu_models": ["RTX 6000 Ada"], "gpu_memory_mib": 44000,
        "local_storage_gib": 60, "gpu_sharing": {"mode": "shared", "margin_mib": 1024, "basis": determinism},
    }
    out = Path(args.dest) / f"k6bh2c-{args.name}.json"
    out.write_text(json.dumps(spec, indent=1) + "\n")
    print(out, items)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
