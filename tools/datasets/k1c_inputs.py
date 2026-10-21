"""Fleet specification for PLAN K1c (K6's bound on egocentric hands; docs/experiments.md, 2026-10-21 K1c).

    python tools/datasets/k1c_inputs.py --h3c2-spec h3c2-P01_107.json --deadline ISO --out k1c-run.json

Every staged input is H3c2's (``h3c-2026-10-11/h3c2-P01_107.json``, identities unchanged): the evaluation set, the
dense archive, fill masks and record, the four pilot videos, LPIPS and DISTS backbones, H2's VISOR poses and
``code.json``, and MANO. One job: the smoke is P01_107's first 48 frames, the stage the four pilot items. Standard
library only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PILOT = ("P01_107_0000003049", "P02_02_0000006080", "P03_10_0000000616", "P09_106_0000006303")
NEEDED = ("eval_set", "dense_archive", "fill_masks", "fill_record", "lpips", "dists", "poses_0", "poses_1", "code", "mano_right")
VIDEOS = ("video_P01_107", "video_P02_02", "video_P03_10", "video_P09_106")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--h3c2-spec", required=True)
    ap.add_argument("--deadline", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    h3c2 = json.loads(Path(args.h3c2_spec).read_text())
    staged = {e["name"]: e for e in h3c2["staged_inputs"]}
    entries = [staged[k] for k in NEEDED + VIDEOS]
    arguments = ["run", "--eval-set", "{staged:eval_set}", "--archive", "{staged:dense_archive}", "--masks", "{staged:fill_masks}",
                 "--mask-record", "{staged:fill_record}", "--lpips-backbone", "{staged:lpips}", "--dists-backbone", "{staged:dists}",
                 "--poses", "{staged:poses_0}", "--poses", "{staged:poses_1}", "--pose-code", "{staged:code}",
                 "--mano-right", "{staged:mano_right}", *[a for v in VIDEOS for a in ("--video", f"{{staged:{v}}}")],
                 "--items", "{items}", "--frames", "{frames}"]
    basis = ("K1c run: P01_107_0000003049's first 48 window frames through the full path (window decode with the sparse-JPEG "
             "gate, fill masks checked against their record, H2's WiLoR rows coded at the 100 ms choice and projected, the hand "
             "skeleton warp at the coded and uncoded joints, LPIPS and DISTS on R as h3c.Scorer, LPIPS over S); the stage runs the "
             "four pilot items at 240 frames.")
    spec = {
        "schema": 1, "entrypoint": ["-m", "experiments.visor.k1c"], "environment": h3c2["environment"], "inputs": h3c2["inputs"],
        "arguments": arguments, "staged_inputs": entries,
        "scale": {"items": {"smoke": PILOT[0], "full": ",".join(PILOT)}, "frames": {"smoke": 48, "full": 240}},
        "smoke": {"seconds": 600, "representative_basis": basis}, "full": {"seconds": 2700},
        "validator": ["{python}", "-m", "experiments.visor.k1c", "validate"], "validator_seconds": 300,
        "required_commands": [], "budget_seconds": 3600, "deadline": args.deadline, "stall_seconds": 1200,
        "cpu_threads": 16, "device": "gpu", "hosts": h3c2["hosts"], "gpu_models": ["RTX 6000 Ada", "RTX A6000"],
        "gpu_memory_mib": 16000, "local_storage_gib": 30,
        "gpu_sharing": {"mode": "shared", "basis": "the warp is CPU arithmetic and LPIPS/DISTS on R are deterministic; "
                        "wall time is not evidence"},
    }
    Path(args.out).write_text(json.dumps(spec, indent=1) + "\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
