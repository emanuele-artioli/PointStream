"""Fleet specifications for PLAN K1f (the hand bound's scope on held-out items; docs/experiments.md, 2026-10-21 K1f).

    python tools/datasets/k1f_inputs.py --h3c2-spec h3c2-P01_107.json --h3d-score-spec h3dc-score-P01_107-dcvc0_2.json \\
        --eval-set experiments/visor/eval_set.json --deadline ISO --part N --size K --dest DIR [--cover --hand-box]

B2's 30 non-pilot items in a content-blind order (sha256 of the item id); part N (from 0) is items N*K .. N*K+K-1.
Staged inputs are H3c2's (window, masks, LPIPS, DISTS, DCVC-UF, libaom inspect, H2's poses and code, MANO, WiLoR) and
H3d's G networks (the generic jobs' archives), identities unchanged. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PILOT = ("P01_107_0000003049", "P02_02_0000006080", "P03_10_0000000616", "P09_106_0000006303")
NEEDED = ("eval_set", "dense_archive", "fill_masks", "fill_record", "lpips", "dists", "dcvc_image", "dcvc_video", "inspect",
          "poses_0", "poses_1", "code", "mano_right")


def held_out(eval_set: dict) -> list[dict]:
    rest = [i for i in eval_set["items"] if i["id"] not in PILOT]
    return sorted(rest, key=lambda i: hashlib.sha256(i["id"].encode()).hexdigest())


def main() -> int:
    ap = argparse.ArgumentParser()
    for flag in ("--h3c2-spec", "--h3d-score-spec", "--eval-set", "--deadline", "--dest"):
        ap.add_argument(flag, required=True)
    ap.add_argument("--part", type=int, required=True)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--cover", action="store_true")
    ap.add_argument("--hand-box", action="store_true")
    args = ap.parse_args()
    h3c2 = json.loads(Path(args.h3c2_spec).read_text())
    score = json.loads(Path(args.h3d_score_spec).read_text())
    staged = {e["name"]: e for e in h3c2["staged_inputs"]}
    generic = [e for e in score["staged_inputs"] if e["name"].startswith("generic_")]
    items = held_out(json.loads(Path(args.eval_set).read_text()))[args.part * args.size:(args.part + 1) * args.size]
    if not items:
        raise SystemExit("no items in that part")
    videos = sorted({f"video_{i['video']}" for i in items})
    extra = ("mano_left", "wilor_checkpoint") if args.cover else ()
    entries = [staged[k] for k in NEEDED + extra] + generic + [staged[v] for v in videos]
    arguments = ["run", "--eval-set", "{staged:eval_set}", "--archive", "{staged:dense_archive}", "--masks", "{staged:fill_masks}",
                 "--mask-record", "{staged:fill_record}", "--lpips-backbone", "{staged:lpips}", "--dists-backbone", "{staged:dists}",
                 "--image-ckpt", "{staged:dcvc_image}", "--video-ckpt", "{staged:dcvc_video}", "--inspect", "{staged:inspect}",
                 "--poses", "{staged:poses_0}", "--poses", "{staged:poses_1}", "--pose-code", "{staged:code}",
                 "--mano-right", "{staged:mano_right}", *[a for e in generic for a in ("--generic", f"{{staged:{e['name']}}}")],
                 *[a for v in videos for a in ("--video", f"{{staged:{v}}}")], "--items", "{items}", "--frames", "{frames}", "--dcvc", "{dcvc}"]
    if args.cover:
        arguments += ["--cover", "--mano-left", "{staged:mano_left}", "--wilor-checkpoint", "{staged:wilor_checkpoint}"]
        arguments += ["--hand-box"] if args.hand_box else []
    ids = [i["id"] for i in items]
    basis = (f"K1f part {args.part}: {ids[0]}'s first 48 window frames through the full path (window decode with the sparse-JPEG gate, "
             "fill masks checked, H3b2's SVT-AV1 low points encoded and accounted by libaom inspect, DCVC-UF HT-L at QP 0, full size and 1/4 "
             "(the stage: seven points) with the per-NAL split, G's networks applied, LPIPS and DISTS on R, then K1c's bound); the stage runs "
             f"{len(ids)} items at 240 frames.")
    spec = {
        "schema": 1, "entrypoint": ["-m", "experiments.visor.k1f"], "environment": h3c2["environment"], "inputs": h3c2["inputs"],
        "arguments": arguments, "staged_inputs": entries, "scale": {"items": {"smoke": ids[0], "full": ",".join(ids)},
                                                                   "frames": {"smoke": 48, "full": 240},
                                                                   "dcvc": {"smoke": "0/1,0/4", "full": "0/1,0/2,21/2,42/2,0/4,21/4,42/4"}},
        "smoke": {"seconds": 600, "representative_basis": basis}, "full": {"seconds": 2700},
        "validator": ["{python}", "-m", "experiments.visor.k1f", "validate"], "validator_seconds": 300, "required_commands": [],
        "budget_seconds": 3600, "deadline": args.deadline, "stall_seconds": 1500, "cpu_threads": 16, "device": "gpu",
        "hosts": h3c2["hosts"], "gpu_models": ["RTX 6000 Ada", "RTX A6000"], "gpu_memory_mib": 32000, "local_storage_gib": 60,
        "gpu_sharing": h3c2["gpu_sharing"],
    }
    out = Path(args.dest) / f"k1f-part{args.part}.json"
    out.write_text(json.dumps(spec, indent=1) + "\n")
    print(out, ids)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
