"""W4: scoring specs from H5p 4a's, with the receiver masks of H5p 4d(b)'s diagnostic jobs in place of F̂.

Each spec is 4a's ``score-<video>-<coder>.json`` (same pairs, backgrounds, foreground archive, judges) with three
changes: the tag is ``w4-<coder>``, the stage reads the diagnostic's saved tracker output and seed, and the limits
and hosts are W4's. Usage: ``python -m experiments.visor.w4_specs SPECS_4A_DIR OUT_DIR DEADLINE [INPUTS_JSON]``; with the 4d scoring
inputs, only the thinned pairs.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

INBOX = "/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox"

#: H5p 4d(b)'s diagnostic jobs (240 frames, one foreground point each): the tracker's saved masks and the seed.
DIAGNOSTICS: dict[str, dict[str, str]] = {
    "P01_107": {"job": "20261024T084521Z-bc72ff4d", "point": "dcvc-0_4",
                "masks": "f0387790033e9322e26976ea99bb859980e953058e4cbf9c41a81f86f6a78d4a",
                "seed": "227d25b37f3606c8f05e08241b4998988d54fa53e73ad9d331e4a21fbe3b034a"},
    "P02_02": {"job": "20261024T084534Z-c700da3b", "point": "dcvc-0_4",
               "masks": "1e56683ea2683bd32a49037c3707814e391f85977577feaa7dcb2f4b49b96994",
               "seed": "1a93a7b7035e04ed1750ccce78ea821327c7f09abc7462e219446eb7896e1bfe"},
    "P03_10": {"job": "20261024T084607Z-e00640d5", "point": "dcvc-0_4",
               "masks": "e62fda8c84bd5878a26fda741d6345a11e2b297c6a24347d4af149c7bf36881d",
               "seed": "dc7efcb426aa1b6acc4570eb3067fc918adb17ac8f17c458d496757470c8b41f"},
    "P09_106": {"job": "20261024T084640Z-668afb3e", "point": "dcvc-42_2",
                "masks": "340006f74010f76e3cf61124e102bae0c29cb23578f96560fb5b5ae5b133e4ae",
                "seed": "d7caf5293f9cd431271eade2934a1392d649ddc8e9573934a4bc5542e5fb844b"},
}

HOSTS = ["gpu6"]  # RTX 6000 Ada; gpu5's worker heartbeat was stale (835 s) on 2026-10-29, gpu2 times out smokes, gpu4 is reserved
MODELS = ["RTX 6000 Ada"]
SMOKE_FRAMES, SMOKE_PAIRS = 24, 2
#: A full stage's cap: setup plus a pair's scoring (4a: 40 s; the first W4 job: 175 s under host contention).
FULL_BASE_SECONDS, FULL_PAIR_SECONDS = 300, 200


def make_spec(base: dict[str, Any], video: str, coder: str, deadline: str, full_seconds: int) -> dict[str, Any]:
    """4a's spec for one item and coder turned into W4's."""
    diag = DIAGNOSTICS[video]
    s = copy.deepcopy(base)
    a = s["arguments"]
    if "--tag" not in a or a[a.index("--tag") + 1] != f"h5p4-{coder}":
        raise ValueError(f"{video}-{coder}: not a 4a {coder} spec")
    a[a.index("--tag") + 1] = f"w4-{coder}"
    a += ["--receiver-masks", "{staged:receiver_masks}", "--receiver-seed", "{staged:receiver_seed}"]
    run = f"{INBOX}/{diag['job']}/full"
    s["staged_inputs"] += [{"name": "receiver_masks", "path": f"{run}/arm0-{diag['point']}.npz",
                            "sha256": diag["masks"], "extract": False},
                           {"name": "receiver_seed", "path": f"{run}/seed.shape", "sha256": diag["seed"],
                            "extract": False}]
    s["hosts"], s["gpu_models"] = list(HOSTS), list(MODELS)
    s["scale"] = {"limit": {"smoke": SMOKE_FRAMES, "full": 0}, "combos": {"smoke": SMOKE_PAIRS, "full": 0}}
    s["smoke"] = {"seconds": 300, "representative_basis": (
        f"{video} on {SMOKE_FRAMES} frames: {SMOKE_PAIRS} of the {coder} coder's weighted-DISTS hull pairs of H5p 4a, "
        f"composited with the uncorrected receiver masks of H5p 4d(b) job {diag['job']} ({diag['point']}), the seed "
        "the only mask byte billed, containers packed and checked, B2's harness with DISTS")}
    s["full"] = {"seconds": full_seconds}
    s["budget_seconds"] = full_seconds + 300 + 120
    s["deadline"] = deadline
    return s


def thinned_pairs(inputs: Path) -> dict[str, set[tuple[str, str]]]:
    """Per video, the fallback pair set of ``w4_report.thin`` from H5p 4d's ``h5p-scoring-inputs.json``."""
    from experiments.visor.w4_report import thin

    base = json.loads(inputs.read_text())
    out: dict[str, set[tuple[str, str]]] = {}
    for item, selection in base["selection"].items():
        chosen = {(pair, coder) for pair, coder, ms in selection if "wdists_fill" in ms}
        rates = {(r["pair"], r["tag"].removeprefix("h5p4-")): r["kbps"] for st in base["scores"]
                 if st["item"] == item for r in st["rows"]}
        out[item.rsplit("_", 1)[0]] = thin(chosen, rates)
    return out


def main() -> int:
    specs, out, deadline = (Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3])
    keep = thinned_pairs(Path(sys.argv[4])) if len(sys.argv) > 4 else None
    out.mkdir(parents=True, exist_ok=True)
    for video in DIAGNOSTICS:
        for coder in ("gho", "venue"):
            path = specs / f"score-{video}-{coder}.json"
            if path.is_file():
                base = json.loads(path.read_text())
                if keep is not None:
                    a = base["arguments"]
                    wanted = [x for x in a[a.index("--pairs") + 1].split(",") if (x, coder) in keep[video]]
                    if not wanted:
                        continue
                    a[a.index("--pairs") + 1] = ",".join(wanted)
                pairs = len(base["arguments"][base["arguments"].index("--pairs") + 1].split(","))
                spec = make_spec(base, video, coder, deadline, FULL_BASE_SECONDS + FULL_PAIR_SECONDS * pairs)
                (out / f"w4-{video}-{coder}.json").write_text(json.dumps(spec, indent=1))
                print(video, coder, spec["full"]["seconds"], len(spec["arguments"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
