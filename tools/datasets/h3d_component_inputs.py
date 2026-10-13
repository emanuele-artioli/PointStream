"""Fleet specifications for H3d's component (docs/experiments.md, 2026-10-12, "H3d component").

    python tools/datasets/h3d_component_inputs.py footage --dest DIR --h3d-dir DIR --videos DIR --sam PT SHA256 \\
        --environment TAR SHA256 --deadline ISO
    python tools/datasets/h3d_component_inputs.py generic --dest DIR --footage-g TAR ... (as above)
    python tools/datasets/h3d_component_inputs.py score --dest DIR --footage ITEM:ARM=TAR ... --generic TAR ... (as above)

``footage`` writes ``h3dc-footage-<V|K>-<video>.json`` for each pilot item and ``h3dc-footage-G.json``; ``generic``
writes ``h3dc-generic-<scale>.json`` (the ¼ and the ½ points) from G's footage archive; ``score`` writes
``h3dc-score-<video>-<point>.json``, one per pilot item and DCVC-UF point, from V's and K's footage and G's networks.
The recordings are chosen here, by sha256 among those of the evaluation's classes (`h3d_component.VERIFIED`), and
hard-linked under ``DEST/inputs/videos``; the window inputs are the oracle's (``--h3d-dir``'s manifest). Content-
addressed manifests, as every H3 step. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

PILOT = ("P01_107", "P02_02", "P03_10", "P09_106")
EXCLUDED = ("P01", "P02", "P03", "P09")
VERIFIED = ("1920x1080@50", "1920x1080@59.94")  # experiments.visor.h3d_component.VERIFIED
MAX_VIDEOS = 6  # h3d_component.FOOTAGE["max_videos"]
SEED = "pointstream-h3d"
POINTS = {"4": ["dcvc@0/4", "dcvc@21/4", "dcvc@42/4"], "2": ["dcvc@0/2", "dcvc@21/2", "dcvc@42/2"]}
GPU_MODELS = ["RTX 6000 Ada", "RTX A6000"]
HOSTS = ["gpu1", "gpu2", "gpu3", "gpu4", "gpu5", "gpu6"]
SHARING = {"mode": "shared", "basis": "SAM 3.1's masks, DCVC-UF's streams and decodes, the seeded network fits and their "
           "scores on the hand region do not depend on wall time; recorded seconds are not evidence"}
MODULE = ["-m", "experiments.visor.h3d_component"]
VALIDATOR = ["{python}", "-m", "experiments.visor.h3d_component", "validate"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def entry(name: str, path: Path, sha: str | None = None, extract: bool = False) -> dict[str, Any]:
    out = {"name": name, "path": str(path), "sha256": sha or file_sha256(path)}
    return {**out, "extract": True} if extract else out


def link(source: Path, target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if os.path.samefile(source, target):
            return target
        raise SystemExit(f"{target} exists and is not {source}")
    os.link(source, target)
    return target


def video_class(path: Path) -> str:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate",
                          "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout.strip()
    width, height, rate = out.split(",")
    num, den = rate.split("/")
    return f"{width}x{height}@{round(int(num) / int(den), 2):g}"


def by_hash(key: str, names: list[str]) -> list[str]:
    return sorted(names, key=lambda v: int(hashlib.sha256(f"{SEED}:{key}:{v}".encode()).hexdigest(), 16))


def recordings(videos_dir: Path) -> dict[str, Path]:
    """Every recording of the evaluation's classes."""
    out = {}
    for path in sorted(videos_dir.glob("*.MP4")):
        if video_class(path) in VERIFIED:
            out[path.stem] = path
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("footage", "generic", "score"))
    parser.add_argument("--dest", required=True)
    parser.add_argument("--h3d-dir", required=True, help="pointstream-data/visor/h3d-2026-10-12 (the oracle's inputs)")
    parser.add_argument("--videos", required=True, help="Datasets/EPIC-KITCHENS-VISOR/epic_kitchens_videos")
    parser.add_argument("--sam", nargs=2, metavar=("PT", "SHA256"), help="SAM 3.1's checkpoint (G1's staged copy)")
    parser.add_argument("--environment", nargs=2, required=True, metavar=("TAR", "SHA256"))
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--steps", default="2000")
    parser.add_argument("--footage-g", help="G's footage job's published.tar")
    parser.add_argument("--footage", action="append", default=[], metavar="VIDEO:ARM=TAR", help="V's and K's footage archives")
    parser.add_argument("--generic", action="append", default=[], help="G's network jobs' published.tar")
    args = parser.parse_args()
    dest = Path(args.dest).resolve()
    oracle = sorted(Path(args.h3d_dir).glob("inputs-*.json"))
    if len(oracle) != 1:
        raise SystemExit(f"{args.h3d_dir}: expected one oracle manifest, found {oracle}")
    window_spec = json.loads((Path(args.h3d_dir) / "h3d-oracle-dcvc-P01_107.json").read_text())
    staged = {e["name"]: e for e in window_spec["staged_inputs"]}
    items = json.loads(oracle[0].read_text())["pilot_items"]
    found = recordings(Path(args.videos))
    chosen: dict[str, list[str]] = {}
    for video in PILOT:
        own = [v for v in found if v.split("_")[0] == video.split("_")[0] and v != video]
        chosen[f"K:{video}"] = sorted(by_hash(f"{items[video]}:K", own)[:MAX_VIDEOS])
    others = [v for v in found if v.split("_")[0] not in EXCLUDED]
    chosen["G"] = sorted(by_hash("generic:G", others)[:MAX_VIDEOS])
    videos_dir = dest / "inputs" / "videos"
    needed = {v for names in chosen.values() for v in names} | set(PILOT)
    for v in sorted(needed):
        key = f"video_{v}"
        if key not in staged:
            staged[key] = entry(key, link(found[v], videos_dir / f"{v}.MP4"))
    if args.sam:
        staged["sam"] = entry("sam", Path(args.sam[0]), args.sam[1])
    if args.footage_g:
        staged["footage_g"] = entry("footage_g", Path(args.footage_g))
    footage_keys = {}
    for spec in args.footage:
        video, _, rest = spec.partition(":")
        arm, _, tar = rest.partition("=")
        key = f"footage_{arm.lower()}_{video}"
        staged[key] = entry(key, Path(tar))
        footage_keys[(video, arm)] = key
    for n, tar in enumerate(args.generic):
        staged[f"generic_{n}"] = entry(f"generic_{n}", Path(tar))
    dest.mkdir(parents=True, exist_ok=True)
    text = json.dumps({"files": {k: {"path": v["path"], "sha256": v["sha256"]} for k, v in sorted(staged.items())},
                       "pilot_items": items, "recordings": chosen}, indent=1, sort_keys=True) + "\n"
    inputs = dest / f"inputs-{hashlib.sha256(text.encode()).hexdigest()[:12]}.json"
    if not inputs.exists():
        inputs.write_text(text)

    def spec(arguments: list[str], names: list[str], scale: dict[str, Any], basis: str, memory: int) -> dict[str, Any]:
        return {
            "schema": 1, "entrypoint": MODULE, "environment": {"path": args.environment[0], "sha256": args.environment[1]},
            "inputs": [{"path": str(inputs), "sha256": file_sha256(inputs)}], "arguments": arguments,
            "staged_inputs": [staged[k] for k in names], "scale": scale, "smoke": {"seconds": 600, "representative_basis": basis},
            "full": {"seconds": 2700}, "validator": VALIDATOR, "validator_seconds": 300, "required_commands": [],
            "budget_seconds": 4500, "deadline": args.deadline, "stall_seconds": 1800, "cpu_threads": 16, "device": "gpu",
            "hosts": HOSTS, "gpu_models": GPU_MODELS, "gpu_memory_mib": memory, "local_storage_gib": 80, "gpu_sharing": SHARING,
        }

    def videos_args(names: list[str]) -> list[str]:
        return [a for v in names for a in ("--video", f"{{staged:video_{v}}}")]

    written = []
    if args.phase == "footage":
        if not args.sam:
            raise SystemExit("footage needs --sam")
        jobs = [(f"V-{v}", items[v], "V", [v]) for v in PILOT] + [(f"K-{v}", items[v], "K", chosen[f"K:{v}"]) for v in PILOT] + \
            [("G", "generic", "G", chosen["G"])]
        for label, target, arm, names in jobs:
            arguments = ["footage", "--target", target, "--arm", arm, *videos_args(names), "--eval-set", "{staged:eval_set}",
                         "--sam-checkpoint", "{staged:sam}", "--segments", "{segments}"]
            basis = (f"H3d footage {label}: the first 2 of its 24 content-blind segments through the full path (decode with the "
                     "verified-class check, JPEG chunks at G1's analysis size, SAM 3.1 'hand' and 'arm', packed masks); the stage "
                     f"runs all 24 over {len(names)} recording(s).")
            body = spec(arguments, ["eval_set", "sam", *[f"video_{v}" for v in names]],
                        {"segments": {"smoke": 2, "full": 24}}, basis, 40000)
            (dest / f"h3dc-footage-{label}.json").write_text(json.dumps(body, indent=1) + "\n")
            written.append(f"h3dc-footage-{label}.json")
    coder = ["--image-ckpt", "{staged:dcvc_image}", "--video-ckpt", "{staged:dcvc_video}", "--lpips-backbone", "{staged:lpips}",
             "--steps", "{steps}", "--segments", "{segments}"]
    if args.phase == "generic":
        if not args.footage_g:
            raise SystemExit("generic needs --footage-g")
        for scale, points in POINTS.items():
            arguments = ["generic", "--footage", "{staged:footage_g}", *videos_args(chosen["G"]), "--point", "{points}", *coder]
            basis = (f"H3d G networks at {points[0]}: G's footage (first 2 segments) re-decoded and checked against the footage "
                     "job's sha256, coded by DCVC-UF HT-L, upscaled, cropped on R; the network fitted 20 steps; the stage fits "
                     f"{', '.join(points)} on all 24 segments for {args.steps} steps.")
            body = spec(arguments, ["footage_g", "dcvc_image", "dcvc_video", "lpips", *[f"video_{v}" for v in chosen["G"]]],
                        {"points": {"smoke": points[0], "full": ",".join(points)}, "steps": {"smoke": 20, "full": int(args.steps)},
                         "segments": {"smoke": 2, "full": 1000}}, basis, 32000)
            (dest / f"h3dc-generic-{scale}.json").write_text(json.dumps(body, indent=1) + "\n")
            written.append(f"h3dc-generic-{scale}.json")
    if args.phase == "score":
        if not args.generic or len(footage_keys) != 2 * len(PILOT):
            raise SystemExit("score needs --generic and V's and K's --footage for every pilot video")
        generic = [f"generic_{n}" for n in range(len(args.generic))]
        for video in PILOT:
            names = [video, *chosen[f"K:{video}"]]
            for point in [p for ps in POINTS.values() for p in ps]:
                arguments = ["score", "--eval-set", "{staged:eval_set}", "--archive", "{staged:dense_archive}",
                             "--masks", "{staged:fill_masks}", "--mask-record", "{staged:fill_record}", "--items", "{items}",
                             "--frames", "{frames}", "--dists-backbone", "{staged:dists}", "--point", point,
                             "--footage", f"{{staged:{footage_keys[(video, 'V')]}}}", "--footage", f"{{staged:{footage_keys[(video, 'K')]}}}",
                             *[a for g in generic for a in ("--generic", f"{{staged:{g}}}")], *videos_args(names), *coder]
                basis = (f"H3d component score: {items[video]} at {point}, its first 48 window frames through the full path "
                         "(window decode with the sparse-JPEG gate, the point coded with its rate on R, V's and K's footage "
                         "(first 2 segments) re-decoded, checked and coded, V-L/M/S and K fitted 20 steps, G's network loaded "
                         "by sha256, every output scored); the stage runs 240 frames, 24 segments, "
                         f"{args.steps} steps.")
                needed_names = ["eval_set", "dense_archive", "fill_masks", "fill_record", "dists", "dcvc_image", "dcvc_video",
                                "lpips", footage_keys[(video, "V")], footage_keys[(video, "K")], *generic, *[f"video_{v}" for v in names]]
                body = spec(arguments, needed_names, {"items": {"smoke": items[video], "full": items[video]},
                                                      "frames": {"smoke": 48, "full": 240}, "steps": {"smoke": 20, "full": int(args.steps)},
                                                      "segments": {"smoke": 2, "full": 1000}}, basis, 32000)
                label = f"h3dc-score-{video}-{point.replace('@', '').replace('/', '_')}.json"
                (dest / label).write_text(json.dumps(body, indent=1) + "\n")
                written.append(label)
    print(json.dumps({"inputs": str(inputs), "specs": written, "recordings": chosen}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
