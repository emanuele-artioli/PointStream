"""Freeze K6's inputs (docs/experiments.md, 2026-10-21 K6).

``features``: K4's nine broadcast windows (the outro card left out, as K4) as a K4b.1 feature manifest, each clip
listed with frame 120 as its only review frame, which ``k4b_features.window_span`` places at frames 0-239 (K4's
window); and the feature job's specification, K4b.1's in everything but the windows. No labels are involved.

``oracle``: the oracle's inputs archive (K4b.1's and K4's validated window features, K8b's fits inside each
window, the source clips) and its job specifications: the pilot, then parts (`plan`).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

from experiments.tennis.k4b_features import WINDOW, plan_windows
from tools.datasets.k4_inputs import GPU_MODELS, HOSTS, SHARING, file_sha256

DATA_ROOT = "/home/itec/emanuele/Datasets/pointstream-data/tennis/k6-2026-10-21"
#: K4's outro card (`k4 report --exclude`): no tennis, no person.
K4_EXCLUDED = ("broadcast/medvedev_struff/042",)
#: The frame whose `window_span` is K4's window (frames 0-239) in a clip of at least 240 frames.
K4_FRAME = WINDOW // 2


def k4_manifest(windows: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
    clips = []
    for w in windows["windows"]:
        if w["dataset"] != "broadcast" or w["id"] in K4_EXCLUDED:
            continue
        staged = inputs["files"][w["source"]["staged"]]
        match, clip = w["id"].split("/")[1:]
        clips.append({"match": match, "clip": clip, "id": f"{match}/{clip}", "path": staged["path"], "width": w["size"][0],
                      "height": w["size"][1], "fps": w["fps"], "frames": w["source_frames"], "sha256": staged["sha256"],
                      "staged": {"name": "clip", "path": staged["path"], "sha256": staged["sha256"]},
                      "review_frames": [K4_FRAME], "k4_window": w["id"]})
    manifest = {"schema": 1, "contains_human_marks": False, "purpose": "K6: K4's broadcast windows through K4b.1's features",
                "window_rule": "frame 120 of each clip, so window_span gives K4's first 240 frames", "clips": clips}
    for w in plan_windows(manifest).values():
        if (w["start"], w["frames"]) != (0, WINDOW):
            raise ValueError(f"{w['id']} is not K4's window")
    return manifest


def feature_spec(manifest: dict[str, Any], manifest_sha: str, base: dict[str, Any], deadline: str) -> dict[str, Any]:
    """K4b.1's feature job (`tools.datasets.k4b_feature_inputs.spec`) on K4's windows, one part; the manifest's
    identity is checked when the full stage is validated locally (`k4b_features validate --manifest`)."""
    from tools.datasets.k4b_feature_inputs import spec

    remote = f"{DATA_ROOT}/features/manifest.json"
    windows = sorted(plan_windows(manifest).values(), key=lambda w: (-w["frames"], w["id"]))
    clips = {c["id"]: c for c in manifest["clips"]}
    return spec(windows, clips, base, remote, manifest_sha, deadline, 2700)


FEATURE_FILES = ("features.json", "sam-masks.rle", "pose.json.zst")
MIN_SCORED = 24
REMOTE = f"{DATA_ROOT}/oracle"


def registrations(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """Clip id -> {sha256, fits: {clip frame: homography}} (K8b's frames with fits true)."""
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        for clip in json.loads(path.read_text())["clips"]:
            out[clip["id"]] = {"sha256": clip["sha256"],
                               "fits": {r["frame"]: r["homography"] for r in clip["registration"] if r["fits"]}}
    return out


def oracle_windows(jobs: list[Path], regs: dict[str, dict[str, Any]], sources: dict[str, dict[str, str]],
                   k4_ids: dict[str, str]) -> dict[str, dict[str, Any]]:
    """Every window of the validated feature jobs with its files, K8b fits inside its span and its source clip."""
    out: dict[str, dict[str, Any]] = {}
    for job in jobs:
        summary = json.loads((job / "k4b-features.json").read_text())
        git_head = json.loads((job / "ready.json").read_text())["git_head"]
        for w in summary["windows"]:
            root = job / "x" / "publish" / w["id"].replace("/", "__")
            files = {n: file_sha256(root / n) for n in FEATURE_FILES}
            if any(files[n] != w["files"][n] for n in FEATURE_FILES):
                raise ValueError(f"Changed feature artifact: {root}")
            reg = regs[w["clip"]]
            if reg["sha256"] != w["sha256"] or sources[w["clip"]]["sha256"] != w["sha256"]:
                raise ValueError(f"{w['clip']}: K8b, the source and the features name different files")
            fits = {str(f): h for f, h in reg["fits"].items() if w["start"] <= f < w["start"] + w["frames"]}
            out[w["id"]] = {"window": {k: w[k] for k in ("id", "clip", "frame", "start", "frames", "target", "fps", "sha256")},
                            "files": files, "fits": fits, "k4_window": k4_ids.get(w["clip"]), "root": str(root),
                            "feature_job": job.name, "feature_git_head": git_head, "source": sources[w["clip"]]}
    return out


def order_key(wid: str) -> str:
    return hashlib.sha256(f"pointstream-k6:{wid}".encode()).hexdigest()


def plan(windows: dict[str, dict[str, Any]], part_size: int) -> dict[str, list[str]]:
    """Pilot: the 2 of K4's windows first by hash among those scored on every frame; then K4's other windows (all 9:
    amendment 2 needs each), then K4b.1's windows with >= MIN_SCORED scored frames, each group by hash, in parts."""
    k4w = sorted((w for w in windows if windows[w]["k4_window"]), key=order_key)
    full = [w for w in k4w if len(windows[w]["fits"]) == windows[w]["window"]["frames"]]
    pilot = full[:2]
    rest_k4 = [w for w in k4w if w not in pilot]
    k4b = sorted((w for w in windows if not windows[w]["k4_window"] and len(windows[w]["fits"]) >= MIN_SCORED), key=order_key)
    out = {"pilot": pilot}
    for group, ids in (("k4", rest_k4), ("k4b", k4b)):
        for n in range(0, len(ids), part_size):
            out[f"{group}-part{n // part_size + 1}"] = ids[n:n + part_size]
    return out


def oracle_spec(ids: list[str], windows: dict[str, dict[str, Any]], weights: dict[str, dict[str, str]], environment: dict[str, str],
                archive_sha: str, deadline: str, full_seconds: int) -> dict[str, Any]:
    names: dict[str, str] = {}
    for w in ids:
        names.setdefault(windows[w]["window"]["clip"], f"c{len(names)}")
    staged = [{"name": "inputs", "path": f"{REMOTE}/inputs.tar", "sha256": archive_sha, "extract": True}]
    staged += [{"name": n, **r} for n, r in weights.items()]
    staged += [{"name": n, "path": windows[next(w for w in ids if windows[w]["window"]["clip"] == c)]["source"]["path"],
                "sha256": windows[next(w for w in ids if windows[w]["window"]["clip"] == c)]["source"]["sha256"]} for c, n in names.items()]
    first = ids[0]
    return {
        "schema": 1, "entrypoint": ["-m", "experiments.tennis.k6"], "environment": environment,
        "inputs": [{"path": f"{REMOTE}/inputs.tar", "sha256": archive_sha}], "staged_inputs": staged,
        "arguments": ["run", "--inputs", "{staged:inputs}", "--pose", "{staged:yolo11x_pose}", "--inspect", "{staged:inspect}",
                      "--image-ckpt", "{staged:dcvc_image}", "--image-sha256", weights["dcvc_image"]["sha256"],
                      "--video-ckpt", "{staged:dcvc_video}", "--video-sha256", weights["dcvc_video"]["sha256"],
                      "--lpips-backbone", "{staged:lpips}", "--dists-backbone", "{staged:dists}",
                      "--frames", "{frames}", "--max-windows", "{windows}", "--windows", *ids,
                      "--sources", *[f"{{staged:{names[windows[w]['window']['clip']]}}}" for w in ids]],
        "scale": {"frames": {"smoke": 48, "full": 240}, "windows": {"smoke": 1, "full": 0}},
        "smoke": {"seconds": 600, "representative_basis": (
            f"The part's first window ({first}), its first 48 frames, through the full path: K4's conversion, (c)+(d) "
            "players, pose, K4b.3's four codes, all ten anchors (SVT-AV1 with aom-inspect accounting, DCVC-UF with its "
            f"bit estimate), the nine oracle composites and every judge. Full: {len(ids)} windows of 240 frames.")},
        "full": {"seconds": full_seconds},
        "validator": ["{python}", "-m", "experiments.tennis.k6", "validate"], "validator_seconds": 600,
        "budget_seconds": 600 + full_seconds + 2 * 600 + 600, "deadline": deadline, "stall_seconds": 1500,
        "device": "gpu", "hosts": HOSTS, "gpu_models": GPU_MODELS, "gpu_memory_mib": 24000, "cpu_threads": 16,
        "local_storage_gib": 60, "required_commands": [], "gpu_sharing": {**SHARING, "margin_mib": 4096},
    }


def build_oracle(args: argparse.Namespace) -> int:
    """inputs.tar (per window: the feature files; inputs.json: window records, K8b fits, sources) and the job specs."""
    regs = registrations(args.registration)
    sources: dict[str, dict[str, str]] = {}
    k4_ids: dict[str, str] = {}
    for c in json.loads(args.k4b_manifest.read_text())["clips"]:
        sources[c["id"]] = {"path": c["staged"]["path"], "sha256": c["sha256"]}
    for c in json.loads(args.k4_manifest.read_text())["clips"]:
        sources[c["id"]] = {"path": c["staged"]["path"], "sha256": c["sha256"]}
        k4_ids[c["id"]] = c["k4_window"]
    windows = oracle_windows(args.jobs, regs, sources, k4_ids)
    record: dict[str, Any] = {"schema": 1, "registration_sha256": {str(p): file_sha256(p) for p in args.registration},
                              "k4_manifest_sha256": file_sha256(args.k4_manifest),
                              "k4b_manifest_sha256": file_sha256(args.k4b_manifest), "windows": {}}
    archive = args.dest / "inputs.tar"
    with tarfile.open(archive, "w") as tar:
        for wid in sorted(windows):
            w = windows[wid]
            for name in FEATURE_FILES:
                tar.add(Path(w["root"]) / name, arcname=f"windows/{wid.replace('/', '__')}/{name}")
            record["windows"][wid] = {k: v for k, v in w.items() if k != "root"}
        data = (json.dumps(record) + "\n").encode()
        info = tarfile.TarInfo("inputs.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    sha = file_sha256(archive)
    base = json.loads(args.base_spec.read_text())
    dcvc = {r["name"]: {"path": r["path"], "sha256": r["sha256"]} for r in base["staged_inputs"] if r["name"].startswith("dcvc_")}
    weights = {**json.loads(args.weights.read_text()), **dcvc}
    parts = plan(windows, args.part_size)
    for name, ids in parts.items():
        spec = oracle_spec(ids, windows, weights, base["environment"], sha, args.deadline, args.full_seconds)
        (args.dest / f"{name}.json").write_text(json.dumps(spec, indent=1) + "\n")
        print(name, ids, file_sha256(args.dest / f"{name}.json"))
    (args.dest / "plan.json").write_text(json.dumps({"inputs_sha256": sha, "parts": parts,
                                                     "scored": {w: len(windows[w]["fits"]) for w in windows}}, indent=1) + "\n")
    print("inputs.tar", sha, len(windows), "windows")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    o = sub.add_parser("oracle")
    o.add_argument("--jobs", type=Path, nargs="+", required=True, help="locally validated feature job directories "
                   "(ready.json, k4b-features.json, x/publish)")
    o.add_argument("--registration", type=Path, nargs="+", required=True)
    o.add_argument("--k4b-manifest", type=Path, required=True)
    o.add_argument("--k4-manifest", type=Path, required=True, help="K6's features/manifest.json")
    o.add_argument("--base-spec", type=Path, required=True, help="K5's pilot spec (environment, DCVC-UF checkpoints)")
    o.add_argument("--weights", type=Path, required=True, help="JSON: name -> {path, sha256} for yolo11x_pose, inspect, lpips, dists")
    o.add_argument("--dest", type=Path, required=True)
    o.add_argument("--deadline", required=True)
    o.add_argument("--part-size", type=int, default=3)
    o.add_argument("--full-seconds", type=int, default=2700)
    f = sub.add_parser("features")
    f.add_argument("--k4-windows", type=Path, required=True, help="K4's windows.json")
    f.add_argument("--k4-inputs", type=Path, required=True, help="K4's inputs-*.json")
    f.add_argument("--base-spec", type=Path, required=True, help="one K4b.1 feature spec (features-v2), for environment and weights")
    f.add_argument("--dest", type=Path, required=True)
    f.add_argument("--deadline", required=True)
    args = p.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    if args.command == "oracle":
        return build_oracle(args)
    manifest = k4_manifest(json.loads(args.k4_windows.read_text()), json.loads(args.k4_inputs.read_text()))
    path = args.dest / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1) + "\n")
    sha = file_sha256(path)
    spec = feature_spec(manifest, sha, json.loads(args.base_spec.read_text()), args.deadline)
    (args.dest / "features-job.json").write_text(json.dumps(spec, indent=1) + "\n")
    print("manifest", sha, len(manifest["clips"]), "clips;", "spec", file_sha256(args.dest / "features-job.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
