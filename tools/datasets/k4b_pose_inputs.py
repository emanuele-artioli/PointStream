"""Freeze K4b.2's inputs: the hand-identified players on K4b.1's validated windows, and the job specs.

Reads K4b.1's feature index (locally validated full stages), the user's marks and
the review masks. Writes ``inputs.tar`` (per window: SAM masks, features, K4's
whole-frame detections and K4b.1's window record; ``inputs.json`` with the
players the marks select), the pilot's specification and the parts'.
"""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path
from typing import Any

from experiments.tennis import k4
from experiments.tennis.k4b_pose import RULE, hand_players
from experiments.tennis.k4b_score import load_windows
from tools.datasets.k4_inputs import GPU_MODELS, HOSTS, file_sha256
from tools.datasets.k4b_feature_inputs import DATA_ROOT, PILOT, split

FILES = ("sam-masks.rle", "features.json", "pose.json.zst", "window.json")
REMOTE = f"{DATA_ROOT}/pose-v1"


def collect(index: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Window id -> its validated job directory and record (load_windows checks every identity first)."""
    load_windows(index["jobs"])
    out = {}
    for job in index["jobs"]:
        root = Path(job["directory"])
        summary = json.loads((root / "k4b-features.json").read_text())
        for w in summary["windows"]:
            out[w["id"]] = {"directory": root / "publish" / k4.safe(w["id"]), "window": w,
                            "job": job["job_id"], "git_head": job["git_head"]}
    return out


def build(index_path: Path, marks_path: Path, review_dir: Path, archive: Path) -> dict[str, Any]:
    windows = collect(json.loads(index_path.read_text()))
    review_path = review_dir / "instances.json"
    players = hand_players(json.loads(marks_path.read_text()), json.loads(review_path.read_text()), review_path, review_dir,
                           {k: v["directory"] for k, v in windows.items()})
    if set(players) != set(windows):
        raise ValueError("Every window needs its marked frame")
    record: dict[str, Any] = {"schema": 1, "rule": RULE, "marks_sha256": file_sha256(marks_path),
                              "review_sha256": file_sha256(review_path), "features_index_sha256": file_sha256(index_path),
                              "windows": {}}
    with tarfile.open(archive, "w") as tar:
        for wid in sorted(windows):
            w = windows[wid]
            files = {}
            for name in FILES:
                path = w["directory"] / name
                if name != "window.json" and file_sha256(path) != w["window"]["files"][name]:
                    raise ValueError(f"Changed K4b.1 artifact: {path}")
                files[name] = file_sha256(path)
                tar.add(path, arcname=f"windows/{k4.safe(wid)}/{name}")
            record["windows"][wid] = {"window": w["window"], "job": w["job"], "git_head": w["git_head"], "files": files,
                                      **players[wid]}
        data = (json.dumps(record, indent=1) + "\n").encode()
        info = tarfile.TarInfo("inputs.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return record


def spec(windows: list[dict[str, Any]], clips: dict[str, Any], base: dict[str, Any], archive_sha: str,
         deadline: str, full_seconds: int) -> dict[str, Any]:
    """One part: smoke is its first window whole; full is every window."""
    names: dict[str, str] = {}
    for w in windows:
        names.setdefault(w["clip"], f"c{len(names)}")
    pose = next(r for r in base["staged_inputs"] if r["name"] == "yolo11x_pose")
    staged = [{"name": "inputs", "path": f"{REMOTE}/inputs.tar", "sha256": archive_sha, "extract": True}, pose]
    staged += [{"name": n, "path": clips[c]["path"], "sha256": clips[c]["sha256"]} for c, n in names.items()]
    frames = sum(w["frames"] for w in windows)
    return {"schema": 1, "entrypoint": ["-m", "experiments.tennis.k4b_pose"], "environment": base["environment"],
            "inputs": [{"path": f"{REMOTE}/inputs.tar", "sha256": archive_sha}], "staged_inputs": staged,
            "arguments": ["run", "--inputs", "{staged:inputs}", "--pose", "{staged:yolo11x_pose}", "--max-windows", "{windows}",
                          "--windows", *[w["id"] for w in windows],
                          "--sources", *[f"{{staged:{names[w['clip']]}}}" for w in windows]],
            "scale": {"windows": {"smoke": 1, "full": 0}},
            "smoke": {"seconds": 600, "representative_basis": (
                f"The part's first window whole ({windows[0]['frames']} frames, {windows[0]['id']}) through the full path: "
                "K4's conversion (checked against K4b.1's), every player-frame's crop at x1, x2, x4, the synthetic far "
                f"player and the review images, so the smoke measures the real per-window cost. Full: {len(windows)} "
                f"windows, {frames} frames.")},
            "full": {"seconds": full_seconds},
            "validator": ["{python}", "-m", "experiments.tennis.k4b_pose", "validate"], "validator_seconds": 600,
            "budget_seconds": 600 + full_seconds + 2 * 600 + 600, "deadline": deadline, "stall_seconds": 1200,
            "device": "gpu", "hosts": HOSTS, "gpu_models": GPU_MODELS, "gpu_memory_mib": 12000,
            "cpu_threads": 8, "local_storage_gib": 20, "required_commands": [],
            "gpu_sharing": {"mode": "shared", "margin_mib": 2048, "basis": (
                "YOLO's detections on fixed crops do not depend on wall time; the recorded latency is measured on a "
                "shared GPU and is indicative only")}}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features-index", type=Path, required=True)
    p.add_argument("--marks", type=Path, required=True)
    p.add_argument("--review", type=Path, required=True, help="the review directory holding instances.json")
    p.add_argument("--base-spec", type=Path, required=True, help="one K4b.1 feature spec, for environment, weights and clips")
    p.add_argument("--manifest", type=Path, required=True, help="K4b.1's clip manifest (features-v1/manifest.json)")
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--deadline", required=True)
    p.add_argument("--parts", type=int, default=2)
    p.add_argument("--full-seconds", type=int, default=2700)
    args = p.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)
    archive = args.dest / "inputs.tar"
    record = build(args.features_index, args.marks, args.review, archive)
    sha = file_sha256(archive)
    clips = {c["id"]: {"path": c["staged"]["path"], "sha256": c["sha256"]}
             for c in json.loads(args.manifest.read_text())["clips"]}
    base = json.loads(args.base_spec.read_text())
    plan = [{**v["window"], "players": v["players"]} for v in record["windows"].values()]
    jobs = {"pilot": [w for w in plan if w["id"] in PILOT]}
    jobs.update({f"part{n + 1}": part for n, part in enumerate(split([w for w in plan if w["id"] not in PILOT], args.parts))})
    for name, windows in jobs.items():
        windows = sorted(windows, key=lambda w: (-w["frames"], w["id"]))
        (args.dest / f"{name}.json").write_text(json.dumps(spec(windows, clips, base, sha, args.deadline, args.full_seconds), indent=1) + "\n")
        print(name, len(windows), sum(w["frames"] for w in windows), file_sha256(args.dest / f"{name}.json"))
    print("inputs.tar", sha, {k: len(v["players"]) for k, v in record["windows"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
