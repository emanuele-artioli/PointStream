"""Source-image judge qualification for registered TrackNet clips.

Only source JPEGs are judged here, using OpenCV's JPEG-to-RGB path. This is not
K5c's video-transform parity or a codec quality result. No model is loaded at
module import or on --help.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

from experiments.tennis.k5d_analysis import score_frames


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha(path: Path) -> str:
    """Digest names and content hashes of the Python/YAML model code tree."""
    files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix in (".py", ".yaml", ".yml"))
    if not files:
        raise ValueError("empty model code tree")
    h = hashlib.sha256()
    for p in files:
        h.update(p.relative_to(path).as_posix().encode() + b"\0" + bytes.fromhex(sha(p)))
    return h.hexdigest()


def bind_config(wasb_code: Path, expected_hash: str) -> None:
    if sha(wasb_code / "src/configs/dataset/tennis.yaml") != expected_hash:
        raise ValueError("WASB tree dataset config differs from declared config")


def validate_output(output: Path, scratch: Path, input_paths: list[Path]) -> None:
    target, root = output.resolve(), scratch.resolve()
    if not target.is_relative_to(root) or target == root:
        raise ValueError("output must be beneath PS_SCRATCH_DIR")
    for path in input_paths:
        source = path.resolve()
        if target == source or (source.is_dir() and target.is_relative_to(source)):
            raise ValueError("output overlaps an input")


def select(manifest: dict[str, Any], clip_id: str, start: int, frames: int) -> dict[str, Any]:
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported selection manifest")
    matches = [c for c in manifest["tracknet"]["clips"] if c["clip_id"] == clip_id]
    if len(matches) != 1 or "full_retrospective_benchmark" not in matches[0]["roles"]:
        raise ValueError("clip is not uniquely registered")
    clip = matches[0]
    if start < 0 or frames <= 0 or start + frames > clip["label_rows"]:
        raise ValueError("range outside registered clip")
    return clip


def validate_rows(rows: list[tuple[str, int, float, float]], expected: dict[str, Any]) -> None:
    if len(rows) != expected["label_rows"] or len(rows) != expected["jpg_entries"]:
        raise ValueError("label and JPEG row count mismatch")
    counts = {str(i): 0 for i in range(4)}
    names: set[str] = set()
    for name, kind, x, y in rows:
        if (not name or Path(name).name != name or name in names or kind not in range(4)
                or not math.isfinite(x) or not math.isfinite(y)):
            raise ValueError("invalid or duplicate source label row")
        if kind in (1, 2) and not (0 <= x < 1280 and 0 <= y < 720):
            raise ValueError("visible label outside source geometry")
        counts[str(kind)] += 1
        names.add(name)
    if counts != expected["visibility_counts"]:
        raise ValueError("visibility counts differ from manifest")


def validate_report(report: dict[str, Any]) -> dict[str, Any]:
    """Recompute all counts from saved image-only predictions and registered labels."""
    if report.get("schema_version") != 1 or not report.get("clip") or not report.get("exposure"):
        raise ValueError("missing report provenance")
    device = report.get("device", {})
    if (device.get("visible_count") != 1 or len(device.get("affinity", [])) != 1 or not device.get("name")
            or device.get("models_on_cuda") != ["wasb-step1", "balltrack"]
            or len(device.get("capability", [])) != 2):
        raise ValueError("invalid device evidence")
    identities = report.get("identities", {})
    for name in ("manifest", "archive", "balltrack_weight", "wasb_config", "exposure", "balltrack_code", "wasb_code", "wasb_weights"):
        identity = identities.get(name, {})
        digest = identity.get("tree_sha256", identity.get("sha256", ""))
        if not identity.get("path") or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("missing identity")
    rows = report["rows"]
    start, frames = report["start_frame"], report["frames"]
    if len(rows) != frames or [r["frame"] for r in rows] != list(range(start, start + frames)):
        raise ValueError("missing or duplicate frame rows")
    if any(not r.get("member") or len(r["source_sha256"]) != 64
           or any(c not in "0123456789abcdef" for c in r["source_sha256"]) for r in rows):
        raise ValueError("missing image hash")
    labels = [(r["label_kind"], r["label_x"], r["label_y"]) for r in rows]
    scores: dict[str, Any] = {}
    for judge in ("wasb", "balltrack"):
        predictions = [(r[judge]["detected"], r[judge]["x"], r[judge]["y"]) for r in rows]
        scores[judge] = score_frames(labels, predictions)
        if scores[judge] != report["scores"][judge]:
            raise ValueError("saved count mismatch")
    recall = scores["wasb"]["recall10"]
    qualified = recall is not None and recall >= 0.8
    if report["qualified"] != qualified:
        raise ValueError("incorrect source qualification")
    return {"qualified": qualified, "scores": scores}


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    # Imports remain here so CLI discovery never loads model frameworks.
    import torch
    from experiments.tennis import k5b_trackers as trk

    if len(os.sched_getaffinity(0)) != 1 or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("requires one visible CUDA device and single-CPU affinity")
    identities = {}
    for name in ("manifest", "archive", "balltrack_weight", "wasb_config", "exposure"):
        path = Path(getattr(args, name))
        digest = sha(path)
        if digest != getattr(args, name + "_sha256"):
            raise ValueError(f"{name} SHA256 mismatch")
        identities[name] = {"path": str(path), "sha256": digest}
    for name in ("balltrack_code", "wasb_code"):
        path = Path(getattr(args, name))
        digest = tree_sha(path)
        if digest != getattr(args, name + "_sha256"):
            raise ValueError(f"{name} tree SHA256 mismatch")
        identities[name] = {"path": str(path), "tree_sha256": digest}
    bind_config(Path(args.wasb_code), identities["wasb_config"]["sha256"])
    weights = Path(args.wasb_weights) / "wasb_tennis_best.pth.tar"
    weight_hash = sha(weights)
    if weight_hash != args.wasb_weights_sha256:
        raise ValueError("WASB weights SHA256 mismatch")
    identities["wasb_weights"] = {"path": str(weights), "sha256": weight_hash}
    manifest = json.loads(Path(args.manifest).read_text())
    if manifest["tracknet"]["source_archive_sha256"] != identities["archive"]["sha256"]:
        raise ValueError("archive differs from selection")
    expected = select(manifest, args.clip, args.start_frame, args.frames)
    exposure = json.loads(Path(args.exposure).read_text())
    game = args.clip.split("/")[1]
    if (game not in exposure.get("test_games", []) or game in exposure.get("train_games", [])
            or exposure.get("checkpoint_sha256_in_original_k5b_run") != weight_hash
            or exposure.get("files", {}).get("src/configs/dataset/tennis.yaml", {}).get("sha256")
            != identities["wasb_config"]["sha256"]):
        raise ValueError("WASB training provenance mismatch")
    game_number = int(game[4:])
    found = [c for c in trk.tracknet_clips(Path(args.archive), [game_number]) if c["id"] == args.clip]
    if len(found) != 1:
        raise ValueError("clip missing or duplicate in archive")
    clip = found[0]
    validate_rows(clip["rows"], expected)
    with zipfile.ZipFile(args.archive) as z:
        if hashlib.sha256(z.read(expected["label"])).hexdigest() != expected["label_sha256"]:
            raise ValueError("label hash mismatch")
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive members")
        selected_rows = clip["rows"][args.start_frame:args.start_frame + args.frames]
        hashes = [hashlib.sha256(z.read(f"{clip['dir']}/{row[0]}")).hexdigest() for row in selected_rows]
    subclip = {**clip, "rows": selected_rows}
    labels = trk.tracknet_labels(subclip)
    images = trk.tracknet_frames(Path(args.archive), subclip)
    zoo = trk.Zoo(Path(args.wasb_code), Path(args.wasb_weights), names=("wasb-step1",))
    bt = trk.BallTrack(Path(args.balltrack_code), Path(args.balltrack_weight))
    if next(zoo.settings["wasb-step1"]["net"].parameters()).device.type != "cuda" or next(bt.net.parameters()).device.type != "cuda":
        raise ValueError("model is not on CUDA")
    predictions = {"wasb": zoo.run("wasb-step1", images), "balltrack": bt.run(images)}
    scores = {name: score_frames(labels, preds) for name, preds in predictions.items()}
    rows = []
    for i, (label, digest) in enumerate(zip(labels, hashes)):
        row: dict[str, Any] = {"frame": args.start_frame + i, "member": f"{clip['dir']}/{selected_rows[i][0]}",
                               "source_sha256": digest, "label_kind": label[0], "label_x": label[1], "label_y": label[2]}
        for name, preds in predictions.items():
            detected, x, y = preds[i]
            row[name] = {"detected": bool(detected), "x": float(x), "y": float(y)}
        rows.append(row)
    recall = scores["wasb"]["recall10"]
    props = torch.cuda.get_device_properties(0)
    device_uuid = getattr(props, "uuid", None)
    report = {"schema_version": 1, "clip": args.clip, "start_frame": args.start_frame, "frames": args.frames,
              "scope": "retrospective full clip" if args.start_frame == 0 and args.frames == expected["label_rows"]
              else "development subrange; not an independent clip", "identities": identities,
              "source_transform": "OpenCV JPEG decode BGR converted to RGB; no video transform; no K5c parity claim",
              "label_policy": "TrackNet 1/2 visible, 0 absent, 3 occluded unknown",
              "device": {"name": torch.cuda.get_device_name(0), "visible_count": torch.cuda.device_count(),
                         "affinity": sorted(os.sched_getaffinity(0)),
                         "capability": list(torch.cuda.get_device_capability(0)),
                         "uuid": str(device_uuid) if device_uuid is not None else None,
                         "models_on_cuda": ["wasb-step1", "balltrack"]},
              "tools": {"python": sys.version, "torch": torch.__version__, "torch_cuda": torch.version.cuda}, "exposure": exposure,
              "elapsed_seconds": time.monotonic() - started, "rows": rows, "scores": scores,
              "qualified": recall is not None and recall >= 0.8}
    validate_report(report)
    return report


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--validate-report":
        result = validate_report(json.loads(Path(sys.argv[2]).read_text()))
        print(json.dumps(result, allow_nan=False))
        return 0 if result["qualified"] else 1
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "archive", "balltrack-code", "balltrack-weight", "wasb-code", "wasb-weights", "wasb-config", "exposure"):
        parser.add_argument("--" + name, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("PS_SCRATCH_DIR"):
        parser.error("PS_SCRATCH_DIR is required")
    try:
        validate_output(args.output, Path(os.environ["PS_SCRATCH_DIR"]), [Path(getattr(args, key)) for key in
                        ("manifest", "archive", "balltrack_code", "balltrack_weight", "wasb_code", "wasb_weights", "wasb_config", "exposure")])
        local_root = os.environ.get("PS_LOCAL_ROOT")
        if local_root and not Path(args.archive).resolve().is_relative_to(Path(local_root).resolve()):
            raise ValueError("archive is outside PS_LOCAL_ROOT")
    except ValueError as exc:
        parser.error(str(exc))
    try:
        report = run(args)
    except Exception as exc:
        report = {"schema_version": 1, "qualified": False, "error": f"{type(exc).__name__}: {exc}"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        json.dump(report, out, allow_nan=False, indent=2)
        out.write("\n")
    return 0 if report["qualified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
