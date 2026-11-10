"""One registered TrackNet clip per admitted fleet shard; no scheduling or budget authority.

The input fidelity gate is a frozen receipt from an independent calibration and
complete-cost review. It permits source-negative clips only for fidelity, never
task qualification. Fleet enforces shard deadlines and process-tree budgets.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

from experiments.tennis import k5d_encoder as encoder, k5d_fleet as fleet, k5d_source_oracle as oracle


def build(args: argparse.Namespace) -> argparse.Namespace:
    if oracle.sha(fleet.SOURCE) != fleet.SOURCE_SHA or oracle.sha(fleet.EXPOSURE) != fleet.EXPOSURE_SHA:
        raise ValueError("frozen selection or exposure changed")
    manifest = json.loads(fleet.SOURCE.read_text())
    selected = oracle.select(manifest, args.clip, 0, args.frames)
    stage = os.environ.get("PS_STAGE")
    if not ((stage == "smoke" and args.frames == 16) or
            (stage == "full" and args.frames == selected["label_rows"])):
        raise ValueError("shard requires first 16 frames in smoke and all frames in full")
    encoder.fidelity_gate(Path(args.fidelity_gate), args.fidelity_gate_sha256, fleet.EXPECTED["archive"],
                      purpose=getattr(args, 'fidelity_purpose', 'full-retrospective-benchmark'),
                      clip=args.clip, frames=args.frames)
    data: dict[str, Any] = vars(args).copy()
    stage_dir = Path(os.environ["PS_STAGE_DIR"])
    dispatch_code = json.loads((stage_dir / "dispatch.json").read_text())["code"]
    data["snapshot_identity"] = {name: dispatch_code.get(name) for name in
                                 ("git_head", "source_sha256", "snapshot_sha256", "tracked_worktree_patch_sha256", "included_untracked_sha256")}
    data.update(code_revision=fleet.snapshot_head(stage_dir), manifest=str(fleet.SOURCE), manifest_sha256=fleet.SOURCE_SHA,
                exposure=str(fleet.EXPOSURE), exposure_sha256=fleet.EXPOSURE_SHA, start_frame=0, benchmark=True,
                output=Path(os.environ["PS_SCRATCH_DIR"]) / "publish", feasibility_no_metrics=False)
    for name, digest in fleet.EXPECTED.items():
        data[name + "_sha256"] = digest
    entries = {item["name"]: item for item in json.loads((stage_dir / "staging.json").read_text())["inputs"]}
    for name in ("wasb_code", "balltrack_code"):
        entry = entries[name]
        archive = Path(data[name]).parent / Path(entry["source"]).name
        if oracle.sha(archive) != entry["sha256"]:
            raise ValueError("model archive hash mismatch")
        digest = fleet.archive_tree_sha(archive)
        if oracle.tree_sha(Path(data[name])) != digest:
            raise ValueError("model code tree mismatch")
        data[name + "_sha256"] = digest
    for name, binary in (("ffmpeg", "ffmpeg"), ("encoder", "SvtAv1EncApp"), ("decoder", "dav1d")):
        data[name] = str(Path(sys.prefix) / "bin" / binary)
    return argparse.Namespace(**data)


def validate_smoke() -> None:
    target = Path(os.environ["PS_VALIDATION_PATH"])
    try:
        stage = Path(os.environ["PS_STAGE_DIR"])
        with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"], prefix="k5d-shard-validation-") as tmp:
            with tarfile.open(stage / "published.tar", "r:") as bundle:
                members = bundle.getmembers()
                if len(members) > 200 or sum(m.size for m in members) > 2_000_000_000:
                    raise ValueError("unexpected published smoke archive")
                if any(not (m.isfile() or m.isdir()) for m in members):
                    raise ValueError("unsafe smoke archive")
                bundle.extractall(tmp, filter="data")
            result = encoder.validate(Path(tmp) / "publish")
        if result["frames"] != 16 or result["points"] != 6 or not result["full_metrics"]:
            raise ValueError("incomplete benchmark smoke")
        spec = json.loads((stage.parent / "spec.json").read_text())
        execution = json.loads((stage / "execution.json").read_text())
        dispatch = json.loads((stage / "dispatch.json").read_text())
        command = dispatch["command"]
        clip = command[command.index("--clip") + 1]
        manifest = json.loads(fleet.SOURCE.read_text())
        clips = [c for c in manifest["tracknet"]["clips"] if c["clip_id"] == clip]
        if len(clips) != 1 or clips[0]["label_rows"] != spec["scale"]["frames"]["full"]:
            raise ValueError("full shard frame count differs from registered clip")
        seconds = execution["seconds"]
        now = time.monotonic()
        clock = dispatch["campaign_clock"]
        end = clock["budget_end"]
        if (execution.get("exit_code") != 0 or not isinstance(seconds, (int, float)) or isinstance(seconds, bool)
                or not math.isfinite(seconds) or seconds <= 0 or not (0 < clock["started"] <= clock["phase_started"] <= now < end)):
            raise ValueError("invalid measured smoke cost")
        predicted = seconds * clips[0]["label_rows"] / 16 * 1.25 + 60
        if predicted > spec["full"]["seconds"] or predicted + 60 > end - now:
            raise ValueError("measured full-shard forecast exceeds approved bounds")
        result = {"passed": True, "checks": {"six_wire_and_native_points": True, "independent_source_qualification": True,
                  "all_metrics_and_judges": True, "cost_projection": True},
                  "projected_full_seconds_with_reserve": predicted, "clip": clip}
    except Exception as exc:
        target.write_text(json.dumps({"passed": False, "error": f"{type(exc).__name__}: {exc}"}) + "\n")
        raise
    target.write_text(json.dumps(result, allow_nan=False) + "\n")


def main() -> None:
    if sys.argv[1:] == ["--validate-smoke"]:
        validate_smoke()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "balltrack-code", "balltrack-weight", "wasb-code", "wasb-weights", "wasb-config",
                 "image-ckpt", "video-ckpt", "lpips-backbone", "dists-backbone", "fidelity-gate", "fidelity-gate-sha256", "clip"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--frames", type=int, required=True)
    args = parser.parse_args()
    encoder.run(build(args))


if __name__ == "__main__":
    main()
