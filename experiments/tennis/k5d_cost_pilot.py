"""Prospective 16→48 frame cost pilot; original 188-frame gate stays failed.

Both stages run the real six-point path. The smoke must exactly match the saved
ed28 source/pixel/quality contract before a separate 48-frame cost forecast.
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
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from experiments.tennis import k5d_encoder as encoder, k5d_fleet as fleet, k5d_source_oracle as oracle
from experiments.tennis.k5d_timing import Timer

ACCEPTED_SHA = "dc1bc14133cf210589f2452785fbaf6d4905e91aae05fdc6d7ac0f25da8ec066"
ACCEPTED_PILOT_SHA = "c46864671036e7435a572b9c05793ad79639927897adce2f89bd0bb62d3fff12"
ACCEPTED_JOB = "20261104T223611Z-ed28cf8e"
CLIP = "tracknet/game10/Clip3"


@contextmanager
def unpack(archive: Path, expected_sha: str, timer: Timer | None = None, prefix: str = "archive") -> Iterator[Path]:
    digest = timer.measure(f"{prefix}.hash", lambda: oracle.sha(archive)) if timer is not None else oracle.sha(archive)
    if digest != expected_sha:
        raise ValueError("archive SHA256 mismatch")
    with tempfile.TemporaryDirectory(prefix="k5d-cost-audit-", dir=os.environ.get("TMPDIR")) as folder:
        with tarfile.open(archive, "r:") as bundle:
            members = bundle.getmembers()
            if len(members) > 250 or sum(m.size for m in members) > 2_000_000_000:
                raise ValueError("unexpected archive size")
            if any(not (m.isfile() or m.isdir()) for m in members):
                raise ValueError("unsupported archive member")
            if timer is None:
                bundle.extractall(folder, filter="data")
            else:
                timer.measure(f"{prefix}.extract", lambda: bundle.extractall(folder, filter="data"))
        yield Path(folder) / "publish"


def reference(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if oracle.sha(path / "pilot.json") != ACCEPTED_PILOT_SHA:
        raise ValueError("accepted 16-frame pilot identity mismatch")
    result = encoder.validate(path)
    if result["frames"] != 16 or result["points"] != 6 or not result["full_metrics"]:
        raise ValueError("accepted pilot is incomplete")
    report = json.loads((path / "pilot.json").read_text())
    source = json.loads((path / "source-qualification.json").read_text())
    if report["clip"] != CLIP or not source["qualified"] or source["device"]["capability"] != [8, 6]:
        raise ValueError("accepted source or GPU class differs")
    return report, source


def compare_contract(prior: dict[str, Any], original: dict[str, Any], current: dict[str, Any],
                     source: dict[str, Any], *, exact: bool) -> None:
    if current["clip"] != CLIP or source["clip"] != CLIP or source["device"]["capability"] != [8, 6]:
        raise ValueError("clip or GPU class differs from accepted reference")
    for key in ("manifest", "archive", "balltrack_weight", "wasb_config", "exposure", "balltrack_code", "wasb_code", "wasb_weights"):
        a, b = original["identities"][key], source["identities"][key]
        if a.get("sha256", a.get("tree_sha256")) != b.get("sha256", b.get("tree_sha256")):
            raise ValueError(f"reference identity differs: {key}")
    for name in ("image_ckpt", "video_ckpt", "lpips_backbone", "dists_backbone"):
        if prior["identities"][name]["sha256"] != current["identities"][name]["sha256"]:
            raise ValueError(f"reference component weight differs: {name}")
    def native_tools(report: dict[str, Any]) -> dict[str, str]:
        return {Path(value["path"]).name: value["sha256"] for value in report["tools"].values()}
    if native_tools(prior) != native_tools(current):
        raise ValueError("reference native tools differ")
    for model, fields in (("lpips", ("heads_sha256", "backbone_sha256")),
                          ("dists", ("weights_sha256", "backbone_sha256"))):
        if any(prior["metric_models"][model][key] != current["metric_models"][model][key] for key in fields):
            raise ValueError(f"reference metric model differs: {model}")
    if (prior["fps"], prior["size"]) != (current["fps"], current["size"]):
        raise ValueError("reference video format differs")
    if len(source["rows"]) < 16:
        raise ValueError("source prefix is incomplete")
    for previous, now in zip(original["rows"], source["rows"][:16]):
        row_fields = ("frame", "member", "source_sha256", "label_kind", "label_x", "label_y")
        if any(previous[k] != now[k] for k in row_fields):
            raise ValueError("reference source frame or label differs")
    if not exact:
        return
    if current["frames"] != 16 or source["rows"] != original["rows"] or source["scores"] != original["scores"]:
        raise ValueError("16-frame source prediction or score differs")
    if current["conversion"]["yuv_sha256"] != prior["conversion"]["yuv_sha256"]:
        raise ValueError("common converted native input differs")
    for a, b in zip(prior["points"], current["points"]):
        for key in ("codec", "point", "native_sha256", "wire_sha256", "encoder_decoded_sha256", "encoder_rendered_sha256"):
            if a[key] != b[key]:
                raise ValueError(f"16-frame native/wire/pixel parity failed: {key}")
        if a["baseline"]["zero_ball_wire_sha256"] != b["baseline"]["zero_ball_wire_sha256"]:
            raise ValueError("16-frame empty-BALL stream differs")
        for arm in ("baseline", "streak"):
            if a[arm]["judges"] != b[arm]["judges"]:
                raise ValueError("16-frame judge output differs")
            for left, right in zip(a[arm]["metrics"], b[arm]["metrics"]):
                keys = ("frame", "frame_psnr", "full_lpips", "full_dists", "crop_lpips", "crop_dists")
                if any(left[key] != right[key] for key in keys):
                    raise ValueError("16-frame metric output differs")
        if a["streak"]["guard_off"] != b["streak"]["guard_off"]:
            raise ValueError("16-frame PSNR guard differs")


def validate_timing(out: Path, frames: int) -> None:
    report = json.loads((out / "pilot.json").read_text())
    geo = report["metric_geometry"]
    source = json.loads((out / "source-qualification.json").read_text())
    if (not report.get("cost_pilot") or report["frames"] != frames or geo["frames"] != frames
            or geo["visible_crop_frames"] != source["scores"]["wasb"]["visible"]
            or (geo["width"], geo["height"], geo["crop_size"], geo["arms_per_point"]) != (1280, 720, 64, 2)):
        raise ValueError("cost-pilot geometry differs")
    data = json.loads((out / "timing.json").read_text())
    events = data["events"]
    names = {e["name"] for e in events}
    required = {"encoder.run_total", "source.oracle_total", "source.image_decode", "source.hash.archive",
                "source.hash.selected_frames", "source.wasb_init", "source.balltrack_init", "source.wasb_judge",
                "source.balltrack_judge", "source.encoder_rgb_decode", "source.rgb_to_yuv", "source.yuv_to_rgb_floor",
                "model.balltrack_init", "model.wasb_init", "model.lpips_init", "model.dists_init",
                "encoder.source_track", "encoder.fit_streak", "source.hash.yuv", "source.conversion_floor", "judge.wasb", "judge.balltrack",
                "bootstrap.source_manifest_hash", "bootstrap.exposure_hash", "bootstrap.accepted16.hash", "bootstrap.accepted16.extract",
                "bootstrap.fidelity_gate_validate", "bootstrap.accepted16_validate", "reference.compare"}
    for kind, point in encoder.POINTS:
        required.update({f"native.{kind}.{point}.total", f"native.{kind}.{point}.yuv_to_rgb", f"receiver.{kind}.{point}",
                         f"metrics.{kind}.{point}.baseline", f"judges.{kind}.{point}.baseline",
                         f"encoder.{kind}.{point}.guard", f"encoder.{kind}.{point}.ball_encode"})
        if kind == "dcvc":
            required.update({f"native.dcvc.{point}.encode", f"native.dcvc.{point}.decode"})
            folder = out / f"{kind}-{point}"
            for filename in ("encode.json", "decode.json", "receiver-native-decoder-report.json"):
                native_report = json.loads((folder / filename).read_text())
                value = native_report.get("load_seconds")
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError("DCVC model load timing missing")
        else:
            record = next(p for p in report["points"] if p["codec"] == kind and p["point"] == point)
            native_timing = record["native_record"].get("native_timing", {})
            if (set(native_timing) != {"encode_seconds", "encode_cpu_seconds", "decode_seconds", "decode_cpu_seconds"}
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
                           for v in native_timing.values())):
                raise ValueError("SVT native timing incomplete")
    if not required.issubset(names):
        raise ValueError(f"missing timing components: {sorted(required - names)}")
    for event in events:
        if not event["completed"] or any(not isinstance(event[k], (int, float)) or not math.isfinite(event[k]) or event[k] < 0
                                          for k in ("wall_seconds", "self_cpu_seconds", "waited_children_cpu_seconds")):
            raise ValueError("invalid timing event")
    judge_calls = len(report["points"]) + sum(not p["streak"]["guard_off"] for p in report["points"])
    if any(sum(e["name"] == f"judge.{judge}" for e in events) != judge_calls for judge in ("wasb", "balltrack")):
        raise ValueError("judge timing coverage differs from scored arms")
    for p in report["points"]:
        path = out / f"{p['codec']}-{p['point']}" / "point.json"
        if json.loads(path.read_text()) != p:
            raise ValueError("point checkpoint differs from pilot report")
        if not p["streak"]["guard_off"]:
            for name in (f"metrics.{p['codec']}.{p['point']}.streak", f"judges.{p['codec']}.{p['point']}.streak"):
                if name not in names:
                    raise ValueError("missing streak timing")


def projection(stage: Path, now: float) -> dict[str, float]:
    execution = json.loads((stage / "execution.json").read_text())
    clock = json.loads((stage / "dispatch.json").read_text())["campaign_clock"]
    sec = execution["seconds"]
    started, phase, end = clock["started"], clock["phase_started"], clock["budget_end"]
    if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in (sec, started, phase, end, now)):
        raise ValueError("invalid measured campaign clock")
    if execution.get("exit_code") != 0 or sec <= 0 or not (0 < started <= phase <= now < end):
        raise ValueError("inconsistent campaign clock")
    estimate = sec * 3 * 1.25 + 60
    if estimate > 900 or estimate + 60 > end - now:
        raise ValueError("48-frame forecast exceeds stage or campaign budget")
    return {"smoke_command_seconds": sec, "multiplier": 3., "margin": 1.25,
            "full_estimate_with_stage_reserve_seconds": estimate, "remaining_seconds": end - now,
            "campaign_reserve_seconds": 60.}


def validate_smoke() -> None:
    target = Path(os.environ["PS_VALIDATION_PATH"])
    try:
        stage = Path(os.environ["PS_STAGE_DIR"])
        staging = json.loads((stage / "staging.json").read_text())
        entries = {x["name"]: x for x in staging["inputs"]}
        published_sha = staging["published"]["sha256"]
        with unpack(Path(entries["accepted16"]["path"]), ACCEPTED_SHA) as old, unpack(stage / "published.tar", published_sha) as new:
            prior, original = reference(old)
            result = encoder.validate(new)
            if result["frames"] != 16 or not result["full_metrics"]:
                raise ValueError("new 16-frame smoke incomplete")
            compare_contract(prior, original, json.loads((new / "pilot.json").read_text()),
                             json.loads((new / "source-qualification.json").read_text()), exact=True)
            validate_timing(new, 16)
        cost = projection(stage, time.monotonic())
        result = {"passed": True, "checks": {"six_complete_real_points": True, "reference_exact_contract": True,
                  "per_component_timing": True, "48_frame_cost_gate": True}, "cost_projection": cost,
                  "reference_job": ACCEPTED_JOB, "scope": "cost pilot only; original 188-frame gate remains failed"}
    except Exception as exc:
        target.write_text(json.dumps({"passed": False, "error": f"{type(exc).__name__}: {exc}"}) + "\n")
        raise
    target.write_text(json.dumps(result, allow_nan=False) + "\n")


def make_spec(base: dict[str, Any], gate_path: str, gate_sha256: str, deadline: str) -> dict[str, Any]:
    """Bind the actually approved gate before rendering a concrete fleet spec."""
    data_root = Path("/home/itec/emanuele/Datasets/pointstream-data")
    if not Path(gate_path).is_absolute() or not Path(gate_path).is_relative_to(data_root):
        raise ValueError("receipt path must be an absolute data-root file")
    if len(gate_sha256) != 64 or any(c not in "0123456789abcdef" for c in gate_sha256):
        raise ValueError("receipt SHA256 must be exact lowercase hex")
    spec = json.loads(json.dumps(base))
    spec["entrypoint"] = ["-m", "experiments.tennis.k5d_cost_pilot"]
    spec["gpu_models"] = ["RTX A6000"]
    spec["arguments"] += ["--accepted16", "{staged:accepted16}", "--fidelity-gate", "{staged:fidelity_gate}",
                          "--fidelity-gate-sha256", gate_sha256]
    spec["scale"]["frames"] = {"smoke": 16, "full": 48}
    accepted = {"name": "accepted16", "path": "/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261104T223611Z-ed28cf8e/smoke/published.tar",
                "sha256": ACCEPTED_SHA}
    gate = {"name": "fidelity_gate", "path": gate_path, "sha256": gate_sha256}
    spec["staged_inputs"].extend([accepted, gate])
    spec["inputs"].extend([{key: row[key] for key in ("path", "sha256")} for row in (accepted, gate)])
    spec["smoke"] = {"seconds": 600, "representative_basis": "First 16 frames of game10/Clip3; all six actual native/PSBL paths, receiver, judges and full metrics, exact pixel/score parity to accepted ed28 and per-component timing. The 48-frame stage has identical code and inputs and three times the frames; validator requires the frozen 1.25x smoke cost gate plus reserves."}
    spec["full"] = {"seconds": 900}
    spec["validator"] = ["{python}", "-m", "experiments.tennis.k5d_cost_pilot", "--validate-smoke"]
    spec["validator_seconds"] = 60
    spec["budget_seconds"] = 1700
    spec["deadline"] = deadline
    return spec


def make_spec_cli() -> None:
    parser = argparse.ArgumentParser(description="Bind an actual approved receipt into the 16/48 cost pilot spec")
    parser.add_argument("--gate-path", required=True)
    parser.add_argument("--gate-sha256", required=True)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[2:])
    base_path = fleet.ROOT / "docs/experiments/k5d-inputs/calibration-fleet-spec-20261104.json"
    base = json.loads(base_path.read_text())
    spec = make_spec(base, args.gate_path, args.gate_sha256, args.deadline)
    from experiments.jobs.inbox import validate_spec
    validate_spec(spec)
    with args.output.open("x") as output:
        json.dump(spec, output, indent=2, allow_nan=False)
        output.write("\n")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--make-spec":
        make_spec_cli()
        return
    if sys.argv[1:] == ["--validate-smoke"]:
        validate_smoke()
        return
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "balltrack-code", "balltrack-weight", "wasb-code", "wasb-weights", "wasb-config", "image-ckpt",
                 "video-ckpt", "lpips-backbone", "dists-backbone", "accepted16", "fidelity-gate", "fidelity-gate-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--frames", type=int, required=True)
    args = parser.parse_args()
    if {"smoke": 16, "full": 48}.get(os.environ.get("PS_STAGE", "")) != args.frames:
        raise ValueError("stage and frame count mismatch")
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    timer = Timer(scratch / "cost-bootstrap-timing.json")
    if timer.measure("bootstrap.source_manifest_hash", lambda: oracle.sha(fleet.SOURCE)) != fleet.SOURCE_SHA:
        raise ValueError("source manifest differs")
    if timer.measure("bootstrap.exposure_hash", lambda: oracle.sha(fleet.EXPOSURE)) != fleet.EXPOSURE_SHA:
        raise ValueError("exposure differs")
    data = vars(args).copy()
    stage = Path(os.environ["PS_STAGE_DIR"])
    dispatch_code = json.loads((stage / "dispatch.json").read_text())["code"]
    data["snapshot_identity"] = {name: dispatch_code.get(name) for name in
                                 ("git_head", "source_sha256", "snapshot_sha256", "tracked_worktree_patch_sha256", "included_untracked_sha256")}
    data.update(code_revision=fleet.snapshot_head(stage), manifest=str(fleet.SOURCE), manifest_sha256=fleet.SOURCE_SHA,
                exposure=str(fleet.EXPOSURE), exposure_sha256=fleet.EXPOSURE_SHA, clip=CLIP, start_frame=0,
                output=scratch / "publish", feasibility_no_metrics=False, cost_pilot=True, timer=timer)
    for name, digest in fleet.EXPECTED.items():
        data[name + "_sha256"] = digest
    # The receipt is required before either stage; it is limited to this clip and <=48 frames.
    timer.measure("bootstrap.fidelity_gate_validate", lambda: encoder.fidelity_gate(
        Path(args.fidelity_gate), args.fidelity_gate_sha256, fleet.EXPECTED["archive"],
        purpose="48-frame-cost-pilot", clip=CLIP, frames=args.frames))
    entries = {item["name"]: item for item in json.loads((stage / "staging.json").read_text())["inputs"]}
    for name in ("wasb_code", "balltrack_code"):
        entry = entries[name]
        archive = Path(data[name]).parent / Path(entry["source"]).name
        if timer.measure(f"bootstrap.hash.{name}_archive", lambda: oracle.sha(archive)) != entry["sha256"]:
            raise ValueError("model code archive identity differs")
        expected = timer.measure(f"bootstrap.tar_tree.{name}", lambda: fleet.archive_tree_sha(archive))
        if timer.measure(f"bootstrap.installed_tree.{name}", lambda: oracle.tree_sha(Path(data[name]))) != expected:
            raise ValueError("cached code tree differs")
        data[name + "_sha256"] = expected
    for name, binary in (("ffmpeg", "ffmpeg"), ("encoder", "SvtAv1EncApp"), ("decoder", "dav1d")):
        data[name] = str(Path(sys.prefix) / "bin" / binary)
    with unpack(Path(args.accepted16), ACCEPTED_SHA, timer, "bootstrap.accepted16") as old:
        prior, original = timer.measure("bootstrap.accepted16_validate", lambda: reference(old))
        timer.measure("encoder.run_total", lambda: encoder.run(argparse.Namespace(**data)))
        current = json.loads((data["output"] / "pilot.json").read_text())
        source = json.loads((data["output"] / "source-qualification.json").read_text())
        timer.measure("reference.compare", lambda: compare_contract(prior, original, current, source, exact=args.frames == 16))
        validate_timing(data["output"], args.frames)
        (data["output"] / "reference-parity.json").write_text(json.dumps({"reference_job": ACCEPTED_JOB,
            "reference_archive_sha256": ACCEPTED_SHA, "mode": "exact-16" if args.frames == 16 else "source-prefix-48",
            "passed": True}) + "\n")


if __name__ == "__main__":
    main()
