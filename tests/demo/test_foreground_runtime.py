from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pytest

from demo.experiments.foreground_runtime import (
    av1_square_args,
    check_campaign_budgets,
    check_job_budget,
    classify_execution_provider,
    combined_count,
    coordinates_agree,
    run_bounded_python,
    stage_rates,
    stages_reproduce_total,
    PlanCapError,
)
from demo.experiments.foreground_smoke import (
    RECORDINGS,
    _require_job_output,
    audit_dataset,
    build_parser,
    compare_packet_cuts,
    freeze_training_split,
    run_packet,
    validate_limits,
)
from demo.experiments.foreground_validate import validate_report
from demo.pipeline.foreground_codec_v2 import TrackedHand
from experiments.jobs import fleet, inbox, monitor


@pytest.mark.parametrize("argv", [
    ["audit", "--output-dir", "out", "--max-candidates-per-recording", "25"],
    ["packet", "--input", "in.json", "--output-dir", "out", "--max-frames", "9"],
    ["fit", "--manifest", "in.json", "--output-dir", "out", "--max-steps", "121"],
    ["fit", "--manifest", "in.json", "--output-dir", "out", "--seconds-per-arm", "121"],
    ["profile", "--images", "images", "--output-dir", "out", "--max-images", "17"],
    ["profile", "--images", "images", "--output-dir", "out", "--sam-frames", "31"],
    ["profile", "--images", "images", "--output-dir", "out", "--sam-frames", "8"],
    ["compare", "--manifest", "in.json", "--output-dir", "out", "--resolutions", "128", "256"],
])
def test_cli_rejects_any_work_larger_than_smoke_limits(argv):
    args = build_parser().parse_args(argv)
    with pytest.raises(ValueError):
        validate_limits(args)


def test_default_cli_limits_are_exactly_the_plan_caps():
    cases = [
        ["audit", "--output-dir", "out"],
        ["packet", "--input", "in.json", "--output-dir", "out"],
        ["fit", "--manifest", "in.json", "--output-dir", "out"],
        ["profile", "--images", "images", "--output-dir", "out"],
        ["compare", "--manifest", "in.json", "--output-dir", "out"],
    ]
    for argv in cases:
        validate_limits(build_parser().parse_args(argv))


def test_outputs_stay_in_the_stage_directory(tmp_path, monkeypatch):
    stage = tmp_path / "smoke"
    stage.mkdir()
    job = tmp_path / "job"
    job.mkdir()
    monkeypatch.setenv("PS_STAGE_DIR", str(stage))
    monkeypatch.setenv("PS_JOB_DIR", str(job))
    assert _require_job_output(stage) == stage.resolve()
    with pytest.raises(ValueError, match="PS_JOB_DIR"):
        _require_job_output(job)
    with pytest.raises(ValueError, match="PS_STAGE_DIR"):
        _require_job_output(tmp_path / "elsewhere")
    monkeypatch.delenv("PS_STAGE_DIR")
    with pytest.raises(ValueError, match="PS_STAGE_DIR"):
        _require_job_output(tmp_path / "out")


def test_audit_samples_at_most_24_candidates_and_keeps_labels_pending(tmp_path):
    dataset = tmp_path / "dataset"
    source_hashes = {}
    for clip, (relative, _total_frames) in RECORDINGS.items():
        folder = dataset / relative
        (folder / "original").mkdir(parents=True)
        (folder / "masks").mkdir()
        (folder / "sam_poses").mkdir()
        pose_rows = []
        for index in range(30):
            frame = index * 30
            name = f"{frame:06d}.jpg"
            image = np.full((100, 100, 3), 90, dtype=np.uint8)
            mask = np.zeros((100, 100), dtype=np.uint8)
            mask[10:90, 10:90] = 255
            assert cv2.imwrite(str(folder / "original" / name), image)
            assert cv2.imwrite(str(folder / "masks" / f"{frame:06d}.png"), mask)
            points = [[20 + i % 5 * 8, 20 + i // 5 * 8] for i in range(21)]
            pose_rows.append({
                "frame_idx": frame, "file": name, "aisle": False,
                "look": bool(clip == "clip_03" and frame == 0),
                "hands": [{"selected": True, "side": "Left", "confidence": 1.2,
                           "box": [10, 10, 90, 90], "landmarks_pixel": points}],
            })
        pose_path = folder / "sam_poses" / "rtmw-l.json"
        pose_path.write_text(json.dumps(pose_rows))
        source_hashes[pose_path] = hashlib.sha256(pose_path.read_bytes()).hexdigest()

    output = tmp_path / "out"
    manifest_path = audit_dataset(dataset, output, seed=1234, max_per_recording=24)
    manifest = json.loads(manifest_path.read_text())
    by_clip = {clip: [row for row in manifest["candidates"] if row["recording"] == clip]
               for clip in RECORDINGS}
    assert all(len(rows) == 24 for rows in by_clip.values())
    assert all(row["review_label"] is None and row["review_status"] in {
        "pending_human", "not_on_contact_sheet", "pixel_review_unavailable"
    } for rows in by_clip.values() for row in rows)
    for rows in by_clip.values():
        for group in ("plausible", "flagged", "ambiguous"):
            assert sum(row["sample_group"] == group and row["review_status"] == "pending_human"
                       for row in rows) <= 8
    assert all(row["raw_mean_score"] == 1.2 and row["raw_joint_scores"] is None
               for rows in by_clip.values() for row in rows)
    assert all(row["review_label"] is None and row["training_eligible"] is False
               for row in by_clip["factory002"])
    zero = next(row for row in manifest["candidates"]
                if row["recording"] == "clip_03" and row["frame_idx"] == 0)
    assert zero["clip3_second_zero_look_override"] is True
    assert zero["training_eligible"] is True
    assert (output / "audit-sheet-clip_01.jpg").is_file()
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest
               for path, digest in source_hashes.items())


def test_freeze_requires_human_visible_labels_and_keeps_seconds_disjoint(tmp_path):
    candidates = []
    for clip in ("clip_01", "clip_03"):
        for second in range(12):
            for side in ("left", "right"):
                points = [[20 + i % 5 * 8, 20 + i // 5 * 8] for i in range(21)]
                candidates.append({
                    "recording": clip, "frame_idx": second * 30, "source_second": second,
                    "candidate_id": f"{clip}-{second}-{side}", "review_status": "reviewed",
                    "review_label": "visible_hand", "training_eligible": True,
                    "image_sha256": "a" * 64, "mask_sha256": "b" * 64,
                    "bbox": [10, 10, 90, 90], "landmarks_pixel": points,
                    "handedness": side,
                })
    candidates.extend([
        {**candidates[0], "candidate_id": "nonhand", "review_label": "non_hand"},
        {**candidates[0], "candidate_id": "factory002", "recording": "factory002"},
    ])
    result = freeze_training_split({
        "schema": "pointstream.foreground.audit.v1", "candidates": candidates,
    }, tmp_path / "frozen")
    fit_keys = {(row["recording"], row["source_second"]) for row in result["fit"]}
    val_keys = {(row["recording"], row["source_second"]) for row in result["validation"]}
    assert len(result["fit"]) == 16 and len(result["validation"]) == 8
    assert not (fit_keys & val_keys)
    assert {row["recording"] for row in result["fit"]} == {"clip_01", "clip_03"}
    assert all(row["decoded_track_id"] == 0 and row["packet_bytes"] > 36
               for row in result["fit"] + result["validation"])
    assert result["gate"]["passed"] is True


def test_freeze_stops_below_eight_fit_and_four_validation_crops(tmp_path):
    rows = []
    for second in range(10):
        rows.append({
            "recording": "clip_01", "frame_idx": second * 30, "source_second": second,
            "candidate_id": f"c-{second}", "review_status": "reviewed",
            "review_label": "visible_hand", "training_eligible": True,
            "image_sha256": "a" * 64, "mask_sha256": "b" * 64,
            "bbox": [10, 10, 90, 90],
            "landmarks_pixel": [[20 + i % 5 * 8, 20 + i // 5 * 8] for i in range(21)],
            "handedness": "left",
        })
    with pytest.raises(ValueError, match="need >=8 fit and >=4 validation"):
        freeze_training_split({"schema": "pointstream.foreground.audit.v1", "candidates": rows}, tmp_path / "nope")


def test_packet_runner_counts_actual_files_in_all_three_modes(tmp_path):
    joints = [[20 + index % 5, 20 + index // 5] for index in range(21)]
    one_hand = {"track_id": 0, "handedness": "left", "bbox": [10, 10, 40, 40], "joints": joints}
    source = tmp_path / "input.json"
    source.write_text(json.dumps({
        "width": 64, "height": 48, "start_frame": 30,
        "frames": [[one_hand] for _ in range(8)],
    }))
    result = run_packet(source, tmp_path / "packet-output", max_frames=8)
    assert set(result["methods"]) == {"raw", "zlib", "delta_zlib"}
    assert result["codes_identical"] is True
    for method, values in result["methods"].items():
        path = tmp_path / "packet-output" / f"segment-{method}.psfg"
        assert path.stat().st_size == values["bytes"] == values["file_bytes"]
        assert values["roundtrip_frames"] == 8


def _hands(count: int = 8) -> list[list[TrackedHand]]:
    frames = []
    for index in range(count):
        joints = tuple((20 + index + joint % 5, 30 + joint // 5) for joint in range(21))
        frames.append([TrackedHand(0, "left", (10, 8, 30, 28), joints)])
    return frames


def test_lossless_methods_match_and_packaged_bytes_are_the_file_size(tmp_path):
    table = compare_packet_cuts(_hands(), width=64, height=48, start_frame=120)
    assert table["codes_identical"] is True
    assert table["start_index"] == 120
    assert table["sum_one_frame_bytes"] == sum(table["one_frame_file_bytes"])
    path = tmp_path / "cut.psfg"
    from demo.pipeline.foreground_codec_v2 import encode_segment
    packet = encode_segment(_hands(), width=64, height=48, start_frame=120, method="raw")
    path.write_bytes(packet)
    assert path.stat().st_size == len(packet) == table["methods"]["raw"]["bytes"]


def test_timing_distinguishes_crops_from_frames_and_the_realtime_budget():
    rates = stage_rates({"read": 0.01, "inference": 0.02}, n_crops=4, n_frames=2)
    assert rates["crops_per_s"] == pytest.approx(4 / 0.03)
    assert rates["frames_per_s"] == pytest.approx(2 / 0.03)
    assert rates["crops_per_s"] != rates["frames_per_s"]
    assert stages_reproduce_total(rates["stages_s"], rates["total_s"], overhead_s=1e-9)
    slow = stage_rates({"pipeline": 0.05}, n_crops=1, n_frames=1)
    assert slow["ms_per_frame"] == pytest.approx(50.0)
    assert slow["meets_24fps"] is False
    fast = stage_rates({"pipeline": 0.04}, n_crops=2, n_frames=1)
    assert fast["meets_24fps"] is True
    assert fast["ms_per_frame"] <= 41.67


def test_provider_fallback_and_batch_coordinates():
    assert classify_execution_provider(["CUDAExecutionProvider"], ["CPUExecutionProvider"]) == "cpu_fallback"
    assert classify_execution_provider(["CUDAExecutionProvider"], ["CUDAExecutionProvider"]) == "cuda"
    single = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert coordinates_agree(single, single + 0.4)
    assert not coordinates_agree(single, single + 1.5)


def test_plan_rejects_over_cap_budgets_and_doubled_work():
    spec = {
        "budget_seconds": 500,
        "smoke": {"seconds": 30},
        "full": {"seconds": 30},
        "validator_seconds": 10,
        "stall_seconds": 20,
    }
    with pytest.raises(PlanCapError):
        check_job_budget(spec)
    ok = {**spec, "budget_seconds": 80}
    check_job_budget(ok)
    with pytest.raises(PlanCapError):
        check_campaign_budgets([{**ok, "budget_seconds": 480} for _ in range(4)])
    with pytest.raises(PlanCapError):
        combined_count(120, 120, 120, "fit steps")
    combined_count(20, 100, 120, "fit steps")


def test_av1_recipe_keeps_crf_preset_and_segment_gop():
    args = av1_square_args(64, 8)
    assert "63" in args and "7" in args and "-g" in args and "64:64" in " ".join(args)
    with pytest.raises(ValueError):
        av1_square_args(32, 8)


def test_interpreter_propagates_env_and_times_out(tmp_path):
    script = tmp_path / "show_env.py"
    script.write_text("import os,sys\nprint(os.environ.get('PS_STAGE',''))\nprint(os.environ.get('CUDA_VISIBLE_DEVICES',''))\nsys.exit(0)\n")
    result = run_bounded_python(
        sys.executable, [str(script)], timeout_s=10,
        env={"PS_STAGE": "smoke", "CUDA_VISIBLE_DEVICES": "0", "PATH": os.environ.get("PATH", "")},
        allowed={sys.executable},
    )
    assert result.returncode == 0
    assert "smoke" in result.stdout and "0" in result.stdout
    sleeper = tmp_path / "sleep.py"
    sleeper.write_text("import time\ntime.sleep(5)\n")
    with pytest.raises(TimeoutError):
        run_bounded_python(sys.executable, [str(sleeper)], timeout_s=0.2, env={"PATH": os.environ.get("PATH", "")}, allowed={sys.executable})
    with pytest.raises(ValueError, match="inline"):
        run_bounded_python(sys.executable, ["-c", "print(1)"], timeout_s=2, allowed={sys.executable})


def test_validator_requires_substantive_checks():
    empty = validate_report({
        "schema": "pointstream.foreground.packet.v1", "kind": "packet", "stage": "smoke",
        "citable": False, "label": "diagnostic", "methods": {},
    })
    assert empty["substantive"] is False
    assert empty["passed_gate"] is False


def _spec(data: Path, work: str, validate: str) -> dict:
    identity = data / "input.json"
    identity.write_text('{"immutable":true}')
    return {
        "schema": 1, "hosts": ["gpu1"], "gpu_models": ["RTX A6000"], "gpu_memory_mib": 12000,
        "cpu_threads": 4, "entrypoint": ["work.py"],
        "arguments": ["--frames", "{frames}"],
        "scale": {"frames": {"smoke": 2, "full": 4}},
        "inputs": [{"path": str(identity), "sha256": inbox.file_digest(identity)}],
        "smoke": {"seconds": 2, "representative_basis": "same packet path on a nested frame count"},
        "full": {"seconds": 2}, "budget_seconds": 20,
        "deadline": datetime.fromtimestamp(time.time() + 300, timezone.utc).isoformat(),
        "validator": ["{python}", "validate.py"], "validator_seconds": 2,
        "required_commands": [], "stall_seconds": 10,
    }


def test_spec_rejects_missing_hash_and_partial_placeholder(tmp_path):
    spec = _spec(tmp_path, "", "")
    spec["inputs"] = [{"path": "/tmp/x.json", "sha256": "zz"}]
    with pytest.raises(fleet.FleetError, match="SHA256"):
        inbox.validate_spec(spec, now=time.time())
    spec = _spec(tmp_path, "", "")
    spec["arguments"] = ["--frames={frames}"]
    with pytest.raises(fleet.FleetError, match="whole argument"):
        inbox.validate_spec(spec, now=time.time())


def test_validator_failure_blocks_the_second_stage(tmp_path, monkeypatch):
    directory = tmp_path / "jobs" / "fleet" / "inbox" / "20261004T000000Z-12345678"
    source = directory / "source"
    source.mkdir(parents=True)
    (directory / "run").mkdir()
    repo = Path(__file__).resolve().parents[2]
    (source / "work.py").write_text(
        "import json,os\nfrom pathlib import Path\n"
        "Path(os.environ['PS_STAGE_DIR'],'report.json').write_text(json.dumps({\n"
        "'schema':'pointstream.foreground.packet.v1','kind':'packet','stage':os.environ['PS_STAGE'],\n"
        "'citable':False,'label':'diagnostic','methods':{}}))\n"
    )
    (source / "validate.py").write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(repo)!r})\n"
        "from demo.experiments.foreground_validate import main\n"
        "raise SystemExit(main())\n"
    )
    spec = inbox.validate_spec(_spec(tmp_path, "", ""), now=time.time())
    monitor.write_json(directory / "spec.json", spec)
    monitor.write_json(directory / "ready.json", {"spec_sha256": inbox.digest(spec), "source_sha256": inbox.source_identity(source)})
    monkeypatch.setenv("PS_JOB_DIR", str(directory / "run"))
    assert inbox.campaign(directory) == 1
    assert not (directory / "full").exists()
