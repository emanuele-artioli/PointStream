from __future__ import annotations

from pathlib import Path

import pytest
import cv2
import numpy as np
import hashlib

from demo.experiments.foreground_smoke import (
    RECORDINGS, audit_dataset, build_parser, validate_limits, _require_job_output,
    freeze_training_split, run_packet,
)


@pytest.mark.parametrize("argv", [
    ["audit", "--output-dir", "out", "--max-candidates-per-recording", "25"],
    ["packet", "--input", "in.json", "--output-dir", "out", "--max-frames", "9"],
    ["fit", "--manifest", "in.json", "--output-dir", "out", "--max-steps", "121"],
    ["fit", "--manifest", "in.json", "--output-dir", "out", "--seconds-per-arm", "121"],
    ["profile", "--images", "images", "--output-dir", "out", "--max-images", "17"],
    ["profile", "--images", "images", "--output-dir", "out", "--sam-frames", "31"],
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


def test_outputs_are_confined_to_dispatcher_assigned_job_dir(tmp_path, monkeypatch):
    job = tmp_path / "job"
    job.mkdir()
    monkeypatch.setenv("PS_JOB_DIR", str(job))
    assert _require_job_output(job / "foreground") == (job / "foreground").resolve()
    with pytest.raises(ValueError, match="below PS_JOB_DIR"):
        _require_job_output(job)
    with pytest.raises(ValueError, match="below PS_JOB_DIR"):
        _require_job_output(tmp_path / "elsewhere")
    monkeypatch.delenv("PS_JOB_DIR")
    with pytest.raises(ValueError, match="PS_JOB_DIR"):
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
        pose_path.write_text(__import__("json").dumps(pose_rows))
        source_hashes[pose_path] = hashlib.sha256(pose_path.read_bytes()).hexdigest()

    output = tmp_path / "out"
    manifest_path = audit_dataset(dataset, output, seed=1234, max_per_recording=24)
    manifest = __import__("json").loads(manifest_path.read_text())
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
    # A nonvisible label and the audit-only domain cannot enter either split.
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
    source.write_text(__import__("json").dumps({
        "width": 64, "height": 48, "start_frame": 30,
        "frames": [[one_hand] for _ in range(8)],
    }))
    result = run_packet(source, tmp_path / "packet-output", max_frames=8)
    assert set(result["methods"]) == {"raw", "zlib", "delta_zlib"}
    for method, values in result["methods"].items():
        path = tmp_path / "packet-output" / f"segment-{method}.psfg"
        assert path.stat().st_size == values["bytes"] == values["file_bytes"]
        assert values["roundtrip_frames"] == 8
