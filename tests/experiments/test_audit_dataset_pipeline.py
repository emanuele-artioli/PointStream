from __future__ import annotations

import json
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from scripts.audit_dataset_pipeline import (
    _build_quality_chunks,
    _finalize_quality_dataset,
    _has_tracking_seed,
    _load_quality_chunk,
    _mask_tracking_point,
    _prepare_quality_build_root,
    _quarantine_incomplete_quality_chunk,
    _resize_mask_nearest,
    _resolve_existing_scene_frames,
    _safe_propagate_sam3,
    _sam_input_dimensions,
    _tracking_refinement_points,
    _validated_quality_chunk_summary,
)


def test_sam_inference_caps_4k_dimensions_without_changing_aspect_ratio() -> None:
    assert _sam_input_dimensions(3840, 2160, 1920) == (1920, 1080)
    assert _sam_input_dimensions(1280, 720, 1920) == (1280, 720)


def test_only_the_requested_prompt_object_can_seed_tracking() -> None:
    missing = type("Observation", (), {"object_id": "target", "mask": None, "tracker_id": None})()
    extra = type("Observation", (), {"object_id": "racket:3", "mask": object(), "tracker_id": 3})()
    observed = type("Observation", (), {"object_id": "target", "mask": object(), "tracker_id": 4})()
    assert not _has_tracking_seed([missing], "target")
    assert not _has_tracking_seed([extra], "target")
    assert _has_tracking_seed([missing, extra, observed], "target")


def test_mask_tracking_point_always_lands_on_foreground() -> None:
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[1:3, 1:4] = 1
    mask[5:7, 7:9] = 1
    point = _mask_tracking_point(mask)
    assert point is not None
    x, y = map(int, point)
    assert mask[y, x] != 0


def test_racket_seed_refinement_uses_legacy_click_and_marks_drifting_mask_negative() -> None:
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[80:90, 75:85] = 1
    target = SimpleNamespace(mask=mask, tracker_id=12)
    seed = {"bbox": [100, 100, 140, 140], "point_xy": [120.0, 120.0]}

    points, labels, negative, source = _tracking_refinement_points(
        seed, target, scale_x=0.5, scale_y=0.5
    )

    assert points[0] == (60.0, 60.0)
    assert labels == [1, 0]
    assert negative == points[1]
    assert source == "legacy_racket_mask_points"


def test_quality_chunk_uses_legacy_racket_point_but_not_for_player() -> None:
    scene = {
        "source_id": "video_scene_000",
        "video": "video",
        "scene": "scene_000",
        "sample_frame_ids": [0, 1],
        "tracks": [{
            "object_id": "track_0001",
            "records": [
                {
                    "frame_id": frame_id,
                    "bbox": [100, 200, 220, 380],
                    "racket_bbox_crop": [0, 78, 84, 128],
                    "racket_mask_points": {
                        "p1": [91.0, 131.0], "p2": [4.0, 77.0],
                        "p3": [43.0, 77.0], "p4": [14.0, 131.0],
                    },
                }
                for frame_id in (0, 1)
            ],
        }],
    }

    chunk = _build_quality_chunks(
        scene, [Path("frame_000000.png"), Path("frame_000001.png")], (400, 400)
    )[0]

    assert "point_xy" not in chunk["frame_prompts"]["0"]["player"][0]
    assert chunk["frame_prompts"]["0"]["racket"][0]["point_xy"] == [138.0, 304.0]
    assert [
        prompt["frame_index"]
        for frame in chunk["frame_prompts"].values()
        for prompt in frame["player"]
    ] == [0, 1]
    assert [
        prompt["frame_index"]
        for frame in chunk["frame_prompts"].values()
        for prompt in frame["racket"]
    ] == [0, 1]


def test_empty_mask_has_no_tracking_point() -> None:
    assert _mask_tracking_point(np.zeros((4, 4), dtype=np.uint8)) is None


def test_no_points_propagation_is_withheld_but_other_runtime_errors_surface() -> None:
    class NoPoints:
        def propagate(self, role: str, **kwargs: object) -> list[object]:
            raise RuntimeError("No points are provided; please add points first")

    class Broken:
        def propagate(self, role: str, **kwargs: object) -> list[object]:
            raise RuntimeError("CUDA kernel failure")

    assert _safe_propagate_sam3(NoPoints(), "player")[0] == ()
    assert _safe_propagate_sam3(NoPoints(), "player")[1] == "sam3_1_tracking_state_has_no_accepted_points"
    with pytest.raises(RuntimeError, match="CUDA kernel failure"):
        _safe_propagate_sam3(Broken(), "player")


def test_resumed_quality_chunk_requires_matching_metadata_and_masks(tmp_path: Path) -> None:
    mask_dir = tmp_path / "chunk_000000"
    mask_dir.mkdir()
    provenance = {
        "name": "sam3.1_multiplex", "model_revision": "a" * 40,
        "checkpoint_sha256": "b" * 64, "config_sha256": "c" * 64,
        "policy": "offline_bidirectional",
    }
    payload = {
        "schema": "pointstream.sam31-existing-dataset-chunk.v1",
        "source_id": "video_scene_000",
        "video": "video",
        "scene": "scene_000",
        "frame_start": 0,
        "frame_count": 2,
        "source_frame_size": [10, 8],
        "sam_provenance": provenance,
        "outputs": [
            {"role": "player", "frame_index": 0, "object_id": "track_0001", "mask_path": None},
            {"role": "player", "frame_index": 1, "object_id": "track_0001", "mask_path": None},
        ],
        "chunk_seconds": 2.5,
        "suppressed_nonlegacy_sam_masks": {"player": 0, "racket": 0},
    }
    (mask_dir / "chunk.json").write_text(json.dumps(payload))
    chunk = {
        "source_id": "video_scene_000",
        "video": "video",
        "scene": "scene_000",
        "frame_start": 0,
        "frame_count": 2,
        "frame_width": 10,
        "frame_height": 8,
        "mask_dir": str(mask_dir),
        "target_frames": {
            "0": {"player": ["track_0001"], "racket": []},
            "1": {"player": ["track_0001"], "racket": []},
        },
    }

    summary = _validated_quality_chunk_summary(chunk, expected_provenance=provenance)
    assert summary["output_count"] == 2
    assert summary["seconds"] == 2.5

    payload["frame_count"] = 3
    (mask_dir / "chunk.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="different frame_count"):
        _validated_quality_chunk_summary(chunk, expected_provenance=provenance)


def test_quality_chunk_uses_original_sam_score_after_point_refinement(tmp_path: Path) -> None:
    from PIL import Image

    mask_dir = tmp_path / "chunk_000000"
    mask_dir.mkdir()
    provenance = {
        "name": "sam3.1_multiplex", "model_revision": "a" * 40,
        "checkpoint_sha256": "b" * 64, "config_sha256": "c" * 64,
        "policy": "offline_bidirectional",
    }
    mask_path = mask_dir / "racket.png"
    Image.fromarray(np.ones((2, 2), dtype=np.uint8) * 255).save(mask_path)
    payload = {
        "sam_provenance": provenance,
        "frame_shape": [8, 10],
        "outputs": [{
            "role": "racket", "frame_index": 0, "object_id": "racket:track_0001",
            "tracker_id": 7, "score": 1.0, "status": "observed", "reason": None,
            "mask_path": str(mask_path),
            "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
            "bbox_xyxy": [3, 2, 5, 4],
        }],
        "prompts": [{
            "role": "racket", "frame_index": 0, "object_id": "racket:track_0001",
            "sam_score": 0.42,
        }],
    }
    metadata_path = mask_dir / "chunk.json"
    metadata_path.write_text(json.dumps(payload))
    chunk = {
        "metadata_path": str(metadata_path),
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "frame_shape": [8, 10],
    }

    _, outputs = _load_quality_chunk(chunk)

    assert outputs["racket"][(0, "racket:track_0001")].score == 0.42


def test_incomplete_quality_chunk_is_preserved_when_quarantined(tmp_path: Path) -> None:
    shard_dir = tmp_path / "shard-00"
    incomplete = shard_dir / "sam_masks" / "video" / "scene_000" / "chunk_000000"
    incomplete.mkdir(parents=True)
    partial = incomplete / "partial.png"
    partial.write_bytes(b"partial output")

    destination = _quarantine_incomplete_quality_chunk(incomplete, shard_dir)

    assert not incomplete.exists()
    assert (destination / "partial.png").read_bytes() == b"partial output"


def test_resume_reuses_existing_reconstructed_source_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.audit_dataset_pipeline as pipeline

    data_root = tmp_path / "data"
    raw_video = data_root / "assets" / "raw_4k" / "video.mp4"
    raw_video.parent.mkdir(parents=True)
    raw_video.write_bytes(b"raw video identity")
    metadata_path = data_root / "assets" / "dataset" / "video" / "scene_metadata.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(json.dumps({"scenes": [{"t_start": 0.0, "t_end": 1.0, "duration": 1.0}]}))
    shard_dir = tmp_path / "shard-04"
    frame_dir = shard_dir / "reconstructed_sources" / "video" / "scene_000" / "extract_24"
    frame_dir.mkdir(parents=True)
    (frame_dir / "frame_000001.png").write_bytes(b"frame one")
    (frame_dir / "frame_000002.png").write_bytes(b"frame two")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(stdout="ffmpeg version test\n")

    monkeypatch.setattr(pipeline.shutil, "which", lambda command: "/usr/bin/ffmpeg")
    monkeypatch.setattr(pipeline.subprocess, "run", fake_run)
    scene = {
        "source_id": "video_scene_000", "video": "video", "scene": "scene_000",
        "sample_frame_ids": [0, 1],
    }

    directory, paths, provenance = _resolve_existing_scene_frames(data_root, scene, shard_dir)

    assert directory == frame_dir
    assert [path.name for path in paths] == ["frame_000001.png", "frame_000002.png"]
    assert provenance["source_kind"] == "reconstructed_from_raw_4k"
    assert calls == [["/usr/bin/ffmpeg", "-version"]]


def test_mask_restore_uses_nearest_neighbor_and_preserves_binary_values() -> None:
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    restored = _resize_mask_nearest(mask, 4, 4)
    assert restored.shape == (4, 4)
    assert set(np.unique(restored)) == {0, 1}
    assert restored.tolist() == [
        [0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]
    ]


def test_quality_shards_share_only_the_same_incomplete_build_root(tmp_path) -> None:
    output_root = tmp_path / "quality-dataset"
    plan = {
        "schema": "pointstream.sam31-existing-training-build-plan.v1",
        "selection_sha256": "selection-hash",
        "shard_count": 5,
        "sam_checkpoint_sha256": "checkpoint-hash",
    }

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(_prepare_quality_build_root, output_root, plan=plan) for _ in range(5)]
        for future in futures:
            future.result()

    assert json.loads((output_root / "build_plan.json").read_text()) == plan

    mismatched_plan = {**plan, "shard_count": 4}
    with pytest.raises(ValueError, match="does not match existing output root"):
        _prepare_quality_build_root(output_root, plan=mismatched_plan)

    (output_root / "dataset_manifest.json").write_text("{}\n")
    with pytest.raises(FileExistsError, match="refusing to overwrite completed quality dataset"):
        _prepare_quality_build_root(output_root, plan=plan)


def test_quality_finalizer_records_the_selected_manifest_revision(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    selection_path = repo_root / "manifests" / "sam31_quality_dataset_v3.json"
    selection_path.parent.mkdir(parents=True)
    selection_path.write_text('{"version":"sam31-quality-existing-training-v3"}\n')
    output_root = tmp_path / "external" / "dataset-v3"
    for index in range(2):
        shard_dir = output_root / "shards" / f"shard-{index:02d}"
        shard_dir.mkdir(parents=True)
        (shard_dir / "training_samples.jsonl").write_text("")
        (shard_dir / "quality_audit.jsonl").write_text("")
        (shard_dir / "shard_manifest.json").write_text(json.dumps({
            "status": "complete",
            "source_dataset": "/external/assets/dataset",
            "source_scenes": [],
            "track_metadata_sha256": {},
            "scene_reviews": [],
            "sam31": {"name": "SAM3.1"},
            "dwpose": {"name": "DWPose"},
            "sam_runtime": {"host": "gpu", "gpu": "Ada"},
            "shard_index": index,
            "shard_count": 2,
            "scene_count": 0,
            "track_count": 0,
            "track_frame_count": 0,
            "unique_frame_count": 0,
            "chunk_count": 0,
            "accepted_by_class": {},
            "rejected_by_class": {},
            "elapsed_seconds": 0.0,
        }))

    manifest = _finalize_quality_dataset(
        output_root,
        shard_count=2,
        selection={
            "version": "sam31-quality-existing-training-v3",
            "split": "development_exposed_existing_train",
            "quality_policy": {},
            "sam_inference": {},
            "selection_policy": {},
        },
        selection_manifest_path=selection_path,
        repo_root=repo_root,
    )

    assert manifest is not None
    assert manifest["schema"] == "pointstream.sam31-quality-dataset.v3"
    assert manifest["selection_manifest"] == "manifests/sam31_quality_dataset_v3.json"
    assert manifest["selection_manifest_sha256"] == hashlib.sha256(selection_path.read_bytes()).hexdigest()
    assert manifest["active"] is False
