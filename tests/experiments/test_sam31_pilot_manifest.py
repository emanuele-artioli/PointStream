from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from scripts.audit_dataset_pipeline import (
    _frame_paths_by_position,
    _scene_local_frame_index,
    _verify_catalog_frame_anchors,
    validate_pilot_manifest,
)


ROOT = Path(__file__).resolve().parents[2]


def _inputs() -> tuple[dict, dict]:
    pilot = json.loads((ROOT / "manifests/sam31_pilot_v1.json").read_text())
    catalog = json.loads((ROOT / pilot["scene_manifest"]).read_text())
    return pilot, catalog


def test_pilot_pins_three_consecutive_exposed_development_scenes() -> None:
    pilot, catalog = _inputs()
    scenes = validate_pilot_manifest(pilot, catalog)
    assert len(scenes) == 3
    assert {scene["contrast"] for scene in scenes} == {
        "near_static",
        "panning",
        "small_or_partially_occluded_racket_candidate",
    }
    assert all(scene["frame_count"] == 16 for scene in scenes)
    assert [scene["frame_directory"] for scene in scenes] == [
        "extract_24",
        "extract_24",
        "extract_24",
    ]
    assert all(scene["catalog_interval"]["end_frame"] - scene["catalog_interval"]["start_frame"] == 48 for scene in scenes)
    assert all(set(scene["catalog_interval"]["frame_hashes"]) == {"first", "mid", "last"} for scene in scenes)
    assert scenes[-1]["visual_racket_case_verified"] is False


def test_pilot_rejects_reserved_confirmation_sources() -> None:
    pilot, catalog = _inputs()
    with pytest.raises(ValueError, match="reserved confirmation"):
        validate_pilot_manifest(
            pilot,
            catalog,
            reserved_source_ids={"alcaraz_highlights"},
        )


def test_pilot_rejects_frame_span_outside_eligible_interval() -> None:
    pilot, catalog = _inputs()
    pilot["scenes"][0]["frame_start"] = 10000
    with pytest.raises(ValueError, match="outside its eligible"):
        validate_pilot_manifest(pilot, catalog)


def test_pilot_rejects_unpinned_frame_directory() -> None:
    pilot, catalog = _inputs()
    pilot["scenes"][0]["frame_directory"] = "../../other-split"
    with pytest.raises(ValueError, match="unsupported frame directory"):
        validate_pilot_manifest(pilot, catalog)


def test_global_observation_index_maps_to_chunk_local_wire_index() -> None:
    scene = {"frame_start": 361}
    assert _scene_local_frame_index(371, scene) == 10
    with pytest.raises(ValueError, match="precedes its selected"):
        _scene_local_frame_index(360, scene)


def test_catalog_frames_are_selected_by_zero_based_position_and_hash_anchors(tmp_path: Path) -> None:
    directory = tmp_path / "extract_24"
    directory.mkdir()
    for file_id in range(1, 61):
        (directory / f"frame_{file_id:06d}.png").write_bytes(f"frame-{file_id}".encode())

    catalog_interval_paths = _frame_paths_by_position(directory, 1, 48)
    hashes = {
        "first": hashlib.sha256(catalog_interval_paths[0].read_bytes()).hexdigest(),
        "mid": hashlib.sha256(catalog_interval_paths[24].read_bytes()).hexdigest(),
        "last": hashlib.sha256(catalog_interval_paths[-1].read_bytes()).hexdigest(),
    }
    scene = {
        "source_id": "test-scene",
        "catalog_interval": {"start_frame": 1, "end_frame": 49, "frame_hashes": hashes},
    }

    verified_interval = _verify_catalog_frame_anchors(directory, scene)
    pilot_paths = _frame_paths_by_position(directory, 1, 16)

    assert [path.name for path in (verified_interval[0], verified_interval[24], verified_interval[-1])] == [
        "frame_000002.png",
        "frame_000026.png",
        "frame_000049.png",
    ]
    assert [path.name for path in (pilot_paths[0], pilot_paths[-1])] == [
        "frame_000002.png",
        "frame_000017.png",
    ]

    scene["catalog_interval"]["frame_hashes"]["first"] = "0" * 64
    with pytest.raises(ValueError, match="first catalog frame hash mismatch"):
        _verify_catalog_frame_anchors(directory, scene)
