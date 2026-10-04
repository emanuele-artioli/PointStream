"""Egocentric hands as a first-class domain: schema, profile and splits."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.components.domain.datasets import load_manifest
from src.components.domain.datasets.catalog import (
    SPLIT_FINAL_HOLDOUT,
    ClipSpec,
    iter_dataset,
    parse_manifest,
)
from src.contracts.domain import (
    BACKGROUND_NONE,
    BACKGROUND_PANORAMA_FULL,
    EGOCENTRIC,
    CameraMotion,
)
from src.contracts.errors import ConfigValueError
from src.contracts.keypoints import (
    COCO_WHOLEBODY_133,
    HAND_21,
    hand_from_wholebody,
    schema,
)


def test_hand_schema_is_registered_with_a_connected_skeleton() -> None:
    assert schema("hand-21") is HAND_21
    assert len(HAND_21) == 21
    assert len(HAND_21.edges) == 20
    touched = {joint for edge in HAND_21.edges for joint in edge}
    assert touched == set(HAND_21.joints)


@pytest.mark.parametrize("side", ["left", "right"])
def test_wholebody_hand_bank_projects_losslessly(side: str) -> None:
    projection = hand_from_wholebody(side)
    assert projection.is_lossless
    index = COCO_WHOLEBODY_133.index_of
    assert projection.direct[0] == index[f"{side}_hand_00"]
    assert projection.direct[20] == index[f"{side}_hand_20"]


def test_unknown_hand_side_is_rejected() -> None:
    with pytest.raises(ValueError):
        hand_from_wholebody("both")


def test_egocentric_profile_forbids_a_panorama_background() -> None:
    assert EGOCENTRIC.camera_motion is CameraMotion.FREE_MOVING
    EGOCENTRIC.assert_background_valid(BACKGROUND_NONE)
    with pytest.raises(ConfigValueError):
        EGOCENTRIC.assert_background_valid(BACKGROUND_PANORAMA_FULL)


def test_hand_carries_keypoints() -> None:
    EGOCENTRIC.assert_motion_supported("hand", "keypoints")
    assert EGOCENTRIC.schema_for("hand") is HAND_21


def test_shipped_manifest_freezes_disjoint_windows_per_source() -> None:
    manifest = load_manifest("egocentric")
    by_source: dict[str, list[ClipSpec]] = {}
    for clip in manifest.clips:
        by_source.setdefault(clip.path, []).append(clip)
    finals = [clip for clip in manifest.clips if clip.split == SPLIT_FINAL_HOLDOUT]
    assert len(finals) == 1
    for path, clips in by_source.items():
        if clips[0].split == SPLIT_FINAL_HOLDOUT:
            assert len(clips) == 1
            continue
        assert {clip.split for clip in clips} == {
            "fit",
            "validation",
            "development-holdout",
        }
        windows = sorted((clip.start_s, clip.start_s + clip.duration_s) for clip in clips)
        for (_, end), (start, _) in zip(windows, windows[1:]):
            assert end == pytest.approx(start), path


def _manifest_with_final(tmp_path: Path):
    (tmp_path / "dev.mp4").write_bytes(b"x")
    (tmp_path / "final.mp4").write_bytes(b"x")
    return parse_manifest(
        {
            "domain": "egocentric",
            "search_roots": [str(tmp_path)],
            "clips": [
                {"id": "dev", "kind": "video", "path": "dev.mp4", "split": "fit"},
                {
                    "id": "final",
                    "kind": "video",
                    "path": "final.mp4",
                    "split": "final-holdout",
                },
            ],
        }
    )


def test_iteration_never_opens_the_final_holdout_by_default(tmp_path: Path) -> None:
    manifest = _manifest_with_final(tmp_path)
    assert [item.clip_id for item in iter_dataset("egocentric", manifest=manifest)] == ["dev"]
    opted = iter_dataset("egocentric", manifest=manifest, include_final_holdout=True)
    assert [item.clip_id for item in opted] == ["dev", "final"]


def test_split_filter_and_validation(tmp_path: Path) -> None:
    manifest = _manifest_with_final(tmp_path)
    assert list(iter_dataset("egocentric", manifest=manifest, splits=["validation"])) == []
    with pytest.raises(ValueError):
        list(iter_dataset("egocentric", manifest=manifest, splits=["test"]))
    with pytest.raises(ValueError):
        ClipSpec(id="bad", kind="video", path="x.mp4", split="test")
    with pytest.raises(ValueError):
        ClipSpec(id="bad", kind="video", path="x.mp4", start_s=1.0)
    with pytest.raises(ValueError):
        ClipSpec(id="bad", kind="frames", path="x", start_s=0.0, duration_s=1.0)
