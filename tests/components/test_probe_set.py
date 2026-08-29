"""Probe-set regenerator and verifier.

Behaviour the caller relies on: a harness reading the manifest finds the
frames the view actually holds, in track-local indices, and the view holds
exactly those tracks. Conditioning directories are paired by position in the
sorted ``frame_*.png`` list, never by reconstructing a filename — crop and
``_skeleton`` do not share names.

Plausible misuse is the silent kind — a wrong symlink that still exists, a
v1 schema that looks complete, a held-out video leaking into the training
view, a conditioning dir whose frame count does not match the crop (the
fault that left 5 of 12 v2 clips with 48 colour frames and 0 skeletons).

Deliberately not tested: PNG contents, argparse help text, third-party
filesystem iteration order, renaming anything in ``assets/dataset``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.probe_set.materialize import regenerate
from experiments.probe_set.schema import (
    COORDINATE_SYSTEM,
    HELD_OUT_VIDEOS,
    LEGACY_SCHEMA_ID,
    SCHEMA_ID,
    TRAINING_SPLIT_VIDEOS,
    ProbeSetError,
)
from experiments.probe_set.select import discover_candidate_tracks
from experiments.probe_set.verify import collect_violations, locked_split_violations, verify
from src.contracts import paths as ps_paths

BROKEN_PROBE_SET = ps_paths.assets() / "probe_set.broken-v1"
UNALIGNED_V2_PROBE_SET = ps_paths.assets() / "probe_set.broken-v2-unaligned"
LIVE_PROBE_SET = ps_paths.assets() / "probe_set"
LIVE_DATASET = ps_paths.assets() / "dataset"


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-png")


def _make_track(
    dataset_root: Path,
    video: str,
    scene: str,
    track: str,
    source_frame_ids: list[int],
    *,
    with_skeleton: bool = True,
) -> None:
    scene_dir = dataset_root / video / "segmentations" / scene
    track_dir = scene_dir / track
    track_dir.mkdir(parents=True, exist_ok=True)
    for fid in source_frame_ids:
        _write_png(track_dir / f"frame_{fid:06d}.png")
    if with_skeleton:
        skel = scene_dir / f"{track}_skeleton"
        skel.mkdir(parents=True, exist_ok=True)
        # Match the live dataset: skeleton files are track-local and zero-based,
        # not named with the crop's global source ids.
        for local_id, _fid in enumerate(source_frame_ids):
            _write_png(skel / f"frame_{local_id:06d}.png")
    (scene_dir / f"{track}_caption.json").write_text('{"caption": "test"}')
    (scene_dir / f"{track}_metadata.json").write_text("[]")
    (scene_dir / f"{track}_keypoints.json").write_text("[]")


@pytest.fixture
def fake_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    _make_track(root, "alcaraz_ruud", "scene_002", "track_0021", list(range(100, 160)))
    _make_track(root, "alcaraz_ruud", "scene_004", "track_0100", list(range(200, 250)))
    _make_track(root, "sinner_alcaraz", "scene_001", "track_0005", list(range(0, 80)))
    _make_track(root, "sinner_alcaraz", "scene_002", "track_0006", list(range(0, 5)))
    _make_track(root, "djokovic_federer", "scene_003", "track_0010", list(range(50, 120)))
    _make_track(
        root,
        "federer_djokovic",
        "scene_001",
        "track_0099",
        list(range(0, 50)),
        with_skeleton=False,
    )
    _make_track(root, "alcaraz_perricard", "scene_002", "track_0002", list(range(0, 40)))
    return root


def _regenerated(tmp_path: Path, dataset: Path, **kwargs: object) -> Path:
    output = tmp_path / "probe_set"
    defaults: dict[str, object] = {
        "seed": 7,
        "num_clips": 3,
        "clip_len_frames": 8,
        "min_frames": 8,
        "training_videos": (
            "alcaraz_ruud",
            "sinner_alcaraz",
            "djokovic_federer",
        ),
    }
    defaults.update(kwargs)
    regenerate(dataset, output, **defaults)  # type: ignore[arg-type]
    return output


class TestVerifierCatchesTheOriginalFaults:
    def test_view_tracks_disagreeing_with_the_manifest_fails(self, tmp_path: Path) -> None:
        view = tmp_path / "clips"
        named = view / "alcaraz_ruud" / "scene_002" / "track_0021"
        other = view / "alcaraz_ruud" / "scene_002" / "track_0001"
        _write_png(named / "frame_000000.png")
        _write_png(other / "frame_000000.png")
        manifest = {
            "schema": SCHEMA_ID,
            "coordinate_system": COORDINATE_SYSTEM,
            "view": "clips",
            "training_videos": ["alcaraz_ruud"],
            "held_out_videos": list(HELD_OUT_VIDEOS),
            "num_probe_clips": 1,
            "probe_clips": [
                {
                    "video": "alcaraz_ruud",
                    "scene": "scene_002",
                    "track": "track_0021",
                    "key": "alcaraz_ruud/scene_002/track_0021",
                    "frame_ids": [0],
                    "num_frames": 1,
                }
            ],
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        (tmp_path / "training_view").mkdir()
        violations = collect_violations(tmp_path)
        joined = "\n".join(violations)
        assert "track_0001" in joined
        assert "does not name" in joined

    def test_named_frame_missing_on_disk_fails(self, tmp_path: Path) -> None:
        track = tmp_path / "clips" / "alcaraz_ruud" / "scene_002" / "track_0021"
        _write_png(track / "frame_000000.png")
        manifest = {
            "schema": SCHEMA_ID,
            "coordinate_system": COORDINATE_SYSTEM,
            "view": "clips",
            "training_videos": ["alcaraz_ruud"],
            "held_out_videos": list(HELD_OUT_VIDEOS),
            "num_probe_clips": 1,
            "probe_clips": [
                {
                    "video": "alcaraz_ruud",
                    "scene": "scene_002",
                    "track": "track_0021",
                    "frame_ids": [0, 1],
                    "num_frames": 2,
                }
            ],
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        (tmp_path / "training_view").mkdir()
        violations = collect_violations(tmp_path)
        assert any("not files" in item or "not a real file" in item for item in violations)

    def test_frame_count_mismatch_fails(self, tmp_path: Path) -> None:
        track = tmp_path / "clips" / "alcaraz_ruud" / "scene_002" / "track_0021"
        _write_png(track / "frame_000000.png")
        _write_png(track / "frame_000001.png")
        manifest = {
            "schema": SCHEMA_ID,
            "coordinate_system": COORDINATE_SYSTEM,
            "view": "clips",
            "training_videos": ["alcaraz_ruud"],
            "held_out_videos": list(HELD_OUT_VIDEOS),
            "num_probe_clips": 1,
            "probe_clips": [
                {
                    "video": "alcaraz_ruud",
                    "scene": "scene_002",
                    "track": "track_0021",
                    "frame_ids": [0],
                    "num_frames": 1,
                }
            ],
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        (tmp_path / "training_view").mkdir()
        violations = collect_violations(tmp_path)
        assert any("view has 2 frames, manifest names 1" in item for item in violations)

    def test_legacy_schema_is_rejected(self, tmp_path: Path) -> None:
        (tmp_path / "clips").mkdir()
        (tmp_path / "training_view").mkdir()
        (tmp_path / "manifest.json").write_text(
            json.dumps({"schema": LEGACY_SCHEMA_ID, "probe_clips": [], "view": "clips"})
        )
        violations = collect_violations(tmp_path)
        assert any(LEGACY_SCHEMA_ID in item for item in violations)
        assert any("not trustworthy" in item for item in violations)

    def test_wrong_symlink_fails_identity_even_when_the_path_exists(
        self, tmp_path: Path, fake_dataset: Path
    ) -> None:
        named_src = fake_dataset / "alcaraz_ruud" / "segmentations" / "scene_002" / "track_0021"
        other_src = fake_dataset / "alcaraz_ruud" / "segmentations" / "scene_004" / "track_0100"
        view_track = tmp_path / "clips" / "alcaraz_ruud" / "scene_002" / "track_0021"
        view_track.mkdir(parents=True)
        (view_track / "frame_000000.png").symlink_to((other_src / "frame_000200.png").resolve())
        manifest = {
            "schema": SCHEMA_ID,
            "coordinate_system": COORDINATE_SYSTEM,
            "view": "clips",
            "training_videos": ["alcaraz_ruud"],
            "held_out_videos": list(HELD_OUT_VIDEOS),
            "dataset_root": str(fake_dataset),
            "num_probe_clips": 1,
            "probe_clips": [
                {
                    "video": "alcaraz_ruud",
                    "scene": "scene_002",
                    "track": "track_0021",
                    "frame_ids": [0],
                    "num_frames": 1,
                    "source_track": str(named_src),
                }
            ],
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        (tmp_path / "training_view").mkdir()
        assert (view_track / "frame_000000.png").exists()
        violations = collect_violations(tmp_path, dataset_root=fake_dataset)
        assert any("not under the named track" in item for item in violations)

    def test_broken_symlink_is_a_distinct_failure(self, tmp_path: Path) -> None:
        view_track = tmp_path / "clips" / "alcaraz_ruud" / "scene_002" / "track_0021"
        view_track.mkdir(parents=True)
        (view_track / "frame_000000.png").symlink_to(tmp_path / "does-not-exist.png")
        manifest = {
            "schema": SCHEMA_ID,
            "coordinate_system": COORDINATE_SYSTEM,
            "view": "clips",
            "training_videos": ["alcaraz_ruud"],
            "held_out_videos": list(HELD_OUT_VIDEOS),
            "num_probe_clips": 1,
            "probe_clips": [
                {
                    "video": "alcaraz_ruud",
                    "scene": "scene_002",
                    "track": "track_0021",
                    "frame_ids": [0],
                    "num_frames": 1,
                }
            ],
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        (tmp_path / "training_view").mkdir()
        violations = collect_violations(tmp_path)
        assert any("broken symlink" in item for item in violations)

    def test_held_out_video_in_training_view_fails(self, tmp_path: Path) -> None:
        track = tmp_path / "clips" / "alcaraz_ruud" / "scene_002" / "track_0021"
        _write_png(track / "frame_000000.png")
        (tmp_path / "training_view" / "alcaraz_highlights").mkdir(parents=True)
        manifest = {
            "schema": SCHEMA_ID,
            "coordinate_system": COORDINATE_SYSTEM,
            "view": "clips",
            "training_videos": ["alcaraz_ruud"],
            "held_out_videos": list(HELD_OUT_VIDEOS),
            "num_probe_clips": 1,
            "probe_clips": [
                {
                    "video": "alcaraz_ruud",
                    "scene": "scene_002",
                    "track": "track_0021",
                    "frame_ids": [0],
                    "num_frames": 1,
                }
            ],
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        violations = collect_violations(tmp_path)
        assert any("held-out" in item for item in violations)

    def test_missing_manifest_fails(self, tmp_path: Path) -> None:
        violations = collect_violations(tmp_path)
        assert any("missing manifest" in item for item in violations)
        with pytest.raises(ProbeSetError, match="missing manifest"):
            verify(tmp_path)

    def test_conditioning_frame_count_must_match_crop(self, tmp_path: Path) -> None:
        """The assertion that would have caught 48 colour / 0 skeleton."""
        track = tmp_path / "clips" / "alcaraz_ruud" / "scene_002" / "track_0021"
        skel = tmp_path / "clips" / "alcaraz_ruud" / "scene_002" / "track_0021_skeleton"
        _write_png(track / "frame_000000.png")
        _write_png(track / "frame_000001.png")
        skel.mkdir(parents=True)
        manifest = {
            "schema": SCHEMA_ID,
            "coordinate_system": COORDINATE_SYSTEM,
            "view": "clips",
            "training_videos": ["alcaraz_ruud"],
            "held_out_videos": list(HELD_OUT_VIDEOS),
            "num_probe_clips": 1,
            "probe_clips": [
                {
                    "video": "alcaraz_ruud",
                    "scene": "scene_002",
                    "track": "track_0021",
                    "frame_ids": [0, 1],
                    "num_frames": 2,
                }
            ],
        }
        (tmp_path / "manifest.json").write_text(json.dumps(manifest))
        (tmp_path / "training_view").mkdir()
        violations = collect_violations(tmp_path)
        assert any("_skeleton has 0 frames, crop has 2" in item for item in violations)


class TestRegenerator:
    def test_manifest_is_derived_from_the_view(
        self, tmp_path: Path, fake_dataset: Path
    ) -> None:
        output = _regenerated(tmp_path, fake_dataset)
        verify(output, dataset_root=fake_dataset)
        manifest = json.loads((output / "manifest.json").read_text())
        assert manifest["schema"] == SCHEMA_ID
        assert manifest["coordinate_system"] == COORDINATE_SYSTEM
        view_tracks = {
            f"{path.parent.parent.name}/{path.parent.name}/{path.name}"
            for path in (output / "clips").glob("*/*/track_*")
            if path.is_dir() and path.name[6:].isdigit()
        }
        manifest_tracks = {clip["key"] for clip in manifest["probe_clips"]}
        assert view_tracks == manifest_tracks
        assert manifest["excluded_training_keys"] == sorted(manifest_tracks)

    def test_frames_are_track_local_with_recoverable_global_offset(
        self, tmp_path: Path, fake_dataset: Path
    ) -> None:
        output = _regenerated(tmp_path, fake_dataset, num_clips=3, clip_len_frames=8)
        manifest = json.loads((output / "manifest.json").read_text())
        for clip in manifest["probe_clips"]:
            assert clip["frame_ids"] == list(range(clip["num_frames"]))
            assert clip["global_offset"] == clip["global_frame_ids"][0]
            track_dir = output / "clips" / clip["video"] / clip["scene"] / clip["track"]
            for local_id, source_id in zip(clip["frame_ids"], clip["global_frame_ids"]):
                link = track_dir / f"frame_{local_id:06d}.png"
                assert link.is_symlink()
                assert link.resolve().name == f"frame_{source_id:06d}.png"
                source_track = (
                    fake_dataset / clip["video"] / "segmentations" / clip["scene"] / clip["track"]
                )
                assert link.resolve().parent == source_track.resolve()

    def test_same_seed_selects_the_same_clips(
        self, tmp_path: Path, fake_dataset: Path
    ) -> None:
        first = _regenerated(tmp_path / "a", fake_dataset, seed=11)
        second = _regenerated(tmp_path / "b", fake_dataset, seed=11)
        manifest_a = json.loads((first / "manifest.json").read_text())
        manifest_b = json.loads((second / "manifest.json").read_text())
        assert [clip["key"] for clip in manifest_a["probe_clips"]] == [
            clip["key"] for clip in manifest_b["probe_clips"]
        ]
        assert [clip["global_frame_ids"] for clip in manifest_a["probe_clips"]] == [
            clip["global_frame_ids"] for clip in manifest_b["probe_clips"]
        ]

    def test_held_out_videos_are_absent_from_the_training_view(
        self, tmp_path: Path, fake_dataset: Path
    ) -> None:
        _make_track(
            fake_dataset, "alcaraz_highlights", "scene_001", "track_0001", list(range(0, 20))
        )
        output = _regenerated(tmp_path, fake_dataset)
        present = {path.name for path in (output / "training_view").iterdir()}
        assert present.isdisjoint(set(HELD_OUT_VIDEOS))
        verify(output, dataset_root=fake_dataset)

    def test_probe_tracks_are_omitted_from_the_training_view(
        self, tmp_path: Path, fake_dataset: Path
    ) -> None:
        output = _regenerated(tmp_path, fake_dataset)
        manifest = json.loads((output / "manifest.json").read_text())
        for key in manifest["excluded_training_keys"]:
            video, scene, track = key.split("/")
            leaked = output / "training_view" / video / "segmentations" / scene / track
            assert not leaked.exists(), key

    def test_refuses_to_sample_held_out_videos(
        self, tmp_path: Path, fake_dataset: Path
    ) -> None:
        with pytest.raises(ValueError, match="held-out"):
            regenerate(
                fake_dataset,
                tmp_path / "out",
                training_videos=("alcaraz_highlights",),
                num_clips=1,
            )

    def test_short_tracks_and_tracks_without_skeleton_are_ineligible(
        self, fake_dataset: Path
    ) -> None:
        candidates = discover_candidate_tracks(
            fake_dataset,
            ("alcaraz_ruud", "sinner_alcaraz", "federer_djokovic"),
            min_frames=8,
        )
        keys = {candidate.key for candidate in candidates}
        assert "sinner_alcaraz/scene_002/track_0006" not in keys
        assert "federer_djokovic/scene_001/track_0099" not in keys
        assert "alcaraz_ruud/scene_002/track_0021" in keys

    def test_skeleton_pairs_by_track_position_not_filename(
        self, tmp_path: Path, fake_dataset: Path
    ) -> None:
        output = _regenerated(tmp_path, fake_dataset, num_clips=3, clip_len_frames=8)
        manifest = json.loads((output / "manifest.json").read_text())
        diverged = False
        for clip in manifest["probe_clips"]:
            source_crop = (
                fake_dataset / clip["video"] / "segmentations" / clip["scene"] / clip["track"]
            )
            source_skel = source_crop.with_name(f"{clip['track']}_skeleton")
            crop_ids = [
                int(path.name[6:12]) for path in sorted(source_crop.glob("frame_*.png"))
            ]
            window_start = crop_ids.index(clip["global_frame_ids"][0])
            view_skel = (
                output / "clips" / clip["video"] / clip["scene"] / f"{clip['track']}_skeleton"
            )
            assert len(list(view_skel.glob("frame_*.png"))) == clip["num_frames"]
            for local_id, source_id in enumerate(clip["global_frame_ids"]):
                link = view_skel / f"frame_{local_id:06d}.png"
                positional = source_skel / f"frame_{window_start + local_id:06d}.png"
                assert link.is_symlink()
                assert link.resolve() == positional.resolve()
                if source_id != window_start + local_id:
                    diverged = True
                    reconstructed = source_skel / f"frame_{source_id:06d}.png"
                    assert not reconstructed.exists()
        assert diverged, "fixture must include a track whose crop names are not 0-based"

    def test_condition_count_mismatch_raises_instead_of_skipping(
        self, tmp_path: Path
    ) -> None:
        dataset = tmp_path / "dataset"
        _make_track(dataset, "alcaraz_ruud", "scene_002", "track_0021", list(range(100, 160)))
        skel = dataset / "alcaraz_ruud" / "segmentations" / "scene_002" / "track_0021_skeleton"
        for path in list(skel.glob("frame_*.png"))[8:]:
            path.unlink()
        with pytest.raises(ProbeSetError, match="_skeleton has 8 frames, crop has 60"):
            regenerate(
                dataset,
                tmp_path / "probe_set",
                seed=7,
                num_clips=1,
                clip_len_frames=8,
                min_frames=8,
                training_videos=("alcaraz_ruud",),
            )


class TestLockedSplit:
    def test_training_and_held_out_are_disjoint(self) -> None:
        assert set(TRAINING_SPLIT_VIDEOS).isdisjoint(set(HELD_OUT_VIDEOS))
        assert len(TRAINING_SPLIT_VIDEOS) == 5
        assert len(HELD_OUT_VIDEOS) == 2

    def test_locked_split_helper_flags_drift(self) -> None:
        violations = locked_split_violations(
            {
                "training_videos": list(TRAINING_SPLIT_VIDEOS)[:-1],
                "held_out_videos": list(HELD_OUT_VIDEOS),
            }
        )
        assert violations


class TestAgainstTheRealTrees:
    """Drive the verifier at the on-disk snapshot and the rebuilt live set.

    Skip when the trees are not mounted — CI without ``assets/`` is not a pass.
    """

    def test_verifier_fails_on_the_broken_v1_snapshot(self) -> None:
        if not (BROKEN_PROBE_SET / "manifest.json").is_file():
            pytest.skip("assets/probe_set.broken-v1 is not present")
        violations = collect_violations(
            BROKEN_PROBE_SET, dataset_root=LIVE_DATASET, check_locked_split=True
        )
        assert violations, "a verifier that passes on the broken set is not a verifier"
        joined = "\n".join(violations)
        assert LEGACY_SCHEMA_ID in joined
        assert "missing tracks the manifest names" in joined

    def test_verifier_fails_on_the_unaligned_v2_snapshot(self) -> None:
        if not (UNALIGNED_V2_PROBE_SET / "manifest.json").is_file():
            pytest.skip("assets/probe_set.broken-v2-unaligned is not present")
        violations = collect_violations(
            UNALIGNED_V2_PROBE_SET, dataset_root=LIVE_DATASET, check_locked_split=True
        )
        assert violations, "a verifier that passes on the unaligned v2 set is not a verifier"
        joined = "\n".join(violations)
        assert "_skeleton has 0 frames" in joined
        assert "crop has 48" in joined

    def test_verifier_passes_on_the_rebuilt_set(self) -> None:
        if not (LIVE_PROBE_SET / "manifest.json").is_file():
            pytest.skip("assets/probe_set is not present")
        schema = json.loads((LIVE_PROBE_SET / "manifest.json").read_text()).get("schema")
        if schema != SCHEMA_ID:
            pytest.skip("live probe set has not been rebuilt to v2 yet")
        verify(LIVE_PROBE_SET, dataset_root=LIVE_DATASET, check_locked_split=True)
