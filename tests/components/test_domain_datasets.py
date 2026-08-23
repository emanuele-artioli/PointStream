"""Minimal-dataset manifests and loader behaviour when files are missing or present."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.components.domain import REGISTRY as DOMAINS
from src.components.domain.datasets import (
    DatasetMissingError,
    first_sample,
    iter_dataset,
    load_manifest,
    manifest_path,
    parse_manifest,
    smoke,
)
from src.contracts.errors import UnknownBackendError


def _write_frames(directory: Path, names: tuple[str, ...]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"fixture-frame")


def _synthetic_manifest(tmp_path: Path, *, domain: str = "tennis") -> Path:
    frames = tmp_path / "clip_a"
    _write_frames(frames, ("frame_000000.png", "frame_000001.png"))
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"not-a-real-video")
    payload = {
        "domain": domain,
        "selector": "heuristic",
        "search_roots": [str(tmp_path)],
        "clips": [
            {
                "id": "clip_a",
                "kind": "frames",
                "path": "clip_a",
                "pattern": "frame_*.png",
            },
            {"id": "clip_video", "kind": "video", "path": "clip.mp4"},
        ],
    }
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_minimal_manifests_exist_for_both_profiles() -> None:
    for name in ("tennis", "general"):
        path = manifest_path(name)
        assert path.is_file()
        manifest = load_manifest(name)
        assert manifest.domain == name
        assert manifest.clips
        spec = DOMAINS.spec(name)
        assert manifest.selector == spec.defaults["selector"]


def test_football_has_no_dataset_manifest() -> None:
    with pytest.raises(UnknownBackendError, match="football"):
        load_manifest("football")


def test_loader_yields_frames_tagged_with_the_domain_when_files_are_present(tmp_path: Path) -> None:
    manifest = load_manifest("tennis", path=_synthetic_manifest(tmp_path))
    items = list(iter_dataset("tennis", manifest=manifest, missing="error"))
    assert [item.clip_id for item in items] == ["clip_a", "clip_video"]
    assert all(item.domain == "tennis" for item in items)
    assert items[0].frames[0].name == "frame_000000.png"
    assert items[0].sample_path.is_file()
    assert items[1].kind == "video"
    assert items[1].source == tmp_path / "clip.mp4"


def test_loader_names_the_path_when_a_required_clip_is_missing(tmp_path: Path) -> None:
    manifest = parse_manifest(
        {
            "domain": "general",
            "selector": "identity",
            "search_roots": [str(tmp_path / "empty")],
            "clips": [{"id": "parkour", "kind": "frames", "path": "parkour", "pattern": "*.jpg"}],
        }
    )
    with pytest.raises(DatasetMissingError) as excinfo:
        list(iter_dataset("general", manifest=manifest, missing="error"))
    message = str(excinfo.value)
    assert "parkour" in message
    assert "does not download" in message
    assert "general" in message


def test_loader_skips_missing_clips_instead_of_pretending_they_loaded(tmp_path: Path) -> None:
    present = tmp_path / "lucia"
    _write_frames(present, ("00000.jpg",))
    manifest = parse_manifest(
        {
            "domain": "general",
            "selector": "identity",
            "search_roots": [str(tmp_path)],
            "clips": [
                {"id": "parkour", "kind": "frames", "path": "parkour", "pattern": "*.jpg"},
                {"id": "lucia", "kind": "frames", "path": "lucia", "pattern": "*.jpg"},
            ],
        }
    )
    items = list(iter_dataset("general", manifest=manifest, missing="skip"))
    assert [item.clip_id for item in items] == ["lucia"]


def test_smoke_loads_one_sample_per_profile_when_files_exist(tmp_path: Path) -> None:
    tennis_dir = tmp_path / "tennis" / "clip_a"
    _write_frames(tennis_dir, ("frame_000000.png",))
    tennis_manifest = parse_manifest(
        {
            "domain": "tennis",
            "selector": "heuristic",
            "search_roots": [str(tmp_path / "tennis")],
            "clips": [
                {
                    "id": "clip_a",
                    "kind": "frames",
                    "path": "clip_a",
                    "pattern": "frame_*.png",
                }
            ],
        }
    )
    sample = first_sample("tennis", manifest=tennis_manifest, missing="error")
    assert sample is not None
    assert sample.domain == "tennis"
    assert sample.sample_path.is_file()
    # Smoke against the real manifests must not invent an encode when files
    # are absent from this worktree; when they are present it returns paths.
    result = smoke(require=False)
    for domain, item in result.items():
        assert item.domain == domain
        assert item.sample_path.is_file()


def test_smoke_require_fails_clearly_when_nothing_is_on_disk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty = {
        "domain": "tennis",
        "selector": "heuristic",
        "search_roots": [str(tmp_path / "nowhere")],
        "clips": [{"id": "real_tennis", "kind": "video", "path": "real_tennis.mp4"}],
    }

    def _fake_load(domain: str, *, path: Path | None = None):
        del path
        data = dict(empty)
        data["domain"] = domain
        if domain == "general":
            data["clips"] = [{"id": "parkour", "kind": "frames", "path": "parkour"}]
        return parse_manifest(data)

    monkeypatch.setattr("src.components.domain.datasets.catalog.load_manifest", _fake_load)
    with pytest.raises(DatasetMissingError, match="not on disk"):
        smoke(require=True)


def test_a_manifest_with_no_clips_is_rejected() -> None:
    with pytest.raises(ValueError, match="no clips"):
        parse_manifest({"domain": "tennis", "selector": "heuristic", "clips": []})
