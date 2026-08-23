"""The trained ControlNet text channel reaches inference, or it does not."""

from __future__ import annotations

import json

import numpy as np

from experiments.probe.clips import (
    CodingSample,
    bundle_coding,
    caption_names_a_colour,
    load_track_caption,
)
from experiments.probe.run import _coding_bundle, _prompt_meta


def test_load_track_caption_reads_the_sidecar_next_to_the_track(tmp_path) -> None:
    """Training's path: ``{track_dir.parent}/{track_dir.name}_caption.json``."""
    scene = tmp_path / "segmentations" / "scene_000"
    track = scene / "track_1"
    track.mkdir(parents=True)
    payload = {"caption": "a man in a purple shirt playing tennis"}
    (scene / "track_1_caption.json").write_text(json.dumps(payload))
    assert load_track_caption(track) == payload["caption"]


def test_load_track_caption_falls_back_to_the_dataset_copy(tmp_path, monkeypatch) -> None:
    """Probe-set clips built before the sidecar was copied still find the caption."""
    from experiments.probe import clips as clips_mod

    monkeypatch.setattr(clips_mod, "repo_root", lambda: tmp_path)
    track = tmp_path / "assets" / "probe_set" / "clips" / "match" / "scene_000" / "track_1"
    track.mkdir(parents=True)
    dataset = (
        tmp_path / "assets" / "dataset" / "match" / "segmentations" / "scene_000"
    )
    dataset.mkdir(parents=True)
    (dataset / "track_1_caption.json").write_text(
        json.dumps({"caption": "a woman in a white dress playing tennis"})
    )
    assert (
        load_track_caption(track, video="match", scene="scene_000", track="track_1")
        == "a woman in a white dress playing tennis"
    )


def test_empty_caption_sidecar_is_missing_not_an_empty_string(tmp_path) -> None:
    """Whitespace would look like a live prompt and hide the fallback."""
    scene = tmp_path / "scene_000"
    track = scene / "track_1"
    track.mkdir(parents=True)
    (scene / "track_1_caption.json").write_text(json.dumps({"caption": "   "}))
    assert load_track_caption(track) is None


def test_caption_names_a_colour_is_about_kit_not_any_substring() -> None:
    assert caption_names_a_colour("a man in a purple shirt playing tennis")
    assert not caption_names_a_colour("photorealistic tennis player, broadcast sports shot")
    assert not caption_names_a_colour(None)
    assert not caption_names_a_colour("")


def test_coding_bundle_carries_the_track_caption() -> None:
    """The defect: generate had no field to put the caption in."""
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    mask = np.ones((4, 4), dtype=bool)
    caption = "a man in a purple shirt playing tennis"
    sample = CodingSample(
        key="clip",
        video="match",
        scene="scene_000",
        track="track_1",
        appearance_frame_index=0,
        target_frame_index=1,
        offset=1,
        n_frames=8,
        appearance_rgb=rgb,
        reference_rgb=rgb,
        object_mask=mask,
        pose_rgb=rgb,
        canny=np.zeros((4, 4), dtype=np.uint8),
        motion_field=np.zeros((2, 4, 4), dtype=np.float32),
        split="train",
        caption=caption,
    )
    payload = bundle_coding(sample)
    assert payload["caption"] == caption
    bundle = _coding_bundle(sample)
    assert bundle.caption == caption


def test_prompt_meta_reads_what_the_generator_last_sent() -> None:
    class _Gen:
        last_prompt = "a man in a purple shirt playing tennis"
        last_prompt_source = "caption"

    prompt, source = _prompt_meta(_Gen())
    assert prompt == _Gen.last_prompt
    assert source == "caption"
    assert _prompt_meta(object()) == (None, None)
