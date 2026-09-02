"""Integration safeguards: no stale checkpoints or shortened fallback controls.

Codec internals are not tested here. Real codec and geometry smoke tests live
in the existing background suite and low_rate_smoke command.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.tier.low_rate_checkpoint import (
    completion_counts, guard_checkpoints, implementation_digest,
    load_checkpoint, save_checkpoint, source_identity,
)


def test_resume_rejects_changed_identity(tmp_path: Path) -> None:
    identity = {"preset": "0", "config": "a", "source": "hash-a", "code": "v1"}
    guard_checkpoints(tmp_path, identity)
    save_checkpoint(tmp_path, "point", {"bytes": 123})
    guard_checkpoints(tmp_path, dict(identity))
    resumed = load_checkpoint(tmp_path, "point")
    assert resumed is not None and resumed["bytes"] == 123
    for key in identity:
        with pytest.raises(SystemExit, match="identity changed"):
            guard_checkpoints(tmp_path, {**identity, key: "changed"})


def test_input_digest_catches_unsampled_frame_change() -> None:
    clip = SimpleNamespace(context_id="court", frames=np.zeros((8, 2, 2, 3), np.uint8))
    before = source_identity([clip])
    clip.frames[2, 0, 0, 0] = 1
    assert source_identity([clip]) != before


def test_code_digest_includes_dirty_source(tmp_path: Path) -> None:
    import subprocess
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "src").mkdir()
    source = tmp_path / "src" / "example.py"
    source.write_text("x = 1\n")
    before = implementation_digest(tmp_path)
    source.write_text("x = 2\n")
    assert implementation_digest(tmp_path) != before


def test_failed_points_are_counted() -> None:
    assert completion_counts([
        {"pointstream": {"usable": True}}, {"pointstream_error": "failed"},
        {"pointstream": {"usable": False}},
    ]) == {"submitted": 3, "succeeded": 1, "failed": 2}


@pytest.mark.parametrize("available", [47, 48])
def test_fallback_loader_requires_exact_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, available: int,
) -> None:
    from experiments.long_scenes import loader
    directory = tmp_path / "video" / "scene" / "extract_24"
    directory.mkdir(parents=True)
    for index in range(available):
        (directory / f"frame_{index:06d}.png").touch()
    monkeypatch.setattr(loader, "BP46_CLIPS", tmp_path)
    monkeypatch.setattr(loader, "DATASET", tmp_path / "no-tracks")
    monkeypatch.setattr(loader, "get_long_scene_manifest", lambda _: {"scenes": [{
        "video": "video", "scene": "scene", "context_id": "crowd",
        "eligibility": {"route": "conventional_fallback"},
        "intervals": {"48": {"status": "ineligible", "start_frame": 0,
                              "end_frame": 48, "failure_reasons": ["crowd"]}},
    }]})
    monkeypatch.setattr(loader, "load_rgb_stack", lambda paths: np.zeros((len(paths), 2, 2, 3), np.uint8))
    if available < 48:
        with pytest.raises(loader.LongSceneError, match="expected 48"):
            loader.load_long_scene_clip("video", "scene", 48, allow_ineligible=True)
    else:
        clip = loader.load_long_scene_clip("video", "scene", 48, allow_ineligible=True)
        assert clip.frames.shape[0] == clip.n_frames == 48
        assert clip.route == "conventional_fallback" and not clip.is_eligible


def test_fallback_delivery_accounts_for_route(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.runner import fallback
    from src.contracts.config import FallbackConfig
    from src.components.codec.measure import TimedRoundtrip
    frames = np.zeros((2, 64, 64, 3), np.uint8)
    calls = []
    def encode(source, *, request, fps):
        calls.append((request.codec_name, fps))
        return TimedRoundtrip(100, source.copy(), 1.0, 0.1, "test", "test", "0", 63)
    monkeypatch.setattr(fallback, "timed_roundtrip", encode)
    config = FallbackConfig(codec="av1", preset="0")
    result = fallback.deliver_fallback(frames, config, route="conventional_fallback")
    assert calls == [("av1", 24.0)]
    assert result.transport_bytes == result.trip.size_bytes + len(result.routing_header) == 101
    np.testing.assert_array_equal(result.trip.frames, frames)
    with pytest.raises(ValueError, match="explicit"):
        fallback.deliver_fallback(frames, config, route="pointstream")


@pytest.mark.parametrize("contexts", [("court", "court"), ("court", "replay")])
def test_static_pan_and_context_change_through_runner(contexts: tuple[str, str]) -> None:
    from dataclasses import replace
    from src.runner import run
    from tests.runner.test_background_panorama import _config
    from tests.components.background.test_canonical_canvas import _court_pair
    from src.contracts.lattice import ART_BACKGROUND_MODEL
    from src.pipeline.reconstruction.background import BackgroundModelView

    static, pan = _court_pair(n_static=48, n_pan=48, step=1)
    base = _config(method="panorama-stream")
    config = replace(base, background=replace(base.background, canvas="canonical"))
    result = run(config, [static, pan], context_ids=contexts)
    assert result.delivered_frames.shape == (96, 96, 128, 3)
    assert result.sizes.is_rate and result.sizes.transport_total > 0
    assert result.sizes.parts_sum == result.sizes.transport_total
    views = [chunk.bag[ART_BACKGROUND_MODEL] for chunk in result.chunks]
    if contexts[0] == contexts[1]:
        first, second = views
        assert isinstance(first, BackgroundModelView) and isinstance(second, BackgroundModelView)
        assert first.width == second.width
        assert first.height == second.height
