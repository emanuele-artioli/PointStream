"""Recovery must preserve evidence, not merely avoid rerunning a scene.

Real small background streams exercise continuation; no codec-internal resume
or coverage-only cases. Native runtime remains a separate operational gate.
"""
from dataclasses import replace
import importlib
import json
from threading import Event

import numpy as np
import pytest

from src.components.background.strategy import PanoramaStream
from src.pipeline.encoder.encoder import Encoder
from src.runner import run
from src.runner.chunk_checkpoint import completed_indices, load_chunk
from src.runner.recovery import RecoverySession, atomic_json
from tests.components.background.test_canonical_canvas import _court_pair
from tests.runner.test_background_panorama import _config
from tests.runner.test_run import _all_off, _clip


@pytest.mark.parametrize("change", ["source", "config", "contexts", "implementation"])
def test_changed_identity_cannot_reuse_finished_pixels(tmp_path, monkeypatch, change):
    config, clips, contexts = _all_off(), [_clip(40)], ["court"]
    run(config, clips, context_ids=contexts, checkpoint_dir=tmp_path)
    if change == "source":
        clips[0][0, 0, 0, 0] += 1
    elif change == "config":
        config = replace(config, background=replace(config.background, stream_crf=41))
    elif change == "contexts":
        contexts = ["replay"]
    else:
        monkeypatch.setattr("src.runner.recovery.file_digest", lambda path: "changed-code")
    with pytest.raises(ValueError, match="identity mismatch"):
        run(config, clips, context_ids=contexts, checkpoint_dir=tmp_path)


def test_single_scene_restores_exact_quality_records(tmp_path):
    config, clips = _all_off(), [_clip(40)]
    first = run(config, clips, checkpoint_dir=tmp_path)
    restored = run(config, clips, checkpoint_dir=tmp_path)
    assert restored.quality == first.quality
    assert restored.delivered_quality == first.delivered_quality
    assert restored.chunks[0].reconstruction.quality == first.chunks[0].reconstruction.quality
    assert restored.delivered_quality.scoped  # not a placeholder with no region scores
    assert restored.timing["attempts"] == 2
    assert restored.timing["run_seconds"] > restored.timing["invocation_seconds"]


@pytest.mark.parametrize("contexts", [("court", "court"), ("court", "replay")])
def test_interrupted_background_matches_uninterrupted_run(tmp_path, monkeypatch, contexts):
    clips = list(_court_pair(n_static=2, n_pan=3))
    config = _config(method="panorama-stream")
    config = replace(config, background=replace(config.background, canvas="canonical"))
    expected = run(config, clips, context_ids=contexts, checkpoint_dir=tmp_path / "baseline")
    encode = Encoder.encode
    calls = 0

    def interrupt_second(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return encode(self, *args, **kwargs)

    root = tmp_path / "resume"
    with monkeypatch.context() as patch:
        patch.setattr(Encoder, "encode", interrupt_second)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            run(config, clips, context_ids=contexts, checkpoint_dir=root)
    assert completed_indices(root) == (0,)

    def no_prepass(*args, **kwargs):
        pytest.fail("resume must use the saved canonical canvas")

    monkeypatch.setattr(PanoramaStream, "prepare_contexts", no_prepass)
    restored = run(config, clips, context_ids=contexts, checkpoint_dir=root)
    assert np.array_equal(restored.frames, expected.frames)
    assert np.array_equal(restored.delivered_frames, expected.delivered_frames)
    assert restored.sizes == expected.sizes
    assert restored.quality == expected.quality
    assert restored.delivered_quality == expected.delivered_quality
    assert restored.symmetry == expected.symmetry
    for actual, wanted in zip(restored.chunks, expected.chunks):
        assert actual.sizes == wanted.sizes
        assert actual.quality == wanted.quality
        assert actual.delivered_quality == wanted.delivered_quality
        assert np.array_equal(actual.encoder_frames, wanted.encoder_frames)
        assert np.array_equal(actual.reconstruction.frames, wanted.reconstruction.frames)
        assert actual.reconstruction.quality == wanted.reconstruction.quality
    # Includes reference chains, payload bytes, canvas, and scene coordinates.
    actual = json.loads((root / "chunk_01/meta.json").read_text())
    wanted = json.loads((tmp_path / "baseline/chunk_01/meta.json").read_text())
    assert actual["background_state"] == wanted["background_state"]
    assert restored.timing["attempts"] == 2


def test_preparation_snapshot_survives_failure_before_first_scene(tmp_path, monkeypatch):
    config = _config(method="panorama-stream")
    config = replace(config, background=replace(config.background, canvas="canonical"))
    clips = list(_court_pair(n_static=2, n_pan=3))
    with monkeypatch.context() as patch:
        def interrupt(*args, **kwargs):
            raise RuntimeError("before first scene")
        patch.setattr(Encoder, "encode", interrupt)
        with pytest.raises(RuntimeError, match="before first scene"):
            run(config, clips, checkpoint_dir=tmp_path)
    assert (tmp_path / "prepared/done").is_file()
    assert completed_indices(tmp_path) == ()
    def no_prepass(*args, **kwargs):
        pytest.fail("preparation must not repeat after interruption")
    monkeypatch.setattr(PanoramaStream, "prepare_contexts", no_prepass)
    assert len(run(config, clips, checkpoint_dir=tmp_path).chunks) == 2


@pytest.mark.parametrize("phase", ["runner_identity", "bind_backends", "load_chunk", "_assemble"])
def test_whole_invocation_reports_progress(tmp_path, monkeypatch, phase):
    config, clips = _all_off(), [_clip(40)]
    run(config, clips, checkpoint_dir=tmp_path)
    module = importlib.import_module(
        "src.runner.recovery" if phase == "runner_identity" else
        "src.runner.chunk_checkpoint" if phase == "load_chunk" else "src.runner.run"
    )
    original = getattr(module, phase)
    entered, reported = Event(), Event()
    def emit(message):
        if entered.is_set() and "runner (including" in message and "still running" in message:
            reported.set()
    def blocked(*args, **kwargs):
        entered.set()
        assert reported.wait(2), f"no heartbeat during {phase}"
        return original(*args, **kwargs)
    monkeypatch.setattr("src.pipeline.dag.heartbeat._stdout", emit)
    monkeypatch.setattr(module, phase, blocked)
    run(config, clips, checkpoint_dir=tmp_path, heartbeat_interval=0.01)


def test_timing_counts_failed_attempts_and_flags_hourly_overrun(tmp_path):
    now = [0.0]
    session = RecoverySession(tmp_path, "same", clock=lambda: now[0])
    now[0] = 10.0
    session.checkpoint()
    now[0] = 15.0
    assert session.finish(success=False)["run_seconds"] == 15.0
    session = RecoverySession(tmp_path, "same", clock=lambda: now[0])
    now[0] = 3620.0
    result = session.finish(success=True)
    assert result["run_seconds"] == 3620.0
    assert result["invocation_seconds"] == 3605.0
    assert result["hourly_checkpoint_budget_met"] is False


def test_hard_interruption_cannot_masquerade_as_complete_time(tmp_path):
    atomic_json(tmp_path / "identity.json", {"schema": 2, "identity": "same"})
    atomic_json(tmp_path / "timing.json", [
        {"status": "running", "seconds": 20.0, "max_checkpoint_gap": 20.0},
    ])
    session = RecoverySession(tmp_path, "same", clock=lambda: 100.0)
    result = session.finish(success=True)
    assert result["run_seconds"] is None
    assert result["run_seconds_lower_bound"] == 20.0
    assert result["timing_complete"] is False
    assert result["hourly_checkpoint_budget_met"] is None


@pytest.mark.parametrize("damage", ["changed", "missing", "legacy", "manifest"])
def test_corrupt_snapshot_is_rejected(tmp_path, damage):
    run(_all_off(), [_clip(40)], checkpoint_dir=tmp_path)
    directory = tmp_path / "chunk_00"
    if damage == "changed":
        (directory / "frames.npy").write_bytes(b"corrupt")
    elif damage == "missing":
        (directory / "delivered.npy").unlink()
    elif damage == "legacy":
        (directory / "done").write_text("1")
    else:
        manifest = json.loads((directory / "done").read_text())
        del manifest["frames.npy"]
        (directory / "done").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="incomplete or corrupt checkpoint"):
        load_chunk(tmp_path, 0)


def test_unpublished_snapshot_is_not_complete_and_legacy_root_is_rejected(tmp_path):
    (tmp_path / ".pending-interrupted").mkdir()
    assert completed_indices(tmp_path) == ()
    with pytest.raises(ValueError, match="legacy checkpoint"):
        RecoverySession(tmp_path, "same")
