"""Diagnostic matrix reporter: identity, reuse, and generator-comparison validity."""

from __future__ import annotations

import sqlite3  # noqa: F401
from dataclasses import dataclass, replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from experiments.tier.diagnostic_report import (
    REQUIRED_CORNER_KEYS,
    REQUIRED_REPORT_KEYS,
    assess_generator_comparison,
    build_run_identity,
    identity_matches,
    resolved_configuration,
    reusable_corners,
)
from scripts.run_diagnostic_matrix import (
    _augment_objects_with_pose,
    assemble_matrix_report,
    resolve_clip_start_frame,
    run_matrix,
)
from src.contracts.config import PointstreamConfig
from src.pipeline.reconstruction.reconstruct import ObjectRequest

# The checkout that contains this test file. Do not pin a local worktree path;
# CI has no /tmp/pointstream-wave1-c.
_REPO = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FakeClip:
    video: str
    scene: str
    context_id: str
    frames: np.ndarray
    objects: tuple[Any, ...] = ()
    start_frame: int | None = None


class FakeSizes:
    def __init__(
        self,
        *,
        residual: int = 0,
        panorama: int = 20,
        actor_reference: int = 30,
        metadata: int = 5,
        transport_total: int = 55,
        subledger: dict[str, int] | None = None,
    ) -> None:
        self.residual = residual
        self.panorama = panorama
        self.actor_reference = actor_reference
        self.metadata = metadata
        self.transport_total = transport_total
        self.subledger = subledger

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "residual": self.residual,
            "panorama": self.panorama,
            "actor_reference": self.actor_reference,
            "metadata": self.metadata,
            "transport_total": self.transport_total,
        }
        if self.subledger is not None:
            payload["subledger"] = dict(self.subledger)
        return payload


class FakeBackend:
    def generate(self, *args: Any, **kwargs: Any) -> np.ndarray:
        return np.zeros((4, 4, 3), dtype=np.uint8)


class FakeGeneratorRef:
    def __init__(self) -> None:
        self.backend = FakeBackend()


class FakeResult:
    def __init__(self, delivered: np.ndarray, sizes: FakeSizes) -> None:
        self.delivered_frames = delivered
        self.sizes = sizes
        self.timing = {
            "encoder_seconds": 0.11,
            "client_seconds": 0.07,
            "evaluation_seconds": 0.03,
        }
        self.encoder_seconds = 0.11
        self.client_seconds = 0.07
        self.evaluation_seconds = 0.03
        wire = b"\x00" * int(sizes.transport_total)
        self.chunks = (SimpleNamespace(bag={"wire_request": wire}, sizes=sizes),)


def _clip() -> FakeClip:
    frames = np.zeros((2, 16, 16, 3), dtype=np.uint8)
    frames[0, :, :, 0] = 40
    frames[1, :, :, 1] = 80
    return FakeClip(video="alcaraz_highlights", scene="scene_000", context_id="ctx", frames=frames)


def _score(reference: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {"psnr_y": 32.0, "ssim": 0.91, "vmaf": 70.0}


def _factory(*args: Any, **kwargs: Any) -> FakeGeneratorRef:
    return FakeGeneratorRef()


def _run_changing_pixels(
    cfg: Any,
    sources: Any,
    objects: Any = None,
    context_ids: Any = None,
    generator: Any = None,
) -> FakeResult:
    frames = np.asarray(sources[0])
    delivered = frames.copy()
    gen_on = bool(getattr(cfg.lattice, "generation", False))
    res_on = bool(getattr(cfg.lattice, "residual", False))
    if generator is not None:
        backend = getattr(generator, "backend", generator)
        backend.generate(None)
        delivered = delivered.copy()
        delivered[0, 0, 0] = (int(delivered[0, 0, 0, 0]) + 17) % 256
        if getattr(cfg.generator, "backend", "") and "shuffle" in str(objects):
            delivered[0, 1, 1] = 200
    residual = 40 if res_on else 0
    if gen_on and res_on:
        residual = 25
    total = 20 + 30 + 5 + residual
    return FakeResult(
        delivered,
        FakeSizes(
            residual=residual,
            transport_total=total,
            subledger={"residual": residual, "other": total - residual},
        ),
    )


def test_reporter_includes_required_keys(tmp_path: Path) -> None:
    clip = _clip()
    ckpt = tmp_path / "pix2pix_generator.pt"
    ckpt.write_bytes(b"dummy-weights")
    report = run_matrix(
        clip,
        PointstreamConfig(),
        generator_arch="pix2pix",
        residual_qp=32,
        checkpoint=ckpt,
        shuffled_control=True,
        device="cpu",
        frames=2,
        run_fn=_run_changing_pixels,
        score_fn=_score,
        generator_factory=_factory,
        repo=_REPO,
    )
    for key in REQUIRED_REPORT_KEYS:
        assert key in report, f"missing report key {key}"
    assert report["video"] == "alcaraz_highlights"
    assert report["scene"] == "scene_000"
    assert report["identity"]["checkpoint_sha256"]
    assert report["identity"]["code_revision"]["commit"]
    assert report["source_manifest"]["frame_hashes"]
    assert report["generator_comparison_valid"] is True
    corners = {row["corner"] for row in report["matrix"]}
    assert "gen_on_shuffled_conditioning" in corners
    for row in report["matrix"]:
        for key in REQUIRED_CORNER_KEYS:
            assert key in row, f"{row.get('corner')} missing {key}"
        assert row["timing"]["encoder_seconds"] == 0.11
        assert row["timing"]["client_seconds"] == 0.07
        assert row["timing"]["evaluation_seconds"] == 0.03
        assert row["wire_reconciliation"]["matched"] is True
        assert row["byte_subledger"] is not None
        assert row["failure"] is None


def test_missing_checkpoint_marks_comparison_invalid(tmp_path: Path) -> None:
    clip = _clip()
    missing = tmp_path / "not-a-checkpoint.pt"
    report = run_matrix(
        clip,
        PointstreamConfig(),
        generator_arch="pix2pix",
        residual_qp=32,
        checkpoint=missing,
        shuffled_control=False,
        device="cpu",
        frames=2,
        run_fn=_run_changing_pixels,
        score_fn=_score,
        generator_factory=_factory,
        repo=_REPO,
    )
    assert report["checkpoint_sha256"] is None
    assert report["generator_comparison_valid"] is False
    gen_rows = [row for row in report["matrix"] if row["generation_on"]]
    assert gen_rows
    assert all(row["failure"] is not None for row in gen_rows)
    assert all(row["failure"]["type"] == "FileNotFoundError" for row in gen_rows)


def test_noop_generator_marks_comparison_invalid(tmp_path: Path) -> None:
    clip = _clip()
    ckpt = tmp_path / "weights.pt"
    ckpt.write_bytes(b"dummy-weights")

    def run_without_calling_generator(
        cfg: Any,
        sources: Any,
        objects: Any = None,
        context_ids: Any = None,
        generator: Any = None,
    ) -> FakeResult:
        frames = np.asarray(sources[0])
        return FakeResult(frames.copy(), FakeSizes(transport_total=55))

    report = run_matrix(
        clip,
        PointstreamConfig(),
        generator_arch="pix2pix",
        residual_qp=32,
        checkpoint=ckpt,
        shuffled_control=False,
        device="cpu",
        frames=2,
        run_fn=run_without_calling_generator,
        score_fn=_score,
        generator_factory=_factory,
        repo=_REPO,
    )
    assert report["generator_comparison_valid"] is False
    reasons = " ".join(report["generator_comparison"]["reasons"])
    assert "no-op" in reasons or "did not change delivered pixels" in reasons


def test_reuse_refuses_mismatched_identity(tmp_path: Path) -> None:
    clip = _clip()
    ckpt = tmp_path / "weights.pt"
    ckpt.write_bytes(b"dummy-weights")
    first = run_matrix(
        clip,
        PointstreamConfig(),
        generator_arch="pix2pix",
        residual_qp=32,
        checkpoint=ckpt,
        shuffled_control=False,
        device="cpu",
        frames=2,
        run_fn=_run_changing_pixels,
        score_fn=_score,
        generator_factory=_factory,
        repo=_REPO,
    )
    other_ckpt = tmp_path / "other.pt"
    other_ckpt.write_bytes(b"other-weights")
    reused = reusable_corners(
        first,
        {
            **first["identity"],
            "checkpoint_sha256": "0" * 64,
        },
    )
    assert reused == {}
    assert identity_matches(first["identity"], first["identity"]) is True
    mismatched = dict(first["identity"])
    mismatched["source_frame_hashes"] = ["deadbeef"]
    assert identity_matches(first["identity"], mismatched) is False
    mismatched_rev = dict(first["identity"])
    mismatched_rev["code_revision"] = {"commit": "abc", "dirty": False}
    assert identity_matches(first["identity"], mismatched_rev) is False

    calls = {"n": 0}

    def counting_run(*args: Any, **kwargs: Any) -> FakeResult:
        calls["n"] += 1
        return _run_changing_pixels(*args, **kwargs)

    reuse_path = tmp_path / "prior.json"
    reuse_path.write_text(json.dumps(first))
    second = run_matrix(
        clip,
        PointstreamConfig(),
        generator_arch="pix2pix",
        residual_qp=32,
        checkpoint=other_ckpt,
        shuffled_control=False,
        device="cpu",
        frames=2,
        reuse_path=reuse_path,
        run_fn=counting_run,
        score_fn=_score,
        generator_factory=_factory,
        repo=_REPO,
    )
    assert calls["n"] == len(second["matrix"])
    before_reuse = calls["n"]
    matching = run_matrix(
        clip,
        PointstreamConfig(),
        generator_arch="pix2pix",
        residual_qp=32,
        checkpoint=ckpt,
        shuffled_control=False,
        device="cpu",
        frames=2,
        reuse_path=reuse_path,
        run_fn=counting_run,
        score_fn=_score,
        generator_factory=_factory,
        repo=_REPO,
    )
    assert matching["matrix"][0]["corner"] == first["matrix"][0]["corner"]
    assert calls["n"] == before_reuse


def test_assemble_report_lists_required_top_level_keys() -> None:
    clip = _clip()
    cfg = PointstreamConfig()
    matrix = [
        {
            "corner": "gen_off_res_off",
            "generation_on": False,
            "residual_on": False,
            "shuffled_conditioning": False,
            "delivered_frame_hashes": ["aa"],
            "base_frame_hashes": ["bb"],
            "parts": {
                "residual": 0,
                "panorama": 1,
                "actor_reference": 1,
                "metadata": 1,
                "transport_total": 3,
            },
            "byte_subledger": None,
            "wire_reconciliation": {"verdict": "wire_request_absent"},
            "timing": {"encoder_seconds": 0.1, "client_seconds": 0.1, "evaluation_seconds": 0.1},
            "failure": None,
            "model_invocation_count": 0,
            "control": "pasted_reference",
        }
    ]
    report = assemble_matrix_report(
        video=clip.video,
        scene=clip.scene,
        frames=2,
        generator="pix2pix",
        residual_qp=32,
        clip=clip,
        base_config=cfg,
        matrix=matrix,
        checkpoint_path=None,
        checkpoint_sha256=None,
        device="cpu",
        shuffled_control=False,
        repo=_REPO,
    )
    for key in REQUIRED_REPORT_KEYS:
        assert key in report
    assert report["generator_comparison_valid"] is False


def test_assess_generator_comparison_requires_identity() -> None:
    verdict = assess_generator_comparison(
        checkpoint_sha256=None,
        generator_backend="pix2pix",
        matrix=[],
    )
    assert verdict["generator_comparison_valid"] is False
    assert "missing checkpoint SHA-256" in verdict["reasons"]


def test_identity_independent_variations_cause_cache_miss() -> None:
    clip = _clip()
    cfg = PointstreamConfig()
    repo_rev = {"commit": "commit123", "dirty": False, "diff_sha256": None}
    base_identity = build_run_identity(
        code_revision=repo_rev,
        checkpoint_sha256="a" * 64,
        source_frame_hashes=["hash1", "hash2"],
        config=resolved_configuration(cfg, device="cpu", objects=clip.objects),
        video=clip.video,
        scene=clip.scene,
        frames=2,
        generator="pix2pix",
        residual_qp=32,
    )
    prior = {
        "identity": base_identity,
        "matrix": [
            {
                "corner": "gen_off_res_off",
                "generation_on": False,
                "residual_on": False,
                "shuffled_conditioning": False,
                "delivered_frame_hashes": ["hash1", "hash2"],
            }
        ],
    }

    # 1. Identical complete identity reuses
    assert identity_matches(base_identity, base_identity) is True
    assert len(reusable_corners(prior, base_identity)) == 1

    # 2. Background quality change causes cache miss
    cfg_bg = replace(cfg, background=replace(cfg.background, jpeg_quality=95))
    id_bg = dict(base_identity)
    id_bg["config"] = resolved_configuration(cfg_bg, device="cpu", objects=clip.objects)
    assert identity_matches(id_bg, base_identity) is False
    assert reusable_corners(prior, id_bg) == {}

    # 3. Device change causes cache miss
    id_device = dict(base_identity)
    id_device["config"] = resolved_configuration(cfg, device="cuda:0", objects=clip.objects)
    assert identity_matches(id_device, base_identity) is False
    assert reusable_corners(prior, id_device) == {}

    # 4. Pose/mask change causes cache miss
    obj_orig = ObjectRequest(
        object_id="track_0001",
        appearance=np.zeros((10, 10, 3), dtype=np.uint8),
        bbox=(0, 0, 10, 10),
        mask=np.zeros((2, 10, 10), dtype=bool),
        frame_index=0,
    )
    obj_mod_mask = replace(obj_orig, mask=np.ones((2, 10, 10), dtype=bool))
    id_orig_obj = dict(base_identity)
    id_orig_obj["config"] = resolved_configuration(cfg, device="cpu", objects=(obj_orig,))
    id_mod_mask = dict(base_identity)
    id_mod_mask["config"] = resolved_configuration(cfg, device="cpu", objects=(obj_mod_mask,))
    assert identity_matches(id_mod_mask, id_orig_obj) is False

    # 5. Code changes cause cache miss
    # a. Different commit
    id_code_commit = dict(base_identity)
    id_code_commit["code_revision"] = {"commit": "commit456", "dirty": False, "diff_sha256": None}
    assert identity_matches(id_code_commit, base_identity) is False

    # b. Dirty code without diff hash (unhashed dirty reuse refused)
    id_code_dirty_unhashed = dict(base_identity)
    id_code_dirty_unhashed["code_revision"] = {
        "commit": "commit123",
        "dirty": True,
        "diff_sha256": None,
    }
    assert identity_matches(id_code_dirty_unhashed, base_identity) is False

    # c. Dirty code with differing diff hash
    id_code_dirty1 = dict(base_identity)
    id_code_dirty1["code_revision"] = {
        "commit": "commit123",
        "dirty": True,
        "diff_sha256": "1" * 64,
    }
    id_code_dirty2 = dict(base_identity)
    id_code_dirty2["code_revision"] = {
        "commit": "commit123",
        "dirty": True,
        "diff_sha256": "2" * 64,
    }
    assert identity_matches(id_code_dirty1, id_code_dirty2) is False

    # d. Dirty code with matching diff hash matches
    assert identity_matches(id_code_dirty1, id_code_dirty1) is True


def test_failed_paste_corners_with_successful_generation_invalidates_comparison() -> None:
    matrix = [
        {
            "corner": "gen_off_res_off",
            "generation_on": False,
            "residual_on": False,
            "shuffled_conditioning": False,
            "delivered_frame_hashes": ["paste_hash1", "paste_hash2"],
            "failure": {"type": "RuntimeError", "message": "paste failed"},
            "model_invocation_count": 0,
            "control": "pasted_reference",
        },
        {
            "corner": "gen_off_res_on",
            "generation_on": False,
            "residual_on": True,
            "shuffled_conditioning": False,
            "delivered_frame_hashes": ["paste_hash1", "paste_hash2"],
            "failure": None,
            "model_invocation_count": 0,
            "control": "pasted_reference",
        },
        {
            "corner": "gen_on_res_off",
            "generation_on": True,
            "residual_on": False,
            "shuffled_conditioning": False,
            "delivered_frame_hashes": ["gen_hash1", "gen_hash2"],
            "failure": None,
            "model_invocation_count": 1,
        },
        {
            "corner": "gen_on_res_on",
            "generation_on": True,
            "residual_on": True,
            "shuffled_conditioning": False,
            "delivered_frame_hashes": ["gen_hash1", "gen_hash2"],
            "failure": None,
            "model_invocation_count": 1,
        },
    ]
    verdict = assess_generator_comparison(
        checkpoint_sha256="a" * 64,
        generator_backend="pix2pix",
        matrix=matrix,
    )
    assert verdict["generator_comparison_valid"] is False
    assert "paste control corner failed" in verdict["reasons"]


def test_two_scenes_distinct_frame_offsets_and_absent_pose_rejection() -> None:
    # Test two scenes with distinct frame offsets from manifest
    clip_alcaraz = FakeClip(
        video="alcaraz_highlights",
        scene="scene_000",
        context_id="ctx",
        frames=np.zeros((48, 16, 16, 3), dtype=np.uint8),
    )
    offset_alcaraz = resolve_clip_start_frame(clip_alcaraz, n_frames=48)
    assert offset_alcaraz == 38

    clip_federer = FakeClip(
        video="federer_djokovic",
        scene="scene_007",
        context_id="ctx",
        frames=np.zeros((48, 16, 16, 3), dtype=np.uint8),
    )
    offset_federer = resolve_clip_start_frame(clip_federer, n_frames=48)
    assert offset_federer == 68

    assert offset_alcaraz != offset_federer

    # Test absent pose: missing required conditioning fails closed (no synthetic fallback)
    obj = ObjectRequest(
        object_id="track_nonexistent_9999",
        appearance=np.zeros((32, 32, 3), dtype=np.uint8),
        bbox=(0, 0, 32, 32),
        mask=np.zeros((2, 32, 32), dtype=bool),
        frame_index=0,
    )
    clip_missing_pose = FakeClip(
        video="alcaraz_highlights",
        scene="scene_000",
        context_id="ctx",
        frames=np.zeros((2, 32, 32, 3), dtype=np.uint8),
        objects=(obj,),
    )
    with pytest.raises(FileNotFoundError) as exc_info:
        _augment_objects_with_pose(clip_missing_pose, shuffle=False, seed=42)
    assert "Missing required pose conditioning skeleton" in str(exc_info.value)


def test_augment_objects_with_pose_resolves_positionally(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import cv2

    monkeypatch.setenv("PS_DATA_ROOT", str(tmp_path))
    scene_dir = tmp_path / "assets" / "dataset" / "fake_video" / "segmentations" / "fake_scene"
    crop_dir = scene_dir / "track_0001"
    skel_dir = scene_dir / "track_0001_skeleton"
    crop_dir.mkdir(parents=True)
    skel_dir.mkdir(parents=True)

    # Frame 29 is position 0
    img = np.zeros((32, 24, 3), dtype=np.uint8)
    cv2.imwrite(str(crop_dir / "frame_000029.png"), img)
    cv2.imwrite(str(skel_dir / "frame_000000.png"), img)

    obj = ObjectRequest(
        object_id="track_0001",
        appearance=np.zeros((32, 24, 3), dtype=np.uint8),
        bbox=(0, 0, 32, 24),
        mask=np.zeros((1, 32, 24), dtype=bool),
        frame_index=0,
    )
    clip = FakeClip(
        video="fake_video",
        scene="fake_scene",
        context_id="ctx",
        frames=np.zeros((1, 32, 24, 3), dtype=np.uint8),
        objects=(obj,),
        start_frame=29,
    )
    augmented = _augment_objects_with_pose(clip, shuffle=False, seed=42)
    assert len(augmented) == 1
    assert augmented[0].conditioning is not None
    assert augmented[0].conditioning.pose.shape == (3, 32, 24)
