"""Diagnostic matrix reporter: identity, reuse, and generator-comparison validity."""

from __future__ import annotations

import sqlite3  # noqa: F401
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from experiments.tier.diagnostic_report import (
    REQUIRED_CORNER_KEYS,
    REQUIRED_REPORT_KEYS,
    assess_generator_comparison,
    identity_matches,
    reusable_corners,
)
from scripts.run_diagnostic_matrix import assemble_matrix_report, run_matrix
from src.contracts.config import PointstreamConfig


@dataclass(frozen=True)
class FakeClip:
    video: str
    scene: str
    context_id: str
    frames: np.ndarray
    objects: tuple[Any, ...] = ()


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
        payload = {
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
        repo=Path("/tmp/pointstream-wave1-c"),
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
        repo=Path("/tmp/pointstream-wave1-c"),
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
        repo=Path("/tmp/pointstream-wave1-c"),
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
        repo=Path("/tmp/pointstream-wave1-c"),
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
        repo=Path("/tmp/pointstream-wave1-c"),
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
        repo=Path("/tmp/pointstream-wave1-c"),
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
            "parts": {"residual": 0, "panorama": 1, "actor_reference": 1, "metadata": 1, "transport_total": 3},
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
        repo=Path("/tmp/pointstream-wave1-c"),
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
