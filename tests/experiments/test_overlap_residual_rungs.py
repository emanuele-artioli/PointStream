"""Overlap residual rungs must be coarser than H0 and still include H0–H3."""

from __future__ import annotations

import json

from experiments.tier.resolution_adaptive import (
    HIGH_FIDELITY_RESIDUAL_RUNGS,
    OVERLAP_RESIDUAL_RUNGS,
)
from src.contracts import paths as ps_paths


def test_overlap_rungs_extend_high_fidelity_downward() -> None:
    hf_ids = tuple(rung.rung_id for rung in HIGH_FIDELITY_RESIDUAL_RUNGS)
    overlap_ids = tuple(rung.rung_id for rung in OVERLAP_RESIDUAL_RUNGS)
    assert hf_ids == ("H0", "H1", "H2", "H3")
    assert overlap_ids[:3] == ("R63", "R55", "R48")
    assert overlap_ids[-4:] == hf_ids
    qp_by_id = {rung.rung_id: rung.residual_qp for rung in OVERLAP_RESIDUAL_RUNGS}
    assert qp_by_id["R63"] > qp_by_id["H0"] == 42
    assert qp_by_id["R55"] > qp_by_id["H0"]
    assert qp_by_id["R48"] > qp_by_id["H0"]


def test_wave2_overlap_manifest_adds_higher_fidelity_anchor_qps() -> None:
    data = json.loads(
        (ps_paths.repo_root() / "manifests" / "wave2_overlap_ladder.json").read_text(
            encoding="utf-8"
        )
    )
    for codec in ("av1", "vvc"):
        qps = data["anchors"][codec]["qps"]
        assert qps[:4] == [63, 55, 47, 39]
        assert qps[-3:] == [35, 31, 27]
    scenes = {(item["video"], item["scene"]) for item in data["scenes"]}
    assert scenes == {("alcaraz_highlights", "scene_000"), ("federer_djokovic", "scene_007")}
    forbidden = set(data["holdout_protection"]["confirmation_sources_forbidden"])
    for item in data["scenes"]:
        assert item["source_id"] not in forbidden
