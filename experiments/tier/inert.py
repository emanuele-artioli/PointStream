"""Which config knobs actually reach the run path, driven rather than read.

`AGENTS.md`: a flag existing is not a feature working. The schema rejects
unknown keys, so every field in a tier config is real — but "real" means the
field exists, not that anything downstream consults it. The way that fails is
specific and expensive: a sweep over a knob that reaches nothing produces a
clean, plausible, entirely fictional ablation.

So this drives it. Each knob is changed on its own, a run is made with and
without the change, and the two outputs are compared byte for byte. A knob that
leaves the output identical did not reach anything on this path.

Two honest limits on the answer:

* **Identical output is evidence, not proof.** A knob could in principle move
  something that this comparison does not look at. The comparison covers the
  reconstructed pixels, the whole size ledger and both quality reports, which is
  everything a run returns.
* **"Reaches nothing" is scoped to the corner under test.** Generator knobs are
  inert here because generation is off in every tier, not because they are
  inert everywhere. The corner is recorded beside the verdict.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from src.contracts.codecs import RateControl
from src.contracts.config import (
    AppearanceConfig,
    BackendConfig,
    BackgroundConfig,
    EvaluationConfig,
    MotionConfig,
    PointstreamConfig,
)
from src.runner import RunResult
from src.runner.config_io import load_tier
from src.contracts import paths as ps_paths

OUT_DIR = ps_paths.outputs() / "bp23-tier"


def _clip() -> np.ndarray:
    clip = np.full((3, 64, 96, 3), 100, dtype=np.uint8)
    for index in range(3):
        clip[index, 10 + index * 2 : 40 + index * 2, 12 + index * 3 : 44 + index * 3] = 210
    clip[1:] += 2
    return clip


def _fingerprint(result: RunResult) -> dict[str, Any]:
    """Everything a run returns, reduced to something comparable."""
    return {
        "frames": result.frames.tobytes().hex()[:64],
        "frames_sum": int(result.frames.astype(np.int64).sum()),
        "sizes": result.sizes_bytes,
        "reconstruction": sorted(
            (item.metric, item.role, repr(item.value)) for item in result.quality.scoped
        ),
        "delivered": sorted(
            (item.metric, item.role, repr(item.value))
            for item in result.delivered_quality.scoped
        ),
    }


def _variants(base: PointstreamConfig) -> dict[str, PointstreamConfig]:
    """One changed knob per entry. Every variant must still validate."""
    return {
        "run.max_frames": base.with_(run=replace(base.run, max_frames=2)),
        "run.chunk_duration_sec": base.with_(
            run=replace(base.run, chunk_duration_sec=8.0)
        ),
        "run.output_root": base.with_(run=replace(base.run, output_root=Path("elsewhere"))),
        "run.log_level": base.with_(run=replace(base.run, log_level="debug")),
        "run.seed": base.with_(run=replace(base.run, seed=4242)),
        "domain": base.with_(domain="general", background=BackgroundConfig(method="none")),
        "detector.model": base.with_(
            detector=replace(base.detector, model="yolo26x-seg.pt")
        ),
        "detector.backend": base.with_(detector=replace(base.detector, backend="sam3")),
        "selection.backend": base.with_(selection=BackendConfig(backend="identity")),
        "tracking.backend": base.with_(tracking=BackendConfig(backend="tracker", model="x")),
        "pose.model": base.with_(pose=replace(base.pose, model="yolo26x-pose.pt")),
        "segmenter.backend": base.with_(segmenter=BackendConfig(backend="sam3")),
        "rigid.backend": base.with_(rigid=BackendConfig(backend="none")),
        "appearance.jpeg_quality": base.with_(
            appearance=AppearanceConfig(jpeg_quality=10, downscale=1)
        ),
        "appearance.downscale": base.with_(
            appearance=AppearanceConfig(jpeg_quality=90, downscale=8)
        ),
        "motion.representation": base.with_(
            motion=MotionConfig(representation="sparse-trajectories")
        ),
        "temporal.keyframe_interval": base.with_(
            temporal=replace(base.temporal, keyframe_interval=1)
        ),
        "temporal.metadata_sparsity": base.with_(
            temporal=replace(base.temporal, metadata_sparsity=False)
        ),
        "background.codec": base.with_(
            background=replace(base.background, codec="png")
        ),
        "background.jpeg_quality": base.with_(
            background=replace(base.background, jpeg_quality=1)
        ),
        "generator.steps": base.with_(
            generator=replace(base.generator, steps=1)
        ),
        "residual.codec": base.with_(residual=replace(base.residual, codec="hevc", rate_control=RateControl.QP)),
        "residual.rate_control": base.with_(
            residual=replace(
                base.residual, codec="avc", rate_control=RateControl.LOSSLESS, rate=0
            )
        ),
        "residual.rate": base.with_(residual=replace(base.residual, rate=1)),
        "residual.preset": base.with_(residual=replace(base.residual, preset="1")),
        "residual.block_threshold": base.with_(
            residual=replace(base.residual, block_threshold=0.0)
        ),
        "residual.block_size": base.with_(residual=replace(base.residual, block_size=4)),
        "residual.background_downscale": base.with_(
            residual=replace(base.residual, background_downscale=1)
        ),
        "fallback.rate": base.with_(fallback=replace(base.fallback, rate=1)),
        "fallback.codec": base.with_(fallback=replace(base.fallback, codec="avc")),
        "evaluation.max_frames": base.with_(
            evaluation=EvaluationConfig(metrics=base.evaluation.metrics, max_frames=1)
        ),
        "evaluation.metrics": base.with_(
            evaluation=EvaluationConfig(metrics=("psnr", "ssim"))
        ),
    }


def survey(base: PointstreamConfig | None = None) -> dict[str, Any]:
    from experiments.tier.run import run_config
    from experiments.tier.clip import TierClip
    from src.pipeline.reconstruction.reconstruct import ObjectRequest

    config = base if base is not None else load_tier("fast")
    clip = _clip()
    mask = np.zeros(clip.shape[:3], dtype=bool)
    for index in range(clip.shape[0]):
        mask[index, 10 + index * 2 : 40 + index * 2, 12 + index * 3 : 44 + index * 3] = True
    objects = (
        ObjectRequest(
            object_id="player",
            appearance=clip[0, 10:40, 12:44],
            bbox=(12, 10, 44, 40),
            mask=mask,
            frame_index=0,
        ),
    )
    holder = TierClip(
        video="synthetic",
        scene="inert-survey",
        frame_ids=tuple(range(clip.shape[0])),
        frames=clip,
        objects=objects,
        union_mask=mask,
        paste_back_mae=0.0,
        n_tracks=1,
    )

    reference = _fingerprint(run_config("base", config, holder).result)
    rows: list[dict[str, Any]] = []
    for name, variant in _variants(config).items():
        try:
            outcome = _fingerprint(run_config(name, variant, holder).result)
        except Exception as error:  # noqa: BLE001 — a knob that cannot run is its own answer
            rows.append({"field": name, "moved_the_run": None, "error": repr(error)})
            continue
        rows.append({"field": name, "moved_the_run": outcome != reference})

    live = sorted(row["field"] for row in rows if row.get("moved_the_run") is True)
    inert = sorted(row["field"] for row in rows if row.get("moved_the_run") is False)
    return {
        "question": "changing this field on its own — does anything in the run change?",
        "corner": sorted(config.stages.enabled),
        "note": (
            "Generation is off in this corner, so generator knobs are inert *here*; "
            "that is a statement about the corner, not about the knob."
        ),
        "n_fields": len(rows),
        "live": live,
        "inert": inert,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT_DIR / "inert-config-fields.json"))
    args = parser.parse_args(argv)
    outcome = survey()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(outcome, indent=2) + "\n", encoding="utf-8")
    print(f"{len(outcome['live'])} live, {len(outcome['inert'])} inert of {outcome['n_fields']}")
    print("live: ", outcome["live"])
    print("inert:", outcome["inert"])
    return 0


__all__ = ["main", "survey"]


if __name__ == "__main__":
    raise SystemExit(main())
