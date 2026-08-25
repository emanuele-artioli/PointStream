"""Drive the shipped tier configs end to end and write what came back.

The point of this module is that the three layers meet: a config file on disk,
`src.runner.run`, and a scored reconstruction. It is deliberately thin — the
work belongs in the runner, and anything interesting that happens here would be
a sign the runner is missing a seam.

Every run carries its own controls, in the same session as the measurement:

* **the all-off corner**, which must come back bit-identical to the source. It
  is the anchor that says the path did not quietly lose or gain pixels.
* **a call count on every disabled stage**, because a stage configured off that
  still runs makes every ablation number meaningless. A flag existing is not a
  feature working.
* **a generator factory that raises**, so "generation is off" is proved by the
  generator never being constructed rather than asserted.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tier.clip import TierClip, load_tier_clip
from src.contracts.config import PointstreamConfig
from src.contracts.lattice import OPTIONAL_STAGES, SOURCE_PASSTHROUGH
from src.pipeline.reconstruction.dispatch import GeneratorRef
from src.pipeline.reconstruction.quality import QualityReport
from src.pipeline.residual.spectrum import point_for
from src.runner import RunResult, lattice_config_from, run
from src.runner.config_io import load_tier

TIERS = ("fast", "balanced", "quality")

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "outputs" / "bp23-tier"


class StageCounter:
    """Wraps a stage callable and counts invocations. Zero is the assertion."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    def __call__(self, bag: Mapping[str, Any]) -> tuple[()]:
        self.calls += 1
        return ()


def _no_generator() -> GeneratorRef:
    raise AssertionError(
        "generation is off in this config; constructing a generator would mean "
        "the lattice corner is not the one the config names"
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, float):
        return None if math.isnan(value) else (str(value) if math.isinf(value) else value)
    if isinstance(value, (np.floating,)):
        return _jsonable(float(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def quality_record(report: QualityReport) -> dict[str, Any]:
    """Every scored region, labelled. A score whose scope is unstated is unusable."""
    return _jsonable(
        {
            "bit_identical": report.bit_identical,
            "mean_abs_diff": report.closeness.mean_abs_diff,
            "max_abs_diff": report.closeness.max_abs_diff,
            "pixel_psnr_dB": report.closeness.psnr,
            "enforced": list(report.enforced),
            "scoped": [
                {
                    "metric": item.metric,
                    "role": item.role,
                    "value": item.value,
                    "n_pixels": item.n_pixels,
                    "name": item.name,
                }
                for item in report.scoped
            ],
        }
    )


def config_record(config: PointstreamConfig) -> dict[str, Any]:
    point = point_for(config.stages, config.residual)
    return {
        "corner": sorted(config.stages.enabled),
        "disabled": sorted(set(OPTIONAL_STAGES) - set(config.stages.enabled)),
        "residual_point": {
            "coarseness": point.coarseness.value,
            "variant": point.variant.value,
            "describes": point.describe(),
        },
        "requested_metrics": list(config.evaluation.metrics),
        "generator": config.generator.resolved_name,
        "detector": f"{config.detector.backend}:{config.detector.model}",
        "background": config.background.method,
    }


@dataclass(frozen=True)
class TierRun:
    name: str
    config: PointstreamConfig
    result: RunResult
    seconds: float
    disabled_stage_calls: dict[str, int]

    def record(self) -> dict[str, Any]:
        return {
            "tier": self.name,
            "config": config_record(self.config),
            "metrics_reported": sorted(
                {item.metric for item in self.result.delivered_quality.scoped}
            ),
            "wall_clock_seconds": round(self.seconds, 2),
            "reconstruction_quality": quality_record(self.result.quality),
            "delivered_quality": quality_record(self.result.delivered_quality),
            "sizes_bytes": _jsonable(self.result.sizes_bytes),
            "encoder_client_symmetry": _jsonable(
                {
                    "bit_identical": self.result.symmetry.bit_identical,
                    "mean_abs_diff": self.result.symmetry.mean_abs_diff,
                    "max_abs_diff": self.result.symmetry.max_abs_diff,
                    "psnr_dB": self.result.symmetry.psnr,
                }
            ),
            "disabled_stage_calls": self.disabled_stage_calls,
        }


def run_config(
    name: str,
    config: PointstreamConfig,
    clip: TierClip,
    *,
    objects: bool = True,
) -> TierRun:
    """One config, one chunk, with the disabled stages instrumented."""
    disabled = sorted(set(OPTIONAL_STAGES) - set(config.stages.enabled))
    counters = {stage: StageCounter(stage) for stage in disabled}
    started = time.time()
    result = run(
        config,
        [clip.frames],
        backends=dict(counters),
        bind_generator_fn=_no_generator,
        objects=((clip.objects,) if objects else None),
    )
    seconds = time.time() - started
    return TierRun(
        name=name,
        config=config,
        result=result,
        seconds=seconds,
        disabled_stage_calls={stage: counter.calls for stage, counter in counters.items()},
    )


def run_all_off(clip: TierClip) -> TierRun:
    """The null control. All-off is a lattice corner, not a special path."""
    config = PointstreamConfig(lattice=lattice_config_from(SOURCE_PASSTHROUGH))
    return run_config("all-off (control)", config, clip, objects=False)


def run_residual_absent(clip: TierClip) -> TierRun:
    """`tier_fast` with the residual switched off — the unaided reconstruction.

    Named in the Phase C gate (`PLAN.md` §8): a residual-absent run has to
    complete and report its measured quality drop, rather than the residual
    being a stage the pipeline cannot do without. It is also the honest floor
    for the tier ladder: everything the three tiers gain, they gain from here.
    """
    base = load_tier("fast")
    config = base.with_(lattice=replace(base.lattice, residual=False))
    return run_config("residual-absent (control)", config, clip)


def probe_perception_knobs(clip: TierClip) -> dict[str, Any]:
    """Do the detector / pose / segmenter names in a tier config change anything?

    A flag existing is not a feature working. `tier_fast` and `tier_quality`
    name different perception backends, and the runner's default stage
    callables for those axes are pass-throughs that do not load a detector — so
    the prediction is that swapping the names leaves the output byte-identical.
    That is a prediction to drive, not an assumption to state: this runs
    `tier_fast` twice, once with `tier_quality`'s perception names grafted on,
    and reports whether anything moved.
    """
    base = load_tier("fast")
    other = load_tier("quality")
    swapped = base.with_(
        detector=other.detector,
        pose=other.pose,
        segmenter=other.segmenter,
        selection=other.selection,
        tracking=other.tracking,
    )
    first = run_config("fast", base, clip)
    second = run_config("fast+quality-perception-names", swapped, clip)
    identical = bool(np.array_equal(first.result.frames, second.result.frames))
    return {
        "question": "does naming a different detector/pose/segmenter change a run?",
        "swapped": {
            "detector": [
                f"{base.detector.backend}:{base.detector.model}",
                f"{swapped.detector.backend}:{swapped.detector.model}",
            ],
            "pose": [
                f"{base.pose.backend}:{base.pose.model}",
                f"{swapped.pose.backend}:{swapped.pose.model}",
            ],
            "segmenter": [
                f"{base.segmenter.backend}:{base.segmenter.model}",
                f"{swapped.segmenter.backend}:{swapped.segmenter.model}",
            ],
        },
        "frames_bit_identical": identical,
        "sizes_identical": first.result.sizes_bytes == second.result.sizes_bytes,
        "delivered_psnr_dB": [
            _jsonable(first.result.delivered_quality.whole_frame()),
            _jsonable(second.result.delivered_quality.whole_frame()),
        ],
        "verdict": (
            "INERT — the perception names in a tier config reach nothing today. The "
            "runner's default stage callables forward artifacts the caller supplied "
            "and never load a backend."
            if identical
            else "the names moved the output; the assumption that they are inert is wrong"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the shipped tier configs end to end.")
    parser.add_argument("--tiers", nargs="*", default=list(TIERS))
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--video", default=None)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--out", default=str(OUT_DIR / "report.json"))
    parser.add_argument(
        "--probe-perception",
        action="store_true",
        help="only check whether the perception backend names change a run",
    )
    args = parser.parse_args(argv)

    kwargs: dict[str, Any] = {"n_frames": args.frames}
    if args.video:
        kwargs["video"] = args.video
    if args.scene:
        kwargs["scene"] = args.scene
    clip = load_tier_clip(**kwargs)
    print(f"clip {clip.video}/{clip.scene} {clip.describe()}", flush=True)

    if args.probe_perception:
        probe = probe_perception_knobs(clip)
        destination = Path(args.out)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps({"clip": clip.describe(), "probe": probe}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(probe, indent=2), flush=True)
        return 0

    records: list[dict[str, Any]] = []
    control = run_all_off(clip)
    control_record = control.record()
    control_record["source_bit_identical"] = bool(
        np.array_equal(clip.frames, control.result.frames)
    )
    records.append(control_record)
    print(
        f"control all-off  {control.seconds:.1f}s  "
        f"bit_identical={control_record['source_bit_identical']}",
        flush=True,
    )

    unaided = run_residual_absent(clip)
    records.append(unaided.record())
    print(
        f"control residual-absent {unaided.seconds:.1f}s  "
        f"delivered psnr={unaided.result.delivered_quality.whole_frame():.2f} dB  "
        f"residual={unaided.result.sizes.residual} B",
        flush=True,
    )

    for tier in args.tiers:
        config = load_tier(tier)
        outcome = run_config(tier, config, clip)
        records.append(outcome.record())
        print(
            f"tier {tier:9s} {outcome.seconds:7.1f}s  "
            f"delivered psnr="
            f"{outcome.result.delivered_quality.whole_frame():.2f} dB  "
            f"residual={outcome.result.sizes.residual} B",
            flush=True,
        )

    payload = {
        "brief": "BP23 — one tier config, end to end",
        "bounds_written_before_measurement": "outputs/bp23-tier/bounds-before-run.json",
        "not_a_pointstream_rate_result": (
            "The runner's codec stage is an identity round-trip and no encoder "
            "binary runs. Byte counts here are pixel-payload bytes, not coded "
            "bytes; transport_to_source_ratio is not a compression ratio."
        ),
        "clip": clip.describe(),
        "runs": records,
    }
    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {destination}", flush=True)
    return 0


__all__ = ["TIERS", "TierRun", "main", "run_all_off", "run_config"]
