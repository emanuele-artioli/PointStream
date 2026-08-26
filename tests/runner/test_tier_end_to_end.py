"""Required behaviour: a shipped tier config runs end to end and is scored.

This is the gate for `PLAN.md` §7 P0 item 1. It exists because the three parts
— the config files, `src/runner`, and the metric set — were each green on their
own for weeks while never having met. Unit tests on the parts cannot catch
that; only driving a real file from `config/` can.

What is deliberately *not* asserted here: that the numbers are good. Every
generative engine on the roster loses to pasting the keyframe (`PLAN.md` §2.10)
and these tiers run with generation off, so a quality bar in this file would be
a bar on the residual, not on the platform. The properties below are the ones
whose failure means the platform is broken rather than the model being weak.

The clip is synthetic and tiny on purpose: this is a path gate that has to run
in CI, not a measurement. The measurement lives in `experiments/tier`, which
uses a real 4K clip.
"""

from __future__ import annotations

import dataclasses
import math
import shutil
import subprocess

import numpy as np
import pytest

from src.contracts.config import PointstreamConfig
from src.contracts.lattice import (
    OPTIONAL_STAGES,
    SOURCE_PASSTHROUGH,
    STAGE_GENERATION,
    STAGE_RESIDUAL,
)
from src.pipeline.reconstruction.dispatch import GeneratorRef
from src.pipeline.reconstruction.quality import ROLE_BACKGROUND, ROLE_FRAME, ROLE_OBJECT
from src.pipeline.reconstruction.reconstruct import ObjectRequest
from src.runner import lattice_config_from, run
from src.runner.config_io import CONFIG_DIR, load_tier

TIERS = ("fast", "balanced", "quality")


def _ffmpeg_has_libvmaf() -> bool:
    """CI's apt ffmpeg is not built with libvmaf. The quality tier asks for VMAF."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    result = subprocess.run(
        [ffmpeg, "-hide_banner", "-filters"],
        capture_output=True,
        text=True,
        check=False,
    )
    return "libvmaf" in result.stdout


def _moving_block_clip(frames: int = 3, height: int = 96, width: int = 128):
    """A grey plate with one bright block that moves — the cheapest thing that
    makes a static background reconstruction wrong in a known place."""
    clip = np.full((frames, height, width, 3), 100, dtype=np.uint8)
    mask = np.zeros((frames, height, width), dtype=bool)
    for index in range(frames):
        top, left = 20 + index * 3, 30 + index * 4
        clip[index, top : top + 40, left : left + 40] = 220
        mask[index, top : top + 40, left : left + 40] = True
    return clip, mask


def _objects(clip: np.ndarray, mask: np.ndarray) -> tuple[ObjectRequest, ...]:
    return (
        ObjectRequest(
            object_id="player",
            appearance=clip[0, 20:60, 30:70],
            bbox=(30, 20, 70, 60),
            mask=mask,
            frame_index=0,
        ),
    )


def _never_constructs() -> GeneratorRef:
    raise AssertionError("a tier with generation off must not construct a generator")


def _ffmpeg_has_libvmaf() -> bool:
    """CI's apt ffmpeg is not built with libvmaf. The quality file still names VMAF."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    process = subprocess.run(
        [ffmpeg, "-hide_banner", "-filters"],
        capture_output=True,
        text=True,
        check=False,
    )
    return "libvmaf" in (process.stdout or "")


def _light_perception() -> dict[str, object]:
    """Stand-ins so a tier path test does not load YOLO pose/seg weights."""

    class _SkipPose:
        def estimate(self, frame, detection, **kwargs):  # noqa: ANN001
            _ = (frame, detection, kwargs)
            return None

    class _SkipSeg:
        def segment(self, frame, detection):  # noqa: ANN001
            _ = (frame, detection)
            return None

    return {"pose": _SkipPose(), "segmenter": _SkipSeg()}


def _run_tier(name: str):
    if name == "quality" and not _ffmpeg_has_libvmaf():
        pytest.skip("quality tier asks for VMAF; this ffmpeg has no libvmaf")
    clip, mask = _moving_block_clip()
    config = load_tier(name)
    asked = tuple(config.evaluation.metrics)
    if "vmaf" in asked and not _ffmpeg_has_libvmaf():
        # Path gate, not a metric gate. Main is red on these tests for the same
        # reason (PR #22). Stream E / BP27 owns making VMAF run in CI.
        config = config.with_(
            evaluation=dataclasses.replace(
                config.evaluation,
                metrics=[metric for metric in asked if metric != "vmaf"],
            )
        )
    counters = {
        stage: _Counter() for stage in set(OPTIONAL_STAGES) - set(config.stages.enabled)
    }
    result = run(
        config,
        [clip],
        backends=dict(counters),
        bind_generator_fn=_never_constructs,
        objects=(_objects(clip, mask),),
        components=_light_perception(),
    )
    return config, result, counters, clip


class _Counter:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, bag) -> tuple[()]:  # noqa: ANN001
        self.calls += 1
        return ()


@pytest.mark.parametrize("tier", TIERS)
def test_shipped_tier_config_parses_and_validates(tier: str) -> None:
    """The file on disk, not a dataclass built in the test.

    The previous generation of these files was written in the retired flat
    schema and every key in them was rejected. Nothing noticed, because nothing
    loaded them.
    """
    assert (CONFIG_DIR / f"tier_{tier}.yaml").is_file()
    config = load_tier(tier)
    assert isinstance(config, PointstreamConfig)
    assert config.evaluation.metrics


@pytest.mark.parametrize("tier", TIERS)
def test_tier_config_runs_end_to_end_and_produces_a_scored_result(tier: str) -> None:
    """The gate. A run that comes back without both quality views is a failed run."""
    config, result, _counters, clip = _run_tier(tier)

    assert result.frames.shape == clip.shape
    # Both views, because they answer different questions: the reconstruction
    # against the source, and the delivered payload against the source.
    for report in (result.quality, result.delivered_quality):
        frame = report.whole_frame()
        assert math.isfinite(frame) or math.isinf(frame)
        assert report.for_role(ROLE_FRAME)
    # Region scores, because a whole-frame number hides a broken object.
    assert result.quality.for_role(ROLE_OBJECT)
    assert result.quality.for_role(ROLE_BACKGROUND)
    # One ledger, and it has to add up.
    assert result.sizes.transport_total > 0
    assert result.sizes.parts_fit()
    assert result.sizes.residual > 0, "the residual stage is on; it must cost something"
    assert config.stages.is_enabled(STAGE_RESIDUAL)


@pytest.mark.parametrize("tier", TIERS)
def test_tier_reports_every_metric_its_config_asks_for(tier: str) -> None:
    """A flag existing is not a feature working.

    `evaluation.metrics` was a field nothing read: a config naming SSIM and
    VMAF produced PSNR and said nothing about it. The assertion is on the names
    that come back, not on the field parsing.
    """
    config, result, _counters, _clip = _run_tier(tier)
    reported = {item.metric for item in result.delivered_quality.scoped}
    assert set(config.evaluation.metrics) <= reported, (
        f"tier {tier} asked for {sorted(config.evaluation.metrics)} and reported "
        f"{sorted(reported)}"
    )


@pytest.mark.parametrize("tier", TIERS)
def test_a_stage_switched_off_in_a_tier_is_never_invoked(tier: str) -> None:
    """Configured off has to mean not run, or every ablation number is fiction."""
    config, _result, counters, _clip = _run_tier(tier)
    assert not config.stages.is_enabled(STAGE_GENERATION)
    assert counters, "this test is vacuous if the tier disables nothing"
    for stage, counter in counters.items():
        assert counter.calls == 0, f"{stage} ran while switched off in tier {tier}"


@pytest.mark.parametrize("tier", TIERS)
def test_encoder_and_client_agree_when_nothing_generative_is_in_the_path(
    tier: str,
) -> None:
    """Symmetry is measured, not asserted — but with generation off both sides
    run the same deterministic composite over the same residual, so a
    difference here means the two clips are being compared at different points
    in the pipeline rather than that generation is statistical."""
    _config, result, _counters, _clip = _run_tier(tier)
    assert result.symmetry.bit_identical, (
        f"tier {tier} encoder/client differ by mean {result.symmetry.mean_abs_diff} "
        "with no generator in the path"
    )


def test_the_all_off_corner_is_the_source_and_anchors_the_tier_runs() -> None:
    """The control that has to run beside the tiers, not after being asked.

    Without it, "a tier config produced numbers" says nothing about whether the
    path preserves pixels at all.
    """
    clip, _mask = _moving_block_clip()
    config = PointstreamConfig(lattice=lattice_config_from(SOURCE_PASSTHROUGH))
    result = run(config, [clip], bind_generator_fn=_never_constructs)
    assert np.array_equal(clip, result.frames)
    assert result.quality.bit_identical
    assert math.isinf(result.quality.whole_frame())
    assert result.sizes.transport_total == clip.nbytes
    assert result.sizes.residual == 0


def test_a_residual_absent_run_reports_its_quality_drop_instead_of_the_source() -> None:
    """Named in the Phase C gate, and it used to be silently false.

    With the residual switched off the codec stage fell through to "deliver the
    source", so the corner whose whole point is *unaided* quality reported an
    infinite PSNR and a bit-perfect copy of the video. Nothing failed; the
    number was simply the wrong number. The delivered clip has to be the
    reconstruction the client would build.
    """
    clip, mask = _moving_block_clip()
    base = load_tier("fast")
    config = base.with_(lattice=dataclasses.replace(base.lattice, residual=False))
    result = run(
        config,
        [clip],
        bind_generator_fn=_never_constructs,
        objects=(_objects(clip, mask),),
        components=_light_perception(),
    )
    assert result.sizes.residual == 0
    assert not result.delivered_quality.bit_identical, (
        "a residual-absent run delivered a bit-perfect copy of the source"
    )
    delivered = result.delivered_quality.whole_frame()
    assert math.isfinite(delivered)
    # Delivered and reconstruction are the same pixels on this corner: with no
    # residual there is nothing between them.
    assert delivered == pytest.approx(result.quality.whole_frame(), rel=1e-9)


def test_the_all_off_corner_still_delivers_the_source_after_that_fix() -> None:
    """The fix must not turn the baseline corner into an approximation.

    All-off is the one corner where delivering the source is correct, and it is
    correct because the corner *is* the source — not because the codec stage
    has a special case for it.
    """
    clip, _mask = _moving_block_clip()
    config = PointstreamConfig(lattice=lattice_config_from(SOURCE_PASSTHROUGH))
    result = run(config, [clip], bind_generator_fn=_never_constructs)
    assert result.delivered_quality.bit_identical


def test_the_tier_ladder_is_a_ladder_and_not_three_names_for_one_setting() -> None:
    """Coarseness must move with the tier, and delivered quality must move with it.

    Pre-registered before the first run (`outputs/bp23-tier/bounds-before-run.json`):
    fast <= balanced <= quality on delivered whole-frame PSNR. An inversion means
    a coarseness knob does not do what it claims.
    """
    rungs = []
    for tier in TIERS:
        if tier == "quality" and not _ffmpeg_has_libvmaf():
            continue
        config, result, _counters, _clip = _run_tier(tier)
        rungs.append((tier, config.residual, result.delivered_quality.whole_frame()))

    assert len(rungs) >= 2, "a ladder needs at least two rungs that can run here"

    coarseness = [(item[1].block_threshold, item[1].background_downscale) for item in rungs]
    assert len(set(coarseness)) == len(rungs), (
        f"the three tiers name the same residual settings: {coarseness}"
    )
    psnrs = [item[2] for item in rungs]
    assert psnrs == sorted(psnrs), (
        f"delivered PSNR is not monotonic across the tier ladder: "
        f"{[(item[0], item[2]) for item in rungs]}"
    )
