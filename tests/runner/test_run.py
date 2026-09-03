"""The runner is one loop: every path scores, skipped stages stay idle.

Behaviour the brief named. Plausible misuse is a missing quality record or
a sizes ledger that does not add up — those are the silent wrong answers.
"""

from pathlib import Path

import math

import numpy as np
import pytest

from src.contracts.codecs import RateControl
from src.contracts.config import GeneratorConfig, PointstreamConfig, ResidualConfig
from src.contracts.errors import ConfigValueError
from src.contracts.lattice import (
    ART_DELIVERED,
    ART_QUALITY,
    OPTIONAL_STAGES,
    SOURCE_PASSTHROUGH,
    STAGE_CODEC,
    STAGE_DETECTION,
    STAGE_GENERATION,
    STAGE_METRICS,
    STAGE_RESIDUAL,
    WHOLE_FRAME_RESIDUAL,
    StageLattice,
)
from src.pipeline.encoder.encoder import SOURCE
from src.pipeline.reconstruction import (
    GeneratorRef,
    ObjectRequest,
    bit_identical,
)
from src.pipeline.reconstruction.quality import ROLE_BACKGROUND, ROLE_FRAME, ROLE_OBJECT, score
from src.pipeline.residual import ResidualResult, ResidualVariant
from src.runner import lattice_config_from, run
from src.runner.stages import _delivered_frames
from tests.pipeline.clocks import ClockedStage


def _clip(value: int, *, frames: int = 2, size: int = 32) -> np.ndarray:
    return np.full((frames, size, size, 3), value, dtype=np.uint8)


def _all_off() -> PointstreamConfig:
    return PointstreamConfig(lattice=lattice_config_from(SOURCE_PASSTHROUGH))


def _residual_only() -> PointstreamConfig:
    return PointstreamConfig(
        lattice=lattice_config_from(WHOLE_FRAME_RESIDUAL),
        residual=ResidualConfig(
            codec="avc",
            rate_control=RateControl.LOSSLESS,
            rate=0,
            block_size=1,
            background_downscale=1,
        ),
    )


def _object(*, size: int = 16) -> ObjectRequest:
    return ObjectRequest(
        object_id="player",
        appearance=np.full((size, size, 3), 200, dtype=np.uint8),
        bbox=(0, 0, size, size),
        mask=np.ones((size, size), dtype=bool),
    )


def test_one_chunk_and_many_chunks_are_the_same_loop() -> None:
    """A lone chunk is not a branch that skips the runner. Count codec calls."""

    class _Count:
        """All-off delivers the source, so the stand-in is the whole stage.

        It does not call the real codec callable: that one is built from a
        `StageContext` now, because a corner with the residual switched off has
        to deliver the reconstruction rather than the source.
        """

        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, bag):  # noqa: ANN001
            self.calls += 1
            frames = bag[SOURCE]
            return {"frames": frames, "byte_count": int(np.asarray(frames).nbytes)}

    one_clock = _Count()
    many_clock = _Count()
    clip_a = _clip(40, frames=1)
    clip_b = _clip(80, frames=1)
    one = run(_all_off(), [clip_a], backends={STAGE_CODEC: one_clock})
    many = run(_all_off(), [clip_a, clip_b], backends={STAGE_CODEC: many_clock})
    assert one_clock.calls == 1
    assert many_clock.calls == 2
    assert len(one.chunks) == 1
    assert len(many.chunks) == 2
    assert many.frames.shape[0] == 2


def test_all_off_is_the_source_and_never_constructs_a_generator() -> None:
    """Bounds: identity, PSNR infinite, residual absent. A finite PSNR or a
    constructed generator means the corner grew a special-case path."""

    def _must_not_construct() -> GeneratorRef:
        raise AssertionError("all-off must not construct a generator")

    source = _clip(91)
    source[0, 3, 5] = (4, 50, 200)
    result = run(_all_off(), [source], bind_generator_fn=_must_not_construct)
    assert result.lattice.is_source_passthrough
    assert bit_identical(source, result.frames)
    assert result.quality.bit_identical
    assert math.isinf(result.quality.whole_frame())
    assert math.isinf(result.delivered_quality.whole_frame())
    assert result.chunks[0].bag.get(STAGE_RESIDUAL) is None
    residual = result.chunks[0].bag.get("residual-stream")
    assert residual is None


def test_disabled_stage_callables_are_not_invoked() -> None:
    clocks = {name: ClockedStage() for name in OPTIONAL_STAGES}
    run(_all_off(), [_clip(10, frames=1)], backends=clocks)
    for name, clock in clocks.items():
        assert clock.calls == 0, name
        assert clock.cost == 0, name


def test_residual_only_runs_without_a_generator_and_keeps_both_quality_views() -> None:
    """Bounds written first. 2×32×32×3 source at grey 80, reconstruction is
    zeros: reconstruct PSNR ≈ 10.07 dB (10 log10(255²/80²)). Alarm if
    reconstruct PSNR is infinite (passthrough leaked in) or if delivered
    PSNR is finite (lossless residual did not restore). Lossless residual
    of a zero reconstruction is 2*32*32*3*2 = 12288 bytes."""

    def _must_not_construct() -> GeneratorRef:
        raise AssertionError("residual-only must not construct a generator")

    source = _clip(80)
    result = run(_residual_only(), [source], bind_generator_fn=_must_not_construct)
    recon_psnr = result.quality.whole_frame()
    assert 8.0 <= recon_psnr <= 12.0
    assert math.isinf(result.delivered_quality.whole_frame())
    assert bit_identical(source, result.frames)
    payload = result.chunks[0].bag["residual-stream"]
    assert isinstance(payload, ResidualResult)
    assert payload.payload.variant is ResidualVariant.LOSSLESS
    assert payload.payload.byte_count == 2 * 32 * 32 * 3 * 2
    assert result.sizes.residual == 2 * 32 * 32 * 3 * 2


def test_residual_only_reports_region_scores_alongside_the_frame() -> None:
    source = _clip(40, frames=1, size=32)
    source[0, 0:16, 0:16] = 200
    mask = np.zeros((32, 32), dtype=bool)
    mask[0:16, 0:16] = True
    obj = ObjectRequest(
        object_id="player",
        appearance=np.full((16, 16, 3), 200, dtype=np.uint8),
        bbox=(0, 0, 16, 16),
        mask=mask,
    )
    result = run(_residual_only(), [source], objects=((obj,),))
    assert result.quality.for_role(ROLE_OBJECT)
    assert result.quality.for_role(ROLE_BACKGROUND)
    assert result.quality.for_role(ROLE_FRAME)
    assert result.quality.whole_frame() is not None


def test_sizes_parts_sum_to_transport_total() -> None:
    result = run(_residual_only(), [_clip(12)])
    sizes = result.sizes
    assert sizes.transport_total > 0
    assert sizes.parts_fit()
    assert sizes.parts_sum == sizes.residual
    assert sizes.as_dict()["residual"] == sizes.residual


def test_missing_delivered_quality_fails_the_run() -> None:
    def _mute(bag):  # noqa: ANN001
        return None

    with pytest.raises(ConfigValueError, match="ART_QUALITY"):
        run(_all_off(), [_clip(7, frames=1)], backends={STAGE_METRICS: _mute})


def test_missing_reconstruction_quality_fails_the_run() -> None:
    class _Mute:
        def evaluate(self, *args, **kwargs):  # noqa: ANN001
            return None

    def _ok_metrics(bag):  # noqa: ANN001
        return score(bag[SOURCE], _delivered_frames(bag[ART_DELIVERED]))

    with pytest.raises(ConfigValueError, match="QualityReport"):
        run(
            _all_off(),
            [_clip(7, frames=1)],
            backends={STAGE_METRICS: _ok_metrics},
            evaluator=_Mute(),
        )


def test_same_generator_ref_serves_encoder_dispatch_and_client_reconstruct() -> None:
    """Encoder-side generation wraps C1 dispatch. Do not construct a second
    backend. A counter makes the two sides differ so symmetry is measured,
    never asserted equal."""

    class _Paint:
        def __init__(self) -> None:
            self.calls = 0

        def generate(self, conditioning, *, seed, device, params):  # noqa: ANN001
            self.calls += 1
            return np.full((8, 8, 3), 100 + self.calls, dtype=np.uint8)

    backend = _Paint()
    ref = GeneratorRef(backend=backend, name="paint")
    config = PointstreamConfig(
        lattice=lattice_config_from(StageLattice.of(STAGE_DETECTION, STAGE_GENERATION)),
        generator=GeneratorConfig(backend="paint"),
    )
    source = _clip(0, frames=1, size=16)
    result = run(config, [source], generator=ref, objects=((_object(size=8),),))
    assert backend.calls == 2
    assert result.chunks[0].reconstruction.quality is not None
    assert result.chunks[0].bag[ART_QUALITY] is not None
    assert not result.symmetry.bit_identical


def test_empty_chunks_are_refused() -> None:
    with pytest.raises(ValueError, match="at least one"):
        run(_all_off(), [])


def test_objects_misaligned_with_chunks_are_refused() -> None:
    with pytest.raises(ValueError, match="track position"):
        run(_all_off(), [_clip(1, frames=1)], objects=((_object(),), (_object(),)))


# ---------------------------------------------------------------------------
# `delivered_frames` — the array the byte count belongs to
#
# `RunResult.frames` is the client's clip with the residual applied *as the
# residual stage produced it*. Since BP24 codes that residual, the clip the
# pipeline actually delivers is rebuilt from what the codec returned, and the
# two arrays diverge by exactly the residual's coding loss. Pairing a coded rate
# with `frames` is `plans/done/BP24-findings.md` §4 — two real numbers at two
# different operating points, and on a rate ladder the error would not look
# like one, because sweeping the residual's rung is what makes them differ.
# ---------------------------------------------------------------------------


def test_delivered_frames_is_what_transport_handed_over() -> None:
    result = run(_residual_only(), [_clip(80)])
    expected = _delivered_frames(result.chunks[0].bag[ART_DELIVERED])
    assert bit_identical(expected, result.delivered_frames)


def test_delivered_frames_concatenates_every_chunk() -> None:
    """A per-chunk property that silently dropped chunks would still 'work'."""
    chunks = [_clip(80, frames=2), _clip(120, frames=2)]
    result = run(_residual_only(), chunks)
    assert result.delivered_frames.shape[0] == 4
    assert result.delivered_frames.shape == result.frames.shape


def test_delivered_frames_follows_the_codec_stage_not_the_residual() -> None:
    """The divergence itself, forced.

    Without an encoder the two arrays are equal, so a test on the shipped path
    cannot fail for the reason it claims. A codec stage that returns *different*
    pixels — which is exactly what a real one does — separates them: `frames`
    keeps the residual as the residual stage produced it, `delivered_frames`
    follows what the codec handed on, and `delivered_quality` is scored on the
    second. Reading the first beside a coded byte count is findings §4.
    """
    source = _clip(80)
    darker: dict[str, object] = {}

    def fake_codec(bag: object) -> dict[str, object]:
        residual = bag["residual-stream"]  # type: ignore[index]
        assert isinstance(residual, ResidualResult)
        frames = np.clip(residual.reconstructed.astype(np.int16) - 10, 0, 255).astype(
            np.uint8
        )
        darker["frames"] = frames
        return {"frames": frames, "byte_count": 1234, "residual_is_coded": True}

    result = run(_residual_only(), [source], backends={STAGE_CODEC: fake_codec})

    assert bit_identical(darker["frames"], result.delivered_frames)  # type: ignore[arg-type]
    assert not bit_identical(result.frames, result.delivered_frames)
    # And the score follows the delivered array, not the other one.
    assert score(source, result.delivered_frames).whole_frame() == pytest.approx(
        result.delivered_quality.whole_frame()
    )
    assert not math.isinf(result.delivered_quality.whole_frame())


def test_a_second_run_resumes_finished_chunks(tmp_path: Path) -> None:
    class _Count:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, bag):  # noqa: ANN001
            self.calls += 1
            frames = bag[SOURCE]
            return {"frames": frames, "byte_count": int(np.asarray(frames).nbytes)}

    clock = _Count()
    clips = [_clip(40, frames=1), _clip(80, frames=1)]
    ckpt = tmp_path / "chunks"
    run(_all_off(), clips, backends={STAGE_CODEC: clock}, checkpoint_dir=ckpt, checkpoint_identity="count-v1")
    assert clock.calls == 2
    run(_all_off(), clips, backends={STAGE_CODEC: clock}, checkpoint_dir=ckpt, checkpoint_identity="count-v1")
    assert clock.calls == 2
    assert (ckpt / "chunk_00" / "done").is_file()
    assert (ckpt / "chunk_01" / "done").is_file()
