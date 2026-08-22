"""All-off is the source; quality is always scored; regions are not interchangeable.

The all-off corner reducing to the source video is a Phase-C gate. It is
proven here from the first reconstruction commit, not discovered at the end.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.contracts.errors import ConfigValueError
from src.contracts.lattice import (
    STAGE_DETECTION,
    STAGE_GENERATION,
    SOURCE_PASSTHROUGH,
    StageLattice,
)
from src.pipeline.reconstruction import (
    BackgroundModelView,
    GeneratorRef,
    ObjectRequest,
    ReconstructionRequest,
    bit_identical,
    closeness,
    measure_symmetry,
    reconstruct,
    score,
)
from src.pipeline.reconstruction.quality import MIN_REGION_PIXELS, ROLE_BACKGROUND, ROLE_OBJECT


def _clip(value: int, *, frames: int = 2, size: int = 32) -> np.ndarray:
    return np.full((frames, size, size, 3), value, dtype=np.uint8)


def test_all_off_reconstruction_is_the_source_bit_for_bit() -> None:
    """Turn everything optional off, including the residual: what remains is
    the source. Bounds: identity, PSNR infinite. Anything finite is an alarm."""
    source = _clip(91)
    source[0, 3, 5] = (4, 50, 200)
    result = reconstruct(
        ReconstructionRequest(lattice=StageLattice.all_off(), source=source)
    )
    assert result.path == "source-passthrough"
    assert SOURCE_PASSTHROUGH.is_source_passthrough
    assert bit_identical(source, result.frames)
    assert result.quality.bit_identical
    assert math.isinf(result.quality.whole_frame())


def test_source_passthrough_does_not_consult_a_generator_or_background() -> None:
    """A leftover generator must not run; the corner has no generation stage."""

    class _MustNotRun:
        def generate(self, conditioning, *, seed, device, params):  # noqa: ANN001
            raise AssertionError("passthrough must not dispatch generation")

    source = _clip(10)
    plate = np.full((32, 32, 3), 255, dtype=np.uint8)
    result = reconstruct(
        ReconstructionRequest(
            lattice=SOURCE_PASSTHROUGH,
            source=source,
            background=BackgroundModelView(plate=plate),
            generator=GeneratorRef(backend=_MustNotRun()),
            objects=(
                ObjectRequest(
                    object_id="player",
                    appearance=np.full((8, 8, 3), 1, dtype=np.uint8),
                    bbox=(0, 0, 8, 8),
                ),
            ),
        )
    )
    assert bit_identical(source, result.frames)


def test_background_off_does_not_use_the_plate() -> None:
    """when_off: background lands in the residual. The plate must not appear."""
    source = _clip(40, size=32)
    plate = np.full((32, 32, 3), 200, dtype=np.uint8)
    lattice = StageLattice.of(STAGE_DETECTION)
    result = reconstruct(
        ReconstructionRequest(
            lattice=lattice,
            source=source,
            background=BackgroundModelView(plate=plate, deferred_to_residual=False),
        )
    )
    assert np.all(result.frames == 0)


def test_quality_is_present_on_every_path() -> None:
    source = _clip(12)
    result = reconstruct(ReconstructionRequest(lattice=SOURCE_PASSTHROUGH, source=source))
    assert result.quality.scoped
    assert result.quality.whole_frame("psnr") is not None


def test_a_destroyed_object_cannot_hide_in_frame_psnr() -> None:
    """Perfect background, mangled player: frame PSNR looks tolerable, object
    PSNR does not. Bounds written before the numbers: 64×64 frame, 16×16
    object of 255 vs background 40 → object PSNR ≈ 1.5 dB
    (10 log10(255²/215²)); frame PSNR ≈ 13.5 dB because the player is 1/16
    of the pixels. Alarm if object PSNR > 5 (the destruction did not register)
    or frame PSNR > 25 (the object error was averaged away)."""
    size = 64
    source = np.full((1, size, size, 3), 40, dtype=np.uint8)
    source[0, 8:24, 8:24] = 255
    predicted = np.full((1, size, size, 3), 40, dtype=np.uint8)
    mask = np.zeros((size, size), dtype=bool)
    mask[8:24, 8:24] = True
    report = score(source, predicted, object_mask=mask)
    object_psnr = report.for_role(ROLE_OBJECT)[0].value
    background_psnr = report.for_role(ROLE_BACKGROUND)[0].value
    frame_psnr = report.whole_frame()
    assert 1.0 <= object_psnr <= 2.0
    assert math.isinf(background_psnr)
    assert 10.0 <= frame_psnr <= 18.0
    assert frame_psnr > object_psnr


def test_bit_identity_fails_for_one_grey_level() -> None:
    src = _clip(64, frames=1, size=8)
    dirty = src.copy()
    dirty[0, 0, 0, 0] = 65
    assert bit_identical(src, src.copy())
    assert not bit_identical(src, dirty)


def test_generative_closeness_does_not_require_bit_identity() -> None:
    """A noisy sampler is close, not identical. Asserting identity would fail
    for reasons that have nothing to do with correctness."""
    encoder = np.full((2, 8, 8, 3), 128, dtype=np.uint8)
    client = encoder.copy()
    client[0, 0, 0, :] = 129
    relate = measure_symmetry(encoder, client, atol=1.0)
    assert not relate.bit_identical
    assert relate.within_atol
    assert closeness(encoder, client, atol=1.0).mean_abs_diff > 0.0


def test_mismatched_shapes_are_refused_rather_than_broadcast() -> None:
    with pytest.raises(ValueError, match="shape"):
        score(_clip(8, size=8), _clip(8, size=4))


def test_a_tiny_object_region_is_refused_rather_than_scored() -> None:
    source = _clip(10, frames=1, size=32)
    mask = np.zeros((32, 32), dtype=bool)
    mask[0, : MIN_REGION_PIXELS - 1] = True
    with pytest.raises(ValueError, match="small-sample"):
        score(source, source, object_mask=mask)


def test_generation_enabled_without_an_injected_generator_is_rejected() -> None:
    with pytest.raises(ConfigValueError, match="injected"):
        reconstruct(
            ReconstructionRequest(
                lattice=StageLattice.of(STAGE_DETECTION, STAGE_GENERATION),
                source=_clip(7),
                objects=(
                    ObjectRequest(
                        object_id="player",
                        appearance=np.full((8, 8, 3), 1, dtype=np.uint8),
                        bbox=(0, 0, 8, 8),
                    ),
                ),
            )
        )


def test_injected_generator_is_composited_and_skipped_when_generation_is_off() -> None:
    """Generation off: subjects land in the residual, so the crop must not
    appear. Generation on: the injected backend's pixels must appear — a flag
    existing is not a feature working."""

    class _Paint:
        def generate(self, conditioning, *, seed, device, params):  # noqa: ANN001
            return np.full((8, 8, 3), 255, dtype=np.uint8)

    source = _clip(0, frames=1, size=16)
    objects = (
        ObjectRequest(
            object_id="player",
            appearance=np.full((8, 8, 3), 9, dtype=np.uint8),
            bbox=(0, 0, 8, 8),
        ),
    )
    off = reconstruct(
        ReconstructionRequest(
            lattice=StageLattice.of(STAGE_DETECTION),
            source=source,
            generator=GeneratorRef(backend=_Paint()),
            objects=objects,
        )
    )
    on = reconstruct(
        ReconstructionRequest(
            lattice=StageLattice.of(STAGE_DETECTION, STAGE_GENERATION),
            source=source,
            generator=GeneratorRef(backend=_Paint()),
            objects=objects,
        )
    )
    assert np.all(off.frames == 0)
    assert np.all(on.frames[0, 0:8, 0:8] == 255)


def test_empty_source_is_refused() -> None:
    with pytest.raises(ValueError, match="empty"):
        reconstruct(
            ReconstructionRequest(
                lattice=SOURCE_PASSTHROUGH,
                source=np.zeros((0, 8, 8, 3), dtype=np.uint8),
            )
        )
