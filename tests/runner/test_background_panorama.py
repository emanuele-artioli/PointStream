"""Required behaviour: `background.method` reaches `build_plate` (BP29).

`build_plate` existed from the rewrite and the runner never called it, so
`background.method` selected a transmission strategy over the first source
frame. A flag existing is not a feature working, so these tests drive the
config and check the output changed in the way the option claims:

* a panning clip produces a plate bigger than a frame, with real homographies;
* the plate the client decodes reproduces later frames the first frame cannot;
* `span=1` still produces exactly the pre-BP29 plate, which is what makes it a
  usable control for the measurement rather than a second code path;
* `none` still stitches nothing, because `none` sends nothing.

Deliberately not tested: the codec's own fidelity (stream B owns the sidecar),
whether the trade is worth it (a measurement, not an assertion), and the delta
strategy across chunks (the runner is still single-chunk for the background).
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.contracts.config import (
    BackgroundConfig,
    LatticeConfig,
    PointstreamConfig,
    ResidualConfig,
)
from src.contracts.codecs import RateControl
from src.contracts.errors import ConfigValueError
from src.contracts.lattice import ART_BACKGROUND_MODEL, STAGE_BACKGROUND
from src.pipeline.encoder.encoder import SOURCE
from src.pipeline.reconstruction.background import BackgroundModelView, BackgroundResolver
from src.pipeline.reconstruction.dispatch import GeneratorRef
from src.pipeline.reconstruction.quality import QualityEvaluator
from src.runner import run
from src.runner.routing import bind_evaluator, generation_params
from src.runner.stages import OBJECTS, StageContext, make_background
from tests.components.test_plate_registration import (
    _plate_errors,
    to_background_is_clean,
)


def _texture(height: int, width: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, size=(height // 4, width // 4, 3), dtype=np.uint8)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)


def _pan(n_frames: int = 6, height: int = 96, width: int = 128, step: int = 3) -> np.ndarray:
    base = _texture(height, width + step * n_frames + 8)
    return np.stack([base[:, k * step : k * step + width] for k in range(n_frames)])


def _config(*, method: str = "panorama-full", codec: str = "png") -> PointstreamConfig:
    lattice = LatticeConfig(
        scene_classification=False,
        detection=False,
        selection=False,
        tracking=False,
        appearance=False,
        motion=False,
        temporal_policy=False,
        pose=False,
        segmentation=False,
        rigid_objects=False,
        background=True,
        generation=False,
        residual=False,
    )
    return PointstreamConfig(
        lattice=lattice,
        background=BackgroundConfig(method=method, codec=codec),
        residual=ResidualConfig(rate_control=RateControl.CRF, background_downscale=4),
    )


def _context(config: PointstreamConfig) -> StageContext:
    evaluator: QualityEvaluator = bind_evaluator(None, config)
    generator: GeneratorRef | None = None
    return StageContext(
        lattice=config.stages,
        residual=config.residual,
        generator=generator,
        evaluator=evaluator,
        resolver=BackgroundResolver(),
        seed=config.run.seed,
        params=generation_params(config),
        config=config,
    )


def _stage_view(config: PointstreamConfig, clip: np.ndarray, **kwargs: object) -> BackgroundModelView:
    stage = make_background(_context(config), **kwargs)  # type: ignore[arg-type]
    view = stage({SOURCE: clip, OBJECTS: ()})
    assert isinstance(view, BackgroundModelView)
    return view


def test_a_panning_clip_produces_a_plate_bigger_than_a_frame() -> None:
    clip = _pan()
    view = _stage_view(_config(), clip)
    plate = np.asarray(view.plate)
    assert plate.shape[1] > clip.shape[2], (
        "the background stage returned a frame-sized plate on a panning clip, "
        "which is what it did before build_plate was wired in"
    )
    assert len(view.homographies) == clip.shape[0]
    assert not all(
        np.allclose(np.asarray(h, dtype=np.float64).reshape(3, 3), np.eye(3), atol=1e-7)
        for h in view.homographies
    )
    assert view.payload_bytes is not None and view.payload_bytes > 0


def test_the_transmitted_plate_reproduces_a_frame_the_keyframe_cannot() -> None:
    """Decoded pixels, decoded homographies — what the client actually holds."""
    clip = _pan()
    view = _stage_view(_config(), clip)
    last = clip.shape[0] - 1
    matrix = np.asarray(view.homographies[last], dtype=np.float64).reshape(3, 3)
    warped = cv2.warpPerspective(
        np.asarray(view.plate),
        np.linalg.inv(matrix),
        (clip.shape[2], clip.shape[1]),
        flags=cv2.INTER_LINEAR,
    )
    from_plate = float(np.mean(np.abs(warped.astype(np.int32) - clip[last].astype(np.int32))))
    from_first = float(np.mean(np.abs(clip[0].astype(np.int32) - clip[last].astype(np.int32))))
    assert from_plate < from_first / 2.0


def test_span_one_still_transmits_the_first_source_frame() -> None:
    """The control arm. A lossless sidecar means this is an exact claim."""
    clip = _pan()
    view = _stage_view(_config(codec="png"), clip, span=1)
    assert np.array_equal(np.asarray(view.plate), clip[0])
    assert len(view.homographies) == 1


def test_the_span_changes_what_is_transmitted() -> None:
    """Same config, same clip, different span: the payload must differ.

    If these matched, the wiring would be inert and every number taken from it
    would be measuring one arm twice.
    """
    clip = _pan()
    config = _config()
    whole = _stage_view(config, clip)
    one = _stage_view(config, clip, span=1)
    assert whole.payload_bytes != one.payload_bytes
    assert np.asarray(whole.plate).shape != np.asarray(one.plate).shape


def test_background_none_stitches_nothing() -> None:
    clip = _pan()
    view = _stage_view(_config(method="none"), clip)
    assert view.deferred_to_residual is True
    assert view.homographies == ()
    assert view.payload_bytes == 0


def test_a_span_below_one_frame_is_refused() -> None:
    with pytest.raises(ConfigValueError, match="span"):
        make_background(_context(_config()), span=0)


def test_a_still_object_is_kept_out_of_the_plate_through_the_runner() -> None:
    """Masks the runner already holds must reach `build_plate`.

    An object that sits still for half the chunk otherwise burns into the
    median and is transmitted as background.
    """
    from src.pipeline.reconstruction.reconstruct import ObjectRequest

    height, width, n_frames = 96, 128, 6
    background = _texture(height, width)
    clip = np.repeat(background[None, ...], n_frames, axis=0).copy()
    mask = np.zeros((n_frames, height, width), dtype=bool)
    clip[:3, 20:40, 30:50] = 0
    mask[:3, 20:40, 30:50] = True
    subject = ObjectRequest(
        object_id="player",
        appearance=clip[0, 20:40, 30:50].copy(),
        bbox=(30, 20, 50, 40),
        mask=mask,
        frame_index=0,
    )
    stage = make_background(_context(_config(codec="png")))
    view = stage({SOURCE: clip, OBJECTS: (subject,)})
    assert isinstance(view, BackgroundModelView)
    inside, outside, to_object = _plate_errors(
        np.asarray(view.plate), background, (20, 40, 30, 50)
    )
    assert to_background_is_clean(inside, outside, to_object), (
        f"masked region {inside:.2f}, rest of the plate {outside:.2f}, object "
        f"{to_object:.2f}: the runner's masks did not reach build_plate"
    )


def test_the_run_path_transmits_a_stitched_plate() -> None:
    """End to end: `bind_backends` uses the same stage, and the ledger sees it."""
    clip = _pan()
    config = _config()
    result = run(config, [clip])
    view = result.chunks[0].bag.get(ART_BACKGROUND_MODEL) or result.chunks[0].bag.get(
        STAGE_BACKGROUND
    )
    assert isinstance(view, BackgroundModelView)
    assert np.asarray(view.plate).shape[1] > clip.shape[2]
    assert result.sizes.panorama == view.payload_bytes
    assert result.sizes.panorama > 0
