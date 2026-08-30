"""The background stage must carry its stream across chunks.

`plans/BP30-findings.md` §§20-22 measured that coding each scene's plate against
the previous reconstruction costs about half of coding every plate fresh. That
saving exists in the runner only if **one** background model survives the whole
chunk loop. The stage used to bind a fresh model inside its per-chunk body,
which would have given every chunk an empty stream and a full keyframe — the
amortisation configured, reported in the ledger, and entirely absent.

That failure is invisible from the outside: every payload is a valid plate and
every reconstruction is a real image. Only the byte counts differ. So it is
tested by byte counts.

**Deliberately not tested here:** the size of the saving on real 4K plates,
which is `experiments/tier/background_stream.py`'s job against pre-written
bounds; and the codec's own correctness.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.contracts import config as cfg
from src.contracts.domain import BACKGROUND_PANORAMA_FULL, BACKGROUND_PANORAMA_STREAM

HEIGHT, WIDTH = 96, 160


def _panning_plates(count: int = 4, step: int = 3) -> list[np.ndarray]:
    """Structured content that pans — the case inter prediction is for."""
    rng = np.random.default_rng(5)
    canvas = np.zeros((HEIGHT, WIDTH + step * count, 3), dtype=np.uint8)
    canvas[:] = 30
    for _ in range(16):
        top = int(rng.integers(0, HEIGHT - 12))
        left = int(rng.integers(0, canvas.shape[1] - 14))
        canvas[top : top + 12, left : left + 14] = rng.integers(80, 255, 3, dtype=np.uint8)
    return [np.ascontiguousarray(canvas[:, k * step : k * step + WIDTH]) for k in range(count)]


def _model(method: str, **overrides: object):
    from src.components.background.strategy import bind as bind_background

    settings = {"method": method}
    settings.update(overrides)
    return bind_background(cfg.load({"background": settings}))


@pytest.mark.integration
class TestStreamAmortisesAcrossChunks:
    def test_later_scenes_cost_far_less_than_the_first(self) -> None:
        model = _model(BACKGROUND_PANORAMA_STREAM)
        costs = [len(model.transmit(plate).payload) for plate in _panning_plates()]
        assert costs[0] > 0
        for later in costs[1:]:
            assert later < costs[0] / 2, (
                f"scene cost {later} against a first scene of {costs[0]}; the stream "
                "is not carrying its reconstruction across chunks"
            )

    def test_one_model_reused_beats_a_fresh_model_per_chunk(self) -> None:
        """The regression the hoist exists to prevent, measured directly.

        Rebinding per chunk is what the stage used to do. It produces a valid
        run whose every number is defensible except the one that matters.
        """
        plates = _panning_plates()
        carried = _model(BACKGROUND_PANORAMA_STREAM)
        carried_total = sum(len(carried.transmit(p).payload) for p in plates)
        rebound_total = sum(
            len(_model(BACKGROUND_PANORAMA_STREAM).transmit(p).payload) for p in plates
        )
        assert carried_total < rebound_total, (
            f"carrying the stream cost {carried_total} against {rebound_total} for a "
            "fresh model per chunk — the state is not being carried"
        )

    def test_the_stage_binds_one_model_for_the_whole_run(self) -> None:
        """Guards the hoist itself, not just its effect.

        `make_background` builds the model when the stage is created. If it ever
        moves back inside the per-chunk body this fails, rather than quietly
        costing the project its largest rate lever.
        """
        import inspect

        from src.runner import stages

        body = inspect.getsource(stages.make_background)
        stage_body = body.split("def background_stage", 1)[1]
        assert "bind_background" not in stage_body, (
            "make_background rebinds the model inside the per-chunk body; a "
            "stateful background stream cannot survive that"
        )

    def test_a_keyframe_is_reported_as_a_whole_plate_not_an_amortised_one(self) -> None:
        """The ledger has to stay readable about which scenes were not amortised."""
        model = _model(BACKGROUND_PANORAMA_STREAM, keyframe_interval=2, reference_mode="periodic-i")
        modes = [model.transmit(plate).mode for plate in _panning_plates()]
        assert modes[0] == "full"
        assert modes[1] == "stream"
        assert modes[2] == "full"


@pytest.mark.integration
class TestStreamIsDistinctFromDelta:
    """`panorama-stream` and `panorama-delta` are opposite mechanisms.

    Findings §17 measured pixel subtraction costing 1.49-1.70x *more* than a
    fresh plate across scenes; §§18-19 measured inter prediction saving 31-53%.
    Both remain registered so the ablation can compare them — the names are what
    keep them apart.
    """

    def test_both_methods_are_registered_under_different_names(self) -> None:
        from src.components.background import REGISTRY

        assert REGISTRY.spec(BACKGROUND_PANORAMA_STREAM).target.endswith("PanoramaStream")
        assert REGISTRY.spec("panorama-delta").target.endswith("PanoramaDelta")

    def test_the_stream_beats_a_fresh_plate_per_scene_on_panning_content(self) -> None:
        plates = _panning_plates()
        streamed = _model(BACKGROUND_PANORAMA_STREAM)
        streamed_total = sum(len(streamed.transmit(p).payload) for p in plates)
        fresh = _model(BACKGROUND_PANORAMA_FULL, codec="png")
        fresh_total = sum(len(fresh.transmit(p).payload) for p in plates)
        assert streamed_total < fresh_total
