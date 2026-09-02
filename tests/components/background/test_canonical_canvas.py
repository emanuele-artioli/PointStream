"""Required behaviour for the offline canonical background canvas (BP44).

The failure this file exists for is the one BP31 §12 measured at span 24:
two compatible scenes produce different local panorama sizes, and the
predictive background sequence refuses them. Padding to equal size is not
enough on its own — the origin shift has to travel with the homographies,
or reconstruction places the background in the wrong pixels.

Deliberately not tested: causal canvas growth (parked), OpenCV's own
RANSAC, and the 4K byte counts (that is the diagnostic in
``experiments/tier/canonical_canvas.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.components.background.plate import (
    PAD_FILL,
    build_plate,
    even_up,
    prepare_canonical_context,
)
from src.components.background.sidecar import build_sidecar
from src.components.background.strategy import bind as bind_background
from src.components.background.stream import (
    BackgroundStreamReceiver,
    BackgroundStreamTransmitter,
    context_reset_indices,
    segmented_reset_indices,
)
from src.contracts import config as cfg
from src.contracts.domain import BACKGROUND_PANORAMA_FULL, BACKGROUND_PANORAMA_STREAM
from src.pipeline.reconstruction.background import warp_plate
from tests.components.test_plate_registration import _pan, _texture


def _static(n_frames: int = 4, height: int = 96, width: int = 128) -> np.ndarray:
    return np.repeat(_texture(height, width, seed=3)[None, ...], n_frames, axis=0)


def _court_pair(
    *,
    n_static: int = 4,
    n_pan: int = 6,
    height: int = 96,
    width: int = 128,
    step: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Static and panning windows onto one texture — one camera, two motions."""
    base = _texture(height, width + step * n_pan + 8, seed=7)
    static = np.stack([base[:, :width] for _ in range(n_static)])
    panning = np.stack([base[:, k * step : k * step + width] for k in range(n_pan)])
    return static, panning


def _warp_back(plate: np.ndarray, packed: tuple, frames: np.ndarray) -> np.ndarray:
    height, width = int(frames.shape[1]), int(frames.shape[2])
    return warp_plate(plate, packed, height=height, width=width, frame_count=int(frames.shape[0]))


def test_even_up_rounds_odd_edges_up_not_down() -> None:
    """Cropping would drop the last column of a panorama. The canvas pads."""
    assert even_up(2161) == 2162
    assert even_up(3841) == 3842
    assert even_up(128) == 128


def test_two_unequal_local_panoramas_produce_equal_encoded_dimensions() -> None:
    static, panning = _court_pair()
    local_static, _ = build_plate(static)
    local_pan, _ = build_plate(panning)
    assert local_static.shape != local_pan.shape

    canvas, alignments, _bounds = prepare_canonical_context(
        [static, panning], context_id="court"
    )
    assert canvas.width % 2 == 0 and canvas.height % 2 == 0
    first, _ = build_plate(static, canvas=canvas, alignment=alignments[0])
    second, _ = build_plate(panning, canvas=canvas, alignment=alignments[1])
    assert first.shape == second.shape == (canvas.height, canvas.width, 3)


def test_source_frame_reconstruction_stays_aligned_after_the_origin_shift() -> None:
    """The origin shift is not a crop. Warp-back must still land on the source."""
    static, panning = _court_pair(n_static=6, n_pan=6, step=3)
    independent, independent_maps = build_plate(panning)
    from_independent = _warp_back(independent, independent_maps, panning)

    canvas, alignments, _ = prepare_canonical_context([static, panning], context_id="court")
    canonical, canonical_maps = build_plate(
        panning, canvas=canvas, alignment=alignments[1]
    )
    from_canonical = _warp_back(canonical, canonical_maps, panning)

    error = float(np.mean(np.abs(from_canonical.astype(np.int16) - from_independent.astype(np.int16))))
    assert error < 2.0, (
        f"canonical warp-back drifted {error:.2f} MAE from the independent plate. "
        "The origin shift did not travel with the homographies."
    )
    last = int(panning.shape[0] - 1)
    from_plate = float(
        np.mean(np.abs(from_canonical[last].astype(np.int16) - panning[last].astype(np.int16)))
    )
    from_first = float(
        np.mean(np.abs(panning[0].astype(np.int16) - panning[last].astype(np.int16)))
    )
    assert from_plate < from_first / 2.0


def test_padding_uses_the_documented_fill_outside_the_valid_region() -> None:
    static, panning = _court_pair(step=6)
    canvas, alignments, _ = prepare_canonical_context([static, panning], context_id="court")
    placed, _ = build_plate(static, canvas=canvas, alignment=alignments[0])
    assert placed.shape[1] > static.shape[2] or placed.shape[0] > static.shape[1]
    pad_count = int(np.all(placed == PAD_FILL, axis=-1).sum())
    assert pad_count > 0, "canonical padding did not use PAD_FILL"


def test_unaligned_unrelated_images_still_share_a_size_without_registering() -> None:
    """Equal size is not a licence to predict across cameras.

    ``register=False`` keeps each scene in its own frame-0, so the union is
    pad-to-max. Reconstruction stays local; the stream only sees one size.
    """
    a = np.repeat(_texture(64, 80, seed=1)[None, ...], 3, axis=0)
    b = np.repeat(_texture(64, 96, seed=99)[None, ...], 3, axis=0)
    canvas, alignments, _ = prepare_canonical_context(
        [a, b], context_id="mixed", register=False
    )
    assert canvas.width == even_up(96)
    assert canvas.height == even_up(64)
    first, _ = build_plate(a, canvas=canvas, alignment=alignments[0], register=False)
    second, _ = build_plate(b, canvas=canvas, alignment=alignments[1], register=False)
    assert first.shape == second.shape == (canvas.height, canvas.width, 3)


class TestResetBoundaries:
    """Segmented and continuous controls must name the same PointStream splits."""

    def test_one_context_resets_only_at_the_start(self) -> None:
        assert context_reset_indices(("court", "court", "court")) == (0,)

    def test_a_context_change_is_a_reset(self) -> None:
        assert context_reset_indices(("court", "court", "replay", "replay")) == (0, 2)

    def test_the_segmented_control_resets_every_scene(self) -> None:
        ids = ("court", "court", "replay")
        segmented = segmented_reset_indices(len(ids))
        continuous = context_reset_indices(ids)
        assert segmented == (0, 1, 2)
        assert continuous == (0, 2)
        # The continuous control is the PointStream configuration; the
        # segmented control is a stricter reset, not a different split.
        assert set(continuous).issubset(set(segmented))

    def test_empty_is_no_reset(self) -> None:
        assert context_reset_indices(()) == ()


def test_independent_coding_still_uses_a_local_canvas() -> None:
    """Prior 8/16-frame behaviour: panorama-full does not grow to a union."""
    panning = _pan(n_frames=6, height=96, width=128, step=3)
    local, packed = build_plate(panning)
    model = bind_background(cfg.load({"background": {"method": BACKGROUND_PANORAMA_FULL, "codec": "png"}}))
    stitched, stitched_maps = model.stitch(panning)
    assert stitched.shape == local.shape
    assert len(stitched_maps) == len(packed)


def test_an_unknown_canvas_mode_is_refused() -> None:
    with pytest.raises(Exception, match="canonical"):
        cfg.load({"background": {"canvas": "causal-grow"}})


@pytest.mark.integration
class TestCanonicalStream:
    def test_sender_and_receiver_reconstruct_byte_identical_images(self) -> None:
        static, panning = _court_pair(n_pan=5, step=4)
        canvas, alignments, _ = prepare_canonical_context([static, panning], context_id="court")
        plates = [
            build_plate(static, canvas=canvas, alignment=alignments[0])[0],
            build_plate(panning, canvas=canvas, alignment=alignments[1])[0],
        ]
        transmitter = BackgroundStreamTransmitter(mode="last", codec="av1", crf=38)
        receiver = BackgroundStreamReceiver(codec="av1")
        for plate in plates:
            payload = transmitter.push(plate)
            client = receiver.receive(payload, height=canvas.height, width=canvas.width)
            encoder = transmitter.reconstructions[payload.index]
            assert np.array_equal(client, encoder)

    def test_static_and_panning_scenes_share_a_context_without_failure(self) -> None:
        static, panning = _court_pair()
        model = bind_background(
            cfg.load(
                {
                    "background": {
                        "method": BACKGROUND_PANORAMA_STREAM,
                        "canvas": "canonical",
                        "context_id": "court",
                    }
                }
            )
        )
        canvas = model.prepare_context([static, panning], context_id="court")
        assert canvas is not None
        first_plate, first_maps = model.stitch(static)
        first = model.transmit(first_plate, homographies=first_maps, context_id="court")
        second_plate, second_maps = model.stitch(panning)
        second = model.transmit(second_plate, homographies=second_maps, context_id="court")
        assert first.width == second.width == canvas.width
        assert first.height == second.height == canvas.height
        assert first.mode == "full"
        assert second.mode == "stream"

    def test_unrelated_contexts_force_a_new_independently_coded_background(self) -> None:
        plate = _static()[0]
        model = bind_background(
            cfg.load({"background": {"method": BACKGROUND_PANORAMA_STREAM, "context_id": "court"}})
        )
        first = model.transmit(plate, context_id="court")
        second = model.transmit(plate, context_id="court")
        third = model.transmit(plate, context_id="replay")
        assert first.mode == "full"
        assert second.mode == "stream"
        assert third.mode == "full"
        assert third.payload != second.payload

    def test_padding_is_included_in_coded_bytes(self) -> None:
        static, panning = _court_pair(step=6)
        sidecar = build_sidecar("png")
        local, _ = build_plate(static)
        canvas, alignments, _ = prepare_canonical_context([static, panning], context_id="court")
        placed, _ = build_plate(static, canvas=canvas, alignment=alignments[0])
        assert placed.shape[0] * placed.shape[1] > local.shape[0] * local.shape[1]
        local_bytes = len(sidecar.encode(local))
        padded_bytes = len(sidecar.encode(placed))
        assert padded_bytes > local_bytes, (
            f"padded canvas coded to {padded_bytes} B against {local_bytes} B for the "
            "local plate; padding did not reach the bitstream"
        )

    def test_predictive_coding_changes_bytes_against_independent_coding(self) -> None:
        static, panning = _court_pair(n_pan=5, step=3)
        canvas, alignments, _ = prepare_canonical_context([static, panning], context_id="court")
        plates = [
            build_plate(static, canvas=canvas, alignment=alignments[0])[0],
            build_plate(panning, canvas=canvas, alignment=alignments[1])[0],
        ]
        streamed = bind_background(
            cfg.load({"background": {"method": BACKGROUND_PANORAMA_STREAM}})
        )
        stream_total = sum(len(streamed.transmit(plate).payload) for plate in plates)
        independent = bind_background(
            cfg.load({"background": {"method": BACKGROUND_PANORAMA_FULL, "codec": "png"}})
        )
        independent_total = sum(len(independent.transmit(plate).payload) for plate in plates)
        assert stream_total != independent_total
        assert stream_total < independent_total
