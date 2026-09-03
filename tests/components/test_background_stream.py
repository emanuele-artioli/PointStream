"""Required behaviour for the cross-scene background stream.

The property this file exists for is the one `https://github.com/emanuele-artioli/PointStream/blob/ec581e9/plans/prompts/claude-bp30.md`
names as the single place the work can go wrong quietly: **encoder-side and
client-side reconstructions must be bit-identical across a multi-scene
sequence.** A drifted reconstruction is still an image, so nothing else in the
system would notice.

The ffmpeg-driven tests carry `integration`, matching this suite's existing
convention for anything that needs an encoder binary. They are not optional
because of it -- `test_reconstructions_are_bit_identical_across_scenes` is the
required-behaviour test for this component and has to be run and reported, not
merely collected.

**Deliberately not tested here**, because the tests that pay for themselves
check the paper's claim rather than that code runs:

- libaom's, x265's and x264's own correctness. Testing a third party's encoder
  is not this project's job; what is tested is that *this* module's causality
  assumption about them holds, and `_assert_prefix_stable` re-checks that on
  every real encode anyway.
- The hevc and avc arms. They exist as a contrast (findings §19: x265 saves 12%
  on one pair and loses 6% on the other), not as equals, and testing all three
  triples the wall clock for no extra property.
- Real 4K plate content and the size of any saving. That is the measurement in
  `experiments/tier/background_stream.py`, which prices it against pre-written
  bounds; a unit test asserting a compression ratio would be asserting the
  result rather than the behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.components.background.stream import (
    CODECS,
    KEYFRAME_NEVER,
    REFERENCE_BEST_SCORED,
    REFERENCE_FIRST,
    REFERENCE_LAST,
    REFERENCE_MODES,
    REFERENCE_PERIODIC_I,
    BackgroundStreamReceiver,
    BackgroundStreamTransmitter,
    StreamDrift,
    stream_linear,
    canny_edges,
    canny_iou,
)

HEIGHT, WIDTH = 96, 128


def test_transmitter_state_roundtrip_does_not_need_ffmpeg() -> None:
    original = np.zeros((4, 4, 3), dtype=np.uint8)
    recon = np.full((4, 4, 3), 7, dtype=np.uint8)
    tx = BackgroundStreamTransmitter(mode=REFERENCE_LAST, codec="av1", crf=51)
    tx._originals = [original]
    tx._reconstructions = [recon]
    tx._payloads = [b"obu"]
    tx._chains = [(0,)]
    restored = BackgroundStreamTransmitter(mode=REFERENCE_LAST, codec="av1", crf=51)
    restored.import_state(tx.export_state())
    assert restored._payloads == [b"obu"]
    assert restored._chains == [(0,)]
    assert np.array_equal(restored._reconstructions[0], recon)


def test_transmitter_refuses_a_codec_mismatch_on_restore() -> None:
    tx = BackgroundStreamTransmitter(mode=REFERENCE_LAST, codec="av1", crf=51)
    tx._payloads = [b"x"]
    tx._chains = [(0,)]
    tx._originals = [np.zeros((2, 2, 3), dtype=np.uint8)]
    tx._reconstructions = [np.zeros((2, 2, 3), dtype=np.uint8)]
    other = BackgroundStreamTransmitter(mode=REFERENCE_LAST, codec="hevc", crf=51)
    with pytest.raises(ValueError, match="codec"):
        other.import_state(tx.export_state())


def _panning_scenes(count: int = 4, step: int = 4) -> list[np.ndarray]:
    """A background that pans, which is the case inter prediction is for.

    Built from structured content rather than noise: Canny on white noise finds
    edges everywhere and would make the structural score meaningless.
    """
    rng = np.random.default_rng(11)
    canvas = np.zeros((HEIGHT, WIDTH + step * count, 3), dtype=np.uint8)
    canvas[:, :, :] = 40
    for _ in range(18):
        top = int(rng.integers(0, HEIGHT - 12))
        left = int(rng.integers(0, canvas.shape[1] - 18))
        colour = rng.integers(60, 255, 3, dtype=np.uint8)
        canvas[top : top + 12, left : left + 18] = colour
    return [np.ascontiguousarray(canvas[:, k * step : k * step + WIDTH]) for k in range(count)]


class TestCannyProxy:
    """The structural score, calibrated against anchors before it ranks anything."""

    def test_identical_images_score_one(self) -> None:
        plate = _panning_scenes(1)[0]
        assert canny_iou(plate, plate) == pytest.approx(1.0)

    def test_a_nearer_pan_scores_above_a_further_one(self) -> None:
        scenes = _panning_scenes(4, step=8)
        near = canny_iou(scenes[0], scenes[1])
        far = canny_iou(scenes[0], scenes[3])
        assert near > far

    def test_two_edgeless_images_score_one_rather_than_zero(self) -> None:
        flat = np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8)
        assert canny_edges(flat).sum() == 0
        assert canny_iou(flat, flat) == pytest.approx(1.0)


class TestConstruction:
    """Configuration is refused at construction, not discovered mid-stream."""

    def test_unknown_reference_mode_is_refused(self) -> None:
        with pytest.raises(ValueError, match="reference mode"):
            BackgroundStreamTransmitter(mode="nearest")

    def test_unknown_codec_is_refused(self) -> None:
        with pytest.raises(ValueError, match="stream codec"):
            BackgroundStreamTransmitter(codec="vp9")

    def test_negative_keyframe_interval_is_refused(self) -> None:
        with pytest.raises(ValueError, match="keyframe_interval"):
            BackgroundStreamTransmitter(keyframe_interval=-1)

    def test_every_declared_mode_constructs(self) -> None:
        for mode in REFERENCE_MODES:
            assert BackgroundStreamTransmitter(mode=mode).mode == mode

    def test_every_declared_codec_names_a_raw_container(self) -> None:
        # A payload has to be concatenable with the ones before it, so an
        # elementary stream is the format. mkv would hide the framing in the
        # container and the client could not reassemble a chain.
        for name, spec in CODECS.items():
            assert spec.container in {"obu", "hevc", "h264"}, name


@pytest.mark.integration
class TestCausalityAndIdentity:
    """The two properties the scheme stands on."""

    def test_reconstructions_are_bit_identical_across_scenes(self) -> None:
        """The required behaviour: no drift between encoder and client.

        The client is given only the payloads. If it ends up holding different
        pixels from the encoder, every later scene predicts from the wrong
        picture and nothing downstream can tell.
        """
        scenes = _panning_scenes(4)
        for mode in REFERENCE_MODES:
            transmitter = BackgroundStreamTransmitter(
                mode=mode, codec="av1", crf=38, keyframe_interval=2
            )
            receiver = BackgroundStreamReceiver(codec="av1")
            for plate in scenes:
                payload = transmitter.push(plate)
                client = receiver.receive(payload, height=HEIGHT, width=WIDTH)
                encoder = transmitter.reconstructions[payload.index]
                assert np.array_equal(client, encoder), (
                    f"{mode}: scene {payload.index} drifted between encoder and client"
                )

    def test_a_payload_is_never_revised_by_a_later_scene(self) -> None:
        """Prefix stability, which is what makes each payload causal.

        If appending scene n+1 changed scene n's bytes, the encoder would have
        needed the future to emit scene n -- and this would be an offline
        archiver rather than a codec.
        """
        scenes = _panning_scenes(4)
        transmitter = BackgroundStreamTransmitter(mode=REFERENCE_LAST, codec="av1", crf=38)
        emitted: list[bytes] = []
        for plate in scenes:
            emitted.append(transmitter.push(plate).payload)
        # `push` re-encodes the whole chain each time and `_assert_prefix_stable`
        # compares it against what was already sent, so reaching here at all
        # means every prefix survived. Assert the bytes too, so a future change
        # that removes that internal check still fails this test.
        replay = BackgroundStreamTransmitter(mode=REFERENCE_LAST, codec="av1", crf=38)
        for offset, plate in enumerate(scenes):
            assert replay.push(plate).payload == emitted[offset]

    def test_no_arm_emits_a_b_frame(self) -> None:
        """A B-frame references a future picture, so it breaks causality."""
        scenes = _panning_scenes(3)
        for mode in REFERENCE_MODES:
            transmitter = BackgroundStreamTransmitter(mode=mode, codec="av1", crf=38)
            for plate in scenes:
                assert transmitter.push(plate).picture_type in {"I", "P"}


@pytest.mark.integration
class TestReferenceModes:
    """Each mode must actually choose a different reference, not just be named.

    `BackgroundConfig.codec` accepted three values and reached nothing until
    BP24 wired the background stage; a mode that is accepted and ignored is the
    same defect.
    """

    def test_first_always_predicts_from_the_opening_scene(self) -> None:
        transmitter = BackgroundStreamTransmitter(mode=REFERENCE_FIRST, codec="av1", crf=38)
        references = [transmitter.push(p).reference for p in _panning_scenes(4)]
        assert references == [None, 0, 0, 0]

    def test_last_always_predicts_from_the_previous_scene(self) -> None:
        transmitter = BackgroundStreamTransmitter(mode=REFERENCE_LAST, codec="av1", crf=38)
        references = [transmitter.push(p).reference for p in _panning_scenes(4)]
        assert references == [None, 0, 1, 2]

    def test_best_scored_picks_the_structurally_closest_reconstruction(self) -> None:
        """On a monotonic pan the nearest previous scene is the closest one.

        This is the proxy's easy case and is here to prove the search is wired
        to the score at all. Whether Canny tracks *coding* distance on real
        plates is a measurement, not a unit test -- brief §3 requires it to be
        checked against trial encodes, which
        `experiments/tier/canny_validate.py` does.
        """
        transmitter = BackgroundStreamTransmitter(mode=REFERENCE_BEST_SCORED, codec="av1", crf=38)
        references = [transmitter.push(p).reference for p in _panning_scenes(4, step=10)]
        assert references[0] is None
        assert references[1:] == [0, 1, 2]

    def test_periodic_i_forces_a_keyframe_every_k_scenes(self) -> None:
        transmitter = BackgroundStreamTransmitter(
            mode=REFERENCE_PERIODIC_I, codec="av1", crf=38, keyframe_interval=2
        )
        payloads = [transmitter.push(p) for p in _panning_scenes(5)]
        assert [p.is_keyframe for p in payloads] == [True, False, True, False, True]
        assert [p.picture_type for p in payloads] == ["I", "P", "I", "P", "I"]

    def test_periodic_i_with_never_sends_exactly_one_keyframe(self) -> None:
        """The `never` column of the sweep: a pure P-chain."""
        transmitter = BackgroundStreamTransmitter(
            mode=REFERENCE_PERIODIC_I, codec="av1", crf=38, keyframe_interval=KEYFRAME_NEVER
        )
        payloads = [transmitter.push(p) for p in _panning_scenes(4)]
        assert [p.is_keyframe for p in payloads] == [True, False, False, False]

    def test_a_predicted_scene_costs_less_than_a_fresh_one(self) -> None:
        """The control, in miniature: inter prediction must actually be running.

        Findings §19 records why this is not decoration -- a misconfigured x265
        made a P-frame come back *larger* than a fresh intra, and without the
        control that would have been reported as a finding about low delay.
        """
        transmitter = BackgroundStreamTransmitter(mode=REFERENCE_LAST, codec="av1", crf=38)
        payloads = [transmitter.push(p) for p in _panning_scenes(3, step=2)]
        assert payloads[0].is_keyframe
        for later in payloads[1:]:
            assert later.byte_count < payloads[0].byte_count


@pytest.mark.integration
class TestRefusals:
    """Plausible misuse, refused loudly rather than absorbed."""

    def test_a_changed_plate_size_is_refused(self) -> None:
        """Brief §2: inter prediction needs a fixed frame size."""
        transmitter = BackgroundStreamTransmitter(mode=REFERENCE_LAST, codec="av1", crf=38)
        transmitter.push(_panning_scenes(1)[0])
        with pytest.raises(ValueError, match="fixed frame size"):
            transmitter.push(np.zeros((HEIGHT + 2, WIDTH, 3), dtype=np.uint8))

    def test_the_client_refuses_a_chain_it_did_not_receive(self) -> None:
        """A P-chain has no random access; joining mid-stream must fail loudly.

        Brief §3 accepts that for a paper. It does not accept a client that
        quietly returns the wrong picture instead.
        """
        scenes = _panning_scenes(3)
        transmitter = BackgroundStreamTransmitter(mode=REFERENCE_LAST, codec="av1", crf=38)
        payloads = [transmitter.push(p) for p in scenes]
        latecomer = BackgroundStreamReceiver(codec="av1")
        with pytest.raises(StreamDrift, match="never received"):
            latecomer.receive(payloads[2], height=HEIGHT, width=WIDTH)


@pytest.mark.integration
class TestBatchPath:
    """`stream_linear` is an optimisation, so it has to produce the same bytes.

    It is only correct because the encode is prefix-stable. If that ever stops
    holding, the sweep would quietly report payloads no incremental transmitter
    would ever have emitted -- which is the offline archiver BP30 exists to
    avoid claiming is a codec.
    """

    @pytest.mark.parametrize("interval", [KEYFRAME_NEVER, 2])
    def test_the_batch_path_agrees_with_pushing_scene_by_scene(self, interval: int) -> None:
        scenes = _panning_scenes(5)
        mode = REFERENCE_LAST if interval == KEYFRAME_NEVER else REFERENCE_PERIODIC_I
        incremental = BackgroundStreamTransmitter(
            mode=mode, codec="av1", crf=38, keyframe_interval=interval
        )
        one_at_a_time = [incremental.push(p) for p in scenes]
        batched = stream_linear(
            scenes, codec="av1", crf=38, keyframe_interval=interval, mode=mode
        )
        assert [p.payload for p in batched] == [p.payload for p in one_at_a_time]
        assert [p.chain for p in batched] == [p.chain for p in one_at_a_time]
        assert [p.picture_type for p in batched] == [p.picture_type for p in one_at_a_time]

    def test_the_batch_path_refuses_a_mode_whose_chains_are_not_linear(self) -> None:
        with pytest.raises(ValueError, match="linear chains"):
            stream_linear(_panning_scenes(2), mode=REFERENCE_BEST_SCORED)
