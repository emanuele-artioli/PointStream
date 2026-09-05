"""Required behaviour for background stream encoder effort (BP56).

Behaviour
1. Default av1 flags stay realtime cpu-used 8; CODECS is not rewritten.
2. The candidate (good, cpu-used 4) keeps lag-in-frames 0 and bf 0.
3. Changing the option changes the ffmpeg command and the emitted bytes.
4. Independent 2/3/4-frame encodes are prefix-stable on textured, static and
   translated plates.
5. Last-reference push stays prefix-stable; a same-size context reset is a
   new keyframe; a byte-only client matches the encoder reconstruction.
6. Old checkpoints without effort keys still restore on the default transmitter.

Plausible misuse
7. Unsupported usage, non-av1 effort, and a changed-option resume are refused.

Deliberately not tested: native 4K rate, BD-rate, a slower-than-candidate grid,
SVT-AV1 presets, re-encoding BP49–BP53 outputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.components.background.stream import (
    CANDIDATE_STREAM_CPU_USED,
    CANDIDATE_STREAM_USAGE,
    CODECS,
    DEFAULT_STREAM_CPU_USED,
    DEFAULT_STREAM_USAGE,
    REFERENCE_LAST,
    BackgroundStreamReceiver,
    BackgroundStreamTransmitter,
    assert_independent_prefixes_stable,
    independent_prefix_payloads,
    last_encode_record,
    probe_stream_effort,
    resolve_stream_codec,
)
from src.contracts import config as cfg
from src.contracts.domain import BACKGROUND_PANORAMA_FULL, BACKGROUND_PANORAMA_STREAM
from src.contracts.errors import ConfigError

HEIGHT, WIDTH = 48, 64


def _textured(count: int = 4) -> list[np.ndarray]:
    rng = np.random.default_rng(56)
    canvas = np.full((HEIGHT, WIDTH + 4 * count, 3), 36, dtype=np.uint8)
    for _ in range(14):
        top = int(rng.integers(0, HEIGHT - 10))
        left = int(rng.integers(0, canvas.shape[1] - 14))
        canvas[top : top + 10, left : left + 14] = rng.integers(50, 240, 3, dtype=np.uint8)
    return [np.ascontiguousarray(canvas[:, k * 3 : k * 3 + WIDTH]) for k in range(count)]


def _static(count: int = 4) -> list[np.ndarray]:
    plate = _textured(1)[0]
    return [np.ascontiguousarray(plate.copy()) for _ in range(count)]


def _translated(count: int = 4) -> list[np.ndarray]:
    plate = _textured(1)[0]
    out = []
    for shift in range(count):
        rolled = np.roll(plate, shift * 2, axis=1)
        out.append(np.ascontiguousarray(rolled))
    return out


def test_default_av1_spec_is_unchanged_realtime() -> None:
    spec = resolve_stream_codec("av1")
    assert spec.low_delay == CODECS["av1"].low_delay
    assert spec.low_delay == (
        "-cpu-used",
        "8",
        "-usage",
        "realtime",
        "-lag-in-frames",
        "0",
        "-bf",
        "0",
    )
    before = CODECS["av1"].low_delay
    resolve_stream_codec(
        "av1", usage=CANDIDATE_STREAM_USAGE, cpu_used=CANDIDATE_STREAM_CPU_USED
    )
    assert CODECS["av1"].low_delay is before


def test_candidate_keeps_causal_flags() -> None:
    spec = resolve_stream_codec(
        "av1", usage=CANDIDATE_STREAM_USAGE, cpu_used=CANDIDATE_STREAM_CPU_USED
    )
    assert spec.encoder == "libaom-av1"
    assert spec.low_delay == (
        "-cpu-used",
        "4",
        "-usage",
        "good",
        "-lag-in-frames",
        "0",
        "-bf",
        "0",
    )


def test_hevc_rejects_nondefault_effort() -> None:
    with pytest.raises(ValueError, match="libaom-av1"):
        resolve_stream_codec("hevc", usage="good", cpu_used=4)


def test_unknown_usage_is_refused() -> None:
    with pytest.raises(ValueError, match="usage"):
        resolve_stream_codec("av1", usage="allintra", cpu_used=4)


def test_changed_option_resume_is_rejected() -> None:
    tx = BackgroundStreamTransmitter(mode=REFERENCE_LAST, codec="av1", crf=51)
    tx._payloads = [b"x"]
    tx._chains = [(0,)]
    tx._originals = [np.zeros((2, 2, 3), dtype=np.uint8)]
    tx._reconstructions = [np.zeros((2, 2, 3), dtype=np.uint8)]
    state = tx.export_state()
    other = BackgroundStreamTransmitter(
        mode=REFERENCE_LAST,
        codec="av1",
        crf=51,
        stream_usage=CANDIDATE_STREAM_USAGE,
        stream_cpu_used=CANDIDATE_STREAM_CPU_USED,
    )
    with pytest.raises(ValueError, match="effort"):
        other.import_state(state)


def test_default_resume_accepts_legacy_state_without_effort_keys() -> None:
    tx = BackgroundStreamTransmitter(mode=REFERENCE_LAST, codec="av1", crf=51)
    tx._payloads = [b"obu"]
    tx._chains = [(0,)]
    tx._originals = [np.zeros((2, 2, 3), dtype=np.uint8)]
    tx._reconstructions = [np.zeros((2, 2, 3), dtype=np.uint8)]
    state = tx.export_state()
    del state["stream_usage"]
    del state["stream_cpu_used"]
    restored = BackgroundStreamTransmitter(mode=REFERENCE_LAST, codec="av1", crf=51)
    restored.import_state(state)
    assert restored._payloads == [b"obu"]
    assert restored.stream_usage == DEFAULT_STREAM_USAGE
    assert restored.stream_cpu_used == DEFAULT_STREAM_CPU_USED


def test_default_config_keeps_realtime_effort() -> None:
    loaded = cfg.default()
    assert loaded.background.stream_usage == "realtime"
    assert loaded.background.stream_cpu_used == 8
    cfg.validate(loaded)


def test_nondefault_effort_on_a_still_background_is_refused() -> None:
    with pytest.raises(ConfigError, match="stream_usage|effort"):
        cfg.load(
            {
                "background": {
                    "method": BACKGROUND_PANORAMA_FULL,
                    "stream_usage": "good",
                    "stream_cpu_used": 4,
                }
            }
        )


def test_nondefault_effort_on_hevc_stream_is_refused() -> None:
    with pytest.raises(ConfigError, match="libaom-av1|stream_codec"):
        cfg.load(
            {
                "background": {
                    "method": BACKGROUND_PANORAMA_STREAM,
                    "stream_codec": "hevc",
                    "stream_usage": "good",
                    "stream_cpu_used": 4,
                }
            }
        )


@pytest.mark.integration
def test_candidate_options_are_on_the_command_and_change_bytes() -> None:
    probe = probe_stream_effort(
        usage=CANDIDATE_STREAM_USAGE, cpu_used=CANDIDATE_STREAM_CPU_USED
    )
    assert probe["supported"] is True, probe
    assert probe["command_has_usage"] is True
    assert probe["command_has_cpu_used"] is True
    assert "-lag-in-frames" in probe["command"]
    frames = _textured(2)
    default_tx = BackgroundStreamTransmitter(
        mode=REFERENCE_LAST, codec="av1", crf=51
    )
    candidate_tx = BackgroundStreamTransmitter(
        mode=REFERENCE_LAST,
        codec="av1",
        crf=51,
        stream_usage=CANDIDATE_STREAM_USAGE,
        stream_cpu_used=CANDIDATE_STREAM_CPU_USED,
    )
    default_bytes = [default_tx.push(frame).payload for frame in frames]
    default_cmd = last_encode_record()["argv"]
    candidate_bytes = [candidate_tx.push(frame).payload for frame in frames]
    candidate_cmd = last_encode_record()["argv"]
    assert default_bytes != candidate_bytes
    assert "realtime" in default_cmd
    assert "good" in candidate_cmd
    assert "8" in default_cmd
    assert "4" in candidate_cmd


@pytest.mark.integration
@pytest.mark.parametrize("factory", [_textured, _static, _translated])
def test_independent_prefixes_stay_byte_identical(factory: object) -> None:
    frames = factory(4)  # type: ignore[operator]
    payloads = independent_prefix_payloads(
        frames,
        usage=CANDIDATE_STREAM_USAGE,
        cpu_used=CANDIDATE_STREAM_CPU_USED,
        crf=51,
    )
    assert_independent_prefixes_stable(payloads)


@pytest.mark.integration
def test_last_reference_reset_and_byte_only_client() -> None:
    frames = _textured(4)
    transmitter = BackgroundStreamTransmitter(
        mode=REFERENCE_LAST,
        codec="av1",
        crf=51,
        stream_usage=CANDIDATE_STREAM_USAGE,
        stream_cpu_used=CANDIDATE_STREAM_CPU_USED,
    )
    receiver = BackgroundStreamReceiver(codec="av1")
    emitted: list[bytes] = []
    for plate in frames:
        payload = transmitter.push(plate)
        emitted.append(payload.payload)
        client = receiver.receive(payload, height=HEIGHT, width=WIDTH)
        assert np.array_equal(client, transmitter.reconstructions[payload.index])
    replay = BackgroundStreamTransmitter(
        mode=REFERENCE_LAST,
        codec="av1",
        crf=51,
        stream_usage=CANDIDATE_STREAM_USAGE,
        stream_cpu_used=CANDIDATE_STREAM_CPU_USED,
    )
    for offset, plate in enumerate(frames):
        assert replay.push(plate).payload == emitted[offset]

    transmitter.reset()
    receiver.reset()
    first = transmitter.push(frames[0])
    assert first.is_keyframe
    client = receiver.receive(first, height=HEIGHT, width=WIDTH)
    assert np.array_equal(client, transmitter.reconstructions[0])

    packets_only = BackgroundStreamReceiver(codec="av1")
    restored = None
    replay_tx = BackgroundStreamTransmitter(
        mode=REFERENCE_LAST,
        codec="av1",
        crf=51,
        stream_usage=CANDIDATE_STREAM_USAGE,
        stream_cpu_used=CANDIDATE_STREAM_CPU_USED,
    )
    for plate in frames:
        payload = replay_tx.push(plate)
        restored = packets_only.receive(payload, height=HEIGHT, width=WIDTH)
    assert restored is not None
    assert np.array_equal(restored, replay_tx.reconstructions[-1])
