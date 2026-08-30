"""What BP29's plate-codec sweep must get right to be worth reading.

The sweep exists to catch one specific failure — a codec named in a config that
never reaches an encoder, or falls back to a different one — so the properties
tested here are the ones that catch it, plus the refusal to extrapolate a
matched-fidelity claim. No ffmpeg is needed: payloads are literal signatures and
rungs are constructed directly, so this runs anywhere.
"""

from __future__ import annotations

from typing import Any

import pytest

from experiments.tier.plate_codec_sweep import (
    SidecarRung,
    bytes_at_fidelity,
    check_bounds,
    container_kind,
)


def rung(
    codec: str,
    knob: str,
    payload_bytes: int,
    psnr: float,
    *,
    container: str | None = None,
    bit_identical: bool = False,
    detail: dict[str, Any] | None = None,
) -> SidecarRung:
    kinds = {"jpeg": "jpeg", "png": "png", "roi-video": "mp4"}
    return SidecarRung(
        codec=codec,
        knob=knob,
        codec_id=f"{codec}:{knob}",
        payload_bytes=payload_bytes,
        psnr_rgb_db=psnr,
        psnr_y_db=psnr,
        bit_identical=bit_identical,
        container=kinds[codec] if container is None else container,
        seconds=0.0,
        detail=detail or {},
    )


def healthy_rungs() -> list[SidecarRung]:
    """A sweep with nothing wrong with it: the null for every alarm below."""
    return [
        rung("jpeg", "q30", 283_431, 37.98),
        rung("jpeg", "q75", 463_334, 42.79),
        rung("png", "z3", 14_000_000, float("inf"), bit_identical=True),
        rung(
            "roi-video",
            "crf30",
            180_000,
            39.50,
            detail={"ffprobe": {"probed": True, "codec_name": "h264"}},
        ),
        rung(
            "roi-video",
            "crf18",
            320_000,
            43.10,
            detail={"ffprobe": {"probed": True, "codec_name": "h264"}},
        ),
    ]


def test_container_kind_reads_the_payload_not_the_label() -> None:
    assert container_kind(b"\xff\xd8\xff\xe0" + b"\x00" * 8) == "jpeg"
    assert container_kind(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8) == "png"
    assert container_kind(b"\x00\x00\x00\x18ftypisom" + b"\x00" * 8) == "mp4"
    assert container_kind(b"not a container at all") == "unknown"


def test_a_codec_that_produced_the_wrong_container_is_an_alarm() -> None:
    rungs = healthy_rungs()
    rungs[1] = rung("jpeg", "q75", 463_334, 42.79, container="mp4")
    alarms = check_bounds(rungs, [])
    assert any("not a jpeg" in alarm for alarm in alarms)


def test_a_video_payload_from_another_encoder_is_an_alarm() -> None:
    """The §14 failure: ffmpeg falls back and every quality it returns is capped."""
    rungs = healthy_rungs()
    rungs[3] = rung(
        "roi-video",
        "crf30",
        180_000,
        39.50,
        detail={"ffprobe": {"probed": True, "codec_name": "hevc"}},
    )
    alarms = check_bounds(rungs, [])
    assert any("codec_name" in alarm for alarm in alarms)


def test_png_must_be_lossless_and_dearer_than_every_jpeg() -> None:
    lying = [
        rung("jpeg", "q75", 463_334, 42.79),
        rung("png", "z3", 400_000, 51.0, bit_identical=False),
    ]
    alarms = check_bounds(lying, [])
    assert any("bit-identical" in alarm for alarm in alarms)
    assert any("Lossless must cost more" in alarm for alarm in alarms)


def test_a_knob_that_buys_quality_without_costing_bytes_is_an_alarm() -> None:
    rungs = healthy_rungs()
    rungs[1] = rung("jpeg", "q75", 200_000, 42.79)
    alarms = check_bounds(rungs, [])
    assert any("must cost more" in alarm for alarm in alarms)


def test_two_codecs_agreeing_to_the_byte_is_an_alarm() -> None:
    rungs = healthy_rungs()
    rungs[3] = rung(
        "roi-video",
        "crf30",
        283_431,
        39.50,
        detail={"ffprobe": {"probed": True, "codec_name": "h264"}},
    )
    alarms = check_bounds(rungs, [])
    assert any("same 283431 B" in alarm for alarm in alarms)


def test_a_healthy_sweep_raises_nothing() -> None:
    assert check_bounds(healthy_rungs(), []) == []


def test_the_runner_sending_a_different_plate_than_was_measured_is_an_alarm() -> None:
    arms = [
        {
            "arm": "roi-video:crf30",
            "plate_bytes": 175_000,
            "sidecar_bytes_for_same_settings": 180_000,
            "is_rate": True,
        }
    ]
    alarms = check_bounds(healthy_rungs(), arms)
    assert any("is not the plate measured here" in alarm for alarm in alarms)


def test_the_reference_arm_must_reproduce_the_bp24_rung() -> None:
    arms = [
        {
            "arm": "jpeg:75",
            "plate_bytes": 401_000,
            "sidecar_bytes_for_same_settings": 401_000,
            "is_rate": True,
        }
    ]
    alarms = check_bounds(healthy_rungs(), arms)
    assert any("463,334 B" in alarm for alarm in alarms)


def test_a_total_that_is_not_a_rate_is_an_alarm() -> None:
    arms = [{"arm": "png:3", "plate_bytes": None, "is_rate": False}]
    alarms = check_bounds(healthy_rungs(), arms)
    assert any("not a rate" in alarm for alarm in alarms)


def test_a_failed_arm_is_reported_rather_than_alarmed_on() -> None:
    arms = [{"arm": "roi-video:crf30", "error": "RuntimeError('ffmpeg missing')"}]
    assert check_bounds(healthy_rungs(), arms) == []


def test_matched_fidelity_never_extrapolates() -> None:
    """A matched-fidelity claim outside the measured range is a claim about a fit."""
    curve = [rung("jpeg", "q30", 283_431, 37.98), rung("jpeg", "q75", 463_334, 42.79)]
    outside = bytes_at_fidelity(curve, 45.0)
    assert outside["bytes"] is None
    assert "outside measured" in outside["reason"]
    assert outside["measured_range_dB"] == pytest.approx([37.98, 42.79])


def test_matched_fidelity_interpolates_inside_the_measured_range() -> None:
    curve = [rung("jpeg", "q30", 283_431, 37.98), rung("jpeg", "q75", 463_334, 42.79)]
    at_end = bytes_at_fidelity(curve, 37.98)
    assert at_end["bytes"] == pytest.approx(283_431, rel=1e-3)
    middle = bytes_at_fidelity(curve, 40.0)
    assert 283_431 < middle["bytes"] < 463_334


def test_a_lossless_curve_offers_no_matched_fidelity_point() -> None:
    """PNG's PSNR is infinite, so there is no rate at which it 'matches' a target."""
    lossless = [
        rung("png", "z3", 14_000_000, float("inf"), bit_identical=True),
        rung("png", "z9", 13_000_000, float("inf"), bit_identical=True),
    ]
    answer = bytes_at_fidelity(lossless, 42.8)
    assert answer["bytes"] is None
    assert "finite-PSNR" in answer["reason"]
