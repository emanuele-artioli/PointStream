"""Required behaviour for the av1 and vvc intra background sidecars (`BP29`).

The plate is 88-91% of PointStream's payload and it was a JPEG. These are the
properties that have to hold before any of that is believed, and each is here
because it could plausibly fail rather than to execute a line:

1. ``av1`` and ``vvc`` are reachable values of ``background.codec``.
2. The codec is taken from the name, not hardcoded — the paired ladder needs the
   plate on the *same* codec as the anchor, so a sidecar that always built av1
   would silently break pairing.
3. ``codec_id`` separates two encodes that are not the same encode.
4. A real plate round-trips: bytes out, pixels back, same shape.
5. **The byte count moves with the quality knob, and a coarser setting returns
   visibly worse pixels.** A flag existing is not a feature working;
   ``background.codec`` accepted three values and reached nothing at all until
   BP24 wired ``make_background``.
6. The payload is an actual av1/vvc bitstream and not an image the wrapper fell
   back to. A size within a few percent of JPEG's would mean the encoder never
   ran.
7. The quality curve is not flat. BP24 spent a sweep measuring frames that had
   been through the rung's codec *and then* through x264, because a decode
   named no ``-c:v`` (`plans/BP24-findings.md` §14).
8. **The plate decodes exactly as the anchor decodes.** This one caught a real
   defect: decoding the bitstream straight to a PNG is one conversion instead of
   two and was bit-identical on av1, but lost 0.57 dB on vvc, which is 10-bit.
   A plate reconstructed on a different path from the anchor it is paired
   against is not a paired arm.

Deliberately not tested: SVT-AV1's and libvvenc's own rate-distortion
behaviour, ffmpeg's demuxers, and the absolute byte counts on the 4K plate —
those are third-party behaviour and a measurement, not a contract. The 4K
numbers live in ``outputs/bp29-intra-sidecar/``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.components.background.sidecar import (
    ALL_SIDECAR_CODECS,
    DEFAULT_INTRA_QP,
    INTRA_PRESETS,
    INTRA_SIDECAR_CODECS,
    IntraCodecSidecar,
    build_sidecar,
    normalize_sidecar,
)
from src.contracts.errors import ConfigValueError

#: Fine and coarse QP per codec. Far enough apart that a real encoder must
#: separate them, and inside each codec's own range (av1 1-63, vvc 0-63).
QP_PAIRS = {"av1": (30, 58), "vvc": (25, 55)}


def _plate(height: int = 192, width: int = 320, seed: int = 7) -> np.ndarray:
    """A BGR still with structure *and* detail, so a QP change has somewhere to bite."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.stack(
        [
            128 + 90 * np.sin(xx / 5.0) * np.cos(yy / 6.0),
            128 + 90 * np.cos(xx / 9.0 + 1.0) * np.sin(yy / 4.0),
            128 + 90 * np.sin((xx + yy) / 7.0 + 2.0),
        ],
        axis=-1,
    )
    return np.clip(base + rng.normal(0.0, 12.0, size=base.shape), 0, 255).astype(np.uint8)


def _psnr(reference: np.ndarray, candidate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float64)
    got = np.asarray(candidate, dtype=np.float64)
    mse = float(np.mean((ref - got) ** 2))
    return float("inf") if mse == 0.0 else 10.0 * float(np.log10((255.0**2) / mse))


class TestTheCodecIsSelectableAndComesFromConfig:
    def test_av1_and_vvc_are_background_sidecar_codecs(self) -> None:
        assert {"av1", "vvc"} <= ALL_SIDECAR_CODECS
        assert INTRA_SIDECAR_CODECS == {"av1", "vvc"}
        assert normalize_sidecar("av1") == "av1"
        assert normalize_sidecar("VVC") == "vvc"
        assert normalize_sidecar("av1-intra") == "av1"
        assert normalize_sidecar("h266") == "vvc"

    def test_build_sidecar_returns_the_codec_it_was_named(self) -> None:
        """Not hardcoded av1. The ladder pairs the plate with the anchor's codec."""
        for codec in ("av1", "vvc"):
            built = build_sidecar(codec)
            assert isinstance(built, IntraCodecSidecar)
            assert built.name == codec
            assert built.codec_id.startswith(f"{codec}:")
        assert build_sidecar("av1").codec_id != build_sidecar("vvc").codec_id

    def test_the_preset_matches_the_codec_layer_so_a_pair_is_matched_effort(self) -> None:
        from src.components.codec.measure import PRESETS

        for codec, preset in INTRA_PRESETS.items():
            assert PRESETS[codec] == preset
            assert build_sidecar(codec).preset == preset

    def test_codec_id_separates_two_encodes_that_are_not_the_same_encode(self) -> None:
        assert build_sidecar("av1", intra_qp=30).codec_id != build_sidecar(
            "av1", intra_qp=50
        ).codec_id
        assert build_sidecar("av1", intra_preset="8").codec_id != build_sidecar(
            "av1", intra_preset="10"
        ).codec_id
        assert build_sidecar("av1", intra_qp=DEFAULT_INTRA_QP).codec_id == (
            f"av1:intra:qp{DEFAULT_INTRA_QP}:{INTRA_PRESETS['av1']}"
        )


    def test_background_codec_av1_is_reachable_from_a_loaded_config(self) -> None:
        """The whole point of the axis: it has to arrive from config, not a call site.

        ``background.codec`` accepted three values and reached nothing at all
        until BP24 wired ``make_background`` (`plans/BP24-findings.md` §6), so
        "the enum has a new member" is not evidence that anything is wired.
        """
        from src.components.background import REGISTRY as BACKGROUND
        from src.components.background.strategy import bind
        from src.contracts import config
        from src.contracts.config import validate_backends
        from src.contracts.domain import BACKGROUND_PANORAMA_FULL
        from src.components.rigid import REGISTRY as RIGID

        for codec in ("av1", "vvc"):
            loaded = config.load(
                {"background": {"method": BACKGROUND_PANORAMA_FULL, "codec": codec}}
            )
            validate_backends(loaded, registries={"background": BACKGROUND, "rigid": RIGID})
            model = bind(loaded)
            assert model.codec_name == codec
            assert model.codec_id == f"{codec}:intra:qp{DEFAULT_INTRA_QP}:{INTRA_PRESETS[codec]}"


class TestMisuseIsRefusedBeforeAnyEncoderRuns:
    def test_a_qp_outside_the_codecs_range_is_a_config_error(self) -> None:
        for codec in ("av1", "vvc"):
            with pytest.raises(ConfigValueError, match="intra_qp"):
                build_sidecar(codec, intra_qp=64)
        with pytest.raises(ConfigValueError, match="intra_qp"):
            build_sidecar("av1", intra_qp=0)  # av1's floor is 1, not 0

    def test_a_non_intra_codec_cannot_be_forced_into_the_intra_sidecar(self) -> None:
        with pytest.raises(ConfigValueError, match="intra sidecar codec"):
            IntraCodecSidecar("jpeg")

    def test_a_plate_that_is_not_bgr_is_refused(self) -> None:
        sidecar = build_sidecar("av1")
        with pytest.raises(ValueError, match="BGR plate"):
            sidecar.encode(np.zeros((8, 8), dtype=np.uint8))
        with pytest.raises(ValueError, match="BGR plate"):
            sidecar.encode(np.zeros((8, 8, 4), dtype=np.uint8))

    def test_an_empty_payload_is_not_silently_decoded(self) -> None:
        with pytest.raises(RuntimeError, match="Empty"):
            build_sidecar("vvc").decode(b"")


@pytest.mark.integration
@pytest.mark.parametrize("codec", ["av1", "vvc"])
def test_intra_sidecar_round_trips_a_plate(codec: str) -> None:
    """Bytes out, pixels back, same shape — and far smaller than raw."""
    plate = _plate()
    sidecar = build_sidecar(codec, intra_qp=QP_PAIRS[codec][0])
    payload = sidecar.encode(plate)
    decoded = sidecar.decode(payload)

    assert len(payload) > 0
    assert len(payload) < plate.nbytes // 4
    assert decoded.shape == plate.shape
    assert decoded.dtype == np.uint8
    # A real reconstruction, not noise. 20 dB is a floor, not a target.
    assert _psnr(plate, decoded) > 20.0


@pytest.mark.integration
@pytest.mark.parametrize("codec", ["av1", "vvc"])
def test_a_coarser_setting_costs_fewer_bytes_and_returns_visibly_worse_pixels(
    codec: str,
) -> None:
    """The knob has to reach the encoder, and the loss has to reach the pixels.

    Both halves matter. Bytes moving while quality stays flat is the signature
    of a second encoder in the decode path, which is exactly what happened to
    `coded_roundtrip` in BP24.
    """
    plate = _plate()
    fine_qp, coarse_qp = QP_PAIRS[codec]

    fine = build_sidecar(codec, intra_qp=fine_qp)
    coarse = build_sidecar(codec, intra_qp=coarse_qp)
    fine_payload, coarse_payload = fine.encode(plate), coarse.encode(plate)
    fine_psnr = _psnr(plate, fine.decode(fine_payload))
    coarse_psnr = _psnr(plate, coarse.decode(coarse_payload))

    assert len(coarse_payload) < len(fine_payload), (
        f"{codec}: qp{coarse_qp} produced {len(coarse_payload)} B against "
        f"qp{fine_qp}'s {len(fine_payload)} B — the rate knob is not reaching the encoder."
    )
    assert coarse_psnr < fine_psnr - 2.0, (
        f"{codec}: qp{coarse_qp} scored {coarse_psnr:.2f} dB against qp{fine_qp}'s "
        f"{fine_psnr:.2f} dB. A quality curve that stays flat while bytes fall means "
        f"a second encoder is capping the decode (BP24 findings §14)."
    )


@pytest.mark.integration
@pytest.mark.parametrize("codec", ["av1", "vvc"])
def test_the_payload_is_a_real_bitstream_and_not_an_image(codec: str) -> None:
    """A size a few percent from JPEG's would mean the encoder never ran."""
    payload = build_sidecar(codec, intra_qp=QP_PAIRS[codec][0]).encode(_plate())
    assert not payload.startswith(b"\xff\xd8\xff"), "this is a JPEG, not a bitstream"
    assert not payload.startswith(b"\x89PNG"), "this is a PNG, not a bitstream"
    if codec == "av1":
        # SVT-AV1 writes IVF; 'DKIF' is its magic.
        assert payload[:4] == b"DKIF"
    else:
        # A raw VVC Annex-B elementary stream opens on a start code.
        assert payload[:4] in {b"\x00\x00\x00\x01", b"\x00\x00\x01\x00"} or payload[
            :3
        ] == b"\x00\x00\x01"


@pytest.mark.integration
@pytest.mark.parametrize("codec", ["av1", "vvc"])
def test_the_encoder_is_identified_by_path_and_version(codec: str) -> None:
    """A size without the binary that produced it is not evidence."""
    from pathlib import Path

    path, version = build_sidecar(codec).probe_encoder()
    assert Path(path).exists()
    assert version and version != "unknown"


@pytest.mark.integration
@pytest.mark.parametrize("codec", ["av1", "vvc"])
def test_the_plate_decodes_exactly_as_the_anchor_decodes(codec: str) -> None:
    """Same bitstream and same pixels as `coded_roundtrip`, which the ladder uses.

    Guards the vvc defect specifically. libvvenc emits 10-bit 4:2:0, so a
    shorter decode that skips the anchor's 8-bit intermediate lands on different
    pixels — 0.57 dB apart at qp25 on the real 4K plate — while av1, being
    8-bit, shows nothing. A test on av1 alone would have passed through it.
    """
    from src.components.codec.measure import coded_roundtrip
    from src.contracts.codecs import EncodeRequest, RateControl

    plate = _plate()
    qp = QP_PAIRS[codec][0]
    sidecar = build_sidecar(codec, intra_qp=qp)
    payload = sidecar.encode(plate)
    decoded = sidecar.decode(payload)

    anchor_bytes, anchor_frames = coded_roundtrip(
        plate[np.newaxis, :, :, ::-1],  # coded_roundtrip takes RGB
        request=EncodeRequest(
            codec_name=codec,
            rate_control=RateControl.QP,
            rate=qp,
            preset=INTRA_PRESETS[codec],
            pix_fmt="yuv420p",
        ),
    )

    assert len(payload) == anchor_bytes, "the plate and the anchor did not code the same"
    assert np.array_equal(decoded, anchor_frames[0][:, :, ::-1]), (
        f"{codec}: the sidecar's decode differs from the anchor's. "
        f"PSNR {_psnr(plate, decoded):.3f} dB against "
        f"{_psnr(plate, anchor_frames[0][:, :, ::-1]):.3f} dB."
    )
