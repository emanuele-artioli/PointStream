"""What each codec actually honours, declared as data and checked.

Encoders lie by omission. `libsvtav1` accepts `-pix_fmt yuv444p` on the command
line, returns success, and emits yuv420p — so every residual encode that asked
for 4:4:4 silently got 4:2:0, and the pixel-format ablation built on top of it
measured a knob that did nothing. The only defence is to state what each encoder
supports and reject the request before it runs.

This module declares and validates. It does **not** build command lines: that is
the components layer's job, and it differs per driver. Two drivers matter here,
because the two encoders with usable region-of-interest support are only
reachable as standalone binaries:

- `ffmpeg` sub-encoders — `libx264`, `libvvenc`
- standalone binaries — `kvazaar`, `SvtAv1EncApp`

Region-of-interest support is load-bearing rather than decorative. The paper has
to show this system beating baseline codecs *when the baselines are also allowed
to spend their bits where the content is*, or it is beating a strawman.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Final

from src.contracts.errors import CodecConstraintError

# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------


class RateControl(str, Enum):
    """How an encoder is told what quality or bitrate to hit.

    These are not interchangeable, and mixing them across arms of a comparison
    is how a published table came to report a 24.9–40.2% region-of-interest
    saving that turned out to be entirely the *baseline's* bitrate overshoot.
    See `assert_matched_rate_control`.
    """

    CRF = "crf"
    """Constant rate factor. Quality target, variable bitrate."""

    QP = "qp"
    """Fixed quantization parameter. The rate control ROI arms use, because a
    delta-QP map is only meaningful relative to a known base QP."""

    BITRATE = "bitrate"
    """Target bitrate. Note that some encoders overshoot this substantially."""

    LOSSLESS = "lossless"
    """Mathematically lossless. Expensive, and used here as a ceiling
    calibration rather than an operating point."""


class Driver(str, Enum):
    """How the encoder is invoked."""

    FFMPEG = "ffmpeg"
    """An ffmpeg `-c:v` sub-encoder."""

    BINARY = "binary"
    """A standalone executable with its own argument grammar."""


class RoiSupport(str, Enum):
    """How, if at all, an encoder can be told where to spend bits."""

    NONE = "none"
    """No region control. Needs the pixel-domain path to have an ROI arm."""

    DELTA_QP_MAP = "delta-qp-map"
    """A per-block map of QP offsets. Genuine block-level control."""

    FFMPEG_ADDROI = "ffmpeg-addroi"
    """ffmpeg's `addroi` filter. Present in principle; **treat as unverified**
    until a test shows the output actually differs from the same encode without
    it. A prior project abandoned this path unfinished."""


# --------------------------------------------------------------------------
# Capabilities
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CodecCapabilities:
    """What one encoder supports, as checkable data.

    Args:
        name: Registry key used in config, e.g. ``av1``.
        driver: How it is invoked.
        invocation: ffmpeg encoder name, or the executable name.
        pix_fmts: Pixel formats it genuinely emits. Anything else is rejected,
            even if the encoder would silently accept the flag.
        rate_controls: Rate-control modes it supports.
        preset_scale: Which preset vocabulary it uses. `"x264"` for named
            presets, `"svtav1-numeric"` for 0–13, `"vvenc"` for its own names,
            `"none"` where presets do not apply.
        roi: Region-of-interest mechanism.
        lossless_rate: The rate value meaning lossless, where one exists.
        standard: The coding standard, for grouping in tables.
        notes: Anything a reader would otherwise have to discover the hard way.
    """

    name: str
    driver: Driver
    invocation: str
    pix_fmts: frozenset[str]
    rate_controls: frozenset[RateControl]
    preset_scale: str
    roi: RoiSupport
    lossless_rate: int | None = None
    standard: str = ""
    notes: str = ""

    @property
    def supports_roi(self) -> bool:
        """Whether this encoder can be given a region map at all."""
        return self.roi is not RoiSupport.NONE

    @property
    def roi_is_verified(self) -> bool:
        """Whether its ROI mechanism is known to work, not merely documented."""
        return self.roi is RoiSupport.DELTA_QP_MAP


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------

X264 = CodecCapabilities(
    name="avc",
    driver=Driver.FFMPEG,
    invocation="libx264",
    pix_fmts=frozenset({"yuv420p", "yuv422p", "yuv444p", "gray"}),
    rate_controls=frozenset({RateControl.CRF, RateControl.QP, RateControl.BITRATE, RateControl.LOSSLESS}),
    preset_scale="x264",
    roi=RoiSupport.FFMPEG_ADDROI,
    lossless_rate=0,
    standard="H.264/AVC",
    notes="The speed rung — what makes a real-time target reachable at all.",
)

KVAZAAR = CodecCapabilities(
    name="hevc",
    driver=Driver.BINARY,
    invocation="kvazaar",
    pix_fmts=frozenset({"yuv420p"}),
    rate_controls=frozenset({RateControl.QP, RateControl.BITRATE}),
    preset_scale="x264",
    roi=RoiSupport.DELTA_QP_MAP,
    standard="H.265/HEVC",
    notes=(
        "No ffmpeg encoder on this host, so it is driven as a binary. Chosen over "
        "libx265 precisely because its --roi delta-QP map demonstrably works. "
        "Its bitrate mode overshoots the target substantially, which is why ROI "
        "comparisons here are specified at fixed QP."
    ),
)

SVT_AV1 = CodecCapabilities(
    name="av1",
    driver=Driver.BINARY,
    invocation="SvtAv1EncApp",
    pix_fmts=frozenset({"yuv420p"}),
    rate_controls=frozenset({RateControl.CRF, RateControl.QP, RateControl.BITRATE}),
    preset_scale="svtav1-numeric",
    roi=RoiSupport.DELTA_QP_MAP,
    standard="AV1",
    notes=(
        "Driven as a binary because --roi-map-file, the only route to native AV1 "
        "region control, is not exposed through ffmpeg's libsvtav1 wrapper. "
        "Emits yuv420p regardless of any pixel-format request; that silent "
        "substitution is the reason this table exists."
    ),
)

VVENC = CodecCapabilities(
    name="vvc",
    driver=Driver.FFMPEG,
    invocation="libvvenc",
    pix_fmts=frozenset({"yuv420p"}),
    rate_controls=frozenset({RateControl.QP, RateControl.BITRATE}),
    preset_scale="vvenc",
    roi=RoiSupport.NONE,
    standard="H.266/VVC",
    notes=(
        "No region map of any kind — only perceptual QP adaptation — so its ROI "
        "arm has to come from pixel-domain degradation. Expected to be very slow, "
        "which makes it a useful encode-time comparator as well as a quality one."
    ),
)

#: Every codec, by config name.
CODECS: Final[Mapping[str, CodecCapabilities]] = {
    codec.name: codec for codec in (X264, KVAZAAR, SVT_AV1, VVENC)
}


def codec(name: str) -> CodecCapabilities:
    """Look up a codec's capabilities by config name."""
    try:
        return CODECS[name]
    except KeyError:
        raise CodecConstraintError(name, "name", name, sorted(CODECS)) from None


# --------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EncodeRequest:
    """One encode, validated against what the codec actually supports.

    Args:
        codec_name: Key into `CODECS`.
        rate_control: How `rate` is interpreted.
        rate: The CRF, QP, or bitrate in bits per second. Ignored when
            `rate_control` is `LOSSLESS`.
        preset: Encoder effort, in that codec's own preset vocabulary.
        pix_fmt: Requested pixel format.
        roi_map: Per-block QP offsets, when this is a region-of-interest arm.
        extra_args: Escape hatch for one-off experiments. Not validated.
    """

    codec_name: str
    rate_control: RateControl = RateControl.CRF
    rate: int | None = None
    preset: str | None = None
    pix_fmt: str = "yuv420p"
    roi_map: str | None = None
    extra_args: tuple[str, ...] = ()

    @property
    def capabilities(self) -> CodecCapabilities:
        """The capabilities of the codec this request names."""
        return codec(self.codec_name)

    @property
    def is_roi_arm(self) -> bool:
        """Whether this request carries a region map."""
        return self.roi_map is not None

    def validate(self) -> None:
        """Raise if this request asks for something the codec will not honour.

        Raises:
            CodecConstraintError: Naming the offending knob and the legal values.
        """
        caps = self.capabilities

        if self.pix_fmt not in caps.pix_fmts:
            raise CodecConstraintError(caps.name, "pix_fmt", self.pix_fmt, caps.pix_fmts)

        if self.rate_control not in caps.rate_controls:
            raise CodecConstraintError(
                caps.name,
                "rate_control",
                self.rate_control.value,
                sorted(mode.value for mode in caps.rate_controls),
            )

        if self.rate_control is RateControl.LOSSLESS and caps.lossless_rate is None:
            raise CodecConstraintError(
                caps.name, "rate_control", "lossless", sorted(mode.value for mode in caps.rate_controls)
            )

        if self.rate_control is not RateControl.LOSSLESS and self.rate is None:
            raise CodecConstraintError(caps.name, "rate", None, ["an integer"])

        if self.is_roi_arm:
            if not caps.supports_roi:
                raise CodecConstraintError(
                    caps.name,
                    "roi_map",
                    self.roi_map,
                    sorted(other.name for other in CODECS.values() if other.supports_roi),
                )
            if self.rate_control is not RateControl.QP:
                # A delta-QP map offsets a base QP. Under CRF or bitrate the
                # encoder is free to move that base around, so the offsets stop
                # meaning anything fixed and the arm is not interpretable.
                raise CodecConstraintError(
                    caps.name, "rate_control for an ROI arm", self.rate_control.value, ["qp"]
                )

    def replace_rate(self, rate: int) -> EncodeRequest:
        """This request at a different rate, for walking a quality ladder."""
        return EncodeRequest(
            codec_name=self.codec_name,
            rate_control=self.rate_control,
            rate=rate,
            preset=self.preset,
            pix_fmt=self.pix_fmt,
            roi_map=self.roi_map,
            extra_args=self.extra_args,
        )

    def without_roi(self) -> EncodeRequest:
        """The matched baseline for this ROI arm: identical but for the map.

        The *only* legitimate baseline for a region-of-interest arm. Building one
        by hand — picking a comparable rate, or reusing an encode made with
        different rate control — reintroduces exactly the confound that made a
        prior project's ROI table meaningless.
        """
        return EncodeRequest(
            codec_name=self.codec_name,
            rate_control=self.rate_control,
            rate=self.rate,
            preset=self.preset,
            pix_fmt=self.pix_fmt,
            roi_map=None,
            extra_args=self.extra_args,
        )

    @classmethod
    def lossless(cls, codec_name: str = "avc", *, preset: str | None = "veryfast") -> EncodeRequest:
        """A mathematically lossless encode, for clip extraction and ceilings."""
        caps = codec(codec_name)
        return cls(
            codec_name=codec_name,
            rate_control=RateControl.LOSSLESS,
            rate=caps.lossless_rate,
            preset=preset,
            pix_fmt="yuv420p",
        )


# --------------------------------------------------------------------------
# Comparison hygiene
# --------------------------------------------------------------------------


def assert_matched_rate_control(requests: Iterable[EncodeRequest]) -> None:
    """Raise unless every arm of a comparison uses identical rate control.

    The failure this prevents is specific and has happened: region-of-interest
    arms encoded at fixed QP were compared against baselines encoded at a target
    bitrate, using an encoder whose bitrate mode overshoots by 30–45%. The
    resulting "24.9–40.2% fewer bits" was the overshoot, not the region control.
    Once both arms were encoded the same way the saving vanished, and the honest
    claim turned out to be about foreground quality at equal bitrate instead.

    Call this before reporting any cross-arm number, not after.

    Raises:
        CodecConstraintError: If the arms disagree on rate control or preset.
    """
    arms = list(requests)
    if len(arms) < 2:
        return

    modes = {arm.rate_control for arm in arms}
    if len(modes) > 1:
        raise CodecConstraintError(
            "comparison",
            "rate_control across arms",
            sorted(mode.value for mode in modes),
            ["a single mode shared by every arm"],
        )

    presets = {arm.preset for arm in arms}
    if len(presets) > 1:
        raise CodecConstraintError(
            "comparison",
            "preset across arms",
            sorted(str(preset) for preset in presets),
            ["a single preset shared by every arm"],
        )


def describe_ladder() -> str:
    """A readable table of the codec ladder and its region-of-interest story."""
    width = max(len(item.name) for item in CODECS.values())
    lines = ["codec ladder:"]
    for item in CODECS.values():
        verified = "verified" if item.roi_is_verified else "unverified" if item.supports_roi else "none"
        lines.append(
            f"  {item.name.ljust(width)}  {item.standard:<12}  "
            f"{item.driver.value:<7} {item.invocation:<14} roi={item.roi.value} ({verified})"
        )
    return "\n".join(lines)


#: Codecs whose ROI mechanism is confirmed to exist as a real delta-QP map.
#: The others need the pixel-domain arm to be comparable at all.
ROI_CAPABLE: Final[frozenset[str]] = frozenset(
    item.name for item in CODECS.values() if item.roi_is_verified
)

#: Fields that must match between an ROI arm and its baseline.
MATCHED_FIELDS: Final[tuple[str, ...]] = (
    "codec_name",
    "rate_control",
    "rate",
    "preset",
    "pix_fmt",
)
