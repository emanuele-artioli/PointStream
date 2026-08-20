"""The object stream: appearance, motion, and how densely motion is sent.

This is the abstraction the rest of the system is organised around. **Every
salient object is described by three independent choices** — what it looks like
(established once), how that appearance evolves, and how densely that evolution
is actually transmitted versus interpolated. The arrangement this replaces
treated a skeleton, a JPEG crop and an encoded clip as three unrelated special
cases with three unrelated code paths; they are one thing with three settings.

Two consequences are worth stating because they are what make the design work:

**A skeleton is a motion representation, not a privileged one.** Sending
per-frame keypoints is the same act as sending motion parameters — both describe
how an already-established appearance should be transformed. So an object
without a skeleton is a configuration, not a redesign.

**A compressed image is an appearance vector.** A JPEG is a two-dimensional
appearance vector encoded with a DCT; a diffusion latent and a CLIP embedding
are the same kind of thing at different sizes. Its two degradation knobs — JPEG
quality and downscale factor — are deliberately separate fields, because they
are *not* equivalent: one discards high-frequency detail through quantization,
the other through resolution, and which serves generative reconstruction better
is an open question this module has to be able to express.

**Sparse trajectories, not dense flow.** The flow-conditioned animation
literature splits into models consuming dense per-pixel flow and models
consuming a handful of tracked points which they expand to dense motion
themselves. Dense flow costs the same to transmit as classical block motion
vectors, which defeats the entire purpose; sparse trajectories are the same
order of size as a skeleton — 16–100 motion parameters in the generative
face-coding literature against our 17–133 keypoints — and the sparse-to-dense
expansion is exactly the generative decoder's job. `SparseTrajectories` refuses
point counts above `MAX_SPARSE_POINTS` for that reason.

**Not every pairing is decodable.** An appearance carried as Canny edges cannot
be turned back into colour by anything, so structure-only representations belong
on the conditioning side and never as the appearance carrier. More generally a
pairing is usable only if some registered generator declares acceptance of both
halves — `assert_decodable` is that check, and it names what would have worked.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Protocol

from src.contracts.capabilities import (
    ALL_APPEARANCE,
    ALL_MOTION,
    APPEARANCE_COMPRESSED_IMAGE,
    APPEARANCE_DIFFUSION_LATENT,
    APPEARANCE_IMAGE_EMBEDDING,
    MOTION_ENCODED_VIDEO,
    MOTION_KEYPOINTS,
    MOTION_VECTORS,
    NS_APPEARANCE,
    NS_MOTION,
)
from src.contracts.codecs import EncodeRequest
from src.contracts.errors import ConfigValueError, UndecodableStreamError
from src.contracts.keypoints import KeypointSchema, wire_cost
from src.contracts.registry import BackendSpec, Registry

# --------------------------------------------------------------------------
# Wire cost — the currency every component is judged in
# --------------------------------------------------------------------------

#: Above this many tracked points a "sparse trajectory" set costs the same order
#: as classical block motion vectors, at which point the representation has
#: stopped being the cheap thing the design argues for. The published sparse
#: schemes sit two orders of magnitude below this; the ceiling exists to catch a
#: config that quietly turned sparse trajectories into dense flow.
MAX_SPARSE_POINTS: Final = 1024


@dataclass(frozen=True)
class WireCost:
    """What one piece of a stream costs to transmit.

    Args:
        values: Scalars on the wire, where they can be counted exactly. `None`
            for entropy-coded payloads, whose value count is not the thing that
            determines their size.
        byte_count: Bytes on the wire. `None` means *not derivable from the
            configuration* — a JPEG's size depends on the pixels, not only on
            the quality setting.
        exact: Whether the numbers above follow from declared parameters and a
            declared quantization, rather than from a model of the encoder.
        basis: One line saying how the number was arrived at, so a total that
            mixes measured and derived parts can be read back.

    `byte_count` deliberately stays `None` rather than carrying an invented
    estimate. An entropy-coded payload's size is a measurement; a plausible
    model of it would produce exactly the kind of clean, believable, fictional
    ablation table this project has already been burned by once. Fill it in from
    the encoder's actual output via `measured_bytes` on the descriptor.
    """

    values: int | None = None
    byte_count: int | None = None
    exact: bool = True
    basis: str = ""

    def __add__(self, other: WireCost) -> WireCost:
        return WireCost(
            values=_add_optional(self.values, other.values),
            byte_count=_add_optional(self.byte_count, other.byte_count),
            exact=self.exact and other.exact,
            basis=" + ".join(part for part in (self.basis, other.basis) if part),
        )

    def scaled(self, factor: int) -> WireCost:
        """This cost repeated `factor` times — a per-frame cost over a span."""
        if factor < 0:
            raise ValueError(f"WireCost.scaled needs a non-negative factor, got {factor}.")
        return WireCost(
            values=None if self.values is None else self.values * factor,
            byte_count=None if self.byte_count is None else self.byte_count * factor,
            exact=self.exact,
            basis=f"{factor}x({self.basis})" if self.basis else "",
        )

    @property
    def is_stated(self) -> bool:
        """Whether a byte figure exists at all, measured or derived."""
        return self.byte_count is not None


def _add_optional(left: int | None, right: int | None) -> int | None:
    """Sum two counts, propagating "unknown" rather than treating it as zero.

    Silently reading a missing measurement as zero is how a payload total comes
    out smaller than it really is, which is the one direction of error this
    project cannot afford: every component is ranked on payload.
    """
    if left is None or right is None:
        return None
    return left + right


# --------------------------------------------------------------------------
# Appearance representations — what an object looks like, established once
# --------------------------------------------------------------------------


class Representation(Protocol):
    """Anything that goes on the wire as part of an object stream."""

    @property
    def kind(self) -> str:
        """The capability vocabulary value this representation is registered as."""

    def cost(self) -> WireCost:
        """What one transmission of this representation costs."""


@dataclass(frozen=True)
class CompressedImage:
    """The object crop as a compressed image.

    Args:
        width: Source crop width in pixels, before downscaling.
        height: Source crop height in pixels, before downscaling.
        quality: JPEG quality, 1–100. Discards high-frequency detail through
            quantization.
        downscale: Linear scale factor in (0, 1]. Discards it through
            resolution instead.
        measured_bytes: The encoded size, once something has actually encoded
            it. Required for this representation to contribute a byte figure.

    `quality` and `downscale` are separate knobs because they are separate
    mechanisms. Collapsing them into one "degradation" scalar — the tempting
    simplification — would make the comparison the paper wants to run
    unexpressible.
    """

    width: int
    height: int
    quality: int = 85
    downscale: float = 1.0
    measured_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"CompressedImage needs a positive crop size, got {self.width}x{self.height}."
            )
        if not 1 <= self.quality <= 100:
            raise ValueError(f"CompressedImage quality must be 1-100, got {self.quality}.")
        if not 0.0 < self.downscale <= 1.0:
            raise ValueError(
                f"CompressedImage downscale must be in (0, 1], got {self.downscale}. "
                f"Upscaling an object crop adds no information and costs bytes."
            )
        if self.measured_bytes is not None and self.measured_bytes < 0:
            raise ValueError(f"CompressedImage measured_bytes must be >= 0, got {self.measured_bytes}.")

    @property
    def kind(self) -> str:
        return APPEARANCE_COMPRESSED_IMAGE

    @property
    def transmitted_size(self) -> tuple[int, int]:
        """Pixel dimensions actually sent, after `downscale`."""
        return (
            max(1, round(self.width * self.downscale)),
            max(1, round(self.height * self.downscale)),
        )

    def cost(self) -> WireCost:
        sent_width, sent_height = self.transmitted_size
        pixels = sent_width * sent_height
        if self.measured_bytes is None:
            return WireCost(
                values=pixels * 3,
                byte_count=None,
                exact=False,
                basis=(
                    f"jpeg q{self.quality} at {sent_width}x{sent_height}; "
                    f"size is data-dependent and unmeasured"
                ),
            )
        return WireCost(
            values=pixels * 3,
            byte_count=self.measured_bytes,
            exact=True,
            basis=f"jpeg q{self.quality} at {sent_width}x{sent_height}, measured",
        )


@dataclass(frozen=True)
class DiffusionLatent:
    """The crop encoded into a diffusion model's own VAE latent space.

    Args:
        channels: Latent channel count.
        height: Latent grid height.
        width: Latent grid width.
        bytes_per_value: Declared quantization of each latent scalar. Two is
            half precision, which is what the models consume anyway.
        measured_bytes: Size after any further entropy coding, if applied.

    The size of a raw latent *is* derivable, so this representation states its
    cost exactly rather than needing a measurement. That is a genuine advantage
    over the compressed-image route when comparing them.
    """

    channels: int
    height: int
    width: int
    bytes_per_value: int = 2
    measured_bytes: int | None = None

    def __post_init__(self) -> None:
        if min(self.channels, self.height, self.width) <= 0:
            raise ValueError(
                f"DiffusionLatent needs positive dimensions, got "
                f"{self.channels}x{self.height}x{self.width}."
            )
        if self.bytes_per_value <= 0:
            raise ValueError(
                f"DiffusionLatent bytes_per_value must be positive, got {self.bytes_per_value}."
            )

    @property
    def kind(self) -> str:
        return APPEARANCE_DIFFUSION_LATENT

    def cost(self) -> WireCost:
        values = self.channels * self.height * self.width
        if self.measured_bytes is not None:
            return WireCost(
                values=values,
                byte_count=self.measured_bytes,
                exact=True,
                basis=f"latent {self.channels}x{self.height}x{self.width}, measured",
            )
        return WireCost(
            values=values,
            byte_count=values * self.bytes_per_value,
            exact=True,
            basis=(
                f"latent {self.channels}x{self.height}x{self.width} "
                f"at {self.bytes_per_value}B/value"
            ),
        )


@dataclass(frozen=True)
class ImageEmbedding:
    """A CLIP/IP-Adapter-style appearance vector.

    Args:
        dimensions: Length of one embedding vector.
        tokens: How many such vectors. IP-Adapter-style conditioners use more
            than one.
        bytes_per_value: Declared quantization per component.

    The most compact option that still carries colour and texture, and the one
    whose cost is least ambiguous — a few KB, statable exactly.
    """

    dimensions: int
    tokens: int = 1
    bytes_per_value: int = 2

    def __post_init__(self) -> None:
        if self.dimensions <= 0 or self.tokens <= 0:
            raise ValueError(
                f"ImageEmbedding needs positive dimensions and tokens, got "
                f"{self.dimensions} and {self.tokens}."
            )
        if self.bytes_per_value <= 0:
            raise ValueError(
                f"ImageEmbedding bytes_per_value must be positive, got {self.bytes_per_value}."
            )

    @property
    def kind(self) -> str:
        return APPEARANCE_IMAGE_EMBEDDING

    def cost(self) -> WireCost:
        values = self.dimensions * self.tokens
        return WireCost(
            values=values,
            byte_count=values * self.bytes_per_value,
            exact=True,
            basis=f"embedding {self.tokens}x{self.dimensions} at {self.bytes_per_value}B/value",
        )


AppearanceRepresentation = CompressedImage | DiffusionLatent | ImageEmbedding


# --------------------------------------------------------------------------
# Motion representations — how that appearance evolves
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class KeypointMotion:
    """A per-frame pose vector under a declared schema.

    Args:
        schema: The **wire** schema, not the canonical internal one. Sending 133
            joints to a conditioner that reads 18 is wasted payload, and payload
            is the ranking currency.
        values_per_joint: Three by default — x, y, confidence.
        bytes_per_value: Declared quantization per scalar.

    Applies to object classes with a stable skeleton. `DomainProfile` is what
    knows whether a given class has one; pairing this with a rigid class is
    rejected there rather than here.
    """

    schema: KeypointSchema
    values_per_joint: int = 3
    bytes_per_value: int = 2

    def __post_init__(self) -> None:
        if self.values_per_joint <= 0:
            raise ValueError(
                f"KeypointMotion values_per_joint must be positive, got {self.values_per_joint}."
            )
        if self.bytes_per_value <= 0:
            raise ValueError(
                f"KeypointMotion bytes_per_value must be positive, got {self.bytes_per_value}."
            )

    @property
    def kind(self) -> str:
        return MOTION_KEYPOINTS

    def cost(self) -> WireCost:
        values = wire_cost(self.schema, values_per_joint=self.values_per_joint)
        return WireCost(
            values=values,
            byte_count=values * self.bytes_per_value,
            exact=True,
            basis=(
                f"{len(self.schema)} joints x {self.values_per_joint} values "
                f"at {self.bytes_per_value}B ({self.schema.name})"
            ),
        )


@dataclass(frozen=True)
class SparseTrajectories:
    """A handful of tracked points, expanded to dense motion by the decoder.

    Args:
        point_count: Tracked points per object. Published sparse schemes use
            tens; `MAX_SPARSE_POINTS` is the ceiling past which this stops being
            the cheap representation the design argues for.
        values_per_point: Two by default — dx, dy. Three where a visibility or
            confidence channel is carried too.
        bytes_per_value: Declared quantization per scalar.

    Registered under the `motion-vectors` capability value, which is the name
    the vocabulary already uses for "class-agnostic motion field". The class is
    named for what is actually transmitted: **sparse trajectories, never dense
    flow**. Dense per-pixel flow costs what block motion vectors cost, so
    transmitting it would defeat the point of the whole system; the models that
    matter here (MOFA-Video, DragNUWA, Tora) consume sparse points and do the
    expansion themselves.
    """

    point_count: int
    values_per_point: int = 2
    bytes_per_value: int = 2

    def __post_init__(self) -> None:
        if self.point_count <= 0:
            raise ValueError(
                f"SparseTrajectories needs at least one point, got {self.point_count}."
            )
        if self.point_count > MAX_SPARSE_POINTS:
            raise ValueError(
                f"SparseTrajectories was asked for {self.point_count} points, above the "
                f"{MAX_SPARSE_POINTS} ceiling. At that density the payload is the same "
                f"order as classical block motion vectors, which is the cost this "
                f"representation exists to avoid — that is dense flow wearing a sparse "
                f"name. Use motion representation {MOTION_ENCODED_VIDEO!r} if a dense "
                f"per-pixel answer is genuinely wanted."
            )
        if self.values_per_point <= 0:
            raise ValueError(
                f"SparseTrajectories values_per_point must be positive, got "
                f"{self.values_per_point}."
            )
        if self.bytes_per_value <= 0:
            raise ValueError(
                f"SparseTrajectories bytes_per_value must be positive, got {self.bytes_per_value}."
            )

    @property
    def kind(self) -> str:
        return MOTION_VECTORS

    def cost(self) -> WireCost:
        values = self.point_count * self.values_per_point
        return WireCost(
            values=values,
            byte_count=values * self.bytes_per_value,
            exact=True,
            basis=(
                f"{self.point_count} trajectories x {self.values_per_point} values "
                f"at {self.bytes_per_value}B"
            ),
        )


@dataclass(frozen=True)
class EncodedVideoMotion:
    """The object crop as a literal video after its appearance keyframe.

    Args:
        request: How the per-object clip is encoded. Carrying the real
            `EncodeRequest` rather than a codec name means this arm is subject
            to the same constraint checks and the same matched-rate rule as
            every other encode — which is what makes it a fair baseline instead
            of a differently-configured strawman.
        width: Crop width fed to the encoder.
        height: Crop height fed to the encoder.
        measured_bytes_per_frame: Encoded size per frame, once measured.

    The classical codec answer applied per object — MPEG-4 Part 2 object coding
    is the ancestor worth citing — and the baseline the generative motion
    representations have to beat.
    """

    request: EncodeRequest
    width: int
    height: int
    measured_bytes_per_frame: int | None = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"EncodedVideoMotion needs a positive crop size, got {self.width}x{self.height}."
            )
        if self.measured_bytes_per_frame is not None and self.measured_bytes_per_frame < 0:
            raise ValueError(
                f"EncodedVideoMotion measured_bytes_per_frame must be >= 0, got "
                f"{self.measured_bytes_per_frame}."
            )

    @property
    def kind(self) -> str:
        return MOTION_ENCODED_VIDEO

    def cost(self) -> WireCost:
        where = f"{self.request.codec_name} at {self.width}x{self.height}"
        if self.measured_bytes_per_frame is None:
            return WireCost(
                values=None,
                byte_count=None,
                exact=False,
                basis=f"{where}; inter-frame size is data-dependent and unmeasured",
            )
        return WireCost(
            values=None,
            byte_count=self.measured_bytes_per_frame,
            exact=True,
            basis=f"{where}, measured",
        )


MotionRepresentation = KeypointMotion | SparseTrajectories | EncodedVideoMotion


# --------------------------------------------------------------------------
# Temporal policy — how densely motion is actually sent
# --------------------------------------------------------------------------


class FrameAction(str, Enum):
    """What happens to one object on one frame.

    This is the unit that travels **in the payload**. The encoder honours a
    decision by skipping stages and the reconstruction side honours the
    identical decision by interpolating; recomputing it independently on each
    side from the config is how the two drift apart, and that drift has already
    produced one real bug.
    """

    FULL = "full"
    """Perception, transmission and generation all run on this frame."""

    TRANSMIT_ONLY = "transmit-only"
    """Motion is perceived and transmitted, but the generative model does not
    run — the client interpolates the generated output instead."""

    INTERPOLATE = "interpolate"
    """Nothing is transmitted. Both sides interpolate between the anchor and the
    target keyframe."""

    HOLD = "hold"
    """Nothing is transmitted and there is no later keyframe to reach for, so
    the anchor's state is held."""

    @property
    def is_transmitted(self) -> bool:
        """Whether this frame puts motion on the wire."""
        return self in (FrameAction.FULL, FrameAction.TRANSMIT_ONLY)


@dataclass(frozen=True)
class Sparsity:
    """One level of temporal sparsity.

    Args:
        stride: Act every `stride` frames. One means every frame.
        threshold: Act when accumulated motion since the last action exceeds
            this. `None` means stride alone governs.
        threshold_relative_to_scene_motion: Whether `threshold` is a multiple of
            the clip's mean measured motion rather than an absolute figure.
            The arrangement this replaces used one hard-coded constant for every
            clip, so a slow rally and a fast exchange got identical keyframe
            density.

    `stride` and `threshold` compose: either firing acts. A stride alongside a
    threshold is the useful pairing — the threshold adds keyframes where motion
    is fast, the stride caps how far reconstruction can drift where it is slow.
    """

    stride: int = 1
    threshold: float | None = None
    threshold_relative_to_scene_motion: bool = False

    def __post_init__(self) -> None:
        if self.stride < 1:
            raise ValueError(f"Sparsity stride must be >= 1, got {self.stride}.")
        if self.threshold is not None and self.threshold <= 0:
            raise ValueError(f"Sparsity threshold must be positive, got {self.threshold}.")
        if self.threshold_relative_to_scene_motion and self.threshold is None:
            raise ValueError(
                "Sparsity was told its threshold is relative to scene motion, but no "
                "threshold was given."
            )

    @property
    def is_dense(self) -> bool:
        """Whether this level acts on every frame."""
        return self.stride == 1 and self.threshold is None

    @property
    def is_adaptive(self) -> bool:
        """Whether this level needs measured per-frame motion to decide."""
        return self.threshold is not None

    def describe(self) -> str:
        if self.is_dense:
            return "every frame"
        parts = [] if self.stride == 1 else [f"every {self.stride} frames"]
        if self.threshold is not None:
            unit = "x scene motion" if self.threshold_relative_to_scene_motion else " units"
            parts.append(f"or when motion exceeds {self.threshold:g}{unit}")
        return " ".join(parts)


@dataclass(frozen=True)
class FrameDecision:
    """One object's fate on one frame, as it appears in the payload.

    Args:
        frame_index: Absolute frame index.
        object_id: Which object this decision is about.
        action: What happens.
        anchor: The last transmitted frame this frame derives from. Required for
            `INTERPOLATE` and `HOLD`, and must be `None` for the transmitted
            actions, which derive from nothing.
        target: The next transmitted frame an `INTERPOLATE` reaches toward.
    """

    frame_index: int
    object_id: str
    action: FrameAction
    anchor: int | None = None
    target: int | None = None

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError(f"FrameDecision frame_index must be >= 0, got {self.frame_index}.")


@dataclass(frozen=True)
class TemporalSchedule:
    """Every per-frame decision for a span, plus where the span is cut.

    Args:
        decisions: The decisions, in any order. Uniqueness of
            `(object_id, frame_index)` is checked by `validate`.
        discontinuities: Frame indices that *start* a new span — a scene cut, a
            track lost and re-acquired. Nothing may be interpolated or held
            across one: on the far side of a cut the anchor describes different
            content entirely, so an interpolation across it is not a slightly
            worse prediction but a confidently wrong one.
    """

    decisions: tuple[FrameDecision, ...] = ()
    discontinuities: frozenset[int] = frozenset()

    def for_object(self, object_id: str) -> tuple[FrameDecision, ...]:
        """This object's decisions, in frame order."""
        return tuple(
            sorted(
                (item for item in self.decisions if item.object_id == object_id),
                key=lambda item: item.frame_index,
            )
        )

    def transmitted_frames(self, object_id: str | None = None) -> tuple[int, ...]:
        """Frames that put motion on the wire, in order.

        This is the count that multiplies an object's per-frame motion cost, so
        it is the number the whole temporal axis is judged by.
        """
        chosen = (
            self.decisions
            if object_id is None
            else tuple(item for item in self.decisions if item.object_id == object_id)
        )
        return tuple(sorted(item.frame_index for item in chosen if item.action.is_transmitted))

    def crosses_discontinuity(self, start: int, end: int) -> bool:
        """Whether a cut falls in `(start, end]`.

        Half-open at the start because a cut *at* the anchor makes the anchor
        the first frame of the new span, which is a legitimate thing to
        interpolate from.
        """
        return any(start < cut <= end for cut in self.discontinuities)

    def validate(self, *, path: str = "temporal.schedule") -> None:
        """Raise unless every decision is self-consistent and legal.

        Raises:
            ConfigValueError: Naming the offending frame and what is wrong with
                it. Schedules can be written by hand for an ablation, so this
                cannot assume `TemporalPolicy.plan` produced them.
        """
        seen: set[tuple[str, int]] = set()
        for item in sorted(self.decisions, key=lambda entry: (entry.object_id, entry.frame_index)):
            key = (item.object_id, item.frame_index)
            if key in seen:
                raise ConfigValueError(
                    f"{path}[{item.object_id}:{item.frame_index}]",
                    "two decisions for the same object on the same frame.",
                )
            seen.add(key)

            where = f"{path}[{item.object_id}:{item.frame_index}]"
            if item.action.is_transmitted:
                if item.anchor is not None or item.target is not None:
                    raise ConfigValueError(
                        where,
                        f"a {item.action.value} frame is transmitted outright, so it must "
                        f"not name an anchor or a target.",
                    )
                continue

            if item.anchor is None:
                raise ConfigValueError(
                    where,
                    f"a {item.action.value} frame derives from an earlier transmitted "
                    f"frame and must name it as its anchor.",
                )
            if item.anchor >= item.frame_index:
                raise ConfigValueError(
                    where,
                    f"anchor {item.anchor} is not before this frame.",
                )

            if item.action is FrameAction.HOLD:
                if item.target is not None:
                    raise ConfigValueError(
                        where, "a hold reaches toward nothing, so it must not name a target."
                    )
                span_end = item.frame_index
            else:
                if item.target is None:
                    raise ConfigValueError(
                        where, "an interpolate frame must name the keyframe it reaches toward."
                    )
                if item.target <= item.frame_index:
                    raise ConfigValueError(
                        where, f"target {item.target} is not after this frame."
                    )
                span_end = item.target

            if self.crosses_discontinuity(item.anchor, span_end):
                cuts = sorted(
                    cut for cut in self.discontinuities if item.anchor < cut <= span_end
                )
                raise ConfigValueError(
                    where,
                    f"{item.action.value} spans frames {item.anchor}-{span_end}, which "
                    f"crosses a discontinuity at {cuts}. Across a cut the anchor describes "
                    f"different content, so the prediction is not merely worse but wrong. "
                    f"Transmit a keyframe at {cuts[0]} instead.",
                )

    def describe(self) -> str:
        """A readable dump, one line per decision."""
        if not self.decisions:
            return "temporal schedule: (empty)"
        cuts = ", ".join(str(cut) for cut in sorted(self.discontinuities)) or "none"
        lines = [f"temporal schedule: {len(self.decisions)} decisions, cuts at {cuts}"]
        for item in sorted(self.decisions, key=lambda entry: (entry.object_id, entry.frame_index)):
            tail = ""
            if item.anchor is not None:
                tail = f"  <- {item.anchor}"
                if item.target is not None:
                    tail += f"..{item.target}"
            lines.append(
                f"  frame {item.frame_index:>5}  {item.object_id:<12} {item.action.value}{tail}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class TemporalPolicy:
    """The three composing levels of temporal sparsity, and how to apply them.

    Args:
        metadata: When motion is transmitted. Between transmissions the
            reconstruction interpolates.
        generation: When the generative model runs, counted in **keyframes**
            rather than frames — the model runs at keyframes and the client
            interpolates the generated *output* between them. A frame carrying
            no transmitted motion has no fresh conditioning to generate from, so
            generation is a subset of transmission rather than a third schedule
            that could disagree with it.
        pipeline: When perception — detection, pose, segmentation — runs at all.
            This is the level that actually saves encode time, and the one that
            was missing: the other two govern only what is transmitted and what
            is generated, while perception ran on every frame regardless.
        preroll_frames: Frames at the start of a span kept residual-only, before
            temporal generation has enough history to run.

    The levels compose in one direction: a frame whose perception was skipped
    has no motion to transmit, so pipeline sparsity bounds metadata sparsity
    rather than merely sitting alongside it.
    """

    metadata: Sparsity = Sparsity()
    generation: Sparsity = Sparsity()
    pipeline: Sparsity = Sparsity()
    preroll_frames: int = 0

    def __post_init__(self) -> None:
        if self.preroll_frames < 0:
            raise ValueError(
                f"TemporalPolicy preroll_frames must be >= 0, got {self.preroll_frames}."
            )

    @property
    def is_dense(self) -> bool:
        """Whether every frame gets the full pipeline — the maximum-cost corner."""
        return (
            self.metadata.is_dense
            and self.generation.is_dense
            and self.pipeline.is_dense
            and self.preroll_frames == 0
        )

    @property
    def needs_measured_motion(self) -> bool:
        """Whether `plan` will require per-frame motion magnitudes."""
        return (
            self.metadata.is_adaptive
            or self.generation.is_adaptive
            or self.pipeline.is_adaptive
        )

    def plan(
        self,
        *,
        frame_count: int,
        object_id: str = "object",
        motion: Sequence[float] | None = None,
        discontinuities: Iterable[int] = (),
        path: str = "temporal-policy",
    ) -> TemporalSchedule:
        """Turn this policy into the per-frame decisions that go in the payload.

        Args:
            frame_count: Frames in the span, indexed from zero.
            object_id: Which object these decisions are about.
            motion: Per-frame motion magnitude, required when any level is
                adaptive. Units are the caller's; thresholds are compared
                against them directly, or against their mean when the threshold
                was declared relative.
            discontinuities: Frame indices starting a new span.
            path: Config path used in error messages.

        Returns:
            A schedule whose first frame of every span is transmitted, and in
            which nothing interpolates or holds across a cut.

        Raises:
            ConfigValueError: If an adaptive level was configured without the
                motion measurements it needs, or the motion series is the wrong
                length.
        """
        if frame_count < 0:
            raise ValueError(f"plan needs a non-negative frame_count, got {frame_count}.")
        cuts = frozenset(int(cut) for cut in discontinuities)

        if self.needs_measured_motion and motion is None:
            raise ConfigValueError(
                path,
                "a motion-adaptive sparsity threshold was configured, but no measured "
                "per-frame motion was supplied. The threshold is what makes keyframe "
                "density follow the content instead of a constant, so planning without "
                "it would quietly fall back to the fixed-stride behaviour it replaces.",
            )
        if motion is not None and len(motion) != frame_count:
            raise ConfigValueError(
                path,
                f"motion has {len(motion)} entries for {frame_count} frames.",
            )

        scene_motion = (
            sum(motion) / len(motion) if motion and len(motion) > 0 else 0.0
        )

        transmitted = self._transmitted_flags(frame_count, motion, cuts, scene_motion)
        generated = self._generated_flags(frame_count, cuts, transmitted)

        decisions: list[FrameDecision] = []
        for index in range(frame_count):
            if transmitted[index]:
                action = FrameAction.FULL if generated[index] else FrameAction.TRANSMIT_ONLY
                decisions.append(FrameDecision(index, object_id, action))
                continue

            anchor = max(
                position for position in range(index) if transmitted[position]
            )
            target = next(
                (
                    position
                    for position in range(index + 1, frame_count)
                    if transmitted[position] and not _cut_between(cuts, index, position)
                ),
                None,
            )
            if target is None:
                decisions.append(
                    FrameDecision(index, object_id, FrameAction.HOLD, anchor=anchor)
                )
            else:
                decisions.append(
                    FrameDecision(
                        index, object_id, FrameAction.INTERPOLATE, anchor=anchor, target=target
                    )
                )

        return TemporalSchedule(decisions=tuple(decisions), discontinuities=cuts)

    def _transmitted_flags(
        self,
        frame_count: int,
        motion: Sequence[float] | None,
        cuts: frozenset[int],
        scene_motion: float,
    ) -> list[bool]:
        """Which frames put motion on the wire.

        Perception bounds transmission: a frame the pipeline skipped has nothing
        to send, whatever the metadata level would have preferred.
        """
        effective = self._effective_threshold(self.metadata, scene_motion)
        flags = [False] * frame_count
        accumulated = 0.0
        span_start = 0
        for index in range(frame_count):
            if index == 0 or index in cuts:
                span_start = index
                flags[index] = True
                accumulated = 0.0
                continue

            offset = index - span_start
            if motion is not None:
                accumulated += motion[index]

            perceived = self.pipeline.stride == 1 or offset % self.pipeline.stride == 0
            if not perceived:
                continue

            if self.metadata.is_dense:
                send = True
            else:
                send = self.metadata.stride > 1 and offset % self.metadata.stride == 0
                if effective is not None and accumulated >= effective:
                    send = True

            flags[index] = send
            if send:
                accumulated = 0.0
        return flags

    def _generated_flags(
        self, frame_count: int, cuts: frozenset[int], transmitted: Sequence[bool]
    ) -> list[bool]:
        """Which frames the generative model actually runs on.

        Generation sparsity counts **keyframes, not frames**: the model runs at
        keyframes and the client interpolates the generated output between them.
        A frame with no transmitted motion has no fresh conditioning to generate
        from, so generation is a subset of transmission by construction rather
        than a third independent schedule that could disagree with it.
        """
        flags = [False] * frame_count
        span_start = 0
        keyframe_ordinal = 0
        for index in range(frame_count):
            if index == 0 or index in cuts:
                span_start = index
                keyframe_ordinal = 0
            if not transmitted[index]:
                continue
            if index - span_start < self.preroll_frames:
                continue
            ordinal = keyframe_ordinal
            keyframe_ordinal += 1
            flags[index] = self.generation.stride == 1 or ordinal % self.generation.stride == 0
        return flags

    @staticmethod
    def _effective_threshold(level: Sparsity, scene_motion: float) -> float | None:
        """`level`'s threshold in the caller's motion units."""
        if level.threshold is None:
            return None
        if level.threshold_relative_to_scene_motion:
            return level.threshold * scene_motion
        return level.threshold

    def describe(self) -> str:
        """A readable summary of the three levels."""
        preroll = (
            "" if self.preroll_frames == 0 else f", {self.preroll_frames}-frame residual-only preroll"
        )
        return (
            f"temporal policy: metadata {self.metadata.describe()}; "
            f"generation {self.generation.describe()}; "
            f"pipeline {self.pipeline.describe()}{preroll}"
        )


def _cut_between(cuts: frozenset[int], start: int, end: int) -> bool:
    """Whether a cut falls in `(start, end]`."""
    return any(start < cut <= end for cut in cuts)


#: Every frame fully processed: maximum metadata and compute, minimum prediction
#: error. The "temporal policy off" corner of the lattice.
DENSE_POLICY: Final = TemporalPolicy()


# --------------------------------------------------------------------------
# The pairing constraint
# --------------------------------------------------------------------------


def workable_pairings(generators: Registry[Any]) -> tuple[tuple[str, str], ...]:
    """Every `(appearance, motion)` pair some registered generator can decode.

    Sorted, so an error message listing them reads the same every time.
    """
    pairs: set[tuple[str, str]] = set()
    for spec in generators:
        for appearance in spec.accepted(NS_APPEARANCE):
            for motion in spec.accepted(NS_MOTION):
                pairs.add((appearance, motion))
    return tuple(sorted(pairs))


def decodable_by(
    appearance: str, motion: str, generators: Registry[Any]
) -> list[BackendSpec[Any]]:
    """Registered generators accepting both halves of this pairing."""
    return [
        spec
        for spec in generators
        if spec.accepts(NS_APPEARANCE, appearance) and spec.accepts(NS_MOTION, motion)
    ]


def assert_decodable(
    appearance: str, motion: str, generators: Registry[Any]
) -> BackendSpec[Any]:
    """Return a generator that can decode this pairing, or explain why none can.

    The axes are genuinely independent, but not every combination has a decoder,
    and the implemented set is exactly whatever does. Checking here is what stops
    the design sprawling into combinations nothing implements — the failure
    arrives at config validation with the workable alternatives attached, rather
    than as a shape mismatch inside a diffusion pipeline.

    Raises:
        UndecodableStreamError: Listing every pairing that would have worked.
    """
    if appearance not in ALL_APPEARANCE:
        raise ConfigValueError(
            "object-stream.appearance",
            f"{appearance!r} is not an appearance representation. "
            f"Known: {', '.join(sorted(ALL_APPEARANCE))}.",
        )
    if motion not in ALL_MOTION:
        raise ConfigValueError(
            "object-stream.motion",
            f"{motion!r} is not a motion representation. "
            f"Known: {', '.join(sorted(ALL_MOTION))}.",
        )
    candidates = decodable_by(appearance, motion, generators)
    if not candidates:
        raise UndecodableStreamError(appearance, motion, workable_pairings(generators))
    return candidates[0]


# --------------------------------------------------------------------------
# The stream
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectStream:
    """One object's full transmission choice.

    Args:
        object_id: Stable identity across frames, from the tracking stage.
        object_class: The domain's name for what this is — ``player``,
            ``racket``, ``ball``. `DomainProfile` is what knows whether the
            class has a skeleton.
        appearance: What it looks like, sent once.
        motion: How that appearance evolves, sent per transmitted frame.
        policy: How densely `motion` is actually sent.

    The three representation choices are independent by construction: nothing
    here couples them. The one place they *are* coupled — whether a decoder
    exists for the pair — is checked explicitly by `validate` rather than
    discovered at runtime.
    """

    object_id: str
    object_class: str
    appearance: AppearanceRepresentation
    motion: MotionRepresentation
    policy: TemporalPolicy = DENSE_POLICY

    @property
    def appearance_kind(self) -> str:
        """Capability value for the chosen appearance representation."""
        return self.appearance.kind

    @property
    def motion_kind(self) -> str:
        """Capability value for the chosen motion representation."""
        return self.motion.kind

    def setup_cost(self) -> WireCost:
        """What establishing this object's appearance costs, once."""
        return self.appearance.cost()

    def per_frame_cost(self) -> WireCost:
        """What one *transmitted* frame of motion costs.

        Per transmitted frame, not per frame of video: that distinction is the
        entire point of the temporal policy, and conflating the two would make
        metadata sparsity look free.
        """
        return self.motion.cost()

    def total_cost(self, schedule: TemporalSchedule) -> WireCost:
        """Appearance once, plus motion on every frame the schedule transmits."""
        sent = len(schedule.transmitted_frames(self.object_id))
        return self.setup_cost() + self.per_frame_cost().scaled(sent)

    def plan(
        self,
        *,
        frame_count: int,
        motion: Sequence[float] | None = None,
        discontinuities: Iterable[int] = (),
    ) -> TemporalSchedule:
        """This object's schedule under its own policy."""
        return self.policy.plan(
            frame_count=frame_count,
            object_id=self.object_id,
            motion=motion,
            discontinuities=discontinuities,
            path=f"object-stream[{self.object_id}].temporal-policy",
        )

    def validate(self, generators: Registry[Any]) -> BackendSpec[Any]:
        """Check this stream is decodable, returning a generator that can do it.

        Raises:
            UndecodableStreamError: If nothing registered accepts the pairing.
        """
        return assert_decodable(self.appearance_kind, self.motion_kind, generators)

    def describe(self) -> str:
        """A readable summary, for reading an ablation corner back."""
        setup = self.setup_cost()
        per_frame = self.per_frame_cost()
        return (
            f"{self.object_id} ({self.object_class}): "
            f"{self.appearance_kind} + {self.motion_kind}\n"
            f"  appearance: {_render_cost(setup)}  [{setup.basis}]\n"
            f"  motion:     {_render_cost(per_frame)} per transmitted frame  "
            f"[{per_frame.basis}]\n"
            f"  {self.policy.describe()}"
        )


def _render_cost(cost: WireCost) -> str:
    """A cost as a short human-readable figure."""
    values = "?" if cost.values is None else f"{cost.values} values"
    byte_count = "unmeasured" if cost.byte_count is None else f"{cost.byte_count} B"
    return f"{values}, {byte_count}"


def describe_streams(streams: Iterable[ObjectStream]) -> str:
    """A readable block for every stream in a run."""
    rendered = [stream.describe() for stream in streams]
    if not rendered:
        return "object streams: (none)"
    return "object streams:\n" + "\n".join(rendered)


#: Representation classes by their capability value, so a config naming
#: ``appearance: diffusion-latent`` can find the descriptor to build.
APPEARANCE_TYPES: Final[Mapping[str, type[Any]]] = {
    APPEARANCE_COMPRESSED_IMAGE: CompressedImage,
    APPEARANCE_DIFFUSION_LATENT: DiffusionLatent,
    APPEARANCE_IMAGE_EMBEDDING: ImageEmbedding,
}

MOTION_TYPES: Final[Mapping[str, type[Any]]] = {
    MOTION_KEYPOINTS: KeypointMotion,
    MOTION_VECTORS: SparseTrajectories,
    MOTION_ENCODED_VIDEO: EncodedVideoMotion,
}
