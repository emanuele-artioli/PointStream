"""What a generator is conditioned on, with every input separately typed.

This replaces one parameter that meant five different things. In the
arrangement being retired, `BaseGenAIStrategy.generate` took an argument called
`dense_dwpose_tensor` which carried a rendered **pose** for the OpenPose
backends, a binary **mask** for the segmentation backend, a **canny** edge image
for the canny backend, and a **`(pose, mask)` tuple** for multi-ControlNet — and
the compositor decided which by string-matching the backend's *name*
(`compositor.py:243`). `controlnet_engine.py:590-604` still carries the comments
written while someone worked that out. Object geometry travelled the same way,
as trailing positional arguments the callee could only guess the meaning of.

Three mechanisms replace it, and the point of all three is that a mistake
becomes an error at the call site instead of a wrong picture:

**`ConditioningBundle`** carries every possible input in its own typed, optional
field. A generator that wants a mask reads `bundle.mask`; there is no slot that
means different things to different readers. It is deliberately *not* a dict:
encoder and decoder each construct one, and a typo'd dict key would leave the
two sides silently conditioning on different things — exactly the divergence
the Residual Guarantee cannot survive.

**`ConditioningPlan`** is derived once, from the chosen generator's declared
`requires`, and handed to both sides. This is what makes cross-axis effects
derived rather than string-matched. Today `actor_pipeline.py:88` reads whether
the generator's name contains `"canny-controlnet"` and nulls the pose estimator
from an unrelated module; here the generator declares `requires={canny}` and the
plan derives that canny extraction must run and pose estimation need not.

**`FrameGenerator` / `SequenceGenerator`** are protocols, and temporal capability
is read from a declared capability rather than from `isinstance(strategy,
AnimateAnyoneStrategy)` (`compositor.py:194`), which forced any new temporal
backend to subclass one specific class to be recognised at all.

Tensors are described here, never imported. `contracts` must be importable for
config validation on a machine with no torch, so tensor-shaped values are typed
as a structural `ArrayLike` and their expected layout is declared as data.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Final, Protocol, TypeAlias, runtime_checkable

from src.contracts.capabilities import (
    ALL_CONDITIONS,
    CAP_TEMPORAL_SEQUENCE,
    CONDITION_APPEARANCE,
    CONDITION_CANNY,
    CONDITION_MASK,
    CONDITION_MOTION_FIELD,
    CONDITION_POSE,
)
from src.contracts.errors import MissingConditioningError, UnsupportedCapabilityError
from src.contracts.lattice import CONDITION_SOURCES
from src.contracts.registry import BackendSpec

# --------------------------------------------------------------------------
# Structural stand-ins for the things this layer may not import
# --------------------------------------------------------------------------


@runtime_checkable
class ArrayLike(Protocol):
    """A tensor or array, described by what it exposes rather than by its type.

    `torch.Tensor` and `numpy.ndarray` both satisfy this, and neither has to be
    importable for a config to be validated. Only `shape` and `dtype` are
    required, because those are the only attributes this layer ever inspects —
    everything else about a tensor is the components layer's business.
    """

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dtype(self) -> Any: ...


#: Where a generator should run. A `torch.device`, a string, or whatever the
#: backend understands — opaque here, because naming the type would mean
#: importing torch into a layer that must stay installable without it.
Device: TypeAlias = Any


# --------------------------------------------------------------------------
# Stages and wire items a conditioning requirement implies
# --------------------------------------------------------------------------

TRANSMIT_KEYPOINTS: Final = "keypoints"
TRANSMIT_MASK: Final = "mask"
TRANSMIT_CANNY: Final = "canny-edges"
TRANSMIT_APPEARANCE: Final = "appearance"
TRANSMIT_MOTION_FIELD: Final = "motion-field"

#: Which encoder-side stage each conditioning kind makes necessary. The whole
#: cross-axis question — "does the pose estimator need to run?" — is answered by
#: a lookup here against what the generator declared, and nowhere else.
#:
#: Taken from the stage catalogue rather than restated. Two lists of stage names
#: would be two things to keep in step, and one of them would eventually be
#: wrong — precisely the drift this package exists to prevent.
STAGES_FOR_CONDITION: Final[Mapping[str, tuple[str, ...]]] = {
    condition: (stage_name,) for condition, stage_name in CONDITION_SOURCES.items()
}

#: Every stage a conditioning requirement can ask for. A subset of the full
#: catalogue: stages like transport or metrics are never implied by conditioning.
ALL_STAGES: Final[frozenset[str]] = frozenset(CONDITION_SOURCES.values())

#: What each conditioning kind costs on the wire. The decoder cannot recompute
#: any of these — it never sees the source frame — so a required condition is a
#: transmitted condition. The one input that is *not* here is the previously
#: generated frame, which the client already holds; see `ConditioningBundle`.
TRANSMIT_FOR_CONDITION: Final[Mapping[str, tuple[str, ...]]] = {
    CONDITION_POSE: (TRANSMIT_KEYPOINTS,),
    CONDITION_MASK: (TRANSMIT_MASK,),
    CONDITION_CANNY: (TRANSMIT_CANNY,),
    CONDITION_APPEARANCE: (TRANSMIT_APPEARANCE,),
    CONDITION_MOTION_FIELD: (TRANSMIT_MOTION_FIELD,),
}


# --------------------------------------------------------------------------
# The fields, declared as data
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditioningField:
    """One slot of a `ConditioningBundle`, and what belongs in it.

    Args:
        attribute: Field name on the bundle.
        condition: The `CONDITION_*` kind a generator declares to ask for this,
            or None for inputs no generator has to request because they are not
            transmitted.
        layouts: Human-readable axis orders this field accepts, e.g. ``"CHW"``.
            Declared rather than enforced with torch: the rank is checked, the
            semantics are documented.
        dtype: What the values mean — ``"uint8"``, ``"float32 in [0, 1]"``.
        summary: One line, for listings and error messages.
    """

    attribute: str
    condition: str | None
    layouts: tuple[str, ...]
    dtype: str
    summary: str = ""

    @property
    def ranks(self) -> frozenset[int]:
        """Accepted array ranks, derived from the declared layouts."""
        return frozenset(len(layout) for layout in self.layouts)


FIELDS: Final[Mapping[str, ConditioningField]] = {
    spec.attribute: spec
    for spec in (
        ConditioningField(
            attribute="appearance",
            condition=CONDITION_APPEARANCE,
            layouts=("CHW", "HWC"),
            dtype="uint8",
            summary="The object's established appearance — the reference crop.",
        ),
        ConditioningField(
            attribute="pose",
            condition=CONDITION_POSE,
            layouts=("CHW",),
            dtype="uint8",
            summary="A rendered skeleton condition image, not a raw keypoint vector.",
        ),
        ConditioningField(
            attribute="mask",
            condition=CONDITION_MASK,
            layouts=("HW", "CHW"),
            dtype="uint8, 0 or 255",
            summary="Binary segmentation of the object.",
        ),
        ConditioningField(
            attribute="canny",
            condition=CONDITION_CANNY,
            layouts=("HW", "CHW"),
            dtype="uint8, 0 or 255",
            summary="Edge image. Structure only — it carries no colour, so it "
            "conditions generation and can never be the appearance carrier.",
        ),
        ConditioningField(
            attribute="motion_field",
            condition=CONDITION_MOTION_FIELD,
            layouts=("2HW",),
            dtype="float32 displacement in pixels",
            summary="Per-pixel or per-block displacement, for skeleton-less objects.",
        ),
        ConditioningField(
            attribute="previous_frame",
            condition=None,
            layouts=("CHW", "HWC"),
            dtype="uint8",
            summary="The previously generated frame. Never transmitted — the "
            "client already has it — so no generator declares it as a requirement.",
        ),
    )
}

#: Reverse lookup: which bundle field satisfies a declared conditioning kind.
ATTRIBUTE_FOR_CONDITION: Final[Mapping[str, str]] = {
    spec.condition: spec.attribute for spec in FIELDS.values() if spec.condition is not None
}


def _normalise_condition(name: str) -> str:
    """Resolve a condition kind, accepting the field spelling as well.

    `motion-field` and `motion_field` are the same thing said two ways — the
    capability vocabulary hyphenates, Python attributes cannot. Accepting both
    costs nothing; guessing at a third spelling would not, so anything else is
    rejected with the legal set.
    """
    if name in ALL_CONDITIONS:
        return name
    hyphenated = name.replace("_", "-")
    if hyphenated in ALL_CONDITIONS:
        return hyphenated
    raise ValueError(
        f"Unknown conditioning kind {name!r}. "
        f"Known kinds: {', '.join(sorted(ALL_CONDITIONS))}."
    )


# --------------------------------------------------------------------------
# The bundle
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditioningBundle:
    """Everything one generation call may be conditioned on, each in its own field.

    Every field is optional and independently typed. A generator reads the ones
    it declared and never has to ask what the caller meant by a shared slot.

    Args:
        appearance: The object's established appearance.
        pose: A rendered pose condition image.
        mask: Binary segmentation of the object.
        canny: Edge image.
        motion_field: Displacement field.
        previous_frame: The previously generated frame, for temporal backends.
        bbox: Where the object goes in the full frame, as ``(x1, y1, x2, y2)``.
            Carried here rather than as a trailing positional argument, which is
            how it used to reach the generators.
        frame_index: Index within the clip, for logging and for temporal
            backends that need to know where in a window they are.
        object_id: Stable identity of the object across frames.
    """

    appearance: ArrayLike | None = None
    pose: ArrayLike | None = None
    mask: ArrayLike | None = None
    canny: ArrayLike | None = None
    motion_field: ArrayLike | None = None
    previous_frame: ArrayLike | None = None
    bbox: tuple[int, int, int, int] | None = None
    frame_index: int | None = None
    object_id: str | None = None

    def __post_init__(self) -> None:
        if self.bbox is not None:
            if len(self.bbox) != 4:
                raise ValueError(f"bbox must be (x1, y1, x2, y2); got {self.bbox!r}.")
            x1, y1, x2, y2 = self.bbox
            if x2 <= x1 or y2 <= y1:
                raise ValueError(
                    f"bbox {self.bbox!r} is empty or inverted. A zero-area box "
                    f"reaches the generator as a resize to nothing, which fails "
                    f"much further down as an unrelated shape error."
                )
        if self.frame_index is not None and self.frame_index < 0:
            raise ValueError(f"frame_index must be non-negative; got {self.frame_index}.")

    # -- what is here ------------------------------------------------------

    def present(self) -> frozenset[str]:
        """The conditioning kinds this bundle actually carries."""
        return frozenset(
            spec.condition
            for spec in FIELDS.values()
            if spec.condition is not None and getattr(self, spec.attribute) is not None
        )

    def get(self, condition: str) -> ArrayLike | None:
        """The value for a conditioning kind, by its capability name."""
        return getattr(self, ATTRIBUTE_FOR_CONDITION[_normalise_condition(condition)])

    def require(self, *conditions: str) -> None:
        """Raise unless every named conditioning kind is present.

        Call this first thing in `generate`. The failure it produces names the
        missing input — "needs a mask, got none" — instead of surfacing forty
        lines later as a shape mismatch on a tensor nobody expected to be
        absent, which is how the untyped arrangement failed.

        Raises:
            MissingConditioningError: Listing what is missing and what is here.
            ValueError: If a name is not a known conditioning kind at all.
        """
        wanted = [_normalise_condition(name) for name in conditions]
        missing = [name for name in wanted if self.get(name) is None]
        if missing:
            raise MissingConditioningError(missing, sorted(self.present()))

    def validate_shapes(self) -> None:
        """Check each present field's rank against its declared layout.

        A cheap sanity check that costs no dependency: a `(2, H, W)` motion
        field arriving in the `mask` slot is caught here rather than inside a
        diffusion pipeline. It cannot catch everything — a mask and a canny
        image have the same rank — which is exactly why the fields are separate
        in the first place.

        Raises:
            ValueError: Naming the field, the rank found and the layouts allowed.
        """
        for spec in FIELDS.values():
            value = getattr(self, spec.attribute)
            if value is None:
                continue
            shape = getattr(value, "shape", None)
            if shape is None:
                continue
            rank = len(tuple(shape))
            if rank not in spec.ranks:
                allowed = ", ".join(spec.layouts)
                raise ValueError(
                    f"Conditioning field {spec.attribute!r} has shape {tuple(shape)!r} "
                    f"(rank {rank}), which matches none of its layouts: {allowed}."
                )

    # -- deriving a new one ------------------------------------------------

    def with_fields(self, **changes: Any) -> ConditioningBundle:
        """A copy with some fields replaced.

        The bundle is frozen because encoder and decoder both build one and any
        in-place edit on one side is a divergence. Attaching the previously
        generated frame between calls is the common case.
        """
        unknown = sorted(set(changes) - set(FIELDS) - {"bbox", "frame_index", "object_id"})
        if unknown:
            raise ValueError(
                f"ConditioningBundle has no field(s) {unknown}. "
                f"Fields: {', '.join(sorted(FIELDS))}, bbox, frame_index, object_id."
            )
        return replace(self, **changes)

    def describe(self) -> str:
        """One line naming what is carried, for logs and debug artifacts."""
        carried = sorted(self.present())
        extras = [name for name in ("previous_frame",) if getattr(self, name) is not None]
        who = self.object_id or "?"
        where = f"@{self.frame_index}" if self.frame_index is not None else ""
        return f"conditioning[{who}{where}]: {', '.join([*carried, *extras]) or '(empty)'}"


# --------------------------------------------------------------------------
# Per-call parameters
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationParams:
    """Knobs that vary per call rather than per backend.

    These were trailing `*_override` arguments on `generate`, which meant every
    backend had to accept them whether or not it honoured them, and a caller
    could not tell which did. Here they are one object with one meaning, and a
    backend documents which it reads.

    Args:
        steps: Denoising steps, where the backend has any.
        strength: img2img denoising strength in [0, 1]. 0 returns the init image
            unchanged; 1 ignores it.
        guidance_scale: Classifier-free guidance weight.
        width: Generation width in pixels.
        height: Generation height in pixels.
        init_image: Overrides whatever the backend would otherwise start from.
            Distinct from `ConditioningBundle.appearance`, which says what the
            object looks like; this says where the sampler starts.
        extra: Backend-specific knobs. An escape hatch for one-off experiments,
            deliberately unvalidated and deliberately awkward.
    """

    steps: int | None = None
    strength: float | None = None
    guidance_scale: float | None = None
    width: int | None = None
    height: int | None = None
    init_image: ArrayLike | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.steps is not None and self.steps < 1:
            raise ValueError(f"steps must be at least 1; got {self.steps}.")
        if self.strength is not None and not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"strength must be within [0, 1]; got {self.strength}.")
        for name in ("width", "height"):
            value: int | None = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive; got {value}.")


# --------------------------------------------------------------------------
# The generation protocols
# --------------------------------------------------------------------------


@runtime_checkable
class FrameGenerator(Protocol):
    """Produces one frame from one bundle of conditioning.

    Everything after the bundle is keyword-only on purpose. The interface this
    replaces let meaning ride on argument position, so a caller could hand a
    mask to a parameter named for a pose and nothing would complain.
    """

    def generate(
        self,
        conditioning: ConditioningBundle,
        *,
        seed: int,
        device: Device,
        params: GenerationParams,
    ) -> ArrayLike:
        """Generate the object for one frame.

        Implementations call `conditioning.require(...)` for what they declared
        and read only those fields.
        """
        ...


@runtime_checkable
class SequenceGenerator(FrameGenerator, Protocol):
    """A generator that can also produce a temporally coherent run of frames.

    Whether a backend has this is read from `CAP_TEMPORAL_SEQUENCE` in its
    registry entry, not from its class. Under the previous arrangement the
    compositor asked `isinstance(strategy, AnimateAnyoneStrategy)`, so a second
    temporal backend would have had to subclass Animate-Anyone to be offered a
    sequence at all.
    """

    def generate_sequence(
        self,
        conditioning: Sequence[ConditioningBundle],
        *,
        seed: int,
        device: Device,
        params: GenerationParams,
    ) -> Sequence[ArrayLike]:
        """Generate one frame per bundle, in order, as a coherent sequence."""
        ...


def supports_sequence(spec: BackendSpec[Any]) -> bool:
    """Whether this generator declares temporal-sequence generation."""
    return spec.supports(CAP_TEMPORAL_SEQUENCE)


def require_sequence(spec: BackendSpec[Any], generator: object | None = None) -> None:
    """Raise unless this generator both declares *and* implements sequences.

    Two failures, one check. A backend that does not declare
    `CAP_TEMPORAL_SEQUENCE` must not be handed a window of frames. A backend
    that declares it but has no `generate_sequence` is a registry entry that
    lies, which is worse — it would pass config validation and fail mid-run.

    Raises:
        UnsupportedCapabilityError: Naming the capability and what the backend
            does declare.
    """
    if not supports_sequence(spec):
        raise UnsupportedCapabilityError(
            CAP_TEMPORAL_SEQUENCE, spec.name, sorted(spec.capabilities)
        )
    if generator is not None and not isinstance(generator, SequenceGenerator):
        raise UnsupportedCapabilityError(
            CAP_TEMPORAL_SEQUENCE,
            f"{spec.name} (declares it, but has no generate_sequence method)",
            sorted(spec.capabilities),
        )


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditioningPlan:
    """What the chosen generator needs, derived once and used by both sides.

    Derived from the generator's declared `requires`, then handed to the encoder
    — which reads `stages` to know what to run and `transmit` to know what to
    put in the payload — and to the decoder, which reads `conditioning` to know
    what to expect in each bundle. One derivation, two consumers: encoder and
    decoder cannot disagree about what is being conditioned on, because neither
    works it out for itself.

    Args:
        generator: Canonical name of the generator this was derived for.
        conditioning: The `CONDITION_*` kinds every bundle must carry.
        stages: Encoder stages that must run to produce them.
        transmit: Payload items the encoder must include.
        temporal: Whether the generator declares sequence generation.
    """

    generator: str
    conditioning: frozenset[str]
    stages: frozenset[str]
    transmit: frozenset[str]
    temporal: bool = False

    @classmethod
    def derive(cls, spec: BackendSpec[Any]) -> ConditioningPlan:
        """Work out the plan for one registered generator.

        The only supported way to build a plan. Anything else — a module
        inspecting the generator's *name* to decide that pose estimation should
        be switched off, which is what `actor_pipeline.py:88` does today — puts
        the decision somewhere neither the generator nor the encoder can see.

        Raises:
            ValueError: If the spec requires a conditioning kind that is not in
                the vocabulary, which usually means a typo in a registry entry.
        """
        required = frozenset(_normalise_condition(name) for name in spec.requires)
        stages: set[str] = set()
        transmit: set[str] = set()
        for condition in required:
            stages.update(STAGES_FOR_CONDITION[condition])
            transmit.update(TRANSMIT_FOR_CONDITION[condition])
        return cls(
            generator=spec.name,
            conditioning=required,
            stages=frozenset(stages),
            transmit=frozenset(transmit),
            temporal=supports_sequence(spec),
        )

    def needs_stage(self, stage: str) -> bool:
        """Whether `stage` has to run under this plan.

        The replacement for reading a backend name from another axis: a pose
        estimator is built when this says `pose-estimation` is needed, and left
        unbuilt when it does not.
        """
        if stage not in ALL_STAGES:
            raise ValueError(
                f"Unknown stage {stage!r}. Stages: {', '.join(sorted(ALL_STAGES))}."
            )
        return stage in self.stages

    def check(self, bundle: ConditioningBundle) -> None:
        """Raise unless `bundle` carries everything this plan promised.

        Raises:
            MissingConditioningError: Naming the missing kinds.
        """
        bundle.require(*sorted(self.conditioning))

    def describe(self) -> str:
        """A readable summary, for run summaries and logs."""
        return (
            f"plan[{self.generator}]: "
            f"conditioning={', '.join(sorted(self.conditioning)) or '(none)'}; "
            f"stages={', '.join(sorted(self.stages)) or '(none)'}; "
            f"transmit={', '.join(sorted(self.transmit)) or '(none)'}; "
            f"temporal={'yes' if self.temporal else 'no'}"
        )


def unused_stages(plan: ConditioningPlan) -> frozenset[str]:
    """Stages this plan makes unnecessary.

    The measurable half of the cross-axis rule. A canny generator leaves pose
    estimation in here, so the encoder can skip it — and skipping it is a real
    saving in encode time, not just a nominal one.
    """
    return ALL_STAGES - plan.stages


def describe_fields(names: Iterable[str] | None = None) -> str:
    """A readable table of the conditioning fields and what belongs in each."""
    chosen = [FIELDS[name] for name in names] if names else list(FIELDS.values())
    width = max(len(spec.attribute) for spec in chosen)
    lines = ["conditioning fields:"]
    for spec in chosen:
        kind = spec.condition or "(not transmitted)"
        lines.append(
            f"  {spec.attribute.ljust(width)}  {kind:<16} "
            f"{'|'.join(spec.layouts):<8} {spec.dtype:<28} {spec.summary}"
        )
    return "\n".join(lines)
