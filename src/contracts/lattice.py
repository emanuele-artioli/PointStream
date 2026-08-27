"""The ablation lattice — which components are switched on.

This is the organising principle that makes the platform a research instrument
rather than tidy code. **Every stage can be switched off, and the residual
absorbs whatever the disabled stages would have handled.** Turn off subject
detection but keep the background model: metadata shrinks, encode time drops,
and the residual grows to carry the players. Turn off the background model too
and the residual grows again. Turn everything off — residual included — and
there is nothing left but the source video, which is why the whole-frame
baseline is a corner of this lattice rather than a special comparison mode.

That is what gives component ablations in one uniform currency: does racket
tracking pay for itself? Run it on and off; the change in total payload *is* the
answer, measured identically for every component.

Two rules make it real rather than aspirational:

**No stage is structurally required except where the catalogue says so.** Only
the codec, the transport and the metrics are — a run that transmits anything
needs an encoder and a way to deliver it, and quality is measured in every
configuration without exception. `_assert_optionality` enforces that at import,
so a later edit cannot quietly promote a stage to mandatory and shrink the
lattice out from under the ablations.

**Stages declare what they produce and consume**, so the enabled set can be
checked for coherence and the pipeline DAG built from it. The contradiction this
catches: generation enabled with detection disabled, which leaves the generator
with nothing to draw; or a generator that declares it needs pose conditioning
while pose estimation is off. The arrangement this replaces derived neither —
one module reached across axes and nulled the pose estimator whenever the
configured generator's *name* contained a certain substring.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final

from src.contracts.capabilities import (
    CONDITION_APPEARANCE,
    CONDITION_CANNY,
    CONDITION_MASK,
    CONDITION_MOTION_FIELD,
    CONDITION_POSE,
)
from src.contracts.errors import ConfigValueError, UnknownBackendError

# --------------------------------------------------------------------------
# Stage names — the catalogue's sixteen rows
# --------------------------------------------------------------------------

STAGE_SCENE: Final = "scene-classification"
STAGE_DETECTION: Final = "detection"
STAGE_SELECTION: Final = "selection"
STAGE_TRACKING: Final = "tracking"
STAGE_APPEARANCE: Final = "appearance"
STAGE_MOTION: Final = "motion"
STAGE_TEMPORAL: Final = "temporal-policy"
STAGE_POSE: Final = "pose"
STAGE_SEGMENTATION: Final = "segmentation"
STAGE_RIGID: Final = "rigid-objects"
STAGE_BACKGROUND: Final = "background"
STAGE_GENERATION: Final = "generation"
STAGE_RESIDUAL: Final = "residual"
STAGE_CODEC: Final = "codec"
STAGE_TRANSPORT: Final = "transport"
STAGE_METRICS: Final = "metrics"


# --------------------------------------------------------------------------
# Artifact names — what flows between stages
# --------------------------------------------------------------------------

ART_SCENE_SPANS: Final = "scene-spans"
ART_SUBJECTS: Final = "subjects"
ART_SALIENT_SUBJECTS: Final = "salient-subjects"
ART_IDENTITIES: Final = "identities"
ART_APPEARANCE_PAYLOAD: Final = "appearance-payload"
ART_MOTION_PAYLOAD: Final = "motion-payload"
ART_SCHEDULE: Final = "temporal-schedule"
ART_KEYPOINTS: Final = "keypoints"
ART_MASKS: Final = "masks"
ART_RIGID_SHAPES: Final = "rigid-shapes"
ART_BACKGROUND_MODEL: Final = "background-model"
ART_GENERATED_FRAMES: Final = "generated-frames"
ART_RESIDUAL_STREAM: Final = "residual-stream"
ART_BITSTREAM: Final = "bitstream"
ART_DELIVERED: Final = "delivered-payload"
ART_QUALITY: Final = "quality"


# --------------------------------------------------------------------------
# The catalogue
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StageSpec:
    """One row of the component catalogue.

    Args:
        name: Config key and lattice key.
        row: Its number in the catalogue, used only to order listings and to
            break DAG ties deterministically.
        when_off: Where the work goes instead. This is what makes the lattice
            *measurable* rather than merely configurable — every disabled stage
            has a stated destination for its share of the signal, and the
            residual is usually it.
        produces: Artifacts this stage makes available.
        consumes: Artifacts it cannot run without. If no enabled stage produces
            one of these, the enabled set is incoherent.
        optional_inputs: Artifacts it uses when present. These create ordering
            edges in the DAG but never a requirement.
        required: Whether the stage cannot be switched off. True for exactly
            three rows, and `_assert_optionality` keeps it that way.
        variants: The alternatives available on this axis, for listings.
        summary: One line.
    """

    name: str
    row: int
    when_off: str
    produces: frozenset[str] = frozenset()
    consumes: frozenset[str] = frozenset()
    optional_inputs: frozenset[str] = frozenset()
    required: bool = False
    variants: tuple[str, ...] = ()
    summary: str = ""


_CATALOGUE: Final[tuple[StageSpec, ...]] = (
    StageSpec(
        name=STAGE_SCENE,
        row=1,
        when_off="whole input is one span; no semantic-vs-fallback routing",
        produces=frozenset({ART_SCENE_SPANS}),
        variants=("hsv-histogram", "none"),
        summary="Split the input into spans and route them.",
    ),
    StageSpec(
        name=STAGE_DETECTION,
        row=2,
        when_off="no subjects found; they land in the residual",
        produces=frozenset({ART_SUBJECTS}),
        optional_inputs=frozenset({ART_SCENE_SPANS}),
        variants=("yolo26", "sam3", "rf-detr"),
        summary="Find candidate objects.",
    ),
    StageSpec(
        name=STAGE_SELECTION,
        row=3,
        when_off="every detection treated as salient, spectators included",
        produces=frozenset({ART_SALIENT_SUBJECTS}),
        consumes=frozenset({ART_SUBJECTS}),
        variants=("open-vocabulary", "heuristic", "all-detections"),
        summary="Decide which detections are worth modelling.",
    ),
    StageSpec(
        name=STAGE_TRACKING,
        row=4,
        when_off="no cross-frame identity, so no appearance reuse",
        produces=frozenset({ART_IDENTITIES}),
        consumes=frozenset({ART_SUBJECTS}),
        variants=("tracker+recovery", "per-frame"),
        summary="Carry object identity across frames.",
    ),
    StageSpec(
        name=STAGE_APPEARANCE,
        row=5,
        when_off="generator has no appearance cue",
        produces=frozenset({ART_APPEARANCE_PAYLOAD}),
        consumes=frozenset({ART_SUBJECTS}),
        optional_inputs=frozenset({ART_SALIENT_SUBJECTS, ART_IDENTITIES, ART_MASKS}),
        variants=("compressed-image", "diffusion-latent", "image-embedding"),
        summary="Establish what each object looks like, once.",
    ),
    StageSpec(
        name=STAGE_MOTION,
        row=6,
        when_off="object static after appearance is established; motion lands in the residual",
        produces=frozenset({ART_MOTION_PAYLOAD}),
        consumes=frozenset({ART_SUBJECTS}),
        optional_inputs=frozenset({ART_KEYPOINTS, ART_IDENTITIES, ART_SCHEDULE}),
        variants=("keypoints", "sparse-trajectories", "encoded-video"),
        summary="Describe how that appearance evolves.",
    ),
    StageSpec(
        name=STAGE_TEMPORAL,
        row=7,
        when_off=(
            "every frame fully processed — maximum metadata and compute, "
            "minimum prediction error"
        ),
        produces=frozenset({ART_SCHEDULE}),
        optional_inputs=frozenset({ART_SUBJECTS, ART_SCENE_SPANS}),
        variants=("metadata-sparsity", "generation-sparsity", "pipeline-sparsity", "none"),
        summary="Decide, per frame and object, what is transmitted and what is interpolated.",
    ),
    StageSpec(
        name=STAGE_POSE,
        row=8,
        when_off="no keypoints; motion representation must be trajectories or video",
        produces=frozenset({ART_KEYPOINTS}),
        consumes=frozenset({ART_SUBJECTS}),
        optional_inputs=frozenset({ART_SCHEDULE}),
        variants=("dwpose", "yolo-pose", "none"),
        summary="Estimate skeletons for classes that have one.",
    ),
    StageSpec(
        name=STAGE_SEGMENTATION,
        row=9,
        when_off="compositing falls back to heuristic masks",
        produces=frozenset({ART_MASKS}),
        consumes=frozenset({ART_SUBJECTS}),
        optional_inputs=frozenset({ART_SCHEDULE}),
        variants=("yolo-seg", "sam3", "none"),
        summary="Produce per-object masks.",
    ),
    StageSpec(
        name=STAGE_RIGID,
        row=10,
        when_off="rigid objects land in the residual",
        produces=frozenset({ART_RIGID_SHAPES}),
        consumes=frozenset({ART_SUBJECTS}),
        optional_inputs=frozenset({ART_KEYPOINTS, ART_MASKS}),
        variants=("racket-hull", "ball-difference", "ball-segmentation", "none"),
        summary="Per-class strategies for objects with no skeleton.",
    ),
    StageSpec(
        name=STAGE_BACKGROUND,
        row=11,
        when_off="background lands in the residual",
        produces=frozenset({ART_BACKGROUND_MODEL}),
        optional_inputs=frozenset({ART_MASKS, ART_SCENE_SPANS}),
        variants=("panorama-full", "panorama-delta", "none"),
        summary="Model the background once and warp it per frame.",
    ),
    StageSpec(
        name=STAGE_GENERATION,
        row=12,
        when_off="subjects land in the residual",
        produces=frozenset({ART_GENERATED_FRAMES}),
        consumes=frozenset({ART_SUBJECTS}),
        optional_inputs=frozenset(
            {
                ART_APPEARANCE_PAYLOAD,
                ART_MOTION_PAYLOAD,
                ART_KEYPOINTS,
                ART_MASKS,
                ART_BACKGROUND_MODEL,
                ART_SCHEDULE,
            }
        ),
        variants=(
            "controlnet",
            "pix2pix",
            "spade4tennis",
            "animate-anyone",
            "motion-vector-animator",
            "upscale-refine",
            "none",
        ),
        summary="Reconstruct objects from their appearance and motion.",
    ),
    StageSpec(
        name=STAGE_RESIDUAL,
        row=13,
        when_off="nothing corrects generation error; quality rests entirely on generation",
        produces=frozenset({ART_RESIDUAL_STREAM}),
        optional_inputs=frozenset({ART_GENERATED_FRAMES, ART_BACKGROUND_MODEL}),
        variants=("lossy", "lossless", "none"),
        summary="Correct everyone else's error. Consequential, but not privileged.",
    ),
    StageSpec(
        name=STAGE_CODEC,
        row=14,
        when_off="not switchable — any transmitted video stream needs an encoder",
        produces=frozenset({ART_BITSTREAM}),
        optional_inputs=frozenset(
            {
                ART_RESIDUAL_STREAM,
                ART_BACKGROUND_MODEL,
                ART_APPEARANCE_PAYLOAD,
                # Without this the DAG is free to order the codec before the
                # generator, and a generation-on / residual-off corner reaches
                # this stage with no crops to composite. BP23 found that corner
                # silently delivering the SOURCE; BP24 declares the edge so the
                # ordering cannot happen in the first place.
                ART_GENERATED_FRAMES,
            }
        ),
        required=True,
        variants=("avc", "hevc", "av1", "vvc"),
        summary="Encode whatever video is transmitted, with or without ROI.",
    ),
    StageSpec(
        name=STAGE_TRANSPORT,
        row=15,
        when_off="not switchable — a payload that is never delivered cannot be scored",
        produces=frozenset({ART_DELIVERED}),
        consumes=frozenset({ART_BITSTREAM}),
        optional_inputs=frozenset({ART_MOTION_PAYLOAD, ART_SCHEDULE}),
        required=True,
        variants=("disk",),
        summary="Serialize the payload and move it.",
    ),
    StageSpec(
        name=STAGE_METRICS,
        row=16,
        when_off="not switchable — at least PSNR always runs",
        produces=frozenset({ART_QUALITY}),
        consumes=frozenset({ART_DELIVERED}),
        required=True,
        variants=("psnr", "ssim", "vmaf", "lpips", "fvmd"),
        summary=(
            "Measure quality. Mandatory in every configuration: the residual always "
            "carries coarseness and generation is statistical, so correctness is never "
            "something that can be assumed instead of measured."
        ),
    ),
)

#: Every stage, by name, in catalogue order.
STAGES: Final[Mapping[str, StageSpec]] = {spec.name: spec for spec in _CATALOGUE}

#: The only stages the architecture may insist on. Everything else is a knob.
REQUIRED_STAGES: Final[frozenset[str]] = frozenset(
    {STAGE_CODEC, STAGE_TRANSPORT, STAGE_METRICS}
)

#: Which stage supplies each conditioning kind a generator can declare it needs.
#: Canny is derived from the object crop rather than from a stage of its own, so
#: it is detection that has to be enabled for it to exist at all.
CONDITION_SOURCES: Final[Mapping[str, str]] = {
    CONDITION_POSE: STAGE_POSE,
    CONDITION_MASK: STAGE_SEGMENTATION,
    CONDITION_APPEARANCE: STAGE_APPEARANCE,
    CONDITION_MOTION_FIELD: STAGE_MOTION,
    CONDITION_CANNY: STAGE_DETECTION,
}


def _assert_optionality() -> None:
    """Fail at import if any stage outside the catalogue's three is mandatory.

    The lattice's whole value is that every component can be measured by being
    turned off. A stage quietly promoted to required removes a corner from every
    ablation, so this is checked rather than trusted — and checked here, where a
    bad edit stops the process instead of shrinking a results table.
    """
    declared = frozenset(spec.name for spec in _CATALOGUE if spec.required)
    if declared != REQUIRED_STAGES:
        unexpected = sorted(declared - REQUIRED_STAGES)
        missing = sorted(REQUIRED_STAGES - declared)
        raise ValueError(
            f"Stage catalogue disagrees with REQUIRED_STAGES. "
            f"Unexpectedly required: {unexpected or 'none'}. "
            f"Expected required but optional: {missing or 'none'}. "
            f"Only the codec, transport and metrics rows may be structurally "
            f"required; everything else must be ablatable."
        )
    rows = [spec.row for spec in _CATALOGUE]
    if sorted(rows) != list(range(1, len(_CATALOGUE) + 1)):
        raise ValueError(f"Stage catalogue rows are not 1..{len(_CATALOGUE)}: {sorted(rows)}")


_assert_optionality()


def stage(name: str) -> StageSpec:
    """Look up a stage by name.

    Raises:
        UnknownBackendError: With the registered stages and a close-match hint.
    """
    try:
        return STAGES[name]
    except KeyError:
        raise UnknownBackendError("stage", name, sorted(STAGES)) from None


#: Every stage that can be switched off, in catalogue order.
OPTIONAL_STAGES: Final[tuple[str, ...]] = tuple(
    spec.name for spec in _CATALOGUE if not spec.required
)


# --------------------------------------------------------------------------
# The lattice
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StageLattice:
    """Which components are enabled — one corner of the ablation lattice.

    Args:
        enabled: Stage names that are on. The three required stages must be
            among them; a config that switches one off is rejected rather than
            silently corrected, because silently correcting it would mean a run
            labelled "no metrics" quietly had metrics.

    Corners are compared by their enabled set, so two lattices built different
    ways but naming the same stages are equal — which is what lets a results
    table key on the corner.
    """

    enabled: frozenset[str] = REQUIRED_STAGES

    def __post_init__(self) -> None:
        unknown = sorted(set(self.enabled) - set(STAGES))
        if unknown:
            raise UnknownBackendError("stage", unknown[0], sorted(STAGES))
        missing = sorted(REQUIRED_STAGES - set(self.enabled))
        if missing:
            raise ConfigValueError(
                "lattice",
                f"stage(s) {missing} cannot be switched off. The codec and transport "
                f"are needed by any run that transmits anything, and quality is "
                f"measured in every configuration without exception — a run that "
                f"reports no quality number cannot be cited.",
            )

    # -- construction ------------------------------------------------------

    @classmethod
    def of(cls, *names: str) -> StageLattice:
        """A lattice with `names` enabled, plus the required stages."""
        return cls(frozenset(names) | REQUIRED_STAGES)

    @classmethod
    def all_on(cls) -> StageLattice:
        """Every stage enabled — the maximum-metadata, maximum-compute corner."""
        return cls(frozenset(STAGES))

    @classmethod
    def all_off(cls) -> StageLattice:
        """Only the required stages: the source video, encoded and delivered.

        This is the baseline, and it is a corner of the lattice rather than a
        separate comparison mode. Anything the semantic path claims has to beat
        it in the same currency.
        """
        return cls(REQUIRED_STAGES)

    def enable(self, *names: str) -> StageLattice:
        """This corner with `names` additionally on."""
        for name in names:
            stage(name)
        return StageLattice(self.enabled | frozenset(names))

    def disable(self, *names: str) -> StageLattice:
        """This corner with `names` off, leaving everything else untouched.

        May produce an incoherent corner — disabling detection alone leaves
        generation with nothing to draw. Use `prune` for a corner guaranteed to
        stay coherent, or call `assert_coherent` and read the error.
        """
        for name in names:
            stage(name)
        return StageLattice(self.enabled - frozenset(names))

    def prune(self, *names: str) -> StageLattice:
        """This corner with `names` off, plus everything that depended on them.

        Turning off detection turns off every stage that needs subjects, which is
        the honest meaning of "run without detection": those stages have nothing
        to work on, so their share of the signal goes to the residual. This is
        what an ablation sweep should call.
        """
        remaining = set(self.disable(*names).enabled)
        changed = True
        while changed:
            changed = False
            produced = _produced_by(remaining)
            for name in sorted(remaining):
                spec = STAGES[name]
                if spec.required:
                    continue
                if not spec.consumes <= produced:
                    remaining.discard(name)
                    changed = True
        return StageLattice(frozenset(remaining))

    # -- inspection --------------------------------------------------------

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self.enabled

    def is_enabled(self, name: str) -> bool:
        """Whether `name` is on. Raises for a stage that does not exist."""
        stage(name)
        return name in self.enabled

    @property
    def disabled(self) -> frozenset[str]:
        """Stage names that are off."""
        return frozenset(STAGES) - self.enabled

    @property
    def optional_enabled(self) -> tuple[str, ...]:
        """Enabled stages that could have been switched off, in catalogue order."""
        return tuple(name for name in OPTIONAL_STAGES if name in self.enabled)

    @property
    def is_source_passthrough(self) -> bool:
        """Whether this corner reduces to transmitting the source video.

        True only when nothing optional is on — not even the residual, since a
        residual with nothing to correct is the whole frame coded at its own
        coarseness, which is a different arm with a different knob.
        """
        return not self.optional_enabled

    def produced_artifacts(self) -> frozenset[str]:
        """Everything the enabled set makes available."""
        return _produced_by(self.enabled)

    def missing_inputs(self) -> Mapping[str, frozenset[str]]:
        """Enabled stages whose hard inputs nothing enabled produces."""
        produced = self.produced_artifacts()
        return {
            name: STAGES[name].consumes - produced
            for name in sorted(self.enabled)
            if not STAGES[name].consumes <= produced
        }

    def assert_coherent(
        self, *, conditioning: Iterable[str] = (), path: str = "lattice"
    ) -> None:
        """Raise unless the enabled set can actually run.

        Args:
            conditioning: Conditioning kinds the chosen generator declares it
                needs. Passing them here is what makes the cross-axis effect
                *derived*: a generator that wants pose is the reason pose
                estimation must be on, and nothing reads the generator's name to
                work that out.
            path: Config path used in the error message.

        Raises:
            ConfigValueError: Naming the stage, the artifact it is missing, and
                the stage that would have produced it.
        """
        missing = self.missing_inputs()
        if missing:
            name, absent = next(iter(missing.items()))
            producers = sorted(
                other for other, spec in STAGES.items() if spec.produces & absent
            )
            raise ConfigValueError(
                path,
                f"stage {name!r} is enabled but nothing enabled produces "
                f"{sorted(absent)}. Enable {producers} or disable {name!r} — "
                f"whatever {name!r} would have contributed lands in the residual.",
            )

        wanted = sorted(set(conditioning))
        for kind in wanted:
            source = CONDITION_SOURCES.get(kind)
            if source is None:
                raise ConfigValueError(
                    path,
                    f"the chosen generator requires conditioning {kind!r}, which no "
                    f"stage produces. Known conditioning: "
                    f"{', '.join(sorted(CONDITION_SOURCES))}.",
                )
            if source not in self.enabled:
                raise ConfigValueError(
                    path,
                    f"the chosen generator requires {kind!r} conditioning, which comes "
                    f"from the {source!r} stage, but {source!r} is disabled. Enable it, "
                    f"or choose a generator that does not need {kind!r}.",
                )
        if wanted and STAGE_GENERATION not in self.enabled:
            raise ConfigValueError(
                path,
                f"conditioning {wanted} was declared, but the {STAGE_GENERATION!r} "
                f"stage is disabled, so nothing would consume it.",
            )

    def dag(self) -> tuple[str, ...]:
        """The enabled stages in an order that satisfies their dependencies.

        Ordering uses optional inputs as well as hard ones, so a background model
        is built before the generator that would use it. Ties break on catalogue
        row, which keeps the order stable across runs and therefore comparable
        between them.

        Raises:
            ConfigValueError: If the enabled set is incoherent, or if the
                declared dependencies contain a cycle.
        """
        self.assert_coherent()
        produced = self.produced_artifacts()
        pending = {
            name: (STAGES[name].consumes | (STAGES[name].optional_inputs & produced))
            for name in self.enabled
        }
        satisfied: set[str] = set()
        order: list[str] = []
        while pending:
            ready = sorted(
                (name for name, needs in pending.items() if needs <= satisfied),
                key=lambda name: STAGES[name].row,
            )
            if not ready:
                raise ConfigValueError(
                    "lattice",
                    f"the enabled stages {sorted(pending)} depend on each other "
                    f"cyclically, so no run order exists.",
                )
            for name in ready:
                order.append(name)
                satisfied |= STAGES[name].produces
                del pending[name]
        return tuple(order)

    def describe(self) -> str:
        """A readable corner, so an ablation can be read back by a human."""
        width = max(len(name) for name in STAGES)
        lines = [f"lattice corner: {self.label()}"]
        for spec in _CATALOGUE:
            if spec.name in self.enabled:
                mark = "required" if spec.required else "on"
                note = spec.summary
            else:
                mark = "off"
                note = f"-> {spec.when_off}"
            lines.append(f"  {spec.name.ljust(width)}  {mark:<8}  {note}")
        return "\n".join(lines)

    def label(self) -> str:
        """A short name for this corner: a registered one, or its enabled set."""
        for name, corner in NAMED_CORNERS.items():
            if corner == self:
                return name
        enabled = self.optional_enabled
        return "+".join(enabled) if enabled else "(required only)"


def _produced_by(names: Iterable[str]) -> frozenset[str]:
    """Every artifact the given stages make available."""
    produced: set[str] = set()
    for name in names:
        produced |= STAGES[name].produces
    return frozenset(produced)


# --------------------------------------------------------------------------
# Named corners
# --------------------------------------------------------------------------

#: Everything off. What is left is the source video, encoded and delivered — the
#: baseline every semantic claim is measured against.
SOURCE_PASSTHROUGH: Final = StageLattice.all_off()

#: Only the residual. Nothing is predicted, so the residual carries whole frames:
#: this is the Whole-Frame Residual Baseline, and it differs from
#: `SOURCE_PASSTHROUGH` in having a coarseness knob of its own.
WHOLE_FRAME_RESIDUAL: Final = StageLattice.of(STAGE_RESIDUAL)

#: Every stage on. Maximum metadata and maximum compute; the upper end of the
#: compute/bandwidth/quality surface rather than a recommended operating point.
FULL: Final = StageLattice.all_on()

#: Everything except the residual. Quality rests entirely on generation, which
#: is the only configuration where a like-for-like comparison of two encodings
#: of the same object is visible in the quality number rather than absorbed.
GENERATIVE_ONLY: Final = FULL.disable(STAGE_RESIDUAL)

#: Perception and background modelling, with nothing generative: what the
#: semantic metadata costs before any model runs.
NO_GENERATION: Final = FULL.prune(STAGE_GENERATION)

NAMED_CORNERS: Final[Mapping[str, StageLattice]] = {
    "source-passthrough": SOURCE_PASSTHROUGH,
    "whole-frame-residual": WHOLE_FRAME_RESIDUAL,
    "full": FULL,
    "generative-only": GENERATIVE_ONLY,
    "no-generation": NO_GENERATION,
}


def corner(name: str) -> StageLattice:
    """Look up a named corner.

    Raises:
        UnknownBackendError: With the registered corner names.
    """
    try:
        return NAMED_CORNERS[name]
    except KeyError:
        raise UnknownBackendError("lattice corner", name, sorted(NAMED_CORNERS)) from None


def describe_catalogue() -> str:
    """A readable table of every stage and where its work goes when it is off."""
    width = max(len(name) for name in STAGES)
    lines = ["stage catalogue:"]
    for spec in _CATALOGUE:
        flag = "required" if spec.required else "optional"
        lines.append(
            f"  {spec.row:>2}. {spec.name.ljust(width)}  {flag}  "
            f"[{', '.join(spec.variants)}]"
        )
        lines.append(f"      off -> {spec.when_off}")
    return "\n".join(lines)
