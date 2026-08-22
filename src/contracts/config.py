"""The configuration schema — one validated document describing a whole run.

Nested by slot rather than flat, so a section reads as the thing it configures
and a new backend's knobs land in one place instead of accreting a prefix at the
top level. The arrangement it replaces was 81 flat fields whose names carried the
grouping (`controlnet_temporal_strength_min`, `animate_anyone_transparent_threshold`)
and whose unknown keys were silently discarded.

Validation happens in two passes, because they need different things:

**Structural** (`parse`) needs only the schema. Unknown keys, wrong types and
missing required fields are caught here, all of them at once, without importing
anything heavy.

**Contractual** (`validate`) needs the contracts but not the components. It
checks the things that are knowable before any model is loaded: that the domain
profile exists, that the keypoint schema exists, that the codec will actually
honour the requested pixel format and rate control, that the metric set resolves
and is not empty, that the enabled stages are coherent, and that a panorama
background has not been requested under a camera assumption that makes panoramas
meaningless.

A third pass belongs to the components layer, once registries exist: that every
named backend is registered, and that the chosen appearance/motion pairing has a
generator able to decode it. `validate_backends` is where that goes; it takes the
registries as arguments precisely so this module keeps importing nothing heavy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from src.contracts import capabilities, codecs, keypoints, metrics
from src.contracts import domain as domains
from src.contracts import lattice as stages
from src.contracts.errors import (
    ConfigError,
    ConfigValueError,
    ContractError,
)
from src.contracts.parsing import build, to_mapping

# --------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    """How the run is driven, independent of what it encodes."""

    source: str | None = None
    output_root: Path = Path("outputs")
    max_frames: int | None = None
    chunk_duration_sec: float = 2.0
    seed: int = 1337
    log_level: str = "info"


@dataclass(frozen=True)
class BackendConfig:
    """A backend choice plus the weights it loads.

    Shared by every perception slot, because they genuinely have the same shape.
    `backend` names the implementation; `model` names the checkpoint. Splitting
    them is what lets a config say "YOLO, but the extra-large checkpoint" without
    the backend name having to encode the checkpoint.
    """

    backend: str = "none"
    model: str | None = None
    prompt: str | None = None
    """Class prompt for open-vocabulary backends. Ignored by closed-vocabulary
    ones, which is checked rather than assumed once registries exist."""


@dataclass(frozen=True)
class PoseConfig(BackendConfig):
    """Pose estimation, plus which keypoint schema it produces."""

    schema: str = keypoints.CANONICAL_HUMAN.name


@dataclass(frozen=True)
class AppearanceConfig:
    """How an object's appearance is carried (§ object stream)."""

    representation: str = "compressed-image"
    jpeg_quality: int = 90
    downscale: int = 1
    """Both degradation knobs are here because they are not equivalent — one
    discards high-frequency detail, the other resolution — and which serves
    generative reconstruction better is an open question worth sweeping."""


@dataclass(frozen=True)
class MotionConfig:
    """How an object's motion is carried.

    Per class, not global. A tennis racket has no skeleton, so a single
    system-wide `keypoints` setting is not expressible — the player can carry
    keypoints and the racket cannot, in the same run.

    `representation` is the preference. A class that cannot carry it takes the
    best option it can from `FALLBACK_ORDER`, and `resolve` reports that it did
    rather than quietly substituting: a silent fallback would mean a stream
    carrying a different representation than the one requested, and a number
    measured under it is not comparable with one measured under the request.
    """

    representation: str = "keypoints"
    per_class: Mapping[str, str] = field(default_factory=dict)
    max_points: int = 64
    """Trajectory count, for the sparse-trajectory representation."""

    #: What a class falls back to, best first, when it cannot carry the
    #: preferred representation.
    #:
    #: Sparse trajectories come before encoded video deliberately. Both are
    #: class-agnostic, but trajectories are the cheap semantic option — a
    #: skeleton's worth of points, expanded to dense motion by the decoder —
    #: while encoding the object crop as a literal video is the classical answer
    #: this system is trying to beat. A racket with no skeleton should still get
    #: the semantic treatment; dropping it straight to encoded video would
    #: concede the comparison before running it.
    FALLBACK_ORDER: tuple[str, ...] = (
        capabilities.MOTION_SPARSE_TRAJECTORIES,
        capabilities.MOTION_ENCODED_VIDEO,
    )

    def resolve(self, profile: domains.DomainProfile) -> MotionResolution:
        """Work out what each salient class actually carries, and why."""
        chosen: dict[str, str] = {}
        fell_back: list[str] = []
        for salient in profile.salient_classes:
            override = self.per_class.get(salient.name)
            if override is not None:
                chosen[salient.name] = override
                continue
            supported = salient.supported_motion()
            if self.representation in supported:
                chosen[salient.name] = self.representation
                continue
            substitute = next(
                (option for option in self.FALLBACK_ORDER if option in supported),
                None,
            )
            if substitute is None:
                raise ConfigValueError(
                    f"motion.per_class.{salient.name}",
                    f"class {salient.name!r} supports no motion representation this "
                    f"system can fall back to. It supports {sorted(supported)}; the "
                    f"fallback order is {list(self.FALLBACK_ORDER)}.",
                )
            chosen[salient.name] = substitute
            fell_back.append(salient.name)
        return MotionResolution(by_class=chosen, fell_back=tuple(fell_back))


@dataclass(frozen=True)
class MotionResolution:
    """What each salient class ended up carrying.

    `fell_back` exists so the substitution is visible in a run summary. A number
    produced under a silently-substituted representation is not comparable with
    one produced under the requested representation, and nothing else would say
    so.
    """

    by_class: Mapping[str, str]
    fell_back: tuple[str, ...] = ()

    def describe(self) -> str:
        lines = []
        for name, representation in sorted(self.by_class.items()):
            marker = "  (fell back)" if name in self.fell_back else ""
            lines.append(f"  {name}: {representation}{marker}")
        return "\n".join(lines)


@dataclass(frozen=True)
class TemporalConfig:
    """How densely motion is sent versus interpolated."""

    metadata_sparsity: bool = True
    generation_sparsity: bool = False
    pipeline_sparsity: bool = False
    delta_threshold: float = 20.0
    keyframe_interval: int = 8
    preroll_frames: int = 0


@dataclass(frozen=True)
class BackgroundConfig:
    """The background model, and the sidecar codec carrying it.

    Two axes, not one. The knob this replaces conflated transmission strategy
    with still-image codec, which made `{panorama-delta, roi-video}` inexpressible.
    """

    method: str = domains.BACKGROUND_PANORAMA_FULL
    codec: str = "jpeg"
    jpeg_quality: int = 50


@dataclass(frozen=True)
class GeneratorConfig:
    """The generative backend and its sampling knobs."""

    backend: str = "none"
    variant: str | None = None
    checkpoint: str | None = None
    width: int = 512
    height: int = 512
    steps: int = 20
    strength: float = 0.65
    guidance: float = 7.0

    @property
    def resolved_name(self) -> str:
        """Registry key: ``{variant}-{backend}`` when a variant is named.

        Lets `{backend: controlnet, variant: canny}` and `canny-controlnet`
        describe the same thing, so the config reads by slot while the registry
        keeps flat exact-match names.
        """
        if self.variant:
            return f"{self.variant}-{self.backend}"
        return self.backend


@dataclass(frozen=True)
class ResidualConfig:
    """The corrective residual — one component like any other, and switchable off."""

    codec: str = "av1"
    rate_control: codecs.RateControl = codecs.RateControl.CRF
    rate: int | None = 35
    preset: str | None = "8"
    pix_fmt: str = "yuv420p"
    block_size: int = 8
    block_threshold: float = 0.0
    background_downscale: int = 2

    def encode_request(self) -> codecs.EncodeRequest:
        """The residual's encode, as a validatable request."""
        return codecs.EncodeRequest(
            codec_name=self.codec,
            rate_control=self.rate_control,
            rate=self.rate,
            preset=self.preset,
            pix_fmt=self.pix_fmt,
        )


@dataclass(frozen=True)
class FallbackConfig:
    """The codec a run falls back to, and anchors are measured against."""

    codec: str = "av1"
    rate_control: codecs.RateControl = codecs.RateControl.CRF
    rate: int | None = 35
    preset: str | None = "8"
    pix_fmt: str = "yuv420p"
    roi: bool = False

    def encode_request(self, roi_map: str | None = None) -> codecs.EncodeRequest:
        return codecs.EncodeRequest(
            codec_name=self.codec,
            rate_control=self.rate_control,
            rate=self.rate,
            preset=self.preset,
            pix_fmt=self.pix_fmt,
            roi_map=roi_map,
        )


@dataclass(frozen=True)
class EvaluationConfig:
    """Which metrics run. Never empty — see `metrics.ALWAYS_ON`."""

    metrics: tuple[str, ...] = ("psnr",)
    max_frames: int | None = None


@dataclass(frozen=True)
class LatticeConfig:
    """Which components are switched on.

    Everything defaults on except generation, which needs a backend named to do
    anything and so cannot sensibly default to enabled with none chosen. An
    ablation is expressed by turning things off, which reads the way the
    experiment reads.
    """

    scene_classification: bool = True
    detection: bool = True
    selection: bool = True
    tracking: bool = True
    appearance: bool = True
    motion: bool = True
    temporal_policy: bool = True
    pose: bool = True
    segmentation: bool = True
    rigid_objects: bool = True
    background: bool = True
    generation: bool = False
    residual: bool = True

    #: Field name to catalogue stage name. The catalogue owns the names; this is
    #: only the mapping from the config's Python-legal spelling.
    _STAGE_FIELDS = {
        "scene_classification": stages.STAGE_SCENE,
        "detection": stages.STAGE_DETECTION,
        "selection": stages.STAGE_SELECTION,
        "tracking": stages.STAGE_TRACKING,
        "appearance": stages.STAGE_APPEARANCE,
        "motion": stages.STAGE_MOTION,
        "temporal_policy": stages.STAGE_TEMPORAL,
        "pose": stages.STAGE_POSE,
        "segmentation": stages.STAGE_SEGMENTATION,
        "rigid_objects": stages.STAGE_RIGID,
        "background": stages.STAGE_BACKGROUND,
        "generation": stages.STAGE_GENERATION,
        "residual": stages.STAGE_RESIDUAL,
    }

    def to_lattice(self) -> stages.StageLattice:
        """The corner this config names."""
        enabled = {
            stage_name
            for attribute, stage_name in self._STAGE_FIELDS.items()
            if getattr(self, attribute)
        }
        return stages.StageLattice(frozenset(enabled) | stages.REQUIRED_STAGES)


# --------------------------------------------------------------------------
# The document
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PointstreamConfig:
    """One run, fully described.

    Frozen, because encoder and decoder both read it and a configuration edited
    partway through a run is a divergence between the two sides that nothing
    would report.
    """

    run: RunConfig = field(default_factory=RunConfig)
    domain: str = "tennis"
    lattice: LatticeConfig = field(default_factory=LatticeConfig)
    detector: BackendConfig = field(default_factory=lambda: BackendConfig(backend="yolo", model="yolo26n.pt"))
    selection: BackendConfig = field(default_factory=lambda: BackendConfig(backend="heuristic"))
    tracking: BackendConfig = field(default_factory=lambda: BackendConfig(backend="tracker"))
    pose: PoseConfig = field(default_factory=lambda: PoseConfig(backend="yolo", model="yolo26n-pose.pt"))
    segmenter: BackendConfig = field(default_factory=lambda: BackendConfig(backend="yolo", model="yolo26n-seg.pt"))
    appearance: AppearanceConfig = field(default_factory=AppearanceConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    rigid: BackendConfig = field(default_factory=lambda: BackendConfig(backend="tennis"))
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    residual: ResidualConfig = field(default_factory=ResidualConfig)
    fallback: FallbackConfig = field(default_factory=FallbackConfig)
    transport: str = "disk"
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @property
    def profile(self) -> domains.DomainProfile:
        """The resolved domain profile."""
        return domains.profile(self.domain)

    @property
    def stages(self) -> stages.StageLattice:
        """The lattice corner this config names."""
        return self.lattice.to_lattice()

    def with_(self, **overrides: Any) -> PointstreamConfig:
        """A copy with top-level sections replaced, for sweeps."""
        return replace(self, **overrides)


# --------------------------------------------------------------------------
# Parsing and validation
# --------------------------------------------------------------------------


def parse(data: Mapping[str, Any]) -> PointstreamConfig:
    """Structural pass: build the document, rejecting anything unrecognised."""
    return build(PointstreamConfig, data)


def validate(config: PointstreamConfig) -> None:
    """Contractual pass: everything knowable before a model is loaded.

    Collects every problem rather than stopping at the first, so one run of
    validation tells you everything wrong with a config.

    Raises:
        ConfigError: With one entry per problem found.
    """
    problems: list[ContractError] = []

    def note(exc: ContractError) -> None:
        problems.append(exc)

    # Domain, and the keypoint schema it implies.
    profile: domains.DomainProfile | None = None
    try:
        profile = domains.profile(config.domain)
    except ContractError as exc:
        note(exc)
    except ValueError as exc:
        note(ConfigValueError("domain", str(exc)))

    try:
        keypoints.schema(config.pose.schema)
    except ValueError as exc:
        note(ConfigValueError("pose.schema", str(exc)))

    # Codecs: does the encoder actually honour what was asked for?
    for path, request in (
        ("residual", config.residual.encode_request()),
        ("fallback", config.fallback.encode_request()),
    ):
        try:
            request.validate()
        except ContractError as exc:
            note(ConfigValueError(path, str(exc)))

    # A fallback asking for region control needs a codec that has it.
    if config.fallback.roi:
        try:
            capabilities = codecs.codec(config.fallback.codec)
            if not capabilities.supports_roi:
                note(
                    ConfigValueError(
                        "fallback.roi",
                        f"codec {config.fallback.codec!r} has no region-of-interest "
                        f"mechanism. Codecs that do: {sorted(codecs.ROI_CAPABLE)}.",
                    )
                )
        except ContractError as exc:
            note(exc)

    # Metrics: resolvable, and never empty.
    try:
        metrics.resolve(config.evaluation.metrics)
    except ContractError as exc:
        note(exc)

    # The lattice corner has to be coherent.
    corner: stages.StageLattice | None = None
    try:
        corner = config.stages
    except ContractError as exc:
        note(exc)

    # Background method against the domain's camera assumption. A panorama under
    # parallax does not merely score badly — it is meaningless, and silently so.
    if profile is not None and config.lattice.background:
        try:
            profile.assert_background_valid(config.background.method)
        except ContractError as exc:
            note(exc)

    # Whatever each class ends up carrying has to be something it can carry.
    # Only explicit per-class overrides can be wrong here — the preference falls
    # back rather than failing, and reports that it did.
    if profile is not None and config.lattice.motion:
        try:
            resolution = config.motion.resolve(profile)
        except ContractError as exc:
            note(exc)
        else:
            for class_name, representation in resolution.by_class.items():
                if class_name not in config.motion.per_class:
                    continue
                try:
                    profile.assert_motion_supported(
                        class_name, representation, path=f"motion.per_class.{class_name}"
                    )
                except ContractError as exc:
                    note(exc)

    # A generator that is switched on has to be named, and vice versa.
    if corner is not None:
        generation_on = stages.STAGE_GENERATION in corner.enabled
        named = config.generator.backend not in ("", "none")
        if generation_on and not named:
            note(
                ConfigValueError(
                    "generator.backend",
                    "the generation stage is enabled but no generator is named. "
                    "Switch the stage off in the lattice, or name a backend.",
                )
            )
        if named and not generation_on:
            note(
                ConfigValueError(
                    "lattice.generation",
                    f"generator {config.generator.resolved_name!r} is named but the "
                    f"generation stage is switched off, so it would never run.",
                )
            )

    if problems:
        raise ConfigError(problems)


def validate_backends(
    config: PointstreamConfig,
    *,
    generators: Any = None,
    registries: Mapping[str, Any] | None = None,
) -> None:
    """The pass that needs the component registries.

    Kept separate, and taking its registries as arguments, so this module still
    imports nothing heavy: config validation must work on a machine where no
    backend's dependencies are installed.

    Args:
        config: An already-`validate`d document.
        generators: The generator registry, when available. Without it the
            appearance/motion pairing cannot be checked, and is skipped.
        registries: Optional mapping of axis name -> Registry. When given,
            every named backend on that axis is checked to exist. Axes whose
            registry is omitted are skipped, so streams can wire one axis at
            a time. A backend of ``""`` or ``"none"`` is treated as unset.

    Raises:
        ConfigError: With one entry per problem.
        UndecodableStreamError: If no registered generator accepts the chosen
            appearance and motion representations together.
    """
    problems: list[ContractError] = []

    if registries:
        named = {
            "detector": config.detector.backend,
            "selection": config.selection.backend,
            "tracking": config.tracking.backend,
            "pose": config.pose.backend,
            "segmenter": config.segmenter.backend,
            "generator": config.generator.resolved_name,
            "rigid": config.rigid.backend,
            "transport": config.transport,
            "codec": config.fallback.codec,
            "domain": config.domain,
            "background": config.background.method,
            "appearance": config.appearance.representation,
            "motion": config.motion.representation,
        }
        for axis, registry in registries.items():
            name = named.get(axis)
            if name in (None, "", "none"):
                continue
            try:
                registry.spec(name)
            except ContractError as exc:
                problems.append(exc)
        if config.evaluation.metrics and "metric" in registries:
            for metric_name in config.evaluation.metrics:
                try:
                    registries["metric"].spec(metric_name)
                except ContractError as exc:
                    problems.append(exc)

    if generators is not None:
        from src.contracts import objectstream

        # Every representation actually in play, not just the preferred one: a
        # class that fell back to encoded video still needs something able to
        # decode it.
        resolution = config.motion.resolve(config.profile)
        wanted = {config.motion.representation, *resolution.by_class.values()}

        for motion in sorted(wanted):
            try:
                objectstream.assert_decodable(
                    config.appearance.representation, motion, generators
                )
            except ContractError as exc:
                problems.append(exc)

    if problems:
        raise ConfigError(problems)


def load(data: Mapping[str, Any]) -> PointstreamConfig:
    """Parse then validate, which is what a caller almost always wants."""
    config = parse(data)
    validate(config)
    return config


def default() -> PointstreamConfig:
    """The shipped default configuration."""
    return PointstreamConfig()


def render_default() -> dict[str, Any]:
    """The default config as a plain mapping, for writing `config/default.yaml`.

    Generated from the schema rather than maintained by hand. This is what makes
    a documented-but-unreachable key impossible: a key can only appear in the
    file if a field exists to produce it.
    """
    return to_mapping(default())
