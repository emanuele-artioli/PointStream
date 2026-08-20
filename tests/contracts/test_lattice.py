"""Behaviour and plausible misuse of the ablation lattice."""

import pytest

from src.contracts.capabilities import (
    CONDITION_MASK,
    CONDITION_MOTION_FIELD,
    CONDITION_POSE,
)
from src.contracts.errors import ConfigValueError, UnknownBackendError
from src.contracts.lattice import (
    FULL,
    GENERATIVE_ONLY,
    NAMED_CORNERS,
    OPTIONAL_STAGES,
    REQUIRED_STAGES,
    SOURCE_PASSTHROUGH,
    STAGE_APPEARANCE,
    STAGE_BACKGROUND,
    STAGE_CODEC,
    STAGE_DETECTION,
    STAGE_GENERATION,
    STAGE_METRICS,
    STAGE_MOTION,
    STAGE_POSE,
    STAGE_RESIDUAL,
    STAGE_SEGMENTATION,
    STAGE_TRANSPORT,
    STAGES,
    StageLattice,
    corner,
    describe_catalogue,
    stage,
)


# --------------------------------------------------------------------------
# The optionality rule
# --------------------------------------------------------------------------


def test_only_the_codec_transport_and_metrics_are_structurally_required():
    """The lattice's value is that every component can be measured by being
    turned off. Anything else marked required removes a corner from every
    ablation table."""
    declared = {name for name, spec in STAGES.items() if spec.required}
    assert declared == REQUIRED_STAGES
    assert declared == {STAGE_CODEC, STAGE_TRANSPORT, STAGE_METRICS}


def test_the_catalogue_carries_all_sixteen_rows():
    assert len(STAGES) == 16
    assert len(OPTIONAL_STAGES) == 13


def test_every_optional_stage_can_actually_be_turned_off():
    """Declared-optional is not the same as switchable; pruning each in turn is
    what proves the lattice has the corners it claims."""
    for name in OPTIONAL_STAGES:
        pruned = FULL.prune(name)
        assert name not in pruned
        pruned.assert_coherent()


def test_every_stage_states_where_its_work_goes_when_it_is_off():
    """That statement is what makes a disabled stage measurable rather than
    merely absent."""
    for spec in STAGES.values():
        assert spec.when_off


def test_switching_off_metrics_is_rejected_rather_than_silently_corrected():
    """A run that reports no quality number cannot be cited, so a config asking
    for one has to fail rather than quietly get metrics anyway."""
    with pytest.raises(ConfigValueError, match=STAGE_METRICS):
        StageLattice(frozenset({STAGE_CODEC, STAGE_TRANSPORT}))


def test_an_unknown_stage_name_is_rejected_with_the_known_ones():
    with pytest.raises(UnknownBackendError) as excinfo:
        StageLattice.of("segmenation")
    assert STAGE_SEGMENTATION in str(excinfo.value)


def test_stage_lookup_suggests_a_near_miss():
    with pytest.raises(UnknownBackendError, match="Did you mean"):
        stage("detecton")


# --------------------------------------------------------------------------
# Named corners
# --------------------------------------------------------------------------


def test_the_all_off_corner_is_the_source_video():
    assert SOURCE_PASSTHROUGH.is_source_passthrough
    assert SOURCE_PASSTHROUGH.enabled == REQUIRED_STAGES
    assert SOURCE_PASSTHROUGH.optional_enabled == ()


def test_the_whole_frame_residual_corner_is_not_the_all_off_corner():
    """They differ in having a coarseness knob, which is a different arm."""
    whole_frame = corner("whole-frame-residual")
    assert whole_frame != SOURCE_PASSTHROUGH
    assert not whole_frame.is_source_passthrough
    assert whole_frame.optional_enabled == (STAGE_RESIDUAL,)


def test_the_generative_only_corner_leaves_nothing_to_absorb_error():
    assert STAGE_RESIDUAL not in GENERATIVE_ONLY
    assert STAGE_GENERATION in GENERATIVE_ONLY


def test_every_named_corner_is_coherent():
    for name, lattice in NAMED_CORNERS.items():
        lattice.assert_coherent()
        assert lattice.label() == name


def test_an_unregistered_corner_names_the_registered_ones():
    with pytest.raises(UnknownBackendError, match="source-passthrough"):
        corner("everything-off")


def test_corners_compare_by_their_enabled_set():
    assert StageLattice.of(STAGE_RESIDUAL) == corner("whole-frame-residual")
    assert StageLattice.all_on() == FULL


# --------------------------------------------------------------------------
# Coherence
# --------------------------------------------------------------------------


def test_generation_without_detection_is_a_contradiction():
    """The generator would have nothing to draw."""
    broken = FULL.disable(STAGE_DETECTION)
    with pytest.raises(ConfigValueError) as excinfo:
        broken.assert_coherent()
    message = str(excinfo.value)
    assert "subjects" in message
    assert STAGE_DETECTION in message


def test_pruning_detection_cascades_to_everything_that_needed_subjects():
    pruned = FULL.prune(STAGE_DETECTION)
    for name in (
        STAGE_DETECTION,
        STAGE_GENERATION,
        STAGE_POSE,
        STAGE_SEGMENTATION,
        STAGE_APPEARANCE,
        STAGE_MOTION,
    ):
        assert name not in pruned
    pruned.assert_coherent()


def test_pruning_detection_leaves_the_background_model_alone():
    """Background modelling does not need subjects, so it is a separate axis and
    the residual only absorbs what actually went away."""
    assert STAGE_BACKGROUND in FULL.prune(STAGE_DETECTION)


def test_a_generator_needing_pose_with_pose_disabled_is_rejected():
    """The cross-axis effect is derived from the declaration, not from matching
    a substring in the generator's name."""
    lattice = FULL.disable(STAGE_POSE)
    with pytest.raises(ConfigValueError) as excinfo:
        lattice.assert_coherent(conditioning=[CONDITION_POSE])
    message = str(excinfo.value)
    assert STAGE_POSE in message
    assert CONDITION_POSE in message


def test_a_generator_needing_masks_with_segmentation_disabled_is_rejected():
    lattice = FULL.disable(STAGE_SEGMENTATION)
    with pytest.raises(ConfigValueError, match=STAGE_SEGMENTATION):
        lattice.assert_coherent(conditioning=[CONDITION_MASK])


def test_declared_conditioning_is_satisfied_when_its_stage_is_on():
    FULL.assert_coherent(conditioning=[CONDITION_POSE, CONDITION_MASK, CONDITION_MOTION_FIELD])


def test_conditioning_declared_with_generation_off_is_rejected():
    """Nothing would consume it, so the config means something other than what
    it says."""
    lattice = FULL.prune(STAGE_GENERATION)
    with pytest.raises(ConfigValueError, match="disabled"):
        lattice.assert_coherent(conditioning=[CONDITION_POSE])


def test_unknown_conditioning_is_rejected_with_the_known_kinds():
    with pytest.raises(ConfigValueError, match="no stage produces"):
        FULL.assert_coherent(conditioning=["depth-map"])


def test_missing_inputs_reports_every_broken_stage_not_just_the_first():
    broken = FULL.disable(STAGE_DETECTION)
    missing = broken.missing_inputs()
    assert STAGE_GENERATION in missing
    assert STAGE_POSE in missing


# --------------------------------------------------------------------------
# The DAG
# --------------------------------------------------------------------------


def test_the_dag_covers_exactly_the_enabled_stages():
    order = FULL.dag()
    assert set(order) == set(FULL.enabled)
    assert len(order) == len(FULL.enabled)


def test_the_dag_runs_producers_before_consumers():
    order = FULL.dag()
    position = {name: index for index, name in enumerate(order)}
    assert position[STAGE_DETECTION] < position[STAGE_POSE]
    assert position[STAGE_POSE] < position[STAGE_GENERATION]
    assert position[STAGE_BACKGROUND] < position[STAGE_GENERATION]
    assert position[STAGE_GENERATION] < position[STAGE_RESIDUAL]
    assert position[STAGE_CODEC] < position[STAGE_TRANSPORT]
    assert position[STAGE_TRANSPORT] < position[STAGE_METRICS]


def test_the_dag_of_the_all_off_corner_is_just_the_required_spine():
    assert SOURCE_PASSTHROUGH.dag() == (STAGE_CODEC, STAGE_TRANSPORT, STAGE_METRICS)


def test_the_dag_is_stable_across_calls_so_corners_stay_comparable():
    assert FULL.dag() == FULL.dag()
    assert StageLattice.all_on().dag() == FULL.dag()


def test_building_a_dag_from_an_incoherent_corner_raises():
    with pytest.raises(ConfigValueError):
        FULL.disable(STAGE_DETECTION).dag()


# --------------------------------------------------------------------------
# Reading a corner back
# --------------------------------------------------------------------------


def test_describe_shows_where_a_disabled_stages_work_goes():
    text = FULL.prune(STAGE_DETECTION).describe()
    assert "they land in the residual" in text
    assert "off" in text


def test_an_unnamed_corner_labels_itself_by_its_enabled_stages():
    label = StageLattice.of(STAGE_BACKGROUND, STAGE_RESIDUAL).label()
    assert STAGE_BACKGROUND in label and STAGE_RESIDUAL in label


def test_the_catalogue_listing_names_every_stage_and_its_off_behaviour():
    text = describe_catalogue()
    for name in STAGES:
        assert name in text
    assert text.count("off ->") == len(STAGES)


# --------------------------------------------------------------------------
# Immutability
# --------------------------------------------------------------------------


def test_enabling_and_disabling_return_new_corners():
    base = StageLattice.all_off()
    grown = base.enable(STAGE_RESIDUAL)
    assert base.is_source_passthrough
    assert STAGE_RESIDUAL in grown
    assert STAGE_RESIDUAL not in base


def test_enabling_an_unknown_stage_raises_before_building_the_corner():
    with pytest.raises(UnknownBackendError):
        StageLattice.all_off().enable("hallucination")
