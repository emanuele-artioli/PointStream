"""The conditioning contract: typed inputs, derived plans, declared capability.

These test the three failures the module exists to prevent — an input arriving
in the wrong slot, one axis deciding another axis's fate by name, and temporal
capability being read from a class rather than from a declaration.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest

from src.contracts.capabilities import (
    ALL_CONDITIONS,
    CAP_PER_FRAME,
    CAP_TEMPORAL_SEQUENCE,
    CONDITION_APPEARANCE,
    CONDITION_CANNY,
    CONDITION_MASK,
    CONDITION_POSE,
)
from src.contracts.lattice import STAGE_DETECTION, STAGE_POSE, STAGE_SEGMENTATION, STAGES
from src.contracts.conditioning import (
    STAGES_FOR_CONDITION,
    TRANSMIT_CANNY,
    TRANSMIT_FOR_CONDITION,
    ArrayLike,
    ConditioningBundle,
    ConditioningPlan,
    FrameGenerator,
    GenerationParams,
    SequenceGenerator,
    require_sequence,
    supports_sequence,
    unused_stages,
)
from src.contracts.errors import MissingConditioningError, UnsupportedCapabilityError
from src.contracts.registry import BackendSpec


@dataclass(frozen=True)
class FakeArray:
    """Something with a shape, standing in for a tensor this layer cannot import."""

    shape: tuple[int, ...] = (3, 64, 32)
    dtype: str = "uint8"


def _spec(name: str, requires: set[str], capabilities: set[str] | None = None) -> BackendSpec[object]:
    return BackendSpec(
        name=name,
        target="tests.contracts.test_conditioning:FakeArray",
        requires=frozenset(requires),
        capabilities=frozenset(capabilities or {CAP_PER_FRAME}),
    )


class PoseOnlyGenerator:
    """A per-frame generator, written the way a real backend would be."""

    def generate(
        self,
        conditioning: ConditioningBundle,
        *,
        seed: int,
        device: Any,
        params: GenerationParams,
    ) -> ArrayLike:
        conditioning.require(CONDITION_POSE, CONDITION_APPEARANCE)
        assert conditioning.pose is not None
        return conditioning.pose


class TemporalGenerator(PoseOnlyGenerator):
    """A generator that really can produce a sequence."""

    def generate_sequence(
        self,
        conditioning: Sequence[ConditioningBundle],
        *,
        seed: int,
        device: Any,
        params: GenerationParams,
    ) -> Sequence[ArrayLike]:
        return [
            self.generate(item, seed=seed, device=device, params=params) for item in conditioning
        ]


# --------------------------------------------------------------------------
# The bundle
# --------------------------------------------------------------------------


def test_require_names_the_missing_input_and_what_was_supplied() -> None:
    """A generator asking for what it declared fails by name, at the call site.

    The whole point of the replacement: the old shared parameter turned this
    into a tensor-shape error deep inside a diffusion pipeline.
    """
    bundle = ConditioningBundle(appearance=FakeArray(), pose=FakeArray())

    with pytest.raises(MissingConditioningError) as caught:
        bundle.require(CONDITION_POSE, CONDITION_MASK)

    assert caught.value.missing == [CONDITION_MASK]
    assert "mask" in str(caught.value)
    assert "pose" in str(caught.value)  # listed as present


def test_present_reports_only_transmitted_kinds() -> None:
    """`previous_frame` is client-side, so it is not a conditioning kind."""
    bundle = ConditioningBundle(appearance=FakeArray(), previous_frame=FakeArray())
    assert bundle.present() == {CONDITION_APPEARANCE}


def test_unknown_conditioning_kind_is_rejected_with_the_vocabulary() -> None:
    with pytest.raises(ValueError, match="Unknown conditioning kind 'dense_dwpose'"):
        ConditioningBundle().require("dense_dwpose")


def test_motion_field_accepts_both_spellings() -> None:
    """The capability vocabulary hyphenates; Python attributes cannot."""
    bundle = ConditioningBundle(motion_field=FakeArray(shape=(2, 64, 32)))
    bundle.require("motion-field")
    bundle.require("motion_field")


def test_a_two_dimensional_array_in_the_pose_slot_is_caught() -> None:
    """The historical bug, in the shape check: a mask handed to the pose field."""
    bundle = ConditioningBundle(pose=FakeArray(shape=(64, 32)))
    with pytest.raises(ValueError, match="'pose'"):
        bundle.validate_shapes()


def test_valid_shapes_pass_and_unknown_shapes_are_ignored() -> None:
    bundle = ConditioningBundle(
        appearance=FakeArray(shape=(3, 64, 32)),
        mask=FakeArray(shape=(64, 32)),
        motion_field=FakeArray(shape=(2, 64, 32)),
    )
    bundle.validate_shapes()


def test_an_inverted_bbox_is_rejected_at_construction() -> None:
    """A zero-area box resizes to nothing far downstream, as an unrelated error."""
    with pytest.raises(ValueError, match="empty or inverted"):
        ConditioningBundle(bbox=(40, 10, 40, 90))


def test_negative_frame_index_is_rejected() -> None:
    with pytest.raises(ValueError, match="frame_index"):
        ConditioningBundle(frame_index=-1)


def test_with_fields_copies_rather_than_mutates() -> None:
    """Frozen because encoder and decoder each hold one; an edit is a divergence."""
    original = ConditioningBundle(appearance=FakeArray(), object_id="player_1")
    updated = original.with_fields(previous_frame=FakeArray())

    assert original.previous_frame is None
    assert updated.previous_frame is not None
    assert updated.object_id == "player_1"


def test_with_fields_rejects_a_field_that_does_not_exist() -> None:
    """A dict would have accepted this silently, and diverged the two sides."""
    with pytest.raises(ValueError, match="dense_dwpose_tensor"):
        ConditioningBundle().with_fields(dense_dwpose_tensor=FakeArray())


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------


def test_a_canny_generator_does_not_make_pose_estimation_run() -> None:
    """The cross-axis leak, replaced.

    Today a module reads `"canny-controlnet" in genai_backend` and nulls the
    pose estimator from the other side of the system. Here the generator
    declares canny and the plan derives the rest.
    """
    plan = ConditioningPlan.derive(_spec("canny-controlnet", {CONDITION_CANNY, CONDITION_APPEARANCE}))

    assert CONDITION_POSE not in plan.conditioning
    assert not plan.needs_stage(STAGE_POSE)
    assert plan.needs_stage(STAGE_DETECTION)
    assert TRANSMIT_CANNY in plan.transmit
    assert STAGE_POSE in unused_stages(plan)


def test_a_multi_condition_generator_enables_every_stage_it_declared() -> None:
    plan = ConditioningPlan.derive(
        _spec("multi-controlnet", {CONDITION_POSE, CONDITION_MASK, CONDITION_APPEARANCE})
    )

    assert plan.needs_stage(STAGE_POSE)
    assert plan.needs_stage(STAGE_SEGMENTATION)
    assert not plan.needs_stage(STAGE_DETECTION)


def test_plan_check_rejects_a_bundle_missing_what_the_generator_declared() -> None:
    """A generator declaring conditioning nothing supplies — the §10 misuse case."""
    plan = ConditioningPlan.derive(_spec("seg-controlnet", {CONDITION_MASK, CONDITION_APPEARANCE}))
    bundle = ConditioningBundle(appearance=FakeArray())

    with pytest.raises(MissingConditioningError) as caught:
        plan.check(bundle)
    assert caught.value.missing == [CONDITION_MASK]


def test_a_registry_entry_with_a_misspelled_requirement_fails_to_plan() -> None:
    """A typo in `requires` would otherwise mean a stage silently never runs."""
    with pytest.raises(ValueError, match="Unknown conditioning kind 'posee'"):
        ConditioningPlan.derive(_spec("typo", {"posee"}))


def test_an_unknown_stage_name_is_rejected_rather_than_answered_false() -> None:
    """A misspelt stage answering False would read as "skip it", not "you typo'd"."""
    plan = ConditioningPlan.derive(_spec("pose-controlnet", {CONDITION_POSE}))
    with pytest.raises(ValueError, match="Unknown stage 'pose-estimation'"):
        plan.needs_stage("pose-estimation")


def test_stage_names_come_from_the_catalogue() -> None:
    """One vocabulary, not two.

    These names were briefly declared twice — once here and once in the stage
    catalogue — with `pose` in one and `pose-estimation` in the other. Two lists
    of the same thing drift, and a stage named twice is the drift this package
    exists to prevent.
    """
    assert set(STAGES_FOR_CONDITION.values()) <= {(name,) for name in STAGES}
    assert STAGES_FOR_CONDITION[CONDITION_POSE] == (STAGE_POSE,)
    assert STAGES_FOR_CONDITION[CONDITION_MASK] == (STAGE_SEGMENTATION,)


def test_every_conditioning_kind_has_a_stage_and_a_wire_cost() -> None:
    """A new kind must be routed, or a plan derived for it would enable nothing."""
    assert set(STAGES_FOR_CONDITION) == ALL_CONDITIONS
    assert set(TRANSMIT_FOR_CONDITION) == ALL_CONDITIONS


# --------------------------------------------------------------------------
# Temporal capability
# --------------------------------------------------------------------------


def test_temporal_capability_comes_from_the_declaration_not_the_class() -> None:
    """Any backend can be temporal; it does not have to subclass one that is."""
    temporal = _spec("some-new-video-model", set(), {CAP_PER_FRAME, CAP_TEMPORAL_SEQUENCE})

    assert supports_sequence(temporal)
    assert ConditioningPlan.derive(temporal).temporal
    require_sequence(temporal, TemporalGenerator())


def test_a_per_frame_generator_asked_for_a_sequence_is_refused() -> None:
    per_frame = _spec("baseline-controlnet", {CONDITION_POSE})

    assert not supports_sequence(per_frame)
    with pytest.raises(UnsupportedCapabilityError, match="temporal-sequence"):
        require_sequence(per_frame, TemporalGenerator())


def test_a_backend_that_declares_sequences_but_cannot_do_them_is_caught() -> None:
    """A lying registry entry passes config validation and fails mid-run."""
    lying = _spec("claims-temporal", set(), {CAP_PER_FRAME, CAP_TEMPORAL_SEQUENCE})

    with pytest.raises(UnsupportedCapabilityError, match="generate_sequence"):
        require_sequence(lying, PoseOnlyGenerator())


def test_the_protocols_match_structurally() -> None:
    assert isinstance(PoseOnlyGenerator(), FrameGenerator)
    assert not isinstance(PoseOnlyGenerator(), SequenceGenerator)
    assert isinstance(TemporalGenerator(), SequenceGenerator)


# --------------------------------------------------------------------------
# Per-call parameters
# --------------------------------------------------------------------------


def test_generation_params_reject_values_that_would_silently_do_nothing() -> None:
    with pytest.raises(ValueError, match="strength"):
        GenerationParams(strength=1.5)
    with pytest.raises(ValueError, match="steps"):
        GenerationParams(steps=0)
    with pytest.raises(ValueError, match="width"):
        GenerationParams(width=0)
