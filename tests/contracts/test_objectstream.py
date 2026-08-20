"""Behaviour and plausible misuse of the object stream contract."""

import pytest

from src.contracts import capabilities as caps
from src.contracts.codecs import EncodeRequest, RateControl
from src.contracts.errors import ConfigValueError, UndecodableStreamError
from src.contracts.keypoints import COCO_17, COCO_WHOLEBODY_133, OPENPOSE_18, wire_cost
from src.contracts.objectstream import (
    MAX_SPARSE_POINTS,
    CompressedImage,
    DiffusionLatent,
    EncodedVideoMotion,
    FrameAction,
    FrameDecision,
    ImageEmbedding,
    KeypointMotion,
    ObjectStream,
    Sparsity,
    SparseTrajectories,
    TemporalPolicy,
    TemporalSchedule,
    WireCost,
    assert_decodable,
    decodable_by,
    workable_pairings,
)
from src.contracts.registry import BackendSpec, Registry


def generators() -> Registry[object]:
    """A registry standing in for the generator axis.

    Deliberately partial: the pose ControlNet takes an image but not a latent,
    and nothing at all accepts an embedding paired with encoded video. That gap
    is what the pairing check exists to find.
    """
    registry: Registry[object] = Registry("generator")
    registry.register(
        BackendSpec(
            name="controlnet-pose",
            target="fake.module:ControlNetPose",
            capabilities=caps.appearance(caps.APPEARANCE_COMPRESSED_IMAGE)
            | caps.motion(caps.MOTION_KEYPOINTS)
            | {caps.CAP_PER_FRAME},
            requires=frozenset({caps.CONDITION_POSE}),
        )
    )
    registry.register(
        BackendSpec(
            name="mofa-trajectories",
            target="fake.module:Mofa",
            capabilities=caps.appearance(
                caps.APPEARANCE_COMPRESSED_IMAGE, caps.APPEARANCE_DIFFUSION_LATENT
            )
            | caps.motion(caps.MOTION_VECTORS)
            | {caps.CAP_TEMPORAL_SEQUENCE},
            requires=frozenset({caps.CONDITION_MOTION_FIELD}),
        )
    )
    registry.register(
        BackendSpec(
            name="upscale-refine",
            target="fake.module:UpscaleRefine",
            capabilities=caps.appearance(caps.APPEARANCE_COMPRESSED_IMAGE)
            | caps.motion(caps.MOTION_ENCODED_VIDEO),
        )
    )
    return registry


# --------------------------------------------------------------------------
# Wire cost
# --------------------------------------------------------------------------


def test_keypoint_motion_cost_matches_the_schema_wire_cost():
    motion = KeypointMotion(COCO_WHOLEBODY_133)
    assert motion.cost().values == wire_cost(COCO_WHOLEBODY_133)
    assert motion.cost().byte_count == wire_cost(COCO_WHOLEBODY_133) * 2


def test_a_leaner_wire_schema_costs_strictly_less():
    """Schema richness is an ablation axis, so it has to move the number."""
    rich = KeypointMotion(COCO_WHOLEBODY_133).cost()
    lean = KeypointMotion(OPENPOSE_18).cost()
    assert lean.values is not None and rich.values is not None
    assert lean.values < rich.values


def test_downscale_and_quality_are_independent_knobs():
    """Collapsing them into one degradation scalar would lose the comparison."""
    base = CompressedImage(200, 400, quality=90, measured_bytes=8000)
    downscaled = CompressedImage(200, 400, quality=90, downscale=0.5, measured_bytes=8000)
    lower_quality = CompressedImage(200, 400, quality=30, measured_bytes=8000)

    assert base.transmitted_size == (200, 400)
    assert downscaled.transmitted_size == (100, 200)
    assert lower_quality.transmitted_size == base.transmitted_size
    assert downscaled.cost().values != base.cost().values


def test_unmeasured_entropy_coded_appearance_reports_no_byte_figure():
    """A plausible JPEG-size model would be a fictional number, so there is none."""
    cost = CompressedImage(64, 128, quality=75).cost()
    assert cost.byte_count is None
    assert cost.exact is False
    assert "unmeasured" in cost.basis


def test_unknown_bytes_propagate_through_a_total_rather_than_counting_as_zero():
    """Undercounting payload is the one direction of error that cannot be afforded."""
    known = ImageEmbedding(512).cost()
    unknown = CompressedImage(64, 64).cost()
    total = known + unknown
    assert total.byte_count is None
    assert total.exact is False


def test_latent_and_embedding_state_their_size_exactly():
    latent = DiffusionLatent(4, 32, 24, bytes_per_value=2).cost()
    assert latent.values == 4 * 32 * 24
    assert latent.byte_count == 4 * 32 * 24 * 2
    assert latent.exact

    embedding = ImageEmbedding(768, tokens=4).cost()
    assert embedding.byte_count == 768 * 4 * 2


def test_wire_cost_scales_over_a_span():
    scaled = WireCost(values=10, byte_count=20, basis="unit").scaled(5)
    assert (scaled.values, scaled.byte_count) == (50, 100)


def test_scaling_a_cost_by_a_negative_span_is_rejected():
    with pytest.raises(ValueError):
        WireCost(values=1, byte_count=1).scaled(-1)


# --------------------------------------------------------------------------
# Sparse trajectories, not dense flow
# --------------------------------------------------------------------------


def test_sparse_trajectories_are_the_same_order_as_a_skeleton():
    """The claim the design rests on: trajectories cost what keypoints cost."""
    trajectories = SparseTrajectories(point_count=64).cost()
    skeleton = KeypointMotion(COCO_17).cost()
    assert trajectories.values is not None and skeleton.values is not None
    assert 0.1 < trajectories.values / skeleton.values < 10


def test_dense_flow_wearing_a_sparse_name_is_rejected():
    """Above the ceiling this costs what block motion vectors cost, which is
    exactly the expense the representation exists to avoid."""
    with pytest.raises(ValueError, match="dense flow"):
        SparseTrajectories(point_count=MAX_SPARSE_POINTS + 1)


def test_a_trajectory_set_needs_at_least_one_point():
    with pytest.raises(ValueError):
        SparseTrajectories(point_count=0)


def test_encoded_video_motion_carries_a_real_encode_request():
    """So the classical arm is subject to the same constraint checks as any
    other encode, rather than being a differently-configured strawman."""
    request = EncodeRequest(codec_name="av1", rate_control=RateControl.CRF, rate=30)
    motion = EncodedVideoMotion(request=request, width=96, height=192)
    motion.request.validate()
    assert motion.kind == caps.MOTION_ENCODED_VIDEO
    assert motion.cost().byte_count is None


# --------------------------------------------------------------------------
# Descriptor misuse
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"quality": 0},
        {"quality": 101},
        {"downscale": 0.0},
        {"downscale": 1.5},
    ],
)
def test_compressed_image_rejects_impossible_degradation_knobs(kwargs):
    with pytest.raises(ValueError):
        CompressedImage(64, 64, **kwargs)


def test_upscaling_an_object_crop_is_rejected_rather_than_silently_allowed():
    with pytest.raises(ValueError, match="adds no information"):
        CompressedImage(64, 64, downscale=2.0)


# --------------------------------------------------------------------------
# The pairing constraint
# --------------------------------------------------------------------------


def test_a_registered_pairing_resolves_to_the_generator_that_accepts_it():
    spec = assert_decodable(
        caps.APPEARANCE_DIFFUSION_LATENT, caps.MOTION_VECTORS, generators()
    )
    assert spec.name == "mofa-trajectories"


def test_an_undecodable_pairing_is_rejected_and_names_what_would_work():
    registry = generators()
    with pytest.raises(UndecodableStreamError) as excinfo:
        assert_decodable(
            caps.APPEARANCE_IMAGE_EMBEDDING, caps.MOTION_ENCODED_VIDEO, registry
        )
    message = str(excinfo.value)
    assert caps.APPEARANCE_IMAGE_EMBEDDING in message
    assert "compressed-image+keypoints" in message


def test_a_pairing_each_half_of_which_is_supported_can_still_be_undecodable():
    """The trap the check exists for: both halves are registered, but not by the
    same generator, so nothing can actually decode the combination."""
    registry = generators()
    assert decodable_by(caps.APPEARANCE_DIFFUSION_LATENT, caps.MOTION_VECTORS, registry)
    assert decodable_by(caps.APPEARANCE_COMPRESSED_IMAGE, caps.MOTION_KEYPOINTS, registry)
    assert not decodable_by(
        caps.APPEARANCE_DIFFUSION_LATENT, caps.MOTION_KEYPOINTS, registry
    )


def test_pairings_that_exist_only_in_the_vocabulary_are_not_reported_as_workable():
    pairs = workable_pairings(generators())
    assert (caps.APPEARANCE_COMPRESSED_IMAGE, caps.MOTION_KEYPOINTS) in pairs
    assert (caps.APPEARANCE_IMAGE_EMBEDDING, caps.MOTION_KEYPOINTS) not in pairs


def test_an_appearance_outside_the_vocabulary_is_named_as_such():
    """Canny is the motivating case: a tempting appearance option that discards
    colour entirely, so it belongs on the conditioning side and nowhere else."""
    with pytest.raises(ConfigValueError, match="not an appearance representation"):
        assert_decodable("canny", caps.MOTION_KEYPOINTS, generators())


def test_a_motion_outside_the_vocabulary_is_named_as_such():
    with pytest.raises(ConfigValueError, match="not a motion representation"):
        assert_decodable(caps.APPEARANCE_COMPRESSED_IMAGE, "optical-flow", generators())


# --------------------------------------------------------------------------
# Temporal policy
# --------------------------------------------------------------------------


def test_the_dense_policy_transmits_and_generates_every_frame():
    schedule = TemporalPolicy().plan(frame_count=5, object_id="p1")
    assert [item.action for item in schedule.for_object("p1")] == [FrameAction.FULL] * 5
    assert schedule.transmitted_frames("p1") == (0, 1, 2, 3, 4)


def test_metadata_sparsity_interpolates_between_keyframes():
    policy = TemporalPolicy(metadata=Sparsity(stride=3))
    schedule = policy.plan(frame_count=7, object_id="p1")
    actions = [item.action for item in schedule.for_object("p1")]
    assert schedule.transmitted_frames("p1") == (0, 3, 6)
    assert actions[1] is FrameAction.INTERPOLATE
    assert schedule.for_object("p1")[1].anchor == 0
    assert schedule.for_object("p1")[1].target == 3
    schedule.validate()


def test_frames_after_the_last_keyframe_hold_rather_than_interpolating():
    """There is nothing to reach toward, so pretending to interpolate would mean
    inventing a target."""
    schedule = TemporalPolicy(metadata=Sparsity(stride=3)).plan(frame_count=5, object_id="p1")
    tail = schedule.for_object("p1")[4]
    assert tail.action is FrameAction.HOLD
    assert tail.target is None


def test_pipeline_sparsity_bounds_metadata_sparsity():
    """A frame perception skipped has nothing to transmit, whatever the metadata
    level would have preferred."""
    dense_metadata = TemporalPolicy(pipeline=Sparsity(stride=4))
    schedule = dense_metadata.plan(frame_count=9, object_id="p1")
    assert schedule.transmitted_frames("p1") == (0, 4, 8)


def test_generation_sparsity_runs_the_model_at_a_subset_of_keyframes():
    policy = TemporalPolicy(metadata=Sparsity(stride=2), generation=Sparsity(stride=2))
    schedule = policy.plan(frame_count=9, object_id="p1")
    generated = [
        item.frame_index
        for item in schedule.for_object("p1")
        if item.action is FrameAction.FULL
    ]
    transmitted = schedule.transmitted_frames("p1")
    assert set(generated) <= set(transmitted)
    assert generated == [0, 4, 8]


def test_a_preroll_keeps_the_opening_keyframes_residual_only():
    policy = TemporalPolicy(preroll_frames=2)
    actions = [item.action for item in policy.plan(frame_count=5, object_id="p1").for_object("p1")]
    assert actions[:2] == [FrameAction.TRANSMIT_ONLY, FrameAction.TRANSMIT_ONLY]
    assert actions[2] is FrameAction.FULL


def test_motion_adaptive_thresholds_follow_the_content():
    """The point of replacing a hard-coded threshold: a fast passage gets more
    keyframes than a slow one, from the same config."""
    policy = TemporalPolicy(metadata=Sparsity(threshold=2.0))
    slow = policy.plan(frame_count=6, object_id="p1", motion=[0.0] + [0.4] * 5)
    fast = policy.plan(frame_count=6, object_id="p1", motion=[0.0] + [3.0] * 5)
    assert len(slow.transmitted_frames("p1")) < len(fast.transmitted_frames("p1"))


def test_a_relative_threshold_is_measured_against_the_clips_own_motion():
    policy = TemporalPolicy(
        metadata=Sparsity(threshold=1.5, threshold_relative_to_scene_motion=True)
    )
    schedule = policy.plan(frame_count=6, object_id="p1", motion=[1.0] * 6)
    # Mean motion is 1.0, so the threshold is 1.5 and two frames must accumulate.
    assert schedule.transmitted_frames("p1") == (0, 2, 4)


def test_an_adaptive_policy_without_measured_motion_is_rejected():
    """Falling back to the fixed-stride behaviour would silently reinstate the
    constant the adaptive threshold exists to replace."""
    policy = TemporalPolicy(metadata=Sparsity(threshold=5.0))
    with pytest.raises(ConfigValueError, match="no measured"):
        policy.plan(frame_count=4, object_id="p1")


def test_a_motion_series_of_the_wrong_length_is_rejected():
    policy = TemporalPolicy(metadata=Sparsity(threshold=5.0))
    with pytest.raises(ConfigValueError, match="3 entries for 4 frames"):
        policy.plan(frame_count=4, object_id="p1", motion=[1.0, 1.0, 1.0])


def test_a_relative_threshold_without_a_threshold_is_rejected():
    with pytest.raises(ValueError, match="no threshold was given"):
        Sparsity(threshold_relative_to_scene_motion=True)


@pytest.mark.parametrize("stride", [0, -3])
def test_a_stride_below_one_is_rejected(stride):
    with pytest.raises(ValueError):
        Sparsity(stride=stride)


# --------------------------------------------------------------------------
# Discontinuities
# --------------------------------------------------------------------------


def test_planning_restarts_at_every_discontinuity():
    policy = TemporalPolicy(metadata=Sparsity(stride=4))
    schedule = policy.plan(frame_count=10, object_id="p1", discontinuities=[5])
    assert 5 in schedule.transmitted_frames("p1")
    schedule.validate()


def test_nothing_interpolates_across_a_cut_in_a_planned_schedule():
    policy = TemporalPolicy(metadata=Sparsity(stride=4))
    schedule = policy.plan(frame_count=10, object_id="p1", discontinuities=[5])
    for item in schedule.for_object("p1"):
        if item.anchor is None:
            continue
        end = item.target if item.target is not None else item.frame_index
        assert not schedule.crosses_discontinuity(item.anchor, end)


def test_a_hand_written_interpolation_across_a_cut_is_rejected():
    """Ablations get schedules written by hand, so validate cannot assume plan
    produced them. Across a cut the anchor describes different content, so the
    prediction is confidently wrong rather than merely worse."""
    schedule = TemporalSchedule(
        decisions=(
            FrameDecision(0, "p1", FrameAction.FULL),
            FrameDecision(1, "p1", FrameAction.INTERPOLATE, anchor=0, target=4),
            FrameDecision(4, "p1", FrameAction.FULL),
        ),
        discontinuities=frozenset({3}),
    )
    with pytest.raises(ConfigValueError, match="crosses a discontinuity"):
        schedule.validate()


def test_a_hold_across_a_cut_is_rejected_too():
    schedule = TemporalSchedule(
        decisions=(
            FrameDecision(0, "p1", FrameAction.FULL),
            FrameDecision(4, "p1", FrameAction.HOLD, anchor=0),
        ),
        discontinuities=frozenset({3}),
    )
    with pytest.raises(ConfigValueError, match="crosses a discontinuity"):
        schedule.validate()


def test_an_anchor_exactly_at_the_cut_is_allowed():
    """The cut's own frame starts the new span, so it is a legitimate anchor."""
    schedule = TemporalSchedule(
        decisions=(
            FrameDecision(3, "p1", FrameAction.FULL),
            FrameDecision(4, "p1", FrameAction.HOLD, anchor=3),
        ),
        discontinuities=frozenset({3}),
    )
    schedule.validate()


# --------------------------------------------------------------------------
# Schedule misuse
# --------------------------------------------------------------------------


def test_an_interpolation_with_no_anchor_is_rejected():
    schedule = TemporalSchedule(
        decisions=(FrameDecision(2, "p1", FrameAction.INTERPOLATE, target=4),)
    )
    with pytest.raises(ConfigValueError, match="anchor"):
        schedule.validate()


def test_an_interpolation_with_no_target_is_rejected():
    schedule = TemporalSchedule(
        decisions=(FrameDecision(2, "p1", FrameAction.INTERPOLATE, anchor=0),)
    )
    with pytest.raises(ConfigValueError, match="reaches toward"):
        schedule.validate()


def test_an_anchor_that_is_not_earlier_is_rejected():
    schedule = TemporalSchedule(
        decisions=(FrameDecision(2, "p1", FrameAction.HOLD, anchor=5),)
    )
    with pytest.raises(ConfigValueError, match="not before"):
        schedule.validate()


def test_a_transmitted_frame_that_claims_an_anchor_is_rejected():
    """It derives from nothing; an anchor there means the two sides disagree
    about what the frame is."""
    schedule = TemporalSchedule(
        decisions=(FrameDecision(2, "p1", FrameAction.FULL, anchor=0),)
    )
    with pytest.raises(ConfigValueError, match="must not name an anchor"):
        schedule.validate()


def test_two_decisions_for_one_object_on_one_frame_are_rejected():
    schedule = TemporalSchedule(
        decisions=(
            FrameDecision(0, "p1", FrameAction.FULL),
            FrameDecision(0, "p1", FrameAction.TRANSMIT_ONLY),
        )
    )
    with pytest.raises(ConfigValueError, match="two decisions"):
        schedule.validate()


def test_objects_are_scheduled_independently():
    schedule = TemporalSchedule(
        decisions=(
            FrameDecision(0, "p1", FrameAction.FULL),
            FrameDecision(0, "p2", FrameAction.FULL),
            FrameDecision(1, "p1", FrameAction.HOLD, anchor=0),
            FrameDecision(1, "p2", FrameAction.FULL),
        )
    )
    schedule.validate()
    assert schedule.transmitted_frames("p1") == (0,)
    assert schedule.transmitted_frames("p2") == (0, 1)


# --------------------------------------------------------------------------
# The stream
# --------------------------------------------------------------------------


def stream(
    appearance=None,
    motion=None,
    policy=None,
) -> ObjectStream:
    return ObjectStream(
        object_id="p1",
        object_class="player",
        appearance=ImageEmbedding(1024) if appearance is None else appearance,
        motion=KeypointMotion(OPENPOSE_18) if motion is None else motion,
        policy=TemporalPolicy() if policy is None else policy,
    )


def test_total_cost_is_appearance_once_plus_motion_per_transmitted_frame():
    subject = stream(policy=TemporalPolicy(metadata=Sparsity(stride=4)))
    schedule = subject.plan(frame_count=9)
    assert schedule.transmitted_frames("p1") == (0, 4, 8)

    total = subject.total_cost(schedule)
    expected = subject.setup_cost().byte_count + 3 * subject.per_frame_cost().byte_count
    assert total.byte_count == expected


def test_metadata_sparsity_reduces_total_payload():
    """The whole justification for the temporal axis, in the ranking currency."""
    dense = stream()
    sparse = stream(policy=TemporalPolicy(metadata=Sparsity(stride=4)))
    dense_total = dense.total_cost(dense.plan(frame_count=16)).byte_count
    sparse_total = sparse.total_cost(sparse.plan(frame_count=16)).byte_count
    assert sparse_total < dense_total


def test_a_stream_validates_against_the_generator_registry():
    subject = stream(appearance=CompressedImage(64, 128), motion=KeypointMotion(OPENPOSE_18))
    assert subject.validate(generators()).name == "controlnet-pose"


def test_a_stream_nothing_can_decode_is_rejected_at_validation():
    subject = stream(appearance=ImageEmbedding(768), motion=KeypointMotion(OPENPOSE_18))
    with pytest.raises(UndecodableStreamError):
        subject.validate(generators())


def test_describe_names_both_representations_and_the_policy():
    text = stream(policy=TemporalPolicy(metadata=Sparsity(stride=5))).describe()
    assert caps.APPEARANCE_IMAGE_EMBEDDING in text
    assert caps.MOTION_KEYPOINTS in text
    assert "every 5 frames" in text
