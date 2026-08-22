"""Temporal policy: three composing levels, one schedule that travels in the payload."""

from __future__ import annotations

import msgpack
import pytest

from src.components.temporal import REGISTRY as TEMPORAL
from src.components.temporal.policy import ConfigurableTemporalPolicy, SceneCutMarks
from src.components.transport.disk import DiskTransport
from src.components.transport.payload import ChunkPayload, dump_schedule
from src.contracts.config import TemporalConfig
from src.contracts.errors import ConfigValueError
from src.contracts.objectstream import FrameAction


def _policy(**flags: bool) -> ConfigurableTemporalPolicy:
    return ConfigurableTemporalPolicy(
        TemporalConfig(
            keyframe_interval=4,
            delta_threshold=20.0,
            preroll_frames=0,
            metadata_sparsity=flags.get("metadata_sparsity", False),
            generation_sparsity=flags.get("generation_sparsity", False),
            pipeline_sparsity=flags.get("pipeline_sparsity", False),
        )
    )


def _plan(policy: ConfigurableTemporalPolicy, n: int = 12, motion: list[float] | None = None):
    series = motion if motion is not None else [0.0] * n
    return policy.plan(frame_count=n, object_ids=("p1",), motion=series)


def test_three_sparsity_levels_are_selectable_and_compose() -> None:
    n = 12
    dense = _plan(_policy())
    meta = _plan(_policy(metadata_sparsity=True))
    gen = _plan(_policy(generation_sparsity=True))
    pipe = _plan(_policy(pipeline_sparsity=True))
    all_three = _plan(
        _policy(metadata_sparsity=True, generation_sparsity=True, pipeline_sparsity=True)
    )

    dense_actions = [item.action for item in dense.schedule.for_object("p1")]
    assert dense_actions == [FrameAction.FULL] * n
    assert dense.perception_count("p1") == n

    assert meta.schedule.transmitted_frames("p1") == (0, 4, 8)
    assert meta.perception_count("p1") == n
    assert meta.schedule.for_object("p1")[1].action is FrameAction.INTERPOLATE

    generated = [
        item.frame_index
        for item in gen.schedule.for_object("p1")
        if item.action is FrameAction.FULL
    ]
    assert gen.schedule.transmitted_frames("p1") == tuple(range(n))
    assert generated == [0, 4, 8]
    assert set(generated) <= set(gen.schedule.transmitted_frames("p1"))

    assert pipe.perception_frames("p1") == (0, 4, 8)
    assert pipe.schedule.transmitted_frames("p1") == (0, 4, 8)

    assert all_three.perception_frames("p1") == (0, 4, 8)
    assert all_three.schedule.transmitted_frames("p1") == (0, 4, 8)
    full = [
        item.frame_index
        for item in all_three.schedule.for_object("p1")
        if item.action is FrameAction.FULL
    ]
    assert full == [0]
    assert set(full) <= set(all_three.schedule.transmitted_frames("p1"))
    assert set(all_three.schedule.transmitted_frames("p1")) <= set(
        all_three.perception_frames("p1")
    )


def test_reduced_pipeline_sparsity_schedules_fewer_perception_frames() -> None:
    """The encode-time saving is perception skipped, not a flag that nothing honours."""
    n = 16
    dense = _plan(_policy(), n=n)
    sparse = _plan(_policy(pipeline_sparsity=True), n=n)
    assert sparse.perception_count("p1") < dense.perception_count("p1")
    assert sparse.perception_count("p1") == 4

    def honour(planned) -> int:
        perceive = set(planned.perception_frames("p1"))
        calls = 0
        for index in range(n):
            if index in perceive:
                calls += 1
        return calls

    assert honour(sparse) < honour(dense)
    assert honour(sparse) == sparse.perception_count("p1")


def test_named_backends_select_the_three_levels() -> None:
    meta = TEMPORAL.build("metadata-sparsity")
    gen = TEMPORAL.build("generation-sparsity")
    pipe = TEMPORAL.build("pipeline-sparsity")
    none = TEMPORAL.build("none")
    assert meta.config.metadata_sparsity and not meta.config.pipeline_sparsity
    assert gen.config.generation_sparsity and not gen.config.metadata_sparsity
    assert pipe.config.pipeline_sparsity and not pipe.config.generation_sparsity
    assert none.policy.is_dense


def test_schedule_round_trips_bit_identically_in_the_payload(tmp_path) -> None:
    """Encoder and decoder share the serialized decision. They do not re-plan."""
    planned = _plan(_policy(metadata_sparsity=True, pipeline_sparsity=True), n=12)
    transport = DiskTransport(root=tmp_path)
    transport.send(ChunkPayload(chunk_id="c0", schedule=planned))
    received = transport.receive("c0")
    sent_bytes = msgpack.packb(dump_schedule(planned), use_bin_type=True)
    got_bytes = msgpack.packb(dump_schedule(received.schedule), use_bin_type=True)
    assert got_bytes == sent_bytes

    decoder_config = ConfigurableTemporalPolicy(
        TemporalConfig(
            metadata_sparsity=False,
            generation_sparsity=False,
            pipeline_sparsity=False,
            keyframe_interval=4,
            delta_threshold=20.0,
        )
    )
    if_they_replanned = decoder_config.plan(frame_count=12, object_ids=("p1",))
    assert dump_schedule(if_they_replanned) != dump_schedule(received.schedule)


def test_nothing_interpolates_across_a_scene_cut() -> None:
    cuts = SceneCutMarks(frames=(5,)).as_discontinuities()
    planned = _policy(metadata_sparsity=True).plan(
        frame_count=16,
        object_ids=("p1",),
        motion=[0.0] * 16,
        discontinuities=cuts,
    )
    assert 5 in planned.schedule.transmitted_frames("p1")
    for item in planned.schedule.for_object("p1"):
        if item.anchor is None:
            continue
        end = item.target if item.target is not None else item.frame_index
        assert not planned.schedule.crosses_discontinuity(item.anchor, end)
    planned.schedule.validate()


def test_motion_adaptive_threshold_differs_for_slow_vs_fast_motion() -> None:
    """A slow rally and a fast exchange must not share a keyframe density."""
    policy = ConfigurableTemporalPolicy(
        TemporalConfig(
            metadata_sparsity=True,
            generation_sparsity=False,
            pipeline_sparsity=False,
            keyframe_interval=16,
            delta_threshold=8.0,
        )
    )
    slow = [1.0] * 16
    fast = [12.0] * 16
    assert policy.adapted_threshold(slow) != policy.adapted_threshold(fast)

    slow_plan = policy.plan(frame_count=16, object_ids=("p1",), motion=slow)
    fast_plan = policy.plan(frame_count=16, object_ids=("p1",), motion=fast)
    assert slow_plan.adapted_threshold != fast_plan.adapted_threshold
    assert len(fast_plan.schedule.transmitted_frames("p1")) > len(
        slow_plan.schedule.transmitted_frames("p1")
    )

    pipeline = ConfigurableTemporalPolicy(
        TemporalConfig(
            metadata_sparsity=False,
            generation_sparsity=False,
            pipeline_sparsity=True,
            keyframe_interval=16,
            delta_threshold=8.0,
        )
    )
    slow_pipe = pipeline.plan(frame_count=16, object_ids=("p1",), motion=slow)
    fast_pipe = pipeline.plan(frame_count=16, object_ids=("p1",), motion=fast)
    assert fast_pipe.perception_count("p1") > slow_pipe.perception_count("p1")


def test_adaptive_plan_without_motion_is_rejected() -> None:
    """Falling back to stride-only would restore the constant this replaces."""
    with pytest.raises(ConfigValueError, match="no measured"):
        _policy(metadata_sparsity=True).plan(frame_count=4, object_ids=("p1",))


def test_a_motion_series_of_the_wrong_length_is_rejected() -> None:
    with pytest.raises(ConfigValueError, match="3 entries for 4 frames"):
        _policy(metadata_sparsity=True).plan(
            frame_count=4, object_ids=("p1",), motion=[1.0, 1.0, 1.0]
        )


def test_plan_without_an_object_is_rejected() -> None:
    with pytest.raises(ConfigValueError, match="at least one object_id"):
        _policy().plan(frame_count=4, object_ids=())
