from __future__ import annotations

import pytest

from src.encoder import match_orchestrator as mo
from src.shared.schemas import SceneClass, SceneSpan


def _span(t_start: float, t_end: float, scene_class: SceneClass = SceneClass.POINT) -> SceneSpan:
    return SceneSpan(
        t_start=t_start,
        t_end=t_end,
        scene_class=scene_class,
        confidence=1.0,
        avg_score=0.001,
        std_score=0.001,
        max_score=0.001,
    )


class TestSplitSceneIntoSubchunks:
    def test_exact_multiple_of_chunk_duration(self) -> None:
        boundaries = mo.split_scene_into_subchunks(0.0, 6.0, chunk_duration=2.0)
        assert boundaries == [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0)]

    def test_final_subchunk_is_shorter(self) -> None:
        boundaries = mo.split_scene_into_subchunks(0.0, 5.0, chunk_duration=2.0)
        assert boundaries == [(0.0, 2.0), (2.0, 4.0), (4.0, 5.0)]

    def test_scene_shorter_than_chunk_duration_yields_one_subchunk(self) -> None:
        boundaries = mo.split_scene_into_subchunks(10.0, 11.5, chunk_duration=2.0)
        assert boundaries == [(10.0, 11.5)]

    def test_offset_start_time(self) -> None:
        boundaries = mo.split_scene_into_subchunks(100.0, 104.0, chunk_duration=2.0)
        assert boundaries == [(100.0, 102.0), (102.0, 104.0)]

    def test_empty_span_yields_no_subchunks(self) -> None:
        assert mo.split_scene_into_subchunks(5.0, 5.0, chunk_duration=2.0) == []

    def test_nonpositive_chunk_duration_raises(self) -> None:
        with pytest.raises(ValueError):
            mo.split_scene_into_subchunks(0.0, 5.0, chunk_duration=0.0)

    def test_subchunks_tile_the_span_with_no_gaps(self) -> None:
        boundaries = mo.split_scene_into_subchunks(3.3, 17.9, chunk_duration=2.0)
        assert boundaries[0][0] == 3.3
        assert boundaries[-1][1] == pytest.approx(17.9)
        for (_, end_prev), (start_next, _) in zip(boundaries, boundaries[1:]):
            assert end_prev == start_next


class TestChooseRouting:
    def test_semantic_wins_when_smaller(self) -> None:
        assert mo.choose_routing(semantic_bytes=100, fallback_bytes=200) == "semantic"

    def test_fallback_wins_when_smaller(self) -> None:
        assert mo.choose_routing(semantic_bytes=300, fallback_bytes=200) == "fallback"

    def test_ties_go_to_semantic(self) -> None:
        assert mo.choose_routing(semantic_bytes=150, fallback_bytes=150) == "semantic"

    def test_semantic_not_attempted_forces_fallback(self) -> None:
        assert mo.choose_routing(semantic_bytes=None, fallback_bytes=999) == "fallback"


class TestAssertScenesTileVideo:
    def test_contiguous_scenes_pass(self) -> None:
        scenes = [_span(0.0, 4.0), _span(4.0, 8.0), _span(8.0, 10.0)]
        mo.assert_scenes_tile_video(scenes, video_duration_sec=10.0)  # no raise

    def test_unsorted_input_is_still_validated_correctly(self) -> None:
        scenes = [_span(4.0, 8.0), _span(0.0, 4.0), _span(8.0, 10.0)]
        mo.assert_scenes_tile_video(scenes, video_duration_sec=10.0)  # no raise

    def test_gap_between_scenes_raises(self) -> None:
        scenes = [_span(0.0, 4.0), _span(5.0, 10.0)]
        with pytest.raises(ValueError, match="gap/overlap"):
            mo.assert_scenes_tile_video(scenes, video_duration_sec=10.0, tolerance_sec=0.1)

    def test_overlap_between_scenes_raises(self) -> None:
        scenes = [_span(0.0, 5.0), _span(4.0, 10.0)]
        with pytest.raises(ValueError, match="gap/overlap"):
            mo.assert_scenes_tile_video(scenes, video_duration_sec=10.0, tolerance_sec=0.1)

    def test_missing_tail_coverage_raises(self) -> None:
        scenes = [_span(0.0, 4.0), _span(4.0, 6.0)]
        with pytest.raises(ValueError, match="video duration"):
            mo.assert_scenes_tile_video(scenes, video_duration_sec=10.0)

    def test_first_scene_not_starting_at_zero_raises(self) -> None:
        scenes = [_span(3.0, 10.0)]
        with pytest.raises(ValueError, match="expected ~0.0s"):
            mo.assert_scenes_tile_video(scenes, video_duration_sec=10.0)

    def test_no_scenes_raises(self) -> None:
        with pytest.raises(ValueError, match="No scenes"):
            mo.assert_scenes_tile_video([], video_duration_sec=10.0)

    def test_small_rounding_slop_within_tolerance_passes(self) -> None:
        scenes = [_span(0.0, 4.0), _span(4.02, 10.0)]
        mo.assert_scenes_tile_video(scenes, video_duration_sec=10.0, tolerance_sec=0.5)
