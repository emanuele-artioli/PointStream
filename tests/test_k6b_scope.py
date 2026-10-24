"""Boundary indexing matters: the reference must be compared with its shot."""
from experiments.tennis.k6b_scope import parse_scenes, pose_distance, shot_id, spans


def test_cut_reference_membership():
    cuts = [32, 140]
    assert shot_id(31, cuts) == 0
    assert shot_id(32, cuts) == 1
    assert shot_id(140, cuts) == 2
    assert shot_id(144, cuts) == shot_id(140, cuts)
    assert spans([140, 142, 141, 150]) == [[140, 142], [150, 150]]


def test_scene_metadata_and_pose_translation():
    assert parse_scenes('frame:1 pts:1\nlavfi.scene_score=0.432100\n') == [.4321]
    a = [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    b = [[10, 10, 1], [11, 10, 1], [10, 11, 1], [11, 11, 1]]
    assert pose_distance(a, b, 100) == 0
    assert pose_distance(a, [[0, 0, 0]] * 4, 100) is None
