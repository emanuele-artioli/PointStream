import pytest
from demo.evaluation.evaluate_robotics_teleop import score_pose_tracks
from demo.pipeline.hand_keypoints import FrameHandPose, SingleHand


def hand(x=10):
    return SingleHand('Right', .9, [0, 0, 20, 20], [[0, 0, 0]] * 21, [[x, 10]] * 21)


def frames():
    return [FrameHandPose(i, [hand(10 + 500 * i)]) for i in range(3)]


def test_missing_suffix_remains_in_reference_denominator():
    ref = frames()
    result = score_pose_tracks(ref, ref[:1])
    assert result['gt_hands'] == 3
    assert result['missing_prediction_frames'] == 2
    assert result['missing_reference_hands'] == 2
    assert result['detection_rate'] == pytest.approx(1 / 3)
    assert result['pck50_all_gt'] == pytest.approx(1 / 3)
    assert result['pck50_matched'] == 1


def test_empty_prediction_is_all_missing():
    result = score_pose_tracks(frames(), [])
    assert result['registered_frames'] == 3
    assert result['gt_hands'] == result['missing_reference_hands'] == 3
    assert result['pck50_all_gt'] == result['detection_rate'] == 0


def test_reordering_and_sparse_indices_align_by_identity():
    ref = frames()
    assert score_pose_tracks(ref, list(reversed(ref)))['pck50_all_gt'] == 1
    assert score_pose_tracks(ref, [ref[2]])['pck50_all_gt'] == pytest.approx(1 / 3)


@pytest.mark.parametrize('side', ['reference', 'prediction'])
def test_duplicate_indices_rejected(side):
    ref, pred = frames(), frames()
    if side == 'reference':
        ref.append(ref[0])
    else:
        pred.append(pred[0])
    with pytest.raises(ValueError, match=f'duplicate {side}'):
        score_pose_tracks(ref, pred)


def test_outside_registered_frame_rejected():
    with pytest.raises(ValueError, match='outside registered'):
        score_pose_tracks(frames(), [FrameHandPose(99, [])])


def test_empty_registered_window():
    result = score_pose_tracks([], [])
    assert result['registered_frames'] == result['missing_prediction_frames'] == 0


def test_present_empty_prediction_frame_is_missing_hand_not_missing_frame():
    ref = frames()
    pred = [FrameHandPose(p.frame_idx, []) for p in ref]
    result = score_pose_tracks(ref, pred)
    assert result['missing_prediction_frames'] == 0
    assert result['missing_reference_hands'] == 3
    assert result['detection_rate'] == 0


def test_nonconsecutive_reference_window_preserves_ids():
    ref = [FrameHandPose(10, [hand()]), FrameHandPose(42, [hand()])]
    result = score_pose_tracks(ref, [ref[1]])
    assert result['registered_frames'] == 2
    assert result['missing_prediction_frames'] == 1
    assert result['detection_rate'] == .5
