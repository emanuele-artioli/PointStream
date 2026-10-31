"""Paid judging gates: real controls, exact frame coverage, conservative stopping."""
import numpy as np
import pytest

from experiments.tennis.k6b_matte_judge import assess, control_error, receiver_mask


def evidence(n, error=0.6, kp_bytes=250):
    rows, measured, wire, masks = [], [], [], []
    for i in range(12):
        wid = f"w{i}"
        rows.append({"window": wid, "match": "alcaraz_hurkacz", "registered_seconds": 1.,
                     "adapter_guider_bytes": 0, "base_checkpoint_bytes": 1,
                     "anchor_points": {"lo": {"kbps": 2., "lpips": .5, "dists": .5},
                                       "hi": {"kbps": 6., "lpips": .25, "dists": .25}}})
        wire.append({"window": wid, "match": "alcaraz_hurkacz", "keypoints": {"bytes": kp_bytes}, "references": {"png_bytes": 0}})
        masks.append({"window": wid, "settings": {str(s): {"selected": {"bytes": 0}} for s in (8, 16, 32)}})
        if i < n:
            measured.append({"id": wid, "judged": [0, 3], "original_judged": [0, 3], "means": {str(s): {"lpips": error, "dists": error} for s in (8, 16, 32)}})
    return {"rows": rows}, {"rows": wire}, {"rows": masks}, measured


def test_receiver_mask_is_empty_for_a_judged_frame_without_a_crop_and_strict_for_cropped_frames():
    # A judged frame with no pose has no crop and no receiver mask (job 20261031T101138Z-b73a54fa died on frame 84 here).
    shape = (4, 6)
    predicted = {0: {3: np.ones(shape, bool)}}
    crop_frames = {3: None}
    assert receiver_mask(predicted, crop_frames, 0, 3, shape).all()
    assert not receiver_mask(predicted, crop_frames, 0, 84, shape).any()
    with pytest.raises(KeyError):
        receiver_mask({0: {}}, crop_frames, 0, 3, shape)


def test_control_gate_rejects_missing_or_nonfinite_and_drifted_scores():
    for got in (np.array([.1]), np.array([np.nan, .2]), np.array([.1002, .2])):
        with pytest.raises(ValueError):
            control_error(got, np.array([.1, .2]))
    assert control_error(np.array([.10001, .2]), np.array([.1, .2])) < 1e-4


def test_prefix_stop_keeps_last_possible_eighteen_and_counts_unresolved_optimistically():
    three = assess(*evidence(3))
    assert not three["decision_made"]
    assert three["scales"]["8"]["optimistic_final_wins_bound"] == 18
    four = assess(*evidence(4))
    assert four["decision_made"] and four["scales"]["8"]["optimistic_final_wins_bound"] == 16
    six = assess(*evidence(6))
    assert six["decision_made"]
    assert six["scales"]["8"]["optimistic_final_wins_bound"] == 12
    outside = assess(*evidence(6, kp_bytes=1000))
    assert outside["scales"]["8"]["unresolved"] == 12
    assert not outside["decision_made"]


def test_partial_smoke_cannot_decide_full_quality_and_actual_scores_replace_old_quality():
    args = evidence(12, error=.1)
    report = assess(*args)
    assert report["scales"]["8"]["known_wins"] == 24
    args[-1][0]["judged"] = [0]
    with pytest.raises(ValueError, match="Partial smoke"):
        assess(*args)


def test_exclusive_fleet_has_no_shared_memory_override(monkeypatch):
    from experiments.tennis.k6b_matte_judge import gpu_memory_limit
    monkeypatch.delenv("PS_GPU_MEMORY_LIMIT_MIB", raising=False)
    assert gpu_memory_limit() == 24000
    monkeypatch.setenv("PS_GPU_MEMORY_LIMIT_MIB", "12000")
    assert gpu_memory_limit() == 12000
    monkeypatch.setenv("PS_GPU_MEMORY_LIMIT_MIB", "48000")
    assert gpu_memory_limit() == 24000
