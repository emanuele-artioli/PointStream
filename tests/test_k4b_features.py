import numpy as np
import pytest
from types import SimpleNamespace

from experiments.tennis.k4b_features import extract_features, key, mask_parity, offload_tracker_state
from src.segmentation.masks import ClipMasks


def test_parity_rejects_identity_swaps_even_when_union_is_identical():
    a, b = [ClipMasks(("person", "player", "racket"), 4, 6, 1) for _ in range(2)]
    mask = np.zeros((4, 6), bool)
    mask[1:3, 1:3] = True
    a.add(0, "person", 1, mask, .9)
    b.add(0, "person", 2, mask, .9)
    result = mask_parity(a, b)
    assert result["minimum_union_iou"] == 1
    assert not result["track_ids_match"] and result["minimum_instance_iou"] == 0


def test_tracker_cpu_state_adapter_forwards_native_arguments_and_checks_storage():
    def model(tracker):
        def cache(state, t, masks):
            state.setdefault("cached_frame_outputs", {})[t] = dict(masks)
        return SimpleNamespace(tracker=tracker, _cache_frame_outputs=cache)

    class Tracker:
        def init_state(self, *, num_frames, offload_state_to_cpu=False):
            return {"frames": num_frames, "offload_state_to_cpu": offload_state_to_cpu,
                    "storage_device": "cpu" if offload_state_to_cpu else "cuda"}

    tracker = Tracker()
    native = model(tracker)
    record = offload_tracker_state(SimpleNamespace(model=native))
    state = tracker.init_state(num_frames=407)
    assert state["frames"] == 407 and state["storage_device"] == "cpu"
    assert record["initialized"] == record["cpu"] == 1

    class Mask:
        def __init__(self, device="cuda"):
            self.device = device

        def cpu(self):
            return Mask("cpu")

    original = Mask()
    native._cache_frame_outputs(state, 3, {7: original})
    assert original.device == "cuda"
    assert state["cached_frame_outputs"][3][7].device == "cpu"
    assert record["cached_masks"] == record["cached_cpu_masks"] == 1

    class IgnoringTracker:
        def init_state(self, *, offload_state_to_cpu=False):
            return {"offload_state_to_cpu": offload_state_to_cpu, "storage_device": "cuda"}

    bad = IgnoringTracker()
    offload_tracker_state(SimpleNamespace(model=model(bad)))
    with pytest.raises(RuntimeError, match="was not effective"):
        bad.init_state()
    with pytest.raises(RuntimeError, match="lacks native CPU-state"):
        offload_tracker_state(SimpleNamespace(model=SimpleNamespace(tracker=SimpleNamespace(init_state=lambda: {}))))


def test_full_clip_features_merge_duplicate_prompts_and_preserve_target_masks():
    clip = ClipMasks(("person", "player", "racket"), 540, 960, 2)
    detections = []
    for t in range(8):
        player = np.zeros((540, 960), bool)
        player[20:50, 30+t:40+t] = True
        other = np.zeros_like(player)
        other[100:120, 100:110] = True
        racket = np.zeros_like(player)
        racket[30:35, 40+t:43+t] = True
        clip.add(t, "person", 1, player, .9)
        clip.add(t, "player", 9, player, .8)
        clip.add(t, "person", 2, other, .9)
        clip.add(t, "racket", 1, racket, .9)
        kp = np.zeros((1, 17, 3), np.float32)
        kp[:, :, 0] = (35+t)*2
        kp[:, :, 1] = 70
        kp[:, :, 2] = .9
        detections.append(dict(boxes=np.array([[30+t, 20, 40+t, 50]], np.float32)*2, kps=kp))
    record, targets = extract_features(clip, detections, [0, 7])
    player_id, other_id = key("person", 1), key("person", 2)
    assert record["aliases"][key("player", 9)] == player_id
    assert len(record["features"]) == 2
    assert record["features"][player_id]["a"] == 7/30
    assert record["features"][player_id]["e"] == 7/30
    assert record["features"][player_id]["c"] == 1
    assert record["features"][other_id]["c"] == 0
    assert record["picks"]["c"] == [player_id]
    assert len(targets.frames) == 2 and all(len(f) == 2 for f in targets.frames)
    assert np.array_equal(targets.frames[1][0].mask(), clip.frames[7][0].mask())
