import numpy as np

from experiments.tennis.k4b_features import extract_features, key
from src.segmentation.masks import ClipMasks


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
