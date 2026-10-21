import json

import numpy as np
import pytest

from experiments.tennis import k4
from experiments.tennis.k4b_features import (convert_command, default_config_sha256, extract_features, key,
                                             plan_windows, validate, window_span)
from experiments.tennis.k4b_rules import RULES
from experiments.visor.b1 import file_sha256
from src.segmentation.masks import ClipMasks, compress


def test_window_is_k4_length_centred_on_the_marked_frame_and_kept_inside_the_clip():
    assert window_span(1532, 3975) == (1412, 240)
    assert window_span(14, 3975) == (0, 240)
    assert window_span(3970, 3975) == (3735, 240)
    assert window_span(62, 64) == (0, 64)
    assert window_span(218, 294) == (54, 240)
    with pytest.raises(ValueError, match="inside the clip"):
        window_span(64, 64)


def test_plan_has_one_window_per_marked_frame_and_refuses_label_manifests():
    clips = [dict(id="m/c", frames=400, fps=50., sha256="s", review_frames=[10, 390])]
    plan = plan_windows(dict(contains_human_marks=False, clips=clips))
    assert list(plan) == ["m/c@10", "m/c@390"]
    assert (plan["m/c@390"]["start"], plan["m/c@390"]["frames"], plan["m/c@390"]["target"]) == (160, 240, 230)
    with pytest.raises(ValueError, match="human selections"):
        plan_windows(dict(clips=clips))


def test_conversion_selects_the_window_before_k4s_scaling():
    command = convert_command("ffmpeg", "in.mp4", "out.yuv", 160, 240)
    assert command[command.index("-frames:v") + 1] == "240"
    assert command[command.index("-vf") + 1].startswith("select=between(n\\,160\\,399),scale=1920:1080:flags=area:out_range=tv")
    assert "passthrough" in command


def test_features_merge_duplicate_prompts_and_preserve_target_masks():
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


def _window(tmp_path):
    clip = ClipMasks(("person", "player", "racket"), 540, 960, 25)
    detections = []
    for t in range(8):
        mask = np.zeros((540, 960), bool)
        mask[20:50, 30 + t:40 + t] = True
        clip.add(t, "person", 1, mask, .9)
        detections.append(dict(boxes=np.zeros((0, 4), np.float32), kps=np.zeros((0, 17, 3), np.float32)))
    manifest = dict(contains_human_marks=False, clips=[dict(id="m/c", frames=8, fps=25., sha256="s", review_frames=[3])])
    window = plan_windows(manifest)["m/c@3"]
    out = tmp_path / "publish" / k4.safe(window["id"])
    out.mkdir(parents=True)
    record, targets = extract_features(clip, detections, [window["target"]])
    clip.save(out / "sam-masks.rle")
    targets.save(out / "target-masks.rle")
    (out / "pose.json.zst").write_bytes(compress(json.dumps([{k: v.tolist() for k, v in d.items()} for d in detections]).encode()))
    (out / "features.json").write_text(json.dumps(record))
    row = {**window, "conversion": dict(frames=8), "files": {p.name: file_sha256(p) for p in out.iterdir()}}
    summary = dict(requested=[window["id"]], configured_windows=0, windows=[row], manifest_sha256="",
                   clips={"m/c": dict(frames=8, fps=25., review_frames=[3], sha256="s")}, rules=RULES,
                   sam=dict(checkpoint_sha256="0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6",
                            sdpa_backend_policy="native_flash", runtime=dict(capability=[8, 9]),
                            config_sha256=default_config_sha256(), grounding_batch_size=16),
                   pose=dict(on_cuda={"yolo11x-pose": "cuda:0"}), gpu_peak_gib=24., gpu_memory_limit_mib=32000)
    return tmp_path / "publish", summary, out


def test_validator_recomputes_windows_and_catches_tampering(tmp_path):
    directory, summary, out = _window(tmp_path)
    report = validate(directory, summary)
    assert report["passed"], report["checks"]
    assert report["frames"] == 8
    moved = json.loads(json.dumps(summary))
    moved["windows"][0]["start"] = 1
    assert not validate(directory, moved)["checks"]["windows_placed"]
    offloaded = json.loads(json.dumps(summary))
    offloaded["sam"]["config_sha256"] = "0" * 64
    assert not validate(directory, offloaded)["checks"]["sam_default_config"]
    over = json.loads(json.dumps(summary))
    over["gpu_peak_gib"] = 40.
    assert not validate(directory, over)["checks"]["peak_within_declaration"]
    features = json.loads((out / "features.json").read_text())
    features["features"][key("person", 1)]["a"] = 99.
    (out / "features.json").write_text(json.dumps(features))
    checks = validate(directory, summary)["checks"]
    assert not checks["files_identified"] and not checks["features_recomputed"]
