"""The speed gate must reject incomplete measurements and unsafe precision changes."""
import json

import numpy as np

from experiments.background import g6_speed


def test_speed_gate_rejects_precision_and_missing_frames(tmp_path):
    configs = [{"config": name, "score": {"dataset": {"lpips_v": 0.1, "dists_v": 0.2, "psnr_v": 30},
                                          "flicker": {"flicker_v": 1}, "kid": {"kid_v": 2}},
                "timing": {"wall_ms_per_frame": [12, 13], "cnn_ms_mean": 10},
                "rgb_mae_vs_fp32": 0.01} for name in g6_speed.CONFIGS]
    result = {"gpu_capability": [8, 9], "frames": 2, "crfs": [48],
              "points": [{"crf": 48, "decoded_hashes_match": True, "configs": configs}]}
    path = tmp_path / "g6-speed.json"
    path.write_text(json.dumps(result))
    assert all(g6_speed.validate(tmp_path).values())
    configs[1]["rgb_mae_vs_fp32"] = 2
    configs[2]["timing"]["wall_ms_per_frame"] = [12]
    configs[2]["score"]["kid"]["kid_v"] = None
    path.write_text(json.dumps(result))
    checks = g6_speed.validate(tmp_path)
    assert not checks["precision_close"]
    assert not checks["timed_every_frame"]
    assert not checks["finite_scores"]


def test_pixel_error_preserves_unsigned_extremes():
    a = np.array([[[[0, 255, 3]]], [[[255, 0, 7]]]], dtype=np.uint8)
    b = np.array([[[[255, 0, 5]]], [[[0, 255, 7]]]], dtype=np.uint8)
    expected = np.abs(a.astype(np.int64) - b.astype(np.int64))
    assert g6_speed.pixel_error(a, b) == (float(expected.mean()), int(expected.max()))


def test_gate_rejects_missing_rate(tmp_path):
    result = {"gpu_capability": [8, 9], "frames": 2, "crfs": [48], "points": []}
    (tmp_path / "g6-speed.json").write_text(json.dumps(result))
    checks = g6_speed.validate(tmp_path)
    assert not checks["every_crf"]
    assert not checks["native_decode_hashes"]


def test_g6_inference_option_preserves_legacy_default(monkeypatch):
    from experiments.background import g6

    selected = []

    def capture(args):
        selected.append((args.inference, g6.enhance_method(args.inference, args.enhancer_channels)))
        return 0

    monkeypatch.setattr(g6, "command_enhance", capture)
    names = ("eval-set", "archive", "masks", "video", "item", "image-ckpt", "video-ckpt", "lpips-backbone",
             "dists-backbone", "inception", "propainter-tree", "propainter-ckpt", "raft-ckpt", "flow-completion-ckpt",
             "footage", "steps")
    argv = ["enhance", *[v for name in names for v in (f"--{name}", "1")]]
    assert g6.main(argv) == 0
    assert g6.main([*argv, "--inference", "fp16-channels-last"]) == 0
    assert g6.main([*argv, "--inference", "fp16-channels-last-540-residual"]) == 0
    assert g6.main([*argv, "--inference", "fp16-channels-last", "--enhancer-channels", "16"]) == 0
    assert selected == [("legacy", "enh-1080"), ("fp16-channels-last", "enh-1080"),
                        ("fp16-channels-last-540-residual", "enh-correction-540"),
                        ("fp16-channels-last", "enh-1080-b8-c16")]
    assert g6.enhance_method("fp16-channels-last-540-residual", 16) == "enh-correction-540-b8-c16"
