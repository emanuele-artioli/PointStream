"""The speed gate must reject incomplete measurements and unsafe precision changes."""
import json

import numpy as np

from experiments.background import g6_speed


def test_speed_gate_rejects_precision_and_missing_frames(tmp_path):
    configs = [{"config": name, "score": {"dataset": {"lpips_v": 0.1}},
                "timing": {"wall_ms_per_frame": [12, 13], "cnn_ms_mean": 10},
                "rgb_mae_vs_fp32": 0.01} for name in g6_speed.CONFIGS]
    result = {"gpu_capability": [8, 9], "frames": 2,
              "points": [{"decoded_hashes_match": True, "configs": configs}]}
    path = tmp_path / "g6-speed.json"
    path.write_text(json.dumps(result))
    assert all(g6_speed.validate(tmp_path).values())
    configs[1]["rgb_mae_vs_fp32"] = 2
    configs[2]["timing"]["wall_ms_per_frame"] = [12]
    path.write_text(json.dumps(result))
    checks = g6_speed.validate(tmp_path)
    assert not checks["precision_close"]
    assert not checks["timed_every_frame"]


def test_pixel_error_preserves_unsigned_extremes():
    a = np.array([[[[0, 255, 3]]], [[[255, 0, 7]]]], dtype=np.uint8)
    b = np.array([[[[255, 0, 5]]], [[[0, 255, 7]]]], dtype=np.uint8)
    expected = np.abs(a.astype(np.int64) - b.astype(np.int64))
    assert g6_speed.pixel_error(a, b) == (float(expected.mean()), int(expected.max()))
