import copy
import subprocess
import sys
from typing import Any

import pytest

from experiments.tennis.k5d_analysis import score_frames
from experiments.tennis.k5d_source_oracle import select, validate_report, validate_rows


def clip() -> dict:
    return {"clip_id": "tracknet/game9/Clip5", "roles": ["full_retrospective_benchmark"],
            "label_rows": 3, "jpg_entries": 3, "visibility_counts": {"0": 1, "1": 1, "2": 0, "3": 1}}


def test_selection_and_labels() -> None:
    expected = clip()
    manifest = {"schema_version": 1, "tracknet": {"clips": [expected]}}
    assert select(manifest, expected["clip_id"], 1, 2) == expected
    for start, frames in ((-1, 1), (2, 2), (0, 0)):
        with pytest.raises(ValueError):
            select(manifest, expected["clip_id"], start, frames)
    rows = [("1.jpg", 1, 3., 4.), ("2.jpg", 0, 0., 0.), ("3.jpg", 3, 0., 0.)]
    validate_rows(rows, expected)
    with pytest.raises(ValueError):
        validate_rows(rows[:-1] + [rows[0]], expected)


def test_report_recomputed_from_all_frames() -> None:
    labels = [(1, 3., 4.), (0, 0., 0.), (-1, 0., 0.)]
    pred = [(True, 3., 4.), (True, 9., 9.), (True, 1., 1.)]
    rows = [{"frame": i, "member": f"{i}.jpg", "source_sha256": "a" * 64,
             "label_kind": label[0], "label_x": label[1], "label_y": label[2],
             "wasb": {"detected": v, "x": x, "y": y}, "balltrack": {"detected": v, "x": x, "y": y}}
            for i, (label, (v, x, y)) in enumerate(zip(labels, pred))]
    identities = {name: {"path": "resource", "sha256": "a" * 64} for name in
                  ("manifest", "archive", "balltrack_weight", "wasb_config", "exposure", "balltrack_code", "wasb_code", "wasb_weights")}
    scores = score_frames(labels, pred)
    report: dict[str, Any] = {"schema_version": 1, "clip": "test", "exposure": {"basis": "reported"}, "identities": identities,
              "device": {"visible_count": 1, "affinity": [0], "name": "example", "capability": [8, 9], "models_on_cuda": ["wasb-step1", "balltrack"]}, "start_frame": 0, "frames": 3,
              "rows": rows, "scores": {"wasb": scores, "balltrack": scores}, "qualified": True}
    assert validate_report(report)["scores"]["wasb"]["false_on_absent"] == 1
    bad = copy.deepcopy(report)
    bad["rows"][0]["wasb"]["x"] = 100
    with pytest.raises(ValueError, match="count mismatch"):
        validate_report(bad)
    bad = copy.deepcopy(report)
    bad["rows"].pop()
    with pytest.raises(ValueError, match="frame"):
        validate_report(bad)


def test_help_does_not_load_torch() -> None:
    p = subprocess.run([sys.executable, "-c", "import sys; import experiments.tennis.k5d_source_oracle; "
                        "assert 'torch' not in sys.modules"], check=False)
    assert p.returncode == 0


def test_bind_config_and_contain_output(tmp_path) -> None:
    from experiments.tennis.k5d_source_oracle import bind_config, sha, validate_output

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    model = scratch / "model"
    config = model / "src/configs/dataset/tennis.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("test: yes\n")
    bind_config(model, sha(config))
    with pytest.raises(ValueError, match="differs"):
        bind_config(model, "0" * 64)
    validate_output(scratch / "output.json", scratch, [model, config])
    with pytest.raises(ValueError, match="beneath"):
        validate_output(tmp_path / "outside.json", scratch, [model])
    with pytest.raises(ValueError, match="overlaps"):
        validate_output(model / "output.json", scratch, [model])
    (scratch / "escape").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="beneath"):
        validate_output(scratch / "escape" / "output.json", scratch, [model])
