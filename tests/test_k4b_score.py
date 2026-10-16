import json

import numpy as np
import pytest

from experiments.tennis.k4b_rules import RULES, choose
from experiments.tennis.k4b_score import score
from experiments.visor.b1 import file_sha256
from src.segmentation.masks import ClipMasks


def fixture(tmp_path):
    review_dir, feature_dir = tmp_path / "review", tmp_path / "features"
    review_dir.mkdir()
    feature_dir.mkdir()
    mask = np.zeros((540, 960), bool)
    mask[10:40, 10:30] = True
    frames = []
    for t in range(2):
        raw = ClipMasks(("person", "player"), 540, 960, 25)
        raw.add(0, "person", 1, mask, .9)
        path = review_dir / f"{t}.rle"
        raw.save(path)
        frames.append(dict(id=f"m/c/{t}", match="m", clip="c", frame=t, masks=path.name,
                           masks_sha256=file_sha256(path), instances=[dict(id=1)]))
    review = dict(selection_sha256="sel", frames=frames)
    rp = review_dir / "instances.json"
    rp.write_text(json.dumps(review))
    marks = dict(schema=1, selection_sha256="sel", result_sha256=file_sha256(rp), marks={
        "m/c/0": dict(selected=[1], visible=1, missing_instance=False, confirmed=True),
        "m/c/1": dict(selected=[], visible=0, missing_instance=False, confirmed=True)})
    target = ClipMasks(("person",), 540, 960, 25)
    target.add(0, "person", 0, mask, .9)
    target.ensure_frames(2)
    target.save(feature_dir / "target-masks.rle")
    features = {"person:1": dict(presence=.5, a=1., b=1., c=1., e=1., racket_most=True)}
    record = dict(rules=RULES, features=features, picks=choose(features), target_frames=[0, 1],
                  frame_count=2, track_ids={"person:1": 0})
    (feature_dir / "features.json").write_text(json.dumps(record))
    (feature_dir / "local-validation.json").write_text(json.dumps(dict(passed=True, checks=dict(real=True), clip="m/c")))
    (feature_dir / "ready.json").write_text(json.dumps(dict(git_head="commit")))
    (feature_dir / "k4b-features.json").write_text(json.dumps(dict(configured_frames=0, stage="full", frames=2, clip=dict(id="m/c", frames=2))))
    source = dict(directory=str(feature_dir), clip="m/c", job_id="job", git_head="commit",
                  files={p.name: file_sha256(p) for p in feature_dir.iterdir()})
    return marks, review, rp, review_dir, source


def test_score_counts_absent_track_as_none_and_ties_to_simpler_rule(tmp_path):
    marks, review, rp, root, source = fixture(tmp_path)
    result = score(marks, review, rp, root, [source])
    assert result["complete"] and result["scored"] == 2
    assert all(r["correct"] == 2 and r["reaches_bar"] for r in result["table"])
    assert result["selected_by_frozen_rule"] == "a"
    assert result["applied_to_k4"] is False
    assert result["proposal_oracle"]["correct"] == 2


def test_score_rejects_changed_artifacts_and_missing_clips(tmp_path):
    marks, review, rp, root, source = fixture(tmp_path)
    with pytest.raises(ValueError, match="Missing full-clip"):
        score(marks, review, rp, root, [])
    (tmp_path / "features/ready.json").write_text("{}")
    with pytest.raises(ValueError, match="Changed feature"):
        score(marks, review, rp, root, [source])
