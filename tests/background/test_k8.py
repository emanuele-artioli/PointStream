"""K8 fixed rule, scene indexing, missing fits, geometric recovery and review labels."""
from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from experiments.tennis import court, k8


def test_cuts_use_source_pts_and_partition_without_losing_frames():
    text = "frame:0 pts:250 pts_time:5\nlavfi.scene_score=0.33\nframe:1 pts:600 pts_time:12\nlavfi.scene_score=0.42"
    assert [r["frame"] for r in k8.parse_scenes(text, 50)] == [250, 600]
    assert k8.shot_spans([600, 250, 250], 800) == [(0, 250), (250, 600), (600, 800)]
    with pytest.raises(ValueError):
        k8.shot_spans([800], 800)


def test_eighty_percent_counts_every_frame():
    rows = [{"registered": i < 8} for i in range(10)]
    assert k8.shot_prediction(rows)["predicted"] == "court"
    rows.append({"registered": False})
    assert k8.shot_prediction(rows)["predicted"] == "non_court"


def shot(ident, start, end, pred, error=1.0):
    return {"id": ident, "start": start, "end": end, "predicted": pred,
            "registration": [{"chamfer_px": error} for _ in range(end - start)]}


def test_scoring_duration_recall_and_mixed_shot_penalizes_both():
    report = {"review_id": "a", "fps": 10, "shots": [shot("a", 0, 90, "court"),
              shot("b", 90, 100, "non_court"), shot("c", 100, 110, "court")]}
    result = k8.score(report, {"review_id": "a", "marks": {"a": "court", "b": "court", "c": "mixed"}})
    assert result["precision"] == 0.5
    assert result["duration_recall"] == 9 / 11
    assert result["misses_by_class"] == {"court": 1, "mixed": 1}
    assert not result["adopted"]


def test_missing_registrations_fail_error_bar_and_marks_must_be_complete():
    report = {"review_id": "a", "fps": 10, "shots": [shot("a", 0, 100, "court", None)]}
    result = k8.score(report, {"review_id": "a", "marks": {"a": "court"}})
    assert result["precision"] == result["duration_recall"] == 1
    assert result["registration_median_px"] is None
    assert not result["conditions"]["registration"]
    for marks in ({"review_id": "b", "marks": {"a": "court"}}, {"review_id": "a", "marks": {}}):
        with pytest.raises(ValueError):
            k8.score(report, marks)


def synthetic():
    rgb = np.full((1080, 1920, 3), [55, 105, 55], np.uint8)
    corners = np.array([[660, 240], [1260, 240], [1600, 920], [320, 920]], np.float32)
    hom = cv2.getPerspectiveTransform(court.OUTLINE, corners)
    for segment in court.SEGMENTS:
        points = np.rint(court.project(segment, hom)).astype(np.int32)
        cv2.polylines(rgb, [points], False, (240, 240, 240), 4)
    return rgb, hom


def test_ransac_recovers_projected_regulation_court_and_rejects_blank():
    rgb, expected = synthetic()
    result = court.register(rgb)
    assert result["registered"]
    actual = court.project(court.OUTLINE, np.array(result["homography"]))
    assert np.max(np.linalg.norm(actual - court.project(court.OUTLINE, expected), axis=1)) < 5
    assert result["chamfer_px"] <= 2
    assert court.register(np.zeros_like(rgb))["homography"] is None


def test_chamfer_scales_to_1080_and_out_of_frame_is_not_ignored():
    rgb, hom = synthetic()
    mask = court.white_lines(rgb)
    assert court.chamfer(hom, mask)["chamfer_px"] < 1
    hom[0, 2] += 3000
    assert court.chamfer(hom, mask)["chamfer_px"] is None


def test_review_starts_unlabelled_and_downloads_identified_json(tmp_path):
    report = {"review_id": "abc", "fps": 50, "scene_threshold": 0.25,
              "shots": [{"id": "shot-0000", "start": 0, "end": 100, "predicted": "court",
                         "registered_share": 1.0, "images": ["a.jpg", "b.jpg", "c.jpg"]}]}
    path = tmp_path / "review.html"
    k8.review_page(report, path)
    text = path.read_text()
    assert 'value=""' in text and 'selected' not in text
    assert 'new Blob' in text and 'review_id:reviewId' in text
    assert json.dumps(k8.CLASSES) in text
