"""K8b: fit-state cut splits, clip partitions, census scoring, review page and validator."""
from __future__ import annotations

import json
import tarfile

import numpy as np
import pytest

from experiments.tennis import court, k8b


def test_short_state_runs_are_absorbed_and_persistent_changes_split():
    states = [False] * 60 + [True] * 3 + [False] * 10 + [True] * 120
    assert k8b.split_by_state(states, 50) == [(0, 73), (73, 193)]
    assert k8b.split_by_state([True] * 10 + [False] * 5, 50) == [(0, 15)]
    assert k8b.split_by_state([], 50) == []
    flicker = [i % 2 == 0 for i in range(200)]
    assert k8b.split_by_state(flicker, 50) == [(0, 200)]


def rows(states):
    return [{"frame": i, "fits": s, "error_px": 1.0 if s else None, "homography": None} for i, s in enumerate(states)]


def test_clip_shots_cut_first_then_split_a_gradual_transition():
    states = [False] * 100 + [True] * 150 + [True] * 50
    spans = k8b.clip_shots(rows(states), [250], 50)
    assert spans == [(0, 100), (100, 250), (250, 300)]
    assert k8b.prediction(rows(states)[100:250])["predicted"] == "court"


def test_parts_cover_every_clip_once_and_weigh_4k_more():
    clips = [{"id": f"m/{i}", "frames": f, "weight": w} for i, (f, w) in enumerate([(4000, 1.2), (3000, 1.0), (2000, 1.0), (1000, 1.0)])]
    groups = k8b.parts(clips, 2)
    assert sorted(c["id"] for g in groups for c in g) == sorted(c["id"] for c in clips)
    assert [c["id"] for c in groups[0]] == ["m/0", "m/3"]  # 4,800 + 1,000 against 3,000 + 2,000
    chunks = k8b.chunks([{"id": "m/0", "frames": 900}])
    assert [(c["start"], c["end"]) for c in chunks] == [(0, 400), (400, 800), (800, 900)]


def report():
    def clip(cid, fps, shots, regs):
        return {"id": cid, "path": f"{cid}.mp4", "fps": fps, "frames": len(regs), "source_frames": len(regs),
                "shots": shots, "registration": regs}
    regs_a = rows([True] * 100 + [False] * 100)
    shots_a = [{"id": "a/0/shot-00", "start": 0, "end": 100, "seconds": 2.0, "predicted": "court", "fit_share": 1.0,
                "images": ["i1", "i2", "i3"]},
               {"id": "a/0/shot-01", "start": 100, "end": 200, "seconds": 2.0, "predicted": "non_court", "fit_share": 0.0,
                "images": ["i4", "i5", "i6"]}]
    regs_b = rows([True] * 60)
    shots_b = [{"id": "b/0/shot-00", "start": 0, "end": 60, "seconds": 1.0, "predicted": "court", "fit_share": 1.0,
                "images": ["i7", "i8", "i9"]}]
    return {"review_id": "r", "clips": [clip("a/0", 50, shots_a, regs_a), clip("b/0", 60, shots_b, regs_b)]}


def test_census_score_is_exact_and_missed_court_frames_are_infinite():
    marks = {"review_id": "r", "marks": {"a/0/shot-00": "court", "a/0/shot-01": "court", "b/0/shot-00": "replay"}}
    result = k8b.score(report(), marks)
    assert result["precision"] == 0.5 and result["duration_recall"] == pytest.approx(0.5)
    assert result["court_frames"] == 200 and result["registration_median_px"] is None
    assert result["misses_by_class"] == {"court": 1, "replay": 1} and not result["adopted"]
    marks["marks"].update({"a/0/shot-01": "crowd", "b/0/shot-00": "court"})
    result = k8b.score(report(), marks)
    assert result["adopted"] and result["registration_median_px"] == 1.0
    with pytest.raises(ValueError):
        k8b.score(report(), {"review_id": "r", "marks": {"a/0/shot-00": "court"}})


def test_review_page_starts_blank_and_carries_the_review_id(tmp_path):
    k8b.review_page(report(), tmp_path / "review.html")
    page = (tmp_path / "review.html").read_text()
    assert page.count("<article") == 3 and "selected" not in page and '"r"' in page
    assert 'src="clips/a/0.mp4#t=0.000,2.000"' in page


def test_validator_reads_the_published_archive(tmp_path, monkeypatch):
    hom = np.eye(3).tolist()
    regs = [{"frame": i, "fits": True, "error_px": 1.0, "homography": hom, "clutter": 0.01} for i in range(10)]
    clip = {"id": "a/0", "frames": 10, "cuts": [], "registration": regs,
            "shots": [{"id": "a/0/shot-00", "start": 0, "end": 10, "predicted": "court",
                       "images": [f"images/a-0-00-{r}.jpg" for r in ("first", "middle", "last")]}]}
    publish = tmp_path / "scratch" / "publish"
    (publish / "clips").mkdir(parents=True)
    (publish / "images").mkdir()
    (publish / "clips" / "a-0.json").write_text(json.dumps(clip))
    (publish / "k8b-report.json").write_text(json.dumps({"clips": [{"id": "a/0"}]}))
    for r in ("first", "middle", "last"):
        (publish / "images" / f"a-0-00-{r}.jpg").write_bytes(b"x")
    stage = tmp_path / "stage"
    stage.mkdir()
    with tarfile.open(stage / "published.tar", "w") as archive:
        archive.add(publish, arcname="publish")
    monkeypatch.setenv("PS_STAGE_DIR", str(stage))
    monkeypatch.setenv("PS_SCRATCH_DIR", str(tmp_path / "gone"))
    monkeypatch.setenv("PS_VALIDATION_PATH", str(tmp_path / "v.json"))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert k8b.validate() == 0
    regs[3]["fits"] = False
    (publish / "clips" / "a-0.json").write_text(json.dumps(clip))
    with tarfile.open(stage / "published.tar", "w") as archive:
        archive.add(publish, arcname="publish")
    assert k8b.validate() == 1


def test_register_v2_reseats_a_fit_locked_on_the_singles_lines():
    """Court-level camera: the doubles corners leave the frame; a seed on the singles lines must be swapped."""
    import cv2

    true = cv2.getPerspectiveTransform(court.OUTLINE, np.array([[440, 214], [1485, 214], [2374, 770], [-472, 770]], np.float32))
    image = np.zeros((1080, 1920, 3), np.uint8)
    image[:] = (40, 110, 170)
    for segment in court.SEGMENTS:
        cv2.polylines(image, [np.rint(court.project(segment, true)).astype(np.int32)], False, (235, 235, 235), 5)
    locked = true @ court.SWAPS[1]  # doubles sidelines drawn where the singles lines are
    result = court.register_v2(image, locked)
    assert result["fits"] and result["error_px"] < 2.0 and result["segments_supported"] >= 8


def test_register_v2_recovers_a_shaded_court_and_rejects_texture():
    """A court whose lines are only slightly brighter than dark grass (K8's absolute bar misses them)."""
    import cv2

    h = cv2.getPerspectiveTransform(court.OUTLINE, np.array([[560, 300], [1360, 300], [1700, 950], [220, 950]], np.float32))
    image = np.zeros((1080, 1920, 3), np.uint8)
    image[:] = (40, 90, 40)
    for segment in court.SEGMENTS:
        cv2.polylines(image, [np.rint(court.project(segment, h)).astype(np.int32)], False, (120, 135, 120), 5)
    assert court.white_lines(image).sum() == 0  # the shade failure K8 had
    result = court.register_v2(image)
    assert result["fits"] and result["error_px"] < 2.0 and result["clutter"] < 0.05
    noise = np.random.default_rng(0).integers(0, 255, (1080, 1920, 3), dtype=np.uint8)
    assert not court.register_v2(noise)["fits"]


def test_dev_clips_are_neither_reviewed_nor_scored(tmp_path, monkeypatch):
    monkeypatch.setattr(k8b, "DEV_CLIPS", ("b/0",))
    k8b.review_page(report(), tmp_path / "review.html")
    assert (tmp_path / "review.html").read_text().count("<article") == 2
    marks = {"review_id": "r", "marks": {"a/0/shot-00": "court", "a/0/shot-01": "crowd"}}
    result = k8b.score(report(), marks)
    assert result["adopted"] and result["excluded_dev_clips"] == ["b/0"]
