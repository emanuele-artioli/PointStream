from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared import scene_classification as sc
from src.shared.schemas import SceneClass, SceneSpan

REPO_ROOT = Path(__file__).resolve().parents[1]


def _synthetic_signal() -> tuple[np.ndarray, np.ndarray]:
    """~40s of low-motion 'point' rallies separated by high-motion 'interlude' pans.

    Ten scenes, alternating point/interlude, each 4s @ 10 samples/s so there
    are enough valid scenes (>= 6) to exercise the GMM clustering path
    rather than the small-N fallback.
    """
    rng = np.random.default_rng(0)
    times = []
    scores = []
    t = 0.0
    for scene_idx in range(10):
        is_interlude = scene_idx % 2 == 1
        n_samples = 40
        for _ in range(n_samples):
            t += 0.1
            base = 0.4 if is_interlude else 0.01
            noise = rng.uniform(0.0, 0.02)
            scores.append(base + noise)
            times.append(t)
    return np.array(times), np.array(scores)


def test_find_candidate_cuts_returns_floats_within_range() -> None:
    times, scores = _synthetic_signal()
    cuts = sc.find_candidate_cuts(times, scores)
    assert all(isinstance(c, float) for c in cuts)
    assert all(0.0 <= c <= times[-1] for c in cuts)


def test_classify_scene_stats_separates_point_and_interlude() -> None:
    # Explicit scene boundaries (every 4s) rather than derived cuts: this
    # isolates the clustering/thresholding logic from peak detection, which
    # has its own test above and expects real single-frame cut spikes, not
    # this test's synthetic sustained-plateau signal.
    times, scores = _synthetic_signal()
    cuts = [4.0 * i for i in range(1, 10)]
    scene_stats, raw_centroids = sc.classify_scene_stats(times, scores, cuts)

    assert len(scene_stats) >= 6
    classes = {s["classification"] for s in scene_stats}
    # The dynamic threshold must separate at least one low-motion cluster
    # from at least one high-motion cluster.
    assert "cluster_point" in classes
    assert "cluster_interlude" in classes

    spans = sc.to_scene_spans(scene_stats)
    assert any(s.scene_class == SceneClass.POINT for s in spans)
    assert any(s.scene_class == SceneClass.INTERLUDE for s in spans)


def test_classify_scene_stats_small_n_fallback() -> None:
    """Fewer than 6 valid scenes takes the robust fallback branch, not GMM."""
    times = np.linspace(0.1, 10.0, 100)
    scores = np.full_like(times, 0.001)  # uniformly low motion, no cuts
    scene_stats, raw_centroids = sc.classify_scene_stats(times, scores, cut_times=[])

    assert len(scene_stats) == 1
    assert raw_centroids == {}
    assert scene_stats[0]["classification"] == "cluster_point"
    assert scene_stats[0]["cluster_confidence"] == 0.0


def test_classify_scene_stats_blank_scene_is_excluded_from_clustering() -> None:
    times, scores = _synthetic_signal()
    # Prepend a near-zero-motion span that must be classified as blank.
    blank_times = np.linspace(0.01, 0.09, 5)
    blank_scores = np.zeros_like(blank_times)
    all_times = np.concatenate([blank_times, times + 0.1])
    all_scores = np.concatenate([blank_scores, scores])

    cuts = sc.find_candidate_cuts(all_times, all_scores)
    # Force a cut right at the blank/signal boundary so it forms its own scene.
    cuts = sorted(set(cuts) | {0.1})
    scene_stats, _ = sc.classify_scene_stats(all_times, all_scores, cuts)

    blank_scenes = [s for s in scene_stats if s["classification"] == "error_blank"]
    assert len(blank_scenes) >= 1


def test_to_scene_spans_produces_valid_pydantic_models() -> None:
    scene_stats = [
        {
            "t_start": 0.0,
            "t_end": 4.0,
            "avg_score": 0.01,
            "std_score": 0.005,
            "max_score": 0.02,
            "classification": "cluster_point",
            "cluster_confidence": 0.9,
        },
        {
            "t_start": 4.0,
            "t_end": 8.0,
            "avg_score": 0.5,
            "std_score": 0.1,
            "max_score": 0.6,
            "classification": "cluster_interlude",
            "cluster_confidence": 0.8,
        },
        {
            "t_start": 8.0,
            "t_end": 8.1,
            "avg_score": 0.0,
            "std_score": 0.0,
            "max_score": 0.0,
            "classification": "error_blank",
            "cluster_confidence": 0.0,
        },
    ]
    spans = sc.to_scene_spans(scene_stats)
    assert spans == [
        SceneSpan(t_start=0.0, t_end=4.0, scene_class=SceneClass.POINT, confidence=0.9, avg_score=0.01, std_score=0.005, max_score=0.02),
        SceneSpan(t_start=4.0, t_end=8.0, scene_class=SceneClass.INTERLUDE, confidence=0.8, avg_score=0.5, std_score=0.1, max_score=0.6),
        SceneSpan(t_start=8.0, t_end=8.1, scene_class=SceneClass.BLANK, confidence=0.0, avg_score=0.0, std_score=0.0, max_score=0.0),
    ]


def test_load_scene_scores_roundtrip(tmp_path: Path) -> None:
    cache_file = tmp_path / "scene_scores.csv"
    cache_file.write_text("pts_time,scene_score\n0.1,0.02\n0.2,0.5\n", encoding="utf-8")
    scores = sc.load_scene_scores(str(cache_file))
    assert scores == [(0.1, 0.02), (0.2, 0.5)]


@pytest.mark.integration
def test_classify_matches_cached_alcaraz_ruud_dataset_labels() -> None:
    """Regression: the shared module must reproduce the existing curated
    dataset's classification exactly (report 10 dataset inventory), since
    `assets/dataset/alcaraz_ruud` was produced by the pre-refactor algorithm
    this module was extracted from, with the same fixed random_state=42.
    """
    video_path = REPO_ROOT / "assets" / "raw_4k" / "alcaraz_ruud.mp4"
    cache_file = REPO_ROOT / "assets" / "dataset" / "alcaraz_ruud" / "scene_scores.csv"
    metadata_path = REPO_ROOT / "assets" / "dataset" / "alcaraz_ruud" / "scene_metadata.json"
    if not (video_path.exists() and cache_file.exists() and metadata_path.exists()):
        pytest.skip("assets/dataset/alcaraz_ruud not present in this environment")

    scene_stats, raw_centroids = sc.detect_and_classify_scenes(str(video_path), str(cache_file))
    non_blank = [s for s in scene_stats if s["classification"] != "error_blank"]

    expected = json.loads(metadata_path.read_text())
    expected_scenes = expected["scenes"]

    assert len(non_blank) == len(expected_scenes)
    for actual, exp in zip(non_blank, expected_scenes):
        assert actual["t_start"] == pytest.approx(exp["t_start"], abs=1e-3)
        assert actual["t_end"] == pytest.approx(exp["t_end"], abs=1e-3)
        assert actual["classification"] == exp["cluster"]
        assert actual["cluster_confidence"] == pytest.approx(exp["cluster_confidence"], abs=1e-3)
