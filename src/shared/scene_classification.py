"""Point-anchored scene classification (report 2, report 10 Phase 1).

Splits a video's HSV/motion scene-score signal into scenes and classifies
each as a low-motion "point" (eligible for the semantic pipeline) or a
high-motion "interlude"/low-confidence "other" (routed to the fallback
codec). This module is the single source of truth for scene cuts: both the
dataset-curation script (`scripts/process_dataset.py`, quality-tier models)
and the runtime encoder (speed-tier models) must call it so that scene
boundaries never depend on which model tier is running (report 10
"Deterministic shared segmentation").

The classification math here is a faithful extraction of the original
`scripts/process_dataset.py:classify_scenes` implementation, split into pure
functions so it can be unit tested and reused without the frame-thumbnail/
JSON I/O side effects that script also performs.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
from typing import Any

import numpy as np

from src.shared.schemas import SceneClass, SceneSpan


def get_video_duration(video_path: str) -> float:
    cmd_dur = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        return float(subprocess.check_output(cmd_dur, text=True).strip())
    except Exception:
        return 0.0


def extract_scene_scores(
    video_path: str, cache_file: str, threads: int = 1, chunk_duration: int = 60
) -> tuple[str, str]:
    """Compute the per-frame HSV scene-score signal via ffmpeg, caching to CSV.

    Idempotent/resumable: re-running with an existing, fully-covered
    `cache_file` is a no-op, which is what makes the shared cache in report
    10's efficiency architecture work (cuts computed once per video, ever).
    """
    duration = get_video_duration(video_path)
    if duration == 0:
        print(f"[extract_scene_scores] Could not get duration for {video_path}")
        return video_path, cache_file

    last_processed_time = 0.0

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            reader = csv.DictReader(f)
            last_row = None
            for row in reader:
                last_row = row
            if last_row is not None:
                last_processed_time = float(last_row["pts_time"])
                print(
                    f"[extract_scene_scores] Resuming {os.path.basename(video_path)} "
                    f"from {last_processed_time:.2f}s"
                )
    else:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pts_time", "scene_score"])

    current_time = last_processed_time

    while current_time < duration:
        end_time = min(duration, current_time + chunk_duration)
        start_time = max(0.0, current_time - 1.0)

        print(
            f"[extract_scene_scores] Processing {os.path.basename(video_path)}: "
            f"{current_time:.2f}s to {end_time:.2f}s"
        )

        cmd = [
            "ffmpeg", "-hide_banner", "-hwaccel", "cuda", "-threads", str(threads),
            "-ss", str(start_time), "-copyts",
            "-i", video_path,
            "-to", str(end_time),
            "-filter:v", "scale=320:-1,select='gte(scene,0)',metadata=print:key=lavfi.scene_score",
            "-f", "null", "-",
        ]

        process = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
        assert process.stderr is not None

        scores = []
        current_time_val = None
        for line in process.stderr:
            time_match = re.search(r"pts_time:([0-9.]+)", line)
            if time_match:
                current_time_val = float(time_match.group(1))

            score_match = re.search(r"lavfi\.scene_score=([0-9.]+)", line)
            if score_match and current_time_val is not None:
                t = current_time_val
                score = float(score_match.group(1))
                if t > current_time:
                    scores.append((t, score))
                current_time_val = None

        process.wait()

        if scores:
            with open(cache_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(scores)
            current_time = scores[-1][0]
        else:
            current_time = end_time

    print(f"[extract_scene_scores] Finished extracting frames for {os.path.basename(video_path)}")
    return video_path, cache_file


def load_scene_scores(cache_file: str) -> list[tuple[float, float]]:
    scores = []
    with open(cache_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append((float(row["pts_time"]), float(row["scene_score"])))
    return scores


def find_candidate_cuts(times: np.ndarray, scores: np.ndarray) -> list[float]:
    """Dynamic-prominence peak detection over the raw scene-score signal.

    Prominence is anchored to the video's own score distribution (median +
    std, clamped) rather than a fixed threshold, so cut sensitivity adapts
    per video (report 2's "point-anchored" framing).
    """
    from scipy.signal import find_peaks

    dyn_prominence = float(np.median(scores) + np.std(scores))
    dyn_prominence = max(0.01, min(dyn_prominence, 0.2))

    peaks, _properties = find_peaks(scores, prominence=dyn_prominence, distance=15)
    return [float(times[p]) for p in peaks]


def filter_false_cuts(
    video_path: str, end_t: float, base_cuts: list[float]
) -> list[float]:
    """Discard candidate cuts that are actually continuous camera pans.

    Compares HSV 2D histograms just before/after each short-boundary cut
    (report 2's "Pairwise HSV Histogram correlation"); a high correlation
    means the same background persisted through the "cut", so it is fused
    away rather than kept as a hard scene boundary.
    """
    import cv2

    suspect_cuts = set()
    for i in range(len(base_cuts)):
        t = base_cuts[i]
        prev_t = base_cuts[i - 1] if i > 0 else 0.0
        next_t = base_cuts[i + 1] if i < len(base_cuts) - 1 else end_t

        if (t - prev_t < 1.0) or (next_t - t < 1.0):
            suspect_cuts.add(t)

    if not suspect_cuts:
        return base_cuts

    from moviepy.editor import VideoFileClip

    valid_cuts = []
    clip = None
    try:
        clip = VideoFileClip(video_path)
        for t in base_cuts:
            if t in suspect_cuts:
                t1 = max(0.0, t - 0.05)
                t2 = min(end_t, t + 0.05)
                try:
                    frame1 = clip.get_frame(t1)
                    frame2 = clip.get_frame(t2)

                    frame1 = cv2.resize(frame1, (320, 180))
                    frame2 = cv2.resize(frame2, (320, 180))

                    hsv1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2HSV)
                    hsv2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2HSV)

                    hist1 = cv2.calcHist([hsv1], [0, 1], None, [30, 32], [0, 180, 0, 256])
                    hist2 = cv2.calcHist([hsv2], [0, 1], None, [30, 32], [0, 180, 0, 256])

                    cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                    cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

                    similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

                    if similarity > 0.85:
                        continue
                except Exception:
                    pass
            valid_cuts.append(t)
    except Exception:
        valid_cuts = base_cuts
    finally:
        if clip is not None:
            clip.close()

    return valid_cuts


def _scene_stats_for_span(times: np.ndarray, scores: np.ndarray, t_start: float, t_end: float) -> dict[str, Any]:
    duration = t_end - t_start
    idx_start = np.searchsorted(times, t_start, side="right")
    idx_end = np.searchsorted(times, t_end, side="left")
    scene_scores = scores[idx_start:idx_end]

    if len(scene_scores) > 0:
        avg_score = float(np.mean(scene_scores))
        std_score = float(np.std(scene_scores))
        max_score = float(np.max(scene_scores))
        med_score = float(np.median(scene_scores))
        energy_score = float(np.sum(scene_scores))
    else:
        avg_score = std_score = max_score = med_score = energy_score = 0.0

    classification = "error_blank" if max_score < 0.0001 else "unknown"

    return {
        "t_start": t_start,
        "t_end": t_end,
        "duration": duration,
        "avg_score": avg_score,
        "std_score": std_score,
        "max_score": max_score,
        "med_score": med_score,
        "energy_score": energy_score,
        "classification": classification,
    }


def classify_scene_stats(
    times: np.ndarray, scores: np.ndarray, cut_times: list[float]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Classify already-cut scenes into point/interlude/other clusters.

    Pure given `times`/`scores`/`cut_times` (no filesystem or video I/O) so
    it is directly unit-testable and reusable at runtime. Returns
    `(scene_stats, raw_centroids)` mirroring the two structures
    `scripts/process_dataset.py:classify_scenes` used to build its JSON
    output and confidence scores from.
    """
    start_t = 0.0
    end_t = times[-1] if len(times) > 0 else 0.0

    T = [start_t] + cut_times + [end_t]
    T = sorted(set(T))

    scene_stats = [
        _scene_stats_for_span(times, scores, T[i], T[i + 1]) for i in range(len(T) - 1)
    ]

    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import RobustScaler

    valid_scenes = [s for s in scene_stats if s["classification"] != "error_blank"]
    raw_centroids: dict[str, Any] = {}

    if len(valid_scenes) >= 6:
        features = np.array(
            [
                [
                    np.log10(s["duration"] + 1e-6),
                    np.log10(s["avg_score"] + 1e-6),
                    np.log10(s["std_score"] + 1e-6),
                    np.log10(s["max_score"] + 1e-6),
                    np.log10(s["med_score"] + 1e-6),
                    np.log10(s["energy_score"] + 1e-6),
                ]
                for s in valid_scenes
            ]
        )

        scaler = RobustScaler()
        scaled_features = scaler.fit_transform(features)

        pca = PCA(n_components=2, random_state=42)
        pca_features = pca.fit_transform(scaled_features)

        best_gmm = None
        best_bic = np.inf

        max_k = min(6, len(valid_scenes) // 2)
        for k in range(2, max_k + 1):
            gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42, n_init=3)
            gmm.fit(pca_features)
            bic = gmm.bic(pca_features)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm

        assert best_gmm is not None
        labels = best_gmm.predict(pca_features)

        for idx, s in enumerate(valid_scenes):
            s["classification"] = f"cluster_{labels[idx]}"

        cluster_replay_scores = {}
        for k in np.unique(labels):
            c_scenes = [s for i, s in enumerate(valid_scenes) if labels[i] == k]
            score = np.mean([s["std_score"] / (s["duration"] + 1e-6) for s in c_scenes])
            cluster_replay_scores[k] = score

        replay_cluster_id = max(cluster_replay_scores, key=lambda ck: cluster_replay_scores[ck])
        replay_class_name = f"cluster_{replay_cluster_id}"

        merged_scenes = []
        i = 0
        while i < len(scene_stats):
            s = scene_stats[i]

            if s["classification"] != replay_class_name:
                merged_scenes.append(s)
                i += 1
                continue

            j = i + 1
            while j < len(scene_stats):
                if scene_stats[j]["classification"] == replay_class_name:
                    j += 1
                else:
                    break

            if j > i + 1:
                merged = _scene_stats_for_span(
                    times, scores, scene_stats[i]["t_start"], scene_stats[j - 1]["t_end"]
                )
                merged["classification"] = replay_class_name
                merged_scenes.append(merged)
                i = j
            else:
                merged_scenes.append(s)
                i += 1

        scene_stats = merged_scenes

        valid_merged = [s for s in scene_stats if s["classification"] != "error_blank"]
        unique_clusters = list({s["classification"] for s in valid_merged})

        for cname in unique_clusters:
            c_scenes = [s for s in valid_merged if s["classification"] == cname]
            if not c_scenes:
                continue

            durs = [s["duration"] for s in c_scenes]
            avgs = [s["avg_score"] for s in c_scenes]
            stds = [s["std_score"] for s in c_scenes]
            maxs = [s["max_score"] for s in c_scenes]

            raw_centroids[cname] = {
                "count": len(c_scenes),
                "duration": {"mean": float(np.mean(durs)), "min": float(np.min(durs)), "max": float(np.max(durs))},
                "avg_score": {"mean": float(np.mean(avgs)), "min": float(np.min(avgs)), "max": float(np.max(avgs))},
                "std_score": {"mean": float(np.mean(stds)), "min": float(np.min(stds)), "max": float(np.max(stds))},
                "max_score": {"mean": float(np.mean(maxs)), "min": float(np.min(maxs)), "max": float(np.max(maxs))},
            }

        merged_features_raw = np.array(
            [
                [
                    np.log10(s["duration"] + 1e-6),
                    np.log10(s["avg_score"] + 1e-6),
                    np.log10(s["std_score"] + 1e-6),
                    np.log10(s["max_score"] + 1e-6),
                    np.log10(s["med_score"] + 1e-6),
                    np.log10(s["energy_score"] + 1e-6),
                ]
                for s in valid_merged
            ]
        )

        merged_scaled = scaler.transform(merged_features_raw)
        merged_pca = pca.transform(merged_scaled)

        probas = best_gmm.predict_proba(merged_pca)

        for i, s in enumerate(valid_merged):
            cname = s["classification"]
            try:
                cluster_id = int(cname.replace("cluster_", ""))
                s["cluster_confidence"] = float(probas[i][cluster_id])
            except Exception:
                s["cluster_confidence"] = 0.0

        if raw_centroids:
            total_scenes = len([s for s in scene_stats if s["classification"] != "error_blank"])
            valid_cnames = [k for k, v in raw_centroids.items() if v["count"] >= total_scenes * 0.05]
            if not valid_cnames:
                valid_cnames = list(raw_centroids.keys())

            point_cluster = min(valid_cnames, key=lambda k: raw_centroids[k]["avg_score"]["mean"])
            point_mean = raw_centroids[point_cluster]["avg_score"]["mean"]
            point_max = raw_centroids[point_cluster]["avg_score"]["max"]

            dynamic_threshold = max(point_max * 1.5, point_mean * 2.0)

            target_mapping = {}
            for cluster_name, centroid in raw_centroids.items():
                if centroid["avg_score"]["mean"] > dynamic_threshold:
                    target_mapping[cluster_name] = "cluster_interlude"
                else:
                    target_mapping[cluster_name] = "cluster_point"

            conf_scores = [s.get("cluster_confidence", 0.0) for s in scene_stats if s.get("cluster_confidence", 0.0) > 0]
            if conf_scores:
                threshold = np.percentile(conf_scores, 10)
                threshold = min(threshold, 0.98)
            else:
                threshold = 0.98

            for s in scene_stats:
                if s["classification"] == "error_blank":
                    continue
                cname = s["classification"]
                conf = s.get("cluster_confidence", 0.0)

                target_class = target_mapping.get(cname, "cluster_other")
                if conf >= threshold:
                    s["classification"] = target_class
                else:
                    s["classification"] = "cluster_other"

    else:
        for s in valid_scenes:
            if s["avg_score"] < 0.005 and s["duration"] > 3.0:
                s["classification"] = "cluster_point"
            else:
                s["classification"] = "cluster_other"

        for s in scene_stats:
            s["cluster_confidence"] = 0.0

    return scene_stats, raw_centroids


_CLUSTER_TO_SCENE_CLASS = {
    "error_blank": SceneClass.BLANK,
    "cluster_point": SceneClass.POINT,
    "cluster_interlude": SceneClass.INTERLUDE,
    "cluster_other": SceneClass.OTHER,
}


def to_scene_spans(scene_stats: list[dict[str, Any]]) -> list[SceneSpan]:
    """Map the legacy dict/cluster-name representation onto typed `SceneSpan`s."""
    spans = []
    for s in scene_stats:
        cluster = s["classification"]
        scene_class = _CLUSTER_TO_SCENE_CLASS.get(cluster, SceneClass.OTHER)
        spans.append(
            SceneSpan(
                t_start=s["t_start"],
                t_end=s["t_end"],
                scene_class=scene_class,
                confidence=s.get("cluster_confidence", 0.0),
                avg_score=s["avg_score"],
                std_score=s["std_score"],
                max_score=s["max_score"],
            )
        )
    return spans


def detect_and_classify_scenes(
    video_path: str, cache_file: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """End-to-end: cached scores -> candidate cuts -> false-cut filtering -> classification.

    Requires `cache_file` (the `scene_scores.csv` produced by
    `extract_scene_scores`) to already exist. Returns the legacy
    `(scene_stats, raw_centroids)` pair; callers that want typed output
    should pass `scene_stats` through `to_scene_spans`.
    """
    scores_data = load_scene_scores(cache_file)
    if not scores_data:
        return [], {}

    times = np.array([t for t, _s in scores_data])
    scores = np.array([s for _t, s in scores_data])

    candidate_cuts = find_candidate_cuts(times, scores)
    end_t = times[-1] if len(times) > 0 else 0.0
    valid_cuts = filter_false_cuts(video_path, end_t, candidate_cuts)

    return classify_scene_stats(times, scores, valid_cuts)


def classify_video_scenes(video_path: str, cache_file: str) -> list[SceneSpan]:
    """Typed entry point for runtime callers (report 10 Phase 2 orchestrator).

    `cache_file` must already hold the full `scene_scores.csv` for
    `video_path` (call `extract_scene_scores` first) — cuts are deterministic
    and shared across all model tiers, so this never re-derives them per
    tier (report 10 "Deterministic shared segmentation").
    """
    scene_stats, _raw_centroids = detect_and_classify_scenes(video_path, cache_file)
    return to_scene_spans(scene_stats)
