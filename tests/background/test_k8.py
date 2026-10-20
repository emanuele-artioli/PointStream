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
            "registration": [{"frame": t, "chamfer_px": error} for t in range(start, end)]}


def scoring_report(shots):
    frames = shots[-1]["end"]
    return {"review_id": "a", "fps": 10, "frames": frames, "source_frames": frames, "shots": shots}


def test_scoring_duration_recall_and_mixed_shot_penalizes_both():
    report = scoring_report([shot("a", 0, 90, "court"),
              shot("b", 90, 100, "non_court"), shot("c", 100, 110, "court")])
    result = k8.score(report, {"review_id": "a", "marks": {"a": "court", "b": "court", "c": "mixed"}})
    assert result["precision"] == 0.5
    assert result["duration_recall"] == 9 / 11
    assert result["misses_by_class"] == {"court": 1, "mixed": 1}
    assert not result["adopted"]


def test_missing_registrations_fail_error_bar_and_marks_must_be_complete():
    report = scoring_report([shot("a", 0, 100, "court", None)])
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


def test_chamfer_rejects_an_entirely_out_of_frame_projection():
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


def test_chamfer_samples_uniformly_in_projected_pixels():
    _, hom = synthetic()
    model, points, groups = court.samples(hom)
    assert np.max(np.linalg.norm(court.project(model, hom) - points, axis=1)) < 0.001
    for group in range(len(court.SEGMENTS)):
        steps = np.linalg.norm(np.diff(points[groups == group], axis=0), axis=1)
        assert steps.max() <= 1.001
        assert steps.max() - steps.min() < 0.001


def test_ffmpeg_scene_scan_bounds_input_before_scene_selection(tmp_path):
    import os
    import shutil
    import subprocess

    ffmpeg = os.environ.get("PS_K8_TEST_FFMPEG") or shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg integration fixture")
    version = subprocess.run([ffmpeg, "-version"], capture_output=True)
    if version.returncode:
        pytest.skip("installed ffmpeg cannot start")
    video = tmp_path / "three-spans.mkv"
    frames = np.zeros((30, 48, 64, 3), np.uint8)
    frames[5:20] = 255
    subprocess.run([ffmpeg, "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "64x48", "-r", "10",
                    "-i", "pipe:0", "-an", "-c:v", "ffv1", str(video)], input=frames.tobytes(), check=True)
    rows, _ = k8.cuts(video, 10, 10, ffmpeg, tmp_path)
    assert [r["frame"] for r in rows] == [5]  # cut at 20 must NOT be scanned


def test_chamfer_measures_visible_portions_of_a_clipped_court():
    rgb, hom = synthetic()
    shifted = np.roll(rgb, -400, axis=1)
    shifted[:, -400:] = [55, 105, 55]
    hom = np.array([[1, 0, -400], [0, 1, 0], [0, 0, 1]], dtype=float) @ hom
    error = court.chamfer(hom, court.white_lines(shifted))
    assert error["chamfer_px"] < 1
    assert 0.5 < error["visible_fraction"] < 1
    assert court.register(shifted, hom)["registered"]


def test_validator_reads_published_archive_after_scratch_cleanup(tmp_path, monkeypatch):
    import io
    import tarfile

    monkeypatch.setenv("PS_STAGE_DIR", str(tmp_path))
    monkeypatch.setenv("PS_SCRATCH_DIR", str(tmp_path / "deleted-scratch"))
    monkeypatch.setenv("PS_VALIDATION_PATH", str(tmp_path / "validation.json"))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    report = {"size": [1920, 1080], "frames": 1, "all_spans": [[0, 1]], "part": -1, "parts": 1, "geometry": "ITF regulation; net excluded", "shots": [
        {"id": "shot-0000", "start": 0, "end": 1, "predicted": "court", "images": ["images/a.jpg"] * 3,
         "registration": [{"frame": 0, "homography": np.eye(3).tolist(), "registered": True, "chamfer_px": 1.0}]}]}
    with tarfile.open(tmp_path / "published.tar", "w") as archive:
        for name, data in (("publish/k8-report.json", json.dumps(report).encode()), ("publish/images/a.jpg", b"image")):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    assert k8.validate_job() == 0
    assert json.loads((tmp_path / "validation.json").read_text())["passed"]


def test_whole_shot_partitions_cover_all_frames_once_and_preserve_ids():
    spans = k8.shot_spans([100, 130, 200, 205, 280, 340], 400)
    selected = [k8.selected_spans(spans, i, 3) for i in range(3)]
    assert sorted(row for group in selected for row in group) == k8.selected_spans(spans, -1, 3)
    assert all(b > a for group in selected for _, a, b in group)
    loads = [sum(b - a for _, a, b in group) for group in selected]
    assert max(loads) - min(loads) <= max(b - a for a, b in spans)


def test_assembly_requires_complete_nonduplicated_frame_coverage(tmp_path):
    import hashlib
    import io
    import tarfile

    source = tmp_path / "source.mp4"
    source.write_bytes(b"immutable fixture source")
    identity = {"video_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    archives = []
    for part in range(2):
        ident = f"shot-{part:04d}"
        row = shot(ident, part * 2, (part + 1) * 2, "court")
        for frame, reg in enumerate(row["registration"], part * 2):
            reg["frame"] = frame
        row.update(images=[f"images/{ident}.jpg"] * 3, registered_share=1.0)
        report = {"source": identity, "fps": 50, "frames": 4, "source_frames": 4, "runtime_seconds": 10 + part, "scene_threshold": 0.25,
                  "scene_scores": [], "decision": k8.DECISION, "all_spans": [[0, 2], [2, 4]], "shots": [row]}
        path = tmp_path / f"part-{part}.tar"
        with tarfile.open(path, "w") as archive:
            for name, data in (("publish/k8-report.json", json.dumps(report).encode()),
                               (f"publish/images/{ident}.jpg", b"image")):
                info = tarfile.TarInfo(name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        archives.append(path)
    result = k8.assemble(archives, tmp_path / "review", source)
    assert result["frames"] == 4 and result["shots"] == 2
    assembled = json.loads((tmp_path / "review/k8-report.json").read_text())
    assert "runtime_seconds" not in assembled
    assert [r["runtime_seconds"] for r in assembled["part_runs"]] == [10, 11]
    assert (tmp_path / "review/source.mp4").resolve() == source
    assert (tmp_path / "review/READY.json").exists()
    with pytest.raises(ValueError, match="cover every"):
        k8.assemble(archives[:1], tmp_path / "incomplete", source)
    assert not (tmp_path / "incomplete/READY.json").exists()
    with pytest.raises(ValueError, match="duplicate"):
        k8.assemble([archives[0], archives[0]], tmp_path / "duplicate", source)


def test_error_summary_uses_ordinary_median_and_retains_infinite_tail():
    row = shot("a", 0, 4, "court")
    for reg, error in zip(row["registration"], [1.0, 1.0, 3.0, 3.0]):
        reg["chamfer_px"] = error
    report = scoring_report([row])
    marks = {"review_id": "a", "marks": {"a": "court"}}
    assert k8.score(report, marks)["registration_median_px"] == 2.0
    row["registration"][-1]["chamfer_px"] = None
    assert k8.score(report, marks)["registration_p95_px"] is None


def test_review_server_supports_byte_ranges_for_seeking(tmp_path):
    import http.client
    import threading
    from functools import partial
    from http.server import ThreadingHTTPServer
    from tools.datasets.k8_review_server import RangeHandler, byte_range

    assert byte_range("bytes=2-4", 10) == (2, 5)
    assert byte_range("bytes=-3", 10) == (7, 10)
    assert byte_range("bytes=7-", 10) == (7, 10)
    for header in ("bytes=10-", "bytes=-0", "bytes=7-2", "bytes=1-2,4-6"):
        with pytest.raises(ValueError):
            byte_range(header, 10)
    (tmp_path / "source.mp4").write_bytes(b"0123456789")
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), partial(RangeHandler, directory=str(tmp_path)))
    except PermissionError:
        pytest.skip("loopback server unavailable in sandbox")
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        connection.request("GET", "/source.mp4", headers={"Range": "bytes=2-4"})
        response = connection.getresponse()
        assert response.status == 206 and response.getheader("Content-Range") == "bytes 2-4/10"
        assert response.read() == b"234"
        connection.request("GET", "/source.mp4", headers={"Range": "bytes=100-"})
        response = connection.getresponse()
        assert response.status == 416 and response.read() == b""
        connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_scoring_rejects_partial_matches_and_dropped_frame_rows():
    report = scoring_report([shot("a", 0, 10, "court")])
    marks = {"review_id": "a", "marks": {"a": "court"}}
    report["source_frames"] = 20
    with pytest.raises(ValueError, match="whole-match"):
        k8.score(report, marks)
    report["source_frames"] = 10
    report["shots"][0]["registration"].pop()
    with pytest.raises(ValueError, match="every shot frame"):
        k8.score(report, marks)


def sample_report(count=60):
    """Two predicted-court shots, then non-court shots of varied length (fps 10)."""
    shots, start = [], 0
    for i in range(count):
        length = 20 + (i * 37) % 300
        shots.append(shot(f"shot-{i:04d}", start, start + length, "court" if i < 2 else "non_court"))
        start += length
    return scoring_report(shots)


def test_inclusion_is_proportional_to_duration_capped_and_sums_to_expected():
    pi, c = k8.inclusion_probabilities({"a": 100, "b": 1, "c": 1, "d": 2}, 2)
    assert pi == {"a": 1.0, "b": 0.25, "c": 0.25, "d": 0.5} and c == 4
    assert sum(pi.values()) == pytest.approx(2)
    pi, c = k8.inclusion_probabilities({"a": 1, "b": 3}, 5)
    assert pi == {"a": 1.0, "b": 1.0} and c == float("inf")


def test_sample_is_content_blind_nested_and_keeps_every_predicted_court_shot():
    report = sample_report()
    small, large = k8.stratified_sample(report, 8), k8.stratified_sample(report, 25)
    assert small["court_shots"] == large["court_shots"] == ["shot-0000", "shot-0001"]
    assert set(small["sampled_non_court"]) < set(large["sampled_non_court"])
    assert small["review_id"] != large["review_id"] != report["review_id"]
    # The draw depends only on the shot ID, never on its registration or pixels.
    report["shots"][5]["registration"][0]["chamfer_px"] = 99.0
    assert k8.stratified_sample(report, 8)["sampled_non_court"] == small["sampled_non_court"]
    assert k8.sample_uniform("shot-0005") == k8.sample_uniform("shot-0005") != k8.sample_uniform("shot-0006")


def test_sampled_recall_is_horvitz_thompson_with_a_poisson_interval(monkeypatch):
    report = sample_report()
    monkeypatch.setattr(k8, "SAMPLE_STAGES", (8.0, 25.0))
    design = k8.stratified_sample(report, 8)
    sampled = design["sampled_non_court"]
    assert len(sampled) >= 2
    hidden = sampled[0]
    marks = {"shot-0000": "court", "shot-0001": "court", **{k: "crowd" for k in sampled}, hidden: "court"}
    result = k8.score(report, {"review_id": design["review_id"], "marks": marks})
    durations = {s["id"]: (s["end"] - s["start"]) / 10 for s in report["shots"]}
    a = durations["shot-0000"] + durations["shot-0001"]
    pi = design["inclusion"][hidden]
    assert pi < 1 and durations[hidden] / pi == pytest.approx(design["scale_seconds"])
    assert result["duration_recall"] == pytest.approx(a / (a + durations[hidden] / pi))
    lo_k, hi_k = k8.poisson_interval(1)
    c = design["scale_seconds"]
    assert result["duration_recall_ci95"] == pytest.approx(
        [a / (a + c * hi_k), a / (a + max(durations[hidden], c * lo_k))])
    assert result["precision"] == 1.0 and result["misses_by_class"] == {"court": 1}
    assert result["misses_by_class_estimated"]["court"] == pytest.approx(1 / pi)
    # Frames of the hidden court shot weigh 1/pi in the registration percentiles.
    assert result["court_frames_weighted"] == pytest.approx(a * 10 + durations[hidden] * 10 / pi)
    with pytest.raises(ValueError):
        k8.score(report, {"review_id": design["review_id"], "marks": {**marks, "shot-0059": "crowd"}})


def test_straddling_stage_one_extends_and_the_last_stage_decides_on_the_point(monkeypatch):
    report = sample_report()
    monkeypatch.setattr(k8, "SAMPLE_STAGES", (8.0, 25.0))
    for expected, status in ((8.0, "extend_sample"), (25.0, "adopted")):
        design = k8.stratified_sample(report, expected)
        marks = {"shot-0000": "court", "shot-0001": "court", **{k: "crowd" for k in design["sampled_non_court"]}}
        result = k8.score(report, {"review_id": design["review_id"], "marks": marks})
        assert result["duration_recall"] == 1.0 and result["recall_ci_straddles_bar"]
        assert result["status"] == status
    assert k8.poisson_interval(0) == pytest.approx((0.0, 3.6888794541139363))


def test_weighted_percentile_reduces_to_linear_and_keeps_infinite_tail():
    values = [3.0, 1.0, 7.5, 2.0, 4.0]
    for q in (0.0, 0.3, 0.5, 0.95, 1.0):
        assert k8.weighted_percentile(values, [2.0] * 5, q) == pytest.approx(np.percentile(values, 100 * q))
    assert k8.weighted_percentile([1.0, 2.0, float("inf")], [1.0, 1.0, 1.0], 0.95) == float("inf")
    # A heavier small value pulls the median down.
    assert k8.weighted_percentile([1.0, 2.0, 3.0], [5.0, 1.0, 1.0], 0.5) < 2.0


def test_sample_page_shows_only_the_sample_and_keeps_the_full_review(tmp_path):
    report = sample_report()
    report["scene_threshold"] = 0.25
    for s in report["shots"]:
        s.update(images=[f"images/{s['id']}-{r}.jpg" for r in ("first", "middle", "last")], registered_share=0.5)
    (tmp_path / "review").mkdir()
    path = tmp_path / "review" / "k8-report.json"
    path.write_text(json.dumps(report))
    summary = k8.write_sample(path, tmp_path / "review-sample-01", 8)
    page = (tmp_path / "review-sample-01" / "review.html").read_text()
    design = json.loads((tmp_path / "review-sample-01" / "k8-sample.json").read_text())
    assert page.count("<article") == summary["shots"] == 2 + len(design["sampled_non_court"]) < 60
    assert json.dumps(design["review_id"]) in page and 'id="shot-0000"' in page
    assert (tmp_path / "review-sample-01" / "images").readlink() == k8.Path("../review/images")
    assert design["base_report_sha256"] == k8.file_sha256(path)
