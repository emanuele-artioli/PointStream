"""K8 court shots: CPU white-line registration, shot review and fixed-rule scoring.

All generated data belongs under Datasets/pointstream-data (fleet scratch is temporary).
Shot intervals and frame indices are half-open, zero-based decoded presentation order.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from experiments.visor.b1 import file_sha256, write_json

DECISION = {"shot_registered_share": 0.80, "precision": 0.95, "duration_recall": 0.90,
            "median_px": 2.0, "p95_px": 6.0}
CLASSES = ("court", "close_up", "crowd", "replay", "graphics", "other", "mixed")
SAMPLE_SEED = "pointstream-k8"
# Amendment 1: expected sampled non-court shots per nested stage; stage 2 runs only
# when stage 1's recall interval straddles the bar and the other conditions pass.
SAMPLE_STAGES = (50.0, 150.0)


def shot_spans(cuts: list[int], frames: int) -> list[tuple[int, int]]:
    if frames <= 0 or any(c <= 0 or c >= frames for c in cuts):
        raise ValueError("cuts must be interior frame boundaries")
    edges = [0, *sorted(set(cuts)), frames]
    return list(zip(edges, edges[1:]))


def parse_scenes(text: str, fps: float) -> list[dict[str, Any]]:
    """ffmpeg metadata prints selected-frame index, source PTS and scene score.

    Use source PTS rather than the selected-frame index; validate CFR separately.
    """
    rows: list[dict[str, Any]] = []
    pts: float | None = None
    for line in text.splitlines():
        match = re.search(r"pts_time:([\d.e+\-]+)", line)
        if match:
            pts = float(match[1])
        match = re.search(r"lavfi.scene_score=([\d.e+\-]+)", line)
        if match and pts is not None:
            rows.append({"frame": round(pts * fps), "pts_seconds": pts, "score": float(match[1])})
            pts = None
    return rows


def shot_prediction(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("empty shot")
    share = sum(r["registered"] for r in rows) / len(rows)
    return {"predicted": "court" if share >= DECISION["shot_registered_share"] else "non_court",
            "registered_share": share}


def sample_uniform(ident: str) -> float:
    """Content-blind uniform draw in [0, 1), fixed by the shot ID (amendment 1)."""
    return int(hashlib.sha256(f"{SAMPLE_SEED}:{ident}".encode()).hexdigest(), 16) / 2 ** 256


def inclusion_probabilities(durations: dict[str, float], expected: float) -> tuple[dict[str, float], float]:
    """Probability proportional to duration, capped at 1 iteratively; returns (pi, c).

    Below the cap pi = d / c, so every sampled positive adds exactly c seconds to the
    Horvitz-Thompson total. c is infinite when every shot is taken with certainty.
    """
    if expected <= 0 or any(d <= 0 for d in durations.values()):
        raise ValueError("positive durations and expected size required")
    certain: set[str] = set()
    while True:
        rest = {k: d for k, d in durations.items() if k not in certain}
        slots = expected - len(certain)
        if not rest or slots <= 0:
            return {k: 1.0 for k in durations}, float("inf")
        c = sum(rest.values()) / slots
        capped = {k for k, d in rest.items() if d >= c}
        if not capped:
            return {k: (1.0 if k in certain else d / c) for k, d in durations.items()}, c
        certain |= capped


def stratified_sample(report: dict[str, Any], expected: float) -> dict[str, Any]:
    """Every predicted-court shot plus a Poisson PPS sample of predicted non-court shots.

    Nested in `expected`: pi grows with it, and the draws u are fixed per shot.
    """
    fps = report["fps"]
    court = [s["id"] for s in report["shots"] if s["predicted"] == "court"]
    durations = {s["id"]: (s["end"] - s["start"]) / fps for s in report["shots"] if s["predicted"] != "court"}
    pi, c = inclusion_probabilities(durations, expected)
    sampled = [k for k in durations if sample_uniform(k) < pi[k]]
    design = {"schema": 1, "base_review_id": report["review_id"], "seed": f"sha256('{SAMPLE_SEED}:<shot id>')/2^256",
              "expected_non_court": float(expected), "scale_seconds": c if np.isfinite(c) else None,
              "court_shots": court, "sampled_non_court": sampled,
              "inclusion": {k: pi[k] for k in sampled}, "non_court_shots": len(durations),
              "non_court_seconds": sum(durations.values())}
    design["review_id"] = hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()[:16]
    return design


def poisson_interval(k: int, level: float = 0.95) -> tuple[float, float]:
    """Exact (Garwood) interval for a Poisson mean from one count."""
    from scipy.stats import chi2

    alpha = 1 - level
    lo = 0.0 if k == 0 else float(chi2.ppf(alpha / 2, 2 * k)) / 2
    return lo, float(chi2.ppf(1 - alpha / 2, 2 * k + 2)) / 2


def weighted_percentile(values: list[float], weights: list[float], q: float) -> float:
    """Linear interpolation at cumulative-weight positions S_{i-1} / (W - w_N).

    Equal weights give positions i/(N-1), the ordinary linear percentile. Infinite
    (missing) values are retained: a bracket ending at infinity returns infinity.
    """
    if not values:
        return float("inf")
    order = np.argsort(values, kind="stable")
    v, w = np.asarray(values, float)[order], np.asarray(weights, float)[order]
    if len(v) == 1:
        return float(v[0])
    pos = np.concatenate([[0.0], np.cumsum(w)[:-1]]) / (w.sum() - w[-1])
    hi = int(np.searchsorted(pos, q, side="left"))
    if not np.isfinite(v[hi]):
        return float("inf")
    if pos[hi] == q:
        return float(v[hi])
    lo = hi - 1
    return float(v[lo] + (q - pos[lo]) / (pos[hi] - pos[lo]) * (v[hi] - v[lo]))


def score(report: dict[str, Any], marks: dict[str, Any]) -> dict[str, Any]:
    """Mixed shots are wrong for both: a predicted court is FP and their duration is FN.

    Registration includes ALL true-court frames, with missing fits represented by
    infinity, never silently omitted. Use the ordinary median and linearly
    interpolated 95th percentile, retaining infinite failures.

    Marks for the whole review are a census. Marks for a stratified sample
    (amendment 1) are scored by Horvitz-Thompson: each marked shot weighs 1/pi.
    """
    design = None
    if marks.get("review_id") != report["review_id"]:
        design = next((d for d in (stratified_sample(report, n) for n in SAMPLE_STAGES)
                       if d["review_id"] == marks.get("review_id")), None)
        if design is None:
            raise ValueError("marks belong to another review")
    labels = marks["marks"]
    expected_ids = ({s["id"] for s in report["shots"]} if design is None
                    else set(design["court_shots"]) | set(design["sampled_non_court"]))
    if set(labels) != expected_ids or any(v not in CLASSES for v in labels.values()):
        raise ValueError("mark every shot exactly once with an allowed class")
    weight = {k: 1.0 for k in labels} if design is None else {
        **{k: 1.0 for k in design["court_shots"]}, **{k: 1 / p for k, p in design["inclusion"].items()}}
    shots = report["shots"]
    frames = report.get("source_frames", 0)
    if frames <= 0 or report.get("frames") != frames:
        raise ValueError("the rule requires a complete whole-match report")
    if [(s["start"], s["end"]) for s in shots] != shot_spans([s["start"] for s in shots[1:]], frames):
        raise ValueError("shots do not partition the whole match")
    if any([r["frame"] for r in s["registration"]] != list(range(s["start"], s["end"])) for s in shots):
        raise ValueError("registration must cover every shot frame, including failed fits")
    tp = fp = 0
    duration_tp = duration_true_predicted = 0.0
    sampled_true = estimated_true = capped_true = 0.0
    positives = 0
    errors: list[float] = []
    error_weights: list[float] = []
    misses: Counter[str] = Counter()
    missed_weighted: defaultdict[str, float] = defaultdict(float)
    missed_seconds: defaultdict[str, float] = defaultdict(float)
    missed_shots: dict[str, list[str]] = defaultdict(list)
    confusion: Counter[str] = Counter()
    for shot in shots:
        if shot["id"] not in labels:
            continue
        truth, predicted, w = labels[shot["id"]], shot["predicted"] == "court", weight[shot["id"]]
        duration = (shot["end"] - shot["start"]) / report["fps"]
        confusion[f"{truth}/{shot['predicted']}"] += 1
        if predicted:
            tp += truth == "court"
            fp += truth != "court"
        if truth in ("court", "mixed"):
            if predicted:
                duration_true_predicted += duration
                duration_tp += duration * (truth == "court")
            else:
                sampled_true += duration
                estimated_true += duration * w
                if w == 1.0:
                    capped_true += duration
                else:
                    positives += 1
        if (predicted and truth != "court") or (not predicted and truth in ("court", "mixed")):
            misses[truth] += 1
            missed_weighted[truth] += w
            missed_seconds[truth] += duration * w
            missed_shots[truth].append(shot["id"])
        if truth == "court":
            rows = shot["registration"]
            errors.extend(float(r["chamfer_px"]) if r["chamfer_px"] is not None else float("inf") for r in rows)
            error_weights.extend([w] * len(rows))
    precision = tp / (tp + fp) if tp + fp else 0.0
    true_seconds = duration_true_predicted + estimated_true
    recall = duration_tp / true_seconds if true_seconds else 0.0
    if design is None:
        interval = (recall, recall)
    else:
        lo_k, hi_k = poisson_interval(positives)
        c = design["scale_seconds"] or 0.0
        y_lo = max(sampled_true, capped_true + c * lo_k)
        y_hi = capped_true + c * hi_k
        lower, upper = (duration_tp / (duration_true_predicted + y) if duration_true_predicted + y else 0.0
                        for y in (y_hi, y_lo))
        interval = (lower, upper)
    median = weighted_percentile(errors, error_weights, 0.5)
    p95 = weighted_percentile(errors, error_weights, 0.95)
    bar = DECISION["duration_recall"]
    straddles = interval[0] < bar <= interval[1]
    final_stage = design is None or design["expected_non_court"] == SAMPLE_STAGES[-1]
    # None: the stage-1 interval straddles the bar, so recall is not yet decided.
    conditions: dict[str, bool | None] = {
        "precision": precision >= DECISION["precision"],
        "duration_recall": recall >= bar if final_stage or not straddles else None,
        "registration": median <= DECISION["median_px"] and p95 <= DECISION["p95_px"]}
    extend = conditions["duration_recall"] is None and bool(conditions["precision"] and conditions["registration"])
    status = "extend_sample" if extend else ("adopted" if all(conditions.values()) else "not_adopted")
    return {"decision": DECISION, "status": status, "adopted": status == "adopted", "conditions": conditions,
            "precision": precision, "duration_recall": recall, "duration_recall_ci95": list(interval),
            "recall_ci_straddles_bar": straddles,
            "next_stage_expected_non_court": (SAMPLE_STAGES[SAMPLE_STAGES.index(design["expected_non_court"]) + 1]
                                              if extend and design is not None else None),
            "sample": None if design is None else {k: design[k] for k in (
                "review_id", "expected_non_court", "scale_seconds", "non_court_shots", "non_court_seconds")} | {
                "sampled_non_court": len(design["sampled_non_court"]), "sampled_positives_uncapped": positives,
                "sampled_positive_seconds": sampled_true, "estimated_hidden_court_seconds": estimated_true},
            "registration_median_px": median if np.isfinite(median) else None,
            "registration_p95_px": p95 if np.isfinite(p95) else None,
            "missing_registration_is_infinite": True, "court_frames": len(errors),
            "court_frames_weighted": float(sum(error_weights)),
            "misses_by_class": dict(misses), "misses_by_class_estimated": dict(missed_weighted),
            "missed_seconds_by_class_estimated": dict(missed_seconds),
            "missed_shots_by_class": dict(missed_shots), "confusion": dict(confusion),
            "marks_sha256": hashlib.sha256(json.dumps(marks, sort_keys=True).encode()).hexdigest()}


def review_page(report: dict[str, Any], out: Path, design: dict[str, Any] | None = None) -> None:
    """Self-contained page apart from relative JPEGs; labels start blank, never predictions.

    With a stratified-sample design (amendment 1), only its shots are shown, in time
    order, and the marks file carries the sample's review ID.
    """
    chosen = None if design is None else set(design["court_shots"]) | set(design["sampled_non_court"])
    review_id = report["review_id"] if design is None else design["review_id"]
    heading = ("K8 · every broadcast shot" if design is None else
               f"K8 · {len(chosen or ())} shots: every predicted court shot and a duration-weighted sample of the rest")
    cards = []
    for s in report["shots"]:
        if chosen is not None and s["id"] not in chosen:
            continue
        images = "".join(f'<figure><img loading="lazy" src="{html.escape(p)}"><figcaption>{label}</figcaption></figure>'
                         for p, label in zip(s["images"], ("First", "Middle + projected court", "Last")))
        options = '<option value="">Choose true class…</option>' + ''.join(
            f'<option value="{c}">{c}</option>' for c in CLASSES)
        cards.append(f'<article id="{s["id"]}"><h2>{s["id"]} · {s["start"]/report["fps"]:.2f}–'
                     f'{s["end"]/report["fps"]:.2f} s</h2><p>Prediction: <b>{s["predicted"]}</b> · '
                     f'{s["registered_share"]:.1%} registered</p><div class="strip">{images}</div>'
                     f'<label>True class <select data-id="{s["id"]}">{options}</select></label>'
                     '<details><summary>Inspect the shot in the source video</summary>'
                     f'<video controls muted preload="none" src="source.mp4#t={s["start"]/report["fps"]:.3f},'
                     f'{s["end"]/report["fps"]:.3f}"></video></details></article>')
    template = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>K8 court-shot review</title>
<style>body{font:16px system-ui;margin:24px;background:#eee;color:#18212b}header{position:sticky;top:0;background:#fff;padding:12px;z-index:2}article{background:white;margin:20px 0;padding:16px;border-radius:8px}.strip{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}figure{margin:0;min-width:0}img{display:block;width:100%}video{width:70%;max-width:100%}select,button{font:inherit;padding:8px}figcaption{font-size:13px}</style>
<header><b>HEADING</b> <span id="count"></span> <button id="save">Download marks JSON</button>
<label>Restore marks <input id="load" type="file" accept="application/json"></label></header>
<p>Mark the true class of every shot. Court means an ordinary court view intended for PointStream; mark replays as replay even when the court is visible. Use mixed for a shot containing both court and non-court content (bad cut); it counts as wrong for both. Use the source video when three frames are insufficient. Green is the projected model; no overlay means no homography. Predictions are not labels. Labels are saved locally in this browser; also download the JSON.</p>
<p>Scene threshold: THRESHOLD. Initial threshold is provisional until this review. All scoring thresholds remain fixed.</p>
CARDS
<script>const reviewId=REVIEWID, key='pointstream-k8-'+reviewId;
let marks={};try{marks=JSON.parse(localStorage.getItem(key)||'{}')}catch(e){}
const selects=[...document.querySelectorAll('select')];
function update(){for(const s of selects)s.value=marks[s.dataset.id]||'';document.querySelector('#count').textContent=Object.keys(marks).length+' / '+selects.length+' marked';localStorage.setItem(key,JSON.stringify(marks))}
for(const s of selects)s.onchange=()=>{if(s.value)marks[s.dataset.id]=s.value;else delete marks[s.dataset.id];update()};update();
document.querySelector('#save').onclick=()=>{const blob=new Blob([JSON.stringify({schema:1,review_id:reviewId,marks},null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='k8-marks-'+reviewId+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)};
document.querySelector('#load').onchange=async e=>{try{const r=JSON.parse(await e.target.files[0].text());if(r.review_id!==reviewId)throw Error('Different review');if(Object.entries(r.marks).some(([k,v])=>!selects.some(s=>s.dataset.id===k)||!CLASSES.includes(v)))throw Error('Invalid labels');marks=r.marks;update()}catch(err){alert(err.message)}};
</script></html>'''
    out.write_text(template.replace('HEADING', html.escape(heading)).replace('THRESHOLD', str(report['scene_threshold'])).replace('CARDS', '\n'.join(cards))
                   .replace('REVIEWID', json.dumps(review_id)).replace('!CLASSES.includes', f'!{json.dumps(CLASSES)}.includes'))


def write_sample(report_path: Path, out: Path, expected: float) -> dict[str, Any]:
    """Amendment 1 review: a new directory beside the 360-shot review, which stays unchanged.

    Images and the source link point into the sibling review directory (relative links).
    """
    report = json.loads(report_path.read_text())
    design = stratified_sample(report, expected)
    design["base_report_sha256"] = file_sha256(report_path)
    out.mkdir(parents=True, exist_ok=False)
    for name in ("images", "source.mp4"):
        (out / name).symlink_to(Path("..") / report_path.parent.name / name)
    write_json(out / "k8-sample.json", design)
    review_page(report, out / "review.html", design)
    return {"review": str(out / "review.html"), "review_id": design["review_id"],
            "shots": len(design["court_shots"]) + len(design["sampled_non_court"]),
            "sampled_non_court": len(design["sampled_non_court"])}


def cuts(video: Path, fps: float, limit: int, ffmpeg: str, scratch: Path) -> tuple[list[dict[str, Any]], list[str]]:
    log = scratch / "scene-metadata.txt"
    command = [ffmpeg, "-v", "error", "-nostdin", "-threads", "2", "-t", str(limit / fps), "-i", str(video),
               "-vf", f"setpts=PTS-STARTPTS,select='gte(scene,0.10)',metadata=print:file={log}", "-an", "-f", "null", "-"]
    subprocess.run(command, check=True, capture_output=True, timeout=1800)
    return parse_scenes(log.read_text(), fps), command


def process_shot(video: Path, fps: float, start: int, end: int, ident: str, publish: Path, ffmpeg: str, extra_review: dict[str, int] | None = None,
                 image_prefix: str | None = None) -> dict[str, Any]:
    import cv2
    from experiments.tennis import court
    from experiments.tennis import windows as tw

    window = {"id": ident, "size": [1920, 1080], "fps": fps, "source": {"kind": "video"}}
    command = tw.convert_command(ffmpeg, window, video, Path("pipe:1"), end - start)
    command[command.index("-i"):command.index("-i")] = ["-ss", str(start / fps)]
    command[command.index("-i"):command.index("-i")] = ["-threads", "1"]
    # Do not use -y pipe:1 as a filesystem path: ffmpeg recognizes the pipe protocol.
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    frame_size = 1920 * 1080 * 3 // 2
    rows, images = [], {}
    requests = ([(index, role, ident) for index, role in
                 ((0, "first"), ((end - start) // 2, "middle"), (end - start - 1, "last"))]
                if extra_review is None else [(frame - start, role, image_prefix or ident) for role, frame in extra_review.items()])
    previous = None
    try:
        for i in range(end - start):
            data = bytearray()
            while len(data) < frame_size:
                block = proc.stdout.read(frame_size - len(data))
                if not block:
                    raise RuntimeError(f"{ident}: truncated frame {i}")
                data.extend(block)
            yuv = np.frombuffer(data, dtype=np.uint8).reshape(1080 * 3 // 2, 1920)
            rgb = tw.yuv_to_rgb(yuv, 1920, 1080)
            row = court.register(rgb, previous)
            previous = np.array(row["homography"]) if row["homography"] is not None else None
            rows.append({"frame": start + i, **row})
            # Short shots can have coincident first/middle/last frames; write all roles.
            for index, role, prefix in requests:
                if i != index:
                    continue
                image = rgb.copy()
                if role == "middle" and row["homography"] is not None:
                    for segment in court.SEGMENTS:
                        pts = np.rint(court.project(segment, np.array(row["homography"]))).astype(np.int32)
                        cv2.polylines(image, [pts], False, (0, 255, 60), 2, cv2.LINE_AA)
                name = f"images/{prefix}-{role}.jpg"
                if not cv2.imwrite(str(publish / name), image[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88]):
                    raise RuntimeError("JPEG write failed")
                images[role] = name
        if proc.stdout.read(1):
            raise RuntimeError("extra decoded frames")
        stderr = proc.communicate(timeout=30)[1]
        if proc.returncode:
            raise RuntimeError(stderr.decode())
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=30)
    shot = {"id": ident, "start": start, "end": end, "registration": rows,
            "images": [images[r] for r in ("first", "middle", "last")] if extra_review is None else [],
            "extra_images": images if extra_review is not None else {}, "parent_shot": image_prefix,
            "conversion_command": command,
            **shot_prediction(rows)}
    write_json(publish / "shots" / f"{ident}.json", shot)
    return shot


def selected_spans(spans: list[tuple[int, int]], part: int, parts: int) -> list[tuple[int, int, int]]:
    """Balance whole shots by frame count, preserving global shot IDs and boundaries."""
    if parts < 1 or part < -1 or part >= parts:
        raise ValueError("invalid shot partition")
    indexed = [(i, a, b) for i, (a, b) in enumerate(spans)]
    if part == -1:
        return indexed
    loads = [0] * parts
    groups: list[list[tuple[int, int, int]]] = [[] for _ in range(parts)]
    for row in sorted(indexed, key=lambda r: (-(r[2] - r[1]), r[0])):
        group = min(range(parts), key=lambda i: (loads[i], i))
        groups[group].append(row)
        loads[group] += row[2] - row[1]
    return sorted(groups[part])


def assemble(archives: list[Path], out: Path, source: Path) -> dict[str, Any]:
    """Assemble new review artifacts without altering any completed job evidence."""
    out.mkdir(parents=True, exist_ok=False)
    reports, all_shots, copied = [], {}, set()
    for path in archives:
        with tarfile.open(path) as archive:
            handle = archive.extractfile("publish/k8-report.json")
            if handle is None:
                raise ValueError(f"{path}: no report")
            report = json.load(handle)
            reports.append(report)
            for shot in report["shots"]:
                if shot["id"] in all_shots:
                    raise ValueError("duplicate shot across parts")
                all_shots[shot["id"]] = shot
                for image in shot["images"]:
                    # Copy only fixed JPEG basenames; never extract arbitrary archive paths.
                    if Path(image).parent != Path("images") or Path(image).suffix != ".jpg":
                        raise ValueError("unsafe review image name")
                    image_handle = archive.extractfile(f"publish/{image}")
                    if image_handle is None:
                        raise ValueError("missing review image")
                    (out / "images").mkdir(exist_ok=True)
                    if image not in copied:
                        (out / image).write_bytes(image_handle.read())
                        copied.add(image)
    first = reports[0]
    for report in reports:
        for key in ("source", "fps", "frames", "source_frames", "scene_threshold", "scene_scores", "decision", "all_spans"):
            if report[key] != first[key]:
                raise ValueError(f"incompatible reports: {key}")
    shots = sorted(all_shots.values(), key=lambda s: s["start"])
    if [(s["start"], s["end"]) for s in shots] != [tuple(p) for p in first["all_spans"]]:
        raise ValueError("parts do not cover every whole-match shot")
    if [r["frame"] for s in shots for r in s["registration"]] != list(range(first["source_frames"])):
        raise ValueError("parts do not cover every source frame exactly once")
    if file_sha256(source) != first["source"]["video_sha256"]:
        raise ValueError("review video identity changed")
    result = {k: v for k, v in first.items() if k not in ("shots", "review_id", "part", "parts", "runtime_seconds")}
    result.update(shots=shots, archives=[{"path": str(p), "sha256": file_sha256(p)} for p in archives])
    result["part_runs"] = [{"archive": str(path), "part": report.get("part"),
                            "runtime_seconds": report.get("runtime_seconds"), "native_tools": report.get("native_tools")}
                           for path, report in zip(archives, reports)]
    result["review_id"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()[:16]
    (out / "source.mp4").symlink_to(source)
    write_json(out / "k8-report.json", result)
    review_page(result, out / "review.html")
    write_json(out / "READY.json", {"review_id": result["review_id"], "shots": len(shots), "frames": first["frames"],
                                  "status": "awaiting_user_marks", "archives": result["archives"]})
    return {"review": str(out / "review.html"), "review_id": result["review_id"], "shots": len(shots), "frames": first["frames"]}


def run_job(args: argparse.Namespace) -> int:
    import cv2
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from experiments.jobs.monitor import publish_progress

    cv2.setNumThreads(1)
    began = time.monotonic()
    native: dict[str, Any] = {}
    for name in ("ffmpeg", "ffprobe"):
        path = shutil.which(args.ffmpeg if name == "ffmpeg" else name)
        if path is None:
            raise FileNotFoundError(name)
        native[name] = {"path": path, "version": subprocess.check_output([path, "-version"], text=True).splitlines()[:3]}
    args.ffmpeg = native["ffmpeg"]["path"]
    stage, scratch = Path(os.environ["PS_STAGE_DIR"]), Path(os.environ["PS_SCRATCH_DIR"])
    publish = scratch / "publish"
    (publish / "images").mkdir(parents=True)
    (publish / "shots").mkdir()
    identity = json.loads(args.inputs.read_text())
    if file_sha256(args.video) != identity["video_sha256"]:
        raise ValueError("source identity changed")
    probe_command = [native["ffprobe"]["path"], "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                     "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_packets,start_time,codec_name,pix_fmt,color_space,color_range", "-of", "json", str(args.video)]
    meta = json.loads(subprocess.check_output(probe_command))["streams"][0]
    from fractions import Fraction
    fps = float(Fraction(meta["r_frame_rate"]))
    if meta["r_frame_rate"] != meta["avg_frame_rate"]:
        raise ValueError("requires CFR video")
    if [meta["width"], meta["height"]] != [1920, 1080]:
        raise ValueError("K8 whole match must be 1080p")
    total = int(meta["nb_read_packets"])
    frames = min(total, args.frames) if args.frames else total
    scenes, scene_command = cuts(args.video, fps, frames, args.ffmpeg, scratch)
    boundaries = [r["frame"] for r in scenes if r["score"] >= args.threshold and 0 < r["frame"] < frames]
    spans = shot_spans(boundaries, frames)
    selected = selected_spans(spans, args.part, args.parts)
    rows: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_shot, args.video, fps, a, b, f"shot-{i:04d}", publish, args.ffmpeg): i
                   for i, a, b in selected}
        for future in as_completed(futures):
            rows[futures[future]] = future.result()
            publish_progress(os.environ["PS_STAGE"], len(rows))
    report = {"schema": 1, "size": [1920, 1080], "fps": fps, "frames": frames, "source_frames": total, "source": identity,
              "scene_threshold": args.threshold, "scene_scores": scenes, "scene_command": scene_command,
              "decision": DECISION, "shots": [rows[i] for i in sorted(rows)], "part": args.part, "parts": args.parts,
              "all_spans": spans,
              "source_start_time": float(meta.get("start_time", 0)), "stream_metadata": meta, "native_tools": native,
              "runtime_seconds": time.monotonic() - began, "device": "cpu", "opencv": cv2.__version__,
              "numpy": np.__version__, "geometry": "ITF regulation; net excluded", "probe_command": probe_command}
    report["review_id"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()[:16]
    write_json(publish / "k8-report.json", report)
    review_page(report, publish / "review.html")
    write_json(stage / "k8.json", {k: v for k, v in report.items() if k != "shots"})
    print(json.dumps({"shots": len(rows), "court_predictions": sum(s["predicted"] == "court" for s in rows.values()),
                      "frames": frames, "seconds": report["runtime_seconds"]}), flush=True)
    return 0


def validate_job() -> int:
    from experiments.tennis import court
    stage, scratch = Path(os.environ["PS_STAGE_DIR"]), Path(os.environ["PS_SCRATCH_DIR"])
    report_path = scratch / "publish" / "k8-report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        image_names = {str(p.relative_to(scratch / "publish")) for p in (scratch / "publish" / "images").glob("*.jpg")}
    else:
        with tarfile.open(stage / "published.tar") as archive:
            member = archive.extractfile("publish/k8-report.json")
            if member is None:
                raise ValueError("published report missing")
            report = json.load(member)
            image_names = {m.name.removeprefix("publish/") for m in archive.getmembers() if m.isfile() and m.size > 0}
    shots = report["shots"]
    rows = [r for s in shots for r in s["registration"]]
    checks = {"contiguous_frame_coverage": [r["frame"] for r in rows] == [f for _, a, b in selected_spans([tuple(p) for p in report["all_spans"]], report["part"], report["parts"]) for f in range(a, b)],
              "shot_boundaries_partition": [(s["start"], s["end"]) for s in shots] == [(a, b) for _, a, b in selected_spans(
                  [tuple(p) for p in report["all_spans"]], report["part"], report["parts"])],
              "predictions_obey_80_percent": all(s["predicted"] == shot_prediction(s["registration"])["predicted"] for s in shots),
              "every_review_image": all(p in image_names for s in shots for p in s["images"]),
              "some_real_registered_frames": any(r["registered"] for r in rows),
              "failed_frames_retained": all(not r["registered"] or (r["homography"] is not None and r["chamfer_px"] <= 2) for r in rows),
              "finite_nonsingular_homographies": all(r["homography"] is None or (np.isfinite(r["homography"]).all()
                   and abs(np.linalg.det(r["homography"])) > 1e-10) for r in rows),
              "cpu_only": not os.environ.get("CUDA_VISIBLE_DEVICES"),
              "regulation_nine_lines": court.SEGMENTS.shape == (9, 2, 2),
              "native_1080p": report["size"] == [1920, 1080]}
    result = {"passed": all(checks.values()), "checks": checks}
    write_json(Path(os.environ["PS_VALIDATION_PATH"]), result)
    write_json(stage / "k8-validation.json", result)
    print(json.dumps(result), flush=True)
    return 0 if result["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    scoring = sub.add_parser("score")
    scoring.add_argument("--report", type=Path, required=True)
    scoring.add_argument("--marks", type=Path, required=True)
    scoring.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run")
    run.add_argument("--video", type=Path, required=True)
    run.add_argument("--inputs", type=Path, required=True)
    run.add_argument("--frames", type=int, default=0)
    run.add_argument("--threshold", type=float, default=0.25)
    run.add_argument("--part", type=int, default=-1)
    run.add_argument("--parts", type=int, default=1)
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--ffmpeg", default="ffmpeg")
    sampling = sub.add_parser("sample")
    sampling.add_argument("--report", type=Path, required=True)
    sampling.add_argument("--out", type=Path, required=True)
    sampling.add_argument("--expected", type=float, default=SAMPLE_STAGES[0])
    sub.add_parser("validate")
    assembly = sub.add_parser("assemble")
    assembly.add_argument("--archive", type=Path, action="append", required=True)
    assembly.add_argument("--out", type=Path, required=True)
    assembly.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "assemble":
        print(json.dumps(assemble(args.archive, args.out, args.source)))
        return 0
    if args.command == "sample":
        print(json.dumps(write_sample(args.report, args.out, args.expected)))
        return 0
    if args.command == "run":
        return run_job(args)
    if args.command == "validate":
        return validate_job()
    write_json(args.out, score(json.loads(args.report.read_text()), json.loads(args.marks.read_text())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
