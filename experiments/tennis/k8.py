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
import subprocess
import tarfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from experiments.visor.b1 import file_sha256, write_json

DECISION = {"shot_registered_share": 0.80, "precision": 0.95, "duration_recall": 0.90,
            "median_px": 2.0, "p95_px": 6.0}
CLASSES = ("court", "close_up", "crowd", "replay", "graphics", "other", "mixed")


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


def score(report: dict[str, Any], marks: dict[str, Any]) -> dict[str, Any]:
    """Mixed shots are wrong for both: a predicted court is FP and their duration is FN.

    Registration includes ALL true-court frames, with missing fits represented by
    infinity, never silently omitted. Quantiles use nearest rank to retain failures.
    """
    if marks.get("review_id") != report["review_id"]:
        raise ValueError("marks belong to another review")
    labels = marks["marks"]
    shots = report["shots"]
    if set(labels) != {s["id"] for s in shots} or any(v not in CLASSES for v in labels.values()):
        raise ValueError("mark every shot exactly once with an allowed class")
    tp = fp = 0
    duration_tp = duration_true = 0.0
    errors: list[float] = []
    misses: Counter[str] = Counter()
    confusion: Counter[str] = Counter()
    for shot in shots:
        truth, predicted = labels[shot["id"]], shot["predicted"] == "court"
        duration = (shot["end"] - shot["start"]) / report["fps"]
        confusion[f"{truth}/{shot['predicted']}"] += 1
        if predicted:
            tp += truth == "court"
            fp += truth != "court"
        if truth in ("court", "mixed"):
            duration_true += duration
            if truth == "court" and predicted:
                duration_tp += duration
        if (predicted and truth != "court") or (not predicted and truth in ("court", "mixed")):
            misses[truth] += 1
        if truth == "court":
            errors.extend(float(r["chamfer_px"]) if r["chamfer_px"] is not None else float("inf")
                          for r in shot["registration"])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = duration_tp / duration_true if duration_true else 0.0
    median, p95 = (np.quantile(errors, [0.5, 0.95], method="inverted_cdf").tolist()
                   if errors else [float("inf"), float("inf")])
    conditions = {"precision": precision >= DECISION["precision"],
                  "duration_recall": recall >= DECISION["duration_recall"],
                  "registration": median <= DECISION["median_px"] and p95 <= DECISION["p95_px"]}
    return {"decision": DECISION, "adopted": all(conditions.values()), "conditions": conditions,
            "precision": precision, "duration_recall": recall,
            "registration_median_px": median if np.isfinite(median) else None,
            "registration_p95_px": p95 if np.isfinite(p95) else None,
            "missing_registration_is_infinite": True, "court_frames": len(errors),
            "misses_by_class": dict(misses), "confusion": dict(confusion),
            "marks_sha256": hashlib.sha256(json.dumps(marks, sort_keys=True).encode()).hexdigest()}


def review_page(report: dict[str, Any], out: Path) -> None:
    """Self-contained page apart from relative JPEGs; labels start blank, never predictions."""
    cards = []
    for s in report["shots"]:
        images = "".join(f'<figure><img loading="lazy" src="{html.escape(p)}"><figcaption>{label}</figcaption></figure>'
                         for p, label in zip(s["images"], ("First", "Middle + projected court", "Last")))
        options = '<option value="">Choose true class…</option>' + ''.join(
            f'<option value="{c}">{c}</option>' for c in CLASSES)
        cards.append(f'<article id="{s["id"]}"><h2>{s["id"]} · {s["start"]/report["fps"]:.2f}–'
                     f'{s["end"]/report["fps"]:.2f} s</h2><p>Prediction: <b>{s["predicted"]}</b> · '
                     f'{s["registered_share"]:.1%} registered</p><div class="strip">{images}</div>'
                     f'<label>True class <select data-id="{s["id"]}">{options}</select></label>'
                     '<details><summary>Inspect the shot in the source video</summary>'
                     f'<video controls preload="none" src="source.mp4#t={s["start"]/report["fps"]:.3f},'
                     f'{s["end"]/report["fps"]:.3f}"></video></details></article>')
    template = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>K8 court-shot review</title>
<style>body{font:16px system-ui;margin:24px;background:#eee;color:#18212b}header{position:sticky;top:0;background:#fff;padding:12px;z-index:2}article{background:white;margin:20px 0;padding:16px;border-radius:8px}.strip{display:flex;gap:10px}figure{margin:0;width:33.3%}img{width:100%}video{width:70%}select,button{font:inherit;padding:8px}figcaption{font-size:13px}</style>
<header><b>K8 · every broadcast shot</b> <span id="count"></span> <button id="save">Download marks JSON</button>
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
    out.write_text(template.replace('THRESHOLD', str(report['scene_threshold'])).replace('CARDS', '\n'.join(cards))
                   .replace('REVIEWID', json.dumps(report['review_id'])).replace('!CLASSES.includes', f'!{json.dumps(CLASSES)}.includes'))


def cuts(video: Path, fps: float, limit: int, ffmpeg: str, scratch: Path) -> tuple[list[dict[str, Any]], list[str]]:
    log = scratch / "scene-metadata.txt"
    command = [ffmpeg, "-v", "error", "-nostdin", "-threads", "2", "-t", str(limit / fps), "-i", str(video),
               "-vf", f"setpts=PTS-STARTPTS,select='gte(scene,0.10)',metadata=print:file={log}", "-an", "-f", "null", "-"]
    subprocess.run(command, check=True, capture_output=True, timeout=1800)
    return parse_scenes(log.read_text(), fps), command


def process_shot(video: Path, fps: float, start: int, end: int, ident: str, publish: Path, ffmpeg: str) -> dict[str, Any]:
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
            for index, role in ((0, "first"), ((end - start) // 2, "middle"), (end - start - 1, "last")):
                if i != index:
                    continue
                image = rgb.copy()
                if role == "middle" and row["homography"] is not None:
                    for segment in court.SEGMENTS:
                        pts = np.rint(court.project(segment, np.array(row["homography"]))).astype(np.int32)
                        cv2.polylines(image, [pts], False, (0, 255, 60), 2, cv2.LINE_AA)
                name = f"images/{ident}-{role}.jpg"
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
            "images": [images[r] for r in ("first", "middle", "last")], "conversion_command": command,
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
    result = {k: v for k, v in first.items() if k not in ("shots", "review_id", "part", "parts")}
    result.update(shots=shots, archives=[{"path": str(p), "sha256": file_sha256(p)} for p in archives])
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
    stage, scratch = Path(os.environ["PS_STAGE_DIR"]), Path(os.environ["PS_SCRATCH_DIR"])
    publish = scratch / "publish"
    (publish / "images").mkdir(parents=True)
    (publish / "shots").mkdir()
    identity = json.loads(args.inputs.read_text())
    if file_sha256(args.video) != identity["video_sha256"]:
        raise ValueError("source identity changed")
    probe_command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                     "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_packets,start_time", "-of", "json", str(args.video)]
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
              "source_start_time": float(meta.get("start_time", 0)),
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
    sub.add_parser("validate")
    assembly = sub.add_parser("assemble")
    assembly.add_argument("--archive", type=Path, action="append", required=True)
    assembly.add_argument("--out", type=Path, required=True)
    assembly.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "assemble":
        print(json.dumps(assemble(args.archive, args.out, args.source)))
        return 0
    if args.command == "run":
        return run_job(args)
    if args.command == "validate":
        return validate_job()
    write_json(args.out, score(json.loads(args.report.read_text()), json.loads(args.marks.read_text())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
