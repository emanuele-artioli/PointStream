"""K8b court shots on held-out broadcast clips: shade-robust registration, census review, fixed rule.

Rule: docs/experiments.md, "2026-10-20 — K8b". Each clip is decoded from its own start (frames
selected with `trim`, never seeked), so frame n is the clip's n-th decoded frame, the same index
the scene scan uses (setpts=PTS-STARTPTS). Long clips are split into chunks for parallel work;
each chunk warm-starts from its own first frame. Every finished chunk and clip is checkpointed.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import shutil
import subprocess
import tarfile
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import k8
from experiments.visor.b1 import file_sha256, write_json

DECISION = {"shot_fit_share": 0.80, "fit_px": 6.0, "precision": 0.95, "duration_recall": 0.90,
            "median_px": 2.0, "p95_px": 6.0, "scene_threshold": 0.25, "min_state_seconds": 1.0}
# Clips seen while correcting the method (the pilot smoke's clip): registered for K4b, but
# neither reviewed nor scored (rule amendment of 2026-10-21).
DEV_CLIPS = ("djokovic_zverev/000",)
CHUNK = 200
SIZE = (1920, 1080)


def split_by_state(states: list[bool], min_run: int) -> list[tuple[int, int]]:
    """Half-open spans of persistent fit state: runs shorter than `min_run` frames are absorbed.

    The shortest short run merges into its longer neighbour until every run is long enough
    (or one run is left), then spans break where the state changes.
    """
    if not states:
        return []
    runs: list[list[Any]] = []
    for i, state in enumerate(states):
        if runs and runs[-1][0] == state:
            runs[-1][2] = i + 1
        else:
            runs.append([state, i, i + 1])
    while len(runs) > 1:
        short = [i for i, r in enumerate(runs) if r[2] - r[1] < min_run]
        if not short:
            break
        i = min(short, key=lambda j: (runs[j][2] - runs[j][1], j))
        neighbours = [j for j in (i - 1, i + 1) if 0 <= j < len(runs)]
        j = max(neighbours, key=lambda n: (runs[n][2] - runs[n][1], -n))
        lo, hi = min(i, j), max(i, j)
        merged = [runs[j][0], runs[lo][1], runs[hi][2]]
        runs[lo:hi + 1] = [merged]
        # Adjacent runs of equal state after a merge become one.
        k = 0
        while k < len(runs) - 1:
            if runs[k][0] == runs[k + 1][0]:
                runs[k:k + 2] = [[runs[k][0], runs[k][1], runs[k + 1][2]]]
            else:
                k += 1
    return [(r[1], r[2]) for r in runs]


def clip_shots(rows: list[dict[str, Any]], cuts: list[int], fps: float) -> list[tuple[int, int]]:
    """Scene cuts first, then fit-state splits of at least DECISION['min_state_seconds']."""
    spans = k8.shot_spans(cuts, len(rows)) if cuts else [(0, len(rows))]
    min_run = max(1, round(DECISION["min_state_seconds"] * fps))
    out: list[tuple[int, int]] = []
    for a, b in spans:
        out.extend((a + x, a + y) for x, y in split_by_state([bool(r["fits"]) for r in rows[a:b]], min_run))
    return out


def prediction(rows: list[dict[str, Any]]) -> dict[str, Any]:
    share = sum(bool(r["fits"]) for r in rows) / len(rows)
    return {"predicted": "court" if share >= DECISION["shot_fit_share"] else "non_court", "fit_share": share}


def probe(video: Path, ffprobe: str) -> dict[str, Any]:
    meta = json.loads(subprocess.check_output(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
         "stream=width,height,r_frame_rate,avg_frame_rate,nb_read_packets,start_time,codec_name,pix_fmt,color_space,color_range",
         "-of", "json", str(video)]))["streams"][0]
    if meta["r_frame_rate"] != meta["avg_frame_rate"]:
        raise ValueError(f"{video}: requires CFR video")
    return {**meta, "fps": float(Fraction(meta["r_frame_rate"])), "frames": int(meta["nb_read_packets"])}


def decode_command(ffmpeg: str, video: Path, start: int, end: int) -> list[str]:
    """Frames [start, end) of the clip as 1080p limited-range YUV420 (K4's conversion), by index."""
    scale = f"scale={SIZE[0]}:{SIZE[1]}:flags=area:out_range=tv,format=yuv420p"
    return [ffmpeg, "-v", "error", "-nostdin", "-threads", "1", "-i", str(video),
            "-vf", f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS,{scale}",
            "-frames:v", str(end - start), "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"]


def frames(command: list[str], count: int) -> Any:
    """Yield `count` RGB frames from a raw 1080p YUV420 pipe; fail on truncation or extras."""
    from experiments.tennis import windows as tw

    size = SIZE[0] * SIZE[1] * 3 // 2
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    try:
        for i in range(count):
            data = bytearray()
            while len(data) < size:
                block = proc.stdout.read(size - len(data))
                if not block:
                    raise RuntimeError(f"truncated at frame {i} of {count}")
                data.extend(block)
            yield tw.yuv_to_rgb(np.frombuffer(data, np.uint8).reshape(SIZE[1] * 3 // 2, SIZE[0]), *SIZE)
        if proc.stdout.read(1):
            raise RuntimeError("extra decoded frames")
        stderr = proc.communicate(timeout=60)[1]
        if proc.returncode:
            raise RuntimeError(stderr.decode())
    finally:
        if proc.poll() is None:
            proc.terminate()
            proc.wait(timeout=30)


def register_chunk(video: str, start: int, end: int, ffmpeg: str) -> list[dict[str, Any]]:
    import cv2

    from experiments.tennis import court

    cv2.setNumThreads(1)
    rows, previous = [], None
    for i, rgb in enumerate(frames(decode_command(ffmpeg, Path(video), start, end), end - start)):
        row = court.register_v2(rgb, previous)
        previous = np.array(row["homography"]) if row["homography"] is not None else None
        rows.append({"frame": start + i, **row})
    return rows


def review_images(video: Path, ffmpeg: str, shots: list[dict[str, Any]], rows: list[dict[str, Any]],
                  publish: Path, prefix: str) -> None:
    """First, middle (with the projected court when a fit exists) and last frame of each shot."""
    import cv2

    from experiments.tennis import court

    wanted: dict[int, list[tuple[dict[str, Any], str]]] = defaultdict(list)
    for shot in shots:
        a, b = shot["start"], shot["end"]
        for role, n in (("first", a), ("middle", a + (b - a) // 2), ("last", b - 1)):
            wanted[n].append((shot, role))
    order = sorted(wanted)
    select = "+".join(f"eq(n\\,{n})" for n in order)
    scale = f"scale={SIZE[0]}:{SIZE[1]}:flags=area:out_range=tv,format=yuv420p"
    command = [ffmpeg, "-v", "error", "-nostdin", "-threads", "2", "-i", str(video), "-vf", f"select='{select}',{scale}",
               "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"]
    (publish / "images").mkdir(parents=True, exist_ok=True)
    for n, rgb in zip(order, frames(command, len(order))):
        for shot, role in wanted[n]:
            image = rgb.copy()
            hom = rows[n]["homography"]
            if role == "middle" and hom is not None:
                for segment in court.SEGMENTS:
                    pts = np.rint(court.project(segment, np.array(hom))).astype(np.int32)
                    cv2.polylines(image, [pts], False, (0, 255, 60), 2, cv2.LINE_AA)
            name = f"images/{prefix}-{shot['index']:02d}-{role}.jpg"
            if not cv2.imwrite(str(publish / name), image[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85]):
                raise RuntimeError("JPEG write failed")
            shot.setdefault("images", {})[role] = name
    for shot in shots:
        shot["images"] = [shot["images"][r] for r in ("first", "middle", "last")]


def chunks(clips: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for clip in clips:
        for a in range(0, clip["frames"], CHUNK):
            out.append({"clip": clip["id"], "start": a, "end": min(a + CHUNK, clip["frames"])})
    return out


def parts(clips: list[dict[str, Any]], count: int) -> list[list[dict[str, Any]]]:
    """Whole clips balanced by frame count (4K frames weigh more for decoding)."""
    loads = [0.0] * count
    groups: list[list[dict[str, Any]]] = [[] for _ in range(count)]
    for clip in sorted(clips, key=lambda c: (-c["frames"] * c["weight"], c["id"])):
        g = min(range(count), key=lambda i: (loads[i], i))
        groups[g].append(clip)
        loads[g] += clip["frames"] * clip["weight"]
    return [sorted(g, key=lambda c: c["id"]) for g in groups]


def checkpoint(name: str, data: dict[str, Any]) -> None:
    from experiments.jobs.monitor import save_checkpoint

    save_checkpoint(name, json.dumps(data).encode())


def run(args: argparse.Namespace) -> int:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from experiments.jobs.monitor import publish_progress

    began = time.monotonic()
    manifest = json.loads(args.manifest.read_text())
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    checkpoints = Path(os.environ.get("PS_CHECKPOINT_DIR", scratch / "checkpoint"))
    checkpoints.mkdir(parents=True, exist_ok=True)
    os.environ["PS_CHECKPOINT_DIR"] = str(checkpoints)  # save_checkpoint's target, also outside the fleet
    publish = scratch / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    root = scratch / "clips"
    if not root.exists():
        with tarfile.open(args.archive) as archive:
            archive.extractall(root, filter="data")
    native: dict[str, dict[str, Any]] = {}
    for name in ("ffmpeg", "ffprobe"):
        path = shutil.which(name)
        if path is None:
            raise FileNotFoundError(name)
        native[name] = {"path": path, "version": subprocess.check_output([path, "-version"], text=True).splitlines()[:3]}
    selected = parts(manifest["clips"], args.parts)[args.part]
    if args.frames:
        selected = [max(selected, key=lambda c: (c["frames"], c["id"]))]  # smoke: the part's longest clip, truncated
    clips = {}
    for clip in selected:
        video = root / clip["path"]
        if file_sha256(video) != clip["sha256"]:
            raise ValueError(f"{clip['id']}: clip identity changed")
        meta = probe(video, native["ffprobe"]["path"])
        count = min(meta["frames"], args.frames) if args.frames else meta["frames"]
        clips[clip["id"]] = {**clip, "video": str(video), "meta": meta, "frames": count}
    work = chunks(list(clips.values()))
    done: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for saved_path in checkpoints.glob("chunk-*.json"):
        saved = json.loads(saved_path.read_text())
        done[(saved["clip"], saved["start"])] = saved["rows"]
    results: dict[str, dict[str, Any]] = {}

    def finalize(clip_id: str) -> None:
        """Cuts, shots and review images for a clip whose chunks are all done; published at once,
        so a stage stopped later still keeps it in partial.tar."""
        clip = clips[clip_id]
        rows = [r for a in range(0, clip["frames"], CHUNK) for r in done[(clip_id, a)]]
        if [r["frame"] for r in rows] != list(range(clip["frames"])):
            raise ValueError(f"{clip_id}: frames not covered exactly once")
        fps = clip["meta"]["fps"]
        scenes, scene_command = k8.cuts(Path(clip["video"]), fps, clip["frames"], native["ffmpeg"]["path"], scratch)
        cuts = sorted({s["frame"] for s in scenes if s["score"] >= DECISION["scene_threshold"] and 0 < s["frame"] < clip["frames"]})
        shots = []
        for index, (a, b) in enumerate(clip_shots(rows, cuts, fps)):
            shots.append({"id": f"{clip_id}/shot-{index:02d}", "index": index, "clip": clip_id, "start": a, "end": b,
                          "seconds": (b - a) / fps, **prediction(rows[a:b])})
        prefix = clip_id.replace("/", "-")
        review_images(Path(clip["video"]), native["ffmpeg"]["path"], shots, rows, publish, prefix)
        result = {"id": clip_id, "path": clip["path"], "sha256": clip["sha256"], "fps": fps, "frames": clip["frames"],
                  "source_frames": clip["meta"]["frames"], "stream": clip["meta"], "scene_scores": scenes,
                  "scene_command": scene_command, "cuts": cuts, "shots": shots, "registration": rows}
        write_json(publish / "clips" / f"{prefix}.json", result)
        results[clip_id] = {k: result[k] for k in ("id", "frames", "source_frames", "fps", "cuts")} | {"shots": len(shots)}

    def complete(clip_id: str) -> bool:
        return all((clip_id, a) in done for a in range(0, clips[clip_id]["frames"], CHUNK))

    for clip_id in sorted(clips):
        if complete(clip_id):
            finalize(clip_id)
    # Longest clips first, chunk by chunk, so clips finish progressively and the tail is short.
    order = sorted(clips, key=lambda k: (-clips[k]["frames"], k))
    todo = [c for k in order for c in work if c["clip"] == k and (c["clip"], c["start"]) not in done]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(register_chunk, clips[c["clip"]]["video"], c["start"], c["end"], native["ffmpeg"]["path"]): c
                   for c in todo}
        for future in as_completed(futures):
            c = futures[future]
            done[(c["clip"], c["start"])] = future.result()
            key = hashlib.sha256(f"{c['clip']}:{c['start']}".encode()).hexdigest()[:16]
            checkpoint(f"chunk-{key}.json", {"clip": c["clip"], "start": c["start"], "rows": done[(c["clip"], c["start"])]})
            publish_progress(os.environ.get("PS_STAGE", "local"), len(done))
            if complete(c["clip"]) and c["clip"] not in results:
                finalize(c["clip"])
    report = {"schema": 1, "decision": DECISION, "part": args.part, "parts": args.parts, "chunk": CHUNK,
              "manifest_sha256": file_sha256(args.manifest), "clips": [results[k] for k in sorted(results)], "native_tools": native,
              "frames_limit": args.frames, "runtime_seconds": time.monotonic() - began, "device": "cpu",
              "court_code_sha256": file_sha256(Path(__file__).with_name("court.py")),
              "worker_code_sha256": file_sha256(Path(__file__))}
    write_json(publish / "k8b-report.json", report)
    stage = os.environ.get("PS_STAGE_DIR")
    if stage:
        write_json(Path(stage) / "k8b.json", report)
    print(json.dumps({"clips": len(results), "frames": sum(r["frames"] for r in results.values()),
                      "seconds": report["runtime_seconds"]}), flush=True)
    return 0


def load_published(stage: Path, scratch: Path) -> tuple[dict[str, Any], list[dict[str, Any]], set[str]]:
    publish = scratch / "publish"
    if (publish / "k8b-report.json").exists():
        report = json.loads((publish / "k8b-report.json").read_text())
        clips = [json.loads(p.read_text()) for p in sorted((publish / "clips").glob("*.json"))]
        images = {str(p.relative_to(publish)) for p in (publish / "images").glob("*.jpg") if p.stat().st_size > 0}
        return report, clips, images
    with tarfile.open(stage / "published.tar") as archive:
        def read(name: str) -> Any:
            handle = archive.extractfile(name)
            if handle is None:
                raise ValueError(f"missing {name}")
            return json.load(handle)
        members = archive.getmembers()
        report = read("publish/k8b-report.json")
        clips = [read(m.name) for m in sorted(members, key=lambda m: m.name)
                 if m.name.startswith("publish/clips/") and m.name.endswith(".json")]
        images = {m.name.removeprefix("publish/") for m in members if m.isfile() and m.size > 0 and m.name.endswith(".jpg")}
        return report, clips, images


def validate() -> int:
    stage, scratch = Path(os.environ["PS_STAGE_DIR"]), Path(os.environ["PS_SCRATCH_DIR"])
    report, clips, images = load_published(stage, scratch)
    rows = [r for c in clips for r in c["registration"]]
    checks = {
        "every_selected_clip_published": sorted(c["id"] for c in clips) == sorted(r["id"] for r in report["clips"]),
        "frames_once_in_order": all([r["frame"] for r in c["registration"]] == list(range(c["frames"])) for c in clips),
        "shots_partition_clips": all([(s["start"], s["end"]) for s in c["shots"]] == list(zip(
            [0, *[s["end"] for s in c["shots"][:-1]]], [s["end"] for s in c["shots"]])) and c["shots"][-1]["end"] == c["frames"]
            for c in clips),
        "cuts_are_shot_boundaries": all(set(c["cuts"]) <= {s["start"] for s in c["shots"]} for c in clips),
        "predictions_obey_rule": all(s["predicted"] == prediction(c["registration"][s["start"]:s["end"]])["predicted"]
                                     for c in clips for s in c["shots"]),
        "three_review_images_per_shot": all(len(s["images"]) == 3 and set(s["images"]) <= images for c in clips for s in c["shots"]),
        "some_frames_fit": any(r["fits"] for r in rows),
        "fits_have_finite_homographies": all(not r["fits"] or (r["homography"] is not None and np.isfinite(r["homography"]).all()
                                             and abs(np.linalg.det(r["homography"])) > 1e-10) for r in rows),
        "fit_flag_matches_error": all(r["fits"] == (r["error_px"] is not None and r["error_px"] <= DECISION["fit_px"]) for r in rows),
        "clutter_guard_held": all(r["homography"] is None or r["clutter"] <= 0.25 for r in rows),
        "cpu_only": not os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    result = {"passed": all(checks.values()), "checks": checks}
    if os.environ.get("PS_VALIDATION_PATH"):
        write_json(Path(os.environ["PS_VALIDATION_PATH"]), result)
    write_json(stage / "k8b-validation.json", result)
    print(json.dumps(result), flush=True)
    return 0 if result["passed"] else 1


def review_page(report: dict[str, Any], out: Path) -> None:
    """K8's review layout over held-out clips; labels start blank and download with the review ID."""
    cards = []
    for clip in report["clips"]:
        if clip["id"] in DEV_CLIPS:
            continue
        for s in clip["shots"]:
            images = "".join(f'<figure><img loading="lazy" src="{html.escape(p)}"><figcaption>{label}</figcaption></figure>'
                             for p, label in zip(s["images"], ("First", "Middle + projected court", "Last")))
            options = '<option value="">Choose true class…</option>' + "".join(
                f'<option value="{c}">{c}</option>' for c in k8.CLASSES)
            t0, t1 = s["start"] / clip["fps"], s["end"] / clip["fps"]
            cards.append(f'<article id="{html.escape(s["id"])}"><h2>{html.escape(s["id"])} · {t0:.2f}–{t1:.2f} s</h2>'
                         f'<p>Prediction: <b>{s["predicted"]}</b> · {s["fit_share"]:.1%} of frames fit</p>'
                         f'<div class="strip">{images}</div><label>True class <select data-id="{html.escape(s["id"])}">'
                         f'{options}</select></label><details><summary>Play the shot</summary>'
                         f'<video controls muted preload="none" src="clips/{html.escape(clip["path"])}#t={t0:.3f},{t1:.3f}">'
                         '</video></details></article>')
    shots = sum(len(c["shots"]) for c in report["clips"] if c["id"] not in DEV_CLIPS)
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>K8b court-shot review</title>
<style>body{{font:16px system-ui;margin:24px;background:#eee;color:#18212b}}header{{position:sticky;top:0;background:#fff;padding:12px;z-index:2}}article{{background:white;margin:20px 0;padding:16px;border-radius:8px}}.strip{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}}figure{{margin:0;min-width:0}}img{{display:block;width:100%}}video{{width:70%;max-width:100%}}select,button{{font:inherit;padding:8px}}figcaption{{font-size:13px}}</style>
<header><b>K8b · {shots} shots of three held-out broadcasts</b> <span id="count"></span> <button id="save">Download marks JSON</button>
<label>Restore marks <input id="load" type="file" accept="application/json"></label></header>
<p>Mark the true class of every shot, as in K8. Court means an ordinary court view intended for PointStream; mark replays as replay even when the court is visible. Use mixed for a shot containing both court and non-court content (bad cut); it counts as wrong for both. Green is the projected model; no overlay means no fit. Predictions are not labels. Labels are saved locally in this browser; also download the JSON.</p>
{"".join(cards)}
<script>const reviewId={json.dumps(report["review_id"])}, key='pointstream-k8b-'+reviewId, classes={json.dumps(k8.CLASSES)};
let marks={{}};try{{marks=JSON.parse(localStorage.getItem(key)||'{{}}')}}catch(e){{}}
const selects=[...document.querySelectorAll('select')];
function update(){{for(const s of selects)s.value=marks[s.dataset.id]||'';document.querySelector('#count').textContent=Object.keys(marks).length+' / '+selects.length+' marked';try{{localStorage.setItem(key,JSON.stringify(marks))}}catch(e){{}}}}
for(const s of selects)s.onchange=()=>{{if(s.value)marks[s.dataset.id]=s.value;else delete marks[s.dataset.id];update()}};update();
document.querySelector('#save').onclick=()=>{{const blob=new Blob([JSON.stringify({{schema:1,review_id:reviewId,marks}},null,2)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='k8b-marks-'+reviewId+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}};
document.querySelector('#load').onchange=async e=>{{try{{const r=JSON.parse(await e.target.files[0].text());if(r.review_id!==reviewId)throw Error('Different review');if(Object.entries(r.marks).some(([k,v])=>!selects.some(s=>s.dataset.id===k)||!classes.includes(v)))throw Error('Invalid labels');marks=r.marks;update()}}catch(err){{alert(err.message)}}}};
</script></html>'''
    out.write_text(page)


def assemble(args: argparse.Namespace) -> int:
    """Review directory from the parts' published archives; fails on missing or duplicate clips."""
    manifest = json.loads(args.manifest.read_text())
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    clips: dict[str, dict[str, Any]] = {}
    evidence = []
    for path in args.archive:
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                name = member.name.removeprefix("publish/")
                if not member.isfile():
                    continue
                if name.startswith("clips/") and name.endswith(".json"):
                    handle = archive.extractfile(member)
                    assert handle is not None
                    clip = json.load(handle)
                    if clip["id"] in clips:
                        raise ValueError(f"duplicate clip {clip['id']}")
                    clips[clip["id"]] = clip
                elif name.startswith("images/") and name.endswith(".jpg") and "/" not in name.removeprefix("images/"):
                    handle = archive.extractfile(member)
                    assert handle is not None
                    target = out / name
                    if target.exists():
                        raise ValueError(f"duplicate image {name}")
                    target.write_bytes(handle.read())
            handle = archive.extractfile("publish/k8b-report.json")
            assert handle is not None
            part = json.load(handle)
            if part["manifest_sha256"] != file_sha256(args.manifest) or part["frames_limit"]:
                raise ValueError(f"{path}: not a full stage of this manifest")
            evidence.append({"path": str(path), "sha256": file_sha256(path), "part": part["part"],
                             "runtime_seconds": part["runtime_seconds"], "native_tools": part["native_tools"],
                             "court_code_sha256": part["court_code_sha256"], "worker_code_sha256": part["worker_code_sha256"]})
    if sorted(clips) != sorted(c["id"] for c in manifest["clips"]):
        raise ValueError("parts do not cover every manifest clip exactly once")
    for clip in clips.values():
        if clip["frames"] != clip["source_frames"]:
            raise ValueError(f"{clip['id']}: incomplete clip")
        for shot in clip["shots"]:
            if not all((out / p).is_file() for p in shot["images"]):
                raise ValueError(f"{shot['id']}: missing review image")
    report = {"schema": 1, "decision": DECISION, "manifest_sha256": file_sha256(args.manifest), "evidence": evidence,
              "clips": [clips[k] for k in sorted(clips)]}
    report["review_id"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()[:16]
    (out / "clips").symlink_to(args.clips_root)
    write_json(out / "k8b-report.json", report)
    review_page(report, out / "review.html")
    ready = {"review_id": report["review_id"], "clips": len(clips), "shots": sum(len(c["shots"]) for c in clips.values()),
             "frames": sum(c["frames"] for c in clips.values()), "status": "awaiting_user_marks"}
    write_json(out / "READY.json", ready)
    print(json.dumps(ready))
    return 0


def score(report: dict[str, Any], marks: dict[str, Any]) -> dict[str, Any]:
    """Census scoring by K8's rule: exact precision and duration recall; registration percentiles over
    every true-court frame with missing fits infinite; mixed shots wrong for both."""
    if marks.get("review_id") != report["review_id"]:
        raise ValueError("marks belong to another review")
    labels = marks["marks"]
    shots = [(c, s) for c in report["clips"] if c["id"] not in DEV_CLIPS for s in c["shots"]]
    if set(labels) != {s["id"] for _, s in shots} or any(v not in k8.CLASSES for v in labels.values()):
        raise ValueError("mark every shot exactly once with an allowed class")
    tp = fp = 0
    seconds_tp = seconds_true = 0.0
    errors: list[float] = []
    misses: Counter[str] = Counter()
    missed: dict[str, list[str]] = defaultdict(list)
    confusion: Counter[str] = Counter()
    by_match: dict[str, Counter[str]] = defaultdict(Counter)
    for clip, shot in shots:
        truth, predicted = labels[shot["id"]], shot["predicted"] == "court"
        confusion[f"{truth}/{shot['predicted']}"] += 1
        match = clip["id"].split("/")[0]
        by_match[match][f"{truth}/{shot['predicted']}"] += 1
        if predicted:
            tp += truth == "court"
            fp += truth != "court"
        if truth in ("court", "mixed"):
            seconds_true += shot["seconds"]
            seconds_tp += shot["seconds"] * (truth == "court" and predicted)
        if (predicted and truth != "court") or (not predicted and truth in ("court", "mixed")):
            misses[truth] += 1
            missed[truth].append(shot["id"])
        if truth == "court":
            errors.extend(float(r["error_px"]) if r["error_px"] is not None else float("inf")
                          for r in clip["registration"][shot["start"]:shot["end"]])
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = seconds_tp / seconds_true if seconds_true else 0.0
    median = k8.weighted_percentile(errors, [1.0] * len(errors), 0.5)
    p95 = k8.weighted_percentile(errors, [1.0] * len(errors), 0.95)
    conditions = {"precision": precision >= DECISION["precision"], "duration_recall": recall >= DECISION["duration_recall"],
                  "registration": median <= DECISION["median_px"] and p95 <= DECISION["p95_px"]}
    return {"decision": DECISION, "adopted": all(conditions.values()), "conditions": conditions,
            "excluded_dev_clips": list(DEV_CLIPS),
            "precision": precision, "predicted_court_shots": tp + fp, "duration_recall": recall,
            "true_court_seconds": seconds_true, "registration_median_px": median if np.isfinite(median) else None,
            "registration_p95_px": p95 if np.isfinite(p95) else None, "court_frames": len(errors),
            "missing_registration_is_infinite": True, "misses_by_class": dict(misses), "missed_shots_by_class": dict(missed),
            "confusion": dict(confusion), "confusion_by_match": {k: dict(v) for k, v in by_match.items()},
            "marks_sha256": hashlib.sha256(json.dumps(marks, sort_keys=True).encode()).hexdigest()}


def make_manifest(root: Path, matches: list[str], out: Path, pattern: str = "*.mp4") -> dict[str, Any]:
    """Clip identities for the held-out set; 2160p clips weigh 1.2 when balancing parts (registration runs at 1080p; only decoding costs more)."""
    clips = []
    for match in matches:
        for video in sorted((root / match).glob(pattern)):
            meta = json.loads(subprocess.check_output(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=height,nb_frames", "-of", "json",
                 str(video)]))["streams"][0]
            clips.append({"id": f"{match}/{video.stem}", "path": str(video.relative_to(root)), "sha256": file_sha256(video),
                          "frames": int(meta["nb_frames"]), "weight": 1.2 if int(meta["height"]) > 1080 else 1.0})
    manifest = {"schema": 1, "root": str(root), "matches": matches, "pattern": pattern, "clips": clips}
    write_json(out, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("manifest")
    m.add_argument("--root", type=Path, required=True)
    m.add_argument("--match", action="append", required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--pattern", default="*.mp4", help="clip glob inside each match directory, e.g. videos/*.mp4")
    r = sub.add_parser("run")
    r.add_argument("--manifest", type=Path, required=True)
    r.add_argument("--archive", type=Path, required=True)
    r.add_argument("--part", type=int, required=True)
    r.add_argument("--parts", type=int, default=8)
    r.add_argument("--frames", type=int, default=0)
    r.add_argument("--workers", type=int, default=8)
    sub.add_parser("validate")
    a = sub.add_parser("assemble")
    a.add_argument("--manifest", type=Path, required=True)
    a.add_argument("--archive", type=Path, action="append", required=True)
    a.add_argument("--clips-root", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--report", type=Path, required=True)
    s.add_argument("--marks", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "manifest":
        make_manifest(args.root, args.match, args.out, args.pattern)
        return 0
    if args.command == "run":
        return run(args)
    if args.command == "validate":
        return validate()
    if args.command == "assemble":
        return assemble(args)
    write_json(args.out, score(json.loads(args.report.read_text()), json.loads(args.marks.read_text())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
