"""K8 bounded recovery: reuse cuts/completed shots, checkpoint <=240-frame chunks.

The registration algorithm and fixed decision rule are unchanged. Separate CPU
processes avoid the Python-thread bottleneck on difficult Hough hypotheses.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import shutil
import subprocess
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import k8
from experiments.visor.b1 import file_sha256, write_json

CHUNK = 240


def chunks(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for i, a, b in manifest["missing"]:
        for start in range(a, b, CHUNK):
            result.append({"id": f"shot-{i:04d}-chunk-{start:06d}", "parent": f"shot-{i:04d}",
                           "start": start, "end": min(start + CHUNK, b), "shot_start": a, "shot_end": b})
    return result


def groups(items: list[dict[str, Any]], count: int) -> list[list[dict[str, Any]]]:
    result: list[list[dict[str, Any]]] = [[] for _ in range(count)]
    loads = [0] * count
    for item in sorted(items, key=lambda r: (-(r["end"] - r["start"]), r["id"])):
        index = min(range(count), key=lambda n: (loads[n], n))
        result[index].append(item)
        loads[index] += item["end"] - item["start"]
    return [sorted(g, key=lambda r: (r["start"], r["id"])) for g in result]


def worker(video: Path, fps: float, item: dict[str, Any], publish: Path, ffmpeg: str, seek_from: float) -> dict[str, Any]:
    import cv2
    cv2.setNumThreads(1)
    a, b = item["shot_start"], item["shot_end"]
    reviews = {"first": a, "middle": a + (b - a) // 2, "last": b - 1}
    return k8.process_shot(video, fps, item["start"], item["end"], item["id"], publish, ffmpeg,
                           reviews, item["parent"], seek_from=seek_from)


def load_report(stage: Path) -> tuple[dict[str, Any], set[str]]:
    with tarfile.open(stage / "published.tar") as archive:
        handle = archive.extractfile("publish/recovery-report.json")
        if handle is None:
            raise ValueError("no recovery report")
        return json.load(handle), {m.name for m in archive.getmembers() if m.isfile() and m.size > 0}


def run(args: argparse.Namespace) -> int:
    from experiments.jobs.monitor import publish_progress
    manifest = json.loads(args.manifest.read_text())
    base = manifest["base_report"]
    if file_sha256(args.video) != base["source"]["video_sha256"]:
        raise ValueError("source changed")
    selected = groups(chunks(manifest), args.groups)[args.group] if args.group >= 0 else [
        {"id": "smoke-court", "parent": "smoke-court", "start": base["all_spans"][9][0],
         "end": min(base["all_spans"][9][0] + args.frames, base["all_spans"][9][1]),
         "shot_start": base["all_spans"][9][0], "shot_end": base["all_spans"][9][1]}]
    if args.group >= 0 and args.frames:
        selected = selected[:max(1, args.frames // CHUNK)]
    publish = Path(os.environ["PS_SCRATCH_DIR"]) / "publish"
    (publish / "images").mkdir(parents=True)
    (publish / "shots").mkdir()
    native: dict[str, Any] = {}
    for name in ("ffmpeg", "ffprobe"):
        path = shutil.which(name)
        if path is None:
            raise FileNotFoundError(name)
        native[name] = {"path": path, "version": subprocess.check_output([path, "-version"], text=True).splitlines()[:3]}
    began = time.monotonic()
    rows = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(worker, args.video, base["fps"], item, publish, native["ffmpeg"]["path"],
                               k8.seek_offset(args.video, native["ffprobe"]["path"])): item["id"] for item in selected}
        for future in as_completed(futures):
            rows[futures[future]] = future.result()
            publish_progress(os.environ["PS_STAGE"], len(rows))
    report = {"schema": 1, "source": base["source"], "source_frames": base["source_frames"], "size": [1920, 1080],
              "selected": selected, "chunks": [rows[item["id"]] for item in selected], "native_tools": native,
              "runtime_seconds": time.monotonic() - began, "group": args.group,
              "manifest_sha256": file_sha256(args.manifest), "job_dir": os.environ["PS_JOB_DIR"],
              "registration_code_sha256": file_sha256(Path(k8.__file__).with_name("court.py")),
              "worker_code_sha256": file_sha256(Path(__file__))}
    write_json(publish / "recovery-report.json", report)
    write_json(Path(os.environ["PS_STAGE_DIR"]) / "recovery.json", {k: v for k, v in report.items() if k != "chunks"})
    print(json.dumps({"chunks": len(rows), "frames": sum(len(r["registration"]) for r in rows.values()),
                      "seconds": report["runtime_seconds"]}), flush=True)
    return 0


def validate() -> int:
    report, names = load_report(Path(os.environ["PS_STAGE_DIR"]))
    rows = report["chunks"]
    checks = {"requested_chunks": {r["id"] for r in rows} == {s["id"] for s in report["selected"]},
              "every_frame": all([r["frame"] for r in s["registration"]] == list(range(s["start"], s["end"])) for s in rows),
              "bounded_chunks": all(s["end"] - s["start"] <= CHUNK for s in rows),
              "some_real_registration": any(r["registered"] for s in rows for r in s["registration"]),
              "fixed_classification": all(s["predicted"] == k8.shot_prediction(s["registration"])["predicted"] for s in rows),
              "finite_homographies": all(r["homography"] is None or (np.isfinite(r["homography"]).all() and abs(np.linalg.det(r["homography"])) > 1e-10) for s in rows for r in s["registration"]),
              "review_images": all(f"publish/{p}" in names for s in rows for p in s["extra_images"].values()),
              "cpu_only": not os.environ.get("CUDA_VISIBLE_DEVICES"), "native_1080p": report["size"] == [1920, 1080]}
    result = {"passed": all(checks.values()), "checks": checks}
    write_json(Path(os.environ["PS_VALIDATION_PATH"]), result)
    print(json.dumps(result))
    return 0 if result["passed"] else 1


def assemble(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text())
    result = manifest["base_report"]
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    (out / "images").mkdir()
    shots: dict[str, Any] = {}
    pending: dict[str, list[dict[str, Any]]] = {}
    images: dict[str, dict[str, str]] = {}
    evidence = []

    def copy_image(archive: tarfile.TarFile, name: str) -> None:
        if Path(name).parent != Path("images") or Path(name).suffix != ".jpg":
            raise ValueError("unsafe image")
        handle = archive.extractfile(f"publish/{name}")
        if handle is None:
            raise ValueError("missing image")
        target = out / name
        if target.exists():
            raise ValueError("duplicate review image")
        target.write_bytes(handle.read())

    for record in manifest["completed_archives"]:
        path = Path(record["path"])
        if file_sha256(path) != record["sha256"]:
            raise ValueError("completed evidence changed")
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                if member.isfile() and member.name.startswith("publish/shots/") and member.name.endswith(".json"):
                    handle = archive.extractfile(member)
                    if handle is None:
                        raise ValueError("missing shot")
                    shot = json.load(handle)
                    if shot["id"] in shots:
                        raise ValueError("duplicate complete shot")
                    shots[shot["id"]] = shot
                    for name in shot["images"]:
                        copy_image(archive, name)
        evidence.append(record)
    seen = set()
    for path in args.archive:
        with tarfile.open(path) as archive:
            handle = archive.extractfile("publish/recovery-report.json")
            if handle is None:
                raise ValueError("missing recovery report")
            report = json.load(handle)
            if report["manifest_sha256"] != file_sha256(args.manifest):
                raise ValueError("different recovery input")
            for chunk in report["chunks"]:
                if chunk["id"] in seen:
                    raise ValueError("duplicate chunk")
                seen.add(chunk["id"])
                parent = chunk["parent_shot"]
                if parent in shots:
                    raise ValueError("reprocessed completed shot")
                pending.setdefault(parent, []).append(chunk)
                for role, name in chunk["extra_images"].items():
                    if role in images.setdefault(parent, {}):
                        raise ValueError("duplicate review role")
                    images[parent][role] = name
                    copy_image(archive, name)
            evidence.append({"path": str(path), "sha256": file_sha256(path),
                             "job_dir": report["job_dir"], "native_tools": report["native_tools"],
                             "registration_code_sha256": report["registration_code_sha256"],
                             "worker_code_sha256": report["worker_code_sha256"], "runtime_seconds": report["runtime_seconds"]})
    if seen != {c["id"] for c in chunks(manifest)}:
        raise ValueError("missing recovery chunks")
    for parent, parts in pending.items():
        i = int(parent.split("-")[1])
        a, b = result["all_spans"][i]
        rows = [r for part in sorted(parts, key=lambda p: p["start"]) for r in part["registration"]]
        if [r["frame"] for r in rows] != list(range(a, b)):
            raise ValueError("missing or overlapping frame")
        shots[parent] = {"id": parent, "start": a, "end": b, "registration": rows,
                         "images": [images[parent][role] for role in ("first", "middle", "last")],
                         "conversion_commands": [p["conversion_command"] for p in parts], **k8.shot_prediction(rows)}
    ordered = sorted(shots.values(), key=lambda s: s["start"])
    if [(s["start"], s["end"]) for s in ordered] != [tuple(s) for s in result["all_spans"]]:
        raise ValueError("incomplete whole match")
    if [r["frame"] for s in ordered for r in s["registration"]] != list(range(result["source_frames"])):
        raise ValueError("whole-match frame coverage failed")
    if file_sha256(args.source) != result["source"]["video_sha256"]:
        raise ValueError("review source changed")
    # The recovery manifest's base metadata came from the original smoke.
    # Its cut *data* is the saved full scan; label the reconstructed full
    # invocation explicitly rather than retaining the smoke's 30 s command.
    if "scene_command" in result:
        command = list(result["scene_command"])
        command[command.index("-t") + 1] = str(result["source_frames"] / result["fps"])
        result["scene_command"] = [a.replace("-smoke/scene-metadata.txt", "-full/scene-metadata.txt") for a in command]
        result["scene_command_reconstructed"] = True
        result["scene_metadata_sha256"] = manifest["scene_metadata_sha256"]
        result["scene_dispatch"] = str(Path(manifest["completed_archives"][0]["path"]).parent / "dispatch.json")
    result.update(shots=ordered, evidence=evidence, recovery_manifest_sha256=file_sha256(args.manifest))
    result["review_id"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()[:16]
    (out / "source.mp4").symlink_to(args.source)
    write_json(out / "k8-report.json", result)
    k8.review_page(result, out / "review.html")
    ready = {"review_id": result["review_id"], "shots": len(ordered), "frames": result["source_frames"], "status": "awaiting_user_marks"}
    write_json(out / "READY.json", ready)
    print(json.dumps(ready))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--group", type=int, required=True)
    p.add_argument("--groups", type=int, default=12)
    p.add_argument("--frames", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    sub.add_parser("validate")
    a = sub.add_parser("assemble")
    a.add_argument("--manifest", type=Path, required=True)
    a.add_argument("--archive", type=Path, action="append", required=True)
    a.add_argument("--source", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    return {"run": run, "validate": lambda _: validate(), "assemble": assemble}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
