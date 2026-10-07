"""PLAN step B1: verify VISOR's frame mapping, convert the evaluation set, score a trivial candidate.

    python -m experiments.visor.b1 mapping --archive DIR --video V.MP4 ... --frames N
    python -m experiments.visor.b1 evalset --archive DIR --eval-set JSON --items K --frames N
    python -m experiments.visor.b1 validate --kind mapping|evalset

``mapping`` decodes, for each checked sparse frame, the video frames around the
index each candidate rule (`RULES`) predicts and compares them with the released
JPEG (mean absolute difference over RGB, 0–255). A rule holds on a frame when
its frame is within ``TIE`` of the best match; repeated frames make ties. A rule
holds for a video class (codec, size, rate) when it holds on every checked frame
of every video in that class.

``evalset`` converts each item to `ClipMasks`, places its frames on the video
between their run's keyframes (`visor.frame_alignment`), records an
environment-independent hash of its masks, compares the first frame (a dense
keyframe, redrawn at 480p) with the human sparse masks, and scores the trivial
candidate *hold the first frame* against the item with
`src.segmentation.evaluate.compare`.

Writes ``mapping.json`` or ``evalset.json`` to ``PS_STAGE_DIR``; masks go to
``PS_SCRATCH_DIR/publish/masks/<item>/masks.rle``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from src.segmentation import visor
from src.segmentation.evaluate import compare
from src.segmentation.masks import ClipMasks

#: A rule's frame may exceed the best match by this much (grey levels) and still hold.
TIE = 0.5
#: Best-match difference below which the JPEG and the decoded frame show the same image.
MATCH = 3.0
MARGIN = 4
#: Candidate decoded-frame index of a sparse frame from its VISOR number n,
#: its EPIC rgb number k (frame_mapping.json) and the video's rate.
#: ``epic_reader`` is the reader's own rule (`visor.epic_frame_to_video_index`).
RULES = {
    "visor_minus_one": lambda n, k, fps: n - 1,
    "visor_time_60": lambda n, k, fps: int(round((n - 1) * fps / 60.0)),
    "epic_minus_one": lambda n, k, fps: k - 1,
    "epic_time_60": lambda n, k, fps: int(round((k - 1) * fps / 60.0)),
    "epic_time_nominal": lambda n, k, fps: int(round((k - 1) * fps / round(fps))),
    "epic_ceil_60": lambda n, k, fps: int(math.ceil((k - 1) * fps / 60.0)),
    "epic_reader": lambda n, k, fps: visor.epic_frame_to_video_index(k, fps),
}


def stage_dir() -> Path:
    path = Path(os.environ.get("PS_STAGE_DIR") or ".")
    path.mkdir(parents=True, exist_ok=True)
    return path


def progress(completed: int) -> None:
    from experiments.jobs.monitor import publish_progress

    publish_progress(os.environ.get("PS_STAGE", "local"), completed)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 24):
            digest.update(block)
    return digest.hexdigest()


def class_key(info: dict[str, Any]) -> str:
    """The same key as tools/datasets/visor_b1_inputs.py: codec, size and nominal rate."""
    return f"{info['codec']} {info['width']}x{info['height']} {info['fps']:.3f}"


def spaced(items: list[str], count: int) -> list[str]:
    if count >= len(items):
        return list(items)
    if count == 1:
        return [items[-1]]
    picks = sorted({round(i * (len(items) - 1) / (count - 1)) for i in range(count)})
    return [items[i] for i in picks]


# ----------------------------------------------------------------- mapping

def check_video(video: Path, names: list[str], jpegs: Path, epic: dict[str, str], threads: int = 0) -> dict[str, Any]:
    import cv2
    from PIL import Image

    info = visor.video_info(video)
    rows = []
    for name in names:
        number = visor.frame_number(name)
        epic_number = visor.frame_number(epic[name])
        candidates = {rule: f(number, epic_number, info["fps"]) for rule, f in RULES.items()}
        started = time.time()
        decoded = dict(visor.decode_frames(video, candidates.values(), margin=MARGIN, threads=threads))
        # A rule may point past the end of the video; that is a miss, not a decoding gap.
        last = int(info["frames_declared"] or 0) - 1
        wanted = {i + d for i in candidates.values() for d in range(-MARGIN, MARGIN + 1) if 0 <= i + d and (last < 0 or i + d <= last)}
        released = np.asarray(Image.open(jpegs / name).convert("RGB"))
        resized = False
        mae: dict[int, float] = {}
        for index, frame in decoded.items():
            if frame.shape != released.shape:
                frame = cv2.resize(frame, (released.shape[1], released.shape[0]), interpolation=cv2.INTER_AREA)
                resized = True
            mae[index] = float(np.abs(frame.astype(np.int16) - released.astype(np.int16)).mean())
        if not mae:
            # No candidate lies inside the video: every rule misses this frame.
            rows.append({
                "name": name, "visor_frame": number, "epic_frame": epic_number, "candidates": candidates,
                "decoded_complete": False, "missing": sorted(wanted), "best_index": None, "best_mae": None,
                "rule_mae": {rule: None for rule in candidates}, "rule_offset": {rule: None for rule in candidates},
                "rule_holds": {rule: False for rule in candidates}, "ties": [], "repeated_with_previous": [],
                "jpeg_shape": list(released.shape), "resized_decoded": False, "window_mae": {},
                "error": "no candidate frame decoded", "seconds": round(time.time() - started, 2),
            })
            continue
        best = min(mae, key=lambda i: mae[i])
        ordered = sorted(decoded)
        repeats = [
            int(b) for a, b in zip(ordered, ordered[1:])
            if b == a + 1 and float(np.abs(decoded[a].astype(np.int16) - decoded[b].astype(np.int16)).mean()) < TIE
        ]
        rows.append({
            "name": name, "visor_frame": number, "epic_frame": epic_number, "candidates": candidates,
            "decoded_complete": set(decoded) == wanted, "missing": sorted(wanted - set(decoded)),
            "best_index": best, "best_mae": round(mae[best], 3),
            "rule_mae": {rule: round(mae[i], 3) if i in mae else None for rule, i in candidates.items()},
            "rule_offset": {rule: i - best for rule, i in candidates.items()},
            "rule_holds": {rule: i in mae and mae[i] - mae[best] <= TIE and mae[best] < MATCH for rule, i in candidates.items()},
            "ties": sorted(i for i in mae if mae[i] - mae[best] <= TIE),
            "repeated_with_previous": repeats, "jpeg_shape": list(released.shape), "resized_decoded": resized,
            "window_mae": {str(i): round(mae[i], 2) for i in sorted(mae)}, "seconds": round(time.time() - started, 2),
        })
    return {"video": video.stem, "info": info, "class": class_key(info), "frames": rows}


def summarize_rules(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for rule in RULES:
        holds = [row["rule_holds"][rule] for row in rows]
        offsets = Counter(row["rule_offset"][rule] for row in rows if row["rule_offset"][rule] is not None)
        excess = [row["rule_mae"][rule] - row["best_mae"] for row in rows if row["rule_mae"][rule] is not None]
        out[rule] = {
            "frames": len(rows), "holds": int(sum(holds)), "all_hold": bool(rows) and all(holds),
            "offset_to_best": {str(k): v for k, v in sorted(offsets.items())},
            "max_mae_excess": round(max(excess), 3) if excess else None,
        }
    return out


def command_mapping(args: argparse.Namespace) -> int:
    archive = Path(args.archive)
    selection = json.loads((archive / "selection.json").read_text())
    mapping = json.loads((archive / "frame_mapping.json").read_text())
    videos = [Path(v) for v in args.video]
    # One process per video; the decoders share the claimed CPU allowance.
    allowance = max(1, int(os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 1))
    workers = min(allowance, len(videos))
    threads = max(1, allowance // workers)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            path.stem: pool.submit(check_video, path, spaced(selection[path.stem]["checked"], args.frames),
                                   archive / "rgb_frames", mapping[path.stem], threads)
            for path in videos
        }
        for stem, future in futures.items():
            result = future.result()
            result["sha256_expected"] = selection[stem]["video"]["sha256"]
            result["rules"] = summarize_rules(result["frames"])
            results.append(result)
            progress(len(results))
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_class[result["class"]].extend(result["frames"])
    classes = {key: summarize_rules(rows) for key, rows in sorted(by_class.items())}
    write_json(stage_dir() / "mapping.json", {
        "tie": TIE, "match": MATCH, "margin": MARGIN, "frames_per_video": args.frames,
        "processes": workers, "decoder_threads": threads,
        "videos": results, "classes": classes,
        "rule_holds_for_classes": {rule: sorted(k for k, v in classes.items() if v[rule]["all_hold"]) for rule in RULES},
    })
    return 0


# ----------------------------------------------------------------- evalset

def mask_digest(clip: ClipMasks) -> str:
    """sha256 of the decoded masks, independent of the RLE encoder and compressor."""
    digest = hashlib.sha256()
    for index, instances in enumerate(clip.frames):
        digest.update(f"{index}:{(clip.labelled or [True] * len(clip))[index]}".encode())
        for inst in instances:
            digest.update(f"|{inst.class_name}|{inst.track_id}|{inst.label}|{inst.provenance}|".encode())
            digest.update(np.packbits(inst.mask()).tobytes())
    return digest.hexdigest()


def human_agreement(clip: ClipMasks, sparse: dict[str, Any], first: int) -> dict[str, Any]:
    """IoU of the dense masks with the human 1080p masks on every sparse frame inside the clip."""
    human_frames = visor.frames(sparse)
    return {
        str(first + index): frame_agreement(clip, index, human_frames[first + index])
        for index in range(len(clip)) if first + index in human_frames
    }


def frame_agreement(clip: ClipMasks, index: int, frame: visor.Frame) -> dict[str, Any]:
    """Per hand and per shared object label, IoU of dense masks (480p redraw) with human masks."""
    human = visor.frame_instances(frame, dense=False, tracks={})
    out: dict[str, Any] = {"present": True}
    for scope in ("left hand", "right hand"):
        a, b = clip.class_mask(index, scope), np.zeros((clip.height, clip.width), bool)
        for inst in human:
            if inst.class_name == scope:
                b |= inst.mask()
        if a.any() or b.any():
            out[scope] = round(float((a & b).sum() / (a | b).sum()), 4)
    labels = {inst.label for inst in clip.frames[index] if inst.class_name == "active object" and inst.label}
    objects = {}
    for label in sorted(labels & {inst.label for inst in human if inst.label}):
        a = np.zeros((clip.height, clip.width), bool)
        b = np.zeros_like(a)
        for inst in clip.frames[index]:
            if inst.label == label:
                a |= inst.mask()
        for inst in human:
            if inst.label == label:
                b |= inst.mask()
        objects[label] = round(float((a & b).sum() / max((a | b).sum(), 1)), 4)
    out["objects"] = objects
    out["human_only_labels"] = sorted(
        {inst.label for inst in human if inst.label} - {inst.label for inst in clip.frames[index] if inst.label}
    )
    return out


def hold_first(clip: ClipMasks) -> ClipMasks:
    """The trivial candidate: the first frame's masks, unchanged, on every frame."""
    held = ClipMasks(clip.classes, clip.height, clip.width, clip.fps, meta={"candidate": "hold-first"})
    held.frames = [list(clip.frames[0]) for _ in clip.frames]
    return held


def convert_item(item: dict[str, Any], archive: str, frames: int, publish: str) -> dict[str, Any]:
    import cv2

    cv2.setNumThreads(1)
    started = time.time()
    member = Path(archive) / item["dense_member"]
    dense_sha = file_sha256(member)
    doc = visor.load_annotations(member)
    mapping = json.loads((Path(archive) / "frame_mapping.json").read_text())[item["video"]]
    ratio = item["fps"] / visor.extraction_rate(item["fps"])
    alignment = visor.frame_alignment(doc, visor.keyframe_anchors(doc, mapping, item["fps"]), ratio)
    clip = visor.clip_masks(doc, item["first_visor_frame"], frames, fps=item["fps"], alignment=alignment, meta={
        "item": item["id"], "dense_sha256": dense_sha, "video_sha256": item["video_file"]["sha256"],
    })
    converted = time.time() - started
    target = clip.save(Path(publish) / "masks" / item["id"])
    sparse = visor.load_annotations(Path(archive) / f"annotations/{item['video']}.json")
    gaps = visor.hand_gaps(doc, sparse)
    scored = compare(hold_first(clip), clip)
    counts = Counter(inst.class_name for instances in clip.frames for inst in instances)
    return {
        "id": item["id"], "frames": len(clip), "labelled": sum(clip.labelled or []),
        "dense_sha256_matches": dense_sha == item["dense_sha256"],
        "keyframes": clip.meta["keyframes"], "first_is_keyframe": 0 in clip.meta["keyframes"],
        "aligned_exactly": clip.meta["aligned_exactly"], "max_abs_drift": clip.meta["max_abs_drift"],
        "video_indices": [clip.meta["video_indices"][0], clip.meta["video_indices"][-1]],
        "video_indices_match_eval_set": clip.meta["video_indices"][0] == item["first_video_index"],
        "instances_per_class": dict(sorted(counts.items())),
        "provenance": dict(Counter(inst.provenance for f in clip.frames for inst in f)),
        "mask_sha256": mask_digest(clip), "masks_rle_sha256": file_sha256(target),
        "human_agreement": human_agreement(clip, sparse, item["first_visor_frame"]),
        "hand_gap_frames": sum(1 for n in range(item["first_visor_frame"], item["first_visor_frame"] + frames) if n in gaps),
        "hold_first": scored, "convert_seconds": round(converted, 2),
        "total_seconds": round(time.time() - started, 2),
    }


def command_evalset(args: argparse.Namespace) -> int:
    eval_set = json.loads(Path(args.eval_set).read_text())
    items = eval_set["items"][: args.items] if args.items > 0 else eval_set["items"]
    publish = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir()) / "publish"
    workers = max(1, int(os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 1))
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(workers, len(items))) as pool:
        futures = [pool.submit(convert_item, item, args.archive, min(args.frames, item["frames"]), str(publish)) for item in items]
        for future in futures:
            rows.append(future.result())
            progress(len(rows))
    scopes = sorted({scope for row in rows for scope in row["hold_first"]["scopes"]})
    summary = {}
    for scope in scopes:
        values = [row["hold_first"]["scopes"][scope] for row in rows if scope in row["hold_first"]["scopes"]]
        summary[scope] = {key: round(float(np.mean([v[key] for v in values])), 4) for key in ("J", "F", "J&F")}
    hands = [v for row in rows for frame in row["human_agreement"].values() for k, v in frame.items() if k in visor.HANDS]
    objects = [v for row in rows for frame in row["human_agreement"].values() for v in frame.get("objects", {}).values()]
    write_json(stage_dir() / "evalset.json", {
        "eval_set": {"path": args.eval_set, "sha256": file_sha256(Path(args.eval_set)), "name": eval_set["name"]},
        "frames_per_item": args.frames, "items": rows,
        "hold_first_mean_over_items": summary,
        "dense_vs_human_keyframe_iou": {
            "hands": {"n": len(hands), "median": round(float(np.median(hands)), 4) if hands else None, "min": min(hands, default=None)},
            "objects": {"n": len(objects), "median": round(float(np.median(objects)), 4) if objects else None, "min": min(objects, default=None)},
        },
    })
    return 0


# ----------------------------------------------------------------- validate

def validate_mapping(stage: Path) -> dict[str, bool]:
    result = json.loads((stage / "mapping.json").read_text())
    rows = [row for video in result["videos"] for row in video["frames"]]
    return {
        "every_video_checked": bool(result["videos"]) and all(video["frames"] for video in result["videos"]),
        "every_window_decoded_completely": bool(rows) and all(row["decoded_complete"] for row in rows),
        "every_jpeg_matches_a_decoded_frame_mae_below_3": bool(rows) and all(
            row["best_mae"] is not None and row["best_mae"] < MATCH for row in rows
        ),
        "jpegs_are_1080p": all(row["jpeg_shape"] == [1080, 1920, 3] for row in rows),
        "every_class_has_a_decision": set(result["classes"]) == {video["class"] for video in result["videos"]},
    }


def validate_evalset(stage: Path) -> dict[str, bool]:
    result = json.loads((stage / "evalset.json").read_text())
    rows = result["items"]
    hands = result["dense_vs_human_keyframe_iou"]["hands"]
    scores = [s[k] for row in rows for s in row["hold_first"]["scopes"].values() for k in ("J", "F") if s[k] is not None]
    return {
        "items_converted": bool(rows),
        "every_frame_labelled": all(row["labelled"] == row["frames"] == result["frames_per_item"] for row in rows),
        "dense_sources_match_their_sha256": all(row["dense_sha256_matches"] for row in rows),
        "items_have_no_hand_gap": all(row["hand_gap_frames"] == 0 for row in rows),
        "items_contain_a_human_keyframe": all(row["human_agreement"] for row in rows),
        "items_aligned_exactly_on_the_video": all(row["aligned_exactly"] and row["video_indices_match_eval_set"] for row in rows),
        "only_interpolated_provenance": all(set(row["provenance"]) == {visor.INTERPOLATED} for row in rows),
        "hands_present": all(row["instances_per_class"].get("left hand", 0) + row["instances_per_class"].get("right hand", 0) > 0 for row in rows),
        "dense_keyframes_agree_with_human_masks_median_hand_iou_0.9": bool(hands["n"]) and hands["median"] >= 0.9,
        "scores_in_unit_interval": bool(scores) and all(0.0 <= v <= 1.0 for v in scores),
        "masks_hashed": all(len(row["mask_sha256"]) == 64 for row in rows),
    }


def command_validate(args: argparse.Namespace) -> int:
    stage = stage_dir()
    checks = validate_mapping(stage) if args.kind == "mapping" else validate_evalset(stage)
    report = {"passed": all(checks.values()), "checks": checks}
    target = os.environ.get("PS_VALIDATION_PATH")
    if target:
        write_json(Path(target), report)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    mapping = sub.add_parser("mapping")
    mapping.add_argument("--archive", required=True)
    mapping.add_argument("--video", action="append", required=True)
    mapping.add_argument("--frames", type=int, required=True)
    evalset = sub.add_parser("evalset")
    evalset.add_argument("--archive", required=True)
    evalset.add_argument("--eval-set", required=True)
    evalset.add_argument("--items", type=int, required=True, help="first K items; 0 for all")
    evalset.add_argument("--frames", type=int, required=True)
    validate = sub.add_parser("validate")
    validate.add_argument("--kind", choices=("mapping", "evalset"), required=True)
    args = parser.parse_args(argv)
    return {"mapping": command_mapping, "evalset": command_evalset, "validate": command_validate}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
