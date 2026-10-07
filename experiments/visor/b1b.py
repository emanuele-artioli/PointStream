"""PLAN step B1b: fill VISOR's missing hands and objects with SAM 3.1, and validate the fill.

    python -m experiments.visor.b1b fill --eval-set JSON --archive DIR --masks DIR --mask-record JSON \\
        --hand-objects DIR --checkpoint PT --video V.MP4 ... --items all|ID,ID --review K
    python -m experiments.visor.b1b validate
    python -m experiments.visor.b1b report --result b1b.json ... --out DIR

``fill``, per item of evaluation set v2:

1. *Span.* The item's 240-frame window lies inside a dense run; the span runs
   from the run's last human-labelled (sparse) frame at or before the window
   to its first one at or after it. Keyframes are placed on the video through
   ``frame_mapping.json`` (`visor.keyframe_anchors`, exact by B1's rules), so
   the span is decoded as consecutive video frames whatever VISOR's numbering
   does in between. The released sparse JPEGs inside the span are checked
   against the decoded frames, as B1 and B2 did.
2. *Objects.* Every object a human labelled at a keyframe of the span (hands by
   side, objects by label and repeat) is one SAM 3.1 object.
3. *Held-out validation.* For each pair of consecutive keyframes ``a < b``,
   SAM 3.1 is prompted with the human masks at ``a`` only, tracks forward to
   ``b``, and is scored there against the human masks it never saw: J and
   boundary F per object, with "hold the mask of ``a``" as the floor. Objects
   labelled at ``a`` but not at ``b`` score whether SAM also lets them go.
   Objects that VISOR's dense masks drop between ``a`` and ``b`` are flagged:
   they are the ones the fill is for, and the hard cases.
4. *Fill.* Per gap, SAM 3.1 is also prompted with the human masks at ``b``
   and tracks backward to the gap's midpoint; each frame takes the prediction
   from its nearer keyframe (the forward run of step 3 for the first half).
   Multiplex SAM 3.1 takes mask prompts only on the first frame of a fresh
   state, so each direction is its own session. On each window frame, every
   object the dense masks lack
   (`visor.object_key`) and SAM finds (at least ``MIN_AREA`` pixels) is added
   with provenance ``sam_from_label_prompt``; the dense masks are unchanged.
   The mask set is written in B1's ``masks.rle`` format
   (``publish/masks/<item>/masks.rle``), so ``b2 run --streams`` and D1 can
   score with it as a second mask set.
5. *Hand boxes.* Hands are checked against the EPIC-KITCHENS-100 hand-object
   detections (detector output, `src.segmentation.hand_objects`): a hand mask
   agrees when a detected hand of the same side (score >= 0.5) lies mostly
   inside its bounding box. VISOR's hands include the forearm, so the rate is
   read against the same rate for the human and the dense hands.
6. *Review.* A content-blind sample of window frames that are not keyframes
   (``REVIEW_SEED``) is drawn with the dense masks, the fill and SAM's own
   masks, for review by eye (``publish/review/``).

Writes ``b1b.json`` to ``PS_STAGE_DIR``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from experiments.visor.b1 import MATCH, file_sha256, mask_digest, progress, stage_dir, write_json
from src.segmentation import hand_objects, visor
from src.segmentation.evaluate import boundary_f, region_scores
from src.segmentation.masks import ClipMasks, Instance, decode_rle, encode_rle

SAM_TIER = "sam_from_label_prompt"
#: SAM masks smaller than this (pixels at 1080p) count as absent.
MIN_AREA = 64
#: A SAM instance is not added when more than this share of it lies on an instance already in the
#: frame: a track that lost its object and latched onto another (pilot 20261007T205637Z-0ba26d70:
#: 57 of 66 filled hands on P09_106 lay on the other hand).
MAX_OVERLAP = 0.5
#: Fill instances get track ids above the dense masks' own.
FILL_TRACK_BASE = 1000
#: A detected hand agrees with a hand mask when this share of its box lies inside the mask's box.
BOX_INSIDE = 0.5
HAND_SCORE = 0.5
REVIEW_SEED = "pointstream-b1b-review"
#: The adoption rule (docs/experiments.md, 2026-10-07 B1b), applied by ``report``.
DECISION: dict[str, Any] = {
    "hands": {"mean_J": 0.70, "median_J": 0.80, "margin_over_hold": 0.15},
    "objects": {"mean_J": 0.60, "median_J": 0.65, "margin_over_hold": 0.10},
    "hard_subset_min_n": 10, "hard_subset_mean_J": {"hands": 0.60, "objects": 0.50},
    "hands_released_min_n": 5, "hands_released_share": 0.70,
    "box_agreement_ratio_to_dense": 0.90,
}


# ----------------------------------------------------------------- plan

@dataclass
class TrackObject:
    object_id: int
    class_name: str
    label: str
    repeat: int

    @property
    def key(self) -> str:
        return visor.object_key(self.class_name, self.label)

    @property
    def group(self) -> str:
        return "hands" if self.class_name in visor.HANDS else "objects"


@dataclass
class Plan:
    item: dict[str, Any]
    span_start: int
    span_end: int
    keyframes: dict[int, int]  # VISOR sparse frame -> span index
    objects: list[TrackObject]
    human: dict[int, dict[int, np.ndarray]] = field(default_factory=dict)  # span index -> object id -> mask
    shape: tuple[int, int] = visor.FRAME_SIZE

    @property
    def length(self) -> int:
        return self.span_end - self.span_start + 1

    @property
    def window(self) -> tuple[int, int]:
        """Span indices of the window's first and last frame."""
        first = int(self.item["first_video_index"]) - self.span_start
        return first, first + int(self.item["frames"]) - 1

    def pairs(self) -> list[tuple[int, int]]:
        ordered = sorted(self.keyframes.values())
        return list(zip(ordered, ordered[1:]))


def plan_item(item: dict[str, Any], sparse: dict[str, Any], mapping: dict[str, str],
              shape: tuple[int, int] = visor.FRAME_SIZE) -> Plan:
    """The span, keyframes and objects of one item (no video needed); masks are drawn at ``shape``."""
    fps = float(item["fps"])
    anchors = visor.keyframe_anchors(sparse, mapping, fps)
    run_first, run_last = int(item["run"]["first"]), int(item["run"]["last"])
    keys = sorted(n for n in anchors if run_first <= n <= run_last)
    first, last = int(item["first_video_index"]), int(item["first_video_index"]) + int(item["frames"]) - 1
    before = [k for k in keys if anchors[k] <= first]
    after = [k for k in keys if anchors[k] >= last]
    if not before or not after:
        raise ValueError(f"{item['id']}: no human-labelled frame bounds the window in its run")
    start, end = anchors[before[-1]], anchors[after[0]]
    keyframes = {k: anchors[k] - start for k in keys if start <= anchors[k] <= end}
    human_frames = visor.frames(sparse)
    tracks: dict[tuple[str, str, int], int] = {}
    human: dict[int, dict[int, np.ndarray]] = {}
    for number, index in sorted(keyframes.items()):
        instances = visor.frame_instances(human_frames[number], dense=False, tracks=tracks, shape=shape)
        masks: dict[int, np.ndarray] = {}
        for inst in instances:
            masks[inst.track_id + 1] = masks.get(inst.track_id + 1, np.zeros(shape, bool)) | inst.mask()
        human[index] = masks
    objects = [TrackObject(track + 1, key[0], key[1], key[2]) for key, track in sorted(tracks.items(), key=lambda kv: kv[1])]
    return Plan(item, start, end, keyframes, objects, human, shape)


def dense_keys(dense: dict[str, Any]) -> dict[int, set[str]]:
    """Objects (`visor.object_key`) in each dense frame, by VISOR number."""
    return {n: {visor.object_key(visor.native_class(a), a["name"]) for a in f.annotations}
            for n, f in visor.frames(dense).items()}


def video_to_epic_frame(index: int, fps: float) -> int | None:
    """The EPIC rgb frame (1-indexed) whose decoded frame is ``index``, by the reader's verified rule."""
    guess = int(round(index * visor.extraction_rate(fps) / fps)) + 1
    for k in (guess, guess - 1, guess + 1):
        if k >= 1 and visor.epic_frame_to_video_index(k, fps) == index:
            return k
    return None


# ----------------------------------------------------------------- scoring

def box_of(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask)
    if not xs.size:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def inside_share(det: tuple[float, float, float, float], box: tuple[float, float, float, float]) -> float:
    w = max(0.0, min(det[2], box[2]) - max(det[0], box[0]))
    h = max(0.0, min(det[3], box[3]) - max(det[1], box[1]))
    area = max((det[2] - det[0]) * (det[3] - det[1]), 1e-9)
    return w * h / area


def hand_agreement(mask: np.ndarray, side: str, hands: list[hand_objects.Hand]) -> str:
    """``agrees`` (a same-side detection mostly inside the mask's box), ``other_side``, or ``no_detection``."""
    box = box_of(mask)
    if box is None:
        return "empty"
    height, width = mask.shape
    best = {s: max((inside_share(h.box.pixels(width, height), box) for h in hands if h.side == s), default=0.0)
            for s in visor.HANDS}
    if best[side] >= BOX_INSIDE:
        return "agrees"
    other = next(s for s in visor.HANDS if s != side)
    return "other_side" if best[other] >= BOX_INSIDE else "no_detection"


def score_pair(plan: Plan, a: int, b: int, sam_b: dict[int, tuple[np.ndarray, float]],
               dropped: set[str]) -> list[dict[str, Any]]:
    """Rows for one held-out pair: SAM prompted at span index ``a``, scored at ``b``."""
    rows = []
    for obj in plan.objects:
        if obj.object_id not in plan.human[a]:
            continue
        at_a = plan.human[a][obj.object_id]
        mask, prob = sam_b.get(obj.object_id, (np.zeros(plan.shape, bool), 0.0))
        area = int(mask.sum())
        row: dict[str, Any] = {"a": a, "b": b, "gap": b - a, "object_id": obj.object_id, "class": obj.class_name,
                               "group": obj.group, "label": obj.label, "sam_area_b": area,
                               "sam_present_b": area >= MIN_AREA, "presence_prob_b": round(prob, 4)}
        if obj.object_id in plan.human[b]:
            at_b = plan.human[b][obj.object_id]
            row.update({
                "labelled_b": True,
                "J": round(region_scores(mask, at_b)["iou"], 4), "F": round(boundary_f(mask, at_b), 4),
                "hold_J": round(region_scores(at_a, at_b)["iou"], 4), "hold_F": round(boundary_f(at_a, at_b), 4),
                "dense_dropped_between": obj.key in dropped,
            })
        else:
            row.update({"labelled_b": False, "released": area < MIN_AREA})
        rows.append(row)
    return rows


def summarize_validation(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    scored = [r for r in rows if r.get("labelled_b")]
    for name, keep in (("hands", lambda r: r["group"] == "hands"), ("objects", lambda r: r["group"] == "objects"),
                       ("left hand", lambda r: r["class"] == "left hand"),
                       ("right hand", lambda r: r["class"] == "right hand"),
                       ("active object", lambda r: r["class"] == "active object")):
        sel = [r for r in scored if keep(r)]
        hard = [r for r in sel if r["dense_dropped_between"]]
        released = [r for r in rows if not r.get("labelled_b") and keep(r)]
        out[name] = {
            "n": len(sel),
            "mean_J": stat(sel, "J", np.mean), "median_J": stat(sel, "J", np.median),
            "mean_F": stat(sel, "F", np.mean), "median_F": stat(sel, "F", np.median),
            "hold_mean_J": stat(sel, "hold_J", np.mean), "hold_median_J": stat(sel, "hold_J", np.median),
            "hold_mean_F": stat(sel, "hold_F", np.mean),
            "J_p10": stat(sel, "J", lambda v: np.percentile(v, 10)),
            "hard_n": len(hard), "hard_mean_J": stat(hard, "J", np.mean), "hard_mean_F": stat(hard, "F", np.mean),
            "hard_hold_mean_J": stat(hard, "hold_J", np.mean),
            "released_n": len(released),
            "released_share": round(float(np.mean([r["released"] for r in released])), 4) if released else None,
            "mean_gap_frames": stat(sel, "gap", np.mean),
        }
    return out


def stat(rows: list[dict[str, Any]], key: str, fn: Any) -> float | None:
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    return round(float(fn(values)), 4) if values else None


# ----------------------------------------------------------------- fill

def fill_clip(dense: ClipMasks, plan: Plan, sam: dict[int, dict[int, tuple[dict[str, Any], float]]]) -> tuple[ClipMasks, dict[str, Any]]:
    """B1's dense masks plus every object they lack that SAM found, per window frame."""
    window_first, _ = plan.window
    out = ClipMasks(dense.classes, dense.height, dense.width, dense.fps,
                    labelled=list(dense.labelled) if dense.labelled is not None else None, meta=dict(dense.meta))
    out.frames = [list(instances) for instances in dense.frames]
    added: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    frames_with_fill = 0
    for i in range(len(dense)):
        present = {visor.object_key(inst.class_name, inst.label) for inst in dense.frames[i]}
        occupied = [inst.mask() for inst in dense.frames[i]]
        new = False
        for obj in plan.objects:
            if obj.key in present:
                continue
            rle, prob = sam[window_first + i].get(obj.object_id, (None, 0.0))
            if rle is None:
                continue
            mask = decode_rle(rle)
            area = int(mask.sum())
            if area < MIN_AREA:
                continue
            if any((mask & other).sum() > MAX_OVERLAP * area for other in occupied):
                rejected[obj.class_name] += 1
                continue
            occupied.append(mask)
            box = box_of(mask)
            assert box is not None
            out.frames[i].append(Instance(obj.class_name, FILL_TRACK_BASE + obj.object_id, round(prob, 4), rle, box,
                                          SAM_TIER, obj.label))
            added[obj.class_name] += 1
            new = True
        frames_with_fill += new
    out.meta["fill"] = {
        "tier": SAM_TIER, "rule": "objects a human labelled at a keyframe of the span that the dense masks lack on a frame, where SAM 3.1 finds at least MIN_AREA pixels, at most MAX_OVERLAP of them on an instance already in the frame",
        "min_area": MIN_AREA, "max_overlap": MAX_OVERLAP, "track_id_base": FILL_TRACK_BASE,
        "objects": [{"object_id": o.object_id, "class_name": o.class_name, "label": o.label, "repeat": o.repeat,
                     "track_id": FILL_TRACK_BASE + o.object_id} for o in plan.objects],
        "span_video_indices": [plan.span_start, plan.span_end],
        "keyframes": {str(k): v for k, v in sorted(plan.keyframes.items())},
    }
    return out, {"instances_added": dict(sorted(added.items())), "rejected_overlapping": dict(sorted(rejected.items())),
                 "frames_with_fill": frames_with_fill}


def missing_share(clip: ClipMasks, item: dict[str, Any], expected: dict[int, set[str]]) -> dict[str, Any]:
    """B2's rule: objects a human labelled at both keyframes of the frame's run (or at the frame) that the set lacks."""
    shares, hands = [], 0
    labels: Counter[str] = Counter()
    for i in range(len(clip)):
        wanted = expected.get(int(item["first_visor_frame"]) + i, set())
        lacking = wanted - {visor.object_key(inst.class_name, inst.label) for inst in clip.frames[i]}
        labels.update(lacking)
        hands += bool(lacking & set(visor.HANDS))
        if wanted:
            shares.append(len(lacking) / len(wanted))
    return {"mean_share": round(float(np.mean(shares)), 4) if shares else None,
            "frames_with_missing_hand": hands, "missing_labels": dict(sorted(labels.items()))}


# ----------------------------------------------------------------- decoding (own process)

def decode_span(video: str, item: dict[str, Any], start: int, end: int, out: str, archive: str, threads: int) -> dict[str, Any]:
    """Write span frames ``start..end`` as ``<i>.png`` and check the item's sparse JPEGs inside the span."""
    import cv2
    from PIL import Image

    cv2.setNumThreads(max(1, threads))
    began = time.time()
    target = Path(out)
    target.mkdir(parents=True, exist_ok=True)
    mapping = json.loads((Path(archive) / "frame_mapping.json").read_text())[item["video"]]
    jpegs = []
    for member in item["sparse_jpegs"]:
        epic = visor.frame_number(mapping[Path(member).name])
        jpegs.append((member, visor.epic_frame_to_video_index(epic, item["fps"])))
    rgb_at: dict[int, np.ndarray] = {}
    expected = start
    for index, frame in visor.decode_frames(video, range(start, end + 1), threads=threads):
        if index != expected:
            raise RuntimeError(f"{item['id']}: decoded frame {index}, expected {expected}")
        cv2.imwrite(str(target / f"{index - start}.png"), frame[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if any(abs(index - at) <= 2 for _, at in jpegs):
            rgb_at[index] = frame
        expected += 1
    if expected != end + 1:
        raise RuntimeError(f"{item['id']}: decoded {expected - start} of {end - start + 1} span frames")
    gate = []
    for member, at in jpegs:
        released = np.asarray(Image.open(Path(archive) / member).convert("RGB")).astype(np.int16)
        mae = {i: float(np.abs(rgb_at[i].astype(np.int16) - released).mean()) for i in rgb_at if abs(i - at) <= 2}
        best = min(mae, key=lambda i: mae[i])
        gate.append({"jpeg": member, "index": at, "best_index": best, "rule_mae": round(mae[at], 3),
                     "best_mae": round(mae[best], 3), "holds": mae[at] - mae[best] <= 0.5 and mae[best] < MATCH})
    return {"frames": end - start + 1, "jpeg_gate": gate, "seconds": round(time.time() - began, 2)}


# ----------------------------------------------------------------- review

COLOURS = {"left hand": (230, 60, 60), "right hand": (60, 110, 240), "active object": (240, 200, 40)}
FILL_COLOUR = (40, 220, 90)


def review_frames(plan: Plan, count: int) -> list[int]:
    """Content-blind window frames (span indices) that are not keyframes, picked by hash of the item id."""
    first, last = plan.window
    keyframes = set(plan.keyframes.values())
    candidates = [i for i in range(first, last + 1) if i not in keyframes]
    picks: list[int] = []
    salt = 0
    while len(picks) < min(count, len(candidates)):
        digest = hashlib.sha256(f"{REVIEW_SEED}:{plan.item['id']}:{salt}".encode()).hexdigest()
        pick = candidates[int(digest, 16) % len(candidates)]
        if pick not in picks:
            picks.append(pick)
        salt += 1
    return sorted(picks)


def draw_overlay(rgb: np.ndarray, panels: list[tuple[str, list[tuple[np.ndarray, tuple[int, int, int], str, bool]]]],
                 hands: list[hand_objects.Hand], path: Path) -> None:
    """Side-by-side panels at 960x540; each mask tinted, fill outlined thick, detector hands as boxes."""
    import cv2

    tiles = []
    for title, masks in panels:
        tinted = rgb.astype(np.float32)
        for mask, colour, _, _ in masks:
            tinted[mask] = 0.55 * tinted[mask] + 0.45 * np.array(colour, np.float32)
        tile = tinted.astype(np.uint8)
        for mask, colour, text, outline in masks:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(tile, contours, -1, colour if not outline else FILL_COLOUR, 8 if outline else 3)
            box = box_of(mask)
            if box is not None:
                cv2.putText(tile, text, (int(box[0]) + 6, int(box[1]) + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                            (255, 255, 255), 3, cv2.LINE_AA)
        for hand in hands:
            x0, y0, x1, y1 = (int(v) for v in hand.box.pixels(rgb.shape[1], rgb.shape[0]))
            cv2.rectangle(tile, (x0, y0), (x1, y1), (255, 255, 255), 2)
            cv2.putText(tile, f"det {hand.side.split()[0]} {hand.score:.2f}", (x0 + 4, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(tile, title, (24, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4, cv2.LINE_AA)
        tiles.append(cv2.resize(tile, (960, 540), interpolation=cv2.INTER_AREA))
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.concatenate(tiles, 1)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])


# ----------------------------------------------------------------- fill command

def load_inputs(item: dict[str, Any], archive: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    dense = visor.load_annotations(archive / item["dense_member"])
    sparse = visor.load_annotations(archive / f"annotations/{item['video']}.json")
    mapping = json.loads((archive / "frame_mapping.json").read_text())[item["video"]]
    return dense, sparse, mapping


def process_item(tracker: Any, plan: Plan, frames_dir: Path, dense_doc: dict[str, Any], sparse: dict[str, Any],
                 dense_clip: ClipMasks, detections: list[hand_objects.FrameDetections], publish: Path,
                 review: int, profile: bool) -> dict[str, Any]:
    item = plan.item
    timing: dict[str, float] = {}
    began = time.time()
    images, height, width = tracker.load_frames(frames_dir)
    if len(images) != plan.length or (height, width) != plan.shape:
        raise RuntimeError(f"{item['id']}: loaded {len(images)} frames of {height}x{width}, expected {plan.length} of {plan.shape}")
    timing["load_frames"] = round(time.time() - began, 2)
    keys_by_frame = dense_keys(dense_doc)
    inverse = {index: number for number, index in plan.keyframes.items()}
    kernels: dict[str, Any] = {}

    # Per keyframe gap a < b: SAM prompted at a tracks forward to b (the held-out run, scored at b,
    # which it never saw), and prompted at b tracks backward to the midpoint. Each frame of the
    # fill takes the prediction from its nearer keyframe.
    began = time.time()
    rows: list[dict[str, Any]] = []
    hand_check: dict[str, Counter[str]] = defaultdict(Counter)
    fps = float(item["fps"])
    window_first, window_last = plan.window
    sam: dict[int, dict[int, tuple[dict[str, Any], float]]] = {}
    sam_masks_review: dict[int, dict[int, np.ndarray]] = {}
    review_at = set(review_frames(plan, review))
    prompt_agreement: list[float] = []
    tracked = {"forward": 0, "backward": 0}

    def keep(index: int, objects: dict[int, tuple[np.ndarray, float]]) -> None:
        if index in plan.human:
            for obj_id, mask in plan.human[index].items():
                if obj_id in objects:
                    prompt_agreement.append(region_scores(objects[obj_id][0], mask)["iou"])
        if window_first <= index <= window_last:
            sam[index] = {o: (encode_rle(m), p) for o, (m, p) in objects.items() if m.sum() >= MIN_AREA}
        if index in review_at:
            sam_masks_review[index] = {o: m for o, (m, _) in objects.items() if m.sum() >= MIN_AREA}

    for number, (a, b) in enumerate(plan.pairs()):
        middle = a + (b - a) // 2
        forward: dict[int, dict[int, tuple[np.ndarray, float]]] = {}
        if plan.human[a]:
            prompt = {0: dict(plan.human[a])}
            if profile and number == 0:
                # Kernels and attention family from a short separate run; the profiler would bloat on a whole gap.
                from experiments.audit.env_smoke import profile_cuda

                _, kernels = profile_cuda(lambda: list(tracker.track(images[a:a + 4], height, width, prompt)))
            forward = dict(tracker.track(images[a:b + 1], height, width, prompt))
            tracked["forward"] += b - a + 1
            dropped = {o.key for o in plan.objects
                       if any(o.key not in keys_by_frame.get(n, set()) for n in range(inverse[a] + 1, inverse[b]))}
            rows.extend(score_pair(plan, a, b, forward[b - a], dropped))
        backward: dict[int, dict[int, tuple[np.ndarray, float]]] = {}
        if plan.human[b]:
            last = b - (middle + 1)
            backward = dict(tracker.track(images[middle + 1:b + 1], height, width, {last: dict(plan.human[b])},
                                          start=last, reverse=True))
            tracked["backward"] += last + 1
        for index in range(a, b + 1):
            if index <= middle:
                keep(index, forward.get(index - a, {}))
            else:
                keep(index, backward.get(index - middle - 1, {}))
        # SAM's held-out hands on the in-between window frames against the detector; the dense
        # hands on the same frames are the calibration.
        for offset in range(1, b - a):
            i = a + offset - window_first
            if not 0 <= i < len(dense_clip) or offset not in forward:
                continue
            epic = video_to_epic_frame(plan.span_start + a + offset, fps)
            if epic is None:
                continue
            hands = hand_objects.hands_at(detections, epic, min_score=HAND_SCORE)
            for obj in plan.objects:
                if obj.class_name not in visor.HANDS or obj.object_id not in forward[offset]:
                    continue
                mask = forward[offset][obj.object_id][0]
                if mask.sum() >= MIN_AREA:
                    hand_check["sam_heldout"][hand_agreement(mask, obj.class_name, hands)] += 1
                dense_hand = dense_clip.class_mask(i, obj.class_name)
                if dense_hand.sum() >= MIN_AREA:
                    hand_check["dense_same_frames"][hand_agreement(dense_hand, obj.class_name, hands)] += 1
    timing["tracking"] = round(time.time() - began, 2)
    if set(sam) != set(range(window_first, window_last + 1)):
        raise RuntimeError(f"{item['id']}: SAM covered {len(sam)} of {window_last - window_first + 1} window frames")

    began = time.time()
    filled, fill_info = fill_clip(dense_clip, plan, sam)
    target = filled.save(publish / "masks" / item["id"])
    expected = visor.expected_objects(dense_doc, sparse)
    # Hands in the window against the detector, per tier.
    for i in range(len(filled)):
        epic = video_to_epic_frame(int(item["first_video_index"]) + i, fps)
        if epic is None:
            continue
        hands = hand_objects.hands_at(detections, epic, min_score=HAND_SCORE)
        is_key = window_first + i in plan.human
        for inst in filled.frames[i]:
            if inst.class_name in visor.HANDS:
                tier = "dense_keyframe" if is_key and inst.provenance == visor.INTERPOLATED else inst.provenance
                hand_check[f"window_{tier}"][hand_agreement(inst.mask(), inst.class_name, hands)] += 1
        for side in visor.HANDS:
            if any(h.side == side for h in hands):
                for name, clip in (("dense", dense_clip), ("filled", filled)):
                    has = any(inst.class_name == side for inst in clip.frames[i])
                    hand_check[f"detector_{side.split()[0]}_vs_{name}"]["mask_present" if has else "mask_absent"] += 1
    # Human hands at the keyframes of the span: the detector's own agreement with labels.
    for index, masks in plan.human.items():
        epic = video_to_epic_frame(plan.span_start + index, fps)
        if epic is None:
            continue
        hands = hand_objects.hands_at(detections, epic, min_score=HAND_SCORE)
        for obj in plan.objects:
            if obj.class_name in visor.HANDS and obj.object_id in masks:
                hand_check["human_keyframe"][hand_agreement(masks[obj.object_id], obj.class_name, hands)] += 1

    # Review overlays.
    import cv2

    reviews = []
    for index in sorted(review_at):
        rgb = cv2.imread(str(frames_dir / f"{index}.png"))[:, :, ::-1]
        i = index - window_first
        dense_layer = [(inst.mask(), COLOURS[inst.class_name], inst.label or inst.class_name, False)
                       for inst in dense_clip.frames[i]]
        fill_layer = [(inst.mask(), COLOURS[inst.class_name], f"FILL {inst.label}", True)
                      for inst in filled.frames[i] if inst.provenance == SAM_TIER]
        own = {o.object_id: o for o in plan.objects}
        sam_layer = [(mask, COLOURS[own[o].class_name], own[o].label, False) for o, mask in sam_masks_review[index].items()]
        epic = video_to_epic_frame(plan.span_start + index, fps)
        hands = hand_objects.hands_at(detections, epic, min_score=HAND_SCORE) if epic else []
        name = f"{item['id']}_w{i:03d}.jpg"
        draw_overlay(rgb, [(f"VISOR dense + fill (frame {i})", dense_layer + fill_layer), ("SAM 3.1, all objects", sam_layer)],
                     hands, publish / "review" / name)
        reviews.append({"file": f"review/{name}", "window_frame": i, "video_index": plan.span_start + index,
                        "fill_instances": [inst.label for inst in filled.frames[i] if inst.provenance == SAM_TIER],
                        "dense_instances": [inst.label for inst in dense_clip.frames[i]]})
    timing["fill_and_checks"] = round(time.time() - began, 2)
    return {
        "id": item["id"], "video": item["video"], "video_type": "EK-100" if abs(fps - 50) < 0.01 else "EK-55",
        "span_video_indices": [plan.span_start, plan.span_end], "span_frames": plan.length,
        "window_span_indices": [window_first, window_last],
        "keyframes": {str(k): v for k, v in sorted(plan.keyframes.items())},
        "objects": [{"object_id": o.object_id, "class": o.class_name, "label": o.label, "repeat": o.repeat} for o in plan.objects],
        "pairs_scored": len(plan.pairs()), "frames_tracked": tracked, "validation": rows,
        "prompt_reproduced_iou": {"n": len(prompt_agreement), "min": round(min(prompt_agreement), 4) if prompt_agreement else None,
                                  "median": round(float(np.median(prompt_agreement)), 4) if prompt_agreement else None},
        "fill": fill_info, "masks_rle": str(target), "masks_rle_sha256": file_sha256(target),
        "mask_sha256": mask_digest(filled), "dense_mask_sha256": mask_digest(dense_clip),
        "provenance": dict(Counter(inst.provenance for f in filled.frames for inst in f)),
        "missing_objects": {"visor_dense": missing_share(dense_clip, item, expected),
                            "visor_dense_sam_fill": missing_share(filled, item, expected)},
        "hand_check": {k: dict(v) for k, v in sorted(hand_check.items())},
        "review": reviews, "kernels": kernels, "timing": timing,
        "peak_gpu_mib": tracker.peak_gpu_mib(),
    }


def command_fill(args: argparse.Namespace) -> int:
    eval_set = json.loads(Path(args.eval_set).read_text())
    items = eval_set["items"] if args.items == "all" else [i for i in eval_set["items"] if i["id"] in args.items.split(",")]
    if args.items != "all" and len(items) != len(args.items.split(",")):
        raise SystemExit(f"unknown items in {args.items}")
    archive = Path(args.archive)
    videos = {Path(v).stem: v for v in args.video}
    records = {row["id"]: row for row in json.loads(Path(args.mask_record).read_text())["items"]}
    manifest = {f["path"]: f["sha256"] for f in json.loads(Path(args.hand_objects_manifest).read_text())["files"]}
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    plans = []
    for item in items:
        _, sparse, mapping = load_inputs(item, archive)
        plans.append(plan_item(item, sparse, mapping))
    # Decode in fresh processes before torch loads (B2: PyAV stalled in a process that had loaded torchvision).
    allowance = max(1, int(os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 1))
    pool = ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"))
    threads = max(1, allowance // 2)

    def submit(plan: Plan) -> Any:
        return pool.submit(decode_span, videos[plan.item["video"]], plan.item, plan.span_start, plan.span_end,
                           str(scratch / "frames" / plan.item["id"]), str(archive), threads)

    pending = {0: submit(plans[0])}
    if len(plans) > 1:
        pending[1] = submit(plans[1])
    from src.segmentation.sam31_tracker import Sam31MaskTracker, runtime

    began = time.time()
    tracker = Sam31MaskTracker(args.checkpoint)
    load_seconds = round(time.time() - began, 2)
    rows = []
    for number, plan in enumerate(plans):
        decoded = pending.pop(number).result()
        if number + 2 < len(plans):
            pending[number + 2] = submit(plans[number + 2])
        item = plan.item
        dense_doc, sparse, _ = load_inputs(item, archive)
        dense_clip = ClipMasks.load(mask_root(Path(args.masks), item["id"]))
        relative = f"hand-objects/{item['video'][:3]}/{item['video']}.pkl"
        detections_sha256 = file_sha256(Path(args.hand_objects) / relative)
        detections = hand_objects.load_detections(Path(args.hand_objects) / relative)
        row = process_item(tracker, plan, scratch / "frames" / item["id"], dense_doc, sparse, dense_clip, detections,
                           publish, args.review, profile=number == 0)
        row["decode"] = decoded
        row["dense_record_mask_sha256"] = records[item["id"]]["mask_sha256"]
        row["hand_objects"] = {"path": relative, "sha256": detections_sha256, "manifest_sha256": manifest.get(relative)}
        rows.append(row)
        shutil.rmtree(scratch / "frames" / item["id"], ignore_errors=True)
        write_json(publish / "items" / f"{item['id']}.json", row)
        progress(len(rows))
    pool.shutdown()
    validation = [r for row in rows for r in row["validation"]]
    write_json(stage_dir() / "b1b.json", {
        "eval_set": {"path": args.eval_set, "sha256": file_sha256(Path(args.eval_set)), "name": eval_set["name"]},
        "inputs": {"checkpoint": args.checkpoint, "checkpoint_sha256": tracker.load_report.get("checkpoint_sha256"),
                   "hand_objects": args.hand_objects, "hand_objects_manifest_sha256": file_sha256(Path(args.hand_objects_manifest)),
                   "masks": args.masks, "mask_record": args.mask_record,
                   "mask_record_sha256": file_sha256(Path(args.mask_record))},
        "sam": {**tracker.load_report, "load_seconds": load_seconds, "runtime": runtime(),
                "min_area": MIN_AREA, "policy": "per keyframe gap a < b: prompted at a, forward to b (held-out score at b); prompted at b, backward to the midpoint; each frame from its nearer keyframe"},
        "settings": {"review": args.review, "box_inside": BOX_INSIDE, "hand_score": HAND_SCORE,
                     "decision": DECISION},
        "items": rows,
        "validation_summary": summarize_validation(validation),
    })
    return 0


def mask_root(directory: Path, item_id: str) -> Path:
    for candidate in (directory / item_id, directory / "masks" / item_id, directory / "publish" / "masks" / item_id):
        if (candidate / "masks.rle").is_file():
            return candidate / "masks.rle"
    raise FileNotFoundError(f"no masks.rle for {item_id} under {directory}")


# ----------------------------------------------------------------- validate

def validate_result(stage: Path) -> dict[str, bool]:
    result = json.loads((stage / "b1b.json").read_text())
    rows = result["items"]
    sam = result["sam"]
    summary = result["validation_summary"]
    scores = [r[k] for row in rows for r in row["validation"] for k in ("J", "F", "hold_J", "hold_F") if k in r]
    capability = sam["runtime"]["capability"]
    first_kernels = rows[0]["kernels"] if rows else {}
    return {
        "items_processed": bool(rows),
        "sam_on_ada_or_a6000": any(name in sam["runtime"]["gpu"] for name in ("RTX 6000 Ada", "RTX A6000")),
        "sam_weights_load_completely": not sam["missing_keys"] and not sam["unexpected_keys"] and sam["loaded_keys"] == sam["model_keys"],
        "pinned_checkpoint": sam["checkpoint_sha256"] == "0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6",
        "cuda_kernels_ran": first_kernels.get("kernel_launches", 0) > 0,
        "flash_attention_on_sm80_plus": capability[0] < 8 or "flash" in first_kernels.get("attention_family", []),
        "sparse_jpegs_match_decoded_frames": all(g["holds"] for row in rows for g in row["decode"]["jpeg_gate"]),
        "dense_masks_are_b1s": all(row["dense_mask_sha256"] == row["dense_record_mask_sha256"] for row in rows),
        "hand_objects_match_manifest": all(row["hand_objects"]["sha256"] == row["hand_objects"]["manifest_sha256"] for row in rows),
        "window_inside_span": all(0 <= row["window_span_indices"][0] <= row["window_span_indices"][1] < row["span_frames"] for row in rows),
        "mask_prompts_reproduced": all(row["prompt_reproduced_iou"]["n"] > 0 and row["prompt_reproduced_iou"]["median"] >= 0.9 for row in rows),
        "held_out_keyframes_scored": summary["hands"]["n"] > 0,
        "scores_in_unit_interval": bool(scores) and all(0.0 <= v <= 1.0 for v in scores),
        "only_dense_and_fill_tiers": all(set(row["provenance"]) <= {visor.INTERPOLATED, SAM_TIER} for row in rows),
        "fill_never_adds_a_present_object": all(
            (row["missing_objects"]["visor_dense_sam_fill"]["mean_share"] or 0) <= (row["missing_objects"]["visor_dense"]["mean_share"] or 0)
            for row in rows),
        "masks_hashed": all(len(row["masks_rle_sha256"]) == 64 for row in rows),
        "review_frames_drawn": all(len(row["review"]) == result["settings"]["review"] for row in rows),
        "hands_checked_against_detector": any(row["hand_check"].get("human_keyframe") for row in rows),
    }


def validate_merge(stage: Path) -> dict[str, bool]:
    """Checks on the merged set as published: the stage's ``published.tar`` (scratch is gone by then)."""
    import tarfile

    from src.segmentation.masks import decompress

    result = json.loads((stage / "merge.json").read_text())
    rows = result["items"]
    allowed = {visor.INTERPOLATED, SAM_TIER}
    groups = set(result["groups"])
    sam_hands = 0
    matches = []
    with tarfile.open(stage / "published.tar") as tar:
        members = {m.name: m for m in tar.getmembers()}
        for row in rows:
            member = members.get(f"publish/masks/{row['id']}/masks.rle")
            handle = tar.extractfile(member) if member is not None else None
            if handle is None:
                matches.append(False)
                continue
            data = handle.read()
            matches.append(hashlib.sha256(data).hexdigest() == row["masks_rle_sha256"])
            clip = ClipMasks.from_doc(json.loads(decompress(data).decode("utf-8")))
            sam_hands += sum(1 for f in clip.frames for inst in f if inst.provenance == SAM_TIER and inst.class_name in visor.HANDS)
    return {
        "items_merged": bool(rows),
        "every_item_once": len({r["id"] for r in rows}) == len(rows),
        "published_masks_match_their_hashes": bool(matches) and all(matches),
        "only_dense_and_fill_tiers": all(set(r["provenance"]) <= allowed for r in rows),
        "dropped_groups_absent": "hands" in groups or sam_hands == 0,
        "masks_hashed": all(len(r["masks_rle_sha256"]) == 64 and len(r["mask_sha256"]) == 64 for r in rows),
    }


def command_validate(args: argparse.Namespace) -> int:
    checks = validate_merge(stage_dir()) if args.kind == "merge" else validate_result(stage_dir())
    report = {"passed": all(checks.values()), "checks": checks}
    target = os.environ.get("PS_VALIDATION_PATH")
    if target:
        write_json(Path(target), report)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


# ----------------------------------------------------------------- report

def decide(summary: dict[str, Any], hand_rates: dict[str, float | None]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for group in ("hands", "objects"):
        s, rule = summary[group], DECISION[group]
        checks = {
            "mean_J": s["mean_J"] is not None and s["mean_J"] >= rule["mean_J"],
            "median_J": s["median_J"] is not None and s["median_J"] >= rule["median_J"],
            "beats_hold": s["mean_J"] is not None and s["hold_mean_J"] is not None and s["mean_J"] - s["hold_mean_J"] >= rule["margin_over_hold"],
        }
        if s["hard_n"] >= DECISION["hard_subset_min_n"]:
            checks["hard_subset"] = s["hard_mean_J"] >= DECISION["hard_subset_mean_J"][group]
        if group == "hands":
            if s["released_n"] >= DECISION["hands_released_min_n"]:
                checks["releases_departed_hands"] = s["released_share"] >= DECISION["hands_released_share"]
            sam_rate, dense_rate = hand_rates.get("sam_heldout"), hand_rates.get("dense_same_frames")
            checks["detector_agreement"] = sam_rate is not None and dense_rate is not None and \
                sam_rate >= DECISION["box_agreement_ratio_to_dense"] * dense_rate
        out[group] = {"checks": checks, "adopt": all(checks.values())}
    return out


def agree_rate(counter: dict[str, int]) -> float | None:
    total = sum(v for k, v in counter.items() if k != "empty")
    return round(counter.get("agrees", 0) / total, 4) if total else None


def command_report(args: argparse.Namespace) -> int:
    rows = [row for path in args.result for row in json.loads(Path(path).read_text())["items"]]
    validation = [r for row in rows for r in row["validation"]]
    summary = summarize_validation(validation)
    by_type = {t: summarize_validation([r for row in rows if row["video_type"] == t for r in row["validation"]])
               for t in ("EK-100", "EK-55")}
    hand: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for key, counter in row["hand_check"].items():
            hand[key].update(counter)
    rates = {key: agree_rate(c) for key, c in hand.items() if not key.startswith("detector_")}
    missing = {name: float(np.mean([row["missing_objects"][name]["mean_share"] for row in rows
                                    if row["missing_objects"][name]["mean_share"] is not None]))
               for name in ("visor_dense", "visor_dense_sam_fill")}
    report = {
        "results": [{"path": p, "sha256": file_sha256(Path(p))} for p in args.result],
        "items": len(rows), "validation": summary, "validation_by_type": by_type,
        "hand_check": {k: dict(v) for k, v in sorted(hand.items())}, "hand_agreement_rates": rates,
        "missing_objects_mean_share": {k: round(v, 4) for k, v in missing.items()},
        "fill": {"instances_added": dict(sum((Counter(row["fill"]["instances_added"]) for row in rows), Counter())),
                 "rejected_overlapping": dict(sum((Counter(row["fill"].get("rejected_overlapping", {})) for row in rows), Counter())),
                 "frames_with_fill": sum(row["fill"]["frames_with_fill"] for row in rows),
                 "items_with_fill": sum(1 for row in rows if row["fill"]["frames_with_fill"])},
        "decision": decide(summary, rates),
        "review": [{"item": row["id"], **r} for row in rows for r in row["review"]],
    }
    out = Path(args.out)
    write_json(out / "b1b-report.json", report)
    print(json.dumps({k: report[k] for k in ("items", "validation", "hand_agreement_rates", "missing_objects_mean_share", "fill", "decision")}, indent=1))
    return 0


def command_merge(args: argparse.Namespace) -> int:
    """One mask set from the fill jobs' published masks, keeping SAM instances of the adopted groups only."""
    keep = set(args.groups.split(","))
    if not keep <= {"hands", "objects"}:
        raise SystemExit(f"unknown groups {args.groups}")
    publish = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch") / "publish"
    rows = []
    for result_path, masks_dir in zip(args.result, args.masks):
        for row in json.loads(Path(result_path).read_text())["items"]:
            source = mask_root(Path(masks_dir), row["id"])
            if file_sha256(source) != row["masks_rle_sha256"]:
                raise RuntimeError(f"{row['id']}: masks.rle differs from its fill record")
            clip = ClipMasks.load(source)
            dropped = 0
            for i, instances in enumerate(clip.frames):
                kept = [inst for inst in instances if inst.provenance != SAM_TIER
                        or ("hands" if inst.class_name in visor.HANDS else "objects") in keep]
                dropped += len(instances) - len(kept)
                clip.frames[i] = kept
            clip.meta.setdefault("fill", {})["groups_kept"] = sorted(keep)
            target = clip.save(publish / "masks" / row["id"])
            rows.append({"id": row["id"], "source_masks_rle_sha256": row["masks_rle_sha256"], "dropped_sam_instances": dropped,
                         "masks_rle_sha256": file_sha256(target), "mask_sha256": mask_digest(clip),
                         "provenance": dict(Counter(inst.provenance for f in clip.frames for inst in f))})
    if len({r["id"] for r in rows}) != len(rows):
        raise RuntimeError("an item appears in more than one fill result")
    write_json(stage_dir() / "merge.json", {"groups": sorted(keep), "results": [
        {"path": p, "sha256": file_sha256(Path(p))} for p in args.result], "items": rows})
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    fill = sub.add_parser("fill")
    fill.add_argument("--eval-set", required=True)
    fill.add_argument("--archive", required=True, help="extracted B1 archive: frame_mapping.json, annotations/, dense/, rgb_frames/")
    fill.add_argument("--masks", required=True, help="B1's published masks (one masks.rle per item)")
    fill.add_argument("--mask-record", required=True, help="B1's evalset.json, with each item's mask_sha256")
    fill.add_argument("--hand-objects", required=True, help="extracted hand-object detections: hand-objects/<P>/<video>.pkl")
    fill.add_argument("--hand-objects-manifest", required=True, help="Datasets/manifests/EPIC-KITCHENS-hand-objects.json")
    fill.add_argument("--checkpoint", required=True)
    fill.add_argument("--video", action="append", required=True)
    fill.add_argument("--items", required=True, help="all, or comma-separated item ids")
    fill.add_argument("--review", type=int, default=2, help="review overlays per item")
    merge = sub.add_parser("merge")
    merge.add_argument("--result", action="append", required=True, help="a fill job's b1b.json; pairs with --masks")
    merge.add_argument("--masks", action="append", required=True, help="that job's extracted published.tar")
    merge.add_argument("--groups", required=True, help="hands, objects, or hands,objects")
    validate = sub.add_parser("validate")
    validate.add_argument("--kind", choices=("fill", "merge"), default="fill")
    report = sub.add_parser("report")
    report.add_argument("--result", action="append", required=True)
    report.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    commands = {"fill": command_fill, "merge": command_merge, "validate": command_validate, "report": command_report}
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
