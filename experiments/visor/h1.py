"""PLAN step H1: what the foreground is made of, and how much of it motion parameters could explain.

    python -m experiments.visor.h1 run --eval-set JSON --archive DIR --masks NAME DIR --mask-record NAME JSON \\
        --video V.MP4 ... --items all|ID,ID --frames N --wilor-checkpoint CKPT --wilor-detector PT \\
        --mano-left PKL --mano-right PKL --svt-streams DIR --svt-result B2.JSON --inspect BIN
    python -m experiments.visor.h1 validate
    python -m experiments.visor.h1 report --result h1.json ... --b2-report JSON --out DIR

``run``, per item of evaluation set v2 (docs/experiments.md, 2026-10-08 H1):

1. *Window.* Decoded as B2 does, in a fresh process, with the sparse-JPEG gate; both mask sets
   (``visor_dense``, ``visor_dense_sam_fill``) checked against their records.
2. *Hands.* WiLoR's detector finds hand boxes; each VISOR hand takes the box most of which lies in
   the mask's bounding box, and WiLoR fits MANO with VISOR's handedness. The mesh is projected with
   WiLoR's camera and rasterized; the wrist line (through the wrist joint, perpendicular to wrist →
   middle knuckle) splits the VISOR hand mask into hand and forearm.
3. *Parts.* Each pixel: hand, forearm, hand without a fit, handled object (a human labelled it in
   contact with a hand at a keyframe bounding or inside the window), other object, background.
4. *Motion.* Backward DIS flow per frame pair; per object (and per hand and forearm part) a
   similarity and a homography fitted to the flow in its mask; the previous frame warped by each
   model, by no motion and by the flow itself, scored inside the mask; and a reference carried
   forward by the chained homographies, refreshed where it stops holding.
5. *Bits.* B2's SVT-AV1 streams decoded by libaom's ``inspect`` with bit accounting; each block's
   bits spread over its pixels and split by the parts of the frame it displays as.
6. *Parameters.* MANO pose, orientation and camera per hand and frame, the homography's corner
   displacements per object and frame, quantized; the projection error the quantization causes.
7. *Review.* Content-blind frames (``REVIEW_SEED``) drawn with the parts, the rendered hands and the
   warped objects, plus the frame with the worst hand fit, labelled as such.

Writes ``h1.json`` to ``PS_STAGE_DIR``; per-item rows and overlays to ``publish/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from experiments.visor.b1 import MATCH, file_sha256, progress, stage_dir, write_json
from src.segmentation import visor
from src.segmentation.masks import ClipMasks

FILL = "visor_dense_sam_fill"
DENSE = "visor_dense"
#: Parts of a frame, in label order; earlier parts win where masks overlap.
PARTS = ("background", "hand", "forearm", "hand_no_fit", "handled_object", "other_object")
FOREGROUND_PARTS = PARTS[1:]
MI = 4  # AV1 mode-info unit (pixels)
#: WiLoR's own demo settings.
DETECTOR_CONF = 0.3
RESCALE = 2.0
#: A detection belongs to a VISOR hand when this share of its box lies in the mask's bounding box.
BOX_INSIDE = 0.5
EDGE_PX = 4
OCCLUSION_SHARE = 0.25
BLUR_SHARPNESS = 0.5
BLUR_MOTION = 0.10
#: Objects moving more than this many pixels per frame count as motion-blurred.
BLUR_SPEED_PX = 10.0
#: Fewer flow samples than this in a mask: no fit.
MIN_POINTS = 24
SAMPLE_STRIDE = 4
RANSAC_PX = 2.0
#: A reference holds at a frame when PSNR on its covered pixels is at least the threshold and at
#: most MAX_UNCOVERED of the mask is uncovered.
HOLD_PSNR = (30.0, 33.75)
MAX_UNCOVERED = 0.20
#: Quantization of the parameters (docs/experiments.md H1, step 4).
QUANT = {"angle_deg": 1.0, "position_px": 0.25, "log_depth": 0.002, "corner_px": 0.125, "betas": 0.01}
REVIEW_SEED = "pointstream-h1-review"
#: The rule (docs/experiments.md, 2026-10-08 H1), applied by ``report`` to means over the items.
DECISION: dict[str, Any] = {
    "min_foreground_bit_share": 0.10, "bit_share_points": [48, 62],
    "hands": {"fitted_share": 0.90, "median_hand_iou": 0.70, "share_at_0.6": 0.70},
    "rigid": {"held_share": 0.70, "median_reference_life_s": 0.5, "hold_psnr": 30.0},
    "max_parameter_rate_share": 0.10, "parameter_rate_point": 62,
    "forearm_render_share": 0.20,
}
COLOURS = {"hand": (230, 60, 60), "forearm": (250, 150, 60), "hand_no_fit": (200, 0, 200),
           "handled_object": (240, 220, 40), "other_object": (60, 200, 230)}


# ----------------------------------------------------------------- geometry

def wrist_split(mask: np.ndarray, wrist: np.ndarray, knuckle: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(hand, forearm)``: the mask on the fingers' side of the wrist line and beyond it."""
    axis = np.asarray(knuckle, float) - np.asarray(wrist, float)
    norm = float(np.hypot(*axis))
    if norm < 1e-6:
        return mask.copy(), np.zeros_like(mask)
    axis /= norm
    ys, xs = np.nonzero(mask)
    side = (xs - wrist[0]) * axis[0] + (ys - wrist[1]) * axis[1]
    hand = np.zeros_like(mask)
    forearm = np.zeros_like(mask)
    hand[ys[side >= 0], xs[side >= 0]] = True
    forearm[ys[side < 0], xs[side < 0]] = True
    return hand, forearm


def rasterize(points: np.ndarray, faces: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Silhouette of a projected mesh (``points`` in pixels)."""
    import cv2

    height, width = shape
    out = np.zeros(shape, np.uint8)
    lo = np.floor(points.min(0)).astype(int)
    hi = np.ceil(points.max(0)).astype(int)
    x0, y0 = max(lo[0], 0), max(lo[1], 0)
    x1, y1 = min(hi[0] + 1, width), min(hi[1] + 1, height)
    if x1 <= x0 or y1 <= y0:
        return out.astype(bool)
    crop = np.zeros((y1 - y0, x1 - x0), np.uint8)
    shifted = np.round((points - [x0, y0]) * 16).astype(np.int32)
    for tri in shifted[faces]:
        cv2.fillConvexPoly(crop, tri, (1,), lineType=cv2.LINE_8, shift=4)
    out[y0:y1, x0:x1] = crop
    return out.astype(bool)


def iou(a: np.ndarray, b: np.ndarray) -> float | None:
    union = int((a | b).sum())
    return float((a & b).sum()) / union if union else None


def box_of(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask)
    if not xs.size:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def inside_share(det: tuple[float, ...], box: tuple[float, ...]) -> float:
    w = max(0.0, min(det[2], box[2]) - max(det[0], box[0]))
    h = max(0.0, min(det[3], box[3]) - max(det[1], box[1]))
    return w * h / max((det[2] - det[0]) * (det[3] - det[1]), 1e-9)


def match_detections(hands: dict[str, np.ndarray], boxes: np.ndarray) -> dict[str, int]:
    """VISOR hand side -> detection index, one to one; by pixels of the box on the mask."""
    pairs = []
    for side, mask in hands.items():
        mbox = box_of(mask)
        if mbox is None:
            continue
        for j, det in enumerate(boxes):
            if inside_share(tuple(det), mbox) < BOX_INSIDE:
                continue
            x0, y0, x1, y1 = (int(round(v)) for v in det)
            area = max((x1 - x0) * (y1 - y0), 1)
            pairs.append((float(mask[max(y0, 0):y1, max(x0, 0):x1].sum()) / area, side, j))
    out: dict[str, int] = {}
    for _, side, j in sorted(pairs, reverse=True):
        if side not in out and j not in out.values():
            out[side] = j
    return out


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos = (np.trace(a.T @ b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


# ----------------------------------------------------------------- motion

def psnr(sse: float, count: int) -> float | None:
    if count == 0:
        return None
    mse = sse / (3 * count)
    return 99.0 if mse <= 1e-10 else float(10 * math.log10(255.0 ** 2 / mse))


def sample_map(points_src: np.ndarray, image: np.ndarray, nearest: bool = False) -> np.ndarray:
    """``image`` sampled at float ``(x, y)`` points (bilinear, or nearest for masks; 0 outside)."""
    import cv2

    count = len(points_src)
    # remap needs maps under 32767 on each side: fold the points into rows of WIDTH.
    width = 1024
    rows = max(1, -(-count // width))
    padded = np.zeros((rows * width, 2), np.float32)
    padded[:count] = points_src
    mx = np.ascontiguousarray(padded[:, 0].reshape(rows, width))
    my = np.ascontiguousarray(padded[:, 1].reshape(rows, width))
    if nearest:
        out = cv2.remap(image.astype(np.uint8), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0,))
        return out.reshape(-1)[:count] > 0
    out = cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(rows * width, -1)[:count]


def apply_h(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    homog = np.c_[points, np.ones(len(points))] @ h.T
    return homog[:, :2] / homog[:, 2:3]


def fit_models(flow: np.ndarray, mask_t: np.ndarray) -> dict[str, Any]:
    """Similarity and homography (previous frame -> this one) fitted to the backward flow in the mask."""
    import cv2

    ys, xs = np.nonzero(mask_t[::SAMPLE_STRIDE, ::SAMPLE_STRIDE])
    xs, ys = xs * SAMPLE_STRIDE, ys * SAMPLE_STRIDE
    if len(xs) < MIN_POINTS:
        return {"points": int(len(xs))}
    dst = np.c_[xs, ys].astype(np.float32)
    src = dst + flow[ys, xs]
    out: dict[str, Any] = {"points": int(len(xs)), "speed_px": float(np.median(np.hypot(*flow[ys, xs].T)))}
    sim, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_PX)
    if sim is not None:
        out["similarity"] = np.vstack([sim, [0, 0, 1]])
        out["similarity_inliers"] = float(inl.mean())
    hom, inl = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
    if hom is not None and abs(np.linalg.det(hom)) > 1e-8:
        out["homography"] = hom
        out["homography_inliers"] = float(inl.mean())
    return out


def frame_motion(prev: np.ndarray, cur: np.ndarray, flow: np.ndarray, mask_prev: np.ndarray, mask_t: np.ndarray,
                 hands_prev: np.ndarray, fits: dict[str, Any]) -> dict[str, Any]:
    """PSNR inside ``mask_t`` of the previous frame warped by each model, and the homography's error split."""
    ys, xs = np.nonzero(mask_t)
    pts = np.c_[xs, ys].astype(np.float64)
    target = cur[ys, xs].astype(np.float64)
    count = len(pts)
    row: dict[str, Any] = {"pixels": int(count), "points": fits.get("points", 0), "speed_px": fits.get("speed_px")}
    errors: dict[str, np.ndarray] = {}
    sources = {"copy": pts, "flow": pts + flow[ys, xs]}
    for name in ("similarity", "homography"):
        if name in fits:
            sources[name] = apply_h(np.linalg.inv(fits[name]), pts)
    for name, src in sources.items():
        err = ((sample_map(src, prev).astype(np.float64) - target) ** 2).sum(1)
        errors[name] = err
        row[f"psnr_{name}"] = psnr(float(err.sum()), count)
    if "homography" in errors:
        src = sources["homography"]
        tracked = sample_map(src, mask_prev, nearest=True)
        under_hand = sample_map(src, hands_prev, nearest=True) & ~tracked
        new = ~tracked & ~under_hand
        e_h, e_f = errors["homography"], errors["flow"]
        total = float(e_h.sum())
        blur = (fits.get("speed_px") or 0.0) >= BLUR_SPEED_PX
        tracked_dev = float(np.maximum(e_h[tracked] - e_f[tracked], 0).sum())
        tracked_app = float(np.minimum(e_h[tracked], e_f[tracked]).sum())
        row["error_split"] = {
            "sse": total, "new": float(e_h[new].sum()), "hand_occlusion": float(e_h[under_hand].sum()),
            "deformation": 0.0 if blur else tracked_dev, "appearance": 0.0 if blur else tracked_app,
            "motion_blur": tracked_dev + tracked_app if blur else 0.0,
        }
        row["new_share"] = float(new.mean())
        row["hand_occlusion_share"] = float(under_hand.mean())
    return row


@dataclass
class ReferenceChain:
    """A reference carried forward by chained homographies; refreshed where it stops holding."""

    threshold: float
    fps: float
    reference: int | None = None
    to_current: np.ndarray | None = None
    held: int = 0
    frames: int = 0
    refreshes: list[int] = field(default_factory=list)

    def start(self, t: int) -> None:
        self.reference, self.to_current = t, np.eye(3)
        self.refreshes.append(t)
        self.frames += 1

    def step(self, t: int, h_step: np.ndarray | None, frames: np.ndarray, mask_ref: Any, mask_t: np.ndarray) -> dict[str, Any]:
        """Advance to frame ``t``; ``mask_ref(r)`` gives the object's mask at frame ``r``."""
        if self.reference is None or self.to_current is None or h_step is None:
            self.start(t)
            return {"held": False, "refresh": True}
        chained = h_step @ self.to_current
        ys, xs = np.nonzero(mask_t)
        src = apply_h(np.linalg.inv(chained), np.c_[xs, ys].astype(np.float64))
        covered = sample_map(src, mask_ref(self.reference), nearest=True)
        uncovered = 1.0 - float(covered.mean()) if len(covered) else 1.0
        value = None
        if covered.any():
            pred = sample_map(src[covered], frames[self.reference]).astype(np.float64)
            value = psnr(float(((pred - frames[t][ys[covered], xs[covered]].astype(np.float64)) ** 2).sum()),
                         int(covered.sum()))
        holds = value is not None and value >= self.threshold and uncovered <= MAX_UNCOVERED
        if holds:
            self.to_current = chained
            self.held += 1
            self.frames += 1
            return {"held": True, "psnr": value, "uncovered": uncovered}
        self.start(t)
        return {"held": False, "refresh": True, "psnr": value, "uncovered": uncovered}

    def summary(self, last: int) -> dict[str, Any]:
        bounds = self.refreshes + [last + 1]
        lives = [(b - a) / self.fps for a, b in zip(bounds, bounds[1:])]
        return {"frames": self.frames, "held": self.held, "references": len(self.refreshes),
                "reference_life_s": [round(v, 4) for v in lives]}


# ----------------------------------------------------------------- bits (libaom inspect)

BLOCK_RE_PREFIX = "BLOCK_"


def unwrap_order_hints(hints: list[int], bits: int = 7) -> list[int]:
    """Display order from order hints modulo ``2**bits``: each the candidate nearest the previous."""
    period = 1 << bits
    out: list[int] = []
    for hint in hints:
        if not out:
            out.append(hint)
            continue
        base = out[-1] - (out[-1] % period)
        candidates = [base + hint + k * period for k in (-1, 0, 1)]
        out.append(min(candidates, key=lambda c: abs(c - out[-1])))
    return out


def parse_inspect(text: str) -> list[dict[str, Any]]:
    body = text.strip()
    if body.endswith("]"):
        body = body[:-1].rstrip()
    body = body.removesuffix("null").rstrip().rstrip(",")
    if not body.startswith("["):
        raise ValueError("inspect output does not start with '['")
    return [frame for frame in json.loads(body + "]") if frame]


def block_dims(size_map: dict[str, int]) -> dict[int, tuple[int, int]]:
    """Block-size enum -> (width, height) in mode-info units."""
    out = {}
    for name, code in size_map.items():
        w, h = name[len(BLOCK_RE_PREFIX):].lower().split("x")
        out[int(code)] = (int(w) // MI, int(h) // MI)
    return out


def bit_density(frame: dict[str, Any], dims: dict[int, tuple[int, int]]) -> tuple[np.ndarray, float]:
    """Bits per mode-info cell: each block's accounted bits spread evenly over its (clipped) area."""
    sizes = np.asarray(frame["blockSize"], dtype=np.int16)
    rows, cols = sizes.shape
    origin = np.zeros((rows, cols), np.float64)
    x = y = -1
    total = 0.0
    for sym in frame["symbols"]:
        if len(sym) == 2:
            x, y = sym
        else:
            bits = sym[1] / 8.0  # AOM_ACCT_BITRES = 3
            total += bits
            if 0 <= y < rows and 0 <= x < cols:
                origin[y, x] += bits
    density = np.zeros((rows, cols), np.float64)
    ys, xs = np.nonzero(origin)
    for code in np.unique(sizes[ys, xs]):
        w, h = dims[int(code)]
        sel = sizes[ys, xs] == code
        for yy, xx in zip(ys[sel], xs[sel]):
            y1, x1 = min(yy + h, rows), min(xx + w, cols)
            density[yy:y1, xx:x1] += origin[yy, xx] / ((y1 - yy) * (x1 - xx))
    return density, total


def mi_counts(labels: np.ndarray, parts: int) -> np.ndarray:
    """Per part, pixels of each MI x MI cell in that part (uint8, 0..16)."""
    h, w = labels.shape
    rows, cols = h // MI, w // MI
    lab = labels[: rows * MI, : cols * MI].reshape(rows, MI, cols, MI)
    return np.stack([(lab == p).sum((1, 3)) for p in range(parts)]).astype(np.uint8)


def stream_bits(inspect_bin: str, stream: Path, counts: dict[str, np.ndarray], frames: int) -> dict[str, Any]:
    """Bits per part (per mask set) of one stream, over the window's first ``frames`` display frames."""
    began = time.time()
    proc = subprocess.run([inspect_bin, "-a", "-bs", str(stream)], capture_output=True, text=True, check=True)
    decoded = parse_inspect(proc.stdout)
    order = unwrap_order_hints([int(f["orderHint"]) for f in decoded])
    dims = block_dims(decoded[0]["blockSizeMap"])
    per_set: dict[str, np.ndarray] = {name: np.zeros(len(PARTS)) for name in counts}
    per_frame: dict[str, list[list[float]]] = {name: [[0.0] * len(PARTS) for _ in range(frames)] for name in counts}
    accounted = outside = 0.0
    seen: Counter[int] = Counter()
    for frame, display in zip(decoded, order):
        density, total = bit_density(frame, dims)
        accounted += total
        seen[display] += 1
        if not 0 <= display < frames:
            outside += total
            continue
        for name, c in counts.items():
            split = (c[display].astype(np.float64) / (MI * MI) * density[None]).sum((1, 2))
            per_set[name] += split
            per_frame[name][display] = [round(float(v), 1) for v in split]
    return {
        "frames_decoded": len(decoded), "display_frames": sorted(seen), "displayed_twice": sorted(d for d, n in seen.items() if n > 1),
        "accounted_bits": round(accounted, 1), "outside_window_bits": round(outside, 1),
        "bits_by_part": {name: dict(zip(PARTS, (round(float(v), 1) for v in split))) for name, split in per_set.items()},
        "bits_by_frame": per_frame, "seconds": round(time.time() - began, 2),
    }


# ----------------------------------------------------------------- decoding (own process)

def decode_window(video: str, item: dict[str, Any], frames: int, archive: str, out: str, threads: int) -> dict[str, Any]:
    """Window frames as an RGB ``.npy`` at ``out``, with B2's sparse-JPEG gate (best within ±2 frames)."""
    from PIL import Image

    from experiments.visor.b2 import GATE_MARGIN, TIE, jpeg_targets

    began = time.time()
    first = int(item["first_video_index"])
    targets = jpeg_targets(item, Path(archive))
    gate = {t["index"] + d for t in targets for d in range(-GATE_MARGIN, GATE_MARGIN + 1) if t["index"] + d >= 0}
    array = np.lib.format.open_memmap(out, mode="w+", dtype=np.uint8, shape=(frames, *visor.FRAME_SIZE, 3))
    near: dict[int, np.ndarray] = {}
    expected = first
    for index, rgb in visor.decode_frames(video, sorted(set(range(first, first + frames)) | gate), threads=threads):
        if index in gate:
            near[index] = rgb
        if not first <= index < first + frames:
            continue
        if index != expected:
            raise RuntimeError(f"{item['id']}: decoded frame {index}, expected {expected}")
        array[index - first] = rgb
        expected += 1
    array.flush()
    if expected != first + frames:
        raise RuntimeError(f"{item['id']}: decoded {expected - first} of {frames} frames")
    checks = []
    for target in targets:
        released = np.asarray(Image.open(Path(archive) / target["jpeg"]).convert("RGB")).astype(np.int16)
        mae = {i: float(np.abs(near[i].astype(np.int16) - released).mean()) for i in near
               if abs(i - target["index"]) <= GATE_MARGIN}
        best = min(mae, key=lambda i: mae[i])
        checks.append({"jpeg": target["jpeg"], "index": target["index"], "best_index": best,
                       "rule_mae": round(mae[target["index"]], 3), "best_mae": round(mae[best], 3),
                       "holds": mae[target["index"]] - mae[best] <= TIE and mae[best] < MATCH})
    return {"frames": frames, "jpeg_gate": checks, "seconds": round(time.time() - began, 2)}


# ----------------------------------------------------------------- per item

def contact_objects(item: dict[str, Any], sparse: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
    """Objects a human labelled in contact with a hand at a keyframe bounding or inside the window."""
    from experiments.visor.b1b import plan_item

    plan = plan_item(item, sparse, mapping, shape=(8, 8))
    human = visor.frames(sparse)
    handled: set[str] = set()
    contacts = []
    for number in sorted(plan.keyframes):
        annotations = human[number].annotations
        names = {a["id"]: a["name"] for a in annotations}
        for a in annotations:
            if visor.native_class(a) in visor.HANDS and a.get("in_contact_object") in names:
                target = next(x for x in annotations if x["id"] == a["in_contact_object"])
                if visor.native_class(target) not in visor.HANDS:
                    key = visor.object_key(visor.native_class(target), str(target["name"]))
                    handled.add(key)
                    contacts.append([number, visor.native_class(a), key])
    return {"keyframes": sorted(plan.keyframes), "handled": sorted(handled), "contacts": contacts}


class WiLoR:
    """WiLoR (detector + MANO regressor) loaded as the environment audit does."""

    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from ultralytics import YOLO

        from experiments.audit.env_smoke import _hand_model

        ns = argparse.Namespace(wilor_checkpoint=args.wilor_checkpoint, mano_dir=args.mano_dir)
        began = time.time()
        self.model, self.cfg, self.dataset_cls, self.load = _hand_model("wilor", ns)
        self.detector = YOLO(args.wilor_detector)
        self.load["detector_sha256"] = file_sha256(Path(args.wilor_detector))
        self.load["seconds"] = round(time.time() - began, 2)
        self.faces = np.asarray(self.model.mano.faces, dtype=np.int32)
        self.torch = torch
        self.kernels: dict[str, Any] = {}

    def detect(self, frames: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        results = self.detector.predict([np.ascontiguousarray(f[..., ::-1]) for f in frames], conf=DETECTOR_CONF,
                                        device=0, verbose=False)
        out = []
        for r in results:
            if r.boxes is None or not len(r.boxes):
                out.append((np.zeros((0, 4)), np.zeros(0)))
            else:
                out.append((r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy()))
        return out

    def fit(self, frame: np.ndarray, boxes: np.ndarray, right: np.ndarray, profile: bool = False) -> list[dict[str, Any]]:
        torch = self.torch
        from torch.utils.data import default_collate

        dataset = self.dataset_cls(self.cfg, np.ascontiguousarray(frame[..., ::-1]), boxes.astype(np.float32),
                                   right.astype(np.float32), rescale_factor=RESCALE)
        batch = default_collate([dataset[i] for i in range(len(dataset))])
        batch = {k: (v.cuda() if hasattr(v, "cuda") else v) for k, v in batch.items()}
        with torch.no_grad():
            if profile:
                from experiments.audit.env_smoke import profile_cuda

                out, self.kernels = profile_cuda(lambda: self.model(batch))
            else:
                out = self.model(batch)
        multiplier = 2 * batch["right"] - 1
        cam = out["pred_cam"].clone()
        cam[:, 1] = multiplier * cam[:, 1]
        img_size = batch["img_size"].float()
        focal = self.cfg.EXTRA.FOCAL_LENGTH / self.cfg.MODEL.IMAGE_SIZE * img_size.max()
        cam_full = cam_crop_to_full(cam, batch["box_center"].float(), batch["box_size"].float(), img_size, focal)
        params = out["pred_mano_params"]
        rows = []
        for n in range(len(boxes)):
            m = float(multiplier[n])
            verts = out["pred_vertices"][n].float().cpu().numpy().copy()
            joints = out["pred_keypoints_3d"][n].float().cpu().numpy().copy()
            verts[:, 0] *= m
            joints[:, 0] *= m
            rows.append({"verts3d": verts, "joints3d": joints, "cam": cam_full[n].float().cpu().numpy(),
                         "focal": float(focal), "global_orient": params["global_orient"][n].reshape(3, 3).float().cpu().numpy(),
                         "hand_pose": params["hand_pose"][n].reshape(15, 3, 3).float().cpu().numpy(),
                         "betas": params["betas"][n].float().cpu().numpy()})
        return rows

    def mano_joints(self, global_orient: np.ndarray, hand_pose: np.ndarray, betas: np.ndarray) -> np.ndarray:
        """MANO joints (right-hand convention) for batches of rotation matrices."""
        torch = self.torch
        with torch.no_grad():
            out = self.model.mano(global_orient=torch.as_tensor(global_orient, dtype=torch.float32).cuda().reshape(-1, 1, 3, 3),
                                  hand_pose=torch.as_tensor(hand_pose, dtype=torch.float32).cuda().reshape(-1, 15, 3, 3),
                                  betas=torch.as_tensor(betas, dtype=torch.float32).cuda().reshape(-1, 10), pose2rot=False)
        return out.joints.float().cpu().numpy()


def cam_crop_to_full(cam: Any, center: Any, size: Any, img_size: Any, focal: Any) -> Any:
    """WiLoR's crop camera (s, tx, ty) as a translation in the full image (``wilor.utils.renderer``;
    copied because that module imports pyrender, which needs OpenGL)."""
    import torch

    w_2, h_2 = img_size[:, 0] / 2.0, img_size[:, 1] / 2.0
    bs = size * cam[:, 0] + 1e-9
    return torch.stack([2 * (center[:, 0] - w_2) / bs + cam[:, 1], 2 * (center[:, 1] - h_2) / bs + cam[:, 2],
                        2 * focal / bs], dim=-1)


def project(points: np.ndarray, cam: np.ndarray, focal: float, shape: tuple[int, int]) -> np.ndarray:
    p = points + cam
    return focal * p[:, :2] / p[:, 2:3] + np.array([shape[1] / 2.0, shape[0] / 2.0])


def laplacian_var(gray: np.ndarray, box: list[float]) -> float:
    import cv2

    x0, y0, x1, y1 = (int(round(v)) for v in box)
    crop = gray[max(y0, 0):max(y1, 1), max(x0, 0):max(x1, 1)]
    return float(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size > 16 else 0.0


def hand_parameters(fit: dict[str, Any], shape: tuple[int, int]) -> dict[str, Any]:
    """The values sent per frame: axis-angle pose and orientation, camera as origin pixel and log depth."""
    from scipy.spatial.transform import Rotation

    tx, ty, tz = (float(v) for v in fit["cam"])
    f = fit["focal"]
    return {"global_orient": Rotation.from_matrix(fit["global_orient"]).as_rotvec(),
            "hand_pose": Rotation.from_matrix(fit["hand_pose"]).as_rotvec(),
            "camera": np.array([f * tx / tz + shape[1] / 2.0, f * ty / tz + shape[0] / 2.0, math.log(tz)]),
            "betas": fit["betas"]}


def quantized_symbols(params: dict[str, Any]) -> dict[str, list[int]]:
    step = math.radians(QUANT["angle_deg"])
    return {"global_orient": [int(round(v / step)) for v in params["global_orient"]],
            "hand_pose": [int(round(v / step)) for v in params["hand_pose"].reshape(-1)],
            "camera": [int(round(params["camera"][0] / QUANT["position_px"])),
                       int(round(params["camera"][1] / QUANT["position_px"])),
                       int(round(params["camera"][2] / QUANT["log_depth"]))]}


def dequantize(symbols: dict[str, list[int]], focal: float, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.spatial.transform import Rotation

    step = math.radians(QUANT["angle_deg"])
    orient = Rotation.from_rotvec(np.array(symbols["global_orient"]) * step).as_matrix()
    pose = Rotation.from_rotvec(np.array(symbols["hand_pose"]).reshape(15, 3) * step).as_matrix()
    u = symbols["camera"][0] * QUANT["position_px"]
    v = symbols["camera"][1] * QUANT["position_px"]
    tz = math.exp(symbols["camera"][2] * QUANT["log_depth"])
    cam = np.array([(u - shape[1] / 2.0) * tz / focal, (v - shape[0] / 2.0) * tz / focal, tz])
    return orient, pose, cam


def corner_symbols(h: np.ndarray, box: tuple[float, float, float, float]) -> list[int]:
    corners = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]], float)
    moved = apply_h(h, corners)
    return [int(round(v / QUANT["corner_px"])) for v in (moved - corners).reshape(-1)]


def review_frames(item_id: str, frames: int, count: int) -> list[int]:
    picks: list[int] = []
    salt = 0
    while len(picks) < min(count, frames - 1):
        digest = hashlib.sha256(f"{REVIEW_SEED}:{item_id}:{salt}".encode()).hexdigest()
        pick = 1 + int(digest, 16) % (frames - 1)
        if pick not in picks:
            picks.append(pick)
        salt += 1
    return sorted(picks)


def process_item(args: argparse.Namespace, wilor: WiLoR, item: dict[str, Any], frames_path: Path, clips: dict[str, ClipMasks],
                 contact: dict[str, Any], streams: dict[str, Path], publish: Path, profile: bool) -> dict[str, Any]:
    import cv2

    frames = np.load(frames_path, mmap_mode="r")
    count = int(args.frames)
    shape = visor.FRAME_SIZE
    fps = float(item["fps"])
    handled = set(contact["handled"])
    timing: dict[str, float] = {}

    # Hands: detection and MANO fits.
    began = time.time()
    fits: list[dict[str, dict[str, Any]]] = [{} for _ in range(count)]
    hand_rows: list[dict[str, Any]] = []
    for start in range(0, count, 16):
        chunk = list(range(start, min(start + 16, count)))
        detections = wilor.detect([frames[t] for t in chunk])
        for t, (boxes, classes) in zip(chunk, detections):
            hands = {side: clips[FILL].class_mask(t, side) for side in visor.HANDS}
            hands = {s: m for s, m in hands.items() if m.any()}
            matched = match_detections(hands, boxes)
            for side in hands:
                if side not in matched:
                    hand_rows.append({"t": t, "side": side, "fitted": False, "mask_px": int(hands[side].sum())})
            if not matched:
                continue
            order = sorted(matched)
            rows = wilor.fit(frames[t], boxes[[matched[s] for s in order]],
                             np.array([1.0 if s == "right hand" else 0.0 for s in order]), profile=profile and not wilor.kernels)
            for side, fitted_hand in zip(order, rows):
                fitted_hand["box"] = [float(v) for v in boxes[matched[side]]]
                fitted_hand["detector_right"] = bool(classes[matched[side]] > 0.5)
                fits[t][side] = fitted_hand
    timing["hands"] = round(time.time() - began, 2)

    # Parts, hand metrics and per-MI part counts for the bits.
    began = time.time()
    labels_mi: dict[str, list[np.ndarray]] = {DENSE: [], FILL: []}
    composition: list[dict[str, Any]] = []
    split_masks: list[dict[str, np.ndarray]] = []
    object_masks: list[dict[str, np.ndarray]] = []
    hand_union: list[np.ndarray] = []
    grays: list[np.ndarray] = []
    for t in range(count):
        frame = frames[t]
        gray = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2GRAY)
        grays.append(gray)
        parts: dict[str, np.ndarray] = {}
        union = np.zeros(shape, bool)
        for side in visor.HANDS:
            mask = clips[FILL].class_mask(t, side)
            union |= mask
            if not mask.any():
                continue
            fit = fits[t].get(side)
            if fit is None:
                parts[f"hand_no_fit:{side}"] = mask
                continue
            verts = project(fit["verts3d"], fit["cam"], fit["focal"], shape)
            joints2d = project(fit["joints3d"], fit["cam"], fit["focal"], shape)
            silhouette = rasterize(verts, wilor.faces, shape)
            hand, forearm = wrist_split(mask, joints2d[0], joints2d[9])
            parts[f"hand:{side}"], parts[f"forearm:{side}"] = hand, forearm
            fit["joints2d"] = joints2d
            fit["verts2d"] = verts
            box = fit["box"]
            bw, bh = box[2] - box[0], box[3] - box[1]
            objects_here = np.zeros(shape, bool)
            for inst in clips[FILL].frames[t]:
                if inst.class_name not in visor.HANDS:
                    objects_here |= inst.mask()
            x0, y0, x1, y1 = (int(round(v)) for v in box)
            box_px = max((min(y1, shape[0]) - max(y0, 0)) * (min(x1, shape[1]) - max(x0, 0)), 1)
            row: dict[str, Any] = {
                "t": t, "side": side, "fitted": True, "mask_px": int(mask.sum()), "box": [round(v, 1) for v in box],
                "detector_side_agrees": fit["detector_right"] == (side == "right hand"),
                "iou_mask": iou(silhouette, mask), "iou_hand": iou(silhouette, hand),
                "forearm_share": float(forearm.sum()) / float(mask.sum()),
                "render_outside_mask": float((silhouette & ~mask).sum()) / max(int(silhouette.sum()), 1),
                "hand_recall": float((silhouette & hand).sum()) / max(int(hand.sum()), 1),
                "edge": x0 <= EDGE_PX or y0 <= EDGE_PX or x1 >= shape[1] - EDGE_PX or y1 >= shape[0] - EDGE_PX,
                "occlusion_share": float(objects_here[max(y0, 0):y1, max(x0, 0):x1].sum()) / box_px,
                "sharpness": laplacian_var(gray, box), "box_size": float(max(bw, bh)),
                "joints2d": np.round(joints2d, 2).tolist(),
                "joints3d_mm": np.round((fit["joints3d"] - fit["joints3d"][0]) * 1000.0, 2).tolist(),
            }
            params = hand_parameters(fit, shape)
            row["symbols"] = quantized_symbols(params)
            row["betas"] = np.round(params["betas"], 4).tolist()
            row["global_orient_matrix"] = np.round(fit["global_orient"], 6).tolist()
            hand_rows.append(row)
        hand_union.append(union)
        objects: dict[str, np.ndarray] = {}
        for inst in clips[FILL].frames[t]:
            if inst.class_name in visor.HANDS:
                continue
            key = visor.object_key(inst.class_name, inst.label)
            objects[key] = objects.get(key, np.zeros(shape, bool)) | inst.mask()
        object_masks.append(objects)
        split_masks.append(parts)
        for name in (DENSE, FILL):
            label = np.zeros(shape, np.uint8)
            if name == FILL:
                objs = objects
            else:
                objs = {}
                for inst in clips[DENSE].frames[t]:
                    if inst.class_name not in visor.HANDS:
                        key = visor.object_key(inst.class_name, inst.label)
                        objs[key] = objs.get(key, np.zeros(shape, bool)) | inst.mask()
            for key, mask in objs.items():
                code = PARTS.index("handled_object" if key in handled else "other_object")
                label[mask & (label == 0)] = code
            # Hands are painted last so that they win: the same in both sets.
            for part_key, mask in parts.items():
                label[mask] = PARTS.index(part_key.split(":")[0])
            labels_mi[name].append(mi_counts(label, len(PARTS)))
            shares = np.bincount(label.reshape(-1), minlength=len(PARTS)) / label.size
            composition.append({"t": t, "set": name, "frame_share": dict(zip(PARTS, (round(float(v), 6) for v in shares)))})
    timing["parts"] = round(time.time() - began, 2)

    # Stability of the fitted hands over consecutive frames.
    stability = hand_stability(hand_rows)

    # Quantization: projected joints from the quantized parameters against the fitted ones.
    began = time.time()
    fitted_rows = [r for r in hand_rows if r["fitted"]]
    quant_error = []
    if fitted_rows:
        orients, poses, cams = [], [], []
        for r in fitted_rows:
            orient, pose, cam = dequantize(r["symbols"], fits[r["t"]][r["side"]]["focal"], shape)
            orients.append(orient)
            poses.append(pose)
            cams.append(cam)
        betas = np.array([np.round(np.array(r["betas"]) / QUANT["betas"]) * QUANT["betas"] for r in fitted_rows])
        joints = wilor.mano_joints(np.array(orients), np.array(poses), betas)
        for r, j, cam in zip(fitted_rows, joints, cams):
            hand_fit = fits[r["t"]][r["side"]]
            if r["side"] == "left hand":
                j = j.copy()
                j[:, 0] *= -1
            q = project(j, cam, hand_fit["focal"], shape)
            err = float(np.linalg.norm(q - hand_fit["joints2d"], axis=1).mean())
            r["quantization_joint_error_px"] = round(err, 4)
            quant_error.append(err)
    timing["quantization"] = round(time.time() - began, 2)

    # Motion: flow, rigid fits, reference chains, for objects and the hand parts.
    began = time.time()
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)  # type: ignore[attr-defined]
    halves = [cv2.resize(g, (shape[1] // 2, shape[0] // 2), interpolation=cv2.INTER_AREA) for g in grays]
    tracks: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    for t in range(count):
        for key, mask in object_masks[t].items():
            tracks[key][t] = mask
        for key, mask in split_masks[t].items():
            if key.startswith(("hand:", "forearm:")) and mask.any():
                tracks[key][t] = mask
    chains = {key: [ReferenceChain(th, fps) for th in HOLD_PSNR] for key in tracks}
    motion_rows: list[dict[str, Any]] = []
    object_symbols: dict[str, list[list[int]]] = defaultdict(list)
    review_flows: dict[int, np.ndarray] = {}
    review_at = set(review_frames(item["id"], count, args.review))
    reviewed_fits: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for t in range(count):
        flow = None
        if t > 0:
            small = dis.calc(halves[t], halves[t - 1], None)
            flow = cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR) * 2.0
            if t in review_at:
                review_flows[t] = flow
        for key, masks in tracks.items():
            if t not in masks:
                continue
            mask_t = masks[t]
            if not mask_t.any():
                continue
            fitted: dict[str, Any] = {}
            motion_row: dict[str, Any] = {"t": t, "track": key}
            if flow is not None and t - 1 in masks:
                fitted = fit_models(flow, mask_t)
                motion_row.update(frame_motion(frames[t - 1], frames[t], flow, masks[t - 1], mask_t, hand_union[t - 1], fitted))
                if "homography" in fitted:
                    box = box_of(masks[t - 1])
                    if box is not None:
                        object_symbols[key].append(corner_symbols(fitted["homography"], box))
                if t in review_at:
                    reviewed_fits[t][key] = fitted
            h_step = fitted.get("homography") if (flow is not None and t - 1 in masks) else None
            for chain in chains[key]:
                if chain.reference is None or t - 1 not in masks:
                    chain.start(t)
                    motion_row[f"held_{chain.threshold:g}"] = False
                else:
                    step = chain.step(t, h_step, frames, lambda r, m=masks: m[r], mask_t)
                    motion_row[f"held_{chain.threshold:g}"] = step["held"]
                    if chain.threshold == HOLD_PSNR[0]:
                        motion_row["reference_psnr"] = step.get("psnr")
                        motion_row["reference_uncovered"] = step.get("uncovered")
            motion_rows.append(motion_row)
    track_info = {key: {"kind": track_kind(key, handled), "frames": len(masks),
                        "mean_area": float(np.mean([m.sum() for m in masks.values()])),
                        "chains": {f"{c.threshold:g}": c.summary(max(masks)) for c in chains[key]},
                        "corner_symbols": object_symbols.get(key, [])}
                  for key, masks in tracks.items()}
    timing["motion"] = round(time.time() - began, 2)

    # Bits of B2's SVT-AV1 streams per part.
    began = time.time()
    counts = {name: np.stack(v) for name, v in labels_mi.items()}
    bits = {point: stream_bits(args.inspect, path, counts, count) for point, path in sorted(streams.items())}
    timing["bits"] = round(time.time() - began, 2)

    # Review overlays.
    began = time.time()
    reviews = []
    worst = min((r for r in hand_rows if r["fitted"] and r["t"] > 0 and r["iou_hand"] is not None),
                key=lambda r: r["iou_hand"], default=None)
    picks = [(t, "content-blind") for t in sorted(review_at)]
    if worst is not None and worst["t"] not in review_at:
        picks.append((worst["t"], f"worst hand-side IoU ({worst['side']}, {worst['iou_hand']:.2f})"))
    for t, why in picks:
        flow = review_flows.get(t)
        if flow is None:
            small = dis.calc(halves[t], halves[t - 1], None)
            flow = cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR) * 2.0
        fitted_here = reviewed_fits.get(t) or {k: fit_models(flow, m[t]) for k, m in tracks.items() if t in m and t - 1 in m}
        name = f"{item['id']}_t{t:03d}.jpg"
        draw_review(frames[t - 1], frames[t], split_masks[t], object_masks[t], handled, fits[t], wilor.faces,
                    {k: v for k, v in fitted_here.items() if not k.startswith(("hand:", "hand_no_fit:"))}, tracks, t, why,
                    publish / "review" / name)
        reviews.append({"file": f"review/{name}", "t": t, "why": why})
    timing["review"] = round(time.time() - began, 2)

    return {
        "id": item["id"], "video": item["video"], "fps": fps, "frames": count,
        "video_type": "EK-100" if abs(fps - 50) < 0.01 else "EK-55",
        "contact": contact, "composition": composition, "hands": hand_rows, "stability": stability,
        "quantization": {"steps": QUANT, "joint_error_px_mean": round(float(np.mean(quant_error)), 4) if quant_error else None,
                         "joint_error_px_p95": round(float(np.percentile(quant_error, 95)), 4) if quant_error else None},
        "motion": motion_rows, "tracks": track_info, "bits": bits, "review": reviews, "timing": timing,
    }


def track_kind(key: str, handled: set[str]) -> str:
    if key.startswith("hand:"):
        return "hand"
    if key.startswith("forearm:"):
        return "forearm"
    return "handled_object" if key in handled else "other_object"


def hand_stability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per side, over runs of consecutive fitted frames: 3D/2D joint acceleration and orientation change."""
    out: dict[str, Any] = {}
    for side in visor.HANDS:
        by_t = {r["t"]: r for r in rows if r["side"] == side and r["fitted"]}
        acc3, acc2, rot = [], [], []
        for t, r in by_t.items():
            if t - 1 in by_t:
                rot.append(rotation_angle_deg(np.array(by_t[t - 1]["global_orient_matrix"]), np.array(r["global_orient_matrix"])))
            if t - 1 in by_t and t + 1 in by_t:
                j = [np.array(by_t[s]["joints3d_mm"]) for s in (t - 1, t, t + 1)]
                acc3.append(float(np.linalg.norm(j[2] - 2 * j[1] + j[0], axis=1).mean()))
                k = [np.array(by_t[s]["joints2d"]) for s in (t - 1, t, t + 1)]
                acc2.append(float(np.linalg.norm(k[2] - 2 * k[1] + k[0], axis=1).mean()) / max(r["box_size"], 1.0))
        out[side] = {
            "pairs": len(rot), "triples": len(acc3),
            "joint_accel_mm_median": round(float(np.median(acc3)), 3) if acc3 else None,
            "joint_accel_mm_mean": round(float(np.mean(acc3)), 3) if acc3 else None,
            "kp2d_accel_box_median": round(float(np.median(acc2)), 4) if acc2 else None,
            "orient_change_deg_median": round(float(np.median(rot)), 3) if rot else None,
            "orient_flips_45": int(sum(v > 45 for v in rot)),
        }
    return out


# ----------------------------------------------------------------- review drawing

def shaded_mesh(canvas: np.ndarray, verts2d: np.ndarray, verts3d: np.ndarray, faces: np.ndarray, colour: tuple[int, int, int]) -> None:
    """Painter's algorithm, flat shading by the face normal's z (camera looks along +z)."""
    import cv2

    tri3 = verts3d[faces]
    normal = np.cross(tri3[:, 1] - tri3[:, 0], tri3[:, 2] - tri3[:, 0])
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12
    shade = 0.35 + 0.65 * np.abs(normal[:, 2])
    order = np.argsort(-tri3[:, :, 2].mean(1))
    pts = np.round(verts2d[faces] * 16).astype(np.int32)
    for i in order:
        c = tuple(int(v * shade[i]) for v in colour)
        cv2.fillConvexPoly(canvas, pts[i], c, lineType=cv2.LINE_AA, shift=4)


def draw_review(prev: np.ndarray, cur: np.ndarray, parts: dict[str, np.ndarray], objects: dict[str, np.ndarray],
                handled: set[str], fits: dict[str, dict[str, Any]], faces: np.ndarray, object_fits: dict[str, dict[str, Any]],
                tracks: dict[str, dict[int, np.ndarray]], t: int, why: str, path: Path) -> None:
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    shape = cur.shape[:2]
    # 1. Parts.
    tinted = cur.astype(np.float32).copy()
    layers = [(m, COLOURS["handled_object" if k in handled else "other_object"], k.split("|")[-1]) for k, m in objects.items()]
    layers += [(m, COLOURS[k.split(":")[0]], k.split(":")[0]) for k, m in parts.items()]
    for mask, colour, _ in layers:
        tinted[mask] = 0.5 * tinted[mask] + 0.5 * np.array(colour, np.float32)
    panel1 = tinted.astype(np.uint8)
    for mask, colour, text in layers:
        box = box_of(mask)
        if box is not None:
            cv2.putText(panel1, text[:24], (int(box[0]) + 6, int(box[1]) + 36), font, 1.2, (255, 255, 255), 3, cv2.LINE_AA)
    for side, fit in fits.items():
        if "joints2d" in fit:
            w, k = fit["joints2d"][0], fit["joints2d"][9]
            axis = (k - w) / (np.linalg.norm(k - w) + 1e-9)
            normal = np.array([-axis[1], axis[0]])
            a, b = w - 400 * normal, w + 400 * normal
            cv2.line(panel1, tuple(int(v) for v in a), tuple(int(v) for v in b), (255, 255, 255), 4, cv2.LINE_AA)
    # 2. MANO.
    panel2 = cur.copy()
    for side, fit in fits.items():
        if "verts2d" in fit:
            shaded_mesh(panel2, fit["verts2d"], fit["verts3d"] + fit["cam"], faces,
                        (120, 140, 255) if side == "right hand" else (255, 120, 140))
    for key, mask in parts.items():
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel2, contours, -1, COLOURS[key.split(":")[0]], 4)
    # 3. Objects warped from t-1 by their homography; 4. absolute error there (x4).
    panel3 = cur.copy()
    panel4 = np.zeros_like(cur)
    for key, fitted in object_fits.items():
        mask = tracks[key][t]
        if "homography" not in fitted:
            continue
        ys, xs = np.nonzero(mask)
        src = apply_h(np.linalg.inv(fitted["homography"]), np.c_[xs, ys].astype(np.float64))
        pred = sample_map(src, prev).astype(np.uint8)
        panel3[ys, xs] = pred
        panel4[ys, xs] = np.clip(np.abs(pred.astype(np.int16) - cur[ys, xs].astype(np.int16)) * 4, 0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel3, contours, -1, (255, 255, 255), 2)
        err = ((pred.astype(np.float64) - cur[ys, xs].astype(np.float64)) ** 2).sum()
        value = psnr(float(err), len(ys))
        box = box_of(mask)
        if box is not None and value is not None:
            cv2.putText(panel3, f"{key.split('|')[-1][:18]} {value:.1f} dB", (int(box[0]) + 6, int(box[1]) + 36), font, 1.1,
                        (255, 255, 255), 3, cv2.LINE_AA)
    titles = [f"Parts (t={t}; {why})", "WiLoR MANO + VISOR hand outline", "Objects: homography from t-1", "|error| x4 inside objects"]
    tiles = []
    for panel, title in zip((panel1, panel2, panel3, panel4), titles):
        cv2.putText(panel, title, (24, 64), font, 1.7, (255, 255, 255), 5, cv2.LINE_AA)
        tiles.append(cv2.resize(panel, (shape[1] // 2, shape[0] // 2), interpolation=cv2.INTER_AREA))
    grid = np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), grid[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])


# ----------------------------------------------------------------- run command

def named(pairs: list[list[str]] | None) -> dict[str, str]:
    return {k: v for k, v in (pairs or [])}


def stream_paths(args: argparse.Namespace, item_id: str) -> tuple[dict[str, Path], dict[str, Any]]:
    """B2's SVT-AV1 streams of the item, each checked against the sha256 its encoding job recorded."""
    from experiments.visor.b2 import stream_root

    records = {}
    for result in args.svt_result:
        for row in json.loads(Path(result).read_text())["items"]:
            if row["id"] == item_id:
                records = {f"{float(p['point']):g}": p for p in row["points"] if not p.get("variant")}
    out, info = {}, {}
    for directory in args.svt_streams:
        root = stream_root(Path(directory)) / "svtav1" / item_id
        for point in sorted(records):
            path = root / point / "stream.ivf"
            if path.is_file():
                out[point] = path
                info[point] = {"sha256": file_sha256(path), "recorded_sha256": records[point]["stream_sha256"],
                               "rate_bytes": records[point]["rate_bytes"], "kbps": records[point]["kbps"]}
    return out, info


def command_run(args: argparse.Namespace) -> int:
    from experiments.visor.b1b import load_inputs
    from experiments.visor.b2 import load_mask_sets, select_items

    eval_set = json.loads(Path(args.eval_set).read_text())
    items = select_items(eval_set, args.items)
    archive = Path(args.archive)
    videos = {Path(v).stem: v for v in args.video}
    masks, records = named(args.masks), named(args.mask_record)
    if set(masks) != {DENSE, FILL}:
        raise SystemExit(f"--masks must name {DENSE} and {FILL}")
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    # MANO loaders want one directory; staged files keep their cache names.
    mano = scratch / "mano"
    mano.mkdir(parents=True, exist_ok=True)
    for name, source in (("MANO_LEFT.pkl", args.mano_left), ("MANO_RIGHT.pkl", args.mano_right)):
        if not (mano / name).exists():
            (mano / name).symlink_to(Path(source).resolve())
    args.mano_dir = str(mano)
    # The staging cache keeps files read-only and drops the execute bit: run a copy.
    inspect = scratch / "bin" / Path(args.inspect).name
    inspect.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.inspect, inspect)
    inspect.chmod(0o755)
    args.inspect_source = args.inspect
    args.inspect = str(inspect)
    allowance = max(1, int(os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 1))
    # Decode in fresh processes before torch loads (B2: PyAV stalled in a process that had loaded torchvision).
    pool = ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"))

    def submit(item: dict[str, Any]) -> Any:
        target = scratch / "frames" / f"{item['id']}.npy"
        target.parent.mkdir(parents=True, exist_ok=True)
        return pool.submit(decode_window, videos[item["video"]], item, int(args.frames), str(archive), str(target),
                           max(1, allowance // 2))

    pending = {i: submit(items[i]) for i in range(min(2, len(items)))}
    import cv2
    import torch

    from experiments.audit.env_smoke import device_record, model_on_cuda

    cv2.setNumThreads(max(1, allowance))
    device = device_record(torch.cuda.get_device_name(0))
    wilor = WiLoR(args)
    rows = []
    for number, item in enumerate(items):
        decoded = pending.pop(number).result()
        if number + 2 < len(items):
            pending[number + 2] = submit(items[number + 2])
        clips, mask_info = load_mask_sets(item, int(args.frames), masks, records, archive)
        _, sparse, mapping = load_inputs(item, archive)
        contact = contact_objects(item, sparse, mapping)
        streams, stream_info = stream_paths(args, item["id"])
        frames_path = scratch / "frames" / f"{item['id']}.npy"
        row = process_item(args, wilor, item, frames_path, clips, contact, streams, publish, profile=number == 0)
        row["decode"] = decoded
        row["mask_sets"] = mask_info
        row["streams"] = stream_info
        frames_path.unlink()
        rows.append(row)
        write_json(publish / "items" / f"{item['id']}.json", row)
        progress(len(rows))
    pool.shutdown()
    write_json(stage_dir() / "h1.json", {
        "eval_set": {"path": args.eval_set, "sha256": file_sha256(Path(args.eval_set)), "name": eval_set["name"]},
        "inputs": {"wilor_checkpoint_sha256": wilor.load["checkpoint_sha256"], "wilor_detector_sha256": wilor.load["detector_sha256"],
                   "mano_left_sha256": file_sha256(Path(args.mano_left)), "mano_right_sha256": file_sha256(Path(args.mano_right)),
                   "inspect": args.inspect_source, "inspect_sha256": file_sha256(Path(args.inspect)),
                   "svt_result_sha256": [file_sha256(Path(p)) for p in args.svt_result]},
        "wilor": {**wilor.load, "kernels": wilor.kernels, "on_cuda": model_on_cuda(wilor.model)},
        "device": device, "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "settings": {"frames": int(args.frames), "review": args.review, "parts": PARTS, "quantization": QUANT,
                     "hold_psnr": HOLD_PSNR, "max_uncovered": MAX_UNCOVERED, "decision": DECISION,
                     "detector_conf": DETECTOR_CONF, "rescale": RESCALE, "box_inside": BOX_INSIDE,
                     "blur_speed_px": BLUR_SPEED_PX, "ransac_px": RANSAC_PX, "sample_stride": SAMPLE_STRIDE},
        "items": [{k: v for k, v in row.items() if k not in ("composition", "hands", "motion")} | {"published": f"items/{row['id']}.json"}
                  for row in rows],
    })
    return 0


# ----------------------------------------------------------------- validate

def validate_result(stage: Path) -> dict[str, bool]:
    import tarfile

    result = json.loads((stage / "h1.json").read_text())
    with tarfile.open(stage / "published.tar") as tar:
        members = {m.name: m for m in tar.getmembers()}
        full = []
        for row in result["items"]:
            handle = tar.extractfile(members[f"publish/{row['published']}"])
            full.append(json.loads(handle.read()) if handle else {})
        reviews = [m for m in members if m.startswith("publish/review/") and m.endswith(".jpg")]
    rows = result["items"]
    wilor = result["wilor"]
    hands = [h for row in full for h in row.get("hands", [])]
    fitted = [h for h in hands if h["fitted"]]
    ious = [h[k] for h in fitted for k in ("iou_mask", "iou_hand") if h[k] is not None]
    comps = [c for row in full for c in row.get("composition", [])]
    streams = [s for row in rows for s in row["streams"].values()]
    bits = [(b, row["streams"][p]) for row in rows for p, b in row["bits"].items()]
    motion = [m for row in full for m in row.get("motion", [])]
    scored = [m for m in motion if m.get("psnr_homography") is not None]
    # The dense flow explains at least as much as no motion, up to its own interpolation error.
    large = [m for m in motion if m.get("psnr_flow") is not None and m.get("psnr_copy") is not None and m["pixels"] > 5000]
    return {
        "items_processed": bool(rows) and len(full) == len(rows),
        "wilor_on_cuda": bool(wilor["on_cuda"]) and wilor["kernels"].get("kernel_launches", 0) > 0,
        "wilor_weights_load_completely": wilor["missing_count"] == 0 and wilor["unexpected_count"] == 0,
        "pinned_wilor_weights": result["inputs"]["wilor_checkpoint_sha256"] == "3e97aafc7dd08d883a4cc5a027df61fdb6fda6136dbd1319405413862ada6bb2"
        and result["inputs"]["wilor_detector_sha256"] == "5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033",
        "pinned_inspect": result["inputs"]["inspect_sha256"] == "9b2e75a288309a6d91c9dcd2976f30425923a8f6733e74dd65475f7f4e443959",
        "sparse_jpegs_match_decoded_frames": all(g["holds"] for row in rows for g in row["decode"]["jpeg_gate"]),
        "mask_sets_match_records": all(info["masks_rle_sha256"] == info["record_masks_rle_sha256"]
                                       for row in rows for info in row["mask_sets"].values()),
        "streams_match_b2_records": bool(streams) and all(s["sha256"] == s["recorded_sha256"] for s in streams),
        "every_window_frame_has_bits_once": all(set(range(row["frames"])) <= set(b["display_frames"]) and not b["displayed_twice"]
                                                for row in rows for b in row["bits"].values()),
        "accounting_covers_payload": bool(bits) and all(0.9 <= b["accounted_bits"] / (8 * s["rate_bytes"]) <= 1.0 for b, s in bits),
        "hands_fitted": len(fitted) > 0,
        "ious_in_unit_interval": bool(ious) and all(0.0 <= v <= 1.0 for v in ious),
        "part_shares_sum_to_one": bool(comps) and all(abs(sum(c["frame_share"].values()) - 1.0) < 1e-4 for c in comps),
        "quantization_measured": all(row["quantization"]["joint_error_px_mean"] is not None for row in rows),
        "motion_scored": bool(scored),
        "flow_explains_more_than_no_motion": bool(large) and float(np.mean([m["psnr_flow"] >= m["psnr_copy"] for m in large])) >= 0.9,
        "review_drawn": len(reviews) >= result["settings"]["review"] * len(rows),
    }


def command_validate(args: argparse.Namespace) -> int:
    checks = validate_result(stage_dir())
    report = {"passed": all(checks.values()), "checks": checks}
    target = os.environ.get("PS_VALIDATION_PATH")
    if target:
        write_json(Path(target), report)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


# ----------------------------------------------------------------- report

def entropy_bits(symbols: list[int]) -> float:
    """Empirical entropy (bits per symbol) of a list of integers."""
    if not symbols:
        return 0.0
    counts = np.array(list(Counter(symbols).values()), float)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def parameter_rates(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Entropy of first differences, pooled over items, per parameter; bits per hand-frame and object-frame."""
    hand_diffs: dict[str, list[int]] = defaultdict(list)
    starts = 0
    hand_frames = 0
    for row in items:
        for side in visor.HANDS:
            by_t = {h["t"]: h for h in row["hands"] if h["side"] == side and h["fitted"]}
            for t, h in sorted(by_t.items()):
                hand_frames += 1
                flat = h["symbols"]["global_orient"] + h["symbols"]["hand_pose"] + h["symbols"]["camera"]
                if t - 1 in by_t:
                    p = by_t[t - 1]["symbols"]
                    prev = p["global_orient"] + p["hand_pose"] + p["camera"]
                    for i, (a, b) in enumerate(zip(flat, prev)):
                        hand_diffs[str(i)].append(a - b)
                else:
                    starts += 1
    per_value = {i: entropy_bits(v) for i, v in hand_diffs.items()}
    names = [f"orient{i}" for i in range(3)] + [f"pose{i}" for i in range(45)] + ["cam_u", "cam_v", "cam_logz"]
    diff_frames = len(hand_diffs.get("0", []))
    hand_bits_diff = float(sum(per_value.values()))
    # A track start sends every value at 16 bits plus the shape (10 betas at 16 bits).
    start_bits = 16 * 51 + 16 * 10
    obj_diffs: dict[int, list[int]] = defaultdict(list)
    obj_frames = 0
    for row in items:
        for key, info in row["tracks"].items():
            if info["kind"] in ("handled_object", "other_object"):
                symbols = info["corner_symbols"]
                obj_frames += len(symbols)
                for a, b in zip(symbols[1:], symbols):
                    for i, (x, y) in enumerate(zip(a, b)):
                        obj_diffs[i].append(x - y)
    obj_bits = float(sum(entropy_bits(v) for v in obj_diffs.values()))
    return {
        "hand": {"bits_per_frame_entropy": round(hand_bits_diff + 1, 2), "bits_per_frame_fixed16": 16 * 51 + 1,
                 "track_start_bits": start_bits, "hand_frames": hand_frames, "diff_frames": diff_frames, "starts": starts,
                 "per_value_bits": {names[int(i)]: round(v, 3) for i, v in sorted(per_value.items(), key=lambda kv: int(kv[0]))}},
        "object": {"bits_per_frame_entropy": round(obj_bits + 1, 2), "bits_per_frame_fixed16": 16 * 8 + 1,
                   "object_frames": obj_frames},
    }


def summarize_items(items: list[dict[str, Any]], rates: dict[str, Any]) -> dict[str, Any]:
    """Per item: composition, bits by part, hand fit, rigid holding, parameter kbps."""
    out = []
    for row in items:
        fps, count = row["fps"], row["frames"]
        seconds = count / fps
        comp = {}
        for name in (DENSE, FILL):
            rows = [c["frame_share"] for c in row["composition"] if c["set"] == name]
            mean = {p: float(np.mean([r[p] for r in rows])) for p in PARTS}
            fg = 1.0 - mean["background"]
            comp[name] = {"frame_share": mean, "foreground_share": {p: (mean[p] / fg if fg else None) for p in FOREGROUND_PARTS}}
        bits = {}
        for point, b in row["bits"].items():
            entry_bits: dict[str, Any] = {}
            for name in (DENSE, FILL):
                by = b["bits_by_part"][name]
                total = sum(by.values())
                fg = total - by["background"]
                entry_bits[name] = {"kbps": {p: by[p] / seconds / 1000 for p in PARTS},
                               "frame_share": {p: by[p] / total for p in PARTS},
                               "foreground_share": {p: (by[p] / fg if fg else None) for p in FOREGROUND_PARTS},
                               "foreground_bit_share": fg / total}
            entry_bits["accounted_share_of_payload"] = b["accounted_bits"] / (8 * row["streams"][point]["rate_bytes"])
            entry_bits["stream_kbps"] = row["streams"][point]["kbps"]
            bits[point] = entry_bits
        hands = [h for h in row["hands"]]
        fitted = [h for h in hands if h["fitted"]]
        hand = {
            "hand_frames": len(hands), "fitted_share": len(fitted) / len(hands) if hands else None,
            "iou_hand_median": float(np.median([h["iou_hand"] for h in fitted if h["iou_hand"] is not None])) if fitted else None,
            "iou_hand_at_0.6": float(np.mean([(h["iou_hand"] or 0) >= 0.6 for h in fitted])) if fitted else None,
            "iou_mask_median": float(np.median([h["iou_mask"] for h in fitted if h["iou_mask"] is not None])) if fitted else None,
            "forearm_share_mean": float(np.mean([h["forearm_share"] for h in fitted])) if fitted else None,
        }
        kinds: dict[str, dict[str, Any]] = {}
        for kind in ("hand", "forearm", "handled_object", "other_object"):
            infos = [i for i in row["tracks"].values() if i["kind"] == kind]
            motion = [m for m in row["motion"] if track_kind(m["track"], set(row["contact"]["handled"])) == kind]
            entry: dict[str, Any] = {"tracks": len(infos)}
            for th in ("30", "33.75"):
                frames = sum(i["chains"][th]["frames"] for i in infos)
                held = sum(i["chains"][th]["held"] for i in infos)
                lives = [v for i in infos for v in i["chains"][th]["reference_life_s"]]
                entry[f"held_share_{th}"] = held / frames if frames else None
                entry[f"reference_life_median_s_{th}"] = float(np.median(lives)) if lives else None
                entry[f"references_per_s_{th}"] = sum(i["chains"][th]["references"] for i in infos) / seconds
            for model in ("copy", "similarity", "homography", "flow"):
                values = [m[f"psnr_{model}"] for m in motion if m.get(f"psnr_{model}") is not None]
                entry[f"psnr_{model}_median"] = float(np.median(values)) if values else None
                entry[f"share_{model}_30"] = float(np.mean([v >= 30 for v in values])) if values else None
            split: Counter[str] = Counter()
            for m in motion:
                for k, v in (m.get("error_split") or {}).items():
                    split[k] += v
            total = split.pop("sse", 0.0)
            entry["error_split"] = {k: v / total for k, v in split.items()} if total else {}
            kinds[kind] = entry
        hand_kbps = rates["hand"]["bits_per_frame_entropy"] * len(fitted) / seconds / 1000
        obj_frames = sum(len(i["corner_symbols"]) for i in row["tracks"].values() if i["kind"] in ("handled_object", "other_object"))
        obj_kbps = rates["object"]["bits_per_frame_entropy"] * obj_frames / seconds / 1000
        out.append({"id": row["id"], "video_type": row["video_type"], "composition": comp, "bits": bits, "hand": hand,
                    "stability": row["stability"], "quantization": row["quantization"], "rigid": kinds,
                    "parameter_kbps": {"hands": hand_kbps, "objects": obj_kbps}})
    return {"items": out}


def by_label(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Object tracks grouped by VISOR label over items: frames, kind, share held from a reference."""
    groups: dict[str, dict[str, Any]] = {}
    for row in items:
        handled = set(row["contact"]["handled"])
        motion: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for m in row["motion"]:
            motion[m["track"]].append(m)
        for key, info in row["tracks"].items():
            if info["kind"] not in ("handled_object", "other_object"):
                continue
            g = groups.setdefault(key, {"label": key, "items": 0, "frames": 0, "handled_frames": 0, "held_30": 0,
                                        "chain_frames": 0, "area": 0.0, "psnr_homography": [], "psnr_flow": []})
            g["items"] += 1
            g["frames"] += info["frames"]
            g["handled_frames"] += info["frames"] if key in handled else 0
            g["held_30"] += info["chains"]["30"]["held"]
            g["chain_frames"] += info["chains"]["30"]["frames"]
            g["area"] += info["mean_area"] * info["frames"]
            g["psnr_homography"] += [m["psnr_homography"] for m in motion[key] if m.get("psnr_homography") is not None]
            g["psnr_flow"] += [m["psnr_flow"] for m in motion[key] if m.get("psnr_flow") is not None]
    out = []
    for g in groups.values():
        out.append({"label": g["label"], "items": g["items"], "frames": g["frames"],
                    "handled_share": g["handled_frames"] / g["frames"], "mean_area_px": g["area"] / g["frames"],
                    "held_share_30": g["held_30"] / g["chain_frames"] if g["chain_frames"] else None,
                    "psnr_homography_median": float(np.median(g["psnr_homography"])) if g["psnr_homography"] else None,
                    "psnr_flow_median": float(np.median(g["psnr_flow"])) if g["psnr_flow"] else None})
    return sorted(out, key=lambda r: -r["frames"] * r["mean_area_px"])


def mean_of(values: list[float | None]) -> float | None:
    kept = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(kept)) if kept else None


def decide(summary: list[dict[str, Any]]) -> dict[str, Any]:
    rule = DECISION
    out: dict[str, Any] = {}
    mapping = {"hand": "hand", "forearm": "forearm", "handled_object": "handled_object", "other_object": "other_object"}
    for part, rigid_key in mapping.items():
        share = {p: mean_of([s["bits"][f"{p}"][FILL]["foreground_share"][part] for s in summary if f"{p}" in s["bits"]])
                 for p in (str(v) for v in rule["bit_share_points"])}
        a = all(v is not None and v >= rule["min_foreground_bit_share"] for v in share.values())
        if part == "hand":
            fitted = mean_of([s["hand"]["fitted_share"] for s in summary])
            median_iou = mean_of([s["hand"]["iou_hand_median"] for s in summary])
            at06 = mean_of([s["hand"]["iou_hand_at_0.6"] for s in summary])
            b = (fitted or 0) >= rule["hands"]["fitted_share"] and (median_iou or 0) >= rule["hands"]["median_hand_iou"] \
                and (at06 or 0) >= rule["hands"]["share_at_0.6"]
            explained = {"fitted_share": fitted, "iou_hand_median": median_iou, "iou_hand_at_0.6": at06}
            params = mean_of([s["parameter_kbps"]["hands"] for s in summary])
        else:
            held = mean_of([s["rigid"][rigid_key]["held_share_30"] for s in summary])
            life = mean_of([s["rigid"][rigid_key]["reference_life_median_s_30"] for s in summary])
            b = (held or 0) >= rule["rigid"]["held_share"] and (life or 0) >= rule["rigid"]["median_reference_life_s"]
            explained = {"held_share_30": held, "reference_life_median_s_30": life}
            params = mean_of([s["parameter_kbps"]["objects"] for s in summary]) if part.endswith("object") else None
        point = str(rule["parameter_rate_point"])
        codec = mean_of([s["bits"][point][FILL]["kbps"][part] for s in summary if point in s["bits"]])
        c = params is not None and codec is not None and codec > 0 and params / codec <= rule["max_parameter_rate_share"]
        if part == "forearm":
            c = True  # the forearm's parameters ride on the hand's pose
        out[part] = {"foreground_bit_share": share, "a_bits": a, "explained": explained, "b_explained": b,
                     "parameter_kbps": params, "codec_kbps_crf62": codec, "c_rate": c, "parametric": a and b and c}
    forearm_share = mean_of([s["composition"][FILL]["foreground_share"]["forearm"] for s in summary])
    hand_share = mean_of([s["composition"][FILL]["foreground_share"]["hand"] for s in summary])
    out["forearm_of_hand_masks"] = (forearm_share / (forearm_share + hand_share)) if forearm_share and hand_share else None
    out["forearm_rendered_by_H3"] = (out["forearm_of_hand_masks"] or 0) >= rule["forearm_render_share"]
    return out


def command_report(args: argparse.Namespace) -> int:
    import tarfile

    items = []
    sources = []
    for path in args.result:
        stage = Path(path).parent
        result = json.loads(Path(path).read_text())
        sources.append({"path": path, "sha256": file_sha256(Path(path)), "device": result["device"]["name"]})
        with tarfile.open(stage / "published.tar") as tar:
            for row in result["items"]:
                handle = tar.extractfile(f"publish/{row['published']}")
                assert handle is not None
                items.append(json.loads(handle.read()))
    rates = parameter_rates(items)
    summary = summarize_items(items, rates)["items"]
    report = {"results": sources, "items": len(items), "parameter_rates": rates, "per_item": summary,
              "objects_by_label": by_label(items),
              "decision": decide(summary), "rule": DECISION}
    out = Path(args.out)
    write_json(out / "h1-report.json", report)
    print(json.dumps(report["decision"], indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--eval-set", required=True)
    run.add_argument("--archive", required=True)
    run.add_argument("--masks", nargs=2, action="append", metavar=("NAME", "DIR"), required=True)
    run.add_argument("--mask-record", nargs=2, action="append", metavar=("NAME", "JSON"), required=True)
    run.add_argument("--video", action="append", required=True)
    run.add_argument("--items", required=True)
    run.add_argument("--frames", type=int, default=240)
    run.add_argument("--wilor-checkpoint", required=True)
    run.add_argument("--wilor-detector", required=True)
    run.add_argument("--mano-left", required=True)
    run.add_argument("--mano-right", required=True)
    run.add_argument("--svt-streams", action="append", required=True, help="extracted B2 SVT-AV1 published.tar")
    run.add_argument("--svt-result", action="append", required=True, help="that job's b2.json")
    run.add_argument("--inspect", required=True)
    run.add_argument("--review", type=int, default=2)
    sub.add_parser("validate")
    report = sub.add_parser("report")
    report.add_argument("--result", action="append", required=True, help="a run's h1.json, beside its published.tar")
    report.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return {"run": command_run, "validate": command_validate, "report": command_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
