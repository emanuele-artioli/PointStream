"""K6b (hands) 2c: a buildable reference bank, chosen outside the window and billed (docs/experiments.md, 2026-10-22).

    python -m experiments.visor.k6bh2c run --eval-set JSON --archive DIR --masks DIR --mask-record JSON --video MP4 ... \\
        --items IDS --frames N --stride N --lpips-backbone PTH --dists-backbone PTH --k1f-stage TAR ... \\
        --wilor-checkpoint CKPT --mano-right PKL --mano-left PKL
    python -m experiments.visor.k6bh2c validate
    python -m experiments.visor.k6bh2c report --stage DIR ... --k1f-stage TAR ... --out JSON

Per K1f item: VISOR's dense hand masks on the video's frames outside the scored window (and a guard), every
``stride``-th; WiLoR's two-pass hand-box fit on each (K1e's ``k1c.mask_box_rows``); per side a nested bank chosen by
greedy k-medoids on 2b's pose vector; each reference coded as one SVT-AV1 still with its mask as polygons and its
joints, billed once per video. The window's frames are warped from the reference nearest in coded pose (2b's
decoder) with the coded reference (``bank2c@K``) and with the uncoded one (``free2c@K``), beside K1f's bound
(``warp``). Scored on R exactly as K1f and 2b (``h3c.Scorer``); ``report`` reads K1f's hull at each item's rate,
the keypoints' plus the amortized bank's.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.visor import k1c
from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json
from experiments.visor.k6bh import k1f_rows, pose_vector

KS = (16, 32, 64)
GUARD_SECONDS = 1.0  # candidates this close to the window are left out, on each side
NEAR_SECONDS = 10.0  # diagnostic: references (and matches) this close to the window
MIN_INSIDE = 15  # of 21 joints inside the dilated mask, or the fit is dropped
DILATE = 0.10  # of the mask box's longer side
MAX_CANDIDATES = 4000  # per side, for the medoid search (evenly subsampled beyond)
CRF = 40
PRESET = 4
BOX_PAD = 8
MIN_SIDE = 64  # px: SVT-AV1's smallest frame
POLY_EPS = 1.5  # px, OpenCV's approxPolyDP
COORD_BITS = 11  # a box corner or size in a 1920 x 1080 frame
WILOR_CHUNK = 32
GPU_CAP_MIB = 27 * 1024  # PyTorch's cap: the allocator frees its cache before growing past it (declared 30,000 MiB)
WARP_TOLERANCE = 0.002
CRITERION = 0.75
BANK2C_DECISION: dict[str, Any] = {
    "variants": "bank2c@K (coded, billed) and free2c@K (same references uncoded, unbilled), K in {16, 32, 64} per side; "
                "warp (K1f's bound, the check)",
    "candidates": "VISOR dense hand masks outside the window and a 1 s guard, every stride-th VISOR frame, WiLoR's two-pass "
                  "hand-box fit, kept with >= 15 of 21 joints in the mask dilated by 10% of its box",
    "selection": "per side, greedy k-medoids on 2b's pose vector (nested)",
    "billing": f"per reference an SVT-AV1 still (CRF {CRF}, preset {PRESET}) of the padded mask box, the mask as polygons, "
               "the joints, the box and side; payload bits; amortized over the whole video",
    "item_won": "bank2c@K's LPIPS (DISTS) on R below K1f's hull at the keypoint rate plus the amortized bank; a rate above "
                "the hull's highest point counts as lost",
    "score": "mean over {LPIPS, DISTS} of the share of items won",
    "pass": "some K scores >= 0.75: a buildable renderer for hands; next, that K on K1f's other held-out items",
    "fail_free_passes": "free2c reaches 0.75 at some K: the bank's bits fail; next goes to the user",
    "fail": "free2c fails too: the reference route closes at a buildable rate; rung 3 waits for the user",
    "check": f"warp reproduces K1f's recorded LPIPS on R within {WARP_TOLERANCE} per item, or nothing is read",
}


# ----------------------------------------------------------------- selection and coding (CPU)


def pairwise(poses: np.ndarray) -> np.ndarray:
    sq = (poses ** 2).sum(1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None] - 2.0 * poses @ poses.T, 0.0))


def greedy_medoids(poses: np.ndarray, k: int) -> list[int]:
    """Indices of up to ``k`` poses, each step adding the one that most lowers the mean distance from every pose to its
    nearest chosen one. Nested: the first j of a call with k > j are the call with j."""
    n = len(poses)
    if n == 0:
        return []
    dist = pairwise(poses.astype(np.float64))
    best = np.full(n, np.inf)
    chosen: list[int] = []
    while len(chosen) < min(k, n):
        cost = np.minimum(best[:, None], dist).mean(0)
        cost[chosen] = np.inf
        c = int(np.argmin(cost))
        if not np.isfinite(cost[c]):
            break
        chosen.append(c)
        best = np.minimum(best, dist[:, c])
    return chosen


def joints_inside(points: np.ndarray, mask: np.ndarray, frac: float = DILATE) -> int:
    """How many joints fall inside ``mask`` dilated by ``frac`` of its box's longer side."""
    import cv2

    ys, xs = np.nonzero(mask)
    if not len(ys):
        return 0
    r = max(1, int(round(frac * max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))))
    h, w = mask.shape
    y0, y1, x0, x1 = max(0, ys.min() - r), min(h, ys.max() + r + 1), max(0, xs.min() - r), min(w, xs.max() + r + 1)
    grown = cv2.dilate(mask[y0:y1, x0:x1].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    px = np.round(points[:, 0]).astype(int) - x0
    py = np.round(points[:, 1]).astype(int) - y0
    ok = (px >= 0) & (px < x1 - x0) & (py >= 0) & (py < y1 - y0)
    return int(grown[py[ok], px[ok]].sum())


def ref_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """(x0, y0, w, h): the mask's box padded by BOX_PAD, sides even multiples of 8 and >= MIN_SIDE, inside the frame."""
    ys, xs = np.nonzero(mask)
    fh, fw = mask.shape
    x0, x1 = xs.min() - BOX_PAD, xs.max() + 1 + BOX_PAD
    y0, y1 = ys.min() - BOX_PAD, ys.max() + 1 + BOX_PAD

    def side(lo: int, hi: int, limit: int) -> tuple[int, int]:
        size = min(limit - limit % 8, max(MIN_SIDE, int(math.ceil((hi - lo) / 8)) * 8))
        start = int(round((lo + hi - size) / 2))
        return min(max(start, 0), limit - size), size

    bx, w = side(int(x0), int(x1), fw)
    by, h = side(int(y0), int(y1), fh)
    return bx, by, w, h


def smooth_fill(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """``image`` with the pixels outside ``mask`` replaced by a push-pull interpolation of the ones inside (smooth, so
    the coder spends few bits outside the mask)."""
    import cv2

    if mask.all() or not mask.any():
        return image.copy()
    num: Any = image.astype(np.float32) * mask[..., None]
    den: Any = mask.astype(np.float32)
    pyramid = [(num, den)]
    while min(num.shape[:2]) > 1:
        h, w = num.shape[:2]
        num = cv2.resize(num, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA).reshape(max(1, h // 2), max(1, w // 2), 3)
        den = cv2.resize(den, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA).reshape(max(1, h // 2), max(1, w // 2))
        pyramid.append((num, den))
    num, den = pyramid[-1]
    est = num / np.maximum(den[..., None], 1e-6)
    for num, den in reversed(pyramid[:-1]):
        h, w = num.shape[:2]
        up = cv2.resize(est, (w, h), interpolation=cv2.INTER_LINEAR).reshape(h, w, 3)
        a = np.clip(den, 0.0, 1.0)[..., None]
        est = a * (num / np.maximum(den[..., None], 1e-6)) + (1.0 - a) * up
    out = np.clip(np.round(est), 0, 255).astype(np.uint8)
    out[mask] = image[mask]
    return out


def code_polygons(mask: np.ndarray) -> tuple[list[np.ndarray], int]:
    """The mask's outer contours simplified at POLY_EPS px, and their bits: a polygon count (8), then per polygon a
    vertex count (12) and each vertex at ceil(log2 w) + ceil(log2 h) bits."""
    import cv2

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polys = [cv2.approxPolyDP(c, POLY_EPS, True).reshape(-1, 2) for c in contours]
    polys = [p.astype(np.int32) for p in polys if len(p) >= 3]
    h, w = mask.shape
    per_vertex = max(1, math.ceil(math.log2(w))) + max(1, math.ceil(math.log2(h)))
    return polys, 8 + sum(12 + len(p) * per_vertex for p in polys)


def decode_polygons(polys: list[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    import cv2

    canvas = np.zeros(shape, np.uint8)
    if polys:
        cv2.fillPoly(canvas, polys, (1,))
    return canvas > 0


def joint_bits(w: int, h: int) -> int:
    return 21 * (max(1, math.ceil(math.log2(w))) + max(1, math.ceil(math.log2(h))))


def sent_joints(points: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """The joints as the decoder gets them: whole pixels in the box, clipped to it, back in frame coordinates."""
    x0, y0, w, h = box
    local = np.round(points[:, :2] - np.array([x0, y0]))
    local = np.clip(local, 0, [w - 1, h - 1])
    return local + np.array([x0, y0], float)


def with_conf(points: np.ndarray) -> np.ndarray:
    """(21, 3) joints with confidence 1, as ``k6.draw`` reads them (K1f's rows carry the same)."""
    return np.concatenate([np.asarray(points, float)[:, :2], np.ones((len(points), 1))], -1)


def header_bits() -> int:
    return 4 * COORD_BITS + 1  # the box and the side


def payback_seconds(bank_bits: float, kp_kbps: float, hull: list[tuple[float, float]], value: float) -> dict[str, Any]:
    """The shortest video for which ``value`` (lower is better) at the keypoint rate plus the bank stays below the
    hull: bank bits / (hull's rate at that quality - keypoint rate)."""
    from experiments.visor.k1 import rate_for

    need = rate_for(hull, -value)
    if need is None:
        if -value > hull[-1][1]:
            return {"seconds": bank_bits / max(1e-9, (hull[-1][0] - kp_kbps) * 1000), "bound": "at most (beats every hull point)"}
        return {"seconds": None, "bound": "never (worse than every hull point)"}
    if need <= kp_kbps:
        return {"seconds": None, "bound": "never (the hull reaches it at the keypoint rate)"}
    return {"seconds": bank_bits / ((need - kp_kbps) * 1000), "bound": "exact"}


# ----------------------------------------------------------------- candidates (CPU, before torch)


def hand_masks(frame: Any) -> dict[str, np.ndarray]:
    """VISOR's dense masks of one frame per hand side (polygons drawn at 480p, filled at 1080p)."""
    from src.segmentation import visor

    out: dict[str, np.ndarray] = {}
    for a in frame.annotations:
        side = visor.native_class(a)
        if side in visor.HANDS:
            m = visor.polygon_mask(a["segments"], source_size=visor.DENSE_SIZE)
            out[side] = out[side] | m if side in out else m
    return out


def candidate_frames(item: dict[str, Any], doc: dict[str, Any], mapping: dict[str, Any], stride: int) -> list[tuple[int, int]]:
    """(VISOR frame, decoded index) of every ``stride``-th labelled frame with a hand, outside the window and the
    guard; on non-50 fps videos only frames ``visor.frame_alignment`` places exactly."""
    from src.segmentation import visor

    fps = float(item["fps"])
    frames = visor.frames(doc)
    if abs(fps - 50.0) < 0.01:
        index = {n: n - 1 for n in frames}
    else:
        placed = visor.frame_alignment(doc, visor.keyframe_anchors(doc, mapping[item["video"]], fps), fps / visor.extraction_rate(fps))
        index = {n: p["index"] for n, p in placed.items() if p["exact"]}
    guard = int(round(GUARD_SECONDS * fps))
    lo, hi = int(item["first_video_index"]) - guard, int(item["first_video_index"]) + int(item["frames"]) + guard
    out = []
    for n, frame in frames.items():
        if n % stride or n not in index or lo <= index[n] < hi:
            continue
        if any(visor.native_class(a) in visor.HANDS for a in frame.annotations):
            out.append((n, index[n]))
    return sorted(out, key=lambda p: p[1])


def decode_bank_frames(video: str, indices: list[int], out: str, threads: int) -> dict[str, Any]:
    """The 4:2:0 planes of the decoded frames at ``indices`` (sorted) into ``out``, in order. Wanted frames closer than
    50 apart are reached by decoding through (one seek per stretch)."""
    import av

    from src.segmentation import visor

    began = time.time()
    wanted = sorted(set(indices))
    span: list[int] = []
    for a, b in zip(wanted, [*wanted[1:], None]):
        span.append(a)
        if b is not None and b - a <= 50:
            span.extend(range(a + 1, b))
    keep = set(wanted)
    written: list[int] = []
    width = height = 0
    formats: set[str] = set()
    with open(out, "xb") as handle:
        for at, frame in visor.decoded_frames(video, span, threads=threads):
            if at not in keep or (written and at <= written[-1]):
                continue
            fmt = frame.format.name
            if fmt not in ("yuv420p", "yuvj420p"):
                raise RuntimeError(f"{video}: decoded format {fmt} is not 8-bit 4:2:0")
            formats.add(fmt)
            planes = np.ascontiguousarray(frame.to_ndarray())
            height, width = planes.shape[0] * 2 // 3, planes.shape[1]
            handle.write(planes.tobytes())
            written.append(at)
    with av.open(video) as container:
        stream = container.streams.video[0]
        duration = float(container.duration) / 1e6 if container.duration else float(stream.duration * stream.time_base)
        frames_total = int(stream.frames)
        rate = float(stream.average_rate) if stream.average_rate else None
    if written != wanted:
        raise RuntimeError(f"{video}: decoded {len(written)} of {len(wanted)} bank frames")
    return {"path": out, "indices": written, "width": width, "height": height, "formats": sorted(formats),
            "duration_s": duration, "frames_total": frames_total, "fps": rate, "seconds": round(time.time() - began, 2)}


# ----------------------------------------------------------------- run (GPU)


class _Masks:
    """``class_mask(t, side)`` over a chunk of candidate frames, for ``k1c.mask_box_rows``."""

    def __init__(self, masks: list[dict[str, np.ndarray]], shape: tuple[int, int]) -> None:
        self.masks, self.empty = masks, np.zeros(shape, bool)

    def class_mask(self, t: int, side: str) -> np.ndarray:
        return self.masks[t].get(side, self.empty)


def to_rgb(planes: np.ndarray, width: int, height: int, full_range: bool, device: str) -> np.ndarray:
    import torch

    from src.codecs.quality import yuv420_to_rgb

    rgb = yuv420_to_rgb(torch.from_numpy(planes).to(device), width, height, full_range=full_range)
    return rgb.permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()


def fit_candidates(wilor: Any, bank: dict[str, Any], cands: list[tuple[int, int]], frames_doc: dict[int, Any], full_range: bool,
                   device: str) -> dict[str, np.ndarray]:
    """K1e's two-pass hand-box fit on every candidate side, and the inside-the-mask test."""
    from experiments.visor.h2 import NOMINAL_FOCAL
    from src.codecs.quality import read_frames
    from src.segmentation import visor

    shape = visor.FRAME_SIZE
    meta = {"image_shape": list(shape), "focal": NOMINAL_FOCAL * max(shape), "principal": (shape[1] / 2.0, shape[0] / 2.0)}
    empty = {"t": np.zeros(0, int), "right": np.zeros(0, bool), "box": np.zeros((0, 4)), "wilor_rotvec": np.zeros((0, 16, 3)),
             "wilor_betas": np.zeros((0, 10)), "wilor_transl": np.zeros((0, 3)), "wilor_kp2d": np.zeros((0, 21, 2))}
    w, h = bank["width"], bank["height"]
    rows: dict[str, list[Any]] = {"cand": [], "right": [], "kp2d": [], "inside": []}
    for first in range(0, len(cands), WILOR_CHUNK):
        chunk = list(range(first, min(len(cands), first + WILOR_CHUNK)))
        rgb = to_rgb(read_frames(Path(bank["path"]), w, h, first, len(chunk)), w, h, full_range, device)
        masks = [hand_masks(frames_doc[cands[c][0]]) for c in chunk]
        new = k1c.mask_box_rows(wilor, rgb, _Masks(masks, shape), len(chunk), meta, empty, hand_box=True)
        for t, right, kp in zip(new["t"], new["right"], new["wilor_kp2d"]):
            side = visor.HANDS[1] if right else visor.HANDS[0]
            rows["cand"].append(chunk[int(t)])
            rows["right"].append(bool(right))
            rows["kp2d"].append(np.asarray(kp, np.float32))
            rows["inside"].append(joints_inside(np.asarray(kp), masks[int(t)][side]))
    return {"cand": np.array(rows["cand"], int), "right": np.array(rows["right"], bool),
            "kp2d": np.stack(rows["kp2d"]) if rows["kp2d"] else np.zeros((0, 21, 2), np.float32),
            "inside": np.array(rows["inside"], int)}


def code_still(crop: np.ndarray, work: Path, encoder: str, decoder: str) -> tuple[np.ndarray, int, str]:
    """One SVT-AV1 still of ``crop`` (RGB, BT.709 full range 4:2:0), decoded by dav1d: (RGB, payload bytes, sha256)."""
    from experiments.background.g5 import rgb_to_yuv420, yuv420_to_rgb
    from src.codecs import svtav1

    h, w = crop.shape[:2]
    work.mkdir(parents=True, exist_ok=True)
    source, stream, decoded = work / "ref.yuv", work / "ref.ivf", work / "ref-dec.yuv"
    source.write_bytes(rgb_to_yuv420(crop[None]).tobytes())
    svtav1.run(svtav1.encode_command(encoder, source, stream, width=w, height=h, fps=Fraction(1), frames=1, crf=CRF,
                                     preset=PRESET, full_range=True), 600)
    svtav1.run(svtav1.decode_command(decoder, stream, decoded, threads=1), 600)
    data = stream.read_bytes()
    planes = np.fromfile(decoded, np.uint8)
    if planes.size != w * h * 3 // 2:
        raise RuntimeError(f"{work}: decoded {planes.size} bytes, expected {w * h * 3 // 2}")
    rgb = yuv420_to_rgb(planes.reshape(1, h * 3 // 2, w))[0]
    for f in (source, stream, decoded):
        f.unlink()
    return rgb, sum(svtav1.ivf_frames(data)), hashlib.sha256(data).hexdigest()


def build_bank(args: argparse.Namespace, bank_frames: dict[str, Any], cands: list[tuple[int, int]], frames_doc: dict[int, Any],
               fits: dict[str, np.ndarray], full_range: bool, device: str, work: Path
               ) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Per side, the nested medoid bank (up to max(KS)), each reference coded and billed; and the candidates per side."""
    from src.codecs.quality import read_frames
    from src.segmentation import visor

    w, h = bank_frames["width"], bank_frames["height"]
    out: dict[str, list[dict[str, Any]]] = {}
    record: list[dict[str, Any]] = []
    jobs = []
    for right, side in ((False, visor.HANDS[0]), (True, visor.HANDS[1])):
        rows = np.nonzero((fits["right"] == right) & (fits["inside"] >= MIN_INSIDE))[0]
        if len(rows) > MAX_CANDIDATES:
            rows = rows[np.linspace(0, len(rows) - 1, MAX_CANDIDATES).round().astype(int)]
        poses = np.stack([pose_vector(fits["kp2d"][r]) for r in rows]) if len(rows) else np.zeros((0, 42))
        chosen = [int(rows[j]) for j in greedy_medoids(poses, max(KS))]
        refs = []
        for rank, r in enumerate(chosen):
            c = int(fits["cand"][r])
            n, index = cands[c]
            rgb = to_rgb(read_frames(Path(bank_frames["path"]), w, h, c, 1), w, h, full_range, device)[0]
            mask = hand_masks(frames_doc[n])[side]
            box = ref_box(mask)
            x0, y0, bw, bh = box
            crop, crop_mask = rgb[y0:y0 + bh, x0:x0 + bw], mask[y0:y0 + bh, x0:x0 + bw]
            polys, poly_bits = code_polygons(crop_mask)
            refs.append({"rank": rank, "row": r, "visor_frame": n, "index": index, "side": side, "box": box,
                         "kp2d": with_conf(fits["kp2d"][r]), "sent_kp2d": with_conf(sent_joints(fits["kp2d"][r], box)),
                         "image": rgb, "mask": mask, "polys": polys, "poly_bits": poly_bits,
                         "joint_bits": joint_bits(bw, bh), "header_bits": header_bits()})
            jobs.append((refs[-1], smooth_fill(crop, crop_mask), work / f"{side.replace(' ', '-')}-{rank:02d}"))
        out[side] = refs
        record.append({"side": side, "fitted": int((fits["right"] == right).sum()), "kept": int(len(rows))})
    with ThreadPoolExecutor(max_workers=8) as pool:
        coded = list(pool.map(lambda j: code_still(j[1], j[2], args.encoder, args.decoder), jobs))
    for (ref, _, _), (rgb, payload, digest) in zip(jobs, coded):
        x0, y0, bw, bh = ref["box"]
        decoded_mask = np.zeros(ref["mask"].shape, bool)
        decoded_mask[y0:y0 + bh, x0:x0 + bw] = decode_polygons(ref["polys"], (bh, bw))
        image = np.zeros_like(ref["image"])
        image[y0:y0 + bh, x0:x0 + bw] = rgb
        source = ref["image"][y0:y0 + bh, x0:x0 + bw][ref["mask"][y0:y0 + bh, x0:x0 + bw]].astype(float)
        mse = float(((rgb[ref["mask"][y0:y0 + bh, x0:x0 + bw]].astype(float) - source) ** 2).mean())
        inter, union = int((decoded_mask & ref["mask"]).sum()), int((decoded_mask | ref["mask"]).sum())
        ref.update({"coded_image": image, "coded_mask": decoded_mask, "image_bits": 8 * payload, "stream_sha256": digest,
                    "psnr_mask": 10 * math.log10(255 ** 2 / max(mse, 1e-9)), "mask_iou": inter / max(1, union),
                    "bits": 8 * payload + ref["poly_bits"] + ref["joint_bits"] + ref["header_bits"]})
    return out, record


def process_item(args: argparse.Namespace, item: dict[str, Any], decoded: dict[str, Any], bank_frames: dict[str, Any],
                 cands: list[tuple[int, int]], nets: dict[str, Any], k1f_unit: tuple[dict[str, Any], dict[str, np.ndarray]],
                 device: str, publish: Path, scratch: Path) -> None:
    from experiments.tennis.k6 import draw, telea_fill
    from experiments.visor.b2 import load_mask_sets
    from experiments.visor.h2 import save_unit
    from experiments.visor.h3 import FILL, HANDS, rgb_memmap
    from experiments.visor.h3c import Scorer
    from experiments.visor.k1f import release
    from src.segmentation import visor

    began = time.time()
    frames = int(args.frames)
    width, height = decoded["width"], decoded["height"]
    work = scratch / "work" / item["id"]
    work.mkdir(parents=True, exist_ok=True)
    doc = visor.load_annotations(Path(args.archive) / item["dense_member"])
    frames_doc = visor.frames(doc)
    full_range = bool(decoded["full_range"])
    fits = fit_candidates(nets["wilor"], bank_frames, cands, frames_doc, full_range, device)
    fit_seconds = time.time() - began
    save_unit(publish, "k6bh2c-candidates", item["id"],
              {"visor_frame": np.array([n for n, _ in cands], int), "index": np.array([i for _, i in cands], int),
               "row_cand": fits["cand"], "row_right": fits["right"], "row_kp2d": fits["kp2d"], "row_inside": fits["inside"]},
              {"unit": item["id"], "item": item["id"], "video": item["video"], "stride": int(args.stride), "candidates": len(cands),
               "rows": int(len(fits["cand"])), "kept": int((fits["inside"] >= MIN_INSIDE).sum())})
    release()
    banks, candidate_record = build_bank(args, bank_frames, cands, frames_doc, fits, full_range, device, work)
    os.remove(bank_frames["path"])
    clips, mask_info = load_mask_sets(item, frames, {FILL: args.masks}, {FILL: args.mask_record}, Path(args.archive))
    clip = clips[FILL]
    truth = rgb_memmap(Path(decoded["path"]), work / "truth.npy", frames, width, height, full_range, device)
    meta_k, a_k = k1f_unit
    ones = np.ones((len(a_k["row_t"]), 21, 1))
    rows = {"t": np.asarray(a_k["row_t"]).astype(int), "right": np.asarray(a_k["row_right"]).astype(bool),
            "uncoded": np.concatenate([a_k["uncoded_kp2d"], ones], -1), "coded": np.concatenate([a_k["coded_kp2d"], ones], -1)}
    size = np.ptp(rows["coded"][:, :, 1], axis=1) + np.ptp(rows["coded"][:, :, 0], axis=1)
    variants = ["warp"] + [f"{kind}@{k}" for kind in ("bank2c", "free2c") for k in KS]
    rows_at: dict[int, list[int]] = {}
    for i, t in enumerate(rows["t"]):
        if t < frames:
            rows_at.setdefault(int(t), []).append(i)
    regions = [clip.class_mask(t, HANDS[0]) | clip.class_mask(t, HANDS[1]) for t in range(frames)]
    outs = {v: np.lib.format.open_memmap(work / f"{v.replace('@', '-')}.npy", mode="w+", dtype=np.uint8, shape=(frames, height, width, 3))
            for v in variants}
    covered = {v: np.zeros(frames, np.int64) for v in variants}
    poses = {s: {"sent": [pose_vector(r["sent_kp2d"]) for r in refs], "free": [pose_vector(r["kp2d"]) for r in refs]}
             for s, refs in banks.items()}
    matches: dict[str, list[tuple[int, float, int]]] = {v: [] for v in variants if v != "warp"}  # (rank, pose distance, |dt|)
    window_mid = int(item["first_video_index"]) + int(item["frames"]) // 2
    for t in range(frames):
        truth_t = np.asarray(truth[t])
        if not regions[t].any():
            for v in variants:
                outs[v][t] = truth_t
            continue
        sides = {s: clip.class_mask(t, s) for s in HANDS}
        objects = clip.foreground(t) & ~regions[t]
        here = sorted(rows_at.get(t, []), key=lambda i: size[i])
        hands = [(bool(rows["right"][i]), rows["uncoded"][i], rows["coded"][i], -float(size[i])) for i in here]
        frame, mine = k1c.compose(truth_t, sides, objects, hands)
        outs["warp"][t] = frame
        covered["warp"][t] = int((mine & regions[t]).sum())
        filled = telea_fill(truth_t, regions[t])
        for v in variants[1:]:
            kind, k = v.split("@")
            coded = kind == "bank2c"
            canvas = filled.copy()
            drawn = np.zeros(regions[t].shape, bool)
            for i in here:
                side = HANDS[1] if rows["right"][i] else HANDS[0]
                refs = banks.get(side, [])[:int(k)]
                if not sides[side].any() or not refs:
                    continue
                target = pose_vector(rows["coded"][i])
                d = [float(np.linalg.norm(p - target)) for p in poses[side]["sent" if coded else "free"][:len(refs)]]
                j = int(np.argmin(d))
                ref = refs[j]
                mask = ref["coded_mask"] if coded else ref["mask"]
                ys = np.nonzero(mask.any(1))[0]
                draw(canvas, drawn, ref["coded_image"] if coded else ref["image"], mask, ref["sent_kp2d"] if coded else ref["kp2d"],
                     rows["coded"][i], k1c.SIGMA_OF_HEIGHT * float(ys.max() - ys.min() + 1), k1c.HAND_SKELETON)
                matches[v].append((j, d[j], abs(int(ref["index"]) - window_mid)))
            over = drawn & objects
            canvas[over] = truth_t[over]
            outs[v][t] = canvas
            covered[v][t] = int((drawn & regions[t]).sum())
    for v in variants:
        outs[v].flush()
    score = Scorer(nets, truth, regions, device)
    hand = np.array(score.hand, int)
    arrays: dict[str, np.ndarray] = {"pixels_r": np.array([int(r.sum()) for r in regions], np.int64)}
    summary: dict[str, dict[str, Any]] = {}
    fps = float(item["fps"])
    near = NEAR_SECONDS * fps
    for v in variants:
        s = score(outs[v], upscale=False)
        release()
        key = v.replace("@", "_")
        arrays[f"{key}_lpips_r"], arrays[f"{key}_dists_r"], arrays[f"{key}_sse_r"] = s["lpips"], s["dists"], s["sse"]
        arrays[f"{key}_covered_r"] = covered[v]
        summary[v] = {"lpips_r": float(np.nanmean(s["lpips"][hand])), "dists_r": float(np.nanmean(s["dists"][hand])),
                      "covered_share": float(covered[v][hand].sum() / max(1, arrays["pixels_r"][hand].sum()))}
        if v in matches and matches[v]:
            m = np.array(matches[v], float)
            summary[v].update({"matches": len(m), "pose_distance_mean": float(m[:, 1].mean()),
                               "matched_near_share": float((m[:, 2] <= near).mean()), "references_used": int(len(set(m[:, 0].astype(int))))})
    bank_bits = {str(k): int(sum(r["bits"] for refs in banks.values() for r in refs[:k])) for k in KS}
    bank_record = {side: [{"rank": r["rank"], "visor_frame": r["visor_frame"], "index": r["index"], "box": list(r["box"]),
                           "bits": r["bits"], "image_bits": r["image_bits"], "poly_bits": r["poly_bits"], "joint_bits": r["joint_bits"],
                           "header_bits": r["header_bits"], "psnr_mask": r["psnr_mask"], "mask_iou": r["mask_iou"],
                           "mask_px": int(r["mask"].sum()), "stream_sha256": r["stream_sha256"],
                           "seconds_from_window": abs(r["index"] - window_mid) / fps} for r in refs] for side, refs in banks.items()}
    lo, hi = int(item["first_video_index"]), int(item["first_video_index"]) + int(item["frames"])
    meta = {"unit": item["id"], "item": item["id"], "video": item["video"], "frames": frames, "hand_frames": len(hand),
            "summary": summary, "bank_bits": bank_bits, "bank": bank_record, "candidates": candidate_record,
            "candidate_frames": len(cands), "stride": int(args.stride),
            "nearest_candidate_frames": int(min(min(abs(i - lo), abs(i - (hi - 1))) for _, i in cands)) if cands else None,
            "window": [lo, hi], "guard_frames": int(round(GUARD_SECONDS * fps)),
            "video_duration_s": bank_frames["duration_s"], "video_frames": bank_frames["frames_total"], "video_fps": bank_frames["fps"],
            "near_share": {str(k): float(np.mean([abs(r["index"] - window_mid) <= near for refs in banks.values() for r in refs[:k]]))
                           for k in KS},
            "k1f": {"kbps": float(meta_k["cover"]["kbps"]) if meta_k.get("cover") else k1c.POSE_KBPS,
                    "warp": meta_k["summary"]["warp"], "frames": int(meta_k["frames"]), "npz_hash_ok": bool(meta_k["npz_hash_ok"])},
            "mask_set": mask_info, "decode_bank_seconds": bank_frames["seconds"], "fit_seconds": round(fit_seconds, 2),
            "seconds": round(time.time() - began, 2)}
    save_unit(publish, "k6bh2c", item["id"], arrays, meta)
    del outs, truth
    os.remove(decoded["path"])
    for f in work.glob("*.npy"):
        f.unlink()


def command_run(args: argparse.Namespace) -> int:
    from experiments.visor.b2 import select_items
    from experiments.visor.h2 import restore_units
    from experiments.visor.h3c import decode_windows
    from src.segmentation import visor

    items = select_items(json.loads(Path(args.eval_set).read_text()), args.items)
    for it in items:
        it["frames"] = int(it.get("frames", 240))
    units = k1f_rows(args.k1f_stage)
    if missing := [it["id"] for it in items if it["id"] not in units]:
        raise SystemExit(f"no K1f rows for {missing}")
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    restored = restore_units(publish, "k6bh2c")
    todo = [it for it in items if it["id"] not in restored]
    mapping = json.loads((Path(args.archive) / "frame_mapping.json").read_text())
    cands = {it["id"]: candidate_frames(it, visor.load_annotations(Path(args.archive) / it["dense_member"]), mapping, int(args.stride))
             for it in todo}
    videos = {Path(v).stem: v for v in args.video}
    allowance = max(1, int(os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 1))
    pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn"))
    (scratch / "bank").mkdir(parents=True, exist_ok=True)
    bank_jobs = {}
    for it in todo:  # PyAV in a fresh process, started before torch
        target = scratch / "bank" / f"{it['id']}.yuv"
        if target.exists():
            target.unlink()
        bank_jobs[it["id"]] = pool.submit(decode_bank_frames, videos[it["video"]], [i for _, i in cands[it["id"]]], str(target),
                                          max(1, allowance // 2))
    windows = decode_windows(args, todo, int(args.frames), scratch)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # one allocator for crops of every size
    import torch

    from experiments.audit.env_smoke import device_record, model_on_cuda
    from experiments.visor.h2 import Regressor, mano_dir
    from experiments.visor.h3b import load_dists
    from src.codecs import svtav1
    from src.codecs.quality import load_lpips, lpips_record

    if not torch.cuda.is_available():
        raise SystemExit("k6bh2c run needs a GPU")
    device = "cuda"
    total_mib = torch.cuda.get_device_properties(0).total_memory / 2 ** 20
    torch.cuda.set_per_process_memory_fraction(min(1.0, GPU_CAP_MIB / total_mib))
    nets: dict[str, Any] = {"lpips": load_lpips(Path(args.lpips_backbone), device), "dists": load_dists(Path(args.dists_backbone), device)}
    args.mano_dir = mano_dir(args, scratch)
    nets["wilor"] = Regressor("wilor", args)
    on_cuda = {k: next(v.parameters()).is_cuda for k, v in nets.items() if k != "wilor"}
    on_cuda["wilor"] = model_on_cuda(nets["wilor"].model)
    began = time.time()
    done = len(restored)
    for item, decoded in windows:
        process_item(args, item, decoded, bank_jobs[item["id"]].result(), cands[item["id"]], nets, units[item["id"]], device,
                     publish, scratch)
        done += 1
        progress(done)
    pool.shutdown()
    import torchmetrics

    write_json(stage_dir() / "k6bh2c.json", {
        "command": "run", "device": device_record(args.gpu_class), "on_cuda": on_cuda, "seconds": round(time.time() - began, 2),
        "inputs": {"k1f_stages": [file_sha256(Path(t)) for t in args.k1f_stage],
                   "wilor_checkpoint": file_sha256(Path(args.wilor_checkpoint)), "videos": {Path(v).stem: file_sha256(Path(v)) for v in args.video}},
        "wilor": nets["wilor"].load, "lpips": lpips_record(Path(args.lpips_backbone)),
        "dists": {"torchmetrics": torchmetrics.__version__, "backbone_sha256": file_sha256(Path(args.dists_backbone))},
        "gpu_cap_mib": GPU_CAP_MIB, "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2 ** 20, 1),
        "svt": {"encoder": svtav1.tool(args.encoder), "decoder": svtav1.tool(args.decoder), "crf": CRF, "preset": PRESET},
        "settings": {"frames": int(args.frames), "stride": int(args.stride), "items": [it["id"] for it in items], "ks": list(KS),
                     "guard_seconds": GUARD_SECONDS, "min_inside": MIN_INSIDE, "dilate": DILATE, "poly_eps": POLY_EPS,
                     "decision": BANK2C_DECISION}, "restored_units": sorted(restored)})
    return 0


# ----------------------------------------------------------------- validation (CPU)


def validate(stage: Path, result: dict[str, Any]) -> dict[str, bool]:
    from experiments.visor.h2 import load_units

    units = {m["item"]: (m, a) for m, a in load_units([str(stage / "published.tar")], "k6bh2c")}
    settings = result["settings"]
    checks = {"every_item": sorted(units) == sorted(settings["items"]), "on_cuda": all(result["on_cuda"].values()),
              "device_class": bool(result["device"]["class_matches"]), "crf": result["svt"]["crf"] == CRF}
    for item, (m, a) in units.items():
        hand = a["pixels_r"] > 0
        lo, hi = m["window"]
        checks[f"{item}:hash"] = bool(m["npz_hash_ok"]) and bool(m["k1f"]["npz_hash_ok"])
        checks[f"{item}:candidates"] = m["candidate_frames"] > 0 and all(c["kept"] > 0 for c in m["candidates"])
        checks[f"{item}:guard"] = m["nearest_candidate_frames"] is not None and m["nearest_candidate_frames"] > m["guard_frames"]
        refs = [r for side in m["bank"].values() for r in side]
        checks[f"{item}:bank_outside_window"] = all(not lo - m["guard_frames"] <= r["index"] < hi + m["guard_frames"] for r in refs)
        checks[f"{item}:bank_sizes"] = all(1 <= len(side) <= max(KS) for side in m["bank"].values()) and bool(m["bank"])
        checks[f"{item}:refs_coded"] = all(r["image_bits"] > 0 and r["psnr_mask"] > 25.0 for r in refs)
        checks[f"{item}:polygons"] = all(r["mask_iou"] > 0.9 for r in refs)
        checks[f"{item}:bits_nested"] = m["bank_bits"]["16"] <= m["bank_bits"]["32"] <= m["bank_bits"]["64"] and m["bank_bits"]["16"] > 0
        checks[f"{item}:duration"] = 60.0 < m["video_duration_s"] < 7200.0
        for v in m["summary"]:
            key = v.replace("@", "_")
            checks[f"{item}:{v}:finite"] = bool(np.isfinite(a[f"{key}_lpips_r"][hand]).all() and np.isfinite(a[f"{key}_dists_r"][hand]).all())
            checks[f"{item}:{v}:draws"] = m["summary"][v]["covered_share"] > (0.5 if v == "warp" else 0.1)
        checks[f"{item}:coding_shows"] = all(m["summary"][f"bank2c@{k}"]["lpips_r"] != m["summary"][f"free2c@{k}"]["lpips_r"] for k in KS)
        if settings["frames"] == m["k1f"]["frames"]:
            checks[f"{item}:warp_reproduces_k1f"] = abs(m["summary"]["warp"]["lpips_r"] - m["k1f"]["warp"]["lpips_r"]) <= WARP_TOLERANCE
    return checks


def command_validate(args: argparse.Namespace) -> int:
    stage = Path(args.stage) if args.stage else stage_dir()
    result = json.loads((stage / "k6bh2c.json").read_text())
    checks = {k: bool(v) for k, v in validate(stage, result).items()}
    report = {"passed": all(checks.values()), "checks": checks}
    if target := os.environ.get("PS_VALIDATION_PATH"):
        write_json(Path(target), report)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


# ----------------------------------------------------------------- report (CPU)


def won(value: float, hull: list[tuple[float, float]], kbps: float) -> tuple[bool, float | None, str]:
    """Whether ``value`` (lower is better) beats the hull at ``kbps``: (won, the hull's value there, how it was read)."""
    from experiments.visor.k1 import at_rate

    if kbps > hull[-1][0]:
        return False, None, "above the hull's highest rate (lost)"
    q = at_rate(hull, kbps)
    if q is None:  # below the hull's lowest rate: its lowest point is better than the hull there
        return bool(value < -hull[0][1]), -hull[0][1], "below the hull's lowest rate (its lowest point)"
    return bool(value < -q), -q, "on the hull"


def score_items(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    if not rows:
        return {"items": 0}
    lp = float(np.mean([r[variant]["lpips_won"] for r in rows]))
    di = float(np.mean([r[variant]["dists_won"] for r in rows]))
    return {"items": len(rows), "lpips_share": lp, "dists_share": di, "score": (lp + di) / 2, "passes": (lp + di) / 2 >= CRITERION}


def command_report(args: argparse.Namespace) -> int:
    from experiments.background.g5 import envelope
    from experiments.visor.h2 import load_units
    from experiments.visor.k1f import G_VIDEOS

    hulls = {m["item"]: m for m, _ in load_units(args.k1f_stage, "k1f")}
    rows: list[dict[str, Any]] = []
    sources = []
    variants: list[str] = []
    for stage in map(Path, args.stage):
        result = json.loads((stage / "k6bh2c.json").read_text())
        checks = validate(stage, result)
        if not all(checks.values()) or result["settings"]["frames"] != 240:
            raise SystemExit(f"{stage} fails validation or is not a full stage: {[k for k, v in checks.items() if not v]}")
        sources.append({"stage": str(stage), "sha256": file_sha256(stage / "published.tar")})
        for m, _ in load_units([str(stage / "published.tar")], "k6bh2c"):
            kp = m["k1f"]["kbps"]
            variants = list(m["summary"])
            row: dict[str, Any] = {"item": m["item"], "video": m["video"], "kbps_keypoints": kp, "g_overlap": m["video"] in G_VIDEOS,
                                   "video_duration_s": m["video_duration_s"], "bound": m["k1f"]["warp"], "near_share": m["near_share"],
                                   "candidates": m["candidates"], "bank_bits": m["bank_bits"]}
            for v in variants:
                k = v.split("@")[1] if "@" in v else None
                billed = v.startswith("bank2c")
                bits = m["bank_bits"][k] if billed and k else 0
                rate = kp + bits / m["video_duration_s"] / 1000
                row[v] = {"kbps": rate, "bank_kbps": rate - kp, "kbps_bank_per_minute": kp + bits / 60 / 1000 if billed else None,
                          **{f: m["summary"][v].get(f) for f in ("pose_distance_mean", "matched_near_share", "references_used")}}
                for metric in ("lpips", "dists"):
                    hull = envelope([(p["kbps"], -p[metric]) for p in hulls[m["item"]]["points"]])
                    value = m["summary"][v][f"{metric}_r"]
                    ok, at, how = won(value, hull, rate)
                    row[v].update({metric: value, f"{metric}_hull": at, f"{metric}_won": ok, f"{metric}_read": how})
                    if billed:
                        row[v][f"{metric}_payback"] = payback_seconds(bits, kp, hull, value)
            rows.append(row)
    groups = {"all": rows, "clean_of_g": [r for r in rows if not r["g_overlap"]]}
    scores = {g: {v: score_items(rs, v) for v in variants} for g, rs in groups.items()}
    billed_pass = [k for k in KS if scores["all"][f"bank2c@{k}"]["passes"]]
    free_pass = [k for k in KS if scores["all"][f"free2c@{k}"]["passes"]]
    decision = ("pass" if billed_pass else "fail: the bank's bits" if free_pass else "fail")
    out = {"inputs": {"stages": sources, "k1f_stages": [file_sha256(Path(t)) for t in args.k1f_stage], "code_sha256": file_sha256(Path(__file__))},
           "decision_rule": BANK2C_DECISION, "rows": rows, "groups": scores,
           "decision": {"outcome": decision, "billed_passing_k": billed_pass, "free_passing_k": free_pass}}
    write_json(Path(args.out), out)
    for r in rows:
        print(r["item"], f"kp {r['kbps_keypoints']:.2f} kbps", f"{r['video_duration_s']:.0f} s")
        for v in variants:
            x = r[v]
            print(f"  {v:10s} {x['kbps']:6.2f} kbps  L {x['lpips']:.3f}/{x['lpips_hull'] if x['lpips_hull'] is None else round(x['lpips_hull'], 3)}"
                  f" {'W' if x['lpips_won'] else '-'}  D {x['dists']:.3f}/{x['dists_hull'] if x['dists_hull'] is None else round(x['dists_hull'], 3)}"
                  f" {'W' if x['dists_won'] else '-'}")
    print(json.dumps(scores["all"], indent=1), out["decision"])
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    for flag in ("--eval-set", "--archive", "--masks", "--mask-record", "--items", "--lpips-backbone", "--dists-backbone",
                 "--wilor-checkpoint", "--mano-right", "--mano-left"):
        run.add_argument(flag, required=True)
    run.add_argument("--video", action="append", default=[])
    run.add_argument("--k1f-stage", action="append", required=True, help="K1f's published.tar (rows, bound and hull)")
    run.add_argument("--frames", default="240")
    run.add_argument("--stride", default="5", help="every stride-th VISOR frame outside the window is a candidate")
    run.add_argument("--encoder", default="SvtAv1EncApp")
    run.add_argument("--decoder", default="dav1d")
    run.add_argument("--gpu-class", default="RTX 6000 Ada", help="the GPU class the smoke asserts (model-GPU table)")
    val = sub.add_parser("validate")
    val.add_argument("--stage", default="")
    rep = sub.add_parser("report")
    rep.add_argument("--stage", action="append", required=True)
    rep.add_argument("--k1f-stage", action="append", required=True)
    rep.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return {"run": command_run, "validate": command_validate, "report": command_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
