"""PLAN step G2: the background evaluation protocol and its baselines.

    python -m experiments.background.g2 regions --inputs DIR --select all|ID,ID --source NAME PATH ... \\
        --lpips-backbone PTH [--limit-frames N]
    python -m experiments.background.g2 run --codec svtav1|dcvc --inputs DIR --regions DIR --select ... \\
        --source NAME PATH ... --points P,P --variants background[,frame] --spans E[,E2] --lpips-backbone PTH \\
        [--structure htl --image-ckpt PTH --video-ckpt PTH] [--workers N --threads N] [--keep-streams]
    python -m experiments.background.g2 validate
    python -m experiments.background.g2 choose --result g2.json ...
    python -m experiments.background.g2 report --regions-result g2-regions.json --result g2.json ... --out DIR

The protocol is fixed in docs/experiments.md (2026-10-09, background
evaluation protocol); this module implements it. Inputs are the tar of
``tools/datasets/g2_inputs.py`` (extracted): ``clips.json`` with each clip's
warm-up W and spans, G1's SAM 3.1 masks, the ball labels, and OpenTTGames' own
masks. Crowd zones and caption graphics are the hand annotations in
``g2_annotations.json`` next to this file.

``regions`` (CPU), per clip: the foreground F at every analysis frame of the
coded range (players, rackets, dilated; the ball disc is added from the labels
wherever F is used), the clip's plate (per-pixel median of the visible samples
at 2 frames/s, in the source's own 4:2:0 planes), the warm-up plate, the
shadows S on every scored frame, the still-camera check, the check of SAM's
players against OpenTTGames' masks, and the zero-rate static ceiling (the
warm-up plate scored on E as is and with a per-frame gain and offset). The
stage's ``publish/`` is the regions archive every codec job stages.

``run``, per clip: the source frames of the coded range (2 s lead-in before W,
or the clip start, to the end of the last span) as 4:2:0 planes; for the
*background* input the same frames with F filled from the plate (between
analysis frames, the union of the neighbouring analysis frames' F); each
point coded and decoded (SVT-AV1 4.2.0 preset 4, or DCVC-UF HT-L), bytes
attributed to frames (IVF temporal units; DCVC NALs spread over their
frames), and every scored frame of each span scored against the source on
V and its regions (PSNR from exact error sums, LPIPS map means). Each finished
clip is saved to ``PS_CHECKPOINT_DIR`` and restored on a declared resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from experiments.background.g1 import restore_clips, safe, save_clip
from experiments.visor.b1 import file_sha256, stage_dir, write_json
from src.codecs import quality, svtav1

ANNOTATIONS = Path(__file__).with_name("g2_annotations.json")
ANALYSIS_SIZE = (960, 540)
#: Foreground dilation in analysis pixels (32 px at 1080p).
DILATE = 16
#: Ball disc radius as a share of the frame width (12 px at 1080p).
BALL_RADIUS = 1.0 / 160.0
PLATE_FPS = 2.0
SHADOW = {"ratio_low": 0.2, "ratio_high": 0.92, "chroma_levels": 10, "band_of_height": 0.5, "min_pixels": 20}
STILL_MAX_PX = 1.0
OTT_HUMAN_RGB = (0, 255, 0)
OTT_MASK_SIZE = (320, 128)
REGIONS = ("V", "P", "C", "G", "S")
WEIGHT_BG = 0.3
#: Acceptable loss (protocol): PSNR_V and whole-frame LPIPS.
ACCEPT = {"psnr_v_db": 0.33, "lpips_frame": 0.002}
STREAM_NAMES = {"svtav1": "stream.ivf", "dcvc": "stream.psdc"}
PRESET = 4
#: Rate-point rule (docs/experiments.md, G2 run entry).
CHOOSE = {"targets": 6, "cap_db": 45.0, "min_range_db": 3.0, "min_points": 4}


def log(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} [{os.getpid()}] {message}", file=sys.stderr, flush=True)


# ----------------------------------------------------------------- clips and sources


def load_clips(inputs: Path) -> dict[str, Any]:
    return json.loads((inputs / "clips.json").read_text())


def select(doc: dict[str, Any], which: str) -> list[dict[str, Any]]:
    clips = doc["clips"]
    if which == "all":
        return clips
    if which == "pilot":
        which = ",".join(doc["pilot"])
    by_id = {c["id"]: c for c in clips}
    missing = [w for w in which.split(",") if w not in by_id]
    if missing:
        raise SystemExit(f"unknown clips: {missing}")
    return [by_id[w] for w in which.split(",")]


def analysis_indices(clip: dict[str, Any]) -> list[int]:
    """G1's analysis frames: the index of G1's mask frame k is the k-th entry."""
    fps, rate = float(clip["fps"]), float(clip["analysis"]["fps"])
    first, last = int(clip["first_index"]), int(clip["last_index"])
    out, k = [], 0
    while (at := first + int(round(k * fps / rate))) <= last:
        out.append(at)
        k += 1
    return out


def frame_size(clip: dict[str, Any]) -> tuple[int, int]:
    width, height = json.loads(ANNOTATIONS.read_text())["clips"][clip["id"]]["size"]
    return int(width), int(height)


def full_range(clip: dict[str, Any]) -> bool:
    """OpenTTGames is limited-range H.264 (planes as decoded); TrackNet JPEGs are converted to full range."""
    return clip["source"]["kind"] == "jpeg_dir"


def coded_range(clip: dict[str, Any], spans: list[str]) -> tuple[int, int]:
    frames = clip["g2"]["frames"]
    last = max(frames[s][1] for s in spans if s in frames)
    return int(frames["lead"]), int(last)


def scored(clip: dict[str, Any], span: str) -> list[int]:
    if span not in clip["g2"]["frames"]:
        return []
    a, b = clip["g2"]["frames"][span]
    return [i for i in analysis_indices(clip) if a <= i <= b]


def rgb_to_yuv420(rgb: np.ndarray, *, full: bool = True) -> np.ndarray:
    """8-bit RGB (h, w, 3) to BT.709 4:2:0 planes (h*3/2, w), chroma averaged over 2x2, rounded."""
    kr, kb = quality.KR, quality.KB
    x = rgb.astype(np.float64) / 255.0
    y = kr * x[..., 0] + (1 - kr - kb) * x[..., 1] + kb * x[..., 2]
    cb = (x[..., 2] - y) / (2 * (1 - kb))
    cr = (x[..., 0] - y) / (2 * (1 - kr))
    if full:
        planes = (y * 255.0, cb * 255.0 + 128.0, cr * 255.0 + 128.0)
    else:
        planes = (16.0 + y * 219.0, cb * 224.0 + 128.0, cr * 224.0 + 128.0)
    h, w = y.shape
    luma = planes[0]
    chroma = [p.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3)) for p in planes[1:]]
    out = np.concatenate([luma, chroma[0].reshape(h // 4, w), chroma[1].reshape(h // 4, w)], axis=0)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def source_frames(clip: dict[str, Any], sources: dict[str, Path], indices: list[int],
                  threads: int) -> Iterator[tuple[int, np.ndarray]]:
    """(index, 4:2:0 planes) for the wanted source indices, in order."""
    kind = clip["source"]["kind"]
    wanted = sorted(set(indices))
    if kind == "video":
        from src.segmentation import visor

        width, height = frame_size(clip)
        last = -1
        want = set(wanted)
        for at, frame in visor.decoded_frames(sources[clip["source"]["name"]], wanted, threads=threads):
            if at <= last:
                raise RuntimeError(f"{clip['id']}: decoded frame {at} after {last}")
            last = at
            if at in want:
                if frame.format.name != "yuv420p" or (frame.width, frame.height) != (width, height):
                    raise RuntimeError(f"{clip['id']}: frame {at} is {frame.format.name} {frame.width}x{frame.height}")
                yield at, frame.to_ndarray(format="yuv420p")
    elif kind == "jpeg_dir":
        import cv2

        root = sources[clip["source"]["name"]] / clip["source"]["member_dir"]
        names = sorted(p for p in root.iterdir() if p.suffix == ".jpg")
        if len(names) != clip["source"]["frames"]:
            raise RuntimeError(f"{clip['id']}: {len(names)} JPEGs, expected {clip['source']['frames']}")
        for index in wanted:
            bgr = cv2.imread(str(names[index]))
            yield index, rgb_to_yuv420(bgr[:, :, ::-1], full=True)
    else:
        raise ValueError(f"unknown source kind {kind}")


def write_source(clip: dict[str, Any], sources: dict[str, Path], first: int, last: int, out: Path,
                 threads: int) -> dict[str, Any]:
    """Raw 4:2:0 frames ``first..last`` to ``out`` (runs in a fresh process: PyAV stalls next to torchvision)."""
    began = time.time()
    count, digest = 0, hashlib.sha256()
    expected = first
    with out.open("xb") as handle:
        for at, planes in source_frames(clip, sources, list(range(first, last + 1)), threads):
            if at != expected:
                raise RuntimeError(f"{clip['id']}: got frame {at}, expected {expected}")
            data = planes.tobytes()
            handle.write(data)
            digest.update(data)
            count += 1
            expected += 1
    if count != last - first + 1:
        raise RuntimeError(f"{clip['id']}: wrote {count} of {last - first + 1} frames")
    width, height = frame_size(clip)
    return {"path": str(out), "first": first, "last": last, "frames": count, "width": width, "height": height,
            "full_range": full_range(clip), "sha256": digest.hexdigest(), "seconds": round(time.time() - began, 2)}


def in_fresh_process(fn: Any, *args: Any) -> Any:
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        return pool.submit(fn, *args).result()


# ----------------------------------------------------------------- regions


def annotation(clip: dict[str, Any]) -> dict[str, Any]:
    return json.loads(ANNOTATIONS.read_text())["clips"][clip["id"]]


def boxes_mask(boxes: list[list[int]], width: int, height: int, scale: float = 1.0) -> np.ndarray:
    out = np.zeros((height, width), bool)
    for x0, y0, x1, y1 in boxes:
        out[int(round(y0 * scale)):int(round(y1 * scale)), int(round(x0 * scale)):int(round(x1 * scale))] = True
    return out


def captions_at(clip: dict[str, Any], index: int) -> list[list[int]]:
    return [c["box"] for c in annotation(clip)["captions"]
            if c["frames"] is None or c["frames"][0] <= index <= c["frames"][1]]


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    import cv2

    size = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def foreground_analysis(instances: list[Any], crowd: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]], int]:
    """Players and rackets of one analysis frame: (undilated union, players with mask and box, persons in the zone)."""
    width, height = ANALYSIS_SIZE
    union = np.zeros((height, width), bool)
    players, in_zone = [], 0
    for inst in instances:
        mask = inst.mask().astype(bool)
        if mask.shape != (height, width):
            raise RuntimeError(f"mask shape {mask.shape}, expected {(height, width)}")
        if inst.class_name == "person":
            ys, xs = np.nonzero(mask)
            if not len(xs):
                continue
            cy, cx = int(round(ys.mean())), int(round(xs.mean()))
            if crowd[min(cy, height - 1), min(cx, width - 1)]:
                in_zone += 1
                continue
            players.append({"mask": mask, "box": (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)})
            union |= mask
        elif inst.class_name == "racket":
            union |= mask
    return union, players, in_zone


def upscale(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    import cv2

    if mask.shape == (height, width):
        return mask
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST) > 0


def ball_disc(ball: dict[str, list[float]], index: int, width: int, height: int) -> np.ndarray | None:
    import cv2

    at = ball.get(str(index))
    if at is None:
        return None
    out = np.zeros((height, width), np.uint8)
    radius = max(1, int(round(width * BALL_RADIUS)))
    cv2.circle(out, (int(round(at[0])), int(round(at[1]))), radius, (1,), thickness=-1)
    return out > 0


def chroma_mask(mask: np.ndarray) -> np.ndarray:
    """A full-resolution mask on the 4:2:0 chroma grid: any of the 2x2 pixels."""
    h, w = mask.shape
    return mask.reshape(h // 2, 2, w // 2, 2).any(axis=(1, 3))


def chroma_all(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    return mask.reshape(h // 2, 2, w // 2, 2).all(axis=(1, 3))


class Regions:
    """A clip's published regions, turned into full-resolution masks per frame."""

    def __init__(self, clip: dict[str, Any], directory: Path, inputs: Path) -> None:
        data = np.load(directory / "regions.npz")
        self.clip = clip
        self.width, self.height = frame_size(clip)
        self.indices = [int(i) for i in data["indices"]]
        self.position = {index: k for k, index in enumerate(self.indices)}
        self.fill = np.unpackbits(data["fill"], axis=1, count=ANALYSIS_SIZE[0] * ANALYSIS_SIZE[1]).astype(bool)
        self.scored = [int(i) for i in data["scored"]]
        self.shadow_at = {index: k for k, index in enumerate(self.scored)}
        self.shadow = np.unpackbits(data["shadow"], axis=1, count=ANALYSIS_SIZE[0] * ANALYSIS_SIZE[1]).astype(bool)
        self.ball = json.loads((inputs / "labels" / safe(clip["id"]) / "ball.json").read_text())
        self.crowd_zone = boxes_mask(annotation(clip)["crowd"], self.width, self.height)

    def fill_analysis(self, index: int) -> np.ndarray:
        return self.fill[self.position[index]].reshape(ANALYSIS_SIZE[1], ANALYSIS_SIZE[0])

    def foreground(self, index: int) -> np.ndarray:
        """F at an analysis frame (dilated players and rackets, plus the ball disc), full resolution."""
        out = upscale(self.fill_analysis(index), self.width, self.height).copy()
        disc = ball_disc(self.ball, index, self.width, self.height)
        if disc is not None:
            out |= disc
        return out

    def fill_mask(self, index: int) -> np.ndarray:
        """What the background input fills at any coded frame: F of the neighbouring analysis frames, plus the ball."""
        if index in self.position:
            return self.foreground(index)
        before = [i for i in self.indices if i < index]
        after = [i for i in self.indices if i > index]
        if not before and not after:
            raise RuntimeError(f"{self.clip['id']}: no analysis frame near {index}")
        out = np.zeros((self.height, self.width), bool)
        for near in ([before[-1]] if before else []) + ([after[0]] if after else []):
            out |= upscale(self.fill_analysis(near), self.width, self.height)
        disc = ball_disc(self.ball, index, self.width, self.height)
        if disc is not None:
            out |= disc
        return out

    def masks(self, index: int) -> dict[str, np.ndarray]:
        """V and its regions at a scored frame."""
        visible = ~self.foreground(index)
        caption = boxes_mask(captions_at(self.clip, index), self.width, self.height) & visible
        crowd = self.crowd_zone & visible & ~caption
        shadow = upscale(self.shadow[self.shadow_at[index]].reshape(ANALYSIS_SIZE[1], ANALYSIS_SIZE[0]),
                         self.width, self.height) & visible & ~crowd & ~caption
        plain = visible & ~crowd & ~caption & ~shadow
        return {"V": visible, "P": plain, "C": crowd, "G": caption, "S": shadow}


def median_plate(frames: np.ndarray, masks: np.ndarray, width: int, height: int,
                 full: bool) -> tuple[np.ndarray, float]:
    """Per-pixel median of the unmasked samples of (n, h*3/2, w) planes; unseen pixels inpainted.

    Returns the plate and the share of luma pixels never seen.
    """
    import warnings

    import cv2

    warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)  # never-seen pixels: inpainted below
    n = frames.shape[0]
    luma = np.empty((height, width), np.float32)
    chroma = np.empty((2, height // 2, width // 2), np.float32)
    step = 64
    for r in range(0, height, step):
        rows = slice(r, min(r + step, height))
        block = frames[:, rows].astype(np.float32)
        block[masks[:, rows]] = np.nan
        luma[rows] = np.nanmedian(block, axis=0) if n else np.nan
    cmask = np.stack([chroma_mask(m) for m in masks]) if n else np.zeros((0, height // 2, width // 2), bool)
    for p in range(2):
        plane = frames[:, height + p * height // 4:height + (p + 1) * height // 4].reshape(n, height // 2, width // 2)
        block = plane.astype(np.float32)
        block[cmask] = np.nan
        chroma[p] = np.nanmedian(block, axis=0)
    unseen = float(np.isnan(luma).mean())
    planes = []
    for plane in (luma, chroma[0], chroma[1]):
        holes = np.isnan(plane)
        filled = np.where(holes, 0, plane)
        image = np.clip(np.round(filled), 0, 255).astype(np.uint8)
        if holes.any():
            image = cv2.inpaint(image, holes.astype(np.uint8), 5, cv2.INPAINT_TELEA)
        planes.append(image)
    out = np.concatenate([planes[0], planes[1].reshape(height // 4, width), planes[2].reshape(height // 4, width)])
    return out, unseen


def analysis_planes(planes: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Y, Cb, Cr of 4:2:0 planes resized to the analysis grid (area)."""
    import cv2

    y = planes[:height]
    cb = planes[height:height + height // 4].reshape(height // 2, width // 2)
    cr = planes[height + height // 4:].reshape(height // 2, width // 2)
    size = ANALYSIS_SIZE
    return tuple(cv2.resize(p, size, interpolation=cv2.INTER_AREA).astype(np.float32) for p in (y, cb, cr))  # type: ignore[return-value]


def shadow_mask(frame: tuple[np.ndarray, ...], plate: tuple[np.ndarray, ...], players: list[dict[str, Any]],
                excluded: np.ndarray, full: bool) -> np.ndarray:
    """Shadow candidates on the analysis grid (protocol: darker than the plate near a player, same chroma)."""
    import cv2

    width, height = ANALYSIS_SIZE
    band = np.zeros((height, width), bool)
    for player in players:
        distance = cv2.distanceTransform((~player["mask"]).astype(np.uint8), cv2.DIST_L2, 3)
        band |= distance <= SHADOW["band_of_height"] * (player["box"][3] - player["box"][1])
    black = 0.0 if full else 16.0
    ratio = (frame[0] - black) / np.maximum(plate[0] - black, 1.0)
    chroma_ok = (np.abs(frame[1] - plate[1]) <= SHADOW["chroma_levels"]) & (np.abs(frame[2] - plate[2]) <= SHADOW["chroma_levels"])
    candidate = band & ~excluded & chroma_ok & (ratio >= SHADOW["ratio_low"]) & (ratio <= SHADOW["ratio_high"])
    opened = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= SHADOW["min_pixels"]
    return keep[labels]


def gain_offset(reference: np.ndarray, plate: np.ndarray, plain: np.ndarray, width: int,
                height: int) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """The plate with a per-plane gain and offset fitted by least squares on the plain region, and the fits."""
    out = plate.astype(np.float64).copy()
    fits: list[tuple[float, float]] = []
    regions = [(slice(0, height), plain),
               (slice(height, height + height // 4), chroma_all(plain)),
               (slice(height + height // 4, height * 3 // 2), chroma_all(plain))]
    for rows, mask in regions:
        ref = reference[rows].reshape(mask.shape).astype(np.float64)
        src = plate[rows].reshape(mask.shape).astype(np.float64)
        if mask.sum() < 100:
            fits.append((1.0, 0.0))
            continue
        a, b = np.polyfit(src[mask], ref[mask], 1)
        out[rows] = (a * src + b).reshape(out[rows].shape)
        fits.append((float(a), float(b)))
    return np.clip(np.round(out), 0, 255).astype(np.uint8), fits


def region_scores(per_frame: list[dict[str, Any]]) -> dict[str, Any]:
    """Means over frames of each region's PSNR and LPIPS, pooled PSNR, share of V, and the acceptable-loss terms."""
    out: dict[str, Any] = {}
    for region in (*REGIONS, "frame"):
        values = [f[region] for f in per_frame if f[region]["pixels"]]
        if not values:
            out[region] = None
            continue
        sse = sum(v["sse"] for v in values)
        pixels = sum(v["pixels"] for v in values)
        out[region] = {
            "psnr": float(np.mean([v["psnr"] for v in values])),
            "psnr_pooled": quality.psnr(sse / (3 * pixels)),
            "lpips": float(np.mean([v["lpips"] for v in values])),
            "frames": len(values),
            "share_of_v": float(np.mean([f[region]["pixels"] / f["V"]["pixels"] for f in per_frame if f["V"]["pixels"]])),
        }
    out["excess"] = excess(per_frame)
    return out


def excess(per_frame: list[dict[str, Any]]) -> dict[str, Any]:
    """Protocol's acceptable-loss terms per frame, averaged: crowd with graphics, and shadows."""
    terms: dict[str, list[tuple[float, float]]] = {"crowd_graphics": [], "shadows": []}
    for f in per_frame:
        plain, visible = f["P"], f["V"]
        if not plain["pixels"] or not visible["pixels"]:
            continue
        per_px = plain["sse"] / plain["pixels"]
        lp_px = plain["lpips"]
        for name, parts in (("crowd_graphics", ("C", "G")), ("shadows", ("S",))):
            pixels = sum(f[p]["pixels"] for p in parts)
            if not pixels:
                continue
            sse_actual = visible["sse"]
            sse_counter = sse_actual - sum(f[p]["sse"] for p in parts) + per_px * pixels
            d_psnr = 10 * math.log10(max(sse_actual, 1) / max(sse_counter, 1))
            lp_actual = sum(f[p]["lpips"] * f[p]["pixels"] for p in parts if f[p]["pixels"])
            d_lpips_frame = (lp_actual - lp_px * pixels) / f["frame"]["pixels"]
            terms[name].append((d_psnr, d_lpips_frame))
    out: dict[str, Any] = {}
    for name, values in terms.items():
        if not values:
            out[name] = None
            continue
        d_psnr = float(np.mean([v[0] for v in values]))
        d_lp = float(np.mean([v[1] for v in values]))
        out[name] = {"psnr_v_db": d_psnr, "weighted_psnr_db": WEIGHT_BG * d_psnr, "lpips_frame": d_lp,
                     "frames": len(values),
                     "acceptable": d_psnr <= ACCEPT["psnr_v_db"] and d_lp <= ACCEPT["lpips_frame"]}
    return out


def with_source_foreground(reference: np.ndarray, output: np.ndarray, visible: np.ndarray) -> np.ndarray:
    """The output with F (not V) taken from the source, so F cannot reach V's LPIPS through its receptive field.

    Chroma samples are replaced only where all four of their pixels are in F, so no V pixel changes.
    """
    height = visible.shape[0]
    fg = ~visible
    out = output.copy()
    out[:height][fg] = reference[:height][fg]
    cfg = chroma_all(fg)
    for p in range(2):
        rows = slice(height + p * height // 4, height + (p + 1) * height // 4)
        view = out[rows].reshape(cfg.shape)
        view[cfg] = reference[rows].reshape(cfg.shape)[cfg]
        out[rows] = view.reshape(out[rows].shape)
    return out


def score_frames(reference: np.ndarray, output: np.ndarray, masks: list[dict[str, np.ndarray]], clip: dict[str, Any],
                 device: str, net: Any, batch: int = 4) -> list[dict[str, Any]]:
    """Every region's scores; the output is first given the source's foreground (`with_source_foreground`)."""
    width, height = frame_size(clip)
    out: list[dict[str, Any]] = []
    for first in range(0, len(masks), batch):
        part = masks[first:first + batch]
        stacked = {name: np.stack([m[name] for m in part]) for name in REGIONS}
        ref = reference[first:first + batch]
        composite = np.stack([with_source_foreground(ref[k], output[first + k], part[k]["V"]) for k in range(len(part))])
        out.extend(quality.score_batch(ref, composite, stacked, width=width, height=height,
                                       full_range=full_range(clip), device=device, lpips_net=net))
    return out


_LPIPS: dict[str, Any] = {}


def lpips_net(backbone: str, device: str) -> Any:
    key = f"{backbone}|{device}"
    if key not in _LPIPS:
        _LPIPS[key] = quality.load_lpips(Path(backbone), device)
    return _LPIPS[key]


def regions_clip(args: argparse.Namespace, clip: dict[str, Any], work: Path, out: Path) -> dict[str, Any]:
    """Regions, plates, checks and the static ceiling of one clip."""
    import cv2
    import torch

    from src.segmentation.masks import ClipMasks

    torch.set_num_threads(max(1, args.threads))
    cv2.setNumThreads(max(1, args.threads))
    began = time.time()
    inputs = Path(args.inputs)
    width, height = frame_size(clip)
    full = full_range(clip)
    indices = analysis_indices(clip)
    masks = ClipMasks.load(inputs / "masks" / safe(clip["id"]) / "masks.rle")
    if len(masks.frames) != len(indices):
        raise RuntimeError(f"{clip['id']}: {len(masks.frames)} mask frames, {len(indices)} analysis frames")
    scale = ANALYSIS_SIZE[0] / width
    crowd_a = boxes_mask(annotation(clip)["crowd"], *ANALYSIS_SIZE, scale=scale)
    ball = json.loads((inputs / "labels" / safe(clip["id"]) / "ball.json").read_text())
    fill_a, players_at, zone_persons = {}, {}, 0
    for k, index in enumerate(indices):
        union, players, in_zone = foreground_analysis(masks.frames[k], crowd_a)
        fill_a[index] = dilate(union, DILATE)
        players_at[index] = players
        zone_persons += in_zone

    def foreground(index: int) -> np.ndarray:
        f = upscale(fill_a[index], width, height).copy()
        disc = ball_disc(ball, index, width, height)
        if disc is not None:
            f |= disc
        return f

    # Plates: the clip's (2 frames/s over the analysed range) and the warm-up's ([0, W) at 10 frames/s).
    stride = int(round(float(clip["analysis"]["fps"]) / PLATE_FPS))
    plate_idx = indices[::stride]
    warm_end = int(clip["first_index"]) + int(round(clip["g2"]["warmup_s"] * float(clip["fps"])))
    warm_idx = [i for i in indices if i < warm_end] or indices[:1]
    spans = {s: scored(clip, s) for s in ("E", "E2")}
    if args.limit_frames:
        spans = {s: v[:args.limit_frames] for s, v in spans.items()}
        plate_idx = plate_idx[:max(args.limit_frames, 4)]
    wanted = sorted(set(plate_idx) | set(warm_idx) | set(spans["E"]) | set(spans["E2"]))
    log(f"{clip['id']}: decoding {len(wanted)} frames")
    frames_path = work / "frames.npy"
    store = np.lib.format.open_memmap(frames_path, mode="w+", dtype=np.uint8, shape=(len(wanted), height * 3 // 2, width))
    at = {index: k for k, index in enumerate(wanted)}
    done = in_fresh_process(_decode_into, clip, {k: str(v) for k, v in args.sources.items()}, wanted, str(frames_path),
                            args.threads)
    if done != len(wanted):
        raise RuntimeError(f"{clip['id']}: decoded {done} of {len(wanted)} frames")
    store = np.load(frames_path, mmap_mode="r")
    log(f"{clip['id']}: plates")
    plate, unseen = median_plate(store[[at[i] for i in plate_idx]], np.stack([foreground(i) for i in plate_idx]),
                                 width, height, full)
    warm, warm_unseen = median_plate(store[[at[i] for i in warm_idx]], np.stack([foreground(i) for i in warm_idx]),
                                     width, height, full)
    plate_a = analysis_planes(plate, width, height)
    # Shadows, the still-camera check, and SAM against OpenTTGames' masks.
    log(f"{clip['id']}: shadows and checks")
    shadow_a: dict[int, np.ndarray] = {}
    shifts = []
    window = cv2.createHanningWindow(ANALYSIS_SIZE, cv2.CV_32F)
    for index in spans["E"] + spans["E2"]:
        frame_a = analysis_planes(np.asarray(store[at[index]]), width, height)
        excluded = fill_a[index] | crowd_a | boxes_mask(captions_at(clip, index), *ANALYSIS_SIZE, scale=scale)
        shadow_a[index] = shadow_mask(frame_a, plate_a, players_at[index], excluded, full)
        (dx, dy), _ = cv2.phaseCorrelate(frame_a[0], plate_a[0], window)
        shifts.append(math.hypot(dx, dy) / scale)
    iou = ott_iou(inputs, clip, indices, masks, crowd_a)
    # Static ceiling: the warm-up plate on E, as is and with a per-frame gain and offset.
    log(f"{clip['id']}: static ceiling")
    device = "cuda" if args.device == "cuda" else "cpu"
    net = lpips_net(args.lpips_backbone, device)
    e_idx = spans["E"]
    region_masks = [regions_from(clip, foreground(i), shadow_a[i], i, width, height) for i in e_idx]
    reference = np.stack([store[at[i]] for i in e_idx])
    static = score_frames(reference, np.repeat(warm[None], len(e_idx), axis=0), region_masks, clip, device, net)
    fitted = [gain_offset(reference[k], warm, region_masks[k]["P"], width, height) for k in range(len(e_idx))]
    adjusted_frames = np.stack([f[0] for f in fitted])
    gains = np.array([f[1] for f in fitted])  # (frames, plane, (gain, offset))
    adjusted = score_frames(reference, adjusted_frames, region_masks, clip, device, net)
    # Published regions: F on every analysis frame of the codec range (plus one either side), S on scored frames.
    first, last = int(clip["g2"]["frames"]["lead"]), max(v[1] for v in clip["g2"]["frames"].values() if isinstance(v, list))
    inside = [i for i in indices if first <= i <= last]
    k0, k1 = indices.index(inside[0]), indices.index(inside[-1])
    keep = indices[max(0, k0 - 1):k1 + 2]
    scored_all = spans["E"] + spans["E2"]
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "regions.npz", indices=np.array(keep), scored=np.array(scored_all),
        fill=np.packbits(np.stack([fill_a[i].reshape(-1) for i in keep]), axis=1),
        shadow=np.packbits(np.stack([shadow_a[i].reshape(-1) for i in scored_all]), axis=1)
        if scored_all else np.zeros((0, 0), np.uint8),
    )
    (out / "plate.yuv").write_bytes(plate.tobytes())
    (out / "plate_warmup.yuv").write_bytes(warm.tobytes())
    write_views(out, clip, plate, warm, store[at[e_idx[len(e_idx) // 2]]], region_masks[len(e_idx) // 2], full)
    shares = {r: float(np.mean([m[r].sum() / max(1, m["V"].sum()) for m in region_masks])) for r in REGIONS}
    result = {
        "id": clip["id"], "width": width, "height": height, "full_range": full, "warmup_s": clip["g2"]["warmup_s"],
        "frames": {"plate": len(plate_idx), "warmup": len(warm_idx), "E": len(spans["E"]), "E2": len(spans["E2"]),
                   "published_fill": len(keep)},
        "plate_unseen": unseen, "warmup_plate_unseen": warm_unseen,
        "foreground_share_of_frame": float(np.mean([m["V"].size - m["V"].sum() for m in region_masks]) / (width * height)),
        "share_of_v": shares, "persons_in_zone": zone_persons,
        "still": {"max_shift_px": float(max(shifts)) if shifts else None,
                  "median_shift_px": float(np.median(shifts)) if shifts else None, "frames": len(shifts)},
        "ott_iou": iou,
        "static_ceiling": {"as_is": region_scores(static), "gain_offset": region_scores(adjusted)},
        "gain_offset_fits": {plane: {"gain": [float(gains[:, k, 0].min()), float(np.median(gains[:, k, 0])),
                                              float(gains[:, k, 0].max())],
                                     "offset": [float(gains[:, k, 1].min()), float(np.median(gains[:, k, 1])),
                                                float(gains[:, k, 1].max())]}
                             for k, plane in enumerate("YUV")},
        "static_frames": {"as_is": static, "gain_offset": adjusted},
        "plate_sha256": hashlib.sha256(plate.tobytes()).hexdigest(),
        "warmup_plate_sha256": hashlib.sha256(warm.tobytes()).hexdigest(),
        "regions_sha256": file_sha256(out / "regions.npz"), "lpips_device": device,
        "seconds": round(time.time() - began, 2),
    }
    write_json(out / "result.json", result)
    os.remove(frames_path)
    return {k: v for k, v in result.items() if k != "static_frames"}


def regions_from(clip: dict[str, Any], fg: np.ndarray, shadow_a: np.ndarray, index: int, width: int,
                 height: int) -> dict[str, np.ndarray]:
    visible = ~fg
    caption = boxes_mask(captions_at(clip, index), width, height) & visible
    crowd = boxes_mask(annotation(clip)["crowd"], width, height) & visible & ~caption
    shadow = upscale(shadow_a, width, height) & visible & ~crowd & ~caption
    return {"V": visible, "P": visible & ~crowd & ~caption & ~shadow, "C": crowd, "G": caption, "S": shadow}


def _decode_into(clip: dict[str, Any], sources: dict[str, str], wanted: list[int], path: str, threads: int) -> int:
    store = np.load(path, mmap_mode="r+")
    at = {index: k for k, index in enumerate(wanted)}
    count = 0
    for index, planes in source_frames(clip, {k: Path(v) for k, v in sources.items()}, wanted, threads):
        store[at[index]] = planes
        count += 1
    store.flush()
    return count


def ott_iou(inputs: Path, clip: dict[str, Any], indices: list[int], masks: Any, crowd_a: np.ndarray) -> dict[str, Any] | None:
    import cv2
    from PIL import Image

    directory = inputs / "ott_masks" / safe(clip["id"])
    if not directory.is_dir():
        return None
    values = []
    for k, index in enumerate(indices):
        path = directory / f"{index}.png"
        if not path.is_file():
            continue
        label = np.asarray(Image.open(path).convert("RGB"))
        human = np.all(label == np.array(OTT_HUMAN_RGB, np.uint8), axis=-1)
        union, _, _ = foreground_analysis(masks.frames[k], crowd_a)
        ours = cv2.resize(union.astype(np.float32), OTT_MASK_SIZE, interpolation=cv2.INTER_AREA) >= 0.5
        inter, joint = (ours & human).sum(), (ours | human).sum()
        if joint:
            values.append(float(inter / joint))
    if not values:
        return {"frames": 0}
    return {"frames": len(values), "median": float(np.median(values)), "p10": float(np.percentile(values, 10)),
            "mean": float(np.mean(values))}


def to_rgb(planes: np.ndarray, width: int, height: int, full: bool) -> np.ndarray:
    import torch

    rgb = quality.yuv420_to_rgb(torch.from_numpy(np.ascontiguousarray(planes)[None]), width, height, full_range=full)
    return rgb[0].permute(1, 2, 0).numpy().astype(np.uint8)


def write_views(out: Path, clip: dict[str, Any], plate: np.ndarray, warm: np.ndarray, frame: np.ndarray,
                regions: dict[str, np.ndarray], full: bool) -> None:
    """Plates and one E frame with its regions tinted (F magenta, C cyan, G yellow, S blue), for review by eye."""
    import cv2

    width, height = frame_size(clip)
    cv2.imwrite(str(out / "plate.jpg"), to_rgb(plate, width, height, full)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(out / "plate_warmup.jpg"), to_rgb(warm, width, height, full)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])
    rgb = to_rgb(frame, width, height, full).astype(np.float32)
    tints = {"F": (255, 0, 255), "C": (0, 255, 255), "G": (255, 255, 0), "S": (0, 80, 255)}
    masks = {"F": ~regions["V"], "C": regions["C"], "G": regions["G"], "S": regions["S"]}
    for name, colour in tints.items():
        m = masks[name]
        rgb[m] = 0.5 * rgb[m] + 0.5 * np.array(colour, np.float32)
    cv2.imwrite(str(out / "regions.jpg"), np.clip(rgb, 0, 255).astype(np.uint8)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 90])


def command_regions(args: argparse.Namespace) -> int:
    stage = stage_dir()
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage / "scratch")
    publish = scratch / "publish"
    args.sources = {name: Path(path) for name, path in args.source}
    doc = load_clips(Path(args.inputs))
    clips = select(doc, args.select)
    checkpoints = Path(os.environ["PS_CHECKPOINT_DIR"]) if os.environ.get("PS_CHECKPOINT_DIR") else None
    restored = restore_clips(checkpoints, publish) if checkpoints else {}
    results = []
    for clip in clips:
        if clip["id"] in restored:
            log(f"{clip['id']}: restored from the checkpoint")
            results.append(restored[clip["id"]]["result"])
            continue
        work = scratch / "work" / safe(clip["id"])
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        target = publish / "clips" / safe(clip["id"])
        result = regions_clip(args, clip, work, target)
        results.append(result)
        if checkpoints:
            save_clip(checkpoints, target, clip["id"], result, None)
        shutil.rmtree(work)
    record = {
        "command": "regions", "clips_file": {"name": doc["name"], "sha256": file_sha256(Path(args.inputs) / "clips.json")},
        "annotations_sha256": file_sha256(ANNOTATIONS), "select": args.select, "limit_frames": args.limit_frames,
        "settings": {"dilate_analysis_px": DILATE, "ball_radius_of_width": BALL_RADIUS, "plate_fps": PLATE_FPS,
                     "shadow": SHADOW, "still_max_px": STILL_MAX_PX, "accept": ACCEPT},
        "lpips": quality.lpips_record(Path(args.lpips_backbone)), "results": results,
        "restored": sorted(restored),
    }
    write_json(stage / "g2-regions.json", record)
    # The codec jobs stage ``publish/`` alone, so it carries the record too.
    write_json(publish / "g2-regions.json", {k: v for k, v in record.items() if k != "lpips"})
    return 0


# ----------------------------------------------------------------- codecs


def frame_bytes_svtav1(stream: Path, frames: int) -> list[int]:
    sizes = svtav1.ivf_frames(stream.read_bytes())
    if len(sizes) != frames:
        raise RuntimeError(f"{stream}: {len(sizes)} temporal units for {frames} frames")
    return sizes


def frame_bytes_dcvc(encode_report: dict[str, Any], frames: int) -> list[float]:
    """Each NAL's bytes (and any SPS) spread evenly over the frames it carries."""
    out: list[float] = []
    for nal in encode_report["nals"]:
        share = (nal["nal_bytes"] + nal["sps_bytes"]) / nal["display_frames"]
        out.extend([share] * nal["display_frames"])
    if len(out) != frames:
        raise RuntimeError(f"DCVC-UF NALs carry {len(out)} frames, expected {frames}")
    return out


def write_background(source: Path, out: Path, regions: Regions, plate: np.ndarray, first: int, frames: int,
                     width: int, height: int) -> dict[str, Any]:
    """The source with F filled from the plate, frame by frame."""
    size = quality.frame_bytes(width, height)
    filled_share = []
    with out.open("xb") as handle:
        for k in range(frames):
            planes = quality.read_frames(source, width, height, k, 1)[0].copy()
            fill = regions.fill_mask(first + k)
            cfill = chroma_mask(fill)
            planes[:height][fill] = plate[:height][fill]
            for p in range(2):
                rows = slice(height + p * height // 4, height + (p + 1) * height // 4)
                view = planes[rows].reshape(height // 2, width // 2)
                view[cfill] = plate[rows].reshape(height // 2, width // 2)[cfill]
                planes[rows] = view.reshape(height // 4, width)
            handle.write(planes.tobytes())
            filled_share.append(float(fill.mean()))
    if out.stat().st_size != size * frames:
        raise RuntimeError(f"{out}: wrong size")
    return {"path": str(out), "sha256": file_sha256(out), "fill_share_mean": float(np.mean(filled_share))}


def code_point(args: argparse.Namespace, clip: dict[str, Any], source: dict[str, Any], point: float,
               work: Path, cpus: list[int] | None) -> dict[str, Any]:
    width, height, frames = source["width"], source["height"], source["frames"]
    fps = Fraction(clip["fps"]).limit_denominator(1001)
    if args.codec == "svtav1":
        record = svtav1.code(Path(source["path"]), work, width=width, height=height, fps=fps, frames=frames, crf=point,
                             preset=PRESET, threads=args.threads, full_range=source["full_range"],
                             encoder=args.encoder, decoder=args.decoder, cpus=cpus)
        record["frame_bytes"] = frame_bytes_svtav1(Path(record["stream"]), frames)
        if sum(record["frame_bytes"]) != record["payload_bytes"]:
            raise RuntimeError("temporal units do not sum to the payload")
        record["rate_bytes_total"] = record["payload_bytes"]
        record["decode_seconds_per_frame"] = record["decode_seconds"] / frames
        return record
    from experiments.visor.b2 import code_dcvc

    record = code_dcvc(args, source, point, work, None, profile=False, variant={"range": "full"})
    encode = json.loads((work / "encode.json").read_text())
    record["frame_bytes"] = frame_bytes_dcvc(encode, frames)
    record["rate_bytes_total"] = record["rate_bytes"]
    model = [s for s in record["model_decode_seconds"] if s is not None]
    record["decode_seconds_per_frame"] = (min(model) if model else min(record["decode_seconds"])) / frames
    return record


_SLOT: dict[str, list[int] | None] = {"cpus": None}


def _take_slot(slots: Any) -> None:
    _SLOT["cpus"] = slots.get()


def _point_job(payload: tuple[argparse.Namespace, dict[str, Any], dict[str, Any], str, float, str]) -> dict[str, Any]:
    args, clip, source, variant, point, work = payload
    return run_point(args, clip, source, variant, point, Path(work), _SLOT["cpus"])


def run_point(args: argparse.Namespace, clip: dict[str, Any], sources: dict[str, Any], variant: str, point: float,
              work: Path, cpus: list[int] | None) -> dict[str, Any]:
    """Code one input at one point, and score every scored frame of each span against the source."""
    import torch

    torch.set_num_threads(max(1, args.threads))
    began = time.time()
    source = sources[variant]
    reference = sources["frame"]
    width, height = source["width"], source["height"]
    work.mkdir(parents=True, exist_ok=True)
    coded = code_point(args, clip, source, point, work, cpus)
    t_code = time.time()
    regions = Regions(clip, Path(args.regions) / "clips" / safe(clip["id"]), Path(args.inputs))
    fps = float(clip["fps"])
    first = source["first"]
    device = args.metric_device
    net = lpips_net(args.lpips_backbone, device)
    spans: dict[str, Any] = {}
    frames_out: dict[str, Any] = {}
    for span in args.spans:
        idx = [i for i in scored(clip, span) if first <= i <= source["last"]]
        if not idx:
            continue
        a, b = clip["g2"]["frames"][span]
        offsets = [i - first for i in idx]
        ref = np.stack([quality.read_frames(Path(reference["path"]), width, height, o, 1)[0] for o in offsets])
        dec = np.stack([quality.read_frames(Path(coded["decoded"]), width, height, o, 1)[0] for o in offsets])
        masks = [regions.masks(i) for i in idx]
        per_frame = score_frames(ref, dec, masks, clip, device, net)
        span_bytes = float(sum(coded["frame_bytes"][a - first:b - first + 1]))
        seconds = (b - a + 1) / fps
        spans[span] = {"frames": [a, b], "seconds": seconds, "bytes": span_bytes, "kbps": span_bytes * 8 / seconds / 1000,
                       "scored_frames": len(idx), "scores": region_scores(per_frame)}
        frames_out[span] = {"indices": idx, "per_frame": per_frame}
    kept = None
    if args.keep_streams:
        kept_dir = Path(args.publish) / "clips" / safe(clip["id"]) / "streams" / variant / f"{point:g}"
        kept_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(coded["stream"], kept_dir / STREAM_NAMES[args.codec])
        kept = str(kept_dir.relative_to(Path(args.publish)))
    record = {k: v for k, v in coded.items() if k not in ("stream", "decoded", "frame_bytes")}
    row = {
        "variant": variant, "point": point, "spans": spans, "frame_bytes": coded["frame_bytes"],
        "stream_sha256": coded["stream_sha256"], "decoded_frames": coded["decoded_frames"], "coded_frames": source["frames"],
        "codec_record": record, "kept_stream": kept,
        "seconds": {"code": round(t_code - began, 2), "score": round(time.time() - t_code, 2)},
    }
    shutil.rmtree(work)
    return {"row": row, "frames": frames_out}


def prepare_sources(args: argparse.Namespace, clip: dict[str, Any], work: Path) -> dict[str, Any]:
    first, last = coded_range(clip, args.spans)
    log(f"{clip['id']}: decoding source frames {first}..{last}")
    source = in_fresh_process(write_source, clip, args.sources, first, last, work / "source.yuv", args.threads)
    out = {"frame": source}
    if "background" in args.variants:
        regions = Regions(clip, Path(args.regions) / "clips" / safe(clip["id"]), Path(args.inputs))
        plate_path = Path(args.regions) / "clips" / safe(clip["id"]) / "plate.yuv"
        plate = np.fromfile(plate_path, np.uint8).reshape(source["height"] * 3 // 2, source["width"])
        log(f"{clip['id']}: filling the foreground from the plate")
        filled = write_background(work / "source.yuv", work / "background.yuv", regions, plate, first,
                                  source["frames"], source["width"], source["height"])
        out["background"] = {**source, **filled, "plate_sha256": file_sha256(plate_path)}
    return out


def run_clip(args: argparse.Namespace, clip: dict[str, Any], scratch: Path, pool: Any) -> dict[str, Any]:
    began = time.time()
    work = scratch / "work" / safe(clip["id"])
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    sources = prepare_sources(args, clip, work)
    tasks = [(args, clip, sources, variant, point, str(work / f"{variant}-{point:g}"))
             for variant in args.variants for point in args.point_values]
    results = list(pool.map(_point_job, tasks)) if pool else [_point_job(t) for t in tasks]
    target = Path(args.publish) / "clips" / safe(clip["id"])
    target.mkdir(parents=True, exist_ok=True)
    rows = [r["row"] for r in results]
    write_json(target / "frames.json", {"id": clip["id"], "codec": args.codec,
                                         "points": [{"variant": r["row"]["variant"], "point": r["row"]["point"],
                                                     "frames": r["frames"]} for r in results]})
    result = {
        "id": clip["id"], "dataset": clip["dataset"], "codec": args.codec, "fps": clip["fps"],
        "sources": {k: {kk: vv for kk, vv in v.items() if kk != "path"} for k, v in sources.items()},
        "rows": rows, "seconds": round(time.time() - began, 2),
    }
    write_json(target / "result.json", result)
    shutil.rmtree(work)
    return result


def command_run(args: argparse.Namespace) -> int:
    stage = stage_dir()
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage / "scratch")
    args.publish = str(scratch / "publish")
    args.sources = {name: Path(path) for name, path in args.source}
    args.variants = args.variants.split(",")
    args.spans = args.spans.split(",")
    args.point_values = [float(p) for p in args.points.split(",")]
    if args.codec == "dcvc":
        args.image_sha256 = file_sha256(Path(args.image_ckpt))
        args.video_sha256 = file_sha256(Path(args.video_ckpt))
        if args.workers != 1:
            raise SystemExit("DCVC-UF runs one point at a time on its GPU")
    args.metric_device = args.metric_device or ("cuda" if args.codec == "dcvc" else "cpu")
    if (Path(args.regions) / "publish" / "clips").is_dir():  # the extracted archive's root
        args.regions = str(Path(args.regions) / "publish")
    doc = load_clips(Path(args.inputs))
    clips = select(doc, args.select)
    regions_record = json.loads((Path(args.regions) / "g2-regions.json").read_text()) \
        if (Path(args.regions) / "g2-regions.json").is_file() else None
    checkpoints = Path(os.environ["PS_CHECKPOINT_DIR"]) if os.environ.get("PS_CHECKPOINT_DIR") else None
    restored = restore_clips(checkpoints, Path(args.publish)) if checkpoints else {}
    pool = None
    if args.workers > 1:
        cores = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
        per = max(1, len(cores) // args.workers)
        slots = multiprocessing.get_context("spawn").Queue()
        for w in range(args.workers):
            slots.put(cores[w * per:(w + 1) * per])
        pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                                   initializer=_take_slot, initargs=(slots,))
    results = []
    try:
        for clip in clips:
            if clip["id"] in restored:
                log(f"{clip['id']}: restored from the checkpoint")
                results.append(restored[clip["id"]]["result"])
                continue
            result = run_clip(args, clip, scratch, pool)
            results.append(result)
            if checkpoints:
                save_clip(checkpoints, Path(args.publish) / "clips" / safe(clip["id"]), clip["id"], result, None)
    finally:
        if pool:
            pool.shutdown()
    tools: dict[str, Any] = {}
    if args.codec == "svtav1":
        tools = {"encoder": svtav1.tool(args.encoder), "decoder": svtav1.tool(args.decoder), "preset": PRESET}
    else:
        tools = {"structure": args.structure, "image_ckpt": args.image_ckpt, "image_sha256": args.image_sha256,
                 "video_ckpt": args.video_ckpt, "video_sha256": args.video_sha256}
    write_json(stage / "g2.json", {
        "command": "run", "codec": args.codec, "variants": args.variants, "spans": args.spans,
        "points": args.point_values, "select": args.select, "metric_device": args.metric_device,
        "clips_file": {"name": doc["name"], "sha256": file_sha256(Path(args.inputs) / "clips.json")},
        "annotations_sha256": file_sha256(ANNOTATIONS),
        "regions": {"path": args.regions, "record": regions_record and {
            "clips_file": regions_record["clips_file"], "annotations_sha256": regions_record["annotations_sha256"],
            "regions_sha256": {r["id"]: r["regions_sha256"] for r in regions_record["results"]}}},
        "regions_sha256": {c["id"]: file_sha256(Path(args.regions) / "clips" / safe(c["id"]) / "regions.npz")
                           for c in clips},
        "tools": tools, "lpips": quality.lpips_record(Path(args.lpips_backbone)), "results": results,
        "restored": sorted(restored),
    })
    return 0


# ----------------------------------------------------------------- validate


def finite(value: Any, low: float, high: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and low <= value <= high


def published_names(stage: Path) -> set[str]:
    import tarfile

    path = stage / "published.tar"
    if not path.is_file():
        return set()
    with tarfile.open(path) as tar:
        return set(tar.getnames())


def validate_regions(stage: Path, doc: dict[str, Any]) -> dict[str, bool]:
    results = doc["results"]
    names = published_names(stage)
    wanted = ("regions.npz", "plate.yuv", "plate_warmup.yuv", "result.json", "regions.jpg")
    checks = {
        "every_clip_published": bool(results) and "publish/g2-regions.json" in names
        and all(f"publish/clips/{safe(r['id'])}/{n}" in names for r in results for n in wanted),
        "e_spans_scored": all(r["frames"]["E"] > 0 for r in results),
        "cameras_are_still": all(r["still"]["max_shift_px"] is not None and r["still"]["max_shift_px"] < STILL_MAX_PX
                                 for r in results),
        "plates_nearly_fully_seen": all(r["plate_unseen"] < 0.01 for r in results),
        "region_shares_in_range": all(finite(r["share_of_v"][k], 0.0, 1.0) for r in results for k in REGIONS)
        and all(abs(sum(r["share_of_v"][k] for k in ("P", "C", "G", "S")) - 1.0) < 1e-6 for r in results),
        "foreground_share_plausible": all(finite(r["foreground_share_of_frame"], 0.005, 0.5) for r in results),
        "sam_players_match_openttgames": all(r["ott_iou"] is None or r["ott_iou"].get("frames", 0) == 0
                                             or r["ott_iou"]["median"] >= 0.5 for r in results),
        "static_ceiling_scored": all(finite(r["static_ceiling"]["as_is"]["V"]["psnr"], 5.0, 100.0)
                                     and finite(r["static_ceiling"]["as_is"]["V"]["lpips"], 0.0, 1.0) for r in results),
    }
    return checks


def validate_run(stage: Path, doc: dict[str, Any]) -> dict[str, bool]:
    results = doc["results"]
    expected = len(doc["variants"]) * len(doc["points"])
    rows = [row for r in results for row in r["rows"]]

    def rises(r: dict[str, Any], variant: str) -> bool:
        pts = sorted((row["spans"]["E"]["kbps"], row["spans"]["E"]["scores"]["V"]["psnr_pooled"])
                     for row in r["rows"] if row["variant"] == variant and "E" in row["spans"])
        return all(b[1] >= a[1] - 0.05 for a, b in zip(pts, pts[1:]))

    names = published_names(stage)
    checks = {
        "every_clip_published": all(f"publish/clips/{safe(r['id'])}/{n}" in names
                                    for r in results for n in ("result.json", "frames.json")),
        "every_clip_and_point": bool(results) and all(len(r["rows"]) == expected for r in results),
        "decoded_every_frame": all(row["decoded_frames"] == row["coded_frames"] for row in rows),
        "frame_bytes_cover_every_frame": all(len(row["frame_bytes"]) == row["coded_frames"] for row in rows),
        "frame_bytes_sum_to_stream": all(abs(sum(row["frame_bytes"]) - row["codec_record"]["rate_bytes_total"])
                                         <= (16 if doc["codec"] == "dcvc" else 0) for row in rows),
        "e_span_scored_100_frames": all(row["spans"]["E"]["scored_frames"] >= 99 for row in rows),
        "rates_positive": all(finite(s["kbps"], 1e-3, 1e6) for row in rows for s in row["spans"].values()),
        "scores_finite": all(finite(s["scores"]["V"]["psnr"], 5.0, 100.0) and finite(s["scores"]["V"]["lpips"], 0.0, 1.0)
                             for row in rows for s in row["spans"].values()),
        "quality_rises_with_rate": all(rises(r, v) for r in results for v in doc["variants"]),
        "fill_share_plausible": all(r["sources"].get("background", {}).get("fill_share_mean", 0.0) < 0.5
                                              for r in results),
        "regions_match_their_record": doc["regions"]["record"] is None or all(
            doc["regions_sha256"][c] == doc["regions"]["record"]["regions_sha256"].get(c) for c in doc["regions_sha256"]),
    }
    if doc["codec"] == "dcvc":
        checks["dcvc_decode_deterministic"] = all(row["codec_record"]["deterministic"] for row in rows)
        checks["dcvc_decoder_matches_encoder_intra"] = all(row["codec_record"].get("decoder_matches_encoder_intra")
                                                           for row in rows)
        checks["dcvc_same_gpu"] = all(row["codec_record"].get("same_gpu") for row in rows)
    return checks


def command_validate(args: argparse.Namespace) -> int:
    stage = stage_dir()
    if (stage / "g2-regions.json").is_file():
        doc = json.loads((stage / "g2-regions.json").read_text())
        checks = validate_regions(stage, doc)
    else:
        doc = json.loads((stage / "g2.json").read_text())
        checks = validate_run(stage, doc)
    report = {"passed": all(checks.values()), "checks": checks}
    target = os.environ.get("PS_VALIDATION_PATH")
    if target:
        write_json(Path(target), report)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


# ----------------------------------------------------------------- choose and report


def mean_curve(docs: list[dict[str, Any]], codec: str, variant: str, span: str = "E") -> list[tuple[float, float, float]]:
    """(point, mean kbps, mean PSNR_V) over the clips, per point."""
    per_point: dict[float, list[tuple[float, float]]] = {}
    for doc in docs:
        if doc["codec"] != codec:
            continue
        for r in doc["results"]:
            for row in r["rows"]:
                if row["variant"] == variant and span in row["spans"]:
                    s = row["spans"][span]
                    per_point.setdefault(row["point"], []).append((s["kbps"], s["scores"]["V"]["psnr"]))
    return sorted((p, float(np.mean([v[0] for v in vals])), float(np.mean([v[1] for v in vals])))
                  for p, vals in per_point.items())


def choose_points(curves: dict[str, list[tuple[float, float, float]]]) -> dict[str, Any]:
    """The G2 rate-point rule on the pilot means (background input)."""
    ranges = {}
    for codec, curve in curves.items():
        qualities = [q for _, _, q in curve]
        ranges[codec] = (min(qualities), min(max(qualities), CHOOSE["cap_db"]))
    low = max(r[0] for r in ranges.values())
    high = min(r[1] for r in ranges.values())
    targets = list(np.linspace(low, high, int(CHOOSE["targets"]))) if high > low else []
    chosen: dict[str, list[float]] = {}
    for codec, curve in curves.items():
        picks: list[float] = []
        for target in targets:
            ordered = sorted(curve, key=lambda c: abs(c[2] - target))
            for point, _, _ in ordered:
                if point not in picks:
                    picks.append(point)
                    break
        by_rate = sorted(curve, key=lambda c: c[1])
        lowest = min((c for c in by_rate if c[0] in picks), key=lambda c: c[1], default=None)
        if lowest is not None:
            below = [c for c in by_rate if c[1] < lowest[1] and c[0] not in picks]
            if below:
                picks.append(below[-1][0])
        chosen[codec] = sorted(picks)
    refine = (high - low) < CHOOSE["min_range_db"] or any(len(v) < CHOOSE["min_points"] for v in chosen.values())
    return {"range_db": [low, high], "targets": targets, "points": chosen, "refine": refine, "per_codec_range": ranges}


def command_choose(args: argparse.Namespace) -> int:
    docs = [json.loads(Path(p).read_text()) for p in args.result]
    curves = {codec: mean_curve(docs, codec, "background") for codec in sorted({d["codec"] for d in docs})}
    out = {"curves": curves, **choose_points(curves)}
    print(json.dumps(out, indent=1))
    return 0


def bd_rate(rate_a: list[float], q_a: list[float], rate_b: list[float], q_b: list[float]) -> float | None:
    from experiments.visor.b2 import bd_rate as b2_bd_rate

    return b2_bd_rate(rate_a, q_a, rate_b, q_b)


def command_report(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    regions_doc = json.loads(Path(args.regions_result).read_text())
    docs = [json.loads(Path(p).read_text()) for p in args.result]
    clips = sorted({r["id"] for d in docs for r in d["results"]})
    table = []
    for d in docs:
        for r in d["results"]:
            for row in r["rows"]:
                for span, s in row["spans"].items():
                    sc = s["scores"]
                    table.append({"clip": r["id"], "dataset": r["dataset"], "codec": d["codec"], "variant": row["variant"],
                                  "point": row["point"], "span": span, "kbps": s["kbps"],
                                  **{f"psnr_{k}": (sc[k] or {}).get("psnr") for k in REGIONS},
                                  **{f"lpips_{k}": (sc[k] or {}).get("lpips") for k in REGIONS},
                                  "lpips_frame_v_excess": sc["excess"],
                                  "decode_ms_per_frame": 1000 * row["codec_record"]["decode_seconds_per_frame"]})
    bd = {}
    for clip in clips:
        def curve(codec: str) -> tuple[list[float], list[float]]:
            rows = sorted((t["kbps"], t["psnr_V"]) for t in table
                          if t["clip"] == clip and t["codec"] == codec and t["variant"] == "background" and t["span"] == "E")
            return [r[0] for r in rows], [r[1] for r in rows]
        a, b = curve("svtav1"), curve("dcvc")
        if len(a[0]) >= 4 and len(b[0]) >= 4:
            bd[clip] = {"psnr_v": bd_rate(a[0], a[1], b[0], b[1])}
    stationarity = []
    for t in table:
        if t["span"] == "E2":
            e = next((u for u in table if u["clip"] == t["clip"] and u["codec"] == t["codec"] and u["variant"] == t["variant"]
                      and u["point"] == t["point"] and u["span"] == "E"), None)
            if e:
                stationarity.append({"clip": t["clip"], "codec": t["codec"], "variant": t["variant"], "point": t["point"],
                                     "rate_ratio_e2_e": t["kbps"] / e["kbps"],
                                     "stationary": abs(t["kbps"] / e["kbps"] - 1) <= 0.25})
    report = {"clips": clips, "rows": table, "bd_rate_dcvc_vs_svtav1_background": bd, "stationarity": stationarity,
              "regions": [{k: v for k, v in r.items()} for r in regions_doc["results"]],
              "inputs": {"regions": args.regions_result, "results": args.result}}
    write_json(out / "g2-report.json", report)
    # Figure: PSNR_V against rate per clip, background input, both codecs; frame input dashed.
    n = len(clips)
    cols = min(4, n)
    fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(4 * cols, 3.2 * ((n + cols - 1) // cols)), squeeze=False)
    styles = {("svtav1", "background"): ("C0", "-"), ("dcvc", "background"): ("C1", "-"), ("svtav1", "frame"): ("C0", "--")}
    for ax, clip in zip(axes.flat, clips):
        for (codec, variant), (colour, ls) in styles.items():
            rows = sorted((t["kbps"], t["psnr_V"]) for t in table
                          if t["clip"] == clip and t["codec"] == codec and t["variant"] == variant and t["span"] == "E")
            if rows:
                ax.plot([r[0] for r in rows], [r[1] for r in rows], ls, color=colour, marker="o", ms=3,
                        label=f"{'SVT-AV1' if codec == 'svtav1' else 'DCVC-UF'} {variant}")
        ax.set_xscale("log")
        ax.set_title(clip, fontsize=9)
        ax.set_xlabel("kbps over E")
        ax.set_ylabel("PSNR on V (dB)")
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(out / f"g2-background-rd.{ext}", dpi=150)
    plt.close(fig)
    print(json.dumps({"bd": bd, "stationary": sum(s["stationary"] for s in stationarity), "of": len(stationarity)}, indent=1))
    return 0


# ----------------------------------------------------------------- entry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    regions = sub.add_parser("regions")
    regions.add_argument("--inputs", required=True)
    regions.add_argument("--select", default="all")
    regions.add_argument("--source", nargs=2, action="append", default=[], metavar=("NAME", "PATH"))
    regions.add_argument("--lpips-backbone", required=True)
    regions.add_argument("--limit-frames", type=int, default=0)
    regions.add_argument("--threads", type=int, default=16)
    regions.add_argument("--device", default="cpu")
    run = sub.add_parser("run")
    run.add_argument("--codec", choices=("svtav1", "dcvc"), required=True)
    run.add_argument("--inputs", required=True)
    run.add_argument("--regions", required=True, help="extracted regions archive (its publish/ directory)")
    run.add_argument("--select", default="all")
    run.add_argument("--source", nargs=2, action="append", default=[], metavar=("NAME", "PATH"))
    run.add_argument("--points", required=True)
    run.add_argument("--variants", default="background")
    run.add_argument("--spans", default="E")
    run.add_argument("--lpips-backbone", required=True)
    run.add_argument("--metric-device", default="")
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--threads", type=int, default=8)
    run.add_argument("--encoder", default="SvtAv1EncApp")
    run.add_argument("--decoder", default="dav1d")
    run.add_argument("--structure", default="htl")
    run.add_argument("--image-ckpt", default="")
    run.add_argument("--video-ckpt", default="")
    run.add_argument("--keep-streams", action="store_true")
    sub.add_parser("validate")
    choose = sub.add_parser("choose")
    choose.add_argument("--result", action="append", required=True)
    report = sub.add_parser("report")
    report.add_argument("--regions-result", required=True)
    report.add_argument("--result", action="append", required=True)
    report.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    handlers = {"regions": command_regions, "run": command_run, "validate": command_validate,
                "choose": command_choose, "report": command_report}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
