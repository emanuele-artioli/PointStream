"""PLAN step H3: H3b's oracle, perfect appearance at the coded pose (docs/experiments.md, 2026-10-11).

    python -m experiments.visor.h3b oracle --eval-set JSON --archive DIR --masks DIR --mask-record JSON \\
        --svt-streams DIR --svt-result JSON [--svt-streams DIR --svt-result JSON] --lpips-backbone PTH \\
        --dists-backbone PTH --inception PTH --poses TAR [--poses TAR] --code JSON --mano-left PKL \\
        --mano-right PKL --wilor-checkpoint CKPT --video V.MP4 ... --items all|ID,ID --frames N
    python -m experiments.visor.h3b validate
    python -m experiments.visor.h3b report --oracle TAR ... --h1-report JSON --code JSON --out DIR

``oracle`` renders, on every frame of an evaluation-set window with a hand, each fitted hand's MANO mesh
at the coded pose (H2's 100 ms choice per model) with the source's own colours carried from the same
surface points on the uncoded mesh, and composites it over a fill of the hand region (variants in
``variant_names``). Every composite and SVT-AV1's are scored on the same region by LPIPS, PSNR, DISTS and
flicker; every ``JUDGE_EVERY``-th frame also by WiLoR re-run on the composite and Inception features of
the hand crops. Each finished window is published as ``oracle/<item>.npz`` with a ``.json`` and
checkpointed. ``report`` adds rates (H1's SVT-AV1 bits, H2's pose stream) and applies ``DECISION``.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from experiments.visor import mano_np
from experiments.visor import pose_coding as pc
from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json
from experiments.visor.h2 import load_units, restore_units, save_unit
from experiments.visor.h3 import (FILL, HANDS, LPIPS_BATCH, POINTS, RESAMPLES, at_rate, bootstrap_mean, dis_flow,
                                  flow_box, hand_kbps, rgb_memmap, seed_of, svt_rate_at_quality, warp)

MODELS = ("wilor", "hamer")
BUDGET_MS = "100"  # H2's latency budget named for H3
JUDGE_EVERY = 5
CROP_PAD = 64  # px around the region and every mesh's projection
DISTS_PAD = 64  # px around S for DISTS
DISTS_BATCH = 6
FILL_RADIUS = 5  # OpenCV Telea
CROP_GROW = 1.5  # Inception crop: H1's box grown to a square
INCEPTION_SIZE = 299
OCCLUDED_M = 0.005  # a sample is hidden when the uncoded surface in front is nearer by more than this
KID_DEGREE = 3
KID_SUBSET = 1000
KID_SUBSETS = 100
KID_RESAMPLES = 200
SEED = "pointstream-h3b"
FACES = 1538  # MANO's triangles per hand
REMAP_WIDTH = 1024


def variant_names() -> list[str]:
    out = [f"svt@{p:g}" for p in POINTS]
    for m in MODELS:
        out += [f"{m}:hand", f"{m}:arm", f"{m}:arm_uncoded"] + [f"{m}:hybrid@{p:g}" for p in POINTS]
    return out


VARIANTS = variant_names()
JUDGED = ["source"] + [f"svt@{p:g}" for p in POINTS] + [f"{m}:{k}" for m in MODELS for k in ("hand", "arm")]

DECISION = {
    "region": "S: the union of VISOR's hand masks (fill set, forearm included) and every rendered silhouette of "
              "either model, coded or uncoded; objects excluded",
    "render": "MANO at H2's coded pose (100 ms choice per model); each covered pixel takes the source's colour at the "
              "same surface point on the uncoded mesh (nearest region pixel when that point is outside the region); "
              "never over objects",
    "fill": f"OpenCV Telea (radius {FILL_RADIUS}) of the hand region under every composite",
    "variants": {"m:hand": "render over the fill", "m:arm": "forearm's true pixels, then the render",
                 "m:arm_uncoded": "m:arm at the uncoded pose (diagnostic)",
                 "m:hybrid@c": "forearm and unfitted hands as SVT-AV1's pixels at CRF c, then the render",
                 "svt@c": "B2's stream at CRF c inside the hand region"},
    "metric": "LPIPS (B2's AlexNet spatial map, full frame) averaged over S decides; PSNR, DISTS (G5c's masked), "
              "flicker, WiLoR pose fidelity and Inception KID/FID beside",
    "rates": "oracle: the model's pose stream (H2, 100 ms); hybrid: + H1's bits on forearm and hand without fit at c; "
             "SVT-AV1: H1's bits on hand + forearm + hand without fit",
    "gate": "wilor:arm − svt@62 LPIPS per item, mean over items, 95% paired bootstrap by item; passes if the interval "
            "is below zero",
    "hand": "wilor:hand, the same test (must the generator draw the arm?)",
    "hybrid": "wilor:hybrid@c − SVT-AV1's LPIPS at the hybrid's rate (linear in log-rate, held at the end points); "
              "passes if the interval is below zero at some c",
    "escalate": "if nothing passes but wilor:arm_uncoded passes the gate, or wilor:arm is no worse than svt@62 on pose "
                "fidelity, DISTS and KID, the outcome goes to the user before the route is dropped",
    "beside": "HaMeR's composites take the same tests",
    "judge_every": JUDGE_EVERY, "resamples": RESAMPLES,
}


# ----------------------------------------------------------------- poses

def coding_combo(code: dict[str, Any], model: str) -> dict[str, Any]:
    """H2's chosen combination for ``model`` at the named budget."""
    key = code["choices"][model][BUDGET_MS]["key"]
    found = [c for c in pc.combinations() if pc.combo_key(c) == key]
    if len(found) != 1:
        raise SystemExit(f"{model}: coding {key!r} is not one of H2's combinations")
    return found[0]


def item_poses(meta: dict[str, Any], a: dict[str, np.ndarray], mano: mano_np.Mano, combos: dict[str, dict[str, Any]],
               basis: pc.Basis) -> dict[str, dict[str, Any]]:
    """Per model, every hand-frame's uncoded mesh (its own shape, as H2 rendered it), the coded mesh and joints
    (each track sent with its median shape and decoded), and the number of tracks."""
    from experiments.visor.h2 import CodingSet

    out: dict[str, dict[str, Any]] = {}
    for m in MODELS:
        n = int(meta["hands"])
        if not n:
            out[m] = {"uncoded": np.zeros((0, 778, 3)), "coded": np.zeros((0, 778, 3)), "coded_joints": np.zeros((0, 21, 3)), "tracks": 0}
            continue
        data = CodingSet("visor", m, [(meta, a)], mano)
        uncoded = np.stack([mano.mesh(data.rot[r:r + 1], data.betas[r], data.transl[r:r + 1], left=bool(data.left[r]))[0]
                            for r in range(n)])
        coded = np.zeros_like(uncoded)
        joints = np.zeros((n, 21, 3))
        info = data.units[0]
        for tr in data.tracks:
            symbols, sent = pc.encode(tr["x"], combos[m], tr["fps"], basis)
            x = pc.decode(symbols, sent, len(tr["x"]), combos[m], basis)
            rot = mano_np.rodrigues(x[:, :48].reshape(-1, 16, 3))
            transl = pc.root_transl(x[:, 48:51], info["focal"], info["centre"])
            coded[tr["rows"]] = mano.mesh(rot, tr["betas"], transl, left=tr["left"])
            joints[tr["rows"]] = mano.joints(rot, np.repeat(tr["betas"][None], len(rot), 0), transl, np.full(len(rot), tr["left"]))
        out[m] = {"uncoded": uncoded, "coded": coded, "coded_joints": joints, "tracks": len(data.tracks)}
    return out


# ----------------------------------------------------------------- rasterizing with correspondence

def raster(points: np.ndarray, depth: np.ndarray, faces: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Face index per pixel of ``box`` (−1 where empty), nearer faces drawn last (painter's order)."""
    import cv2

    x0, y0, x1, y1 = box
    ids = np.full((y1 - y0, x1 - x0), -1, np.int32)
    local = np.clip(points - [x0, y0], -1e5, 1e5)
    tris = np.round(local[faces] * 16).astype(np.int32)
    for f in np.argsort(-depth[faces].mean(1), kind="stable"):
        cv2.fillConvexPoly(ids, tris[f], (int(f),), lineType=cv2.LINE_8, shift=4)
    return ids


def barycentric(ids: np.ndarray, points: np.ndarray, faces: np.ndarray, box: tuple[int, int, int, int]
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Covered pixels (rows, cols in the box), their face and barycentric weights (n, 3) at the pixel centre."""
    ys, xs = np.nonzero(ids >= 0)
    f = ids[ys, xs]
    p = np.stack([xs + box[0], ys + box[1]], 1).astype(float)
    a, b, c = points[faces[f, 0]], points[faces[f, 1]], points[faces[f, 2]]
    v0, v1, v2 = b - a, c - a, p - a
    den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
    den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    l1 = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
    l2 = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
    w = np.clip(np.stack([1 - l1 - l2, l1, l2], 1), 0, None)
    w /= np.clip(w.sum(1, keepdims=True), 1e-12, None)
    return ys, xs, f, w


def depth_map(ids: np.ndarray, points: np.ndarray, depth: np.ndarray, faces: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    out = np.full(ids.shape, np.inf)
    ys, xs, f, w = barycentric(ids, points, faces, box)
    out[ys, xs] = (w * depth[faces[f]]).sum(1)
    return out


def nearest_lookup(region: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """For every pixel, the label of the nearest ``region`` pixel, and the labels' (row, col)."""
    import cv2

    if not region.any():
        return None
    _, labels = cv2.distanceTransformWithLabels((~region).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    zy, zx = np.nonzero(region)
    lut = np.zeros((int(labels.max()) + 1, 2), np.int64)
    lut[labels[zy, zx]] = np.stack([zy, zx], 1)
    return labels, lut


def render(source: np.ndarray, region: np.ndarray, objects: np.ndarray, target_ids: np.ndarray, target_pts: np.ndarray,
           source_pts: np.ndarray, source_z: np.ndarray, source_zmap: np.ndarray, faces: np.ndarray,
           box: tuple[int, int, int, int], nearest: tuple[np.ndarray, np.ndarray] | None) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """The target pose's covered pixels (box crop, objects excluded) and their colours: the source's colour at the
    same surface point on the source pose (bilinear), or the nearest region pixel's when that point is outside the
    region. ``source``, ``region``, ``objects`` and ``source_zmap`` are crops at ``box``."""
    import cv2

    covered = (target_ids >= 0) & ~objects
    ids = np.where(covered, target_ids, -1)
    ys, xs, f, w = barycentric(ids, target_pts, faces, box)
    colours = np.zeros_like(source)
    stats = {"covered": int(len(ys)), "outside": 0, "occluded": 0}
    if not len(ys):
        return covered, colours, stats
    h, wd = region.shape
    q = (w[:, :, None] * source_pts[faces[f]]).sum(1) - [box[0], box[1]]
    qx = np.clip(q[:, 0], 0, wd - 1)
    qy = np.clip(q[:, 1], 0, h - 1)
    # remap wants maps under 32,767 rows and columns: lay the samples out as rows of REMAP_WIDTH.
    n = len(qx)
    rows = -(-n // REMAP_WIDTH)
    map_x = np.zeros(rows * REMAP_WIDTH, np.float32)
    map_y = np.zeros(rows * REMAP_WIDTH, np.float32)
    map_x[:n], map_y[:n] = qx, qy
    sampled = cv2.remap(source, map_x.reshape(rows, REMAP_WIDTH), map_y.reshape(rows, REMAP_WIDTH), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE).reshape(-1, source.shape[2])[:n]
    qi, qj = np.round(qy).astype(int), np.round(qx).astype(int)
    outside = ~region[qi, qj]
    if outside.any() and nearest is not None:
        labels, lut = nearest
        yx = lut[labels[qi[outside], qj[outside]]]
        sampled[outside] = source[yx[:, 0], yx[:, 1]]
    z = (w * source_z[faces[f]]).sum(1)
    stats["outside"] = int(outside.sum())
    stats["occluded"] = int((z > source_zmap[qi, qj] + OCCLUDED_M).sum())
    colours[ys, xs] = sampled
    return covered, colours, stats


# ----------------------------------------------------------------- metrics

def lpips_reduce(net: Any, frames: list[np.ndarray], truth: np.ndarray, masks: list[list[np.ndarray]], device: str) -> list[list[float]]:
    """LPIPS of each full frame against ``truth``, averaged inside each of that frame's masks (NaN if empty)."""
    import torch

    out: list[list[float]] = []
    ref = torch.from_numpy(np.ascontiguousarray(truth)).to(device).permute(2, 0, 1)[None].float() / 127.5 - 1.0
    for first in range(0, len(frames), LPIPS_BATCH):
        batch = torch.from_numpy(np.stack(frames[first:first + LPIPS_BATCH])).to(device).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        with torch.inference_mode():
            lp = net(batch, ref.expand(len(batch), -1, -1, -1))[:, 0]
        for i in range(len(batch)):
            row = []
            for mask in masks[first + i]:
                n = int(mask.sum())
                row.append(float((lp[i] * torch.from_numpy(mask).to(device)).sum()) / n if n else math.nan)
            out.append(row)
    return out


def load_dists(backbone: Path, device: str) -> Any:
    """torchmetrics 1.9's DISTS (its alpha/beta) on the VGG16 read from ``backbone``, never downloaded (G5c's
    ``src.codecs.quality.load_dists``)."""
    from unittest import mock

    import torch
    import torchvision
    from torchmetrics.functional.image import dists as module

    vgg = torchvision.models.vgg16(weights=None)
    vgg.load_state_dict(torch.load(backbone, map_location="cpu", weights_only=True), strict=True)
    with mock.patch.object(module, "vgg16", lambda weights=None: vgg):
        return module.DISTSNetwork(load_weights=True).eval().to(device)


def masked_dists(net: Any, x: Any, y: Any, mask: Any) -> Any:
    """DISTS of (b, 3, h, w) images in [0, 1] with every layer's statistics weighted by ``mask`` (b, 1, h, w),
    resized to the layer by area average (G5c's ``masked_dists``); DISTS exactly when the mask is all ones."""
    import torch
    import torch.nn.functional as F

    with torch.inference_mode():
        fx, fy = net.forward_once(x), net.forward_once(y)
        total = net.alpha.sum() + net.beta.sum()
        alpha = torch.split(net.alpha / total, net.chns, dim=1)
        beta = torch.split(net.beta / total, net.chns, dim=1)
        structure = texture = torch.zeros(x.shape[0], 1, 1, 1, device=x.device)
        c1 = c2 = 1e-6
        for k, (a, b) in enumerate(zip(fx, fy)):
            w = F.adaptive_avg_pool2d(mask.float(), a.shape[-2:])
            w = w / w.sum((2, 3), keepdim=True).clamp_min(1e-12)

            def mean(t: Any) -> Any:
                return (t * w).sum((2, 3), keepdim=True)  # noqa: B023 (w of this layer)

            a_mean, b_mean = mean(a), mean(b)
            s1 = (2 * a_mean * b_mean + c1) / (a_mean**2 + b_mean**2 + c1)
            a_var, b_var = mean((a - a_mean) ** 2), mean((b - b_mean) ** 2)
            cov = mean(a * b) - a_mean * b_mean
            s2 = (2 * cov + c2) / (a_var + b_var + c2)
            structure = structure + (alpha[k] * s1).sum(1, keepdim=True)
            texture = texture + (beta[k] * s2).sum(1, keepdim=True)
        return 1 - (structure + texture).flatten()


def dists_crops(net: Any, crops: list[np.ndarray], truth: np.ndarray, mask: np.ndarray, device: str) -> list[float]:
    import torch

    def tensor(xs: list[np.ndarray]) -> Any:
        return torch.from_numpy(np.stack(xs)).to(device).permute(0, 3, 1, 2).float() / 255.0

    out: list[float] = []
    m = torch.from_numpy(mask).to(device)[None, None]
    for first in range(0, len(crops), DISTS_BATCH):
        x = tensor(crops[first:first + DISTS_BATCH])
        y = tensor([truth]).expand(len(x), -1, -1, -1)
        out.extend(masked_dists(net, x, y, m.expand(len(x), -1, -1, -1)).double().cpu().tolist())
    return out


def luma(image: np.ndarray) -> np.ndarray:
    return image.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)


def flicker(current: np.ndarray, previous_warped: np.ndarray, truth: np.ndarray, truth_prev_warped: np.ndarray, mask: np.ndarray) -> float:
    """G5c's flicker: the method's temporal residual minus the source's, luma, mean absolute value in ``mask``."""
    d = (luma(current) - luma(previous_warped)) - (luma(truth) - luma(truth_prev_warped))
    return float(np.abs(d[mask]).mean()) if mask.any() else math.nan


def load_inception(weights: Path, device: str) -> Any:
    """torchvision's Inception-v3 (ImageNet weights read from ``weights``) returning the 2048-d pool features."""
    import torch
    import torchvision

    net = torchvision.models.inception_v3(weights=None, aux_logits=True, init_weights=False, transform_input=True)
    net.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    net.fc = torch.nn.Identity()
    return net.eval().to(device)


def square_crop(image: np.ndarray, box: np.ndarray, grow: float, size: int) -> np.ndarray:
    """The box grown to a square of ``grow`` × its longer side (zeros outside the frame), resized to ``size``."""
    import cv2

    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    half = grow * max(box[2] - box[0], box[3] - box[1]) / 2
    x0, y0 = int(math.floor(cx - half)), int(math.floor(cy - half))
    side = max(2, int(math.ceil(2 * half)))
    window = np.zeros((side, side, 3), np.uint8)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x0 + side, image.shape[1]), min(y0 + side, image.shape[0])
    if sx1 > sx0 and sy1 > sy0:
        window[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = image[sy0:sy1, sx0:sx1]
    return np.asarray(cv2.resize(window, (size, size), interpolation=cv2.INTER_AREA))


def inception_features(net: Any, crops: list[np.ndarray], device: str) -> np.ndarray:
    import torch

    x = torch.from_numpy(np.stack(crops)).to(device).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=device)[None, :, None, None]
    with torch.inference_mode():
        return np.asarray(net((x - mean) / std).float().cpu().numpy(), np.float16)


def kid_kernels(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = x.shape[1]
    x, y = x.astype(np.float64), y.astype(np.float64)
    return ((x @ x.T / d + 1) ** KID_DEGREE, (y @ y.T / d + 1) ** KID_DEGREE, (x @ y.T / d + 1) ** KID_DEGREE)


def kid_from(kxx: np.ndarray, kyy: np.ndarray, kxy: np.ndarray, rng: np.random.Generator, subset: int = KID_SUBSET,
             subsets: int = KID_SUBSETS) -> float:
    """Unbiased MMD² with a cubic polynomial kernel, averaged over random subsets (the KID estimator)."""
    n, m = kxx.shape[0], kyy.shape[0]
    size = min(subset, n, m)
    if size < 2:
        return math.nan
    values = []
    for _ in range(subsets):
        i = rng.choice(n, size, replace=False)
        j = rng.choice(m, size, replace=False)
        a, b, c = kxx[np.ix_(i, i)], kyy[np.ix_(j, j)], kxy[np.ix_(i, j)]
        values.append((a.sum() - np.trace(a)) / (size * (size - 1)) + (b.sum() - np.trace(b)) / (size * (size - 1)) - 2 * c.mean())
    return float(np.mean(values))


def fid(x: np.ndarray, y: np.ndarray) -> float:
    from scipy import linalg

    x, y = x.astype(np.float64), y.astype(np.float64)
    mx, my = x.mean(0), y.mean(0)
    cx, cy = np.cov(x, rowvar=False), np.cov(y, rowvar=False)
    root, _ = linalg.sqrtm(cx @ cy, disp=False)
    return float(((mx - my) ** 2).sum() + np.trace(cx + cy - 2 * np.real(root)))


# ----------------------------------------------------------------- one window

def process_item(args: argparse.Namespace, item: dict[str, Any], nets: dict[str, Any], device: str, publish: Path,
                 decoded_source: dict[str, Any], scratch: Path, cross_device: bool) -> None:
    import cv2

    from experiments.visor.b2 import load_mask_sets
    from experiments.visor.h1 import iou, stream_paths, wrist_split
    from experiments.visor.h2 import pinhole
    from src.codecs import svtav1

    began = time.time()
    frames = int(args.frames)
    work = scratch / "work" / item["id"]
    work.mkdir(parents=True, exist_ok=True)
    width, height, full_range = decoded_source["width"], decoded_source["height"], decoded_source["full_range"]
    clips, mask_info = load_mask_sets(item, frames, {FILL: args.masks}, {FILL: args.mask_record}, Path(args.archive))
    clip = clips[FILL]
    truth = rgb_memmap(Path(decoded_source["path"]), work / "source-rgb.npy", frames, width, height, full_range, device)
    streams, stream_info = stream_paths(args, item["id"])
    names = [f"{p:g}" for p in POINTS]
    allowance = max(1, int(os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 1))
    decoded: dict[str, np.ndarray] = {}
    for name in names:
        yuv = work / f"crf{name}.yuv"
        command = svtav1.decode_command(args.decoder, streams[name], yuv, threads=allowance)
        timing = svtav1.run(command, 1800)
        count = yuv.stat().st_size // (width * height * 3 // 2)
        stream_info[name].update({"decode_command": command, "decode_seconds": round(timing["wall"], 2), "decoded_frames": int(count)})
        if count < frames:
            raise RuntimeError(f"{item['id']}: CRF {name} decoded {count} frames, need {frames}")
        decoded[name] = rgb_memmap(yuv, work / f"crf{name}-rgb.npy", frames, width, height, full_range, device)
        yuv.unlink()

    meta_p, a = args.pose_units[item["id"]]
    poses = item_poses(meta_p, a, args.mano, args.combos, args.basis)
    focal, centre = float(meta_p["focal"]), tuple(meta_p["principal"])
    rows_at: dict[int, list[int]] = {}
    for r, t in enumerate(a["t"]):
        if t < frames:
            rows_at.setdefault(int(t), []).append(r)
    proj = {m: {k: np.stack([pinhole(v, focal, centre) for v in poses[m][k]]) if len(poses[m][k]) else np.zeros((0, 778, 2))
                for k in ("uncoded", "coded")} for m in MODELS}
    n_rows = len(a["t"])
    coded_err = {m: np.linalg.norm(np.stack([pinhole(j, focal, centre) for j in poses[m]["coded_joints"]]) - a[f"{m}_kp2d"], axis=-1).mean(1)
                 if n_rows else np.zeros(0) for m in MODELS}
    uncoded_iou = {m: np.full(n_rows, np.nan) for m in MODELS}

    nv = len(VARIANTS)
    lp_s = np.full((nv, frames), np.nan)
    sse_s = np.zeros((nv, frames), np.int64)
    dists = np.full((nv, frames), np.nan)
    flick = np.full((nv, frames), np.nan)
    svt_lp_r = np.full((len(POINTS), frames), np.nan)
    lp_rendered = np.full((len(MODELS), frames), np.nan)
    pixels_s = np.zeros(frames, np.int64)
    pixels_r = np.zeros(frames, np.int64)
    stat_keys = ("covered", "outside", "occluded", "hand_px", "missed", "spill")
    stats = {m: {k: np.zeros(frames, np.int64) for k in stat_keys} for m in MODELS}
    judge_t: list[int] = []
    judge_right: list[bool] = []
    judge_box: list[float] = []
    judge_kp: list[np.ndarray] = []
    judge_feat: list[np.ndarray] = []
    judge_rows: list[int] = []
    checks: dict[str, Any] = {}
    cv2.setNumThreads(allowance)
    previous: dict[str, Any] | None = None
    faces = args.mano.faces
    for t in range(frames):
        sides = {s: clip.class_mask(t, s) for s in HANDS}
        region = sides[HANDS[0]] | sides[HANDS[1]]
        if not region.any():
            previous = None
            continue
        truth_t = np.asarray(truth[t])
        objects = clip.foreground(t) & ~region
        rows = rows_at.get(t, [])
        forearm = np.zeros_like(region)
        nofit = np.zeros_like(region)
        hand_part = np.zeros_like(region)
        fitted = {bool(a["right"][r]): r for r in rows}
        for s in HANDS:
            mask = sides[s]
            if not mask.any():
                continue
            fit_row = fitted.get(s == "right hand")
            if fit_row is None:
                nofit |= mask
                continue
            kp = a["wilor_kp2d"][fit_row]
            hand, arm = wrist_split(mask, kp[0], kp[9])
            hand_part |= hand
            forearm |= arm
        # The crop: the region and every mesh's projection, padded.
        xs_all = [np.nonzero(region.any(0))[0]]
        ys_all = [np.nonzero(region.any(1))[0]]
        for m in MODELS:
            for pose in ("uncoded", "coded"):
                for r in rows:
                    xs_all.append(np.clip(proj[m][pose][r][:, 0], -1e5, 1e5).astype(int))
                    ys_all.append(np.clip(proj[m][pose][r][:, 1], -1e5, 1e5).astype(int))
        x0 = max(0, int(min(v.min() for v in xs_all)) - CROP_PAD)
        y0 = max(0, int(min(v.min() for v in ys_all)) - CROP_PAD)
        x1 = min(width, int(max(v.max() for v in xs_all)) + CROP_PAD + 1)
        y1 = min(height, int(max(v.max() for v in ys_all)) + CROP_PAD + 1)
        box = (x0, y0, x1, y1)
        crop = (slice(y0, y1), slice(x0, x1))
        src = np.ascontiguousarray(truth_t[crop])
        reg, obj = region[crop], objects[crop]
        filled = src.copy()
        hole = cv2.inpaint(src, reg.astype(np.uint8), FILL_RADIUS, cv2.INPAINT_TELEA)
        filled[reg] = hole[reg]
        nearest = nearest_lookup(reg)
        renders: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        silhouettes = np.zeros_like(reg)
        for m in MODELS:
            if not rows:
                renders[m] = renders[f"{m}_uncoded"] = (np.zeros_like(reg), np.zeros_like(src))
                continue
            k_faces = np.concatenate([faces + 778 * k for k in range(len(rows))])
            pts_u = np.concatenate([proj[m]["uncoded"][r] for r in rows])
            pts_c = np.concatenate([proj[m]["coded"][r] for r in rows])
            z_u = np.concatenate([poses[m]["uncoded"][r][:, 2] for r in rows])
            z_c = np.concatenate([poses[m]["coded"][r][:, 2] for r in rows])
            ids_u = raster(pts_u, z_u, k_faces, box)
            ids_c = raster(pts_c, z_c, k_faces, box)
            zmap_u = depth_map(ids_u, pts_u, z_u, k_faces, box)
            for k, r in enumerate(rows):
                own = (ids_u >= k * FACES) & (ids_u < (k + 1) * FACES)
                side = sides["right hand" if a["right"][r] else "left hand"][crop]
                value = iou(own, side)
                uncoded_iou[m][r] = np.nan if value is None else value
            covered, colours, st = render(src, reg, obj, ids_c, pts_c, pts_u, z_u, zmap_u, k_faces, box, nearest)
            renders[m] = (covered, colours)
            renders[f"{m}_uncoded"] = render(src, reg, obj, ids_u, pts_u, pts_u, z_u, zmap_u, k_faces, box, nearest)[:2]
            for key in ("covered", "outside", "occluded"):
                stats[m][key][t] = st[key]
            stats[m]["hand_px"][t] = int(hand_part[crop].sum())
            stats[m]["missed"][t] = int((hand_part[crop] & ~covered).sum())
            stats[m]["spill"][t] = int((covered & ~reg).sum())
            silhouettes |= covered | renders[f"{m}_uncoded"][0]
        s_crop = reg | silhouettes
        crops: dict[str, np.ndarray] = {}
        for i, name in enumerate(names):
            svt = np.asarray(decoded[name][t])[crop]
            out = src.copy()
            out[reg] = svt[reg]
            crops[f"svt@{name}"] = out
            for m in MODELS:
                cov, col = renders[m]
                hyb = filled.copy()
                coded_px = (forearm | nofit)[crop]
                hyb[coded_px] = svt[coded_px]
                hyb[cov] = col[cov]
                crops[f"{m}:hybrid@{name}"] = hyb
        for m in MODELS:
            for label, key in ((f"{m}:hand", m), (f"{m}:arm", m), (f"{m}:arm_uncoded", f"{m}_uncoded")):
                cov, col = renders[key]
                out = filled.copy()
                if label != f"{m}:hand":
                    fa = forearm[crop]
                    out[fa] = src[fa]
                out[cov] = col[cov]
                crops[label] = out
        full_s = np.zeros_like(region)
        full_s[crop] = s_crop
        full_r = region
        pixels_s[t], pixels_r[t] = int(full_s.sum()), int(full_r.sum())
        fulls = []
        for v in VARIANTS:
            frame = truth_t.copy()
            frame[crop] = crops[v]
            fulls.append(frame)
            sse_s[VARIANTS.index(v), t] = int(((crops[v][s_crop].astype(np.int32) - src[s_crop].astype(np.int32)) ** 2).sum())
        masks = [[full_s] + ([full_r] if v.startswith("svt@") else []) for v in VARIANTS]
        for j, m in enumerate(MODELS):
            rendered = np.zeros_like(region)
            rendered[crop] = renders[m][0]
            masks[VARIANTS.index(f"{m}:arm")].append(rendered)
        lp = lpips_reduce(nets["lpips"], fulls, truth_t, masks, device)
        if cross_device and "lpips_cpu_vs_device" not in checks:
            from src.codecs.quality import load_lpips

            cpu = lpips_reduce(load_lpips(Path(args.lpips_backbone), "cpu"), fulls[-1:], truth_t, masks[-1:], "cpu")
            checks["lpips_cpu_vs_device"] = abs(cpu[0][0] - lp[-1][0])
        for i, v in enumerate(VARIANTS):
            lp_s[i, t] = lp[i][0]
        for i, name in enumerate(names):
            svt_lp_r[i, t] = lp[VARIANTS.index(f"svt@{name}")][1]
        for j, m in enumerate(MODELS):
            row = lp[VARIANTS.index(f"{m}:arm")]
            lp_rendered[j, t] = row[1]
        # DISTS on the crop around S.
        sb = (max(0, int(np.nonzero(full_s.any(0))[0].min()) - DISTS_PAD), max(0, int(np.nonzero(full_s.any(1))[0].min()) - DISTS_PAD),
              min(width, int(np.nonzero(full_s.any(0))[0].max()) + 1 + DISTS_PAD), min(height, int(np.nonzero(full_s.any(1))[0].max()) + 1 + DISTS_PAD))
        dcrop = (slice(sb[1], sb[3]), slice(sb[0], sb[2]))
        values = dists_crops(nets["dists"], [f[dcrop] for f in fulls], truth_t[dcrop], full_s[dcrop], device)
        dists[:, t] = values
        if "dists_identity" not in checks:
            checks["dists_identity"] = dists_crops(nets["dists"], [truth_t[dcrop]], truth_t[dcrop], full_s[dcrop], device)[0]
        # Flicker against the previous frame, when it had a hand.
        if previous is not None:
            fb = flow_box(full_s, previous["s"])
            fcrop = (slice(fb[1], fb[3]), slice(fb[0], fb[2]))
            gray_t = cv2.cvtColor(truth_t[fcrop], cv2.COLOR_RGB2GRAY)
            gray_p = cv2.cvtColor(previous["truth"][fcrop], cv2.COLOR_RGB2GRAY)
            flow = dis_flow(gray_t, gray_p)
            truth_warped = warp(np.ascontiguousarray(previous["truth"][fcrop]), flow)
            for i, v in enumerate(VARIANTS):
                flick[i, t] = flicker(fulls[i][fcrop], warp(np.ascontiguousarray(previous["fulls"][i][fcrop]), flow),
                                      truth_t[fcrop], truth_warped, full_s[fcrop])
        previous = {"s": full_s, "truth": truth_t, "fulls": fulls}
        # Judges on every JUDGE_EVERY-th frame: WiLoR re-run and Inception features, per fitted hand.
        if t % JUDGE_EVERY == 0 and rows:
            sources = {"source": truth_t, **{v: fulls[VARIANTS.index(v)] for v in JUDGED if v != "source"}}
            inputs, crops_in = [], []
            for v in JUDGED:
                for r in rows:
                    inputs.append(nets["wilor"].item(sources[v], np.asarray(a["box"][r], np.float32), bool(a["right"][r])))
                    crops_in.append(square_crop(sources[v], a["box"][r], CROP_GROW, INCEPTION_SIZE))
            fit = nets["wilor"].forward(inputs)
            feats = inception_features(nets["inception"], crops_in, device)
            count = len(rows)
            judge_kp.append(fit["kp2d"].reshape(len(JUDGED), count, 21, 2).astype(np.float32))
            judge_feat.append(feats.reshape(len(JUDGED), count, -1))
            judge_box.extend(fit["box_size"][:count].tolist())
            judge_t.extend([t] * count)
            judge_right.extend(bool(a["right"][r]) for r in rows)
            judge_rows.extend(rows)
    os.remove(decoded_source["path"])
    shutil.rmtree(work)
    arrays: dict[str, np.ndarray] = {
        "pixels_s": pixels_s, "pixels_r": pixels_r, "lpips": lp_s, "sse": sse_s, "dists": dists, "flicker": flick,
        "svt_lpips_r": svt_lp_r, "lpips_rendered": lp_rendered,
        "judge_t": np.array(judge_t, int), "judge_right": np.array(judge_right, bool), "judge_row": np.array(judge_rows, int),
        "judge_box_size": np.array(judge_box, float),
        "judge_kp2d": np.concatenate(judge_kp, 1) if judge_kp else np.zeros((len(JUDGED), 0, 21, 2), np.float32),
        "judge_features": np.concatenate(judge_feat, 1) if judge_feat else np.zeros((len(JUDGED), 0, 2048), np.float16),
        **{f"{m}_{k}": v for m in MODELS for k, v in stats[m].items()},
        **{f"{m}_coded_err_px": coded_err[m] for m in MODELS}, **{f"{m}_uncoded_iou": uncoded_iou[m] for m in MODELS},
        "archived_iou_mask": np.stack([a[f"{m}_iou_mask"] for m in MODELS]) if n_rows else np.zeros((len(MODELS), 0)),
        "archived_wilor_kp2d": a["wilor_kp2d"] if n_rows else np.zeros((0, 21, 2)),
    }
    meta = {"unit": item["id"], "video": item["video"], "fps": float(item["fps"]), "frames": frames,
            "hand_frames": int((pixels_r > 0).sum()), "variants": VARIANTS, "judged": JUDGED,
            "tracks": {m: poses[m]["tracks"] for m in MODELS}, "pose_rows": n_rows, "pose_unit_hash_ok": bool(meta_p["npz_hash_ok"]),
            "decode": {k: v for k, v in decoded_source.items() if k != "path"}, "mask_set": mask_info[FILL],
            "streams": stream_info, "checks": checks, "seconds": round(time.time() - began, 2)}
    save_unit(publish, "oracle", item["id"], arrays, meta)


def command_oracle(args: argparse.Namespace) -> int:
    from experiments.visor.b2 import decode_window, select_items
    from experiments.visor.h2 import Regressor, mano_dir

    eval_set = json.loads(Path(args.eval_set).read_text())
    items = select_items(eval_set, args.items)
    videos = {Path(v).stem: Path(v) for v in args.video}
    scratch = Path(os.environ.get("PS_SCRATCH_DIR") or stage_dir() / "scratch")
    publish = scratch / "publish"
    publish.mkdir(parents=True, exist_ok=True)
    restored = restore_units(publish, "oracle")
    todo = [it for it in items if it["id"] not in restored]
    allowance = max(1, int(os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 1))
    frames = int(args.frames)
    code = json.loads(Path(args.code).read_text())
    args.combos = {m: coding_combo(code, m) for m in MODELS}
    args.mano = mano_np.Mano.load(args.mano_right)
    args.basis = pc.Basis(args.mano.hands_mean, args.mano.hands_components)
    args.pose_units = {meta["unit"]: (meta, a) for meta, a in load_units(args.poses, "visor")}
    if missing := [it["id"] for it in items if it["id"] not in args.pose_units]:
        raise SystemExit(f"no H2 poses for {missing}")
    # PyAV decodes in fresh processes, before torch loads (B2).
    pool = ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"))

    def submit(item: dict[str, Any]) -> Any:
        target = scratch / "source" / f"{item['id']}.yuv"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        return pool.submit(decode_window, videos[item["video"]], item, frames, Path(args.archive), target, max(1, allowance // 2))

    pending = {i: submit(todo[i]) for i in range(min(2, len(todo)))}
    import torch

    from experiments.audit.env_smoke import device_record, model_on_cuda
    from src.codecs.quality import load_lpips, lpips_record

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("h3b oracle needs a GPU (WiLoR)")
    torch.set_num_threads(allowance)
    args.mano_dir = mano_dir(args, scratch)
    nets = {"lpips": load_lpips(Path(args.lpips_backbone), device), "dists": load_dists(Path(args.dists_backbone), device),
            "inception": load_inception(Path(args.inception), device), "wilor": Regressor("wilor", args)}
    record = device_record(torch.cuda.get_device_name(0))
    on_cuda = {"lpips": next(nets["lpips"].parameters()).is_cuda, "dists": next(nets["dists"].parameters()).is_cuda,
               "inception": next(nets["inception"].parameters()).is_cuda, "wilor": model_on_cuda(nets["wilor"].model)}
    done = len(restored)
    began = time.time()
    for number, item in enumerate(todo):
        source = pending.pop(number).result()
        if number + 2 < len(todo):
            pending[number + 2] = submit(todo[number + 2])
        process_item(args, item, nets, device, publish, source, scratch, cross_device=number == 0)
        done += 1
        progress(done)
    pool.shutdown()
    import torchmetrics
    from torchmetrics.functional.image import dists as dists_module

    dists_weights = Path(dists_module.__file__).parent / "dists_models" / "weights.pt"
    write_json(stage_dir() / "h3b.json", {
        "command": "oracle", "device": record, "on_cuda": on_cuda, "seconds": round(time.time() - began, 2),
        "peak_gpu_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "lpips": lpips_record(Path(args.lpips_backbone)),
        "dists": {"torchmetrics": torchmetrics.__version__, "weights_sha256": file_sha256(dists_weights),
                  "backbone": args.dists_backbone, "backbone_sha256": file_sha256(Path(args.dists_backbone))},
        "inception": {"path": args.inception, "sha256": file_sha256(Path(args.inception)), "net": "torchvision inception_v3 pool"},
        "wilor": nets["wilor"].load, "decoder": _tool(args.decoder),
        "inputs": {"eval_set_sha256": file_sha256(Path(args.eval_set)), "mask_record_sha256": file_sha256(Path(args.mask_record)),
                   "svt_result_sha256": [file_sha256(Path(p)) for p in args.svt_result],
                   "poses_sha256": [file_sha256(Path(p)) for p in args.poses], "code_sha256": file_sha256(Path(args.code)),
                   "mano_right_sha256": file_sha256(Path(args.mano_right))},
        "settings": {"frames": frames, "items": args.items, "decision": DECISION, "combos": {m: pc.combo_key(c) for m, c in args.combos.items()},
                     "crop_pad": CROP_PAD, "dists_pad": DISTS_PAD, "fill_radius": FILL_RADIUS, "judge_every": JUDGE_EVERY,
                     "crop_grow": CROP_GROW, "occluded_m": OCCLUDED_M},
        "items": [it["id"] for it in items], "restored_units": sorted(restored),
    })
    return 0


def _tool(name: str) -> dict[str, Any]:
    from src.codecs import svtav1

    try:
        return svtav1.tool(name)
    except FileNotFoundError:
        return {"path": name, "missing": True}


# ----------------------------------------------------------------- validate

def validate_result(stage: Path) -> dict[str, bool]:
    if (stage / "h3b-report.json").exists():
        out = json.loads((stage / "h3b-report.json").read_text())
        return {"items_scored": out["items"] > 0, "decision_made": out["decision"]["gate"]["passes"] in (True, False)}
    result = json.loads((stage / "h3b.json").read_text())
    units = load_units([str(stage / "published.tar")], "oracle")
    first = [m for m, _ in units if "lpips_cpu_vs_device" in m["checks"]]
    hand = [(m, a, a["pixels_r"] > 0) for m, a in units]
    rerun = [np.linalg.norm(a["judge_kp2d"][0] - a["archived_wilor_kp2d"][a["judge_row"]], axis=-1).mean(-1) / a["judge_box_size"]
             for _, a in units if len(a["judge_row"])]
    svt62, svt41 = VARIANTS.index("svt@62"), VARIANTS.index("svt@41")
    iou_gap = [float(np.nanmedian(np.abs(a[f"{m}_uncoded_iou"] - a["archived_iou_mask"][j])))
               for _, a in units for j, m in enumerate(MODELS) if np.isfinite(a[f"{m}_uncoded_iou"]).any()]
    err = np.concatenate([a["wilor_coded_err_px"] for _, a in units]) if units else np.zeros(0)
    return {
        "every_item_present": bool(units) and len(units) == len(result["items"]) and all(m["npz_hash_ok"] for m, _ in units),
        "pose_units_intact": all(m["pose_unit_hash_ok"] for m, _ in units),
        "every_net_on_cuda": all(result["on_cuda"].values()),
        "sparse_jpegs_match_decoded_frames": all(g["holds"] for m, _ in units for g in m["decode"]["jpeg_gate"]),
        "mask_set_matches_record": all(m["mask_set"]["masks_rle_sha256"] == m["mask_set"]["record_masks_rle_sha256"] for m, _ in units),
        "streams_match_records": all(s["sha256"] == s["recorded_sha256"] for m, _ in units for s in m["streams"].values()),
        "lpips_cpu_matches_device_1e-3": bool(first) and all(m["checks"]["lpips_cpu_vs_device"] < 1e-3 for m in first),
        "dists_of_identical_is_zero": all(abs(m["checks"]["dists_identity"]) < 1e-4 for m, _ in units if "dists_identity" in m["checks"]),
        "scores_finite_on_hand_frames": all(np.isfinite(a["lpips"][:, h]).all() and np.isfinite(a["dists"][:, h]).all() for _, a, h in hand),
        "svt_lpips_rises_with_crf": bool(units) and float(np.nanmean(np.concatenate([a["lpips"][svt41] for _, a in units])))
        < float(np.nanmean(np.concatenate([a["lpips"][svt62] for _, a in units]))),
        # WiLoR re-run on the source with the same box reproduces H2's fit (the judge's input path is H2's).
        "wilor_rerun_reproduces_h2": bool(rerun) and float(np.median(np.concatenate(rerun))) < 0.005,
        # The uncoded render's silhouette is H2's (same mesh, same projection), up to the other hand in front.
        "uncoded_silhouette_matches_h2_iou": bool(iou_gap) and float(np.max(iou_gap)) < 0.02,
        # H2: the 100 ms choice moves WiLoR's joints 8.4 px from the uncoded pose (with the track's shape).
        "coded_pose_error_near_h2": bool(len(err)) and 3.0 < float(np.mean(err)) < 20.0,
        "renders_cover_pixels": all(int(a["wilor_covered"].sum()) > 0 for _, a in units if len(a["wilor_coded_err_px"])),
    }


def command_validate(args: argparse.Namespace) -> int:
    checks = validate_result(stage_dir())
    report = {"passed": all(checks.values()), "checks": checks}
    target = os.environ.get("PS_VALIDATION_PATH")
    if target:
        write_json(Path(target), report)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


# ----------------------------------------------------------------- report (CPU)

def item_summary(meta: dict[str, Any], a: dict[str, np.ndarray], svt_kbps: dict[str, float], parts_kbps: dict[str, float],
                 pose_kbps: dict[str, float]) -> dict[str, Any]:
    """Per variant: mean LPIPS, DISTS, flicker over hand-frames, PSNR from the summed error, rate; and the
    pose-fidelity error per judged source (mean over judged hands, share of the box)."""
    hand = a["pixels_r"] > 0
    pixels = float(a["pixels_s"][hand].sum())
    out: dict[str, Any] = {"unit": meta["unit"], "hand_frames": int(hand.sum()), "variants": {}}
    for i, v in enumerate(meta["variants"]):
        if v.startswith("svt@"):
            kbps = svt_kbps[v[4:]]
        else:
            m, kind = v.split(":")
            kbps = pose_kbps[m] + (parts_kbps[kind.split("@")[1]] if kind.startswith("hybrid@") else 0.0)
        from src.codecs.quality import psnr

        out["variants"][v] = {"kbps": kbps, "lpips": float(np.nanmean(a["lpips"][i, hand])),
                              "dists": float(np.nanmean(a["dists"][i, hand])),
                              "flicker": float(np.nanmean(a["flicker"][i, hand])) if np.isfinite(a["flicker"][i, hand]).any() else math.nan,
                              "psnr": psnr(float(a["sse"][i, hand].sum()) / (3 * pixels))}
    out["svt_lpips_r"] = {f"{p:g}": float(np.nanmean(a["svt_lpips_r"][j, hand])) for j, p in enumerate(POINTS)}
    if len(a["judge_row"]):
        ref = a["judge_kp2d"][0]
        errors = np.linalg.norm(a["judge_kp2d"] - ref[None], axis=-1).mean(-1) / a["judge_box_size"][None]
        out["pose_error"] = {v: float(errors[j].mean()) for j, v in enumerate(meta["judged"])}
    for j, m in enumerate(MODELS):
        hp = float(a[f"{m}_hand_px"].sum())
        out[m] = {"missed_share": float(a[f"{m}_missed"].sum()) / hp if hp else math.nan,
                  "spill_share": float(a[f"{m}_spill"].sum()) / max(1.0, float(a[f"{m}_covered"].sum())),
                  "outside_samples": float(a[f"{m}_outside"].sum()) / max(1.0, float(a[f"{m}_covered"].sum())),
                  "occluded_samples": float(a[f"{m}_occluded"].sum()) / max(1.0, float(a[f"{m}_covered"].sum())),
                  "lpips_rendered_pixels": float(np.nanmean(a["lpips_rendered"][j, hand])) if np.isfinite(a["lpips_rendered"][j, hand]).any() else math.nan,
                  "coded_err_px": float(a[f"{m}_coded_err_px"].mean()) if len(a[f"{m}_coded_err_px"]) else math.nan,
                  "tracks_per_minute": meta["tracks"][m] / (meta["frames"] / meta["fps"] / 60.0)}
    return out


def equal_quality(svt: list[dict[str, Any]], lpips: float, kbps: float) -> dict[str, Any]:
    if lpips < min(r["lpips"] for r in svt):
        return {"svt_kbps": None, "saving": None, "where": "better than every CRF"}
    rate = svt_rate_at_quality(svt, lpips)
    if rate is None:
        return {"svt_kbps": None, "saving": None, "where": "worse than every CRF"}
    return {"svt_kbps": rate, "saving": 1.0 - kbps / rate, "where": "inside"}


def command_report(args: argparse.Namespace) -> int:
    units = load_units(args.oracle, "oracle")
    h1 = {row["id"]: row for row in json.loads(Path(args.h1_report).read_text())["per_item"]}
    code = json.loads(Path(args.code).read_text())
    pose_kbps = {m: float(code["choices"][m][BUDGET_MS]["visor_kbps"]) for m in MODELS}
    items = []
    for meta, a in units:
        if not (a["pixels_r"] > 0).any():
            continue
        row = h1[meta["unit"]]
        parts = {p: sum(row["bits"][p][FILL]["kbps"][k] for k in ("forearm", "hand_no_fit")) for p in row["bits"]}
        items.append(item_summary(meta, a, hand_kbps(row), parts, pose_kbps))
    names = units[0][0]["variants"]

    def diff(metric: str, v: str, against: str, tag: str) -> dict[str, float]:
        d = np.array([it["variants"][v][metric] - it["variants"][against][metric] for it in items])
        return bootstrap_mean(d, np.random.default_rng(seed_of(f"{SEED}:{tag}:{metric}:{v}:{against}")))

    def gate(v: str) -> dict[str, Any]:
        d = diff("lpips", v, "svt@62", "gate")
        svt_curves = [[{"kbps": it["variants"][f"svt@{p:g}"]["kbps"], "lpips": it["variants"][f"svt@{p:g}"]["lpips"]} for p in POINTS]
                      for it in items]
        eq = [equal_quality(c, it["variants"][v]["lpips"], it["variants"][v]["kbps"]) for c, it in zip(svt_curves, items)]
        savings = [e["saving"] for e in eq if e["saving"] is not None]
        return {"variant": v, "lpips_minus_svt62": d, "passes": bool(d["high"] < 0),
                "items_better": int(sum(it["variants"][v]["lpips"] < it["variants"]["svt@62"]["lpips"] for it in items)),
                "equal_quality": {"better_than_every_crf": sum(e["where"] == "better than every CRF" for e in eq),
                                  "worse_than_every_crf": sum(e["where"] == "worse than every CRF" for e in eq),
                                  "median_saving_inside": float(np.median(savings)) if savings else None, "inside": len(savings)},
                "beside": {k: diff(k, v, "svt@62", "gate") for k in ("dists", "psnr", "flicker")}}

    def hybrid(m: str) -> dict[str, Any]:
        out = {}
        for p in POINTS:
            v = f"{m}:hybrid@{p:g}"
            d = []
            for it in items:
                rates = np.array([it["variants"][f"svt@{q:g}"]["kbps"] for q in POINTS])
                values = np.array([it["variants"][f"svt@{q:g}"]["lpips"] for q in POINTS])
                d.append(it["variants"][v]["lpips"] - at_rate(rates, values, it["variants"][v]["kbps"]))
            b = bootstrap_mean(np.array(d), np.random.default_rng(seed_of(f"{SEED}:hybrid:{v}")))
            out[f"{p:g}"] = {"kbps": float(np.mean([it["variants"][v]["kbps"] for it in items])), "lpips_minus_svt_at_rate": b,
                             "passes": bool(b["high"] < 0)}
        return out

    # Realism: KID and FID of hand crops against the source's, pooled over items; KID's interval by item.
    realism: dict[str, Any] = {}
    feats = [(meta, a["judge_features"]) for meta, a in units if a["judge_features"].shape[1]]
    if feats:
        judged = feats[0][0]["judged"]
        owner = np.concatenate([np.full(f.shape[1], i) for i, (_, f) in enumerate(feats)])
        source = np.concatenate([f[0] for _, f in feats]).astype(np.float64)
        for j, v in enumerate(judged[1:], start=1):
            x = np.concatenate([f[j] for _, f in feats]).astype(np.float64)
            kxx, kyy, kxy = kid_kernels(x, source)
            rng = np.random.default_rng(seed_of(f"{SEED}:kid:{v}"))
            point = kid_from(kxx, kyy, kxy, rng)
            boots = []
            for _ in range(KID_RESAMPLES):
                pick = rng.integers(0, len(feats), len(feats))
                idx = np.concatenate([np.nonzero(owner == i)[0] for i in pick])
                boots.append(kid_from(kxx[np.ix_(idx, idx)], kyy[np.ix_(idx, idx)], kxy[np.ix_(idx, idx)], rng, subsets=10))
            realism[v] = {"kid": point, "kid_low": float(np.percentile(boots, 2.5)), "kid_high": float(np.percentile(boots, 97.5)),
                          "fid": fid(x, source), "crops": int(len(x))}
    pose = {v: bootstrap_mean(np.array([it["pose_error"][v] for it in items if "pose_error" in it]),
                              np.random.default_rng(seed_of(f"{SEED}:pose:{v}"))) for v in JUDGED} if any("pose_error" in it for it in items) else {}
    tests = {m: {"arm": gate(f"{m}:arm"), "hand": gate(f"{m}:hand"), "arm_uncoded": gate(f"{m}:arm_uncoded"), "hybrid": hybrid(m)}
             for m in MODELS}
    w = tests["wilor"]
    hybrid_passes = any(r["passes"] for r in w["hybrid"].values())
    judges_match = bool(pose and realism) and (
        pose["wilor:arm"]["mean"] <= pose["svt@62"]["mean"] and w["arm"]["beside"]["dists"]["mean"] <= 0
        and realism["wilor:arm"]["kid"] <= realism["svt@62"]["kid"])
    if w["arm"]["passes"]:
        verdict = "build the generator" + ("" if w["hand"]["passes"] else ", drawing the arm")
    elif hybrid_passes:
        verdict = "build the generator for the hand; the forearm is coded"
    elif w["arm_uncoded"]["passes"] or judges_match:
        verdict = "to the user: the gate fails only through placement or on LPIPS alone"
    else:
        verdict = "not built; H3c next"
    decision: dict[str, Any] = {"gate": w["arm"], "hand": w["hand"]["passes"], "hybrid": hybrid_passes,
                                "arm_uncoded": w["arm_uncoded"]["passes"], "judges_match": judges_match, "verdict": verdict}
    mean_of = {v: {k: float(np.nanmean([it["variants"][v][k] for it in items])) for k in ("kbps", "lpips", "dists", "psnr", "flicker")}
               for v in names}
    out = {
        "rule": DECISION, "items": len(items), "pose_kbps": pose_kbps, "means": mean_of,
        "svt_lpips_r": {f"{p:g}": float(np.mean([it["svt_lpips_r"][f"{p:g}"] for it in items])) for p in POINTS},
        "tests": tests, "pose_fidelity": pose, "realism": realism,
        "render": {m: {k: float(np.nanmean([it[m][k] for it in items])) for k in items[0][m]} for m in MODELS},
        "decision": decision,
        "per_item": items,
        "inputs": {"oracle": [{"path": p, "sha256": file_sha256(Path(p))} for p in args.oracle],
                   "h1_report_sha256": file_sha256(Path(args.h1_report)), "code_sha256": file_sha256(Path(args.code))},
        "units_hash_ok": all(m["npz_hash_ok"] for m, _ in units),
    }
    target = Path(args.out) if args.out else stage_dir()
    write_json(target / "h3b-report.json", out)
    print(json.dumps({"decision": {k: v for k, v in decision.items() if k != "gate"} | {"gate": w["arm"]["lpips_minus_svt62"]},
                      "means": {v: mean_of[v] for v in names if "hybrid" not in v}}, indent=1, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="experiments.visor.h3b")
    commands = parser.add_subparsers(dest="command", required=True)
    oracle = commands.add_parser("oracle")
    oracle.add_argument("--eval-set", required=True)
    oracle.add_argument("--archive", required=True)
    oracle.add_argument("--masks", required=True)
    oracle.add_argument("--mask-record", required=True)
    oracle.add_argument("--svt-streams", action="append", required=True)
    oracle.add_argument("--svt-result", action="append", required=True)
    oracle.add_argument("--lpips-backbone", required=True)
    oracle.add_argument("--dists-backbone", required=True)
    oracle.add_argument("--inception", required=True)
    oracle.add_argument("--poses", action="append", required=True, help="H2's VISOR published.tar (both jobs)")
    oracle.add_argument("--code", required=True, help="H2's code.json")
    oracle.add_argument("--mano-left", required=True)
    oracle.add_argument("--mano-right", required=True)
    oracle.add_argument("--wilor-checkpoint", required=True)
    oracle.add_argument("--video", action="append", default=[])
    oracle.add_argument("--decoder", default="dav1d")
    oracle.add_argument("--items", default="all")
    oracle.add_argument("--frames", default="240")
    commands.add_parser("validate")
    report = commands.add_parser("report")
    report.add_argument("--oracle", action="append", required=True)
    report.add_argument("--h1-report", required=True)
    report.add_argument("--code", required=True)
    report.add_argument("--out", default="", help="default: the stage directory")
    args = parser.parse_args(argv)
    return {"oracle": command_oracle, "validate": command_validate, "report": command_report}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
