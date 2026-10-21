"""K6: the player oracle, H3b for bodies on broadcast tennis (docs/experiments.md, 2026-10-21 K6).

    python -m experiments.tennis.k6 run --inputs DIR --pose PT --inspect BIN --image-ckpt PTH --image-sha256 HEX \\
        --video-ckpt PTH --video-sha256 HEX --lpips-backbone PTH --dists-backbone PTH --frames N --max-windows N \\
        --windows ID ... --sources MP4 ...
    python -m experiments.tennis.k6 validate [--directory DIR --summary JSON]
    python -m experiments.tennis.k6 report STAGE_DIR ... --out DIR

Per window (``tools/datasets/k6_inputs.py`` freezes the inputs): K4's conversion, the adopted players ((c)+(d) on
K4b.1's features and K8b's homographies), their pose (x1 crops under 15% of the frame height, K4b.1's whole-frame
detections above), K4b.3's keypoint codes on the scored (registered) frames, the pixel anchors (SVT-AV1 and DCVC-UF
over the coding resolution, bits on the players' region R), the oracle composites (``warp``: the frame's own pixels
moved to the coded keypoints; ``ref``: a reference frame's pixels moved there) and the judges on S, R plus every
drawn pixel. ``report`` applies `DECISION` over locally validated stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tarfile
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from experiments.tennis import k4
from experiments.tennis.k4b_court import choose_d, court_features
from experiments.tennis.k4b_features import convert
from experiments.tennis.k4b_pose import Crops, crop_box, fit, fit_whole, frame_dets, magnify, player_masks
from experiments.tennis.k4b_rate import code
from experiments.visor.b1 import file_sha256, progress, stage_dir, write_json
from src.segmentation.masks import ClipMasks, compress, decompress

FRAME = (1920, 1080)
#: Pixel anchors: (codec, value, scale); scale s codes the window at 1/s of 1080p (H3b2's sweep).
ANCHORS: tuple[tuple[str, int, int], ...] = (
    ("svt", 48, 1), ("svt", 62, 1), ("svt", 63, 2), ("svt", 55, 4), ("svt", 63, 4), ("svt", 63, 8),
    ("dcvc", 0, 1), ("dcvc", 0, 2), ("dcvc", 0, 4), ("dcvc", 21, 2),
)
#: K4b.3's codes: (q px, predictor, every); the first is K4b.3's choice.
CODES: tuple[tuple[int, str, int], ...] = ((4, "velocity", 4), (4, "velocity", 2), (4, "velocity", 1), (1, "previous", 1))
CHOSEN = CODES[0]
PRESET = 4
WHOLE_FRAME_ABOVE = 0.15  # median mask height share at or above which the whole-frame detections are used
SIGMA_OF_HEIGHT = 0.08
REF_GAP_S = 1.0
MIN_JOINT_KEYPOINTS = 4
TELEA_RADIUS = 5
FILL_PAD = 20
DISTS_PAD = 64
LPIPS_BATCH = 8
DCVC_ATTEMPTS = 3
SANE_LUMA_PSNR = 20.0
REVIEW_FRAMES = 2
REVIEW_SEED = "pointstream-k6-review"
HEIGHT_BINS = (0.0, 0.10, 0.15, 0.25, 1.01)
#: Parts of the amendment-2 accounting (K4's order, without the ball, whose bits come from K4's record).
PARTS = ("background", "other_person", "player", "racket")
#: The stage's claimed device as the supervisor set it; ultralytics rewrites CUDA_VISIBLE_DEVICES to "0" once loaded,
#: and the DCVC-UF worker refuses to run without the claimed UUID (K5's failed attempt `20261013T134053Z-43fa3fb2`).
STAGE_CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")

DECISION: dict[str, Any] = {
    "version": 1,
    "quality": "LPIPS (B2's AlexNet spatial map, full frame) averaged over S = R plus every oracle variant's drawn pixels; "
               "mean over the window's scored frames with R non-empty",
    "anchors": [list(a) for a in ANCHORS],
    "envelope": "per window, the upper concave hull (g5.envelope) of the anchor points on (kbps on R over scored seconds, "
                "-LPIPS); the rate for a quality is linear along the hull; better than every point: the best point's rate; "
                "worse than every point: saving min(0, 1 - r / lowest anchor rate)",
    "saving": "1 - keypoint kbps / envelope kbps at the variant's LPIPS",
    "bootstrap": {"resamples": 1000, "seed": "pointstream-k6", "level": 0.95, "by": "window"},
    "bound": {"variant": "warp", "code": list(CHOSEN), "ci_low_above": 0.50,
              "fallback": "lowest-rate passing code among every 2nd frame and every frame (q 4, velocity)"},
    "charged": {"variant": "ref", "code": "the bound's operating point", "ci_low_above": 0.25},
    "dists_sign": "a deciding variant whose DISTS mean saving has the other sign goes to the user",
    "stopping": "decided once every gate read has its CI wholly above or below its bar on >= 8 windows from >= 4 matches",
}


def code_name(c: tuple[int, str, int]) -> str:
    return f"q{c[0]}-{c[1]}-e{c[2]}"


def anchor_name(a: tuple[str, int, int]) -> str:
    return f"{a[0]}-{a[1]}-s{a[2]}"


def variant_names() -> list[str]:
    return ([anchor_name(a) for a in ANCHORS] + [f"warp@{code_name(c)}" for c in CODES]
            + [f"ref@{code_name(c)}" for c in CODES] + ["ref@uncoded"])


def scaled_size(scale: int) -> tuple[int, int]:
    from experiments.visor.h3b import scaled_size as size

    return size(FRAME[0], FRAME[1], scale)


# ----------------------------------------------------------------- the adopted players, their region and pose


def window_players(features: dict[str, Any], clip: ClipMasks, start: int, fits: dict[int, np.ndarray]) -> dict[str, Any]:
    """K4b.1's (c)+(d) picks on the window's canonical tracks, with K8b's registration (`k4b_court`)."""
    court = court_features(clip, features["aliases"], start, fits)
    picks = choose_d(features["features"], court)["c+d"]
    return {"players": picks, "court": court}


def touching_rackets(player: np.ndarray, rackets: list[np.ndarray]) -> np.ndarray:
    """Racket instances touching the player's mask (K4b.1's (c): 1 px dilation at SAM's size)."""
    grown = k4.dilate(player, 1)
    out = np.zeros_like(player, dtype=bool)
    for r in rackets:
        if (grown & r).any():
            out |= r
    return out


def frame_regions(frame: list[Any], aliases: dict[str, str], players: list[str]) -> dict[str, Any]:
    """At 1080p: each player's region (its mask and the rackets touching it), the rackets on players, and K4's parts."""
    from experiments.tennis.k4b_features import key

    masks = player_masks(frame, aliases, players)
    rackets = [i.mask().astype(bool) for i in frame if i.class_name == "racket"]
    regions: dict[str, np.ndarray] = {}
    bodies: dict[str, np.ndarray] = {}
    racket_on = np.zeros((FRAME[1], FRAME[0]), bool)
    for p, m in masks.items():
        if not m.any():
            continue
        touch = touching_rackets(m.astype(bool), rackets)
        bodies[p] = k4.upsample(m, FRAME)
        regions[p] = k4.upsample(m.astype(bool) | touch, FRAME)
        racket_on |= k4.upsample(touch, FRAME)
    labels = np.zeros((FRAME[1], FRAME[0]), np.uint8)
    for inst in frame:
        if inst.class_name in ("person", "player") and aliases[key(inst.class_name, inst.track_id)] not in players:
            labels[k4.upsample(inst.mask(), FRAME)] = PARTS.index("other_person")
    for m in masks.values():
        labels[k4.upsample(m, FRAME)] = PARTS.index("player")
    for r in rackets:
        labels[k4.upsample(r, FRAME)] = PARTS.index("racket")
    return {"regions": regions, "bodies": bodies, "racket": racket_on, "labels": labels}


def heights(regions: list[dict[str, np.ndarray]], players: list[str]) -> dict[str, float]:
    """Median height (px at 1080p) of each player's region over the frames it is present."""
    out = {}
    for p in players:
        hs = []
        for r in regions:
            if p in r and r[p].any():
                ys = np.nonzero(r[p].any(1))[0]
                hs.append(float(ys[-1] - ys[0] + 1))
        out[p] = float(np.median(hs)) if hs else 0.
    return out


def pose_window(crops: Crops, rgb: Any, frames: int, masks: list[dict[str, np.ndarray]], whole: list[dict[str, Any]],
                players: list[str], median_height: dict[str, float]) -> dict[str, list[dict[str, Any] | None]]:
    """Per player and frame, K4's fit with the estimator step 3 assigns (x1 crop under 15%, whole frame above), on the
    player's own mask at 1080p (``masks``, no racket), as K4b.2."""
    out: dict[str, list[dict[str, Any] | None]] = {p: [None] * frames for p in players}
    setting = {p: "whole" if median_height[p] >= WHOLE_FRAME_ABOVE * FRAME[1] else "crop1" for p in players}
    queued = False
    whole_fitted: dict[tuple[str, int], bool] = {}
    for t in range(frames):
        for p in players:
            m = masks[t].get(p)
            if m is None or not m.any():
                continue
            frame = {(t, q): x for q, x in masks[t].items()}
            whole_fit = fit_whole(m, frame, t, p, *frame_dets(whole[t]))
            if setting[p] == "whole":
                out[p][t] = {**whole_fit, "setting": "whole", "whole_fitted": whole_fit["fitted"]}
            else:
                whole_fitted[(p, t)] = whole_fit["fitted"]
                from experiments.visor.h1 import box_of

                x0, y0, x1, y1 = crop_box(box_of(m))  # type: ignore[arg-type]
                image, factors = magnify(np.ascontiguousarray(np.asarray(rgb[t])[y0:y1, x0:x1]), 1)
                crops.add(image, factors, (x0, y0), (p, t))
                queued = True
    if queued:
        for (p, t), boxes, kps in crops.run("crop1"):
            out[p][t] = {**fit(masks[t][p], boxes, kps), "setting": "crop1", "whole_fitted": whole_fitted[(p, t)]}
    return out


# ----------------------------------------------------------------- keypoint codes


def code_tracks(poses: dict[str, list[dict[str, Any] | None]], scored: list[int]) -> list[dict[str, Any]]:
    """Runs of consecutive fitted scored frames per player (a gap starts a run), with their uncoded keypoints."""
    scored_set = set(scored)
    runs: list[dict[str, Any]] = []
    for p, seq in sorted(poses.items()):
        current: dict[str, Any] | None = None
        for t, r in enumerate(seq):
            ok = r is not None and r.get("fitted") and t in scored_set
            if not ok:
                current = None
                continue
            if current is None:
                current = {"player": p, "t": [], "kps": []}
                runs.append(current)
            current["t"].append(t)
            current["kps"].append(r["kps"])  # type: ignore[index]
    return [{**r, "kps": np.asarray(r["kps"], np.float64)} for r in runs]


def coded_keypoints(tracks: list[dict[str, Any]]) -> tuple[dict[str, dict[tuple[str, int], np.ndarray]], dict[str, Any]]:
    """Per code: (player, t) -> decoded keypoints (17, 3) with the uncoded confidence (the visibility symbol); and per
    code the symbols the report pools for the rate."""
    decoded: dict[str, dict[tuple[str, int], np.ndarray]] = {}
    symbols: dict[str, Any] = {}
    for c in CODES:
        name = code_name(c)
        decoded[name] = {}
        residuals: list[int] = []
        visibility: list[int] = []
        start_bits = 0.
        for tr in tracks:
            dec, res, vis, start = code(tr["kps"], c[0], c[1], c[2])
            residuals.extend(res)
            visibility.extend(vis)
            start_bits += start
            for i, t in enumerate(tr["t"]):
                decoded[name][(tr["player"], t)] = np.concatenate([dec[i], tr["kps"][i, :, 2:3]], 1)
        symbols[name] = {"residuals": residuals, "visibility": visibility, "start_bits": start_bits,
                         "player_frames": sum(len(tr["t"]) for tr in tracks)}
    return decoded, symbols


# ----------------------------------------------------------------- the skeleton warp


def segment_distance2(px: np.ndarray, py: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    length2 = float(d @ d)
    if length2 < 1e-9:
        return (px - a[0]) ** 2 + (py - a[1]) ** 2
    u = np.clip(((px - a[0]) * d[0] + (py - a[1]) * d[1]) / length2, 0., 1.)
    return (px - a[0] - u * d[0]) ** 2 + (py - a[1] - u * d[1]) ** 2


def backward_map(px: np.ndarray, py: np.ndarray, target: np.ndarray, source: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Source position of each target pixel: the blend of each bone's similarity (target segment onto source segment)
    with weights exp(-d^2 / 2 sigma^2), d the distance to the target bone. Bones need both ends confident in both
    keypoint sets; without one, a translation by the mean displacement of jointly confident keypoints; None without."""
    conf = (target[:, 2] >= k4.KP_CONF) & (source[:, 2] >= k4.KP_CONF)
    bones = [(a, b) for a, b in k4.SKELETON if conf[a] and conf[b]]
    if not bones:
        if not conf.any():
            return None
        shift = (source[conf, :2] - target[conf, :2]).mean(0)
        return px + shift[0], py + shift[1]
    d2 = np.stack([segment_distance2(px, py, target[a, :2], target[b, :2]) for a, b in bones])
    w = np.exp(-(d2 - d2.min(0)) / (2 * sigma * sigma))
    w /= w.sum(0)
    z = px + 1j * py
    out = np.zeros_like(z)
    for k, (a, b) in enumerate(bones):
        p0, p1 = complex(*target[a, :2]), complex(*target[b, :2])
        q0, q1 = complex(*source[a, :2]), complex(*source[b, :2])
        m = (q1 - q0) / (p1 - p0) if abs(p1 - p0) > 1e-6 else 1.
        out += w[k] * (q0 + m * (z - p0))
    return out.real, out.imag


def target_box(target: np.ndarray, source: np.ndarray, region: np.ndarray, pad: float) -> tuple[int, int, int, int] | None:
    """Where the moved region can land: the source region's box relative to its confident keypoints' box, carried to
    the target's keypoints' box (scaled by their heights), padded by ``pad`` of the target's height."""
    conf_t, conf_s = target[:, 2] >= k4.KP_CONF, source[:, 2] >= k4.KP_CONF
    if not conf_t.any() or not conf_s.any():
        return None
    ys, xs = np.nonzero(region)
    if not len(ys):
        return None
    sx0, sy0 = source[conf_s, 0].min(), source[conf_s, 1].min()
    sx1, sy1 = source[conf_s, 0].max(), source[conf_s, 1].max()
    tx0, ty0 = target[conf_t, 0].min(), target[conf_t, 1].min()
    tx1, ty1 = target[conf_t, 0].max(), target[conf_t, 1].max()
    s = max(ty1 - ty0, 1.) / max(sy1 - sy0, 1.)
    h = max((ys.max() - ys.min() + 1) * s, ty1 - ty0, 8.)
    x0 = tx0 - (sx0 - xs.min()) * s - pad * h
    y0 = ty0 - (sy0 - ys.min()) * s - pad * h
    x1 = tx1 + (xs.max() + 1 - sx1) * s + pad * h
    y1 = ty1 + (ys.max() + 1 - sy1) * s + pad * h
    box = (max(0, int(math.floor(x0))), max(0, int(math.floor(y0))), min(FRAME[0], int(math.ceil(x1))), min(FRAME[1], int(math.ceil(y1))))
    return box if box[2] > box[0] and box[3] > box[1] else None


def draw(canvas: np.ndarray, drawn: np.ndarray, image: np.ndarray, region: np.ndarray, source_kp: np.ndarray,
         target_kp: np.ndarray, sigma: float) -> np.ndarray:
    """Draw ``image``'s ``region`` moved from ``source_kp`` to ``target_kp`` onto ``canvas`` (in place); returns this
    player's drawn mask (also OR-ed into ``drawn``)."""
    import cv2

    mine = np.zeros(drawn.shape, bool)
    box = target_box(target_kp, source_kp, region, 0.15)
    if box is None:
        return mine
    x0, y0, x1, y1 = box
    py, px = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    mapped = backward_map(px, py, target_kp, source_kp, sigma)
    if mapped is None:
        return mine
    mx, my = mapped[0].astype(np.float32), mapped[1].astype(np.float32)
    inside = cv2.remap(region.astype(np.uint8), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,)) > 0
    colour = cv2.remap(image, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    view = canvas[y0:y1, x0:x1]
    view[inside] = colour[inside]
    mine[y0:y1, x0:x1] = inside
    drawn |= mine
    return mine


def telea_fill(image: np.ndarray, region: np.ndarray) -> np.ndarray:
    """H3b's background where nothing is drawn: OpenCV Telea inside ``region`` (radius 5), on the region's padded box."""
    import cv2

    out = image.copy()
    if not region.any():
        return out
    ys, xs = np.nonzero(region)
    y0, y1 = max(0, ys.min() - FILL_PAD), min(image.shape[0], ys.max() + 1 + FILL_PAD)
    x0, x1 = max(0, xs.min() - FILL_PAD), min(image.shape[1], xs.max() + 1 + FILL_PAD)
    crop = np.ascontiguousarray(image[y0:y1, x0:x1])
    out[y0:y1, x0:x1] = cv2.inpaint(crop, region[y0:y1, x0:x1].astype(np.uint8), TELEA_RADIUS, cv2.INPAINT_TELEA)
    return out


def normalized(kp: np.ndarray, joint: np.ndarray, height: float) -> np.ndarray:
    pts = kp[joint, :2]
    return (pts - pts.mean(0)) / max(height, 1.)


def choose_reference(t: int, player: str, target: np.ndarray, poses: dict[str, list[dict[str, Any] | None]], scored: list[int],
                     region_height: dict[tuple[str, int], float], gap: int) -> int | None:
    """The player's fitted scored frame at least ``gap`` frames away whose uncoded keypoints are nearest the coded ones
    (each set centred on its confident mean, divided by its region's height); ties by the nearer, then earlier frame."""
    best: tuple[float, int, int] | None = None
    conf_t = target[:, 2] >= k4.KP_CONF
    for r in scored:
        if abs(r - t) < gap:
            continue
        rec = poses[player][r]
        if rec is None or not rec.get("fitted"):
            continue
        src = np.asarray(rec["kps"])
        joint = conf_t & (src[:, 2] >= k4.KP_CONF)
        if joint.sum() < MIN_JOINT_KEYPOINTS:
            continue
        d = float(np.linalg.norm(normalized(target, joint, region_height[(player, t)])
                                 - normalized(src, joint, region_height[(player, r)]), axis=1).mean())
        key = (d, abs(r - t), r)
        if best is None or key < best:
            best = key
    return None if best is None else best[2]


# ----------------------------------------------------------------- pixel anchors


def downscaled_source(yuv: Path, frames: int, scale: int, out: Path) -> Path:
    from experiments.visor.h3c import downscale_planes

    sw, sh = scaled_size(scale)
    raw = np.memmap(yuv, dtype=np.uint8, mode="r", shape=(frames, FRAME[1] * 3 // 2, FRAME[0]))
    with out.open("wb") as handle:
        for t in range(frames):
            handle.write(downscale_planes(np.asarray(raw[t]), FRAME[0], FRAME[1], sw, sh).tobytes())
    return out


def scaled_counts(maps: np.ndarray, parts: int, scale: int) -> np.ndarray:
    """MI-cell pixel counts per part of each frame's label map at the coded size (nearest for labels, as K4's
    accounting splits each block's bits by the parts' pixel shares)."""
    import cv2

    from experiments.tennis.windows import mi_counts

    sw, sh = scaled_size(scale)
    return np.stack([mi_counts(m if scale == 1 else cv2.resize(m, (sw, sh), interpolation=cv2.INTER_NEAREST), parts) for m in maps])


def region_labels(regions: np.ndarray, scale: int) -> np.ndarray:
    """R at the coded size (H3b2's area fraction >= 0.5) as a 0/1 label map per frame."""
    from experiments.visor.h3c import downscale_mask

    sw, sh = scaled_size(scale)
    return np.stack([downscale_mask(r, sw, sh).astype(np.uint8) for r in regions])


def code_svt(args: argparse.Namespace, yuv: Path, frames: int, fps: Fraction, crf: int, scale: int, scored_r: np.ndarray,
             parts: np.ndarray | None, work: Path, cpus: list[int]) -> dict[str, Any]:
    """SVT-AV1 at ``crf`` on the window at 1/``scale`` (K4's settings), decoded by dav1d, with K4's accounting of the bits
    on R over the scored frames (and on K4's parts over every frame at scale 1)."""
    from experiments.tennis.windows import mi_counts, stream_bits
    from src.codecs import svtav1

    sw, sh = scaled_size(scale)
    work.mkdir(parents=True, exist_ok=True)
    raw = downscaled_source(yuv, frames, scale, work / "frames.yuv") if scale > 1 else yuv
    stream, decoded = work / "stream.ivf", work / "decoded.yuv"
    command = svtav1.encode_command(args.encoder, raw, stream, width=sw, height=sh, fps=fps, frames=frames, crf=crf,
                                    preset=PRESET, full_range=False)
    timing = svtav1.run(command, 3600, cpus)
    svtav1.run(svtav1.decode_command(args.decoder, stream, decoded, threads=len(cpus)), 3600, cpus)
    if scale > 1:
        raw.unlink()
    counts = {"scored_r": np.stack([mi_counts(m, 2) for m in region_labels(scored_r, scale)])}
    if parts is not None:
        counts["parts"] = scaled_counts(parts, len(PARTS), scale)
    bits = stream_bits(args.inspect, stream, counts, frames)
    data = stream.read_bytes()
    got = decoded.stat().st_size // (sw * sh * 3 // 2)
    if got != frames:
        raise RuntimeError(f"SVT-AV1 CRF {crf} scale {scale}: decoded {got} of {frames} frames")
    stream.unlink()
    return {"codec": "svt", "value": crf, "scale": scale, "size": [sw, sh], "decoded": str(decoded),
            "payload_bytes": sum(svtav1.ivf_frames(data)), "stream_sha256": hashlib.sha256(data).hexdigest(),
            "region_bits": bits["bits"]["scored_r"][1], "accounted_bits": bits["accounted_bits"],
            "outside_window_bits": bits["outside_window_bits"], "displayed_twice": bits["displayed_twice"],
            "parts_bits": dict(zip(PARTS, bits["bits"]["parts"])) if parts is not None else None,
            "command": command, "encode_seconds": round(timing["wall"], 2), "inspect_seconds": bits["seconds"]}


def code_dcvc(args: argparse.Namespace, yuv: Path, frames: int, qp: int, scale: int, scored_r: np.ndarray, work: Path) -> dict[str, Any]:
    """DCVC-UF HT-L off the shelf (H3c's `code_dcvc`: B2's worker, the own-entropy split of the bits onto R), re-coded
    from scratch up to `DCVC_ATTEMPTS` times until the decode is deterministic, matches the encoder's I frame and is a
    picture of the source (K5's guard); every attempt is recorded."""
    from experiments.tennis.k5 import luma_psnr
    from experiments.visor.h3c import code_dcvc as h3c_code

    sw, sh = scaled_size(scale)
    work.mkdir(parents=True, exist_ok=True)
    source = downscaled_source(yuv, frames, scale, work / "source.yuv")
    region_file = work / "region.npy"
    np.save(region_file, region_labels(scored_r, scale).astype(bool))
    attempts = []
    if STAGE_CUDA_VISIBLE_DEVICES is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = STAGE_CUDA_VISIBLE_DEVICES  # h3c's worker and bit estimate read os.environ
    for attempt in range(DCVC_ATTEMPTS):
        rec = h3c_code(args, source, region_file, (sw, sh), frames, qp, args.video_ckpt, args.video_sha256, work / f"a{attempt}")
        psnr = luma_psnr(Path(rec["decoded"]), source, sw, sh, frames)
        ok = rec["deterministic"] and rec["decoder_matches_encoder_intra"] and psnr >= SANE_LUMA_PSNR
        attempts.append({"stream_sha256": rec["stream_sha256"], "bytes": rec["bytes"], "deterministic": rec["deterministic"],
                         "decoder_matches_encoder_intra": rec["decoder_matches_encoder_intra"], "luma_psnr": psnr})
        if ok:
            break
    else:
        raise RuntimeError(f"DCVC-UF QP {qp} scale {scale}: no sane decode in {DCVC_ATTEMPTS} attempts: {attempts}")
    source.unlink()
    return {"codec": "dcvc", "value": qp, "scale": scale, "size": [sw, sh], "decoded": rec["decoded"],
            "payload_bytes": rec["bytes"], "stream_sha256": rec["stream_sha256"], "region_bits": rec["region_bits"],
            "estimate_over_actual": rec["estimate_over_actual"], "attempts": attempts,
            "seconds": {"encode": rec["encode_seconds"], "decode": rec["decode_seconds"], "estimate": rec["estimate_seconds"]}}


# ----------------------------------------------------------------- one window


def review_frames(window_id: str, scored: list[int]) -> list[int]:
    if not scored:
        return []
    ranked = sorted(scored, key=lambda t: hashlib.sha256(f"{REVIEW_SEED}:{window_id}:{t}".encode()).hexdigest())
    return sorted(ranked[:REVIEW_FRAMES])


def save_review(out: Path, truth: np.ndarray, frames: dict[str, np.ndarray], regions: dict[str, np.ndarray], title: str) -> None:
    """Per player (rows): the source, then each listed composite, cropped around the player's region."""
    import cv2

    rows = []
    for p, r in sorted(regions.items()):
        if not r.any():
            continue
        ys, xs = np.nonzero(r)
        h = ys.max() - ys.min() + 1
        pad = int(0.3 * h) + 8
        y0, y1 = max(0, ys.min() - pad), min(FRAME[1], ys.max() + 1 + pad)
        x0, x1 = max(0, xs.min() - pad), min(FRAME[0], xs.max() + 1 + pad)
        side = 256
        tiles = []
        for name, image in [("source", truth), *frames.items()]:
            crop = image[y0:y1, x0:x1]
            f = side / max(crop.shape[:2])
            tile = cv2.resize(crop, (max(1, round(crop.shape[1] * f)), max(1, round(crop.shape[0] * f))), interpolation=cv2.INTER_CUBIC)
            canvas = np.full((side + 18, side, 3), 255, np.uint8)
            canvas[18:18 + tile.shape[0], :tile.shape[1]] = tile
            cv2.putText(canvas, name[:34], (2, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)
            tiles.append(canvas)
        rows.append(np.concatenate(tiles, 1))
    if rows:
        width = max(r.shape[1] for r in rows)
        sheet = np.concatenate([np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0)), constant_values=255) for r in rows])
        header = np.full((20, width, 3), 255, np.uint8)
        cv2.putText(header, title, (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), cv2.cvtColor(np.concatenate([header, sheet]), cv2.COLOR_RGB2BGR))


def process_window(args: argparse.Namespace, entry: dict[str, Any], source: Path, nets: dict[str, Any], crops: Crops,
                   pool: ThreadPoolExecutor, cpu_sets: list[list[int]], scratch: Path, publish: Path, device: str) -> dict[str, Any]:
    import cv2

    from experiments.visor.h3 import rgb_memmap
    from experiments.visor.h3b import dists_crops, lpips_reduce

    began = time.time()
    w = entry["window"]
    wid = w["id"]
    frames = min(int(args.frames), int(w["frames"]))
    fps = Fraction(w["fps"]).limit_denominator(1001)
    work = scratch / "work" / k4.safe(wid)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    root = Path(args.inputs) / "windows" / k4.safe(wid)
    for name, digest in entry["files"].items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"Changed input: {root / name}")
    if file_sha256(source) != w["sha256"]:
        raise ValueError(f"{wid}: original clip identity mismatch")
    yuv = work / "source.yuv"
    converted = convert(args.ffmpeg, {**w, "frames": frames}, source, yuv)
    truth = rgb_memmap(yuv, work / "truth.npy", frames, FRAME[0], FRAME[1], False, device)

    features = json.loads((root / "features.json").read_text())
    clip = ClipMasks.load(root / "sam-masks.rle")
    whole = json.loads(decompress((root / "pose.json.zst").read_bytes()))
    fits = {int(f): np.asarray(h, np.float64) for f, h in entry["fits"].items()}
    chosen = window_players(features, clip, int(w["start"]), fits)
    players = chosen["players"]
    scored_all = [t for t in range(frames) if int(w["start"]) + t in fits]
    per_frame = [frame_regions(clip.frames[t], features["aliases"], players) for t in range(frames)]
    regions = [f["regions"] for f in per_frame]
    median_height = heights([f["bodies"] for f in per_frame], players)
    poses = pose_window(crops, truth, frames, [f["bodies"] for f in per_frame], whole, players, median_height)
    union = np.stack([np.logical_or.reduce([*f["regions"].values(), np.zeros((FRAME[1], FRAME[0]), bool)]) for f in per_frame])
    scored = [t for t in scored_all if union[t].any()]
    scored_r = np.zeros_like(union)
    scored_r[scored] = union[scored]
    parts = np.stack([f["labels"] for f in per_frame])

    svt_jobs: dict[str, Future[dict[str, Any]]] = {}
    for i, a in enumerate([a for a in ANCHORS if a[0] == "svt"]):
        svt_jobs[anchor_name(a)] = pool.submit(code_svt, args, yuv, frames, fps, a[1], a[2], scored_r,
                                               parts if a[2] == 1 else None, work / anchor_name(a), cpu_sets[i % len(cpu_sets)])
    anchors: dict[str, dict[str, Any]] = {}
    for a in [a for a in ANCHORS if a[0] == "dcvc"]:
        anchors[anchor_name(a)] = code_dcvc(args, yuv, frames, a[1], a[2], scored_r, work / anchor_name(a))
    for a in ANCHORS:
        if a[0] == "svt":
            anchors[anchor_name(a)] = svt_jobs[anchor_name(a)].result()
    decodes = {}
    for name, rec in anchors.items():
        sw, sh = rec["size"]
        decodes[name] = rgb_memmap(Path(rec["decoded"]), work / f"{name}-rgb.npy", frames, sw, sh, False, device)
        Path(rec["decoded"]).unlink()

    tracks = code_tracks(poses, scored)
    decoded, symbols = coded_keypoints(tracks)
    region_height = {(p, t): float(np.ptp(np.nonzero(r[p].any(1))[0]) + 1) if p in r and r[p].any() else 1.
                     for t, r in enumerate(regions) for p in players}
    order = sorted(players, key=lambda p: median_height[p])  # far first
    gap = max(1, round(REF_GAP_S * float(fps)))
    names = variant_names()
    oracle_names = names[len(ANCHORS):]
    n_v = len(names)
    lp = np.full((n_v, frames), np.nan)
    lp_drawn = np.full((n_v, frames), np.nan)
    lp_racket = np.full((n_v, frames), np.nan)
    dists = np.full((n_v, frames), np.nan)
    sse = np.zeros((n_v, frames))
    pixels_s = np.zeros(frames, np.int64)
    pixels_r = np.zeros(frames, np.int64)
    outline: list[dict[str, Any]] = []
    references: dict[str, dict[str, int | None]] = {v: {} for v in oracle_names if v.startswith("ref")}
    identity_max_diff = None
    review_at = set(review_frames(wid, scored))
    for n, t in enumerate(scored):
        truth_t = np.asarray(truth[t])
        r_t = union[t]
        fill = telea_fill(truth_t, r_t)
        composites: dict[str, np.ndarray] = {}
        drawn_all = np.zeros_like(r_t)
        drawn: dict[str, np.ndarray] = {}
        for v in oracle_names:
            kind, code_key = v.split("@")
            canvas = fill.copy()
            mask_v = np.zeros_like(r_t)
            for p in order:
                pose = poses[p][t]
                if p not in regions[t] or pose is None or not pose.get("fitted"):
                    continue
                uncoded = np.asarray(pose["kps"], np.float64)
                target = uncoded if code_key == "uncoded" else decoded[code_key].get((p, t))
                if target is None:
                    continue
                sigma = SIGMA_OF_HEIGHT * region_height[(p, t)]
                if kind == "warp":
                    mine = draw(canvas, mask_v, truth_t, regions[t][p], uncoded, target, sigma)
                else:
                    r = choose_reference(t, p, target, poses, scored, region_height, gap)
                    references[v][f"{p}@{t}"] = r
                    if r is None:
                        continue
                    mine = draw(canvas, mask_v, np.asarray(truth[r]), regions[r][p], np.asarray(poses[p][r]["kps"], np.float64),  # type: ignore[index]
                                target, sigma)
                rp = regions[t][p]
                outline.append({"variant": v, "player": p, "t": t, "region_px": int(rp.sum()), "drawn_px": int(mine.sum()),
                                "missed_px": int((rp & ~mine).sum()), "over_px": int((mine & ~r_t).sum()),
                                "height": region_height[(p, t)]})
            composites[v] = canvas
            drawn[v] = mask_v
            drawn_all |= mask_v
        if identity_max_diff is None:  # the warp at the uncoded keypoints reproduces the source on what it draws
            canvas = fill.copy()
            mask_id = np.zeros_like(r_t)
            for p in order:
                pose = poses[p][t]
                if p in regions[t] and pose is not None and pose.get("fitted"):
                    kp = np.asarray(pose["kps"], np.float64)
                    draw(canvas, mask_id, truth_t, regions[t][p], kp, kp, SIGMA_OF_HEIGHT * region_height[(p, t)])
            if mask_id.any():
                identity_max_diff = int(np.abs(canvas[mask_id].astype(int) - truth_t[mask_id].astype(int)).max())
        s_t = r_t | drawn_all
        racket_t = per_frame[t]["racket"]
        outs = []
        for name in names[:len(ANCHORS)]:
            up = cv2.resize(np.asarray(decodes[name][t]), FRAME, interpolation=cv2.INTER_CUBIC)
            out = truth_t.copy()
            out[r_t] = up[r_t]
            outs.append(out)
        outs += [composites[v] for v in oracle_names]
        masks = [[s_t, drawn.get(v, np.zeros_like(r_t)), racket_t] for v in names]
        reduced = lpips_reduce(nets["lpips"], outs, truth_t, masks, device)
        ys, xs = np.nonzero(s_t)
        box = (slice(max(0, ys.min() - DISTS_PAD), min(FRAME[1], ys.max() + 1 + DISTS_PAD)),
               slice(max(0, xs.min() - DISTS_PAD), min(FRAME[0], xs.max() + 1 + DISTS_PAD)))
        dists[:, t] = dists_crops(nets["dists"], [o[box] for o in outs], truth_t[box], s_t[box], device)
        for i, o in enumerate(outs):
            lp[i, t], lp_drawn[i, t], lp_racket[i, t] = reduced[i]
            sse[i, t] = float(((o[s_t].astype(np.int32) - truth_t[s_t].astype(np.int32)) ** 2).sum())
        pixels_s[t], pixels_r[t] = int(s_t.sum()), int(r_t.sum())
        if t in review_at:
            pick = {k: outs[names.index(k)] for k in (f"warp@{code_name(CHOSEN)}", f"ref@{code_name(CHOSEN)}", "svt-63-s4", "dcvc-0-s4", "svt-62-s1")}
            save_review(publish / k4.safe(wid) / f"review-t{t:03d}.png", truth_t, pick, regions[t], f"{wid} t={t}")
        print(f"phase=score window={wid} frame={n + 1}/{len(scored)}", flush=True)

    out_dir = publish / k4.safe(wid)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "scores.npz", lpips=lp, lpips_drawn=lp_drawn, lpips_racket=lp_racket, dists=dists, sse=sse,
                        pixels_s=pixels_s, pixels_r=pixels_r, scored=np.asarray(scored, np.int64))
    pose_rows = {p: [None if r is None else {k: r[k] for k in ("fitted", "setting", "kps", "inside", "confident", "whole_fitted") if k in r}
                     for r in seq] for p, seq in poses.items()}
    keypoints = {name: {f"{p}@{t}": np.round(v, 4).tolist() for (p, t), v in d.items()} for name, d in decoded.items()}
    for name in ("features.json", "sam-masks.rle"):
        shutil.copyfile(root / name, out_dir / name)
    write_json(out_dir / "fits.json", entry["fits"])
    (out_dir / "poses.json.zst").write_bytes(compress(json.dumps(pose_rows).encode()))
    (out_dir / "keypoints.json.zst").write_bytes(compress(json.dumps({"decoded": keypoints, "symbols": symbols}).encode()))
    (out_dir / "outline.json.zst").write_bytes(compress(json.dumps({"rows": outline, "references": references}).encode()))
    seconds = len(scored) / float(fps)
    row = {
        "id": wid, "k4_window": entry.get("k4_window"), "match": wid.split("/")[0], "fps": w["fps"], "frames": frames,
        "start": w["start"], "convert": converted, "players": players, "court": chosen["court"],
        "median_height": median_height, "scored": scored, "scored_registered": scored_all, "scored_seconds": seconds,
        "variants": names, "identity_max_diff": identity_max_diff,
        "anchors": {k: {**{x: v for x, v in rec.items() if x != "decoded"}, "kbps_r": rec["region_bits"] / seconds / 1000 if seconds else None,
                        "kbps": 8 * rec["payload_bytes"] / (frames / float(fps)) / 1000}
                    for k, rec in anchors.items()},
        "region_pixels_mean": float(pixels_r[scored].mean()) if scored else 0.,
        "parts_pixel_share": {p: float((parts == i).mean()) for i, p in enumerate(PARTS)},
        "seconds": round(time.time() - began, 2),
    }
    row["files"] = {p.name: file_sha256(p) for p in sorted(out_dir.iterdir()) if p.is_file()}
    write_json(out_dir / "window.json", row)
    shutil.rmtree(work, ignore_errors=True)
    return row


def command_run(args: argparse.Namespace) -> int:
    began = time.time()
    import torch

    from src.codecs import svtav1
    from src.codecs.quality import load_lpips, lpips_record
    from experiments.visor.h3b import load_dists

    inputs = json.loads((Path(args.inputs) / "inputs.json").read_text())
    if len(args.windows) != len(args.sources) or len(set(args.windows)) != len(args.windows):
        raise ValueError("Give one source per window, without repeated windows")
    chosen = args.windows[:args.max_windows or None]
    scratch = Path(os.environ["PS_SCRATCH_DIR"])
    publish = scratch / "publish"
    publish.mkdir(exist_ok=True)
    # The staging cache keeps files read-only and drops the execute bit: run a copy (as H1 and K4).
    inspect = scratch / "bin" / Path(args.inspect).name
    inspect.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.inspect, inspect)
    inspect.chmod(0o755)
    args.inspect = str(inspect)
    os.environ["YOLO_CONFIG_DIR"] = str(scratch / "ultralytics")
    device = "cuda"
    limit = int(os.environ.get("PS_GPU_MEMORY_LIMIT_MIB", "24000"))
    torch.cuda.set_per_process_memory_fraction(min(1.0, limit * 2**20 / torch.cuda.get_device_properties(0).total_memory))
    nets = {"lpips": load_lpips(Path(args.lpips_backbone), device), "dists": load_dists(Path(args.dists_backbone), device)}
    poses = k4.Poses([("yolo11x-pose", args.pose)])
    crops = Crops(poses)
    allowance = max(2, int(os.environ.get("PS_CPU_ALLOWANCE") or os.cpu_count() or 2))
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(allowance))
    half = max(1, len(cpus) // 2)
    cpu_sets = [cpus[:half], cpus[half:] or cpus[:half]]
    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for n, wid in enumerate(chosen):
            torch.cuda.reset_peak_memory_stats()
            rows.append(process_window(args, inputs["windows"][wid], Path(args.sources[args.windows.index(wid)]), nets, crops,
                                       pool, cpu_sets, scratch, publish, device))
            print(f"phase=window window={wid} done={n + 1}/{len(chosen)} seconds={rows[-1]['seconds']}", flush=True)
            progress(n + 1)
    summary = {"stage": os.environ.get("PS_STAGE"), "requested": args.windows, "configured_windows": int(args.max_windows),
               "frames_argument": int(args.frames), "windows": rows, "decision": DECISION, "codes": [list(c) for c in CODES],
               "inputs_sha256": file_sha256(Path(args.inputs) / "inputs.json"), "window_records": {w: args_entry
               for w, args_entry in ((w, {k: v for k, v in inputs["windows"][w].items() if k != "fits"}) for w in chosen)},
               "tools": {"svt-av1": svtav1.tool(args.encoder), "dav1d": svtav1.tool(args.decoder), "ffmpeg": svtav1.tool(args.ffmpeg),
                         "inspect_sha256": file_sha256(Path(args.inspect))},
               "weights": {"lpips": lpips_record(Path(args.lpips_backbone)), "dists_sha256": file_sha256(Path(args.dists_backbone)),
                           "dcvc_image_sha256": args.image_sha256, "dcvc_video_sha256": args.video_sha256,
                           "pose_sha256": file_sha256(Path(args.pose))},
               "pose": poses.record(), "on_cuda": {k: next(v.parameters()).is_cuda for k, v in nets.items()},
               "gpu": torch.cuda.get_device_name(0), "gpu_peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
               "gpu_memory_limit_mib": limit, "seconds": round(time.time() - began, 2)}
    with tarfile.open(stage_dir() / "published.tar", "w") as tar:
        tar.add(publish, arcname="publish")
    write_json(stage_dir() / "k6.json", summary)
    return 0


# ----------------------------------------------------------------- validation (recomputed from the published files)


def validate_window(root: Path, row: dict[str, Any]) -> dict[str, bool]:
    files = all(file_sha256(root / n) == h for n, h in row["files"].items())
    features = json.loads((root / "features.json").read_text())
    clip = ClipMasks.load(root / "sam-masks.rle")
    fits = {int(f): np.asarray(h, np.float64) for f, h in json.loads((root / "fits.json").read_text()).items()}
    picks = window_players(features, clip, int(row["start"]), fits)["players"] == row["players"]
    registered = [t for t in range(row["frames"]) if int(row["start"]) + t in fits]
    scored_ok = row["scored_registered"] == registered and set(row["scored"]) <= set(registered)
    poses = json.loads(decompress((root / "poses.json.zst").read_bytes()))
    published = json.loads(decompress((root / "keypoints.json.zst").read_bytes()))
    decoded, symbols = coded_keypoints(code_tracks(poses, row["scored"]))
    again = {name: {f"{p}@{t}": np.round(v, 4).tolist() for (p, t), v in d.items()} for name, d in decoded.items()}
    codes_ok = again == published["decoded"] and json.loads(json.dumps(symbols)) == published["symbols"]
    a = np.load(root / "scores.npz")
    scored = row["scored"]
    names = row["variants"]
    finite = all(np.isfinite(a[k][:, scored]).all() for k in ("lpips", "dists", "sse"))
    unscored = [t for t in range(row["frames"]) if t not in set(scored)]
    nan_elsewhere = all(np.isnan(a["lpips"][:, unscored]).all() for _ in [0])
    covers = bool((a["pixels_s"][scored] >= a["pixels_r"][scored]).all() and (a["pixels_r"][scored] > 0).all())
    anchors = row["anchors"]
    svt_ok = all(0 <= r["region_bits"] <= r["accounted_bits"] and r["payload_bytes"] > 0 for r in anchors.values() if r["codec"] == "svt")
    parts_ok = all(anchors[anchor_name(x)]["parts_bits"] is not None for x in ANCHORS if x[0] == "svt" and x[2] == 1)
    dcvc_ok = all((last := r["attempts"][-1])["deterministic"] and last["decoder_matches_encoder_intra"] and last["luma_psnr"] >= SANE_LUMA_PSNR
                  and r["region_bits"] >= 0 for r in anchors.values() if r["codec"] == "dcvc")
    if scored:
        mean = {n: float(np.mean(a["lpips"][names.index(n), scored])) for n in names}
        ladder = mean["svt-48-s1"] < mean["svt-62-s1"] < mean["svt-63-s8"] and mean["dcvc-0-s1"] < mean["dcvc-0-s4"]
        identity = row["identity_max_diff"] == 0
        review = len(list(root.glob("review-t*.png"))) == min(REVIEW_FRAMES, len(scored))
    else:
        ladder = identity = review = True
    return {"files_identified": files, "players_recomputed": picks, "scored_are_registered_frames": scored_ok,
            "keypoint_codes_recomputed": codes_ok, "scores_finite_on_scored_frames": finite, "no_scores_elsewhere": nan_elsewhere,
            "s_covers_r": covers, "svt_accounting": svt_ok, "parts_accounted_at_full_resolution": parts_ok,
            "dcvc_decodes_sane": dcvc_ok, "lpips_rises_as_the_anchors_coarsen": ladder,
            "identity_warp_reproduces_source": identity, "review_images": review}


def validate(directory: Path, summary: dict[str, Any]) -> dict[str, Any]:
    rows = summary["windows"]
    requested = summary["requested"][:summary["configured_windows"] or None]
    per = {r["id"]: validate_window(directory / k4.safe(r["id"]), r) for r in rows}
    checks = {
        "windows_requested": bool(rows) and [r["id"] for r in rows] == requested,
        "frames_as_configured": all(r["frames"] == min(summary["frames_argument"], r["convert"]["frames"]) for r in rows),
        "decision_frozen": summary["decision"] == DECISION and summary["codes"] == [list(c) for c in CODES],
        "nets_on_cuda": all(summary["on_cuda"].values()),
        "peak_within_declaration": summary["gpu_peak_gib"] * 1024 <= summary["gpu_memory_limit_mib"],
        **{k: all(v[k] for v in per.values()) for k in next(iter(per.values()))},
    }
    return {"passed": all(checks.values()), "checks": checks, "windows": {w: [k for k, x in v.items() if not x] for w, v in per.items()}}


def command_validate(args: argparse.Namespace) -> int:
    summary = json.loads((args.summary or stage_dir() / "k6.json").read_text())
    if args.directory:
        report = validate(args.directory, summary)
    else:
        with tempfile.TemporaryDirectory(prefix="k6-validate-", dir="/tmp") as temp:
            with tarfile.open(stage_dir() / "published.tar") as archive:
                archive.extractall(temp, filter="data")
            report = validate(Path(temp) / "publish", summary)
    if os.environ.get("PS_VALIDATION_PATH"):
        write_json(Path(os.environ["PS_VALIDATION_PATH"]), report)
    print(json.dumps(report))
    return 0 if report["passed"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    for name in ("inputs", "pose", "inspect", "image-ckpt", "image-sha256", "video-ckpt", "video-sha256", "lpips-backbone",
                 "dists-backbone"):
        run.add_argument("--" + name, required=True)
    run.add_argument("--frames", type=int, required=True)
    run.add_argument("--max-windows", type=int, required=True, help="0 runs every listed window")
    run.add_argument("--windows", nargs="+", required=True)
    run.add_argument("--sources", nargs="+", required=True)
    run.add_argument("--encoder", default="SvtAv1EncApp")
    run.add_argument("--decoder", default="dav1d")
    run.add_argument("--ffmpeg", default="ffmpeg")
    val = sub.add_parser("validate")
    val.add_argument("--directory", type=Path)
    val.add_argument("--summary", type=Path)
    rep = sub.add_parser("report")
    rep.add_argument("stages", nargs="+", type=Path, help="locally validated stages: k6.json, local-validation.json, x/publish")
    rep.add_argument("--k4-rows", type=Path, help="K4's published window rows (broadcast__*.json), for the ball's bits")
    rep.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "report":
        from experiments.tennis.k6_report import command_report

        return command_report(args)
    return {"run": command_run, "validate": command_validate}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
