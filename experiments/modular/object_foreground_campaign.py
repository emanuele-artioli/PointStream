# ruff: noqa: E402
"""Separate-object foreground and articulated-warp campaign.

The runner uses the background choices and anchors from the 23 September
background campaign. Every independently moving player gets its own crop,
alpha, bbox sequence, and COCO-17 pose sequence. Target masks are scoring
inputs and encoder-side residual gates, never uncharged decoder inputs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3  # noqa: F401  # host C++ runtime before Torch pose import
import sys
import tempfile
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2

from experiments.modular.appearance_motion_probe import _bbox_affine, _box_payload, _extract_keypoints, _keypoint_affine
from experiments.modular.background_arms import _intra_still, _render_panorama
from experiments.modular.background_campaign import OUT_DIR as BG_OUT, _timed_vvc
from experiments.modular.foreground_campaign import CLIPS, ROOT, _alpha_wire, _correct, _pose_wire, _residual_signal, _row
from experiments.modular.measured_ladder import _direct_vvc_encode, _rgb_to_bgr, _rgb_to_yuv420, _yuv420_to_rgb, load_sequence
from scripts.background_probe import pack_panorama_side_data
from src.components.background.sidecar import IntraCodecSidecar
from src.components.codec.encode import BITSTREAM_SUFFIX, decode
from src.components.codec.frames import even_size
from src.components.codec.y4m import Y4M, read, write
from src.contracts.codecs import EncodeRequest, RateControl
from src.pipeline.residual.lossy import residual_clip_fraction

OUT = Path("/home/itec/emanuele/pointstream-data/outputs/modular/object-foreground-campaign")
N_FRAMES = 48


def split_object_tracks(mask: np.ndarray, *, min_area: int = 100) -> list[np.ndarray]:
    """Track connected foreground objects across a 48-frame binary mask.

    Args:
        mask: Boolean ``(T,H,W)`` source segmentation; never a decoder input.
        min_area: Minimum component size in pixels for an object candidate.

    Returns:
        One boolean ``(T,H,W)`` array per object, ordered by frame-zero
        horizontal position. Tracks remain disjoint on every frame and may
        have an empty frame when an object is absent. A two-player mask must
        produce two independently placed tracks, not one union crop.

    Raises:
        ValueError: Empty clip, wrong rank, or no qualifying object.

    A caller relies on each source player being crop-coded separately and on
    the union of returned tracks equalling the qualifying foreground pixels.
    """
    pixels = np.asarray(mask, dtype=bool)
    if pixels.ndim != 3 or not all(pixels.shape) or min_area < 1:
        raise ValueError("mask must be nonempty (T,H,W) and min_area positive")
    tracks: list[list[np.ndarray | None]] = []
    last: list[tuple[int, np.ndarray, tuple[int, int, int, int]]] = []
    first: list[tuple[int, float]] = []
    for t, frame in enumerate(pixels):
        n, labels, stats, centers = cv2.connectedComponentsWithStats(frame.astype(np.uint8), connectivity=8)
        components = [c for c in range(1, n) if stats[c, cv2.CC_STAT_AREA] >= min_area]
        candidates: list[tuple[float, int, int]] = []
        for c in components:
            for j, (seen, center, box) in enumerate(last):
                if t - seen > 12:
                    continue
                gate = max(120.0, 0.85 * np.hypot(max(box[2], stats[c, cv2.CC_STAT_WIDTH]), max(box[3], stats[c, cv2.CC_STAT_HEIGHT])))
                distance = float(np.linalg.norm(centers[c] - center))
                if distance <= gate:
                    candidates.append((distance / gate, c, j))
        used_c: set[int] = set()
        used_j: set[int] = set()
        primary_area: dict[int, int] = {}
        for _, c, j in sorted(candidates):
            if c in used_c or j in used_j:
                continue
            used_c.add(c)
            used_j.add(j)
            primary_area[j] = int(stats[c, cv2.CC_STAT_AREA])
            tracks[j][t] = labels == c
            last[j] = (t, centers[c].copy(), tuple(int(v) for v in stats[c, :4]))
        # Detached limbs and racquet pixels are part of their player, not new
        # objects. Assign only components close to a recently observed box.
        for c in components:
            if c in used_c:
                continue
            cx, cy = centers[c]
            near: list[tuple[float, int]] = []
            for j, (seen, _center, box) in enumerate(last):
                if t - seen > 12:
                    continue
                if j in primary_area and stats[c, cv2.CC_STAT_AREA] > 0.25 * primary_area[j]:
                    continue
                x, y, w, h = box
                gap = float(np.hypot(max(x-cx, 0, cx-(x+w)), max(y-cy, 0, cy-(y+h))))
                if gap <= max(40.0, 0.12 * np.hypot(w, h)):
                    near.append((gap, j))
            if near:
                _, j = min(near)
                fragment = labels == c
                tracks[j][t] = fragment if tracks[j][t] is None else (tracks[j][t] | fragment)
                used_c.add(c)
        for c in components:
            if c in used_c:
                continue
            items: list[np.ndarray | None] = [None] * len(pixels)
            items[t] = labels == c
            tracks.append(items)
            last.append((t, centers[c].copy(), tuple(int(v) for v in stats[c, :4])))
            first.append((t, float(centers[c, 0])))
    if not tracks:
        raise ValueError("mask has no qualifying object")
    order = sorted(range(len(tracks)), key=lambda j: first[j])
    blank = np.zeros(pixels.shape[1:], dtype=bool)
    return [np.stack([blank if frame is None else frame for frame in tracks[j]]) for j in order]


def _box(frame: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(frame)
    if len(ys) == 0:
        raise ValueError("absent object has no box")
    y1 = max(0, int(ys.min()) - 8)
    y2 = min(frame.shape[0], int(ys.max()) + 9)
    x1 = max(0, int(xs.min()) - 8)
    x2 = min(frame.shape[1], int(xs.max()) + 9)
    y1 -= y1 % 2
    x1 -= x1 % 2
    y2 -= (y2 - y1) % 2
    x2 -= (x2 - x1) % 2
    return y1, y2, x1, x2


def _boxes(track: np.ndarray) -> tuple[list[tuple[int, int, int, int]], np.ndarray, int]:
    visible = np.any(track, axis=(1, 2))
    first = int(np.flatnonzero(visible)[0])
    prior = _box(track[first])
    result = []
    for flag, frame in zip(visible, track, strict=True):
        if flag:
            prior = _box(frame)
        result.append(prior)
    return result, visible, first


def _affine_warp(crop: np.ndarray, alpha: np.ndarray, matrix: np.ndarray, source_box: tuple[int, int, int, int], output_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = output_shape
    y1, _, x1, _ = source_box
    local = np.asarray(matrix, dtype=np.float32).copy()
    local[:, 2] += local[:, :2] @ np.asarray([x1, y1], dtype=np.float32)
    pixels = cv2.warpAffine(crop, local, (width, height), flags=cv2.INTER_LINEAR)
    cover = cv2.warpAffine(alpha.astype(np.uint8), local, (width, height), flags=cv2.INTER_NEAREST)
    valid = cover.astype(bool)
    pixels[~valid] = 0
    return pixels, valid


def warp_articulated(
    crop_bgr: np.ndarray,
    alpha: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_box: tuple[int, int, int, int],
    output_shape: tuple[int, int],
    *,
    target_box: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp one object's crop and alpha using its transmitted COCO-17 joints.

    ``source_points`` and ``target_points`` are decoded float16 wire values
    with ``(17,3)`` layout. Return a full-frame BGR image and boolean cover.
    Use a piecewise or articulated transform where valid joints support it,
    and fall back to that object's bbox affine where they do not. The alpha
    and pixels must undergo the same geometry, with no target-mask access.
    Missing or nonfinite joints must never produce plausible invalid pixels.
    """
    src = np.asarray(source_points)
    dst = np.asarray(target_points)
    if src.shape != (17, 3) or dst.shape != (17, 3):
        raise ValueError("COCO-17 poses must have (17,3) shape")
    if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3 or alpha.shape != crop_bgr.shape[:2]:
        raise ValueError("crop and alpha shapes disagree")
    target_box = source_box if target_box is None else target_box
    pixels, cover = _affine_warp(crop_bgr, alpha, _bbox_affine(source_box, target_box), source_box, output_shape)
    valid = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1) & (src[:, 2] > 0) & (dst[:, 2] > 0)
    sy1, sy2, sx1, sx2 = source_box
    valid &= (src[:, 0] >= sx1) & (src[:, 0] < sx2) & (src[:, 1] >= sy1) & (src[:, 1] < sy2)
    if int(valid.sum()) < 3:
        return pixels, cover
    ty1, ty2, tx1, tx2 = target_box
    src_xy = np.concatenate([src[valid, :2].astype(np.float32), np.asarray([[sx1, sy1], [sx2-1, sy1], [sx2-1, sy2-1], [sx1, sy2-1]], dtype=np.float32)])
    dst_xy = np.concatenate([dst[valid, :2].astype(np.float32), np.asarray([[tx1, ty1], [tx2-1, ty1], [tx2-1, ty2-1], [tx1, ty2-1]], dtype=np.float32)])
    try:
        from scipy.spatial import Delaunay, QhullError
        triangles = Delaunay(src_xy).simplices
    except (ValueError, QhullError):
        return pixels, cover
    height, width = output_shape
    origin = np.asarray([sx1, sy1], dtype=np.float32)
    for tri in triangles:
        source_tri = src_xy[tri] - origin
        target_tri = dst_xy[tri]
        if abs(float(np.cross(target_tri[1] - target_tri[0], target_tri[2] - target_tri[0]))) < 1:
            continue
        x0 = max(0, int(np.floor(target_tri[:, 0].min())))
        x1 = min(width, int(np.ceil(target_tri[:, 0].max())) + 1)
        y0 = max(0, int(np.floor(target_tri[:, 1].min())))
        y1 = min(height, int(np.ceil(target_tri[:, 1].max())) + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        local_target = target_tri - np.asarray([x0, y0], dtype=np.float32)
        matrix = cv2.getAffineTransform(np.ascontiguousarray(source_tri), np.ascontiguousarray(local_target))
        patch = cv2.warpAffine(crop_bgr, matrix, (x1-x0, y1-y0), flags=cv2.INTER_LINEAR)
        matte = cv2.warpAffine(alpha.astype(np.uint8), matrix, (x1-x0, y1-y0), flags=cv2.INTER_NEAREST)
        triangle = np.zeros((y1-y0, x1-x0), dtype=np.uint8)
        cv2.fillConvexPoly(triangle, np.rint(local_target).astype(np.int32), 1)
        update = (triangle > 0) & (matte > 0)
        pixels[y0:y1, x0:x1][update] = patch[update]
        cover[y0:y1, x0:x1][triangle > 0] = matte[triangle > 0].astype(bool)
    pixels[~cover] = 0
    return pixels, cover


def _direct_timed_vvc(frames_rgb: np.ndarray, qp: int) -> tuple[bytes, np.ndarray, float, float, str, str]:
    """Force the original direct vvencapp path for Perricard's fixed background."""
    frames = even_size(frames_rgb)
    with tempfile.TemporaryDirectory(prefix="ps_object_bg_") as directory:
        root = Path(directory)
        luma, chroma = _rgb_to_yuv420(frames)
        source = root / "input.y4m"
        write(source, Y4M(width=int(luma.shape[2]), height=int(luma.shape[1]), fps=25.0, luma=luma, chroma=chroma))
        request = EncodeRequest(codec_name="vvc", rate_control=RateControl.QP, rate=qp, preset="faster", pix_fmt="yuv420p")
        bitstream = root / f"vvc_qp{qp}{BITSTREAM_SUFFIX['vvc']}"
        start = time.perf_counter()
        path, version = _direct_vvc_encode(source, bitstream, request)
        encode_s = time.perf_counter() - start
        payload = bitstream.read_bytes()
        if not payload:
            raise RuntimeError("empty VVC background bitstream")
        decoded_path = root / "decoded.y4m"
        start = time.perf_counter()
        decode(bitstream, decoded_path, request)
        decoded = read(decoded_path)
        if decoded.chroma is None:
            raise RuntimeError("VVC background decode has no chroma")
        rgb = _yuv420_to_rgb(decoded.luma, decoded.chroma)[:len(frames)]
        decode_s = time.perf_counter() - start
    if rgb.shape != frames_rgb.shape:
        raise RuntimeError(f"background decode shape changed: {rgb.shape} vs {frames_rgb.shape}")
    return payload, rgb, encode_s, decode_s, path, version


def _render_objects(background: np.ndarray, objects: list[dict], arm: str) -> tuple[np.ndarray, float]:
    result = background.copy()
    height, width = result.shape[1:3]
    start = time.perf_counter()
    for t in range(len(result)):
        for item in objects:
            if not item["presence"][t]:
                continue
            boxes = item["boxes"]
            first = item["first"]
            if arm == "bbox":
                matrix = _bbox_affine(boxes[first], boxes[t])
                patch, cover = _affine_warp(item["crop"], item["alpha"], matrix, boxes[first], (height, width))
            elif arm == "global_pose":
                matrix, _ = _keypoint_affine(item["poses"][first], item["poses"][t], boxes[first], boxes[t])
                patch, cover = _affine_warp(item["crop"], item["alpha"], matrix, boxes[first], (height, width))
            else:
                source_pose = item["poses"][first]
                target_pose = item["poses"][t]
                if source_pose is None or target_pose is None:
                    matrix = _bbox_affine(boxes[first], boxes[t])
                    patch, cover = _affine_warp(item["crop"], item["alpha"], matrix, boxes[first], (height, width))
                else:
                    patch, cover = warp_articulated(item["crop"], item["alpha"], source_pose, target_pose, boxes[first], (height, width), target_box=boxes[t])
            result[t][cover] = patch[cover, ::-1]  # decoded crop BGR; background is RGB
    return result, time.perf_counter() - start


def run_clip(clip_id: str, *, out_dir: Path = OUT, residuals: bool = True) -> dict[str, object]:
    """Measure separate-object bbox, global-pose, and articulated-pose arms.

    Load only the fixed 48-frame cached backgrounds; byte-check each new QP
    46 decode against its saved row before compositing. For each player send
    one AV1 QP42 crop and alpha, all placement bboxes, and pose bytes on pose
    arms. Evaluate residual-off, foreground-only, background-only, and both
    at coarsened QPs under the same source cap. Save a per-clip JSON ledger
    with B/F/M/R/H, all four PSNRs via ``score_regions``, clip fraction,
    encoder paths/versions, separate encode/decode clocks, and render time.
    Raise on changed background bytes or an empty native bitstream.
    """
    if clip_id not in CLIPS:
        raise ValueError(f"unknown clip {clip_id}")
    relative, source_json, representation, expected = CLIPS[clip_id]
    directory = ROOT / relative
    source, mask = load_sequence(directory / "window_48", directory / "masks_48.npz", N_FRAMES)
    row_file = BG_OUT / source_json
    background_doc = json.loads(row_file.read_text())
    anchor = next(r for r in background_doc["rows"] if r["representation"] == "source" and r["qp"] == 46)
    prior_bg = next(r for r in background_doc["rows"] if r["representation"] == representation and r["qp"] == 46)
    cache_file = BG_OUT / "cache" / f"{clip_id}-n48.npz"
    if not cache_file.is_file():
        raise FileNotFoundError(cache_file)
    cache = np.load(cache_file, mmap_mode="r")
    plate_s = float(cache["build_seconds"])
    print(f"{clip_id}: regenerate fixed {representation} QP46", flush=True)
    if representation == "cleaned_video":
        payload, background, bg_enc, bg_dec, bg_path, bg_version = _direct_timed_vvc(cache["cleaned"], 46)
        bg_side = 10
        bg_render = 0.0
    else:
        payload, decoded_plate, bg_path, bg_version, bg_enc, bg_dec = _intra_still(cache["plate"], 46)
        homographies = tuple(np.asarray(h, dtype=np.float64) for h in cache["homographies"])
        side = pack_panorama_side_data(homographies, plate_shape=tuple(int(x) for x in cache["plate"].shape[:2]), frame_shape=tuple(int(x) for x in source.shape[1:3]), fps=25.0)
        bg_side = len(side)
        started = time.perf_counter()
        background = _render_panorama(decoded_plate, side, N_FRAMES, tuple(int(x) for x in source.shape[1:3]))
        bg_render = time.perf_counter() - started
    B = len(payload) + bg_side
    if B != expected or B != int(prior_bg["total_bytes"]):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / f"{clip_id}-stopped.json").write_text(json.dumps({"clip_id": clip_id, "reason": "background byte mismatch; no composite", "expected_bytes": expected, "actual_bytes": B, "encoder_path": bg_path, "encoder_version": bg_version}, indent=2) + "\n")
        raise RuntimeError(f"{clip_id}: background moved to {B} B from {expected} B; no composite")
    if bg_path != prior_bg["tool_path"]:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / f"{clip_id}-stopped.json").write_text(json.dumps({"clip_id": clip_id, "reason": "background encoder path changed; no composite", "expected_path": prior_bg["tool_path"], "actual_path": bg_path}, indent=2) + "\n")
        raise RuntimeError(f"{clip_id}: background encoder changed to {bg_path}; no composite")
    print(f"{clip_id}: background matched {B} B", flush=True)

    started = time.perf_counter()
    track_masks = split_object_tracks(mask)
    tracking_s = time.perf_counter() - started
    objects: list[dict] = []
    sidecar = IntraCodecSidecar("av1", qp=42)
    crop_path, crop_version = sidecar.probe_encoder()
    crop_enc = crop_dec = 0.0
    prep_s = tracking_s
    for index, track in enumerate(track_masks):
        started = time.perf_counter()
        boxes, presence, first = _boxes(track)
        alpha_wire, alpha = _alpha_wire(track[first], boxes[first])
        presence_wire = np.packbits(presence.astype(np.uint8), bitorder="little").tobytes()
        bbox_wire = _box_payload(boxes)
        prep_s += time.perf_counter() - started
        y1, y2, x1, x2 = boxes[first]
        crop_input = np.ascontiguousarray(source[first, y1:y2, x1:x2, ::-1])
        coded_h = max(64, crop_input.shape[0] + (-crop_input.shape[0] % 8))
        coded_w = max(64, crop_input.shape[1] + (-crop_input.shape[1] % 8))
        padded = np.zeros((coded_h, coded_w, 3), dtype=np.uint8)
        padded[:crop_input.shape[0], :crop_input.shape[1]] = crop_input
        started = time.perf_counter()
        crop_wire = sidecar.encode(padded)
        crop_enc += time.perf_counter() - started
        if not crop_wire:
            raise RuntimeError(f"{clip_id}: empty AV1 crop for object {index}")
        started = time.perf_counter()
        crop = sidecar.decode(crop_wire)[:crop_input.shape[0], :crop_input.shape[1]]
        crop_dec += time.perf_counter() - started
        objects.append({"index": index, "first": first, "presence": presence, "boxes": boxes, "alpha": alpha, "alpha_wire": alpha_wire, "presence_wire": presence_wire, "bbox_wire": bbox_wire, "crop": crop, "crop_wire": crop_wire})
        print(f"{clip_id}: object {index}, first frame {first}, visible {int(presence.sum())}, crop {len(crop_wire)} B", flush=True)
    del track_masks
    F = sum(len(item["crop_wire"]) for item in objects)
    H = 1 + sum(len(item["alpha_wire"]) for item in objects)  # count byte and alpha headers
    bbox_M = sum(len(item["bbox_wire"]) + len(item["presence_wire"]) for item in objects)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{clip_id}.json"
    result: dict[str, object] = {
        "clip_id": clip_id,
        "source": str(directory / "window_48"), "mask": str(directory / "masks_48.npz"),
        "background_cache": str(cache_file), "background_row": str(row_file),
        "foreground_fraction": float(mask.mean()),
        "source_anchor": {key: anchor[key] for key in ("total_bytes", "psnr_overall", "psnr_fg", "psnr_bg", "psnr_weighted", "encode_seconds", "decode_seconds", "tool_path", "tool_version")},
        "background": {"representation": representation, "payload_bytes": len(payload), "side_bytes": bg_side, "total_bytes": B, "encoder_path": bg_path, "encoder_version": bg_version, "encode_seconds": bg_enc, "decode_seconds": bg_dec, "render_seconds": bg_render, "offline_plate_seconds": plate_s},
        "appearance": {"codec": "av1", "qp": 42, "encoder_path": crop_path, "encoder_version": crop_version, "payload_bytes": F, "encode_seconds": crop_enc, "decode_seconds": crop_dec},
        "objects": [{"index": item["index"], "first_frame": item["first"], "visible_frames": int(item["presence"].sum()), "crop_bytes": len(item["crop_wire"]), "alpha_bytes": len(item["alpha_wire"]), "bbox_bytes": len(item["bbox_wire"]), "presence_bytes": len(item["presence_wire"])} for item in objects],
        "rows": [], "residual_encodes": [], "completed": False,
    }

    def save() -> None:
        path.write_text(json.dumps(result, indent=2) + "\n")

    save()
    pose_enc = 0.0
    for arm in ("bbox", "global_pose", "articulated_pose"):
        if arm == "global_pose":
            for item in objects:
                started = time.perf_counter()
                visible_indices = np.flatnonzero(item["presence"])
                visible_poses, pose_info = _extract_keypoints(
                    _rgb_to_bgr(source[visible_indices]),
                    [item["boxes"][int(i)] for i in visible_indices],
                )
                poses = [None] * N_FRAMES
                for i, pose in zip(visible_indices, visible_poses, strict=True):
                    poses[int(i)] = pose
                pose_info["timeline_frames"] = N_FRAMES
                pose_wire, decoded_poses = _pose_wire(poses)
                pose_enc += time.perf_counter() - started
                item["poses"] = decoded_poses
                item["pose_wire"] = pose_wire
                result["objects"][item["index"]]["pose_bytes"] = len(pose_wire)
                result["objects"][item["index"]]["pose"] = pose_info
            save()
        M = bbox_M if arm == "bbox" else bbox_M + sum(len(item["pose_wire"]) for item in objects)
        print(f"{clip_id}: {arm} render", flush=True)
        base, paste_s = _render_objects(background, objects, arm)
        common_enc = bg_enc + crop_enc + prep_s + (pose_enc if arm != "bbox" else 0.0)
        common_dec = bg_dec + crop_dec
        common_render = bg_render + paste_s
        fraction = residual_clip_fraction(source.astype(np.int16) - base.astype(np.int16), mask)
        result.setdefault("clip_fraction", {})[arm] = fraction
        off = _row(arm, "neither", B=B, F=F, M=M, R=0, H=H, source=source, delivered=base, mask=mask, encode_s=common_enc, decode_s=common_dec, render_s=common_render, plate_s=plate_s, source_row=anchor)
        result["rows"].append(off)
        save()
        print(f"{clip_id}: {arm} off {off['total_bytes']} B weighted={off['scores']['weighted']:.3f} clip={fraction:.4f}", flush=True)
        if off["total_bytes"] > int(anchor["total_bytes"]):
            result.setdefault("stopped_arms", {})[arm] = "one crop per object with both residuals off exceeds anchor"
            if arm == "bbox":
                result["stopped"] = "separate-object one-crop residual-off point exceeds anchor; fixed background leaves insufficient headroom"
                save()
                return result
            save()
            continue
        if not residuals:
            del base
            continue
        qps = (54, 62) if fraction > 0.05 else (46, 54, 62)
        payloads: dict[str, dict[int, tuple[int, np.ndarray, float, float, str, str]]] = {"fg": {}, "bg": {}}
        signal_prep: dict[str, float] = {}
        for region, region_mask in (("fg", mask), ("bg", ~mask)):
            started = time.perf_counter()
            signal = _residual_signal(source, base, region_mask)
            signal_prep[region] = time.perf_counter() - started
            for qp in qps:
                print(f"{clip_id}: {arm} {region} residual QP{qp}", flush=True)
                data, pixels, enc_s, dec_s, encoder_path, version = _timed_vvc(signal, qp)
                if not data:
                    raise RuntimeError("empty VVC residual bitstream")
                payloads[region][qp] = (len(data), pixels, enc_s, dec_s, encoder_path, version)
                result["residual_encodes"].append({"arm": arm, "region": region, "qp": qp, "bytes": len(data), "encode_seconds": enc_s, "decode_seconds": dec_s, "encoder_path": encoder_path, "encoder_version": version})
                save()
            del signal
        for region in ("fg", "bg"):
            for qp in qps:
                nbytes, pixels, enc_s, dec_s, _, _ = payloads[region][qp]
                started = time.perf_counter()
                delivered = _correct(base, pixels if region == "fg" else None, pixels if region == "bg" else None)
                correction_s = time.perf_counter() - started
                result["rows"].append(_row(arm, f"{region}_only_qp{qp}", B=B, F=F, M=M, R=nbytes, H=H, source=source, delivered=delivered, mask=mask, encode_s=common_enc+signal_prep[region]+enc_s, decode_s=common_dec+dec_s, render_s=common_render+correction_s, plate_s=plate_s, source_row=anchor, fg_qp=qp if region == "fg" else None, bg_qp=qp if region == "bg" else None))
                save()
        for bg_qp in qps:
            for fg_qp in qps:
                f_bytes, f_pixels, f_enc, f_dec, _, _ = payloads["fg"][fg_qp]
                b_bytes, b_pixels, b_enc, b_dec, _, _ = payloads["bg"][bg_qp]
                started = time.perf_counter()
                delivered = _correct(base, f_pixels, b_pixels)
                correction_s = time.perf_counter() - started
                result["rows"].append(_row(arm, f"both_fg{fg_qp}_bg{bg_qp}", B=B, F=F, M=M, R=f_bytes+b_bytes, H=H, source=source, delivered=delivered, mask=mask, encode_s=common_enc+signal_prep["fg"]+signal_prep["bg"]+f_enc+b_enc, decode_s=common_dec+f_dec+b_dec, render_s=common_render+correction_s, plate_s=plate_s, source_row=anchor, fg_qp=fg_qp, bg_qp=bg_qp))
                save()
        del base, payloads
    result["completed"] = True
    result["residuals_enabled"] = residuals
    save()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", choices=tuple(CLIPS))
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--residuals", choices=("on", "off"), default="on")
    args = parser.parse_args()
    run_clip(args.clip, out_dir=args.out_dir, residuals=args.residuals == "on")
