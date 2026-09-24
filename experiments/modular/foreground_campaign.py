"""48-frame foreground campaign on the three fixed QP-46 backgrounds.

The first crop's binary alpha is transmitted once.  The decoder warps that
alpha with the same motion as the appearance crop; target masks are used only
by the encoder to form residual signals and by the evaluator to score frames.
Each residual is a separate VVC video whose neutral pixel is 128.
"""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3  # noqa: F401  # load host C++ runtime before pose backend imports Torch
import struct
import sys
import time
import zlib

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import numpy as np

from experiments.modular.appearance_motion_probe import (
    _bbox_affine,
    _bboxes,
    _box_payload,
    _extract_keypoints,
    _keypoint_affine,
)
from experiments.modular.background_arms import _intra_still, _render_panorama
from experiments.modular.background_campaign import OUT_DIR, _timed_vvc
from experiments.modular.measured_ladder import (
    _rgb_to_bgr,
    load_sequence,
    score_regions,
)
from scripts.background_probe import pack_panorama_side_data
from src.components.motion.keypoints import KeypointMotionEncoder
from src.components.background.sidecar import IntraCodecSidecar
from src.contracts.keypoints import COCO_17
from src.pipeline.residual.lossy import residual_clip_fraction

OUT = Path("/tmp/pointstream-foreground-campaign")
CLIPS = {
    "perricard002": ("alcaraz_perricard/scene_002", "perricard002.json", "cleaned_video", 86894),
    "alcaraz000": ("alcaraz_highlights/scene_000", "alcaraz000.json", "registered_panorama", 24648),
    "federer007": ("federer_djokovic/scene_007", "federer007-matched.json", "registered_panorama", 35763),
}
ROOT = Path("/home/itec/emanuele/pointstream-data/outputs/bp46-long-scenes/clips")


def _scores(source: np.ndarray, delivered: np.ndarray, mask: np.ndarray) -> dict[str, float | None]:
    # score_regions uses mean per-frame PSNR.  Calling it one frame at a time
    # preserves that definition without materializing two 10 GB float64 clips.
    values = [
        score_regions(source[i:i+1], delivered[i:i+1], mask[i:i+1], fg_weight=0.7, bg_weight=0.3)
        for i in range(source.shape[0])
    ]
    means = []
    for column in range(3):
        finite = [float(row[column]) for row in values if np.isfinite(row[column])]
        means.append(float(np.mean(finite)) if finite else float("inf"))
    overall, fg, bg = means
    weighted = 0.7 * fg + 0.3 * bg if np.isfinite(fg) and np.isfinite(bg) else None
    return {"overall": overall, "foreground": fg, "background": bg, "weighted": weighted}


def _alpha_wire(mask0: np.ndarray, box: tuple[int, int, int, int]) -> tuple[bytes, np.ndarray]:
    y1, y2, x1, x2 = box
    crop = np.ascontiguousarray(mask0[y1:y2, x1:x2], dtype=np.uint8)
    header = struct.pack("<II", int(crop.shape[0]), int(crop.shape[1]))
    payload = header + zlib.compress(np.packbits(crop, bitorder="little").tobytes(), 9)
    height, width = struct.unpack("<II", payload[:8])
    bits = np.unpackbits(np.frombuffer(zlib.decompress(payload[8:]), dtype=np.uint8), bitorder="little")
    restored = bits[: height * width].reshape(height, width).astype(bool)
    if not np.array_equal(restored, crop.astype(bool)):
        raise RuntimeError("alpha mask wire round trip failed")
    return payload, restored


def _warp_and_paste(
    background: np.ndarray,
    crop_bgr: np.ndarray,
    alpha: np.ndarray,
    source_box: tuple[int, int, int, int],
    matrices: list[np.ndarray],
) -> tuple[np.ndarray, float]:
    n, height, width, _ = background.shape
    result = np.empty_like(background)
    y1, _y2, x1, _x2 = source_box
    started = time.perf_counter()
    for index in range(n):
        local = np.asarray(matrices[index], dtype=np.float32).copy()
        local[:, 2] += local[:, :2] @ np.asarray([x1, y1], dtype=np.float32)
        patch = cv2.warpAffine(crop_bgr, local, (width, height), flags=cv2.INTER_LINEAR)
        cover = cv2.warpAffine(alpha.astype(np.uint8), local, (width, height), flags=cv2.INTER_NEAREST).astype(bool)
        result[index] = background[index]
        result[index][cover] = patch[cover]
    return result, time.perf_counter() - started


def _pose_wire(poses: list[np.ndarray | None]) -> tuple[bytes, list[np.ndarray | None]]:
    encoder = KeypointMotionEncoder(schema=COCO_17, values_per_joint=3, bytes_per_value=2)
    wire = bytearray()
    decoded: list[np.ndarray | None] = []
    for pose in poses:
        values = np.zeros((17, 3), dtype=np.float32) if pose is None else pose
        _desc, payload = encoder.encode(values)
        wire.extend(payload)
        decoded.append(None if pose is None else encoder.decode(payload))
    return bytes(wire), decoded


def _residual_signal(source: np.ndarray, base: np.ndarray, region: np.ndarray) -> np.ndarray:
    # RGB is kept throughout the residual color path. Neutral 128 is exact
    # before coding; VVC chroma bleed after decoding is included in scoring.
    signed = source.astype(np.int16) - base.astype(np.int16)
    signal = np.full(source.shape, 128, dtype=np.uint8)
    signal[region] = np.clip(signed[region] + 128, 0, 255).astype(np.uint8)
    return signal


def _correct(base: np.ndarray, foreground: np.ndarray | None, background: np.ndarray | None) -> np.ndarray:
    corrected = base.astype(np.int16)
    if foreground is not None:
        corrected += foreground.astype(np.int16) - 128
    if background is not None:
        corrected += background.astype(np.int16) - 128
    return np.clip(corrected, 0, 255).astype(np.uint8)


def _row(
    arm: str,
    setting: str,
    *,
    B: int,
    F: int,
    M: int,
    R: int,
    H: int,
    source: np.ndarray,
    delivered: np.ndarray,
    mask: np.ndarray,
    encode_s: float,
    decode_s: float,
    render_s: float,
    plate_s: float,
    source_row: dict,
    fg_qp: int | None = None,
    bg_qp: int | None = None,
) -> dict:
    total = B + F + M + R + H
    scores = _scores(source, delivered, mask)
    weighted = scores["weighted"]
    return {
        "arm": arm,
        "setting": setting,
        "B": B,
        "F": F,
        "M": M,
        "R": R,
        "H": H,
        "total_bytes": total,
        "scores": scores,
        "claimable": total <= int(source_row["total_bytes"]) and weighted is not None and weighted >= float(source_row["psnr_weighted"]),
        "fg_qp": fg_qp,
        "bg_qp": bg_qp,
        "encode_seconds": round(encode_s, 3),
        "decode_seconds": round(decode_s, 3),
        "render_seconds": round(render_s, 3),
        "sender_seconds": round(plate_s + encode_s, 3),
        "client_seconds": round(decode_s + render_s, 3),
        "source_encode_seconds": source_row["encode_seconds"],
        "source_decode_seconds": source_row["decode_seconds"],
    }


def run(clip_id: str) -> dict:
    relative, source_json, representation, expected = CLIPS[clip_id]
    row_file = OUT_DIR / source_json
    background_doc = json.loads(row_file.read_text())
    source_row = next(r for r in background_doc["rows"] if r["representation"] == "source" and r["qp"] == 46)
    prior_bg = next(r for r in background_doc["rows"] if r["representation"] == representation and r["qp"] == 46)
    directory = ROOT / relative
    source, mask = load_sequence(directory / "window_48", directory / "masks_48.npz", 48)
    cache = np.load(OUT_DIR / "cache" / f"{clip_id}-n48.npz", mmap_mode="r")
    plate_s = float(cache["build_seconds"])
    print(f"{clip_id}: load {source.shape}, mask fraction {mask.mean():.6f}; re-encode fixed background", flush=True)
    if representation == "cleaned_video":
        payload, decoded, bg_enc, bg_dec, bg_path, bg_version = _timed_vvc(cache["cleaned"], 46)
        side_bytes = 10
        bg_render = 0.0
    else:
        plate = cache["plate"]
        homographies = tuple(np.asarray(h, dtype=np.float64) for h in cache["homographies"])
        payload, decoded_plate, bg_path, bg_version, bg_enc, bg_dec = _intra_still(plate, 46)
        side = pack_panorama_side_data(
            homographies,
            plate_shape=(int(plate.shape[0]), int(plate.shape[1])),
            frame_shape=(int(source.shape[1]), int(source.shape[2])),
            fps=25.0,
        )
        side_bytes = len(side)
        start = time.perf_counter()
        decoded = _render_panorama(decoded_plate, side, 48, (int(source.shape[1]), int(source.shape[2])))
        bg_render = time.perf_counter() - start
    bg_bytes = len(payload) + side_bytes
    print(f"{clip_id}: regenerated {bg_bytes} B, expected {expected} B", flush=True)
    if bg_bytes != expected or bg_bytes != int(prior_bg["total_bytes"]):
        raise RuntimeError(f"{clip_id}: background size changed ({bg_bytes} vs {expected}); no composite")
    if bg_path != prior_bg["tool_path"]:
        raise RuntimeError(f"{clip_id}: background encoder changed ({bg_path} vs {prior_bg['tool_path']}); no composite")

    started = time.perf_counter()
    boxes = _bboxes(mask)
    alpha_wire, alpha = _alpha_wire(mask[0], boxes[0])
    common_prep_s = time.perf_counter() - started
    y1, y2, x1, x2 = boxes[0]
    sidecar = IntraCodecSidecar("av1", qp=42)
    crop_input = np.ascontiguousarray(source[0, y1:y2, x1:x2, ::-1])
    crop_h, crop_w = crop_input.shape[:2]
    coded_h = max(64, crop_h + (-crop_h % 8))
    coded_w = max(64, crop_w + (-crop_w % 8))
    padded = np.zeros((coded_h, coded_w, 3), dtype=np.uint8)
    padded[:crop_h, :crop_w] = crop_input
    start = time.perf_counter()
    crop_wire = sidecar.encode(padded)
    crop_enc = time.perf_counter() - start
    start = time.perf_counter()
    crop_bgr = sidecar.decode(crop_wire)[:crop_h, :crop_w]
    crop_dec = time.perf_counter() - start
    crop_path, crop_version = sidecar.probe_encoder()
    bbox_wire = _box_payload(boxes)
    pose_started = time.perf_counter()
    poses, pose_info = _extract_keypoints(_rgb_to_bgr(source), boxes)
    pose_wire, decoded_poses = _pose_wire(poses)
    pose_encode_s = time.perf_counter() - pose_started
    meta = {
        "clip_id": clip_id,
        "source": str(directory / "window_48"),
        "mask": str(directory / "masks_48.npz"),
        "background_cache": str(OUT_DIR / "cache" / f"{clip_id}-n48.npz"),
        "background_row": str(row_file),
        "foreground_fraction": float(mask.mean()),
        "source_anchor": {k: source_row[k] for k in ("total_bytes", "psnr_overall", "psnr_fg", "psnr_bg", "psnr_weighted", "encode_seconds", "decode_seconds", "tool_path", "tool_version")},
        "background": {"representation": representation, "payload_bytes": len(payload), "side_bytes": side_bytes, "total_bytes": bg_bytes, "encoder_path": bg_path, "encoder_version": bg_version, "encode_seconds": bg_enc, "decode_seconds": bg_dec, "render_seconds": bg_render, "offline_plate_seconds": plate_s},
        "appearance": {"codec": "av1", "qp": 42, "payload_bytes": len(crop_wire), "encoder_path": crop_path, "encoder_version": crop_version, "encode_seconds": crop_enc, "decode_seconds": crop_dec, "alpha_bytes": len(alpha_wire), "alpha_coding": "uint32 height,width + zlib9 packed bits"},
        "pose": pose_info,
        "rows": [],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{clip_id}.json"
    def save() -> None:
        out_path.write_text(json.dumps(meta, indent=2) + "\n")
    save()
    bbox_matrices = [_bbox_affine(boxes[0], box) for box in boxes]
    pose_matrices = [_keypoint_affine(decoded_poses[0], decoded_poses[i], boxes[0], boxes[i])[0] for i in range(48)]
    for arm, motion_wire, matrices in (
        ("bbox", bbox_wire, bbox_matrices),
        ("coco17", bbox_wire + pose_wire, pose_matrices),
    ):
        print(f"{clip_id}: {arm} paste", flush=True)
        base, paste_s = _warp_and_paste(decoded, crop_bgr, alpha, boxes[0], matrices)
        base_enc = bg_enc + crop_enc + common_prep_s + (pose_encode_s if arm == "coco17" else 0.0)
        base_dec = bg_dec + crop_dec
        base_render = bg_render + paste_s
        B, F, M, H = bg_bytes, len(crop_wire), len(motion_wire), len(alpha_wire)
        clip_fraction = residual_clip_fraction(source.astype(np.int16) - base.astype(np.int16), mask)
        meta.setdefault("clip_fraction", {})[arm] = clip_fraction
        nores = _row(arm, "neither", B=B,F=F,M=M,R=0,H=H,source=source,delivered=base,mask=mask,encode_s=base_enc,decode_s=base_dec,render_s=base_render,plate_s=plate_s,source_row=source_row)
        meta["rows"].append(nores)
        save()
        print(f"{clip_id}: {arm} off {nores['total_bytes']} B weighted={nores['scores']['weighted']:.3f} clip={clip_fraction:.4f}", flush=True)
        if nores["total_bytes"] > int(source_row["total_bytes"]):
            meta.setdefault("stopped_arms", {})[arm] = "one crop with both residuals off exceeds anchor"
            save()
            continue
        qps = (54, 62) if clip_fraction > 0.05 else (46, 54, 62)
        fg_signal = _residual_signal(source, base, mask)
        bg_signal = _residual_signal(source, base, ~mask)
        payloads: dict[str, dict[int, tuple[int, np.ndarray, float, float, str, str]]] = {"fg": {}, "bg": {}}
        for region, signal in (("fg", fg_signal), ("bg", bg_signal)):
            for qp in qps:
                print(f"{clip_id}: {arm} {region} residual QP {qp}", flush=True)
                data, pixels, enc_s, dec_s, path, version = _timed_vvc(signal, qp)
                if not data:
                    raise RuntimeError("empty residual bitstream")
                payloads[region][qp] = (len(data), pixels, enc_s, dec_s, path, version)
                meta.setdefault("residual_encodes", []).append({"arm": arm,"region": region,"qp": qp,"bytes": len(data),"encode_seconds": enc_s,"decode_seconds": dec_s,"path": path,"version": version})
                save()
        # Record each single residual and the ordered coarsening path.  Every
        # paired row has two distinct payloads; there is no averaged bitstream.
        for region in ("fg", "bg"):
            for qp in qps:
                nbytes, pixels, enc_s, dec_s, _path, _version = payloads[region][qp]
                delivered = _correct(base, pixels if region == "fg" else None, pixels if region == "bg" else None)
                row = _row(arm, f"{region}_only_qp{qp}", B=B,F=F,M=M,R=nbytes,H=H,source=source,delivered=delivered,mask=mask,encode_s=base_enc+enc_s,decode_s=base_dec+dec_s,render_s=base_render,plate_s=plate_s,source_row=source_row,fg_qp=qp if region=="fg" else None,bg_qp=qp if region=="bg" else None)
                meta["rows"].append(row)
                save()
        for bg_qp in qps:
            for fg_qp in qps:
                f_bytes, f_pixels, f_enc, f_dec, _, _ = payloads["fg"][fg_qp]
                b_bytes, b_pixels, b_enc, b_dec, _, _ = payloads["bg"][bg_qp]
                delivered = _correct(base, f_pixels, b_pixels)
                row = _row(arm, f"both_fg{fg_qp}_bg{bg_qp}", B=B,F=F,M=M,R=f_bytes+b_bytes,H=H,source=source,delivered=delivered,mask=mask,encode_s=base_enc+f_enc+b_enc,decode_s=base_dec+f_dec+b_dec,render_s=base_render,plate_s=plate_s,source_row=source_row,fg_qp=fg_qp,bg_qp=bg_qp)
                meta["rows"].append(row)
                save()
        del base, fg_signal, bg_signal, payloads
    meta["completed"] = True
    save()
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("clip", choices=tuple(CLIPS))
    args = parser.parse_args()
    run(args.clip)
