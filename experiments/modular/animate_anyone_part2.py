"""Wire-conditioned 16-frame diagnostic of the installed tennis Animate Anyone model."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sqlite3  # noqa: F401  # host C++ runtime before Torch
import sys
import time
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import cv2
import numpy as np

from experiments.modular.foreground_part2_common import load_fixed_alcaraz
from experiments.modular.measured_ladder import score_regions
from src.components.background.sidecar import IntraCodecSidecar
from src.components.codec.tools import resolve_ffmpeg
from src.contracts.keypoints import COCO_17_JOINTS, OPENPOSE_18_JOINTS

DEFAULT_OUT = Path("/home/itec/emanuele/pointstream-data/outputs/modular/foreground-part2/animate-anyone")
MODEL_ROOT = Path.home() / "Models/AnimateAnyone/profiles/finetuned_tennis"
CANVAS = 256
STEPS = 20
GUIDANCE = 7.5
SEED = 20260924


def coco17_to_openpose18(pose: np.ndarray) -> np.ndarray:
    """Map decoded COCO-17 joints by name; synthesize neck from both shoulders.

    Nonpositive or nonfinite confidence means absent. Present joints with
    nonfinite coordinates are malformed. An absent joint remains all zero.
    """
    values = np.asarray(pose)
    if values.shape != (17, 3):
        raise ValueError(f"expected decoded COCO-17 (17,3), got {values.shape}")
    values = values.astype(np.float32, copy=False)
    output = np.zeros((18, 3), dtype=np.float32)
    source = {name: index for index, name in enumerate(COCO_17_JOINTS)}
    for index, name in enumerate(OPENPOSE_18_JOINTS):
        if name == "neck":
            continue
        joint = values[source[name]]
        confidence = float(joint[2])
        if not np.isfinite(confidence) or confidence <= 0:
            continue
        if not np.isfinite(joint[:2]).all():
            raise ValueError(f"present {name} has nonfinite position")
        output[index] = joint
    left = output[OPENPOSE_18_JOINTS.index("left_shoulder")]
    right = output[OPENPOSE_18_JOINTS.index("right_shoulder")]
    if left[2] > 0 and right[2] > 0:
        output[OPENPOSE_18_JOINTS.index("neck")] = (
            (left[0] + right[0]) * 0.5,
            (left[1] + right[1]) * 0.5,
            min(float(left[2]), float(right[2])),
        )
    return output


def _object_helpers() -> Any:
    """Import CPU runner helpers and restore GPU visibility it suppresses."""
    previous = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        from experiments.modular import object_foreground_campaign
    finally:
        if previous is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = previous
    return object_foreground_campaign


def _byte_components(objects: list[dict[str, Any]], background_bytes: int) -> dict[str, int]:
    """Count every object payload once in B + F + M + R + H."""
    if background_bytes <= 0 or not objects:
        raise ValueError("positive background and at least one object required")
    parts = {
        "B": background_bytes,
        "F": sum(len(obj["crop_wire"]) for obj in objects),
        "M": sum(len(obj["bbox_wire"]) + len(obj["presence_wire"]) + len(obj["pose_wire"]) for obj in objects),
        "R": 0,
        "H": 1 + sum(len(obj["alpha_wire"]) for obj in objects),
    }
    if min(parts["F"], parts["M"], parts["H"]) <= 0:
        raise ValueError("empty appearance, motion or side payload")
    parts["total_bytes"] = sum(parts.values())
    return parts


def _encode_objects(
    source_rgb: np.ndarray,
    tracks: list[np.ndarray],
    sidecar: Any,
    n_frames: int,
) -> tuple[list[dict[str, Any]], float, float]:
    """Encode separate crop, alpha, bbox, presence and COCO-17 pose wires."""
    helpers = _object_helpers()
    from experiments.modular.appearance_motion_probe import _box_payload, _extract_keypoints
    from experiments.modular.foreground_campaign import _alpha_wire, _pose_wire

    objects: list[dict[str, Any]] = []
    encode_s = decode_s = 0.0
    for index, full_track in enumerate(tracks):
        track = full_track[:n_frames]
        if not np.any(track):
            raise RuntimeError(f"object {index} is absent from this screen")
        start = time.perf_counter()
        boxes, presence, first = helpers._boxes(track)
        alpha_wire, alpha = _alpha_wire(track[first], boxes[first])
        bbox_wire = _box_payload(boxes)
        presence_wire = np.packbits(presence.astype(np.uint8), bitorder="little").tobytes()
        y1, y2, x1, x2 = boxes[first]
        crop_input = np.ascontiguousarray(source_rgb[first, y1:y2, x1:x2, ::-1])
        encode_s += time.perf_counter() - start
        coded_h = max(64, (crop_input.shape[0] + 7) // 8 * 8)
        coded_w = max(64, (crop_input.shape[1] + 7) // 8 * 8)
        padded = np.zeros((coded_h, coded_w, 3), dtype=np.uint8)
        padded[:crop_input.shape[0], :crop_input.shape[1]] = crop_input
        start = time.perf_counter()
        crop_wire = sidecar.encode(padded)
        encode_s += time.perf_counter() - start
        if not crop_wire:
            raise RuntimeError(f"object {index}: empty AV1 QP42 reference")
        start = time.perf_counter()
        crop_bgr = sidecar.decode(crop_wire)[:crop_input.shape[0], :crop_input.shape[1]]
        decode_s += time.perf_counter() - start
        if crop_bgr.shape != crop_input.shape:
            raise RuntimeError(f"object {index}: decoded crop shape changed")
        start = time.perf_counter()
        visible = np.flatnonzero(presence)
        pose_list, pose_info = _extract_keypoints(
            np.ascontiguousarray(source_rgb[visible, :, :, ::-1]),
            [boxes[int(i)] for i in visible],
        )
        poses: list[np.ndarray | None] = [None] * n_frames
        for frame_index, pose in zip(visible, pose_list, strict=True):
            poses[int(frame_index)] = pose
        pose_wire, decoded_poses = _pose_wire(poses)
        encode_s += time.perf_counter() - start
        if not pose_wire:
            raise RuntimeError(f"object {index}: empty COCO-17 wire")
        pose18 = np.stack([
            coco17_to_openpose18(np.zeros((17, 3), dtype=np.float16) if pose is None else pose)
            for pose in decoded_poses
        ])
        objects.append({
            "index": index, "first": first, "boxes": boxes, "presence": presence,
            "alpha": alpha, "crop_bgr": crop_bgr, "decoded_poses": decoded_poses,
            "pose18": pose18, "pose_info": pose_info, "crop_wire": crop_wire,
            "alpha_wire": alpha_wire, "bbox_wire": bbox_wire,
            "presence_wire": presence_wire, "pose_wire": pose_wire,
        })
    return objects, encode_s, decode_s


def _generated_patch(generated_bgr: np.ndarray, obj: dict[str, Any], t: int) -> tuple[np.ndarray, np.ndarray]:
    """Crop the reference letterbox and place it through charged alpha."""
    if generated_bgr.shape != (CANVAS, CANVAS, 3):
        raise RuntimeError(f"unexpected model canvas {generated_bgr.shape}")
    ref_h, ref_w = obj["crop_bgr"].shape[:2]
    scale = min(CANVAS / ref_w, CANVAS / ref_h)
    inset_w, inset_h = max(1, round(ref_w * scale)), max(1, round(ref_h * scale))
    off_x, off_y = (CANVAS - inset_w) // 2, (CANVAS - inset_h) // 2
    center = generated_bgr[off_y:off_y + inset_h, off_x:off_x + inset_w]
    y1, y2, x1, x2 = obj["boxes"][t]
    size = (x2 - x1, y2 - y1)
    pixels = cv2.resize(center[:, :, ::-1], size, interpolation=cv2.INTER_LINEAR)
    alpha = cv2.resize(obj["alpha"].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    return pixels, alpha


def _compose_generated(background_rgb: np.ndarray, objects: list[dict[str, Any]], generations: list[np.ndarray], t: int) -> np.ndarray:
    """Decoder-only composite: no source or target mask argument."""
    result = np.array(background_rgb, copy=True)
    for obj, generated in zip(objects, generations, strict=True):
        if not obj["presence"][t]:
            continue
        pixels, alpha = _generated_patch(generated[t], obj, t)
        y1, y2, x1, x2 = obj["boxes"][t]
        roi = result[y1:y2, x1:x2]
        roi[alpha] = pixels[alpha]
    return result


def _score_frames(source: np.ndarray, mask: np.ndarray, frames: list[np.ndarray]) -> dict[str, float]:
    scores = [
        score_regions(source[t:t + 1], frame[None], mask[t:t + 1], fg_weight=0.7, bg_weight=0.3)
        for t, frame in enumerate(frames)
    ]
    overall, fg, bg = (float(np.mean([float(row[c]) for row in scores])) for c in range(3))
    return {"overall": overall, "foreground": fg, "background": bg, "weighted": 0.7 * fg + 0.3 * bg}


def _clip_fraction(source: np.ndarray, mask: np.ndarray, frames: list[np.ndarray]) -> float:
    clipped = count = 0
    for t, frame in enumerate(frames):
        residual = source[t].astype(np.int16) - frame.astype(np.int16)
        clipped += int((np.any((residual < -128) | (residual > 127), axis=-1) & mask[t]).sum())
        count += int(mask[t].sum())
    if count == 0:
        raise ValueError("empty foreground screen")
    return clipped / count


def _package_versions() -> dict[str, str]:
    result = {}
    for name in ("torch", "diffusers", "transformers"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "missing"
    return result


def run_animate_anyone_screen(out_dir: Path, *, n_frames: int = 16) -> dict[str, object]:
    """Score two decoded-wire references and poses against articulated paste.

    The 16-frame screen has no source-anchor claim. Two four-frame controls
    shift decoded pose and swap decoded references; neither adds wire bytes.
    Runtime or checkpoint failure is written to the ledger and re-raised.
    """
    if not 1 <= n_frames <= 48:
        raise ValueError("n_frames must be 1..48")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = out_dir / "alcaraz000-animate-anyone-screen.json"
    ledger: dict[str, Any] = {
        "clip_id": "alcaraz000", "status": "started", "n_frames": n_frames,
        "scope": "development screen; no 48-frame claim", "model_variant": "finetuned_tennis",
        "model_root": str(MODEL_ROOT), "seed": SEED, "canvas": [CANVAS, CANVAS],
        "steps": STEPS, "guidance": GUIDANCE, "runtime_versions": _package_versions(),
    }

    def save() -> None:
        ledger_path.write_text(json.dumps(ledger, indent=2, allow_nan=False) + "\n")

    save()
    try:
        fixed = load_fixed_alcaraz(out_dir.parent)
        source, mask, background = fixed.source_rgb[:n_frames], fixed.mask[:n_frames], fixed.background_rgb[:n_frames]
        if source.shape != background.shape or mask.shape != source.shape[:3]:
            raise RuntimeError("source, mask and background shape mismatch")
        bg_bytes = int(fixed.background["total_bytes"])
        if bg_bytes != 24648:
            raise RuntimeError(f"background is {bg_bytes} B; expected 24648 B")
        helpers = _object_helpers()
        tracks = helpers.split_object_tracks(fixed.mask)
        if len(tracks) != 2:
            raise RuntimeError(f"expected two independent Alcaraz objects, found {len(tracks)}")
        sidecar = IntraCodecSidecar("av1", qp=42)
        crop_path, crop_version = sidecar.probe_encoder()
        objects, obj_encode_s, obj_decode_s = _encode_objects(source, tracks, sidecar, n_frames)
        ledger["wire"] = _byte_components(objects, bg_bytes)
        ledger["objects"] = [{
            "index": obj["index"], "first_frame": obj["first"],
            "crop_bytes": len(obj["crop_wire"]), "alpha_bytes": len(obj["alpha_wire"]),
            "bbox_bytes": len(obj["bbox_wire"]), "presence_bytes": len(obj["presence_wire"]),
            "pose_bytes": len(obj["pose_wire"]), "pose_info": obj["pose_info"],
        } for obj in objects]
        native_decoder = resolve_ffmpeg()
        ledger["encoder"] = {
            "crop_path": crop_path, "crop_version": crop_version,
            "background_path": fixed.background.get("tool_path"),
            "background_version": fixed.background.get("tool_version"),
            "decoder_path": native_decoder.path,
            "decoder_version": native_decoder.version,
        }
        ledger["source_anchor_48_frames"] = fixed.anchor
        save()

        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable after pose extraction; refusing a CPU diffusion fallback")
        from src.components.generation.animate_anyone import resolve_checkpoint
        from src.components.generation.animate_anyone_runtime import generate_sequence

        checkpoint = resolve_checkpoint(MODEL_ROOT)
        ledger["checkpoint"] = {"path": str(checkpoint), "files": {
            name: (checkpoint / name).stat().st_size for name in (
                "denoising_unet.pth", "reference_unet.pth", "pose_guider.pth", "motion_module.pth"
            )}}
        generations: list[np.ndarray] = []
        model_times: list[float] = []
        for obj in objects:
            start = time.perf_counter()
            generated = generate_sequence(
                reference_image_bgr=obj["crop_bgr"], dense_pose_sequence=obj["pose18"],
                seed=SEED + obj["index"], device="cuda", steps=STEPS, cfg=GUIDANCE,
                width=CANVAS, height=CANVAS, model_dir=str(checkpoint),
                model_variant="finetuned_tennis",
            )
            if generated.shape != (n_frames, CANVAS, CANVAS, 3):
                raise RuntimeError(f"object {obj['index']}: model returned {generated.shape}")
            model_times.append(time.perf_counter() - start)
            generations.append(generated)
            np.save(out_dir / f"object-{obj['index']}-generated-bgr.npy", generated)
        model_s = sum(model_times)
        start = time.perf_counter()
        generated_frames = [_compose_generated(background[t], objects, generations, t) for t in range(n_frames)]
        composite_s = time.perf_counter() - start

        start = time.perf_counter()
        paste_frames = []
        for t in range(n_frames):
            frame = np.array(background[t], copy=True)
            for obj in objects:
                if not obj["presence"][t]:
                    continue
                first_pose, target_pose = obj["decoded_poses"][obj["first"]], obj["decoded_poses"][t]
                if first_pose is None or target_pose is None:
                    matrix = helpers._bbox_affine(obj["boxes"][obj["first"]], obj["boxes"][t])
                    pixels, cover = helpers._affine_warp(
                        obj["crop_bgr"], obj["alpha"], matrix, obj["boxes"][obj["first"]], frame.shape[:2]
                    )
                else:
                    pixels, cover = helpers.warp_articulated(
                        obj["crop_bgr"], obj["alpha"], first_pose, target_pose,
                        obj["boxes"][obj["first"]], frame.shape[:2], target_box=obj["boxes"][t],
                    )
                frame[cover] = pixels[cover][:, ::-1]
            paste_frames.append(frame)
        paste_s = time.perf_counter() - start
        generated_scores = _score_frames(source, mask, generated_frames)
        paste_scores = _score_frames(source, mask, paste_frames)
        ledger["screen"] = {
            "animate_anyone": {"scores": generated_scores, "clip_fraction": _clip_fraction(source, mask, generated_frames)},
            "articulated_paste": {"scores": paste_scores, "clip_fraction": _clip_fraction(source, mask, paste_frames)},
            "foreground_gain_db": generated_scores["foreground"] - paste_scores["foreground"],
            "same_frames": n_frames, "claimable": False,
        }
        ledger["timing"] = {
            "background_encode_seconds": fixed.background.get("reencode_seconds"),
            "background_decode_seconds": fixed.background.get("redecode_seconds"),
            "background_render_seconds": fixed.background.get("rerender_seconds"),
            "offline_plate_seconds": fixed.plate_seconds,
            "foreground_encode_seconds": obj_encode_s,
            "foreground_decode_seconds": obj_decode_s,
            "animate_anyone_render_seconds": model_s + composite_s,
            "model_cold_first_object_seconds": model_times[0],
            "model_warm_second_object_seconds": model_times[1],
            "articulated_render_seconds": paste_s,
            "animate_anyone_client_seconds": float(fixed.background.get("redecode_seconds", 0)) + obj_decode_s + model_s + composite_s + float(fixed.background.get("rerender_seconds", 0)),
        }
        save()

        if n_frames >= 2:
            control_n = min(4, n_frames)
            controls: dict[str, Any] = {
                "correct_condition_prefix": {"frames": control_n, "scores": _score_frames(source[:control_n], mask[:control_n], generated_frames[:control_n])}
            }
            for condition in ("shifted_pose", "swapped_reference"):
                outputs = []
                for index, obj in enumerate(objects):
                    pose = obj["pose18"][:control_n]
                    reference = obj["crop_bgr"]
                    if condition == "shifted_pose":
                        pose = np.roll(obj["pose18"], max(1, n_frames // 2), axis=0)[:control_n]
                    else:
                        reference = objects[1 - index]["crop_bgr"]
                    outputs.append(generate_sequence(
                        reference_image_bgr=reference, dense_pose_sequence=pose,
                        seed=SEED + obj["index"], device="cuda", steps=STEPS, cfg=GUIDANCE,
                        width=CANVAS, height=CANVAS, model_dir=str(checkpoint),
                        model_variant="finetuned_tennis",
                    ))
                frames = [_compose_generated(background[t], objects, outputs, t) for t in range(control_n)]
                controls[condition] = {"frames": control_n, "scores": _score_frames(source[:control_n], mask[:control_n], frames)}
            ledger["controls"] = controls
        ledger["status"] = "completed"
        save()
        return ledger
    except Exception as exc:
        ledger["status"] = "failed"
        ledger["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        save()
        raise RuntimeError(f"Animate Anyone screen failed; see {ledger_path}: {exc}") from exc


def score_saved_oracle_alpha(out_dir: Path) -> dict[str, object]:
    """Isolate generated pixel quality using forbidden target silhouettes.

    This diagnostic never enters a charged bitstream or a claimable row. It
    reuses saved generated frames and the exact two-object screen background.
    """
    out_dir = Path(out_dir)
    ledger_path = out_dir / "alcaraz000-animate-anyone-screen.json"
    ledger = json.loads(ledger_path.read_text())
    if ledger.get("status") != "completed":
        raise RuntimeError("Animate Anyone screen is not complete")
    n_frames = int(ledger["n_frames"])
    fixed = load_fixed_alcaraz(out_dir.parent)
    helpers = _object_helpers()
    tracks = helpers.split_object_tracks(fixed.mask)
    if len(tracks) != 2:
        raise RuntimeError("oracle alpha diagnostic expected two object tracks")
    generations = [
        np.load(out_dir / f"object-{index}-generated-bgr.npy", allow_pickle=False)
        for index in range(2)
    ]
    objects = []
    for track in tracks:
        boxes, presence, first = helpers._boxes(track[:n_frames])
        y1, y2, x1, x2 = boxes[first]
        objects.append({
            "boxes": boxes, "presence": presence,
            "crop_bgr": np.empty((y2 - y1, x2 - x1, 3), dtype=np.uint8),
            "alpha": track[first, y1:y2, x1:x2],
        })
    frames = []
    for t in range(n_frames):
        frame = np.array(fixed.background_rgb[t], copy=True)
        for obj, track, generated in zip(objects, tracks, generations, strict=True):
            if not obj["presence"][t]:
                continue
            pixels, _ = _generated_patch(generated[t], obj, t)
            y1, y2, x1, x2 = obj["boxes"][t]
            roi = frame[y1:y2, x1:x2]
            target_roi = track[t, y1:y2, x1:x2]
            roi[target_roi] = pixels[target_roi]
        frames.append(frame)
    diagnostic = {
        "target_informed": True,
        "claimable": False,
        "description": "saved generated pixels pasted through source object silhouettes",
        "scores": _score_frames(fixed.source_rgb[:n_frames], fixed.mask[:n_frames], frames),
        "clip_fraction": _clip_fraction(fixed.source_rgb[:n_frames], fixed.mask[:n_frames], frames),
    }
    ledger["oracle_alpha_diagnostic"] = diagnostic
    ledger_path.write_text(json.dumps(ledger, indent=2, allow_nan=False) + "\n")
    return diagnostic


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--oracle-alpha-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.oracle_alpha_only:
        print(json.dumps(score_saved_oracle_alpha(arguments.out_dir), indent=2))
    else:
        run_animate_anyone_screen(arguments.out_dir, n_frames=arguments.frames)
