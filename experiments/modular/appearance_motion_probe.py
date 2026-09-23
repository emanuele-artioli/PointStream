"""Measure one appearance plus motion against the per-frame crop control.

The control sends a new AV1 intra crop whenever the measured C1 policy fires.
The two alternatives send one AV1 intra crop from frame 0 and then either:

* a four-int16 placement box per later frame; or
* a COCO-17 pose (x, y, confidence) in float16 per later frame.

The reconstruction is deliberately classical: affine-warp the first crop and
paste it through the source mask. It is not a generative result. This makes
the wire/quality tradeoff of the motion signal measurable before a generator
is allowed to claim a win.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import cv2
import numpy as np

from experiments.modular.image_codec_probe import FRAMES, MASKS
from experiments.modular.measured_ladder import (
    CROP_CODEC,
    CROP_QP,
    PLATE_CODEC,
    PLATE_QP,
    _intra_roundtrip,
    _plate_roundtrip,
    _pack_metadata,
    _rgb_to_bgr,
    load_sequence,
    measure_rungs,
    score_regions,
)
from src.components.detection.geometry import Box
from src.components.detection.types import Detection
from src.components.motion.keypoints import KeypointMotionEncoder
from src.components.pose.yolo import YoloPoseEstimator
from src.contracts.keypoints import COCO_17

OUT = Path(
    "/home/itec/emanuele/pointstream-data/outputs/modular/"
    "appearance-motion/federer007.json"
)
N_FRAMES = 48


def _bboxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return padded ``(y1, y2, x1, x2)`` boxes from the transmitted masks."""
    boxes: list[tuple[int, int, int, int]] = []
    for item in mask:
        ys, xs = np.where(item)
        if len(ys) == 0:
            raise ValueError("appearance-motion probe found an empty foreground mask")
        y1 = max(0, int(ys.min()) - 8)
        y2 = min(mask.shape[1], int(ys.max()) + 9)
        x1 = max(0, int(xs.min()) - 8)
        x2 = min(mask.shape[2], int(xs.max()) + 9)
        y1 -= y1 % 2
        x1 -= x1 % 2
        y2 -= (y2 - y1) % 2
        x2 -= (x2 - x1) % 2
        boxes.append((y1, y2, x1, x2))
    return boxes


def _box_payload(boxes: list[tuple[int, int, int, int]]) -> bytes:
    values = np.asarray(
        [[x1, y1, x2, y2] for y1, y2, x1, x2 in boxes],
        dtype="<i2",
    )
    return np.ascontiguousarray(values).tobytes()


def _extract_keypoints(
    frames_bgr: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
) -> tuple[list[np.ndarray | None], dict[str, Any]]:
    """Run the real pose backend and retain missing-frame accounting."""
    estimator = YoloPoseEstimator(model_name="yolo26n-pose.pt")
    source_indices = [estimator.emits.index_of[name] for name in COCO_17.joints]
    poses: list[np.ndarray | None] = []
    missing = 0
    visible_joints = 0
    for frame, (y1, y2, x1, x2) in zip(frames_bgr, boxes, strict=True):
        detection = Detection(
            class_name="person",
            bbox=Box(float(x1), float(y1), float(x2), float(y2)),
        )
        pose = estimator.estimate(frame, detection)
        if pose is None:
            poses.append(None)
            missing += 1
            continue
        values = np.asarray(pose.values[source_indices], dtype=np.float32).copy()
        present = np.asarray(pose.present[source_indices], dtype=bool)
        values[~present] = 0.0
        visible_joints += int(present.sum())
        poses.append(values)
    return poses, {
        "model": estimator.model_name,
        "schema": COCO_17.name,
        "frames": len(poses),
        "missing_frames": missing,
        "mean_visible_joints": visible_joints / max(1, len(poses) - missing),
    }


def _bbox_affine(
    source: tuple[int, int, int, int],
    target: tuple[int, int, int, int],
) -> np.ndarray:
    sy1, sy2, sx1, sx2 = source
    ty1, ty2, tx1, tx2 = target
    sx = (tx2 - tx1) / max(1.0, float(sx2 - sx1))
    sy = (ty2 - ty1) / max(1.0, float(sy2 - sy1))
    return np.asarray(
        [[sx, 0.0, tx1 - sx * sx1], [0.0, sy, ty1 - sy * sy1]],
        dtype=np.float32,
    )


def _keypoint_affine(
    source: np.ndarray | None,
    target: np.ndarray | None,
    source_box: tuple[int, int, int, int],
    target_box: tuple[int, int, int, int],
) -> tuple[np.ndarray, str]:
    if source is not None and target is not None:
        visible = (source[:, 2] > 0) & (target[:, 2] > 0)
        if int(visible.sum()) >= 2:
            matrix, _inliers = cv2.estimateAffinePartial2D(
                source[visible, :2],
                target[visible, :2],
                method=cv2.RANSAC,
                ransacReprojThreshold=8.0,
                maxIters=1000,
                confidence=0.99,
            )
            if matrix is not None:
                return np.asarray(matrix, dtype=np.float32), "keypoints"
    return _bbox_affine(source_box, target_box), "bbox_fallback"


def _warp_crop(
    crop_bgr: np.ndarray,
    matrix_full: np.ndarray,
    source_box: tuple[int, int, int, int],
    target_mask: np.ndarray,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Warp a crop whose origin is in full-frame coordinates."""
    height, width = output_shape
    y1, _y2, x1, _x2 = source_box
    local = np.asarray(matrix_full, dtype=np.float32).copy()
    local[:, 2] += local[:, :2] @ np.asarray([x1, y1], dtype=np.float32)
    warped = cv2.warpAffine(
        crop_bgr,
        local,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    valid = cv2.warpAffine(
        np.ones(crop_bgr.shape[:2], dtype=np.uint8),
        local,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0,),
    ).astype(bool)
    result = np.zeros((height, width, 3), dtype=np.uint8)
    result[target_mask & valid] = warped[target_mask & valid]
    return result


def _row(
    name: str,
    *,
    reconstruction_bgr: np.ndarray,
    frames_rgb: np.ndarray,
    mask: np.ndarray,
    bytes_background: int,
    bytes_appearance: int,
    bytes_metadata: int,
    bytes_motion: int,
    details: dict[str, Any],
) -> dict[str, Any]:
    overall, fg, bg, weighted = score_regions(
        frames_rgb,
        reconstruction_bgr[..., ::-1],
        mask,
        fg_weight=0.70,
        bg_weight=0.30,
    )
    return {
        "arm": name,
        "bytes_background": bytes_background,
        "bytes_appearance": bytes_appearance,
        "bytes_metadata": bytes_metadata,
        "bytes_motion": bytes_motion,
        "total_bytes": (
            bytes_background
            + bytes_appearance
            + bytes_metadata
            + bytes_motion
        ),
        "psnr_overall": overall,
        "psnr_fg": fg,
        "psnr_bg": bg,
        "psnr_weighted": weighted,
        **details,
    }


def main() -> None:
    frames_rgb, mask = load_sequence(FRAMES, MASKS, N_FRAMES)
    frames_bgr = _rgb_to_bgr(frames_rgb)
    boxes = _bboxes(mask)

    # This is the existing per-frame control, now with the same weighted ledger.
    control = measure_rungs(frames_rgb, mask, include_residuals=False)[1]

    from src.components.background.plate import build_plate

    plate, _maps = build_plate(frames_bgr, masks=mask, register=False)
    plate_payload, plate_decoded = _plate_roundtrip(plate)
    y1, y2, x1, x2 = boxes[0]
    first_crop = frames_bgr[0, y1:y2, x1:x2]
    crop_payload, crop_decoded = _intra_roundtrip(first_crop, CROP_CODEC, CROP_QP)
    initial_metadata = _pack_metadata([(0, y1, x1, y2, x2)])

    keypoints, pose_details = _extract_keypoints(frames_bgr, boxes)
    keypoint_encoder = KeypointMotionEncoder(
        schema=COCO_17,
        values_per_joint=3,
        bytes_per_value=2,
    )
    keypoint_payloads: list[bytes] = []
    for pose in keypoints[1:]:
        values = np.zeros((len(COCO_17), 3), dtype=np.float32) if pose is None else pose
        _descriptor, payload = keypoint_encoder.encode(values)
        keypoint_payloads.append(payload)

    bbox_payload = _box_payload(boxes[1:])
    bbox_reconstruction: list[np.ndarray] = []
    keypoint_reconstruction: list[np.ndarray] = []
    keypoint_modes: dict[str, int] = {}
    frame_shape = (int(frames_rgb.shape[1]), int(frames_rgb.shape[2]))
    for index, target_box in enumerate(boxes):
        bbox_matrix = _bbox_affine(boxes[0], target_box)
        bbox_warp = _warp_crop(
            crop_decoded,
            bbox_matrix,
            boxes[0],
            mask[index],
            frame_shape,
        )
        bbox_frame = plate_decoded.copy()
        valid = np.any(bbox_warp != 0, axis=-1) & mask[index]
        bbox_frame[valid] = bbox_warp[valid]
        bbox_reconstruction.append(bbox_frame)

        matrix, mode = _keypoint_affine(
            keypoints[0],
            keypoints[index],
            boxes[0],
            target_box,
        )
        keypoint_modes[mode] = keypoint_modes.get(mode, 0) + 1
        keypoint_warp = _warp_crop(
            crop_decoded,
            matrix,
            boxes[0],
            mask[index],
            frame_shape,
        )
        keypoint_frame = plate_decoded.copy()
        valid = np.any(keypoint_warp != 0, axis=-1) & mask[index]
        keypoint_frame[valid] = keypoint_warp[valid]
        keypoint_reconstruction.append(keypoint_frame)

    rows = [
        _row(
            "per_frame_crop_control",
            reconstruction_bgr=control.reconstruction[..., ::-1],
            frames_rgb=frames_rgb,
            mask=mask,
            bytes_background=control.bytes_background,
            bytes_appearance=control.bytes_appearance,
            bytes_metadata=control.bytes_metadata,
            bytes_motion=0,
            details={"motion": "none", "pose_oks": None},
        ),
        _row(
            "single_crop_bbox_motion",
            reconstruction_bgr=np.stack(bbox_reconstruction),
            frames_rgb=frames_rgb,
            mask=mask,
            bytes_background=len(plate_payload),
            bytes_appearance=len(crop_payload),
            bytes_metadata=len(initial_metadata),
            bytes_motion=len(bbox_payload),
            details={
                "motion": "bbox",
                "motion_bytes_per_frame": 8,
                "pose_oks": None,
            },
        ),
        _row(
            "single_crop_keypoint_motion",
            reconstruction_bgr=np.stack(keypoint_reconstruction),
            frames_rgb=frames_rgb,
            mask=mask,
            bytes_background=len(plate_payload),
            bytes_appearance=len(crop_payload),
            bytes_metadata=len(initial_metadata),
            bytes_motion=sum(len(item) for item in keypoint_payloads),
            details={
                "motion": "coco-17-float16",
                "motion_bytes_per_frame": 102,
                "keypoint_modes": keypoint_modes,
                "pose_oks": None,
            },
        ),
    ]
    result = {
        "source": str(FRAMES),
        "n_frames": N_FRAMES,
        "metric": {
            "foreground_weight": 0.70,
            "background_weight": 0.30,
            "note": "weighted PSNR is the comparison quality arm",
        },
        "codec": {
            "plate": {"codec": PLATE_CODEC, "qp": PLATE_QP, "bytes": len(plate_payload)},
            "crop": {"codec": CROP_CODEC, "qp": CROP_QP, "bytes": len(crop_payload)},
        },
        "keypoint_wire": {
            "schema": COCO_17.name,
            "bytes_per_frame": len(keypoint_payloads[0]) if keypoint_payloads else 0,
            "motion_frames": len(keypoint_payloads),
            "bbox_bytes_per_frame": 8,
            "pose": pose_details,
        },
        "rows": rows,
        "generator_status": (
            "not measured here; this classical warp is a control. Existing "
            "generative claims were tied to the withdrawn constant table."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        print(row, flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
