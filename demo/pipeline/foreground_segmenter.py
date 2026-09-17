"""Foreground segmenter: extracts hand crops, appearance anchors, and masks out foreground."""

from __future__ import annotations

import cv2
import numpy as np

from demo.pipeline.hand_keypoints import SingleHand


def letterbox_crop(
    image: np.ndarray,
    bbox: list[int],
    target_size: int = 256,
) -> tuple[np.ndarray, dict[str, float]]:
    """Crops the bbox from image and letterboxes/resizes it to (target_size, target_size).

    Follows src.components.generation.pose.fit_to_canvas principles:
    scales by min(target/w, target/h) to preserve exact aspect ratio without clipping.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(x1 + 1, min(w, x2))
    y2 = max(y1 + 1, min(h, y2))

    crop = image[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    scale = float(target_size) / max(cw, ch)
    new_w = max(1, min(target_size, int(round(cw * scale))))
    new_h = max(1, min(target_size, int(round(ch * scale))))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

    meta = {
        "scale": scale,
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
        "new_w": float(new_w),
        "new_h": float(new_h),
        "orig_x1": float(x1),
        "orig_y1": float(y1),
        "orig_w": float(cw),
        "orig_h": float(ch),
    }
    return canvas, meta


def unletterbox_crop(
    canvas: np.ndarray,
    meta: dict[str, float],
    dest_w: int,
    dest_h: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Restores the crop to its original bounding box dimensions."""
    pad_x = int(meta["pad_x"])
    pad_y = int(meta["pad_y"])
    new_w = int(meta.get("new_w", round(meta["orig_w"] * meta["scale"])))
    new_h = int(meta.get("new_h", round(meta["orig_h"] * meta["scale"])))
    orig_w = int(meta["orig_w"])
    orig_h = int(meta["orig_h"])

    extracted = canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w]
    restored = cv2.resize(extracted, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    x1 = int(meta["orig_x1"])
    y1 = int(meta["orig_y1"])
    return restored, (x1, y1, x1 + orig_w, y1 + orig_h)


def create_box_feather_mask(crop_h: int, crop_w: int, margin_fraction: float = 0.08) -> np.ndarray:
    """Generates a smooth outer-margin alpha feathering mask [H, W, 1] in [0, 1].

    Preserves 100% of the interior (wrist, palm, finger anatomy) to satisfy MediaPipe's
    palm detector while smoothly ramping down the outer perimeter to 0 to eliminate square seams.
    """
    m_h = np.ones(crop_h, dtype=np.float32)
    m_w = np.ones(crop_w, dtype=np.float32)
    margin_h = max(1, min(crop_h // 3, int(crop_h * margin_fraction) or 1))
    margin_w = max(1, min(crop_w // 3, int(crop_w * margin_fraction) or 1))

    m_h[:margin_h] = np.linspace(0.0, 1.0, margin_h)
    m_h[-margin_h:] = np.linspace(1.0, 0.0, margin_h)
    m_w[:margin_w] = np.linspace(0.0, 1.0, margin_w)
    m_w[-margin_w:] = np.linspace(1.0, 0.0, margin_w)
    alpha = (m_h[:, None] * m_w[None, :])[:, :, None]
    return alpha


def mask_out_hands(
    frame: np.ndarray,
    hands: list[SingleHand],
    blur_kernel: int = 31,
) -> tuple[np.ndarray, np.ndarray]:
    """Blurs/inpaints hand regions from the frame.

    Note: In egocentric video, pixel-domain blurring saves negligible rate (<2%) while creating
    blur-halo artifacts around wrists. For production codecs, use encoder-native delta-QP maps
    (src/components/codec/roi.py) or clean downscaled background streams instead.
    """
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for hand in hands:
        x1, y1, x2, y2 = hand.bbox
        pad = int(max(x2 - x1, y2 - y1) * 0.1)
        bx1 = max(0, x1 - pad)
        by1 = max(0, y1 - pad)
        bx2 = min(w, x2 + pad)
        by2 = min(h, y2 + pad)
        cv2.rectangle(mask, (bx1, by1), (bx2, by2), 255, -1)

    blurred_bg = cv2.blur(frame, (blur_kernel, blur_kernel))
    masked_frame = np.where(mask[:, :, None] > 0, blurred_bg, frame)
    return masked_frame, mask

