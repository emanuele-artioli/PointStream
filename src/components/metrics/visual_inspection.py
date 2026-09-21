"""Visual inspection and comparison montage generator for generative video evaluation.

Produces side-by-side comparison strips, error heatmaps, skeleton/contour overlays,
and report-ready media (PNG, animated GIF, Markdown carousels) so human eyes can
verify visual quality beyond coincidental metric scores.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

#: Standard COCO-17 limb connections (pairs of joint indices).
COCO_17_LIMBS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # Facial keypoints
    (5, 6),  # Shoulders
    (5, 7),
    (7, 9),  # Left arm
    (6, 8),
    (8, 10),  # Right arm
    (5, 11),
    (6, 12),  # Torso
    (11, 12),  # Hips
    (11, 13),
    (13, 15),  # Left leg
    (12, 14),
    (14, 16),  # Right leg
)


def draw_skeleton(
    canvas: np.ndarray,
    keypoints: np.ndarray,
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    confs: np.ndarray | None = None,
    conf_thresh: float = 0.25,
    thickness: int = 2,
    radius: int = 4,
) -> np.ndarray:
    """Draw COCO-17 skeleton lines and joint circles on an RGB image."""
    img = canvas.copy()
    kpts = np.asarray(keypoints, dtype=np.float32)[:, :2]
    k = len(kpts)

    vis = np.ones(k, dtype=bool)
    if confs is not None:
        vis = np.asarray(confs) >= conf_thresh
    elif keypoints.shape[-1] >= 3:
        vis = np.asarray(keypoints)[:, 2] > 0

    # Draw limbs
    for idx1, idx2 in COCO_17_LIMBS:
        if idx1 < k and idx2 < k and vis[idx1] and vis[idx2]:
            pt1 = (int(round(kpts[idx1, 0])), int(round(kpts[idx1, 1])))
            pt2 = (int(round(kpts[idx2, 0])), int(round(kpts[idx2, 1])))
            cv2.line(img, pt1, pt2, color, thickness, lineType=cv2.LINE_AA)

    # Draw joints
    for i in range(k):
        if vis[i]:
            pt = (int(round(kpts[i, 0])), int(round(kpts[i, 1])))
            cv2.circle(img, pt, radius, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(img, pt, radius - 1, color, -1, lineType=cv2.LINE_AA)

    return img


def draw_mask_contour(
    canvas: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """Draw the boundary contour of a binary mask onto an RGB canvas."""
    img = canvas.copy()
    m_u8 = (np.asarray(mask, dtype=bool).astype(np.uint8)) * 255
    contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, color, thickness, lineType=cv2.LINE_AA)
    return img


def compute_error_heatmap(
    reference: np.ndarray,
    predicted: np.ndarray,
    *,
    amplify: float = 5.0,
    colormap: int = cv2.COLORMAP_VIRIDIS,
) -> np.ndarray:
    """Generate an amplified difference heatmap (|pred - ref| * amplify)."""
    ref = np.clip(reference, 0, 255).astype(np.float32)
    pred = np.clip(predicted, 0, 255).astype(np.float32)

    diff = np.abs(pred - ref)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY) if diff.ndim == 3 and diff.shape[2] == 3 else diff

    amplified = np.clip(diff_gray * amplify, 0.0, 255.0).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(amplified, colormap)
    return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)


def _add_banner(
    panel: np.ndarray, title: str, *, banner_h: int = 30, bg_color: tuple[int, int, int] = (30, 30, 30)
) -> np.ndarray:
    h, w = panel.shape[:2]
    banner = np.full((banner_h, w, 3), bg_color, dtype=np.uint8)
    cv2.putText(
        banner,
        title,
        (10, banner_h - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return np.vstack([banner, panel])


def create_comparison_strip(
    reference: np.ndarray,
    predicted: np.ndarray,
    *,
    conditioning: np.ndarray | None = None,
    ref_kpts: np.ndarray | None = None,
    pred_kpts: np.ndarray | None = None,
    ref_mask: np.ndarray | None = None,
    pred_mask: np.ndarray | None = None,
    metrics_summary: str | None = None,
    amplify_error: float = 5.0,
) -> np.ndarray:
    """Construct a multi-panel comparison strip with overlays and error heatmap.

    Panels:
        1. Ground Truth (with green skeleton / yellow contour)
        2. Conditioning (wire / reference crop, if provided)
        3. Generated Prediction (with cyan skeleton / magenta contour)
        4. Error Heatmap (|pred - gt| * amplify)
    """
    ref_rgb = np.clip(reference, 0, 255).astype(np.uint8)
    pred_rgb = np.clip(predicted, 0, 255).astype(np.uint8)

    # Panel 1: Ground truth with overlays
    p1 = ref_rgb.copy()
    if ref_mask is not None:
        p1 = draw_mask_contour(p1, ref_mask, color=(0, 255, 255))  # yellow
    if ref_kpts is not None:
        p1 = draw_skeleton(p1, ref_kpts, color=(0, 255, 0))  # green
    panel1 = _add_banner(p1, "Ground Truth")

    # Panel 2: Conditioning (or placeholder if None)
    panels = [panel1]
    if conditioning is not None:
        cond_rgb: np.ndarray = np.clip(conditioning, 0, 255).astype(np.uint8)
        if cond_rgb.shape[:2] != ref_rgb.shape[:2]:
            cond_rgb = cv2.resize(cond_rgb, (ref_rgb.shape[1], ref_rgb.shape[0]))
        panel2 = _add_banner(cond_rgb, "Conditioning Wire")
        panels.append(panel2)

    # Panel 3: Prediction with overlays
    p3 = pred_rgb.copy()
    if pred_mask is not None:
        p3 = draw_mask_contour(p3, pred_mask, color=(255, 0, 255))  # magenta
    if pred_kpts is not None:
        p3 = draw_skeleton(p3, pred_kpts, color=(0, 255, 255))  # cyan
    panel3 = _add_banner(p3, "Generated Prediction")
    panels.append(panel3)

    # Panel 4: Error Heatmap
    heatmap = compute_error_heatmap(ref_rgb, pred_rgb, amplify=amplify_error)
    panel4 = _add_banner(heatmap, f"Error Map (|x-x_hat| x {int(amplify_error)})")
    panels.append(panel4)

    # Combine horizontally
    combined = np.hstack(panels)

    # Add bottom metric bar if provided
    if metrics_summary:
        bar_h = 28
        total_w = combined.shape[1]
        footer = np.full((bar_h, total_w, 3), (20, 20, 20), dtype=np.uint8)
        cv2.putText(
            footer,
            metrics_summary,
            (15, bar_h - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (100, 255, 100),
            1,
            cv2.LINE_AA,
        )
        combined = np.vstack([combined, footer])

    return combined


def save_montage_image(strip: np.ndarray, output_path: Path | str) -> Path:
    """Save an RGB comparison strip to disk as PNG."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(strip, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)
    return path


def generate_carousel_markdown(image_paths: Sequence[Path | str], titles: Sequence[str]) -> str:
    """Format multiple comparison images into an Antigravity carousel block."""
    if not image_paths:
        return ""
    slides = []
    for p, title in zip(image_paths, titles):
        abs_p = str(Path(p).resolve())
        slides.append(f"![{title}]({abs_p})")
    carousel_body = "\n<!-- slide -->\n".join(slides)
    return f"````carousel\n{carousel_body}\n````"

