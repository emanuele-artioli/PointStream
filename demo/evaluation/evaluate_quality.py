"""Quality evaluator: PSNR, SSIM, LPIPS, DISTS, VMAF, and Scoped Hand-ROI metrics."""

from __future__ import annotations

import sqlite3  # noqa: F401
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch

try:
    import lpips
    _HAS_LPIPS = True
except ImportError:
    _HAS_LPIPS = False

try:
    from DISTS_pytorch import DISTS
    _HAS_DISTS = True
except ImportError:
    _HAS_DISTS = False

from demo.pipeline.hand_keypoints import FrameHandPose


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return float(10.0 * math.log10((255.0 ** 2) / mse))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    # Simplified fast SSIM over grayscale luminance
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = cv2.GaussianBlur(gray1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(gray2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(gray1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(gray2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(gray1 * gray2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(np.mean(ssim_map))


class QualityEvaluator:
    def __init__(self, device_str: str = "cuda:1") -> None:
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        self.lpips_fn = None
        self.dists_fn = None

        if _HAS_LPIPS:
            self.lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(self.device)
            self.lpips_fn.eval()

        if _HAS_DISTS:
            try:
                self.dists_fn = DISTS().to(self.device)
                self.dists_fn.eval()
            except Exception:
                self.dists_fn = None

    def evaluate_frames(
        self,
        ref_frames: list[np.ndarray],
        rec_frames: list[np.ndarray],
        poses: list[FrameHandPose] | None = None,
    ) -> dict[str, float]:
        n = min(len(ref_frames), len(rec_frames))
        if n == 0:
            return {}

        psnr_list, ssim_list, lpips_list, dists_list = [], [], [], []
        hand_psnr_list, hand_ssim_list = [], []

        for i in range(n):
            r_img = ref_frames[i]
            d_img = rec_frames[i]
            if d_img.shape[:2] != r_img.shape[:2]:
                d_img = cv2.resize(d_img, (r_img.shape[1], r_img.shape[0]), interpolation=cv2.INTER_LANCZOS4)

            psnr_val = compute_psnr(r_img, d_img)
            ssim_val = compute_ssim(r_img, d_img)
            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)

            # PyTorch perceptual tensors: [1, 3, H, W] in [-1, 1]
            if self.lpips_fn is not None or self.dists_fn is not None:
                # Downsample for fast feature extraction
                small_r = cv2.resize(r_img, (512, 288), interpolation=cv2.INTER_AREA)
                small_d = cv2.resize(d_img, (512, 288), interpolation=cv2.INTER_AREA)

                t_ref = torch.from_numpy(small_r).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 127.5 - 1.0
                t_rec = torch.from_numpy(small_d).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 127.5 - 1.0

                with torch.no_grad():
                    if self.lpips_fn is not None:
                        val = float(self.lpips_fn(t_ref, t_rec).item())
                        lpips_list.append(val)
                    if self.dists_fn is not None:
                        val = float(self.dists_fn(t_ref, t_rec).item())
                        dists_list.append(val)

            # Scoped Hand-ROI metrics
            if poses and i < len(poses) and poses[i].hands:
                for hand in poses[i].hands:
                    x1, y1, x2, y2 = hand.bbox
                    h_crop_ref = r_img[y1:y2, x1:x2]
                    h_crop_rec = d_img[y1:y2, x1:x2]
                    if h_crop_ref.size > 0 and h_crop_rec.size > 0:
                        hand_psnr_list.append(compute_psnr(h_crop_ref, h_crop_rec))
                        hand_ssim_list.append(compute_ssim(h_crop_ref, h_crop_rec))

        return {
            "psnr_dB": float(np.mean(psnr_list)),
            "ssim": float(np.mean(ssim_list)),
            "lpips": float(np.mean(lpips_list)) if lpips_list else 0.0,
            "dists": float(np.mean(dists_list)) if dists_list else 0.0,
            "hand_roi_psnr_dB": float(np.mean(hand_psnr_list)) if hand_psnr_list else float(np.mean(psnr_list)),
            "hand_roi_ssim": float(np.mean(hand_ssim_list)) if hand_ssim_list else float(np.mean(ssim_list)),
        }

