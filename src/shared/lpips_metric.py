"""LPIPS-like perceptual distance — an uncalibrated VGG-19-bn feature metric.

**This is not the official LPIPS** ("The Unreasonable Effectiveness of Deep
Features as a Perceptual Metric", Zhang et al. 2018). Real LPIPS applies a
small *learned linear calibration* on top of backbone features, fit against a
large human-judgment dataset (BAPPS) — those calibration weights are not
available in this repo and are not trivial to source offline on this host.
Report 10's 2026-07-11 findings entry recorded that LPIPS was previously
(falsely) claimed done and is genuinely absent from `src/`; this module closes
that gap with an explicit, documented approximation rather than silently
skipping it or mislabeling this as the real thing.

What this computes instead: L2 distance between normalized VGG-19-bn
(`assets/weights/vgg19-bn.pth` — already used as the perceptual-loss backbone
in `scripts/train_spade4tennis.py`'s `VGG19PerceptualLoss`) feature maps at
the same four layers used there (relu1_1/2_1/3_1/4_1), averaged per layer
with unit layer weights (no learned calibration). Lower is better; 0 for
identical images. Reported metric key is `lpips_vgg_uncalibrated` everywhere
in this harness so it is never confused with a calibrated LPIPS score from
another codebase.

Mirrors `src/shared/fvd.py`'s split: pure-tensor math is cheap and unit
tested directly; the real backbone load is lazy/cached so importing this
module never requires the checkpoint to exist.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

_LAYER_INDICES = [3, 8, 17, 26]  # relu1_1, relu2_1, relu3_1, relu4_1 (matches VGG19PerceptualLoss)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Frames are resized to this before hitting VGG (mirrors src/shared/fvd.py's
# I3D_FRAME_SIZE convention). Without this, full-resolution (e.g. 4K) source
# frames blow up VGG's early conv activations to tens of GiB and OOM the GPU
# — this bit for real during this workstream's real-data validation.
LPIPS_FRAME_SIZE = 256

_MODEL_CACHE: dict[tuple[str, str], "VGGFeatureExtractor"] = {}


def default_weights_path() -> Path:
    """Resolve the project-relative path to the symlinked VGG-19-bn checkpoint."""
    return Path(__file__).resolve().parents[2] / "assets" / "weights" / "vgg19-bn.pth"


def normalize_for_vgg(frames_rgb01: torch.Tensor) -> torch.Tensor:
    """frames_rgb01: [N, 3, H, W] in [0, 1] -> ImageNet-normalized tensor of the same shape."""
    if frames_rgb01.ndim != 4 or frames_rgb01.shape[1] != 3:
        raise ValueError(f"expected [N, 3, H, W], got {tuple(frames_rgb01.shape)}")
    mean = torch.tensor(_IMAGENET_MEAN, dtype=frames_rgb01.dtype, device=frames_rgb01.device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, dtype=frames_rgb01.dtype, device=frames_rgb01.device).view(1, 3, 1, 1)
    return (frames_rgb01 - mean) / std  # Shape: [N, 3, H, W]


def perceptual_distance_from_features(
    features_a: list[torch.Tensor], features_b: list[torch.Tensor]
) -> float:
    """Mean per-layer L2 distance between two equal-length feature-map lists.

    Each features_*[i]: [N, C_i, H_i, W_i]. Pure-tensor math, no model
    dependency — this is what the unit tests exercise directly.
    """
    if len(features_a) != len(features_b):
        raise ValueError(f"feature list length mismatch: {len(features_a)} vs {len(features_b)}")
    if not features_a:
        raise ValueError("no feature layers supplied")

    per_layer = []
    for layer_a, layer_b in zip(features_a, features_b):
        if layer_a.shape != layer_b.shape:
            raise ValueError(f"feature shape mismatch: {tuple(layer_a.shape)} vs {tuple(layer_b.shape)}")
        # Mean-squared distance per sample, then averaged over the batch.
        diff_sq = (layer_a - layer_b) ** 2  # Shape: [N, C, H, W]
        per_layer.append(diff_sq.mean().item())
    return float(sum(per_layer) / len(per_layer))


class VGGFeatureExtractor:
    """Loads VGG-19-bn and extracts the four relu feature maps used by
    `VGG19PerceptualLoss` (train_spade4tennis.py) — same backbone, reused as
    a perceptual-distance metric rather than a training loss."""

    def __init__(self, weights_path: Path | None = None, device: str | None = None) -> None:
        self._weights_path = weights_path or default_weights_path()
        self._device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model: torch.nn.Module | None = None
        self._slices: list[torch.nn.Sequential] | None = None

    @property
    def device(self) -> str:
        return self._device

    def _load_model(self) -> list[torch.nn.Sequential]:  # pragma: no cover - requires real VGG weights, exercised by integration tests
        if self._slices is not None:
            return self._slices

        import torchvision.models as models

        if self._weights_path.exists():
            vgg = models.vgg19_bn(weights=None)
            vgg.load_state_dict(torch.load(str(self._weights_path), map_location="cpu"))
        else:
            raise FileNotFoundError(
                f"VGG-19-bn weights not found at '{self._weights_path}'. "
                "Expected the symlink assets/weights/vgg19-bn.pth (see CLAUDE.md's "
                "weights convention) to already exist — it is also used by "
                "scripts/train_spade4tennis.py's perceptual loss."
            )

        features = vgg.features
        slices = []
        prev = 0
        for idx in _LAYER_INDICES:
            slices.append(torch.nn.Sequential(*[features[i] for i in range(prev, idx + 1)]))
            prev = idx + 1

        model = torch.nn.ModuleList(slices).to(self._device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        self._model = model
        self._slices = list(model)
        return self._slices

    @torch.no_grad()
    def extract_features(
        self, frames_rgb01: torch.Tensor, batch_size: int = 8
    ) -> list[torch.Tensor]:  # pragma: no cover - requires real VGG weights
        """frames_rgb01: [N, 3, H, W] in [0, 1] -> list of 4 relu feature maps, processed in batches."""
        slices = self._load_model()
        per_batch_outputs: list[list[torch.Tensor]] = []
        for start in range(0, frames_rgb01.shape[0], batch_size):
            x = normalize_for_vgg(frames_rgb01[start : start + batch_size].to(self._device))
            batch_features = []
            for layer in slices:
                x = layer(x)
                batch_features.append(x)
            per_batch_outputs.append(batch_features)
        # Concatenate each layer's feature maps back across the batch dimension.
        return [torch.cat([batch[layer_idx] for batch in per_batch_outputs], dim=0) for layer_idx in range(len(slices))]


def get_cached_extractor(weights_path: Path | None = None, device: str | None = None) -> VGGFeatureExtractor:
    resolved_weights = str(weights_path or default_weights_path())
    resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    cache_key = (resolved_weights, resolved_device)
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = VGGFeatureExtractor(weights_path=weights_path, device=resolved_device)
    return _MODEL_CACHE[cache_key]


def compute_lpips_from_frames(  # pragma: no cover - orchestrates real VGG inference, exercised by integration tests
    reference_frames_rgb01: torch.Tensor,
    predicted_frames_rgb01: torch.Tensor,
    weights_path: Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Compute the uncalibrated VGG perceptual distance between two frame stacks.

    Each input: [N, 3, H, W], float32 in [0, 1]. Frames are matched 1:1 by
    index (resize/frame-count alignment is the caller's job, matching the
    convention already used by `_compute_psnr`/`compute_fvd_from_frames`).
    """
    if reference_frames_rgb01.shape[0] != predicted_frames_rgb01.shape[0]:
        return {
            "lpips_vgg_uncalibrated": None,
            "note": (
                f"frame count mismatch: reference={reference_frames_rgb01.shape[0]} "
                f"predicted={predicted_frames_rgb01.shape[0]}"
            ),
        }
    if reference_frames_rgb01.shape[0] == 0:
        return {"lpips_vgg_uncalibrated": None, "note": "no frames to compare"}

    # Always resize to a fixed, small size before VGG (see LPIPS_FRAME_SIZE's
    # docstring) — this also makes the reference/predicted resolution mismatch
    # case a no-op special case rather than a separate code path.
    target_size = (LPIPS_FRAME_SIZE, LPIPS_FRAME_SIZE)
    reference_frames_rgb01 = F.interpolate(reference_frames_rgb01, size=target_size, mode="bilinear", align_corners=False)
    predicted_frames_rgb01 = F.interpolate(predicted_frames_rgb01, size=target_size, mode="bilinear", align_corners=False)

    extractor = get_cached_extractor(weights_path=weights_path, device=device)
    features_ref = extractor.extract_features(reference_frames_rgb01)
    features_pred = extractor.extract_features(predicted_frames_rgb01)
    distance = perceptual_distance_from_features(features_ref, features_pred)

    return {
        "lpips_vgg_uncalibrated": distance,
        "lpips_backbone": "vgg19_bn_uncalibrated",
        "note": "approximation: uncalibrated VGG-19-bn feature L2, not the learned-linear-calibration LPIPS",
    }
