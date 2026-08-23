"""LPIPS on the pipeline path — frames in, a distance out.

The published LPIPS metric applies a learned linear calibration on VGG/AlexNet
features. That calibration is not a dependency of this package. What this
backend computes, named ``lpips`` to match the contract, is the uncalibrated
feature distance (unit-normalized per-layer L2) used as a perceptual proxy.
A caller injects ``extractor`` to mock the network; the default loads VGG-19-bn
lazily so importing this module does not require torch or weights.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from src.components.metrics.frames import paired, to_clip
from src.contracts.metrics import LPIPS

FeatureMaps = Sequence[np.ndarray]
FeatureExtractor = Callable[[np.ndarray], FeatureMaps]

_LAYER_INDICES = (3, 8, 17, 26)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_FRAME_SIZE = 256


class LpipsMetric:
    """Perceptual distance in ``[0, 1]``-ish units. Lower is better. 0 if identical."""

    name = LPIPS.name

    def __init__(
        self,
        extractor: FeatureExtractor | None = None,
        weights_path: Path | None = None,
    ) -> None:
        self._extractor = extractor
        self._weights_path = weights_path

    def score(self, reference: np.ndarray, predicted: np.ndarray) -> float:
        ref, pred = paired(reference, predicted)
        extract = self._extractor or _vgg_extractor(self._weights_path)
        return perceptual_distance(extract(ref), extract(pred))


def perceptual_distance(features_a: FeatureMaps, features_b: FeatureMaps) -> float:
    """Mean per-layer mean-squared distance. Pure numpy, no network."""
    if len(features_a) != len(features_b):
        raise ValueError(
            f"feature list length mismatch: {len(features_a)} vs {len(features_b)}"
        )
    if not features_a:
        raise ValueError("no feature layers supplied")
    distances: list[float] = []
    for layer_a, layer_b in zip(features_a, features_b, strict=True):
        left = np.asarray(layer_a, dtype=np.float64)
        right = np.asarray(layer_b, dtype=np.float64)
        if left.shape != right.shape:
            raise ValueError(f"feature shape mismatch: {left.shape} vs {right.shape}")
        distances.append(float(np.mean((left - right) ** 2)))
    return float(sum(distances) / len(distances))


def default_weights_path() -> Path:
    """Project-relative VGG-19-bn checkpoint, same file the training loss uses."""
    return Path(__file__).resolve().parents[3] / "assets" / "weights" / "vgg19-bn.pth"


def _vgg_extractor(weights_path: Path | None) -> FeatureExtractor:
    path = weights_path or default_weights_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"VGG-19-bn weights not found at {path}. "
            "Pass extractor=... to score without the checkpoint, or install "
            "assets/weights/vgg19-bn.pth."
        )

    import torch
    import torch.nn.functional as torch_f
    import torchvision.models as models

    vgg = models.vgg19_bn(weights=None)
    state = torch.load(str(path), map_location="cpu")
    vgg.load_state_dict(state)
    features = vgg.features
    slices = []
    prev = 0
    for index in _LAYER_INDICES:
        slices.append(features[prev : index + 1].eval())
        prev = index + 1
    for slice_net in slices:
        for parameter in slice_net.parameters():
            parameter.requires_grad_(False)

    def extract(clip: np.ndarray) -> FeatureMaps:
        rgb = to_clip(clip).astype(np.float32) / 255.0
        nchw = np.transpose(rgb, (0, 3, 1, 2))
        tensor = torch.from_numpy(nchw)
        tensor = torch_f.interpolate(
            tensor, size=(_FRAME_SIZE, _FRAME_SIZE), mode="bilinear", align_corners=False
        )
        mean = torch.from_numpy(_IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.from_numpy(_IMAGENET_STD).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        maps: list[np.ndarray] = []
        current = tensor
        with torch.no_grad():
            for slice_net in slices:
                current = slice_net(current)
                maps.append(current.detach().cpu().numpy())
        return maps

    return extract
