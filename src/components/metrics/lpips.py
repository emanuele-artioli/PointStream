"""LPIPS on the pipeline path — frames in, a calibrated perceptual distance out.

This wraps the published ``lpips`` package, which applies the learned linear
calibration on top of AlexNet features. That calibration is what makes the
number comparable to LPIPS figures in the literature.

**History, because it cost us a wrong conclusion.** This module previously
computed an *uncalibrated* VGG-19-bn feature MSE under the name ``lpips``. It
was honest about that in its docstring and still wrong to ship, because it was
registered, reported and read as LPIPS. It had almost no dynamic range —
measured on 2026-08-23, an *unrelated image* scored 0.083 while a good
reconstruction scored 0.085, so it could not tell them apart. Engine rankings
taken on it are void.

Calibration anchors for the current implementation, asserted in the tests:

| pair | this module | old VGG-MSE |
|---|---|---|
| identical | 0.000 | 0.000 |
| mild noise | 0.250 | 0.009 |
| heavy blur | 0.430 | 0.032 |
| unrelated image | 0.645 | 0.083 |

A caller injects ``extractor`` to mock the network; the real one loads lazily so
importing this module does not require torch.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from src.components.metrics.frames import paired, to_clip

_MODEL_CACHE: dict[tuple[str, str], Any] = {}


def _load_model(net: str, device: str) -> Any:
    key = (net, device)
    if key not in _MODEL_CACHE:
        import lpips as lpips_pkg
        import torch

        model = lpips_pkg.LPIPS(net=net, verbose=False).eval()
        _MODEL_CACHE[key] = model.to(torch.device(device))
    return _MODEL_CACHE[key]


class LpipsMetric:
    """Calibrated LPIPS. Lower is better; 0 if identical."""

    name = "lpips"

    def __init__(
        self,
        *,
        net: str = "alex",
        device: str = "cpu",
        extractor: Callable[[np.ndarray, np.ndarray], float] | None = None,
    ) -> None:
        self._net = net
        self._device = device
        self._extractor = extractor

    def score(self, reference: np.ndarray, predicted: np.ndarray) -> float:
        ref, pred = paired(reference, predicted)
        if self._extractor is not None:
            return float(self._extractor(ref, pred))
        return self._score_frames(ref, pred)

    def _score_frames(self, ref: np.ndarray, pred: np.ndarray) -> float:
        import torch

        model = _load_model(self._net, self._device)
        device = torch.device(self._device)

        def batch(clip: np.ndarray) -> Any:
            rgb = to_clip(clip).astype(np.float32) / 127.5 - 1.0
            nchw = np.transpose(rgb, (0, 3, 1, 2))
            return torch.from_numpy(np.ascontiguousarray(nchw)).to(device)

        with torch.no_grad():
            scores = model(batch(ref), batch(pred))
        return float(scores.mean().item())


def build(**kwargs: Any) -> LpipsMetric:
    return LpipsMetric(**kwargs)
