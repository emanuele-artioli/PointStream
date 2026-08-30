"""Pix2Pix UNet wrapper. Pose + appearance, per frame."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.components.generation._numpy import as_chw, prepare_letterboxed
from src.components.generation.base import BaseFrameGenerator
from src.contracts.capabilities import CONDITION_APPEARANCE, CONDITION_POSE
from src.contracts.conditioning import ConditioningBundle, Device, GenerationParams
from src.contracts import paths

_DEFAULT_WEIGHT = "pix2pix_generator.pt"


class Pix2PixGenerator(BaseFrameGenerator):
    """6-channel (pose RGB + appearance RGB) UNet, matching the training script."""

    required = (CONDITION_POSE, CONDITION_APPEARANCE)

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        checkpoint: str | None = None,
        model: Any = None,
    ) -> None:
        self.width = width
        self.height = height
        self.checkpoint = checkpoint
        self._model = model
        self.loaded_checkpoint: str | None = None
        self.last_seed: int | None = None

    def prepare(
        self, conditioning: ConditioningBundle, params: GenerationParams
    ) -> dict[str, Any]:
        canvas_width, canvas_height = self.canvas_size(params)
        return prepare_letterboxed(
            conditioning.appearance,
            conditioning.bbox,
            canvas_width,
            canvas_height,
            extras={"pose": conditioning.pose},
        )

    def _generate(
        self,
        conditioning: ConditioningBundle,
        *,
        seed: int,
        device: Device,
        params: GenerationParams,
    ) -> np.ndarray:
        prepared = self.prepare(conditioning, params)
        model = self._model if self._model is not None else self._load_model(device)
        self.last_seed = int(seed)
        stacked = np.concatenate(
            [as_chw(prepared["pose"]), as_chw(prepared["appearance"])], axis=0
        )
        output = model(stacked)
        return as_chw(output)

    def _load_model(self, device: Device) -> Any:
        path = _resolve_weight(self.checkpoint, _DEFAULT_WEIGHT)
        net = build_pix2pix_unet()
        net.load_state_dict(_load_state_dict(path))
        net.to(device)
        net.eval()
        self.loaded_checkpoint = str(path)
        self._model = _Pix2PixForward(net, device)
        return self._model


class _Pix2PixForward:
    """nn.Module → numpy callable, so injected test doubles keep the same slot."""

    def __init__(self, module: Any, device: Device) -> None:
        self.module = module
        self.device = device

    def __call__(self, stacked: np.ndarray) -> np.ndarray:
        import torch

        tensor = _uint8_chw_to_batch(stacked).to(self.device)
        with torch.no_grad():
            output = self.module(tensor)
        return _batch_to_uint8_chw(output)


def build_pix2pix_unet() -> Any:
    """UNet matching ``scripts/train_pix2pix.py`` (6-channel pose+appearance)."""
    import torch
    import torch.nn as nn

    class UNetDown(nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            normalize: bool = True,
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            layers: list[nn.Module] = [
                nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=False))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            self.model = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x)

    class UNetUp(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
            super().__init__()
            layers: list[nn.Module] = [
                nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False),
            ]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            self.model = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor, skip_input: torch.Tensor) -> torch.Tensor:
            return torch.cat((self.model(x), skip_input), 1)

    class UNetGenerator(nn.Module):
        def __init__(self, in_channels: int = 6, out_channels: int = 3) -> None:
            super().__init__()
            self.down1 = UNetDown(in_channels, 64, normalize=False)
            self.down2 = UNetDown(64, 128)
            self.down3 = UNetDown(128, 256)
            self.down4 = UNetDown(256, 512, dropout=0.5)
            self.down5 = UNetDown(512, 512, dropout=0.5)
            self.down6 = UNetDown(512, 512, dropout=0.5)
            self.down7 = UNetDown(512, 512, dropout=0.5)
            self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)
            self.up1 = UNetUp(512, 512, dropout=0.5)
            self.up2 = UNetUp(1024, 512, dropout=0.5)
            self.up3 = UNetUp(1024, 512, dropout=0.5)
            self.up4 = UNetUp(1024, 512, dropout=0.5)
            self.up5 = UNetUp(1024, 256)
            self.up6 = UNetUp(512, 128)
            self.up7 = UNetUp(256, 64)
            self.final = nn.Sequential(
                nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1),
                nn.Tanh(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            d1 = self.down1(x)
            d2 = self.down2(d1)
            d3 = self.down3(d2)
            d4 = self.down4(d3)
            d5 = self.down5(d4)
            d6 = self.down6(d5)
            d7 = self.down7(d6)
            d8 = self.down8(d7)
            u1 = self.up1(d8, d7)
            u2 = self.up2(u1, d6)
            u3 = self.up3(u2, d5)
            u4 = self.up4(u3, d4)
            u5 = self.up5(u4, d3)
            u6 = self.up6(u5, d2)
            u7 = self.up7(u6, d1)
            return self.final(u7)

    return UNetGenerator()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_weight(checkpoint: str | None, default_name: str) -> Path:
    if checkpoint:
        path = Path(checkpoint)
        if not path.is_absolute():
            planted = paths.assets() / "weights" / checkpoint
            path = planted if planted.exists() else path
    else:
        path = paths.assets() / "weights" / default_name
    if not path.is_file():
        raise FileNotFoundError(
            f"pix2pix has no model loaded and weight file is missing at {path}. "
            f"Pass model=... for tests, or place {default_name} under assets/weights/."
        )
    return path.resolve()


def _load_state_dict(path: Path) -> Any:
    import torch

    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and any(str(key).startswith("module.") for key in state):
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
    return state


def _uint8_chw_to_batch(stacked: np.ndarray) -> Any:
    import torch

    array = np.asarray(stacked)
    if array.ndim != 3:
        raise ValueError(f"pix2pix expected CHW stacked input, got {array.shape}.")
    tensor = torch.from_numpy(array.astype(np.float32) / 255.0).unsqueeze(0)
    return (tensor - 0.5) * 2.0


def _batch_to_uint8_chw(batch: Any) -> np.ndarray:
    image = (batch.detach().cpu().squeeze(0) + 1.0) / 2.0
    return (image.clamp(0, 1).numpy() * 255.0).astype(np.uint8)
