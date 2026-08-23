"""SPADE4Tennis wrapper. Tennis-specific SPADE generator, per frame."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.components.generation._numpy import as_chw, as_hwc, prepare_letterboxed
from src.components.generation.base import BaseFrameGenerator
from src.contracts.capabilities import CONDITION_APPEARANCE, CONDITION_POSE
from src.contracts.conditioning import ConditioningBundle, Device, GenerationParams

_DEFAULT_WEIGHT = "spade4tennis_lite_generator.pt"


class Spade4TennisGenerator(BaseFrameGenerator):
    """SPADE-conditioned ResNet-9. Domain-specific; scores do not generalise."""

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
        output = model(prepared["appearance"], prepared["pose"])
        return as_chw(output)

    def _load_model(self, device: Device) -> Any:
        from src.components.generation.spade4tennis_arch import SPADEResNet9Generator

        path = _resolve_weight(self.checkpoint, _DEFAULT_WEIGHT)
        net = SPADEResNet9Generator(in_nc=3, out_nc=3, ngf=64, n_blocks=9)
        net.load_state_dict(_load_state_dict(path))
        net.to(device)
        net.eval()
        self.loaded_checkpoint = str(path)
        self._model = _SpadeForward(net, device)
        return self._model


class _SpadeForward:
    """Maps ``(appearance, pose)`` numpy to the module's ``(skeleton, reference)``."""

    def __init__(self, module: Any, device: Device) -> None:
        self.module = module
        self.device = device

    def __call__(self, appearance: np.ndarray, pose: np.ndarray) -> np.ndarray:
        import torch

        skeleton = _hwc_to_batch(pose).to(self.device)
        reference = _hwc_to_batch(appearance).to(self.device)
        with torch.no_grad():
            output = self.module(skeleton, reference)
        return _batch_to_uint8_chw(output)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_weight(checkpoint: str | None, default_name: str) -> Path:
    if checkpoint:
        path = Path(checkpoint)
        if not path.is_absolute():
            planted = _repo_root() / "assets" / "weights" / checkpoint
            path = planted if planted.exists() else path
    else:
        path = _repo_root() / "assets" / "weights" / default_name
    if not path.is_file():
        raise FileNotFoundError(
            f"spade4tennis has no model loaded and weight file is missing at {path}. "
            f"Pass model=... for tests, or place {default_name} under assets/weights/. "
            f"Lite ResNet-9 is cheap to wire; a missing file is the only reason to drop it."
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


def _hwc_to_batch(image: np.ndarray) -> Any:
    import torch

    hwc = as_hwc(image)
    tensor = torch.from_numpy(np.transpose(hwc, (2, 0, 1)).astype(np.float32) / 255.0)
    return ((tensor - 0.5) * 2.0).unsqueeze(0)


def _batch_to_uint8_chw(batch: Any) -> np.ndarray:
    image = (batch.detach().cpu().squeeze(0) + 1.0) / 2.0
    return (image.clamp(0, 1).numpy() * 255.0).astype(np.uint8)
