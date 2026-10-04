"""Four-channel smoke generators. Legacy three-channel modules stay loadable."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from demo.models.unet_generator import HandPix2PixUNet, HandSPADEUNet

CHECKPOINT_SCHEMA = "pointstream.foreground_smoke_checkpoint.v1"


class _RGBATranspose(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose2d(128, 4, kernel_size=4, stride=2, padding=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        raw = self.conv(features)
        return torch.cat((torch.tanh(raw[:, :3]), torch.sigmoid(raw[:, 3:4])), dim=1)


class SmokePix2Pix(nn.Module):
    """Pix2pix body with a new RGBA head. An old 3-channel final layer does not fit."""

    def __init__(self) -> None:
        super().__init__()
        self.body = HandPix2PixUNet(out_channels=3)
        self.body.final = _RGBATranspose()

    def forward(self, appearance_and_pose: torch.Tensor) -> torch.Tensor:
        return self.body(appearance_and_pose)


def build_smoke_model(name: str) -> nn.Module:
    if name == "spade":
        return HandSPADEUNet(out_channels=4)
    if name == "pix2pix":
        return SmokePix2Pix()
    raise ValueError(f"unknown smoke architecture {name}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def initialization_sha256(model: nn.Module) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name, parameter in model.state_dict().items():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def save_smoke_checkpoint(path: Path, model: nn.Module, *, architecture: str, objective: str, step: int) -> None:
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "architecture": architecture,
            "objective": objective,
            "out_channels": 4,
            "step": int(step),
            "state_dict": model.state_dict(),
        },
        path,
    )


def checkpoint_kind(blob: dict) -> str:
    """Label a checkpoint. Legacy RGB checkpoints are not given an oracle alpha."""
    if not isinstance(blob, dict):
        raise ValueError("checkpoint must be a dict")
    if blob.get("schema") == CHECKPOINT_SCHEMA and blob.get("out_channels") == 4:
        return "smoke_rgba"
    channels = blob.get("out_channels")
    state = blob.get("state_dict", blob)
    if channels == 3 or any(str(key).endswith("final.0.weight") and getattr(state[key], "shape", [0])[0] == 3 for key in state):
        return "legacy"
    raise ValueError("unrecognized checkpoint; refusing to invent an alpha channel")


def load_smoke_checkpoint(path: Path, model: nn.Module, *, architecture: str) -> dict:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint_kind(blob) != "smoke_rgba":
        raise ValueError("refusing to load a non-RGBA checkpoint as a smoke model")
    if blob.get("architecture") != architecture or blob.get("out_channels") != 4:
        raise ValueError("checkpoint shape or architecture mismatch")
    model.load_state_dict(blob["state_dict"], strict=True)
    return blob
