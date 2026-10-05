"""Recognize the two saved hand-generator formats without importing Torch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def hand_checkpoint_contract(
    checkpoint: Mapping[str, Any], *, expected_factory: str | None = None
) -> tuple[str, int, Mapping[str, Any]]:
    if "state_dict" in checkpoint:
        architecture = checkpoint.get("model")
        if architecture not in ("pix2pix", "spade"):
            raise ValueError("unsupported trained hand-generator architecture")
        if expected_factory is None or checkpoint.get("factory") != expected_factory:
            raise ValueError("trained checkpoint must match the explicitly selected factory")
        channels = 4 if architecture == "spade" else 3
        state = checkpoint["state_dict"]
    else:
        state = checkpoint.get("model_state_dict")
        if not isinstance(state, Mapping) or not state:
            raise ValueError("missing hand-generator state")
        architecture = (
            "spade"
            if checkpoint.get("model_type") == "spade" or "enc1.0.weight" in state
            else "pix2pix"
        )
        channels = checkpoint.get("out_channels", 3)
    if type(channels) is not int or channels not in (3, 4):
        raise ValueError("unsupported hand-generator output channels")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("missing hand-generator state")
    return architecture, channels, state
