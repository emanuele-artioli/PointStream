"""Archived still-image background fills: SDXL, FLUX Fill, and Qwen Image Edit.

DiffuEraser is the kept filler because it is a video model. These per-frame
runners stay here so they can be rerun later. They are not on the encode path.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from demo.experiments.inpaint_background_smoke import PROMPT, _composite
from demo.experiments.neural_bg import feather_mask


def fill_sdxl(frames, masks, checkpoint: Path, device: str) -> list[np.ndarray]:
    import torch
    from diffusers import AutoPipelineForInpainting

    pipe = AutoPipelineForInpainting.from_pretrained(checkpoint, torch_dtype=torch.float16)
    pipe.to(device)
    out = []
    for frame, mask in zip(frames, masks):
        soft = feather_mask(mask)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pipe(
            prompt=PROMPT,
            image=Image.fromarray(rgb),
            mask_image=Image.fromarray((soft * 255).astype(np.uint8)),
            num_inference_steps=20,
            strength=0.99,
        ).images[0]
        filled = cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR)
        filled = cv2.resize(filled, (frame.shape[1], frame.shape[0]))
        out.append(_composite(frame, filled, soft))
    return out


def fill_flux(frames, masks, checkpoint: Path, device: str) -> list[np.ndarray]:
    import torch
    from diffusers import FluxFillPipeline

    pipe = FluxFillPipeline.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
    pipe.to(device)
    out = []
    for frame, mask in zip(frames, masks):
        soft = feather_mask(mask)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pipe(
            prompt=PROMPT,
            image=Image.fromarray(rgb),
            mask_image=Image.fromarray((soft * 255).astype(np.uint8)),
            num_inference_steps=20,
            guidance_scale=30,
        ).images[0]
        filled = cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR)
        filled = cv2.resize(filled, (frame.shape[1], frame.shape[0]))
        out.append(_composite(frame, filled, soft))
    return out


def fill_qwen(frames, masks, checkpoint: Path, device: str) -> list[np.ndarray]:
    import torch
    from diffusers import QwenImageEditPipeline

    pipe = QwenImageEditPipeline.from_pretrained(checkpoint, torch_dtype=torch.bfloat16)
    pipe.to(device)
    out = []
    for frame, mask in zip(frames, masks):
        soft = feather_mask(mask)
        hole = frame.copy()
        hole[soft > 0.5] = (114, 114, 114)
        rgb = cv2.cvtColor(hole, cv2.COLOR_BGR2RGB)
        result = pipe(
            image=Image.fromarray(rgb),
            prompt=PROMPT,
            num_inference_steps=20,
            true_cfg_scale=4.0,
        ).images[0]
        filled = cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR)
        filled = cv2.resize(filled, (frame.shape[1], frame.shape[0]))
        out.append(_composite(frame, filled, soft))
    return out
