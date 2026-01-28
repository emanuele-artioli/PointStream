import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from diffusers import AutoencoderKL, MotionAdapter, UNet2DConditionModel
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer


class ControlNeXtPoseGuider(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32, out_channels: int = 320):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(base_channels * 4, out_channels, kernel_size=3, stride=1, padding=1)
        self.act = nn.SiLU()
        self.zero_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.to_latent = nn.Conv2d(out_channels, 4, kernel_size=1)
        nn.init.zeros_(self.zero_conv.weight)
        nn.init.zeros_(self.zero_conv.bias)
        nn.init.zeros_(self.to_latent.weight)
        nn.init.zeros_(self.to_latent.bias)

    def forward(self, pose_image: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(pose_image))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.conv4(x)
        return self.to_latent(self.zero_conv(x))


class ReferenceAttentionStore:
    def __init__(self) -> None:
        self.hidden_states: List[torch.Tensor] = []
        self.cached_states: List[torch.Tensor] = []
        self.index: int = 0

    def clear(self) -> None:
        self.hidden_states.clear()
        self.index = 0

    def push(self, hidden_states: torch.Tensor) -> None:
        self.hidden_states.append(hidden_states)

    def cache_current(self) -> None:
        self.cached_states = list(self.hidden_states)

    def reset_index(self) -> None:
        self.index = 0

    def pop(self) -> torch.Tensor:
        if self.index >= len(self.cached_states):
            # During gradient checkpointing recompute, reset index if exceeded
            self.index = self.index % max(len(self.cached_states), 1)
        item = self.cached_states[self.index]
        self.index += 1
        return item


class ReferenceAttentionProcessor(nn.Module):
    def __init__(self, store: ReferenceAttentionStore, mode: str) -> None:
        super().__init__()
        self.store = store
        self.mode = mode
        self.base = AttnProcessor2_0()

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
    ):
        if self.mode == "write":
            output = self.base(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
            self.store.push(output)  # Store the output
            return output
        if self.mode == "read":
            ref_output = self.store.pop()
            output = self.base(attn, hidden_states, encoder_hidden_states, attention_mask, temb)
            # Blend reference and current output
            return output + ref_output * 0.5
        raise ValueError(f"Unknown mode: {self.mode}")


@dataclass
class TurboNextConfig:
    sd_path: str = "runwayml/stable-diffusion-v1-5"
    motion_adapter_path: str = "guoyww/animatediff-motion-adapter-v1-5-2"
    dtype: torch.dtype = torch.float16


class TurboNextModel(nn.Module):
    def __init__(self, config: TurboNextConfig) -> None:
        super().__init__()
        self.config = config

        self.tokenizer = CLIPTokenizer.from_pretrained(config.sd_path, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(config.sd_path, subfolder="text_encoder")
        self.vae = AutoencoderKL.from_pretrained(config.sd_path, subfolder="vae")

        self.ref_unet = UNet2DConditionModel.from_pretrained(config.sd_path, subfolder="unet")
        self.ref_unet.requires_grad_(False)

        # Use standard UNet for training (not motion model to avoid 5D tensor requirements)
        self.unet = UNet2DConditionModel.from_pretrained(config.sd_path, subfolder="unet")

        self.pose_guider = ControlNeXtPoseGuider()
        self.reference_store = ReferenceAttentionStore()
        self._setup_reference_attention()

        # Freeze base models
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        # Disable gradient checkpointing due to reference attention incompatibility
        # self.unet.enable_gradient_checkpointing()
        # self.ref_unet.enable_gradient_checkpointing()

    def _setup_reference_attention(self) -> None:
        ref_processors: Dict[str, nn.Module] = {}
        main_processors: Dict[str, nn.Module] = {}

        # Get ref_unet processors (standard UNet2D)
        for name in self.ref_unet.attn_processors.keys():
            if name.endswith("attn1.processor"):
                ref_processors[name] = ReferenceAttentionProcessor(self.reference_store, mode="write")
            else:
                ref_processors[name] = AttnProcessor2_0()

        # Get main unet processors (UNetMotionModel - has more processors)
        for name in self.unet.attn_processors.keys():
            if name.endswith("attn1.processor"):
                main_processors[name] = ReferenceAttentionProcessor(self.reference_store, mode="read")
            else:
                main_processors[name] = AttnProcessor2_0()

        self.ref_unet.set_attn_processor(ref_processors)
        self.unet.set_attn_processor(main_processors)

    def encode_prompt(self, prompt: List[str], device: torch.device) -> torch.Tensor:
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            input_ids = text_inputs.input_ids.to('cpu')
            self.text_encoder.to('cpu')
            embeds = self.text_encoder(input_ids)[0]
            self.text_encoder.to(device)
        return embeds.to(device)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        device = image.device
        with torch.no_grad():
            image_cpu = image.cpu()
            self.vae.to('cpu')
            output = self.vae.encode(image_cpu)
            # Handle both standard VAE and TinyVAE outputs
            if hasattr(output, 'latent_dist'):
                latents = output.latent_dist.sample()
            else:
                latents = output.latents
            self.vae.to(device)
        return (latents * 0.18215).to(device)

    def cache_reference(self, ref_image: torch.Tensor, timesteps: torch.Tensor, prompt: Optional[List[str]] = None) -> None:
        device = ref_image.device
        prompt = prompt or [""] * ref_image.shape[0]
        text_embeds = self.encode_prompt(prompt, device)
        ref_latents = self.encode_image(ref_image)

        self.reference_store.clear()
        _ = self.ref_unet(
            ref_latents,
            timesteps,
            encoder_hidden_states=text_embeds,
            return_dict=False,
        )
        self.reference_store.cache_current()
        self.reference_store.reset_index()

    def forward(
        self,
        pixel_values: torch.Tensor,
        pose_images: torch.Tensor,
        ref_image: torch.Tensor,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt: Optional[List[str]] = None,
    ) -> torch.Tensor:
        device = pixel_values.device
        prompt = prompt or [""] * pixel_values.shape[0]

        text_embeds = self.encode_prompt(prompt, device)
        pose_cond = self.pose_guider(pose_images)
        noisy_latents = noisy_latents + pose_cond
        
        self.cache_reference(ref_image, timesteps, prompt)
        self.reference_store.reset_index()

        model_output = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_embeds,
            return_dict=False,
        )[0]
        return model_output


def load_lcm_lora(unet: nn.Module, lora_path: str) -> None:
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LCM LoRA not found at {lora_path}")
    state = torch.load(lora_path, map_location="cpu")
    unet.load_state_dict(state, strict=False)
