#!/usr/bin/env python3
"""Simple test of TurboNext model components"""
import torch
from turbonext_model import TurboNextConfig, TurboNextModel

print("Creating model...")
config = TurboNextConfig()
model = TurboNextModel(config)
model.eval()

print("✓ Model created successfully")
print(f"  - VAE type: {type(model.vae).__name__}")
print(f"  - UNet type: {type(model.unet).__name__}")
print(f"  - Pose guider params: {sum(p.numel() for p in model.pose_guider.parameters()):,}")

print("\nTesting pose guider forward...")
pose = torch.randn(1, 3, 512, 512)
pose_features = model.pose_guider(pose)
print(f"✓ Pose features shape: {pose_features.shape}")

print("\nTesting reference caching...")
ref_image = torch.randn(1, 3, 512, 512)
timesteps = torch.tensor([500])
model.cache_reference(ref_image, timesteps)
print(f"✓ Reference cached: {len(model.reference_store.cached_states)} attention states")

print("\nTesting full forward pass...")
pixel_values = torch.randn(1, 3, 512, 512)
pose_images = torch.randn(1, 3, 512, 512)
noisy_latents = torch.randn(1, 4, 64, 64)
timesteps = torch.tensor([500])
output = model(pixel_values, pose_images, ref_image, noisy_latents, timesteps)
print(f"✓ Forward pass output shape: {output.shape}")

print("\n✅ All tests passed!")
