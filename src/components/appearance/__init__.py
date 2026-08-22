"""Appearance representations.

Implementations live in sibling modules; this module holds the registry.
Construction targets are import strings, so importing this module does not load
torch, cv2, or encoder binaries. Do not change ``REGISTRY`` or its axis string
— the parent package and the shared smoke test key on both.
"""

from src.contracts.capabilities import (
    APPEARANCE_COMPRESSED_IMAGE,
    APPEARANCE_DIFFUSION_LATENT,
    APPEARANCE_IMAGE_EMBEDDING,
    appearance,
)
from src.contracts.registry import BackendSpec, Registry

REGISTRY: Registry[object] = Registry("appearance")

REGISTRY.register(
    BackendSpec(
        name=APPEARANCE_COMPRESSED_IMAGE,
        target="src.components.appearance.compressed:CompressedImageAppearance",
        capabilities=appearance(APPEARANCE_COMPRESSED_IMAGE),
        summary="JPEG crop. Quality and downscale are independent knobs.",
    )
)
REGISTRY.register(
    BackendSpec(
        name=APPEARANCE_DIFFUSION_LATENT,
        target="src.components.appearance.latent:DiffusionLatentAppearance",
        capabilities=appearance(APPEARANCE_DIFFUSION_LATENT),
        summary="4 x H/8 x W/8 float16 pack. Exact cost; VAE encode is a later swap.",
    )
)
REGISTRY.register(
    BackendSpec(
        name=APPEARANCE_IMAGE_EMBEDDING,
        target="src.components.appearance.embedding:ImageEmbeddingAppearance",
        capabilities=appearance(APPEARANCE_IMAGE_EMBEDDING),
        summary="Compact colour/texture vector. CLIP weights are not required to encode.",
    )
)
