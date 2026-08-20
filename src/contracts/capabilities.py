"""The capability vocabulary every backend declares itself in terms of.

Capabilities are how one axis learns what another needs without inspecting its
name. A generator that declares ``requires={CONDITION_MASK}`` is the reason
segmentation runs; nothing anywhere reads the generator's name to decide that.
The arrangement this replaces had a module reach across axes and null the pose
estimator whenever the configured generator's *name* happened to contain
``"canny-controlnet"``, which is invisible from either end.

Two kinds of entry, one namespace convention:

- **Abilities** are plain strings — ``temporal-sequence``, ``open-vocabulary``.
  What a backend can do.
- **Acceptances** are ``namespace:value`` — ``motion:keypoints``,
  ``appearance:diffusion-latent``. What a backend can consume.

Keeping both in one set means the pairing checks and the ability checks are the
same lookup, and a backend's whole declaration is one field.
"""

from __future__ import annotations

from typing import Final

# --------------------------------------------------------------------------
# Namespaces
# --------------------------------------------------------------------------

NS_APPEARANCE: Final = "appearance"
NS_MOTION: Final = "motion"
NS_DOMAIN: Final = "domain"

ALL_NAMESPACES: Final = frozenset({NS_APPEARANCE, NS_MOTION, NS_DOMAIN})


# --------------------------------------------------------------------------
# Appearance representations — what an object looks like, established once
# --------------------------------------------------------------------------

#: The object crop as a compressed image, degraded by JPEG quality or by
#: downscaling. The two are not equivalent — one discards high-frequency detail
#: through quantization, the other through resolution — so both are knobs on
#: this one representation rather than separate representations.
APPEARANCE_COMPRESSED_IMAGE: Final = "compressed-image"

#: The crop encoded into a diffusion model's own VAE latent space. Compact, and
#: no format mismatch for a decoder that consumes latents natively.
APPEARANCE_DIFFUSION_LATENT: Final = "diffusion-latent"

#: A CLIP/IP-Adapter-style appearance vector — a few KB, and the most compact
#: option that still carries colour and texture.
APPEARANCE_IMAGE_EMBEDDING: Final = "image-embedding"

ALL_APPEARANCE: Final = frozenset(
    {
        APPEARANCE_COMPRESSED_IMAGE,
        APPEARANCE_DIFFUSION_LATENT,
        APPEARANCE_IMAGE_EMBEDDING,
    }
)


# --------------------------------------------------------------------------
# Motion representations — how that appearance evolves
# --------------------------------------------------------------------------

#: Per-frame pose vector. Requires the object class to have a stable skeleton,
#: so it applies to humans and animals but not to vehicles or arbitrary shapes.
MOTION_KEYPOINTS: Final = "keypoints"

#: Per-frame dense or block motion field. Class-agnostic — the option that makes
#: skeleton-less objects a configuration rather than a redesign.
MOTION_VECTORS: Final = "motion-vectors"

#: The object crop encoded as a literal video after its appearance keyframe.
#: The classical codec answer applied per object, and the baseline the
#: generative representations have to beat.
MOTION_ENCODED_VIDEO: Final = "encoded-video"

ALL_MOTION: Final = frozenset({MOTION_KEYPOINTS, MOTION_VECTORS, MOTION_ENCODED_VIDEO})


# --------------------------------------------------------------------------
# Conditioning kinds — what a generator needs the encoder to produce
# --------------------------------------------------------------------------

CONDITION_POSE: Final = "pose"
CONDITION_MASK: Final = "mask"
CONDITION_CANNY: Final = "canny"
CONDITION_APPEARANCE: Final = "appearance"
CONDITION_MOTION_FIELD: Final = "motion-field"

ALL_CONDITIONS: Final = frozenset(
    {
        CONDITION_POSE,
        CONDITION_MASK,
        CONDITION_CANNY,
        CONDITION_APPEARANCE,
        CONDITION_MOTION_FIELD,
    }
)


# --------------------------------------------------------------------------
# Abilities
# --------------------------------------------------------------------------

#: Generates one frame at a time. Nearly every generator has this.
CAP_PER_FRAME: Final = "per-frame"

#: Generates a temporally-coherent sequence in one call, conditioned on a window
#: of motion rather than a single frame's worth. Declared, not inferred: the
#: arrangement this replaces detected temporal capability with an `isinstance`
#: check against one specific class, so a new temporal backend had to subclass
#: that class to be recognised at all.
CAP_TEMPORAL_SEQUENCE: Final = "temporal-sequence"

#: Accepts a free-text class prompt, so subject selection can be expressed as
#: "tennis player" rather than as hand-written heuristics over generic
#: detections. Whether that is as good as the heuristics is an open question and
#: one of the comparisons this vocabulary exists to make possible.
CAP_OPEN_VOCABULARY: Final = "open-vocabulary"

#: Emits per-object masks, not just boxes.
CAP_INSTANCE_MASKS: Final = "instance-masks"

#: Maintains object identity across frames without a separate tracker.
CAP_NATIVE_TRACKING: Final = "native-tracking"

#: Supports genuine block-level region-of-interest bit allocation. Load-bearing
#: for the codec comparison: baselines must be allowed ROI too, or beating them
#: proves nothing.
CAP_ROI: Final = "roi"

#: Can encode mathematically losslessly.
CAP_LOSSLESS: Final = "lossless"

ALL_ABILITIES: Final = frozenset(
    {
        CAP_PER_FRAME,
        CAP_TEMPORAL_SEQUENCE,
        CAP_OPEN_VOCABULARY,
        CAP_INSTANCE_MASKS,
        CAP_NATIVE_TRACKING,
        CAP_ROI,
        CAP_LOSSLESS,
    }
)


# --------------------------------------------------------------------------
# Construction helpers
# --------------------------------------------------------------------------


def appearance(*values: str) -> frozenset[str]:
    """Namespaced capability entries for the appearance representations given.

    ``appearance(APPEARANCE_COMPRESSED_IMAGE)`` → ``{"appearance:compressed-image"}``.
    """
    return frozenset(f"{NS_APPEARANCE}:{value}" for value in values)


def motion(*values: str) -> frozenset[str]:
    """Namespaced capability entries for the motion representations given."""
    return frozenset(f"{NS_MOTION}:{value}" for value in values)


def domains(*values: str) -> frozenset[str]:
    """Namespaced capability entries for the domains a backend is valid in.

    Most backends declare nothing here, meaning domain-agnostic. A subject
    selector encoding tennis-specific rules declares ``domains("tennis")`` so
    selecting it under another profile fails at validation.
    """
    return frozenset(f"{NS_DOMAIN}:{value}" for value in values)
