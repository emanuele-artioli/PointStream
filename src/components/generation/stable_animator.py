"""StableAnimator: identity-preserving pose-to-video. One modern candidate.

Licence check, 2026-08-22, first-hand:

- Hugging Face model card ``FrancisRing/StableAnimator`` YAML frontmatter and
  Hub API ``cardData.license`` both say **apache-2.0**. Adapter files under
  ``Animation/`` (``unet.pth``, ``pose_net.pth``, ``face_encoder.pth``) sit on
  that card.
- GitHub ``Francis-Rings/StableAnimator`` ``LICENSE`` is **MIT** (Copyright
  Shuyuan Tu), not Apache-2.0. Code and weights are licensed separately.
- Inference still loads **Stable Video Diffusion XT** (the repo's ``SVD/`` /
  ``stable-video-diffusion-img2vid-xt/`` tree, including ``LICENSE.md``). That
  backbone is Stability AI Community / SVD research-licensed. Same class of
  block that strands MOFA-Video: those weights are **not bundled** and cannot
  ship as a flagship.
- The same snapshot also vendors InsightFace ``antelopev2`` ONNX detectors,
  which are typically non-commercial. Not bundled.

Construction succeeds. Loading the real stack raises until a caller injects a
runtime or supplies a local SVD tree that they already hold. Extra packages
(torch 2.5.1+cu124, xformers, insightface, the upstream ``animation`` package)
must live in a **separate conda env**, never in ``pointstream``.

Contract fit: reference image + pose-image sequence maps onto
``CONDITION_APPEARANCE`` + ``CONDITION_POSE``. The model also wants per-frame
face crops for the ID path; ``ConditioningBundle`` has no face field. Faces
travel only as ``params.extra["faces"]``. That is a contract finding, not a
silent extra channel.

MTVCrafter was not wrapped this wave: StableAnimator is the one adopted
candidate. MTVCrafter is Apache-2.0 with public weights, but wants raw 4D/SMPL
motion tokens rather than 2D pose images, so it fights this contract harder.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from src.components.generation._numpy import as_chw, as_hwc
from src.components.generation.base import BaseFrameGenerator
from src.components.generation.pose import letterbox_from_bbox, letterbox_image
from src.contracts.capabilities import CONDITION_APPEARANCE, CONDITION_POSE
from src.contracts.conditioning import ConditioningBundle, Device, GenerationParams

HF_MODEL_CARD = "https://huggingface.co/FrancisRing/StableAnimator"
HF_LICENSE_CHECKED = "apache-2.0"
GITHUB_LICENSE_CHECKED = "MIT"
LICENSE_CHECKED_ON = "2026-08-22"

SVD_LICENSE_BLOCK = (
    "StableAnimator adapter weights are Apache-2.0 on the Hugging Face card "
    f"({HF_MODEL_CARD}, checked {LICENSE_CHECKED_ON}). Inference depends on "
    "Stable Video Diffusion XT under the Stability AI Community License, which "
    "is not bundled and must not be copied into this repo. InsightFace "
    "antelopev2 weights in the same snapshot are also not bundled. Pass "
    "runtime=... for tests, or a caller-held SVD tree from a separate conda env."
)

LICENCE_NOTES = (
    f"HF card {HF_MODEL_CARD}: {HF_LICENSE_CHECKED} (YAML + Hub API cardData, "
    f"{LICENSE_CHECKED_ON}). GitHub Francis-Rings/StableAnimator LICENSE: "
    f"{GITHUB_LICENSE_CHECKED}. SVD-XT backbone: Stability AI, not bundled. "
    "InsightFace antelopev2: not bundled."
)

PACKAGES_NOT_IN_POINTSTREAM = (
    "torch==2.5.1 (cu124 wheel, as upstream requests)",
    "xformers (matching that torch)",
    "insightface",
    "upstream StableAnimator `animation` package (not pip-installed here)",
)

BP5_NOTES = """
Quality-flagship comparison for Wave-2 BP5
==========================================
Checked 2026-08-22. Live GPU inference of StableAnimator was not run: leftover
VRAM on the shared RTX 6000 Ada pair was ~11.6 GB with another user's tokengs
jobs holding ~37 GB each, and VAE decode wants ~16 GB. Do not treat a skipped
run as a quality result.

Incumbent — Animate-Anyone (`finetuned_tennis`)
- Wired to ~/Models/AnimateAnyone/profiles/finetuned_tennis.
- Training meta is assets/dataset/pointstream_aa_meta.json: 7 matches, 114
  tracks. The registry line "single tennis match" is stale.
- In-set probe (alcaraz_highlights/scene_035/track_0096, frames 69-77, 256px,
  4 frames). 3 DDIM steps: melted frames, region PSNR 9.65 dB (ALARM: below
  the 12-28 bound; the config default of 3 steps is not an evaluable
  setting). 20 steps: recognizable player, region PSNR 14.04 dB (per-frame
  17.4 / 13.4 / 13.0 / 12.4), inside bounds. Copying the letterboxed first
  crop onto later frames scored 14.6-15.5 dB — the generator lost to that
  trivial baseline on frames 1-3 of this short window.
- Peak VRAM 5.00 GiB (bound 6-14 was pessimistic at 256px). Cold load wall
  267 s; warm 20-step wall 32.5 s (bound 8-90 missed disk cache: first load
  of the 8 GB .pth files dominated).
- Contract fit is clean: appearance crop + rendered pose images.
- Case for keeping it: only temporal engine fine-tuned on this tennis set;
  pretrained 2026 models can still lose on broadcast tennis. That comparison
  is eval-general, not a reason to delete the incumbent before measuring.
- Case for demoting it: old, known face/body distortion, tennis-set-scoped,
  and on this probe it lost a pixel PSNR contest to copying frame 0.

Candidate — StableAnimator
- Adopted as the one modern wrap this wave. Licence: adapter Apache-2.0 on
  the HF card; GitHub code MIT; SVD-XT Stability-AI — cannot ship, same
  class as MOFA. Do not quietly make this the comparison backbone, and do
  not publish it as the flagship until the SVD dependency is licence-cleared
  or replaced.
- Contract mostly fits (ref image + pose images). Gap: no face-crop field on
  ConditioningBundle; ID path needs params.extra['faces'] or re-extraction.
- Packages: not in the pinned pointstream env; extra tooling belongs in a
  separate conda env.

Sparse2Dense (arXiv 2509.23169, DCC 2026)
- Rechecked 2026-08-22: still no public code or weights. The GitHub hit
  stevewongv/Sparse2Dense is a 2022 3D-detection paper, not this work. Do
  not reimplement. Strong backend *if* released, because VVC keyframe + 3D
  keypoints already matches our lattice corner.

MTVCrafter
- Deferred. Apache-2.0, weights on HF, but wants 4D/SMPL motion tokens.
  That is a motion-representation arm for eval-object more than a drop-in
  flagship, and the 7B/17B DiTs will not fit leftover VRAM.

Roles
- Quality flagship vs comparison backbone must stay distinct until BP5 has
  numbers. A newer model is not automatically better on broadcast tennis.
"""


class StableAnimatorGenerator(BaseFrameGenerator):
    """Sequence generator. SVD weights are not loaded by construction."""

    required = (CONDITION_POSE, CONDITION_APPEARANCE)
    licence_notes = LICENCE_NOTES
    svd_block = SVD_LICENSE_BLOCK

    def __init__(
        self,
        width: int = 512,
        height: int = 512,
        steps: int = 25,
        guidance: float = 3.0,
        checkpoint: str | None = None,
        runtime: Any = None,
    ) -> None:
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance = guidance
        self.checkpoint = checkpoint
        self._runtime = runtime
        self.last_run: dict[str, Any] = {}

    def generate_sequence(
        self,
        conditioning: Sequence[ConditioningBundle],
        *,
        seed: int,
        device: Device,
        params: GenerationParams,
    ) -> Sequence[np.ndarray]:
        if not conditioning:
            raise ValueError("generate_sequence needs at least one ConditioningBundle.")
        for bundle in conditioning:
            bundle.require(*self.required)
            bundle.validate_shapes()
        prepared = [self._prepare(bundle, params) for bundle in conditioning]
        faces = params.extra.get("faces") if params.extra else None
        if self._runtime is not None:
            output = self._runtime(
                list(conditioning),
                seed=seed,
                device=device,
                params=params,
                prepared=prepared,
                faces=faces,
            )
            return tuple(as_chw(frame) for frame in output)
        return self._run_runtime(
            tuple(conditioning),
            prepared=prepared,
            faces=faces,
            seed=seed,
            device=device,
            params=params,
        )

    def _generate(
        self,
        conditioning: ConditioningBundle,
        *,
        seed: int,
        device: Device,
        params: GenerationParams,
    ) -> np.ndarray:
        frames = self.generate_sequence(
            (conditioning,), seed=seed, device=device, params=params
        )
        return frames[0]

    def _prepare(
        self, conditioning: ConditioningBundle, params: GenerationParams
    ) -> dict[str, Any]:
        canvas_width, canvas_height = self.canvas_size(params)
        appearance = as_hwc(conditioning.appearance)
        src_h, src_w = appearance.shape[:2]
        box = letterbox_from_bbox(
            conditioning.bbox, src_w, src_h, canvas_width, canvas_height
        )
        pose = as_hwc(conditioning.pose)
        if pose.shape[:2] != (src_h, src_w):
            import cv2

            pose = cv2.resize(pose, (src_w, src_h), interpolation=cv2.INTER_NEAREST)
        return {
            "appearance": letterbox_image(appearance, box),
            "pose": letterbox_image(pose, box),
            "letterbox": box,
            "licence_notes": LICENCE_NOTES,
        }

    def _run_runtime(
        self,
        conditioning: Sequence[ConditioningBundle],
        *,
        prepared: Sequence[dict[str, Any]],
        faces: Any,
        seed: int,
        device: Device,
        params: GenerationParams,
    ) -> Sequence[np.ndarray]:
        del conditioning, prepared, faces, seed, device, params
        raise RuntimeError(
            SVD_LICENSE_BLOCK
            + " Packages that must not be pip-installed into pointstream: "
            + ", ".join(PACKAGES_NOT_IN_POINTSTREAM)
            + f". checkpoint={self.checkpoint!r}."
        )
