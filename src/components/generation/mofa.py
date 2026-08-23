"""MOFA-Video: trajectory-conditioned SVD. Candidate, not a commitment.

Licence (checked 2026-08-21 against https://github.com/MyNiuuu/MOFA-Video):

- MOFA-Video's own code is Apache-2.0 (Tencent), *except* third-party components.
- GitHub still labels the repo licence as "Other" because of that carve-out.
- Inference depends on Stable Video Diffusion XT weights, which are under the
  Stability AI Community License, and a CMP checkpoint from a third path.
- Weights are **not** copied into this repo.

Until that third-party stack is licence-cleared for this project, construction
raises. The spec stays registered so appearance/motion pairing can name
sparse-trajectories as a workable half. DragNUWA, Motion-I2V and Tora are not
registered here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from src.components.generation.base import BaseFrameGenerator
from src.contracts.capabilities import CONDITION_APPEARANCE, CONDITION_MOTION_FIELD
from src.contracts.conditioning import ConditioningBundle, Device, GenerationParams

LICENCE_BLOCK = (
    "mofa-video is registered as a candidate, not integrated. "
    "MOFA-Video code is Apache-2.0 (Tencent) except third-party components; "
    "it depends on Stable Video Diffusion weights under the Stability AI "
    "Community License, which are not bundled and must not be copied into "
    "this repo. Construction is refused until that stack is licence-cleared. "
    "DragNUWA / Motion-I2V / Tora are not registered."
)


class MofaVideoGenerator(BaseFrameGenerator):
    """Trajectory-conditioned sequence generator. Build is refused on purpose."""

    required = (CONDITION_APPEARANCE, CONDITION_MOTION_FIELD)
    licence_block = LICENCE_BLOCK

    def __init__(self, **_kwargs: Any) -> None:
        raise RuntimeError(LICENCE_BLOCK)

    def generate_sequence(
        self,
        conditioning: Sequence[ConditioningBundle],
        *,
        seed: int,
        device: Device,
        params: GenerationParams,
    ) -> Sequence[np.ndarray]:
        del conditioning, seed, device, params
        raise RuntimeError(LICENCE_BLOCK)

    def _generate(
        self,
        conditioning: ConditioningBundle,
        *,
        seed: int,
        device: Device,
        params: GenerationParams,
    ) -> np.ndarray:
        del conditioning, seed, device, params
        raise RuntimeError(LICENCE_BLOCK)
