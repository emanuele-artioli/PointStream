"""The engines this probe drives, and how densely.

SAM3 and RF-DETR are not generators. ``mofa-video`` refuses SVD by design.
``canny-controlnet`` and ``multi-controlnet`` are wired but not in the BP5
drive list — they are not ranking candidates for §6.2.
"""

from __future__ import annotations

from dataclasses import dataclass

from experiments.probe.clips import DIFFUSION_FRAME_INDEX, ONE_PASS_FRAME_INDICES

SEED = 42
CANVAS = 512
DEVICE = "cuda:0"


@dataclass(frozen=True)
class EnginePlan:
    name: str
    kind: str
    frame_indices: tuple[int, ...]
    steps: int | None = None
    refuse_at: str | None = None
    notes: str = ""


PLANS: tuple[EnginePlan, ...] = (
    EnginePlan(
        name="pose-controlnet",
        kind="diffusion",
        frame_indices=(DIFFUSION_FRAME_INDEX,),
        steps=20,
        notes="comparison backbone, keypoints arm",
    ),
    EnginePlan(
        name="seg-controlnet",
        kind="diffusion",
        frame_indices=(DIFFUSION_FRAME_INDEX,),
        steps=20,
        notes="comparison backbone, mask arm; mask is crop alpha, not a separate segmenter",
    ),
    EnginePlan(
        name="ip-adapter-controlnet",
        kind="diffusion",
        frame_indices=(DIFFUSION_FRAME_INDEX,),
        steps=20,
        notes="txt2img floor ~11 dB is known; not a path bug unless identity",
    ),
    EnginePlan(
        name="trajectory-controlnet",
        kind="diffusion",
        frame_indices=(DIFFUSION_FRAME_INDEX,),
        steps=20,
        notes="same OpenPose ControlNet as pose; control image is Farneback flow sticks",
    ),
    EnginePlan(
        name="pix2pix",
        kind="one-pass",
        frame_indices=ONE_PASS_FRAME_INDICES,
        notes="speed rung; extra frames because a forward pass is cheap",
    ),
    EnginePlan(
        name="spade4tennis",
        kind="one-pass",
        frame_indices=ONE_PASS_FRAME_INDICES,
        notes="domain-specific control; judge on numbers, not in advance",
    ),
    EnginePlan(
        name="upscale-refine",
        kind="one-pass",
        frame_indices=ONE_PASS_FRAME_INDICES,
        notes="non-generative floor",
    ),
    EnginePlan(
        name="animate-anyone",
        kind="temporal",
        frame_indices=(DIFFUSION_FRAME_INDEX,),
        steps=20,
        notes=(
            "in-domain only: fine-tune set includes both held-out videos "
            "(PLAN.md §2.5). Option 2."
        ),
    ),
    EnginePlan(
        name="stable-animator",
        kind="refuse-generate",
        frame_indices=(DIFFUSION_FRAME_INDEX,),
        refuse_at="generate",
        notes="constructs; SVD-XT not bundled, generate is refused by design",
    ),
    EnginePlan(
        name="mofa-video",
        kind="refuse-construct",
        frame_indices=(),
        refuse_at="construct",
        notes="SVD licence block; do not reimplement",
    ),
)


def plan_for(name: str) -> EnginePlan:
    for plan in PLANS:
        if plan.name == name:
            return plan
    known = ", ".join(item.name for item in PLANS)
    raise KeyError(f"unknown probe engine {name!r}. Driven engines: {known}.")
