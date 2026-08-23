"""The engines this probe drives, and how densely.

SAM3 and RF-DETR are not generators. ``mofa-video`` refuses SVD by design.
``canny-controlnet`` and ``multi-controlnet`` are wired but not in the drive
list — they are not ranking candidates.

Static copy is a permanent arm, not an optional engine. It lives here so
``--engine static-copy`` has a plan, and so every other plan shares the same
offsets. Stream A owns ``cuda:0``; this harness defaults to ``cuda:1``.
"""

from __future__ import annotations

from dataclasses import dataclass

from experiments.probe.clips import DEFAULT_OFFSETS

SEED = 42
CANVAS = 512
DEVICE = "cuda:1"
STATIC_COPY = "static-copy"


@dataclass(frozen=True)
class EnginePlan:
    name: str
    kind: str
    offsets: tuple[int, ...]
    steps: int | None = None
    refuse_at: str | None = None
    notes: str = ""


STATIC_COPY_PLAN = EnginePlan(
    name=STATIC_COPY,
    kind="baseline",
    offsets=DEFAULT_OFFSETS,
    notes="paste the keyframe forward, no model. The floor any generator must beat.",
)

PLANS: tuple[EnginePlan, ...] = (
    STATIC_COPY_PLAN,
    EnginePlan(
        name="pose-controlnet",
        kind="diffusion",
        offsets=DEFAULT_OFFSETS,
        steps=20,
        notes="comparison backbone, keypoints arm",
    ),
    EnginePlan(
        name="seg-controlnet",
        kind="diffusion",
        offsets=DEFAULT_OFFSETS,
        steps=20,
        notes="comparison backbone, mask arm; mask is crop alpha, not a separate segmenter",
    ),
    EnginePlan(
        name="ip-adapter-controlnet",
        kind="diffusion",
        offsets=DEFAULT_OFFSETS,
        steps=20,
        notes="trained as a segmentation ControlNet, not an IP-Adapter (PLAN.md §2.3)",
    ),
    EnginePlan(
        name="trajectory-controlnet",
        kind="diffusion",
        offsets=DEFAULT_OFFSETS,
        steps=20,
        notes="same OpenPose ControlNet as pose; control image is Farneback flow sticks",
    ),
    EnginePlan(
        name="pix2pix",
        kind="one-pass",
        offsets=DEFAULT_OFFSETS,
        notes="speed rung",
    ),
    EnginePlan(
        name="spade4tennis",
        kind="one-pass",
        offsets=DEFAULT_OFFSETS,
        notes="domain-specific control; judge on numbers, not in advance",
    ),
    EnginePlan(
        name="upscale-refine",
        kind="one-pass",
        offsets=DEFAULT_OFFSETS,
        notes="non-generative floor",
    ),
    EnginePlan(
        name="animate-anyone",
        kind="temporal",
        offsets=DEFAULT_OFFSETS,
        steps=20,
        notes=(
            "in-domain only: fine-tune set includes both held-out videos "
            "(PLAN.md §2.5). Option 2."
        ),
    ),
    EnginePlan(
        name="stable-animator",
        kind="refuse-generate",
        offsets=DEFAULT_OFFSETS,
        refuse_at="generate",
        notes="constructs; SVD-XT not bundled, generate is refused by design",
    ),
    EnginePlan(
        name="mofa-video",
        kind="refuse-construct",
        offsets=(),
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
