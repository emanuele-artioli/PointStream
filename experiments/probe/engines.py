"""The engines this probe drives, how densely, and which of them are temporal.

SAM3 and RF-DETR are not generators. ``mofa-video`` refuses SVD by design.
``canny-controlnet`` and ``multi-controlnet`` are wired but not in the drive
list — they are not ranking candidates.

**Two baselines are permanent arms, not optional engines.**

* ``static-copy`` pastes this clip's own keyframe forward. It is the floor, and
  it is *real pixels in the wrong pose*, which MSE structurally favours — so it
  bounds PSNR from below without being a pass/fail gate (``PLAN.md`` §2.4).
* ``unrelated-image`` pastes *another* clip's keyframe. It is the null control:
  the score a method gets for showing the wrong person. Every ranking is read
  against it in the same run, because every wrong conclusion in this project was
  a pleasing number reported before its control.

**Clip mode.** A temporal engine is driven once per clip through
``generate_sequence`` over a contiguous run of frames. Animate-Anyone was
evaluated frame-by-frame for three rounds despite carrying a motion module
(``PLAN.md`` §2.7); ``sequence=True`` is what says a plan must not be driven that
way, and the runner refuses rather than silently falling back.

Stream A owns ``cuda:0``; this harness defaults to ``cuda:1``.
"""

from __future__ import annotations

from dataclasses import dataclass

from experiments.probe.clips import CLIP_MODE_OFFSETS

SEED = 42
CANVAS = 512
DEVICE = "cuda:1"
STATIC_COPY = "static-copy"
UNRELATED_IMAGE = "unrelated-image"
BASELINES = (STATIC_COPY, UNRELATED_IMAGE)


@dataclass(frozen=True)
class EnginePlan:
    name: str
    kind: str
    offsets: tuple[int, ...]
    steps: int | None = None
    refuse_at: str | None = None
    sequence: bool = False
    notes: str = ""


STATIC_COPY_PLAN = EnginePlan(
    name=STATIC_COPY,
    kind="baseline",
    offsets=CLIP_MODE_OFFSETS,
    notes="paste this clip's keyframe forward, no model. The published floor.",
)

UNRELATED_IMAGE_PLAN = EnginePlan(
    name=UNRELATED_IMAGE,
    kind="baseline",
    offsets=CLIP_MODE_OFFSETS,
    notes=(
        "paste another clip's keyframe, no model. The null control: what a "
        "score looks like when the appearance is the wrong person."
    ),
)

PLANS: tuple[EnginePlan, ...] = (
    STATIC_COPY_PLAN,
    UNRELATED_IMAGE_PLAN,
    EnginePlan(
        name="pose-controlnet",
        kind="diffusion",
        offsets=CLIP_MODE_OFFSETS,
        steps=20,
        notes="comparison backbone, keypoints arm",
    ),
    EnginePlan(
        name="seg-controlnet",
        kind="diffusion",
        offsets=CLIP_MODE_OFFSETS,
        steps=20,
        notes="comparison backbone, mask arm; mask is crop alpha, not a separate segmenter",
    ),
    EnginePlan(
        name="ip-adapter-controlnet",
        kind="diffusion",
        offsets=CLIP_MODE_OFFSETS,
        steps=20,
        notes="trained as a segmentation ControlNet, not an IP-Adapter (PLAN.md §2.3)",
    ),
    EnginePlan(
        name="trajectory-controlnet",
        kind="diffusion",
        offsets=CLIP_MODE_OFFSETS,
        steps=20,
        notes="same OpenPose ControlNet as pose; control image is Farneback flow sticks",
    ),
    EnginePlan(
        name="pix2pix",
        kind="one-pass",
        offsets=CLIP_MODE_OFFSETS,
        notes="speed rung",
    ),
    EnginePlan(
        name="spade4tennis",
        kind="one-pass",
        offsets=CLIP_MODE_OFFSETS,
        notes="domain-specific control; judge on numbers, not in advance",
    ),
    EnginePlan(
        name="upscale-refine",
        kind="one-pass",
        offsets=CLIP_MODE_OFFSETS,
        notes="non-generative floor",
    ),
    EnginePlan(
        name="animate-anyone",
        kind="temporal",
        offsets=CLIP_MODE_OFFSETS,
        steps=20,
        sequence=True,
        notes=(
            "ReferenceNet + pose guider + motion module. Clip mode is mandatory: "
            "the frame-by-frame path cost this project three rounds of wrong "
            "verdicts (PLAN.md §2.7). In-domain only — the fine-tune set includes "
            "both held-out videos (§2.8)."
        ),
    ),
    EnginePlan(
        name="stable-animator",
        kind="refuse-generate",
        offsets=CLIP_MODE_OFFSETS,
        refuse_at="generate",
        sequence=True,
        notes="constructs; SVD-XT not bundled, generate is refused by design",
    ),
    EnginePlan(
        name="mofa-video",
        kind="refuse-construct",
        offsets=(),
        refuse_at="construct",
        sequence=True,
        notes="SVD licence block; do not reimplement",
    ),
)


def plan_for(name: str) -> EnginePlan:
    for plan in PLANS:
        if plan.name == name:
            return plan
    known = ", ".join(item.name for item in PLANS)
    raise KeyError(f"unknown probe engine {name!r}. Driven engines: {known}.")
