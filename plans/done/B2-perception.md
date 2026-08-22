# B2 — Scene, detection, selection, tracking, pose, segmentation

**Owns exclusively:** `src/components/{scene,detection,selection,tracking,pose,segmentation}/**`
and their tests.
**Implements:** the per-axis protocols in `src/contracts/`, plus
`src/contracts/keypoints.py` for the pose schema.

## What to build

Registry entries replacing substring dispatch. The arrangement being retired
selects backends with `"yolo" in name` (`src/encoder/actor_pipeline.py:74-120`),
which is ordering-dependent and misroutes any name that merely contains a
registered one.

Backends: **YOLO26** (default, and the fallback comparator — ubiquitous with
genuinely fast variants, which matters for the real-time rung), **SAM3**
(`assets/weights/sam3.pt` is on disk; supersedes the SAM2 a reviewer suggested),
and **RF-DETR**.

**Decompose the track-recovery logic.** It currently lives on `Yolo26Detector`
and reaches `YoloEDetector` by inheritance, so a non-YOLO backend cannot reuse
it. Make it a composed policy.

**Subject selection is itself an experiment.** Open-vocabulary detectors accept a
class prompt directly — asking for "tennis player" may be enough with no bespoke
code. But not every detector supports it and it is unknown how good the ones that
do actually are. Build both: an open-vocabulary prompt variant and the ad-hoc
heuristic variant (the current `heuristics.py` logic separating players from ball
kids and crowd). Having both is what makes the comparison possible.

**Scene classification — scoped deliberately.** Its only use is routing point
scenes to the semantic pipeline and interludes to the fallback codec. That is
real and answers a reviewer question, but it does not need more investment than
it has, and for short DAVIS clips it may be irrelevant. Keep it as an optional
lattice row; do not build on it without a reason.

## Traps specific to this stream

**Keep one conda environment.** RF-DETR is not installed. Try the existing
`pointstream` env first; only if it genuinely cannot coexist with the pinned
versions, split into the smallest possible set and document exactly why.

**Weight symlinks go stale silently.** Seven links under `assets/weights/` were
dangling because models moved into a subdirectory, and ultralytics quietly
auto-downloads replacements into the repo root when that happens. **Add a check
that every weight a config names actually resolves** to the behaviour suite —
that is this stream's job.

**The transmitted keypoint schema is not the canonical one.** Canonical is
COCO-WholeBody-133 internally; what goes on the wire is derived from what the
chosen generator consumes. Sending 133 joints to a conditioner reading 18 is
wasted payload, and payload is the ranking currency.

## Done when

- Every backend is reachable by exact name; an unknown name raises with the
  registered set and a close-match suggestion.
- Recovery logic is reusable by a non-YOLO backend.
- Open-vocabulary and heuristic selection are both selectable.
- Every weight named by a shipped config resolves, checked in tests.
- `ruff`, `mypy`, tests pass; import direction clean.

---

## Delivered — 2026-08-22

Landed in `src/components/{scene,detection,selection,tracking,pose,segmentation}/`.
Registered: detectors `yolo` / `sam3` / `rf-detr`, pose `yolo` (COCO-17 stored as
canonical WholeBody-133), selection `heuristic` / `identity` / `prompt`,
segmentation `yolo` / `sam3`, scene `hsv` / `routing`, tracking with recovery.

**Two backends cannot construct, both honestly reported:**

- **SAM3** — `ModuleNotFoundError: torch.nn.attention`. Torch 2.2.2 is too old.
  This blocks `PLAN.md` §7 P1 item 10 (detector comparison including SAM3) and
  the SAM3 segmenter with it. Needs a second conda env, which is a scoped
  decision, not a quick fix.
- **RF-DETR** — needs `transformers>=5.1`, this env pins 4.46.3.

Both are registered and say exactly why they fail, which is the right behaviour;
neither is usable today. YOLO26 loads and runs on GPU.
