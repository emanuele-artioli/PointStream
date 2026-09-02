# BP34 — Encode/decode time, the operating point, and whether the title may stay

**Owns:** `experiments/timing/**` (new), `outputs/bp34-timing/**`,
`sections/evaluation.tex` §`subsec:eval-operating`, and the title decision in
`main.tex`. **Does not own** `experiments/tier/**` or `src/runner/**` while
PR #45 is open — drive the runner, do not edit it.

**Read first:** `AGENTS.md` · `PLAN.md` §2.16 and §7 P1 item 12 ·
`sections/evaluation.tex` `HOLE(subsec:eval-operating)` and `NOTE(sec:evaluation)`
item (a) · `main.tex` `NEXT(abstract)`.

**No result dependency.** Run it whenever.

---

## 0. Why this is a P0-shaped item wearing a P1 label

`PLAN.md` files encode-time comparison as P1 item 12, but three things in the
paper's front matter are blocked on it and none of them is optional:

- **The title still says "for Live Video Streaming".** `NEXT(abstract)` records
  that this is the last real-time promise left in the front matter, that it
  survived the 2026-08-21 reframing only because the title was out of scope for
  that edit, and that the resolution is binary: *either a real-time tier lands
  and is measured, or the title changes with the framing.* Nobody has measured
  it, so nobody can change it, so it sits there promising something the abstract
  is explicitly forbidden from promising.
- **`HOLE(subsec:eval-operating)`** wants both operating points reported
  honestly, including a quality tier whose framerate is well below the source.
- **The abstract commits the paper** to reporting encode and decode time *"on the
  same axes rather than in a separate discussion"*. That is a structural promise
  about the evaluation, and today there is no table it could be kept in.

The one number on record is **~0.09 fps encode at 4K**, which appears as a
`NOTE`, has no provenance line, and is a single figure for a pipeline with at
least six stages.

## 1. What to measure

Per stage, per tier config, per resolution, on the same clips the ladder uses so
the numbers can sit beside rate and quality rather than in their own section.

1. **Encode**, broken down by stage — detection, pose, segmentation, tracking,
   plate construction, plate encode, residual encode, packing. A single
   end-to-end figure cannot be acted on; a breakdown says which stage to attack
   and lets the paper say what a real-time variant would have to drop.
2. **Decode**, broken down by warp, composite, generative synthesis (when on),
   residual application.
3. **The anchor's encode and decode** on the same clips and the same machine, at
   the preset the ladder uses. Without it the PointStream figures are
   uninterpretable — and the comparison must state that av1 preset 10 is a speed
   preset, so the anchor is being given its fast configuration and PointStream is
   not being flattered.
4. **The device and its state.** GPUs here are shared. Record the GPU, whether
   anything else was resident, and run each measurement at least three times with
   the spread reported. A timing taken during a co-tenant's job is not a timing.

## 2. Bounds — `outputs/bp34-timing/bounds-before-run.json` before the first run

- **End-to-end encode at 4K lands in [0.02, 0.5] fps** with generation off. The
  0.09 on record is inside that; well above it means the note is stale in the
  good direction and something has been optimised without anyone noticing, which
  is worth knowing. Well below means a stage is pathological.
- **Perception dominates.** Detection + pose + segmentation is expected to be
  **50–85%** of encode wall clock with generation off. If it is under 50%, the
  plate or the codec path is the cost, which changes what a real-time variant
  would look like.
- **Decode is at least 5x faster than encode.** The architecture's asymmetry
  claim depends on it. If decode is within 2x of encode, the client-side story
  in System Design needs re-reading.
- **Generation, when on, costs ~1 s/frame** (`ENGINE-ROSTER.md`: 20 diffusion
  steps, and the ranking means nothing at 4). Anything under 0.2 s/frame means
  the step count is not what the config says.

## 3. The decision this brief must return

Write it as one paragraph, with the numbers in it:

> **Does PointStream have a real-time tier at 4K, or at any resolution?**

- **If yes** — a configuration reaching ≥24 fps encode end to end — name it, put
  it in the operating-point table, and the title stays. Say what it gives up.
- **If no**, which the 0.09 fps figure strongly suggests, then the title changes.
  Propose the replacement in the same pass; it must not promise speed, and
  `GOAL(abstract)` bans substituting a different performance promise in its
  place. Something naming the object-centric decomposition and the ablation
  lattice is the honest shape. Then clear `NEXT(abstract)` and promote
  `NOTE(sec:evaluation)` item (a) from a footnote to a first-class limitation,
  which `NEXT(contributions)` in `sections/introduction.tex` already asks for.

**Do not soften this.** An abstract forbidden from promising speed, under a title
that promises it, is the kind of inconsistency a referee reads as carelessness
about the rest.

## Done when

- A stage-level timing table exists for both arms with n≥3 and the spread, on a
  recorded device, cited by a `CLAIM(subsec:eval-operating)` line.
- `HOLE(subsec:eval-operating)` is cleared by the edit that lands the data.
- The title question is answered in writing and acted on, and `NEXT(abstract)` is
  either cleared or restated with the reason it survives.
