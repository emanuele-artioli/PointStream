# B′6 — Update Related Work, and repair the novelty claim

**Wave 1.** Touches **only the paper repo** (`67a9ea6275d3d9785ce57026/`), so it
has zero file contention with any code stream and can run fully in parallel.

**Owns exclusively:** `sections/related_work.tex`, `sections/introduction.tex`,
`appendices/related_work_extended.tex`, `ref.bib`.
**Read first:** that repo's own `AGENTS.md` — it is a **separate git repo** with
its own rules and its own commits. Then `PLAN.md` §6.3.

## Why this is urgent rather than cosmetic

A survey on 2026-08-22 found that the field moved while this paper was being
written, and one of the new entries **directly threatens the novelty framing**.

The Introduction currently argues: generative face video coding works because
faces are cooperative, its own survey names *generalisation beyond faces, to
human bodies and natural scenes* as the principal open problem, and **"that is
the gap this work addresses."**

**Sparse2Dense** (arXiv 2509.23169) is generative human-*body* video coding
driven by sparse 3D keypoints, reporting **74.5% BD-rate reduction against VVC**
on DISTS, 73.7% on LPIPS, 75.4% on FVD. A reviewer who knows it will read our
"beyond faces" claim as overclaiming, because that specific boundary has now been
crossed by someone else.

**The gap survives, but it has to be narrowed to the one we actually fill.**
Sparse2Dense is single-subject and human-centric: 3D keypoints for one person,
no background model, no multiple independently moving objects, no corrective
residual, no fallback to a conventional codec. The honest and still-defensible
claim is not *beyond faces* — it is **beyond the isolated subject**: full scenes
with several objects, a background that must itself be coded, a corrective
channel for generative error, and a graceful handoff when the scene does not
suit the semantic path. Plus the representation *comparison* (`eval-object`)
that nobody, including Sparse2Dense, has run.

## What to change

### 1. Add the competing generative codecs

`\subsection{Generative coding of general video}` already handles T-GVC
(`wang2025tgvc`) well and honestly, including that it is an unreviewed preprint.
Extend it:

| Work | What it is | How to position |
|---|---|---|
| **Sparse2Dense** (2509.23169) | Sparse 3D keypoints → dense motion, plus vertex prediction. 74.5% BD-rate vs VVC. | The closest prior work to our keypoint arm. Single subject, no background, no residual. **This one must be engaged with directly, not merely listed.** |
| **GVC-RT** (2608.04891) | Real-time generative compression at ultra-low bitrate | Bears on `eval-operating`, where our measured speed is poor. Cite honestly. |
| **ReGenVC** (2607.28144) | End-to-end real-time talking-head codec, ~26 kB per 77 frames | Talking-head scope; a scale reference for what a semantic channel costs |

### 2. Refresh the pose-conditioned animation list

`\subsection{Trajectory- and flow-conditioned video animation}` cites
AnimateAnyone, MagicAnimate, Champ, UniAnimate and MimicMotion — all 2024. Add
the 2025–26 generation: **StableAnimator**, **MTVCrafter**, **DisPose**,
**Animate-X**. The existing sentence that none of these treats the size of the
driving signal as a cost to be budgeted **stays true and stays** — it is the
hinge of our positioning, and it survives the newer work.

### 3. Repair the Introduction's gap paragraph

Rewrite the `NOTE(sec:intro-gap)` paragraph so the claimed gap is *beyond the
isolated subject*, not *beyond faces*. Keep the GFVC framing, which is still
correct and useful; change what we say is left open.

## Traps specific to this stream

**Do not overclaim in the other direction.** Sparse2Dense's 74.5% is on an
easier problem — one person, no background, no residual. That is a reason our
numbers are not directly comparable, **not** a reason ours are better. Say the
scopes differ; do not imply we win.

**Their number bounds ours.** Record in the paper that a PointStream BD-rate far
beyond 74.5% on harder content is an alarm to be investigated, not a triumph to
be reported. This is the `bound before believing` rule reaching the manuscript.

**Verify every citation from the source.** These came from a web survey. Confirm
each paper's venue, year and headline number from its own PDF or listing before
it enters `ref.bib`. A wrong number in Related Work is worse than an omission.

**Marker discipline.** This repo uses `STATUS`/`GOAL`/`HOLE`/`NOTE`/`NEXT`/`CLAIM`
comments. Update them with the prose — an edit that closes a `HOLE` deletes it in
the same edit. Do not touch the Evaluation section's markers; that is not this
stream's work.

**Commit in the paper repo.** It is a separate repo. Changes there do not appear
in the main repo's `git status`.

## Done when

- Sparse2Dense, GVC-RT and ReGenVC are cited and positioned, with Sparse2Dense
  engaged directly rather than listed.
- The animation list reaches 2026.
- The Introduction claims *beyond the isolated subject*, and the claim is one we
  can defend against every work cited.
- Every new `ref.bib` entry verified from source.
- Committed in the paper repo with its markers updated.
