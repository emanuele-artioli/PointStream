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

**The gap survives, and the right repair is stronger than a narrowing.**

Sparse2Dense codes a **key-reference frame with VVC as a texture reference** and
drives it with sparse **3D keypoints**. In this project's vocabulary that is an
*appearance representation* plus a *motion representation* plus a generator —
with detection, selection, tracking, rigid objects, background, residual and
codec fallback all switched off. It is one subject, no background model, no
non-person objects, no corrective channel.

That is **a corner of the lattice this paper defines**, not a competing system.
So the claim to make is not merely "we go beyond faces" and not merely "we do
harder content". It is:

> The generative human-coding literature has converged on one construction —
> code a reference appearance once, send a compact per-frame motion signal,
> synthesise the rest. Each such system corresponds to a single configuration of
> the component lattice this paper defines. PointStream is the framework in
> which that configuration is one cell among many, adding the components those
> systems set aside — a background model, non-person rigid objects, a corrective
> residual, and a fallback to conventional coding — and asking the comparison
> none of them asks: which representation of an object is actually cheapest at a
> given quality.

**Say *corresponds to*, never *is a special case of ours*.** These are
independent systems whose designs happen to land on corners we also define.
Claiming they are instances of our framework would be both wrong and rude, and a
reviewer would notice.

## What to change

### 1. Add the recent generative codecs

`\subsection{Generative coding of general video}` already handles T-GVC
(`wang2025tgvc`) well and honestly, including that it is an unreviewed preprint.
Extend it:

| Work | What it is | How to position |
|---|---|---|
| **Sparse2Dense** (arXiv 2509.23169, **DCC 2026**; Fudan / DAMO Academy / Hupan Lab / CityU HK) | VVC-coded key-reference frame + sparse **3D** keypoints → dense motion → synthesis, with joint vertex prediction. 74.5% BD-rate vs VVC (DISTS), 73.7% (LPIPS), 75.4% (FVD). | The closest prior work to our keypoint arm, and the clearest example of the lattice-corner framing above. **Engage with it directly, not as a list entry.** |
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

Rewrite the `NOTE(sec:intro-gap)` paragraph around the lattice-corner argument
above. Keep the GFVC framing — it is still correct and still useful — and change
what we say is left open: not *beyond faces*, which others have now reached, but
**the framework in which those designs are configurations rather than systems**,
plus the components they set aside and the representation comparison none of
them runs.

### 4. Note the one idea worth borrowing

Sparse2Dense uses **3D** keypoints where our pose axis currently carries 2D
COCO-17 stored as canonical WholeBody-133. That is a candidate arm for
`subsec:eval-object` and it costs a keypoint schema rather than a new model.
Raise it as a `NEXT` marker in the Evaluation section for a later decision — do
**not** add it to the evaluation plan unilaterally.

## Traps specific to this stream

**Do not overclaim in the other direction.** Sparse2Dense's 74.5% is on an
easier problem — one person, no background, no residual. That is a reason our
numbers are not directly comparable, **not** a reason ours are better. Say the
scopes differ; do not imply we win.

**Their design being a lattice corner is an argument about framing, not a claim
of priority.** They published first and their result stands on its own. The
contribution being claimed here is the framework and the components they set
aside, never their construction.

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
- The lattice-corner argument is made explicitly, in the *corresponds to*
  formulation, and every system named under it is one we can defend that reading
  of from its own paper.
- Whether Sparse2Dense has released code or weights is checked once and recorded
  — it changes whether it is only related work or also a candidate backend.
- The animation list reaches 2026.
- The Introduction claims *beyond the isolated subject*, and the claim is one we
  can defend against every work cited.
- Every new `ref.bib` entry verified from source.
- Committed in the paper repo with its markers updated.
