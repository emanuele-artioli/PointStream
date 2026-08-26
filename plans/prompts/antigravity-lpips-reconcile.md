# Prompt for Antigravity — reconcile the paper's LPIPS calibration

Paste everything below the line. Paper repo only; small, self-contained.

---

You are fixing a **contradiction between the PointStream paper and the
PointStream repository's own test invariants**. Both are in the project; they
disagree; one of them will be wrong in print.

**You own exclusively:** the paper repo `67a9ea6275d3d9785ce57026/`. It is a
**separate git repo with its own `AGENTS.md` and its own commits** — read those
rules and commit there. **Touch no code file** in the PointStream repo; two code
streams (BP24, BP25) are live right now.

**Read first:** the paper repo's `AGENTS.md`, then
`/home/itec/emanuele/pointstream/plans/wave5-report.md` §3, which is where this
was found.

## The contradiction

`sections/evaluation.tex:255` currently states that LPIPS calibrates to
**0.000 / 0.250 / 0.430 / 0.645** across identical, mild-noise, heavy-blur and
unrelated-donor images at 4K.

The repository's invariant, landed by BP27 in
`tests/invariants/test_metric_calibration.py:243-245`, pins the 4K anchors at
**0.000 / 0.0171 / 0.2982 / 0.5493**.

The unrelated anchor differs by ~0.10 and heavy blur by ~0.13. These are not
rounding.

## What to do

1. **Work out which anchor set each number came from.** They look like two
   different measurements — most likely the earlier probe-set calibration
   (referenced in commented-out prose at `sections/evaluation.tex:28, 47, 73`)
   versus BP23's tier anchors on the 4K tennis clip. **Do not assume; establish
   it**, and if you cannot establish it from the repo and its outputs, say so
   plainly rather than picking one.
2. **State the provenance next to every anchor number that stays in the paper** —
   which clip set, which resolution, which date. The paper currently presents its
   figures as *the* 4K calibration with no source, which is what let the
   contradiction hide.
3. **Add the half of the finding the paper is missing.** The text says LPIPS
   "holds reliably at 4K". The more consequential result is that **its ordering
   inverts at 960×540** — severe blur scored 0.613 against unrelated 0.522 — so
   **calibration anchors do not transfer across resolution.** That is what would
   let a future cross-resolution comparison go quietly wrong, and it belongs in
   the methods text, not just in a test file.
4. **Check whether any LPIPS number elsewhere in the paper depends on the anchor
   set you are correcting** — `sections/evaluation.tex:169` quotes 0.570 against
   "unrelated donor 0.645" and "heavy blur 0.430". If those anchors change, that
   sentence's comparison changes with them.

## Rules

- **Every claim must match real measured evidence.** Cite run paths; never paste
  `outputs/` contents into the paper.
- **Quote the instrument's range with the number.** "0.570" means nothing on its
  own; "0.570, where an unrelated donor scores X" means something — and X must be
  the X from the same anchor set.
- This matters more than a normal consistency fix: **two metrics in this project
  were broken until 2026-08-23 and every perceptual ranking before that date is
  void.** A paper that quotes an unsourced calibration is one reviewer question
  away from that history.
- Follow the marker convention (`STATUS`/`GOAL`/`HOLE`/`NOTE`/`NEXT`/`CLAIM`) and
  update the reviewer checklist if this closes an item.

Report: which anchor set each figure came from, what you changed, and whether any
other number in the paper moved as a consequence.
