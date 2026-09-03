# B'28 — Does the paste ever lose? Extend the offset ladder

**Cheap, decisive, and it uses a harness that already exists.** One GPU run.
It either finds the first regime in this project where a generative engine beats
a paste, or it closes that direction with evidence.

**Owns:** `scripts/bp25_rescore.py`, `outputs/bp28-offset/**`,
`plans/ENGINE-ROSTER.md` (the offset section only).
**Read first:** `AGENTS.md`, `plans/ENGINE-ROSTER.md`, `plans/done/RESEARCH-HISTORY.md` §2.10 and
§2.17, `plans/done/BP25-ip-adapter-rescore.md`.

**Does not own** `src/runner/**`, `src/pipeline/**`, `src/contracts/lattice.py`,
`config/tier_*.yaml` — `BP24` is live in those. Report, do not edit.

## The observation

`outputs/bp25-ip-adapter/rows.json` carries per-offset rows. Grouped by offset,
over 12 clips:

| offset | paste | ip-adapter ep1 | gap |
|---|---|---|---|
| 1 | 0.239 ± 0.053 | 0.682 ± 0.025 | +0.443 |
| 4 | 0.452 ± 0.060 | 0.664 ± 0.030 | +0.212 |
| 8 | 0.582 ± 0.043 | 0.720 ± 0.025 | +0.138 |

**The paste degrades about ten times faster than the model** — +0.0458 LPIPS per
offset against +0.0049. A linear fit crosses at **offset ≈ 10.4**, just outside
the measured range.

## What to do

Extend the same protocol to **offsets 12, 16, 24** — same 12 clips, same seed 42,
same 20 steps, same object-bbox LPIPS, same `reid`. Include, at minimum, the
`static-copy`, `checkpoint-epoch-1`, and `unrelated-image` arms. Add
`upscale-refine` and `seg-controlnet` if it is cheap to do so: they are the two
best engines on the roster and the crossover question is really about them, not
about IP-Adapter specifically.

Then answer one question in writing: **at what offset, if any, does the best
engine beat the paste, with n and standard error on clip means (n=12), not on
items.**

## Bounds — write to `outputs/bp28-offset/bounds-before-run.json` first

- **The paste keeps degrading.** Expect offset 12 in **[0.60, 0.72]** and offset
  24 in **[0.66, 0.78]**, from the +0.0458/offset slope with saturation toward
  the unrelated level. A paste that stops degrading, or improves, is an alarm —
  check the offsets are real and the keyframe is where you think it is.
- **The model stays roughly flat.** Expect **[0.65, 0.76]** at every offset. A
  model that degrades as fast as the paste means the offsets are not doing what
  this brief assumes.
- **Neither arm should beat 0.4505** (the offset-1 paste) at any long offset. If
  something does, that is an alarm, not a triumph — check for leakage of the
  target frame into the reference.

## The trap this brief exists to avoid

**A crossover is not a victory.** The paste at offset 8 is already heading toward
0.74, which *is* the unrelated-image anchor, and the model sits flat at ~0.70.
If they cross at offset ~10, they cross in a regime where **both arms are about
as good as handing over a photo of a different player.**

So report the crossover offset *and* the absolute LPIPS at which it happens, and
say plainly whether that quality is usable. The useful conclusion may well be
"the paste wins everywhere anyone would operate", and that is a result worth
having, not a failure of the experiment.

## Done when

- Offsets 12/16/24 are measured for at least three arms, with clip-mean n and SE.
- The crossover offset is stated with the absolute LPIPS there, or its absence is.
- `plans/ENGINE-ROSTER.md`'s offset section is updated with the measured rows.
- The report says whether keyframe interval is a lever worth pulling.
