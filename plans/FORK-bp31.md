# FORK — the three papers, written before the number that chooses between them

**Written 2026-09-02, before BP31's N-scene ladder reports.** That is the whole
point of the document.

`AGENTS.md` permits scoping the headline claim to the regime where it holds, and
forbids *"choosing the configuration after seeing the numbers and not saying
so"*. The distinction between those is **when the decision rule was written**. So
the rule is written here, now, with no result in hand: each branch names the
condition that selects it, the paper it produces, and the work items that
activate. When BP31 and `plans/BP32-rate-budget.md` report, the branch is read
off rather than argued for.

**If a later session changes a branch's condition, it must say so and why.**
Editing this file after the numbers arrive is not forbidden — bounds get revised
for reasons — but an unrecorded edit is the failure mode this document exists to
prevent.

---

## The decision variable

**BD-rate of PointStream against an av1 anchor, both arms on the same footage,
same low-delay treatment, reported per video with the spread, at the best
configuration reachable by the axes in `plans/prompts/next-session-bp31.md` plus
span (`plans/BP33-span-amortisation.md`).**

Quoted on **all three quality axes** once `plans/BP35-perceptual-bdrate.md`
lands, because "PointStream wins" that is true on VMAF and false on PSNR is a
scoped result and must be stated as one, not chosen silently.

Current value: **+90.97%**, N=2 scenes, 8 frames per scene, Y-PSNR, one video
(`plans/BP31-findings.md` §9).

---

## Branch A — a regime is found: BD-rate ≤ −5% on ≥6 videos

**Condition.** A named configuration reaches BD-rate **≤ −5%** against the av1
anchor, on **at least six videos** (`presley`'s bar, which BP30's five did not
clear), with the per-video spread reported and the joint/separate anchor control
(`BP31` §5) reading below 1.0.

**Then the paper is:** *an object-centric codec that beats a conventional anchor
in a named regime, plus the ablation lattice that says which of its components
earn their bytes.*

The headline is scoped to the regime in the abstract itself — content type, scene
count, span, rate range, quality axis — because a claim true in a named regime is
a result and one asserted everywhere is usually a mistake.

**What activates:**

1. **`BP41` immediately.** With a winning corner, the lattice stops being a
   catalogue and becomes the explanation of *why* it wins. It is the paper's
   central contribution and it is the thing that makes the win reproducible
   rather than lucky.
2. **`BP36`**, promoted from "reviewer item" to load-bearing: a win found only on
   self-curated tennis will be read as a win found by searching. A UVG sequence
   in the same regime is the answer.
3. **`BP34`**, because a win at 0.09 fps encode needs the cost stated in the same
   table, not in a limitations paragraph.
4. **`BP42`** writes the Conclusion, puts a number in the abstract for the first
   time (which `HOLE(abstract)` says only a real `outputs/` path may authorise),
   and resolves the title.

**The extra checks this branch owes**, because `AGENTS.md`'s asymmetry rule bites
hardest here:

- the anchor got the same N scenes, the same span, jointly encoded;
- quality is measured on `delivered_frames`, not `RunResult.frames`;
- the decode names `-c:v` (a decode that does not re-encodes and caps quality);
- the quality axis is not RGB-PSNR against a 4:2:0 codec;
- the search that found the regime is reported in full, including the
  configurations that lost;
- **the null control**: the same regime with the object stream off, so the win is
  attributed to the decomposition and not to the plate alone.

## Branch B — a boundary is found: no win, but a named crossover

**Condition.** No configuration reaches −5%, **but** the curves cross, or
approach crossing, along an axis that can be named — a rate below which
PointStream wins on a perceptual metric, a scene count above which the trend
projects to crossing, a content class where the gap is a fraction of what it is
elsewhere. Concretely: any configuration under **+20%**, or any axis where the
gap falls monotonically and the extrapolation is defensible.

**Then the paper is:** *a measured account of where an object-centric codec
becomes competitive, and what would have to change for it to win.*

This is a real TOMM paper and it should not be treated as a consolation. It has
a positive contribution (the lattice, the decomposition, the measured levers), a
quantitative boundary, and a mechanism. What it must not do is *imply* the win —
`GOAL(abstract)` already forbids stating an expected outcome, and that constraint
survives into this branch.

**What activates:** the same list as Branch A, with two changes.

- **`BP41` is promoted to the headline**, not the explanation. If the system does
  not beat the anchor, the contribution is the instrument: a platform where every
  component is a configuration choice, and a per-component BD-rate table nobody
  else has produced for this class of system.
- **The boundary needs its own experiment**, not an extrapolation. Whichever axis
  the crossover lies on, sample past the crossing point even if the operating
  points there are impractical. A crossover you have measured is a result; a
  crossover you have projected is a discussion paragraph.

## Branch C — no regime wins on any tried axis

**Condition.** The axes in `plans/prompts/next-session-bp31.md` (content, scene
count, quality axis, rate regime) plus **span** are each tried far enough to say
where the gap is smallest, and the best configuration anywhere is above **+20%**.

**Then stop and talk to the user, immediately.** Both `AGENTS.md` and that prompt
require it, and both require it *early*: this is a finding about the approach, it
changes what the paper argues, and it must not be discovered at submission time.

**Bring to that conversation**, not afterwards:

- the ledger from `plans/BP32-rate-budget.md` — where the bits went against where
  the headroom said they could go, per term;
- the per-axis map of where the gap was smallest and by how much;
- the honest ceiling: given the measured headroom (foreground **22.9% ± 3.0%** of
  av1's rate, background **34–69%** of the background's), what the best possible
  version of this system could achieve, and how far the implementation is from
  it;
- **what a paper could still be.** Three candidates, in order of how much of the
  existing work they keep:
  1. **The platform paper.** The lattice, the component catalogue, the measured
     per-component BD-rates, and the negative result stated plainly. The
     contribution is a way to ask the question, plus the first honest answers.
  2. **The headroom paper.** `PLAN.md` §2.14 is a solid, self-contained, n=8
     measurement: players are ~1% of the pixels and ~23% of the bitrate; a
     panorama saves 34–69% of the background's rate. That is a motivation
     section that stands on its own as a short paper about where the
     compressible structure in broadcast sport actually is.
  3. **The instruments paper.** Two metrics here passed casual inspection while
     measuring nothing usable; a "fraction of the oracle" metric had a floor at
     0.402; a bound fired against a correct result because its units were wrong;
     a share statistic could not detect the fix for the problem it stated. That
     is a real methodological contribution to an area full of papers that do not
     do these checks.
- **and the calendar.** 30 September. Branch C is only survivable if it is
  reached with weeks in hand, which is the entire reason `plans/BP32-rate-budget.md`
  and `plans/BP33-span-amortisation.md` are marked ⭐ and scheduled first.

---

## The second fork, which resolves sooner

`plans/BP32-rate-budget.md` produces a ledger before any of the above. It has its
own two outcomes, and they change what the axes are worth:

- **The four terms explain most of the ~150-point gap.** Then the levers are
  understood, they are ranked, and BP31's remaining axes can be spent in order of
  value. This is the good case and it is the likely one.
- **A large term is unattributed.** Then something in the implementation costs
  far more than the headroom experiment's equivalent, and the next move is a
  bug hunt rather than a sweep. The leading candidates are already named in that
  brief: a headroom plate built non-causally from the frames it was scored on,
  and the 48-versus-8 frame window. **A sweep run before that is resolved is a
  sweep of the wrong parameter.**
