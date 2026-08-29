# B'29 — How cheap can the plate get before the residual stops covering for it?

**Why this exists.** The BP24 ladder measured the plate at **88–91% of
PointStream's payload at every rung of every sweep**
(`plans/BP24-ladder-report.md`). PointStream loses to av1 by +116.8% BD-rate on
the friendliest clip available, and the plate is where the bytes are. Nothing
else in the project is a large enough lever to close that gap.

There are two ways to make the plate cheaper. This brief is the one that needs
**no new code** — compress the still harder — and it should run before the one
that does (stitching a real panorama, `plans/prompts/next-session.md`), because
it is a sweep rather than an implementation and it tells the panorama work what
target to aim at.

## The question, precisely

The background is 99.4% of the pixels and, per BP23, already reconstructs at
39.46 dB while the objects sit at 14.30 dB — a 25 dB gap on 0.57% of the frame.
So the plate is being sent at a fidelity nothing is asking for. The intuition is
that a viewer would not notice a coarser background, and that the residual is
there to catch what they would.

Both halves of that are testable and neither has been tested:

1. **Does the residual absorb a worse plate?** As the plate coarsens, the
   residual's input grows everywhere in the background. If the block gate keeps
   that error the residual grows to match and the total barely moves — the trade
   is a wash. If the gate drops it, the total falls and the quality falls with
   it. **Which of those happens is the experiment.**
2. **Does frame PSNR see it the way a viewer would?** Almost certainly not, and
   that is the trap below.

## Why the existing data cannot answer it

The BP24 payload sweep moved plate quality **and** the residual's rate together,
so the two effects are confounded:

| rung | total | plate | residual | Y-PSNR |
|---|---:|---:|---:|---:|
| jpeg30 / qp55 | 318,077 | 283,483 | 10,285 | 39.21 |
| jpeg50 / qp46 | 390,889 | 345,947 | 20,633 | 41.45 |
| jpeg75 / qp38 | 525,462 | 463,334 | 37,819 | 43.59 |
| jpeg90 / qp28 | 808,573 | 713,320 | 70,944 | 45.39 |
| jpeg98 / qp18 | 1,548,393 | 1,408,247 | 115,837 | 46.55 |

The residual *grew* as the plate improved, which looks backwards for an
absorption story — but the residual's own quantizer went from QP 55 to QP 18
across the same rows, so the table says nothing about absorption either way.

## What to do

1. **Sweep the plate alone.** Hold `residual` completely fixed at the tier's
   settings and sweep `background.jpeg-quality` over roughly
   `{10, 20, 30, 50, 75, 90, 98}`. One line in
   `experiments/tier/ladder.py` — `payload_rung` already takes the two knobs
   separately, so this is a third `--sweep` mode, not a new harness.
   Report, per rung: plate bytes, residual bytes, total, and **whether the
   residual grew to cover the plate's loss**. That last column is the finding.
2. **Add downscaling as the second axis.** `AppearanceConfig`-style downscale
   already exists for appearance; the background sidecar takes
   `background.jpeg-quality` only. Quality-versus-downscale is `PLAN.md` §7 **P2
   item 15**, written before anyone knew the plate was 90% of the rate. It is
   not a P2 item any more.
3. **Score it on more than frame PSNR** — this is the part that decides whether
   the answer means anything. See below.
4. **Re-run the paired ladder at the best rung found** and report the BD-rate
   beside +116.8%.

## The trap, and it is the whole difficulty

**Frame PSNR cannot reward this trade.** The background is 99.4% of the pixels,
so a coarser plate moves frame PSNR almost one-for-one, and the ladder's axis is
frame Y-PSNR. A sweep scored only on that will conclude "compressing the plate
costs quality", which is arithmetically true and answers the wrong question.

The claim being tested is *perceptual* — a viewer tolerates a soft background —
and the project already owns the instruments to test it:

- **Region-scoped scoring.** `QualityReport` already reports object, background
  and frame roles separately (BP23 used it for the 25 dB gap). The object score
  must not move as the plate coarsens; if it does, the plate is leaking into the
  objects and that is a defect, not a trade.
- **VMAF and LPIPS.** Both work (§2.7, §2.16) and both are closer to a viewer
  than PSNR. **Calibrate first**: §2.16 recorded VMAF's ceiling at 97.54 on this
  content and a floor of 0.00 for both severe blur and an unrelated clip, and
  LPIPS's ordering *inverted* at 960x540 while holding at 4K. Anchors do not
  transfer across resolution — re-anchor at the resolution actually used.

**State which metric answers which question before running**, or the result will
be a table where the reader picks the column that suits them.

## Bounds — write to `outputs/bp29-plate/bounds-before-run.json` first

- **The plate at jpeg10 must be far smaller than at jpeg98.** BP24 measured
  283,483 B at q30 and 1,408,247 B at q98, so expect roughly 100–200 KB at q10.
  A plate that does not shrink means `background.jpeg-quality` is not reaching
  the sidecar — check before believing anything else.
- **The total must fall as the plate coarsens.** If it does not, the residual is
  absorbing the loss byte-for-byte and the trade is a wash — a real result, and
  the one that would kill this direction.
- **Frame Y-PSNR will fall.** Expect roughly 1–4 dB from q75 to q10; a fall
  under 0.5 dB means the plate is not what the frame score is reading, and a
  fall over 8 dB means the residual is not covering anything at all.
- **The object-region score must not move.** The objects are pasted crops; the
  plate should not touch them. Movement over ~0.5 dB is an alarm.
- **The best case for the whole direction:** closing +116.8% needs the total to
  roughly halve at unchanged perceptual quality. If the plate can drop to
  ~150 KB with VMAF holding, PointStream's total lands near 200 KB against av1's
  86 KB at 39.45 dB — still losing, but by 2.3x rather than the plate alone
  costing 3.3x av1's whole bitstream. **Note that this is unlikely to be enough
  on its own**, which is the honest expectation to write down first.

## Done when

The plate-quality sweep is run with the residual held fixed, scored on frame
PSNR *and* region-scoped scores *and* at least one calibrated perceptual metric,
and the report says whether the residual absorbs a coarser plate — with the
paired BD-rate at the best rung stated beside BP24's +116.8%.

## Read first

`plans/BP24-ladder-report.md` · `plans/BP24-findings.md` §§6, 13 ·
`PLAN.md` §2.16 (metric calibration, and the two findings that outlive it),
§2.20 · `src/components/background/sidecar.py` ·
`src/runner/stages.py::_panorama_coded_bytes`
