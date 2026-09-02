# PointStream handoff — 2 September 2026

Target: **submit a defensible ACM TOMM manuscript by 30 September 2026. The date
is hard because the author's contract ends then.**

## Read in this order

1. `AGENTS.md`
2. `plans/ROADMAP.md`
3. `plans/TERMINOLOGY.md`
4. one assigned brief only
5. `PLAN.md` only when historical evidence is needed

Every dispatched task follows `plans/SESSION-REPORT.md`.

## Current state

- The contracts, components, end-to-end runner and BP31 multi-scene experiment
  are merged on `main`; PR #45 passed tests, lint and typing.
- The best current system result is **+90.97% BD-rate against AV1** on one tennis
  video, two scenes and eight frames per scene. PointStream is about 20× the
  reference codec's end-to-end wall time in that run.
- Increasing scene length from 8 to 16 frames improved the single-point
  PointStream/AV1 byte ratio from 2.17× to 2.01×. This confirms amortization but
  does not establish the long-run slope.
- At 24 frames, independently built scene panoramas acquire different
  dimensions. Predictive background-sequence coding requires equal dimensions.
  The next implementation is an offline canonical background canvas per
  compatible camera/background context.
- The current ~43 dB tests are high quality. The first winning-regime search
  moves to ultra-low bitrate and long eligible scenes.
- No reconstruction model beats the reference-image paste control. Generation
  training is parked until the background and lean non-generative payload can
  win on rate--quality.
- The second domain, learned-codec baseline and speed optimization begin only
  after the first-domain rate--quality result is confirmed.

## Optimization order

1. Beat AV1 and VVC on size at matched quality in a named tennis regime.
2. Confirm on at least six independent videos/matches.
3. Explain it with a core component ablation matrix.
4. Add DCVC-RT and one independent domain.
5. Optimize speed on the frozen winning configuration.

Time is measured and shown from step 1, but it is not a gate until step 5. A
slow win must be framed as offline or compute-intensive, never live.

## Immediate work

The first implementation wave is:

- **M1:** extend AV1/VVC to their lowest useful rate, fix metric-direction
  typing, and calibrate the primary VMAF comparison;
- **B1:** implement canonical background canvases and context resets;
- **D1:** extract and validate 2/4/8/16-second eligible tennis scenes;
- **M2:** produce an exact byte ledger and fit per-additional-frame cost from at
  least three durations;
- **P1:** restore reproducible manuscript rendering and keep a live page budget.

Then run E1, the low-rate × duration search. See `plans/ROADMAP.md` for dates,
dependencies, harness assignment and the required report from every session.

## Standing safeguards

- Write two-sided bounds and null controls before reading a result.
- Every result reports rate, all declared quality axes, encode time and decode
  time, even while only rate--quality gates the search.
- The same source frames, resolution, frame rate and colour convention go to
  PointStream, AV1 and VVC.
- Report every searched configuration. A scoped win found by search is valid;
  presenting it as predicted is not.
- YouTube-derived source footage is not redistributed. The paper and artifact
  must not promise a releasable dataset without a rights review.
- The evidence freezes on 20 September.
