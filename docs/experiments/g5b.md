# Experiments: G5b

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md).

### 2026-10-10 — G5b: scene-adapted DCVC-UF on VISOR's stretches
- Question (PLAN G5b): with the scene model delivered beforehand and not
  billed, does DCVC-UF fine-tuned on the scene code a held-out stretch of the
  same scene for fewer bits than SVT-AV1 (and than DCVC-UF unadapted) at
  equal LPIPS on the visible background?
- Held-out segment. On each of G1d's 10 stretches (120 s at 10 frames/s;
  P03_10 103 s) it is G5's 240-frame excerpt (24 s), not 96–120 s as PLAN
  first said: VISOR's dense masks lie in 96–120 s on only one stretch
  (P09_106), and the gate needs them. The excerpt holds the stretch's
  dense-mask window (40–48 frames), and G5's SVT-AV1 streams and DCVC-UF
  setup already code exactly these frames. *Scene frames*: the rest of the
  stretch, less 5 s on each side of the excerpt (no near-duplicate of a
  held-out frame is fitted); 910 frames on most stretches. On 4 stretches
  they lie on both sides of the excerpt; that is the deployment's
  assumption (a model of the scene, made from other footage of it), not
  causal coding.
- Protocol. As G5 (`experiments/background/g5.py`): 960×540, every method
  codes the excerpt's *filled* input (foreground inpainted) or, for SVT-AV1,
  the better of frame and filled; quality on V with the foreground pasted
  back; rate = every byte sent for the excerpt over its 24 s. The fine-tuned
  model is not billed; its fp16 size is reported, with the playback after
  which it would be repaid at each operating point (8 × bytes / rate saved
  against SVT-AV1 at equal LPIPS_V).
- **Gate metric**: LPIPS (AlexNet) on V of the dataset tier (the dense
  frames), mean over frames; BD-rate is G5's (PCHIP, Pareto envelopes) on
  q = −100 × LPIPS_V, so G5's 0.5 overlap is 0.005 LPIPS. PSNR_V on the
  same frames, and LPIPS_V on all 240 frames against G1's `sam_text`
  masks, are reported beside it.
- Arms. *Anchor*: SVT-AV1 (G5's streams, decoded again and scored with
  LPIPS). *Reference*: DCVC-UF HT-L off the shelf, QP 9/27/45/63 (G5's
  `dcvc`). *Upper bound*: HT-L fine-tuned on the excerpt itself. *Component*:
  fine-tuned on the scene frames. *Control*: fine-tuned on another
  stretch's scene frames (P26_02 ↔ P06_03), to tell the scene from
  egocentric footage, the mask or the loss. Fine-tuning
  (`experiments/background/g5b_train.py`, in DCVC's tree): HT-L stage 2's
  recipe (cascaded, 33 frames = I + 4 groups of 8, 512×512 patches, batch
  4, AdamW at 1e-5, a random QP per sample with DCVC's λ from 1 to 768, the
  image model frozen), distortion on V only with DCVC's YUV/RGB weighting,
  optionally plus β·LPIPS_V. β is set once from the smoke so that β times
  the monitor set's mean LPIPS_V equals twice its mean masked distortion
  at step 0. Steps are set from the smoke so that training takes at most
  30 minutes on Ada. A fixed monitor set (from the fitted frames only,
  4 QPs) gives the curve.
- Decision rule (`g5b.DECISION`), on the pilot stretches P26_02 (G5's pilot
  stretch) and P06_03 (the best `sam_text` recall, 0.84). An arm *passes*
  if on some pilot stretch, from a converged run (monitor loss down less than
  3% over the last fifth of the steps), it beats SVT-AV1 (BD < 0) and the
  unadapted DCVC-UF (BD < −10%). It *fails* if on both it does neither.
  Otherwise it is *unclear* (more steps, once). Order: (1) the upper bound,
  β = 0 and β > 0. If it fails, the component is not built and G5b stops.
  (2) The component and the control, with the better β. If the component
  passes and the control reaches at least 2/3 of its gain against the
  unadapted model, the gain is not the scene's, and G5b reports an
  egocentric-adapted DCVC-UF instead. (3) The component on the other 8
  stretches: the gate passes if the median stretch's BD-rate against
  SVT-AV1 is below 0.
- Hypothesis: the upper bound passes with large margins (BD −40% or better
  against the unadapted model), because the decoder and the entropy model
  absorb the excerpt's textures. The component keeps a third or less of
  that gain: the excerpt's background is mostly seen elsewhere in the
  stretch, but lighting, clutter and viewpoint differ. Against SVT-AV1 on
  LPIPS the component is near parity: DCVC-UF off the shelf was +41% on
  PSNR_V on P26_02, and DCVC-UF's LPIPS is usually better than its PSNR
  rank. β > 0 helps LPIPS by 10–30% at a small PSNR cost.
- Competing explanation: whatever the component gains comes from adapting
  to egocentric kitchen footage at 960×540, to the filled input or to the
  loss on V, not from knowing the scene. The control measures that. A
  second one: LPIPS gains with β > 0 come from texture that LPIPS rewards
  but that is wrong, which PSNR_V beside it and a visual check expose.
- Side check (PLAN G5b, cheap): G5's warp-only arm (refreshes coded by
  SVT-AV1, oracle DIS flow at 128 B per frame between) with refreshes every
  0.3 s and 1 s on G5's pilot windows (P02_12, P03_120), CRF 40/50/59.
  If either is below 0 BD on PSNR_V or LPIPS_V against SVT-AV1 on either
  window, G1e's refreshed reference reopens as conditioning for G5b;
  otherwise it stays closed. Hypothesis: both stay above +30%, since
  SVT-AV1's inter coding already uses what a refresh carries, and the warp
  error grows with the reference's age (G1e).
- Budget. Rescore ≤ 1 GPU-hour (one job, all 44 clips). Reference ≤ 0.5
  (one job). Smoke ≤ 10 minutes. Pilot ≤ 6 GPU-hours: the upper bound
  (2 stretches × 2 β) and the component and control (2 + 2), each run
  ≤ 45 minutes on Ada. The rest ≤ 6 GPU-hours (8 runs). The warp check ≤
  0.5 (two jobs). Ceiling 14 GPU-hours, stage by stage; a stage that
  fails the rule stops what follows.
- Inputs: as G5 (`pointstream-data/background/g5-2026-10-09/inputs.json`
  `fe56577d…baecf`), the DCVC-UF checkpoints (image `b3b900de…3302`, HT-L
  `934bde4a…b1a6b3`, the latter 482 MB at fp32), and
  G5's SVT-AV1 streams: `20261009T221753Z-64ab816f` `partial.tar`
  (`e8b27b50…245b`, 14 windows), `20261009T230937Z-17f60d9f`
  (`825625d6…9598`), `…230943Z-a6010e7f` (`e317bd7c…1244`),
  `…230949Z-be5deada` (`8523c9b4…6bc82`), `…230954Z-4c28fc16`
  (`ba0b804f…85b87`). Environment `pointstream-20261009T214002Z`
  (`32e68872…4661`): DCVC-UF's training code needs nothing new
  ([resources](../resources.md), DCVC-UF).
- Calibration (before the pilot). Smoke and a 40-step run at the pilot's
  settings, upper bound on P26_02, β = 0 (`20261010T062353Z-1ed3e72b`,
  Ada, code `60ad72e`): HT-L has 120.5 M parameters (241 MB at fp16); the
  fine-tuned checkpoint loads strictly into B2's worker, decodes
  deterministically and matches the encoder's intra frame. 3.28 s per step,
  42.7 GiB peak. At step 0 the monitor set's masked distortion is 0.00264
  and its LPIPS_V 0.2034, so **β = 0.026**; 30 minutes of training is
  **540 steps**. Infrastructure, not evidence. Warp check smokes and runs:
  `20261010T062342Z-5d5103f4` (P02_12), `…062347Z-917cfaaa` (P03_120).
- Anchors and reference (code `60ad72e`). SVT-AV1 rescored with LPIPS:
  `20261010T062332Z-92f06774` (exceeded its 2,700 s after 23 windows,
  kept in its `partial.tar` `4d18c877…a927`), `…071155Z-7cdba7a4` (the
  10 stretches), `…071200Z-882d6d58` (the other 11 windows). DCVC-UF off
  the shelf on the 10 stretches: `20261010T062337Z-78a7cdcc`. Against
  SVT-AV1 it is worse on LPIPS_V than on PSNR_V: median BD +151% on
  LPIPS_V (range +26% to +545%), against about +100% on PSNR_V. The
  hypothesis that DCVC-UF ranks better on LPIPS was wrong.
- Warp check, outcome: **closed**. Against SVT-AV1 on LPIPS_V: 0.3 s
  +52% (P02_12) and +51% (P03_120); 1 s +42% and +54%. On PSNR_V: 0.3 s
  +34% and dominated; at 1 s, unclear (the cheapest points lie below
  SVT-AV1's cheapest rate, and every point in its range lies below it).
  As predicted.
- Upper bound, outcome (540 steps, all converged by the rule; Ada).
  β = 0: P26_02 `20261010T063842Z-29a4262e`, P06_03 `…063853Z-c1d2154d`.
  β = 0.026: P26_02 `…063847Z-3990302d`, P06_03 `…063858Z-93923e53`.
  BD-rate on LPIPS_V (PSNR_V) against SVT-AV1 | against DCVC-UF:
  - β = 0, P26_02: +37% (−1%) | −37% (−33%).
  - β = 0, P06_03: +11% (+11%) | −36% (−31%).
  - β = 0.026, P26_02: unclear (+82%) | dominates (+15%). Its three
    points inside SVT-AV1's rate range all lie far below SVT-AV1's best
    LPIPS_V (0.063 at 31 kbps against 0.132 at 264 kbps); the cheapest
    (8.7 kbps) lies below SVT-AV1's range, which G5's verdict calls unclear.
  - β = 0.026, P06_03: −83% (+72%) | −92% (+7%).
  By the rule the upper bound **passes** (β = 0.026 on P06_03); β = 0
  alone would be unclear (it beats the reference on both stretches,
  SVT-AV1 on neither). Scene adaptation buys a steady third of the rate
  against the unadapted model on both metrics. β > 0 trades PSNR_V for a
  large LPIPS_V gain, as the second competing explanation (texture that
  LPIPS rewards) would also produce; the visual check decides that, from
  decoded frames that the component's jobs now publish. So the component
  runs at both β (beyond the rule's "better β", within the pilot budget),
  and the control at β = 0.026. Pilot GPU time so far: about 2.3 hours.
- Component and control, outcome (540 steps, all converged by the rule;
  Ada; code `fb39288`). Scene frames, β = 0: P26_02
  `20261010T080746Z-23963f61`, P06_03 `…080758Z-032b73ee`; β = 0.026:
  `…080752Z-37856ad4`, `…080803Z-38efc9c3`. Control (β = 0.026, fitted
  on the other pilot stretch's scene frames): P26_02 from P06_03
  `…080809Z-e1acbcd8`, P06_03 from P26_02 `…080814Z-72742e68`. BD-rate on
  LPIPS_V (PSNR_V) against SVT-AV1 | against DCVC-UF:
  - Component β = 0: P26_02 +82% (+16%) | −20% (−21%); P06_03 +44%
    (+36%) | −11% (−11%). Unclear by the rule: it beats the reference on
    both, SVT-AV1 on neither. It keeps about half the upper bound's gain
    (−37%, −36%), more than the third predicted.
  - Component β = 0.026: P26_02 unclear (+87%) | dominates (+20%), every
    point in SVT-AV1's range below SVT-AV1's best LPIPS_V; P06_03 −74%
    (+111%) | −87% (+32%). **Passes** by the rule. Payback of the 241 MB
    model against SVT-AV1 at equal LPIPS_V: 2.8 h (P26_02, 10 kbps), 14 h
    and 5 h (P06_03, 12 and 44 kbps); at higher rates SVT-AV1 never
    reaches its LPIPS_V.
  - Control: P26_02 −85% (+117%) | −90%; P06_03 −41% (+81%) | −69%.
    The scene model against its control: −54% (P26_02) and −65% (P06_03).
    The control's share of the component's gain against the unadapted
    model, in log rate: 75% on P26_02 (over the rule's 2/3) and 52% on
    P06_03 (under it). The rule fixed no aggregation over the pilot
    stretches, so it does not settle this: most of the LPIPS gain comes
    from fine-tuning for LPIPS on egocentric footage (the first competing
    explanation), and knowing the scene halves the rate again on both
    stretches.
  - Visual check (P26_02, frame 20, foreground pasted back; frames in the
    jobs' `published.tar`). At equal or lower rate the β = 0.026 output
    keeps the countertop's grain and the jar's inner structure where
    β = 0 shows a smooth wash; no periodic or grid artifact. The grain is
    synthesized, not the source's (lines and stains move or vanish), which
    is the PSNR_V cost. A perception–distortion trade, not an adversarial
    one; the paper has to say the background texture is generated.
- Report: [G5b Scene Model Pilot](https://claude.ai/artifact/H9yE6iBUGWau6K1HQBLcek)
  (curves, attribution, frames, payback, review answers).
- Pilot GPU time: about 5.6 hours (calibration 0.1, upper bound 4 × 0.55,
  component and control 6 × 0.55), within the 6 allowed. The rest (8
  stretches) waits for the user: the result changes what the rest should
  measure (per-scene against a pooled egocentric model).
