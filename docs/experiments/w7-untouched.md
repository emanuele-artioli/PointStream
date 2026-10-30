# Experiments: W7

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md). Context:
[W1](w1-roi-pareto.md), [W5](w5-med-pareto.md), the
[fresh-evaluation card](../cards/k6b-fresh-evaluation-plan.md) and PLAN's
foreground-first claim.

### 2026-10-30 — W7: the foreground-first claim on untouched footage (rule)
- Question. W1 (Al) and W5 (Med) found that no SVT or DCVC-UF point reaches
  PointStream's (whole-frame, player) quality pair at equal total rate, and
  that PointStream's player region is far better. Both windows were used
  for development many times. Does the player win hold on footage that no
  experiment has looked at, and on a match the tennis base never trained on?
- What "seen" means here. The tennis base (decoder weights, rank-64 LoRA on
  AnimateAnyone `original`) trained on 36 windows of five matches. Each
  match adapter (sent in the bitstream and billed) trained on other clips
  of its own match. A per-match adapter fitted by the encoder on the same
  match is the codec's normal encode step, not overlap, as long as no
  evaluated frame is in its fit. The base must not have seen the evaluated
  footage for the clean tier.
- Tier 1, seen matches, untouched clips. Alcaraz–Hurkacz and
  Medvedev–Struff clips with at least 240 native frames that are outside
  every old manifest window, every step-5 fit clip and the twelve
  2026-10-25 confirmation windows (k6b-players.md, about 3 and 7 clips).
  Three windows per match, chosen mechanically before any quality is
  computed: every admissible Al clip in ID order up to three; Med clips by a
  seeded draw (seed 20261030) from the admissible list; in each clip the
  first 240-frame interval the frozen registration rule of the 2026-10-25
  windows admits. Clips that fail registration are listed, not replaced by
  hand. Existing Al and Med adapters and reference banks.
- Tier 2, new match. `nadal_kyrgios` (archive, 2,310 s), after a lineage
  check that no base, adapter or reference input contains its frames (W8
  shares this check). Adaptation split: the first half of the broadcast
  (adapter fit with the frozen step-5 recipe, reference bank from the same
  half, all billed). Evaluation: three windows from the second half by the
  same mechanical rule, seed 20261030. Its court marks were used in K8/K8b
  design; registration exposure is disclosed, generator exposure is none.
- Pipeline: the frozen W5 recipe (scale-4 paid SVT base, KPS2 keypoints,
  4-bit adapter, 12-image reference bank, receiver pose-prompted SAM masks,
  generated players on every third frame), unless W6's full-rate generation
  has passed its own rule before W7's first GPU job; the choice is recorded
  here before that job, never after seeing W7 quality.
- Anchors per window: plain SVT-AV1 at half and quarter resolution, SVT
  half with +20 QP outside R (the strongest player policy in W1/W5), and
  DCVC-UF, each with points chosen on rate alone to bracket the hybrid's
  rate. Judges: LPIPS and DISTS (ImageNet networks; no tennis frames seen),
  whole frame and fixed player region R, same controls as W1.
- Decision rule (primary, per tier). Per window, the player gap
  Δ = (best anchor's player-region score at the hybrid's rate) − (hybrid's),
  log-rate interpolation between bracketing points only. The player win
  holds on a tier if the mean Δ over its windows has a 95% t-interval above
  zero on LPIPS and on DISTS, with windows of one match reported as a
  cluster beside the pooled interval. Secondary, reported for every window:
  W1's dominance outcome (A or B) per anchor family and the whole-frame
  difference at equal rate.
- Narrowing, written before results. Tier 1 holds, Tier 2 fails: the claim
  narrows to matches the decoder's base has seen other footage of (a
  tournament- or league-specific decoder); next test: per-match adaptation
  strength on the new match (adapter rank, fit length) within the billed
  rate. Both fail: the claim narrows to the development windows and the
  paper reports a negative generalization result; next test: which stage
  fails (registration, pose, generator) from the per-window breakdown.
  Tier 1 fails on one match only: report the match, narrow to the other,
  next test: what distinguishes them (player size, motion, registration).
- Hypothesis: the player gap holds on both tiers; it shrinks on Tier 2
  because the base never saw those players or that court.
- Competing explanation: the development windows are easy cases (large,
  well-lit players, stable registration) chosen and tuned over many runs,
  and the gap collapses on new windows.
- Budget. CPU stage (inventory, lineage, selection, registration, base and
  SVT encodes): CPU only. GPU stage, staged: one Tier 1 window end to end
  first (≤1 GPU-hour, also the smoke), then the other five (≤4 GPU-hours),
  then Tier 2 including the adapter fit (≤3 GPU-hours). Every stage ≤45 min,
  checkpointed per window.
- Jobs: (none yet)
