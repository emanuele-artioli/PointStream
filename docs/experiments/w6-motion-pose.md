# Experiments: W6

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md). Context:
[assessment 2026-10-29](../assessment-2026-10-29.md), J's experience
judges ([j.md](j.md)).

### 2026-10-29 — W6: are PointStream's players stable and in the right pose? (rule)
- Question. The visual check (2026-10-29) showed sharp generated players
  but a wrong pose and a missing racket at Al frame 120, and the hybrid
  exists only on every third frame. Are the players stable over time at
  the full frame rate, and is their pose as faithful as the anchor's?
- Step 1, full-rate hybrid. Generate the missing 160 frames per player on
  Al with the frozen generator, seeds and steps of the real-base recipe;
  first regenerate 6 of the saved ticks and require them to match the
  saved crops (max abs diff ≤ 1 level, else report the difference and
  continue with the rerun as the reference for all 240 frames), receiver
  masks for all frames, composite on the paid scale 4 base. Then Med the
  same way if the Al stage fits the budget.
- Step 2, J's judges on all 240 frames, unchanged (`j_players.py`, user's
  readings of 2026-10-22): pose — YOLO11x-pose on the method's frames and
  the source, mean joint distance as a share of the box; time —
  `h3b.flicker` against the anchor, and residual energy against 1.25 × the
  source's. Anchor: plain SVT-AV1 at the hybrid's rate (W1's half
  resolution encode at the bracketing CRFs), and DCVC-UF where W3b gives
  a point at that rate. Racket: reported beside, as the share of frames
  in which the source racket region (from the source mask) overlaps the
  generated player.
- Decision rule (J's, per window): pose passes if the method's error ≤
  max(the anchor's, 1.5 × the coded keypoints' own error against the
  source); time passes if `h3b.flicker` ≤ the anchor's OR residual energy
  ≤ 1.25 × the source's. Report each check separately; a fail narrows the
  claim (e.g. "sharper players, less faithful pose") rather than ending
  it.
- Hypothesis: pose passes against SVT (SVT's players are blurred, so the
  estimator fails on them); time fails on residual energy (J found
  generated players change 1.3–6× as much as the source) but may pass
  against SVT's flicker.
- Competing explanation: the generator's pose errors exceed SVT's blur
  errors, and its frame-to-frame changes exceed SVT's flicker.
- Budget: ≤1.5 GPU-hours (generation ≈820 GPU-s per window on A6000 plus
  masks and judges); every stage ≤45 min, saving each finished frame.
- Jobs: (none yet)
