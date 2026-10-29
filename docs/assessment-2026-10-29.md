# Where PointStream stands against its baselines (2026-10-29)

A review of the work done between 2026-10-25 and 2026-10-28 (mostly Codex
sessions), the main obstacles to beating the baselines, and the directions
started on 2026-10-29 (PLAN item W).

## Short version

No configuration has beaten SVT-AV1 on whole-frame LPIPS and DISTS once every
bit is paid for, in either domain. The tennis studies of the last four days
all hit the same wall from different sides: at the tested operating point
(about 23 kbps for 1080p, players a few percent of the frame) whole-frame
metrics are decided by the background, and SVT spends its bits on the
background more efficiently than PointStream spends them on the players.
Perfect players would not change that. The foreground itself wins by a wide
margin at every measured rate. The paper's claim therefore depends on a
comparison that credits the foreground without ignoring the background
(W1), on a side stream about five times cheaper (W2), and on a working
neural baseline (W3), which the paper needs whatever the outcome.

## The work of the last four days

What was done well:

- Every result is reproducible and audited: job IDs, code revisions, input
  checksums, independent CPU recomputation of the scores, controls that
  reproduce earlier numbers exactly. Negative results are reported as such.
- The CPU side-rate feasibility analysis
  ([report](k6b-side-feasibility-report-2026-10-28.md)) is the most useful
  result of the period: it turns the failures into one number, the side rate
  the players may cost before the whole frame loses.

What cost more than it returned:

- **Order of studies.** The feasibility analysis took 0.03 s of CPU and
  explains in advance why the ideal-player, clean-background and real-base
  GPU studies had to fail. Running it first would have saved those studies.
- **Drift away from the thesis.** The single-stream temporal filter and the
  persistent block prediction are generic codec post-processing. They do not
  use the foreground/background split, so even a win would not support the
  paper. Both lost.
- **Process overhead.** Most commits record ledgers, receipts, publication
  recovery and budget arithmetic for one-GPU-hour studies. The documents
  became hard to read (numbers without spaces, unexplained codes such as
  `Al031@129`), against AGENTS.md's plain-language rule.
- **The paper's prerequisites did not move.** The neural baseline (DCVC)
  still fails on decode determinism, the fair region-of-interest SVT
  comparison stopped on a cost projection, and there is no untouched test
  footage.

## Main problems with respect to winning

Window names: `Al` is Alcaraz–Hurkacz, shot 031 from frame 129; `Med` is
Medvedev–Struff, shot 014 from frame 194. Both are 240-frame development
windows used many times; neither is untouched footage.

1. **Whole-frame metrics cannot see the players at this rate (decisive).**
   On Al at 22.74 kbps the SVT envelope scores LPIPS 0.234. PointStream's
   quarter-resolution SVT base alone, at 17.13 kbps, scores 0.283, which is
   already on the SVT envelope. Pasting the *source* pixels into the player
   region improves it only to 0.271; SVT buys that 0.012 with about 1.3 kbps.
   The actual player stream costs 5.6 kbps. Because the base is SVT itself,
   PointStream's background can never be better than the anchor's, so the
   whole frame is a losing trade by construction
   ([ideal-player report](k6b-ideal-player-full-report-2026-10-28.md)).
2. **The side stream is about five times too expensive.** Keypoints cost
   3.53 kbps for two players; the per-match adapter about 1.65 kbps over
   two hours; the robust allowance for the actual overlay is 0.67 kbps.
   Lossless re-packing of the keypoints makes them larger; changing their
   representation (prediction, quantization, frame rate) is untested.
3. **The foreground win is large but unclaimed.** Inside the fixed player
   region PointStream scores LPIPS 0.346 against SVT's 0.613 at the same total
   rate, and wins at every rate measured (15.8–33.7 kbps). No fair baseline
   has been run that also moves bits to the players: SVT with a region-of-
   interest quantizer map at full or half resolution. The one ROI test
   (quarter resolution, background degraded) never completed.
4. **The neural baseline does not run reliably.** DCVC streams decode
   differently on different GPUs; only a 24-frame reproduction on gpu3
   passed. Without it the paper has one baseline of the two required.
5. **Granted inputs remain.** The player matte comes from the source and is
   unbilled; receiver-side pose-prompted SAM masks (4/4 in a pilot, no bits)
   are the candidate replacement. The 49 MB tennis base model is unbilled,
   which is acceptable only as part of the decoder, trained on other
   matches, as neural codecs ship their weights.
6. **Egocentric:** with the mask free, DISTS won by 15% against the
   venue-adapted DCVC; billing the mask removed every win. Masks estimated at
   the receiver and scored directly (without the exact correction stream)
   were never tested.

## Directions (PLAN item W)

Ranked by value to the paper. Decision rules, hypotheses and budgets are in
each item's file.

| Item | Question | Cost | File |
|---|---|---|---|
| W1 | Does PointStream reach a (whole-frame, foreground) quality pair that no SVT bit allocation reaches at the same rate? | CPU encodes, ≤1 GPU-h of scoring | [w1](experiments/w1-roi-pareto.md) |
| W2 | Can keypoints be coded at ≤0.5 kbps for two players without moving them more than the estimator's own noise? | CPU only | [w2](experiments/w2-pose-rate.md) |
| W3 | A deterministic neural baseline: encode and decode in one job, same GPU, Al and Med, rates bracketing 15–60 kbps | ≤2 GPU-h | [w3](experiments/w3-neural-baseline.md) |
| W4 | Egocentric: do zero-bit receiver masks keep the DISTS win? | ≤1.5 GPU-h | [w4](experiments/w4-ego-receiver-masks.md) |

Not started, for the user to decide: whether the paper's headline should be
a two-dimensional claim (foreground quality at equal rate and equal or
bounded whole-frame loss) rather than whole-frame BD-rate alone. W1 provides
the evidence for that decision; it does not make it.
