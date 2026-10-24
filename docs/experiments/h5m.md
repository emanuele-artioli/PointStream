# Experiments: H5m

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md).

### 2026-10-17 — H5m: coding F (the mask PointStream's decoder needs)
- Question (after H5's pilot, user 2026-10-17): H5's gain holds only with
  F granted; sent losslessly at 1080p, F and its class map cost 0.5–2.9
  Mbps per window and flip every comparison. Can F be coded cheaply
  enough that PointStream still beats B2's anchors with every byte,
  F's included, counted?
- Candidate (cheapest first): the class map (0 background, 1 hand, 2
  object) at 1/s of the frame size (s = 2, 4, 8, 16; F̂ where at least
  half of a cell's area is F, hand where hand pixels outnumber object
  pixels there), its two bit planes XOR-ed with the previous frame and
  compressed with lzma, losslessly; the client upsamples F bilinearly
  and thresholds at 0.5, and gives each pixel of F̂ the class of the
  nearest low-resolution F cell (`src/pointstream/mask.py`). F̂ replaces
  F in the compositor (the ring blend, G's slot); the bytes go into the
  container's MASK segment, so the rate counts them.
- Method (`h5 score --mask-scale s`): the published foreground and
  background decodes of H5's pilot re-composited with F̂ and scored as
  in H5 (B2's harness on the dataset masks, which stay the reference).
  Pairs: each item's H5 hull pairs on weighted LPIPS (12–17 per item),
  at each s. Bias, stated: the streams were coded for the exact F (the
  background filled F, the foreground kept K = F ⊕ (8 + 2·scale) px), so
  where F̂ leaves K the foreground shows padding; a pipeline built for
  F̂ would code K and the fill from F̂. This measures the candidate on
  today's streams, an estimate of the built component, not a bound.
  The H5 hull's pairs are chosen with F granted; with F̂ another pair
  could win, which only helps the candidate (the hull here is a lower
  bound of the candidate's own hull).
- Decision rule, fixed now. The candidate is adopted if, at some s (or
  the hull over s), PointStream with F̂ billed passes H5's rule (a)
  against the same pooled anchors: weighted LPIPS (fill) BD-rate −10% or
  better with at least 3 of 4 items below 0. Then the mask coder is
  built at that grid (encoder codes K and the fill from F̂). Otherwise
  the next candidate, with its own rule: F derived at the client from
  what it already decodes (the foreground stream coded with a key
  outside F), or contours with temporal prediction.
- Hypotheses. At s = 8 the class map costs 10–60 kbps per window and
  costs little quality (F̂'s boundary errors of up to 8 px fall in the
  ring's blend); the rule passes at s = 8 or 16, with a smaller margin
  than H5's −15% (the mask rate matters most at PointStream's low rates,
  where most of its hull lies but the anchors do not reach).
- Competing explanation for a failure: F's boundary is where the
  composite is judged hardest (hands against the background, the 70%
  weight on F); missing F pixels show the background's fill (inpainted,
  wrong content) and extra ones show the foreground's padding, so even
  small boundary errors cost more than the bits they save.
- Budget: 4 scoring jobs (one per item, about 60 pairs at 30–40 s), about
  3 GPU-hours, shared GPUs; each under 45 minutes.
- Runs (`h5 score --mask-scale s --pairs <H5 hull>`, code `802be6c`, shared
  Ada and A6000; every full stage 9/9 checks locally, every MASK payload
  decoding to its reduced grid exactly). P01_107 s = 2, 4, 8, 16:
  `20261017T150931Z-e433c92d`, `…150940Z-83ccfcf4`, `…150949Z-43d7a289`,
  `…150958Z-db776057`; P02_02: `…151010Z-7a75681d`, `…151019Z-494b6010`,
  `…151028Z-d9ada6bd`, `…151038Z-dd172972`; P03_10: `…151052Z-d0800d65`,
  `…151104Z-c9ed195b`,
  `…151116Z-ddb03427`, `…151127Z-ad0e1bbc`; P09_106: `…151140Z-c5e09039`,
  `…151149Z-33f8ba99`, `…151158Z-6374d0b3`, `…151207Z-d2be1d00`.
- Mask cost (kbps) and IoU of F̂ with F, P01_107 | P02_02 | P03_10 |
  P09_106: s = 2: 529 | 971 | 662 | 219 (IoU 0.98–0.99); s = 4: 166 | 320
  | 229 | 89 (0.97–0.99); s = 8: 51 | 116 | 75 | 36 (0.96–0.99); s = 16:
  16 | 39 | 23 | 14 (0.83–0.97). F costs about 1/s² of its lossless
  cost; at s = 8 it is still as much as PointStream's whole stream at
  its low end on the items where F is large.
- Outcome (BD-rate against the pooled anchors, as H5; per item P01_107 |
  P02_02 | P03_10 | P09_106):

  | F | weighted LPIPS | mean (CI) | < 0 | frame LPIPS | frame DISTS |
  |---|---|---|---|---|---|
  | granted (H5) | −12 \| +0 \| −26 \| −23% | −15% (−25 to −5) | 3 | −38% | −48% |
  | s = 2 | +123 \| +150 \| +122 \| −2% | +98% | 1 | | |
  | s = 4 | +33 \| +52 \| +27 \| −14% | +25% | 1 | +11% | +19% |
  | s = 8 | +3 \| +21 \| −8 \| −18% | −1% (−16 to +13) | 2 | −15% | −16% |
  | s = 16 | +21 \| +13 \| −11 \| +11% | +9% | 1 | −21% | −28% |
  | hull over s | +3 \| +13 \| −14 \| −18% | −4% (−17 to +8) | 2 | −21% (4/4) | −28% (4/4) |
  | lossless 1080p | | +279% | 0 | +261% | +348% |

  - **The rule fails**: no s, nor the hull over s, reaches −10% with 3 of
    4 items below 0 on weighted LPIPS; the candidate is not adopted as
    it is. Hypothesis wrong in cost (s = 8 costs 36–116 kbps, not 10–60)
    and in quality (the margin vanishes rather than shrinking).
  - The competing explanation holds on the F-weighted metric: F̂'s
    boundary errors (the fill showing where F̂ misses F, padding where it
    overshoots K) cost the 70%-weighted foreground more than the mask's
    bits save. On the whole frame, where F's boundary is a small part,
    PointStream with F coded at s = 16 still needs 21% less rate for the
    same LPIPS and 28% less for DISTS, on 4 of 4 items.
  - Bias recalled: the streams were coded for the exact F, so this
    under-states a pipeline built for F̂ (K and the fill from F̂); that
    component is the obvious next test of this candidate, cheaper than a
    new coder: re-encode the foreground with K from F̂ at s = 8 and 16.
  - Next, by the rule: F derived from what the client already decodes
    (no MASK segment: the foreground stream carries F, e.g. a key outside
    F or an alpha plane coded with it), or contours with temporal
    prediction; each with its own rule, after the F̂-consistent re-encode
    above says how much of the loss was the bias.
- Code: the report's per-key bootstrap (a comparison's CI no longer
  depends on what else is in the report; H5's headline CI is −25 to −5,
  first printed as −25 to −3), rows keyed by pair and mask scale.
- Report: `pointstream-data/visor/h5-2026-10-16/report-h5m-7ffa414/`.
- GPU time: about 2.5 GPU-hours (16 jobs).
