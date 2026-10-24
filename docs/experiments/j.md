# Experiments: J

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md).

### 2026-10-22 — J: experience-track re-judging (rule)
- Question (user, 2026-10-22; protocol "Two tracks"). Four routes closed on
  fidelity only. Does any of them give the same experience as the anchors
  at its own rate: realistic, the same content, as steady over time? A pass
  reopens that route on the experience track only (its next step goes to
  the user); a fail closes it on both tracks. Fidelity verdicts stand
  either way.
- Results, cheapest first. Each is judged only from outputs that exist,
  except FoundHand, whose re-run is budgeted below. Nothing else is
  regenerated under this entry.
  1. **H3e** V1 (JointControlVideo; 4 pilot items, generations and judge
     archives saved; `h3e-report.json` `49af833b…`). Its nine V1 points
     differ only in the DCVC-UF context, so the encoder may choose: the
     item's operating point is the lowest-rate V1 point that passes all
     three checks, else its lowest-rate point.
  2. **FoundHand** `gen` (K6b hands rung 2; positions 0–3 of K1f's order).
     The crops were not kept, so `gen` is re-run with rung 2's code and
     settings (50 steps, seed as recorded), keeping the composites and
     crops, on all 4 items (P09_103 included). Its rate is the item's
     keypoint rate (reference unbilled, as rung 2). `ref` and `warp` are
     judged from the same job, beside it.
  3. **H3a / H3a2** (texture transfer; mesh renders against SVT-AV1). Only
     if their stages' renders or composites are on the server. If not, the
     entry records "not re-judged: outputs not kept", and they stay
     untested on experience.
  4. **K6 `ref`** on players (K4's 8 scored windows, every 4th frame code,
     3.6 kbps for two). Its renders are recomputed if not kept: a CPU warp,
     no generator.
- Judges, per item at the method's rate, against the anchor hull at that
  rate. Anchors: DCVC-UF + G and SVT-AV1 for hands (H3c's perceptually
  tuned DCVC-UF where it exists), K6's envelope for players. Each anchor
  quantity is read on its own hull (lower convex hull of the quantity over
  log rate), linear in log rate at the method's rate, clamped to the
  anchor's range (an item outside the range is flagged).
  - **Realism.** KID (`h3b`'s Inception crops and `kid_from`, bootstrap
    95% interval) of the object's crops against a fixed reference set of
    real crops of the class from footage that neither the method nor the
    anchor was fitted on. Hands: VISOR videos outside the 34-item
    evaluation set and outside G's and G_HO's training footage, with the
    list and its sha256 recorded before any score (FoundHand has seen
    VISOR, so its overlap is recorded, as rung 2). Players: K4b's matches
    other than the window's. Pass: the method's KID ≤ the anchor's upper
    95% bound.
  - **Same content.** The object's estimator on the output against the
    same estimator on the source: WiLoR's 2D joints for hands (mean joint
    distance, % of the box, H3c2's judge), YOLO11x-pose for players (K4b.2's
    path), as a share of the box. Pass: the method's error ≤ max(the
    anchor's, 1.5 × the coded keypoints' own error against the source).
  - **Same over time.** On the region, over consecutive judged frames.
    The accepted threshold names two quantities, so each is compared in
    its own form: pass if the method's `h3b.flicker` (its temporal residual
    minus the source's) ≤ the anchor's, **or** the method's own residual
    energy (mean absolute motion-compensated luma change, `g5c.flicker`'s
    energies) ≤ 1.25 × the source's. *This reading is the drafting
    session's; the user may change it before the first score.* Identity
    drift along a track is reported beside it and does not decide.
- Decision rule, per result (fixed before any score). An item passes when
  all three checks pass. **Pass**: at least 75% of the result's scored
  items pass (3 of 4 for H3e and FoundHand; 6 of 8 for K6 `ref`). **Fail**
  otherwise. Items in a judge's or generator's training footage are
  reported apart beside the pooled share. Each check's per-item pass rate
  is reported too, so a fail names the check that fails.
- Hypothesis. All four fail. H3e on same content (V1's pose error 1.8–6.5%
  of the box, where the pixel hull is at 0.3–1 kbps for that error) and on
  time (flicker 5–23 against DCVC-UF's 3–7); FoundHand on time (its change
  19.7 against the source's 14.4 on P06_108) and on content; K6 `ref` on
  content (it draws 32–46% of R from a fixed reference).
- Competing explanation. The thresholds are lenient where the anchor is
  poor: at 2–4 kbps the pixel anchors blur hands and limbs, so their pose
  error and KID are high and a generator clears them while looking wrong.
  The visual check of one frame per item is reported beside each pass for
  that reason. Or the reference set differs in domain (lighting, kitchens)
  enough that KID ranks every method alike.
- Budget. CPU: the reference set, the hulls, the reading of saved
  archives. GPU: Inception and WiLoR or YOLO on saved outputs, ≤ 1 GPU-hour
  for H3e, H3a/H3a2 and K6 `ref` together; FoundHand's re-run ≤ 3
  GPU-hours (rung 2 took about 0.45 GPU-hours per item at 50 steps on
  Ada, plus its failed attempts), declared at 40,000 MiB, which rung 2's
  scoring needed. Each stage ≤ 45 minutes.
- Time reading (user, 2026-10-22, before any score): the OR reading as
  drafted. Pass if the method's `h3b.flicker` ≤ the anchor's, or its own
  residual energy ≤ 1.25 × the source's.
- Order of checks (session, 2026-10-22; protocol: stop when the decision
  is made). Each anchor quantity is read on its own hull. Adding anchor
  points can only lower a hull inside the rate range the smaller set
  already spans. So a check read against SVT-AV1's points alone (CPU)
  is conclusive where it *fails* and the method's rate is inside their
  range: SVT-AV1 + DCVC-UF + G would be at most as high. A subset pass
  decides nothing. Content and time were scored first. Realism (KID)
  was to be scored only for items they left undecided, so the held-out
  reference sets were not built: no result needed them. Code
  `experiments/visor/j.py` (checks, hull, decision),
  `experiments/tennis/j_players.py`, `experiments/visor/j_hands.py`.
- **H3e V1: fail** (content). From `h3e-report.json` (`49af833b…41d4`), CPU,
  code `b46b81be`: `visor/j-2026-10-22/report-h3e-b46b81be/j-h3e.json`
  (`6985b3d1…f208`). Against SVT-AV1's points alone (H3c2's judge), every
  one of V1's nine points fails same content on P01_107 (1.77% of the
  box against the SVT-AV1 hull's 0.45–1.60% at its rates, and 1.5 × the
  keypoints' 0.77% = 1.16%) and on P03_10 (2.0% against 1.37–1.63%;
  floor 1.46%). So at most 2 of 4 items can pass, and 3 are needed.
  P09_106 passes content through the keypoints' floor (1.5 × 4.49%) at
  six points. P02_02 is open at its lowest point (5.45% at 2.71 kbps,
  under SVT-AV1's 5.91%). Realism and time were not scored. H3e's own
  flicker (V1 5–23 against DCVC-UF's 3–7) is reported beside.
- **H3a / H3a2: not re-judged: outputs not kept.** Their published units
  (`h3-2026-10-10`, jobs `20261010T073916Z-5058a257` and the H3a2 parts)
  hold per-frame scores only (about 55 KB per 240-frame item), and
  `experiments/visor/h3.py` writes no renders. They stay untested on
  experience.
- **K6 `ref` on players: fail** (time, 8 of 8). Jobs (shared GPUs; code
  from each `ready.json`): part 1 `20261022T123630Z-716f81c3`
  (`a947277d`, gpu3 RTX A6000), part 2 `20261022T123851Z-898cda0b`
  (`80133781`, gpu3 RTX A6000), part 3 `20261022T123907Z-d4dd728e`
  (`80133781`, gpu6 RTX 6000 Ada). K6's pipeline unchanged (players,
  pose, the chosen code, `ref` drawn as K6 draws it), K6's inputs
  (`20e595d5…e703`); anchors SVT-AV1's four low points (CRF 63 at ⅛, ¼,
  ½; CRF 55 at ¼), which span every window's keypoint rate and reproduce
  K6's recorded rates on R within 0.1% on all 8 windows. Every full stage
  passes `j_players validate` locally (89 checks). Report
  `tennis/j-2026-10-22/report-players-80133781/j-players.json`
  (`6b849d00…6933`), validations `j-players-validation.tar`
  (`f5120908…f8115`). Per window, at its keypoint rate (1.9–9.8 kbps):

  | window | kbps | `ref` flicker / SVT-AV1 hull | `ref` energy / 1.25 × source | `ref` pose / SVT-AV1 hull (box) | coded keypoints |
  |---|---:|---:|---:|---:|---:|
  | alcaraz_hurkacz/036 | 1.94 | 28.6 / 15.0 | 23.4 / 16.3 | 0.191 / 0.113 | 0.027 |
  | alcaraz_hurkacz/045 | 2.07 | 25.4 / 15.4 | 20.1 / 15.3 | 0.135 / 0.190 | 0.026 |
  | djokovic_federer/038 | 2.74 | 18.1 / 7.1 | 16.0 / 6.8 | 0.263 / 0.134 | 0.014 |
  | djokovic_federer/072 | 2.92 | 16.0 / 7.6 | 13.7 / 7.0 | 0.163 / 0.107 | 0.014 |
  | djokovic_zverev/008 | 3.20 | 15.9 / 6.6 | 13.6 / 6.4 | 0.070 / 0.037 | 0.006 |
  | djokovic_zverev/012 | 3.32 | 18.4 / 7.5 | 15.6 / 7.4 | 0.137 / 0.027 | 0.007 |
  | medvedev_struff/033 | 3.04 | 25.6 / 16.0 | 18.6 / 17.4 | 0.324 / 0.316 | 0.022 |
  | rublev_auger-aliassime/052 | 9.80 | 17.2 / 9.8 | 14.1 / 8.8 | 0.273 / 0.199 | 0.049 |

  Time fails on 8 of 8 windows, on both branches: `ref` changes more
  between frames than the source does (1.3–3.0 × its residual energy)
  and more than SVT-AV1 at its rate. Same content fails on 7 of 8 (045
  is open: `ref` 0.135 under SVT-AV1's 0.190). The content error counts
  a player that YOLO does not fit on an output as one box (`j.MISSED`,
  the session's convention): 2–28% of `ref`'s player-frames, and the
  anchors' too. The decision does not rest on it, since time alone
  fails every window. Realism was not scored. GPU: 1,769 s of stage
  execution (0.49 GPU-hours, the three failed smokes included: part
  1's first attempt `20261022T123215Z-4a8565be` wrote its gate to the
  wrong path, and parts 2 and 3's first attempts `…123634Z-ec2ceea4`
  and `…123639Z-6fd8908a` checked for references a 48-frame smoke cannot
  have; infrastructure, no score).
- **FoundHand `gen` (K6b hands rung 2): fail** (time, 2 of 2 items
  scored, which decides 4). Re-run with rung 2's code path and settings
  (`k6bh.render_item`, 50 steps, item-keyed seeds, its crops and
  composites; `ref` and `warp` beside), with K1f's seven SVT-AV1 low
  points coded as K1f codes them (their rates on R reproduce K1f's within
  0.1%), WiLoR (H3c2's error, box from each row's uncoded keypoints grown
  ×1.2), and the flicker judges. Jobs (gpu5 RTX 6000 Ada, shared, code
  `a947277d` from `ready.json`): P09_103 `20261022T123720Z-c74f647b`,
  P22_107 `20261022T123731Z-683a932b` (G's training footage). Both full
  stages pass `j_hands validate` locally (10 checks each). Report
  `visor/j-2026-10-22/report-hands-80133781/j-hands.json`
  (`8dc07d63…9f70`), validations `j-hands-validation.tar`
  (`6325f2ce…5fe9`). At each item's keypoint rate:

  | item | kbps | `gen` flicker / SVT-AV1 hull | `gen` energy / 1.25 × source | `gen` pose / SVT-AV1 hull / 1.5 × keypoints | `ref` (pose, flicker) | `warp` (pose, flicker, energy) |
  |---|---:|---:|---:|---:|---:|---:|
  | P09_103 | 2.03 | 10.7 / 3.8 | 10.5 / 2.1 | 5.4% / 7.4% / 5.7% (passes) | 8.3%, 5.0 | 2.5%, 5.5, 5.6 |
  | P22_107 (G) | 1.93 | 6.1 / 2.1 | 5.8 / 1.6 | 3.9% / 4.7% / 3.5% (open) | 4.5%, 3.3 | 1.4%, 4.4, 4.2 |

  `gen` fails time on both items, on both branches: its residual energy
  is 4–6 × the source's, and its flicker 3 × SVT-AV1's at its rate. Two
  failed items of four leave at most 2 passing, under the 3 needed, so
  the remaining items (P06_108, P25_09) were cancelled once the decision
  was made (`…123735Z-abbc683f` in its full stage, `…123749Z-a8815161` in
  its smoke; no score read). Same content passes or is open: on P09_103
  `gen`'s pose error (5.4% of the box) is under the keypoints' floor, and
  on P22_107 it is under SVT-AV1's hull, which decides nothing. Realism
  was not scored. FoundHand may have seen this footage (rung 2: the
  overlap could not be resolved), and P22_107 is in G's footage; the
  decision does not depend on either, since both items fail.
  Beside: `ref` and `warp` fail time on both items too. `warp`, K1f's
  bound (the true pixels moved to the coded keypoints), changes 3.2–3.4 ×
  as much as the source between frames and 1.5–2.1 × SVT-AV1's flicker at
  its rate. The coded keypoints themselves (100 ms coding, interpolated
  between sends) make the hand move unlike the source, before any
  renderer. GPU: about 0.76 GPU-hours of claimed time (2 items of about
  19 min, the cancelled jobs 6 and 1 min), within the 3.
- Visual check: no item passed, so none is shown beside a pass. One frame
  per scored item is kept (`publish/frames/` in each job's
  `published.tar`; K6 `ref` in `look-t*.jpg` per window). On P22_107
  (frame 120) `gen` draws a plausible right hand, softer than the source
  and in a slightly different grip; `ref` shows the warped reference's
  doubled fingers; SVT-AV1 at ¼ and ⅛ blurs the hands.
- **Decision (by the rule): all four routes stay closed.** H3e V1 fails
  (content), FoundHand `gen` fails (time), K6 `ref` fails (time), and
  H3a/H3a2 are untested on experience (outputs not kept). By the rule a
  fail closes the route on both tracks. Fidelity verdicts stand.
- Hypothesis ("all four fail"): held for the three judged. H3e failed on
  content as predicted; its time was not needed. FoundHand failed on time
  as predicted; its content passed or stayed open, not failed as
  predicted. K6 `ref` was predicted to fail on content; it fails there
  on 7 of 8 windows, but time decides alone. The competing explanation
  (lenient thresholds where the anchor is poor) did not arise: no item
  passed. Not tested: KID's domain sensitivity (no reference set built).
- Reading. On the experience track the binding check is time, not
  realism or content. Generated and warped objects driven by coded
  keypoints change between frames 1.3–6 × as much as the source does,
  while the pixel coders at the same rate stay near the source's own
  change. Even K1f's bound fails it on both judged items. So a renderer
  for this track needs temporally smooth keypoints or a temporally
  conditioned renderer, and the keypoint code's own jitter needs its own
  measure first. Which, if any, to pursue is the user's call.
- Budget: GPU 0.49 GPU-hours (players) of the 1 shared with H3e and
  H3a/H3a2 (no GPU), and about 0.76 of FoundHand's 3. CPU: the reports.
