# Experiments: B1b

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md).

### 2026-10-07 — B1b: SAM 3.1 fill of VISOR's missing hands and objects
- Question: can SAM 3.1, prompted with the human masks at a run's keyframes,
  supply the hands and objects VISOR's dense masks lack, accurately enough to
  score B2 and D1 with a second mask set? On evaluation set v2 the dense masks
  lack 33% of each frame's labelled objects (B2) and no labelled hand (by
  construction); elsewhere 22–30% of dense val frames lack a labelled hand.
- Method (`experiments/visor/b1b.py`, `src/segmentation/sam31_tracker.py`).
  SAM 3.1's public multiplex API takes no mask prompts at the pinned commit,
  so its own tracker is used directly (sam3's `build_sam3_multiplex_video_model`,
  the checkpoint's `tracker.model.*` and `detector.backbone.vision_backbone.*`
  weights; every key must load). Per item, the *span* runs from the last
  human-labelled frame at or before the window to the first at or after it
  (251–516 frames, 2–11 keyframes; 98 keyframe pairs over the 34 items; a CPU
  survey, not evidence), decoded as consecutive video frames from keyframes
  placed by B1's verified rules; the sparse JPEGs inside the span are checked
  as in B1. Every object a human labelled at a keyframe of the span is a SAM
  object (at most 14 per item).
  *Held out*: for each consecutive keyframe pair a < b, SAM is prompted with
  the human masks at a only, tracks forward, and is scored at b against the
  human masks it never saw (J, boundary F at 0.8% of the diagonal), beside the
  floor "hold the mask of a"; objects labelled at a but not at b score whether
  SAM lets them go (`released`: under 64 pixels). Objects the dense masks drop
  between a and b form the *hard subset*: the ones the fill is for.
  *Fill*: per gap, SAM is also prompted with the human masks at b and tracks
  backward to the gap's midpoint; each frame takes the prediction from its
  nearer keyframe (the held-out forward run for the first half). On each
  window frame, an object the dense masks lack (`visor.object_key`) and SAM
  finds (at least 64 pixels) is added as tier `sam_from_label_prompt`; dense
  masks are unchanged. Output: `masks.rle` per item in B1's format. (Changed
  after the first two smokes, before any evidence: the plan was one session
  prompted at every keyframe, but multiplex SAM 3.1 takes mask prompts only on
  the first frame of a fresh state, `20261007T204154Z-9531c0f7`.)
  *Hand boxes*: EPIC-KITCHENS-100 hand-object detections (Shan et al. 2020;
  `Datasets/manifests/EPIC-KITCHENS-hand-objects.json`, detector output, not
  labels). A hand mask agrees when a detected hand of its side (score ≥ 0.5)
  lies at least half inside the mask's bounding box (VISOR hands include the
  forearm, the detector's do not). Rates are read against the human hands at
  keyframes and the dense hands on the same frames.
  *Review*: two window frames per item that are not keyframes, picked by
  sha256("pointstream-b1b-review:<item>:<n>"), drawn with the dense masks, the
  fill and SAM's own masks, published as an artifact for review by eye.
- Decision rule (fixed before any run; `b1b.DECISION`, applied by
  `b1b report` over all 34 items). Hands are adopted when, over all held-out
  pairs: mean J ≥ 0.70, median J ≥ 0.80, mean J at least 0.15 above the hold
  floor; on the hard subset (if n ≥ 10) mean J ≥ 0.60; hands labelled at a and
  not at b released in ≥ 70% of cases (if n ≥ 5); and SAM's held-out hands
  agree with the detector at ≥ 0.9 times the rate of the dense hands on the
  same frames. Objects are adopted when mean J ≥ 0.60, median J ≥ 0.65, mean J
  at least 0.10 above the floor, and on the hard subset (if n ≥ 10) mean J ≥
  0.50. If only one group passes, the fill keeps only that group (SAM
  instances of the other class dropped from the published masks on CPU). The
  review by eye can veto adoption for a systematic failure; it is recorded
  here. Only if a group is adopted are B2's published streams rescored with
  both mask sets (`b2 run --streams`, GPU scoring), and every B2 number is
  reported with and without the fill, with the share of labelled objects still
  missing.
- Hypothesis: across keyframe gaps of 1.3–1.7 s, SAM 3.1 from human masks
  reaches hand J ≥ 0.8, far above the hold floor; objects score lower (small,
  occluded, handled) but pass; the fill cuts the missing-object share from
  33% to under 10%, and moves B2's weighted PSNR by under 1 dB and its BD-rates
  by a few percent.
- Competing explanations: (1) VISOR dropped exactly the tracks its
  interpolation found hard (fast motion, occlusion, small objects), so SAM's
  accuracy on all objects overstates it on the filled ones; the hard subset
  tests this. (2) One-sided tracking over a whole keyframe gap is harder than
  the fill (each frame at most half a gap from its prompt), so held-out scores
  understate the fill; the rule accepts that bias as conservative. (3)
  Detector agreement may track the detector's own failures on blurred frames;
  the dense hands on the same frames calibrate it. Pairs within an item are
  correlated; results are also given per video type and per item.
- Budget: GPUs Ada and A6000 only (model–GPU table). Smoke: one item
  (P30_110, span 251, one pair), ≤ 600 s. Pilot: the four B2 pilot items
  (P01_107, P09_106, P02_02, P03_10; 1,272 span frames, 11 pairs), ≤ 1 h.
  Full: the other 30 items as two jobs of 15, ≤ 2 h each. Rescoring (if
  adopted): B2's five full jobs' streams, SVT-AV1 scored on any Ada/A6000,
  DCVC-UF decoded on RTX A6000 (its encode class), ≤ 1.5 h per job. Ceiling 9
  GPU-hours and 12 h wall including staging.
- Smokes that failed (not evidence): `20261007T203825Z-0e10810c` (gpu6, Ada,
  `b64332b`): the standalone tracker asked its backbone for the detector's
  head, whose output the tracker cannot read; it now asks only for its own two
  heads. `20261007T204154Z-9531c0f7` (gpu3, A6000, `25db974`): a second mask
  prompt on a fresh multiplex state is refused (above). Every checkpoint key
  loaded in both (931 of 931).
- Pilot `20261007T205637Z-0ba26d70` (gpu3, RTX A6000, `4d7f28b`; smoke on
  P30_110 and validator passed, 18 checks; full stage 573 s, about 5 frames/s
  tracked including the backbone, peak 5.9 GiB): held out, hands mean J 0.90,
  median 0.96 (n = 22; hold floor 0.28); objects 0.88, 0.91 (n = 30; floor
  0.31), hard subset 0.83 (n = 10). Detector agreement: SAM's held-out hands
  0.73 (1,387 of 1,908), dense hands on the same frames 0.83 (1,592 of 1,929).
  Finding: on P09_106 57 of the 66 filled hands lay more than half on the
  other hand, which the dense masks already had (a track that lost its hand
  and latched onto the other; the detector called 32 of them the other side).
  Fill rule added before the full run (adoption rule unchanged): a SAM
  instance more than half on an instance already in the frame is not added
  (`MAX_OVERLAP`). The full run repeats the four pilot items with it.
- Full jobs (commit `1d33976`, environment `pointstream-20261006T113321Z`,
  inputs `pointstream-data/visor/b1b-2026-10-07/inputs/`: detections tar
  `949ea65d…97c3`, manifest `711870f2…1ee5`, SAM checkpoint `0567debe…1cb6`,
  B1 masks `0cf83b8c…37b5`; smokes and validators passed, 19 checks each, and
  the validator passes on both full outputs too): share A
  `20261007T211731Z-f7a21e9f` (17 items, gpu3, RTX A6000, 2,876 s full stage,
  `b1b.json` `b6009493…527a`) and share B `20261007T211914Z-9402b935` (17
  items, gpu6, RTX 6000 Ada, 2,018 s, `39744ab2…c2db`); 17,535 frames
  tracked, 2,344 s of tracking, peak 6.8 GiB; no contention. Report
  (`b1b report`, `pointstream-data/visor/b1b-2026-10-07/report-1d33976/`,
  `b1b-report.json` `b6567477…d2a5`):

  | Held out (prompt at a, score at b) | n | mean J | median J | mean F | hold floor J | hard n | hard mean J |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | Hands | 178 | 0.931 | 0.969 | 0.965 | 0.360 | 1 | 0.974 |
  | Hands, EK-100 / EK-55 | 99 / 79 | 0.965 / 0.888 | 0.972 / 0.965 | 0.991 / 0.932 | 0.424 / 0.279 | | |
  | Objects | 407 | 0.828 | 0.911 | 0.905 | 0.252 | 158 | 0.704 |
  | Objects, EK-100 / EK-55 | 221 / 186 | 0.858 / 0.793 | 0.922 / 0.889 | 0.928 / 0.879 | 0.314 / 0.179 | 75 / 83 | 0.746 / 0.666 |

  Mean gap between keyframes 111–117 frames (about 2 s). Hands labelled at a
  and gone at b: SAM let 4 of 6 go (0.67); objects 24 of 49. Detector
  agreement (hand box at least half inside the mask's box): SAM's held-out
  hands 0.941 (13,471 of 14,317), dense hands on the same frames 0.958
  (13,705 of 14,313), ratio 0.98; human hands at keyframes 0.943 (232 of
  246). Frames where the detector sees a hand that the masks lack: 41 of
  14,544 detected hands (0.3%), the same with and without the fill.
- Decision (by the rule): **objects adopted, hands not**. Hands pass every
  accuracy check by a wide margin but fail one: SAM released 4 of 6 departed
  hands, under 0.70 (n = 6, so the failure rests on two cases). The fill
  therefore keeps objects only. On this set that costs little: the overlap
  rule had already left 9 filled hand instances (57 rejected), none of which
  the detector confirmed, and the review frames show the failure mode (a
  hand track moving onto the other hand). The hard subset (objects the dense
  masks drop) is 0.70 against a 0.10 floor. The review by eye
  ([artifact](https://claude.ai/artifact/GSkm3caejx3bNJYUJnQP8K), 68 frames;
  user, 2026-10-08) flagged 5 frames in 4 items (P01_107 w083, P04_24 w055
  and w096, P08_17 w128, P27_105 w169), all hands in SAM's own tracks: the
  hand kept but the forearm lost, or hand and forearm lost. None is in the
  adopted object fill, so there is no veto; they confirm that SAM's hand
  tracks are not fit to fill hands. Also visible there, not flagged: on
  P27_105 the filled fridge is fragmented into speckle, and large touched
  surfaces (sink, fridge, cupboard) enter the foreground as VISOR labels
  them.
- Fill (objects only; merge job `20261007T222646Z-3a7ac6ef`, CPU, `339c918`,
  validator passed; `published.tar` `049977f0…799a`, `merge.json`
  `1c705e4a…4bea`): 18,267 object instances added on 7,504 of 8,160 window
  frames in 33 of 34 items; 833 object instances rejected for overlap. The
  share of each frame's labelled objects missing falls from 33.3% (dense) to
  1.6% (with the fill), mean over items. Limit: on 8 items a prompt is not
  reproduced on its own frame (minimum IoU 0.00–0.76, every item's median ≥
  0.96), most likely thin or tiny masks lost at the tracker's mask-input
  resolution.
- B2 rescored with both mask sets (commit `339c918`; `b2 run --streams` on
  each B2 full job's published streams, every stream checked against B2's
  recorded sha256; all validators passed; no contention): SVT-AV1
  `20261007T223938Z-cc5aa830` and `20261007T224051Z-2891acbc` (gpu6, RTX 6000
  Ada for scoring; dav1d on CPU; 520 and 554 s), DCVC-UF
  `20261007T223639Z-74f60a0f`, `20261007T223737Z-ab077fea`,
  `20261007T223830Z-2139df4c` (gpu3, RTX A6000, B2's encode class; 2,306,
  2,044 and 2,094 s). Reproduction: `visor_dense` scores equal B2's, within
  9e-6 dB for SVT-AV1 (GPU now, CPU then) and exactly for DCVC-UF. Report
  `pointstream-data/visor/b1b-2026-10-07/b2-rescore-339c918/` (`b2-report.json`
  `ce234b4f…9f29`, `b2-rd.png` `83966155…eb80`), means over the 34 items:

  | Codec, point | kbps | wPSNR dense / fill (dB) | fg PSNR dense / fill | bg PSNR dense / fill | wLPIPS dense / fill |
  |---|---:|---|---|---|---|
  | SVT-AV1 CRF 62 | 434 | 34.58 / 34.18 | 34.42 / 33.75 | 34.96 / 35.20 | 0.160 / 0.162 |
  | SVT-AV1 CRF 55 | 1,022 | 36.87 / 36.47 | 36.85 / 36.19 | 36.94 / 37.11 | 0.120 / 0.121 |
  | SVT-AV1 CRF 48 | 1,576 | 37.84 / 37.44 | 37.88 / 37.26 | 37.73 / 37.86 | 0.105 / 0.106 |
  | SVT-AV1 CRF 41 | 2,498 | 38.67 / 38.28 | 38.78 / 38.18 | 38.41 / 38.51 | 0.094 / 0.094 |
  | DCVC-UF QP 27 | 329 | 34.47 / 33.95 | 34.58 / 33.76 | 34.22 / 34.41 | 0.175 / 0.179 |
  | DCVC-UF QP 36 | 584 | 35.97 / 35.45 | 36.09 / 35.28 | 35.67 / 35.84 | 0.142 / 0.145 |
  | DCVC-UF QP 45 | 1,032 | 37.21 / 36.70 | 37.35 / 36.56 | 36.87 / 37.02 | 0.119 / 0.121 |
  | DCVC-UF QP 54 | 1,846 | 38.26 / 37.76 | 38.42 / 37.64 | 37.88 / 38.02 | 0.102 / 0.104 |
  | DCVC-UF QP 63 | 3,283 | 39.14 / 38.63 | 39.31 / 38.53 | 38.74 / 38.87 | 0.090 / 0.091 |

  Labelled objects missing, mean over items: 33.2% dense, 1.6% with the fill
  (EK-100 30.8% / 1.8%, EK-55 35.2% / 1.4%). BD-rate of DCVC-UF HT-L against
  SVT-AV1 on weighted PSNR, mean / median: dense −13.3% / −18.3%, with the
  fill −10.2% / −16.0% (EK-100 −18.3 / −29.9 → −14.5 / −25.6; EK-55 −9.4 /
  −11.3 → −6.8 / −7.6). Whole-frame PSNR (−0.5% / −3.6%) and VMAF (+7.5% /
  +5.0%) use no masks and do not change.
- Reading: the hypothesis holds for objects. The fill cuts the missing
  objects from 33% to 1.6% and moves weighted PSNR by 0.4–0.5 dB (it lowers
  it: the added objects are handled, moving and harder to code than the
  background they were counted as). It narrows DCVC-UF's weighted advantage
  over SVT-AV1 by about 3 points of BD-rate without reversing it. For hands the
  rule held them back on a small-n release check, and the pilot found the
  failure the check guards against. B1b's done-when holds: the numbers from
  recorded jobs decided adoption. GPU use about 4 GPU-hours (fill 1.4,
  rescoring 2.1, smokes and pilot 0.3) of the 9 budgeted; wall time about 5.5 h.
  Not done: train-split filling for E1/F1 (hands need a better release
  rule first).
- Train split without a hand fill (CPU survey, gpu1, 2026-10-08, code
  `2eeb7a2`, not a fleet job; script and output in
  `pointstream-data/visor/b1b-2026-10-07/train-gap-survey-2eeb7a2/`,
  `train_gaps.json` `79beec63…db92`): of the 2,277,946 dense frames of the
  115 train videos, F1 admits 2,171,781 (drift ≤ 1; P17_01 at 47.95 fps has
  no frame rule). Excluding the frames where `hand_gaps` reports a missing
  hand keeps 1,631,454 (75%; EK-100 77%, EK-55 59.94 fps 73%), with 89% of
  the hand masks and 82% of the object masks, and 4,425 of 5,000 dense runs
  keep at least one frame. Per video the loss has median 26%, 90th
  percentile 40%, maximum 55%. Decision: no hand fill for now. F1 and E1
  exclude the gap frames; 1.6 million frames at 50–60 fps from 115 videos is
  far more than either will sample. Revisit only if E1 or F1 turns out data
  limited, or if their errors concentrate on fast motion and occlusion, the
  conditions under which VISOR's interpolation, and so the excluded frames,
  fail.
- Review by eye, fragmentation (CPU, same day, merged fill
  `049977f0…799a`, every 4th frame): 20% of fill instances have more than a
  tenth of their area outside their largest connected piece, against 16% of
  the dense instances (hands occluding objects split masks legitimately);
  12% have more than a quarter outside. Most often fragmented: cheese, bread,
  pot, mozzarella (small or translucent, often held). The speckled fridge on
  P27_105 is the extreme case, not the norm. No change to the fill.
