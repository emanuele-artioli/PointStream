# Experiments: H4b

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md).

### 2026-10-13 — H4b: a bank of reference views for handled objects (oracle, VISOR)
- Question ([PLAN H4](../../PLAN.md#h4-handled-objects-rescoped-after-h1-user-2026-10-08),
  item 2; H4a dropped grip pose). H1 carried each object from one
  reference by chained homographies and refreshed it where it stopped
  holding: handled objects held on 49.9% of frames, median reference life
  0.02 s. A bank keeps the views already sent: each frame is predicted from
  whichever view predicts it best, and a new view is sent only when none
  does. An object turned back, or reappearing from under a hand, then costs
  no new view. This oracle bounds how much a bank can save before any bank
  is coded: if even an unlimited bank, choosing views with hindsight,
  needs a new view about as often as H1's single reference, the component
  is not built and handled objects stay pixels (item 3).
- Data: evaluation set v2 (34 windows of 240 frames), fill masks
  (`visor_dense_sam_fill`, as H1), H1's objects: each object label's union
  of instances per frame, *handled* when a human labelled it in contact
  with a hand at a keyframe bounding or inside the window (H1's
  `contact_objects`). Frames decoded as H1 (sparse-JPEG gate), full
  resolution. Code `experiments/visor/h4b.py` (to be written after this
  entry), reusing H1's flow, homography fit and hold test.
- Method. Per object track, frame by frame, as H1: DIS flow (medium preset,
  half resolution), a homography fitted to it in the object's mask
  (RANSAC 2 px). A view r *predicts* frame t through a homography H
  (r → t) when, warping view r's pixels onto the object's mask at t, at
  most 20% of the mask is uncovered and PSNR on the covered pixels is
  ≥ 30 dB (H1's test; 33.75 dB beside). Two ways to reach a view:
  *chained* (the frame-to-frame homographies since r, broken where a step
  has no fit or the object is absent, as H1), and *direct* (SIFT features
  inside each mask, dilated 8 px; ratio test 0.75; homography by RANSAC
  3 px with ≥ 15 inliers; the 4 views with most inliers are tried). Coders:
  1. *single*: H1's single reference, chained only (must reproduce H1's
     recorded counts; the correctness check);
  2. *single + direct*: one reference, reached by either way;
  3. *bank-4*: the last 4 views used, either way;
  4. *bank*: every view sent so far, either way (the oracle).
  At each frame the coder takes the view and way with the highest PSNR
  among those that predict t; if none does, t becomes a new view. When a
  coder has more than 4 candidates (chained views plus direct tries), they
  are first screened on every 4th mask pixel in each direction and only
  the 4 best get the full test (added with the code, before any run: an
  unlimited bank can hold a hundred live chains; *single* always gets the
  full test, as H1). Environment and inputs are H1's
  (`pointstream-20261006T113321Z`), so *single* is checked against H1's
  recorded chains per track (within 2% of frames on ≥ 95% of tracks). Per
  track: frames *held* (predicted) and the new views, with the interval
  between consecutive new views (s). Other objects are run and reported
  beside; the decision is on handled objects.
- Decision rule (fixed before any run; 30 dB; means over the 34 items of
  per-item values pooled over the item's handled tracks). The bank is
  worth building when the *bank* oracle (a) holds ≥ 70% of handled-object
  frames, (b) has a median interval between new views ≥ 0.5 s (H1's
  thresholds for a rigid part), and (c) the bank is what helps: it sends at
  most half the new views of *single + direct* (a single reference that
  can also re-register). Pass: build the bank component (finite views,
  coded views and residual) and score it on handled-object pixels against
  SVT-AV1 and DCVC-UF at equal quality (BD-rate per metric against their
  envelope). Fail: handled objects stay coded pixels (item 3), and H4 is
  done. Unclear ((a) and (b) pass without (c), or a pass at 30 dB that
  fails at 33.75 dB): reported, and the user decides.
- Hypothesis: the bank helps where H1's error split says it can, with new
  pixels (17% of the homography's error on handled objects) and occlusion
  by hands (10%), but not with appearance change (28%), deformation (27%)
  and motion blur (18%). Held frames rise from 50% to about 60%, the
  interval between new views stays near 0.1 s, and the bank fails (a) and (b).
- Competing explanations. (1) H1's single reference fails because its chain
  breaks (no fit for a frame, the object absent for a frame), not because
  the appearance changes. Then *single + direct* recovers most of the gain
  and (c) fails; the bank itself is not what helps. (2) 30 dB on covered
  pixels is the wrong bar against coded pixels; 33.75 dB (B2's foreground
  PSNR at SVT-AV1 CRF 62) is reported beside. (3) A pixel codec already
  predicts from several past frames (SVT-AV1 up to 7 references, DCVC-UF
  from its decoded past), so a pass bounds the representation, not its
  gain over the codecs; the BD-rate stage decides that.
- Budget: CPU only (`"device": "cpu"`, 16 threads); decode, flow and SIFT
  per item. Smoke: one item, 48 frames, ≤ 600 s. Pilot: B2's 4 pilot items,
  ≤ 45 min, which also times an item. Full: the other 30 items, in parts of
  at most 45 min each. Ceiling 8 CPU-hours.
- Jobs: pilot `20261013T074546Z-821ee185` (code `2880e27`, environment
  `pointstream-20261006T113321Z` as H1; CPU only, 16 threads on gpu4;
  inputs `pointstream-data/visor/h4b-2026-10-13/inputs-7f2aebab5a6a.json`,
  spec `h4b-pilot.json` beside it). Smoke P01_107, 48 frames (98 s); full
  stage B2's 4 pilot items at 240 frames (1,260 s; per item 188–1,257 s,
  the bank's candidates dominating). Validators passed on the smoke and,
  locally, on the full stage's `published.tar` (`e14d7f86…`; 6 checks,
  including *single* against H1's recorded chains: exact on all 28
  track-thresholds). Report: `…/h4b-2026-10-13/report-pilot-2880e27/h4b-report.json`
  (`1e10dfb0…`). Not evidence: `20261013T074250Z-24ac003d` (the run
  finished its smoke; the validator was given a `{staged:…}` placeholder,
  which the fleet does not expand for validators).
- Outcome (pilot, means over the 4 items; 6 handled tracks, 1,342
  handled-object frames):

  | coder | held at 30 dB | median interval | new views per s | held at 33.75 dB |
  |---|---|---|---|---|
  | single (H1) | 54.6% | 0.032 s | 31.8 | 26.2% |
  | single + direct | 56.2% | 0.032 s | 31.0 | 26.4% |
  | bank-4 | 56.7% | 0.032 s | 30.6 | 26.4% |
  | bank (every view) | 55.9% | 0.032 s | 31.1 | 26.3% |

  The unlimited bank sends 1.00× the new views of *single + direct*
  (rule ≤ 0.5×). Per handled track (frames; new views single / single +
  direct / bank): soy milk 240; 140 / 125 / 130, glass 240; 77 / 77 / 77,
  tap 240; 60 / 59 / 56, pancake 240; 63 / 63 / 63, pancake package 240;
  128 / 128 / 128, fork 142; 78 / 78 / 78. Direct registration reaches 8%
  of the bank's held frames. Other objects behave the same (held 66.7% →
  67.3%, ratio 1.02).
- Decision (by the rule, on the pilot): **the bank fails every check**:
  held 56% (rule 70%), median interval 0.03 s (rule 0.5 s), and it sends
  as many new views as one re-registering reference (rule half). The rule's
  unit is the 34 items; the other 30 were not run, because no mean over
  them could pass: (c) needs the bank to halve the new views, and on every
  pilot track it changes them by 0–7%. H4b stops at the pilot; handled
  objects stay coded pixels (H4 item 3).
- Reading. The hypothesis held, and more strongly than predicted: the bank
  adds nothing, not even the ~10 points expected from occlusion and new
  pixels. What *held* demands: the warped view alone, with no residual,
  within 30 dB of the frame; a frame that is not held falls back to its
  pixels. In frames, the median gap between new views per handled track
  is 1, 2, 4, 3, 2 and 2 (soy milk, glass, tap, pancake, pancake package,
  fork); the mean over items of 0.032 s is about two frames at 50–60 fps,
  so these objects need their pixels on about every other frame, and the
  held frames are below the codec's own quality (at 33.75 dB, B2's
  foreground at CRF 62, 26% are held). The coders agree because they share
  the latest view and differ only in older ones: a view sent earlier
  almost never predicts a later frame better than the latest view does
  (the unit tests show the bank does reuse views when an object turns
  back and forth, and *single* reproduces H1 exactly): within a 4.8 s window the objects' appearance
  changes one way (turning, being cut, filled, lit differently, blurred),
  so the most recent view is always the best one and older views are not
  revisited. Competing explanation (1) does not hold either: direct
  registration recovers 1.6 points over H1's chain, so broken chains are
  not why the reference dies. What fails is appearance: at 30 dB a view
  predicts the next frame on half the frames and is resent every 0.03 s,
  which is what a pixel codec's own inter prediction already does.
- Scope. 4 pilot items, 6 handled tracks. Views are warped by homographies;
  a richer warp (dense flow, a 3D model) was H1's flow bound (1–2 dB over
  the homography) and is not a bank question. The screen (4 candidates
  kept from a 1/16-pixel test) could miss a view; it never applies to
  *single*, and *single + direct* differs from the bank by under 1 point.
