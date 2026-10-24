# Experiments: K8b

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md).

### 2026-10-20 — K8b: court recall in shade and across venues (rule)
- Question (PLAN K8, after K8 was not adopted). Can a revised registration
  meet K8's unchanged decision rule on footage the method was not
  designed on?
- Diagnosis on dev data: K8's 82 marked shots, which may now be used for
  design. K8's line detector keeps only pixels with value > 170 and
  saturation < 80. In shade (court grass at value ≈ 90, a line at ≈ 133)
  it finds no lines. Late in the match the near half of the court is in
  shade, so those shots get a wrong fit (one line off) or none (0172,
  0201, 0223, 0259). The remaining misses fail only the per-frame 2 px
  bar of the shot rule (2–4 px during pans). Four of the six mixed shots
  switch from no-fit to fit with no scene-score cut, which looks like a
  gradual transition.
- Method (each piece checked on one clean frame from each of the 82 dev
  shots, decoded without warm start):
  1. *Relative line detector.* A line pixel is brighter than its local
     background (morphological opening, 31 px) by at least
     max(10, 0.10·background), and less saturated than its surroundings
     (closing, ≤ 0.75 of them). Support from long Hough segments is kept
     as in K8.
  2. *Hypotheses.* The previous frame (warm start, reset at cuts), then,
     when it fails or exceeds 2 px, the seeds from both K8's detector and
     the relative one. Each is refined against both masks, and the fit
     with the lowest error is kept. A bank of the match's own good
     homographies added one frame at twice the cost, so it is not used.
  3. *Ruler.* The error is the directed chamfer from projected model
     lines to the 1 px centerlines of the detected Hough segments.
     Measuring against a mask's full width would let a dense mask
     flatter the fit; on the dev frames the two agree within 0.7 px.
  4. *Spurious-fit guard.* Clutter is the share of the projected court's
     interior, more than 6 px from any model line, that is covered by
     detected pixels. A fit needs clutter ≤ 0.25, at least 6 supported
     segments and plausible geometry (as in K8). On the dev frames, court
     fits had clutter ≤ 0.085, and the two spurious fits (a close-up and
     a crowd shot) had 0.52–0.66.
  5. *Shot rule.* A frame *fits* at error ≤ 6 px, the p95 bar. A shot is
     court when at least 80% of its frames fit. The per-frame 2 px bar
     no longer gates the shot; registration accuracy is judged only by
     the decision rule below.
  6. *Cuts.* Scene score ≥ 0.25 as in K8. In addition, a shot is split
     where its per-frame fit state changes and the new state lasts at
     least 1 s. Shorter runs are absorbed into their neighbours.
  7. *Decoding.* Each clip is decoded sequentially from its start, with
     no seeks, so the decode-offset bug cannot arise.
  On the dev frames, 53/58 court frames fit within 2 px and 56/58 within
  6 px; K8's method got 42 and 52. The 18 non-court frames had 0 fits.
  These dev numbers are design evidence only.
- Held-out data: the 70 scene clips of the three other archived
  broadcasts (`tennis_games/alcaraz_hurkacz`, 21 clips, 6,968 frames at
  1080p50; `djokovic_zverev`, 8 clips, 20,320 frames at 2160p60, converted
  to 1080p by K4's conversion; `medvedev_struff`, 41 clips, 10,854 frames
  at 1080p50). These are different venues, not seen in design, and they
  are the clips K4b's windows come from.
  nadal_kyrgios is not re-scored: its marks were used for design.
- Marks: a review page, as in K8, of every shot the method produces in
  those clips (a census, so no sampling; about 100 shots, about 15–20
  minutes). The user marks the true class with K8's classes. The page
  shows predictions and the projected court as before.
- Decision rule (K8's thresholds, unchanged): adopted when, on the
  held-out shots as marked, court-view precision ≥ 0.95, court-view recall
  ≥ 0.90 weighted by duration, and on all true court frames (missing fits
  infinite) a median error ≤ 2 px and a 95th percentile ≤ 6 px with the
  ruler above. Mixed shots count wrong for both. Otherwise the misses are
  reported by class and the user decides. Limitation, stated beforehand:
  the clips were cut by the old pipeline around points and are mostly
  court views, so precision is weakly tested there. The 0/18 dev
  non-court fits are the only other precision evidence. If K8b is
  adopted, the nadal_kyrgios whole-match run (v1 of the court-shot list)
  comes next, and its predicted-court shots are spot-checked.
- Hypothesis: the shade failure and the 2 px gate cause most of K8's
  misses. Recall and registration pass on other venues, whose lighting
  and surfaces differ.
- Competing explanation: the relative detector is tuned to Wimbledon
  grass. Hard or clay courts with logos, shadows or different line
  contrast fail registration, or spurious fits appear on textured
  surroundings that the clutter guard does not catch.
- Budget: CPU only, on the fleet. At about 0.5–1.6 s per frame per
  process, 38k frames take about 20–60k CPU seconds. Parts are whole
  clips, 8 processes, checkpointing every finished clip. Smoke ≤ 600 s
  on one short clip; full stages ≤ 2,700 s; parts sized from the smoke's
  measured rate. No GPU.
- Pilot smoke `20261020T215039Z-fdd6bf8c` (`78f6fa0`, gpu4, CPU).
  An earlier submission, `20261020T215018Z-5357cd95`, snapshotted the
  main checkout (`dabfa5c`, which has no K8b code) and was cancelled
  before it ran anything. The smoke processed 240 frames of
  `djokovic_zverev/000` in 524 s and failed its validator: no frame fit.
  That clip is a court-level practice camera (4K60). Diagnosis: seven
  model lines fit within 2.4 px, but both singles sidelines were off by
  39 px. The fit had locked the model's doubles sidelines onto the court's
  singles lines. The real doubles lines leave the frame, the seed only
  takes sidelines that cross 60% of the frame height, and K8's geometry
  guard (corners within ±10% of the width) rejected the correct fit,
  whose near corners lie 25% beyond each side. Unguarded, the correct fit
  is 1.22 px on all nine lines. It is not evidence.
- Amendment 1 (2026-10-21, before any held-out shot is reviewed).
  (a) K8b's geometry guard allows corners up to 50% of the frame width
  beyond either side; K8's guard is unchanged. (b) When the best fit
  exceeds 2 px, it is re-seated on the model's four line-swap pairings:
  singles↔doubles sidelines and service lines↔baselines, as model-space
  scalings about the court centre. This is the same one-line-off
  ambiguity as K8's class-2 misses. (c) Each frame computes the distance
  transform once per mask, which changes speed only. No threshold
  changes. Because the fix was found on a held-out clip,
  `djokovic_zverev/000` (3,975 frames, 66 s) becomes dev: it is still
  registered (K4b needs it) but is neither reviewed nor scored. The
  held-out census is the other 69 clips.
- Runs (CPU only, no GPU; packed environment `32e68872…`; manifest SHA256
  `82a582e0246f1a3a0fce2fbe1f694097fe7b47dca8cf774cb3f897b3a51ae00a`).
  The corrected 4-part pilot `20261020T220814Z-c4b9d013` (`0465976`,
  gpu1) passed its smoke (240 frames, 524 s, about 2.2 s per frame per
  process) and reached its 2,700 s full-stage cap at 25 of 34 chunks.
  That code published clips only at the end, so `partial.tar` is empty;
  its `checkpoint.tar` keeps the finished chunks. The artifacts are
  retained and nothing was replayed. Resized to 8 parts × 16 processes
  with 200-frame chunks; each clip is finalized and published as soon as
  its chunks finish, longest clips first (`913b815`). Pilot part 0,
  `20261020T225829Z-3cb8b9aa` (gpu6): smoke 160 frames in 183 s; full
  stage 6 clips, 4,484 frames, in 499 s. Its validator passed on the
  smoke (fleet) and on the full stage (locally, from `published.tar`).
  Parts 1–7 were then submitted on the same revision (`git_head` checked
  in each `ready.json`):

  | Part | Job | Archive SHA256 |
  |---|---|---|
  | 0 | `20261020T225829Z-3cb8b9aa` | `3910621c…` |
  | 1 | `20261020T231141Z-bff7fd07` | `8e61d922…` |
  | 2 | `20261020T231147Z-a44fc8b0` | `86c30e3e…` |
  | 3 | `20261020T231153Z-d859538d` | `8e7d98c6…` |
  | 4 | `20261020T231200Z-23e16ed2` | `d1652905…` |
  | 5 | `20261020T231206Z-50159441` | `db7ad8b0…` |
  | 6 | `20261020T231213Z-19ea5d2d` | `c0317166…` |
  | 7 | `20261020T231220Z-a7110d20` | `0e4df826…` |

  All eight full stages completed (CPU, no GPU). They took 485–564 s on
  gpu5 and gpu6, and 2,446–2,599 s for parts 5–7, which shared gpu3
  (64 cores). Each passed its validator on the smoke (fleet) and on the
  full stage (locally). Assembled with `k8b assemble` (`a87dd2e`): 70
  clips and 38,142 frames, each clip exactly once, every review image
  present. Review `review/` under the run root, ID `13f69f53873866a2`,
  `k8b-report.json` SHA256 `8230bf1a2239978dfb7912bd5f3ed8367f5c85d1ea9e75c42e8ab8fb3fdc3314`,
  `review.html` SHA256 `a02a45a1a3a064fac69a95693699869650bdc2da3cd029b61d11685988b498f7`.
  Every clip forms a single shot: no cut at ≥ 0.25 and no persistent
  change of fit state inside a clip. 69 shots go to the user (the dev
  clip is excluded). 44 are predicted court (723 s) and 25 non-court
  (45 s, all medvedev_struff). These are predictions, not accuracy. The
  page is served read-only on gpu1 loopback (port 17690) through a
  tunnel, and was checked at 0 / 69 marks; the agent supplied no labels.
- **Outcome (2026-10-21): adopted.** Scored with `k8b score`
  (`76a7db2`) on the user's marks `k8b-marks-13f69f53873866a2.json`
  (SHA256 `8f7f665c…3c0dcd`, 69/69 marked) against `k8b-report.json`
  (`8230bf1a…`). Score `k8b-score.json` under the run root, SHA256
  `a914eceed27daf30c46881ee336b4c5505695afa9093830c4db990d2d00b2675`.
  The jobs, inputs and revision are in the table above (CPU only).

  | Condition | Bar | Measured | Passes |
  |---|---|---|---|
  | Precision (44 predicted court shots) | ≥ 0.95 | 1.00 (44/44) | yes |
  | Duration recall (723 s of true court) | ≥ 0.90 | 1.00 | yes |
  | Registration median, 31,914 true court frames | ≤ 2 px | 0.77 px | yes |
  | Registration 95th percentile | ≤ 6 px | 2.71 px | yes |

  Every shot matches the user's class. alcaraz_hurkacz has 21 court
  shots and djokovic_zverev 7. medvedev_struff has 16 court shots plus
  21 close-ups, 2 crowd shots and 2 graphics, all predicted non-court.
  The hypothesis holds: shade and the per-frame 2 px gate caused K8's
  misses. The competing explanation (failure on other surfaces, or
  spurious fits on texture) is not observed: the method held on hard
  courts and a court-level camera. Scope, as stated in the rule: precision
  rests on 44 shots (one-sided 95% lower bound 0.934, Clopper–Pearson).
  Only 25 held-out shots are non-court, and no replay or picture-in-picture
  appears. The clips were pre-cut by the old pipeline, so they test
  in-clip cuts and fit-state splits only weakly. Registration is measured
  with the centerline ruler (rule, item 3).
  **What K4b's candidate (d) can use now:** per-frame court homographies
  (ITF metres to 1080p pixels; null where there is no fit) and the court
  outline for every frame of these 70 clips, including the dev clip
  `djokovic_zverev/000`. They are in `clips/*.json` inside each part's
  `published.tar` and in the assembled `review/k8b-report.json`
  (`registration[n].homography`, n = the clip's decoded frame index).
  Still to run with the same adopted method: the clips of
  djokovic_federer and rublev_auger-aliassime (the two other matches K4b
  draws from), and nadal_kyrgios's whole match (v1 of the court-shot
  list).
- Application to K4b's other two matches (2026-10-21, the adopted method
  unchanged; a production run, not new evidence for the rule).
  djokovic_federer (23 clips, 4K) and rublev_auger-aliassime (66 clips),
  `<match>/videos/*.mp4`: 89 clips, 31,390 frames. Manifest
  `k8b-2026-10-20/k4b-matches/manifest.json` (SHA256 `6c2be1a4…`). Eight
  parts of whole clips (`3aca368`; CPU only, 16 processes each):
  `20261021T092159Z-9b8ef899`, `…092204Z-1ee9605a`, `…092208Z-804e7d30`,
  `…092315Z-1a7be77c` (part 3), `…092233Z-e808e648`, `…092238Z-019a88a8`,
  `…092243Z-41be5f46`, `…092247Z-91b9e709`. The first submission of part 3
  (`20261021T092213Z-112d5db2`) stopped during source upload: the
  directory holds only a partial `source.tar`, with no spec, ready
  record, owner or run. No worker could claim it and nothing executed,
  so part 3 was submitted once more. The partial directory is left as
  it is.
  All eight full stages completed and passed the validator on the smoke
  (fleet) and on the full stage (locally). Assembled at
  `k4b-matches/registration/k8b-report.json` (SHA256
  `eb7b34b9f8117f5926fb9e15d34bde9a209ce43213e19dc9d9641b38b552e3e0`):
  89 clips, 92 shots, 31,390 frames. Frames that fit: djokovic_federer
  13,710/13,901, rublev_auger-aliassime 10,264/17,489. The six
  rublev_auger-aliassime clips K4b needs that do not fit (003, 006, 039,
  054, 062, 063) were inspected visually. They are close-ups, a tunnel,
  and a player in front of a wall or by the sideline: not court views,
  so the absence of a fit is correct. 061 (fits on 34% of frames) is a low
  courtside view. These shots are not marked (the method is adopted), so
  this is an inspection, not evidence. Consumers use frames whose `fits`
  flag is true; a non-null homography above 6 px (e.g. in 003) is kept
  for inspection only.
