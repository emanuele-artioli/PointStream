# Experiments: G1e

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md).

### 2026-10-09 — G1e: reference refresh on VISOR
- Question: G1d ruled out a VISOR background reference that is sent once and
  reused for seconds. Binned by age, G1d's records explain far more with
  fresh references (windows under 0.1 s: 70% with `epi`, 49% with four
  planes). Can a reference refreshed often enough reach the 2 px bar, and
  how often must it be refreshed? If a sendable warp passes at a useful age,
  G2 prices the refreshes in bytes. If not, egocentric video goes to G5.
- Why a new measurement and not G1d's bins. In G1d, a frame's age was G1's
  choice: G1 started a keyframe where tracking failed, so fresh-reference
  frames are not a random sample. A bin also mixes clips. G1e fixes the
  targets and forces the age, so every age is scored on the same frames.
- Clips: G1d's 34 windows and 10 stretches, from G1d's prepared lossless
  archive (`20261009T075810Z-295529d4`, `published.tar` `5a9c47db…5d62`),
  with the same foreground. Measurement floor: the 7 OpenTTGames test clips
  of G1 (120 s at 10 frames/s, G1's `sam_text` masks of persons and rackets),
  prepared the same way in one more CPU job.
- Pairs. Targets are fixed before any run, on a stride, so that correlated
  neighbours do not count as separate samples. Windows: every 10th frame from
  1 s (about 20 per window, 0.2 s apart). Stretches and OpenTTGames: one per
  second from 1 s (119 per clip). Each target t is warped from the reference
  r = t − g for each age g in {1 frame, 0.1 s, 1 s}. On windows (50 or 59.94
  frames/s) these are 1, 5 or 6, and 50 or 60 frames. On stretches and
  OpenTTGames (10 frames/s) one frame is 0.1 s, so they get two ages. Every
  age uses the same targets. A pair is matched by SIFT and a homography with
  at least 30 inliers, as in G1. A pair that does not match counts as not
  explained, and the matched share is reported.
- Methods, all from G1d (`depth.py`), with the reference r as the keyframe:
  `h1` (one homography; also its region defines holes, as in G1d), `planes4`
  (the sendable warp: a label map on the reference, fitted to the dense flow
  to the reference's companion 0.5 s later, then four homographies per
  frame), `epi` (dense flow projected onto epipolar lines; not sendable) and
  `tri` (depth of the reference triangulated from its companions, PnP per
  frame). Companions are chosen as in G1d, without G1's registration (the
  six nearest candidates to each companion time). An encoder that refreshes
  references can see 0.5–1 s ahead at the cost of that much latency, which
  the rate step (G2) would have to state.
- The oracle, and why it bounds what it gates. It gates refreshed
  static-scene warps: `planes4`, `tri`, and the depth representations G1d did
  not run (Depth Anything 3 depth, depth-augmented keyframes, 3DGS). G1d
  found that `epi` alone does not dominate (`tri` beat it on P03_120). So
  the oracle is the per-pair best of `epi`, `planes4` and `tri`: a pair is
  explained if any of the three explains it. By construction it dominates
  `planes4` and `tri`. It does not dominate unmeasured depth (DA3, 3DGS)
  by construction. The argument for them is that after `epi`, 87–89% of
  G1d's residual energy was independent motion, remainder and exposure, and
  a static-scene warp of any depth cannot remove those. Per pair, the
  share where `epi` alone matches the oracle is reported, as a check of
  that argument at fresh ages.
- Measures per pair and method: G1d's (`camera.residual` with
  `depth.complete`): p90 residual flow on textured background at 1080p,
  hole share against `h1`'s region, shares within 1/2/4 px, PSNR after flow,
  energy attribution, near-foreground share. A pair is *explained* when
  p90 ≤ 2 px and holes ≤ 10%. Rate is per frame: `planes4` 128 bytes plus its
  label map PNG per reference, and `tri` 24 bytes plus its 8-bit
  inverse-depth PNG per reference. The reference image itself is G2's to
  price. Render time: one CPU thread, warp construction. Two checks beside it:
  - *Measurement floor, two ways.* (i) OpenTTGames, a static camera, at
    0.1 s and 1 s: `h1` p90 and the explained share. (ii) On every VISOR
    pair, the forward-backward consistency of the DIS flow on textured
    background: the p90 at 1080p of |f(x) + b(x + f(x))|, where f is the
    flow from the target to the warped reference and b the flow back. It is
    computed for every warp, and the rule reads `epi`'s, the closest
    alignment. A pair whose own flow disagrees with itself by more than 2 px
    cannot certify 2 px.
  - *Mask confound.* The one-per-second targets rarely fall inside a
    stretch's evaluation window, so this check takes every stretch frame
    there (10 per second, from 1 s). VISOR's dense masks exist there, read
    from the B1b fill already staged for G1d. Each pair is scored twice:
    with G1's `sam_text` foreground and with the dense masks (dilated as in
    G1). The reference also needs a dense mask, so at 1 s only targets at
    least 1 s into the window count.
- Decision rule (`g1e.DECISION`, fixed before any fleet run). As in G1d, a
  method is meaningful in a group at an age when its median clip explains at
  least 50% of targets. A method's *refresh age* in a group is the largest
  tested age at which it is meaningful. Windows and stretches are judged
  separately.
  1. *Oracle gate.* If the oracle's refresh age on the windows is below
     0.1 s (only one frame, or none), a refreshed reference is no cheaper
     than coding every frame. No refresh scheme is built, and egocentric
     video goes to G5.
  2. *Sendable warp.* Otherwise, if `planes4` (or `tri`) has a refresh age of
     at least 0.1 s on the windows *and* on the stretches, G2 prices that
     warp refreshed at that age against SVT-AV1 and DCVC-UF on the same
     background frames. If several qualify, the one with fewer bytes per
     frame goes forward. The stretch verdict stands only if the mask check
     agrees: the dense and `sam_text` explained shares on the same pairs
     differ by at most 10 points. Otherwise the stretch verdict is the
     dense-mask one, on those pairs only, and is flagged as resting on
     fewer frames.
  3. If the oracle qualifies at ≥ 0.1 s but no sendable warp does, the gap
     is recorded, and the next step is a sendable warp between them, with
     DA3 or depth-augmented keyframes from G1d's (c) and (d), at the oracle's
     refresh age. That step gets its own entry.
  4. *Refinement.* If `planes4` or the oracle is meaningful at 0.1 s but
     not at 1 s on the windows, one more age, 0.3 s, is run on the same
     targets. This places the refresh age within a factor of about 3, since
     G2's rate scales with the refresh rate. No other ages are added.
  5. *Floor.* The growth with age counts as real only if OpenTTGames'
     median `h1` p90 stays ≤ 2 px at 1 s and VISOR's forward-backward p90
     stays ≤ 2 px at the age in question. Where it does not, that age's
     verdict is reported as *not measurable*, and it is not counted as a
     failure.
- Hypothesis: on the windows the oracle is meaningful at one frame and at
  0.1 s (about 65–75%), but not at 1 s. `planes4` is meaningful at one frame
  and borderline at 0.1 s (40–55%), so step 4 runs. The stretches stay below
  the windows at 0.1 s. The mask check closes part of that gap, not all of
  it. The floor holds at 0.1 s, and OpenTTGames holds at 1 s.
- Competing explanations: (1) G1d's fresh bins were easy frames chosen by
  G1's keyframing. Then forced ages give lower shares than G1d's bins at
  the same age. (2) A measurement floor: DIS between distant views degrades.
  Then forward-backward error grows with age as fast as the residual does.
  (3) Masks: the stretches' residual is unmasked hands. Then dense masks
  raise the stretch share on the same pairs.
- Budget: CPU only, 0 GPU-hours. OpenTTGames preparation ≤ 20 min. Dev check
  on gpu6 (packed environment, one window and one stretch, a few targets;
  not evidence) to time the pairs and check forward-backward consistency on
  synthetic and real pairs. Smoke ≤ 600 s. Pilot (2 windows, 1 stretch,
  1 OpenTTGames clip, every target) ≤ 20 min. Full run of the rest sized
  from the pilot and split into shards of ≤ 45 min each, checkpointed per
  clip (about 2,100 window pairs, 2,400 stretch pairs and 1,700 OpenTTGames
  pairs; G1d spent about 5.5 CPU-s per frame for eight methods with shared
  keyframes, and here every pair has its own reference). Refinement (0.3 s),
  if reached: one more job of ≤ 45 min. Ceiling 6 CPU-job-hours and 6 h
  wall.
- Before any fleet run (dev check on gpu6, packed environment, `9ae7334`;
  window P02_12 and stretch P26_02, three targets each; not evidence). The
  run works end to end, and the oracle is at or above each of its parts on
  every pair. A VISOR pair takes about 4.5 CPU-s, so the full run is about
  5,500 VISOR pairs (25,000 CPU-s), and fits one job of under 45 min at 48
  threads without shards. On P02_12, at one frame all four warps explain all
  three targets; at 0.1 s `epi`, `planes4` and `tri` do; at 1 s none does
  (`planes4` p90 5–18 px). Forward-backward p90 is 0.1–0.2 px at one frame,
  about 0.3 px at 0.1 s and 1.1–1.9 px at 1 s on the window (`epi`). On the
  stretch it is 0.8–2.5 px at 0.1 s and 3–14 px at 1 s, so step 5 may well declare the
  stretch's 1 s not measurable. `tri` has no reference depth on 7 of the
  stretch's 16 pairs: the relative pose to the companions fails (a near
  rotation), as in G1d. Inputs: `pointstream-data/background/g1e-2026-10-09/`
  (`inputs.json` `3d29aea5…b4e1`). It holds the seven OpenTTGames results and
  masks from G1's jobs `20261008T162927Z-d9372b99` and
  `20261008T203714Z-fd5d2f5b`, and G1d's prepared archive (`5a9c47db…5d62`).
  Another session had moved the fleet directory to
  `jobs/fleet-archive-20261009/`. It was moved back (doctor passed, workers
  restarted), and the recorded paths in `inputs.json` were corrected, which
  is why its sha256 changed from the first build (`643ee3b1…991f`).
- OpenTTGames preparation `20261009T194103Z-e71866e4` (gpu1, CPU, `9b0f8de`;
  smoke gate passed; full 211 s at 16 threads; validator passed, 9 checks):
  7 clips (300–1,200 frames; test_2, test_3, test_5 and test_6 are shorter
  than 120 s), lossless, foreground equal to G1's on every frame.
  `published.tar` `52d46d36…bffc`.
- Pilot `20261009T195037Z-d2a08aa1` (gpu1, CPU, `9b0f8de`; smoke gate
  passed; full 123 s at 32 threads; validator passed, 10 checks, including
  the oracle at or above each part on every pair): windows P02_12 and
  P03_120, stretch P26_02, OpenTTGames test_3. Explained share per clip
  (one frame / 0.1 s / 1 s):

  | Clip | planes4 | epi | tri | oracle | epi forward-backward p90 |
  |---|---|---|---|---|---|
  | window P02_12 | 1.00 / 1.00 / 0.11 | 1.00 / 1.00 / 0.39 | 0.94 / 0.89 / 0.11 | 1.00 / 1.00 / 0.39 | 0.13 / 0.21 / 0.75 px |
  | window P03_120 | 1.00 / 1.00 / 0.26 | 1.00 / 0.95 / 0.63 | 0.95 / 0.79 / 0.21 | 1.00 / 1.00 / 0.68 | 0.22 / 0.39 / 0.66 px |
  | stretch P26_02 (0.1 s = one frame) | 0.40 / 0.08 | 0.41 / 0.08 | 0.11 / 0.00 | 0.49 / 0.10 | 1.20 / 2.60 px |

  On the stretch, 31% of the 0.1 s pairs and 42% of the 1 s pairs do not
  match (fewer than 30 homography inliers), and by the rule they count as
  not explained. On the pairs that do match, `planes4` explains 58%
  at 0.1 s. Mask check on its 30 window-frame targets at 0.1 s: `planes4`
  explains 0.50 with the dense masks and 0.60 with `sam_text`, and the
  oracle 0.57 and 0.77, so the automatic masks do not hold this stretch
  down. OpenTTGames test_3: `h1` explains 97% at both ages, with p90 0.94
  and 1.20 px, so the measure holds on a static camera at 1 s. Nothing in
  the pilot changes the run's settings, so the other 41 clips run as
  specified: `20261009T195921Z-a5d08578`.
- Full run `20261009T195921Z-a5d08578` (gpu5, CPU, `9b0f8de`; smoke gate
  passed; full 903 s at 48 threads, no contention; validator passed, 10
  checks). Refinement at 0.3 s, reached by step 4: `20261009T201916Z-9abe4e90`
  (gpu5, CPU, `e5099d7`, all 51 clips; 441 s; validator passed). Report:
  `pointstream-data/background/g1e-2026-10-09/report-1ae0ded/`
  (`g1e-report.json` `fdd2ede3…8ff1`, from the pilot's, the full run's and
  the refinement's `g1e.json`: `a5727e3f…bd19`, `5a71bc95…a96b` and
  `6174d3d2…7e73`). Median clip's explained share (p90 ≤ 2 px and holes
  ≤ 10%; an unmatched pair counts as not explained):

  | Group | Age | h1 | planes4 | epi | tri | oracle | oracle p90 | epi forward-backward p90 |
  |---|---|---:|---:|---:|---:|---:|---:|---:|
  | windows (34) | one frame (0.02 s) | 1.00 | 1.00 | 1.00 | 0.68 | 1.00 | 0.43 px | 0.22 px |
  | windows | 0.1 s | 0.45 | **0.78** | 0.84 | 0.46 | **0.92** | 0.93 px | 0.41 px |
  | windows | 0.3 s | 0.11 | 0.36 | 0.44 | 0.11 | **0.54** | 1.80 px | 0.65 px |
  | windows | 1 s | 0.00 | 0.06 | 0.16 | 0.00 | 0.17 | 3.47 px | 1.30 px |
  | stretches (10) | 0.1 s (one frame) | 0.31 | 0.39 | 0.49 | 0.27 | **0.51** | 1.53 px | 0.55 px |
  | stretches | 0.3 s | 0.06 | 0.18 | 0.28 | 0.08 | 0.31 | 2.87 px | 1.08 px |
  | stretches | 1 s | 0.01 | 0.03 | 0.09 | 0.03 | 0.10 | 4.69 px | 1.99 px |
  | OpenTTGames (7), `h1` | 0.1 / 0.3 / 1 s | 1.00 / 1.00 / 0.99 | | | | | 0.84 / 0.97 / 1.03 px | 0.51 / 0.58 / 0.66 px (`h1`) |

  Clips with at least half explained, `planes4` / oracle: windows at 0.1 s
  22 / 27 of 34, stretches at 0.1 s 4 / 6 of 10. Per clip at 0.1 s,
  `planes4` explains 0–100% of a window's targets (11 windows all of them)
  and 16–88% of a stretch's. The stretches' pairs mostly match (median
  clip 95% at 0.1 s, 75% at 1 s). Their shortfall is residual on matched
  pairs: `planes4` explains 49% of them and the oracle 65%. `epi` alone
  gives the oracle's verdict on 95–100% of a median clip's pairs, so the
  best-of construction mattered little here. `planes4` costs
  128 bytes per frame plus a label map per reference.
- Outcome by the rule: **step 3**.
  - **Refresh ages.** The oracle's are 0.3 s on the windows and 0.1 s on
    the stretches. `planes4`'s is 0.1 s on the windows (78%) and none on
    the stretches (39% at 0.1 s). `tri`'s is one frame on the windows, and
    none on the stretches.
  - **No sendable warp qualifies in both groups**, so G2 does not yet
    price a refresh scheme. The rule's next step is a sendable warp
    between `planes4` and the oracle, at the oracle's refresh age (DA3 or
    depth-augmented keyframes), with its own entry. Its ceiling is the
    oracle's: on the stretches, 51% at 0.1 s, only just above the bar, and
    31% at 0.3 s.
  - **Floor (step 5).** It holds at every age: OpenTTGames' `h1` p90 is
    1.03 px at 1 s, and VISOR's forward-backward p90 is at most 1.99 px.
    The growth with age is therefore real. The residual grows from 0.93
    to 3.47 px on the windows between 0.1 s and 1 s, while the flow's
    self-disagreement grows from 0.41 to 1.30 px.
  - **Mask check.** It holds: dense and `sam_text` masks give the same
    shares within 2 points, so the stretches' gap to the windows is
    content, not masks.
- What the remaining residual is made of (`epi`, energy share of the raw
  squared error, mean over pairs). On the windows at 0.1 s: remainder 70%,
  independent motion 21%, parallax 7%, exposure 2%. At 1 s the shares are
  46%, 34%, 13% and 7%. On the stretches at 0.1 s: 63%, 27%, 7% and 2%.
  Misaligned pixels near the foreground: 5–7%. A fresh reference removes the
  geometric part almost entirely. What stays is photometric (blur, noise,
  lighting, disocclusion) and things that move by themselves, which no
  static-scene warp can remove.
- Hypothesis: partly right. The oracle was meaningful at one frame and at
  0.1 s on the windows, and not at 1 s, as predicted. It was stronger than
  predicted (92% at 0.1 s, against 65–75%), and so was `planes4` (78%,
  against 40–55%). The stretches stayed below the windows, as predicted.
  The mask check closed none of the gap, where part was predicted. The
  floor held, as predicted.
- Competing explanations. (1) Selection by G1's keyframing: rejected. Forced
  ages give *higher* shares than G1d's age bins (window bin under 0.1 s:
  `epi` 70%, four planes 49%). (2) A measurement floor: rejected. The flow's
  self-disagreement stays under 2 px and grows a third as fast as the
  residual, and OpenTTGames holds at 1 s. (3) Masks: rejected (above).
- Budget: CPU only, 0 GPU-hours. CPU jobs: preparation 211 s, pilot 123 s,
  full 903 s, refinement 441 s, so about 28 min of CPU-job wall (ceiling 6
  CPU-job-hours). Wall time from the first submission to the report: about
  50 min (ceiling 6 h).
- After review (user, 2026-10-09): the rule's step-3 component is *not*
  built. Its ceiling on the stretches is the oracle's 51% at 0.1 s, a
  narrow pass at best, for a reference refreshed ten times a second.
  Egocentric background work moves to G5. The parked options and what
  would reopen them are in PLAN (G1e).
