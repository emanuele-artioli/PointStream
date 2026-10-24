# Experiments: K6b (hands)

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md).

### 2026-10-22 — K6b (hands) 1: audit of hand renderers
- Question ([PLAN K6b](../../PLAN.md), hands after K1f). K1f's bound moves
  the hand's own pixels to the 21 coded keypoints. A renderer has to draw
  them from a reference instead. Which released models can do that, and
  which one does the zero-shot rung test?
- Interface, fixed by K1f. Per hand and frame the renderer gets a
  reference crop of that hand and the 21 coded 2D keypoints (H2's 100 ms
  code, 1.93–2.72 kbps per item). Its pixels are composited inside R, the
  granted outline, with the source outside R (`k1c.compose`). So it has to
  draw a hand crop, not a scene. H3e's failure was that the scene drifted:
  JointControlVideo generated the whole frame from the first frame and
  never saw the camera's motion. The local interface removes that failure.
- Criteria: conditioning on a reference image and 2D (or 3D) hand
  keypoints; released weights; licence for research; inference cost per
  frame; whether it has seen VISOR (overlap is allowed, recorded, and the
  clean subset reported, per AGENTS.md).

  | model | conditioning | weights, licence | cost | verdict |
  |---|---|---|---|---|
  | [FoundHand](https://github.com/arthurchen0518/FoundHand) (CVPR 2025) | reference image (mask optional), 2D keypoints as heatmaps (OpenPose 21 per hand); per frame, with a released zero-shot video procedure (first frame plus recent generated frames) | released (Google Drive), GPL-3.0 code, CC BY-NC data | DiT on SD-VAE latents, 256² (32² latent); timing not published | **chosen** |
  | K6's `ref` (no learning) | the reference crop's pixels warped from its keypoints to the coded ones | in repo | CPU | **floor**: run beside, so the generator has to beat a warp of the same reference |
  | Moore-AnimateAnyone, original and tennis fine-tune (in `Models`) | reference image, DWPose maps of the full body | in `Models`, Apache-2.0 | SD 1.5, about 1 s per frame | rejected for hands: whole-body model, and FoundHand's paper reports it at LPIPS 0.35 against FoundHand's 0.17 on hand videos (their Table 3b, 12 phone videos). It stays K6b's body candidate |
  | SD 1.5 + ControlNet OpenPose (in `Models`) | text, OpenPose hand map; a reference only through an adapter | in `Models` | per frame | reserve: no reference conditioning trained for hands |
  | JointControlVideo (H3e, in `Models`) | first frame, 3D joints, camera intrinsics | Apache-2.0 | 27 min per 81 frames | rejected: H3e failed (−297%), and the cause was the scene drift |
  | [Generated Reality](https://github.com/LafouCC/generated-reality) (2026) | first frame, 21 3D joints per hand in camera space, head pose | MIT code; trained checkpoint not released (base Wan2.2-Fun-5B-Control) | 5B video DiT | not zero-shot: needs training on HOT3D/GigaHands first |
  | Re-HOLD (CVPR 2025), iDiT-HOI (2025), ManiVideo, RHanDS | hand–object layouts or meshes | no released weights found | | not usable |

- Overlap. FoundHand's repository lists VISOR among FoundHand-10M's
  sources, and VISOR's public annotations are its train and val splits,
  which is where B2's items come from. Before any score is read, the
  zero-shot rung checks the released data lists for each item's video.
  Items whose video is listed are reported as seen. If the lists cannot
  tell, every item counts as seen, and the claim is scoped to "a renderer
  that may have seen this footage". The judges (LPIPS's AlexNet, DISTS's
  VGG) have not seen it.
- Decision: FoundHand is the zero-shot candidate, with `ref` as the floor.
  Rung 2 needs its source at a pinned commit and its weights in
  `Models/FoundHand` (with a sha256 manifest), and an environment check.
  Its pins (Python 3.9, torch 2.3.0, lightning 2.3.0, timm 1.0.7) are older
  than the main environment's (torch 2.10, timm 1.0.30). It is plain DiT
  code, so the main environment plus lightning is tried first. A second
  environment is made only if an import or the smoke's output check fails.
- Searches (2026-10-22): FoundHand's repository and paper, Generated
  Reality's project page and repository, Re-HOLD's project page, and web
  searches for hand reenactment with released weights. Nothing newer than
  H3e's audit has released weights for a reference-plus-keypoint hand
  renderer, other than the fact that Generated Reality's code is out.

### 2026-10-22 — K6b (hands) 2: zero-shot oracle (rule)
- Question. With one unbilled reference crop per hand and the coded
  keypoints, does FoundHand, off the shelf, keep K1f's win over the pixel
  hull? K1f's bound (0.084 median LPIPS on R over its first 22 items;
  0.086 over all 30) is the ceiling, and the hull at each item's keypoint
  rate (median 0.175; 0.174 over 30) is the bar.
- Items. Positions 0–7 of K1f's content-blind order
  (`tools/datasets/k1f_inputs.py`), 240 frames each. All 8 have K1f's
  bound and hull recorded, so nothing is re-coded. Items in G's training
  footage and items whose video FoundHand has seen are reported apart,
  beside the pooled numbers. Development (crop rule, keypoint order,
  sampler settings) is done on K1c's pilot item P01_107 only. It is not
  among the 8.
- Keypoints and fits. K1f's, unchanged: H2's WiLoR rows plus the two-pass
  hand-box fit, coded with H2's 100 ms choice and projected to 21 joints.
- Reference (granted, unbilled, as K6b's step 2). For each hand side, the
  window's first frame where that side is fitted. The crop is the same as
  for the targets.
- Crop. Per hand and frame, a square around the union of the coded
  keypoints' box and that side's mask, ×1.25, resized to 256². The
  keypoints are mapped into the crop, in OpenPose order, with the right
  hand in joints 0–20 as FoundHand's labels have it. The output is
  resized back to the box (Lanczos).
- Variants, each composited by `k1c.compose`: inside that side's R the
  rendered pixels, Telea fill where no fit exists (K1c's charge), and the
  source outside R. R is the generator's matte, granted, as `warp`
  granted the outline.
  - `gen`: FoundHand with its released video procedure (the reference
    as first frame, then recent generated frames), at its released
    sampler settings.
  - `gen@frame` (diagnostic): FoundHand per frame from the reference
    alone. It separates the autoregressive drift from the per-frame
    error.
  - `ref` (floor): K6's warp of the reference crop's pixels from its
    uncoded keypoints to the coded ones.
  - `warp` reproduced (check): K1f's bound. It must match K1f's recorded
    LPIPS on R within 0.002 per item, or nothing is read.
- Judges, as K1f: LPIPS on R and DISTS on R (`h3c.Scorer`), mean over the
  window's frames with R non-empty. Reported beside: PSNR on R, flicker
  (mean absolute luma change between consecutive frames inside R,
  against the source's), WiLoR on the rendered crop against the coded
  keypoints (does it follow its input), and a visual check of one frame
  per item.
- Decision rule (`k6bh.DECISION`, fixed before any run; the user's
  criterion). An item is won on a metric when `gen`'s value on R is below
  the hull's at that item's keypoint rate (K1f's report). Score = mean
  over {LPIPS, DISTS} of the share of the 8 items won.
  - **Pass** (score ≥ 75%): the zero-shot route holds with the reference
    granted. The overfit rung is skipped. Next is the same oracle with
    the reference billed (coded once per window and its bits added), the
    interface K6c would use.
  - **Fail** (score < 75%): the overfit rung (a fit of FoundHand, or a
    LoRA on it, to the window's own frames) gets its rule and budget,
    with the gap from `gen` to the bound reported per item.
  - If `ref` scores ≥ 75%, a warp of one reference already suffices.
    That goes to the user, with the better of `ref` and `gen` per item.
  - If the score is under 75% on the unseen items alone but at or above
    on the seen ones (or the reverse), the outcome goes to the user.
- Hypothesis. `gen` lands near the bar, with a score around 50%. FoundHand
  reports LPIPS 0.17–0.20 on reposing at 256², which is about the hull's
  median, and DISTS forgives texture more than LPIPS does. `ref` scores
  lower: it is K6's failure mode (what the pose cannot describe).
- Competing explanations. (1) Having seen VISOR makes FoundHand better
  here than on new footage. The seen/unseen split shows it if the lists
  allow one. (2) The granted matte hides the generator's outline errors,
  as it did for `warp`, so a pass is still a bound on the matte. (3) The
  256² crop is the limit, not the generator. Large hands (box > 512 px)
  are compared with small ones beside the result. (4) Per-frame metrics
  ignore flicker, so flicker is reported, and a visual check is made of
  any item that passes with flicker above the source's.
- Before the run (CPU): the FoundHand source at a pinned commit and its
  weights into `Models/FoundHand` with a sha256 manifest; its import in
  the main environment (plus lightning); the overlap check above.
- Budget. GPU, RTX 6000 Ada (FoundHand's first run on the class). A smoke
  of at most 600 s on P01_107, 24 frames: it asserts CUDA, the dtype,
  the attention kernel and the output's range, and times one frame. The
  timing is recorded here before the pilot. The released step count is
  kept if one 240-frame item (both hands, `gen` and `gen@frame`) fits in
  40 minutes; otherwise the most steps that fit, not below half the
  released number. Then a pilot on P01_107 (development, not scored for
  the rule), and the 8 items in parts of at most 45 minutes, each saving
  every finished item. Ceiling 3 GPU-hours for the rung, smoke included.
  If the budget runs out, the items finished in the fixed order are
  reported, and so is the shortfall.
- Revision (2026-10-22, after reading FoundHand's source, before any run).
  - Source and weights. FoundHand at `ae0e980` (archive
    `visor/k6bh-2026-10-22/inputs/foundhand-src-ae0e980.tar.gz`,
    `fe8ff8a0…7f35`). The released checkpoint
    `DINO_EMA_11M_b50_lr1e-5_epoch6_step320k.ckpt` (11.6 GB, `66434caf…62d7`)
    and SD's `vae-ft-mse-840000-ema-pruned.ckpt` (`c6a580b1…3dab`, as Hugging
    Face lists it) are in `Models/FoundHand` with a `MANIFEST.json`,
    hard-linked into the run's `inputs/`. Inference uses the DiT-XL/2's
    `ema_state_dict` (as the demos do) and needs only torch, numpy and
    timm. Lightning, SAM and mediapipe serve training and the demos' own
    keypoints and masks, which K1f supplies. So the main environment
    (`pointstream-20261009T214002Z`) runs it, and no environment is added.
    The DiT runs under bf16 autocast, and the VAE in float32.
  - Steps. The released demos sample with 250 respaced DDPM steps and
    guidance 2.5. At that count the 8 items cannot fit the 3 GPU-hours. A
    pilot on P01_107 (development, not scored) runs all four variants at
    50 steps and records seconds per step. The step count for the 8 items
    is the largest of {250, 100, 50} whose projected cost (both FoundHand
    variants) fits the remaining ceiling with a 25% margin. If 50 does not
    fit, `gen@frame` runs on the first 2 items only. If `gen` alone does not
    fit at 50, the rung goes back to the user. The count is recorded here
    before the 8 items run.
  - Keypoints. Read from K1f's published rows (`row_t`, `row_right`,
    `uncoded_kp2d`, `coded_kp2d` of each item's K1c unit; P01_107 from the
    reproduction job `20261022T013208Z-9e048ecf`). They are exactly the
    keypoints K1f scored, with no second WiLoR run. K1f did not publish
    joint depths, so overlapping hands are drawn smaller first (smaller
    taken as farther). The `warp` check against K1f's recorded value
    tests that too.
  - Overlap check. FoundHand-10M's file lists are only on Globus, which
    needs a login, so the lists cannot tell. Every item counts as seen, and
    the claim is scoped to a renderer that may have seen this footage
    (VISOR's annotated frames, not necessarily these windows).
  - Dropped before any run: "WiLoR on the rendered crop" (reported beside).
    It would add WiLoR to the job for a number that decides nothing.
    Flicker and the visual check stay.
  - Handedness. The README and the demos put the true right hand in slots
    0–20, un-mirroring MediaPipe, so no ordering check is needed.
- Attempts and pilot (development on P01_107, not scored for the rule;
  timings from a shared GPU, contaminated, not evidence).
  - `20261022T080212Z-35880d37` (`2344973`, gpu5): the smoke stopped while
    loading the checkpoint. It pickles the training script's `HandDiffOpts`
    from `__main__`. Fixed in `796e6e0` by loading with `__main__` classes
    stubbed (inference reads only `ema_state_dict`). Infrastructure, no
    score.
  - `20261022T080535Z-df3082c3` (`796e6e0`, gpu5, RTX 6000 Ada, shared):
    the smoke and the full stage pass (the full stage locally, 17 checks).
    First run on Ada: flash attention, bf16 CUTLASS GEMMs on sm80 kernels,
    35 ms per guided step (two samples), 675.6 M parameters, no missing
    keys. `warp` reproduces K1f exactly (LPIPS on R 0.08647 against
    0.08647). The full item took 2,045 s: `gen` 1,008 s, `gen@frame`
    488 s, and decoding, `ref`, `warp` and scoring about 550 s.
  - Pilot values on R (P01_107, 1.95 kbps; K1c's hull there reads 0.258
    LPIPS and 0.223 DISTS): `gen` 0.496 / 0.255, `gen@frame` 0.463 /
    0.258, `ref` 0.575 / 0.279, `warp` 0.086 / 0.069. Flicker (luma, R):
    source 6.6, `gen` 10.8, `gen@frame` 31.0. On its first 24 frames
    (the smoke), `gen` read 0.314 / 0.201. The later frames are much worse
    than the first ones.
  - Visual check (frame 120). Where a crop ran past the frame's edge (the
    forearms at the bottom), the zero padding showed up as a black band in
    FoundHand's output and was pasted onto R. Fixed in the next commit:
    the square crop is shifted inside the frame. The generated hands are
    plausible but not the source's: the fingers and the forearm's position
    differ.
  - Budget. Spent so far: about 0.7 GPU-hours of the 3. At 50 steps, `gen`
    with `ref` and `warp` costs about 26 min per item, so the 8 items need
    about 3.5 GPU-hours. **`gen` alone does not fit at 50 steps. By the
    rule, the rung goes back to the user.**
- Decision (user, 2026-10-22, before any scored run): **4 items at 50
  steps**: positions 0–3 of K1f's order (P09_103, P25_09, P22_107 (G's
  training footage), P06_108), with `gen`, `ref` and `warp`, and without
  `gen@frame`. One item per job, about 1.7 GPU-hours, within the 3. The
  rule is unchanged otherwise: the score is the mean over {LPIPS, DISTS}
  of the share of the 4 items won, and it passes at ≥ 75% (3 of 4 on
  both, or 4 and 2). Code `466f73f` (crops inside the frame).
- Jobs (RTX 6000 Ada, shared, so timings are contaminated; code from each
  job's `ready.json`). P06_108 `20261022T084910Z-abd3eff7` (`1320465`).
  P22_107 `20261022T084905Z-09c2f882` (`1320465`). P25_09
  `20261022T090959Z-8ccce208` (`7e75675`: the GPU cache is released
  between rendering and scoring, and 32,000 MiB is declared). The
  rendering is the same in both revisions. Every smoke passed the fleet
  validator, and every full stage passes `k6bh validate` locally. Each
  stage's `warp` reproduces K1f's recorded LPIPS on R within 0.0001.
  P25_09's `ref` covers 49% of R, under the validator's first coverage
  threshold (50%), which was meant to catch an empty drawing. It was
  lowered to 10% for `ref` after that stage was read (`1f47d1d`). The
  decision reads `gen` only, which covers 100%.
  Infrastructure failures (no score): P25_09 `20261022T084900Z-7e08ab8f`
  overran its declared 16,000 MiB (24,886), and P09_103
  `20261022T084856Z-539368dd` overran it too (22,058). P09_103's rerun
  `20261022T092110Z-d37eedfb` overran 32,000 MiB (38,926) while scoring.
  Report `visor/k6bh-2026-10-22/report/k6bh-report.json` (`e32546d7…2b4f`),
  with the local validations in `k6bh-validation.tar` (`8c97643f…0a0f`).
  GPU time: about 2.7 GPU-hours of the 3, failed attempts included.
- Outcome. 3 of the 4 items were scored. P09_103 was not: by then it
  could not change the decision, and the budget was nearly spent. On R,
  at each item's keypoint rate:

  | item | kbps | `gen` LPIPS / hull | `gen` DISTS / hull | `ref` LPIPS / DISTS | `warp` LPIPS / DISTS | crop (median px) | flicker `gen` / source |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | P06_108 | 1.94 | 0.308 / 0.172 | 0.273 / 0.206 | 0.391 / 0.307 | 0.091 / 0.123 | 550 | 19.7 / 14.4 |
  | P22_107 (G) | 1.93 | 0.281 / 0.236 | **0.200** / 0.204 | **0.191** / **0.125** | 0.059 / 0.049 | 742 | 6.5 / 3.8 |
  | P25_09 | 2.14 | 0.524 / 0.260 | 0.353 / 0.240 | 0.429 / 0.297 | 0.097 / 0.102 | 1,067 | 11.0 / 3.2 |

  (Bold: below the hull. G: in G's training footage. FoundHand may have
  seen every item, so the overlap could not be resolved.)
  - `gen` wins 0 of 3 items by LPIPS and 1 of 3 by DISTS: **score 17%**.
    With P09_103 won on both, it would be at most 37.5%.
  - `ref` (a warp of one reference) wins 1 of 3 on both metrics (33%), and
    beats `gen` on P22_107 and P25_09 by LPIPS. The pilot item
    P01_107 (development) read `gen` 0.496 / 0.255 against a hull of
    0.258 / 0.223.
  - `warp`, the bound, wins 3 of 3 on both (it reproduces K1f).
- Decision (by the rule): **fail.** FoundHand off the shelf does not keep
  K1f's win. A zero-shot renderer with one unbilled reference loses to
  the pixel hull at about 2 kbps on every scored item by LPIPS. The next
  rung (overfit) gets its rule and budget below. The decision does not
  depend on the items FoundHand may have seen, since none could be
  separated.
- Reading. The gap from `gen` to the bound (0.19–0.43 LPIPS) is much
  larger than the gap from the bound to the hull. FoundHand's hands are
  plausible but not the source's: the fingers and the forearm are drawn
  where its 256² crop puts them, and it flickers more than the source
  (6.5–19.7 against 3.2–14.4). That `ref` sometimes beats it says that
  where one reference is close in pose, real pixels beat generated ones.
  The forearm, which is in R, is a large part of what neither renders
  (K1c's R includes it). The largest crops (P25_09, about 1,070 px down
  to 256) lose the most.
- Hypothesis ("`gen` lands near the bar, score around 50%"): refuted.
  The score is 17%. Competing explanation (3), the 256² crop, fits the
  ordering by crop size but was not isolated. (1), having seen VISOR,
  cannot be tested, and the result is a fail despite it.

### 2026-10-22 — K6b (hands) 2b: a bank of free references (rule)
- Question (user, 2026-10-22, after rung 2). Rung 2's `ref`, one free
  reference warped to the coded keypoints, beat FoundHand on 2 of 3 items
  and the hull on 1. Can a few real references, each frame warped from
  the one nearest in pose, carry the hand's appearance well enough to beat
  the hull at the keypoint rate? H3a2 already showed that refreshed
  references coded as their own stream lose to SVT-AV1 at 40–140 kbps.
  This tests the corner H3a2 left open: very few references, at about
  2 kbps.
- Why it bounds. The references are the window's own frames, uncoded and
  unbilled. A real bank (coded, billed, or sent once per video from
  outside the window) can only do worse. If this fails, the reference
  route for hands closes at every rate this side of H3a2. If it passes,
  the next test is the buildable form: a bank chosen outside the window,
  coded, and billed once per video.
- Variants, on rung 2's 3 scored items (P06_108, P22_107, P25_09), with
  P01_107 for the smoke. K1f's rows, R as the matte, and Telea fill where
  nothing is drawn, as rung 2.
  - `bank@K`, K ∈ {1, 4, 16} references per hand side. Chosen at the
    encoder by farthest-point sampling on the uncoded poses (wrist-centred,
    scaled by the pose's extent), starting from the side's first fitted
    frame. Per frame, the decoder picks the reference whose coded pose is
    nearest, and warps its side mask's pixels from the reference's coded
    keypoints to the frame's (`k6.draw`). Only what a decoder holding the
    references would have is used.
  - `warp`: K1f's bound, the check (within 0.002 of K1f's LPIPS on R).
- Decision rule (`k6bh.BANK_DECISION`, fixed before any run; the user's
  criterion). Per K, an item is won on a metric when `bank@K` on R is
  below K1f's hull at the item's keypoint rate (the references free).
  Score = mean over {LPIPS, DISTS} of the share of the 3 items won.
  - **Pass** (some K scores ≥ 75%, i.e. all 3 items on one metric and at
    least 2 on the other): the next test is the buildable bank (outside
    the window, coded, billed once per video), with its own rule.
  - **Fail** (no K reaches 75%): no bank of real references beats the
    hull at the keypoint rate. With H3a2, the reference route for
    egocentric hands closes. Rung 3 (a generator fitted per clip) is then
    the last open rung, and it waits for the user.
- Hypothesis. `bank@16` passes, and `bank@1` does not (rung 2's `ref`
  was 1 of 3). Hands show few distinct poses within 4.8 s, so 16
  references cover most frames.
- Competing explanation. The coding error of the keypoints (7–12 px)
  plus the warp's blending of fingers, not the reference's pose, keeps
  the warp above the hull. Then `bank@16` stays near `ref` and far above
  `warp`, even with nearer references.
- Budget. No generation. GPU only for LPIPS and DISTS (four scorings
  per item). One job per item, ≤ 45 min each, declared at 40,000 MiB (rung
  2's scoring peaked at 38,926). Each job's smoke (at most 600 s) runs
  24 frames of its own item, so P01_107 is not needed. Ceiling 0.75
  GPU-hours.
- Jobs (RTX 6000 Ada, gpu6, shared; code `e0d4df5` from `ready.json`):
  P06_108 `20261022T105912Z-875474c1`, P22_107 `20261022T105916Z-287e2bd3`,
  P25_09 `20261022T105921Z-160d3f85`. Every smoke passed the fleet
  validator, and every full stage passes `k6bh validate` locally, with
  `warp` reproducing K1f. Report
  `visor/k6bh-2026-10-22/report/k6bh-bank-report.json` (`c4c0b472…5b70`),
  with the validations in `k6bh-bank-validation.tar` (`377b2202…7c39`).
  GPU time: 2,080 s of stages, about 0.6 GPU-hours with the smokes,
  within the 0.75.
- Outcome. On R, at each item's keypoint rate, the references free:

  | item | hull LPIPS / DISTS | `bank@1` | `bank@4` | `bank@16` | `warp` |
  |---|---:|---:|---:|---:|---:|
  | P06_108 | 0.172 / 0.206 | 0.416 / 0.322 | 0.344 / 0.276 | 0.227 / 0.209 | 0.091 / 0.123 |
  | P22_107 (G) | 0.236 / 0.204 | **0.198 / 0.128** | **0.173 / 0.116** | **0.100 / 0.078** | 0.059 / 0.049 |
  | P25_09 | 0.260 / 0.240 | 0.429 / 0.296 | 0.293 / **0.221** | **0.190 / 0.159** | 0.097 / 0.102 |

  Scores (mean over {LPIPS, DISTS} of the share of items won): `bank@1`
  33%, `bank@4` 50%, `bank@16` 67%. Clean of G's footage (P06_108,
  P25_09): 0%, 25%, 50%.
- Decision (by the rule): **fail.** No K reaches 75%. `bank@16` misses by
  one item-metric: P06_108's DISTS, 0.209 against 0.206.
- Reading. The rule expected a flat curve if the reference route were
  closed. Instead quality improves steeply with K: from K = 1 to 16, LPIPS
  on R falls by 45–56% on every item, and 16 references reach the hull on
  2 of 3 items. The competing explanation (the coding error and the warp's
  blending, not the reference's pose, keep the warp above the hull) is
  refuted. Nearer references close most of the gap to `warp`, so the
  pose distance to the reference is what costs. The hypothesis held for
  `bank@1`, and fell one item-metric short for `bank@16`.
  So the route does not close as the rule's fail branch states. What is
  open is a bank larger than 16, which only makes sense if it is
  affordable: sent once per video (VISOR's videos run for minutes) and
  chosen outside the window. That is the buildable test, with K up to 64
  and its billed rate. It needs keypoints for frames outside the windows
  (H2's poses cover the windows only), so it is a task of its own, not
  run here. Rung 3 (a generator fitted per clip) waits behind it.

### 2026-10-22 — K6b (hands) 3: overfit oracle (rule, not run)
- Question. Can FoundHand, fitted on the window's own frames, draw that
  hand well enough to beat the hull at about 2 kbps? It has seen the
  frames it is judged on (H3d's oracle form), so it bounds every renderer
  fitted elsewhere in the same video. If it fails, the generative route
  closes for egocentric hands at the keypoint stream's rate: K1f's win
  needs the hand's own appearance per frame, which only pixels carry.
- Model. FoundHand's EMA weights, fine-tuned per item (full DiT, or a
  LoRA if the smoke shows full tuning cannot fit 48 GB) with its own
  diffusion loss (`training_losses` in its `gaussian_diffusion`; no
  training script is released, so the loop is written here). Pairs
  (reference crop, target crop) come from the window's own frames and
  the same crops as rung 2 (K1f's uncoded keypoints for training, coded
  ones at test).
- Items. The 3 items scored in rung 2 (P06_108, P22_107, P25_09), with
  P01_107 for the smoke and the pilot. The same judges, hull and
  criterion. Pass (≥ 75%): rung 4 (a fit on other shots of the same
  video, billed). Fail: the generative route closes for hands, and K1f's
  bound stays a bound.
- Training discipline (PLAN K6b). A smoke of at most 10 min: the loss
  falls over 200 steps, and a sample changes towards the target. Then a
  learning-curve pilot on P01_107: fit on frames 0–179 at 25%, 50% and
  100% of a 30-minute step budget, judge `gen` on the held-out frames
  180–239. The scaled fit runs only if the curve, extrapolated, reaches
  the hull's LPIPS on R at P01_107's rate within 45 minutes per item.
  Every stage is at most 45 minutes and checkpointed.
- Hypothesis. It fails. Rung 2's gap is mostly pose-to-pixel error and
  the forearm, not identity, and fitting 240 frames of one hand helps
  identity most.
- Competing explanation. Most of the gap is appearance (skin, sleeves,
  lighting), which the fit learns. Then the held-out frames improve
  steeply in the pilot.
- Budget. A ceiling of 4 GPU-hours (smoke, pilot, 3 fits), RTX 6000 Ada,
  declared at ≥ 40,000 MiB. **Waits for the user's approval:** it is
  training, and rung 2's ceiling was nearly spent.

### 2026-10-22 — K6b (hands) 2c: a buildable bank, chosen outside the window and billed (rule)
- Question ([PLAN K6b](../../PLAN.md), hands, after 2b). Rung 2b's free,
  in-window bank reached the hull on 2 of 3 items with 16 references and
  improved steeply with K. Does a bank a codec could send keep that: chosen
  from the video **outside** the scored window, coded, and billed once per
  video, with the rate amortized over the video?
- Items. Positions 0–3 of K1f's content-blind order: P09_103, P25_09,
  P22_107 (G's training footage) and P06_108, 240 frames each, with K1f's
  rows, R, bound and hull (nothing in the window is re-coded). All four are
  planned. If P09_103 cannot run for infrastructure reasons, the rule reads
  the other three, as 2b did.
- Candidates for the bank (encoder side). VISOR's dense masks (the
  `visor_dense` interpolations; B1b's fill exists only inside windows) on
  every frame of the item's video outside the window and outside a 1 s guard
  on each side, at a stride of 5 frames, exact frames only on P25_09 (59.94
  fps, `visor.frame_alignment`). Per side with a mask whose box has both
  sides ≥ 16 px: WiLoR's two-pass hand-box fit (K1e's `mask_box_rows`,
  unchanged) and its 2D joints. A candidate is kept when at least 15 of its
  21 joints fall inside its mask dilated by 10% of the mask box's longer
  side (drops fits off the hand). This is why rung 2c is its own task: H2
  and K1f fitted the windows only.
- Selection. Per side, greedy k-medoids on 2b's pose vector
  (wrist-centred, scaled by extent): each step adds the candidate that most
  lowers the mean distance from every candidate to its nearest reference. It
  is nested, so the bank of 16 is the first 16 of the bank of 64, and it
  weights the poses the video shows often, where 2b's farthest-point
  sampling (one window, every pose scored) chased the extremes. K ∈ {16, 32,
  64} per side.
- What is sent per reference, and billed: the side mask's box (padded by 8
  px, even sides ≥ 64 px) as one SVT-AV1 still (CRF 40, preset 4, 4:2:0 full
  range; the box outside the mask filled smoothly first, so the background
  costs few bits), its mask as polygons (OpenCV contours simplified at
  1.5 px, coordinates in the box at ⌈log2 side⌉ bits each, plus a count per
  polygon), its 21 joints in the box at the same precision, the box's
  position (4 × 11 bits) and side (1 bit). Payload bits only (the AV1 OBUs,
  as `svtav1` counts them). The decoder uses the decoded pixels and the
  decoded polygon mask. CRF 40 is fixed now, before any run, from a probe of
  two sparse-frame hand crops (P06_108: CRF 35 0.18–0.36 bpp, CRF 45
  0.11–0.22 bpp). A lossless mask (bilevel PNG, about 750 bytes) cost as
  much as the image, hence polygons.
- Decoder, per window frame and fitted side (K1f's rows): the reference
  whose sent joints are nearest the frame's coded joints in pose (2b's
  matching), warped from its joints to the coded ones with `k6.draw`, inside
  R (granted, as K1f and 2b), Telea fill where nothing is drawn, objects
  kept.
- Rate. Item rate = K1f's keypoint rate for the item + the bank's bits
  (both sides, K references each) / the video's duration (ffprobe's, the
  whole video). Reported beside: the rate with the bank amortized over one
  minute, and the payback per metric, the shortest video for which the
  bank's quality still sits below the hull (T* = bank bits / (hull's rate
  at the bank's quality − keypoint rate); an upper bound where the bank
  beats every hull point; none where it never does).
- Variants. `bank2c@K` (coded, billed: the decision) and `free2c@K` (the
  same references uncoded and unbilled: separates the coding and billing
  from the move outside the window), K ∈ {16, 32, 64}, and `warp` (K1f's
  bound, the check: within 0.002 of K1f's LPIPS on R, or nothing is read).
- Decision rule (`k6bh.BANK2C_DECISION`, fixed before any run; the user's
  criterion). Per K, an item is won on a metric when `bank2c@K` on R is
  below K1f's hull at the item's rate. A rate above the hull's highest point
  counts as lost (the hull is not known there). Score = mean over {LPIPS,
  DISTS} of the share of items won.
  - **Pass** (some K scores ≥ 75%): hands have a buildable renderer at the
    keypoint rate plus the bank. Rung 3 is not needed on fidelity. Next: the
    passing K on K1f's other held-out items, then the component.
  - **Fail** with `free2c` at some K ≥ 75%: the bank's bits are what fail.
    Next goes to the user (a cheaper bank: lower resolution, one stream).
  - **Fail** with `free2c` failing too: references from elsewhere in the
    video do not carry the hand well enough. With 2b, the reference route
    closes at a buildable rate, and rung 3 is the last open rung for hands
    (it waits for the user).
- Hypothesis. Fails, with the score at 50% or below at every K. The bank's
  rate (estimated 1–9 kbps at K = 16–64) lifts the bar well above 2b's,
  and references from other activities in the video differ in lighting and
  sleeves. `free2c@64` comes near 2b's `bank@16`.
- Competing explanation. The same person in the same kitchen looks alike
  across the video, and 64 references from thousands of candidates sit
  nearer in pose than 2b's 16 from one window. Then `free2c` beats 2b's
  `bank@16` and `bank2c@16` or `@32` passes, with the bank under 3 kbps.
  The share of references within 10 s of the window, and of window frames
  matched to them, are reported, to tell a near-duplicate bank apart.
- Budget. One job per item, ≤ 45 min each, declared at 44,000 MiB (2b's
  scoring peaked at 38,926 on P09_103, here with seven scorings). Each job's
  smoke (≤ 600 s) runs 24 window frames and candidates at a stride of 200 on
  its own item. Estimate: about 5 min of decoding and WiLoR, then 7 scorings
  (about 150 s each in 2b) per item, so about 30 min, and about 2.6
  GPU-hours with the smokes. Ceiling 3.5 GPU-hours, failed attempts
  included. If it runs out, the items finished in the fixed order are
  reported, and so is the shortfall.
- Revision (2026-10-22, before any run; infrastructure only). The first job,
  `20261022T122902Z-6db2cc7e` (P06_108), stayed pending: other users held
  15–20 GB on every Ada GPU, so none had 44 GB free. It was cancelled before
  it started. The 38,926 MiB was rung 2's, with FoundHand resident, and the
  model–GPU table records 14.2 GiB for WiLoR, LPIPS and DISTS together. So
  PyTorch is capped at 27 GiB (`set_per_process_memory_fraction`, with
  expandable segments: the allocator frees its cache before growing, and a
  real need fails loudly), 30,000 MiB is declared, and the peaks are
  recorded in `k6bh2c.json`. Nothing in the rule changes.
- Revision 2 (2026-10-22, before any score was read; infrastructure). Job
  `20261022T125017Z-174bee4e` (P06_108, `ec72be4`) passed its smoke (28
  checks), but in the full stage the candidates alone took 33 min: 3,121
  frames at stride 5 and 5,361 fitted hands, with the fits waiting for the
  whole decode (about 17,000 1080p HEVC frames). Bank, render and seven
  scorings could not fit the same 45 min. So each item runs as two stages:
  **extract** (bank frames decoded in chunks of 256 in fresh processes while
  WiLoR fits the earlier ones; publishes the candidates unit) and **score**
  (restores it, decodes only the chosen references again, then codes,
  renders and scores). The stride is 10, not 5: about 1,500 candidates per
  video, still over 20 times K = 64. A stride-5 unit is cut to stride 10 on
  restore (the frames whose VISOR number is a multiple of 10, with their
  fits, each fitted on its own), so P06_108's 33 min are kept if that stage's
  `partial.tar` holds them.
- Revision 3 (2026-10-22, validator only, before P22_107 was scored).
  P22_107's score smoke `20261022T135926Z-ea68c004` failed one check: one
  of its 128 references (right hand, rank 55) decodes its polygon mask at
  IoU 0.874 against the source mask, under the check's 0.9 (outer contours
  only, so a hole is filled). That is lossy coding, not a fault: the
  decoder warps with the decoded mask, and its bits are billed. The check,
  meant to catch broken polygon coding, is now median IoU ≥ 0.95 and every
  reference ≥ 0.8. The three items already scored pass it (37 of 37 checks
  each).
- Jobs (RTX 6000 Ada, shared, gpu5 and gpu6; code from each `ready.json`).
  Extract: P09_103 `20261022T133541Z-972661c4`, P25_09
  `20261022T133548Z-9c159ae6`, P22_107 `20261022T133554Z-69d44a0b` (all
  `297927f`); P06_108's stride-5 fits from `20261022T125017Z-174bee4e`
  (`ec72be4`, its `partial.tar` `6ba9ac36…349b`, cut to stride 10). Score:
  P06_108 `20261022T134036Z-c2a4a791`, P09_103 `20261022T134327Z-87c2fe46`,
  P25_09 `20261022T140213Z-6e5dedbc` (all `4f2cb8c`), P22_107
  `20261022T172554Z-d766d9e1` (`83aa02d`; the validator change only). Every
  smoke that ran to the end passed the fleet validator. Every full stage
  passes `k6bh2c validate` locally (11 checks per extract, 37 per score), and
  `warp` reproduces K1f on every item. Candidates (stride 10): 529–1,723
  frames per video, 870–2,529 kept fits. References: SVT-AV1 at CRF 40,
  median PSNR on the mask 37–40 dB, median 10–14 kbit per reference with
  its polygons and joints, median polygon IoU 0.99. Peak GPU memory
  reserved 4.9 GB (P06_108). Report
  `visor/k6bh-2026-10-22/report/k6bh2c-report.json` (`3e46782b…c0b`). GPU
  time: 9,269 s of stages, failed and timed-out attempts included (P06_108's
  first full stage counted at its 2,700 s), about 2.6 GPU-hours of the 3.5.
- Outcome. On R, at each item's rate (keypoints plus the bank amortized over
  the whole video), against K1f's hull there:

  | item (video s) | kp kbps | `bank2c@16` | `bank2c@32` | `bank2c@64` | `free2c@64` (free) | `warp` |
  |---|---:|---:|---:|---:|---:|---:|
  | P09_103 (744) | 2.03 | 0.334 / 0.256 at 2.56 kbps (hull 0.197 / 0.213) | 0.330 / 0.253 at 3.16 (0.186 / 0.204) | 0.331 / 0.251 at 4.43 (0.163 / 0.186) | 0.320 / 0.238 (0.206 / 0.220) | 0.100 / 0.102 |
  | P25_09 (876) | 2.14 | 0.500 / 0.334 at 2.48 (0.249 / 0.237) | 0.489 / 0.327 at 2.86 (0.237 / 0.234) | 0.483 / 0.325 at 3.60 (0.233 / 0.233) | 0.467 / 0.302 (0.260 / 0.240) | 0.097 / 0.102 |
  | P22_107 (581, G) | 1.93 | 0.262 / **0.169** at 2.62 (0.221 / 0.197) | 0.237 / **0.156** at 3.31 (0.206 / 0.191) | 0.233 / **0.153** at 4.48 (0.179 / 0.180) | **0.208 / 0.134** (0.236 / 0.204) | 0.059 / 0.049 |
  | P06_108 (729) | 1.94 | 0.374 / 0.300 at 2.53 (0.161 / 0.199) | 0.358 / 0.289 at 3.09 (0.150 / 0.194) | 0.364 / 0.290 at 4.30 (0.137 / 0.187) | 0.366 / 0.286 (0.172 / 0.206) | 0.091 / 0.123 |

  (LPIPS / DISTS on R; bold: below the hull. G: in G's training footage.)
  Scores: `bank2c@16`, `@32` and `@64` all 12.5% (LPIPS 0 of 4, DISTS 1 of
  4); `free2c` 25% at every K. Clean of G (3 items): 0% for every variant
  but `warp`. The bank costs 0.3–2.5 kbps (0.29–1.78 Mbit), and on P22_107's
  DISTS it pays back after 109–234 s of video.
- Decision (by the rule): **fail, and `free2c` fails too.** A bank chosen
  outside the window, at any K up to 64, does not keep K1f's win: the
  references from elsewhere in the video do not carry the hand well enough,
  coded or not. With 2b, the reference route for egocentric hands closes
  at a buildable rate. Rung 3 (a generator fitted per clip) is the last
  open rung for hands on fidelity, and it waits for the user.
- Reading.
  - Coding and billing are not what fail. `free2c` is within 0.02 LPIPS of
    `bank2c` at the same K, and the bank adds at most 2.5 kbps.
  - K barely matters. From 16 to 64 references the mean pose distance falls
    by 15–27% on every item, but LPIPS on R moves by 0.00–0.03. In 2b, from 1
    to 16 in-window references, it fell by 45–56%. So the nearest pose is
    not what is missing. What is missing is the rest of the hand's
    appearance: the forearm's position and extent, the sleeve, the lighting
    and the hand's place in the frame.
  - The warped references cover 46–91% of R (K1f's own pixels: 95–97%),
    and the order follows the result: P22_107 0.83–0.91, P06_108 0.79,
    P09_103 0.72–0.76, P25_09 0.46–0.49. Where a reference's mask does not
    reach R's shape, Telea fills the rest. Against 2b's in-window `bank@16`
    (free), the gap is 0.10 to 0.23 LPIPS: P22_107 0.100 → 0.208, P25_09
    0.190 → 0.467, P06_108 0.227 → 0.366.
  - It is not a near-duplicate effect either way. On P09_103 and P22_107,
    50–72% of the matches use references within 10 s of the window, and
    they still lose. On P06_108 and P25_09, 0–19% do.
- Hypothesis ("fails, score ≤ 50% at every K; the bank's rate lifts the
  bar"): the fail held, at 12.5%. The stated reason did not: the bar moves
  little (0.3–2.5 kbps), and the free bank fails as well. The competing
  explanation (the same person and kitchen look alike across the video)
  is refuted: `free2c` is far above 2b's in-window bank on every item.
- Scope and caveats. 4 items (positions 0–3 of K1f's order), one of them in
  G's footage. One reference quality (CRF 40), but `free2c` bounds every
  quality. Candidates come only from VISOR's dense spans, at stride 10, with
  VISOR's dense masks (480p polygons) as the references' mattes, where the
  window uses B1b's fill. Matching is by the 2D pose shape alone (2b's
  matcher), not by position or scale in the frame. R is granted, as in K1f.
- Report (2026-10-22, to the root session).
  - Stopped because: goal reached.
  - Decision by the rule: fail, with `free2c` failing too. The reference
    route for egocentric hands closes at a buildable rate.
  - Numbers: score 12.5% for `bank2c@16`, `@32` and `@64`, 25% for
    `free2c` at every K, 0% clean of G. Table above. Jobs above, code
    `297927f` (extract), `4f2cb8c` and `83aa02d` (score), report
    `visor/k6bh-2026-10-22/report/k6bh2c-report.json` `3e46782b…c0b`.
  - Budget: about 2.6 GPU-hours of 3.5.
  - Deviations: stride 10, not 5, and two stages per item (time); the GPU
    memory cap; one validator threshold (one lossy mask). All recorded
    above before the affected scores were read. None changes the rule.
  - Open for the user: whether to approve rung 3 (overfit oracle, 4
    GPU-hours), the last open rung for hands on fidelity.
  - Proposed next step: rung 3 as written, or close hands on fidelity and
    leave them to the experience track (item J).

### 2026-10-22 — K6b (hands) 3: overfit oracle (run)
- Approved (user, 2026-10-22: "Rung 3 approved: run the overfit oracle as
  written, within 4 GPU-hours"), after rung 2c failed. The rule is the
  entry "K6b (hands) 3: overfit oracle (rule, not run)", unchanged: the same
  3 items, judges, hull and criterion; P01_107 for the smoke and the pilot.
- Fit (`experiments/visor/k6bh3.py`, written before any run). FoundHand's
  EMA weights, the full DiT fine-tuned per item: AdamW, learning rate 1e-5
  (the released checkpoint's), batch 8, bf16 autocast over float32 weights,
  FoundHand's own `training_losses` (MSE on the noise plus its learned
  variance term, 1,000 steps). Pairs: a target crop and another crop of the
  same hand from the fitting frames, both with K1f's uncoded keypoints and
  rung 2's crop rule. The reference's mask is dropped half the time (the
  video procedure's later references carry none), and 10% of pairs train
  the sampler's unconditional branch (zero conditions, `nvs` 2). The
  sin-cos position embedding stays frozen. The fitted weights are sampled
  directly (no EMA over a fit this short). Judged as rung 2's `gen` (50
  steps, guidance 2.5, from the window's first fitted frame, on the coded
  keypoints).
- Smoke checks (the rule's): at least 200 steps; the loss on a fixed probe
  (16 pairs, fixed timesteps and noise) falls; a sample of the probe's
  training pairs moves towards its target (LPIPS of the crops falls).
- Deviations, before any run:
  - The pilot judges the held-out frames 180–209 (30, not 60) at 25%, 50%
    and 100% of the 30-minute fit (0% keeps only the probes). By rung 2's
    timing (about 2.5 s per generated hand), four judgements of 60 frames
    plus the fit would not fit a 45-minute stage.
  - Both hands' chains are sampled in lockstep (one batch), so the video
    procedure's per-step random choice of reference is shared by the two
    hands.
  - Declared at 30,000 MiB with PyTorch capped at 27 GiB, not ≥ 40,000,
    as in 2c. Other users hold 15–20 GB on every Ada GPU.
  - The scaled fits' length is the pilot's extrapolated time to the hull,
    at most 45 minutes. If that and the judging of 240 frames do not fit one
    stage, fitting and judging are split, with the weights published
    between them.
- P01_107's bar: K1f's hull at its rate, from the reproduction stage
  `20261022T013208Z-9e048ecf` (K1c recorded 0.258 LPIPS and 0.223 DISTS).
- Smoke and pilot (development, P01_107; not scored for the rule). Job
  `20261022T175637Z-b7d48144` (`a3b261b`, gpu6, RTX 6000 Ada, shared). The
  smoke passed 12 checks: 591 steps in 120 s, the probe loss fell from
  0.093 to 0.078, the training-pair samples from 0.446 to 0.314 LPIPS.
  The full stage passes 12 checks locally: 8,428 steps in 1,800 s of
  fitting (about 4.7 steps per second), peak 21.2 GB reserved. The probe
  loss fell from 0.093 to 0.050. On the held-out frames 180–209, LPIPS /
  DISTS on R:

  | fit | steps | LPIPS on R | DISTS on R |
  |---|---:|---:|---:|
  | 7.5 min | 2,180 | 0.225 | 0.152 |
  | 15 min | 4,341 | 0.189 | 0.144 |
  | 30 min | 8,428 | 0.168 | 0.135 |

  K1f's hull at P01_107's rate (1.95 kbps) reads 0.257 / 0.222. The fit in
  log time reaches it at 3.3 min (DISTS from the start). Report
  `visor/k6bh-2026-10-22/report/k6bh3-pilot-report.json` (`ab0b58c7…bb2`).
  **By the rule, the scaled fits run.** Rung 2's zero-shot `gen` read
  0.496 / 0.255 on all of P01_107.
- Revision (before the scored fits). The run notes set the scaled fit's
  length to the pilot's extrapolated time to the hull, which would be about
  3 min: a fit sized to reach the bar sits on it by construction, which is
  not an upper bound. Each scored item is fitted for 20 min instead
  (between the pilot's 15- and 30-minute points), the longest fit that
  leaves time to judge its 240 frames and K1f's `warp` in one 45-minute
  stage. Lockstep rendering took about 1.6 s per frame for both hands in
  the pilot.
- Jobs (RTX 6000 Ada, shared, gpu5 and gpu6; `a268c0c` from each
  `ready.json`; 20 min of fitting each): P25_09 `20261022T183807Z-3795e867`,
  P22_107 `20261022T183815Z-59e06155`, P06_108 `20261022T183822Z-ef0d3c3c`.
  Every smoke passed the fleet validator, and every full stage passes `k6bh3
  validate` locally (11 checks), with `warp` reproducing K1f. Steps reached:
  5,579–5,708. Report `visor/k6bh-2026-10-22/report/k6bh3-report.json`
  (`07ac515b…064`). GPU time for rung 3, smokes and pilot included: 8,951 s
  of stages, about 2.5 GPU-hours of the 4.
- Outcome. On R, at each item's keypoint rate (1.93–2.14 kbps), the fitted
  weights free (an oracle: fitted on the frames it is judged on):

  | item | `gen` fitted, LPIPS / hull | `gen` fitted, DISTS / hull | rung 2's `gen` (zero-shot) | `warp` (bound) |
  |---|---:|---:|---:|---:|
  | P25_09 | 0.291 / 0.260 | **0.234** / 0.240 | 0.524 / 0.353 | 0.097 / 0.102 |
  | P22_107 (G) | **0.174** / 0.236 | **0.151** / 0.204 | 0.281 / 0.200 | 0.059 / 0.049 |
  | P06_108 | **0.162** / 0.172 | **0.177** / 0.206 | 0.308 / 0.273 | 0.091 / 0.123 |

  (Bold: below the hull.) LPIPS 2 of 3, DISTS 3 of 3: **score 83%**. Clean
  of G's footage (P06_108, P25_09): LPIPS 1 of 2, DISTS 2 of 2, 75%.
- Decision (by the rule): **pass.** A FoundHand fitted on the window's own
  frames draws the hand from the coded keypoints (about 2 kbps) well enough
  to beat the SVT-AV1 + DCVC-UF + G hull on R. Next by the rule: rung 4, a
  fit on other shots of the same video, billed. The generative route for
  hands stays open.
- Reading.
  - The fit roughly halves rung 2's gap: LPIPS on R falls by 0.11–0.23 on
    every item against the zero-shot `gen`. It still sits 0.07–0.19 above
    K1f's bound, the hand's own pixels.
  - The margins are thin where the item is hard. P06_108 wins LPIPS by 0.010
    and P25_09 wins DISTS by 0.006, while P22_107 wins both by 0.05–0.06.
    The pass at 83% rests on these; clean of G it is exactly at the bar.
  - P25_09, the largest hands (crops of about 1,070 px drawn at 256²), is
    the one LPIPS loss, as in rung 2. Competing explanation (3), the crop
    size, still fits and is still not isolated.
  - The pilot's held-out curve (P01_107, frames 180–209) was still falling
    at 30 min (0.168 LPIPS), and these fits stopped at 20. Longer fits would
    likely lower every number. That is an oracle's question, not rung 4's.
- Hypothesis ("fails: the gap is pose-to-pixel error and the forearm, not
  identity"): refuted. The competing explanation (most of the gap is
  appearance, which the fit learns) holds. The pilot's held-out frames fell
  steeply (0.225 → 0.168 between 7.5 and 30 min).
- Scope and caveats. An oracle: fitted on the frames it is judged on, and
  its weights (1.35 GB per item in bf16) are not billed. A real codec has
  to fit elsewhere and send the fit, which is rung 4. FoundHand may have seen
  every item's footage (rung 2's overlap note). 3 items. Flicker and a visual
  check, which rung 2 reported beside, were not computed here, and the
  rendered frames were not kept. A visual check of a few frames is the
  cheapest next step before rung 4 is written.
- Report (2026-10-23, to the root session).
  - Stopped because: goal reached.
  - Decision by the rule: pass (83%; 75% clean of G).
  - Numbers: the table above; jobs above; code `a268c0c`; report
    `k6bh3-report.json` `07ac515b…064`; pilot report `ab0b58c7…bb2`.
  - Budget: about 2.5 GPU-hours of 4.
  - Deviations, all before the scored fits: the pilot judged 30 held-out
    frames, not 60; both hands sampled in lockstep; 30,000 MiB with a 27 GiB
    cap; the scored fits ran 20 min, not "the time to the hull" (about 3
    min, which would sit on the bar by construction).
  - Open for the user: whether rung 4 (fit elsewhere in the video, billed)
    gets a rule and budget, and whether a visual check comes first.
  - Proposed next step: a visual check (the frames were not kept, so one
    item re-fitted and rendered with its frames saved, about 0.6
    GPU-hours), then rung 4's rule.
- Visual check (user, 2026-10-23: "Yes, run the visual check: re-fit one
  item for 20 min with its rendered frames saved"). P06_108 (clean of G, the
  thinnest LPIPS win), re-fitted with the same seeds and settings: job
  `20261023T062736Z-14fa0c21` (`2910f1c`, gpu6), 13 checks pass locally,
  `published.tar` `9101f00b…89fe4`. It reproduces the scored fit: 5,652
  steps, LPIPS / DISTS on R 0.163 / 0.180 (scored: 0.162 / 0.177; hull 0.172
  / 0.206; GPU nondeterminism). Flicker in R (rung 2's measure): fitted
  `gen` 18.0, source 14.4, `warp` 15.3 (rung 2's zero-shot `gen`: 19.7).
  Strips of source | fitted `gen` | `warp` at frames 40, 80, 120, 159 and
  199 (evenly spaced over the hand frames) are in `publish/frames/`.
  - The fitted hands follow the source's pose, finger placement, skin tone
    and lighting in every strip. No fingers are broken, added or missing,
    and the scene outside R is untouched. Rung 2's zero-shot hands were
    plausible but not the source's. These are the source's.
  - The texture is softer than the source's: tendons, veins and skin creases
    are smoothed, as a 256² crop drawn back at about 550 px would be. On the
    reaching hand at frame 40 the colour is slightly warmer.
  - At frame 120 (both hands twisting a jar lid), the fingers over the lid
    are blurred and the lid's outline shifts. `warp` shows doubled edges in
    the same place.
  - Flicker is 25% above the source's, which per-frame LPIPS and DISTS do
    not see. Rung 2's rule asked for a visual check of any pass with flicker
    above the source's: this is it, and nothing in it contradicts the pass.
  GPU time: 2,169 s; rung 3 in all, about 3.1 GPU-hours of the 4.

### 2026-10-23 — K6b (hands) 4: the fit moved outside the window (transfer, rule)
- Approved (user, 2026-10-23: "I agree, go ahead", to running rung 4 as a
  transfer test with the fit treated as delivered beforehand, the billed
  rate and payback reported beside it, about 2.5–3 GPU-hours).
- Question. Rung 3's FoundHand, fitted on the window's own frames, beat the
  hull (83%). Does a fit on other shots of the same video do the same, with
  nothing from the window in its training? That is the fit an encoder could
  make and send before the window plays.
- Billing. The fit is treated as delivered beforehand and not billed, as
  venue models are (user, 2026-10-20), so the item's rate is K1f's keypoint
  rate, as in rungs 2 and 3. Reported beside, not decided on: the rate with
  the fitted weights billed once per video (675.6M parameters in bf16,
  1.35 GB, about 15 Mbps over these 10–15-minute videos) and its payback
  (2c's `payback_seconds`). Per-video billing of full weights fails by
  arithmetic, so a pass here asks how small a fit can be, not whether this
  one ships.
- Items. Rung 3's three (P06_108, P25_09, P22_107; P22_107 in G's
  footage), 240 frames each, with K1f's rows, R, bound and hull. Same
  judges and hull as rungs 2 and 3. Nothing in the window is re-coded.
- Fit data (the only change from rung 3). 2c's candidates (VISOR's dense
  hand masks outside the window and a 1 s guard, every 10th VISOR frame,
  WiLoR's two-pass hand-box fit, kept with ≥ 15 of 21 joints inside the
  mask dilated by 10%), from 2c's published units (P25_09 and P22_107 from
  their extract stages, P06_108 from its score stage, cut to stride 10). Of
  the candidate frames with at least one kept fit, 240 evenly spaced by
  position (content-blind; rung 3 fitted 240 frames), decoded again, and
  every kept hand on them cropped by rung 2's rule with its VISOR mask and
  WiLoR joints. The share of crops within 10 s of the window is reported.
- Everything else is rung 3's, unchanged: the released EMA weights, full
  fine-tune, AdamW 1e-5, batch 8, bf16 autocast, `training_losses`, the
  pairing and dropout, 20 min of fitting, the same seeds. Judged as rung 2's
  `gen` (50 steps, guidance 2.5, both hands in lockstep) on all 240 window
  frames, from the window's first fitted frame per side as reference
  (granted and unbilled, as in rungs 2 and 3). `warp` reproduced (check:
  within 0.002 of K1f's LPIPS on R, or nothing is read). Flicker and the
  visual-check strips are saved, as in rung 3's check.
- Decision rule (`k6bh3.TRANSFER_DECISION`, fixed before any run; the
  user's criterion). An item is won on a metric when the transferred
  `gen`'s value on R is below K1f's hull at its keypoint rate. Score = mean
  over {LPIPS, DISTS} of the share of the 3 items won.
  - **Pass** (≥ 75%): a fit made elsewhere in the same video carries the
    hand. Generative hands are buildable if the fit can be delivered. Next
    goes to the user: how small a fit can be (LoRA rank, fewer layers, one
    fit per person across videos) against the payback.
  - **Fail** (< 75%): the fit does not transfer across shots. Rung 3's
    win needs the window's own frames, which only pixels carry, so the
    generative route for hands closes at the keypoint rate. With 2c, hands
    on fidelity close, and they go to the experience track (item J).
  - Reported beside: the gap to rung 3's in-window fit per item (how much
    of the fit's gain transfers) and to rung 2's zero-shot `gen`.
- Hypothesis. It fails, landing between rung 2 and rung 3 (score ≤ 50%).
  2c showed that what other shots lack is the hand's appearance in the
  window (forearm extent, sleeve, lighting, place in the frame), and a fit
  learns the shots it sees.
- Competing explanation. The fit learns the person (skin, hand shape,
  sleeve) and the generator places it from the keypoints, which a warped
  bank could not. Then most of rung 3's gain transfers and the margins on
  P06_108 and P22_107 survive. A transfer that holds only on the items whose
  crops sit near the window (the 10 s share) would be a near-duplicate
  effect, as 2c checked.
- Budget. One job per item, ≤ 45 min each (decoding the 240 fit frames,
  20 min of fitting, rendering and scoring 240 frames, `warp`), RTX 6000
  Ada, shared, declared at 30,000 MiB with PyTorch capped at 27 GiB (rung
  3's peak: 21.2 GB). Each job's smoke (≤ 600 s) fits 60 frames for 120 s
  and judges 24. Estimate about 2.4 GPU-hours with the smokes. Ceiling 3
  GPU-hours, failed attempts included.
- Dry run (CPU, gpu6, before any GPU stage): P06_108's candidates restored
  from 2c's score stage, 240 frames decoded in 46 s, 396 crops, all outside
  the window and its guard. The masks cover the hands and the joints sit on
  them. One early frame's fit is loose but passed 2c's inside-the-mask
  filter.
- Jobs (RTX 6000 Ada, shared, gpu6; `9bc7003` from each `ready.json`; 20
  min of fitting each): P06_108 `20261023T082231Z-f2ae58cf`, P22_107
  `20261023T082238Z-9b03b134`. Both smokes passed the fleet validator, and
  both full stages pass `k6bh3 validate` locally (16 checks), with `warp`
  reproducing K1f. Steps reached: 5,759 and 5,659 (rung 3: 5,579–5,708).
  Crops: 396 and 378 (from 240 of 1,471 and of P22_107's candidate frames
  with a kept fit), 3% and 7% within 10 s of the window. Report
  `visor/k6bh-2026-10-22/report/k6bh4-report.json` (`3690a96f…e9a`).
- Stopped early (2026-10-23): P25_09 (`20261023T082245Z-e3a8572a`) was
  cancelled while its smoke ran. With P06_108 losing both metrics and
  P22_107 losing LPIPS, the score could reach at most 50% even if P25_09
  won both, so the decision was made. P25_09 also lost LPIPS in rung 3.
- Outcome. On R, at each item's keypoint rate, the fit unbilled:

  | item | transferred fit, LPIPS / DISTS | hull | rung 3 (in-window fit) | rung 2 (zero-shot) | `warp` |
  |---|---:|---:|---:|---:|---:|
  | P06_108 | 0.321 / 0.268 | 0.172 / 0.206 | 0.162 / 0.177 | 0.308 / 0.273 | 0.091 / 0.123 |
  | P22_107 (G) | 0.283 / **0.194** | 0.236 / 0.204 | 0.174 / 0.151 | 0.281 / 0.200 | 0.059 / 0.049 |

  (Bold: below the hull.) LPIPS 0 of 2, DISTS 1 of 2: **score 25%**. Clean
  of G (P06_108): 0%. Flicker in R: 35.7 against the source's 14.4
  (P06_108), 15.7 against 3.8 (P22_107). With the weights billed once per
  video (1.35 GB) the rate is 14.8 and 18.6 Mbps. Payback never comes,
  except on P22_107's DISTS: about 113 days of video.
- Decision (by the rule): **fail.** A FoundHand fitted on other shots of the
  same video does not carry the hand in the window. With 2c, hands on
  fidelity close at the keypoint rate and go to the experience track (item
  J). K1f's bound stays a bound.
- Reading.
  - None of rung 3's gain transfers. The transferred fit lands on rung 2's
    zero-shot `gen` (within 0.013 LPIPS and 0.006 DISTS on both items),
    against rung 3's 0.11–0.15 LPIPS better. The fit learned this window,
    not the hand.
  - The strips show the person: skin tone, the watch and forearm hair on
    P22_107, the sleeve-less arms of P06_108. What breaks is the hand in the
    window's action: at P06_108's frame 120 (twisting a jar lid) the fingers
    over the lid are a smear, and on P22_107 the fingers around the spoon
    are placed differently from the source's. That is the pose-to-pixel
    error rung 3's hypothesis named, and only the window's own frames fixed
    it there.
  - Not a near-duplicate effect: 3–7% of the crops sit within 10 s of the
    window, so the fit saw almost nothing close to it.
- Hypothesis ("fails, between rung 2 and rung 3, score ≤ 50%"): the fail
  held, at 25%. It landed on rung 2, not between. The competing explanation
  (the fit learns the person and the keypoints place it) is refuted: the
  person transfers, and it is not enough.
- Scope and caveats. 2 of the 3 planned items (stopped once decided), one in
  G's footage. 240 fit frames and 20 min, as rung 3. A longer fit or more
  frames could move the numbers, but rung 3's comparison shows the gap is
  the window's own content, not the fit's length. Fit crops use VISOR's
  480p dense masks, and the window uses B1b's fill. The reference stays
  granted from the window, as in rungs 2 and 3. FoundHand may have seen the
  footage.
- Report (2026-10-23, to the root session).
  - Stopped because: goal reached (decided after 2 items).
  - Decision by the rule: fail, 25% (0% clean of G).
  - Numbers: the table above; jobs above; code `9bc7003`; report
    `k6bh4-report.json` `3690a96f…e9a`.
  - Budget: 4,373 s of stages plus P25_09's cancelled smoke, about 1.3
    GPU-hours of the 3.
  - Deviations: P25_09 not run (the decision was made without it).
  - Open for the user: nothing on fidelity for hands. The generative route
    for hands is closed at the keypoint rate; whether the experience track
    takes hands up is the root session's call.
  - Proposed next step: none on this ladder.
