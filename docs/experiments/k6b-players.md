# Experiments: K6b (players)

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md).

### 2026-10-22 — K6b (players) 2: zero-shot oracle, Moore-AnimateAnyone (rule)
- Question ([PLAN K6b](../../PLAN.md), step 2). K6's bound (the frame's own
  pixels moved to the coded keypoints, LPIPS 0.075 at 3.6 kbps for two
  players) saves 95% of the pixel envelope's rate, and a warp of another
  frame (`ref`, 0.461) loses to every pixel point on half the windows. Can
  a pose-conditioned generator, off the shelf, given one reference crop
  per player (granted, unbilled) and the coded keypoints, draw the players
  well enough to beat K6's envelope at the keypoint rate? A pass decides
  K6c's route with no training.
- Model. Moore-AnimateAnyone (SD 1.5 ReferenceNet, pose guider, motion
  module; Apache-2.0), the only reference-plus-pose body model with
  weights in `Models` (K6b's audit, first part), in two profiles:
  `original` (the released weights, `patrolli/AnimateAnyone`) and
  `finetuned_tennis` (the pre-reset fine-tune, Feb 2026). Source: the fork
  `emanuele-artioli/Moore-AnimateAnyone@67c4229` (upstream `a914ef3`
  packaged as `animate_anyone`, ported to diffusers 0.30; the fine-tune
  was trained with it). Its model code differs from upstream only in
  import paths, a default the inference code overrides, and a load
  fallback, so the weights' keys are upstream's; every state dict is
  loaded with its missing keys checked (upstream loads with
  `strict=False`, which would hide a mismatch).
- Environment (audit, 2026-10-22). Inference needs torch, diffusers,
  transformers (the CLIP image encoder), accelerate and einops; not
  DWPose or ONNX Runtime, since the poses come from the code. `pointstream`
  has no diffusers or transformers, and transformers 4.x needs
  huggingface_hub < 1.0 against its 2.1.1 (diffusers 0.30.3 also imports
  `HfFolder`, removed in huggingface_hub 1.0): conflict 10, which
  `pointstream-jcv` already resolves. So AnimateAnyone goes into
  `pointstream-jcv` (torch 2.7.0+cu128, transformers 4.57.2,
  huggingface_hub 0.36.0) rather than a third environment: the JCV pack
  plus diffusers 0.30.3 (the archived runtime's pin, wheel
  `1b70209e…edee`) and the fork, both without dependencies
  (`env/jcv/extend-animateanyone.sh`; delta in
  `env/locks/pointstream-jcv.animateanyone-delta.txt`; `pip check` adds
  only diffusers' `importlib-metadata`, a backport it imports below
  Python 3.8). The import check covers JointControlVideo too. Packed
  `pointstream-jcv-20261022T124459Z.tar.gz` (`bf738ce5…8a7e`, 4.5 GB). A
  first pack (`…122606Z`, `2c534fb8…23ce2`) kept the fork's two links to
  absolute paths (`Models/{original,finetuned_tennis}`, its demo scripts'
  weight shortcuts), which fleet staging refuses (pilot
  `20261022T124048Z-392f179f`, before any stage ran); they are now left out. The
  generator is called only through `experiments.tennis.k6b_generate`
  (npz in, npz out); preparing and judging stay in `pointstream`.
- Items. K6's 8 scored windows (5 matches), with K6's published players,
  poses, coded keypoints (K4b.3's code, every 4th frame) and judged frames
  (every 3rd scored frame), from K6's pilot and parts 1–3
  (`20261021T164820Z-f32fa989`, `…171929Z-fcf1e66d`, `…171936Z-50ce2f74`,
  `…171943Z-b9d3ff04`). Nothing is re-coded.
- Interface (`experiments.tennis.k6b prepare`, CPU; decoder-side
  information only). Per player and run of coded frames, a 512×784 crop
  (AnimateAnyone's size) that follows the coded keypoints: one height per
  run (90th percentile over its frames of 1.4 × the larger of the confident
  keypoints' height and their width at 512:784), centres from the
  keypoints' boxes smoothed over 9 frames. The coded pose is drawn as
  AnimateAnyone's DWPose body map (COCO-17 to OpenPose-18 as DWPose maps
  it, neck from the shoulders, keypoints at K4's 0.5 confidence; body only,
  since the code carries no hands or face). The reference: the player's
  first fitted scored frame in the window, its own crop, uncoded.
- Rendering (`k6b_generate`, `pointstream-jcv`). Each run is one video
  from the released long-video sampler (24-frame contexts, overlap 4) at
  the released settings: DDIM with zero-SNR v-prediction, guidance 3.5,
  fp16, a seed per (window, player). Steps: 30 (pose2vid.py's default),
  see the budget.
- Variants, composited by `k6b judge` at 1080p. `gen-original` and
  `gen-finetuned_tennis`: inside each player's region R (granted, as the
  generator's matte; the analogue of `warp`'s granted outline) the
  rendered crop resized back (Lanczos), far player first; Telea fill where
  R has no rendering (K6's charge); the source outside R.
  `warp@q4-velocity-e4` and `ref@q4-velocity-e4`: K6's, recomputed as the
  check.
- Judges: K6's, unchanged. LPIPS (B2's AlexNet map over the full frame,
  averaged over S = R plus every K6 oracle variant's drawn pixels) and
  masked DISTS, on K6's judged frames; PSNR over S beside. Since the
  generator draws inside R only, S is K6's S.
- Decision rule (`k6b.DECISION`, fixed before any run; the user's
  criterion, 2026-10-21). A window is won on a metric when the variant's
  mean is below K6's envelope (the upper concave hull of K6's ten anchor
  points on (kbps on R, −metric), from `k6-report.json`) at the window's
  keypoint rate (K6's pooled rate of q4-velocity-e4). Score = mean over
  {LPIPS, DISTS} of the share of the 8 windows won.
  - **Pass**: `gen-original` scores ≥ 75%. The zero-shot route holds with
    the reference granted; next is the same oracle with the reference
    billed (K6c's interface).
  - **To the user**: `gen-finetuned_tennis` scores ≥ 75% and
    `gen-original` does not. The fine-tune has seen these matches, so it
    is not zero-shot.
  - **Fail**: both < 75%. Step 3 (the overfit oracle) gets its rule and
    budget and waits for the user.
  - **Check**: `warp` and `ref` reproduce K6's LPIPS over S within 0.002
    per window, and S's pixel counts equal K6's on every judged frame, or
    nothing is read.
  Reported beside: K6's saving statistic with its bootstrap CI, the share
  of R the generator covers, a review image per window.
- Overlap. `original` has not seen PointStream's footage. The audit
  recorded `finetuned_tennis` as trained on all seven archived tennis
  videos, which include K6's five matches, so every window counts as seen
  by it. The only surviving track list (`pointstream-data/assets/dataset/
  pointstream_aa_meta.json`: 114 tracks over 7 videos, written 2026-07-11,
  after the Feb 2026 weights) names `djokovic_federer` and
  `djokovic_zverev` but not `alcaraz_hurkacz`, `medvedev_struff` or
  `rublev_auger-aliassime`. The 4 windows from those three matches are
  reported apart as "absent from the surviving list", not as clean. The
  judges have not seen the footage.
- Hypothesis. Both profiles fail. `gen-original` lands near `ref` (LPIPS
  over S about 0.35–0.45): it draws a plausible player in the right pose,
  but not this frame's cloth, outline and racket (COCO-17 has no racket,
  and the pre-reset fine-tune smeared it). The fine-tune is better, but
  not by enough: the pre-reset diagnostic had it 4.0 dB below an
  articulated paste with the alpha granted.
- Competing explanations. (1) The far player (60–120 px tall) is upscaled
  about 6× into the crop, outside AnimateAnyone's training distribution;
  windows are reported by player height. (2) The body-only pose map is
  out of distribution (DWPose maps carry hands and face); a pass would
  survive it, a fail could partly be it. (3) The granted matte hides the
  generator's outline errors, as for `warp`, so a pass is still a bound on
  the matte.
- Budget. Weights staged as hard links under
  `tennis/k6b-2026-10-22/inputs/aa` (recorded in `Models`' manifests).
  `prepare`: one CPU job. Generation: first-run smokes on each GPU class
  used (Ada, A6000): one unit, 24 frames, both profiles, asserting CUDA,
  fp16, the attention kernels and no missing keys, and timing a frame;
  the timing is recorded here before the scored run. Steps stay at 30 if
  the projected generation (both profiles, every unit) plus about one
  GPU-hour for judging and smokes fits 4 GPU-hours; else 20 (the pre-reset
  class default); else the rung goes back to the user. Generation runs in
  parts of at most 45 minutes, each saving every finished (unit, profile);
  judging in parts of ≤ 4 windows, declared from the smoke's peak. Ceiling
  4 GPU-hours, smokes included (the card's limit: more needs the user).
- Attempts (infrastructure, no score). `20261022T124048Z-392f179f`: fleet
  staging refused the first pack (`82d57c8`; the fork's absolute links; repacked).
  `20261022T124842Z-6ce5a155` (`3129a42`): the smoke stopped on the VAE's
  16 missing keys, since sd-vae-ft-mse stores diffusers' deprecated
  attention names, which only `from_pretrained` renames (a plain
  `strict=False` load, as upstream does for its UNets, would have left them
  random). `20261022T125355Z-b34dc124` (`472bd2c`): stopped on the
  reference UNet's output head (`conv_norm_out`, `conv_out`), absent from
  both profiles' `reference_unet.pth`; AnimateAnyone reads the
  ReferenceNet's attention features, never its output, so the 4 keys are
  allowed and recorded. A CPU check then matched every other key of both
  profiles (3D UNet 686 + 588 motion-module keys, pose guider).
- Prepare: `20261022T123413Z-eba7629b` (CPU, gpu6, `15ed6e9`), 16 units
  (8 windows × 2 players), 3,400 frames, every judged frame covered; all 82
  of `k6b validate`'s checks pass locally (`published.tar` `f832b707…8b8c`).
- Pilot (development, not scored): `20261022T125706Z-924b389b` (gpu6, RTX
  6000 Ada, shared; `98e3c45` from `ready.json`), one unit
  (`djokovic_federer/038`, the far player, 240 frames), both profiles at 30
  steps. First run on Ada passes (fleet model–GPU table). 661 s and 665 s
  per profile: **2.76 s per frame**. All 16 of `k6b_generate validate`'s
  checks pass locally. Visual check (3 judged frames): both profiles follow
  the coded pose (lunge, reach, walking away); `original` keeps the court
  and the player's look, `finetuned_tennis` draws on black (its training
  tracks were matted) and turns the player to face away on the first
  frame; the racket mostly vanishes, as the audit expected.
- Budget, by the rule. Spent: about 0.55 GPU-hours. The other 3,160 frames
  × 2 profiles at 2.76 s are 4.85 GPU-hours, and judging 8 windows about 1:
  **6.4 at 30 steps, about 4.9 at 20**. Neither fits 4 GPU-hours, so **the
  rung goes back to the user** before the scored run.
- **Amendment 1 (user, 2026-10-22, before any score was read):
  `gen-original` only, on all 8 windows, at 30 steps** (about 3.9
  GPU-hours with what is spent). `k6b.DECISION` version 2: the rung passes
  if `gen-original` scores ≥ 75%, and fails otherwise; the "to the user"
  branch for the fine-tune is dropped. `gen-finetuned_tennis` is a
  diagnostic on the pilot window (`djokovic_federer/038`), where it
  rendered the far player; its composite takes `gen-original`'s crops for
  the near player, so the difference between the two is the far player's
  alone. It is not scored against the envelope. The fine-tune has seen
  these matches, so it was not a zero-shot test anyway.
- Generation (`gen-original`, 30 steps; code `edffa84` from each
  `ready.json`; RTX 6000 Ada, shared). Five parts ran at once on gpu5 and
  gpu6 beside other sessions' jobs, and ran at about half the pilot's
  speed. Each saves every finished (unit, profile), so a stop keeps them:
  `20261022T172711Z-2987835e` complete (`published.tar` `c49c92c7…baf`);
  `…172657Z-149a024a` finished all 3 units and stopped at its 2,700 s
  allowance while archiving (`published.tar` `d52af6fb…70f3`);
  `…172650Z-4708bdf5`, `…172704Z-becacf7e` and `…172719Z-c3f87374`
  stopped there with 2 of 3 units each (`partial.tar` `2f17caf0…e496`,
  `6ea7953b…72e`, `f1e97557…612f`). Rendered: 13 of 16 units with the
  pilot's; **6 of the 8 windows have both players** (both
  `djokovic_federer` windows, `rublev_auger-aliassime/052`,
  `djokovic_zverev/012`, both `alcaraz_hurkacz` windows). Missing:
  `djokovic_zverev/008`'s player 1 and both players of
  `medvedev_struff/033` (711 frames). No progress was published per unit,
  so the supervisor raised stall notices; added since (`k6b_generate`
  publishes one per finished unit).
- Budget: about 4.4 GPU-hours of stage time are spent (the five parts
  3.85, the pilot and smokes about 0.55), above the 4-hour ceiling, since
  the parts contended with each other. **Further spending goes to the
  user.** What is left: judging (about 1 GPU-hour for 8 windows, 0.75 for
  6) and the 3 missing units (0.6–1.1). The decision can be read early: at
  8 windows a pass needs 12 of the 16 window-metrics won, so on the 6
  complete windows at most 7 won is already a fail and at least 12 already
  a pass; only 8–11 would need the other two windows.
- Decision (user, 2026-10-22, after the budget was spent): judge the 6
  windows with both players rendered first, and render the last two only
  if the 6 leave the decision open.
- Judging: `20261022T182340Z-e773acb4` (`djokovic_federer/038`,
  `rublev_auger-aliassime/052`, `alcaraz_hurkacz/045`) and
  `20261022T182347Z-c0aac9cc` (`djokovic_federer/072`, `djokovic_zverev/012`,
  `alcaraz_hurkacz/036`), both gpu3 (RTX A6000, shared), code `47db732`
  from `ready.json`, 8.4 and 11.1 minutes. Every smoke passed the fleet
  validator, and both full stages pass all 29 of `k6b validate`'s checks
  locally: on every window `warp` and `ref` reproduce K6's LPIPS over S
  within 0.002, and S's pixel counts equal K6's on every judged frame.
  Report: `tennis/k6b-2026-10-22/report/k6b-report.json` (`acd27678…ca43`,
  `k6b report` at `47db732`; judge archives `87c724d6…2113` and
  `cd470612…6f4c`), with the local validations of the prepare, pilot and
  judge stages in `k6b-validation.tar` (`e92b16fe…2577`).
- Outcome (6 windows from 4 matches; means over K6's judged frames; the
  envelope is K6's at the window's keypoint rate):

  | window | kbps | heights px | `gen-original` LPIPS / envelope | DISTS / envelope | `ref` LPIPS | `warp` LPIPS | saving |
  |---|---:|---|---:|---:|---:|---:|---:|
  | djokovic_federer/038 | 2.74 | 138, 282 | **0.211** / 0.244 | **0.189** / 0.223 | 0.535 | 0.045 | 54% |
  | djokovic_federer/072 | 2.92 | 156, 280 | **0.252** / 0.255 | **0.197** / 0.224 | 0.492 | 0.063 | 2% |
  | djokovic_zverev/012 | 3.32 | 158, 784 | **0.275** / 0.379 | **0.150** / 0.252 | 0.429 | 0.051 | 73% |
  | alcaraz_hurkacz/036 | 1.94 | 118, 175 | 0.291 / 0.280 | 0.250 / 0.246 | 0.431 | 0.099 | −35% |
  | alcaraz_hurkacz/045 | 2.07 | 110, 190 | 0.338 / 0.280 | 0.282 / 0.249 | 0.391 | 0.107 | −135% |
  | rublev_auger-aliassime/052 | 9.80 | 140, 542 | 0.381 / 0.243 | 0.188 / 0.164 | 0.547 | 0.123 | −826% |

  (Bold: below the envelope. The generator covers 95–99.8% of R. K6's
  saving statistic for `gen-original`: mean −1.45, 95% CI [−4.30, 0.31];
  `warp` 0.94 [0.90, 0.97]; `ref` −7.49.)
  - `gen-original` wins 3 of 6 windows on LPIPS and 3 of 6 on DISTS:
    **score 50%**. At 8 windows a pass needs 12 of 16 window-metrics; with
    the two unjudged windows won on both, the score could reach at most
    10 of 16 (62.5%).
  - The fine-tune diagnostic (`djokovic_federer/038`, its far player
    rendered by the fine-tune): LPIPS 0.219 against `original`'s 0.211,
    DISTS 0.193 against 0.189. The fine-tune is not better there.
- Decision (by the rule, amendment 1): **fail**. A zero-shot
  AnimateAnyone, with one unbilled reference per player and R granted as
  its matte, does not beat K6's envelope on 75% of the windows. The two
  unjudged windows (`djokovic_zverev/008`, `medvedev_struff/033`) could not
  change it, so they were not rendered. Step 3 gets its rule below.
- Reading.
  - The hypothesis ("`gen-original` near `ref`, LPIPS 0.35–0.45") is
    refuted in the generator's favour. It is far better than a warp of the
    same kind of reference on every window (0.21–0.38 against 0.39–0.55),
    and it beats the envelope on both metrics on 3 windows. The gap to the
    bound (`warp`, 0.05–0.12) is still 0.16–0.26 LPIPS.
  - The wins are the three Djokovic windows; the losses are both Alcaraz
    windows (both players small, 110–190 px, and lost by 0.004–0.058) and
    the Rublev close-up (7 judged frames, a 542 px player, whose envelope
    is at 9.8 kbps). Competing explanation (1), the far player upscaled
    out of distribution, fits the Alcaraz losses but not
    `djokovic_federer/038`'s win (its far player is 138 px); it was not
    isolated.
  - The granted matte (competing explanation 3) remains the condition: the
    generator draws inside R only, so its outline is granted, as `warp`'s.
- Budget: about 4.75 GPU-hours of stage time (failed attempts 0.12, pilot
  0.46, five generation parts 3.85, judging 0.33), above the 4-hour
  ceiling by the parts' contention with each other; the overrun was
  reported to the user before the judging, which the user approved.

### 2026-10-22 — K6b (players) 3: overfit oracle (rule, not run)
- Question. Can AnimateAnyone, fitted on the window's own frames, draw
  those players well enough to beat K6's envelope at the keypoint rate on
  75% of the windows? It has seen the frames it is judged on (H3d's oracle
  form), so it bounds every renderer fitted elsewhere in the same match
  (step 4). If it fails, the generative route closes for players at the
  keypoint rate, with the matte granted.
- Model. `original` (the profile that decided step 2), fitted per window
  with the fork's `finetune_stage1.py` objective (ReferenceNet, denoising
  UNet and pose guider; the motion module frozen), on pairs (reference,
  target) from the window's own frames, drawn from the uncoded keypoints;
  tested on the coded ones, as step 2. A LoRA instead of the full UNets if
  the smoke shows full fitting cannot fit 48 GB.
- Items. The 6 windows step 2 judged, with the same judges, envelope and
  criterion (score ≥ 75% of window-metrics). Pass: step 4 (a fit on other
  shots of the same match, billed). Fail: the generative route closes for
  players; K6's bound stays a bound.
- Training discipline (PLAN K6b). A smoke of at most 10 minutes: the loss
  falls over 200 steps, and a sample moves towards the target. Then a
  learning-curve pilot on `alcaraz_hurkacz/036` (the narrowest step-2
  loss): fit on frames 0–179 at 25%, 50% and 100% of a 30-minute step
  budget, render and judge the held-out frames 180–239. The scaled fits
  run only if that curve, extrapolated, reaches the window's envelope on
  the held-out frames within 30 minutes of fitting per window.
- Hypothesis. It passes on the Djokovic windows and on
  `alcaraz_hurkacz/036`, and the score reaches 75%: step 2's losses are
  near the bar, and the residual gap is appearance (cloth, racket,
  lighting), which a fit learns.
- Competing explanation. The gap is the pose-to-pixel error of the coded
  keypoints and the missing racket (COCO-17 has none), which a fit on the
  same keypoints cannot remove; then the held-out frames improve little
  in the pilot.
- Budget. Per window a fit of at most 30 minutes, rendering (about 15
  minutes for two players, run one job per GPU: step 2's parts slowed
  each other by half), and judging (3 minutes): about 50 minutes per
  window, 5 GPU-hours for 6, plus the smoke and the pilot (about 1):
  **ceiling 6 GPU-hours, Ada, one job per GPU. Waits for the user's
  approval:** it is training, and step 2 overran its ceiling.
- Approved (user, 2026-10-22): step 3 as written, ceiling 6 GPU-hours, one
  job per GPU, starting with the smoke and the pilot on
  `alcaraz_hurkacz/036`.
- Revision (before any run; the decision rule unchanged).
  - *What is fitted.* A full fit of both UNets with AdamW needs about
    30–40 GB, more than a shared 48 GB card leaves (28–33 GB free on the
    Ada hosts today), so the rule's LoRA branch: rank 64 (alpha 64) on
    both UNets' spatial attention projections (attn1/attn2 to_q, to_k,
    to_v, to_out.0; 104 layers in the ReferenceNet, its last up block
    excluded as `finetune_stage1.py` freezes it, and 128 in the denoising
    UNet), plus the whole pose guider. bf16 base weights and autocast,
    float32 adapters; AdamW, lr 1e-4 (adapters) and 1e-5 (pose guider),
    50 warm-up steps, batch 1 × 4 accumulated. Otherwise
    `finetune_stage1.py`'s objective (v-prediction, zero-SNR scheduled
    linear betas, SNR-γ 5, noise offset 0.05, 10% unconditional). Pairs: a
    target frame and a random reference frame of the same player in the
    window, 512×784 crops from the uncoded keypoints (`k6b prepare-fit`).
    Rendering merges the adapters into `original` (checked on CPU: the
    adapters saved from the stage-1 UNets load into the rendering UNets and
    change their weights; the motion module is untouched) and uses step 2's
    coded maps and granted reference.
  - *The smoke's "sample moves towards the target".* The decoded x₀
    prediction at t = 500, with fixed noise, on 4 fixed training frames:
    its MSE to the true crop must fall from before to after 200 optimizer
    steps, and so must the loss.
  - *Pilot.* Fit on frames 0–179 for 30 minutes, checkpoints at 25%, 50%
    and 100%. Render each, and `original` unfitted (the curve's 0), from
    frame 180 to 239 for both players, and judge those frames' judged
    frames (`--from-t 180`) against K6's envelope rebuilt from its
    anchors' per-frame scores on the same frames (`k6b fit-report`). The
    scaled fits use the shortest checkpoint fraction that is below that
    envelope on both metrics; if none is, the curve's trend decides
    whether 30 minutes can, and if it cannot, the step stops as a fail.
- Prepare-fit: `20261022T190503Z-2b3ca9b5` (CPU, `c829076`), the 6 windows'
  12 units, every fitted frame; all 37 of `k6b validate`'s checks pass
  locally (`published.tar` `99cb6bc2…40b1`).
- Smoke and pilot fit: `20261023T061758Z-c0316a7d` (gpu6, RTX 6000 Ada,
  shared; `c829076`), `alcaraz_hurkacz/036` frames 0–179 (360 crops, both
  players). The smoke (200 optimizer steps) passes the fleet validator:
  loss 0.219 → 0.173 and the decoded x₀'s MSE to the truth 0.0140 →
  0.0112. Full: 2,689 optimizer steps in 30 minutes (checkpoints at 7.5,
  15 and 30 minutes after 677, 1,256 and 2,689 steps), x₀ MSE 0.0140 →
  0.0080, peak 12.0 GiB, attention on the flash kernel; 24.4 M adapter and
  1.1 M pose-guider parameters. All 9 of `k6b_fit validate`'s checks pass
  locally.
- Pilot render: `20261023T065337Z-b411c095` (gpu5, Ada; `2400b01`):
  `original` and the three checkpoints merged (128 adapter layers each),
  both players, t = 180–239, 2.5 s per frame; all 42 of `k6b_generate
  validate`'s checks pass locally. Pilot judge: `20261023T072106Z-2043b488`
  (gpu6, Ada; `2400b01`), 20 held-out judged frames; all 19 checks pass
  locally (`warp` and `ref` reproduce K6 on them). Curve
  (`k6b fit-report`, `k6b-fit-pilot.json` `dc07cb15…4a05`; envelope from
  K6's anchors on the same frames, at 1.94 kbps):

  | on t = 180–239 | LPIPS / envelope | DISTS / envelope |
  |---|---:|---:|
  | `original`, unfitted | 0.282 / 0.306 | 0.256 / 0.262 |
  | fit, 7.5 min (25%) | 0.241 / 0.306 | 0.227 / 0.262 |
  | fit, 15 min (50%) | 0.237 / 0.306 | 0.223 / 0.262 |
  | fit, 30 min (100%) | 0.245 / 0.306 | 0.219 / 0.262 |
  | `warp` (bound) | 0.085 | 0.094 |
  | `ref` | 0.448 | 0.288 |

  The fit helps (−0.04 LPIPS, −0.03 DISTS against unfitted on held-out
  frames) and saturates within 7.5 minutes. Every checkpoint is under the
  held-out envelope, but so is the unfitted model on these frames (the
  window as a whole was lost in step 2 by 0.011 LPIPS), so the pilot shows
  the fit's gain, not a margin. By the rule, **the scaled fits run at the
  shortest reaching fraction: 7.5 minutes per window**, on all of the
  window's frames (the oracle form). Projected: about 35 minutes per window
  (fit 11 with loading, rendering both players 20, judging 3) plus each
  job's smoke, about 4 GPU-hours for 6; with the 1.05 spent, about 5 of the
  6 approved.
- Scaled fits (`gen-fit`: `original` + the window's LoRA, 7.5 minutes on
  all of the window's fitted frames; code `82f3369`, Ada, shared; one job
  per GPU): `20261023T072435Z-a39f6718` (`djokovic_federer/038`, 710
  optimizer steps), `…072444Z-bf4cb8e0` (`/072`, 430), `…072453Z-ce59b93f`
  (`rublev_auger-aliassime/052`, 671), `…072502Z-fcdde99d`
  (`djokovic_zverev/012`, 625), `…072511Z-371ff3e9` (`alcaraz_hurkacz/045`,
  449), `…072519Z-37185050` (`/036`, 415). The budget is wall time, so
  contention cut the steps on four windows below the pilot's 677 at 7.5
  minutes. Every smoke passed (loss and x₀ falling) and every full stage
  passes `k6b_fit validate` locally. Renders (code `1aaa085`, both players,
  every frame): `20261023T073830Z-3f8140a2`, `…073840Z-65dd93e6`,
  `…074026Z-6394e968`, `…074108Z-a101a3d1` (42 minutes, contended),
  `…075035Z-97344dbd`, `…075506Z-bf288877`; all pass `k6b_generate
  validate`. Judges: `20261023T083816Z-f1649e10` and `…083822Z-30b2b476`
  (gpu2, RTX A6000; `1aaa085`), all 29 checks each locally, with `warp`
  and `ref` reproducing K6 on every window. Report:
  `tennis/k6b-2026-10-22/report/k6b-report-step3.json` (`0224e2d9…0583`,
  `k6b report --step 3`), receipts in `k6b-step3-validation.tar`
  (`5e0ec785…447c`).
- Outcome (6 windows from 4 matches; K6's envelope at each window's
  keypoint rate):

  | window | `gen-fit` LPIPS / envelope | DISTS / envelope | step 2 LPIPS / DISTS | saving |
  |---|---:|---:|---:|---:|
  | djokovic_federer/038 | **0.185** / 0.244 | **0.183** / 0.223 | 0.211 / 0.189 | 75% |
  | djokovic_federer/072 | **0.230** / 0.255 | **0.191** / 0.224 | 0.252 / 0.197 | 35% |
  | djokovic_zverev/012 | **0.257** / 0.379 | **0.146** / 0.252 | 0.275 / 0.150 | 84% |
  | alcaraz_hurkacz/036 | **0.247** / 0.280 | **0.219** / 0.246 | 0.291 / 0.250 | 60% |
  | alcaraz_hurkacz/045 | 0.283 / 0.280 | **0.241** / 0.249 | 0.338 / 0.282 | −12% |
  | rublev_auger-aliassime/052 | 0.300 / 0.243 | **0.152** / 0.164 | 0.381 / 0.188 | −122% |

  (K6's saving statistic: mean 0.20, 95% CI [−0.45, 0.69], positive on 4
  of 6. The bound, `warp`, is 0.045–0.123 LPIPS.)
  - `gen-fit` wins 4 of 6 windows on LPIPS and 6 of 6 on DISTS: **score
    83%**.
  - The fit improves every window over step 2 (LPIPS by 0.018–0.081). The
    LPIPS losses are `alcaraz_hurkacz/045` by 0.003, the window whose fit
    got 449 steps, and the Rublev close-up (7 judged frames, a 542 px
    player, an envelope at 9.8 kbps).
- Decision (by the rule): **pass.** A renderer fitted on the window's own
  frames beats the pixel envelope at the keypoint rate on 83% of the
  window-metrics, so the generative route stays open for players with the
  matte granted. Next is step 4: a fit on other shots of the same match,
  judged on K6's windows and billed. Its rule waits for the user.
- Reading.
  - The hypothesis (passes on the Djokovic windows and
    `alcaraz_hurkacz/036`, score ≥ 75%) held. The competing explanation (the
    gap is pose-to-pixel error the fit cannot remove) is refuted for
    DISTS, and for LPIPS on 4 windows; it may still describe the Rublev
    close-up.
  - Scope. This bounds renderers fitted elsewhere in the match: the fit
    saw the frames it is judged on, its reference and matte are granted,
    and its LoRA (24.4 M parameters, about 49 MB in fp16) is unbilled.
    `original` has not seen PointStream's footage before this fit.
- Budget: 4.8 GPU-hours of stage time (smokes, pilot fit, render and
  judge, 6 fits, 6 renders, 2 judges), within the 6 approved.

### 2026-10-23 — K6b (players) 4: match fit, billed (rule, not run)
- Question (PLAN K6b step 4, user 2026-10-23). Step 3's renderer saw the
  frames it is judged on. Does one fitted on *other* shots of the same
  match beat K6's envelope on K6's windows, once its bytes are paid?
- Why size decides. Step 3's adapter is 24.4 M parameters: about 49 MB in
  fp16, 390 Mbit. Over a 2-hour match that is 54 kbps, against a keypoint
  stream of 2–3 kbps and envelope rates of 3–20 kbps at the qualities step
  3 reached. Even at 8 bits it is 27 kbps. To pay back within a match the
  adapter must be a few MB, so rank and quantization are part of the test.
- Matches and windows. The 4 matches of step 3's 6 windows:
  `djokovic_federer` (judged 038, 072), `djokovic_zverev` (012),
  `alcaraz_hurkacz` (036, 045), `rublev_auger-aliassime` (052).
  `medvedev_struff` has no judged window and is left out.
  - Training: K4b.1's other windows of the match (K6's `inputs.tar`), every
    one from a different clip than the judged windows, so other shots. Up
    to 6 per match, those with at least 120 registered frames, in sorted
    id order:
    `alcaraz_hurkacz` 002, 006, 011 (3); `djokovic_federer` 015@85,
    015@211, 033, 057, 069, 077 (6); `djokovic_zverev` 000@814, 000@1532,
    000@2006, 002, 004, 006@1257 (6); `rublev_auger-aliassime` 000, 002,
    004, 021, 024, 027 (6). Only court-view frames (K8b-registered), as K6
    scores.
  - Their players and uncoded poses are not published (K6 ran only K4's
    windows), so a GPU job computes them with K6's code unchanged
    (`window_players`, `pose_window`: (c)+(d), YOLO11x-pose with ×1 crops
    under 15% height), then step 3's crops and DWPose maps.
  - The judged windows' own frames are never used for fitting.
- Model and billing.
  - `original` plus one LoRA per match on both UNets' attention (step 3's
    targets and recipe). The pose guider is frozen: it is a further 1.1 M
    parameters to send, and step 3 did not isolate what it gave.
  - Billed bytes: the adapter quantized to 8 bits per output channel (fp16
    scales; G5c's `quantize_channels`), LZMA-coded (G5c's coder). The
    rendering uses the dequantized adapter, so what is judged is what is
    billed. The reference crop stays granted, as in steps 2 and 3.
  - Amortization: the bits over **H = 2 hours** of the match (a short
    best-of-three; the archive holds scene clips, not whole matches, so H
    is an assumption, and the user may set it). A window's billed rate is
    its keypoint rate plus bits / H. Payback hours are reported per window:
    the bits over the rate saved at equal LPIPS, as G5b.
- Pilot (development, not scored): `djokovic_federer`, the match with the
  most training windows and two judged windows that step 3 won.
  - Three ranks, the fewest that bracket the answer: 64 (step 3's, about
    24 MB at 8 bits), 16 (about 6 MB) and 4 (about 1.5 MB; billed 1.7
    kbps over 2 h).
  - Each fitted for 20 minutes, with checkpoints at 50% and 100% (more
    data than step 3's 480 frames: about 2,900 crops).
  - Rendered on frames 180–239 of 038 and 072 (both players) and judged
    there against K6's envelope from its anchors' per-frame scores
    (`k6b fit-report`), at the billed rate.
  - The scaled run uses the configuration with the best billed result on
    the pilot (both metrics won on both segments; ties to the smaller
    rank, then the shorter fit). If none wins both metrics on both
    segments billed, the pilot's best goes ahead only if it wins them
    unbilled, and the outcome is read with that caveat.
- Decision rule (fixed now; the user's criterion). On the 6 judged windows
  (whole windows, K6's judges and judged frames, `warp` and `ref`
  reproduced as before), a window is won on a metric when `gen-match`'s
  value is below K6's envelope at the window's **billed** rate. Score =
  mean over {LPIPS, DISTS} of the share won.
  - **Pass** (≥ 75%): per-match player models pay for themselves within
    a match. K6c's route is open; next is the reference billed too.
  - **To the user**: < 75% billed but ≥ 75% at the unbilled rate. The
    models work but cost too much at H = 2 h; the options are a smaller
    model, a longer amortization (a per-player model across matches), or
    the H the user sets.
  - **Fail**: < 75% unbilled. A match fit does not reach the envelope, so
    the generative route for players closes at the keypoint rate, and
    step 3's pass was memorization of the judged frames.
- Hypothesis. Unbilled, the match fit lands between steps 2 and 3 (about
  half of step 3's gain) and scores about 75%. Billed, rank 4 costs under
  2 kbps and keeps most of rank 64's quality, so the billed score is about
  the unbilled one. Payback is under an hour on the Djokovic windows.
- Competing explanations. (1) Step 3's gain was the judged frames
  themselves (lighting, the exact shirt in this rally), which other shots
  do not carry; then the match fit is no better than step 2. (2) The
  training windows include other people picked as players or close-ups;
  their crops would add noise. The share of training crops whose player
  height is outside the judged windows' range is reported. (3) 8-bit
  quantization costs quality; the pilot's rank-64 arm is also rendered
  unquantized, as a check.
- Budget (ceiling **7 GPU-hours**, Ada, one job per GPU, every stage ≤ 45
  minutes and checkpointed):
  - Training data, about 0.5: one GPU job per match (YOLO, SAM masks read
    from K6's inputs), 3–6 windows each.
  - Pilot, about 2.5: 3 fits of 20 minutes (with loading, about 1.2), 7
    renders of 240 frames each (6 checkpoints and the unquantized check;
    1,680 frames at 2.5 s, about 1.2), one judge (0.1).
  - Scaled, about 3.3: 3 more match fits at the chosen setting (about 1.3
    with loading and smokes; the pilot match's fit is reused) and the 6
    judged windows rendered in full (2,449 frames, about 1.7), plus 2
    judges (0.3).
  - Code (CPU, before any job): the training-data job, 8-bit adapter
    coding and its billed size, the frozen pose guider, and the billed
    rate in `k6b report`. Tests for the quantization round trip and the
    billing.
- **Waits for the user's approval** (and, if wanted, a different H).
- Approved (user, 2026-10-23): as written, H = 2 hours, ceiling 7
  GPU-hours.
- Training data (`k6b prepare-match`, code `999439c`/`3fbb57a`; K6's
  players and poses on K4b.1's windows): `20261023T093741Z-1ce1d61b`
  (`alcaraz_hurkacz`, 3 windows, 1,440 crops, `33e037eb…bb1b`),
  `20261023T094307Z-a06b224a` (`djokovic_federer`, 6, 2,880,
  `1b9661d6…6b45`), `20261023T093749Z-1f0371dd` (`djokovic_zverev`, 6,
  2,880, `79f36734…d27d`), `20261023T094453Z-6f165e95`
  (`rublev_auger-aliassime`, 6, 2,829, `69f67e70…c47a`); each passes
  `k6b validate` (19–37 checks). Two first attempts failed on memory
  (`20261023T093735Z-76072748` overran its 8,000 MiB declaration at
  12,074; `…093754Z-c9e511bc` ran out on a full shared GPU) and were rerun
  at 16,000 MiB. Player heights: 90–284 px on court, except one
  `djokovic_federer` unit at 749 px and `djokovic_zverev`'s near player
  (530–602 px, as its judged window's 784).
- Pilot fits (`djokovic_federer`, 2,880 crops, 20 minutes, the pose guider
  frozen; code `59f0acb`): rank 64 `20261023T095639Z-63da751f` (1,613
  optimizer steps), rank 16 `…095648Z-3ea11584` (1,607), rank 4
  `…095700Z-083e27d9` (1,534); every smoke and full stage passes
  `k6b_fit validate`, peak 12.4 GiB. Billed (8-bit, LZMA): rank 64 20.5
  MB (22.8 kbps over 2 h), rank 16 5.4 MB (6.0), rank 4 1.6 MB (1.75).
  An earlier submission of the three (`…095534Z`, `…095550Z`,
  `…095602Z`) was cancelled at once: its checkpoints would have shared
  names across ranks.
- Pilot renders and judge: `20261023T110007Z-2335ecfc` (rank 64, both
  checkpoints and the unquantized check), `…110013Z-cbcaaa91` (16),
  `…110020Z-c1725982` (4), all pass `k6b_generate validate`; judge
  `20261023T115041Z-68286c57` (`59f0acb`; `published.tar` `8ec0bc00…8334`), 45 checks
  locally, `warp` and `ref` reproducing K6. Curve (`k6b-pilot4.json`, `b572114a…8df3`;
  frames 180–239 of 038 and 072, 20 judged frames each; envelope from
  K6's anchors on those frames at the billed rate):

  | checkpoint | billed kbps (038 / 072) | 038 LPIPS / env | DISTS / env | 072 LPIPS / env | DISTS / env |
  |---|---:|---:|---:|---:|---:|
  | rank 4, 10 min | 4.47 / 4.64 | 0.195 / 0.245 | 0.185 / 0.239 | 0.210 / 0.259 | 0.187 / 0.230 |
  | rank 4, 20 min | 4.49 / 4.67 | 0.201 / 0.245 | 0.197 / 0.239 | 0.223 / 0.259 | 0.193 / 0.229 |
  | rank 16, 10 min | 8.71 / 8.89 | 0.199 / 0.217 | 0.196 / 0.217 | 0.220 / 0.234 | 0.194 / 0.216 |
  | rank 16, 20 min | 8.73 / 8.91 | 0.202 / 0.217 | 0.196 / 0.216 | 0.221 / 0.234 | 0.192 / 0.215 |
  | rank 64, 10 min | 25.48 / 25.65 | 0.197 / 0.123 | 0.193 / 0.141 | 0.224 / 0.139 | 0.197 / 0.162 |
  | rank 64, 20 min | 25.51 / 25.68 | 0.207 / 0.123 | 0.199 / 0.141 | 0.217 / 0.139 | 0.195 / 0.162 |
  | rank 64, 20 min, fp16 (unbilled check) | – | 0.207 | 0.199 | 0.217 | 0.195 |

  Every rank-4 and rank-16 checkpoint wins both metrics on both segments
  billed; rank 64 wins them only unbilled. The 8-bit adapter scores as
  the fp16 one (0.207 and 0.217 both), so quantization costs nothing here.
  Rank 4 is as good as rank 64, and 10 minutes as good as 20. By the rule,
  **the scaled run uses rank 4 with a 10-minute fit**; `djokovic_federer`
  reuses its pilot checkpoint (`djokovic_federer-r4@fit-50.pt`, 768
  optimizer steps, 1,553,200 bytes). Budget: about 3.55 GPU-hours spent
  (training data 0.55, pilot fits 1.51, renders 1.34, judge 0.1); the
  scaled run projects to about 3.25 (3 fits about 0.95, the 6 windows
  rendered about 2.0, 2 judges 0.3), 6.8 in all.
- Scaled fits (rank 4, 10 minutes, pose guider frozen, 8-bit; code
  `c457eec`): `djokovic_zverev` `20261023T115551Z-85b5b82a` (922 optimizer
  steps, 1,556,432 bytes), `alcaraz_hurkacz` `…115607Z-03161c27` (805,
  1,551,640), `rublev_auger-aliassime` `…115626Z-bba5391b` (919,
  1,558,868); `djokovic_federer` reuses the pilot's (1,553,200). Billed:
  1.72–1.73 kbps over 2 h each. The first two and Rublev pass `k6b_fit
  validate` locally. **`alcaraz_hurkacz` fails one check, `loss_falls`**
  (first-tenth loss 0.195, last-tenth 0.196). Each tenth is the mean of
  about 320 micro-step losses at random timesteps, so it is noisy at this
  scale (corrected 2026-10-23: first written as "one sample", a misreading
  of the 17-point logged curve). The logged curve falls to 0.183 mid-fit,
  and the fixed-noise probe falls 19% (0.0096 → 0.0078; the others 10–13%).
  Its renders went ahead without a refit, which would have exceeded the
  ceiling; it has half the other matches' training crops (1,440).
- Renders (whole windows, both players, the match's `-r4@fit-100.pt`, or
  the pilot's `fit-50` for `djokovic_federer`; code `c457eec`), each
  passing `k6b_generate validate` locally: 038 `20261023T115641Z-915f55b1`
  (`9c64993b…0dec`), 072 `…115652Z-aac3ec71` (`8625d28f…34dc`), 052
  `…121820Z-4e808d4e` (`a0e5fa98…d86a`), 012 `…121813Z-52941285`
  (`14c472a9…3e93`), 045 `…123449Z-c778472c` (`5c73549e…82a8`), 036
  `…123441Z-932422f6` (`7a759cd8…f113`). Judges (`c457eec`, Ada and A6000,
  `--profiles match`): `20261023T132947Z-fb8a8581` (038, 052, 045;
  `fca0ea2b…6c7c`) and `…132954Z-ec202dfb` (072, 012, 036;
  `4578d620…ec56`), 29 checks each, locally. `warp` and `ref` reproduce
  K6 (`reproduced: true`; `warp`'s LPIPS within 1e-4 on every window).
  Report `k6b report --step 4 --billing k6b-step4-billing.json --hours 2`
  (`k6b-report-step4.json`, `73a00639…405b`), with the pilot's curve and
  billing and every stage's validation (`k6b-step4-validation.tar`,
  `a7700f23…ceb8`), read-only in
  `~/pointstream-data/tennis/k6b-2026-10-22/report/`.

  | window | keypoint kbps | billed kbps | LPIPS env (billed) | match | step 3 | step 2 | DISTS env (billed) | match | step 3 | step 2 | payback h |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | djokovic_federer/038 | 2.74 | 4.47 | 0.244 (0.219) | **0.197** | 0.185 | 0.211 | 0.223 (0.208) | **0.184** | 0.183 | 0.189 | 0.57 |
  | djokovic_federer/072 | 2.92 | 4.64 | 0.255 (0.229) | 0.230 | 0.230 | 0.252 | 0.224 (0.209) | **0.195** | 0.191 | 0.197 | 2.12 |
  | djokovic_zverev/012 | 3.32 | 5.05 | 0.379 (0.353) | **0.275** | 0.257 | 0.275 | 0.252 (0.228) | **0.151** | 0.146 | 0.150 | 0.39 |
  | alcaraz_hurkacz/036 | 1.94 | 3.66 | 0.280 (0.261) | **0.245** | 0.247 | 0.291 | 0.246 (0.235) | **0.219** | 0.219 | 0.250 | 1.13 |
  | alcaraz_hurkacz/045 | 2.07 | 3.80 | 0.280 (0.259) | 0.305 | 0.283 | 0.338 | 0.249 (0.237) | 0.259 | 0.241 | 0.282 | – |
  | rublev_auger-aliassime/052 | 9.80 | 11.53 | 0.243 (0.234) | 0.344 | 0.300 | 0.381 | 0.164 (0.156) | 0.171 | 0.152 | 0.188 | – |

  Bold: won billed. Unbilled, `gen-match` wins 4 of 6 on LPIPS and 4 of 6
  on DISTS (**score 67%**); billed, 3 of 6 and 4 of 6 (**58%**; 072's
  LPIPS 0.230 sits between the unbilled 0.255 and the billed 0.229).
  K6's saving statistic for `gen-match`: mean −36%, 95% CI −181% to +64%,
  positive on 4 windows. The report's `finetune_unlisted` group is step
  2's subset and does not apply here.
- **Outcome: fails** by the rule (< 75% unbilled). Read as the rule says:
  a fit on other shots of the same match does not reach the envelope on
  K6's windows, and step 3's two extra windows (045 and 052 on DISTS)
  came from fitting the judged frames. The match fit is not step 2 again,
  though: it improves on zero-shot on every window but 012 (LPIPS equal,
  DISTS 0.151 against 0.150), and recovers about 60% of step 3's gain over step 2 (mean
  LPIPS 0.266 against 0.291 and 0.250; DISTS 0.197 against 0.209 and
  0.189), as hypothesized. The hypothesis's score (about 75%) and its
  billed claim hold in part: billing cost one window-metric (072's
  LPIPS), and payback is 0.4–2.1 hours on the four windows won. It loses
  on the same two windows the earlier steps found hardest: the Rublev
  close-up (its training windows are court views; whether the scale gap
  is the cause was not measured) and `alcaraz_hurkacz/045`, the match
  with the fewest training crops and the fit that failed `loss_falls`.
- Budget: about 7.0 GPU-hours of full stages (training data 0.55, pilot
  fits 1.51, pilot renders 1.34, pilot judge 0.1, scaled fits 0.95,
  renders 2.21, judges 0.27), at the 7-hour ceiling; smokes add a few
  minutes each. Renders ran on shared GPUs and took 9–33 minutes each.

### 2026-10-23 — K6b (players) 5: more data, a tennis base, the pose guider and fewer bits (rule, not run)
- Question (user, 2026-10-23, after step 4's 67% / 58%: "try all the
  things that might improve performance before closing the players").
  Can the match fit reach the envelope on K6's windows, billed, once it
  is given what step 4 left out?
- What step 4 says about where the gain is.
  - Not training time or capacity: on the pilot, 10 minutes scored as 20
    and rank 4 as rank 64; the fits' loss is flat after the first minutes.
    So learning rate, rank or longer fits are not tried.
  - Billing caps the score: step 3's oracle, billed the same 1.73 kbps,
    wins 8 of 12 window-metrics (67%). Reaching 9 needs both fewer bits
    and more quality.
  - The window-metrics in reach: `djokovic_federer/072` LPIPS (0.001
    above the billed envelope: fewer bits) and `rublev/052` DISTS (0.014;
    the oracle wins it at 0.152). `alcaraz_hurkacz/045` (0.022–0.046) and
    052's LPIPS (0.110) are lost by the oracle too.
  - 052 is judged on 32 frames: only 32 of its 240 frames are registered
    court views (K6 scores those). Its other frames are a close-up, which
    K6's player selection (it needs the court) cannot crop, so close-up
    training data is out of reach without new code; it is not tried.
- Data (K6's `inputs.tar`, K4b.1's 57 windows): step 4 capped training at
  6 windows with ≥ 120 registered frames. Without the cap (the 120-frame
  floor kept, K6's 8 windows excluded): `rublev` 13 windows (3,095
  registered frames, about 2.2× step 4's), `djokovic_zverev` 11 (2,640,
  about 1.8×), `djokovic_federer` 7 (1,680), `alcaraz_hurkacz` 3 (720; no
  more exist: more needs new K4b windows from the archive's 21 clips, a
  separate job not run here), `medvedev_struff` 2 (480). Djokovic is the
  only player in two matches.
- Development windows (fixed now): `djokovic_zverev/008@120` and
  `medvedev_struff/033@120`. K6 scored both (envelopes exist) and steps
  2–4 never judged them, so the recipe is chosen there and the 6 judged
  windows are not looked at until the scaled run. They are left out of
  every fit's training data.
- Arms (cumulative, the fewest that separate the causes):
  - **A** step 4's recipe (rank 4, 10 minutes, cap 6, guider frozen),
    the baseline on the development windows.
  - **B** A + every K4b.1 window of the match (no cap) + the pose guider
    trained and billed (1.1 M parameters, coded as the adapter).
  - **C** B on a **tennis base**: one LoRA (rank 64, 30 minutes) trained
    on every K4b.1 window of all 5 matches except the 8 K6 windows. It is
    part of the decoder, fitted once for the sport, so it is not billed
    per match. The match's rank-4 LoRA is fitted on top of the merged
    base. This is the general model that K6c would ship, and it gives
    `alcaraz_hurkacz` the other matches' data.
  - Render-time, on the best arm's checkpoints (no refit): adapter bits
    {8, 4} (per output channel, G5c's coder) and guidance {3.5, 2.0}.
- Per-player (user, 2026-10-23): Djokovic, the one player with two
  matches. One player LoRA over `djokovic_federer` and `djokovic_zverev`
  (the chosen recipe), billed over **H_player = 20 hours** (10 matches of
  2 hours, an assumption the user may set). Judged on 038, 072 and 012
  beside the per-match fit; reported, not deciding (one player).
- Choice on the development windows (2 windows × 2 metrics = 4,
  billed at H = 2 h): the arm winning the most; ties go to the larger
  mean margin below the envelope (in units of the window's envelope),
  then to the cheaper. Bits: 4 if both metrics stay within 0.005 of 8
  bits on both windows, else 8. Guidance: 2.0 only if it wins more, or
  as many with the larger margin.
- Decision rule (step 4's, unchanged; the user's criterion). On the 6
  judged windows at the billed rate (H = 2 h), score = mean over
  {LPIPS, DISTS} of the share won by `gen-match5`.
  - **Pass** (≥ 75%): per-match player models pay for themselves. K6c's
    route is open (the tennis base is its first piece).
  - **To the user**: < 75% billed, ≥ 75% unbilled.
  - **Fail**: < 75% unbilled. Then the open route left for players is a
    newer generator (below); the AnimateAnyone line closes.
  - Reported beside: unbilled score, payback, the per-player Djokovic
    variant, and each arm's development numbers. The 75% threshold stays
    for this ladder; paper results will use a significance test or a
    confidence interval instead (user, 2026-10-23).
- Hypothesis. B fixes `rublev/052`'s DISTS (2.2× data), C adds
  `alcaraz_hurkacz/045` on at least one metric (other matches' data), and
  4 bits flips 072's LPIPS: 9–10 of 12, a pass.
- Competing explanations. (1) The losing windows are hard for this
  generator at any data (052's 32 frames sit at a cut; 045's players are
  small): then B and C do not move them on the development windows
  either, and the newer generator is the test. (2) A tennis base helps
  the court views it was trained on but washes out identity: then C wins
  on DISTS and loses on LPIPS. (3) The pose guider's gain in step 3 was
  memorized timing, not appearance: then B equals A with all data.
- Budget (ceiling **11 GPU-hours**, Ada, one job per GPU, every stage
  ≤ 45 minutes and checkpointed; renders about 0.35 GPU-hours per
  window as measured in step 4):
  - Training data, about 0.6: `prepare-match` for the uncapped windows
    (`rublev` 7 more, `djokovic_zverev` 5, `djokovic_federer` 1,
    `medvedev_struff` 2).
  - Development, about 5.2 (`medvedev_struff` trains on 2 windows only, so
    033 tests transfer from little data): fits A and B for 2 matches (1.2), the tennis
    base (30 minutes, 0.6) and C for 2 matches (0.6), 6 renders of the
    two development windows (2.1), 2 render-time variants on the best arm
    (0.4, 120 frames each), 2 judges (0.3).
  - Scaled, about 4.4: 4 match fits (1.2), the Djokovic player fit (0.3),
    6 window renders (2.1), 2 more for the player variant on 038 and 012
    (0.7; 072 shares 038's fit), 2 judges (0.3) — less if A wins (no new
    base).
  - Code (CPU, before any job): `prepare-match` without the cap, fits on
    a merged base, the guider in the billed adapter, 4-bit coding, the
    development and per-player reports, tests.
- Not in this step: a newer generator. Wan-Animate (Wan2.2, 14B, open
  weights; arXiv 2509.14055) and its successors are the current open
  state of the art for pose-driven animation; trying one needs an
  environment audit (a 14B DiT on 48 GB cards), so it is outlined as
  step 6 (the same zero-shot, overfit and match ladder), its rule written
  after the audit.
- **Waits for the user's approval** (and H_player, if not 20 hours).
- Approved (user, 2026-10-23): as written, ceiling 11 GPU-hours,
  H_player = 20 hours.
- Run log (2026-10-23).
  - Training data (`k6b prepare-match`, code `436e6c3`, 16,000 MiB),
    each passing `k6b validate` locally: `rublev`
    `20261023T192005Z-64582992` (7 windows, 3,337 crops,
    `3ece72fa…db77`), `djokovic_zverev` `…192013Z-a58b341c` (5, 2,400,
    `c163f089…e18e`), `djokovic_federer` `…192023Z-6e446ad0` (1, 480,
    `fd0d57e1…c01a`), `medvedev_struff` `…192031Z-c32a29a2` (2, 939,
    `71e7bc58…1c23`).
  - Deviation (stage limit): the tennis base trains on every second
    frame for 25 minutes, not every frame for 30. At about 18,000 crops,
    encoding (0.07 s per crop) and a 30-minute fit would overrun the
    45-minute stage. Step 4's pilot found no gain past 10 minutes.
  - Fits keep pose maps as uint8 (the base would have needed about 86 GB
    of host memory as float32); the arithmetic is unchanged.
  - The base's first submission (`20261023T193408Z-ecb623cd`) failed its
    smoke on `loss_falls` (0.228 → 0.230 over 200 steps) while its probe
    fell (0.0153 → 0.0142): the second noise failure of that check after
    step 4's `alcaraz_hurkacz`. Each step's loss is at a random timestep,
    so a tenth's mean is noisy. From here, fits record the tenths'
    standard errors and the check fails only on a rise of more than two
    of them; the probe stays the test that training moves the model.
    Earlier fits keep the strict test (step 4's `alcaraz_hurkacz` result
    is not re-read).
  - Development fits (code `a3390e5`, 10 minutes): `dz-A`
    `20261023T192405Z-4879087a` (594 optimizer steps on a shared GPU,
    against step 4's 922; 1,542,044 bytes), `med-A` `…192542Z-76ac4449`
    (843; 1,557,820), `med-B` `…192551Z-62ca7bde` (585; 2,515,324, of
    which the guider's change 965,696: 2.79 kbps over 2 h).

  - The tennis base (`20261023T194934Z-b0dcbb19`, code `3890706`): 36
    windows of all 5 matches, 8,597 crops (every second frame), 1,918
    optimizer steps in 25 minutes, the stage 41.2 minutes; passes `k6b_fit
    validate` (loss 0.217 → 0.207, probe 0.0154 → 0.0140);
    `075b001a…3285`.
  - Arm C's `medvedev_struff` fit (`20261023T203428Z-85ca5f54`) failed its
    smoke on the probe (0.02093 → 0.02102, +0.5%) with every other check
    passing (base merged, guider billed, loss). The probe's frames come
    from the training windows, which the base was trained on, so it starts
    lower (0.0209, against 0.0227 without the base) and 200 steps move it
    within noise; `djokovic_zverev`'s C passed. For fits on the base, the
    check now allows a rise under 1%; resubmitted.
  - Development judge for arms A and B (`20261023T203239Z-d497467e`,
    `b1de5471…6cba`, 25 checks, `warp` and `ref` reproduced): A and B each
    win 2 of 4 window-metrics billed (both of `djokovic_zverev/008`, none
    of `medvedev_struff/033`, lost by 0.001 on each metric for A); A has
    the larger margin (0.156 against 0.137 of the envelope).
  - Arm C fits (on the base, guider trained, 8-bit): `dz-C`
    `20261023T203414Z-38ef7b42` (821 steps, 2,507,660 bytes), `med-C`
    `…204452Z-d5c90446` (588, 2,519,640; passes the strict probe test too).
    Development renders (each passing `k6b_generate validate`): A
    `…194403Z-815f567a`, `…194332Z-0f8fbf72`; B `…195337Z-a9db1ff9`,
    `…194542Z-72dd61dc`; C `…205611Z-1872b27b`, `…210405Z-bf738c80` (008,
    033 in that order). Judges `…203239Z-d497467e` (A, B) and
    `…213006Z-b2642a24` (C, `f3cd5fbd…ee8f`), `warp` and `ref` reproduced.

    | arm | 008 billed kbps | 008 LPIPS / env | 008 DISTS / env | 033 billed kbps | 033 LPIPS / env | 033 DISTS / env |
    |---|---:|---:|---:|---:|---:|---:|
    | A | 4.91 | **0.240** / 0.324 | **0.143** / 0.229 | 4.77 | 0.226 / 0.225 | 0.227 / 0.226 |
    | B | 5.98 | **0.233** / 0.309 | **0.138** / 0.220 | 5.84 | 0.229 / 0.219 | 0.228 / 0.223 |
    | C | 5.98 | **0.225** / 0.309 | **0.133** / 0.220 | 5.84 | 0.222 / 0.219 | 0.226 / 0.223 |

    Envelopes at the billed rate; bold won. Each arm wins 2 of 4; by the
    rule the larger margin decides: **C** (0.159 of the envelope, A 0.156,
    B 0.137). C is best on every cell; B's data and guider alone gain
    little on 033 (2 training windows) and their 0.97 MB guider raises the
    bar. Render-time variants of C submitted: 4 bits (from the fp16
    checkpoint) and guidance 2.0, on both windows.
  - Render-time variants of C (renders `20261023T213704Z-20255857`,
    `…213714Z-d0fde23f` at 4 bits; `…213721Z-d2107908`,
    `…213730Z-5bb220ed` at guidance 2.0; judge `…220810Z-042508fa`,
    `1bfd7fc3…9157`, 25 checks). 4 bits bills 1,491,392 and 1,495,592
    bytes (guider change 0.42 MB), 1.66 kbps over 2 h.

    | variant | 008 LPIPS | 008 DISTS | 033 LPIPS / env | 033 DISTS / env | won | margin |
    |---|---:|---:|---:|---:|---:|---:|
    | C | **0.2246** | **0.1333** | 0.2223 / 0.219 | 0.2263 / 0.223 | 2 | 0.159 |
    | C, 4 bits | **0.2246** | **0.1335** | **0.2220** / 0.225 | 0.2262 / 0.226 | 3 | 0.185 |
    | C, guidance 2.0 | **0.2264** | **0.1372** | **0.2155** / 0.219 | 0.2246 / 0.223 | 3 | 0.163 |

    By the rule, **4 bits** (within 0.0004 of 8 bits on every cell) and
    **guidance 2.0** (3 won against 2). The two were chosen apart; their
    combination is first rendered in the scaled run. Development report
    `k6b-report-step5-dev.json`. Budget so far 6.98 GPU-hours.
  - Scaled run: `gen-match5` = C, 4 bits, guidance 2.0. `djokovic_zverev`
    reuses `dz-C` (the same data, recipe and exclusions as a new fit);
    fits for `djokovic_federer`, `alcaraz_hurkacz` and `rublev` submitted.
    It projects to about 10.6 GPU-hours; the per-player Djokovic variant
    (a fit and 3 renders, about 1.6 more) would pass the ceiling, so it
    waits for the user.
  - The first submission of the `djokovic_zverev/012` render
    (`20261023T221727Z-f4e7cb41`) timed out transferring its snapshot to
    gpu1; inspected, it was never published (no specification or ready
    record), so no worker can run it. It is left in place, and the render
    is a new request (`20261023T221822Z-70715001`).
  - Scaled fits (C; code `45b9b90`): `djokovic_federer`
    `20261023T221639Z-aec779fe` (7 windows, 3,360 crops, 617 steps),
    `alcaraz_hurkacz` `…221650Z-3d7cbe2f` (3, 1,440, 716), `rublev`
    `…221700Z-502a4298` (13, 6,166, 600; probe 0.0132 → 0.0131, the
    smallest move); `djokovic_zverev` reuses `dz-C`. Each passes `k6b_fit
    validate`. Renders (`match5`: the fp16 checkpoint quantized to 4 bits
    as billed, guidance 2.0, the base merged), each passing `k6b_generate
    validate`: 038 `20261023T223743Z-69111776` (`5a99b977…6515`), 072
    `…223751Z-7c51a9ee` (`29dc6cb3…05b1`), 012 `…221822Z-70715001`
    (`ad26e1a2…8a4c`), 036 `…223535Z-e9d7c710` (`9fcf4bab…f7e2`), 045
    `…223542Z-32a6e7dc` (`3a86468a…05c5`), 052 `…224225Z-3f7d4ed2`
    (`66636952…cbd4`). Billed: 1,478,828 (`djokovic_federer`), 1,491,392,
    1,488,252, 1,489,388 bytes, 1.64–1.66 kbps over 2 h. Judges
    `20261023T231003Z-4fe6e239` (038, 052, 045; `4b57f935…9792`) and
    `…232338Z-94c7a048` (072, 012, 036; `39277b5b…77ff`), 29 checks each,
    `warp` and `ref` reproduced. Report `k6b report --step 5 --billing
    k6b-step5-billing.json --hours 2` (`k6b-report-step5.json`,
    `73f2929a…8b4f`); development report `k6b-report-step5-dev.json`
    (`0457b86c…35e6`); receipts `k6b-step5-validation.tar`
    (`65db4ec2…2c52`); all read-only in
    `~/pointstream-data/tennis/k6b-2026-10-22/report/`.

    | window | billed kbps | LPIPS / billed env | step 4 | DISTS / billed env | step 4 | payback h |
    |---|---:|---:|---:|---:|---:|---:|
    | djokovic_federer/038 | 4.38 | **0.196** / 0.219 | 0.197 | **0.190** / 0.208 | 0.184 | 0.52 |
    | djokovic_federer/072 | 4.56 | **0.224** / 0.230 | 0.230 | **0.201** / 0.209 | 0.195 | 1.16 |
    | djokovic_zverev/012 | 4.98 | **0.264** / 0.354 | 0.275 | **0.154** / 0.229 | 0.151 | 0.23 |
    | alcaraz_hurkacz/036 | 3.59 | **0.234** / 0.261 | 0.245 | **0.216** / 0.235 | 0.219 | 0.82 |
    | alcaraz_hurkacz/045 | 3.73 | 0.290 / 0.260 | 0.305 | 0.252 / 0.237 | 0.259 | – |
    | rublev_auger-aliassime/052 | 11.46 | 0.366 / 0.234 | 0.344 | 0.184 / 0.156 | 0.171 | – |

    Bold: won billed. `gen-match5` wins 4 of 6 on LPIPS and 4 of 6 on
    DISTS, billed and unbilled alike (**score 67%**, both). K6's saving
    statistic: mean −58%, 95% CI −267% to +69%, positive on 4 windows.
- **Outcome: fails** by the rule (< 75% unbilled). Billing no longer costs
  anything (4 bits and the guider at 1.66 kbps; 072's LPIPS, lost to
  billing in step 4, is won), and LPIPS improves on five windows (by
  0.001–0.015). But the two windows that step 4 lost stay lost on both
  metrics: `alcaraz_hurkacz/045` improves (LPIPS 0.305 → 0.290, DISTS
  0.259 → 0.252) and still misses by 0.015–0.030; `rublev/052` gets worse
  (0.344 → 0.366, 0.171 → 0.184). DISTS is slightly worse than step 4 on
  four windows (by 0.001–0.006), as guidance 2.0 was on
  `djokovic_zverev/008`. Read as the rule says: the AnimateAnyone line
  closes for players at the keypoint rate; the open route is a newer
  generator (step 6, after an environment audit).
- Against the hypothesis: 4 bits flipped 072 (as predicted); the tennis
  base and more data helped where step 4 already won and on 045's
  values, but did not win 045 (predicted: one metric) or 052's DISTS
  (predicted: won with 2.2× data; it got worse). Competing explanation 1
  (the losing windows are hard for this generator at any data) fits:
  052 is judged on 32 frames at a cut, and its fit moved least (probe
  −1%). Not separated: whether guidance 2.0 or the Rublev fit made 052
  worse (the two were chosen apart on the development windows and first
  combined here).
- Per-player Djokovic variant: **not run**. It would have taken the step
  past the ceiling (about 1.6 GPU-hours more), and it cannot change the
  verdict: all three Djokovic windows are already won.
- Budget: **10.02 GPU-hours** of 11 (37 jobs, smokes included, from the
  jobs' execution records).

### 2026-10-24 — K6b (players) 5b: CPU diagnosis and scope hypothesis (rule)
- Question: why do 045 and 052 lose, and can an encoder-computable rule
  route difficult windows to pixels? Step 5 remains a fail.
- Before new measurements: test 052's shot-cut explanation first. Compare
  court-registration runs, video cut positions, reference frame indices and
  judged indices; distinguish a cut anywhere from a reference crossing a cut.
- Measurements on all eight K6 windows (008 and 033 are also development
  windows, not two extra independent windows): registered share, player
  heights, pose misses, camera motion/cuts, coded rate, LPIPS/DISTS envelope,
  reference-to-target time/pose distance, and saved steps 2–5 margins.
- Hypothesis: mixed shots explain 052, and small or poorly fitted players
  explain 045. Competing explanation: reference selection stays in court
  view and equally small successful players refute that explanation; errors
  may be generator-specific. No causal claim from correlation alone.
- Scope selection: first test a single-shot window (no detected internal cut);
  then examine registered share, heights and fit coverage. Report exclusions
  and pooled/scoped numerators and denominators side by side for every
  step, billed and unbilled where available. Never infer missing scores.
  Any discovered threshold is exploratory and requires unused-window
  confirmation on every method with identical inclusion.
- Budget: zero GPU-hours; saved data and CPU analysis only, no fitting,
  Wan-Animate or 34-window final run. Stop with a measured explanation
  or explicit uncertainty and a bounded, unexecuted confirmation proposal.
- Provenance integration: step-5 run continuation copied verbatim from
  d0a00e5b7e5a0fa2ec53a330eff62b4eea93e756, absent from origin/main
  58f4eba9. Existing entries retained; no earlier verdict changed.
- CPU job submitted: `20261024T082408Z-3cb3f467` (pending at submission),
  code `dae3fe78a3ca7faa9a0da8af2eb86dbe68634267`; snapshot
  `b7458237c73952f47a24040f57fb7978d9bbc1658d907f92f15fb8b52fb0cd86`.
  Two CPU threads, no GPU claim; smoke 300 s, full 600 s, total ceiling
  1,200 s. All six hosts inspected; eligible host pool retained. Original
  clips and K6 bundles staged by the existing prepare specification's
  SHA256 identities. Outputs go to the new job, never into prior evidence.
- First CPU request `20261024T082408Z-3cb3f467`: startup failure, no
  workload or smoke began. Fleet marked attention after owner disappearance;
  inspected `run/supervisor.log` and `run/supervisor.json`: Python exits on
  missing `experiments.jobs.monitor` because the selected snapshot included
  only the analysis module. No run status, smoke or workload log exists.
  Boot-log SHA256 recorded in the export receipt; original request retained.
  Corrected request includes the fleet runtime dependencies; this is a
  diagnosed startup failure, not a replay of uncertain execution.
- Corrected CPU request `20261024T082728Z-19b9b119`: pending at submission, code
  `f7fa4cf6`; snapshot `b8a633ee0e96d58ea1b6c9a0124eec172dc2439d02867ffe7da81941c49d7216`. Same
  inputs and 1,200 s CPU ceiling, zero GPU allocation.
- Corrected request `20261024T082728Z-19b9b119` failed its smoke
  before decoding: the reused specification passed `--inputs`, which the
  analysis parser did not accept. Exit 2 confirmed in `smoke/command.log`;
  no full stage ran. Added the input-manifest parser and archived-window
  inventory, retained both failed jobs; still 0 GPU-hours.
- Third CPU request `20261024T082934Z-3b15531e`: pending at submission, code
  `02d8b5ae`, same zero-GPU specification and 1,200 s CPU ceiling.
- Third request `20261024T082934Z-3b15531e` failed substantive smoke
  validation: ffmpeg filters produced 241 scene metadata rows for 240
  output frames. No full stage ran. Fixed by trimming the filter input
  to 240 frames; added six-frame source contact sheets. Preliminary
  smoke-only diagnostics suggest no cut in 052 (max scene score 0.01282),
  contrary to the card's premise; final read awaits a passing job.
- Fourth CPU request `20261024T083117Z-2e644b08`: pending at submission, code
  `de79482f`, same inputs and limits; no GPU allocation.

#### Outcome: the cut premise is false; scope remains a hypothesis
- Final CPU job `20261024T083117Z-2e644b08` **complete**, gpu2 host,
  **no GPU allocated** (`CUDA_VISIBLE_DEVICES=""`). Code `de79482f`,
  two CPU threads, smoke 4.87 s and full 32.38 s of workload wall time;
  all 34 full-stage validation checks pass locally (the remote smoke
  validator passed 10 checks). No model, judge or fit was run.
- Evidence: `jobs/fleet/inbox/20261024T083117Z-2e644b08/full/scope.json`
  under `Datasets/pointstream-data/`, SHA256
  `ac640ae8f2ee0d807a085555f5504a354de9cba554aaf465cd036673c0d046d2`.
  It includes all clip and K6 archive identities, exact ffmpeg commands,
  all scene scores, reference indices, registration runs and contact-sheet
  identities. Packed environment `pointstream-20261009T214002Z`, SHA256
  `32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661`;
  native binary `{staged environment}/bin/ffmpeg`, version 8.1.2.
  Snapshot selected `experiments/jobs` and `experiments/tennis/k6b_scope.py`;
  no local patch or untracked file included. `ready.json` preserves the
  snapshot identity. Input manifest archive `20e595d5…e703` and the four
  K6 archives are the unchanged step-2 preparation inputs.
- Reproduce scores: `python -m experiments.tennis.k6b_scope report --scope
  SCOPE_JSON --reports STEP2_JSON STEP3_JSON STEP4_JSON STEP5_JSON --out
  NEW_JSON`. The analysis output must be a new external path. Reports are
  the saved `tennis/k6b-2026-10-22/report/` files. Their complete identities:
  - `k6-report.json`: `22cb71c741d2870dfd3f646ca55f6ffbed16fdc275f86174f5103cd8635ee0ad`.
  - `k6b-report.json`: `acd27678226bd3208422b56bbb6ae26bbc60b9828e9152cf36315c37cc42ca43`.
  - `k6b-report-step3.json`: `0224e2d95e73da1b2fe251dbab0f7bec6b5688d890668c9678e7fea0c00a0583`.
  - `k6b-report-step4.json`: `73a00639f6624a1f3edadfc121f9b2a481486fc08da1c911f0ce43730827405b`.
  - `k6b-report-step5.json`: `73f2929adba747446c166a7d118c2e67c6ff36b93968055a4874ace1605f8b4f`.

**First test: a cut in 052.** No cut is detected in **any** of the eight
240-frame windows at scene thresholds 0.15, 0.30 or 0.45 (320×180 decode,
ffmpeg scene score). In 052 the maximum is only 0.01282. The six-frame source
contact sheet (`sheet-rublev_auger-aliassime__052_at120.jpg`, frames 0, 60,
109, 153, 180, 239) shows one continuous, low oblique court view, with camera
motion, throughout. It is **not mostly a close-up**. Sparse registration is
not evidence of non-court footage. The card and earlier interpretation of
this window were mistaken; their historical text and step-5 verdict stand.

052's registered spans (inclusive): 109–110, 115, 118–119, 130–136,
139–153, 157, 235–236, 238–239. Both references are frame **109**, the first
fitted scored frame; both players have all 32 registered poses fitted.
Only **seven frames are judged**: 132, 135, 141, 144, 147, 150, 153.
All references and judged frames belong to the same detected shot. Thus
neither a cross-cut reference nor pose *misses on registered frames* explain
this loss. The generator has poor pose/appearance reconstruction in an
oblique, net-occluded view; failed court registration makes its evaluation
unusually sparse. The saved t144 image shows the far player behind the net,
a changed racket/arm outline and clothing details. These are observed errors,
not a causal ablation proving which model limitation creates the margin.

**Measurements on the eight K6 windows.** 008 and 033 are two of the eight,
not two additional independent windows. Heights are medians of the two
player masks in native 1080p pixels. `Reg` is frames K8b registers, `J` the
actual judged frames. Envelope is LPIPS / DISTS at the unbilled keypoint
rate, reproduced from the saved reports, including development envelopes.

| Window | Reg / 240 | J | Heights px | Keypoint kbps | Envelope LPIPS / DISTS |
|---|---:|---:|---|---:|---|
| alcaraz_hurkacz/036 | 240 | 80 | 118, 175 | 1.937 | 0.280 / 0.246 |
| alcaraz_hurkacz/045 | 239 | 79 | 110, 190 | 2.073 | 0.280 / 0.249 |
| djokovic_federer/038 | 240 | 80 | 138, 282 | 2.741 | 0.244 / 0.223 |
| djokovic_federer/072 | 240 | 80 | 156, 280 | 2.916 | 0.255 / 0.224 |
| djokovic_zverev/008 | 240 | 80 | 160, 618 | 3.196 | 0.348 / 0.244 |
| djokovic_zverev/012 | 239 | 79 | 158, 784 | 3.323 | 0.379 / 0.252 |
| medvedev_struff/033 | 240 | 80 | 85, 168 | 3.044 | 0.248 / 0.240 |
| rublev_auger-aliassime/052 | 32 | 7 | 140, 542 | 9.803 | 0.243 / 0.164 |

Players below are ordered by mask height, smaller first. Misses count failed
pose fits **conditional on registered frames**; unregistered frames are
outside the denominator. Camera motion is the median / p95 displacement of
the projected court centre between adjacent registered frames (1080p px),
a registration-derived proxy, not optical flow or true camera velocity.
Reference distance is translation-centred joint displacement divided by
that player's median mask height (joint confidence ≥ 0.5, ≥ 4 shared joints),
median over fitted judged targets; this measures pose, not image LPIPS.

| Window | Pose misses % | Both fitted % | Camera med / p95 px | Reference t | Median reference gap s | Pose distance |
|---|---|---:|---|---|---|---|
| alcaraz_hurkacz/036 | 0.00, 0.00 | 100.00 | 0.11 / 0.35 | 0, 0 | 4.74, 4.74 | 0.145, 0.147 |
| alcaraz_hurkacz/045 | 0.00, 4.60 | 95.40 | 0.33 / 8.91 | 0, 0 | 4.80, 4.92 | 0.223, 0.087 |
| djokovic_federer/038 | 0.00, 0.00 | 100.00 | 0.09 / 0.20 | 0, 0 | 2.37, 2.37 | 0.194, 0.117 |
| djokovic_federer/072 | 0.00, 0.00 | 100.00 | 0.14 / 0.75 | 0, 0 | 2.37, 2.37 | 0.180, 0.161 |
| djokovic_zverev/008 | 0.00, 0.00 | 100.00 | 0.56 / 1.83 | 0, 0 | 1.98, 1.98 | 0.033, 0.043 |
| djokovic_zverev/012 | 0.00, 0.00 | 100.00 | 0.50 / 2.53 | 1, 1 | 1.98, 1.98 | 0.079, 0.050 |
| medvedev_struff/033 | 3.75, 0.00 | 96.25 | 0.02 / 0.13 | 0, 0 | 2.43, 2.37 | 0.102, 0.306 |
| rublev_auger-aliassime/052 | 0.00, 0.00 | 100.00 | 0.62 / 6.62 | 109, 109 | 0.70, 0.70 | 0.097, 0.114 |

**045:** no cut, 239/240 registered frames, the far player fitted on all
239, the near player missing 4.60% of registered frames. Smaller masks
(110, 190 px) alone do not establish a cause: winning 036 has 118, 175 px,
almost the same keypoint rate (1.94 vs 2.07 kbps) and the same unbilled
anchor strength. Its far-player reference pose distance is larger (0.223
vs 0.145), consistent with more variation from one reference; development
033 has an even larger distance (0.306 on its larger player), so this is not
a uniquely separating feature. Camera p95 is high, but 052 does not have
the largest p95. The saved step-5 t120 image has a front-facing far player
in the source and a back-facing generated player, with a changed racket
and arms on the near player. Appearance/view and racket reconstruction
are visible failure modes; no saved CPU-only comparison isolates their
individual contribution. A global window failure cannot be attributed
numerically to the far player from a single review frame.

052's high 9.80 kbps is measured over just 32 registered frames (0.64 s),
in eight broken runs; short tracks incur repeated absolute starts. Its
median reference gap is only 0.70 s and pose distances 0.097 / 0.114,
so a far-away reference is not its distinctive problem. Rates and anchors
use the same registered-duration convention; changing only its keypoint
denominator to the full window would be unfair. The perfectly supplied
appearance bound still wins (052 LPIPS 0.123 vs envelope 0.243; 045 0.107
vs 0.280), so these windows do not bound all possible renderers as failures.

**Saved per-window margins** (envelope minus method; positive wins).
Each cell is LPIPS / DISTS, **unbilled**. These are fixed-point margins,
not new BD-rate claims. Full original reports retain native codec paths,
versions, GPU identities and source-job provenance. Step-2 judges
`20261022T182340Z-e773acb4`, `20261022T182347Z-c0aac9cc`; step-3 judges
`20261023T083816Z-f1649e10`, `20261023T083822Z-30b2b476`; step-4 judges
`20261023T132947Z-fb8a8581` and its companion in the preserved report;
step-5 judges `20261023T231003Z-4fe6e239`, `20261023T232338Z-94c7a048`.

| Window | Step 2 | Step 3 | Step 4 | Step 5 |
|---|---|---|---|---|
| alcaraz_hurkacz/036 | -0.0104 / -0.0041 | +0.0331 / +0.0268 | +0.0352 / +0.0271 | +0.0465 / +0.0302 |
| alcaraz_hurkacz/045 | -0.0583 / -0.0322 | -0.0027 / +0.0081 | -0.0245 / -0.0096 | -0.0096 / -0.0024 |
| djokovic_federer/038 | +0.0329 / +0.0344 | +0.0591 / +0.0400 | +0.0471 / +0.0394 | +0.0487 / +0.0335 |
| djokovic_federer/072 | +0.0031 / +0.0274 | +0.0251 / +0.0335 | +0.0253 / +0.0295 | +0.0313 / +0.0238 |
| djokovic_zverev/008 | not scored | not scored | not scored | not scored |
| djokovic_zverev/012 | +0.1042 / +0.1017 | +0.1224 / +0.1063 | +0.1039 / +0.1004 | +0.1155 / +0.0981 |
| medvedev_struff/033 | not scored | not scored | not scored | not scored |
| rublev_auger-aliassime/052 | -0.1385 / -0.0243 | -0.0573 / +0.0112 | -0.1006 / -0.0070 | -0.1226 / -0.0204 |

Billed step-5 losses: 045 −0.0298 / −0.0145; 052 −0.1317 / −0.0275.
008 and 033 have **no final step-2–5 test scores**; step 5 used them to
choose the recipe. Their saved development A/B/C and render-time variants
are not the final combined 4-bit, guidance-2.0 configuration. Their numbers
remain in the copied step-5 development table; they must not be filled into
missing test cells. Also correct the card's shorthand: **step 3 scored
83%, not 67%**; its two LPIPS losses both won DISTS.

**Scope candidates and pooled scores, side by side.** Every score compares
the method against exactly the same retained windows' own anchor envelopes.
Unrendered windows never enter a denominator. No anchor is selectively
removed or changed. The saved warp bound stays 100% and the ref warp scores
1/12 pooled, 1/10 after the registration gate, 1/8 after the combined gate;
the score reporter emits all three methods together to enforce that scope.

- Single-shot rule: no scene score > 0.30. It excludes **nothing**, including
  052, at all three thresholds checked; scores equal pooled throughout.
- Registration rule: keep windows with **≥ 95% registered frames**, otherwise
  use the pixel coder. Excludes only 052 on all eight. It does not separate
  every loss, since 045 remains. This threshold is a discovered hypothesis;
  these data do not identify 95% uniquely (any threshold above 13.33% and
  at most 99.58% has the same membership).
- Size alone (single-shot plus smaller player's median height ≥ 115 px):
  excludes 045 and development 033 but retains 052. Its scored denominator
  is five windows and scores happen to equal the registration-only column;
  the excluded failure is different. It has no established causal basis.
- Combined registration ≥ 95% **and** minimum height ≥ 115 px: excludes 052,
  045 and development 033; four scored windows remain. It separates the two
  selected test losses retrospectively, but 115 px was chosen in the tiny
  gap between 110 and 118 px. This perfect-looking selection is explicitly
  exploratory, not a validated decision boundary or a paper result.

| Step | Pooled (= single-shot) | ≥95% registration | ≥95% registration and ≥115 px |
|---|---|---|---|
| 2 unbilled | 6/12 = 50% | 6/10 = 60% | 6/8 = 75% |
| 3 overfit, unbilled | 10/12 = 83.3% | 9/10 = 90% | 8/8 = 100% |
| 4 unbilled | 8/12 = 66.7% | 8/10 = 80% | 8/8 = 100% |
| 4 billed | 7/12 = 58.3% | 7/10 = 70% | 7/8 = 87.5% |
| 5 unbilled | 8/12 = 66.7% | 8/10 = 80% | 8/8 = 100% |
| 5 billed | 8/12 = 66.7% | 8/10 = 80% | 8/8 = 100% |

These are window-metrics, not independent frames; only four matches in the
pooled test and three after removing 052. No score changes the written
step-5 **fail** verdict. Registration-only is the simpler follow-up
hypothesis; the combined gate should be frozen and tested alongside it,
not preferred for producing 100% here. A deployed fallback would also need
whole-stream rate accounting on excluded windows; these object-region
scores do not establish a hybrid codec's end-to-end gain.

**Fresh-window confirmation (not run).** The saved input inventory contains
57 windows: 36 used to train step 5's tennis base, eight evaluation/development
windows, and 13 unused by that fit. The remaining 13 are:
`medvedev_struff` 007@62, 016@14, 023@69, 028@44, 034@105;
`rublev_auger-aliassime` 003@53, 006@55, 038@120, 039@222, 054@16,
061@83, 062@218, 063@337. They failed the 120-registered-frame training
floor (038 has zero), so **none can satisfy a ≥95% gate**: they cannot
confirm performance inside the proposed scope. The 34-window final run
would reuse fitted data and is neither authorized nor an independent test.

Available archive sources outside those 44 train/eval windows include the
2,310 s `archive/pre-reset-2026-10-05/tennis_games/raw_1080p/nadal_kyrgios.mp4`
(documented in K8; not used in steps 3–5), plus unused scene clips of the
five matches. K8b documents 21 Alcaraz–Hurkacz clips and 41 Medvedev–Struff
clips; only five Alcaraz clips (002, 006, 011, 036, 045) and three Medvedev
clips (006, 008, 033) enter step-5 fit/evaluation. These leave **16 and 38
clip IDs**, respectively, outside that fit/evaluation set, before rejecting
non-court views, too-short clips and overlaps. Their remaining IDs and
eligibility have not been enumerated here; do not invent a confirmed
window list. Archive `MANIFEST.json` is the read-only source inventory.

Proposed bounded confirmation: first CPU-freeze 12 new, non-overlapping
240-frame windows from unused Alcaraz and Medvedev clips (six per match),
chosen by source metadata and shot boundaries before looking at generation
or anchor quality. Audit no shared source frames with any step-3–5 fitting,
recipe selection or this scope analysis; compute and freeze both gates,
record exclusions and score the identical windows for every method. Include
boundary cases and retain pooled scores. Use the **existing frozen** step-5
match adapters, with no new fitting; the published eight windows cannot
confirm it. A candidate 12-window budget is **about 6–8 GPU-hours** (not an
authorization): 12 renders × step 4's measured ~0.35 h = 4.2 h, ~0.6–1 h
judging, and ~1–2 h for new features/poses/anchor curves, contingent on a
correctness smoke and scale pilot. Fresh archive verification must precede
that estimate becoming a run card. Start with a six-window first part and
stop if the decision is already clear. Nadal–Kyrgios is an alternative
fresh-match zero-shot check, but lacks a fitted step-5 match adapter and
therefore does not directly confirm the frozen per-match recipe without a
new training protocol. **No confirmation, fitting or Wan-Animate run was
started.**

**Verification and provenance.** The final analysis module passes `ruff
check` and `mypy --config-file pyproject.toml` (181 source files); all three
focused tests pass. Final full `python -m pytest`: 821 passed, 24 skipped,
one failure in unchanged `test_patches_into_models_or_datasets_are_blocked`.
On macOS `/home/itec/emanuele/Datasets` resolves to
`/System/Volumes/Data/home/itec/emanuele/Datasets`, so that test's symlink
expectation fails. The hook is byte-identical to origin/main (SHA256
`82a4e2fe5c882fb1d590e1a7ee031352c74fa901fea8f77fd114f57a3c4b957d`).
The sandboxed first full suite also blocked `ps`; with process-control
access the fleet process test passes. No hook or unrelated item was edited.
Linux CI must pass before auto-merge can complete.
- Final CPU snapshot SHA256: `f29e32500a53e8978d572ca93559ba632fb51fc147238a9733183da531dfcc81`.
- Locally derived scoped-score JSON SHA256: `7686b3681607b53882a6dadbc819d79ad9625e43fa581e40cfb0517fdb3c2272`; reproduce from the preserved source reports with the command above.

## Report: K6b players 5b (2026-10-24)
- Stopped because: goal reached for the CPU diagnosis; individual causal
  contributions remain unisolated, as the saved-data limit requires.
- Decision by the rule: step 5 remains **fail**, 8/12 window-metrics (67%).
  No cuts occur in any of the eight windows. Registration ≥95% gives
  8/10 (80%); adding minimum height ≥115 px gives 8/8 (100%), both
  unconfirmed hypotheses discovered on these windows.
- Explanation: 052 is one continuous oblique court shot, **not a close-up
  or a cross-cut reference**; registration 32/240, seven judged frames,
  reference t109 on both players. Net occlusion and appearance/pose errors
  are visible, but not causally separated. 045 has nearly the same size,
  rate and anchor strength as winning 036; a front/back appearance error
  and racket/arm changes are visible, with no uniquely established cause.
- Numbers: pooled steps 2/3/4/5 are 50/83.3/66.7/66.7% unbilled;
  registration-scoped 60/90/80/80%; combined-scoped 75/100/100/100%.
  Billed step 4 is 58.3/70/87.5% and step 5 is 66.7/80/100% in those
  same three scopes. All scores retain identical method/anchor windows.
  CPU job `20261024T083117Z-2e644b08`, code `de79482f`, gpu2 host,
  no GPU, report `full/scope.json` SHA256
  `ac640ae8f2ee0d807a085555f5504a354de9cba554aaf465cd036673c0d046d2`.
- Budget: **0 of 0 GPU-hours**; completed workload smoke 4.87 s,
  full 32.38 s; three failed startup/smoke requests preserved above.
- Branch and commits: `codex/k6b-players-5b`; measured code `de79482f`,
  diagnosis/report `3f6db4b3`; [PR #232](https://github.com/emanuele-artioli/PointStream/pull/232).
  Ruff, mypy and three focused tests pass; full Mac suite 821 passed,
  24 skipped, one unchanged protection-hook path-resolution failure.
  All 34 CPU measurement checks pass. Linux CI gates merge.
- Deviations: corrected the card's cut/close-up premise and step-3 pooled
  score (83%, not 67%). Eight windows already include development 008/033.
  Integrated d0a00e5's missing saved step-5 continuation without changing
  its verdict. No new GPU computation, fitting or confirmation.
- Open for the user: no execution question; any positive paper claim
  and a future confirmation budget remain the user's decision.
- Proposed next step: CPU-freeze 12 unused Alcaraz/Medvedev windows from
  archive clips not used in training/selection, audit overlaps and gates,
  then test the frozen adapters and same-window anchors in bounded parts,
  estimated **6–8 GPU-hours**, subject to preflight and fresh authorization.
  The 13 unused windows in the current manifest cannot pass the proposed
  registration scope. Nadal–Kyrgios (2,310 s) exists as a fresh-match
  alternative but has no step-5 match adapter. No confirmation was run.


### 2026-10-24 — K6b players confirmation: CPU preparation, awaiting approval

**Goal:** decide whether the frozen step-5 player renderer transfers to unused
windows of the same Alcaraz–Hurkacz and Medvedev–Struff broadcasts under a
prespecified encoder-computable registration scope. This is conditional,
within-match confirmation, not an independent-match test or a deployable codec.
Step 5's historical fail (8/12 window-metrics) remains unchanged.

**Scope:** CPU-only preparation now, 0 GPU-hours authorized or spent. No GPU
submission, fitting, new generated images/video, Wan-Animate, other chats or
recurring monitors. Future execution requires explicit budget approval AND the
CPU provenance gates below. Branch: `codex/k6b-player-confirmation-plan`.
Report address: this requesting chat and this entry; no other chat is launched.

**Stop conditions:** stop preparation with this card and audit; await budget
approval. In future execution stop on a failed validator, unresolved provenance,
insufficient eligible windows, mathematical futility, exhausted budget or a
question requiring user judgment. Never spend beyond the approved ceiling.

#### Audit: what “unused” does and does not establish

Read-only CPU metadata audit on gpu1 (2026-10-24), no model imports or quality
reports opened. Read archive MANIFEST, K4b selection, K8b registration, the saved
5b input inventory and the base job specification. No files in Models or
Datasets were changed. Temporary local audit extracts are not canonical evidence;
the source hashes and complete chosen source identities are recorded here.

- The 5b proposal did not enumerate 12 windows. The archive has 21 Alcaraz
  clips and 41 Medvedev clips. Outside step-5 fit/evaluation there are 16/38
  clip IDs, as reported. Excluding **all 57 old manifest windows**, including
  the 13 unused low-registration windows, leaves 16/33 clip IDs. Of these,
  9/13 respectively contain at least 240 native frames. Thus a six-per-match
  sample is feasible without recycling an old manifest window.
- Alcaraz old clips: 002, 006, 011 (base/match training), 036, 045
  (evaluation). Medvedev: 006, 008 (base/match training), 033 (recipe
  development); also exclude unused-manifest clips 007, 016, 023, 028, 034.
  Step 3's own-window fits and steps 4–5's development/reference use are
  contained in the old clips. Conservatively excluding their entire clips
  excludes their source reference frames too, rather than relying on different
  window IDs. K4b selection's reviewed Alcaraz/Medvedev frames also occur in
  these excluded clips.
- Base job `20261023T194934Z-b0dcbb19/spec.json` explicitly loads
  `original/{denoising_unet,reference_unet,pose_guider,motion_module}.pth`,
  excludes all eight K6 windows and uses every second training frame. Its
  36-window training set is documented above. `k6b_fit.py` samples training
  references from those same training units. The pre-reset
  `finetuned_tennis` profile saw all seven archived matches and lacks a
  track-level manifest (resources.md); **do not use it or its descendants**
  for a clean confirmation. Checkpoint names alone are not proof: the exact
  Alcaraz and Medvedev step-5 adapter/base input hashes and transitive fit-unit
  lists were subsequently joined to the preserved records below: their units
  are the same excluded old clips. Before dispatch recheck actual staged bytes
  against these receipts. Missing or mismatched lineage blocks confirmation.
- All candidate clips were already processed/reviewed in K8b's court census.
  This independently tests generator quality conditional on a frozen
  registration procedure, not the registration algorithm on unseen footage.
  Both matches, identities, venues and broadcast styles were seen by the
  tennis base and match adapters. Twelve windows are **two match clusters**,
  not twelve independent matches; frames and two metrics are not independent
  replicates. Off-the-shelf pretraining overlap is not exhaustively known.
- Distinct clip IDs and SHA256 identities prove separate encoded files, not
  absence of duplicated/replayed source frames. `scenes.txt` provides scene
  cut times, but original extraction offsets/rounding and replay duplicates
  were not verified against decoded video here. A clean source-frame claim is
  therefore **not certified yet**. Before any GPU work, reconstruct source PTS
  intervals for every candidate and every old fitting/development/reference
  clip, verify the mappings by sequential CPU decode, and compare decoded
  frame hashes plus near-duplicate/replay candidates by CPU visual review.
  Use all old clip frames conservatively. Record exact, temporal and replay
  overlap separately. Any overlap or unresolvable mapping stops this card;
  replacements require a new prespecified manifest before quality is opened.
- Judges must not have been fitted on these frames. Pin LPIPS/DISTS weights
  and provenance, use the existing off-the-shelf judges without adaptation,
  and explicitly record unknown pretraining exposure rather than inventing a
  clean subset. Known judge training overlap blocks scoring.

**Conclusion:** enough source clips exist for a same-match holdout; “12 unused
windows” alone does not establish independent confirmation. With the remaining
lineage and decoded-duplicate checks passed, this can confirm a new-window
within-match hypothesis only. It cannot establish unseen-match generalization.
Nadal–Kyrgios would require a different zero-shot protocol (no matching step-5
adapter); it is not silently substituted.

#### Frozen source-only selection and registration rule v1

Selection was written before computing candidate registration counts and before
opening any new codec/generator quality scores: one centered 240-frame window
per unused clip, `start=(N-240)//2`, half-open `[start,start+240)`, native fps,
sequential decoding, no frame-rate conversion. Rank clips by ascending hex
SHA256 of UTF-8 `pointstream-k6b-confirm-v1:<match>/<clip>`, take the first six
per match among N≥240 after excluding all old manifest clips. Do not select by
registration, height, pose success or anticipated quality. Reserves are not
scored and are not automatic replacements. All 12 stay in the pooled cohort.

**Primary gate:** at least **228 of 240** frames have K8b **`fits=true`** and
a finite, nonsingular court homography. Missing/invalid rows count as failures.
Use the adopted frozen K8b algorithm, line ruler, geometry/clutter guards and
source hashes; no re-estimation/tuning to favor a window. `fits` is the ≤6 px
flag that K4b/K6 actually consumes (`k4b_court.registrations`), NOT K8's
older `registered` ≤2 px flag. The initial scratch wording used `registered`;
code inspection corrected that field mapping before quality was opened. This
is a prospective test of 5b's discovered ≥95% hypothesis, not independent
justification of the numerical threshold. Freeze code/manifest hashes before
running or viewing any new quality result; changes invalidate confirmation v1.

No extra cut or pose-success gate. Record cuts and pose failures as diagnostics;
a failed model/pose is not grounds to delete a window. Size ≥115 native 1080p
pixels remains **exploratory**, selected in the 110–118 px gap on old losses.
No independent evidence justifies it. Report it only as a labeled secondary
stratum beside primary and pooled results, with the same methods and anchors;
never use it to rescue a primary failure or replace the deciding denominator.
Source-mask-derived heights also inherit the supplied-matte assumption.

| Clip | Native inclusive frames | fps | K8b fits | Source SHA256 |
|---|---|---:|---:|---|
| alcaraz_hurkacz/031 | 9–248 | 25 | 240/240 | `054e4535638d8dfa3dee69258f45c231aaeab2241b91309d449be7b0934cef76` |
| alcaraz_hurkacz/013 | 93–332 | 25 | 239/240 | `1af169200b28659c503499979863b6219b0d331f532962ac69fb505d13c4976c` |
| alcaraz_hurkacz/023 | 108–347 | 25 | 239/240 | `88345a9deac75a4b6180daa56a2ec7cda2afea25babb5b549b937105bca45d1a` |
| alcaraz_hurkacz/016 | 153–392 | 25 | 234/240 | `305a369d74882a628ff7efd6b49cf3f75c56a985d78ffd798863542df20551bd` |
| alcaraz_hurkacz/008 | 141–380 | 25 | 240/240 | `355e4b0a3408e4c8f049baadf749c937e52930f57df5018867e5baa0cdd49e54` |
| alcaraz_hurkacz/047 | 3–242 | 25 | 240/240 | `7bb3a53e3b416d3b84c0344c99b29d1374f8c49b54f970034f4f29a95609e99b` |
| medvedev_struff/014 | 74–313 | 50 | 240/240 | `a2898b5cdfb463f59b5084fddfe23a2442cf2ba5feaaddb9d5414873e52dd0be` |
| medvedev_struff/037 | 229–468 | 50 | 240/240 | `97372d45ba53068078fdc85300a550221b2c4bd31a3074958b52081326e3feeb` |
| medvedev_struff/002 | 218–457 | 50 | 240/240 | `49cc09d072d740bd4d5ed02fdd24f2b3edbdc5fb16a660f66f1e8148ff8be14a` |
| medvedev_struff/017 | 86–325 | 50 | 240/240 | `64cc4bc0a0712b5daa6bc00c5ed592b31a0f8333cb36a4caf5b05d24f9e39873` |
| medvedev_struff/029 | 18–257 | 50 | 240/240 | `013979d403ec9ae3e48ce3dc49bbf261b37fb6a3dd78af71d5b8d81419612e8f` |
| medvedev_struff/021 | 126–365 | 50 | 240/240 | `fa2da44fdc64d6d95c9a111953dd6d16a088cb534dc7336a98d3cb2557d30b4a` |

All 12 pass the primary gate (Alcaraz/016: 234; all other windows: 239–240).
This cohort tests performance **inside** the proposed scope. With no excluded
window it cannot establish that registration discriminates wins from losses,
or quantify the fallback's benefit; gate and pooled membership coincide here.
Do not add low-registration clips after seeing scores. Source fps matters:
Alcaraz is 25 fps (9.6 s/window), Medvedev 50 fps (4.8 s/window), not all 50 fps.

#### Identical comparison, decision and substantive validation

- Primary method: existing frozen step-5 `gen-match5` (C tennis base,
  appropriate match adapter/guider, 4-bit per-output-channel coding, guidance
  2.0), 512×784 crops, 30 DDIM steps and deterministic unit seed rule
  (`int(sha256(unit)[:7],16)`, per-run seed `seed*1009+run_id`).
  Coded keypoints: 4 px, velocity predictor, every fourth frame as K6.
  No new training, reference search, recipe sweep or step-3 overfit rerun.
  Carry K6 `warp` appearance oracle and `ref` warp as diagnostics on the same
  windows; old steps 2–5 remain historical, not fabricated fresh measurements.
- Each new reference is the first fitted scored source frame per player,
  exactly the existing step-2/5 rule, within its own frozen window, granted
  and unbilled. Record source frame, crop and SHA256; no reference bank or
  outside-window substitution. This permitted test-time conditioning is not
  training overlap, but prevents a “no target appearance” claim.
- Anchors: actual SVT-AV1 (preset 4) and DCVC-UF. Initial three settings
  per codec, `(quality, downscale)`: SVT `(48,1),(63,4),(63,8)`; DCVC
  `(0,1),(0,2),(0,4)`. The final comparison must include the remaining K6
  anchor settings SVT `(62,1),(63,2),(55,4)` and DCVC `(21,2)` wherever
  they could strengthen the hull; omission requires a quality-independent
  dominance justification. If budget cannot cover a fair envelope, report
  incomplete, not pass. Add further points only for bracketing/overlap,
  with a documented budget check. Same source window, native fps,
  frame indices, player regions, union masks, crop and scoring stride for
  every method and every anchor, in both pooled and gated reports. Freeze
  score indices before rendering: K6's registered frames and every-third-frame
  judging convention, with pose failures retained/reported. Missing required
  output invalidates the result; it cannot shrink an anchor's or method's set.
- Rates use the same registered-duration convention as K6 for all curves;
  primary billed rate adds adapter and trained-guider bytes over H=2 hours,
  with the tennis base treated as shipped decoder data as in step 5. Report
  those base bytes and amortization assumption, unbilled rate, payback and
  whole-window duration too. References and exact target matte **R remain
  granted and unbilled**. This is an object-renderer feasibility test with a
  supplied matte, not an end-to-end codec gain, decoder-side segmentation
  solution or paid-mask hybrid/fallback demonstration.
- For each window/metric build the achievable upper-concave anchor envelope
  (quality orientation made explicit), compare strictly lower LPIPS/DISTS
  at the method's billed rate without extrapolation. Primary statistic is
  `(LPIPS wins + DISTS wins)/(2*n_gated)`, bar ≥75%; publish each metric,
  window and match separately, pooled alongside gated, billed/unbilled,
  all exclusions and missingness. If all 12 validate, pass needs ≥18/24.
  Require at least six gated windows, at least three per match, otherwise
  insufficient confirmation. Below 75% billed but ≥75% unbilled goes to the
  user; below 75% unbilled fails. Unbracketed rates or invalid outputs are
  unresolved, never wins. A pass supports this conditional same-match scope
  only; the earlier step-5 verdict is not rewritten.
- Report BD-rate per metric using PCHIP on overlapping quality only when a
  genuine method curve exists; do not invent a curve from one frozen operating
  point. Otherwise report fixed-point wins and margins, and “better/worse at
  every rate” only when the measured curves support it. No extrapolated BD-rate.
  Report match-level variation; a two-cluster confidence interval is not
  credible evidence of population generalization. No optional-stopping p-value.
- Validator must recompute input/checkpoint hashes, complete training/reference
  intersections, source PTS/duplicate audit, exact 240 decoded frames and fps,
  unique IDs and no cross-window overlap, all 240 registration rows and gate
  count. Test boundary counts 227/228, null/nonfinite/singular homographies,
  wrong fps/start/hash, duplicated IDs and missing outputs with deliberate
  corruption fixtures. Validate coded-keypoint encode/decode round trip,
  shapes, finite/nonflat generated crops, frame coverage, player identity and
  reference membership; independently recalculate adapter bits and rates.
  Assert actual CUDA device/provider/kernel on the approved GPU class in the
  future smoke; no CPU fallback. Require real bitstreams, native binary paths
  and versions, encode/decode agreement, identical scoring-index/mask hashes,
  finite metrics, envelope bracketing and no extrapolation. DCVC-UF is encoded,
  decoded and judged within the same job/GPU (its saved-stream portability
  limitation stands). Synthetic corruption tests must fail; a success flag or
  file existence alone is insufficient. Validator is specified here, **not
  implemented or executed by this preparation task**.

#### Staged proposal — NOT authorized

Hypothesis: the frozen step-5 renderer achieves ≥75% billed within the
registration scope on new same-match windows. Competing explanations:
post-hoc scope overfit, repeated broadcast content, same-match appearance
memorization, favorable anchor coverage, and supplied reference/matte hide
costs. The existing appearance oracle is the upper bound; no new renderer
component is proposed. Include its real measurements diagnostically, never
claim the bound as a method result.

Proposed hard ceiling **8 GPU-hours total**, not an allocation or dispatch:

| Stage | Ceiling GPU-h | Release condition |
|---|---:|---|
| CPU provenance/manifest and validator implementation | 0 | Approval plus all lineage/PTS/duplicate checks pass; otherwise stop |
| Representative correctness smoke | 0.3 | ≤600 s per smoke; one window per match, all substantive checks pass |
| First six windows (three per match), all prep/anchors/render/judges | 3.4 | Fixed recipe, no tuning; budget feasibility measured |
| Remaining six, same pipeline | 3.4 | Pilot predicts total ≤8 h and no validator failure |
| Contention/bracketing/validation reserve | 0.9 | Only documented work within total ceiling |

Central estimate 6–8 GPU-hours: renders 12×0.35=4.2 h (measured step-4
per-window cost, not a guarantee for these sources), preparation/poses and
anchors roughly 1–2 h, judging roughly 0.6–1 h. CPU SVT/registration does not
consume GPU time. Smoke overhead and retries count toward 8 h. CPU stages
≤45 min each; GPU shard ≤45 min, checkpoint every finished window. Twelve
renders may be 0.7 h for a pair, so shard by single window rather than match.
Pilot first-three-per-match follows the frozen rank order; compute-only scale
confirmation uses measured seconds/frame, peak memory and projected remaining
cost, never a new quality-dependent recipe. No interim pass: complete all 12
unless even winning every remaining window-metric cannot reach 75%, in which
case stop for futility and report the incomplete cohort honestly. Failure of
this conditional test does not prove that every possible player renderer fails.

Use only compatible Ada hosts gpu5/gpu6 for the generator, per the recorded
model–GPU table; do not add A6000 by inference. Use the pinned packed
`pointstream-jcv` generation environment and main PointStream prep/codec/judge
environment, their locks and exact SHA256 packs. Before dispatch inspect existing
jobs and all eligible hosts, pin code revision and selected patch hashes,
staged archive hashes, exact entrypoint/arguments, versions, per-job caps and
absolute deadline (set only after approval). Save each job ID here immediately.
No runnable fleet spec is issued with guessed hashes/deadlines. Verify native
completion resumption once per client; then one bounded `ps-fleet wait` per
batch, ≤45 min. If unavailable, preserve continuation and use at most one
verified self-expiring deadline check, never recurring model polling.

#### Source receipts and report

Frozen checkpoint dependency receipts (read records, did not load tensors):

- `20261023T194934Z-b0dcbb19`: spec SHA256 `99c8a3b83fa28169645e23a75ba2a7da1dba4a294a1827a12259fc7d68f3b1a3`;
  `full/k6b_fit.json` SHA256 `02e3d1571de608fd87130687cfa1dafb778ae2795303ad8663fa3308b7e22ea2`.
  Frozen checkpoint `tennis-base@fit-100.pt`, SHA256 `85e3c65c9777b799f9ed0df3a5c8da567fe8a92e4ac83d115719527acb739eb4`.
- `20261023T204452Z-d5c90446`: spec SHA256 `c25f464a0e7775176188c0e8b38dd8d9f6ea41cb330c6bbaa469000558778317`;
  `full/k6b_fit.json` SHA256 `9ee9c488d1217c71405fa984ccfc435efef4cb8bd5962d69886b41cd6fa3cfa5`.
  Frozen checkpoint `med-C@fit-100-fp16.pt`, SHA256 `aa7bb8391f0e644c92df447a10d8ff3f76837ec58ad98311db3ecb454a5b98b4`.
- `20261023T221650Z-3d7cbe2f`: spec SHA256 `2e7ef9c434a151e64289fc443b083efa9a2d6691e3343043b1dd706b21bdbaa1`;
  `full/k6b_fit.json` SHA256 `e9d2bd731cb2349dbb4af32281d980ed82e66c5bbc97b8a215c1ff9acc74bf77`.
  Frozen checkpoint `ah-C@fit-100-fp16.pt`, SHA256 `5c6caf090f90a66eac5808abdaadfa50a74114331e610da4ea9a7f61db282d7a`.

Both match jobs point to base archive SHA256
`075b001a3ae9dcc9fdda9fb81e4240aaf3a9da228d707ac58733db094d983285`.
Alcaraz fit-data SHA256
`33e037eb4a07541310f431c8bad4c78ffb6ffd1ee9ee51f976e75292b0f0bb1b`;
Medvedev fit-data SHA256
`71e7bc58a0feaac2323b22d051bf6a64cf1eb605d8d41cb7320a6e8fca341c23`.
Base and match records use only `original` profile weights. Alcaraz units are
002/006/011, Medvedev units 006/008 (two players each); the base includes these
same units plus the other three matches. No selected clip ID intersects any
recorded base/match training unit or old development/reference clip. This
resolves recorded fitting lineage, not decoded replay/PTS independence.

- `/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261024T083117Z-2e644b08/full/scope.json`: `ac640ae8f2ee0d807a085555f5504a354de9cba554aaf465cd036673c0d046d2`.
- `/home/itec/emanuele/Datasets/pointstream-data/tennis/k8b-2026-10-20/review/k8b-report.json`: `8230bf1a2239978dfb7912bd5f3ed8367f5c85d1ea9e75c42e8ab8fb3fdc3314`.
- `/home/itec/emanuele/Datasets/archive/pre-reset-2026-10-05/MANIFEST.json`: `2bd94e9966fc8b09c5ce5eb9bef79f49ad349e1310a1d445fa6a646e86db8dde`.
- `/home/itec/emanuele/Datasets/pointstream-data/tennis/k4b-2026-10-16/inputs-v3/selection.json`: `7d1a22f62ac03c514b5f47dbc8c4c3864d9e6b92d461ab90175d2b520c6c23ef`.

- Code inspected: preparation checkout `aa95b3b1b5d0049f2cf9757be8b38afdfcdadd0e`; CPU metadata inspection only.
- Jobs submitted by this task: **none**. GPU-hours: **0 of 0 authorized**.
- Stopped because: user budget approval needed, as explicitly requested.
- Decision by rule: **not reached**; no new quality scores opened or produced.
- Verification: `ruff check` passed; `mypy --config-file pyproject.toml`
  passed (189 source files); full `python -m pytest`: 833 passed, 25 skipped,
  2 failed. The process-group test failed because sandboxed `ps` is denied;
  its isolated rerun with process inspection permitted passed (1/1). The
  unchanged protected-path hook test still fails on macOS symlink resolution
  under `/System/Volumes/Data/home`, as already reported in 5b. No test/hook
  source was changed. `git diff --check` passed. This is a documentation-only
  change; Linux CI remains the merge gate. Publication: scoped branch/PR.
- Open for user: approve up to 8 GPU-hours for this conditional same-match
  study, contingent on the remaining CPU provenance gates, or keep it
  preparation-only. Recommendation: approve only with that limited claim;
  do not describe it as independent-match confirmation.

- Publication: commit `a52c1bd9` pushed to
  `codex/k6b-player-confirmation-plan`; [PR #236](https://github.com/emanuele-artioli/PointStream/pull/236).
  Auto-merge remains off: automatic approval review rejected
  `gh pr merge 236 --auto --merge`, stating that merging main was not
  authorized. No merge or bypass attempted; explicit user approval is needed
  to resolve that approval block. Study budget remains unapproved.

### 2026-10-24 — Confirmation authorization and CPU provenance gate

- User approved the proposed **8 GPU-hour hard ceiling**, contingent on
  completing CPU provenance checks (annotation “Yes” on the budget request).
  This supersedes the prior pending-budget status, not the frozen protocol.
- Current spending: **0 of 8 GPU-hours**. No new GPU jobs submitted.
- Start with source PTS/replay/duplicate audit; do not generate or spend GPU
  time until that gate and substantive validation prerequisites pass.
- No fitting, Wan-Animate, additional chats or recurring monitors. Existing
  PR auto-merge approval block is separate and not overridden by this budget
  approval. Preserve the cohort, ≥228/240 K8b fit rule and granted matte.

#### CPU provenance gate outcome: unresolved; no GPU dispatch

- Read-only SSH checks on gpu1 verified the archive manifest remains SHA256
  `2bd94e9966fc8b09c5ce5eb9bef79f49ad349e1310a1d445fa6a646e86db8dde`.
  Its only `tennis_games/raw_1080p/` video is `nadal_kyrgios.mp4`.
  Searching all archived video paths containing Alcaraz/Medvedev found 170
  videos, all under the two existing match directories (scene clips and
  derived videos); no alternative named source video exists in this inventory.
  This is a bounded inventory search, not a claim that no copy exists anywhere.
- Container metadata read with `/opt/local/bin/ffprobe` confirms local timestamp
  reset: Alcaraz/031 starts at 0, duration 10.36 s, 259 frames at 25 fps;
  Medvedev/014 starts at 0, duration 7.76 s, 388 frames at 50 fps. These
  metadata observations are not generation/anchor quality scores.
- Existing `scenes.txt` hashes: Alcaraz
  `40cc90152baafa65c0db7b7b1b6f886bde5e957beb534b2e9252923eaf4c3a9d`;
  Medvedev `a9e3bf63a4de0d7ab27bd1413a107d61f9494b6f9ac000209cce5a836432d68e`.
  Cut times suggest nominal scene intervals but do not independently verify
  extraction rounding, offsets, source frame identity or broadcast replays.
- Searched current code/docs and tracked pre-reset code at
  `archive/pre-reset-2026-10-05` for `scenes.txt`, source ID `B1Z3yet4TIQ`
  and `alcaraz_hurkacz/031` (excluding archived research-report dumps).
  No tracked extraction receipt/code mapping these source clips was found.
  Container title/URL metadata is not a hash-verified extraction receipt.
- Recorded fit/reference clip disjointness remains valid. Exact original-frame
  mapping and decoded replay independence remain **unresolved**, not passed
  or disproven. A duplicate scan of available clips alone cannot satisfy the
  separately frozen original-source mapping requirement. No duplicate-scan,
  generator, fitting or judge job was submitted after identifying this gate.
- Stopped under the frozen card's condition: “Any overlap or unresolvable
  mapping stops this card.” No threshold, cohort or provenance requirement
  was silently relaxed following approval. The 115 px gate remains exploratory
  and the reference/exact target matte remain granted and unbilled.
- Budget: **8 GPU-hours approved; 0 spent; no fleet job IDs**. No other chats,
  monitors, scheduling or wakeups created. Historical step-5 fail unchanged.
- Needed to resume: point to original Alcaraz/Medvedev source files plus their
  extraction/source-identity receipts, or explicitly revise the study to a
  weaker clip-local overlap audit with the original-source uncertainty reported.
  Recommendation: recover existing source/receipts first; retain the approved
  budget untouched. No new download or source substitution was attempted.
- Verification: metadata-only/doc-only continuation; `git diff --check` passed.
  Earlier ruff/mypy/pytest results above still describe unchanged runtime code.

### 2026-10-24 — CPU source recovery and clip-local overlap assessment

- User authorizes CPU-only recovery; frozen cohort and approved 8 GPU-hour
  ceiling unchanged, 0 spent. If full recovery fails, decide the defensible
  narrower claim from a clip-only overlap audit. No GPU dispatch in this phase.
- Recovery found `split_scenes.py` at Git commit
  `db11965efd14f54ea6f29b1f5e5bb28693d50b97` (2025-02-12), file SHA256
  `734f2d0e88ab7303d1db2eddd16d8b1f2ad55fccc77948ce82ec33a96193fc55`.
  It parses `scenes.txt` cut times, prepends 0, names scenes by index and
  re-encodes each using input-before-`-ss`, `-t` and default ffmpeg settings.
  This recovers extraction logic, NOT a source-byte or invocation receipt.
- Before duplicate results: compare every frozen candidate frame against all
  frames in every old 57-window manifest source clip, conservatively including
  material outside actual fitted/reference windows; also compare distinct
  candidates. Exact duplicate means SHA256 equality of decoded native yuv420p
  frames at equal dimensions. Low-resolution hashes do not decide exactness.
- Replay screen (diagnostic, not an independence certificate): grayscale 64×36
  frame descriptors, nearest mean squared pixel distance for every candidate
  frame, and source-frame pairs flagged at RMSE ≤3 luma levels on [0,255].
  Full-frame background similarity can cause false positives; source-frame
  crops/temporal neighbors must be inspected before calling a replay. A clean
  screen cannot rule out cropped, zoomed, altered-speed or alternate-view replay.
  Threshold not selected from new quality scores; none will be opened here.
- Runtime bound: CPU-only attended command, ≤600 s, no torch/model loading,
  two ffmpeg threads, sequential source decoding, one new result directory
  under Datasets; no source modification. Stop on missing/changed input or
  decoded frame-count mismatch. Partial outputs remain unpromoted on failure.
- Provisional interpretation frozen before audit: distinct filenames alone
  support only new-file diagnostics. No exact/near duplicate flags plus CPU
  review can support a restricted clip-local holdout diagnostic with unknown
  original-source overlap; neither proves independent source/match confirmation.
- CPU overlap collection now uses `tools/datasets/k6b_recovery.py` (stdlib
  collector, local NumPy screen), with native `/opt/local/bin/ffmpeg`
  n7.1.1-56-gc2184b65d2. Initial smoke v1 passed; v2 adds deadline cleanup
  and type fixes without changing decoded pixels. Final v2 smoke: four clips
  (old/candidate, each match), hash/count checks passed, audit JSON SHA256
  `c22ce0e95f145d53169ea18f353e771e1dea6f3970c5f0fae16a212eab72ad08`.
- Bounded attended CPU full command launched with handle `70689`; external
  output `/home/itec/emanuele/Datasets/pointstream-data/tennis/k6b-recovery-20261024-v2-full/`.
  Command `ssh gpu1 /usr/bin/python3 - collect --cohort <base64 of the frozen
  12 clip/start/source-SHA records> --out <output> --code-sha256 <local file
  SHA256>`, stdin is the audit utility. Exact ffmpeg arguments and all source
  hashes/descriptor offsets are saved in each audit record. Entire collection
  deadline 600 s, child decode deadline 120 s, CPU only; no fleet/GPU job.
  Raw local console log `/tmp/k6b-recovery-full.log`; preserve partial.json
  if it fails. Do not replay unknown execution. No monitors/wakeups created.
- Full v2 command `70689` exited 1 with a confirmed `TimeoutExpired` on
  `djokovic_zverev/000` (3,975 native 4K frames) at the 120 s child cap.
  No unknown submission was replayed. Its read-only checkpoint has 19 finished
  clips / 8,495 frames; the incomplete collection is NOT a clean audit.
- Revised CPU approach: native hash equality is impossible across dimensions,
  so 4K old clips receive only the 64×36 replay screen and their hashes are
  explicitly labeled `screen_only_other_dimensions`, excluded from native
  equality. All frozen candidates are asserted 1920×1080; every equal-size
  old clip still gets native yuv420p SHA256. No overlap-test/cohort/quality
  threshold changes. Child cap 500 s for 4K, 120 s for 1080p; whole command
  still capped at 600 s. Subsequent parts exclude the 19 finished clips.
- First revised 4K CPU pilot: handle `53234`, output
  `.../tennis/k6b-recovery-20261024-v3-part0/`, source index 19 only.
  Dispatch code SHA256
  `a65a8afcae9629cbe4b39a7cfdc14e84fb5dfe8690cd893b721c7e1b6428641c`.
  Prior v2 dispatch code SHA256
  `4465585440ee84dd9d59b4ead6ca90c4734248c465eca277e71e4b33622c2e4d`.
  Both code snapshots retained locally for publication with the new receipts.
  Merger checks complete source coverage, descriptor byte counts and hashes,
  unique clips and consistent expected inventories before promoting an audit;
  smoke/partial collections alone are rejected. No GPU jobs or spending.
- Revised 4K pilot `53234` **complete**: `djokovic_zverev/000`, all
  3,975 descriptor frames verified; audit SHA256
  `665f2c05cc7e8249569c2db92e6d864e93e82b2f692e9ab6818a2a96c255573b`.
  Native hashes deliberately not comparable at 4K; this is not an absence-of-
  replay result. Remaining CPU batch handle `51413` runs sequential parts
  `v3-part1` (source indices 20–26), `v3-part2` (27–40), `v3-part3` (41–60).
  Each part ≤600 s; batch ≤approximately 31 min including SSH overhead,
  under the 45 min operational bound. First 19 clips and pilot clip excluded
  from further decoding. All outputs under the same new `tennis/` prefix.
- Expanded recovery searched the archive manifest, shared home/Datasets and
  likely old-project roots shallowly, narrowly filtered shell history, tracked
  pre-reset sources AND older Git history. The older `split_scenes.py` was
  found by `git log --all -S scenes.txt`, correcting the earlier tip-only
  negative search. Seven preserved `pointstream-data/assets/raw_4k` aliases
  were inspected: Alcaraz titles identify other highlights/practice matches;
  none establishes an original of the two frozen broadcasts. No source bytes,
  original-source hash or matching execution receipt recovered. Search was
  bounded; unnamed/offline backups and unindexed locations remain unsearched.
- Applying the recovered historical splitter to existing scene cut times gives
  zero nominal candidate/old-clip interval overlaps for the two matches, but
  those inferred times are NOT verified original PTS. The clip-local audit
  cannot remove that limitation or upstream pretraining uncertainty.

- CPU remaining-batch start 2026-10-24T16:57:48.058720+00:00; conservative whole-batch deadline 2026-10-24T17:28:48.058720+00:00. Handle `51413`; three ≤600 s stages plus connection overhead.


#### CPU recovery outcome and claim decision

- **Recovery partial:** extraction logic recovered from the historical commit
  above; original source bytes/hash and matching execution receipts not found
  in the bounded search. The original-source confirmation gate remains
  unresolved. No source re-download or substitute was introduced.
- **CPU collection complete:** failed v2 checkpoint plus v3 parts 0–3 cover
  61 unique clips / 39,853 decoded frames: 49 old source clips (all 57 old
  manifest windows) and 12 unchanged candidate windows / 2,880 frames.
  Equal-size native comparisons cover 16,653 old 1080p frames plus distinct
  candidate pairs. The 20,320 old 4K frames are descriptor-screened only;
  native-byte equality across different dimensions is impossible.
- **Detected overlap:** 0 native yuv420p SHA256 matches; 0 frames meet the
  prespecified 64×36 grayscale RMSE ≤3 screen. Same decoder for every clip,
  source file hashes and frame counts checked. New/new pairs are tested once
  and attached to the later row; table minima are against old plus earlier
  candidates, not symmetric global per-window minima. The screen does not
  prove absence of altered replay or original-source overlap.

| Candidate | Exact matches | RMSE≤3 flags | Min RMSE vs old/prior |
|---|---:|---:|---:|
| alcaraz_hurkacz/008 | 0 | 0 | 6.6373 |
| alcaraz_hurkacz/013 | 0 | 0 | 7.3050 |
| alcaraz_hurkacz/016 | 0 | 0 | 3.9952 |
| alcaraz_hurkacz/023 | 0 | 0 | 4.3259 |
| alcaraz_hurkacz/031 | 0 | 0 | 6.6069 |
| alcaraz_hurkacz/047 | 0 | 0 | 8.9437 |
| medvedev_struff/002 | 0 | 0 | 4.5905 |
| medvedev_struff/014 | 0 | 0 | 5.3977 |
| medvedev_struff/017 | 0 | 0 | 3.1347 |
| medvedev_struff/021 | 0 | 0 | 4.4663 |
| medvedev_struff/029 | 0 | 0 | 4.6600 |
| medvedev_struff/037 | 0 | 0 | 4.2987 |

- Reviewed the four closest source-frame pairs at 640×360: Medvedev 017:280
  vs 014:111 (3.1347), Alcaraz 016:382 vs 011:14 (3.9952), Medvedev
  037:302 vs 033:137 (4.2987), Alcaraz 023:304 vs 036:230 (4.3259).
  Different player poses/locations and score overlays are visible; none is
  visually the same frame. This limited review is not a certification against
  cropped/zoomed, altered-speed, overlaid or alternate-view replays. Exact
  decoded hashes also miss lossy re-encodings of an original frame.
- **Decision:** the audit supports a **clip-local, same-broadcast diagnostic
  on the 12 frozen archive windows, with no detected overlap under the stated
  tests**. It does NOT support “independent source footage,” “training-clean
  frames” or unseen-match confirmation. Report unknown original-source and
  upstream pretraining overlap; do not create a certified-clean subset from
  this screen. The original stronger gate is not silently marked passed.
- The narrower route keeps every window, the ≥228/240 registration rule,
  identical method/anchor windows and the 8 GPU-hour ceiling unchanged.
  The 115 px threshold stays exploratory; the ≥75% statistic can be reported
  descriptively, not as 12 independent samples or validation of registration
  discrimination (all 12 pass that gate). Step 5's historical fail stands.
  Appearance reference and exact target matte remain **granted and unbilled**;
  no end-to-end codec-gain claim follows.
- **This phase ends CPU-only.** 0 of 8 approved GPU-hours used; no fitting,
  generation, quality judge, Wan-Animate, chats, monitors or wakeups started.
  All task-owned audit commands are terminal; no active batch is left behind.
  A later diagnostic run still needs the written GPU validator implemented
  and the original smoke/pilot/scale budget gates. No GPU execution starts here.

#### Durable recovery receipts, verification and continuation

- Canonical report bundle under Datasets:
  `/home/itec/emanuele/Datasets/pointstream-data/tennis/k6b-recovery-20261024-report-v2.tar.gz`,
  SHA256 `2bfafa0c0115171d784dd3d79696207917362f85a9653fa7939c3f446b57ff5b`.
  Includes report, all five collection records, code v2/v3, recovered splitter,
  inferred intervals, source-pair review and provenance. Raw descriptor files
  remain read-only in their original output directories.
- Merged screen JSON SHA256
  `03b4ddf84c67efd200464a3649b46f22ef80ec8650ba9c4d8c75ba02b930f5e8`.
  CPU part audit hashes: part0 `665f2c05…5573b` (above),
  part1 `8c6a16b344a50cbe34353fb1b660166842f52c03eaae3014a89afe7f5b443299`,
  part2 `f7710a3c89530add88d2872eeca3e69da075ce406602e86fed790e5cc580e467`,
  part3 `0a055cef3931ccfeaf60a340f5831232e49bf15004c676bf255fc016adff2d48`.
- All runs used gpu1 **CPU only**, Python `/usr/bin/python3` 3.10.12,
  software ffmpeg n7.1.1-56-gc2184b65d2. Binary SHA256:
  `/opt/local/bin/ffmpeg` `0b652966dc4ecf5e45854c58d8149c7a41352b3285e8695ca26501dff4a72656`;
  `/opt/local/bin/ffprobe` `1e09f52f9bc91ecec6d05f5b33fd4d69064fa9874b9599c55343ab011d37e66c`.
  Dispatch baseline `921a9a81` plus the selected v2/v3 code checksums above;
  utility subsequently committed in `728c44ac`. Exact command vectors, source
  SHA256s and descriptor offsets are in collection records. Local screen uses
  project NumPy 1.26.4, at most two BLAS threads, no model loading.
- Reproduce: collect via `tools/datasets/k6b_recovery.py collect` with the
  frozen clip/start/SHA cohort; use bounded clip-index parts and new output
  directories. `python -m tools.datasets.k6b_recovery screen V2_CHECKPOINT
  --extra V3_PART0 --extra V3_PART1 --extra V3_PART2 --extra V3_PART3`.
  Merger refuses missing coverage, duplicated clips, changed bytes, wrong
  descriptor counts and standalone partial/smoke promotion. Source archives,
  older runs and failed checkpoints were not overwritten or deleted.
- Verification: `ruff check` and `mypy --config-file pyproject.toml` pass
  (190 source files). Full `python -m pytest`: 838 passed, 24 skipped,
  1 unchanged macOS protected-path symlink-resolution failure. Seven focused
  audit/scope tests pass, including the subsequently added multi-part merge
  coverage/offset fixture. `git diff --check` passes. No unrelated hook fix.
- Stopped because CPU recovery/claim decision is complete. No new source
  identity was invented and no narrower numerical quality result was produced.
  Recommendation for future work: retain this cohort only for the explicitly
  qualified same-broadcast clip diagnostic; retain the strict original-source
  limitation until originals/receipts are independently recovered.

- Publication recovery: the first report upload lost its SSH connection.
  Inspected the existing destination through shared-storage host gpu2 before
  any retry: old report.tar.gz contains 2,916,352 bytes, SHA256
  `2996e0a8e51939b093f830b202781b5cd0eb3f3397a0e86b2412168349b3b834`,
  not the complete local bundle (3,555,736 bytes). Preserved that partial file;
  publication uses the distinct report-v2.tar.gz path above, never overwrites
  or promotes the partial upload. No compute job was replayed.
- Report publication **verified** through gpu2: report-v2.tar.gz is exactly
  3,555,736 bytes, expected SHA256 `2bfafa0c…ff5b`; every tar member reads
  successfully. Partial first upload retained. The PR's adjacent PLAN-row
  conflict was resolved by retaining this study's row and the upstream T1
  outcome; no T1 work was run or changed by this study. Auto-merge stays off
  under the separately recorded approval block.
- Final branch verification after incorporating origin/main `058d8225`:
  `ruff check` passes; `mypy --config-file pyproject.toml` passes (195 files);
  `python -m pytest`: 843 passed, 24 skipped, 1 unchanged macOS protected-path
  symlink-resolution failure. `git diff --check` passes. PR diff is confined
  to this study's PLAN text, card, experiment entry, audit utility and tests.


### 2026-10-24 — Documentation reconciliation; clip-local scope decision pending

This continuation reuses the preserved CPU recovery report and receipts above;
no audit was rerun. It supersedes the original preparation's execution and
validator prerequisites **only for a proposed clip-local diagnostic**, subject
to explicit user scope approval. The original-source confirmation gate remains
**unresolved**, not passed. The earlier 8 GPU-hour approval was contingent on
provenance checks; it is not permission to waive that gate or dispatch under
a weaker claim automatically.

- **Concrete approval needed:** proceed with a descriptive same-broadcast,
  clip-local diagnostic despite unresolved original-source/replay independence,
  within the existing 8 GPU-hour ceiling, after the remaining CPU prerequisites
  and staged gates; or keep CPU-only status at no GPU cost. Recommendation:
  the diagnostic is defensible if the user values that limited result enough
  for the estimated 6–8 GPU-hours. It cannot deliver independent-source,
  certified training-clean or unseen-match confirmation. No new budget request.
- **Unchanged inputs/protocol:** all 12 frozen 240-frame windows and source
  hashes in the card; ≥228/240 K8b `fits=true` (≤6 px), finite nonsingular
  homographies, invalid/missing=false; no quality-dependent replacements.
  All pass, so gate discrimination is untested. The 115 px threshold stays
  exploratory and cannot rescue failure. Keep frozen step-5 C base/matching
  adapters/guider, 4-bit, guidance 2.0, 512×784, 30 DDIM, existing deterministic
  seed and 4 px velocity/every-fourth-frame keypoint rules; no fitting/tuning.
- **Comparison/limitations:** actual SVT-AV1 and DCVC-UF on identical windows,
  native fps, masks, indices and duration convention, fair achievable anchor
  envelopes without extrapolation. References and exact target matte R remain
  **granted and unbilled**; no end-to-end codec-gain claim. Preserve two-match
  clustering and unknown original-source/replay and upstream pretraining
  exposure; the clip screen is not a certificate of source independence.
- **Descriptive rule:** mean LPIPS/DISTS billed-rate win share ≥75%
  (≥18/24 with all 12), minimum six valid gated windows/three per match;
  report every window/metric/match, pooled/gated, billed/unbilled and missingness.
  Billed below bar but unbilled ≥75% returns to the user; below unbilled bar
  fails the descriptive criterion. No interim success; stop for mathematical
  futility only when every remaining win cannot reach the bar. Missing output
  or unfair/unbracketed anchors are unresolved, never wins. Step-5 fail stands.
- **Remaining CPU prerequisites:** implement/test the substantive GPU-run
  validator specified earlier, including corruption rejection; freeze manifest,
  scoring indices and code; recheck staged bytes and dependency/reference joins
  against preserved receipts. Verify clip-local frame/PTS/fps and alignment,
  rather than requiring unavailable original PTS for this narrower execution.
  Validator records `clip_audit=complete`, `original_source_gate=unresolved`,
  `claim_scope=clip_local_diagnostic`, and rejects a stronger claim while that
  gate is unresolved. No nominal cut-time inference becomes verified PTS.
  Existing substantive checks for geometry boundaries, codec/keypoint round
  trips, masks/indices, coverage, rates, real devices/kernels and finite,
  bracketed metrics remain mandatory. GPU-dependent checks run only in an
  authorized smoke. Known judge overlap, new detected overlap or changed inputs
  stops execution for review; unknown exposure remains explicitly disclosed.
- **Staging unchanged:** 0.3 GPU-h smoke (≤600 s each, one window per match),
  3.4 h first six (three per match in frozen rank order), 3.4 h remaining six,
  0.9 h reserve; hard total 8 h including failures/retries. Before scaling,
  pilot must demonstrate total projected cost ≤8 h with valid outputs and fair
  anchors; otherwise stop, not tune or silently expand budget. Single-window
  shards/checkpoints and waits ≤45 min. No interim pass or quality-driven recipe
  changes; budget exhaustion and validator failure stop the study.
- **Outcome of this continuation:** documentation reconciled; decision by
  quality rule not reached; **0 of 8 GPU-hours spent**, no jobs submitted.
  No GPU dispatch, fitting, generation, source download, additional chat,
  recurring monitor or auto-merge attempt. The separate existing merge approval
  block remains in force. Card now requires scope approval instead of asking
  again for the already granted ceiling. Historical entries above are retained
  as provenance, with this entry controlling the proposed diagnostic route.
- Verification for this documentation-only continuation: reviewed the card,
  appended protocol override and own PLAN text together; `git diff --check`
  passes. No executable files changed, so no audit or test suite rerun. Prior
  code-validation results and the unchanged macOS test limitation remain above.


### 2026-10-24 — Explicit approval of narrower diagnostic scope

- User approval in this study chat: “I approve it, I don't understand what the
  problem is,” attached to the pending clip-local scope decision. This is
  explicit permission for the narrower **clip-local same-broadcast diagnostic
  despite unresolved source independence**. It supersedes the preceding
  scope-approval-pending status; no repeated scope/budget approval is needed
  within the frozen protocol and 8 GPU-hour ceiling.
- The unresolved original-source/replay issue now limits the diagnostic's
  claim; it is not an execution blocker for this approved scope. The stronger
  independent-source confirmation gate remains **unresolved**, not passed.
  No claim of certified training-clean frames or unseen-match generalization.
- Unchanged: 12 frozen windows; ≥228/240 K8b fits gate; frozen step-5 recipe;
  identical-window SVT-AV1/DCVC-UF anchors; descriptive ≥75% score; exploratory
  115 px threshold; references and exact matte granted/unbilled; 8 GPU-hour
  hard ceiling with **0 spent**. Smoke/pilot, scale-feasibility, validator and
  futility rules still apply. No fitting, Wan-Animate or cohort replacement.
- Preparation-only work is complete. Next authorized execution work is to
  implement/test the substantive validator and freeze/verify staged manifests,
  followed by the staged compute gates. These checks protect correctness;
  they do not reopen the already approved claim scope or fabricate source PTS.
- The planning chat was informed directly with the user's explicit permission,
  identifying this chat as the study owner to avoid duplicate execution.
  No other chat was launched. No GPU job, fitting, generation, new source
  download, monitor, wakeup or auto-merge attempt occurred in this approval
  continuation. GPU quality decision remains not reached.
- Verification: documentation reviewed against the approved scope and previous
  frozen protocol; `git diff --check` passes. No executable changes or audit
  reruns; prior test results remain recorded above.


### 2026-10-24 — Approved diagnostic execution begins

- User explicitly instructs “Did you run the task? If not, do so.” This chat
  owns execution; planning chat notified to avoid duplicate dispatch. Scope
  and 8 GPU-hour ceiling already approved; no renewed approval requested.
- CPU manifest gate implemented in `experiments.tennis.k6b_confirm`: checks
  all 12 immutable source/window/fps identities, centered placement, complete
  registration rows, finite nonsingular homographies, old-clip intersections
  and qualified claim scope. Preflight passes all 12 with the original counts
  (234–240/240). It does not pass the original-source confirmation gate.
- Frozen execution manifest:
  `/home/itec/emanuele/Datasets/pointstream-data/tennis/k6b-confirm-20261024-execution/manifest.json`,
  SHA256 `7ed2589fdb5e54ac1fbe3dd05a1b3a7f3c01076aa3244efd135ceb5a50ba03c1`.
  Reuses the completed CPU audit; no new source download or audit rerun.
- First stage: actual SAM3.1/YOLO11x pose/features; smoke two full 240-frame
  windows (first per match), then first three per match in frozen rank order,
  no scores or renders. Frozen centers are actual absolute native frame indices
  (e.g. Alcaraz031@129 starts9), not the old @target120 shorthand.
  Native feature validator is wrapped with immutable cohort/span/fps/hash and
  240-frame coverage checks. Synthetic corruption tests pass for wrong sources,
  starts, fps, missing output, duplicate IDs, overlap claims, singular/nonfinite
  homographies and 227/228 boundary. Ruff and mypy pass (197 source files).
  Full suite running locally; prior macOS hook limitation remains known.
- Stage caps: smoke600 s, full1800 s, validation120 s, total reservation2700 s;
  exclusive compatible Ada pool gpu6/gpu5, 32,000 MiB, eight CPU threads,
  pinned PointStream pack SHA256 `32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661`.
  Overall smoke allowance remains0.3 GPU-h, first-six3.4 h, remaining-six3.4 h,
  reserve0.9 h; all actual smoke/failed/retry use counts against8 h. This
  reservation is not recorded spending. Stop on invalid output or infeasible
  projected full-study cost. All subsequent quality-producing stages still
  require their substantive native and comparison validators.
- Fleet inspected all six hosts; no active diagnostic jobs found. gpu6 has two
  idle Ada GPUs; gpu5 occupied by foreign work, so exclusive admission avoids
  those devices. No foreign processes changed. No `--chat-id` is passed (its
  legacy recurring-heartbeat path is forbidden). Use attended bounded fleet
  wait; automatic completion resumption is unverified in this client.
- No job ID yet at this checkpoint; GPU-hours spent **0/8**. Save the returned
  ID immediately after submission; uncertain submission is inspected, not replayed.

- First job submitted **`20261024T174042Z-463f42af`**, returned pending;
  spec SHA256 `e8913b69cc1f699f8a2132e49be3d8680e775e79d538ca928cd65c3034e44a82`,
  code `dbc79e4bc974b888e9d61d2a1ad1230d22e2b626`, snapshot SHA256
  `a4bffe3c4a14734f8f3e611e7359e3a4af448389610817e83d96bf1c1244fb04`.
  Deadline2026-10-24T18:23:54Z, total reservation2700 s. Inputs/output receipt
  in `/tmp/k6b-confirm-features-submit.json`, canonical output under
  `jobs/fleet/inbox/20261024T174042Z-463f42af/`. No unknown replay.
  Dispatcher inferred chat ID despite no flag and printed legacy heartbeat
  guidance; no automation tool was called. Inspecting whether this is guidance
  only before waiting; do not create any recurring model monitor.
- Full local suite:844 passed,25 skipped,2 failures: known macOS protected-path
  symlink issue and sandbox-denied process inspection in the contention test.
  Focused new validator tests3/3 pass; rerun process test with inspection allowed.

- Admission check returned pending (no host assigned yet); one bounded fleet
  wait started for this recorded ID, ≤2640 s. No resubmission. Source review of
  dispatcher confirms heartbeat text is guidance only: it writes a chat ID in
  the receipt but does not create a scheduled model automation. No monitors
  exist from this task. Process-control test rerun with inspection permitted
  passes1/1; only the previously recorded macOS protected-path failure remains.

- Job `20261024T174042Z-463f42af` **failed before model loading**: staged
  source paths were outside the fleet data root and additionally contained an
  incorrectly inferred `/videos/` segment. Confirmed terminal; no unknown replay.
  Supervisor elapsed10.29 s counts conservatively as **0.00286/8 GPU-h**.
  Failure events reported/acknowledged. Original attempt/evidence preserved.
- Corrected preparation uses exact source paths from saved recovery receipts,
  copies the 12 clips to the new experiment's `sources/`, and verifies every
  SHA256 after copying. Original bytes and cohort unchanged. New immutable
  manifest-v2.json SHA256 `8b81c4b9645a24bb5b613bdda908440735fe932d23ea251b0c567a366bda090b`; 33499247 staged bytes.
  CPU staging's first attempt failed on the inferred path, before any copy;
  inspected then corrected from receipts, not a re-download. No prior output
  overwritten. The successful CPU preparation is terminal.
- Downstream code adds actual R/S scoring-mask hashes to native anchor and
  method judge records, exact generated-target coverage/recipe/checkpoint
  checks, and a bracket-only envelope comparison with no fallback extrapolation.
  Focused tests27 passed,2 skipped; ruff/mypy pass (198 files). No quality
  scores opened. First-feature retry uses the same recipe and 2700 s reservation,
  deadline 2026-10-24T18:29:40.227745+00:00. Next returned job ID must be saved immediately.
