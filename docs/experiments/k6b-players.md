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
