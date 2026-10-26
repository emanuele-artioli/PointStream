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

- Corrected feature job **`20261024T174553Z-ecc5ca36`**, returned pending. Spec SHA256
  `1806ad8b15db6c45784b70b8146239677f535493a324bfcff8bfe32db48af49b`, code `9473f65c`, deadline2026-10-24T18:29:40Z,
  reservation2700 s. Receipt `/tmp/k6b-confirm-features-submit-v2.json`;
  canonical outputs `jobs/fleet/inbox/20261024T174553Z-ecc5ca36/`. Explicit empty chat ID
  disables legacy heartbeat guidance. GPU spending before this job0.00286/8 h.

- Corrected job admission verified running on gpu6; one bounded fleet wait
  handle5839, result file `/tmp/k6b-confirm-features-wait-v2.json`, expires
  within44 min. Original wait79316 already returned terminal failure; no
  recurring model calls or wakeups. Continue independent CPU code preparation.
- Geometry-only input-pack builder now freezes player IDs, scored/every-third
  judged indices and actual R mask hashes from completed feature artifacts
  before anchor/generator quality is produced. Its CPU validator rebuilds the
  pack from the immutable feature archive and compares every artifact byte;
  missing, extra or repeated content cannot pass. Native keypoint round trips,
  generator coverage/recipe/checkpoint/guider billing and R/S mask agreement
  checks remain separate substantive gates. No original-source PTS is inferred.

- Job `20261024T174553Z-ecc5ca36` **failed its smoke validator**, with no full
  promotion. Actual mask/pose extraction completed both windows in253.33 s;
  supervisor total262.14 s. The installed manager expands staged placeholders
  in workload arguments but not validator arguments; retained validator log
  confirms literal `{staged:manifest}` FileNotFoundError. No source/model or
  quality failure. Both failed attempts total272.43 s = **0.07568/8 GPU-h**.
- Read the retained0.91 MB smoke masks/poses and independently ran the repaired
  validator on CPU before another GPU attempt: **all19 checks pass**,480 frames,
  including actual feature/mask recomputation, source placement/hash, native
  SAM/CUDA policy and peak limits. Local receipt
  `/tmp/k6b-confirm-smoke-v2-validation.json`; raw smoke artifacts remain
  read-only in that job. No generator/anchor quality opened.
- Corrected validator arguments use the immutable canonical manifest path,
  with recorded manifest SHA rechecked, rather than an unsupported placeholder.
  Future native stages run through a thin wrapper that validates full output
  before return and independently revalidates saved artifacts for the fleet
  gate. Full-stage checks therefore do not rely on the manager's smoke-only
  validation. CPU input-pack reconstruction and reference-membership checks
  implemented; focused checks and ruff/mypy pass (199 source files).
- Third feature attempt keeps all recipe/scale/input identities; only tested
  validation plumbing changes. Reservation2700 s; deadline2026-10-24T18:40:16.710230+00:00.
  Based on measured two-window253 s smoke, six-window feature stage fits1800 s;
  total study still projects6–8 GPU-h including paid failures, with exact
  remaining projection required before renderer scale. No fitting or monitors.

- Execution interrupted before third submission: automatic approval review hit
  an account usage limit and did not execute the commit/push. Both jobs were
  already terminal; no GPU task or recurring monitor was left running. User
  explicitly instructs “Continue.” On resume, recheck saved job state/hosts;
  old unsubmitted third-attempt deadline is expired and must be refreshed.
- Code review also keeps the generator wrapper independent of main-environment
  mask dependencies: the JCV lock lacks zstandard, so generator-scope checks
  use NumPy/stdlib and the existing standalone generator's JSON/path helpers.
  No new environment or provider, no model recipe change. Native mask/pose
  validation remains in the pinned main environment. Original-source gate
  stays unresolved; exact matte/reference remain granted.

- Resumed commit/push now succeeds after the usage-limit reset. Third feature
  job **`20261024T193537Z-12dffb55`**, code `769833b5`, returnedpending; spec SHA256
  `4a12b3442a59916c5b5677e4f5797c062f1e61f3a31a3a1a5ba7e3752f67d1f0`. Renewed deadline2026-10-24T20:19:25Z,
  reservation2700 s; GPU spend before this job0.07568/8 h. Receipt
  `/tmp/k6b-confirm-features-submit-v3.json`. No implicit replay of old jobs.

- CPU geometry-pack smoke/reconstruction passes on the two retained source
  windows:10 checks, every file byte and geometry-only scoring index/R mask
  recomputed. Pack SHA256
  `348a32de99c9e6c43f6f3f67340a478e8b72ea101f93ad9a6fb554041a0f1d42`;
  this is CPU plumbing evidence, not a new quality result or a full-cohort pack.
- Reporter fixes the denominator at24 cells, forbids interim success and
  extrapolation, separates billed/unbilled and base-charged comparisons,
  records per-window/match metrics/missingness and both duration conventions.
  Keypoint entropy rates pool **all12 source-code symbol sets** even for a
  pilot/futility report; compute those before deciding on partial generations.
  No source-code frequency choice depends on generated quality. No BD-rate is
  fabricated from the single frozen renderer point.
- Frozen checkpoint archives freshly hashed without loading tensors: base
  `075b001a…3285`, Alcaraz `f1a488f2a3ddecf57cff451801f64bc07ae652c2b8b998d4ac844ba8685f5fa2`,
  Medvedev `5fc200fc50f5af68e561fa10baba54520043e5770555712caf6e89092b175427`.
  The actual shipped tennis-base checkpoint is48,878,567 bytes; primary treats
  it as common decoder data, alternative reports that payload charged over2 h.
  Trained adapter/guider still billed once per match over2 h; reference and
  exact target matte remain granted/unbilled in every comparison.

- Full local suite with process inspection allowed:850 passed,24 skipped,1
  unchanged macOS protected-path symlink test failure. Ruff and mypy pass
  (200 files). Later small validation/report refinements pass focused tests.
- CPU receipt archive published separately as
  `tennis/k6b-confirm-20261024-execution/cpu-preflight-20261024T1952Z.tar.gz`,
  SHA256 `60d90908eb26061969e16a54863d3d0e3b6a47dbbd2d1e046d7517410b2035be`,
  1,523 bytes. It identifies the native validator and source feature job; it
  is infrastructure/CPU proof, not a paper quality result. Prior raw jobs and
  archives retained. Native registration-frame alignment now checked directly;
  all12 pass the strengthened CPU gate without changing the manifest.

- Feature job `20261024T193537Z-12dffb55` **complete**, gpu6 Ada; substantive
  smoke and full validators pass on all six windows/1440 frames. Full component
  time698.57 s; complete claimed-job wall1046.78 s =0.29077 GPU-h.
  Total spent **0.36645/8 h** including both failed attempts. Archive
  `jobs/fleet/inbox/20261024T193537Z-12dffb55/full/published.tar`, SHA256
  `e5547615a2680b1ee9d2345ea8bc8d554f1bf4cbbed4327d8042bf923e29c7c9`,2,529,280 bytes. Raw metadata/receipt preserved; no quality opened.
- Resource-only audit of preserved exact K6 pipeline (no quality read) finds
  Ada window times361–443 s for the comparable Alcaraz/Djokovic cases, versus
  662–751 s on A6000. Therefore anchor/source-code jobs will be **three-window
  shards**, not a six-window1800 s guess; full1800 s, smoke≤600 s. This changes
  only scheduling, not any source, mask, score, codec setting or renderer recipe.
  All12 source-code symbol sets are prepared before interpreting pilot quality
  so pooled-rate denominators are fixed; remaining-six generations still await
  pilot cost/futility checks. Small source preparation for those windows is
  necessary to freeze the codec input set and remains charged within8 h.
- Next dispatches: CPU geometry pack for first six (0 GPU-h), and fixed remaining
  six feature inputs (reservation2700 s) using the same validated configuration.
  Their actual returned IDs must be saved before waiting. Total study forecast
  remains6–8 GPU-h; new staged cost proof required before generation scale.

- Submitted inputs-first6: **`20261024T200247Z-b63794f9`**, spec SHA256 `c02707988e907513b279adcc181d98923497e0b231307f8bca58c61cef07e59f`, code `855a0caf`; returnedpending. Receipt `/tmp/k6b-confirm-inputs-first6-submit.json`. CPU-only,0 GPU-h.

- Submitted features-remaining: **`20261024T200318Z-12faa9a8`**, spec SHA256 `c26265ea5ffe7a6a47e37412c88945968ab77d040ad02393c151d03132230d22`, code `855a0caf`; returnedpending. Receipt `/tmp/k6b-confirm-features-remaining-submit.json`. GPU reservation2700 s; pre-job spending0.36645/8 h.

- CPU input job `20261024T200247Z-b63794f9` **complete**,0 GPU-h. First-six
  geometry pack `/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261024T200247Z-b63794f9/full/inputs.tar`, SHA256 `f3b946816950c0b605d865cb97e5dacc716588969552a252f56ac880653426f9`;
  smoke reconstruction gate passes. Remaining-feature job
  `20261024T200318Z-12faa9a8` was running on gpu6 at this batch event; keep
  its ID and inspect before any replay. Current GPU spend0.36645/8 h plus that
  running reservation. No remaining-six renderer release yet.
- Anchor execution now explicitly schedules the frozen rough three settings
  per codec before the remaining four potentially hull-improving K6 points.
  Native defaults remain unchanged for other tasks; opt-in
  `--rough-anchors-first` changes scheduling only, keeps all ten points and
  the exact encoders/decoders/settings/masks. Tests cover no duplicate/missing
  point and the exact initial lists; focused tests, ruff and mypy pass.
  This is committed before any new codec/generator quality is produced.
- Next: two first-six anchor/source-code shards of three windows each,1800 s
  full/≤600 s smoke and2700 s job reservation, exclusive compatible Ada pool.
  Native full output is substantively validated before return; same-job DCVC
  encode/decode/judge retained. Save both IDs before the next batch wait.

- First anchor shard submitted **`20261024T201059Z-29586984`**, code `3fb9ef18`, spec
  SHA256 `f4b16cfead40fc0f2133bdc685e33b6567461ec75a8cbe21792f8340d44f4907`; returnedpending;
  deadline2026-10-24T20:54:56Z,2700 s reservation. Receipt `/tmp/k6b-confirm-anchors-pilot1-submit.json`.
  Scope: Alcaraz031@129, Medvedev014@194, Alcaraz013@213, all240 native
  frames; smoke first48 frames of the first window per match. Real ten-point
  source encodes and source-code/oracle path, no renderer fitting/generation.
  Only one anchor job submitted while the remaining-feature job occupies the
  other Ada GPU; issue the second shard when capacity frees so its full-stage
  deadline can be met. At submission known spent0.36645/8 h, other active
  reservation2700 s; combined worst-case reservations remain below8 h.

- Remaining feature job `20261024T200318Z-12faa9a8` **complete**, full native
  scientific validation passes. All12 frozen windows/2880 unique target frames
  now have masks/poses; no cohort replacement. Remaining archive SHA256
  `ba4534f5e0ed0df5c589cd31c4bb7153225589b970357b8272823e1e59cdb523` at
  `jobs/fleet/inbox/20261024T200318Z-12faa9a8/full/published.tar`.
  Claimed-job wall1056.15 s; total GPU spent **0.65982/8 h**, excluding
  currently running first anchor shard. Native full feature pipeline complete.
- First anchor shard `20261024T201059Z-29586984` was running full after its
  smoke gate passed; no human quality interpretation yet. Ready next source
  jobs: all12 CPU comparison pack (0 GPU-h), then the second first-six anchor
  shard on the now-free Ada GPU. Both planned caps remain2700 s or less, so
  current spending plus both anchor reservations stays below8 h.
- Strengthened preparation validator reconstructs every coded pose bitmap,
  crop box and run ID, checks raw source-decode identity, and first fitted
  source reference placement. Tested on retained real Alcaraz036 preparation
  and source-code artifacts (old jobs `20261022T123413Z-eba7629b`,
  `20261021T171936Z-50ce2f74`): exact reconstruction passes in2.43 CPU seconds.
  No new source-video download or quality result; this is correctness evidence.

- Submitted inputs-all12: **`20261024T203302Z-332cab62`**, spec SHA256 `7fa35cdae4a5e3953343f11c95a62c89405de0d445742a7655b67c24835ab149`, code `e35d3b3c`, returnedpending. Receipt `/tmp/k6b-confirm-inputs-all12-submit.json`. CPU-only,0 GPU-h.

- Submitted anchors-pilot2: **`20261024T203305Z-855eccf9`**, spec SHA256 `72194f4550a3dad06a66d990cc16fcd96a5484b18b270c126113c27df02e1bbe`, code `e35d3b3c`, returnedpending. Receipt `/tmp/k6b-confirm-anchors-pilot2-submit.json`. 2700 s reservation; all-source GPU spending before running anchor jobs0.65982/8 h.

- First anchor/source-code job `20261024T201059Z-29586984` **complete**,
  full scientific validator passes. Real same-window codecs/decodes, source
  q4-velocity-e4 codes and oracle masks on all three240-frame windows;
  per-window times363.62/329.54/343.78 s, total component1047.21 s. Claimed
  GPU job1347.10 s =0.37420 h. Known total **1.03402/8 GPU-h**,
  plus active second anchor `20261024T203305Z-855eccf9`.
  Archive SHA256 `67444a2a8f56c8700fb073a1184ce207ec3ad38cc003dbafa6710bb33836dff9` under that job's full/
  published.tar; no quality interpretation before all12 pooled source rates.
- All12 CPU geometry-pack job `20261024T203302Z-332cab62` **complete**,
  0 GPU-h; inputs.tar SHA256 `8bf1cc899b4597d1bfa7e7d86d3ce60d7516ebcfd948eb9e22eefccf1be780e4`. Frozen per-window masks,
  sources, fps and judging indices ready for every comparison. Source pack
  and first-six pack differ only by included windows, not per-window data.
- Source timing now predicts four three-window anchor shards about1.50 GPU-h
  total; features including failures0.65982 h. With old measured same-recipe
  renderer1252 s/window plus per-job smokes/validation, all12 renders about
  4.5 h and judging about0.2–0.4 h: still within8 h. New renderer cost measured
  before scaling; its recipe remains fixed. Initial correctness smoke0.3 h is
  the one-time campaign; repeated fleet gate work is charged to its associated
  pilot/scale window allocation, with every second counted once in total8 h.
- Next: third source-code/anchor shard on the free Ada GPU and CPU preparation
  of the first three windows using their completed native source-code artifact.
  Target pose maps/crop boxes/run IDs/reference placement are recomputed by the
  strengthened validator before any generation. Save returned IDs before wait.

- Anchor/source shard submission assigned **`20261024T204512Z-8a7d6449`**
  but returned **submission outcome unknown**: snapshot extracted, temporary
  archive cleanup did not finish. Do not replay or assume a GPU job exists.
  Inspect this exact ID and presence of signed spec/ready state first. No new
  code/model/data choice is involved; source snapshot transfer used90 s cap.
- CPU prepare first-three submitted **`20261024T204514Z-f427e93b`**, spec SHA256
  `39a4d82d16606aad47c04b7118a7f2bfaad9fdf7fa0013660f1497e386dd76ce`, code `e35d3b3c`, returnedpending;0 GPU-h.
  Receipt `/tmp/k6b-confirm-prepare-first3-submit.json`. Retain its ID independently of the uncertain anchor request.

- Inspected uncertain request `20261024T204512Z-8a7d6449`: broker reports
  unpublished/unknown; shared directory has source/,source.tar,events only,
  **no spec.json, no ready.json, no run/**. Submission process exited2.
  Therefore it was never runnable and spent0 GPU-h. Preserve that snapshot;
  no manual cleanup or publication. A new replacement request is justified,
  not a replay of an unknown running job. Transfer will explicitly select the
  complete workload trees (experiments/src/env/scripts/tools plus pyproject),
  excluding unrelated docs/tests; same code revision and all runtime inputs.
- CPU preparation `20261024T204514Z-f427e93b` **complete**,0 GPU-h; second
  anchor `20261024T203305Z-855eccf9` running full with two completed windows
  at this inspection. Next source request uses a fresh44-min deadline.

- Confirmed replacement third anchor submitted **`20261024T205303Z-4928c016`**, spec SHA256 `5ed8b884b56b57e5a871d68542fe059958aea0ce4d6a024cda57c7b2fe8660b0`, code `e35d3b3c`, selected runtime trees recorded in snapshot; returnedpending. Receipt `/tmp/k6b-confirm-anchors-remaining1-v2-submit.json`. The old unpublished `20261024T204512Z-8a7d6449` remains untouched/non-runnable.

- First-three CPU preparation `20261024T204514Z-f427e93b` complete with full
  scientific validation. Archive SHA256 `674c40981a7d805a2d241b12bf499ca1a6e99351234581edebe30667cb6bfa0c`.
  First two windows have two240-frame/one-run player units each; Alcaraz013
  has two239-frame/two-run units (one excluded registration frame). References
  all first fitted scored frame0. Actual decoded keypoint maps, crop boxes,
  run IDs, source YUV identity and reference membership/placement verified.
- First renderer pilot ready: Alcaraz031@129, both actual240-frame players,
  exact existing C base and matching adapter/guider,4-bit, guidance2.0,
  512×784,30DDIM, source-deterministic seeds. Smoke8 frames per real unit,
  fullall coded frames,1700 s full/180 s smoke/2100 s total reservation.
  No tuning, fitting or alternative profile. Current source-data scope and
  references/exact matte remain granted; original-source gate unresolved.

- First renderer submitted **`20261024T205730Z-869e62f2`**, spec SHA256 `0532d7e391b3580740f0f1c9799287a4f10223d6941c6f9898462c563230624a`, code `e35d3b3c`, selected runtime trees; returnedpending. Receipt `/tmp/k6b-confirm-generate-pilot1-submit.json`. Alcaraz031@129 both240-frame player units,2100 s reservation, deadline2026-10-24T21:41:26Z. No new fitting.

- Second anchor shard `20261024T203305Z-855eccf9` **complete**, full
  scientific validation passes; archive SHA256 `d77d38a9fec4b09d2cc733a4181fef83f7c7619f5a68aa98e32e6cf4aa824878`.
  This finishes all six pilot windows' real anchors and source codes. Claimed
  GPU wall1315.90 s; known total **1.39955/8 h**, plus active
  source shard `20261024T205303Z-4928c016` and renderer
  `20261024T205730Z-869e62f2`. No quality decision until all12 pooled rates.
- First renderer's initial status/wait confirms it passed the substantive
  real-model smoke and entered full sampling on gpu6. Expected body units,
  frozen recipe/checkpoints/guider billing, device/kernels and actual generated
  target coverage validated before promotion. No fitting or alternative model.
- Next CPU-only preparation covers the second three pilot windows from their
  completed source-code artifact. Remaining-six rendering stays unreleased.

- Second-three CPU preparation submitted **`20261024T210930Z-91f893e4`**, spec SHA256 `a2a312639c80150b6f0264316da9c97a3a2017b305d5cd4859f184a5a10f583c`, code `e35d3b3c`, selected runtime trees, returnedpending;0 GPU-h. Receipt `/tmp/k6b-confirm-prepare-second3-submit.json`.

- Second-three CPU preparation `20261024T210930Z-91f893e4` complete (0 GPU-h).
  At this batch event, replacement source shard `20261024T205303Z-4928c016`
  and renderer pilot `20261024T205730Z-869e62f2` were both running full on
  gpu6; renderer completed one real player unit. No resubmission or recipe
  change. Completed anchor GPU subtotal remains1.39955/8 h before active jobs.
  Native bounded batch wait result `/tmp/k6b-confirm-pilot-next-wait.json`.

- Third source/anchor shard `20261024T205303Z-4928c016` complete with full scientific validation; archive SHA256 `543d4aa07500a89cfa0a65f9381ebff1d69114f527cce102043645cd6aaae162`. Charged GPU wall 0.351748 h; completed total **1.75129/8 h**, renderer pilot still active. No quality interpretation.

- Final source/anchor shard submitted **`20261024T211935Z-d45d6fbb`**, same fixed ten anchors, Med029@138/Al047@123/Med021@246, 2700 s reservation/fresh44-minute deadline, code e35d3b3c. Receipt `/tmp/k6b-confirm-anchor4-submit.json`. Save ID before bounded wait.

- CPU preparation of Al016/Med017/Al008 submitted **`20261024T212000Z-7d4ddcf5`**,0 GPU-h. Rendering these remains gated on first-six futility and cost. Receipt `/tmp/k6b-confirm-prepare-third3-submit.json`.

- First full renderer `20261024T205730Z-869e62f2` complete; substantive full validation passes every check, exact frozen recipe/coverage/device and adapter+guider billing. Archive SHA256 `8a927ae4b80feaa7cb95f9fb61d5f3bafb1d2ce744a310fe07a1d58745a5ec06`. Claimed wall0.371330 h, completed total **2.12262/8 h** plus final anchor active. At0.3713 h/render, remaining11 renders4.085 h + final anchors about0.37 h + judging0.2–0.4 h projects6.8–7.0 h total, within8. Continue first-six pilot; remaining-six rendering still gated on quality futility.

- Second renderer submitted **`20261024T212140Z-f1459bd9`**, Med014@194 both240-frame player units; same30DDIM/4bit/C recipe with matching Med adapter. Spec SHA256 `592b68c6169884e87d1be4d004676d87d4c82c33ab1c00c5a30ce4f368df6a66`, code6ccc5ec2,2100 s reservation/fresh44-minute deadline. Receipt `/tmp/k6b-confirm-gen2-submit.json`.

- CPU recovery checkpoint: `tennis/k6b-confirm-20261024-execution/execution-state-20261024T2129Z.tar.gz`, SHA256 `f138ad9b5f2136e11313f3fde7ec1dda8ebdc88b1f8ddef2a277a0a372d20505`,1,916,710 bytes,13 members. Contains verified receipts, current ledger and exact spec factory; remote copy hash verified. New path only, no evidence modified. GPU stages continue under one bounded native wait.

- Final anchor/source shard `20261024T211935Z-d45d6fbb` complete; all12 source symbol sets and fixed ten-point anchor curves now validated. Archive SHA256 `8f03bf7aef11443ef03ebb9ddfa369ce1148c5120b07404bf5a7bcdba9990e7f`. Charged wall0.380167 h; completed total **2.50279/8 h**, second renderer active. Pooled rates can now be frozen over all12 before interpreting pilot quality.

- Third renderer submitted **`20261024T214351Z-3f29c4ca`**, Al013@213 both239-frame/two-run player units, fixed C recipe; spec SHA256 `f6f07af6b090f001e3275339a0c9baf7b87db7b33a8e8f1802061c47736d04ef`, code6ccc5ec2,2100 s/fresh44-min deadline. Receipt `/tmp/k6b-confirm-gen3-submit.json`.

- Final CPU preparation submitted **`20261024T214359Z-53bb8f6f`**, Med029/Al047/Med021,0 GPU-h; receipt `/tmp/k6b-confirm-prepare-fourth3-submit.json`.

- Second full renderer `20261024T212140Z-f1459bd9` complete, full scientific validation passes. Archive SHA256 `41d02a6ecb0bb693acdee431b8ebaecfd1075c85f7eb3091410c70679d63ea90`. Charged0.374634 h; completed total **2.87743/8 h**, third renderer active. All12 source-only pooled entropy rates frozen before any judge quality: `tennis/k6b-confirm-20261024-execution/source-rates-frozen-20261024T2145Z.json`, SHA256 `b84eed26aafa7a8298c5be52ba2356e996d4394617bfb9819b3f61b3261fd8cf`.

- Fourth renderer submitted **`20261024T214544Z-7d7c0bb1`**, Med037@349 both240-frame/one-run units, fixed matching C recipe; spec SHA256 `ac44b5b6d5d4bbaecf032ddc77c83cfcae9e147d86718d143fee7d189d5802bc`, code6ccc5ec2,2100 s/fresh44-min deadline. Receipt `/tmp/k6b-confirm-gen4-submit.json`.

- CPU report integration probe succeeded with all12 real anchor sources and both completed generation receipts; zero judged rows/no quality verdict, output `/home/itec/emanuele/Datasets/pointstream-data/tennis/k6b-confirm-20261024-execution/report-input-probe-20261024T2147Z.json`, SHA256 `d946015f67695c6367f0614cb78f39ae30f586bcdfef7b83c9e68cfd2386ac0c`. All source artifact checksums and substantive smoke/full receipts accepted by the final report path. No CUDA visible/no GPU use.

- Renderer3 `20261024T214351Z-3f29c4ca` complete; full scientific validation passes. Archive SHA256 `e01f05ebff35a4cf56f6e6b50e55d931567ded5ee205690d48183c4f35e8c3b5`; charged wall0.394257 h.

- Renderer4 `20261024T214544Z-7d7c0bb1` complete; full scientific validation passes. Archive SHA256 `bf6a7b3ed374cdd8efe951b313dcfbf643369ceeb3905b4247748fc5b192b253`; charged wall0.371825 h.

- Fifth renderer submitted **`20261024T220850Z-3f03ad01`**, Al023@228; spec SHA256 `c7c0bc1839046877c174297c3fca660c22e1ba43f36bf71ee07c46d04ff4bf70`, code48a7f136,2100 s/fresh44-min deadline. Receipt `/tmp/k6b-confirm-gen5-submit.json`. Completed total **3.64351/8 h** before pilot windows5–6.

- Sixth renderer submitted **`20261024T220858Z-76fbb3a9`**, Med002@338; spec SHA256 `1f9d62c14f4487fbc10bca662e0d2e419d79b647c3b7dc4189198cc38086332e`, code48a7f136,2100 s/fresh44-min deadline. Receipt `/tmp/k6b-confirm-gen6-submit.json`. First-six pilot now fully submitted; remaining-six render jobs not released.

- CPU preparation third3 complete with substantive full pose-map/run/crop/reference/source-decode validation; archive SHA256 `6bef89419065f5c7a28883027deb0f8828890a2982a804b9efd8893bb902e3fc`,0 GPU-h. All12 are prepared; remaining-six rendering remains unreleased pending the first-six quality decision.

- CPU preparation fourth3 complete with substantive full pose-map/run/crop/reference/source-decode validation; archive SHA256 `1843fae75e58df685baa9b3e01b5ebad933595190ae27b1ec7996485d0fb2300`,0 GPU-h. All12 are prepared; remaining-six rendering remains unreleased pending the first-six quality decision.

- Additional CPU preflight resolved saved-coordinate precision concern: native judge reconstructed the exact source R/S mask hashes on **all480 judged frames across all six pilot windows**, zero mismatches and no quality metrics. Initial null-image probe stopped on a hard-coded shape error; corrected to native `FRAME=(1920,1080)` before the successful audit;0 GPU-h. No scientific/source/renderer change. Evidence script and both logs archived as `tennis/k6b-confirm-20261024-execution/mask-preflight-20261024T2220Z.tar.gz`, SHA256 `f5a476fa9c3a06d4dff4ddf0671599e0d69b755568137f6cb2eca103d5a37591`. Judge work will be two three-window shards using the exact original preparation packs and corresponding generated archives, combined in one frozen-cohort report.

- Fifth renderer `20261024T220850Z-3f03ad01` complete; full scientific validation passes, archive SHA256 `7a52a050f632b8f6277df7901efccffb8325a921934f27912201d15b5f768ff0`, charged0.392057 h. Completed total **4.03556/8 h**, sixth renderer active. First three pilot windows now enter actual fixed-profile scoring; no quality decision before both judge shards complete.

- First pilot judge shard submitted **`20261024T223427Z-7029113c`**, Al031/Med014/Al013, actual LPIPS/DISTS and exact R/S/index checks; spec SHA256 `b520410809d0bcac0f52f978f1894499c984dc4d1c3990b40094c2bc19115586`, code58ecca22,1800 s reservation/fresh44-min deadline. Smoke first3 judged frames on first two mixed-match windows, full all240 judged frames across threewindows. Receipt `/tmp/k6b-confirm-judge1-submit.json`.

- Sixth renderer `20261024T220858Z-76fbb3a9` complete; full scientific validation passes, archive SHA256 `1ebe6d44c31a946e537e914d521e52c82ec7b1692b02615c6189c5a1e3c255f2`, charged0.386331 h. **All six pilot windows fully rendered**; completed total **4.4219/8 h**, first judge active. Remaining-six renderer work is still unreleased.

- Second pilot judge shard submitted **`20261024T223557Z-e5de30f5`**, Med037/Al023/Med002, exact matched R/S and indices; spec SHA256 `994875315501d2b7c88a62ab065f95f17c7427178c7490a93f8f553e48bbeb4d`, code58ecca22,1800 s/fresh44-min deadline. Receipt `/tmp/k6b-confirm-judge2-submit.json`.

- First pilot judge `20261024T223427Z-7029113c` complete; all substantive full checks pass (identical actual R/S/index sets, finite real metrics/device, native source-oracle reproduction, exact frozen profile). Archive SHA256 `aa6198d4f8876c2f6b801c7d85f23b0cdebc52b64371f2f67639eec7bf5799c6`, charged0.095142 h; completed total **4.51704/8 h**, second judge active. No partial quality interpretation.

- Second pilot judge `20261024T223557Z-e5de30f5` complete; all substantive full checks pass, archive SHA256 `e8a568808cd13390856ca07e6d5c77c9dcc5873b2a45cd567760ed843b999030`, charged0.09235 h. Completed total **4.60939/8 h**, **no active GPU jobs**. Six pilot windows now have validated exact-mask real method/anchor quality; next is CPU fixed-rule report, before releasing any remaining-six renderer.

- Frozen CPU pilot report `tennis/k6b-confirm-20261024-execution/pilot-report-20261024T2243Z.json`, SHA256 `589ce8bf984737d9a17946f54820ac9adf30be993f9d45c8133f8b437a174cd8`: **12/12 evaluated billed cells and12/12 unbilled cells win**, all bracketed/exact-mask/fair ten-point envelopes, three windows per match. This is **12/24 frozen cells so far, incomplete/no interim pass**, not a completed12-window result. Source rates exactly equal the pre-quality frozen all12 rate file (independently checked within1e-12). No futility stop. Completed **4.609387/8 h**, no active GPU jobs. Remaining six render estimate6×0.39=2.34 h and judging≈0.19 h projects≈7.14 h total; reserve≈0.86 h. Promote remaining six unchanged, single-window≤45-min stages. No new fit/profile/height selection.

- Seventh renderer submitted **`20261024T224350Z-0cc91c89`**, Al016@273, fixed existing C recipe; spec SHA256 `640fd71611f3cc8f556625f062bf3ff93cefd0a26f64fef45c269055c3787346`, code58ecca22,2100 s/fresh44-min deadline. Receipt `/tmp/k6b-confirm-gen7-submit.json`. Source-only scale audit also counts all remaining run lengths to verify padding cost before further release.

- CPU scale-layout audit: Al016 has234/223 coded player frames across7/10 runs; Med017240/229 across1/8. Read the exact pinned `animate_anyone/pipelines/context.py` and long pipeline on gpu6: short runs≤24 use their real length (no24-frame padding); long runs use stride1/overlap4. Denoiser context-frame work Al016545 and Med017574 is no greater than the pilot one-run480-frame window576; only modest per-run setup overhead differs. Keep the same1700 s full cap/2100 s reservation and frozen renderer; no tuning/cancellation/replay. Original conservative24-padding count was not a sampler cost estimate. Remaining four windows each two240-frame/one-run units.

- Eighth renderer submitted **`20261024T224543Z-9e1145bb`**, Med017@206, fixed existing C recipe; spec SHA256 `9f0ecf0aa03ed03dad8ac992403b3281492a1370818228ef7fa9bb3b1fa240c2`, code58ecca22,2100 s/fresh44-min deadline. Receipt `/tmp/k6b-confirm-gen8-submit.json`. Completed total4.609387/8 h before remaining-six render work; renderers7–8 active.

- Seventh renderer `20261024T224350Z-0cc91c89` complete; full exact recipe/coverage validation passes despite17 source runs across both players. Archive SHA256 `ff98cf212e6ddd38e0adec696003f8a6a2b19433086a8741fcf173b5f17d43cb`, charged0.34983 h. Completed total **4.95922/8 h**, eighth renderer active; same source-only cost reasoning confirmed, no model/configuration change.

- Ninth renderer submitted **`20261024T230641Z-31aa2383`**, Al008@261, both240-frame/one-run units, fixed C recipe; spec SHA256 `119ca7ddaff1383203e9aeb107461fcd661d1d43e5787559e9a105a41cb4a8c6`, code903e033a,2100 s/fresh44-min deadline. Receipt `/tmp/k6b-confirm-gen9-submit.json`.

- Eighth renderer `20261024T224543Z-9e1145bb` complete; full scientific validation passes, exact coded coverage retained including missing-pose gaps. Archive SHA256 `9c679c0e12c1f69ca1fafad52ed503c14a897df90bc7d766196851d440bad540`, charged0.371886 h. Completed total **5.3311/8 h**, ninth renderer active.

- Tenth renderer submitted **`20261024T230918Z-b01ae9ae`**, Med029@138, two240-frame/one-run units, fixed C recipe; spec SHA256 `6b88c4fa4fbfdf5c93c9f8e4e85c34ac7034d05cb74e93329dfe462424e4353c`, code903e033a,2100 s/fresh44-min deadline. Receipt `/tmp/k6b-confirm-gen10-submit.json`.

- Remaining-six CPU exact-mask preflight passed: all476 judging frames reconstruct identical source R/S hashes, zero mismatches and no quality metrics; **956 frames across all12 windows** now covered. Four CPU threads/no CUDA. Archive `tennis/k6b-confirm-20261024-execution/mask-remaining-20261024T2317Z.tar.gz`, SHA256 `853cf3109711fad844d228f8484759a9380452c957578d7133408d884e5f0bc0`, includes source-run cost audit.

- CPU comparison-completeness join confirms **all12 windows have all10 valid native anchor points**, zero failed/missing codec points (metadata only, no new quality decision). Thus no source curve is silently reduced before the remaining renderer/judge stages. Receipt `/tmp/k6b-confirm-all12-anchor-availability.json`.

- Ninth renderer `20261024T230641Z-31aa2383` complete; full scientific validation passes, archive SHA256 `5958b32dd0f59e0dd94bf6a6d31fc18eaf690f19b7073025c5fb8153c22c45b2`, charged0.366273 h. Completed total **5.69738/8 h**, tenth renderer active.

- Eleventh renderer submitted **`20261024T233009Z-28f16db9`**, Al047@123, two240-frame/one-run units, fixed C recipe; spec SHA256 `3f73094e5e929eda47dc422628b78e2e88e4b3dc949aefbcced37f53000e03f8`, code4c5cbea3,2100 s/fresh44-min deadline. Receipt `/tmp/k6b-confirm-gen11-submit.json`.

- Tenth renderer `20261024T230918Z-b01ae9ae` complete; full scientific validation passes, archive SHA256 `b2f54da54f7f5f6214df96d95f6487ced899d862355e1fb06bbf1b7a2a3d1400`, charged0.374804 h. Completed total **6.07218/8 h**, eleventh renderer active.

- Twelfth/final renderer submitted **`20261024T233245Z-ebafc4f5`**, Med021@246, two240-frame/one-run units, fixed C recipe; spec SHA256 `ca802978925191bcff232602229e6e96387d50a754efea64c95d516ecb936794`, code4c5cbea3,2100 s/fresh44-min deadline. Receipt `/tmp/k6b-confirm-gen12-submit.json`. All12 renderer windows now submitted under the unchanged cohort/recipe; completed total6.072180/8 h before active11–12 and final judging.

- Eleventh renderer `20261024T233009Z-28f16db9` complete; full scientific validation passes, archive SHA256 `99bf174660f2106a6bed58c7b7bd7c0a09542e3b31d7d50b252e588c97ffc438`, charged0.366318 h. Completed total **6.4385/8 h**, final renderer active; first remaining-three scoring shard ready using exact original preparation/source/generation artifacts.

- Third judge shard submitted **`20261024T235346Z-9b0af845`**, Al016/Med017/Al008, exact matched R/S and indices; spec SHA256 `d93159e5a574828557f7050e8af02c8e6b9222d885c83d3e0baa4393e2ca1364`, codee1277745,1800 s/fresh44-min deadline. Receipt `/tmp/k6b-confirm-judge3-submit.json`.

- Final/twelfth renderer `20261024T233245Z-ebafc4f5` complete; full scientific validation passes, archive SHA256 `6557f1814ba1e7250b8c11bf15e23c1d5b17d6994f1dc4a272fe2f2dd30c0542`, charged0.375184 h. **All12 frozen windows now rendered and validated**, no new fitting. Completed total **6.81368/8 h**, third judge active; final judge ready.

- Fourth/final judge shard submitted **`20261024T235616Z-1a85923b`**, Med029/Al047/Med021, exact matched R/S and indices; spec SHA256 `2509b835b81cccda501afd41c9cbf392607374844f2e1185036020b5f367bb81`, codee1277745,1800 s/fresh44-min deadline. Receipt `/tmp/k6b-confirm-judge4-submit.json`. Current completed6.813681/8 h, only judges3–4 active.

- Third judge `20261024T235346Z-9b0af845` complete; all substantive full checks pass (exact R/S/indices and native source-oracle reproduction), archive SHA256 `07db4b9c20c6e318644aeaae94e3d98b114b685384277777e0ae2344de4ef45a`, charged0.089406 h. Completed total **6.90309/8 h**, only final judge active. No nine-window quality interpretation; full12 result follows its validation.

- Final judge `20261024T235616Z-1a85923b` complete; all substantive full checks pass, archive SHA256 `d1c851f87e17ea748ee61ed9895ead7a5df73b00fcee96d775a207f694eb2c76`, charged0.092341 h. **All12 windows fully rendered, scored and validated; no active GPU jobs. Final spend 6.995428/8 h**, including smokes, validation reservations and both initial failed requests; no fitting/Wan-Animate. Next CPU final report only.

## 2026-10-25 — completed frozen clip-local tennis-player diagnostic

**Decision reached:** 23/24 window–metric comparisons win at the billed adapter/guider rate (95.83%, above the frozen 18/24 bar). LPIPS wins 11/12 and DISTS 12/12. Unbilled wins 24/24. Alcaraz–Hurkacz contributes 12/12 wins; Medvedev–Struff 11/12. All 24 primary cells resolve on complete native anchor envelopes; no window or failed cell was dropped. The full 12-window result follows a 12/12-cell first-six pilot; there was no interim pass.

**Claim:** a descriptive result on clip-disjoint moments in two familiar broadcasts, with no overlap detected by the recorded clip tests. Original-source/extraction identity and replay independence remain unresolved. This is not independent-source, training-clean or unseen-match confirmation, and does not overturn the historical step-5 failure. References and exact target matte R remain granted and unbilled. The primary rate assumes a shipped common tennis-base decoder.

### Frozen question and comparison

Test whether the unchanged step-5 C renderer beats the achievable SVT-AV1/DCVC-UF anchor envelope on 12 previously unused clip-local 240-frame windows after a source-only court-registration gate. The competing explanation is that old view/registration and appearance difficulties, familiar-source adaptation, or granted side information explain the apparent gain. The present cohort cannot isolate those causes.

The cohort, source hashes and inclusive native frame ranges are frozen in the [task card](../cards/k6b-players-confirmation.md). The execution manifest is `tennis/k6b-confirm-20261024-execution/manifest-v2.json`, SHA256 `8b81c4b9645a24bb5b613bdda908440735fe932d23ea251b0c567a366bda090b`. Each `@N` names the absolute review frame, not the extraction start. Source-only gate: at least 228/240 K8b `fits=true` rows (≤6 px), finite nonsingular homography; missing/invalid=false. All 12 pass, so pooled and gated sets coincide and gate discrimination/fallback benefits are untested. **115 px remains exploratory and was never a selection or rescue rule.**

Native C settings: existing tennis base plus matching Alcaraz/Medvedev adapter and trained pose guider; 4-bit, guidance 2.0, 512×784, 30 DDIM steps, 24-frame contexts/overlap 4, source-deterministic unit/run seeds. No fitting, profile search, Wan-Animate or other study was run. Geometry-only preparation froze players, registration/scoring indices, pose runs/crop boxes and first-fitted-scored references before quality. Missing poses retain their native fallback; no quality-dependent frame filtering was introduced.

Anchors and method use identical source windows, R/S masks, native 25/50 fps and every third predeclared scoring frame. All 120 anchor points are valid: `svt-48-s1`, `svt-62-s1`, `svt-63-s2`, `svt-55-s4`, `svt-63-s4`, `svt-63-s8`, `dcvc-0-s1`, `dcvc-0-s2`, `dcvc-0-s4`, `dcvc-21-s2`. Native codec round trips, R accounting, source keypoint codes, finite LPIPS/DISTS and CUDA/kernels were validated. CPU reconstruction and actual judge hashes agree on all 956 judging frames. There are 2872 registered scoring frames within 2880 unique target frames; registration exclusions were frozen before quality.

Primary rate = the q4-velocity-e4 keypoint entropy bound pooled over all 12 windows plus adapter+trained-guider bytes amortized over 7200 s. These are optimistic pooled symbol costs, not a new deployed entropy bitstream. Alcaraz payload 1,488,252 bytes; Medvedev 1,495,592 bytes. Rates exactly reproduce the source-only freeze `source-rates-frozen-20261024T2145Z.json` (SHA256 `b84eed26aafa7a8298c5be52ba2356e996d4394617bfb9819b3f61b3261fd8cf`).

Each metric is compared at the method rate with the upper concave achievable anchor envelope. Interpolation is permitted only inside the measured rate bracket; unbracketed cells stay unresolved. Strictly lower error wins. No BD-rate is reported from a single method operating point. The frozen descriptive bar is ≥18/24 cells; below the billed bar but ≥18 unbilled requires user judgment. Futility would stop only if all remaining unbilled wins could not reach 18. All 12 were completed. Two match clusters are not 12 independent samples.

### Full-cohort measurements

Lower error is better. Each metric cell shows method / matched-rate anchor envelope; bold identifies the sole loss. Foreground rates exclude the granted matte/reference and common base.

| Window | Billed kbps | LPIPS method / envelope | DISTS method / envelope | Judged frames |
|---|---:|---:|---:|---:|
| `alcaraz_hurkacz/031@129` | 3.604463 | 0.245173 / 0.276848 | 0.223041 / 0.241554 | 80 |
| `alcaraz_hurkacz/013@213` | 3.643203 | 0.263237 / 0.277799 | 0.251764 / 0.256243 | 80 |
| `alcaraz_hurkacz/023@228` | 3.559647 | 0.211330 / 0.316932 | 0.214115 / 0.263014 | 80 |
| `alcaraz_hurkacz/016@273` | 3.947866 | 0.231613 / 0.253207 | 0.212485 / 0.255857 | 76 |
| `alcaraz_hurkacz/008@261` | 3.463794 | 0.209767 / 0.299849 | 0.216253 / 0.261582 | 80 |
| `alcaraz_hurkacz/047@123` | 3.549074 | 0.231017 / 0.270606 | 0.210635 / 0.243777 | 80 |
| `medvedev_struff/014@194` | 4.653833 | 0.170541 / 0.218405 | 0.192080 / 0.220090 | 80 |
| `medvedev_struff/037@349` | 4.466766 | 0.223077 / 0.251592 | 0.201723 / 0.239426 | 80 |
| `medvedev_struff/002@338` | 4.760868 | 0.223823 / 0.255021 | 0.194962 / 0.220377 | 80 |
| `medvedev_struff/017@206` | 4.941293 | 0.188240 / 0.208610 | 0.203322 / 0.211307 | 80 |
| `medvedev_struff/029@138` | 4.924546 | 0.274818 / 0.286920 | 0.220742 / 0.238160 | 80 |
| `medvedev_struff/021@246` | 4.773184 | **0.265830 / 0.253451** | 0.202030 / 0.223712 | 80 |

**Base-charged alternative:** charging the actual 48,878,567-byte tennis-base checkpoint over the same 2 h adds54.309519 kbps. It produces0 wins among 12 bracketed cells (the six Medvedev windows×two metrics); the 12 Alcaraz cells are above their measured rate brackets and remain unresolved. This alternative does not establish the bar. The primary result therefore cannot be presented as a fully charged end-to-end codec saving. Reference transport, target matte transport/estimation and background costs remain outside this diagnostic.

### Resource ledger and evidence

**Final charged GPU wall: 6.995428306 of 8 GPU-hours; 1.004571694 h unused.** Independently summed from all 24 terminal `run/status.json` receipts, including both initial failed requests, every smoke, setup and validation reservation. No GPU jobs remain active. The unpublished `20261024T204512Z-8a7d6449` had no spec/ready/run and used 0 h; its snapshot was retained, never replayed. Each native stage stayed within its ≤45-minute recovery bound; real Ada hardware/device/kernel checks passed. CPU audits used 0 GPU-hours.

| Fleet job | Stage | Terminal state | Charged GPU-h |
|---|---|---|---:|
| `20261024T174042Z-463f42af` | initial invalid-source request | failed | 0.002858 |
| `20261024T174553Z-ecc5ca36` | initial validator-placeholder failure | failed | 0.072818 |
| `20261024T193537Z-12dffb55` | features first6 | complete | 0.290773 |
| `20261024T200318Z-12faa9a8` | features remaining6 | complete | 0.293374 |
| `20261024T201059Z-29586984` | anchors/source codes shard1 | complete | 0.374196 |
| `20261024T203305Z-855eccf9` | anchors/source codes shard2 | complete | 0.365527 |
| `20261024T205303Z-4928c016` | anchors/source codes shard3 | complete | 0.351748 |
| `20261024T205730Z-869e62f2` | render alcaraz_hurkacz/031@129 | complete | 0.371330 |
| `20261024T211935Z-d45d6fbb` | anchors/source codes shard4 | complete | 0.380167 |
| `20261024T212140Z-f1459bd9` | render medvedev_struff/014@194 | complete | 0.374634 |
| `20261024T214351Z-3f29c4ca` | render alcaraz_hurkacz/013@213 | complete | 0.394257 |
| `20261024T214544Z-7d7c0bb1` | render medvedev_struff/037@349 | complete | 0.371825 |
| `20261024T220850Z-3f03ad01` | render alcaraz_hurkacz/023@228 | complete | 0.392057 |
| `20261024T220858Z-76fbb3a9` | render medvedev_struff/002@338 | complete | 0.386331 |
| `20261024T223427Z-7029113c` | judge shard1 | complete | 0.095142 |
| `20261024T223557Z-e5de30f5` | judge shard2 | complete | 0.092350 |
| `20261024T224350Z-0cc91c89` | render alcaraz_hurkacz/016@273 | complete | 0.349830 |
| `20261024T224543Z-9e1145bb` | render medvedev_struff/017@206 | complete | 0.371886 |
| `20261024T230641Z-31aa2383` | render alcaraz_hurkacz/008@261 | complete | 0.366273 |
| `20261024T230918Z-b01ae9ae` | render medvedev_struff/029@138 | complete | 0.374804 |
| `20261024T233009Z-28f16db9` | render alcaraz_hurkacz/047@123 | complete | 0.366318 |
| `20261024T233245Z-ebafc4f5` | render medvedev_struff/021@246 | complete | 0.375184 |
| `20261024T235346Z-9b0af845` | judge shard3 | complete | 0.089406 |
| `20261024T235616Z-1a85923b` | judge shard4 | complete | 0.092341 |

CPU fleet preparation/pack jobs were `20261024T200247Z-b63794f9`, `20261024T203302Z-332cab62`, `20261024T204514Z-f427e93b`, `20261024T210930Z-91f893e4`, `20261024T212000Z-7d4ddcf5`, `20261024T214359Z-53bb8f6f`; each completed with substantive full validation and0 GPU-h. The clip-only recovery audit covered 61 clips/39,853 decoded frames, with 0 native equal-size frame-hash matches and 0 pairs at 64×36 RMSE ≤3; re-encoded/cropped/alternate-view replays and upstream pretraining are not ruled out. Its preserved bundle SHA256 is `2bfafa0c0115171d784dd3d79696207917362f85a9653fa7939c3f446b57ff5b`.

Every native artifact is under `/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/JOB_ID/full/`; `ready.json`, `spec.json`, `gate.json`, `scientific-validation.json`, native metadata and published tarballs retain code revision/selected patch checksums, commands, encoder/decoder paths/versions, model/input hashes, pinned environment and actual GPU identities. Main environment archive SHA256 `32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661`; JCV `bf738ce596552875e81d067de933ba1ae610c970dba915570efdb2784add8a7e`. No weights or raw experiment outputs were added to the checkout.

Final CPU JSON: `/home/itec/emanuele/Datasets/pointstream-data/tennis/k6b-confirm-20261024-execution/final-report-20261025T000250Z.json`, SHA256 `e12e296fcb505bc8d5da9a9f0d4398b99a29071da8d0c8a5b193527449a72e3f`. It validates all 20 comparison/generation stage receipts and every referenced native artifact checksum. Runtime: Python 3.10.20, NumPy 1.26.4, SciPy 1.10.1, PyTorch 2.2.2+cu121 with CUDA hidden; source file hashes retained. Native GPU jobs used their separate pinned packed environments.

Final reproducibility bundle: `/home/itec/emanuele/Datasets/pointstream-data/tennis/k6b-confirm-20261024-execution/final-evidence-20261025T072817Z.tar.gz`, SHA256 `f0b68ee73f98d4a6e8f659c758664a14f4616593755bc02611fd05db7b2b824b`, 1,960,964 bytes / 49 members; exclusive creation and remote hash verification. It contains report/runtime, all 24-job budget audit, receipts, exact spec factory, source-only mask/rate proofs and `k6b-confirm-reproduce-report.py NEW_OUTPUT_JSON`. Existing evidence remains untouched.

Checks: substantive native smoke and full validators passed for every retained feature/anchor/prepare/generation/judge result; all 120 codec points and 956 mask identities passed. Local `ruff check`, configured mypy (200 files), focused scientific tests passed. Full local suite: 850 passed / 24 skipped / one unchanged macOS `/home`-symlink hook failure; no hook change was made. The PR Linux `checks` job passed for the same scientific source. No recurring model monitors, other chats, fitting or Wan-Animate were launched.

**Stop condition reached:** the complete frozen-cohort descriptive decision is made within budget. Strong provenance confirmation remains unresolved; registration-gate discrimination and 115 px remain unconfirmed. Auto-merge remains off following the earlier automatic approval-review rejection of merge authorization; no merge retry or bypass was attempted. Root-owned PLAN standings/dispatch narrative are left for the planning chat.

## 2026-10-25 — CPU-only side-information rate-budget audit (rule frozen)

**Authorization and scope:** new bounded CPU-only entry requested through the planning chat with the user's explicit approval. Reuse the completed frozen 12-window diagnostic and its saved source masks, prepared references/keypoints, generations and anchor curves. This does not extend its completed 8 GPU-hour run (6.995428306 h spent). No GPU jobs, fitting, generation, model/cohort/quality-rule changes, new chats or recurring model monitors. Existing branch `codex/k6b-player-confirmation-plan` remains isolated and its PR is open; upstream PLAN-only updates have been merged before starting.

**Question:** can the fixed-output player-rendering advantage survive deliverable reference, matte and keypoint side information under explicit accounting?

**Hypothesis:** a fully specified, round-trip-verified transport of the saved conditioning data may fit the measured rate headroom for at least 18 of the frozen 24 window–metric cells when common base weights are shipped. **Competing explanation:** the apparent advantage is consumed by reference refreshes, target mattes, real keypoint coding/initialization or required geometry/timing metadata; favorable long-duration accounting may depend on untested reuse. Familiar-match adaptation and unresolved source/replay overlap remain additional limitations.

**Inputs frozen before measurements:** final report `final-report-20261025T000250Z.json`, SHA256 `e12e296fcb505bc8d5da9a9f0d4398b99a29071da8d0c8a5b193527449a72e3f`, under `Datasets/pointstream-data/tennis/k6b-confirm-20261024-execution/`; the same 4 anchor and 4 preparation archives identified in the completed entry. Verify input archive/member hashes before use. Preserve all originals read-only and publish only new audit outputs. The saved method has one quality/rate point per window; there is no renderer RD curve and no invented BD-rate.

**Decision rule, frozen before new rates/headroom:**

1. For each window and LPIPS/DISTS separately, invert its saved achievable upper-concave pixel-anchor envelope at the fixed generated quality. Report the supremum rate that still wins strictly and the signed additional-rate margin relative to the original billed rate. Equality is not a win. Never extrapolate: retain left/right-censored thresholds, unbracketed scenarios and unresolved cases. A method better than the best measured anchor gets only a measured-bracket lower bound on tolerated extra rate, not unlimited headroom. Retain the existing sole losing cell.
2. Re-score accounting scenarios using the same fixed qualities at their new total rates, only within measured brackets. Keep all 24 cells in the denominator and report each metric/window/match; the descriptive bar stays at least 18/24. No statistical inference from 12 independent samples: there are two familiar match clusters. Pooled and gated sets remain identical; all 12 passed registration, so gate discrimination stays untested and 115 px stays exploratory.
3. Replace (subtract once) the pooled keypoint entropy bound with an actual decoder-verified stream. Count framing, code tables/initialization, track/run/frame presence, fps/timestamps, reference associations, crop/geometry/compositing metadata and integrity headers required by the declared decoder. Avoid double billing existing adapter/guider bytes. Verify source keypoints/visibility semantics and reconstructed native pose maps/crop placement against prepared inputs; if bit equivalence is not established, mark quality reuse unresolved.
4. Measure concrete reference costs on the saved full conditioning crops using lossless PNG (exact round trip); JPEG at qualities70/85/95 is an explicitly rate-only sensitivity. Measure exact per-player target matte transport, preserving native timing and overlap/compositing semantics, with existing lossless mask/RLE compression and a cheap temporal/predictive representation if justified. Check whether the native 960×540 grid exactly reconstructs the saved 1920×1080 masks before using it. Approximate half/quarter-grid masks are rate/distortion diagnostics only; source-mask IoU or reference PSNR never substitutes for new generated/composite LPIPS/DISTS.
5. Show both per-window reference refresh and explicit 60/600/7200-second asset amortization. A hypothetical once-per-match bank must pay for every distinct saved reference actually used by that match, not silently treat the 24 window-specific player references as one image. Label longer reuse as an accounting assumption whose future quality is unmeasured. Keep the existing two-hour adapter/guider primary charge and shipped-base assumption; show charging the actual 48,878,567-byte base checkpoint over explicit durations separately. Normalize decision rates over registered scoring seconds like the saved report, and also show whole-native-window wire rates; disclose charges for unscored frames rather than dropping them for free.
6. Bit-exact conditioning transport permits fixed-saved-output rate accounting, not a new regeneration or end-to-end quality measurement. Any lossy references, approximate/decoder-derived mattes or altered keypoints that change conditioning/compositing invalidate transfer of the original quality scores. Their unchanged-quality evaluations are explicitly optimistic counterfactuals. Background/fallback transport, independent source/replay identity and full deployment remain separate prerequisites.
7. First compute the free-side-information fixed-output bound, then a representative round-trip smoke before scaling any coder. Stop building a proposed route if its valid, configuration-specific optimistic gate cannot reach 18/24; do not claim this rules out all possible coders or quality-changing methods. Report a concrete route as plausible only if its fully enumerated measured costs fit and required equivalence checks pass; otherwise report tested-route failure or unresolved prerequisites. If a concrete route passes, propose a separate correctness smoke/pilot and budget, but run no GPU work.

**Budget:** 0 GPU-hours; at most 2 CPU core-hours for measurement/validation workers, at most 4 CPU threads, and 3 h wall for this bounded audit including engineering/checks. Start `2026-10-25T08:13:38.738874+00:00`; wall deadline `2026-10-25T11:13:38.738874+00:00`. Planned stages: envelope/accounting oracle ≤10 core-min; representative transport smoke/pilot ≤25; full 12 saved-input transport ≤45; round-trip/report verification ≤25; reserve 15. Each remote command≤30 min and every wait≤45 min; use smaller checkpointed shards as needed. Record actual user+system CPU and wall time, commands/runtime/input hashes. Reuse existing environments and real CPU coders; no downloads or paid APIs. Stop at the decision, exhausted budget, required user judgment or missing essential evidence; preserve partial results.

**Durable state:** this entry and [audit card](../cards/k6b-player-rate-budget.md). No new measurements or remote audit jobs have run at this pre-registration point. The completed diagnostic remains unchanged.

- CPU oracle completed before transport measurement: original 23/24 decisions reproduce exactly, all 24 rate crossings are inside measured envelopes; winning-cell extra-rate margins range 0.7387–9.8342kbps, with the existing Med021 LPIPS cell at−1.6296kbps. Free-side-information fixed-output bound reaches18/24, so a concrete transport smoke is justified. Oracle input/output SHA256 `e12e296fcb505bc8d5da9a9f0d4398b99a29071da8d0c8a5b193527449a72e3f` / `2165b4819d8ff0b2ab654d63e79fbdb5dcd567908ea2923be2cfddfb4b0da2f2`; 2.558 CPU-seconds,0 GPU-h. Exact command/source checksum in `/tmp/k6b-rate-audit-oracle-receipt.json`, to be included in the durable final bundle.
- Code pinned at `dfc06164`: strict inverse/bracket/equality tests and real keypoint/mask/reference corruption/round-trip fixtures pass (9 tests); full configured ruff and mypy pass (204 source files). The keypoint transport includes every-frame visibility, run/frame presence, timing, reference association, crop-rule/compositing and model-profile initialization, with no free pooled codebook. Native pose-map/crop equality will be checked on every saved coded frame.
- Project-native CPU fleet doctor passed on gpu2/gpu6; the standalone `agentctl` doctor reported gpu2 not enrolled, so no enrollment or remote agent was started. Execution uses PointStream's existing native fleet adapter and packed main environment. The representative smoke covers Al031/Med014/Al016 (both fps and17-run case), then the exact same path scales to all 12 only after substantive packet/native equivalence validation. CPU-only device declaration,4 threads,240s smoke/900s full/120s validator,1500s job bound; no GPU model initialization.

- CPU-only transport audit submitted **`20261025T082944Z-29f021a5`**, spec SHA256 `5436cbbbf1e68a6bd49f05344959a2be5c012b2cb31909baa6819f8a69b91525`, source `dfc06164`, nine immutable staged inputs, CPU4 threads/no GPU. Input/output job path follows `jobs/fleet/inbox/JOB_ID/`; returnedpending. Receipt `/tmp/k6b-rate-audit-transport-submit.json`. Never replay an uncertain submission. This job is independent of the completed 8-GPU-hour diagnostic ledger.

- Accounting clarification before opening transport costs: equal-rate results outside measured brackets remain unresolved. Separately record an actual measured anchor at no greater rate and no worse error as a fixed-output dominance witness. This is not an interpolated/extrapolated quality at the new rate; it can rule out that exact saved-output configuration under a rate budget. Such dominance cannot gate a lossy/altered-conditioning route whose new quality is unknown. All primary24-cell bracketed counts remain unchanged.

- CPU transport job `20261025T082944Z-29f021a5` **complete** on gpu2's CPU, no GPU allocation. Representative3-window smoke and all 12-window full run passed 57/219 substantive checks. All5718 coded player-frame pose maps and crop boxes match prepared inputs exactly; all 24 RGB PNG references are pixel-exact and distinct. Every-frame visibility, timing/runs and actual packet initialization are included. Full measurement wall 412.48 s / CPU 612.44 s, with full decoder revalidation CPU 670.84 s including measurement; smoke CPU 146.98 s including its revalidation, fleet smoke validator 16.29 s. All remain below the new2-core-hour/3h cap; old GPU ledger unchanged.
- Custom workload archive exists at `jobs/fleet/inbox/20261025T082944Z-29f021a5/full/published.tar`, SHA256 `6865035e3132b6e07600fdaabc8c9d7152b8b6e3c3260d9f3df21cab33ca33c7`, 11,018,240 bytes / 231 members. The native staging index did not auto-register this custom publication (`published=null`); it was independently hashed and bulk-copied without modifying that record. Independent CPU decoding of the complete archive passed all 219 checks (28.66 CPU-seconds). The measured source is `dfc06164`. A later storage-only correction puts per-packet writes on local scratch and validates/publishes bulk archives; original completed evidence is preserved.
- First CPU accounting export encountered undefined auxiliary anchor diagnostics while serializing a dominance witness; no rate/quality measurement failed. Witnesses now contain only finite rate/error/name fields, with a regression fixture; the partial local JSON is preserved separately. Accounting includes independent measured-point dominance without replacing unbracketed equal-rate cells by invented numbers.

### Completed CPU rate-budget audit — 2026-10-25

**Decision:** the tested quality-preserving transport fails the frozen 18/24 descriptive bar. Actual keypoints + a two-hour PNG reference bank + exact target mattes retain **0 known wins/24 cells**: 22 bracketed losses and 2 equal-rate cells above their measured bracket (Al047, both metrics), which remain unresolved. All 24 nevertheless have an actual measured pixel-anchor point at no greater rate and no worse error, so the valid configuration-specific fixed-output achievability upper bound is 0. No quality was extrapolated outside a bracket. This rejects these tested packets at the fixed saved output, not all possible coders or future renderings.

Even **free mattes** do not rescue the selected actual-keypoint/PNG-bank route at 2 h: 17/24 wins (LPIPS 7/12, DISTS 10/12; Alcaraz–Hurkacz 9/12, Medvedev–Struff 8/12). Conversely, the chosen exact-matte format alone, with free keypoints and references, already leaves 0/24, all bracketed. Thus improving only reference/keypoint transport cannot rescue that matte configuration, and improving only matte transport cannot rescue the chosen 2 h keypoint/PNG configuration. A deployable alternative remains unresolved; no complete route passed a valid gate, so **no GPU smoke/pilot is proposed or run**.

#### Frozen-output headroom and residual budget

All 24 crossing rates lie inside the measured achievable envelopes. The original sole loss remains a negative allowance. Equality at a crossing is not a win. Allowances below are maximum extra kbps relative to the original billed point; residuals subtract the real keypoint replacement delta, the 7200 s PNG bank charge and exact matte packets. A negative residual is a rate gap to a measured crossing, not an extrapolated quality at the new rate.

| Window | LPIPS extra kbps | DISTS extra kbps | Residual LPIPS kbps | Residual DISTS kbps |
|---|---:|---:|---:|---:|
| `alcaraz_hurkacz/031@129` | 2.977 | 3.002 | -47.255 | -47.229 |
| `alcaraz_hurkacz/013@213` | 1.253 | 0.739 | -43.093 | -43.607 |
| `alcaraz_hurkacz/023@228` | 6.461 | 6.715 | -37.156 | -36.902 |
| `alcaraz_hurkacz/016@273` | 1.896 | 6.126 | -50.256 | -46.026 |
| `alcaraz_hurkacz/008@261` | 6.209 | 6.249 | -36.908 | -36.868 |
| `alcaraz_hurkacz/047@123` | 3.028 | 4.519 | -48.724 | -47.233 |
| `medvedev_struff/014@194` | 8.565 | 9.573 | -65.556 | -64.549 |
| `medvedev_struff/037@349` | 4.633 | 9.834 | -91.046 | -85.844 |
| `medvedev_struff/002@338` | 5.988 | 9.318 | -99.203 | -95.873 |
| `medvedev_struff/017@206` | 3.591 | 2.822 | -70.691 | -71.461 |
| `medvedev_struff/029@138` | 2.179 | 5.754 | -111.280 | -107.705 |
| `medvedev_struff/021@246` | -1.630 | 8.021 | -115.973 | -106.323 |

#### Measured payload costs

All byte counts include actual self-delimiting packet framing, compressed metadata and codec initialization; the minimum of the three lossless matte coders is selected with its format tag paid. COCO-RLE+Zstd wins for all 12 exact windows; packed and temporal-XOR bitplane alternatives are also preserved. The existing pixel-by-pixel H5p binary arithmetic shape coder was inspected but not run: this keypoint/PNG-bank configuration already fails even with a zero-cost matte, so a more expensive matte implementation is not justified by its gate. This does not bound other reference/keypoint configurations or all possible mask coders. All 240 native frames and two per-player masks are charged, including registration-excluded frames. The 960×540 native grid reproduces every 1920×1080 target matte exactly by nearest repetition.

| Window | Real keypoint bytes | PNG pair bytes | Exact matte bytes | Quarter-grid bytes | Full exact rate kbps |
|---|---:|---:|---:|---:|---:|
| `alcaraz_hurkacz/031@129` | 4,236 | 170,112 | 57,189 | 14,096 | 53.836 |
| `alcaraz_hurkacz/013@213` | 4,311 | 100,787 | 49,871 | 12,389 | 47.989 |
| `alcaraz_hurkacz/023@228` | 4,072 | 143,605 | 49,139 | 11,956 | 47.177 |
| `alcaraz_hurkacz/016@273` | 4,546 | 154,774 | 57,992 | 14,193 | 56.100 |
| `alcaraz_hurkacz/008@261` | 3,941 | 183,434 | 48,777 | 12,073 | 46.580 |
| `alcaraz_hurkacz/047@123` | 4,050 | 142,775 | 59,133 | 13,900 | 55.301 |
| `medvedev_struff/014@194` | 3,542 | 172,134 | 41,940 | 10,190 | 78.775 |
| `medvedev_struff/037@349` | 3,252 | 168,791 | 55,052 | 12,730 | 100.145 |
| `medvedev_struff/002@338` | 3,567 | 208,533 | 60,621 | 14,165 | 109.952 |
| `medvedev_struff/017@206` | 3,734 | 200,703 | 42,017 | 9,605 | 79.224 |
| `medvedev_struff/029@138` | 3,859 | 212,541 | 65,388 | 15,452 | 118.384 |
| `medvedev_struff/021@246` | 3,602 | 216,448 | 66,085 | 14,711 | 119.117 |

Decision rates use the original registered scoring duration: 86.08 s total, within 86.40 s of whole native windows. The pooled full exact rate is **67.805 kbps**: keypoints/metadata 4.341, existing adapter+guider 1.656, reference bank 1.100, target matte 60.707. Whole-native-window wire accounting is separately reported in the CSV/JSON (pooled 67.563 kbps). The descriptive win criterion remains equally weighted over 24 window–metric cells; rates are duration-weighted here only to summarize traffic.

Real keypoint streams cost 3.284–6.432 kbps; the pooled entropy bound is subtracted exactly once and replaced, never charged twice. Existing adapter/guider billing remains 1,488,252 bytes (Alcaraz) / 1,495,592 bytes (Medvedev), amortized over 7200 s. Exact mattes cost 40.648–110.142 kbps, compared with 0.739–9.834 kbps of positive original extra-rate headroom. Per-window PNG refresh alone costs 84.341–360.747 kbps.

PNG reference banks pay **all 12 distinct conditioning crops per match** (24 unique source-pixel hashes): 895,487 bytes for Alcaraz–Hurkacz, 1,179,150 bytes for Medvedev–Struff. At 7200 s these add 0.995/1.310 kbps, not a whole bank every 4.8/9.6 s and not one free image shared by unrelated window-specific conditioning. Longer asset lifetimes are accounting assumptions; future quality/reference reuse over those lifetimes is unmeasured.

| Reference accounting, target matte free | Wins /24 | Equal-rate unresolved /24 | Valid fixed-output upper bound |
|---|---:|---:|---:|
| Refresh actual pair per window | 0 | 24 | 0 |
| All match references over 60s | 0 | 24 | 0 |
| All match references over 600s | 0 | 0 | 0 |
| All match references over 7200s | 17 | 0 | 17 |

With the same actual keypoint packets and free mattes, all-PNG reference-bank amortization would have to **exceed  14570.99s (4.047h)** to reach 18 cells; the existing model charge is still fixed at 2 h in that sensitivity. Only 19 cells can win even with an indefinitely cheap reference bank under this keypoint implementation. This is not evidence of four-hour usable references or future rendering quality.

#### Ablations, approximate transports and base sensitivity

| Fixed-output accounting | LPIPS wins /12 | DISTS wins /12 | Alcaraz wins /12 | Medvedev wins /12 | Pooled wins /24 |
|---|---:|---:|---:|---:|---:|
| Actual keypoints; refs/mattes free | 9 | 10 | 10 | 9 | 19 |
| PNG bank 2 h; keypoints/mattes free | 12 | 12 | 12 | 12 | 24 |
| Exact matte; keypoints/refs free | 0 | 0 | 0 | 0 | 0 |
| Actual keypoints + PNG bank 2 h; matte free | 7 | 10 | 9 | 8 | 17 |
| Full exact transport, PNG bank 2 h | 0 | 0 | 0 | 0 | 0 |

Half/quarter native-grid mattes (480×270 /240×135) are real decodable transports, but **lossy** relative to target R. Their source-mask IoUs are 0.896–0.923 /0.740–0.802; these are mask diagnostics, not LPIPS/DISTS. Quarter-grid matte rates remain 10.005–25.753 kbps. With actual keypoints and a 2 h PNG bank, both half- and quarter-grid variants yield 0/24 only in the **unchanged-quality counterfactual**; all those equal-rate cells are bracketed. No changed-matte composite was judged here.

JPEG70/85/95 references were encoded/decoded and their packet costs measured. If the old output quality were unchanged and target mattes were free, the 2 h-bank counts would be 19/18/17 respectively. These are **not measured wins**: decoded reference pixels change the renderer input, so no original generated LPIPS/DISTS can be transferred as a new result. The apparent 18/24 JPEG85 case is not a passing deployable route. Every combination of those JPEG references and the tested transmitted exact/half/quarter mattes still has 0/24 under unchanged quality; actual changed quality is unresolved.


Measured reference-bank packet costs (all 12 actual crops for each match):

| Format | Alcaraz bytes | Medvedev bytes | Added kbps at 2 h, Al / Med | Free-matte frozen-quality count /24 |
|---|---:|---:|---:|---:|
| png | 895,487 | 1,179,150 | 0.995 / 1.310 | 17 (optimistic bound) |
| jpeg70 | 249,631 | 331,943 | 0.277 / 0.369 | 19 (counterfactual) |
| jpeg85 | 372,575 | 479,293 | 0.414 / 0.533 | 18 (counterfactual) |
| jpeg95 | 680,615 | 859,765 | 0.756 / 0.955 | 17 (counterfactual) |

The primary keeps the common tennis base shipped. Charging its actual 48,878,567-byte checkpoint over 60/600/7200 s adds 6517.142/651.714/54.310 kbps. Base-only sensitivities retain 0 known wins: 0/0/12 bracketed cells respectively, all other equal-rate cells unresolved; every fixed-output cell has a real cheaper/no-worse anchor witness. Adding measured side information cannot rescue this configuration. No new base weights were fitted or substituted.

The prescribed primary keeps the saved pixel-anchor envelopes fixed while charging renderer-side information. If an eventual foreground pixel anchor also needs the identical matte transport and both sides pay exactly that same common cost, it cancels from matched-rate coordinates; the corresponding 2 h PNG/actual-keypoint result is still 17/24. This sensitivity does not implement a fully billed pixel/background anchor. No end-to-end codec claim follows from either accounting convention.

#### Actual decoder contract and limitations

Each KPS1/REF1/MSK1 envelope is 44 bytes (type, raw/coded sizes, SHA256), plus a real Zstd frame containing metadata and payload. PNG/JPEG internal headers, RLE initialization, compression tables and checksums are included. Reference packets are keyed by unit identity; ordered window delivery is KPS1 then MSK1, whose player axes follow KPS1 unit order. Research TAR and measurement JSON files are not wire packets. The actual profile/fps/frame-run presence, clip-local start/time base, registration bitmap, reference frame/box association, full conditioning crop, crop-rule constants, compositing order, model control and seed rule are charged. Common codec/profile implementation is assumed installed, like the frozen decoder.

The keypoint decoder sends q4 closed-loop velocity samples every fourth fitted frame, with explicit run starts/lengths and **every-frame** confident-joint bits. The old pooled bound counted visibility at sent frames only and did not deliver a complete initialized stream. Confidence magnitudes are not transmitted; they are unused after the native confident-joint predicate. Native regeneration of every pose map and target box proves model-conditioning equivalence on all 5718 coded player-frames, including source gaps/single-frame runs. All 24 PNG references round-trip pixel/color exactly. These are fixed-saved-output rate results, not new neural generations.

The existing adapter/guider bill already includes per-tensor LZMA framing and quantization scales; its full model bootstrap/tensor-directory container was not independently implemented in this audit. Any extra needed metadata only worsens the failed bound. Full-run crop geometry uses future poses/noncausal buffering; live latency is not established. Decoder background/fallback delivery and decoded-background quality are outside this foreground accounting. Approximate/decoder-derived mattes or lossy references require new composite/generated-quality evidence. Original source/replay identity remains unresolved; the familiar two-match scope, granted-side-information history, 115 px exploratory status and step-5 historical failure remain unchanged. All 12 pass registration, so discrimination of that gate is still untested.

#### Budget, reproducibility and stop

CPU job **`20261025T082944Z-29f021a5`** is terminal, device CPU on host gpu2, CUDA visibility empty, four admitted cores, no staged model weights and **0 GPU-hours**. Source measurement revision `dfc06164`; final analysis `b129ce04`. Native job wall 625.42 s. Recorded measurement/decoder/oracle/accounting CPU 896.09 s; conservative four-core native-job upper bound 2501.68 s plus separately recorded local checks/decodes, within the 7200 CPU-second cap. The audit completed before its 11:13:38 UTC wall deadline. The old 6.995428306/8 GPU-hour ledger was not extended.

Native smoke/full proofs passed 57/219 checks; independent decoding of the complete actual archive passed 219, and a standard fleet publication wrapper around unchanged packet bytes also passed 219. The custom measured archive was not auto-indexed by native staging; its actual file remains `jobs/fleet/inbox/20261025T082944Z-29f021a5/full/published.tar`, SHA256 `6865035e3132b6e07600fdaabc8c9d7152b8b6e3c3260d9f3df21cab33ca33c7`, 11,018,240 bytes / 231 members. Nothing in the completed run was overwritten or deleted. Future packet I/O uses local scratch and standard bulk publication.

New audit evidence: `/home/itec/emanuele/Datasets/pointstream-data/tennis/k6b-rate-budget-20261025T091041Z/cpu-rate-audit-evidence.tar.gz`, SHA256 `80d20f1838d3ae2dc2d39662201b64b6b313d57785b2cbc3cc5a3089d5c31376`, 10,573,727 bytes / 25 members, exclusively created and remotely hash-verified. Includes original diagnostic, transport/packet archive, all 64 scenario tables, per-window CSV, source/commands/runtime/input hashes, budget and decoder/test proofs. `accounting.json` SHA256 `f4e0de425a525f34a8debee19b8f6cbbe1ce242b095ec606390665b7d0c3b093`; `windows.csv` SHA256 `7d61246674e34f71efb76c1887d0998e03b2666be406dc454b397139e09886fb`. The input diagnostic SHA256 remains `e12e296fcb505bc8d5da9a9f0d4398b99a29071da8d0c8a5b193527449a72e3f`. Reproduce with `python -m experiments.tennis.k6b_rate_budget --diagnostic diagnostic.json --transport transport.json --out NEW.json`; decoder check is `CUDA_VISIBLE_DEVICES='' python -m experiments.tennis.k6b_rate_transport --check-archive packets.tar --proof-out NEW_PROOF.json`.

Configured ruff/mypy pass (204 source files), 11 focused tests pass; complete local suite: 862 passed / 24 skipped / one unchanged macOS `/home`-symlink hook failure. No hook change was made. A CPU JSON export failure from copying undefined auxiliary anchor diagnostics was corrected with finite witness fields and a regression fixture; no rate/quality measurement was rerun. The partial local export is retained in the evidence bundle. No new chats, recurring monitors, fitting, generation or GPU jobs were created.

**Stop condition reached:** the complete measured fixed-output transport configurations fail the budget gate; broader deployable alternatives remain unresolved. No qualified complete route supports a GPU smoke/pilot proposal. Any better lossless/reference/keypoint coder or decoder-derived-matte route needs its own new question, valid configuration-specific bound and quality evidence before expensive work. This entry and CPU budget are closed.


### 2026-10-25 — CPU-first compressed-matte quality follow-up (preregistered)

**Authorization:** user-authorized planning dispatch; at most 7200 aggregate CPU seconds and 3 h wall from first measurement; stages at most45 min, no GPU, fitting, regeneration, segmenter inference, other chats or recurring monitors. Previous diagnostic (6.995428306/8 GPU-h) and CPU audit remain closed. Branch `codex/k6b-compressed-matte`.

**Question/hypothesis:** can stronger compressed binary player mattes produce a useful actual rate/quality tradeoff when recompositing the frozen12 saved generations? Hypothesis: area reduction and the existing H5 LZMA temporal-XOR coder materially reduce matte cost while retaining useful output quality. Competing explanation: boundary loss, missing generated RGB outside saved crop boxes, or total keypoint/reference traffic consumes the apparent advantage. The previous unchanged-quality lossy-matte calculation cannot decide this question.

**Frozen settings:** full-frame reductions8/16/32, independent player planes; original all 240 native frames and all player identities billed. Native H5 area-majority reduction (background-padded partial edge cells), temporal-XOR/LZMA encode/decode, native bilinear-threshold upsampling. Compare real COCO-RLE/Zstd transport of the same reduced planes; select the shorter declared packet with paid codec/shape/frame/identity metadata. The previous240×135 stride/nearest point is a bridge diagnostic, not silently equated with H5 area/bilinear reconstruction. No temporal search beyond these existing coders in this initial stage.

**Artifacts and judge:** audit saved rectangular RGB crop coverage against every candidate matte, including expansions outside original R. Never supply target foreground to missing generated pixels. Native paste clips to each saved crop box, retaining the identical original Telea background elsewhere; report unsupported matte area separately and retain this target-derived-background grant. Original exact R and oracle-union S, original judged/scored indices, original source RGB conversion, crop boxes, far-to-near compositing order, same windows and saved pixel-anchor envelope remain fixed. Recompute and verify original R/S hashes before judging.

**Decision:** a complete configuration needs >=18/24 equally weighted window–metric wins, with actual full fixed-frame LPIPS/DISTS and strictly bracketed achievable anchor comparison at its fully billed rate. No missing/unbracketed cell counts as a win; show each match and base-shipped/base-charged separately. PNG bank over 7200 s plus actual keypoint stream is primary; old-quality JPEG pricing is conditional only. Actual KP+PNG with free mattes was17/24; a mask coder alone cannot improve identical quality to18, but actual recompositing may change quality. No per-window quality-based codec/scale selection.

**Stages/budget:** reserve1500 CPU-s for all-window mask packets/IoU/boundary/coverage and validation,1000 for two representative full-resolution metric/recompositing smoke frames (one per match),4200 for complete scoring only if conservative smoke projection plus remaining work fits,500 contingency. Record both self+children CPU and a conservative threads×wall bound. Save per-window output; hard stop on budget/deadline. If projected full fixed-frame scoring exceeds remaining CPU, finish mask measurements and smoke quality, leave full24-cell decision unresolved, and prepare a concrete GPU judging preflight/request without dispatch. A small smoke is not a cohort result.

**Validator:** independently decode every candidate packet, prove reduced-grid equality, full240-frame/player order/timing contract, finite IoU/boundary/coverage counts and exact source identities. Metric smoke must verify fixed R/S hashes, finite actual network outputs and original composite control against corresponding saved judge scores (absolute tolerance1e-4); record versions, code/weight/input hashes and deviceCPU. No proxy replaces a neural judge.

**Limits:** familiar two-match clusters, unresolved source/replay/pretraining independence, original granted exact matte history, target-derived background/fallback, common-base assumption, exploratory115 px threshold and historical step5 failure remain explicit. Approximate transmitted masks do not establish an end-to-end codec.

**Reference follow-up (user question):** PNG preserved the original conditioning pixels; losslessness is not required for a future route. Existing JPEG70 bank cost249631/331943 bytes versus PNG895487/1179150 (Alcaraz/Medvedev), but no JPEG-conditioned quality was measured. Downsample+JPEG/AVIF, JPEG AI and same-player inter-frame reference updates are candidate separate experiments; do not spend GPU or transplant saved quality. Inter-frame coding must pay initial images, updates, identities, ordering, refresh/random access and required dependencies, without assuming correspondence across clip-local player IDs. JPEG AI is still-image coding, not itself temporal prediction (ITU-T T.840.1).

CPU measurement start no earlier than2026-10-25T09:52:29.948451+00:00; overall deadline2026-10-25T12:52:29.948451+00:00. First native CPU job smoke<=360 s/full<=900 s/validators<=60 s each, total supervisor cap1440 s with4 threads (conservative5760 CPU-s including repeat pilot). No completion-triggered model wakeup has been verified for this client; use attended one-batch native fleet wait, no monitor. Initial measurement code and checks are pinned before submission.

Native CPU job `20261025T095256Z-b7692db7` submitted, source `c2f8e670`, spec SHA256 `54b9d2d22202b4286b223bc7c2e193ece52cca2e0769e4ee0306bdbbfd8c72d4`. Status pending at submission; deadline and all SHA256-identified inputs in `/tmp/k6b-matte-spec.json` and native request. CUDA hidden; no GPU resource claim. Inspect this existing job before any further action; never replay uncertain submission. Four focused tests and configured ruff/mypy passed before submission.

Reference-design context: the repository already tested a separate SVT-AV1 stream of sparse references in VISOR H3a2 (`docs/experiments/h3a2.md`): references predicted from the previous reference and decoded references used for rescoring. That hand texture-transfer route failed; it neither validates nor rejects tennis reference coding. Its accounting lesson applies: charge the actual sparse stream and judge its actual decoded conditioning, never borrow dense-video inter-frame cost while retaining dense-video reference quality. Candidate CPU pricing pilot for a future reference experiment: native-size lossy image coding; half-size coding followed by a fixed upsampler to512×784; same-player ordered reference updates encoded with existing SVT-AV1. Start with three settings total, match bytes where feasible, and retain PNG as control. Audit JPEG AI's pinned implementation/dependencies/CPU cost before including it; no new weights, provider or paid API is authorized. Decoder image similarity is a diagnostic, not a substitute for actual regenerated LPIPS/DISTS. All initial/keyframe/update/dependency bytes and true asset lifetimes must be charged.

Local verification at source `c2f8e670`: four focused transport/geometry checks, configured ruff and mypy pass. Full suite864 passed/25 skipped/two platform restrictions: sandbox blocks `ps` in process-suspension test; rerunning that test outside sandbox passes. The unchanged protected-path test fails on macOS's unresolved `/home/itec/...` symlink (same prior audit failure). No unrelated hook/fleet implementation was modified.

Before reference sensitivity measurements: user asks about downsampling/lossy/inter-frame. Add only a cheap CPU rate diagnostic within this task's remaining cap: all 24 frozen PNG reference pixels, JPEG70 at scale1/0.5/0.25 (area downsample, fixed bicubic restoration), paid output-size/filter metadata; at most60 CPU-s. Compare real sealed packet bytes, record native-size PSNR only as an input diagnostic. No generation, learned codec installation or quality transfer. Inter-frame coding remains a design candidate until same-player correspondence and ordered update/lifetime semantics are established.

First CPU job 20261025T095256Z-b7692db7 failed in smoke after one valid window. The source identity guard caught a worker selection bug: the report groups matches rather than interleaving them, so the second source argument did not match the second row. No mismatched pixels were judged. Preserved checkpoint includes Al031's real two-network pilot, exact R/S and original GPU control within1e-4. Failed execution consumed182.806146 CPU-s (121.886372 user+60.919774 system),99.484736 s command wall;0 GPU. Corrective implementation selects pilot IDs explicitly, maps sources by verified SHA256 rather than row position, and independently decodes/reuses the preserved completed-window checkpoint. New source/order regression test added. CPU full-neural projection from this first frame alone is35622.65 CPU-s for 956 frames, already beyond7200; no full CPU cohort judging is promoted. Finish all-window mask audit and the second match's smoke only; retain complete decision unresolved.

Reference CPU sensitivity completed: all 24 original PNG reference images, JPEG70 packets at native/half/quarter dimensions with charged size/upsampler metadata. Native JPEG70:250299/332658 bytes; half256×392:92631/117396; quarter128×196:39544/47446 (Alcaraz/Medvedev). PNG baseline895487/1179150. Half-size packets save89.66%/90.04%; their2 h-bank charges are0.102923/0.130440 kbps. Recorded2.913435 CPU-s. These are measured input-packet rates only: no new generation, no measured output-quality preservation, no18/24 pass. Fixed bicubic restores512×784 conditioning. Source script and packet metadata/evidence retained separately; JPEG AI and temporal reference coding have not been run.

Corrected CPU continuation `20261025T095755Z-f5de5e93` submitted, source `a47ae7d7`, original deadline/cap unchanged; preserved Al031 checkpoint SHA2569b861f44ab838c4d4e617925606a5d8f5e20b7e4da0e05edc2687a9c0ea08ae1. Prior job is terminal failed, never replayed. New source mapping checked by five focused tests and configured ruff/mypy. Reference evidence `/home/itec/emanuele/Datasets/pointstream-data/tennis/k6b-compressed-matte-20261025T095900Z/reference-pricing-evidence.tar.gz`, SHA2564be024c022292b370eb1aaa807a502956fa3da44a2e16ffa0f31b9ab2367bb80.


#### CPU outcome — complete; full quality decision awaits separate budget

Native continuation `20261025T095755Z-f5de5e93` **complete** on gpu6 CPU (Intel Xeon Gold 6418H), CUDA hidden, four threads; no GPU resource or GPU execution. All12 windows/all 240 native frames/two players are coded at each8/16/32 reduction. The earlier failed job remains preserved; its valid Al031 checkpoint was independently decoded and reused. Original generation receipts certify exact judged-frame coverage for all 12; new raw RGB crops were reopened for the two real smoke frames, while all-window crop support uses the previously bit-equivalent decoded keypoint geometry. No extra RGB was invented.

| Full-frame reduction | Actual matte kbps, range across12 | Total kbps with actual KP, PNG/2 h and adapter/guider | Mask IoU | Boundary F,2 px Chebyshev tolerance | H5 / RLE selected windows |
|---|---:|---:|---:|---:|---:|
|1/8|9.995–25.493|15.969–34.897|0.857–0.892|0.904–0.922|2 / 10|
|1/16|4.531–10.490|10.575–19.894|0.692–0.738|0.429–0.506|8 / 4|
|1/32|1.341–3.620|7.283–13.024|0.485–0.572|0.223–0.253|12 / 0|

Rates include self-delimiting MTQ1 framing, format/scale/shape/player identities, H5 native headers/LZMA initialization or real COCO-RLE/Zstd headers and checksums. Both coders decode the identical small masks; select only by paid packet length. All240 frames are charged even when not registered/judged. Frame timing/fps remains in the verified, separately charged KPS1 contract. Per-window rates, full-native-window normalization and common-base-charged sensitivities are in the CSV. Charging the 48,878,567-byte common base over 7200 s adds54.309519 kbps. Background/fallback transmission is still absent, so these are foreground-accounting diagnostics.

The1/8 area-majority/bilinear route is not the old stride/nearest240×135 route: its IoU is higher at a similar rate. This bridge identifies a changed reconstruction, not a claim that coder changes preserved old output.

**Crop support:** all-window judged matte area outside available native crop boxes totals123414/97204/42519 per-player pixel incidences at 8/16/32;14807/26676/15357 are expansions outside original R. These are not unique full-frame pixels and include original missing-pose/crop support. Native clipping leaves the identical Telea/background canvas there. This is a valid restricted saved-crop reconstruction, not unrestricted new RGB beyond the saved rectangles. Original exact-R target-derived Telea fill is still granted and reused; no new segmentation inference or target-foreground fill is used.

**Actual neural smoke only:** first original judged frame (t=0) from Al031 and Med014; same fixed R/S hashes, original native source conversion, saved generated RGB and pose-derived boxes, same far-to-near order. CPU original controls differ from saved GPU scores by at most 0.00001717 LPIPS and0.00001580 DISTS, below frozen 1e-4.

| Window | Matte | LPIPS | DISTS |
|---|---|---:|---:|
|alcaraz_hurkacz/031@129|exact control|0.097575|0.126217|
|alcaraz_hurkacz/031@129|1/8|0.092429|0.132927|
|alcaraz_hurkacz/031@129|1/16|0.113696|0.165996|
|alcaraz_hurkacz/031@129|1/32|0.219319|0.220734|
|medvedev_struff/014@194|exact control|0.135688|0.151058|
|medvedev_struff/014@194|1/8|0.136862|0.150513|
|medvedev_struff/014@194|1/16|0.180015|0.183605|
|medvedev_struff/014@194|1/32|0.324694|0.279773|

At1/8 each smoke frame improves one metric slightly and worsens the other;1/16 and1/32 worsen both on these two frames. No24-cell tally or full-cohort win is inferred. Actual compositing can improve a metric, confirming why unchanged-quality lossy-mask calculations were insufficient.

Boundary/background inspection on the two saved smoke frames:1/8 modifies1096/836 pixels outside original R (Al/Med); the control changes zero. Their2 px boundary-ring RGB MSE rises838.72→918.82 and1604.40→1630.57. Stronger reductions also affect background and erase foreground; the complete per-frame SSE/ring diagnostics and images are retained, not substituted for neural judges.

**CPU limit and next stage:** actual four-variant metric time 24.44–37.26 CPU-s/frame projects23365–35623 CPU-s for 956 frames, before full reconstruction overhead. This exceeds the entire 7200 s budget, so full CPU neural judging was not promoted. The frozen >=18/24 decision is **unresolved**, not passed or failed. A concrete separate saved-render-only GPU request is in the card: estimated 0.40 GPU-h, proposed hard cap0.75 h in four <=675 s three-window parts, including real correctness smokes/validation, no new generation/fitting. Not authorized and not dispatched; no old GPU ledger is reopened. Lossy-reference conditioning would require its own generation/quality experiment and budget.

**Validation/provenance:** source `a47ae7d7` (retained first-window source `c2f8e670`); packed main SHA25632e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661, Python3.12/NumPy1.26.4/OpenCV4.10.0, PyTorch2.10.0+cu128 with all netsCPU, torchmetrics1.9.0. LPIPS AlexNet backbone SHA2567be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02, DISTS VGG16 SHA256397923af8e79cdbb6a7127f12361acd7a2f83e06b05044ddf496e83de57a5bf0; native H5 code `src/pointstream/mask.py`, native paste `experiments/tennis/k6b.py`, actual metric functions unchanged. All source/model/archive/packet identities and actual commands are in native receipts. Full published archive SHA2568a720640feb2d53f9c7703a7b430238b38c71f20d7936b36ba41a10f395d2f17. Independent full archive decode 77 checks and 120 statistical/prior-generation-coverage checks pass. Five focused tests/configured ruff/mypy pass; local full-suite platform restrictions documented above.

Native cumulative CPU 897.795831 s across both jobs (do not double-count corrected smoke). Conservative whole-supervisor four-thread bound plus 60 CPU-s local analysis allowance 2429.956 s, below 7200. All native jobs terminal; 0 new GPU-h; no monitors. CPU phase closed at its scoring feasibility stop condition. Source/replay uncertainty, two familiar match clusters, historical step5 failure, exploratory115 px and granted matte/background history remain unchanged.

Complete CPU evidence `/home/itec/emanuele/Datasets/pointstream-data/tennis/k6b-compressed-matte-20261025T095900Z/cpu-matte-evidence.tar.gz`, SHA256 `3cde843eca365323424ad659cac53914f02a2a0feaa222400a8604c1e372ba63`; self-contained native packet/pilot archive, per-window CSV, all input/weight/native command receipts, independent proofs, boundary/reference diagnostics, source snapshots and checks. No original results were overwritten.

Publication retry succeeded through normal automatic approval review after the user asked to continue. Both evidence files were verified by remote SHA256 and made read-only; terminal events acknowledged. The earlier review failure was a usage-limit error, not an unsafe-action determination.

CPU report published in PR #239 (`081f96c6` result commit), merge-commit auto-merge enabled; Linux CI pending at publication. Full report delivered to the planning chat. The earlier PR #236 merged after the user explicitly requested auto-merge; no merge approval remains blocked. CPU measurements ended before the original wall deadline; later usage-limit recovery resumed publication only, with no new compute.


### 2026-10-25 — Approved GPU judging of saved compressed-matte composites

**Authorization:** user explicitly approved the proposed maximum **0.75 GPU-hours (2700 GPU-seconds)**, estimated0.40 h, to finish the frozen saved-render judging. Separate ledger; old6.995428306/8 h diagnostic and completed CPU ledgers remain closed. Branch`codex/k6b-matte-judging` from merged CPU PR#239. No fitting, new generations, lossy-reference conditioning, segmentation inference, new chats/agents or monitors.

**Question/hypothesis/alternative:** measure actual LPIPS/DISTS versus total rate at each predeclared1/8,1/16,1/32 H5 area/bilinear reconstruction using saved rectangular generated RGB. Hypothesis: at least one fixed scale reaches>=18/24 strictly bracketed window–metric wins with delivered matte, actual keypoints and PNG banks over 7200 s. Competing explanation: reduced mask/boundary fidelity and paid side information erase the original granted-matte advantage. Two CPU smoke frames showed actual metric changes; neither old unchanged-quality scores nor JPEG rate sensitivities decide this stage.

**Frozen contract:** identical original 12 windows, original scored/judged frame sets (956 judged frames total), original exact R and oracle-union S with frame hashes, original source conversion, native far-to-near saved-crop clipping and same original exact-R Telea background. Decode the selected, already measured MTQ1 packets; verify units/shape/scale/hash against CPU artifact and actual KPS1 geometry. Keep generated pixels unchanged and preserve missing-crop fallback; never fill from target foreground. Score exact original control plus all three scales on every judged frame. Verify control against saved original GPU scores at absolute tolerance1e-4 per metric/frame, finite output for every expected variant/frame, original R/S identities, native model weights/versions/CUDA class, input hashes and model-memory cap.

**Decision/accounting:** each scale uses all its windows, no per-window quality-based scale or coder choice. Fixed pixel-anchor achievable envelopes; strict interpolation only inside measured rate brackets, no extrapolated wins. Actual keypoint packet replaces pooled entropy once, plus all 240-frame matte packet bits, all 24 PNG references in match banks over 7200 s and existing adapter/guider over 7200 s. Common base shipped primary; actual48,878,567 bytes over 7200 s separate. Report per-match and per-cell. Lossy-reference pricing stays conditional and receives no generated-quality claim. Preserve source/replay uncertainty, two familiar broadcast clusters, granted matte history/target-derived background/fallback, exploratory 115 px and historical step 5 failure. This is not end-to-end compression.

**Stages and hard budget:** four resumable parts, each three frozen windows, requested in original approved interleaved groups. Each native job reserves<=675 GPU-s total (sum<=2700), smoke<=60 s on first two windows/one original judging frame each, full<=540 s, substantive CPU validator<=30 s, remaining allowance for staging/validation. Only first part is promoted initially; its validated measured runtime must project remaining work within remaining allowance. Submit later parts sequentially; inspect existing IDs/host pool before dispatch. All wait stages<=45 min; attended bounded native fleet waits, no recurring wakeups. Save every completed window and partial per-frame checkpoints on scratch; preserve results on failure. No automatic replay.

**Stop:** validity gate fails; remaining hard GPU budget cannot cover the next stage; any complete scale reaches the descriptive rule; or all scales are mathematically unable to reach18 even granting every unprocessed (and unresolved) cell a win. If the decision is already settled, report the measured prefix and honest optimistic bound rather than invent the unmeasured remainder. Any full-cohort descriptive number requires all original judging frames; partial results remain explicitly partial.

Pre-dispatch implementation: new native module`experiments.tennis.k6b_matte_judge`, real original-score tolerance/partial-frame/optimistic-prefix tests;19 focused tests and configured ruff/mypy pass. GPU models are existing validated LPIPS/DISTS plain convolutions from fleet model–GPU table (Ada); code asserts actual parameters onCUDA, class and memory. Compatible poolgpu5/gpu6 inspected: both gpu5 GPUs lack24 GB free, gpu6 has an idle Ada with48 GB; dispatcher retains full compatible pool and claims the available device. Prior CPU job20261025T095755Z-f5de5e93 inspected terminal complete; existing fleet jobs preserved. Internal job time guard645 s sits inside675 s reserved per part, leaving30 s per-part reserve for supervisor overhead. Source, staged inputs and native commands pinned before dispatch.

GPU part1 submitted: `20261025T135736Z-c830e267`, source`679f0f94`, spec SHA256`efc2b52c4c89e5874f1d913b9b441fa583ff98e4f1b6b97dbf7e9a0f42a6db22`. Windows Al031/Med014/Al013;675 GPU-s reserved of2700, zero known spent at submission. Only this part is active/promoted; later parts remain unsubmitted. Native request contains all immutable inputs and645 s supervised cap. Inspect this existing ID before any further action; no uncertain replay.

GPU part1 job20261025T135736Z-c830e267 **failed in smoke before metric weights/scoring**: exclusive fleet does not export the shared-mode `PS_GPU_MEMORY_LIMIT_MIB` override. The new worker had indexed it unconditionally; existing native judge uses a24000 MiB fallback. Corrected to the explicit declared24 GB contract, honoring any smaller shared override and never exceeding24 GB; regression test added. No composite quality result was produced. Actual admitted interval10.262219 GPU-s is charged to this new ledger. Preserve the terminal job and artifacts. Corrected part1 is a new explicitly diagnosed attempt, not an uncertain replay. Tighten each remaining part's reserved occupancy to670 s and internal guard640 s;10.262219 +4×670=2690.262219 s stays within2700. Validated source/matte/rate/quality rules unchanged. Full local suite before this fix:870 passed/24 skipped, only unchanged macOS `/home` protected-path assertion failed.

Corrected GPU part1 submitted `20261025T140128Z-2dfd517f`, source`a8faea75`, spec SHA256`2a9c8c59005396f12c59bbb653342d260fea94464bd67461ffd45ac434a2ebf6`.20 focused tests/configured ruff/mypy pass.10.262219 GPU-s known spent,670 s reserved for this part, later parts unsubmitted; overall cap2700 s. Prior failed job inspected and preserved.

GPU corrected part1 job20261025T140128Z-2dfd517f **complete**, gpu6 Ada GPU1. Admitted occupancy432.463000 GPU-s; cumulative including failed smoke442.725219 s (0.122979228 h) of2700 s. All three windows' original judging frames complete; independent CPU report verifies full score arrays and original archived frame/R/S/control values, max control discrepancy2.80142e-5. All scales have0 known wins/6 measured cells,0 unresolved; optimistic total18/24 remains attainable, so do not stop yet. Pilot runtime379.68 s for three windows plus smoke/setup projects remaining stages within cap. Promote part2 only. Tighten operational stopping to each completed full window (same predeclared conservative bound), with effective/requested prefix and original three-window plan both recorded. Prior part1 immutable archive is SHA256-staged and independently validated before use; partial smoke never decides. Quality/model/packet/scoring settings unchanged.

GPU part2 submitted `20261025T141239Z-93456964`, source`052df444`, spec SHA256`8b0a3815ebc8996d67b6893282df41be647a101d3370fcbefe460f8eafaaffec`. Planned Med037/Al023/Med002 with per-completed-window conservative stopping; prior part1 archive SHA256`d24508668cd4208cb0eaca54d9ce612113f92c26f60dc9d6017a098af528cef7`.442.725219 GPU-s spent,670 s reserved for this part,2257.274781 s total remaining before it. Parts3/4 remain unsubmitted.20 focused tests and configured ruff/mypy pass after checkpoint/independent-original-reference validation changes.


#### GPU outcome — frozen criterion rejected; stopped after four complete windows

**Decision:** each fixed matte scale (1/8,1/16,1/32) has **0 wins in 8 measured window–metric cells**, all strictly bracketed, no unresolved primary cells. Even if all 16 cells of the eight windows without complete scores won, each scale could reach at most **16/24**, below frozen 18/24. The predeclared optimistic-prefix rule therefore decides all three tested configurations; stop paid work. This is a valid negative bound on the declared12-window criterion, **not a claim that all12 were judged or that every remaining window loses**. No full-cohort mean or independent-match confirmation is reported.

Measured windows: Alcaraz–Hurkacz031@129 and013@213; Medvedev–Struff014@194 and037@349. All original judged frames completed in each (320 total), original R/S hashes unchanged. Alcaraz0/4 and Medvedev0/4 measured cells per scale. Source/replay independence remains unresolved; these are two familiar broadcast clusters, not four independent matches.

**Execution and budget:**

| Native job | Result | Admitted GPU seconds |
|---|---|---:|
|`20261025T135736Z-c830e267`|setup smoke failed before scoring; preserved|10.262219|
|`20261025T140128Z-2dfd517f`|part1 complete,3 windows|432.463000|
|`20261025T141239Z-93456964`|part2 stopped after1 complete window by frozen bound|160.904566|

Total **603.629785 GPU-s = 0.167674940 GPU-h of 0.75**, including failed setup. Remaining 2096.370215 s is unspent; ledger closed at decision, not a transferable generation budget. Parts3/4 and the remaining two full-window evaluations of part2 were never run. Part2 also ran its mandatory Al023 one-frame smoke; that is not a complete-window result and is excluded from the win tally. All three jobs terminal, claims released by native completion, terminal events acknowledged; no recurring monitors. Previous diagnostic/CPU ledgers stay closed.

**Actual rate/quality, measured prefix only:** every point includes actual keypoints, all 240-frame mask packet bits, all 24 distinct PNG references in the two match banks over 7200 s, existing adapter/guider over 7200 s. Original pixel-anchor envelope and original scoring durations are unchanged. Lower LPIPS/DISTS is better; each cell is actual method / matched-rate anchor.

| Window | Scale | Total kbps | LPIPS method / anchor | DISTS method / anchor |
|---|---:|---:|---:|---:|
|alcaraz_hurkacz/031@129|1/8|17.783|0.242458 / 0.170321|0.231960 / 0.177733|
|medvedev_struff/014@194|1/8|25.782|0.173668 / 0.118906|0.201090 / 0.166316|
|alcaraz_hurkacz/013@213|1/8|16.608|0.258683 / 0.176121|0.261187 / 0.198260|
|medvedev_struff/037@349|1/8|29.309|0.225030 / 0.115292|0.207884 / 0.154115|
|alcaraz_hurkacz/031@129|1/16|11.810|0.264402 / 0.189543|0.243423 / 0.190956|
|medvedev_struff/014@194|1/16|15.745|0.197322 / 0.156424|0.221046 / 0.187637|
|alcaraz_hurkacz/013@213|1/16|11.147|0.286992 / 0.194024|0.273951 / 0.212178|
|medvedev_struff/037@349|1/16|16.220|0.242397 / 0.179250|0.226919 / 0.194365|
|alcaraz_hurkacz/031@129|1/32|8.005|0.323660 / 0.230026|0.268397 / 0.214418|
|medvedev_struff/014@194|1/32|11.095|0.263701 / 0.182409|0.267241 / 0.201242|
|alcaraz_hurkacz/013@213|1/32|7.776|0.342936 / 0.229786|0.298952 / 0.231185|
|medvedev_struff/037@349|1/32|10.939|0.319508 / 0.211758|0.271012 / 0.214614|

1/8 modestly improves mean LPIPS relative to the exact original control on the two Alcaraz windows, while worsening DISTS; other tradeoffs are in the raw arrays. These actual changes were measured, not inherited from an unchanged-quality counterfactual. The total-rate penalty nevertheless leaves every measured primary cell with higher error than its matched-rate anchor.

**BD-rate, measured four-window prefix only:** the three actual matte settings form each method curve; native tennis PCHIP log-rate integration over strictly overlapping quality ranges versus the achievable upper-concave anchor envelope. Positive means more bits at equal quality. This secondary curve summary does not alter the frozen fixed-scale18/24 decision or create eight missing complete-window results.

| Window | LPIPS BD-rate | DISTS BD-rate |
|---|---:|---:|
|alcaraz_hurkacz/031@129|+424.90%|+540.08%|
|medvedev_struff/014@194|+343.54%|+521.12%|
|alcaraz_hurkacz/013@213|+618.77%|+879.12%|
|medvedev_struff/037@349|+363.29%|+308.45%|

Common base shipped is primary. Charging its actual48,878,567-byte checkpoint over 7200 s adds 54.309519 kbps to every point. Per scale the measured prefix then has0 known wins, 4 bracketed losses and 4 unbracketed cells; the latter stay unresolved rather than extrapolated. No base-charged full-cohort claim is made. Actual rates and every comparison/dominance witness are in the JSON/CSV.

**Validation:** independent CPU reporting reopens original judge archives, requires their SHA256 identities in each new native job's inputs, verifies exact original judged-frame lists and every R/S hash, and compares the new exact-control score arrays directly with the original arrays (not just a producer-reported flag). Maximum absolute difference across 320 frames/LPIPS+DISTS is2.8014183e-5, below 1e-4. All four variants finite on all original judging frames; full-window means reproduced from saved arrays. Actual decoded selected MTQ1 packet hashes/dimensions/player identities match the CPU audit, and raw generator archives prove exact retained RGB coverage. Ada CUDA parameters asserted for both real networks, unchanged frozen backbones/main packed environment, peak 6.964 GiB within 24 GB. Plain convolution kernels already validated in fleet model–GPU table; no CPU metric fallback. Worker sources a8faea75/052df444, failed setup679f0f94; exact code-file hashes, model/input/archive hashes, native dispatch commands and immutable score arrays retained.

**Scope and limits:** this rejects the tested H5 area/bilinear resolutions and paid PNG-bank/actual-keypoint route at the frozen criterion. It does not reject all lossy masks, all mask coders, lossy appearance references, JPEG AI or inter-frame reference streams. No JPEG-conditioned generation ran; the prior approximately90% half-size-JPEG byte saving is still only reference-packet pricing. No fitting, generations, new segmentation/pose inference, cohort changes or quality-based per-window scale selection occurred. Native saved-crop clipping/fallback remains; pixels beyond saved generated rectangles receive the identical original canvas, never target foreground. Original exact-R target-derived Telea background is still granted, as is the earlier diagnostic's granted-matte history; this remains foreground accounting rather than a delivered end-to-end codec. Familiar-match/source-replay/pretraining uncertainty, exploratory 115 px and historical step 5 failure remain explicit.

**Stop condition met:** the conservative bound decides the frozen criterion after four windows; no further GPU work is proposed or dispatched in this task. Any reference-conditioned generations or redesigned mask route require a separately scoped question and budget.

Final immutable audit bundle `/home/itec/emanuele/Datasets/pointstream-data/tennis/k6b-matte-judge-20261025T141700Z/k6b-matte-gpu-evidence.tar.gz`, SHA256`201e78036a7c2bd2354ddee110155354b9eb1a3faf3632970f211e7c276e3bbc` (6385271 bytes,167 members); all166 payload-member checksums independently verified. Native score/command/input/model/control receipts, original judge archives, delivered reference/keypoint/matte packets, source snapshots, failed trace, local tests and closed ledger retained. Large source clips/generated RGB/model weights remain SHA256-referenced immutable inputs. Final JSON in the same directory `k6b-matte-gpu-final-report.json`, SHA256`55112f6b10254197231b379ee0408a9924c07fcc846fbc47c053d6ee938ea606`; comparison CSV and resource ledger alongside. Remote SHA256 values verified before write protection. No old artifact overwritten.20 focused tests and configured ruff/mypy pass; final full local suite871 passed/24 skipped, only unchanged macOS protected-path symlink assertion fails (Linux CI verifies supported platform).

GPU result published in PR #240 (result commit`7e7f0028`), merge-commit auto-merge enabled. Independent comparison/166-member checksum validation complete, all task jobs terminal, events acknowledged. Planning report includes the stopped prefix, rigorous16/24 optimistic bound and closed0.167674940/0.75 h ledger. No pending compute approval or follow-on run in this task.

### 2026-10-25 — Reference/alpha video and temporal matte CPU comparison

User explicitly requested color plus auxiliary alpha in one container versus
3-channel black segmentation, foreground black lifted to a nonzero guard band,
lightweight references, and an actual generated-quality experiment. The current
codec sends sparse full RGB references and dynamic per-target-frame masks via
RLE/Zstd or XOR/LZMA; it has not coded tennis masks as video. Reference-frame
alpha cannot replace changing target silhouettes. New scope/card:
[reference-video](../cards/k6b-player-reference-video.md), branch
`codex/k6b-reference-video`; old closed budgets are not reopened.

Hypothesis: motion compensation/alignment and reduced, temporally predicted
reference banks lower delivered side information enough to justify a real
changed-conditioning quality test. Alternative: alpha costs/black-key errors,
conditioning distortion or poor inter-reference correlation erase savings.
The saved exact-quality free-reference/free-matte result19/24 is a feasibility
oracle for reducing reference cost at fixed quality, not a bound on changed
conditioning or all lossy silhouettes; the17/24 PNG result cannot close those.
No new quality scores are inspected before this entry.

Freeze: same12 windows/anchors/R/S/judged frames/renderer/seeds and original
>=18/24 strict bracketed decision;7200s reference/adaptation lifetime; common
shipped base primary, actual base bytes separately. Actual CPU sparse-bank tests:
all24 references, half-size256x392, bicubic512x784 restore; SVT-AV1 preset8,
CRF20/35/50, full BT709 RGB/YUV420 conversion. Compare unmasked bank, segmented
color+explicit alpha (two actual AV1 tracks, Matroska container with counted
metadata), and black RGB with floors1/8/32, thresholds ceil(floor/2). Ordering
by unit ID within each match is recorded and charged; no claim of source
chronology/player identity. Saved encoder SAM at recorded reference tick0,
native reference-box crop supplies alpha; no new segmenter or manual labels.
Half-size JPEG70/PNG pricing controls come from the previous immutable audit.
Record actual post-container track decodes, frame/ID/hash fidelity, foreground
RGB MSE, silhouette IoU/FP/FN, all delivered bytes. Pixel MSE is only screening,
not a learned-judge/generation surrogate. No JPEG AI implementation is introduced.

Dynamic-mask pilot: Al031/Med014,240frames and both players each, scale8;
binary/SDF video at the same3 CRFs and reversible bounding-box-center alignment
with circular roll before XOR/LZMA (all shift/header bytes included). Compare
mask IoU and boundary F at2 full pixels beside unaligned H5. All methods/anchors
have identical windows; no quality-driven cohort replacement. Lossy candidates
require actual composite quality before any rate-quality win claim.

CPU authorization interpreted from "go ahead": new cap7200CPU-s including
children,0GPU-h; smoke<=300s then full<=1200s, validator<=60s, overall supervised
<=1680s on4 CPUs (conservative6720CPU-s plus480local/analysis allowance). Save
each setting. Smoke only first match bank/Al031 atCRF35, full both banks/both
mask windows/3CRFs. Validator checks complete real decodes/container tracks,
source/packet identities, finite diagnostics, exact aligned-mask reversibility,
and synthetic pure-black foreground guard-band contract. Inspect old terminal
job20261025T141239Z-93456964 before dispatch and all six hosts; CPU admission
requires no GPU. No new chats/agents/monitors. Generation is user-requested but
requires a concrete new GPU cap after CPU screening, not unused old allowance.

Stop on invalid decode/contract, cap exhaustion or completed bounded pilot.
Keep byte/fidelity Pareto candidates; choose one fixed recipe before opening new
generated quality. Report uncertain tradeoffs instead of declaring a conclusive
close. Familiar two-match/source-replay limitations, exploratory115px, historical
step5 failure, exact-matte history and granted target-derived background remain.

CPU job submitted `20261025T163506Z-af342506`, source`b3d39171`, spec
SHA256`a76e748dd74ca27f27d6fd5c20a137680e56cfa8a67caed0262bdb8273d7ac99`.
Deadline2026-10-25T17:18:03.043086+00:00,1680s supervisor cap,4CPU threads,
0GPU. Native fleet doctor passes; gpu1/2/3/5/6 verified reachable with CPU
headroom, gpu4 refused SSH and excluded. Old final job inspected complete.
Frozen packed main environment and previous CPU packet archive SHA256
6865035e3132b6e07600fdaabc8c9d7152b8b6e3c3260d9f3df21cab33ca33c7.
No completion-resume capability verified for this client; use one attended
bounded native wait for this batch, no recurring or one-shot model monitor.
Preserved spec/submit receipts locally `/tmp/k6b-refvideo-{spec,submit}.json`;
output native `/jobs/fleet/inbox/20261025T163506Z-af342506/{smoke,full}/published.tar`.
Three focused tests/configured ruff/mypy pass before dispatch. Inspect this ID
before acting after interruption; never replay an uncertain submission.

CPU job20261025T163506Z-af342506 failed at smoke validator: native fleet
releases scratch before invoking it. Codec smoke itself completed9 rows with
valid actual container track decodes,59.019865CPU-s; all artifacts preserved
(smoke TAR4cc81a7466e3137fd833c54aaec92fa0f7738456aa61798cfab3613f3a6b56fe).
The defect was validator artifact access, not codec output. Corrected validation
extracts the immutable published TAR and independently decodes each delivered
track, checks original native-decoded hashes and saved conditioning hashes,
plus exact aligned packets and frozen packet input SHA. Both full and smoke
now run these checks before publication as well. Original encoding path/settings
unchanged. Failed admitted interval70.207587s charged conservatively as280.830348
CPU-s. Retry is diagnosed, not replay of uncertain work; new cap1500s at4CPUs,
plus failed280.831 and480local allowance =6760.831CPU-s <7200.
Full local suite872passed/25skipped; two platform constraints: macOS protected
symlink assertion and sandbox denial of ps process-state inspection. Focused
contracts/configured ruff/mypy pass. No GPU used.

Corrected CPU job submitted `20261025T163852Z-1d6ee39a`, source`b257ee03`,1500s cap
inside unchanged7200CPU-s total. Both stages independently validate actual
published codec evidence. No new GPU authority is implied.
Retry spec SHA256`e51b1a762307ea18f79bf6c461a7c332a051beef0d59420c73486d96ee5240d5`; native deadline remains17:18:03UTC.

Before dense carrier measurements, clarify user scope: also compare continuous
foreground appearance, not only sparse reference banks. Dense CPU pilot is
preregistered in the same card (Al031/Med014,240 identical original frames,
960x540 area RGB, exact saved SAM union;3CRFs20/35/50;actual color+alpha container
versus black RGB floors1/32;native source conversion, same decode/restore rule).
No generator/keypoint/adapter cost applies to this separate foreground-video
representation; all container bytes are paid over each window, common background
still granted. Initial8frame smoke, full240; actual CPU reservation must fit the
remaining unchanged7200CPU-s total after prior attempts. No new quality-based
window/parameter choices. CPU FP/FN/IoU/MSE are not neural-judge evidence.

Corrected sparse CPU job20261025T163852Z-1d6ee39a complete; full54 settings,
4 exact alignment packets, substantive smoke/full checks passed. Native admitted
191.603102s plus failed70.207587s => conservative1047.242756CPU-s so far.
Dense reservation1200s x4CPUs +480local allowance gives total6327.242756CPU-s,
inside unchanged7200. Native packed environment unchanged; dense sourceab54b032,
smoke180s,full850s,validator60s,overall1200s. Before dispatch verified previous
job terminal and re-inspected complete eligible CPU pool. Dense source native
conversion/frame hashes and actual post-container video decodes are required.
Sparse full TAR SHA2562e80a76d8e7ba1bc5bd6dcdadf6892cf465a5568c1377fa06f2760822f424e1d;
no new neural quality result. No GPU budget has been spent or reopened.

Dense CPU job `20261025T164351Z-3d27fbbf` submitted, source`ab54b032`, specSHA256`fd8534155bac7bc26c084d9a4563698a880a9bc2936f5613155b390b87ecb8ea`.
1200s total supervisor cap,4threads,0GPU; native deadline2026-10-25T17:18:29.510627+00:00.
Record immutable native source+packet hashes, original conversion/frame identity,
delivered container RGB/alpha splits; no uncertain replay after interruptions.

### Sparse/reference CPU outcome (dense carrier still in progress)

Complete job20261025T163852Z-1d6ee39a, codeb257ee03;54 points =30 reference-bank
settings plus24 dynamic-mask settings,4 reversible alignment packets. Both
native stages independently check actual delivered container decodes and
conditioning hashes; no new neural metric/generation. Independent local88
packet, alignment and saved RGB/alpha conditioning checks pass. CPU report can
be reproduced with `python -m experiments.tennis.k6b_reference_report --archive
PUBLISHED_TAR --diagnostic ORIGINAL_DIAGNOSTIC --transport ORIGINAL_TRANSPORT
--out REPORT` (paths as native receipts).

Reference bank bytes (all12 references per match,256x392 restored512x784,
actual container/index/restore metadata paid, preset8):

| Recipe | CRF | Alcaraz bytes | Medvedev bytes | reference-mask IoU Al/Med | foreground RGB MSE Al/Med |
|---|---:|---:|---:|---|---|
| unmasked RGB |20|53793|75843|not applicable|25.52 /33.44|
| unmasked RGB |35|28054|39040|not applicable|44.21 /53.85|
| unmasked RGB |50|14149|18291|not applicable|100.14 /123.97|
| color+alpha |20|89838|95327|.9865 /.9868|35.74 /51.67|
| color+alpha |35|50905|54892|.9865 /.9868|58.31 /77.47|
| color+alpha |50|28210|30393|.9865 /.9867|137.45 /170.58|
| black floor1 |20|61608|66634|.8386 /.8452|35.76 /51.57|
| black floor1 |35|32572|36317|.7690 /.7632|58.46 /77.33|
| black floor1 |50|16887|18743|.5933 /.6058|137.29 /174.56|
| black floor32 |20|61458|66513|.9833 /.9806|51.38 /62.81|
| black floor32 |35|32077|35977|.9783 /.9761|73.81 /88.57|
| black floor32 |50|16670|18660|.9538 /.9540|150.23 /183.25|

Intermediate floor8 has IoU.9466/.9496 atCRF20,.9149/.9185 at35,.8184/.8399 at50;
its banks are essentially the same size asfloor1. Raising all foreground RGB
channels to1 disambiguates pure black *before coding*, but measured lossy output
creates substantial false positives;32 improves the guard band at the expense
of dark foreground colors. Explicit alpha costs more but avoids that clipping.
These are pooled reference-pixel diagnostics, not temporal matte/generation
quality. Unmasked references need no alpha; their all-one comparison mask is
not an error claim. Native post-container track pixels equal dav1d output.

Original PNG banks895487/1179150B; half-JPEG70 prior92631/117396B. The new CRF20
unmasked bank is94.0%/93.6% belowPNG; CRF35 is96.9%/96.7% belowPNG. All15 new
bank recipes give19/24 strict cells at *unchanged original generated quality*
with actual KPS/adaptation,7200s bank lifetime and free dynamic mattes. This is
only a sensitivity; actual changed-conditioning quality is untested. Sparse
bank order is unitID, not proven chronology or same-player tracking. This does
not isolate the causal gain of inter prediction from the codec/resize change;
no all-intra ablation or JPEG AI implementation was tested.

Dynamic masks: atscale8, all tested binary/SDF AV1 points cost more than the
existing H5 route. Al031 two-player payload totals (including packet metadata):
binary CRF20/35/50=164123/90565/37031B; SDF=71717/37283/18110B. Med014:
binary200564/87028/25630B; SDF54141/26143/13476B. Plain separate H5 streams sum
16414/10658B before shared timing metadata, substantially below video. The
cheapest SDF is also much less accurate: Al IoU.6638/.7382 and boundary F.5801/.6171;
Med.7396/.6257,F.6328/.6076 versus respective H5 AlIoU.8466/.8796,F.9325/.9326,
MedIoU.8693/.8136,F.9295/.9336. These mask diagnostics are on960x540 native grid;
boundary tolerance1native pixel=2 full pixels. No matched-accuracy video saving.
Exact translation alignment also loses: Al7412+10216 vs6451+9963B; Med7209+5216
vs6335+4323B. Alignment packets include shifts/headers; no claim that all
pose-aware predictors or residual coders have been exhausted.

Candidate freeze before new generated quality:half-size CRF20 preset8, one
fixed recipe each: unmasked bank, explicit-alpha segmented bank, blackfloor32
segmented bank. CRF20 is highest tested image fidelity and all its bank rates
already fit the optimistic free-matte headroom. Segmented client conditioning
zeros outside recovered full-size alpha after restoration. No tuning against
new generated scores, no per-window recipe selection. Native original-reference
control required. Detailed proposed new GPU budget/stages in card:0.5h dense
judging plus4.0h two-window actual-generation pilot, combined4.5h maximum,
estimate3.35h. This request is not authorized by the older closed ledgers.

Platform verification: the isolated process-state test passes with read-only ps
allowed, confirming its earlier failure was the sandbox restriction. The old
macOS protected-path assertion remains the only unverified local platform check;
Linux CI remains required. Sparse banks are preshipped/offline in unitID order,
with no chronological/live reference-availability claim. No BD-rate is inferred
from single-point reference recipes;3point dense curves use true overlap only.

Before affine guard-band outcomes: user invited a better way to distinguish
foreground black. Add one bounded CPU refinement, still inside7200CPU-s:
foreground round(32+223*RGB/255), background0; client detects max(channel)>=16,
then inverse round(clip((channel-32)*255/223)). This avoids the floor32 clamp's
dark-color clipping; precodec inverse rounding error<=1 is explicitly tested.
Same24 refs,halfsize,order,3CRFs20/35/50,preset8,actual container bytes and decoded
RGB/alpha checks; smokeAl bankCRF35,fullbothbanks. Reserve<=200supervised s at4CPU
(max800CPU-s), after dense ledger inspection. Even the previously reserved
worst case6327.242756+800=7127.242756CPU-s including480local fits7200.
Select affine rather than clamp as the single black recipe for future generation
only if its CRF20 foreground RGB MSE improves on both matches and mask IoU is
no more than.002 worse on either. Otherwise keep clamp. This refines CPU
screening before any new generated scores; GPU request does not expand beyond
three reference recipes. Sparse16-level threshold remains frozen. No new
source/quality-driven window selection, no GPU or new learned models.

Dense CPU attempt20261025T164351Z-3d27fbbf reached850s full-stage deadline;
not a successful full matrix. Seven completed Al031 settings preserved in
partial TAR SHA25645be9e1a39030246235fec3335b23b7ca798f55ead579639539893656a23996e.
Short smoke39.926484s did not establish the assumed full-window runtime; the
promotion's extrapolation was inadequate. No replay. Completed points are
all240frames: color+alpha CRF20/35/50=784.783/417.661/176.287kbps; blackfloor1
CRF20/35=412.312/197.941; blackfloor32=408.105/196.148. These are all above the
Al031 highest measured anchor57.705kbps, so do not blindly finish the same
high-rate matrix. CPU-only MSE/IoU remains diagnostic; no learned quality win.

Preregister targeted operating-point refinement before its scores: same two
240frame windows,960x540,preset8, color+alpha and robust blackfloor32 only,
CRF63 (the existing anchor's low-rate end). This adds one justified lower-rate
point rather than completing expensive irrelevant high-rate settings. Smoke8
frames,full240,actual per-point projection required before new promotion. New
full cap600s,smoke60s,validator30s,overall750s x4CPUs=3000CPU-s. Preserve all
old settings; no genotype/seed/window substitution. This is not a full3point
curve for Med014; report only measured coverage/overlap.

Refined failure-inclusive CPU bound uses recorded CPU rather than assuming all
four CPUs busy throughout: last full checkpoint900.022310CPU-s,wall818.758641s;
charge unfinished tail4*(850-818.758641)=124.965436CPU-s. Smoke native47.071113;
validator4*2.771559=11.086235. Charge all other admitted overhead at4CPUs:
4*(924.007124-850-39.926484-2.771559)=125.236327. Dense bound1208.381421CPU-s.
Together with earlier conservative1047.242756,480local,800affine reserve and
3000refinement reserve =>6535.624177CPU-s < unchanged7200CPU-s. The earlier
1800s aggregate wall-reservation shortcut is replaced by this auditable CPU
bound; each job remains<=45min and no GPU/paid API is used. Old CPU results
and GPU ledgers are immutable. If any counter/budget check fails, stop.

Refinement cost gate uses the real failed full-size pilot:818.758641/7 x4
settings +150s setup/variance =617.862081s <660s full cap, unlike the earlier
short-smoke-only assumption. Smoke must also finish under55s. Overall750s
reservation unchanged. Foreground MSE now selects foreground pixels *before*
float conversion/squaring, avoiding three needless full-frame float arrays;
identical per-pixel squared errors and summation order, no metric definition
change. Record each point's actual wall time. New stage adds only CRF63 for
alpha/floor32; old high-rate partial points remain immutable and reported.

dense-lowrate CPU job `20261025T170625Z-c041d6b7` submitted, source`5469e9c9`, specSHA256`7a192defcd87d25a6374f7ff75ae51482387de94f0859443515275b9dfbd7609`.
Caps750s(dense low-rate)/200s(affine),4CPU threads,0GPU; inside unchanged7200CPU-s.

affine CPU job `20261025T170628Z-2ae1c4f3` submitted, source`5469e9c9`, specSHA256`10a337b3d2c800f4d27f86351663b3f908d5f4da086fdab3cc93ef10ba8e64ac`.
Caps750s(dense low-rate)/200s(affine),4CPU threads,0GPU; inside unchanged7200CPU-s.

Client format clarification: actual Matroska files carry independently decoded
AV1 color and named alpha video tracks with paid custom layout/restore metadata;
not an asserted standardized RGBA decoder. Dense alpha is the union of both
visible players. Proposed quality judging reports full-frame/outside-original-R
SSE and FP/FN alongside unchanged fixed-S LPIPS/DISTS, retaining background grant.

Affine CPU job20261025T170628Z-2ae1c4f3 complete,6full bank points plus1smoke;
native cumulative CPU62.621371s before final validation/overhead,0GPU. Native
gate and post-container hashes pass, precodec inverse maximum RGB error1.
Full TAR SHA2560afc0fa2bed6bb3c2825e877751dbaa52687814d5861434b5e2354794dd3ea26.
CRF20 affine bank Al61400B,IoU.98404,foreground MSE38.988; Med64795B,IoU.98183,
MSE55.024. Clamp32 was61458/66513B,IoU.9833/.9806,MSE51.38/62.81. Thus affine
passes both predeclared selection conditions and becomes the one black-key
reference recipe before any generated quality. AtCRF35:32227/35327B,IoU.98006/
.97798,MSE66.939/87.370. AtCRF50:17168/18897B,IoU.95780/.95897,MSE160.044/
198.157; inverse mapping amplifies heavy quantization, so do not claim universal
superiority over clamp. Threshold16 is applied in the encoded lifted domain,
then RGB is unlifted and restored, preserving a recovered alpha for black shoes.
Only reference arrays are materialized for future generator input; all original
motion/timing/seed arrays remain exactly hash-equivalent. Old clamp artifacts
are preserved, not overwritten. GPU recipe count and proposed4.5h cap unchanged.

### CPU phase closed — 2026-10-25

Low-rate job20261025T170625Z-c041d6b7 complete; native smoke/full validity gates
pass. Full wall581.031201s < measured617.862081s projection and660s cap.
Four complete240frame points, actual container bytes and decoded track/source
identities, no neural quality:

| Window | CRF63 carrier | whole-window kbps | mask IoU | foreground RGB MSE |
|---|---|---:|---:|---:|
|Al031|color+alpha|45.7717|.94045|1765.662|
|Al031|black clamp32|18.8675|.73465|1752.227|
|Med014|color+alpha|71.0517|.93961|1925.201|
|Med014|black clamp32|31.7100|.79818|1939.018|

These rates enter each window's actual anchor bracket (max57.705/109.089kbps),
so rate alone cannot close the dense route. Mask/color degradation is substantial;
LPIPS/DISTS on identical original R/S plus outside-R diagnostics is required.
No full-cohort success, equal-quality saving or generated-quality preservation
is inferred. All11 completed dense points (7 earlier Al +4 new) remain available;
other original high-rate matrix points were never completed and stay missing.
Al alpha has4 coded qualities, black32 has3; Med has onlyCRF63, not a curve.

Independent local checks:88 sparse packet/decoded-conditioning/alignment checks
plus43 new affine/dense/prepared/source checks, all pass. Original full-source
RGB frame hashes and half-size encoder-input RGB agree exactly across old/new
CPU revisions for all Al points, confirming the MSE allocation optimization did
not change source pixels. Every saved dense point has240 actual decoded frames.
Source clips and prepared inputs remain frozen. Native full low-rate TAR SHA256
1f9cd10f4e95c0446ae2c4103b2cbbd6b9b65e1bf71697e83fc36fd80b5bfeee.

Selected generation inputs: unmasked/explicit-alpha CRF20 from sparse video and
affine32 CRF20 from affine video, allhalf-size. Six units each in original
first-group prepared archive; only reference arrays changed, all other arrays
hash-identical. New affine prepared TAR SHA256
300a8888827af98e423648c5ba241f9a1c7f40518330e1803495f7fd89b65651.
Original and clamp variants preserved; no model call or fitting occurred.

CPU ledger closed: conservative total5708.866274CPU-s of7200, including480local
allowance and failed-stage unfinished-tail/overhead bounds;0 newGPU-h. The five
native jobs are terminal:163506-af342506 failed validator;163852-1d6ee39a complete;
164351-3d27fbbf failed full deadline with7points;170625-c041d6b7 complete low-rate;
170628-2ae1c4f3 complete affine (all20261025T prefixes as above). No recurring
monitors, new chats or agents. Full final local suite874passed/24skipped with
only unchanged macOS protected-path failure;24focused tests/configured ruff/mypy
pass after final changes. Linux CI required for PR. CPU phase reached its bounded
screening stop; broader route remains open.

Next proposed budget remains4.5newGPU-h (estimated3.35):0.5 dense native quality
judging and4.0 actual reference-conditioned two-window generation/control pilot.
The dense substage may add two CRF63 affine32 carrier points (one per pilot
window) inside its unchanged0.5h cap to transfer the tested reference guard-band
idea to dense video; record their actual encoded input bytes/quality, never
borrow clamp rates. Prioritize both matches atCRF63; score saved higher points
within allowance. Reference substage tests exactly3fixed recipes plus original
controls with exact dynamic matte granted; paid-matte/end-to-end claims remain
separate and limited. Existing closed GPU budgets grant no new spending. Await
explicit new budget approval; no GPU submission, generation or Wan-Animate.

Important scope of compression search: these CPU video points used fast preset8.
They are not proof of the best attainable SVT compression. Proposed approved-only
continuation includes a separate<=3600CPU-s preset4/preset1 preflight (tennis anchors actually use4 in k6.PRESET;1 is a
slower comparison) before interpreting a negative quality result; compare
actual bytes/decoded fidelity on fixed inputs and freeze the global preset
before new neural scores. Three recipe families, GPU cap4.5h unchanged. If that
preflight cannot fit, retain the tested-config limitation. No JPEG AI or learned
mask predictor was tested. Current7200CPU-s ledger is closed and not reopened.

CPU evidence published and remotely SHA256-verified in new read-only directory
`/home/itec/emanuele/Datasets/pointstream-data/tennis/k6b-reference-video-20261025T171900Z/`.
Bundle `k6b-reference-video-evidence.tar.gz`,141164474bytes,72members,
SHA256`6ba5e8ce6ce5cb454d0f07ff68ab5a1f945f1ad04c2a85462b263d76fd3df827`;
70payload hashes independently verified. Final report SHA256
`b5cc10c0d40bafa4f04ac548af91ee6c37cd983f472dbc7d02c84a23ec9d43bc`;
resource ledger SHA256`941d6a57859e403ca7354d0e0829fc5adc18df73e2120907dc9271a622975208`.
Bundle preserves native raw publications, full partial settings, code snapshots,
source/command/environment receipts, original prepared inputs, materialized
reference variants and reproducible report helpers. Original clips/weights are
SHA-referenced, not duplicated. Task terminal events acknowledged, no monitors.
The bundle captures completed CPU outcome; subsequent proposed preset preflight
clarifications are in the current card/repository, not retroactive measurements.

Publication push initially encountered an automatic approval-review usage-limit
failure; the command was not executed and was not judged unsafe. After the user
asked to continue, the same normal review path succeeded. No approval bypass.

CPU code/report published as PR#241 (https://github.com/emanuele-artioli/PointStream/pull/241).
Native evidence is remotely verified/read-only; local checks above pass with
known platform exception. Linux CI pending; enable merge-commit auto-merge per
AGENTS. New4.5GPU-h plus<=3600CPU-s preflight budget remains unapproved; no
GPU submission/generation. This publication does not grant compute authority.


## Approved two-window compression-quality pilot (2026-10-25)

User approved <=4.5 NEW GPU-h plus <=3600CPU-s slower-preset preflight.
Hypothesis: actual compressed reference conditioning can retain sufficient
generated quality to pass >=3/4 strictly bracketed cells; competing explanation
is that compression/segmentation damages identity or motion. Dense foreground
is tested separately with all delivered appearance/alpha container bits paid.
Card: docs/cards/k6b-player-reference-video.md, approved continuation.
Old ledgers closed. NewCPU cap3600s, GPUdense1800s and GPUreference14400s.
Freeze sameAl031/Med014 windows, R/S, seed/context/steps/adapters and exact-matte
reference oracle before neural outcomes. Preset4/1 native bank comparison
precedes quality; selection by global bytes/fidelity Pareto with preset4 fallback.
No full12-window confirmation, fitting, Wan-Animate, source-independence or
ungranted-background claim. New jobs recorded on submission, never replayed.
Prior terminal affine job20261025T170628Z-2ae1c4f3 inspected complete before
new dispatch; all6hosts inspected. New branch codex/k6b-reference-quality-pilot.
CPUreservation800supervisedseconds x4 +400local=3600CPU-s maximum.

CPU preset job `20261025T183121Z-18d2a212` submitted; source`7c8187b3`, specSHA256`0baa80a986dbe1410e7b18c892887390308dba587317ffd8a49789b66fac889c`. Deadline in /tmp/k6b-preset-quality-spec.json;800s x4CPU bound,0GPU. One attended native bounded wait, no monitors.

Operational bound refinement before dispatch: original complete PNG-control smoke
was78.946s, so the proposed60s bound would truncate valid initialization. Use
<=110s smoke,<=1800s full,20s validator,2000s total for each initial control;
all within45min and unchanged14400s reference subledger. Later reservations
use actual control cost before admission. 14,000MiB shared cap +4096MiB margin
is supported by original12.21GiB measured peak and revalidated by native smoke.
All Ada GPUs busy; gpu5 has18777MiB free on both, gpu6 has19971MiB on one,
so full compatible gpu5/6 pool is retained. No latency claim. Full controls
may run while presetCPU executes; no new neural quality scores opened.

PNG-control Al031 job `20261025T183259Z-16af35e8` submitted sourcee4b7a3a9 specSHA256`9e4e86be961d104ebc2b3806ea6beeb5a0aa351cd68fb2546998edbbb37e7288`; <=2000GPU-s reservation,14,000MiB, bothAda hosts. No candidate rendering yet.

PNG-control Med014 job `20261025T183318Z-34d149af` submitted sourcee4b7a3a9 specSHA256`d6a7fb8fffadfd827091fdcb01939542f3253746903a81fdb53e5c064dc3166a`; <=2000GPU-s reservation. Combined controls reserve4000/14400referenceGPU-s; no candidate submission.

CPU preset jobcomplete gpu3; full12rows, all native checks pass. ArchiveSHA256
984f9b256e69c9b13643d3381b7a5e4b95fba3f641f22f41c510dc90839d93d1.
Measured native executionCPU smoke12.535246/full59.181003s; conservative
admitted4CPU bound320.786816s. Full programwall31.416s. Preset1 dominates
preset4 AND prior8 on allthree families/bothmatches: fewer bytes, lowerFGMSE,
segmentedIoU nondecreasing. Freeze preset1/CRF20/halfsize before quality.
BankbytesAl/Med full48164/68193; alpha54934/58797; affine43870/47493.
Decoded prepared variants materialized for original6units; only reference arrays
change, all motion/geometry/timing/seed identity retained. NewSHA full03953043e4b4d3d4e4f70591afbc462a385d9347862985e43117cfea3b404547,
alpha13230a0620906ae8f0dc834c76860268411bf7d3c29ba0ed1444141a9723db41,
affine3fbcc34e8b174ee139981526e3d07afe8d52d3cfd06a7c124304cc98e62b0e7e.
RemainingCPU permits representative8frame dense preset4/1 alpha+affine checks,
430s supervised x4=1720CPU-s reservation; totalbound320.786816+1720+400local
=2440.786816 <3600. These shortGOPs cannot establish full240frame rates.

Dense8frame presetCPU job `20261025T183643Z-dfd14984` submitted source65dc374d specSHA256`669220c35ec5a28aa82519099f518a9d755d4912845b1d0c0d4eab1127dfbe6b`;430s4CPU reservation, noGPU. Bothpreset4/1andbothwindows, alpha/affine atCRF63.

MedPNGcontrol20261025T183318Z-34d149af terminalfailed, nofullstarted.
Confirmed native reason: commandexceeded110s smoke allowance after both8frame
player samples. No OOM/data/configfailure observed. Charge entireadmitted
131.172530651GPU-s; preservepartialsmoke. Correctedattempt160s smoke/1750sfull,
20svalidator/2000stotal, sameinputs/renderer/memory. Totalreferencefuture
reservation failed131.173+Al2000+Medretry2000+6x1450+600judge=13431.173s
<14400; actual costs checked before promotion. Not an uncertain replay.
Dense8frame CPUjob20261025T183643Z-dfd14984complete: preset1/4tradeoffs
(Alalpha1slightlyworseMSE; Medaffine1slightlylargerpacket). Freezeglobaldense
preset4 fallback BEFORE neural scores; short8frameGOP not full240rateevidence.
CPUadmittedbound200.560948s, combinedwithprior320.786816+400local=921.347764/3600CPU-s.

Remaining newCPU supports full dense preset4 scale confirmation before quality:
4full240frame alpha/affine32CRF63points, sameAl031/Med014. Actual8frame
preset4wall8.501167s x30+150margin=405.035s <520fullcap. Newnative
smoke must re-establish this projection.30ssmoke/520sfull/20svalidator,
580supervisedseconds x4=2320CPU-s reservation; combined conservative
CPUbound320.786816+200.560949+400local+2320=3241.347765<3600.
Saveeverypoint; ifcapfails preservecompletedpoints and state missingcurves,
no blind retry. This completes selecteddensecarrier codec prep onCPU rather
than reserving aGPU for conventional encoding, within approvedsameCPUcap.

control-med-retry job `20261025T184001Z-2e64d93e` source10556523 specSHA256`df6b0e38ad89aff4d8591a0e5552c72fb56ab7e5e5267c20c16581973423f9aa`; budgets as recorded above. Inspect this ID after interruptions, never replay unknown.

dense-full job `20261025T184009Z-0787c1cf` source10556523 specSHA256`1a12a4a194823baa23e6239f9f196e6341961531aa7cacb7fcfb52d29edb1e28`; budgets as recorded above. Inspect this ID after interruptions, never replay unknown.

Full dense preset4job20261025T184009Z-0787c1cf COMPLETE, all4points240frames.
ArchiveSHA2565d27c429c92e437134f068a440eba0e9b5be5cc54941b2cf24d8fa33f2a1f32b.
Programwall173.972397s, processCPU290.636557s; conservativeadmitted882.829598CPU-s.
ActualratesAlalpha42.791667/affine18.994167kbps; Medalpha68.566667/affine31.666667.
No newquality scores opened. CPUboundsofar320.786816+200.560949+882.829598
+400local=1804.177363CPU-s leaves1795.822637. Use<=430superviseds4CPU
(1720CPU-s) for fullpreset1dense confirmation, same4points, nofamily/window
expansion. Measuredfullpreset4wall x measured8framepreset1/4ratio +75smargin
=306.1s<350fullcap;20ssmoke recheck,20svalidator. Totalreservation3524.177363
<3600CPU-s. Densepreset4fallbackremainsunlessfullpreset1dominatesglobal
bytes/IoU/FGerror; choosebeforeANYnewdensequalityscores. This resolves
shortGOPuncertainty with available approvedCPU, withoutnewGPUvariants.

Finaldensepreset1 CPUjob `20261025T184630Z-ace7057c` sourceecf496b1 specSHA256`d619552b0480fa40acec137b89721208cd02d8161b5f048f5fcbe358dc75ee92`;430s4CPU reservation; remainingcap as above.

FinalCPU preset1densejob20261025T184630Z-ace7057c COMPLETE,4full240framepoints;
archiveSHA25667d6cbfa616d3ef3acda140c25154c8633667b9e812c37e69583167f97781bd6.
Affinepreset1 dominatespreset4 bothmatches: Al22464B/18.72kbps/IoU.809915/MSE1855.657;
Med18858B/31.43kbps/IoU.840381/MSE1994.777. Alpha1larger with better fidelity:
Al53354B/44.461667kbps/IoU.963230/MSE1559.672; Med42747B/71.245kbps/IoU.960539/MSE1697.876.
FreezeONEglobalpreset per densefamily before newquality: alpha4 byfallback,
affine1 bydominance. Referenceall3families remainpreset1CRF20. No extraGPU
variant for the CPU Pareto alternatives; oldpreset8savedcarriers remainhistory.
NewCPUledger COMPLETE conservative bound2647.198625565/3600CPU-s incl400local;0GPU inCPUphase.

DenseGPU preregistration: firsttwo<=600s jobs, onefullwindoweach, selected
alpha4/affine1 atCRF63 beside savedalpha8/clamp32preset8CRF63. Fourdelivered
carriers+oldgeneratorcontrol perwindow, alloriginaljudgedindices. Smoke1judged
frame but all240packetdecodes; validatefullcoverage/hash/source/R/S,control
within1e-4, actualCUDAnets and12000MiBcap.60ssmoke/500sfull/20svalidator
inside600total.<=600sleft for7historicalhighrateAlpoints ifbudget/runtimeallow.
No additionalreferencefamily/presetvariant. Allselecteddense/refpresetsfrozen
beforeANYnewquality. Currentreferencecontrols stillrunning, no candidate
submission until fullcontrol cost/validity checked.

DenseactualqualityAl job `20261025T185239Z-1df3ec6f` sourcee182ad15 specSHA256`0fec287d438057d72f0b7ec94fa929f1d02a74b0f88d7ff33259eb9d4ba8b5cf`;600GPU-s reserved/1800densecap. Nativepoolgpu5/6,12000MiB, firstactualnewqualitystage; allpresetchoices frozen above.

DenseAlquality20261025T185239Z-1df3ec6f FAILED correctnesssmoke beforeany
qualityvalues, nofullstarted. Native27.006871s execution/44.123715s admitted
charged to1800denseGPU-s. NativecontainerYUVhashes passed; combinedRGB/source
pixelhash gatefailed. Diagnose separately instead of weakening packet/pixel
hashes. CarrierCPU originalconversion was torchCPU; originalmetrictruth uses
torchCUDA. Corrected gate computes exactCPUsource independently fromsameYUV,
retains originalnativeCUDAtruth for metrics/control, records bothhashes and
channel/maxdifference, requires<=1levelCPUvsCUDAroundoff. DecodedcarrierRGB
andalpha still require EXACT originalCPUpixelhashes. Ifthese fail, logeach
check and stop; no unproven continuation. Existingfulloriginalcontrolwithin1e-4
remainsunchanged. Correctedjob mustprove thisdiagnosis in native smoke.
Densefuture reservation44.123715+600Alretry+600Med+550historical=1794.123715
<1800GPU-s; nobudgetincrease or oldledgerreuse.

CorrectedDenseAlquality job `20261025T185724Z-884dd4ce` source13b2c9ec specSHA256`0fec287d438057d72f0b7ec94fa929f1d02a74b0f88d7ff33259eb9d4ba8b5cf`;600GPU-sreservation afterterminalfailed44.123715s. ExactdecodedRGB/alpha hashes not relaxed; native source CPU/CUDA distinction explicit.

CorrectedDenseAlsmoke20261025T185724Z-884dd4ce PASSED allchecks. Source
CPUvsCUDA diagnosisconfirmed: only1channel differs by1level on samplet0;
allactualdecodedRGB/alphahashes EXACTmatchCPUpackets. Originalneuralcontrol
maxabsdiff0.0, CUDApeak5.703743GiB, realnative smoke37.938989s. Full
Alstage running. PromoteMed throughsamevalidator, remaining600sreservation.

DenseactualqualityMed job `20261025T190034Z-01c74a65` source7a364580 specSHA256`5f34e27b416ba2251492ff10bafb18cb391969925d59e6ef8800eac4afbd1cd2`;600GPU-sreserved inside1800densecap. Samefrozen4deliveredcarrierpoints andnativeoriginalcontrol.

Operational configuration for remainingreference renders: exclusiveadmission
on unchangedfullcompatibleAda poolgpu5/6, original24000MiB allocation.
Sharedcontrols materially slowed while acompatibleGPU becameidle ininventory;
exclusiveadmission avoids wasting fixedbudgetoncontention. Nohostpin or
changedrenderer/quality recipe. Originalexclusivefull-window evidence
1232.62/1247.34s provides projection; everynewjobstillhasactualsmoke.
Futurecandidate spec160ssmoke/1600sfull/20svalidator,1800stotal; reserve
only affordableparts from14400referenceGPU-s afteractualcontrolcharges.

Alactualquality20261025T185724Z-884dd4ce COMPLETE,80originaljudgedframes,
allnativevalidation passes,252.160562GPU-s admitted. ArchiveSHA256
377641e79cc06d4bd81c03d1a732817316ed604d4eba9304a2293c86391dc50c.
Eachselected/historicalCRF63carrier0/2strictcells,resolved. Alpha4at42.791667kbps
LPIPS.359325093/DISTS.270400329; affine1at18.72kbps .445457180/.318546036.
Historicalalpha8at45.771667 .363348240/.274332323; clamp32p8at18.8675
.472553771/.318922050. Cheaperachievableanchors dominateeachmetric.
CurrentdenseGPUactual44.123715failed+252.160562complete; Medreserve600,
historical7pointAlreserve550 =>1446.284277s<1800. Original5variantAlcost
252.161 incl38ssmoke; full8variant historicalprojection214*8/5+40<383s,
under460fullcap;60ssmoke/20svalidator inside550s. Preserve distinctpreset8
curves and selectedsinglepoints, never combinepreset1/4into oldcurves.

Historicalstage submission20261025T190533Z-524c8807 NOTPUBLISHED,0GPU.
Nativeverify_inputs rejected missingfull/published.tar; originalfailedCPU
correctly retainedfull/partial.tar. Statusreportsunpublished; independent
SSHinspection confirmedrequest.json ABSENT, directoryonlysource/events, so
no worker canadmit it. Preserveunpublishedsource; no uncertainreplay. Correct
only artifactpath tofull/partial.tar, remotelyverifiedSHA256matches
45be9e1a39030246235fec3335b23b7ca798f55ead579639539893656a23996e.
Subsequentcorrectedsubmission is newlypublishedrequest, not replayofrunningjob.

CorrectedHistorical7pointDenseAl job `20261025T190813Z-810bcaba` source2f91b731 specSHA256`9155b47c1e41ceae3942df7a991ae27d96b6718c49c10929a92785ad49e5f72b`;550GPU-sreserve, correctpartial.tar inputverified; unpublished524c8807neverrun.

AlPNGcontrol20261025T183259Z-16af35e8 terminalFAILED1800s full deadline,
1931.056334GPU-sadmittedcharged. Retainedfarplayer006complete240context/80
judgedcrops, fullpartial.tarSHA2566008a1f00799029f865c040dbf9a28239e66ca916389243d5bad560fa7144ce6.
IndependentCPUcomparisonagainstoriginalgen1archive8a927ae4...:tandRGBcrops
BITEXACT, proof/tmp/k6b-quality-control-al-checkpoint-proof.json. No quality
claimedforunfinishednearplayer001. ContinueONLY001 viaexplicitunit-subset
validator(newoptionalflag;defaultstillfullwindow), exactknownfrozenunitset,
full240context/30steps/CFG2/4bit/unmodifiedPNGinput. Separatecheckpoint
archivescombinedlater, completebothplayers mandatorybeforecontrolclaim.
Nativegen module unchanged; subsetcannotincludeunknown/duplicate/emptyIDs,
focusedtestpasses. ExclusiveAdafullpoolgpu5/6,24000MiB,100ssmoke/800sfull/
20svalidator/1000stotal basedoriginal598.9sfullunit. No replayoffinished006.
Referenceledgerknownfailed131.172531+1931.056334=2062.228865GPU-s; Med
controlstillrunningwith2000reservation,Almissing1000reservation. Future
parts admittedonlyafteractualcosts andcompletecontrolgates.

Dense2windowselected/historicalCRF63 complete: bothselectedalpha4/affine1
0/4strictcells,no unresolved. Medjob20261025T190034Z-01c74a65 COMPLETE
207.220136GPU-s, archiveSHA256ba072c0b90af5dec468479fa96888a0a482ed909d25ee9a61bccd00de5facf35.
Medselectedalpha4at68.566667kbps LPIPS.219821640/DISTS.239993266;
affine1at31.43 .294500768/.282842942. Historicalalpha8at71.051667
.229989768/.248466691; clamp32p8at31.71 .313356146/.287229845.
DenseGPUactualthroughbothwindows503.504413s, plus550historicalreserve
=1053.504413<1800. Wholeframe/outsideRerrors and FP/FN saved; black-key
FPmeanAl2635.4/Med1520.2pixels perjudgedframe versusalpha257.2/186.45.
Allnativecontrols/pixel/R/S/coveragechecks passed; nofull12windowclaim.

MissingAlplayer001 control job `20261025T191545Z-f67eebcb` sourceea6fd44e specSHA256`ce2aa7eef86db2f7d7e34a1f78212c3ba557ad787012b4963724b410fa41a97d`;1000GPU-sreservation, exclusivefullAda pool, retained006bitexact checkpoint not replayed.

MedPNGcontrol20261025T184001Z-2e64d93e terminalFAILED1750sfull deadline,
1892.709538GPU-sadmittedcharged. Retained003full240context/80judgedcrops;
partialSHA2563caf5d7101ae059f60c142f0786711778cc480089d4e1a4b17d15d29b488ef0e.
Independentcomparison withoriginalfullgen2:tBITEXACT, RGBmeanabsolute
difference.059822591/channel,max44,5.6194%channelsdifferent. Thiscontrolis
NOTpixelbitexact (Al006wasbitexact); originalmetrics1e-4gate remainsstrict
andmustbecheckedbeforecandidateclaims. No newRGBtolerance invented.
Retain003andcompleteONLYmissing004 onexclusiveAda,100ssmoke/800sfull/20s
validator/1000stotal. FinalPNGassembled4unitsacrossbothwindows mustpass
fulloriginalneuralcontrolwithin1e-4 orinvestigate/stop. No oldcropreplacement
for newcontrol. Referenceknownspent3954.938403GPU-s (3failedjobs);
Almissing1000+Medmissing1000reserves =>5954.938403/14400s beforecandidates.

MissingMedplayer004 control job `20261025T192327Z-591cb699` sourceee0de5da specSHA256`44d1be9f15f30c7d6051a80e379725a290d5d8e578d355c8ba9fdb78e46d39bf`;1000GPU-sreserved, full240context, existing003checkpoint retained; strictPNGscoregate unchanged.

HistoricalAlstage20261025T190813Z-810bcaba terminalFAILED CUDAOOM in
DISTSdefaultbatch6 duringfullrun;141.266623GPU-sadmittedcharged,
81.831284sfullprocess. Smoke1framepeak6.829663GiB didnotcoverlarger
laterfrozenS crop. No completecurve used. Correctresourceconfiguration
ONLY: chunkDISTSinto4imagesfor>5variants; samefunction/weights/pixels/R/S,
controltolerance1e-4unchanged. FullAl5variantnativepeak8.686928GiB across
all80framesbounds chunk4byshape. Retest4originaljudgedframes instead1,
75ssmoke/480sfull/20svalidator/600stotal. Actualdense503.504413+failed
141.266623+reserve600=1244.771036<1800s. NoGPUcapincrease, no curve
quality/CRF/presetselectionchange. Preservefailedpartialmetricsseparately.

2026-10-25T19:34UTC capacityaudit: Almissingf67eebcb PENDING, noallocation
orcharge; pairedMed591cb699queuedexclusive. Currentsharedtimings1.6–1.8x
originalexclusive would exceedfullmatrixbudget, so do notpromotefull
comparisononcontendedGPUs. Requestedread-onlycapacity/ETAinformation from
existingplanningchat01a16cfc-1a88-7013-bd79-8489fe7c1eb2 (userauthorized
contact); explicitly nojobchanges/newchats/monitors. RetainqueuedIDs and
nativeboundedwaits. GPUclass/cohort/recipe/budget unchanged.

BoundedDISTS4 historicalAl job `20261025T193417Z-2387267a` source50d6d383 specSHA256`7c2e294e31c396487268a861c064aff8eea7cc4dfd2de4f3eda7fd9c3b33eace`;600GPU-sreserve,4framesmoke, all7actualcomplete240framecarrierpoints.

Pre-new-reference-quality sanitycheck uses ONLY originalsavedquality with
actualselectedpreset1bank/KPS/adaptercosts: all3families19/24counterfactual
strictwins, including4/4pilotcells. Alrates5.232358–5.244651kbps, Med
7.617872–7.640872kbps, exactdynamicmattegranted. This is NOT newgenerated
quality, prooforbound forchangedconditioning; actualpilot stillneeded.
Receipt/tmp/k6b-quality-p1-fixed-quality-sensitivity.json.
Planningread-onlycapacityreply: all4AdaGPUsbusy; bulkloads have no fleet
claims, humanowner/ETAunidentified. Nootherjobchanged. Existingexclusive
requestsremainpending; native1100swaittimeoutconfirmedAlpending withno
allocation. Onefurtherbounded2700sbatchwaitcoversbothqueueddeadlines.
No recurringLLMmonitor/newchat created; optionaluserETAquestionpending.

2026-10-26 user instructed keepqueued/noETA. Canonicalstatus nowconfirms
20261025T191545Z-f67eebcb and20261025T192327Z-591cb699 EXPIREDwithout
admission(remote_statusabsent),0GPU-s. Preservedoriginalrequests, no unknown
replay. Allhostsaudited07:03UTC: gpu5bothAdaGPUsidle48,645MiBfree;gpu6busy.
Publishfreshmissing-unitrequests,samefrozeninputs/caps,fullcompatiblegpu5/6
pool and45minfreshdeadlines. Useexclusiveadmission/stoponnewforeigncontention
to preservebudget; no automaticreplay. ExistingcompletedAl006/Med003 retained.
CPU2647.198626/3600s andGPUknownterminal4901.949513/16200s unchanged.

Freshalmissingcontrol job `20261026T070441Z-5b0eaf69` sourceeb64c64c specSHA256`accca36139dff4d39f4924bf2d7e4e4ef08e974c6b7bfad2dc2e52f792f40bf3`;1000GPU-sreserved, exclusiveAdapool, priorunstartedexpiredrequest preserved.

Freshmedmissingcontrol job `20261026T070444Z-299584f1` sourceeb64c64c specSHA256`fa2a9573903ed6b0cd2432c004c24038708d88cda49e2d4107aede9faafa1f0d`;1000GPU-sreserved, exclusiveAdapool, priorunstartedexpiredrequest preserved.

### Dense carrier phase closed (2026-10-26; reference pilot continues)

Boundedbatch4historicaljob20261025T193417Z-2387267a COMPLETE,302.240074
GPU-sadmitted; archiveSHA2561dbf827e0d490f40f67834e67ffcf6e271c0acb85823df1cb45c40526c5bb349.
All7historicalpoints/all80Aljudgedframes pass exactpixel/source/control gates.
Totaldense947.011110/1800GPU-s including2failedattempts; nofitting/generation.
Nativehistoricalpreset8Alcurves: alpha4points BD-rate +3550.7471%LPIPS,
+3043.7336%DISTS; clamp32threepoints +6447.7569%/+3061.1668%. PCHIPlograte
overactualqualityoverlap againstachievableanchorconcavehull. Medonlyone
historicalpointperrepresentation, noBD-rate. Selectedalpha4/affine1 remain
singlepointperwindow andeach0/4strictcells. Nevermixdifferentpresetsintocurve.
Thiscloses testedcontinuousforeground-video carriers onthese2windows, not
allcompressionmethods orfull12cohort. Referencegenerationquality stillunmeasured.

Actualreport/tmp/k6b-quality-dense-final-report.json,resourceledgers
/tmp/k6b-quality-cpu-final-ledger.json and/tmp/k6b-quality-gpu-ledger.json.
Knownterminaltotal4901.949513/16200GPU-s=1.361652642GPU-h; oldledgersclosed.
Freshcontrolrequests documentedabove; approval persists, capacity/reproduction
gates remain. Full12confirmation notauthorized by this2windowpilotbudget.

Freshoriginalcontrolmissing-unitjobs COMPLETE onidleexclusivegpu5:
20261026T070441Z-5b0eaf69 Al001:731.520607GPU-sadmitted,240context/80crops,
fullrender596.1s. ArchiveSHA256ce5c54e14db39d2beaf8e8f0c9c5627ca8d6a6420613dc96453fcef33877e297.
20261026T070444Z-299584f1 Med004:748.762351GPU-sadmitted,240context/80crops,
fullrender613.5s. ArchiveSHA256c448d784a2c84a2482bf4f72d74449ecce4842c88ee4f0d38b00100f4af7921c.
Bothnativevalidatorspassedexplicitknownunit scope; allweights/recipe/class/kernels
checked. Assemblewithretained006/003ONLY, nooriginaloldcrop substitution.
Referenceactual5435.221361GPU-s; combinedterminal6382.232472GPU-s.
Beforecompressed-referencegeneration, PNGQC<=350GPU-s (60smoke/240full/20
validator)checksnewboth-playercontrolagainstoriginalper-frameLPIPS/DISTS
within1e-4 onall160judgedframes. OriginalR/S,source andsameactualCUDAjudges.
Thisispre-candidatecorrectness verificationinsideapproved4.5GPU-h; no new
qualitymethod/family/cohort or training. Firstcompressed-referenceconfiguration
projection2x613.5s+nativeoverhead<1300sfull,110ssmoke+20validator within1450s.

PNGwholecontrolQC job `20261026T072949Z-aa61b482` source1a15ddb4 specSHA256`47976c5da48ba848311a0c254fb194ad1063a04fc516388f19a58007d113eef2`;350GPU-sreserved, bothassembledfullwindows, unchanged1e-4scoregate.

Newbudget bookkeeping: densephase CLOSED at947.011110GPU-s, belowits1800s
planningallocation. ReallocateonlytheunusedNEWdensecontingencytooriginal
controlrecovery/referencepilot: remainingreferenceceiling16200-947.011110
=15252.988890s, while TOTALapproved4.5GPU-h remainsunchanged. Old8h/0.75h
ledgersstayclosed. No newfamily/window/seed/steps/fit or search settings.
Referenceactual5435.221361 +QC350 +6x1450referencewindowjobs +finaljudge550
=15035.221361s; withclosed947.011110dense, combinedmaxreservation
15982.232471<16200s, leaving217.767529soverheadcontingency. Admissionstill
checksactualcostbeforeeachbatch; stopifcombinedcap cannotcoverremainingwork.
Thisupdatesinternal0.5/4hplanningreservesusingactualcompletedcost, notthe
user'sfrozenoverallbudget. Candidatebudgets110ssmoke/1300sfull/20validator,
1450total, sameexclusivecompatibleAdapool,stoponnewforeigncontention.

### Pre-candidate control amendment (2026-10-26; no candidate quality opened)

PNGQC20261026T072949Z-aa61b482 failedsmoke atnewAlPNG t0score
difference0.000183224678, above1e-4. Charged30.384579GPU-s. SavedRGB
re-scoringcontrolpassedunchanged1e-4; failureisfreshgenerativere-render,
notmetric/decoded-packet drift. Al001old/newcrops:tEXACT, RGBmeanabs.053333
levels/255,max46; Med004mean.052967,max29. PriorMed003mean.059823/max44.
Weights,pose/timing/geometry,seeds,30DDIM/CFG2/4bit andpinnedenvunchanged;
GPUUUID/workspace differ. Pinnedpipeline clearsattentionbanks atlines556/557
aftereachrun; generator isexplicitlyseededperunit/run.

Correcta conflatedoperationalguard BEFORE anycompressed-referencequality: retain
1e-4for re-scoringidenticalsavedRGB, use1e-3forfreshFP16originalPNG rendering
(<0.6%ofpilotmetricvalues, >5xobservedfirstsampledrift), requirefull160frame
calibrationto pass before candidateadmission. Thisisnot acandidatepass-bar
change: primarystrict3/4rule,anchors,R/S,registration/115pxstatus,cohort,all
3recipes andbudgetunchanged. ReportpairednewPNGqualitydifferences and±1e-3
renderer-envelope sensitivitybesideprimarystrictcells. No pixelhashrelaxation
for deliveredRGB/alpha. FullPNGcalibration maystillfail andstop; no further
toleranceadjustment aftercandidatequality. Bothold/new tolerancestested.

FullPNGcalibration job `20261026T073722Z-4ec3f83c` source20a319b6 specSHA256`599a80a04a19b39020b999e6cb74d64798092ac6f451b2d943abcdc51e15a9ca`;350GPU-sreserve, distinct1e-4savedRGB/1e-3freshFP16checks, all160frames beforecandidatequality.

FullPNGcalibration4ec3f83c FAILED240sfullruntimecap oncontendedGPU6,
293.338359GPU-sadmitted; no tolerancefailure. Alcomplete80frameprefix
PASSED(savedRGB1e-4,newPNGmax0.000587463<1e-3); Medunfinished. RetainAl
completepointusingexplicitcompleted-prefixvalidation, defaultfullcollection
validatorstillrejectspartialcollections andper-windowfullframecoverage never
waived. CompleteONLYMed onexclusiveidleAda,45ssmoke/170sfull/15svalidator/
250stotal. Combinedcalibrationmustcoverbothuniqueoriginalwindows/all160
frameswithallgates beforeANYcompressed-referencequality. No newtolerance
change or data replacement. Actualreference5758.944299s beforeMedQC;
combinedterminal6705.955410/16200s. Admitfuturebatchesagainstactualcost,
remainingworkforecast andfinaljudgingreserve, notunspentoldbudgets.

Med-only remaining PNG calibration job `20261026T075201Z-db1ea276`, source10aaffd7, specSHA256`330431ae6f7f2d354100f341a0b52b55c04d123d82af0edc3a0f135f7596ee22`; 250GPU-s cap. Al retained partial archive SHA256`cc6d0e794369fad3b62f008abee9afff32664277fd642df2390630f23fea81af`; explicit prefix validator passed all complete-window checks, collection remains incomplete until Med passes.

Med calibration `20261026T075201Z-db1ea276` COMPLETE; 151.315546 GPU-s. Full 80 frames pass, freshPNG maximum metric difference 0.000727654. Archive SHA256`fbaf2326e748732bdfd4da7796394f1d4f1e16fcd7237f36c60820418b3acdb0`. Combined Al+Med calibration covers all 160 original judged frames; both guard types pass. No candidate quality opened before this receipt. Combined actual 6857.270955 GPU-s; all six 1450-second candidate reservations plus 550-second judging fit the unchanged 16200-second cap.

Frozen reference full, al: job `20261026T075716Z-f0549307`, source`8cca9b19c007532e24bcc33347e874f2573212c7`, specSHA256`7d40ae27267cab96b51586ae416ec95758f1cae14cfd041918240fcdaf9c0a8f`; 1450 GPU-second cap, exclusive compatible Ada, full two-player 240-context-frame window. Complete calibration passed before admission.

Frozen reference full, med: job `20261026T075723Z-e962542f`, source`8cca9b19c007532e24bcc33347e874f2573212c7`, specSHA256`f4323d6de8789fdede6fb5c73228bc7ee354a67b93df15456eba50c3295fd221`; 1450 GPU-second cap, exclusive compatible Ada, full two-player 240-context-frame window. Complete calibration passed before admission.

Frozen reference alpha, al: job `20261026T075730Z-e8d8f772`, source`8cca9b19c007532e24bcc33347e874f2573212c7`, specSHA256`7b96d0c05d499bdfaced6adff3f63049c5a6d287781644f838b273e97e099e31`; 1450 GPU-second cap, exclusive compatible Ada, full two-player 240-context-frame window. Complete calibration passed before admission.

Frozen reference alpha, med: job `20261026T075737Z-a900b575`, source`8cca9b19c007532e24bcc33347e874f2573212c7`, specSHA256`c3d58daedf6ebb443ff4dde02755eb4872c90703c38937e73f74a6bd259fde3e`; 1450 GPU-second cap, exclusive compatible Ada, full two-player 240-context-frame window. Complete calibration passed before admission.

Frozen reference black-affine, al: job `20261026T075746Z-ca0a46ed`, source`8cca9b19c007532e24bcc33347e874f2573212c7`, specSHA256`86af6ade26458455bf0886d469c4e43646dd1827d49c0817041f5b37d6ef8ba2`; 1450 GPU-second cap, exclusive compatible Ada, full two-player 240-context-frame window. Complete calibration passed before admission.

Frozen reference black-affine, med: job `20261026T075753Z-75fd93fc`, source`8cca9b19c007532e24bcc33347e874f2573212c7`, specSHA256`00e7dd1393a7a8e55fb19a6e93eed4c8a37120c088a40e270b3c2ec3c17c2653`; 1450 GPU-second cap, exclusive compatible Ada, full two-player 240-context-frame window. Complete calibration passed before admission.

Reference full al `20261026T075716Z-f0549307` COMPLETE; admitted 1368.586064 GPU-s; archive SHA256`6a3f620b394871154290340d7700831aea62f098e2569bf0d130e6735a4a1ef1`. Native validator status retained; actual quality still pending joint judging.

Reference alpha al `20261026T075730Z-e8d8f772` COMPLETE; admitted 1399.461311 GPU-s; archive SHA256`2335636c70cda0b2fbc27d6636f64d3727725b4b15396679d8586daeae122759`. Native validator status retained; actual quality still pending joint judging.

The first two candidate jobs passed native full-window validation. Combined
actual GPU cost is 9625.318329 seconds. Four outstanding renders reserve
5800 seconds. Retain 650 seconds for final joint judging (60-second smoke,
540-second full, 20-second validator plus overhead), using measured complete
Med calibration as the timing basis: two generated variants plus truth,
versus five generated variants plus truth in the final comparison; the same
original R/S and metric implementation remain in use. This
raises only the internal judging reservation from 550; combined worst remaining
reservation is 16075.318329/16200 seconds. No new GPU budget, recipes,
quality thresholds, sources or fitting. Preserve all failed and complete jobs.

Reference full med `20261026T075723Z-e962542f` COMPLETE; admitted 1360.567544 GPU-s; archive SHA256`ba752111fb9e2604a34dc560a06586cf820013ab398b6c834aac54546e6881e1`. Native validator status retained; actual quality still pending joint judging.

Reference alpha med `20261026T075737Z-a900b575` COMPLETE; admitted 1400.275787 GPU-s; archive SHA256`e11f553f69aa4f3fb1347326b9d30afceed04ed9fdfa32328a60cca2d6f4b4fa`. Native validator status retained; actual quality still pending joint judging.

Full RGB and explicit-alpha reference families now each have both complete native windows. Combined terminal cost 12386.161660 GPU-s; remaining two affine-black renders plus 650-second joint judging reserve imply at most 15936.161660/16200 seconds. All candidate quality remains unopened.

Reference black-affine al `20261026T075746Z-ca0a46ed` COMPLETE; admitted 1375.243978 GPU-s; archive SHA256`c017d4b502577215cc31a15c0ef9d04a663cfc73e92d812dfd6b77a925b6d066`. Native validator status retained; actual quality still pending joint judging.

Reference black-affine med `20261026T075753Z-75fd93fc` CONTENDED; admitted 1050.864920 GPU-s; archive SHA256`not-published`. Native validator status retained; actual quality still pending joint judging.

### Interrupted affine reference recovery, unchanged total cap

Med affine job `20261026T075753Z-75fd93fc` is terminal CONTENDED after a
foreign process appeared on its GPU. Charged 1050.864920 GPU-seconds; do not
resume/replay it. Retained Med003 has all240 context frames and all80 judged
crops, original seed/30steps/profile, exact NPZ hash/indices/shape and passed
smoke recipe/device gate. Explicit checkpoint proof passes; parent job remains
incomplete. Partial archive SHA256
`a9075cd896af27a0bf46f13c6f0a6be85699ee9adab8f6d8b1a32a70903a0d40`.
Al affine `20261026T075746Z-ca0a46ed` COMPLETE,1375.243978 GPU-seconds.

Combined terminal charge14812.270558/16200 seconds. Recover ONLY unfinished
Med004 with100-second smoke/650-second full/20-second validator,800 total.
Measured original Med004 used748.762351 admitted/613.5full seconds through the
same production renderer/recipe. All hosts re-inspected; one exclusive Ada GPU
is idle. Retain completed003 and validate the new004 separately; then combine
archives without replacing raw evidence. No family/seed/source/quality change.

Restore the earlier550-second final-judge reservation (60smoke/450full/20
validator), rather than the optional650 contingency: historical eight-variant
Al judging completed in241.553530 full seconds even under contention; five
variants over two windows project241.554*5/8*2+60=361.942seconds, below450.
This is a resource projection, not a latency claim; native smoke rechecks
correctness and memory. Original Med two-variant window took99.621seconds.
Worst combined reservation14812.270558+800+550=16162.270558<16200. Stop on
failed checks or inability to finish inside the total cap; no new budget.

Affine Med004-only continuation job `20261026T091049Z-e429637f`, source`9db350ec0a6a50c3d9365c66de17c27d0541aaa1`, specSHA256`6d129b78640b02fec758ffbe7a8a55258d2195075c474b61752581d26e6d5d15`;800 GPU-second cap; retained003 is never replayed. Native partial proof and final two-archive manifest must pass before judging.

Med004 continuation `20261026T091049Z-e429637f` FAILED its650-second full finalization
limit after saving the complete unit (240 context frames,80 judged crops).
Charged 735.208909 GPU-seconds. CPU checkpoint qualification passes exact
NPZ hash, original judged indices/shape/dtype, original seed/profile/steps,
nonconstant crops and native smoke recipe/base/adapter/guider/device/kernel
gates. Archive SHA256`0d71eef9c9e519f62d6f1954fcb98eb6917a86118f287d0accf98f244260594f`. Do not mark either
parent job complete or fabricate missing global metadata. Two disjoint complete
unit checkpoints qualify the full Med affine window payload; both original
parent states remain contended/failed in the manifest. The final native judge
must verify all original R/S, source pixels and full80-frame score coverage.
No further generation. Combined actual 15547.479467
GPU-seconds; with550 final judging, maximum 16097.479467
remains below16200. No scope, recipe, tolerance or total-budget change.

Final joint native judge job `20261026T092648Z-ccf0633e`, source`1c1918e47f399990b1ba70376f8a5bccbee7f11d`, specSHA256`d0bfac4dfba2e51445bf723a7bfdf717a610924eab5a1bd36af37fe7a3ae5940`;550 GPU-second cap. All3 fixed families/both windows, original160 judged frames/R/S, savedRGB1e-4/freshPNG1e-3 controls, actual decoded-reference renders. Two complete affine Med unit checkpoints are explicitly identified as partial archives; parents remain failed/contended.

## 2026-10-26 — Two-window compression-quality pilot: complete

**Decision:** compressed, unmasked full RGB reference banks pass the frozen >=3/4 pilot rule (4/4). Explicit-alpha and affine-black segmented reference banks each fail (1/4). Each selected dense foreground carrier fails (0/4). These are actual generated/rendered quality results, not unchanged-quality rate predictions. The full 12-window >=18/24 claim remains untested.

### Frozen comparison and accounting

The two windows are Alcaraz–Hurkacz 031@129 (source frames 9–248, 25 fps, 9.6 seconds) and Medvedev–Struff 014@194 (74–313, 50 fps, 4.8 seconds). Each uses both original players, all 240 coded/context frames and its original 80 judged frames. Original method and anchors share those indices and court registration, R/S masks and target-derived Telea background. The 115 px threshold remains exploratory; original-source/replay independence remains unresolved. This is a familiar-broadcast, clip-local diagnostic.

The three global reference recipes were frozen before new quality: native SVT-AV1 preset 1, CRF 20, half-size 256×392 decoded back to 512×784. The slower-preset CPU comparison established delivered-byte/decoded-fidelity dominance over presets 4 and 8 on both match banks. Both encoded color plus auxiliary alpha in a single Matroska container and a three-channel affine black-key video were actually decoded and tested. The black-key recipe reserves a guard band rather than relying on RGB 1 surviving lossy compression. Sparse reference masks accompany the reference bank; they do not deliver the per-target-frame dynamic matte. Dense foreground video, tested separately, delivers both appearance and silhouette over all 240 target frames.

All generations retain the same base, 4-bit match adapter/pose guider, 30 DDIM steps, CFG 2, seed, geometry and 24/4 context. No fitting, segmentation inference or Wan-Animate. The primary rate pays keypoints, the actual delivered reference-bank container and adapter/guider over the original 7200-second match horizon; a common shipped base and exact dynamic matte remain granted. Single method operating points have no BD-rate curve; the decision uses the same achievable anchor envelope at the measured method rate. No quality extrapolation or per-window recipe selection.

### Actual quality

| Reference bank | Window | Paid rate, kbps | LPIPS | DISTS | Strict wins |
|---|---|---:|---:|---:|---:|
| full | alcaraz_hurkacz/031@129 | 5.237129 | 0.245873570 | 0.220013805 | 2/2 |
| full | medvedev_struff/014@194 | 7.640872 | 0.172194882 | 0.186108082 | 2/2 |
| alpha | alcaraz_hurkacz/031@129 | 5.244651 | 0.294567408 | 0.253369965 | 0/2 |
| alpha | medvedev_struff/014@194 | 7.630432 | 0.203314784 | 0.207102623 | 1/2 |
| black-affine | alcaraz_hurkacz/031@129 | 5.232358 | 0.293400037 | 0.253710111 | 0/2 |
| black-affine | medvedev_struff/014@194 | 7.617872 | 0.202739390 | 0.205025694 | 1/2 |

Full RGB banks are 48,164 bytes (Alcaraz) and 68,193 bytes (Medvedev), versus 895,487 and 1,179,150 original PNG bytes: 94.62% and 94.22% smaller. Relative to the fresh original-PNG control, full RGB changes LPIPS by +0.000704/+0.001648 and DISTS by -0.003048/-0.005982. Its 4/4 result survives both signs of the frozen ±0.001 renderer-envelope sensitivity. Alpha references lose 0.049398/0.032768 LPIPS and 0.030308/0.015013 DISTS versus the fresh PNG control; the two segmented families therefore do not preserve conditioning quality. The data support retaining background/context in the sparse reference crop for this model, not a universal claim about alpha containers.

**Granted-matte limitation:** charging the measured exact dynamic matte raises full-RGB rates to 52.894629/77.540872 kbps and removes all four wins. All three reference families are 0/4 with that matte packet paid. Charging the 48,878,567-byte base adds 54.309519 kbps: each reference family has zero wins, two resolved losses and two unbracketed/unresolved cells. Thus this is not a fully charged codec win. The historical step-5 failure and unresolved source independence remain.

The selected alpha-preset-4 and affine-black-preset-1 dense foreground videos each remain 0/4. The historical Alcaraz preset-8 multi-rate curves lose by +3550.75%/+3043.73% BD-rate (alpha, LPIPS/DISTS) and +6447.76%/+3061.17% (black32), PCHIP over the actual overlapping quality range against the achievable anchor envelope. Selected preset-1/4 single points are not mixed into those curves; Medvedev lacks a multi-point curve. These bounded tests do not establish global compression optimality. JPEG AI, learned mask prediction and exhaustive rate/resolution searches remain untested.

### Correctness, recovery and provenance

Both full original-PNG windows calibrated before any candidate quality was opened. Identical saved-RGB metric re-scoring retains the original 1e-4 guard; fresh FP16 generation uses the explicitly recorded pre-candidate 1e-3 amendment. The final joint judge passes all 160 original judged frames, source and R/S hashes, exact saved-output hashes, CUDA/class/memory and finite-score/mean checks. Final saved-RGB maximum errors are 0.000004709/0.000018775; fresh-PNG maxima are 0.000587463/0.000740707. Peak allocated GPU memory is 8.686927 GiB. No further tolerance changes followed candidate quality.

The Medvedev affine job was stopped on foreign GPU contention after completing player 003. Its 004-only continuation saved the complete unit before a finalization timeout. Both raw parent states remain CONTENDED/FAILED. Explicit CPU qualification verifies each disjoint unit’s complete 240-frame context, all 80 original judged indices, NPZ hash/shape/dtype, seed/steps/profile and native smoke recipe/device/kernel gates. The final full-window judge consumes those two partial archives without replacing data or inventing global generation metadata. Both players and all original score frames are present. No completed unit was replayed.

Final native judge: `20261026T092648Z-ccf0633e`, code revision `1c1918e4`, source SHA256 `d1a2350572c9127261fe6e45e7c576b18c6138fd11a26cefaa0d54cf5acbda7b`, gpu5 / NVIDIA RTX 6000 Ada Generation, UUID `GPU-07aa7586-7116-8c7b-1e9c-e5dbdc74b162`. Native archive SHA256 `d5a9670d5a223d6c231476cbf7211442aa2ab77dfc01ba1cffbfb7265dee7457`. Native report includes pinned metric/backbone versions, input hashes and per-frame scores; frozen submitted specifications bind generation inputs/environments/weights and exact code revisions. CPU codec receipts record SVT-AV1/dav1d/ffmpeg paths and versions.

Final reference report SHA256 `e9bbb4aca54cd5453f6c7e3472097881c37a200f6b685ec31221b5aff6c10797`; dense report SHA256 `86acf4a379594c1630507f8a608a350efc077b200ec5db9e8db302d329d7be85`. All raw complete and failed outputs are retained; the immutable publication index below supplies canonical paths and checksums.

### Closed resource ledger

New GPU total: **15879.667029/16200 seconds = 4.411018619/4.5 GPU-hours**, including every failed smoke, timeout, contention and recovery. Dense phase: 947.011110 seconds; reference/control/judge: 14932.655919. Slower-preset CPU preflight: **2647.198626/3600 CPU-seconds**, conservatively including the local allowance. Old 8-hour, 0.75-hour and CPU ledgers stay closed. No active GPU jobs or recurring model monitors remain.

| GPU job | Role | Native terminal state | Admitted GPU seconds |
|---|---|---|---:|
| `20261025T183259Z-16af35e8` | al-control-full | failed | 1931.056334 |
| `20261025T183318Z-34d149af` | med-control-smoke | failed | 131.172531 |
| `20261025T184001Z-2e64d93e` | med-control-full | failed | 1892.709538 |
| `20261025T185239Z-1df3ec6f` | al-pixel-smoke | failed | 44.123715 |
| `20261025T185724Z-884dd4ce` | al-low | complete | 252.160562 |
| `20261025T190034Z-01c74a65` | med-low | complete | 207.220136 |
| `20261025T190813Z-810bcaba` | high-batch6 | failed | 141.266623 |
| `20261025T193417Z-2387267a` | high-batch4 | complete | 302.240074 |
| `20261026T070441Z-5b0eaf69` | al-original-missing-unit | complete | 731.520607 |
| `20261026T070444Z-299584f1` | med-original-missing-unit | complete | 748.762351 |
| `20261026T072949Z-aa61b482` | png-smoke-strict | failed | 30.384579 |
| `20261026T073722Z-4ec3f83c` | png-calibration-two-windows | failed | 293.338359 |
| `20261026T075201Z-db1ea276` | med-only-png-calibration | complete | 151.315546 |
| `20261026T075716Z-f0549307` | full-al | complete | 1368.586064 |
| `20261026T075730Z-e8d8f772` | alpha-al | complete | 1399.461311 |
| `20261026T075723Z-e962542f` | full-med | complete | 1360.567544 |
| `20261026T075737Z-a900b575` | alpha-med | complete | 1400.275787 |
| `20261026T075746Z-ca0a46ed` | black-affine-al | complete | 1375.243978 |
| `20261026T075753Z-75fd93fc` | black-affine-med | contended | 1050.864920 |
| `20261026T091049Z-e429637f` | affine-med004-recovery | failed | 735.208909 |
| `20261026T092648Z-ccf0633e` | joint-actual-reference-judge | complete | 332.187562 |

Unstarted requests 20261025T191545Z-f67eebcb and 20261025T192327Z-591cb699 expired at zero GPU cost; unpublished 20261025T190533Z-524c8807 had no request/admission and zero charge. These IDs were inspected rather than replayed.

**Stop:** the authorized two-window comparison is complete and its decision is reached. Retain compressed unmasked references as the candidate; do not advance segmented-reference or dense-foreground variants under this rule. A full-cohort quality confirmation would need a separately scoped budget and must retain the granted-matte, background, base-cost and source-independence limitations. No full-12 run is submitted.

Immutable evidence publication: `/home/itec/emanuele/Datasets/pointstream-data/tennis/k6b-reference-quality-20261026T093850Z`. Metadata bundle SHA256
`75b489c2897f44c35a1cec12fbb0545e03c7dc51e65fcc5aba906141d108ce2e`; all22 copied raw artifacts and110 metadata
files passed checksum verification and are read-only. Existing source data,
weights, checkpoints and prior publications were not overwritten. Source
requests/statuses/specifications and completed-unit qualification proofs remain
inside the bundle; failed parent states remain explicit.

Verification: configured `ruff check` and `mypy --config-file pyproject.toml`
pass;21 focused tests pass, including explicit partial-collection rejection and
completed-window recovery. Prior full local suite recorded873 passed/25 skipped
plus the documented macOS protected-path assertion and sandbox process test;
the latter passes when allowed read-only process inspection. Linux PR CI passes
on the preceding code revision; final documentation commit is checked again.
This is the fidelity track (LPIPS/DISTS); the experience track is untested.

Saved checkpoints contain80 judged crops per player, while the native
generation counter verifies all240 context frames were generated. They do not
store240 decoded crops. Final primary decision and rates use original judged
indices and original registered clip duration.

## 2026-10-26 — Joint compressed RGB references/mattes and edge ramps (frozen before measurement)

User explicitly requested the joint saved-render test and a CPU check of smoothing
the foreground/key boundary, as recorded in planning chat. New scoped branch
`codex/k6b-joint-matte-edge-pilot`; self-contained execution card
`docs/cards/k6b-joint-matte-edge.md`. All prior ledgers are CLOSED.

Primary hypothesis: the new FULL compressed-reference generations may retain
enough quality when paired with actual1/8,1/16,1/32 matte packets to pass>=3/4
strict LPIPS/DISTS cells on frozen Al031/Med014. Competing explanations: matte
rate/shape error still dominates, or apparent benefit comes from missing crop
pixels/changed regions/unpaid side information. All240 coded frames are paid;
original80 judged ticks/window and R/S are unchanged. PR242 FULL saved-RGB
control must reproduce composite hashes and scores within1e-4. Native missing
pixel fallback stays Telea, never target foreground. Exact matte paid, common
base and base charged are reported separately; no full12/independence claim.

Secondary hypothesis: width2/4 full-pixel ramps reduce transform cost without
material edge damage; competing explanation is threshold displacement/color
contamination or an underdetermined RGB/alpha inverse. Freeze width0/2/4,
preset4/CRF35/yuv420p/halfsize; first8 stored judged ticks of both windows with
actual rational timing (smoke Al first4). RGB guard-key ramps and separate paid
luma-mask ramps are labeled separately. Decoder uses decoded channels/constant
metadata only. Zero-codec/anti-leak gates and an explicit non-identifiability
example precede codec claims. Report IoU, symmetric edge mean/p95, temporal edge
error changes, RGB boundary error and container rate. Global engineering
promotion requires>=10% saving on both excerpts, IoU drop<=.005, mean boundary
error rise<=.25px and temporal instability rise<=.005; not a paper verdict.
No secondary neural quality run or sweep.

Existing evidence audited before new measurements: H5p2 signed-distance/binary
SVT video failed its matched-accuracy rule; XOR/LZMA remained cheaper. H5p4c
flat RGB key carriers had accurate shapes but failed the paid-rate oracle.
These concern VISOR/fixed old configurations, not a universal impossibility.
The new check isolates small ramps on the saved tennis pixels and does not
repeat that full sweep or assume deblurring can recover discarded information.

New resource caps: GPU<=1800 admitted seconds, CPU<=3600 aggregate seconds and
2h wall. First GPU reservation750s (60smoke/600full/20validator/70overhead);
real prior five-variant judging took332.188s, new four-variant run fits. CPU
reservation800s at4threads +400local=3600CPU-s (60smoke/650full/30validator/60
overhead); measured projection gates full. All stages<=45min, qualified native
Ada metrics only, exclusive/stop-on-contention, native bounded fleet waits.
No new generation/fitting/reference change/segmenter inference/Wan-Animate,
chats, parallel agents or recurring model monitors. Source uncertainty, granted
background, exploratory115px and historicalstep5 failure remain explicit.

Joint saved-render GPU job `20261026T122430Z-945020c3`, source`e0958ad2c27193af80ffe506c217708074e20d42`, specSHA256`bc9ad969b0b2fe024e3e4e5a393df97556742d6de69ff7ab8d5591d864c5b49c`;750s reservation of new1800s cap. Both original windows, four settings, all original source/R/S hashes and FULL saved-pixel controls. No generator is loaded.

CPU edge spec first attempt rejected locally before publication: missing required CPU-schema fields gpu_models/gpu_memory_mib. No job ID or admission,0CPU/GPU charge. Add empty gpu_models and zero GPU memory; algorithm/inputs/resource cap unchanged.

CPU edge mechanism job `20261026T123044Z-a13c06af`, source`85ac55cebba5b1ea6dd4c1e447296a2cdd48bac5`, specSHA256`7c04d9a00afad56f090694c2fa6d7673a90216c930342278823684ad2d190c18`;800s at4threads reserved (=3200CPU-s) plus400local. Al4tick smoke gates both8tick excerpts; all width/family/codec settings frozen before results.

CPU job a13c06af FAILED smoke before codec work: wire compositing IDs use
`person:...` while saved crop unit files use `person_...`; suffix matching
failed. New normalizer checks unique identity explicitly and has a focused
regression test; no frame/order substitution. Conservative charge40.305241
CPU-seconds. Corrected CPU reservation780admitted seconds x4 +40.305241
spent +400local=3560.305241<3600, same60smoke/650full/30validator.

Queued GPU 945020c3 was inspected unstarted and cancelled with no admission,
0GPU cost, to add explicit source RGB/player-mask frame hashes and decoded
matte hashes for all240 coded ticks before first judging. No pixels, quality
rule, recipe, inputs or tolerance change; replacement will have a fresh ID.
All Ada GPUs remain occupied. No uncertain replay or foreign-process action.

Pre-codec source-coverage clarification (before any neural/codec result): local
provenance construction shows some original matte pixels lie outside the saved
generated crop. Primary test retains original masks, original R/S and native
Telea fallback and reports every missing pixel. Secondary mechanism uses the
native written support (exact mask intersect available crop) as its edge target,
reports uncovered original-mask pixels separately, and averages only available
generated colors at half resolution. It never fills a gap with target foreground
or claims those colors exist. Hard baseline is binary at the coded grid; zero
codec full-resolution geometry loss is reported separately. Width2/4 ramps are
computed from the same written support. No quality-based selection or threshold
change; this bounds the mechanism's scope rather than fabricating RGB evidence.

Corrected GPU job `20261026T124013Z-b543788f`, source`1579e807a468dc292334bdafabcdc5031e23147b`, specSHA256`3751d7ef50442991e2a7233187ad18c3fe7a24e4ca7d91a888a39d80d733062f`; 750 admitted-second cap (8 CPU threads). Prior failed/cancelled IDs retained, never replayed.

Corrected CPU job `20261026T124022Z-49e6c45a`, source`1579e807a468dc292334bdafabcdc5031e23147b`, specSHA256`0311cacaf8d1dc9bc7bd3b5ffc8ae163eef0c72f2a89aeb62a03158951d65523`; 780 admitted-second cap (4 CPU threads). Prior failed/cancelled IDs retained, never replayed.

CPU mechanism COMPLETE: corrected job49e6c45a verified all12 fixed points,
zero-codec/anti-leak and independent container-only client mask reconstruction.
Admitted four-core bound 281.219577CPU-s; with40.305241failed smoke and400local
allowance, new ledger 721.524817/3600CPU-s. No further CPU sweep. Native
archiveSHA256938dbc2f6fd4428322f2ce8509c5085a686121f23aea6b8680959763d732c493.
All four ramp/family alternatives fail the frozen engineering promotion bar.
Full results will be reported separately from the pending primary neural test.

Before any primary neural scores, resource-only amendment: exclusive Ada
request b543788f remained unstarted and was cancelled at0GPU cost. The same
Ada LPIPS/DISTS path is qualified in shared mode; saved RGB hashes and1e-4
controls must pass, and no latency claim is made. Use shared admission with
12000MiB process declaration plus8192MiB free-memory margin (excludes cramped
GPUs), continue-on-contention. First reservation1000s (100smoke/800full/20
validator/80overhead), leaving800 for concrete recovery inside unchanged1800s
cap. Timing forecast332.188*(4/5)*2.5=664.376<800; native smoke must also
extrapolate both complete frame loops plus60s to<800 before promotion. This
changes scheduling/reservations, not pixels, models, masks, rates or tolerances.
Shared timings are resource checks only, never performance evidence.

Shared resource smoke separates source/setup time from measured frame-loop time, extrapolates actual scored-frame cost to80/window with1.5x allowance for varying ROI size, then adds60s model/finalization reserve. It does not multiply fixed source preparation by40. Full is gated at projection<800; no scoring formula changes.

Shared joint judge job `20261026T124922Z-b47be797`, source`c2fa3b4862bac25bada2502aef0e3fc315b739eb`, specSHA256`f6eb1212d8c6ed6dabee928333d72da0a139e5b784714582e47e12113f2dcbd5`;1000s reservation of unchanged1800s new cap,12GB process plus8GiB free margin. Prior exclusive queue b543788f confirmed cancelled/unstarted at0GPU cost. Fixed160 frames/4settings; no generation.

### CPU edge mechanism result (complete; separate from neural joint test)

Native job `20261026T124022Z-49e6c45a`, source `1579e807`, gpu6 CPU-only, SVT-AV1 preset4/CRF35, 960×540 4:2:0. All12 fixed points and container-only client reconstruction passed. Widths are in full-resolution pixels. Each excerpt contains the first8 stored judged ticks (0,3,…,21), coded at the actual25/3 or50/3 fps:0.96s Alcaraz and0.48s Medvedev. These container rates include headers/metadata and are NOT full240-frame stream-rate estimates.

The RGB carrier mixes guard-lifted generated RGB into black with a known width but arbitrary foreground color. Decoder uses only decoded RGB and the fixed max-channel>16 test, then guard inverse; it does not know alpha. The mask-only control thresholds decoded luma at128 and is a separately paid stream with no foreground color. No oracle mask is available to either client; a fresh container decode alone reproduces each stored decoded-mask hash.

| Excerpt | Ramp px | RGB container B | kbps | IoU | Edge mean/p95 px | Edge error flips | Boundary RGB MAE | FP RGB RMS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Alcaraz | 0 | 8634 | 71.950 | 0.982795 | 0.163479/2.000 | 0.212668 | 12.188 | 20.914 |
| Alcaraz | 2 | 7914 | 65.950 | 0.910214 | 0.904284/2.000 | 1.027755 | 15.039 | 17.897 |
| Alcaraz | 4 | 7208 | 60.067 | 0.861151 | 1.548112/2.828 | 1.393094 | 19.729 | 16.621 |
| Medvedev | 0 | 7857 | 130.950 | 0.979787 | 0.191872/2.000 | 0.228670 | 13.668 | 26.727 |
| Medvedev | 2 | 7002 | 116.700 | 0.896763 | 1.053817/2.000 | 1.200524 | 17.098 | 20.990 |
| Medvedev | 4 | 6524 | 108.733 | 0.847947 | 1.720738/2.828 | 1.632526 | 22.834 | 21.142 |

Edge-error flips mean changed boundary-error pixels per true boundary pixel between consecutive sampled ticks; no motion alignment or perceptual interpretation is claimed. RGB errors use only available generated colors and the fixed4px boundary band; FP RGB RMS quantifies color written outside the mechanism target.

| Excerpt | Ramp px | Mask-only container B | kbps | IoU | Edge mean/p95 px | Edge error flips |
|---|---:|---:|---:|---:|---:|---:|
| Alcaraz | 0 | 5669 | 47.242 | 0.999881 | 0.001027/0.000 | 0.001378 |
| Alcaraz | 2 | 5883 | 49.025 | 0.999881 | 0.001027/0.000 | 0.001378 |
| Alcaraz | 4 | 5746 | 47.883 | 0.999761 | 0.002147/0.000 | 0.003005 |
| Medvedev | 0 | 5355 | 89.250 | 0.999292 | 0.006080/0.000 | 0.013517 |
| Medvedev | 2 | 5386 | 89.767 | 0.999155 | 0.007334/0.000 | 0.015726 |
| Medvedev | 4 | 5384 | 89.733 | 0.999110 | 0.007939/0.000 | 0.015875 |

**Decision:** none of the four ramp/family alternatives passes the frozen global engineering bar. RGB width4 saves16.52%/16.97% bytes but IoU falls from0.982795/0.979787 to0.861151/0.847947; mean displacement increases from0.163/0.192px to1.548/1.721px, and temporal boundary error/color contamination increase. Width2 saves8.34%/10.88%, also failing the accuracy/global saving bar. Luma ramps are accurate but slightly larger than hard luma on both excerpts.

The RGB edge problem exists before compression: width2 zero-codec IoU is0.921484/0.881378 and width4 is0.864715/0.848374. Hard zero-codec IoU is0.999881/0.999292 at full resolution (small half-grid geometry loss), with exact coded-grid hard mask recovery and guard inverse RGB error<=1. The luma ramp zero crossing preserves its target well; this does not imply recovery of the original boundary after quantization. A mixed RGB value64 is consistent with opaque foreground64 or half-opacity foreground128, even under the legal guard range. Arbitrary foreground RGB and alpha are generally not uniquely invertible from three received color channels; spatial/color priors may estimate an edge, but cannot guarantee an exact inverse.

Four-pixel ramps do not earn promotion merely because they save bytes. Binary shapes are not inherently incompressible: H5 showed strong XOR/LZMA prediction, and here hard luma is already accurate/cheaper at the fixed setting. This is a bounded mechanism diagnostic, not a proof that all smooth carriers fail or that a different decoder/resolution/codec cannot improve. No secondary LPIPS/DISTS result is claimed.

Target for these diagnostics is the native written support inside the saved RGB crop, not invented color outside it. Original source-mask pixels outside that support total44 (Alcaraz) and538 (Medvedev) over the8 sampled frames, explicitly retained in source-coverage records. The primary joint experiment retains original masks/R/S and native Telea fallback instead of changing its target.

CPU conservative ledger:40.305241CPU-s failed identity-parser smoke +281.219577CPU-s corrected complete job +400 local allowance =721.524817/3600CPU-s. Corrected full program measured35.197832CPU-s /29.192039s wall; these narrower process timings exclude the separately supervised smoke/validators. All failure artifacts retained. Raw published archive SHA256 `938dbc2f6fd4428322f2ce8509c5085a686121f23aea6b8680959763d732c493`.

## 2026-10-26 — Joint saved-render matte pilot: complete, tested recipes fail

**Decision:** actual PR242 unmasked compressed-reference generations paired with paid1/8,1/16 or1/32 dynamic mattes each score0/4 strict LPIPS/DISTS wins (no unresolved primary cells), below the frozen>=3/4 pilot rule. The exact/granted-matte control reproduces4/4 and exactly the saved composite RGB hashes. All results use the original two windows/all240 coded frames billed/original80 judged frames per window, both players, R/S regions and crop order. No fitting, generation, new reference, new segmenter or Wan-Animate.

### Actual joint rate and quality

| Matte scale | Window | Whole paid rate kbps | Matte kbps | LPIPS | DISTS | Wins |
|---|---|---:|---:|---:|---:|---:|
| exact granted control | alcaraz_hurkacz/031@129 | 5.237129 | 47.657500 | 0.245873571 | 0.220013846 | 2/2 |
| exact granted control | medvedev_struff/014@194 | 7.640872 | 69.900000 | 0.172194882 | 0.186108105 | 2/2 |
| 1/8 | alcaraz_hurkacz/031@129 | 16.841296 | 11.604167 | 0.244257053 | 0.230332728 | 0/2 |
| 1/8 | medvedev_struff/014@194 | 24.547539 | 16.906667 | 0.174497564 | 0.194600713 | 0/2 |
| 1/16 | alcaraz_hurkacz/031@129 | 10.868796 | 5.631667 | 0.266268707 | 0.240286078 | 0/2 |
| 1/16 | medvedev_struff/014@194 | 14.510872 | 6.870000 | 0.197085623 | 0.214699723 | 0/2 |
| 1/32 | alcaraz_hurkacz/031@129 | 7.063796 | 1.826667 | 0.324610962 | 0.263390148 | 0/2 |
| 1/32 | medvedev_struff/014@194 | 9.860872 | 2.220000 | 0.262879380 | 0.260517818 | 0/2 |

For the granted control the matte column is its separately measured exact-packet rate, excluded from the displayed primary total; every reduced scale pays its actual packet in the total. Rates also pay actual keypoints, the PR242 preset1/CRF20 unmasked reference-bank containers and the adapter/guider over7200 seconds, with a common shipped base. No PNG reference-price estimate or quality-invariance assumption replaces the actual decoded-reference generations.

The fine1/8 matte largely preserves LPIPS but costs11.604/16.907kbps on top of the reference/keypoint/adapter method. The coarse1/32 matte costs only1.827/2.220kbps, but quality falls substantially: LPIPS0.324611/0.262879 and DISTS0.263390/0.260518. The intermediate scale also loses. The measured exact-matte-paid control at52.894629/77.540872kbps has0/4 wins. All three reduced-scale decisions remain0/4 under either sign of the ±0.001 renderer-envelope sensitivity.

Charging the base for each variant adds54.309519kbps (48,878,567bytes/7200s) and has zero wins, two resolved losses and two unbracketed/unresolved cells. There is one method operating point per scale/window; no method BD-rate curve is defined, and no rate or quality extrapolation is used. Comparisons use the same achievable SVT-AV1/DCVC anchor envelope.

### Pixel, coverage and provenance gates

Both original source clip hashes, decoded source RGB frame hashes, per-player source-mask frame hashes and the decoded matte frame hashes for ALL240 coded ticks are recorded. Packet bytes/CRC/header/unit-order/grid and native decoded-array hashes are checked. Every saved generated crop hashes and covers exactly its expected original judged timestamps. Original R/S hashes match; no region or source changes. The client compositor pastes only saved generated RGB inside the unchanged crop box. Requested pixels without available crop RGB retain native Telea fill; no target foreground is copied into the output.

| Window | Missing union pixel-frames, exact | 1/8 | 1/16 | 1/32 |
|---|---:|---:|---:|---:|
| alcaraz_hurkacz/031@129 | 9104 | 8763 | 6303 | 1766 |
| medvedev_struff/014@194 | 3798 | 3230 | 1963 | 560 |

These are sums across80 judged frames, not unique spatial pixels or invented coverage. Native per-player counts and requested/written/missing unions are also saved. Even the exact control has crop gaps; reproducing its saved composite hashes proves that the fallback remains identical to PR242.

Exact saved-RGB metric maximum differences are0.000001609 (Alcaraz) and0.000010014 (Medvedev), both below the unchanged1e-4 guard. No fresh generative re-render is performed here; the prior1e-3 fresh-render variation remains only a reported sensitivity. GPU CUDA/device/class, finite full-frame score coverage and declared memory pass; peak allocated memory6.963979GiB. Native both-window smoke projection549.249628seconds passed the800-second full bound. Shared timings are contaminated and are only resource/cost records, not performance evidence. One advisory progress-stall event occurred; the job completed and all substantive validators passed.

Native judge `20261026T124922Z-b47be797`, code revision `c2fa3b48`, source SHA256 `27c5a38deed86cd1cfd9d8a511164b2e8af9f7145a9b13327e1d9f58115b3a84`, gpu5 / NVIDIA RTX 6000 Ada Generation, UUID `GPU-07aa7586-7116-8c7b-1e9c-e5dbdc74b162`. Published archive SHA256 `60e8c42b40bbb93c47288dec31afc1663cc4663031ed81ebd8df0da4c41ef75e`. The result includes LPIPS/DISTS backbone hashes, environment/model provenance, all input hashes, per-frame scores and hashes of original/scored masks; source snapshot/specification checksum are preserved with the job.

New GPU ledger CLOSED: **373.340391/1800 admitted seconds = 0.103705664/0.5 GPU-hours**. Both replaced exclusive requests were verified cancelled before admission and cost0GPU. CPU preprocessing/edge ledger CLOSED at721.524817/3600 conservative CPU-seconds, including failed smoke and400local allowance. GPU judge additionally records439.585437 child CPU-seconds across smoke/full, separately available; even adding these to the conservative preprocessing ledger gives1161.110254CPU-seconds before small supervision overhead. Prior budgets remain closed; no unused allocation is transferred to further work.

Validation: configured Ruff and mypy pass;11 focused tests pass. Full local suite:887 passed,24 skipped, one pre-existing macOS protected-path hook assertion fails unchanged. The process-visibility test passes with read-only escalation. Linux CI is the merge gate. Native GPU and CPU full validators pass on the real data/components, including independent container-only edge-mask reconstruction.

### Scope and stop

This closes the three tested joint matte recipes under the two-window pilot criterion and rejects the fixed small-ramp carrier variants for promotion. It does NOT establish a full12-window statistic: ten windows remain untested for this changed generation/matte combination. It is not a universal compression optimum or proof that every smooth carrier/decoder fails. Binary masks are not inherently incompressible, and a sharpened decoded boundary is not guaranteed to equal the original boundary.

Target-derived Telea background remains granted. Source/replay independence remains unresolved; both familiar broadcasts, exploratory115px and historical step5 failure remain explicit. This is fidelity-only (LPIPS/DISTS), not the experience track, independent confirmation or an end-to-end codec claim. Stop because both authorized decisions are reached. No further computation, fitting, generation or full12 run is launched.

Post-run observability repair: publish native progress after each completed
window/archive, using the existing fleet helper, to avoid the advisory stall
seen while the completed study was healthy. This adds no recurring model call
and changes no source pixels, masks, scores, rates or decision. Native evidence
remains pinned to c2fa3b48; the read-only bundle retains that executed code and
its input/output hashes. No GPU re-run is needed or launched for this logging
change; normal static/focused checks cover the final source.

Immutable joint-study publication: `/home/itec/emanuele/Datasets/pointstream-data/tennis/k6b-joint-matte-edge-20261026T131206Z`. Metadata bundle SHA256
`e39c6031cbc0bd54fffc06d2dcca25dacb4d585c993fb587740fc3ac07ca48be`. All8 raw archives and54 metadata files
passed checksum verification and are read-only. The bundle's GPU/CPU executed
code-file hashes exactly match the two native result receipts. Existing source
files, weights and earlier publications were not overwritten. Final repository
source adds only the documented native progress callback after those results.

Publication PR: https://github.com/emanuele-artioli/PointStream/pull/243. Experiment stopped because both authorized decisions are reached; merge-commit auto-merge follows final Linux CI. All current jobs are terminal; no further computation or recurring model monitor.

## 2026-10-26 — Receiver-silhouette audit: preregistration

Card: [receiver-derived silhouette](../cards/k6b-receiver-silhouette.md).
New branch `codex/k6b-receiver-silhouette-audit`, base58fec838. Human request
verified in planning turn01a178c8-70d6-7ee3-91ad-3c0380fa4328: audit prior
answers, pursue a selective semantic win, keep players and ball as the broader
system objective. No new chats/agents. Prior ledgers remain closed.

Before opening any new quality: audit actual full-generation archive identity,
all saved timestamps/RGB shapes, native input geometry and prior experiment
contracts. CPU measurement start2026-10-26T15:20:57Z; deadline17:20:57Z.
CPU cap3600 aggregate seconds; GPU cap2700 admitted seconds, reserved450 smoke
+1650 pilot+300 validation+300 failures, only if the complete anti-circular
receiver contract passes. Stage bounds<=45min. The hypothesis, competing
explanations, >=3/4 strict four-cell bar, exact all240 correction/control and
paid accounting are frozen in the card. No third setting. No inference before
input coverage and isolation gates. Missing required RGB is a stop, not a
license to substitute source pixels or an undeclared temporal predictor.

Frozen raw archives from PR242 (source8cca9b19): Al job
`20261026T075716Z-f0549307`, SHA256
`6a3f620b394871154290340d7700831aea62f098e2569bf0d130e6735a4a1ef1`;
Med job`20261026T075723Z-e962542f`, SHA256
`ba752111fb9e2604a34dc560a06586cf820013ab398b6c834aac54546e6881e1`.
Canonical records are cited for provenance only, no submissions/replays.
Original immutable publication:
`/home/itec/emanuele/Datasets/pointstream-data/tennis/k6b-reference-quality-20261026T093850Z/raw/generated-full-{al,med}-0.tar`.
Frozen window source SHA256 Al
`054e4535638d8dfa3dee69258f45c231aaeab2241b91309d449be7b0934cef76`,
Med`a2898b5cdfb463f59b5084fddfe23a2442cf2ba5feaaddb9d5414873e52dd0be`.
They are evaluator inputs only, forbidden to the proposed predictor.

### Evidence matrix: prior answers, verified against reports and code

| Question / evidence | Predictor input and decoder information | Observed result | What remains unanswered |
|---|---|---|---|
| H5p4d: segment decoded pixels, pay initialization and exact corrections ([report](h5p.md#2026-10-24--h5p-4d-final-accuracy-and-report); `experiments/visor/h5_decoder_mask.py`) | Real raw pre-enhancement DCVC RGB; installed B1b SAM3.1 tracker; actually coded first s=8 union; later masks only in encoder XOR residual. Temporal shape coder, seed and two config bytes billed. | One valid stock point/item:27.30/61.33/30.00/13.29kbps, all above2kbps. Corrections exactly recover reduced union. Full curve stopped on native codec repeatability; initially circular enhanced-RGB attempt excluded. | Independent fresh receiver SAM-mask repeatability unproven; no uncorrected neural quality/full BD-rate curve. Different domain, generated RGB, prompts and target grid here: related negative evidence, not the exact tennis answer. |
| H3e: decoder-derived hand/forearm outlines ([report](h3e.md); `experiments/visor/h3e.py`) | V1 coded MANO pose supplies hand silhouette; V2 uses target R for hand+forearm and is an oracle. | V1 fidelity failed; report explicitly calls decoder-side forearm mask untested. | No generated-RGB silhouette plus correction transport test. |
| K6 tennis pose warp ([report](k6.md), oracle variants; `experiments/tennis/k6.py`) | Skeleton warp of target/reference appearance; source-player mask and mask-height bandwidth; references selected per target from granted bank. | Pose/appearance upper bounds and reference-warp quality, with granted outline/history. | Not a fresh segmenter of unmasked generated RGB; cannot supply a free target-R initializer here. |
| K6b hands3m (PLAN) | Proposed decoder matte for hand renderer. | Withdrawn after rung4 failed even with matte granted. | Not a completed matte experiment; cannot count as negative segmentation evidence. |
| K6b player steps2–5 and clip-local diagnostic ([this file](k6b-players.md); `k6b.py`, `k6b_generate.py`) | RGB generation from reference+decoded pose maps; target R used in judge/compositor. | Conditional renderer evidence and historical step5 failure. | No mask prediction on raw generated RGB. Region/granted-background history is retained. |
| PR239/240 compressed dynamic mattes (`k6b_matte_cpu.py`, `k6b_matte_judge.py`) | Receiver decodes actual transmitted reduced target mattes. | Judging stopped at0/8; full-cohort maximum16/24<18. | Transmitted mattes are not inferred silhouettes. |
| PR241/242 reference/foreground video (`k6b_reference_video.py`, `k6b_reference_prepare.py`) | Sparse FULL RGB/alpha/key reference banks; separately, dense original segmented foreground video. | FULL-reference generated quality4/4 with exact matte granted; other banks1/4; dense carriers0/4. | Sparse reference alpha does not supply target silhouettes; no segmenter on FULL raw generated crops. Dense original foreground is not generated-RGB inference. |
| PR243 joint matte/edge (`k6b_joint_matte.py`, `k6b_edge_ramp.py`) | Paid decoded target matte packets or explicit alpha/key carrier. | Reduced mattes each0/4; bounded RGB ramps lose shape before compression; accurate luma ramps not smaller. | No new segmenter. RGB-key ambiguity is not a proof learned segmentation fails. |

Repository search covered `docs/experiments/k6*.md`, related cards, H3e/H5p,
PLAN and tennis generation/transport/judging code. No overlooked completed
experiment answering the exact raw-FULL-generated-tennis configuration was
found. This is a scoped repository audit, not a claim that the general idea
has never been studied elsewhere.

### CPU input gate result: stopped, no new GPU work

Both actual PR242 FULL-generation archive hashes match the preregistered
identities; all4 NPZ member hashes match native metadata. Every unit reports
240 generated frames but contains **only80 RGB crops at0,3,...,237**, shape
`(80,784,512,3)` uint8. Every player lacks160/240 required prediction ticks
(1,2,4,5,...,238,239),640 missing player-ticks total across both windows.
Each archive contains only the two NPZs and their two metadata JSONs, with no
retained full-context RGB member. The audit hashes every saved RGB frame and
records every present/missing tick; it never opens original masks or new
quality scores. This checks the specified publications, not every possible
unregistered temporary file on the fleet.

Executed generator revision8cca9b19 and current generator are byte-identical,
SHA256`b25e8a1bd3a9c4444244df3a1010cf8652cf374971af55ee55089cb03982a4e2`.
`k6b_generate.py` renders each native pose run, then retains only
`judged[row]` (unless max_frames smoke is requested). Native metadata says
240 generated/80 kept for every unit. Context RGB at the other ticks was
transient and is not in these publications. PR243's card already explicitly
states that saved crops contain only judged frames; its all240 matte billing
used externally transmitted masks and never required segmentation there.
We do not reinterpret “frames_generated=240” as retained RGB coverage.

Anti-circularity audit: `k6b.py:prepare_window` derives full-run crop boxes and
pose maps from decoded keypoints (`run_boxes`), not R. Reference selection/
association and geometry were coded in the prior transport. The reference
preparer replaces only `reference`; FULL uses decoded unmasked RGB without
multiplying by alpha. Although that preparation reads alpha for other recipes,
alpha does not modify the FULL reference. `Renderer.render` takes reference
and pose maps, not target masks. Generation therefore has no direct target-R
pixel gate. Training/adaptation/source provenance remains a separate limitation.
The later exact-R composite is unsuitable segmentation input, as is its
source-driven Telea canvas. Missing raw RGB must not be recovered from either.

**Decision:** the exact new tennis question remains unanswered, but the
specified saved-output/all240-frame correction configuration fails input
completeness before segmentation. No candidate is promoted, no smoke submitted,
no new quality or correction-rate number is invented. It is an input/receiver-
contract blocker, not evidence of bad silhouette quality, excessive measured
correction rate, nondeterministic SAM/YOLOE, or an exhausted GPU allowance.
No installed-model choice, isolation/reset test, cost projection or model
payback has been empirically validated for this new configuration. Therefore
there is no basis for admitting the proposed full pilot under2700s.

A valid narrower question is whether a silhouette estimated from each of the
80 saved raw crops preserves the same80-frame LPIPS/DISTS advantage. That can
answer sampled composite quality, but cannot alone answer all240 exact
correction transport. A declared sparse-RGB predictor (for example, pose-
registered hold/interpolation, or an explicit empty predictor on unsaved ticks)
could define all240 masks using paid inputs. This would be a distinct receiver
profile, with its own timestamp/run-boundary behavior, fresh-reset proof,
correction bytes for **all** ticks and metadata bill; it must not inherit the
native all240 generated-RGB prediction claim. Neither version is silently
substituted or run here. The alternative is recover genuine retained raw RGB
elsewhere, if it exists; regenerating it is outside this authorization.

The next decision is consequently a concrete receiver-profile choice or raw
artifact recovery, not another generator/model sweep or a claim that semantic
players are conclusively closed. The broader objective remains a selective
fully accounted win that can grow to players+ball. No background/ball
integration is run because this pilot has no pass. Any later integration must
use an achievable coded background informed by T1's billed-split failure,
retain ball mode, and compare equal total rate with fair-adaptation anchors.

All prior source, training/checkpoint and result artifacts remain unchanged.
Target-derived background is still granted; familiar two-match/source/replay
uncertainty,115px exploratory status, historical step5 failure and granted-
matte history remain explicit. No full12, independent confirmation, whole-
codec win or universal rejection of decoder-derived silhouettes is claimed.

### Closed resources and verification

New GPU ledger CLOSED at0/2700 admitted seconds: no new fleet job, inference,
judging, fitting or generation. The two input jobs were independently checked
with native `ps-fleet status`: both remain complete with passed gates. Native
`ps-fleet inspect` returned all6 hosts with0 probe errors; no admission followed.
There is no task-owned running process, pending request or model monitor.

CPU archive audit measured4.503882CPU-s /1.825985wall-s. Configured Ruff/mypy/
full pytest measured160.596312 child CPU-s /88.505228wall-s. Reserving600CPU-s
for the remaining local inspection, isolated process-test recheck, Git and
publication gives **765.100194/3600 conservative CPU-seconds**; closed within
2h wall. Unused new/old budgets do not transfer to another test.

Ruff and mypy pass (224 sources). Full pytest:885 pass,25 skip,2 known macOS
limitations: sandbox denies `ps`, and the unchanged protected-path test expects
Linux `/home/.../Datasets` symlink resolution. The process-suspension test
passes with read-only process visibility outside that sandbox. No production
code changed; the audit's actual archive SHA, member SHA, complete array shape,
timestamp uniqueness/range and all4 stride3 schedules pass. Linux CI remains
the merge gate. No new tests mirroring documentation were added.

Immutable audit publication:
`/home/itec/emanuele/Datasets/pointstream-data/tennis/k6b-receiver-audit-20261026T152605Z`.
Bundle SHA256`475f703b1812fde02e2644055c03dd126e5489d35e4aa5075596668c0e877f71`.
All13 manifest-listed files hash-verified, files444/directories555. Contains
standalone audit/check scripts, full timestamp/missing-tick/frame hashes,
code-file hashes, six-host probe, closed ledger, card/report and check logs.
The upload initially stopped after extraction because remote Python3.10 lacks
`hashlib.file_digest`; the existing directory was inspected, verified using
SHA256 read-bytes and finalized read-only without replay/overwrite. This
publication-only recovery is within the600CPU-s allowance; no inference ran.
The original raw generation archives remain in their prior immutable directory.

Publication PR: https://github.com/emanuele-artioli/PointStream/pull/244.
Merge-commit auto-merge enabled, subject to required Linux CI. Experiment
stopped at the CPU input gate; no further task computation is scheduled.

## 2026-10-26 — Sampled receiver-silhouette diagnostic: scope amendment, rule

Planning explicitly selected the narrower80-tick quality diagnostic after the
completed CPU input audit, under continuing human “Get started” authority.
Audit and its765.100194CPU-s/0GPU remain preserved. Cumulative receiver-study
caps stay3600CPU-s/2700GPU-s; no new budget is created. New scoped branch
`codex/k6b-receiver-sampled-pilot`. The final amendment in the existing
[card](../cards/k6b-receiver-silhouette.md#scope-amendment-sampled-zero-matte-diagnostic-planning2026-10-26)
freezes one YOLOE26x candidate, keypoint-only instance choice, read-isolated
fresh reset reproduction, exact control<=1e-4, measured smoke resource gate,
2000s initial reservation and>=3/4 strict cells BEFORE inference or quality.

This specifically tests zero-dynamic-matte composited quality on actual saved
raw crops at0,3,...237 only. It does not repair or answer missing240-frame RGB,
exact correction rate, full-stream determinism or wire delivery. Hypothesis:
receiver-visible generated shape preserves enough fidelity to keep>=3/4 cells.
Competitor: generated shape/context segmentation deviates enough to erase the
advantage even without a matte bit bill. Target masks/source/canvas stay
unreadable to the predictor; only evaluation gets originalR/S/Telea after masks
are frozen. All original scope/provenance/background limitations remain.

Frozen model hashes (CPU availability audit, no inference): YOLOE26x
171640453bytes SHA256`d08d390a08f98195f7c87807839fe4ff93a5491645fef1bc3bf0700efafdd639`;
MobileCLIP2 253794476bytes
SHA256`35d7f213e4d75f38514e4656ad3cb91158bd33e3805d8ac349f23b186f66982f`.
Both already installed in Models/YOLO; main packed environment remains
`32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661`.
Actual Linux read-isolation function passed a CPU allowed/denied-file test on
LandlockABI4/kernel6.8. No weights were loaded. Static checks and9 focused
regressions pass. Geometry projection clips to saved crop support; no source
pixel invention. Independent KPS decode confirmed units already have full
native names, preserving original player order.

Pre-admission six-host inspection: compatible Ada hosts5/6 considered; all
were busy, gpu5 had23–28GiB free. Freeze shared12GB+8GiB margin, no timing claim,
fresh reset pixel/mask hashes mandatory. First native reservation2000s
(300 smoke/1500 full/40 validator/160 overhead),700GPU-s recovery unspent.
GPU-stage CPU validators additionally require smoke<300CPU-s and full<1700CPU-s;
combined with765.100194 prior conservative audit cost and remaining local/
validation/check allowance this fits3600. No recurrent model wakeup; attended
native fleet wait only, no verified completion-triggered model wakeup available.

Submission ID`20261026T154114Z-808dc4f0`, source`a7f62e1e`: CLI reports
outcome unknown after publisher rejected a Models path outside the permitted
external-data input root. No automatic replay. Inspect canonical status before
any replacement; existing weights must be staged by hash under Datasets first.

Recovery inspection for808dc4f0: canonical status says unpublished; direct
transfer inspection proves only`source/` and`events/`, no ready/spec/run record.
Workers cannot admit it, so0GPU; retain its transfer. Existing immutable weights
were copied/hash-verified into new read-only
`Datasets/pointstream-data/tennis/k6b-receiver-inputs-20261026T154300Z/`.
Only input locations changed, original Models preserved. This is a verified
new submission after an unpublished transfer, not an unknown-job replay.

Corrected native job`20261026T154300Z-b22eabc3`, source`a7f62e1e`, admitted
reservation<=2000GPU-s of unchanged2700 cap. Published pending;
sourceSHA256`47638c4e5d896b785d416c53d86fe01d06ef657664774fc219480baeb06d7522`,
native specSHA256`b4f91f4d66061f7f6ea62ff37604fa2a6e6907f8767afea1958a7c8477dfb76b`.
Local spec`/tmp/k6b-receiver-sampled-spec.json`; corrected submission receipt
`/tmp/k6b-receiver-sampled-corrected-submission.json`. No new automation was
created: the legacy CLI inherited chat_id and prints a heartbeat suggestion,
which is intentionally not followed; waiting is a native bounded CLI command.
Full local suite887pass/25skip, same2 Mac restrictions as audit; Ruff/mypy and
9 focused tests pass. Full-stage CPU validator<1700, smoke<300; actual ledger
will include all admitted setup/failures/validators, plus measured local checks.

Accounting sensitivities fixed before reading new quality: primary remains7200s
with common base and segmenter. Also show base-only, segmenter-only and combined
installation charges, plus1800/7200/28800s amortization of actual bank/adapter
and installation bytes with keypoint rate unchanged. These are fixed-quality
accounting counterfactuals, not claims the observed80 frames generalize to those
durations. Use the original measured anchor envelope; no quality extrapolation.
A supported measured rate crossing can bound payback; censored crossings stay
unresolved. These sensitivities cannot select or promote a different candidate.

Job b22eabc3 FAILED smoke before model output. Canonical admitted GPU duration
10.398987293243408s; native command0.717812s /child CPU1.485176s. Parent logged
only a child exit, so no predictor failure detail or scientific result is
claimed. The initial logging gap is retained rather than invented after fact.
A CPU-only reproduction from/tmp proves the packed child cannot locate the new
module without an explicit frozen-source PYTHONPATH, while `--help` succeeds
with it. Correct this bootstrap, publish child logs before raising, and permit
one bounded instrumented recovery within the same cumulative cap. No model,
threshold, images, geometry, metric or decision-rule change. Remaining GPU
2689.601013s before recovery;2000s new reservation fits. Failure metadata in
`/tmp/k6b-receiver-sampled-terminal.json`; bootstrap proof in
`/tmp/k6b-receiver-sampled-bootstrap-proof.json`.

Recovery job`20261026T154832Z-50c37ae0`, source`9e9fb2ca`, published pending,
<=2000GPU-s reservation. SourceSHA256
`b446dcbe6c0d3392d7cb05c7306a3d86d5fc124386168a663bd36cf39ee0768e`,
native specSHA256`bbda0c7b02db9784ebe299d15c63f7f9d990b19bde1534b36280a0682e6cd21d`.
Receipt`/tmp/k6b-receiver-sampled-recovery-submission.json`. Explicit empty
chat_id disables the legacy heartbeat suggestion; no automation created.
Cumulative admitted GPU before this request10.398987293243408/2700s.
Additional configured local checks measured164.913063CPU-s; audit765.100194,
failed smoke1.485176 and300 new local/preflight allowance leave ample measured
headroom for gated smoke/full CPU and actual validators under3600. Same primary
criterion, no inference outputs or new quality scores opened yet.

Instrumented recovery50c37ae0 FAILED the required privacy probe before model
loading: recursive/dev allowance also exposed the fleet's/dev/shm source cache.
This is an isolation implementation error caught by the gate, not segmentation
or quality evidence. The probe read one source byte, then aborted before any
prediction; nothing from that attempt is accepted. Native admitted duration
10.275501728057861s, cumulative20.67448902130127/2700GPU-s. Full traceback is
preserved in native smoke command/worker log; no unisolated fallback.

Narrow device allowlist now contains only concrete GPU/runtime nodes, never/dev
recursively. A CPU-only test using the actual fleet source-video and target-
mask cache paths proves both reads denied while an allowed file still reads.
The candidate/model/threshold/geometry/scoring remain unchanged. This concrete
privacy repair is checked before one further native smoke, using released
unspent reservation; remaining2679.325511GPU-s supports a2000s reservation.
Ruff/mypy and9 focused tests pass. Full prior tests remain valid apart from this
narrow allowance change; required real smoke must prove all4 forbidden reads
and fresh reset outputs before evaluation. No quality scores opened.

Narrow-device job`20261026T155233Z-07cca552`, source`7aba4282`, published
pending,2000GPU-s reservation. SourceSHA256
`d804c19d8ae1d17ff08b5f6031f3b998b648dc73387e60a31463fe0b6d972dd5`,
native specSHA256`b0fc0755d9b371c735e5390799146f69f36b3a718adb39febed5959c0465a62d`.
Receipt`/tmp/k6b-receiver-sampled-private-submission.json`; cumulative prior
admitted20.67448902130127/2700GPU-s. Explicit empty chat metadata, no monitor.
Actual isolation-cache CPU proof`/tmp/k6b-receiver-isolation-cache-probe.json`.

Job07cca552 completed actual isolated prediction twice: all4 forbidden reads
denied,4players×2ticks, exact fresh-reset input/output hashes, CUDA. Then the
evaluator failed before scores because it assumed a`publish/` wrapper that the
native transport archive does not have. Correct to the same root detection as
prior native judges; actual archive CPU extraction confirms the KPS paths,
plus a regression checks both supported layouts. No mask/model/prompt/quality
change. Prediction receipt/masks retained in canonical smoke/partial.tar;
this is valid infrastructure smoke evidence, not a quality result/full pilot.

Native admitted60.83878016471863GPU-s; cumulative81.5132691860199/2700.
Native child CPU53.848476s. Remaining2618.486731GPU-s supports2000s after the
CPU-proven path repair. No full stage or neural quality scores completed.

Scoring-layout job`20261026T155754Z-bbc4a300`, source`694dd06c`, published
pending,2000GPU-s reservation. SourceSHA256
`eeff18011d4c7b901e700bd38991adda6ae9a538b99629e49666f33bdf622945`,
native specSHA256`0f0082564e266c7563bef2da4081ce7ce68ba47b5364931de332f9d4037edabe`.
Receipt`/tmp/k6b-receiver-sampled-scoring-submission.json`; cumulative prior
admitted81.5132691860199/2700GPU-s. No configuration changes after model output.
Visual implementation check of the retained two-frame Alcaraz partial confirms
actual masks cover generated players and leave crop background visible; this
is not quantitative quality evidence and does not select a new profile.

Job bbc4a300 scoring command completed both-window smoke successfully:
86.014590CPU-s, full projection1345.326117s<1500. The template mistakenly kept
`k6b_joint_matte` as validator; it searched for judge.json rather than the new
receiver.json and stopped the fleet before full. Parent job remains FAILED.
Native admitted90.88955974578857GPU-s; cumulative172.40282893180847/2700.

Canonical scored smoke archive SHA256
`0d7397fd25f6c20d09494af5d71064353d40d335853e1743f610491431aa1637`
passes the intended receiver validator independently on CPU:31/31 checks.
Additional per-frame neural-array/mean and source/shape-hash checks pass35/35.
Both real controls, denied reads, fresh resets, actual masks, coverage, GPU
class/memory and the measured resource projection pass. This is qualified
infrastructure smoke evidence, not the full quality result. Only validator
invocation/strength changes; no image/model/prompt/metric/rule change. Next
native job rechecks this same representative smoke before full. Remaining
2527.597171GPU-s supports2000s; no old ledger is reopened.

Correct-validator job`20261026T171827Z-cf4da9e0`, source`115b9988`, published
pending,2000GPU-s reservation. SourceSHA256
`71542310799c36f127ece0e1ac36772105c1ec5aca836bfda425e8466732bb7f`,
native specSHA256`dc2f5a8e84421c58b83cc8a1778728bb2089bebfb0b5d9606af867a9624d7d33`.
Receipt`/tmp/k6b-receiver-sampled-qualified-submission.json`; prior admitted
172.40282893180847/2700GPU-s. Correct module validator and stronger35 checks;
no scientific configuration change. Re-inspect current native deadline/state
before waiting after the client interruption; never replay this request.

Correct native smoke gate cf4da9e0 PASSED35/35 substantive checks; full stage
started on gpu5. Both denied-read and fresh-reset mask checks, exact controls,
actual per-frame score arrays and measured full projection passed natively.
Native current state recorded in
`/tmp/k6b-receiver-sampled-qualified-gate-status.json`. No quality-dependent
configuration choice. Resume bounded wait after acknowledging only the expected
shared-contention event; job's own2000s/44min bounds stay unchanged.

## 2026-10-26 — Sampled receiver silhouette: complete, fixed candidate fails

**Decision:1/4 strict wins, below the frozen>=3/4 pilot bar.** This is a real
inference-and-recomposition result on both original windows/all80 archived
judged ticks each, not an IoU proxy or unchanged-quality substitution. No
further segmentation, generation, fitting, full240 recovery or integration is
launched. This closes this YOLOE26x/person/selection/projection profile; it does
not prove every decoder-derived silhouette route fails.

### Actual quality and conditional paid rate

| Window | Rate kbps | Predicted LPIPS | Predicted DISTS | Strict wins |
|---|---:|---:|---:|---:|
| Alcaraz–Hurkacz031@129 | 5.237128889 | 0.262862106748 | 0.222578307986 | 1/2 (DISTS) |
| Medvedev–Struff014@194 | 7.640872222 | 0.242911524930 | 0.236507502943 | 0/2 |

All4 primary cells are bracketed; no unresolved primary cell or extrapolated
quality. Actual exact/granted-matte controls remain4/4: Alcaraz LPIPS/DISTS
0.245873567472/0.220013449341; Medvedev0.172194861869/0.186108257622. Every
saved control composite RGB hash matches; maximum per-frame metric differences
are3.939867e-5/2.163649e-5, below the unchanged1e-4 guard. No fresh generative
re-render and no tolerance relaxation. There is one method operating point per
window, so no method BD-rate curve is defined. Original achievable SVT-AV1/DCVC
envelope and originalR/S remain unchanged.

Rates pay the actual PR242 FULL reference banks, original KPS geometry/timing/
identity streams and match adapter/guider over7200s. Alcaraz components are
3.530000 keypoints+0.053515556 references+1.653613333 adaptation kbps; Medvedev
5.903333333+0.075770000+1.661768889. Dynamic-matte payload and content-specific
prompt/initialization bits are0 for this fixed installed person profile; no
source mask, source-box prompt or source-driven crop is supplied to the client
predictor. Common base+segmenter/text runtime installation is an explicit
assumption. These are conditional sampled-frame rates, not a demonstrated
complete240-frame wire/decode profile. All240 KPS ticks remain billed; there
are only80 archived generated-RGB ticks, and no new240-frame matte/correction
stream was constructed.

### Receiver contract and failure diagnostics

Both independent fresh-process predictions reproduce all inputs and masks
bit-for-bit for each of the4players×80 archived ticks. Both actual native
workers use CUDA and deny original source videos, target mask-bearing transport
and quality-control archive reads. They see only unmasked generated crops,
sanitized paid KPS, installed fixed profile/weights and runtime files. Raw crop
context was retained. Evaluation reads originalR/S/canvas only after masks are
frozen. Predictor determinism is demonstrated for these saved pixels on this
hardware, not for new generative outputs or an independent all240-frame client.

| Window | Mean per-tick union IoU | FP pixel-frames | FN pixel-frames | Missing RGB exact / predicted |
|---|---:|---:|---:|---:|
| Alcaraz | 0.650181056 | 79,499 | 272,155 | 9,104 / 0 |
| Medvedev | 0.504735064 | 25,922 | 312,532 | 3,798 / 0 |

These union IoUs describe shape, not the decision metric. Empty masks occur on
0/80 and2/80 Alcaraz unit001/unit006 ticks, and1/80 and63/80 Medvedev unit003/
unit004 ticks. In particular, the fixed detector frequently misses Medvedev's
unit004 generated person. This is a measured cause of lost support for this
profile, not evidence that no pose-prompted segmenter could recover it. No
post-score confidence/threshold/model sweep was run.

Predicted masks are projected and clipped using only the already-billed pose
crop rule, so their requested support has no unavailable RGB. Exact target-R
control still has the previously documented crop gaps; those pixels retain the
identical native Telea fill. Zero predicted missing-RGB counts do not mean full
original-body coverage: false negatives and empty predictions are explicitly
counted. No source foreground pixels filled gaps. Per-frame predicted-mask,
source RGB, source-player-mask, R/S, control/composite and score hashes are
preserved with the actual mask arrays and all160 judged-frame scores.

### Installation and amortization sensitivities

The actual installed detector171640453B+local text encoder253794476B total
425434929B. Charging that installation over7200s adds472.705477kbps; base alone
48878567B adds54.309519kbps, combined527.014996kbps. At7200s: base-only has0 wins
(2 bracketed losses/2 unbracketed), segmenter-only and combined have0 wins/all4
unbracketed. In all installation-charged cases a measured cheaper/no-worse
anchor witnesses dominance in every cell; unbracketed cells are not relabelled
as interpolated losses. Combined rates are532.252124/534.655868kbps.

Fixed-quality common-installation accounting at1800/7200/28800s yields0/4,
1/4 and2/4 wins; none meets3/4. This changes byte amortization only, not the
observed quality or cohort. For the measured Alcaraz quality, combined-install
formal crossing horizons are>2740699s (31.72days, LPIPS) and>1207774s (13.98days,
DISTS). They are algebraic fixed-quality sensitivities, not evidence the same
profile/adapter works for those durations. Medvedev's measured crossings are
2.260237/2.283348kbps, already below its5.903333kbps KPS floor; no positive
amortization horizon repairs those two cells at the measured quality. This
does not change the primary7200s failure or establish any temporal generality.

### Actual native evidence, checks and closed budget

Completed native job`20261026T171827Z-cf4da9e0`, executed source`115b9988`,
gpu5 / RTX6000Ada, source/spec hashes in the preregistered job entry. Actual
smoke projection1163.881537s passed the1500s full bound; full command325.814103s.
These shared-GPU times are resource-cost records only, never latency evidence.
Peak allocated3.553785GiB stays within the12000MiB declaration. Runtime Torch,
backbone hashes, weight hashes, all input identities and executed code-file
hashes accompany the result. Full raw archive SHA256
`cc67ec2cc4bb58ca2ec713002a13e63bcc8f01b16e89656e59e6a1f3b2c1a01c`.

Native smoke passes35 checks. The actual full archive independently passes34
checks locally and on CPU in the original packed environment/frozen executed
source: frame/mask arrays and hashes, fresh resets, source/hash coverage,
per-frame score shapes/finite values/means, exact control, both complete original
80-tick sets, memory/class and support accounting. No stand-in score is used.
Full-stage process CPU366.902379s; successful supervisor's448.768725 child-CPU
snapshot already includes smoke and its validator, so it is not added twice.

| Job | Native state | Admitted GPU-s |
|---|---|---:|
| 20261026T154114Z-808dc4f0 | unpublished, inspected no ready/spec/run | 0 |
| 20261026T154300Z-b22eabc3 | failed child bootstrap | 10.398987293 |
| 20261026T154832Z-50c37ae0 | failed privacy gate before model loading | 10.275501728 |
| 20261026T155233Z-07cca552 | private/reset inference passed, evaluator path failed | 60.838780165 |
| 20261026T155754Z-bbc4a300 | scored smoke passed intended checks; wrong template validator failed | 90.889559746 |
| 20261026T171827Z-cf4da9e0 | complete; native smoke and independent full validators pass | 414.611974239 |

**GPU ledger CLOSED:587.014803171/2700s=0.163059668/0.75GPU-h**, all admitted
setup/smokes/failures/validation included. CPU ledger CLOSED at conservative
2334.860723/3600s: closed audit765.100194+additional measured configured checks
164.913063+actual native child snapshots592.741714+local full validation2.977481
+native full validation9.128271+600 local/preflight/analysis/publication allowance
+200 native supervisor/unrecorded-validator allowance. Failed source/model
transfers, source/checkpoint data and all partial results remain preserved.
Unused allocations close with this study and do not transfer. No running or
queued task job, generator, other chat/agent or recurring model monitor remains.

Configured Ruff/mypy pass (226 sources),10 focused regressions pass. Full local
suite887pass/25skip with the same2 known macOS restrictions (sandbox ps denial
and protected Linux symlink expectation); the process test passes with read-only
visibility outside the sandbox. Linux CI is the merge gate. Additive judge API
preserves old default four-matte behavior; the receiver path uses two variants.
No existing source data, weights, training or checkpoints were changed.

### Scope and stopping decision

The deciding failure is actual zero-matte silhouette/recomposition quality,
not prediction repeatability or exhausted GPU budget. The test answers the
narrow80-tick question for one fixed profile. Independent240-frame prediction,
exact correction transport/init/framing/identity cost and full wire evidence
remain unmeasured because160 raw ticks/player are absent. This does not infer
that corrections are cheap or expensive, nor a full12 result.

Target-derived exact-R Telea background remains granted. The broader purpose is
still a fully accounted tennis system with players and ball; ball-only is not
the intended endpoint. Source/replay independence, familiar two-match scope,
115px exploratory status, historical step5 failure and granted-matte history
remain explicit. No coded-background/ball integration or new full240 generation
is launched after this failure. This is neither an end-to-end win nor a
universal closure of receiver-derived silhouettes. Stop because the authorized
candidate's>=3/4 decision is now reached, preserve its evidence and report it
to planning.

Immutable publication:
`/home/itec/emanuele/Datasets/pointstream-data/tennis/k6b-receiver-sampled-20261026T173715Z`.
Bundle SHA256`f106e7235bd214e23845bf59d6fc3879166d67431ace070cd43566583829fce2`.
All100 files independently hash-verified; files444/directories555. Includes real
full/qualified-smoke/private-partial raw archives, source-identified mask/score
arrays, native execution/environment/gate records, failed traces, fresh decoder
proof, fixed-quality sensitivities, closed ledger, checks and report/card.
Published code-file hashes match both executed receiver and judge hashes.
Existing generated/source archives, Models and earlier publications were not
overwritten or deleted. All reported events were acknowledged; no wakeup exists.

Publication PR: https://github.com/emanuele-artioli/PointStream/pull/245.
Merge-commit auto-merge follows required Linux CI. The scientific study and
its cumulative resource ledgers are closed; no further inference is scheduled.

## 2026-10-26 — Pose-prompted receiver SAM: frozen bounded diagnostic

New study authorized through planning after PR245 closed. Branch
`codex/k6b-receiver-sam-pilot`; [self-contained card](../cards/k6b-receiver-sam.md).
Fresh caps2700 admitted GPU-s/3600 aggregate CPU-s/2h wall. Previous ledgers
remain closed. No new GPU submitted or quality opened at this preregistration.
Hypothesis: paid-pose initialization removes detector nonrecognition while
retaining rendered-player fidelity. Competing explanations: shape still differs
from source; tracking includes generated context/background; pose prompt
misaligns; nonrepeatable predictions invalidate receiver correction decoding.
The exact granted control4/4 is the upper bound; SAM is tested only as the
silhouette component. Pass>=3/4 strict bracketed LPIPS/DISTS cells on the same
Al031/Med01480 ticks at actual conditional total rate; no tuning after scores.

CPU audit:8 generated Med004 crops at ticks0,3,30,60,90,120,180,237 visibly
contain the player with broadly aligned paid keypoints. YOLO selection cannot
produce an empty mask from a nonempty candidate, so empty output is compatible
with detection/confidence failure rather than absence of rendered person in
these inspected frames; no all-frame detector-cause assertion. No target masks
or new quality used. Figure and raw pose metadata `/tmp/k6b-sam-med004-pose.png`
and `/tmp/k6b-sam-visual.json`; measured3.465341CPU-s plus conservative audit
allowance to be charged. Existing standalone mask wrapper is not itself the
point API; CPU source inspection verifies installed pinned
`VideoTrackingMultiplexDemo.add_new_points` explicitly converts points through
the interactive head then adds tracker state. Native corner labels2/3 plus
positive torso points are frozen in card. No new dependency/model substitution.
Checkpoint `/home/itec/emanuele/Models/SAM/sam3.1_multiplex.pt`,3502755717bytes,
SHA256`0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`;
source2345a4ad109ac29c569da749c91d84f10dc08c40, existing packed environment
32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661.
Existing table qualifies Ada/A6000 mask tracking; fresh pose path must assert
all loaded weights/CUDA/native flash kernels and independently repeated masks.

The new module preserves historical YOLO implementation unchanged. Predictors
read only independently decoded KPS/raw generated crops/checkpoint in the same
Landlock contract, never target/source/mask/control. Evaluator opens R/S/Telea
only after two matching prediction subprocesses. Exact control tolerance1e-4,
original composites/crop order/quality/anchors retained. Empty/FP/FN/IoU and
missing RGB diagnostic accompany actual quality. Fixed profile reset20 saved
ticks is2.4s Al/1.2s Med, not dense240-frame tracking. Original missing160 raw
ticks/player and all240 correction costs remain unmeasured. Common base+SAM
installation explicit;7200s installation charges/payback sensitivity separate.
Target-derived Telea background remains GRANTED, source/replay independence
unresolved,115px exploratory, historicalstep5 and granted-matte limitation
retained; no end-to-end or independent confirmation claim.

Submission: job **20261026T181328Z-57b68814**, pending at receipt, executed
revision20b16dad177302998415722616fb16b0f156c7bb,
sourceSHA`cd24b1509be1197da3f1a9dc16eb1f0cd28d9a9940c4f2fce8cd106230f88014`,
specSHA`48afe7fc599e003012216489a109e682183491ca2018487a312033675aa51a85`.
Compatible pool gpu2/3/5/6 (A6000/Ada), all6 inspected, job ledger checked.
Native2000s reservation300smoke/1500full/40validator/160overhead; remaining700s
recovery reserve within cumulative2700 only after known failure. Receipt
`/tmp/k6b-sam-submission.json`, spec`/tmp/k6b-sam-spec.json`. Read-only staging
of existing3502755717B SAM checkpoint hash-verified,5.572351CPU-s; no Models
mutation/download. Static Ruff/mypy and5 focused prompt/projection tests pass.
No automatic completion wakeup verified: attended native bounded batch wait,
no recurring model monitor. Conservative preflight allowance150CPU-s plus
visual3.465341 and weight check5.572351, actual check/runtime snapshots later.

Smoke gate PASSED on gpu6 RTX6000Ada: native93.832730s, actual child CPU
113.830771s; application snapshot113.190485CPU-s. All931 weights load, actual
bf16 `pytorch_flash::flash_fwd_kernel` events observed, peak3.469430447GiB,
4 forbidden reads denied/run and two fresh predictions bitwise identical.
Exact-control maximum per-frame differences8.940696716308594e-7 Al and
1.168571836795973e-7 Med, unchanged tolerance1e-4. Conservative projected
full931.171563s<1500, native substantive validator passed and full promoted.
No prompt/profile/tolerance changes after smoke. Shared timings resource cost
only. Metadata `/tmp/k6b-sam-smoke-deadline/`; native continuing wait handle27105.

Native wait raised stall event3c525df3123b5a72b0a950d1c51d4752 while full
prediction/scoring had not yet published a window checkpoint. Authoritative
status after inspection remained running and `last_progress` advanced to
1793038934.0818994, after the alert. Event acknowledged, same job continued;
no replay, migration, profile change or extra allocation. Wait handle48947,
remaining native bound1800s. This is supervision metadata, not quality evidence.

## 2026-10-26 — Pose-prompted receiver SAM: complete, tracking profile fails

Decision: **2/4 strict bracketed wins <3/4**, so the frozen zero-matte SAM
profile fails this pilot. Medvedev–Struff's detection failure is removed, but
Alcaraz–Hurkacz unit006 loses its tracked mask. This rejects this single
prompt/reset/tracker profile, not every receiver segmenter or prompt schedule.
No generator, fitting, new weight/environment, full240 recovery, full12 or
background/ball integration was run. All compute is terminal; budgets CLOSED.

### Actual inference, receiver contract and quality

Job **20261026T181328Z-57b68814**, COMPLETE, executed
`20b16dad177302998415722616fb16b0f156c7bb`, gpu6 RTX6000Ada, original packed
runtime SHA32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661.
Source/spec hashes and actual native commands/inputs are in the submission
entry above and immutable raw archive. Full native command453.892643s; peak
3.613847733GiB. Shared timings describe resource cost only, not performance.
Full raw archive SHA256
`c44b2ab0828612dcf3602746443a4831c88d265318ace0e54c717de2a51e732a`.

All4 player units have80 raw unmasked generated RGB ticks0,3,...237. The
worker independently decodes paid KPS and computes geometry/prompts. Both fresh
subprocesses reproduce every input/output mask hash bit-for-bit, use actual
CUDA/bf16 flash kernels, load all931 weights, deny all4 forbidden source/control/
mask-bearing inputs. The evaluator sees R/S/source/Telea only after prediction
is frozen. Native smoke36checks, local independent full35checks and original-
packed-runtime/frozen-source CPU full35checks pass. No oracle mask initializes
SAM. Profile details and20-tick resets remain exactly as preregistered.

| Window | Actual conditional total kbps | LPIPS | DISTS | Strict cells |
|---|---:|---:|---:|---|
| Alcaraz–Hurkacz031@129 |5.237128888889|0.32516148852074284|0.25150804296135904|LOSS / LOSS|
| Medvedev–Struff014@194 |7.640872222222224|0.18295907434004696|0.19509378224611282|WIN / WIN|

These are real native LPIPS/DISTS on the original R/S scoring masks, all80
judged frames/window, identical crop order, source conversion and achievable
SVT-AV1/DCVC anchor envelope. Envelope errors at these rates are
Al0.25947728608433873/0.23148708141753563 and
Med0.2017129652600654/0.2113499321414575. No unbracketed primary cells, quality
extrapolation or method BD-rate (only one point/window). Al LPIPS additionally
has a real cheaper/no-worse DCVC21s2 witness1.445384305kbps/0.2998191253.

Exact/granted-matte control remains4/4 with actual means
Al0.24587356747176417/0.22001344934105874 and
Med0.17219486186921676/0.18610825762152672. Every saved composite RGB hash
matches. Maximum per-frame metric deviations3.93986701965332e-5 and
2.1636486053466797e-5 remain below unchanged1e-4. No tolerance relaxation or
fresh generative rerender.

### What failed

Empty crop masks: Al0010/80, **Al00651/80**, Med0030/80, Med0040/80.
For Al006 all four reset frames0,60,120,180 yield nonempty masks, then each
track collapses after6,10,8,5 saved ticks respectively; empty counts by segment
14/10/12/15. Thus the observed failure is tracking/presence loss after successful
initialization, not missing initial pose support or receiver nondeterminism.
Inspection of raw generated RGB at lost ticks18,90,144,195 still shows the
player. This is an observation on those inspected frames, not an assertion
about every empty tick. No post-score prompt/reset/threshold tuning.

Mean per-tick union IoU0.5183798257 Al /0.6881396313 Med; pixel-frame FP
73,790/55,484 and FN405,164/168,338. These diagnose shape and do not substitute
for quality. Projected predicted masks never request pixels outside paid crop
support because projection clips to that rectangle; this does not imply target
body coverage. Original exact control still has unsupported requested pixels;
no source foreground is invented to fill unavailable generated RGB.

### Real sampled correction payload, separate CPU follow-on

After predictions were frozen, one CPU codec encodes temporal-XOR of the
predicted-vs-exact-R residual at the **80 archived ticks only**. It bit-packs
full1080p player masks and uses the existing Zstandard framing implementation,
with window/unit identity, explicit native ticks/shape, predictor and decoded
mask checksums, and constant recomputable prompt/reset profile. Independent
fresh decoder subprocess receives only the actual predictor archive, paid KPS
and transmitted packet, no target/source mask input. Every decoded R frame
hash equals exact target R. Separate exact sampled-R payloads use the same
coder, are retained and round-trip verified. This is one coder, no sweep or
claim of optimal correction compression.

| Window | Residual bytes, all framing | Sampled residual kbps | Same-coder exact-R bytes | Exact-R kbps |
|---|---:|---:|---:|---:|
| Al031,9.6s |147144|122.620000|80884|67.403333|
| Med014,4.8s |127477|212.461667|61649|102.748333|

Native timing is physical0,3,...237 at25/50fps. These rates are sampled80-tick
payloads divided by the original9.6/4.8s windows, **not all240 measured rates**
or dense extrapolations. Initial prompts and resets are recomputable from
already-billed KPS/profile; packet identity/framing/checksums are included.
The exact-R baseline is sampled equally; neither substitutes for the original
all240 matte transport. This specific residual is larger than direct sampled R.

Adding these actual sampled residual bits to the unchanged billed KPS/reference/
adaptation would give127.857128889/220.102538889kbps. Exact decoded masks would
recover the existing exact-control composition under the same available RGB/
Telea/crop-order rule; no fresh GPU corrected-quality run is claimed. All4
resulting accounting points are outside the measured anchor rate bracket, so
no interpolated loss or extrapolated BD-rate is reported. All4 have actual
cheaper/no-worse SVT62s1 witnesses (Al11.996104167kbps, Med21.722354167kbps).
This coder offers no promising correction accounting at the sampled horizon.

One CPU wrapper initially expected a `window` field in the native MATTE header;
actual native timing is `shared_KPS1_window_header`. It stopped before encoding.
The fix validates native shape/output grid/timing and binds target/KPS to the
same SHA-verified transport member directory. Failed artifacts retained;
conservative120CPU-s charged. No GPU rerun. A successful15.824936CPU-s run is
retained, followed by17.660931CPU-s to also preserve/verify exact-R baselines;
packets are byte-identical, no algorithm or quality/profile change.

### Installation and limits

Primary rates bill original all240 KPS3.53/5.903333333kbps, actual FULL banks
0.053515556/0.07577kbps at7200s, adapter/guider1.653613333/1.661768889kbps;
no dynamic matte or nonrecomputable prompt/init bits. Common tennis base+SAM
installation is assumed shipped. Unique SAM checkpoint3502755717B costs
3891.950796667kbps at7200s; base48878567B costs54.309518889kbps. Charging both
produces3951.497444444/3953.901187778kbps,0strict wins/all4unbracketed with
real cheaper/no-worse witnesses. `segmenter_and_text` in the generic accounting
schema here denotes only the single SAM checkpoint; no text encoder is loaded
or billed twice (`--text` is an unused compatibility alias to that same path).

Fixed observed quality with common installation gives1/2/2wins of4 at
1800/7200/28800s. Al quality crossings1.242075166/1.990218435kbps lie below its
3.53kbps KPS floor, so longer amortization alone cannot rescue those cells.
For Med, formal combined-installation horizons must exceed5580723.954s LPIPS
or3897487.388s DISTS (about64.6/45.1days), fixed-quality algebra only. There is
no evidence quality, tracking or per-match adapter generalizes for those times.

Retain **GRANTED target-derived Telea background and historical granted-matte
limitation**, unresolved source/replay independence, familiar two broadcasts,
115px exploratory threshold and historicalstep5 failure. Missing160 raw ticks
per player mean neither all240 receiver correctness/correction cost nor a
complete delivery stream is measured. Determinism is demonstrated on these
saved pixels/environment/GPU, not across hardware or fresh generations. This
sampled player diagnostic is not independent confirmation or an end-to-end
codec win; the intended broader system still includes players+ball.

### Closed resource ledger and report

One GPU job, no failed/replayed GPU allocations. Native admitted run
577.565911s; receipt-to-terminal637.731448s includes queue/transfer/finalization.
Conservatively charge **640/2700GPU-s=0.177777778/0.75GPU-h**. Full cumulative
native child CPU snapshot631.598968s already includes smoke, so not counted
twice. Conservative aggregate **1429.969250/3600CPU-s**, including static/
focused/preflight allowances, full suite168.593484s, both CPU validations,
known failed wrapper upper charge, both successful residual runs, visuals and
300s analysis/publication/validator allowance. All stages and study wall fit
bounds; unused GPU/CPU allocations expire, no carryover. Exact ledger and all
raw job/validation/correction/accounting data retained with immutable publication.

Ruff/mypy without paths PASS;6 focused tests PASS. Full local suite890PASS/
25SKIP/2known Mac restrictions: process visibility and protected Linux home
symlink; process test independently passes with read-only process visibility.
Linux CI will check the complete final revision. No task-owned wakeups/recurring
model calls created; all native batch waits ended and events acknowledged.

Stopped because decision reached: fixed pose-prompted SAM tracking fails2/4,
this sampled residual coder is costly, and the contract/repeatability controls
pass. No further model/schedule search or full240 protocol launched. No open
user question. Any new tracker/profile or full240 data capture is a separate
planning decision and budget, not an automatic continuation of this failed pilot.
