# Experiments: W2

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md). Context:
[assessment 2026-10-29](../assessment-2026-10-29.md).

### 2026-10-29 — W2: how cheap can the player keypoints be? (rule)
- Question. The side-rate feasibility analysis
  ([report](../k6b-side-feasibility-report-2026-10-28.md)) allows the
  actual player overlay 0.67 kbps in total before the whole frame loses on
  Al. Keypoints alone cost 3.53 kbps for two players (4236 bytes over
  9.6 s). Lossless wrappers made them larger. Can a different
  representation reach a fraction of the allowance without moving the
  joints more than the pose estimator's own noise?
- Input: the original keypoint packets (KPS) of Al and Med from transport
  job `20261025T082944Z-29f021a5`, `full/published.tar` SHA256
  `6865035e3132b6e07600fdaabc8c9d7152b8b6e3c3260d9f3df21cab33ca33c7`,
  decoded with the existing reader; reference = the decoded original
  keypoints. CPU only, no images generated.
- Coders, three settings each first: (a) temporal prediction (previous
  frame and constant-velocity) with an adaptive arithmetic coder of the
  residuals at the current quantization; (b) the same at 2× and 4× the
  quantization step; (c) keyframes every 2 and 4 frames, decoder-side
  linear and cubic interpolation, coded as (a). Real encode/decode
  round trips; rate from the produced bytes including headers.
- Error: per-joint displacement from the reference, normalized by the
  player's box height; mean and p95; plus temporal jitter (second
  difference) relative to the reference. J measured the estimator's own
  per-frame noise at 1.32% of the box.
- Decision rule. A coder is a candidate if, on both windows, the two
  players' keypoints cost ≤0.5 kbps with mean displacement ≤0.66% of the
  box (half the estimator noise) and p95 ≤1.32%. Any candidate → propose a
  separately budgeted GPU check that generation quality is unchanged with
  the coded keypoints. No candidate → report the best rate at each error
  level; the whole-frame route for the players at this operating point
  closes, because the 0.67 kbps must also cover adapter and references.
- Hypothesis: 25 fps keypoints of slowly moving players are highly
  predictable; prediction plus interpolation reaches ~0.3–0.5 kbps.
- Competing explanation: the packet is already near the entropy of the
  estimator's per-frame noise, which prediction cannot remove without
  smoothing beyond the error bar.
- Also record, from existing numbers only: the other side terms (adapter
  1.65 kbps over 2 h, references) and what remains of the 0.67 kbps.
- Budget: CPU only, ≤1 hour; no GPU.
- Jobs: none (CPU).

### 2026-10-29 — W2: operational definitions, written before any run
The rule above stays as frozen. These definitions fill the gaps it leaves;
they were fixed before the first coder ran.

- Windows. `Al` = `alcaraz_hurkacz/031@129` (240 frames at 25 fps, 9.6 s,
  packet 4236 B, SHA256 `38c29a95…ed50`); `Med` = `medvedev_struff/014@194`
  (240 frames at **50 fps, 4.8 s**, packet 3542 B, SHA256 `0246c32c…f915`).
  A window's rate is stream bytes × 8 / (frames / fps). The packet contract
  also carries a zstd-compressed JSON of window constants (about 416 B,
  names, seeds, model profile); it is not pose information and is reported
  as a separate line, not charged to the coders; the rule's 0.5 kbps is on
  the coder stream including its own header.
- Reference. The decoded original packets (`decode_keypoints`): 240 frames
  per player, coordinates on a 1-pixel grid (linear interpolation of
  4-pixel samples every fourth frame), visibility every frame.
- Quantization step 1× = 4 px (the packet's q). Family (a) codes **every
  frame** (key interval 1) at 4 px with previous-frame or constant-velocity
  prediction; (b) the same at 8 and 16 px with the better (a) predictor;
  (c) keyframes every 2 and 4 frames at 4 px with that predictor, linear
  and Catmull-Rom cubic decoder interpolation. (c) with key 4 and linear
  interpolation is by construction an exact re-coding of the reference: its
  rate is the entropy-coding floor at zero error.
- Visibility is coded losslessly for every frame in all first-stage
  settings (it defines the reference's confident-joint set).
- Entropy coder: adaptive binary range coder with count-based
  probabilities, contexts from the neighbouring residuals, shared across
  both players of a window; trailing zero bytes of the stream are dropped
  and the decoder pads zeros. Rate = produced bytes incl. the stream
  header (frame count, fps, runs, q, key interval, predictor, flags).
- Error: displacement of each decoded joint from the reference joint over
  the reference's confident joints (the renderer draws only those; J's
  `pose_error` convention), normalized by the player's keypoint-extent
  height (median over frames of the confident joints' vertical extent, one
  constant per player). The rule is applied per player and window, mean and
  p95 over frames × confident joints. Secondary, reported only: all 17
  joints, and the crop-box height (1.4 × the larger of the extent height
  and width × 784/512, 90th percentile; always larger, hence more lenient).
  Jitter: mean norm of the second difference over time, normalized by the
  same height, for decoded and reference, and their ratio.
- Candidate: every player of both windows has mean ≤0.66% and p95 ≤1.32%,
  and each window's two players together cost ≤0.5 kbps.
- First stage: (a) 2 settings, (b) 2, (c) 4: eight runs. Because the first
  stage fixes only the family, a second stage, written here so it is not a
  post-hoc search, maps the rate–error frontier on the full grid
  q ∈ {1, 2, 4, 8, 16, 32} px × key interval ∈ {1, 2, 4, 8} × predictor ∈
  {previous, velocity} × interpolation ∈ {linear, cubic}, plus
  visibility coded only on keyframes (held between) for key interval > 1.
  A rule-passing point found there counts as a candidate only if it is
  reported as found in the second stage. "Best rate at each error level":
  for mean thresholds 0.33, 0.66, 1.32, 2.64% (p95 ≤ twice the mean
  threshold), the lowest window rate over settings whose every player
  passes. Untested levers, named so they are not mistaken for ruled out:
  skipping the coordinates of invisible joints, trained (not adaptive)
  probability tables, joint-structured transforms, frame-rate reduction
  below one key in eight.

### 2026-10-29 — W2: additions after the first run (disclosed)
The first run (code `698b4e50`, 32 s) gave no candidate in either stage;
the exact re-coding of the reference (key 4, 4 px, linear) is the cheapest
setting at every error level up to 2.64%. Two additions were made after
seeing that, both analysis of the same inputs and neither able to create a
candidate: (1) supplementary error levels 5.28% and 10.56% (p95 ≤ twice
the mean) so the table shows where the curve leaves the exact point, and
the unbeaten (bytes, worst-player mean) frontier per window; (2) a
lower-bound diagnostic for the exact stream: the zeroth-order empirical
entropy of the key residuals with a static optimal table (no learning
cost), with and without subtracting each key frame's median residual
(global motion compensation, its own entropy added), and the conditional
entropy of visibility. It is not a coder and has no round trip; it bounds
how much a better probability model or a body-motion predictor could save.

### 2026-10-29 — W2: outcome. No candidate; the player keypoints cannot reach 0.5 kbps at the error bar
- **Verdict by the frozen rule: no coder is a candidate**, in the first stage
  (8 settings) or the second (156 settings per window). Every coder that
  meets the error bar costs 1.7–2.8 kbps; the one setting at or below 0.5
  kbps (Al only) moves the joints by 11.7% of the box on average, 18 times
  the bar. The hypothesis (prediction plus interpolation reaches 0.3–0.5
  kbps) is rejected; the competing explanation holds: the samples (every
  fourth frame, 6.25 Hz for Al and 12.5 Hz for Med) already carry the
  motion, and nothing between or below them can be dropped without leaving
  the estimator-noise bar.
- Inputs. Archive `20261025T082944Z-29f021a5/full/published.tar` SHA256
  `6865035e3132b6e07600fdaabc8c9d7152b8b6e3c3260d9f3df21cab33ca33c7`
  (checked before any member was read; copied read-only to the session
  scratchpad, members read in memory). Al packet `38c29a95…ed50` (4236 B,
  240 frames at 25 fps = 9.6 s); Med packet `0246c32c…f915` (3542 B,
  240 frames at **50 fps = 4.8 s**, so the packet is **5.90 kbps**, not
  3.53). Code revision `67cade29` (branch `claude/w2-pose-rate`,
  `experiments/tennis/w2_pose_rate.py`, SHA256 `72f7067b…e3ba`), CPU only
  (31 s wall for both stages on the Mac), no GPU, no images. Full result:
  [JSON](w2-pose-rate-2026-10-29.json). Every row is a real encode and a
  decode from the produced bytes alone, checked equal to the encoder's
  expectation; rate = produced bytes including the 13-byte header.
- First stage (predictor for (b) and (c): previous frame, which beat
  constant velocity by total bytes, 6724 against 6833). Rate in kbps;
  error as mean / p95 % of the box height, worst of the two players (Al
  box heights 96 and 136 px, Med 134 and 68 px); jitter = decoded over
  reference second-difference magnitude (worst player).

  | setting | Al kbps | Al mean / p95 | Al jitter | Med kbps | Med mean / p95 | Med jitter |
  |---|---:|---:|---:|---:|---:|---:|
  | (a) every frame, 4 px, previous | 3.27 | 1.25 / 2.95 | 4.0 | 4.68 | 1.62 / 4.16 | 5.8 |
  | (a) every frame, 4 px, velocity | 3.10 | 1.25 / 2.95 | 4.0 | 5.19 | 1.62 / 4.16 | 5.8 |
  | (b) every frame, 8 px | 2.40 | 3.40 / 5.89 | 6.5 | 3.44 | 4.72 / 8.32 | 10.1 |
  | (b) every frame, 16 px | 1.77 | 6.52 / 10.42 | 10.1 | 2.40 | 9.21 / 14.71 | 12.4 |
  | (c) key 2, linear | 2.44 | 0.87 / 2.95 | 1.5 | 3.52 | 1.23 / 4.16 | 1.9 |
  | (c) key 2, cubic | 2.44 | 1.01 / 2.95 | 1.8 | 3.52 | 1.35 / 4.16 | 2.3 |
  | **(c) key 4, linear (exact)** | **1.81** | **0 / 0** | 1.0 | **2.77** | **0 / 0** | 1.0 |
  | (c) key 4, cubic | 1.81 | 1.07 / 3.18 | 1.3 | 2.77 | 0.72 / 2.37 | 1.4 |

  Coding every frame at the packet's 4 px grid is worse than the packet's
  own every-fourth-frame samples: the reference between samples is linear,
  and a per-frame coder pays for detail the reference never had. Only the
  exact re-coding (key 4, 4 px, linear: the original sample structure) meets
  the error bar, at 1.81 / 2.77 kbps against the original 3.53 / 5.90. The
  arithmetic coder alone saves 49% / 53%; visibility is 157 / 171 bytes of
  it (7–10%), the rest coordinates.
- Frontier (second stage; unbeaten in rate and worst-player mean error;
  visibility held between keys, which mismatches 3–6% of joint-frames, so
  the exact point with lossless visibility is the 1.81 / 2.77 above). Format:
  setting, kbps, worst-player mean / p95 % of the box.
  - Al: 4 px key 4 linear 1.74, 0 / 0; 8 px key 4 linear 1.33, 3.5 / 5.9;
    4 px key 8 cubic 1.10, 4.6 / 14.6; 16 px key 4 linear 0.96, 5.6 / 9.3;
    8 px key 8 cubic 0.86, 6.1 / 15.1; 16 px key 8 linear 0.65, 7.6 / 15.5;
    32 px key 8 linear 0.48, 11.7 / 20.8.
  - Med: 4 px key 4 linear 2.60, 0 / 0; 4 px key 8 cubic 1.72, 3.6 / 12.8;
    8 px key 8 linear 1.28, 6.3 / 13.2; 16 px key 8 linear 0.95, 9.3 / 17.7;
    32 px key 8 linear 0.65, 16.2 / 29.4.

  Best rate at each pre-registered error level (mean ≤ L, p95 ≤ 2L): at
  0.33, 0.66, 1.32 and 2.64% it is the exact point, 1.74 kbps (Al) and 2.60
  (Med) with keyframe-only visibility; nothing cheaper lies within 2.64%.
  Supplementary levels (added after the first run): 5.28% gives 1.33 / 1.92
  kbps (8 px, key 4, linear); 10.56% gives 0.65 / 0.95 kbps (16 px, key 8,
  linear). The Al 0.48 kbps point has 11.7% mean error and mismatches 6% of
  the visibility bits; Med is 0.65 kbps at its cheapest.
- Entropy diagnostic (not a coder; zeroth-order empirical entropy with a
  static optimal table, key residuals of the exact stream, bytes excluding
  first key and header). Al / Med: previous 2259 / 1719, velocity 2079 /
  1537; with each key frame's median residual subtracted (global-motion
  compensation, the medians' entropy added) 1715 / 1310 for previous and
  1901 / 1491 for velocity; visibility 162 / 183. The context coder used
  here already reaches 2015 / 1479 coordinate bytes, below the context-free
  numbers, so these are not lower bounds for every model. Read as the most
  a body-motion predictor could add on top of the contexts, they are
  about 300 / 170 bytes of coordinates, i.e. at most **0.25 kbps (Al) and
  0.28 (Med)**. Even a 24% cut of the whole stream (the context-free
  previous against its global-motion-compensated form) leaves about 1.4 /
  2.1 kbps, 2.8 and 4.2 times the 0.5 kbps limit. Untested levers stay
  untested: skipping invisible joints' coordinates, trained probability
  tables, transforms.
- Remaining allowance (Al, 9.6 s, all from existing reports). The robust
  ceiling for the whole side stream with the actual overlay is 0.6687 kbps
  (central 0.7843). The terms inside it today: keypoints 3.530; adapter and
  guider about 1.65 (the 2-hour amortization of about 1.5 MB, as in the
  assessment; not re-measured); bootstrap header 444 B = 0.370; match bank
  by subtraction 0.057. The non-keypoint terms sum to 2.077 (feasibility
  JSON). Remaining with free keypoints: 0.669 − 2.077 = **−1.41 kbps**. With
  the best exact keypoints (1.81): −3.22. With a hypothetical 0.5 kbps
  keypoint stream: −1.91. Keypoints at 0.5 would need the other terms at or
  below 0.17 kbps; the adapter alone (11.9 Mbit) would need an amortization
  of about 70,000 s (19.5 h) even with bootstrap and bank at zero.
- Decision (the rule stated the consequence): the whole-frame route for the
  players at this operating point closes on the side rate. Neither the
  keypoints (best 1.7 kbps at the error bar against 0.5) nor the other side
  terms (2.1 against the 0.67 total) fit, and no coder bridges both. Scope:
  the SVT-AV1 scale-4 base, the Al window's allowance (Med's was never
  measured), 17-joint 1080p keypoints and these coder families; both windows
  are development footage. No GPU check of generation quality is proposed
  because there is no candidate.
- Caveats. The reference is the decoded packets, already a 4 px,
  every-fourth-frame representation, so "error" is distance from that
  representation, not from raw estimator output; the rule fixed this
  reference. Box height is the player's confident-joint vertical extent
  (68–136 px); the crop-box normalization is in the JSON and is 1.6–2.0
  times larger, which changes no verdict because the exact point is the only
  setting under 2.6%.
- Jobs: none (CPU).
