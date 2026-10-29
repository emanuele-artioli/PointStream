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
- Jobs: (none yet)

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
