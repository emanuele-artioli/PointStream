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
