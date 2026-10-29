# Experiments: W3

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md). Context:
[assessment 2026-10-29](../assessment-2026-10-29.md).

### 2026-10-29 — W3: a deterministic neural baseline on the tennis windows (rule)
- Question. The paper needs a state-of-the-art neural video codec beside
  SVT-AV1. DCVC streams decoded differently across GPUs and with the default
  CUDA stream; only a 24-frame reproduction on gpu3 passed
  ([single-stream report](../single-stream-report-2026-10-28.md)). Can the
  installed DCVC-UF (pinned in
  [resources](../resources.md#neural-video-codec-dcvc-uf), worker
  `src/codecs/dcvc_uf_worker.py`) produce real bitstreams that decode
  exactly in the same job on the same GPU, for Al and Med?
- Design: one job per window on one GPU (gpu3, gpu5 or gpu6; not gpu2,
  whose smokes time out). Encode 240 native frames to bitstream files with
  the native entropy coder, decode those files in the same job, compare the
  decoded frames with the encoder's reconstruction, take the rate from the
  file sizes. Deterministic settings (cuDNN deterministic, benchmark off,
  deterministic algorithms where supported); record which were needed.
  Three operating points (QP, scale) chosen in the smoke on rate alone to
  bracket 15–60 kbps. Score whole and fixed-foreground LPIPS/DISTS on the
  same 80 ticks as W1 with the real-base judge's code.
- Decision rule. The baseline is accepted on a GPU class if, for every
  point, the decoded frames equal the encoder reconstruction (max absolute
  difference ≤1 8-bit level on every frame); record it in the model–GPU
  table of [fleet](../fleet.md). Then report the DCVC envelope against the
  SVT envelope per metric (BD-rate, PCHIP, overlapping quality only;
  otherwise better/worse at every rate), whole frame and foreground. If
  decoding is not exact even within one job, stop and report the
  first diverging frame and kernel settings; no retry with other settings
  without a new entry.
- Hypothesis: within one process on one GPU, with deterministic kernels,
  decoding is exact; earlier failures came from cross-GPU decoding and the
  default CUDA stream.
- Competing explanation: nondeterministic kernels inside the entropy model
  make even same-job decoding diverge.
- Budget: ≤2 GPU-hours including smokes and failures; every stage ≤45 min.
- Jobs: (none yet)
