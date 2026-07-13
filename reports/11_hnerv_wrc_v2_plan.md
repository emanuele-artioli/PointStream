# HNeRV Competitive Baseline v2 (byte-budgeted + Weight Residual Coding) — Plan
*Status: Active | Last updated: 2026-07-13 | Code: `src/shared/hnerv_arch.py`, `scripts/hnerv_baseline.py`*

## Scope

Redo the HNeRV learned-codec baseline so the paper's comparison is
defensible in either direction. The 2026-07-12 WRC run
(report 9, superseded entry) failed twice over: the P-chunk fine-tuning
collapsed (2 of 4 chunks decode to a constant frame), and the protocol was
structurally unable to compete (12.4 M params for 15-frame chunks, fp16
weights, no real sparse/entropy serialization, no byte budget). This plan
fixes the protocol and adds **hard self-check gates** so a broken run halts
instead of publishing numbers.

**Executor discipline (read first):**
- Work through the phases in order. Every gate marked **HARD STOP** means:
  if the check fails, stop the experiment, write the failure and the
  numbers into report 9's findings log, and do not proceed or publish
  aggregates. A failed gate is a *result to report*, not an obstacle to
  route around.
- Do not change the evaluation path: payload bytes = the serialized file
  sizes; quality = `evaluate_run_summary` on the decoded frames vs the 4K
  source, exactly as `outputs/codec_baselines/` and the two previous HNeRV
  runs did. If you believe the eval is wrong, stop and write that up
  instead of changing it silently.
- Run `ruff check` + `mypy` on every touched file and the fast pytest
  suite **before** launching any multi-minute training run, and commit the
  code before the long run starts (repo rule: never leave the code under
  test uncommitted while a slow run validates it).
- No new CLI entry points. Extend `scripts/hnerv_baseline.py` flags only.

## The question being answered

For the 60-frame `assets/real_tennis.mp4` clip, the AV1 anchor curve
(`outputs/codec_baselines/`, report 9) includes, e.g.:

| anchor | bytes | PSNR | VMAF |
|---|---|---|---|
| AV1 CRF50/slow | 453,237 | 38.6 | 93.1 |
| AV1 CRF20/slow | ~2.5 MB | higher | higher |

The only meaningful HNeRV claim is a **rate-distortion point at a matched
byte budget**: "at N bytes, HNeRV reaches quality Q vs AV1's Q′". Payload
5.7× the raw source answers nothing. Target budgets: **0.5 MB, 1 MB, 2 MB,
4 MB** total for all 60 frames.

## Phase 0 — Serialization fixes (before any training)

The current format wastes bytes and makes HNeRV look worse than it is:

1. **int8 weights for the anchor.** `save_hnerv_checkpoint` ships decoder
   weights as fp16 (~1.85 B/param after gzip — this is where 22.9 MB for
   12.4 M params came from). Add per-tensor int8 quantization of decoder
   weights (reuse `quantize_tensor_int8`, per-tensor scale/zero-point),
   matching what the embeddings already get. This mirrors the HNeRV
   paper's own 8-bit protocol. Keep the old fp16 path behind a flag for
   comparison, default to int8.
2. **Real sparse serialization for deltas.** `save_hnerv_residual`
   currently stores dense fp16 with zeros and gzips — the 20 % surviving
   values cost the full 2 B each (12.4 M × 0.2 × 2 B ≈ 5 MB = exactly the
   observed P-chunk sizes; the pruning bought nothing). Replace with:
   flatten per-tensor delta → nonzero mask as a bit-packed `uint8` array
   (`numpy.packbits`) → surviving values quantized int8 with per-tensor
   scale → gzip the lot. Expected cost per surviving value ≈ 1 B + mask
   overhead (numel/8 bytes).
3. **Symmetry stays mandatory.** Keep the existing in-place apply of the
   *lossy* (pruned + quantized) delta to the encoder-side decoder before
   evaluation — that part of the 07-12 code was correct. Extend the unit
   test to assert save→load round-trips bit-identically for the new int8
   sparse format.

**Gate P0 (HARD STOP):** unit test — for a random small decoder, the
serialized-then-loaded weights must equal the encoder-side post-apply
weights exactly, and `bytes_per_param = payload_bytes / params` for a
dense int8 checkpoint must be ≤ 1.3. If you measure ≥ 1.8 B/param, you
are still shipping fp16 somewhere — find it before training.

## Phase 1 — Byte-budgeted single-checkpoint RD curve (no chunking)

Train **one model per budget on all 60 frames** (the 2026-07-11 protocol,
which trained healthily), sized so the *serialized int8 payload* lands
near each budget. Sizing rule of thumb after Phase 0: params ≈ budget in
bytes (≈1 B/param) minus the int8 embedding cost — compute the embedding
bytes first from the grid config, subtract, then choose channel widths.
Suggested ladder: ~0.4 M, ~0.9 M, ~1.8 M, ~3.5 M params. Keep 640×360
training resolution for the two small budgets, 1280×720 for the two
large; 15,000 epochs (the 07-11 run's budget — 2,000 was not converged:
its curve was still flat-lining only at the very end at 720p).

**Gate P1 (HARD STOP per model):** training loss at the final epoch must
be < 0.5× the epoch-0 loss AND monotonically non-increasing over the last
20 % of epochs (tolerance: +2 % noise). Final train PSNR ≥ 28 dB at
640×360 or ≥ 27 dB at 720p. A model failing this is reported as
"diverged/undertrained", not scored.

**Deliverable:** one table, HNeRV budgets vs the nearest AV1/HEVC anchor
points (bytes, PSNR, SSIM, VMAF), appended to report 9. Verdict language:
state the gap in dB/VMAF at matched bytes. Words like "conclusively",
"definitively", "closes the book" are banned — one clip, one architecture.

## Phase 2 — WRC v2 (only after Phase 1 passes)

Chunked variant on the best-performing budget from Phase 1 (and only
budgets whose Phase-1 model reached the P1 gate):

1. **Carry the whole model state** between chunks — encoder AND decoder
   (`model.load_state_dict`), never a fresh `HNeRVModel(config)` with a
   random encoder (root-cause suspect for the 07-12 collapse).
2. **Fine-tune LR = 0.1 × initial LR**, cosine over the fine-tune epochs.
   Fine-tune epochs: 2,000 minimum (500 was visibly still converging).
3. **Delta pipeline:** sparsity sweep {0.5, 0.8, 0.95} × int8-quantized
   surviving values, serialized per Phase 0 item 2. The anchor chunk uses
   the Phase-0 int8 checkpoint format.
4. **Per-chunk gate (HARD STOP):** every P-chunk must end with
   (a) final loss < epoch-0 loss (it starts from a warm model — if loss
   *rises*, training diverged); and (b) final train PSNR ≥ anchor's train
   PSNR − 3 dB. On failure: halt, dump the last checkpoint + the
   `progress.jsonl` excerpt into the report, do not aggregate.
5. **Byte accounting per chunk** printed and saved to `report.json`:
   anchor bytes, each delta's bytes, mask overhead, embedding bytes.

**Deliverable:** WRC RD points (total payload vs quality) on the same
table as Phase 1, with the per-chunk PSNR trajectory attached. The honest
comparison for VOD framing: WRC total bytes vs (a) Phase 1 single model
at the same total budget, (b) AV1 at the same bytes.

## Self-verification checklist (run after each phase, paste into report 9)

- [ ] `progress.jsonl` plotted/inspected for **every** chunk/model — no loss increase, no PSNR plateau below gate.
- [ ] `bytes_per_param` computed and ≤ 1.3 for every serialized artifact.
- [ ] Decoded PNGs spot-checked: open 3 frames per chunk (first/middle/last) — a near-uniform gray/green frame means collapse; check *before* running VMAF.
- [ ] `evaluate_run_summary` numbers pasted next to the AV1 anchor rows from `outputs/codec_baselines/` (same source, same scorer).
- [ ] ruff + mypy + fast pytest green; code committed before the long run.

## Anti-goals

- Do not conclude anything about "learned codecs" as a class.
- Do not tune on VMAF (train on L1 as-is; VMAF is eval-only).
- Do not add new dependencies (no external entropy coders; gzip over
  int8+bitmask is the approved approximation, and its approximation error
  must be stated in the report entry).
- Do not delete or overwrite `outputs/hnerv_vod_sweep*` — superseded runs
  move to `outputs/_superseded/` per repo rules, and only if space demands.
