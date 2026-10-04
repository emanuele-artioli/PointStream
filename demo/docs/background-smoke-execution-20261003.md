# Background smoke execution status — 2026-10-03

This is an execution record for `background-smoke-plan.md`. It records the
bounded work completed from the Mac-side checkout and the evidence gates that
could not be reached. It is not codec-quality or paper evidence.

## Decision

**GPU stages B0–B4 are blocked by fleet DNS. GPU allocation used: 0 of the
30-minute cap. No training, checkpoint inference, encoding, or model decode
was started.** Continue only after a fresh complete fleet inspection and
verified access to the saved preview and canonical data root.

The required inspection was run for gpu1–gpu6. All six returned unavailable
because SSH could not resolve `gpuN.itec.aau.at`. The one status request for
the existing preview job, `20261003T101656Z-61d46aea`, failed for the same
reason on gpu5. Its manifest, images, native decodes, and bitstreams were
therefore not read or re-rendered in this execution. Earlier notes embedded
in the plan remain leads; this run did not independently verify them.

## Work completed

- Used an isolated checkout at `/private/tmp/pointstream-background-smokes-clean`
  on branch `codex/demo-background-smokes`, based on plan commit `e74130a`.
  The shared PointStream checkout and its dirty files were left unchanged.
- Copied only the four untracked background seed files named by
  `smoke-plan-source-inventory.json`. Their bytes and SHA-256 values matched
  the inventory: `factory_bg_rd.py` (`3c0c98c8…`),
  `hnerv_holdout_eval.py` (`8929e1f1…`),
  `render_bg_visual_snapshot.py` (`e19ec999…`), and
  `test_factory_bg_rd.py` (`6dd9cd95…`).
- Added `demo/experiments/background_smoke.py` with bounded helpers and
  artifact-only commands for explicit file inventory, eight-frame RGB
  scoring, 32-frame drift summaries, latent packetization, and record
  collection. It rejects wrong cut lengths/order, checks source frame hashes,
  requires provenance in score manifests, derives rates from actual stream
  file sizes, and refuses evidence writes outside `PS_JOB_DIR` or over an
  existing output. Eight-frame rows explicitly remain `inconclusive` because
  LPIPS Alex and optional aligned-mask errors are not yet wired into this
  helper. It contains no training path and does not invoke DCVC.
- Added `demo/experiments/hnerv_latent_packet.py`. It packages uint8 six-bit
  latent codes into complete 1-, 8-, and 32-frame packets using packed codes,
  zlib, or modular temporal deltas plus zlib. Headers preserve tensor shape,
  ordered frame IDs, checkpoint hash, exact quantizer-array dtype/shape/bytes,
  timebase, and separately reported decoder setup size. Header, payload, and
  decoded-code checksums are verified. Packet file sizes include headers.
- Added synthetic tests for separate stage epochs; hash change and timeout
  behavior; strict state-dict keys/shapes; train/hold-out and temporal-reset
  identity checks; known RGB PSNR and temporal-error calculations; actual-byte
  rate accounting; ordered cut validation; and packet round trips, metadata,
  resets, corruption, and size accounting.

## Evidence and gates

| Gate | Status | Evidence / reason |
|---|---|---|
| B0: reuse and verify saved preview | `blocked` | gpu5 DNS failure prevented status and artifact reads. |
| B1: remote source/checkpoint reconciliation | `blocked` | No remote checkout, checkpoint, training manifest, or environment was readable. Local seed hashes were verified only. |
| B2: pretrained/fine-tuned eight-frame codec pairs | `blocked` | No eligible inspected GPU or installed-codec access. 12 GPU minutes unspent. |
| B3: 32-frame LD drift/reset comparison | `blocked` | Requires a valid B2 pair. 5 GPU minutes unspent. |
| B4: HNeRV extraction, decoder-only proof, and latent RD | `blocked` | No HNeRV checkpoint/environment or GPU access. Packet codec tests cover arrays only; 8 GPU minutes unspent. |
| B5: evidence-based model decision | `inconclusive` | Missing B0–B4 evidence prevents a model ranking or follow-up spend recommendation. |

The bounded test command passed **42 tests**:

```bash
python -m pytest -q tests/demo/test_factory_bg_rd.py tests/demo/test_neural_bg.py tests/demo/test_background_provenance.py tests/demo/test_background_codec_smoke.py tests/demo/test_hnerv_latent_packet.py
```

These are CPU tests with synthetic inputs; they do not establish DCVC or
HNeRV inference correctness. In particular, this execution did not verify
the historical checkpoint hashes, the fine-tuned-versus-pretrained result,
the old long-cut failure, LPIPS, source-free HNeRV reconstruction, or actual
quality/rate of a newly encoded stream.

## Resume order

1. Re-run the required six-host inspection. Do not reuse the failed probe as
   an admission result. Check the old preview job's status and manifest once
   DNS works; verify its ordered eight frames and inspect its saved panels.
2. Complete B1 within its CPU cap: record remote code/diff/environment and
   exact checkpoint hashes, strict model compatibility, separate stage
   epochs, and train/hold-out sequence identities. If any loaded checkpoint
   cannot be hashed and strictly matched, stop model work.
3. Only after B1 passes, run B2's exact primary eight-frame matrix with the
   plan's per-case and 12-minute aggregate limits. Reuse only artifacts whose
   input and checkpoint identities match. Count actual stream files and use
   RGB metrics with one common decoder path.
4. Finish the LPIPS Alex scorer and the installed-version DCVC encode/decode
   adapter before counting B2 as complete. Run B3 only for a structurally
   valid B2 pair; prioritize LD and obey its five-minute ceiling. Run B4
   only with the trained HNeRV architecture and checkpoint verified; prove
   source-frame independence before reporting any latent result. Keep all
   allocations within the shared 30-minute cap.
5. Generate B5 only from machine-readable run records. Preserve blocked and
   negative results; do not use this CPU implementation test as a reason to
   start training or a longer sweep.
