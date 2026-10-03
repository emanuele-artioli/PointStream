# Background: implementation and decision plan, smoke only

## Assignment and limits

Determine whether the background failures come from checkpoint/source
mismatch, inference state, training degradation, or the tested representation.
Measure whether temporal packaging improves HNeRV's latent rate. Produce
bounded evidence for selecting a later experiment. This assignment does not
train or fine-tune any background model, regenerate SAM/DiffuEraser inputs,
rebuild the website, run whole-video scoring, or establish a paper result.

Hard limits:

- **30 GPU allocation minutes cumulatively**, including loads, warmup,
  failures, and filesystem waits after GPU admission. Every job <= 8 minutes.
- At most 10 minutes of remote CPU preparation, with a 90-second timeout on
  each blocking file read/hash/transfer. No broad scans of the dataset,
  environments, or third-party repositories. A second identical failure ends
  the operation. Stop at the aggregate budget; do not start a fresh budget
  merely because a new job ID or host is used.
- Do not call `train-uf`, `train-hnerv`, `fill-holdout`, or an unrestricted
  `score` CLI. No epochs, gradient updates, latent optimization, distillation,
  new checkpoints/downloads, third-party upgrades, or extension rebuilds.
- Reuse completed previews where their input identity is known. New runs go
  to new directories. Never modify completed scientific/infrastructure
  evidence or the active/paused main and confirmation checkouts.

Implement on `codex/demo-background-smokes`, from the Mac, under AGENTS.md.
Keep foreground/model-loss/packet changes out of this branch. Do not start GPU
host agent sessions, merge, deploy, or open a PR. Commit/push coherent code
and tests; provide a decision report and remaining blockers.

## Starting material and execution contract

Read `demo/docs/internal-audit-20261003.md` and the adjacent
`smoke-plan-source-inventory.json`. Needed source files:

- `demo/experiments/factory_bg_rd.py`: source sets, held-out cuts, codec
  commands/configuration, checkpoint paths, saved score locations.
- `demo/experiments/hnerv_holdout_eval.py`: exact trained architecture,
  quantization/decode, current estimated `_bits` rate.
- `demo/experiments/render_bg_visual_snapshot.py`: eight-frame rendering
  helper; its HNeRV labels incorrectly use weights-inclusive `total_bits` as
  an ordinary streaming rate and must be corrected in a new report.
- `demo/pipeline/maps/av1_crf.py` and
  `tests/demo/test_factory_bg_rd.py`, `tests/demo/test_neural_bg.py`.

Some experiment files/tests are untracked in the original Mac checkout.
Verify/copy only the background seed files listed in the inventory into your
scoped checkout and record hashes/revision. A clean checkout alone is not
complete. If hashes differ, inspect/document the selected version; never
reset, stash, clean, or implicitly dispatch unrelated dirty work. Preserve a
reviewed baseline of required seed files on your branch before implementation.

Canonical remote locations (old aliases still resolve, but use these):

```text
data:       /home/itec/emanuele/Datasets/pointstream-data
dataset:    /home/itec/emanuele/Datasets/pointstream-demo
work:       <data>/jobs/factory-bg-rd
DCVC:       <data>/jobs/neural-bg/src/DCVC
HNeRV:      <data>/jobs/neural-bg/src/HNeRV
UF Python:  /home/itec/emanuele/.conda/envs/pointstream-dcvc/bin/python
HNeRV Python: /home/itec/emanuele/.conda/envs/pointstream/bin/python
```

Recordings/hold-out stems:

```text
factory001: clip_01_factory001_worker001_00001_last10s
factory001: clip_03_factory001_worker001_00000_last10s
factory002: factory002_worker001_00000_last10s
```

Factory001's learned background combines sampled press seconds from clips 1
and 3, excluding clip 3 seconds 210/240/420. Clip 3 second 0 stays. Factory002
is separate. Do not mix rooms. Hold-outs are 300/300/299 frames, not three
identical 300-frame videos. Existing filled inputs/checkpoints/scores are
read-only. No data symlinks in the code tree and no cleanup of compatibility
aliases, dataset originals, or worktrees as part of this task.

Before selecting a GPU:

```bash
python -m experiments.jobs.fleet inspect --hosts gpu1 gpu2 gpu3 gpu4 gpu5 gpu6
```

Use only complete probes: no compute processes, memory <= idle baseline
(default 256 MiB), utilization <= 5%, estimate + 4 GiB free, sufficient CPU
headroom. Prefer an inspected idle A6000/Ada for these native 1080p checks;
do not infer GPU availability from old reports. Launch with the project's
`fleet launch`, 4 CPU threads, explicit stage budget/memory estimate, required
input paths, and only reviewed change/untracked paths. An 8-minute allowance
is `--budget-hours 0.1333`. Outputs stay under new `PS_JOB_DIR` directories.
Capture code/patch/seed/checkpoint hashes, exact command/environment, GPU UUID,
encoder/decoder versions, timings, and peak memory. If disconnected, inspect
the existing job; do not replay/migrate it or cancel another user's process.
No eligible GPU means finish independent code/tests and report blocked smoke.

Use an initial conservative memory request of **24,000 MiB** for these
1080p codec/model smokes (plus the dispatcher's 4 GiB margin), and unload one
model before loading the next. Lower that request only from recorded
same-path peak-memory evidence. Never shrink the source images to fit. Model
loads must respect the allocation/time caps, even when compute itself is fast.

Add a bounded entry point, e.g. `demo/experiments/background_smoke.py`, with
subcommands `inventory`, `codec`, `drift`, `latent`, and `summarize`. It must
reject out-of-plan segment lengths/counts and never call an unrestricted old
scoring/training loop. Unit tests may run locally; real codec/model processing
runs on GPU servers via the dispatcher. Do not silently install dependencies
into a shared environment; missing dependencies are explicit blockers.

## B0 — Inspect and reuse the existing preview

Existing bounded job **20261003T101656Z-61d46aea** completed successfully on
gpu5 during preparation of this plan. Verify its current saved status and
manifest before doing anything with it; do not launch another writer there.

```text
<work's data root>/jobs/fleet/runs/20261003T101656Z-61d46aea/
    canvas/manifest.json
    canvas/factory001_frame_{00,02,04,06}.jpg
    canvas/factory002_frame_{00,02,04,06}.jpg
```

The preview uses hold-out indices 120..127 for factory001 clip 3 and
factory002, and compares filled input, AV1 240p/1080p, HNeRV, and fine-tuned
UF LD/HT-S/HT-L at QP 21. Read its manifest to locate native decoded frames
and bitstreams. Verify exact frame count/order/dimensions and inspect the
panels. Copy/export panels for review, keeping the saved files unchanged.
The display numbers are diagnostic, not full-stream metrics; HNeRV's current
display includes setup weights, and no checkpoint hashes were captured in
that old helper. Do not invent historical hashes from today's files.

Important new observation: the fresh eight-frame preview is healthy for LD
and HT-S, unlike the old long-cut averages. On factory001 clip 3 at QP 21,
LD is 36.72 dB / 401.94 kbps and HT-S 34.14 dB / 303.81 kbps on those eight
filled frames. These are fresh short-cut diagnostics, not matched-quality
baseline wins or proof of historical checkpoint identity. Prioritize source/
checkpoint reconciliation and prediction drift; do not assume the weights
fail on every input or spend time retraining immediately.

**Output:** reuse table with path, input identity, what is verified, what is
unknown, and corrected rate labels in a new report. If absent/incomplete,
record which part is missing. Re-render only that necessary subset after B1;
do not restart a whole completed matrix.

## B1 — Reconcile checkpoints, source patches, and training inputs

No GPU inference until required current input/checkpoint identities are
verified. CPU preparation budget includes these reads. Start with current
UF image/pretrained/fine-tuned HT-L checkpoints; only inventory LD/HT-S when
needed by B2. Do not deserialize every optimizer checkpoint as a first step.

1. Save DCVC/HNeRV Git HEAD, dirty status, exact tracked diff, relevant
   inference/training files' hashes, extension/library versions and paths,
   architecture/configuration. Current DCVC was observed at
   `cbdae87a5445114cdc7f48816da63ea80bdeac40`, modified in
   `src/layers/extensions/inference/dmc_common.cpp` and `train_video.py`.
   This observation is a lead, not permission to assume historical identity.
2. Resolve every checkpoint to a canonical path; hash once in a bounded
   streaming read, recording bytes and timeout. Cache hashes with a complete
   successful receipt. Filename, size/mtime, or partial hash is not a verified
   content identity. If NFS is slow, a task-specific local scratch copy may
   be used only with bounded transfer and verified original/copy hash; do not
   overwrite or move the source. Stop blocked reads after the stated limits.
3. Map `model_path_i` and `model_path_p` for each saved/new score to the exact
   checkpoint and structure. Check architecture, key names, tensor shapes,
   normalization, lambdas, QP mapping, precision, extension path, and strict
   state loading. No `strict=False` to conceal incompatible weights.
4. Read embedded epoch and stage separately. Factory001 HT-L saved status
   files contain stage-0 epoch 19 and stage-1 epoch 10; indices are zero-based.
   `s1/ckpt.pth.tar` contains model weights, not a trustworthy combined epoch
   count. Compare its `state_dict` tensors to the last stage-1 status `net`
   only if the bounded read fits the CPU/memory budget. State-dict key mapping
   must remove only a documented `module.` prefix, not arbitrary strings.
   Record whether tensors match exactly. Do not infer 46 completed epochs
   from today's defaults or assert that a checkpoint is historical from its
   path. Preserve unresolved resume/log contradictions.
5. Inspect the specific training description/batches manifests, not every
   image: factory membership, excluded seconds, source frame identities,
   absence of final hold-out identities, and sequence boundaries. DCVC must
   not accidentally predict across independent sampled seconds/rooms.
   HNeRV's per-frame encoder/decoder architecture must come from its trained
   configuration, not the evaluation segment length. Record observed counts;
   do not substitute assumed 1440/1200 counts if files differ.

**Tests** in `tests/demo/test_background_provenance.py`:

- Stage-0 epoch 19 and stage-1 epoch 10 remain separate; no summed epoch claim.
- Changed bytes invalidate a cached identity; timeout/partial read never
  produces a complete hash receipt.
- Wrong structure, state keys/shapes, missing checkpoint, and incompatible
  precision/configuration fail explicitly before dispatch/inference.
- Training/hold-out identity intersection, excluded clip3 seconds, and
  cross-room/cross-second temporal sequences are detected.
- Source diff and checkpoint identity are included in every new score schema.

**Gate:** new runs require verified hashes for their actual loaded checkpoints,
strict load, and exact selected inputs. Historical provenance may remain
unknown and must be labeled as such. An unknown historical hash does not
invalidate a properly recorded new smoke, but prevents claiming it reproduces
the old run exactly. If current checkpoint reads cannot be verified within
budget, stop model work and deliver the provenance blocker.

## B2 — Small pretrained versus fine-tuned codec diagnosis

Use native **1920x1080 RGB PNG**, **30 fps**, indices **120..127 inclusive**.
Primary cut: factory001 clip 3. Secondary cut: factory002. Keep an explicit
ordered list of exactly eight source frame hashes/identities. Use the same
filled sources for all background codec arms. No temporal/spatial subsampling,
new inpainting, padded 299th frame, or alternative favorable scene.

Budget <= **12 GPU minutes** total, each component <= 4 minutes including
load/encode/decode. Matrix order, with a gate after each case:

1. Primary HT-L pretrained, then HT-L fine-tuned, QP 21.
2. Primary LD pretrained/fine-tuned, QP 21.
3. Primary HT-S pretrained/fine-tuned, QP 21.
4. Secondary HT-L pretrained/fine-tuned, QP 21, only if budget remains.

Thus at most 8 new cases, fewer when existing verified decodes can be reused.
Load a model once per required checkpoint and release before the next;
preserve the GPU memory margin. For every structure, use its matching public
video checkpoint from the existing `factory_bg_rd.UF_VIDEO` mapping and the
same UF image checkpoint in both pretrained/fine-tuned arms. Use the exact
installed DCVC CLI/config supported by its recorded revision; inspect its
`--help` before invoking, not a different revision's web documentation.

Run actual bitstream encode and independent decode, recording all file
bytes/header bytes, frame order, range, dimensions, times, and peak memory.
Keep display order distinct from hierarchical coding order. For LD expect
one initial I frame with prediction thereafter; HT must use its documented
eight-frame structure, not forced all-intra. Do not change reset/QP settings
between pretrained and fine-tuned arms. Verify that decoding can run without
source PNGs; test config requiring source images for scoring does not prove
that the decoder needs or does not need them. Add a decoder-only adapter to
the installed codec if necessary and test it on these eight frames only.

For comparison, reuse exact verified AV1 streams or encode only 240p and
1080p, CRF 63/preset 7, GOP 8 using the repository helper. Decode/upscale with
the same documented resampler for every AV1 comparison. Do not use ffmpeg
YUV-average PSNR against DCVC RGB PSNR. Compute common metrics yourself from
decoded RGB PNGs against the exact full-resolution filled input:

- Per-frame RGB MSE/PSNR and mean per-frame PSNR, consistently aggregated.
- LPIPS Alex with RGB in [-1,1], resolution and implementation recorded.
- Mean absolute temporal difference error:
  `mean(abs((rec[t]-rec[t-1])-(ref[t]-ref[t-1]))) / 255`.
  Label this temporal reconstruction error, not motion-compensated flicker.
- Foreground-mask versus outside-mask errors if saved aligned masks exist.
  These measure fill reconstruction, not distance to a real empty room.
- Actual kbps = `8 * stream_bytes * 30 / (1000 * decoded_frame_count)`.
  Report decoder setup weights separately. Include failed arms and reasons.

Never determine output normalization by looking at whether values happen to
fall outside a threshold. Derive it from model/TransformInput source and
assert the expected range; apply one documented conversion. Do not clamp an
incorrectly normalized result until it appears plausible.

**Tests** in `tests/demo/test_background_codec_smoke.py`:

- Synthetic identical/known-error RGB frames yield independently calculated
  MSE/PSNR; distinguish mean-frame PSNR from PSNR of pooled MSE.
- Swapped/dropped/padded frames and different source hashes make a comparison
  fail. HT coding/display order is restored correctly.
- Rate uses actual file bytes and one common duration, including headers.
- Changing source images after encoding cannot affect independent decode.
- CLI/config uses PNG/RGB sources, exact structure/QP/checkpoints, one initial
  intra policy and no automatic full-run fallback.
- Output normalization, BGR/RGB order, channel shape, and LPIPS normalization
  fail on wrong conventions rather than silently converting by range guesses.

**Failure gates:** stop a case on nonfinite tensors/metrics, frame mismatch,
OOM, missing provenance, decode/source dependency, or time cap. A structurally
valid fine-tune losing > 2 dB or using > 2x pretrained bytes at the same QP
is a degradation flag. Check exact source/checkpoint/lambda/normalization/
extension correspondence and save diagnostic frames; do not retrain it.
This threshold triggers investigation, not a claim about all fine-tunes.

Inspect first-I and subsequent-frame errors. If only prediction frames
collapse, test one **single-frame force-intra** round trip at QP 21 with the
same image checkpoint (<= 1 minute, inside this 12-minute budget). Compare
image-only reconstruction, not a fabricated expectation of bit-for-bit
identity across different containers. If pretrained also fails, investigate
the inference/configuration path before attributing failure to training.
Save at most 4 source/decoded rows per case, not whole-video image dumps.

## B3 — Check short versus longer prediction without a full sweep

Budget <= **5 GPU minutes**, and run only if B2 has a structurally valid pair.
Use primary source indices **120..151**, exactly 32 native frames. No new
training. Prioritize **LD pretrained/fine-tuned**, because its short preview
is healthy and its historical long cut failed most severely. Test HT-S only
if LD's cases finish and >= 2 minutes of this stage remain; otherwise report
HT-S drift as untested. HT-L's existing long results are supporting evidence,
not a reason to divert this stage away from the failure under investigation.

Compare one 32-frame stream with four independently reset eight-frame streams
covering those exact 32 frames, QP 21. Reuse previously verified eight-frame
output for 120..127 only when identity and checkpoint hashes match; do not
concatenate compressed files and call that one 32-frame codec stream.
Reuse/encode matching AV1 240p/1080p cuts only within the remaining budget.
Report actual summed bytes, decoded quality, and first/last-frame errors;
look for growing prediction error and reset overhead. All segment tails are
explicit; no 296-versus-300 long-cut comparison in this assignment.

**Tests:** four independent packet decodes have no state dependence;
ordered decoded frames cover exactly the same identities as the 32-frame
case; changing an earlier eight-frame stream cannot change a later reset
stream; setup is not charged four times in steady-state tables. An untrained
or incompatible architecture must never be substituted to fit a segment.

**Gate:** increasing drift or fine-tune degradation means stop and recommend
targeted state/normalization/checkpoint diagnosis. A healthy trend means a
later 64/128-frame drift smoke may be worthwhile if the historical 300-frame
failure remains unexplained; it does not authorize one in this assignment.
If all current short/32-frame cases are healthy but historical long cases
are bad, report the cause as unresolved and distinguish checkpoint/source
drift from temporal codec drift. Do not add a QP grid here;
one QP cannot establish BD-rate or a general baseline win.

## B4 — Package HNeRV latents and test temporal redundancy

Budget <= **8 GPU minutes**, plus the bounded CPU cap. No HNeRV training or
optimization of held-out embeddings. First use the primary 32-frame cut;
factory002 is optional only after correctness passes and budget remains.

Reconstruct the **trained** architecture/configuration, including channel
count, decoder width, strides, output bias, and quantized decoder weights.
Do not use the evaluation frame count to rebuild model capacity. Factory001
has been observed with 9x16x3 embeddings and factory002 with 9x16x4; assert
the loaded architecture's actual tensor shapes instead of hardcoding both to
3 channels. Load once per factory, strict state, freeze/eval, and encode each
selected frame once. Save float embeddings and quantization metadata in new
artifacts so packet experiments do not reload the checkpoint.

At the existing six-bit latent precision compare these packaged methods:

1. Six-bit packed quantized codes, with full decoding header.
2. The same packed codes compressed by zlib.
3. Causal modular temporal deltas of the six-bit codes, packed and zlib-coded.
   `delta = (code[t]-code[t-1]) mod 64`; decoder uses cumulative addition mod
   64. First frame is absolute. Predictor state resets at segment boundaries.

Use independent segments of 1, 8, and 32 frames on the same 32-frame input.
Count complete files, not estimated entropy lengths. The versioned header
must include dimensions/order, frame count/start, quantizer bit depth,
min/scale arrays and their exact stored dtypes/shapes/endianness, checkpoint
identity, compressor identifier, payload length, and integrity check. A zlib
file carries its own entropy coding overhead; do not invent a zero-cost
Huffman code table. Preserve the quantizer's actual broadcast semantics.
Do not downcast min/scale metadata while calling the result lossless.

Round-trip packets to quantized codes exactly, then dequantize and reconstruct
using only decoded codes/metadata and the declared shared decoder. HNeRV's
current model call accepts source `gt` alongside an embedding: isolate the
embedding-to-image decoder and prove output independence from any source
tensor, rather than assuming that unused-looking argument is harmless.
Lossless packing must give identical reconstructed pixels within the verified
deterministic tolerance. Report latent-only bytes, separately packaged
decoder setup bytes, and setup-inclusive totals as different fields.

Only after that gate, and if >= 2 minutes of this stage remain, run a tiny
**4-bit versus existing 6-bit** latent quantization comparison on the primary
cut, using the same frozen decoder, no latent fitting, and identical
quantization-parameter rules. Decode both packets and score the B2 RGB/LPIPS/
temporal errors. Do not re-quantize an already quantized map and describe it
as a sweep of original float embeddings. No extra bit depths or codecs.

**Tests** in `tests/demo/test_hnerv_latent_packet.py`:

- Random/constant tensors with 3 and 4 channels, extremes 0/63, and wraparound
  temporal differences round-trip exactly for all three lossless methods.
- Min/scale shapes, dtypes, endian representation, and quantizer inversion
  match an independently specified fixture; zeros/constant ranges are finite.
- Truncated/wrong-shape/wrong-checkpoint/corrupt packets fail explicitly.
- Independent segment decoding needs no previous packet; future code changes
  cannot affect earlier output; header costs are counted in file sizes.
- Changing/zeroing/removing source tensors does not change reconstruction.
- Latent-only rates never include decoder weights; setup-inclusive tables do.
- Existing `_bits` estimates are labeled estimates and never substituted for
  file sizes. Lossless packaging changes rate, not quality.

**Gate:** a temporal packet saving >= 10% over the packaged six-bit raw
baseline with identical codes/pixels merits a later temporal-code study.
If only zlib helps, report that; do not claim motion prediction. If no method
helps, stop. Quality near the old 18–21 dB remains a decoder/generalization
issue even when a smaller packet is found. A lossy point cannot be promoted
from reduced bytes alone; report its measured quality loss and geometry.

## B5 — Summarize causes and spending decisions

No additional GPU allocation. Build a new table with factory/segment source
hashes, checkpoint/source identity, structure, QP or latent precision,
actual file bytes/kbps, setup bytes, RGB PSNR, LPIPS, temporal error, timing,
and pass/failure reason. Include native decoded-frame sheets. Every scalar
links to a machine-readable record; preserve negative and inconclusive arms.

Answer these decision questions separately:

1. Can the currently loaded checkpoint/source be identified and decoded
   without source frames, with correct RGB metrics?
2. Does pretrained LD/HT-S work while the fine-tune fails, or do both fail?
   Is the failure on I frames, prediction frames, or after a reset?
3. Is HT-L healthy on the eight-frame cut? What remains untested about its
   longer prediction chains and historical resume/checkpoint provenance?
   Do not imply a 32-frame HT-L test: B3 prioritizes LD, with optional HT-S.
4. Does HNeRV temporal packaging lower actual latent bytes, and does its
   frozen reconstruction quality justify more representation work?
5. Which **at most two** follow-up experiments have enough evidence to merit
   GPU time, and which paths should remain parked?

Do not choose a codec from a single favorable frame, mean PSNR alone, or an
unmatched duration. Do not combine background reconstruction with foreground
metrics and call it a full PointStream score. Make no reuse-across-days,
real-time, BD-rate, or general compression-win claim from these smokes.

Run focused tests:

```bash
python -m pytest -q tests/demo/test_factory_bg_rd.py tests/demo/test_neural_bg.py tests/demo/test_background_provenance.py tests/demo/test_background_codec_smoke.py tests/demo/test_hnerv_latent_packet.py
```

Add any shared-module tests affected by your changes. Do not modify the fleet
or monitor merely to get through a smoke failure; if infrastructure changes
are necessary, stop and explain the scope. Their mandated checks are
`tests/experiments/test_resource_claims.py`, `test_gpu_fleet.py`, and
`test_job_monitor.py`.

Deliver scoped code/tests, provenance/input manifests, packets/decoder
fixtures, budget ledger, contact sheets, reports, and a short decision memo.
Each gate is `passed`, `failed`, `inconclusive`, or `blocked`. Stop when this
bounded evidence is collected or budgets/prerequisites stop progress. A
passing smoke never authorizes full training or a larger hold-out sweep.
