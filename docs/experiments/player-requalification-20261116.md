# Experiments: player requalification implementation and release

**Current status:** all three approved GPU replays and independent audits pass.
The source-aligned [user review](player-replay-20261116/human-review-import.json)
is imported: Player1 uncertain; Player2 prefers common tennis and reports a
stock torso-facing hallucination. Common tennis is the provisional next
candidate. See the [motion proposal](player-motion-20261116.md) and
[live ledger](player-replay-20261116/release.json); the new allocation is unreleased.

### 2026-11-16 — CPU audit complete; paired GPU replay awaiting release

The next research action is a **paired comparison of three existing player
profiles**, followed by human viewing before choosing a representation. No new
training is needed. This record implements PLAN stage A; it does not report a
new player-quality or combined player-and-ball result. The previous automatic
compute window ended November15 at12:07:34CET. No new GPU job was submitted.

## What is now checked

- The [saved receipt join](player-profile-join-requalification-20261116.md)
  finds agreement across12 selected match/adapted generation jobs: recorded
  fit checkpoint, match, merge hash and original weights agree; all six
  tennis-base cases agree on base85e3c65c. This is receipt agreement, not
  authentication of every historical crop/mask or training interval.
- The [W8 byte audit](w8-inputs/receiver-20261101/player-conditioning-byte-audit-20261116.json)
  independently authenticates the packet, profile, CPU-proof archive and both
  actual conditioning NPZ members. Both units contain240 consecutive native
  ticks in one run, valid boxes, RGB references and nonempty pose maps.
- The [source/checkpoint byte audit](w8-inputs/receiver-20261101/source-checkpoint-byte-audit-20261116.json)
  hashes the actual78,396,317-byte source video,48,878,567-byte common tennis
  checkpoint and7,562,875-byte match checkpoint, plus their complete archives.
  Hashes match the retained identities. No model weights were loaded by it.
- An actual preprocessing mismatch was found and fixed for **future fitting**:
  inference first resizes the reference to224×224 before CLIP processing;
  fitting previously sent the tall crop directly to CLIP's centre-crop path.
  The [conditioning audit](player-conditioning-requalification-20261116.md)
  distinguishes the corrected recipe from existing checkpoints. No retraining
  or retrospective replacement occurred.
- The [processor-tensor receipt](player-processor-tensor-preflight-20261116.json)
  compares the actual corrected `Trainer.encode` boundary against pinned
  diffusers/CLIP processor expressions on an asymmetric RGB fixture. VAE,
  pose and CLIP tensors agree exactly in float32 and after fp16 conversion.
  This uses recording model substitutes only to capture boundary tensors,
  never learned outputs. It is Mac CPU evidence, not Linux/CUDA or quality
  qualification. The old CLIP path differs; that alone proves no quality effect.
- The [receiver reuse audit](player-receiver-witness-design-20261116.md)
  corrects a stale account of W8: later job`20261106T005057Z-77c276fa`
  successfully sampled the real model at1/3steps under the reviewed isolation
  policy. Its kernel, tensor and denial receipts are useful leads. A fresh
  production30step complete learned receiver is still unproved.

The reference box and the first generated-frame box differ intentionally:
preparation derives the former from uncoded reference-frame pose and the latter
from coded run-level geometry. Equality is not an invariant. The future-pose
geometry/lookahead and target-frame reference remain disclosed offline inputs.
Body-only maps omit hands, face detail and racket control; the CPU renderer
comparison cannot certify the native DWPose distribution or all model quality.

## Frozen next test

**Hypothesis:** after authenticating inputs and conditioning, at least one of
the existing generic, tennis-only or match-adapted recipes yields usable player
reconstruction worth carrying into the paid combined-codec test. The competing
explanation is that the saved positives or negatives depend on incorrect
profile attribution, different conditioning/CFG, source-granted geometry,
reference/state costs, or temporal behavior rather than a deployable benefit.

The [frozen config](player-requalification-20261116/config.json) chooses
`nadal_kyrgios/000@120`, an already opened development source,1920×1080 at50/1Hz.
Use both players and mechanically select ticks0–23, one complete24frame context
per player. This is **0.48seconds**, not a prefix-equivalent rerender of the old
240frame sampler, a representative confirmation cohort, or a temporal-quality
conclusion. Save every generated tick; do not judge only every third tick.

| Profile | State loaded on the same released original weights |
|---|---|
| `original` | No tennis or match fit |
| `tennis_only` | Frozen common tennis checkpoint85e3c65c, once |
| `match_adapted` | Same tennis checkpoint plus nk-C525830aa, using the original4bit match quantizer and guider delta |

The common-base fit receipt declares its pose guider frozen; the tennis-only
profile retains the original guider. This is consistent with the base/match
merge path, and does not attribute match-trained guider improvements to the
common tennis LoRA.

All profiles use identical authenticated references, poses, boxes, ticks,
unit-derived paired seeds,512×784 output,30DDIMsteps, context24/overlap4 and
**CFG3.5**, the stock default. This changes the historical matchCFG2 recipe;
do not label the new output a reproduction of its old pixels. No new fitting,
seed search, resolution reduction, or lower-step quality surrogate is admitted.

The [input inventory](player-requalification-20261116/inputs.json) names exact
archives and member hashes. Only logical staged paths enter the config; all
runtime inputs are resolved to explicit host-local bindings and rehashed. The
common model is an installed frozen resource; match adaptation and references
remain content-specific. Original multi-GB weights are presently existence/
size-checked and historically hash-declared; actual fleet staging and the
worker must authenticate their full bytes before model loading.

## Implemented production-path controller

`experiments.tennis.player_requalification` reuses the actual `Renderer` and
sampler. Each per-profile fleet job runs a one-player24frame correctness smoke
and a two-player24frame full context stage. Across three profiles this is
216generated frames, including repeated smoke, rather than144 unique outputs.
It checks settings, input/member hashes, rational FPS, profile/base lineage,
all expected ticks and geometry before importing Torch, then recorded model
loading and CUDA residency before sampling. All available loading checks run
before spending30steps. PEFT unexpected-key facts remain unrecorded by the
old loader and are explicitly unverified.

The worker preserves each completed NPZ/sidecar, validates full as well as
smoke output, and seals a publication. Missing frames, wrong settings/hashes,
changed input/profile, invalid model loading, or stale output fail. Identical
static frames remain valid and are reported as a diagnostic. A separate paired
receipt comparison requires identical conditioning plans across all profiles.
The validator recovers staged bindings from the actual fleet dispatch, because
fleet validators do not expand staged placeholders. Synthetic tests exercise
that plumbing and publication; they are not GPU qualification.

This controller is **not an isolation boundary or complete codec endpoint**.
The [receiver contract](player-receiver-witness-design-20261116.md) separately
specifies native-base→players→ball composition, every-frame fallback, paid
state, declared installed resources and fresh source-denied decoding. Retained
W8 runtime proofs do not certify changed code/grants automatically.

## Budget proposed for this test only

**Release requested: at most5,400GPUseconds(1.5GPUhours) and7,200CPUseconds
(2CPUhours), including preparation, validation, publication, audit and failures.**
This does not authorize training, additional sports, bulk data acquisition,
independent confirmation, submission, or the subsequent combined experiment.

| Per profile, three sequential requests maximum | Cap |
|---|---:|
| GPU-held whole campaign, including staging and both phases | 1,800s |
| Correctness smoke, one complete context | 600s |
| Full phase, two complete contexts | 900s |
| Validator | 60s inside the campaign, not extra GPU allowance |
| CPU, one-core campaign plus separately capped external preparation/audit | 2,400s |

The three campaign GPU caps sum to5,400s. Their one-core CPU lifecycle caps
sum to5,400s, with at most600external CPU seconds per profile(1,800total).
No stage or external attended audit may exceed45minutes; no automatic retry or
budget transfer. Observe actual complete lifecycle, not just sampling/profiler
time. Initial shared-GPU wall times cannot establish throughput or real time.

Historical Ada timing2.76s/frame gives an illustrative596s steady sampling
for216frames,894s with1.5margin, **before** startup, hashing, profiling,
publication and validation. It is not a measured new forecast or lower bound.
The [cache metadata snapshot](player-requalification-20261116/cache-metadata.json)
finds11/12 required data/runtime files on **both** compatible Ada hosts; only
the small conditioning archive is absent. Both use separate local filesystems
with ample free space. These are metadata observations, not byte validation.
Reinspect all eligible hosts, caches, jobs and worker capabilities at release.
Use the complete compatible Ada pool with memory admission; preserve other
users' work. Stop before a cold-copy change or measured first job breaks the
complete remaining forecast. Do not turn an unavailable host into a GPU retry.

The pure [spec builder](../../experiments/tennis/player_requalification_specs.py)
produces actual schema-checked requests, never submits them:

```bash
python -m experiments.tennis.player_requalification_specs \
  --config docs/experiments/player-requalification-20261116/config.json \
  --inputs docs/experiments/player-requalification-20261116/inputs.json \
  --deadline ACTUAL_RELEASE_DEADLINE_WITH_TIMEZONE --output NEW_LOCAL_SPEC_DIRECTORY
```

After approval, freeze the code revision; publish/readback the exact config
under the external data root; refresh the deadline and compatible hosts;
qualify/record actual workers; then submit **one** first-profile job. Record its
ID, bounds and result directory here before one canonical bounded fleet wait.
Unknown submissions are inspected, never replayed. Admit profiles2/3 only if
the first output, cost and remaining allocation pass; close failures explicitly.

## What this can decide, and what follows

First establish authentic complete outputs and inspect all three profiles
without a10% quality floor. Then present randomized, explicitly labelled
source/reference/player views to the user before discarding a representation.
A24frame one-source pilot cannot select a model for all tennis or establish
statistical significance; it admits a longer source-spread development test.
Use existing full-rate successes/failures with their own qualified identities
for longer viewing where compatible. Source-only annotation is a separate task.

Actual paid startup matters. The [packet accounting replay](player-requalification-20261116/retained-packet-accounting.json)
counts the retained W8 packet at2,289,836bytes:1,897,395match bytes,
366,161reference bytes,23,876frame bytes and2,404window/framing bytes.
Its complete240frame/4.8second cold rate is3,816.39kbps. A declared7200second
reuse arithmetic sensitivity is46.32kbps; no such live reuse behavior is
demonstrated here. These are W8 numbers, not W7al028 corrections or new codec
quality results. Do not assign the whole4.8second bill to a24frame context or
hide state under a short clip's modeled low rate. The next combined test must
freeze actual reference coding/refreshes, session duration and cold/steady
accounting before outcomes; confirm any benefit at supported matched objectives.

Same-source ball hints/labels, receiver masks, actual complete learned decoding,
three paid base rates, effective ROI/DCVC and player-only/ball-only ablations
remain subsequent work. No matching ball stream is presently authenticated for
this source. Preserve absence/unknowns and select visible-ball windows by
source-only rules; never import the different RacketVision results as its ball
output. Rackets remain optional; egocentric reference completion does not block
this test. Fresh matches and source-level uncertainty remain confirmation.

## Verification and continuation

The CPU processor check uses a separate temporary environment with the pinned
JCV processor libraries on Mac CPU. Project gates use a clean CI-style Python
environment with the declared project/dev pins, so optional Torch libraries do
not silently change configured type checking. Configured Ruff and mypy pass
(668files); full pytest passes **2,650tests,39skipped** in136.02seconds, with
the existing multiprocessing/fork deprecation warning. The sole initial full-
suite failure was experiment-document header formatting; that was corrected
and the complete suite rerun. [Verification hashes and versions](player-requalification-20261116/verification.json)
bind these checks to the actual implementation. No local CPU test certifies a
GPU result.

Implementation review: [PR431](https://github.com/emanuele-artioli/PointStream/pull/431).

No model-choice, human-review, complete-codec or submission gate is passed by
this implementation. The immediate external input needed after code review is
the **new bounded compute release above**. Human viewing follows actual outputs.

### 2026-11-16 — user released the bounded replay

The user approved at most5,400GPUseconds and7,200CPUseconds, no training, no
automatic retries, with subsequent profiles gated on the first result. The
[authoritative release ledger](player-replay-20261116/release.json) records
per-profile reservations, conservative externalCPU charges and actual job states.
The immutable config was published/readback with its original eb67e4a9 hash.
All six fleet hosts were inspected, both compatible Ada workers are fresh, and
there were no nonterminal existing requests; large input/runtime caches remain
present on both hosts. This is metadata qualification, not byte verification.

First request: **20261116T122900Z-2a9a4035**, profile`original`, frozen code
`e9d0d000d0e7470d7b5d9b1edc878d7a835d9cdb`, sourceSHA
3a02487149f5b03a57fc2b2c9b26b2c20f8fac90e464dfa35c690d2e02351ba0. GPU campaign cap1,800s, one-core plus externalCPU
cap2,400s, deadline2026-11-16T13:10:51.533592Z; result directory
`/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261116T122900Z-2a9a4035`. Submission returned pending; no duplicate was
created. One canonical wait of at most2,700s follows. This client has no verified
model-resuming callback, so monitoring remains attended with no recurringLLM
timer. The other profiles remain held until the first correctness/cost audit.

The first source/output/cost gate passed after local frozen validation of both
actual24frame outputs. Whole-monitor conservative charge421GPUseconds and
1,021CPUseconds(including the full600externalCPU reservation); no phase
sampling counter was added again. The next tennis-only request is **20261116T124347Z-407d92df**, same sourceSHA/code/config, GPUcap1,800s/CPUcap2,400s,
deadline2026-11-16T13:26:30.385840+00:00. Its result directory is `/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261116T124347Z-407d92df`. The match-adapted profile remains held.

The tennis-only source/output/cost gate also passed, charged421GPUseconds and
1,021CPUseconds. Both completed profiles total842GPU/2,042CPUseconds. Final
match-adapted request: **20261116T125758Z-5dd33301**, same frozen code/config/sourceSHA,
GPUcap1,800s/CPUcap2,400s, deadline2026-11-16T13:40:43.724043+00:00; result directory
`/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261116T125758Z-5dd33301`. No training or extra candidate was released.

### 2026-11-16 — all three fresh replays pass; human pilot ready

| Profile | Job | Conservative GPU-s | Conservative CPU-s including600external | Source/output/cost gate |
|---|---|---:|---:|---|
| Released stock | 20261116T122900Z-2a9a4035 | 421 | 1,021 | PASS |
| Common tennis-only | 20261116T124347Z-407d92df | 421 | 1,021 | PASS |
| Existing match-adapted | 20261116T125758Z-5dd33301 | 361 | 961 | PASS |
| **Total** | Three completed requests | **1,203** | **3,003** | Within5,400GPU/7,200CPUmaximum |

All full outputs contain both players at each of24native ticks. Exact paired
plan equality passes for source, references, poses, boxes, seeds, config and
settings. Each job's repeated first-player smoke/full crops are also bit-exact
across28,901,376RGBvalues. This is repeatability within the same job/device,
not a cross-device test, source-denied reconstruction or240frame prefix proof.
The three jobs used different physical Ada GPUs; shared wall timing remains
contaminated and supports cost bounding only. No training, retries, new cohort,
extra model, subsequent codec experiment or unused-allocation transfer occurred.

See the [paired audit](player-replay-20261116/aggregate-paired-audit.json),
[repeat audit](player-replay-20261116/duplicate-context-replay-audit.json), and
per-profile audits in that directory. Raw publications remain immutable under
the fleet's data root; local viewing copies are separate. Task processes and
claims are released, and reported terminal events are acknowledged.

The private viewing tool authenticates all three sealed plans and outputs. It
also hashes the actual source video, checks native geometry/cadence/offset,
reproduces the original limited-YUV420→BT.709 RGB conversion and confirms both
frame0reference crops exactly(maxRGBdelta0). Source and generated videos have
identical24frame timelines at50Hz and optional labelled10Hz playback; PNGs
preserve exact tick12RGB. All16videos passed ffprobe checks and loaded in the
browser; source and generated playback were exercised with preferences blank
and labels hidden. There is no human decision or annotation approval yet.

The [delivery record](player-replay-20261116/review-delivery.json) names the
durable private HTML, bounded45minute loopback server, screenshot and exact
asset hashes. The server is read-only, has a65CPU-second hard cap, and has no
recurring model callback. Its cost is covered by the conservative externalCPU
reservation; browser elapsed time is not active human labour.

**Decision:** implementation/input/repeatability qualification is complete for
this24frame construction. Preserve all three profiles until human inspection
and a longer motion/source-spread test justify selection. No new rate–quality
curve, ball result, population preference, complete combined codec or submission
readiness follows. The next scientific test must resolve longer-context temporal
behavior and paid references/refreshes/session costs against native/ROI/DCVC
under the common player-and-ball contract. No new GPU stage is released here.
