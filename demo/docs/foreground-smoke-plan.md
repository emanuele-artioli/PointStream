# Foreground: implementation and decision plan, smoke only

## Assignment and limits

Implement this plan in order. Its output is trustworthy diagnostics and a
recommendation for the next experiment, not a trained production encoder or a
claim that PointStream beats AV1. Do not run full training, distillation,
dataset regeneration, a full hold-out sweep, website deployment, or fusion
with the main PointStream campaign. Do not download another model or dataset.

Hard limits for this assignment:

- At most **30 GPU allocation minutes cumulatively**, including model loads,
  warmup, failed attempts, and idle time while a claimed GPU waits for files.
- No GPU job may have a runtime allowance above **8 minutes**. Budget each
  stage below; do not launch all stages as independent 8-minute jobs.
- Training is restricted to the explicit tiny diagnostic runs in F4. Never
  call the current trainer's default 80-epoch loop.
- Bound remote CPU preparation to 10 minutes total; each file read/hash or
  transfer gets a 90-second timeout. A second identical failure ends that
  operation. Do not scan every conda environment or the whole shared disk.
- Caps are ceilings, not targets. Record actual allocation time in a ledger.
  Stop when the cap is reached; report missing evidence. Do not silently raise
  budgets, lower image resolution, or replace the execution environment.

Use an implementation branch `codex/demo-foreground-smokes`. Work from the Mac
checkout and follow its AGENTS.md. Do not start an agent session on a GPU host.
Do not change the background experiments, dispatcher, main campaign, or any
existing evidence. Commit coherent code/tests and push; do not open a PR,
merge, or deploy unless the user asks.

## Starting material and execution contract

Read `demo/docs/internal-audit-20261003.md`. The reference files are:

- `demo/experiments/compare_pose_on_sam_crops.py`: components, pose parsing,
  current permissive `choose_hand`.
- `demo/experiments/train_rtmw_hands.py`: crop/anchor loader and trainer.
- `demo/models/hand_objective.py`, `demo/models/unet_generator.py`, and
  `demo/pipeline/foreground_segmenter.py`: loss, architectures, letterboxing.
- `demo/experiments/hand_packet_rate.py`: legacy packet and delta codec.
- `demo/experiments/holdout_hand_rd.py`, `score_holdout_generators.py`, and
  `fair_crop_av1.py`: timing and the current scoring/selection errors.
- Existing tests in `tests/demo/test_hand_packet_rate.py`,
  `test_holdout_hand_rd.py`, `test_sam_crop_pose.py`, and tests covering the
  shared hand loss/model.

Several experiment files/tests are **untracked in the source Mac checkout**;
a clean Git checkout alone will omit them. Before implementing, read the
adjacent `smoke-plan-source-inventory.json`. Copy only this plan's listed seed
files into your scoped checkout, compare SHA-256 values, and record the seed
revision/checksums. Never stash, clean, reset, move, or commit unrelated work.
If a listed file is missing or differs, inspect the delta and document which
version you used; do not manufacture an empty substitute or assert the old
hash. Preserve a reviewed baseline of the needed seed files on your branch.

Canonical remote roots:

```text
/home/itec/emanuele/Datasets/pointstream-demo
/home/itec/emanuele/Datasets/pointstream-data
```

Training folders are `clip_01_factory001_worker001_00001/f000000-f035129`,
`clip_03_factory001_worker001_00000/f000000-f012629`, and
`factory002_worker001_00000/f000000-f035129`. Pose candidates are in
`sam_poses/rtmw-l.json`; images/masks are in `original/` and `masks/`.
Existing final hold-out material is under
`pointstream-data/jobs/hand-rd/holdout/<stem>/`, where stems end in `_last10s`.
Existing checkpoints/reports are under `jobs/hand-rd/train/` and `compare/`.
These locations are read-only inputs. Do not delete the compatibility data
symlink or rewrite old reports, checkpoints, or manifests.

Read [the current job workflow](../../docs/workflow/long-jobs.md) before
implementation. It supersedes the former direct Python fleet launcher. Use
only the absolute, standalone allow-listed entry point from the Mac; do not
wrap it in `bash -c`, change `PYTHONPATH`, invoke general SSH, or edit allow
rules. If the shell sandbox blocks networking, invoke that same absolute
command with `require_escalated`, as the workflow specifies.

Run these as separate commands (not one shell chain):

```bash
/Users/manu/Desktop/PointStream/scripts/ps-fleet doctor
/Users/manu/Desktop/PointStream/scripts/ps-fleet workers status
/Users/manu/Desktop/PointStream/scripts/ps-fleet inspect --hosts gpu1 gpu2 gpu3 gpu4 gpu5 gpu6
```

If workers are missing, use `workers start` only after `doctor` passes; it
reuses live workers. Do not restart healthy workers for this workload. A worker
upgrade, if actually required, uses verified `workers restart` and preserves
supervisors; never kill workers manually. Failed doctor/probes make those
hosts unavailable. No eligible capacity means a pending request until its
absolute deadline, not a reason to bypass admission or use the Mac for model
inference. Workers perform distributed admission and resource claims; do not
pin a GPU UUID from an old inspection. Require no compute processes, memory
<=256 MiB, utilization <=5%, declared peak +4 GiB free, and sufficient aggregate
CPU headroom. Encode compatibility with `gpu_models`, not a performance claim.

### Smoke-only requests under the two-stage schema

Implement schema-1 JSON specs for each bounded task. Submit with:

```text
/Users/manu/Desktop/PointStream/scripts/ps-fleet submit /absolute/local/path/task.json --chat-id CHAT_ID
```

Replace `CHAT_ID` with the submitting chat's actual ID (or use the documented
`CODEX_THREAD_ID` default); never invent it. Specs live outside the code tree.
Submission snapshots clean HEAD by default. Commit the scoped implementation
and needed seed files first, or append individually reviewed `--include-change
PATH` / `--include-untracked PATH` arguments. Never dispatch the whole dirty
checkout, particularly the unrelated local `experiments/jobs/fleet.py` patch.
The historical source inventory identifies seeds; its infrastructure checksum
is not authorization to include that patch. Use the current tracked manager
and record the current implementation revision and selected patch hashes.

The dispatcher always executes `smoke`, its validator, then a stage named
`full`. **Here `full` means only the remainder of this bounded diagnostic; it
never authorizes full training, a dataset sweep, or any longer experiment.**
Make both stages use the same runner, subcommand, input manifest, processing
path and fixed settings. Vary only whole-argument scale placeholders for a
representative subset/count. Do not make the first stage a trivial import
check and hide all meaningful processing in the second.

Required spec fields and constraints:

- `schema: 1`; `hosts` chosen from gpu1–gpu6; compatible `gpu_models`;
  `gpu_memory_mib` from the estimates below; `cpu_threads: 4`.
- `entrypoint` is one `[-m, module]` array or one relative `.py` script;
  `arguments` is an argv array. No shell command, inline Python, or alternate
  unrestricted launcher. Use real JSON strings, e.g. `["-m", "demo.experiments.foreground_smoke"]`.
- Nonempty `scale`, with `smoke` and `full` values per key; each placeholder
  such as `{frames}` occupies a whole argument. Freeze source identities,
  checkpoints, resolution, seeds, codec settings and selection policy across
  stages. Count work in **both** stages against this plan's caps; do not double
  the specified matrix or silently reset the training budget. A precheck may
  use a nested subset, but repeated processing still consumes time/updates.
- Nonempty `inputs`: absolute paths under the external data root plus actual
  lowercase SHA-256 values. Use an immutable selected-input manifest, with
  constituent hashes checked by the workload validator. No placeholder hashes
  or mutable latest-file identities.
- `smoke.seconds`, `full.seconds`, `validator_seconds`, and `budget_seconds`
  must reserve validation, model loads and overhead. `budget_seconds` <=480
  and <=this task's remaining stage allowance; the sum of submitted budgets
  must fit the plan's 1800-second ceiling. A two-minute task has a <=120-second
  total budget, not two two-minute stages. `smoke.seconds` <=600 is the schema
  limit, but this plan's tighter total limit always wins.
- Nonempty `smoke.representative_basis` describes the fixed representative
  subset and scientific path exercised. Supply `required_commands` for tools
  actually invoked, a future `deadline` ISO timestamp with explicit timezone,
  and `stall_seconds` no larger than the task budget. Choose a finite deadline
  for the current attended work window and record it; never use a stale date
  copied from a prior report or extend it automatically.
- `validator` is an argv array, using `{python}` only for the manager's Python
  where appropriate. It must exit zero and write `passed: true` and nonempty
  substantive `checks` to `PS_VALIDATION_PATH` only when the relevant gate
  passes. Missing/corrupt artifacts, nonfinite metrics, wrong input identity,
  or decoder/source dependence fail validation; exit zero alone is insufficient.

Implement runner/validator tests before submitting: validate specs locally
with `experiments.jobs.inbox.validate_spec`; reject over-cap combined work,
missing hashes, invalid scale arguments and absent substantive checks; verify
validator failure blocks the second stage. Test the same bounded runner on
fixtures without CUDA locally. Do not launch `selftest` automatically: it is
additional infrastructure compute, not the model smoke requested here.

The entrypoint starts in the configured worker interpreter; schema 1 has no
per-job Python/environment override. If the model needs an existing dedicated
interpreter below, implement an explicit bounded Python adapter in the workload
runner: use an argv subprocess with that exact installed interpreter, propagate
`PS_*` and `CUDA_VISIBLE_DEVICES`, retain the owned process group, and enforce
remaining time/count caps. Never use inline `-c`, a shell, installations, or a
new remote session to bypass the job contract. Record both interpreters and
native tool versions. Test argument/environment propagation and failure/timeout
handling. If the installed environment cannot execute that path, report blocked.

Write workload artifacts under `PS_STAGE_DIR`, not the shared supervisor root
`PS_JOB_DIR`. Keep smoke and diagnostic-stage outputs separate, preserve
completed evidence, and publish progress through
`experiments.jobs.monitor.publish_progress(stage, completed)` only when actual
work completes. The validator reads smoke artifacts; reports identify the stage
for each scalar. Dispatcher metadata, code/spec/input identity gates, tool paths,
GPU UUID and resource usage remain preserved. Publish measured peak GPU memory
and a cumulative ledger; model loading, validation, failures and filesystem waits
after admission all count. Label both stages' outputs diagnostic/non-citable.

### Monitoring and recovery

After submission, register or update **one native Codex heartbeat per submitting
chat**, every five minutes, covering all its jobs. Follow the exact saved prompt
in `docs/workflow/long-jobs.md`: run the absolute `ps-fleet watch CHAT_ID`, stay
quiet for unchanged/non-actionable state, combine terminal/stall/contention/
budget/deadline/decision events, and acknowledge stable event IDs only after
reporting them (`ps-fleet ack EVENT_ID ...`). Pause it when all watched jobs are
terminal. Do not create one automation per job. If native automation tooling is
unavailable, report that monitoring prerequisite as blocked before submission;
do not substitute a shell cron or an unmonitored detached job.

Use the same absolute entrypoint with `status JOB_ID` and `events JOB_ID` for
read-only recovery, including existing legacy job IDs. Workers/supervisors survive
Mac sleep and disconnection. On uncertain status, inspect saved request,
supervisor/claim identity and logs; never automatically replay, migrate, extend
budgets, or cancel. Recovery from worker death/reboot uses `doctor` then
`workers start`. Cancellation, when authorized, uses `cancel JOB_ID` and only
owned processes. Preserve every interrupted/completed directory. No heartbeat
may submit new work or cancel jobs.

For SAM requests restrict `gpu_models` to `["RTX A6000", "RTX 6000 Ada"]`;
RTX 8000/GV100 remain excluded for this path.

Initial conservative memory requests: 12,000 MiB for generator/pose smokes,
38,000 MiB for an optional SAM smoke. SAM must be a separate job/process so
its weights are not resident beside generator/pose models. Use the existing
`/home/itec/emanuele/.conda/envs/pointstream/bin/python` as the initial training
candidate; verify `torch.cuda.is_available()` and the actual imported versions
before fitting. SAM uses the existing `pointstream-sam31` environment. For
ONNX, use an installed environment that actually passes the per-session CUDA
profile check, not one picked merely because its name sounds appropriate.
Do not inject a different cuDNN into a running PyTorch process or silently
fall back to CPU. If none of the existing environments passes, mark the
runtime stage blocked and preserve the import/provider diagnostics. Requests
may be lowered only using measured same-path peak-memory evidence, not a
smaller input or optimistic estimate.

Create a new bounded runner, e.g. `demo/experiments/foreground_smoke.py`, with
subcommands `audit`, `packet`, `fit`, `profile`, and `compare`. Its defaults
must be small, and it must reject counts/steps exceeding this plan. It must
not delegate to an unbounded old CLI. Unit tests may run locally; all real
model/image processing smokes run on a GPU server through the dispatcher.

## F1 — Audit the candidates and define the tiny input manifest

Use existing masks/poses; do not run SAM over the dataset again. Read the
three pose JSON files (roughly 2730 frame records); inspect image pixels only
for a deterministic sample of at most **24 candidates per recording**.
Budget: <= 2 GPU allocation minutes if dispatch claims a card; CPU work is
also included in the preparation cap. Cache only the selected inputs.

Write an append-only audit manifest with source recording, source frame index,
component/candidate ID, bbox, raw per-joint scores when available, raw mean
score, selection reasons, split, and image/mask hashes. Existing JSON lacks
per-joint scores; write `null`, never fabricate 21 values from a mean.

Compute separate diagnostic flags:

1. Invalid/nonfinite/wrong-shaped points or bbox; zero-area crop.
2. Fewer than 8 distinct joint positions (coordinates rounded to 0.1 pixel).
3. Joint containment below 0.5, measured against the candidate's own mask
   component, not every hand/arm component in the image.
4. Zero/collapsed bones, extreme relative bone lengths, bbox-edge clipping,
   duplicate candidates, and abrupt temporal changes. Record measurements;
   do not pretend these heuristic thresholds are labeled ground truth.

The pose head returning 21 locations is not evidence of 21 detected joints.
Mean scores often exceed 1; do not interpret them as probabilities or reuse
the old 0.25 threshold as a calibrated hand classifier. Do not reject an
occluded hand solely because some joints lie outside the visible mask.

Build sheets with original crop, local component mask, overlaid joints, and
the appearance anchor. Sample up to 8 apparently plausible, 8 flagged, and 8
ambiguous candidates per recording, uniformly across sampled seconds. If a
stratum is smaller, keep its actual size rather than duplicate examples.
Visually inspect the sheets and label only these sampled items
`visible_hand`, `non_hand`, or `uncertain`, with a short reason. Model outputs
or mask containment alone must not supply those labels. Keep all source data;
the manifest is a proposed selection, not a rewritten dataset.

Clip 3 source seconds 210, 240, and 420 must be excluded from fitting; 0 stays
eligible under the user-approved recipe. Inspect `batches.json` and the old
`look` flags: the pose script defines 0 as `look`, so do not blindly inherit
that older policy. Preserve the original flags and record the explicit
override. The last 10 seconds of each recording are final diagnostic
hold-outs and never become fit/validation/anchor inputs.

From reviewed sampled-training candidates, freeze at most 16 fit crops and
8 validation crops for factory001. Use different source seconds for those
sets, include both sides when available, and avoid adjacent-frame leakage.
Anchors come only from fit seconds, per recording/track where appropriate.
Do not pick the highest raw confidence crop without checking that it is a
hand. Record setup anchor files/bytes. Factory002 is a data audit only in
this assignment; do not run another training matrix there.

**Gate:** at least 8 reviewed visible-hand fit crops and 4 validation crops,
with no shared source second or final hold-out identity. Otherwise stop F4:
report that correct training inputs are not yet established. Uncertain cases
remain available for diagnostics, not silently relabeled as hands.

## F2 — Repair the selection and decoder/scorer contract

Implement a canonical selection/track manifest reused by packet encoding,
AV1 reference extraction, generator conditioning, and metrics. Do not key
identity exclusively by left/right labels: two distinct hands can receive
the same label. Use explicit stable integer `track_id`, a separate nullable
handedness field, and presence events. For the smoke, a deterministic
association using bbox overlap plus normalized joint-center distance is
enough; log its costs and every reset. No future frame may affect a causal
association. Resolve duplicate overlapping detections once, before encoding.
If more than two unresolved physical hands remain, mark the frame unsupported
and report coverage; do not silently score a convenient pair as all hands.

Prefer a **new versioned experimental codec** over silently changing v1:

- Fixed-endian binary header: magic/version, frame dimensions, rational fps,
  segment start/count, coordinate units, records/counts, payload length,
  compression method, and CRC. Define exact field widths in a schema.
- Decoder output includes track ID, handedness, presence, bbox, and joints.
- For the correctness baseline use uint16 full-frame coordinates in units of
  1/16 pixel, for bbox and all 21 x/y pairs. At 1920x1080 this fits uint16;
  rounding error is <= 1/32 pixel per axis. Reject out-of-frame/nonfinite
  coordinates with an explicit reason rather than silently clipping them.
  The baseline may cost more than v1; it is a diagnostic, not a final choice.
- All masks, reference images, original bboxes/points, and candidate lists are
  evaluator/encoder data. A generator decoder receives only the decoded
  packet, checkpoint, and explicitly declared setup anchors.
- Crop the source reference using the **decoded** bbox and the same integer
  rounding/letterbox transform used to render decoded joints. Never use the
  original bbox at the decoder. Never fall back to unsent original joints.
- Generator and AV1 score identical track/frame/crop identities. Empty slots
  remain empty, and missed/unsupported frames appear in a coverage table.
  Keep source foreground coverage separate from conditional crop quality.

Use an explicit schema for setup metadata (anchor and checkpoint identities)
and report actual serialized anchor bytes separately. Ground-truth masks may
define evaluation regions/targets but may not mask a generated prediction at
the decoder unless their transmitted bytes are included. Predicted alpha is
the intended decoder-side mask.

**Required tests** in a new `tests/demo/test_foreground_contract.py`:

- Duplicate same-label candidates; two legitimate distinct same-label hands;
  third unresolved candidate; left/right label swap; track disappearance and
  reappearance. Verify identical IDs used by AV1 and generator.
- Encode/decode 0/1/2 hands, boundary coordinates, negative/nonfinite inputs,
  odd box dimensions, and non-square crops. Compare against independently
  specified expected decoded records and <= 1/32 pixel quantization error.
- Poison/delete original points and bboxes after encoding: the decoder's
  skeleton/crop transform and prediction must be unchanged.
- A candidate not present in the packet cannot produce a scored prediction.
- Joint marker and source-pixel marker align after letterboxing, including
  clipping/rounding at image edges. Match the geometry helper's actual crop.
- Truncated packet, invalid version, wrong CRC, excess count/length, trailing
  bytes, and impossible bbox fail explicitly before inference.
- Every segment decodes independently; the first event after a reset contains
  complete state. Changing future frames does not change earlier decoding.

**Gate:** all tests pass and the runner produces a decoder-only smoke from a
single saved packet. Otherwise stop fitting and RD comparisons.

## F3 — Correct alpha supervision and comparison semantics

Use a smoke-specific objective/adapter so the main campaign's loss is not
silently changed. Keep legacy model/checkpoint loading explicit and intact.
For a fair tiny fit, construct both architectures with RGB+predicted-alpha
outputs; the new four-channel pix2pix head cannot load an old three-channel
checkpoint as if compatible. Version checkpoint metadata and fail on shape
mismatch rather than `strict=False`.

Use RGB in [-1,1], alpha in [0,1], and black background -1. Compose
`shown = alpha * rgb + (1-alpha) * (-1)`. Train with:

```text
L = masked RGB MAE + balanced alpha MAE + full-crop composite MAE
```

Masked RGB MAE divides by `3 * sum(target_alpha)` with an empty-mask guard.
Retain the original, un-matted source RGB crop for this target construction;
apply the soft target alpha once. Do not multiply an already black-matted
RGB target by soft alpha a second time and call that the same reference.
Balanced alpha MAE is the average of foreground and background region MAEs
over regions that exist, against the soft target alpha; do not give a missing
region zero weight in a two-region denominator. Composite MAE compares to the
source RGB composited with target alpha using the same convention. Use unit
weights for this diagnostic; no hyperparameter search, GAN, or LPIPS training
loss. LPIPS is evaluation-only. Apply the same objective to both new models.
Evaluate predicted-alpha composites, and separately report inside RGB error,
foreground opacity error, outside alpha leak, and mask IoU at alpha >= 0.5.

**Required tests** in `tests/demo/test_foreground_objective.py`:

- Exact RGB+alpha target yields zero loss within numerical tolerance.
- All-zero alpha with correct foreground RGB has positive loss; its gradient
  pushes foreground alpha upward. All-one alpha on a mixed mask has positive
  outside penalty and the correct downward gradient there.
- Empty/full masks remain finite and produce correct region denominators.
- Arbitrary RGB outside an exactly zero predicted alpha cannot affect the
  displayed composite. A leak can affect it and is penalized.
- RGB channel normalization, black convention, BGR/RGB conversion, and
  per-image versus batch aggregation are consistent; test unequal mask areas.
- Decoder-side compositing requires no source mask. An RGB-only checkpoint is
  labeled legacy; it is not assigned an oracle alpha for a fair comparison.

**Gate:** tests pass. Re-render at most 8 existing-checkpoint crops for raw RGB
versus predicted-alpha diagnostics (<= 2 GPU minutes). Call these diagnostic
renders, not a new fair architecture ranking.

## F4 — Tiny optimization/capacity smoke, not model selection

Budget <= **8 GPU minutes** for this stage; load once per arm. Freeze F1's
16/8 manifests and anchors before observing losses. Use seed 1234, 256x256,
batch size <= 8, Adam lr 2e-4, betas (0.5,0.999). Save initialization identity,
parameter count, train/validation metrics at steps 0, 20, 60, and 120, and
source/checkpoint hashes. Optimizer steps, not epochs, are the cap.

Run only three arms: legacy-objective SPADE diagnostic, corrected-objective
SPADE, and corrected-objective four-channel pix2pix. Each arm stops at the
first of 120 updates or **120 seconds including model initialization**. No
resuming the old full-set checkpoint, extra seeds, or increasing capacity.
The legacy arm is a loss ablation, not a fair competitor; distinguish it in
the report. Do not compare raw legacy summed-channel L1 with normalized new
MAE. Measure all arms with the common corrected evaluation metrics.

Stop an arm immediately on nonfinite loss/gradient, OOM, empty batches,
wrong alpha range, identity mismatch, or exhausted cumulative budget. Preserve
the failed checkpoint/log; do not try a smaller resolution or another card.

Diagnostic thresholds, fixed before the run:

- Fit masked RGB MAE and composite MAE should each fall >= 25% from step 0.
- On visible-hand fit crops, predicted foreground opacity MAE and outside
  alpha MAE should each be <= 0.15 at the last saved step.
- Validation composite MAE worsening > 20% from the best recorded validation
  point flags overfit; it does not authorize more training or a new split.

Failure means investigate input alignment/objective/optimization. Passing
means the path can learn these few crops, not that it generalizes or beats
AV1. If the 120-second cap prevents a meaningful result, report inconclusive;
do not complete 120 steps outside the cap. Save the first/last predictions,
anchors, skeletons, target/predicted alpha, and reference crops on one sheet.

## F5 — Profile the existing runtime before considering another model

Budget <= **6 GPU minutes**. Use 16 fixed saved images across the recordings,
native resolution, preloaded image/mask arrays; then repeat with disk reads
included. Record per-stage times for read, component extraction, input
preprocessing, H2D, inference, D2H/postprocess, crop/skeleton, packet encode,
generator, and composite. Count actual crops and frames separately.

Inspect each actual ONNX session. Merely listing CUDA as an available provider
is insufficient: record configured providers and a bounded ORT profile of
actual node execution. CPU fallback must be explicit and separately reported.
Do not call it GPU timing. Record warmup (3 calls), median/p95, total time,
thread counts, memory, CUDA synchronization, and model load time separately.

Compare current per-component calls with batching supported by that exact
model wrapper. Test batched outputs against single outputs before timing;
if unsupported, report it rather than hacking shape assumptions. Compare
RTMW-l and the already-present RTMPose-m hand head on the same crop set only
after provider correctness. No model download or distillation in this plan.

SAM: reuse saved timings first. If enough budget remains, one 30-frame native
training second, the existing hand/arm union path, max **2 GPU minutes**; stop
if that cap expires. Do not use an 8-frame SAM input if the production path
requires longer chunks. Do not lower SAM resolution/prompts to claim speed.
This SAM allocation must also fit within the 6-minute profiling budget.

**Tests:** single/batch coordinates agree within the model's documented
tolerance (initial diagnostic tolerance 1 pixel); provider fallback is detected;
timing rates distinguish crops/s from frames/s; stage sums reproduce total
within timer overhead; claimed 24 fps requires the complete encoder pipeline
<= 41.67 ms/frame including all hands, not just a pose or generator call.

**Gate:** if IO/provider/batching explains the bottleneck, recommend that fix.
If corrected pose remains slow, recommend a bounded smaller-head/distillation
study later. If SAM alone exceeds 41.67 ms/frame, state that the present full
pipeline is not real time regardless of pose improvements.

## F6 — Lossless packet smoke and a small diagnostic RD table

Budget <= **4 GPU minutes**, plus bounded CPU packet work. Use exactly the
same **8 frames starting at hold-out index 120** in each of the three existing
hold-outs. Do not fit, choose anchors, or tune thresholds on these final cuts.
If those inputs are missing, stop; do not substitute a favorable segment.

Compare actual packaged bytes for the new correctness baseline: raw packed
records, zlib over those records, and causal temporal residuals plus zlib.
For residual coding, uint16 modular deltas must reconstruct quantized codes
exactly and reset at each independently decodable segment. Keep coordinate
precision unchanged. Test segment lengths 1 and 8 on the same eight frames;
sum eight one-frame files versus one eight-frame file. Count all headers,
CRC, padding, presence/reset events, and codec state. Report setup anchors and
weights separately. Do not discard joints, add 4/6-bit quantization, or import
the main campaign's codec yet.

Encode the exact canonical matted reference tracks with AV1 at square 64,
128, and 256, CRF 63/preset 7, using the shared AV1 helper, GOP equal to the
segment. SVT-AV1 cannot encode below 64. Decode to 256 with one documented
resampler. Score the corrected smoke models only where comparable identity
and coverage are defined, plus explicit unsupported/missing counts. Report
masked/composite MAE, predicted-mask IoU, LPIPS, packet joint errors, and a
sheet. LPIPS alone is not a hand-pose correctness metric. Pixel-based joint
comparison measures agreement with saved model predictions, not true joints.
Label all RD numbers as tiny, diagnostic cuts; do not extrapolate them to
ten-second rates or compute BD-rate from three unmatched points.

**Tests:** independent decode agrees with raw quantized records; packaged
byte count equals file size; no hidden future state or source-coordinate
fallback; unchanged coordinate codes yield identical predictions across
lossless packet methods; FPS/duration counts include both hands once.

**Promotion signal:** a lossless alternative saving >= 10% actual bytes over
the new raw baseline with identical reconstructed codes merits a later larger
packet experiment. Failure is a negative result, not permission to reduce
precision. Joint pruning/learned motion codes remain proposed experiments,
requiring topology and error/residual escape tests in a later assignment.

## Verification, artifacts, and final stop

Run existing focused tests plus the new contract/objective/runtime tests:

```bash
python -m pytest -q tests/demo/test_hand_packet_rate.py tests/demo/test_holdout_hand_rd.py tests/demo/test_sam_crop_pose.py tests/demo/test_foreground_contract.py tests/demo/test_foreground_objective.py tests/demo/test_foreground_runtime.py
```

Run any existing tests covering a shared module you changed; enumerate them
with `rg` rather than guessing a filename. If dispatcher/monitor code changes
were somehow necessary, stop and justify that scope first; the required
infrastructure checks are:

```bash
python -m pytest -q tests/experiments/test_resource_claims.py tests/experiments/test_gpu_fleet.py tests/experiments/test_job_monitor.py tests/experiments/test_fleet_inbox.py
```

Deliver code/tests, the frozen audit/fit/validation/cut manifests, packet schema,
actual packets and decode fixtures, provider/timing reports, smoke loss curves,
contact sheets, and a decision report. Use statuses `passed`, `failed`,
`inconclusive`, or `blocked`, with evidence path and budget consumption for
every gate. Final report must answer: are the crops credible; is conditioning
decoder-only; can corrected SPADE learn; is opacity fixed; which runtime
stage dominates; does temporal packet coding help at unchanged precision?

Stop once those questions have bounded evidence or a recorded blocker. Never
start a full run because a smoke passes. Recommend at most two next GPU
experiments with a hypothesis and the evidence that justifies their cost.
