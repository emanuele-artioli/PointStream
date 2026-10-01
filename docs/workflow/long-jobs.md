# Long jobs: local dispatch and bounded research

## Dispatch from the Mac

Use `experiments.jobs.fleet` for each remote CUDA request. The local coordinator
inspects all selected hosts, rejects incomplete probes, estimates job resources,
selects a compatible idle GPU, snapshots the code, and launches the existing
remote supervisor. There is no persistent queue and no remote Codex event
delivery dependency.

```bash
python -m experiments.jobs.fleet inspect --hosts gpu1 gpu2 gpu3 gpu4 gpu5 gpu6
python -m experiments.jobs.fleet launch --hosts gpu5 gpu6 \
  --gpu-memory-mib 12000 --cpu-threads 8 --budget-hours 2 \
  --require-path /home/itec/emanuele/pointstream-data/assets/dataset/alcaraz_highlights/segmentations/scene_000 \
  --require-path /home/itec/emanuele/pointstream-data/outputs/bp21-headroom/clips/alcaraz_highlights/scene_000/window \
  -- /home/itec/emanuele/.conda/envs/pointstream/bin/python -c \
  'import os; from experiments.tier.run import main; raise SystemExit(main(["--tiers", "fast", "--frames", "8", "--out", os.path.join(os.environ["PS_JOB_DIR"], "report.json")]))'
```

Admission requires no GPU compute process, at most 256 MiB memory use, at most
5% utilization, enough free memory for the declared estimate plus 4 GiB, and
CPU headroom for the requested threads. The monitor claims by canonical host
and GPU UUID and repeats the occupancy/memory check immediately before launch.
Pass only compatible servers with `--hosts`; the dispatcher cannot infer
application-specific GPU requirements from an arbitrary command. Do not infer
availability from utilization or scheduler state. Hardware ordering
(Ada, A6000, RTX 8000, GV100) is a fallback heuristic until comparable workload
timings exist. When they do, pass `--prefer-gpu-name SUBSTRING` once per model
family in measured performance order; the selected order is recorded in the
manifest. The present pilot has no comparable cross-model timing.

The default snapshot is clean `HEAD`. Explicitly add only intended changes with
`--include-change PATH` and new files with `--include-untracked PATH`. The local
manifest records code and patch checksums, command, runtime environment, GPU,
and remote run directory. Jobs use the shared external data root and unique
remote run directories; they never overwrite a working checkout.

If unpacking code on shared storage is slow, pass an owned server-local root,
for example `--snapshot-root /var/tmp/emanuele-pointstream-fleet-snapshots`.
Only the unique, checksum-verified frozen code snapshot moves there. Inputs,
run logs, outputs and cooperative resource claims stay on the external data
root, and the manifest records the actual code location. Retain the normal
admission checks and smoke requirements; a failed transfer is not a codec result.

The remote supervisor survives SSH disconnection and laptop sleep. Inspect a
job with `python -m experiments.jobs.fleet status JOB_ID`; stop it with
`python -m experiments.jobs.fleet cancel JOB_ID`. Retrieve result files from the
reported remote directory with `scp`. The supervisor never silently replays or
migrates a running job. If an outside GPU process appears, it stops only its own
affected child, preserves artifacts, and marks timing contaminated. Claims
coordinate participating PointStream jobs but cannot stop other users allocating
a GPU later; free memory also cannot guarantee that an oversized job avoids OOM.

Use the lower-level `experiments.jobs.monitor` directly only for CPU-only local
work or existing integration tests. Its remote supervisor writes `command.log`,
`status.json`, and durable state in the job directory. Trainers remain
responsible for their own checkpoints and verified resume behavior.

## Bounded paired codec ladders

`python -m experiments.jobs.codec run POLICY_JSON CAMPAIGN_DIRECTORY` wraps the
existing paired ladder, including PointStream's QP and joint JPEG/QP payload
sweeps. Both arms retain the existing matched encoder preset and pixel format.
The cheap axis is clip length; this avoids assuming a fast preset predicts a
slower preset. Choose representative scenes, not only the easiest static scene.

The execution order is short pilots across all selected scenes, optional bounded
spacing changes and another pilot, longer confirmation across all scenes, then
final-length runs. A short-clip success is not a long-clip compression claim.
Amortization and temporal context can change the ordering at longer durations.

Supply these policy fields explicitly:

| Fields | Meaning |
|---|---|
| `codec`, `tier`, `sweep` | Existing codec/tier names; `qp` or `payload` |
| `scenes` | Unique `[video, scene]` pairs from the existing TierClip cache |
| `dataset_revision` | Immutable dataset/manifest revision |
| `pilot_frames`, `confirmation_frames`, `final_frames` | Integers, `2 <= pilot < confirmation <= final`; must fit cached clips |
| `qps`, `qp_min`, `qp_max`, `qp_step` | At least three increasing QPs, allowed interval and widening step |
| `jpegs`, `jpeg_min`, `jpeg_max`, `jpeg_step` | Payload only: decreasing JPEG quality aligned with increasing QP, bounds in 1–100 |
| `min_gap_db`, `max_adjustments` | Minimum adjacent Y-PSNR spacing and maximum pilot revisions |
| `budget_seconds`, `pair_timeout_seconds` | Cumulative worker wall-time budget; per-pair timeout at most 3500 seconds |
| `bands` | Two-sided `[low, high]` bands for `psnr_dB`, `coded_bytes`, `seconds` |
| `bounds_basis`, `controls_evidence` | Pre-measurement rationale and absolute path to control evidence JSON |

There is deliberately no ready-to-run policy with invented scientific bounds.
The control record must contain `calibrated: true` and `null_checked: true`, plus
the evidence establishing those assertions: identical/mild/severe/unrelated
anchors, absolute metric scale, source identities, and a no-generator or shuffled
control as appropriate. These flags attest reviewed evidence; the controller
cannot verify that prose or a control experiment is truthful.

Every scene must return the full requested point count on both arms. Missing,
nonfinite, out-of-band, nonmonotone, uncoded, or failed results pause the campaign.
Close qualities cause wider endpoints and redistributed intermediate points,
within the approved QP/JPEG bounds. Exhausted bounds or failed longer-clip
confirmation pause for a decision rather than launching the expensive stage.
Similar quality can coexist with useful size/speed differences: this gate asks
for usable quality coverage and does not declare a model inferior.

The controller saves inputs, pair outputs, elapsed cost, decisions and inflight
state. Resuming requires the same policy, control record and code identity. A
completed pair is reused; an interrupted worker requires inspection and is not
silently replayed. An attention verdict is terminal for that campaign: change the
plan in a new directory and preserve the original evidence. No worker is allowed
more than 3500 seconds without a campaign checkpoint; jobs needing longer pairs
need finer-grained runner checkpoint integration before increasing that limit.

Exploratory completion always has `citable: false`. Final paper evidence still
requires frozen settings, held-out evaluation, full quality metrics, size and
encode/decode timing, calibrated controls, and uncertainty across independent
scenes/videos. Do not count correlated frames as independent samples.

## Training candidates: proposed evaluation protocol

This is the protocol for a future training campaign, not a repaired training
launcher. `scripts/train_campaign.py` still calls the retired evaluator and must
be rewired to the runner before real training. Do not enable its unattended mode
as a substitute for the gates below.

For ten videos, first freeze two as the test set and use the other eight for
training/validation development. Group by original source/match/camera where
necessary; different excerpts of the same source are not independent videos.

| Stage | Data and task | Decision it supports |
|---|---|---|
| Reconstruction sanity | Fit a tiny scene and reconstruct the training scene using the intended decoder conditioning | Can the implementation learn, use conditioning, and reproduce motion? Repair failures before ranking models. |
| Same-video validation | Train on other scenes of one video; validate on a disjoint scene with no overlapping frames | Does it work beyond memorized frames within this domain? |
| Cross-video screening | Train on small, diverse subsets of development videos; validate on a held-out development video | Which candidates transfer to new content? Rotate held-out development videos for finalists if affordable. |
| Full development training | Choose configuration and training schedule using development validation; refit on all eight videos if appropriate | Freeze finalists and operating points. |
| Final evaluation | Evaluate once on the two untouched test videos | Evidence for the scoped generalization claim, with the limitation of only two independent test videos. |

Use nested, fixed subsets across candidates. Increase data diversity, training
steps and temporal sequence length deliberately; reducing all of them at once
can hide why rankings change. Compare model families at meaningful minimum
budgets, record examples/steps and wall time, and reserve an extension budget for
ambiguous or slowly improving candidates. Do not automatically halve the field
on tiny score differences. A scene-overfit checkpoint can warm-start later stages
only if that curriculum is recorded and applied consistently; finalists should
confirm the intended full training recipe.

For PointStream, judge the whole codec: actual total delivered bytes at a quality
target (or quality at matched bytes), encoding/decoding runtime, and temporal
fidelity. Include model updates/adapters in transmission accounting when adapted
per content. A better-looking generator may require more residual bits. Compare
against the no-generator/reference reconstruction path, validate conditioning
with a shuffled/null control, and use temporal models on sequences. Preserve
Pareto tradeoffs rather than averaging unrelated metrics into a changing score.

There is no established single best candidate selector for this exact task.
Multi-fidelity allocation is the relevant method family:

- [Hyperband](https://www.jmlr.org/beta/papers/v18/16-558.html) allocates increasing budgets through successive halving.
- [BOHB](https://proceedings.mlr.press/v80/falkner18a.html) combines budget allocation with Bayesian configuration search.
- [FABOLAS](https://proceedings.mlr.press/v54/klein17a.html) explicitly models validation error and cost as dataset size changes.
- [In-context freeze-thaw BO](https://proceedings.mlr.press/v235/rakotoarison24a.html) is a more recent learning-curve-based approach; its benchmark claims do not establish superiority on PointStream.
- [DCVC-UF's training recipe](https://github.com/microsoft/DCVC/blob/main/training.md) progressively increases sequence length and patch size. This supports retaining temporal context in later stages, not using one-scene replication as final codec evaluation.

For ten heterogeneous models, begin with the transparent staged protocol above.
Only adopt a learned search controller after checking that cheap-stage evidence
predicts the deployment objective on this candidate population.
