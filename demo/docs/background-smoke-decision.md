# Background smoke recovery and decision — 4 October 2026

## Decision

Latest status after the approved 40-minute GPU / 40-minute CPU amendment: see
[the final bounded attempt](background-smoke-final-attempt-20261004.md).
The gpu5 inventory started and preserved four passing parts plus a matching
DCVC source/extension check, but exceeded its 90-second smoke allowance during
the environment part. No substantive gate passed; B2/B3/B4 remain unrun.
The background plan is experimentally incomplete. No further request was
submitted after this failure. The earlier 25-minute execution is retained in
[the historical report](background-smoke-approved-execution-20261004.md).

For the later artifact recovery, visual review, passing two-host fleet check
and explicit budget decision, see [the continuation report](background-smoke-continuation-20261004.md). The observations below remain the record of the first recovery window.

B1 has no passing provenance gate, so B2/B3/B4 remain **blocked**. Do not
rank DCVC-UF or HNeRV, retrain either model, or start a longer experiment
from these results. No new GPU job was submitted during this recovery.
The implementation is locally tested; its model paths are not GPU-verified.

The original assignment is `background-smoke-plan.md`. This report supplements
rather than rewrites `background-smoke-execution-20261003.md`, whose earlier
zero-allocation/DNS observations describe a different execution window.

## Recovered work and execution

Recovered Cursor session `e8acab4a-01cd-43c0-a1e0-eb484cd9e29e` from its local
agent transcript. The Codex chat lookup returned no match. Continued its
existing worktree `/private/tmp/pointstream-bg-smokes-fleet` on
`codex/demo-background-smokes`, initially at pushed revision `277adcb`
(after implementation commit `8aca488`). Preserved the dirty Desktop checkout,
its fleet edit, background seed copies, unrelated experiments, and all other
worktrees. No foreground/model-loss/foreground-packet changes were added.

| Job | Recovered state | Evidence and consequence |
|---|---|---|
| `20261003T101656Z-61d46aea` | Complete, gpu5, exit 0 | Saved management status and source snapshot identities verified. Preview manifest, panel contents, native decodes and bitstreams were not retrieved in this recovery. |
| `20261004T082204Z-4515f8fa` | Failed, gpu5, exit 1 | No `gate.json`; campaign error: smoke validator did not publish a passing substantive result. Cursor described an adapter-import timeout. That description was not independently checked against its part files. |
| `20261004T083413Z-e373aa74` | Failed, gpu5, exit 1 | Revised inventory really was published before Cursor stopped. Smoke command timed out after 170 seconds; no gate passed and no diagnostic remainder ran. |

Both inventory attempts claimed gpu5 GPU
`GPU-7754164a-33cd-265c-3696-c20966d228fe` (RTX 6000 Ada). They are preparation
failures, not codec-quality evidence. Status does not expose the part files;
this recovery cannot assert which checkpoint hashes, if any, were completed.
No codec, drift, latent-extraction, training or fine-tuning job was submitted
by this recovery. No existing request was replayed, migrated or cancelled.

The recovered specifications reserved **340 + 340 = 680 seconds** against
the common 1800-second ceiling. Conservatively retain all 680 seconds as
spent reservation; at most 1120 seconds remain without enlarging that ceiling.
This is not a measured GPU-active time. Start-to-terminal metadata update
intervals are approximately 256 and 309 seconds, but include management
latency and do not prove exact claim release or CPU-preparation duration.
Do not replace them with a zero-allocation claim. The exact allocation/CPU
ledger requires preserved supervisor and stage artifacts. Repeating the
340-second inventory reservation would exceed the plan's 600-second CPU
preparation ceiling even without any further diagnostic work; no further
preparation is justified without reconciling actual elapsed receipts.

Fresh standalone `ps-fleet doctor`, `workers status`, and six-host `inspect`
were run. Doctor failed: gpu2/gpu5/gpu6 timed out and gpu4 was measurement
locked. A later inspection reached gpu2/gpu5/gpu6; it did not turn the failed
cross-host doctor into a passing one. gpu3/gpu5 workers had fresh heartbeats.
Healthy workers were not restarted, and no selftest was run.

A native five-minute recovery heartbeat was registered, covering the old
submitting chat and this chat. After both recovered requests became terminal,
the revised job's failure/decision events were reported and acknowledged and
the heartbeat was paused. No periodic submission or cancellation was enabled.

## Implementation completed in this recovery

- Validation now requires actual nonempty streams, matching SHA-256/byte
  counts, strict model-load and source-check receipts, decoder independence,
  full per-frame metrics including LPIPS, unchanged input manifests and
  matching ordered source identities. Drift validates its 32-frame stream
  and all four reset streams. Latent validation reads real packet files and
  the decoder setup package. Status-only records cannot pass.
- Drift requires a hash-pinned B2 LD pretrained/fine-tuned result before
  inference, matched to the current eight source identities, checkpoint
  hashes, QP and reset policy. Missing environment or cached LPIPS evidence
  blocks model work. HNeRV checks its installed source hashes again and
  missing installed dependencies block rather than being replaced by stubs.
- Inventory saves each completed part and a partial selected-input manifest,
  publishes completed work as it happens, and preserves exact source diffs.
  Hash/copy children now use only the Python standard library instead of
  importing NumPy for every file. The verified scratch-copy check shares
  the transfer's remaining time allowance. These changes reduce avoidable
  startup overhead but have not been timed on the lab filesystem.
- The preview reader now uses the saved helper's `segment_length` and
  `methods` schema. Missing panels are inconclusive. HNeRV display rates
  are labeled **setup-inclusive estimates**, not ordinary latent streaming
  rates. A metadata audit does not claim visual inspection or historical
  checkpoint identity.
- Spec checks enforce finite, nonnegative prior spend, per-job and cumulative
  stage caps, and the dispatcher schema. The broken default 4-bit full
  stage was replaced by the optional factory002 6-bit correctness diagnostic:
  it runs only after the primary 6-bit gate passes and fits within 420 seconds
  total. A 4-bit arm remains optional within a stage after its same-cut
  6-bit result; it is not selected without rate/quality evidence. Default
  drift spends its remainder on LD fine-tuned; HT-S is left untested unless
  a separately reviewed budget leaves its required two minutes.

All five background-only inventory files (including the tracked neural
background test) still match their recorded SHA-256 identities. Background
seed code was already retained on the branch; unrelated dirty infrastructure
was not selected or dispatched.

## Verification and limits

Focused CPU tests cover metrics, packet round trips, actual-byte rates,
checkpoint identities, source changes, strict-load checks, malformed or
missing artifacts, cumulative caps, missing dependencies, prerequisite gates,
and a real two-stage dispatcher fixture where failed validation blocks the
remainder. Synthetic adapter and LPIPS fixtures do not establish model or
LPIPS correctness on the servers. The torch-dependent packet test is skipped
because this Mac interpreter has no PyTorch. Ruff is unavailable in this
interpreter; it was not installed.

The focused six-file suite passed **96 tests**, with **one torch-dependent
skip**. After the final cumulative-stage-budget change, the spec suite was
rerun and passed **14 tests**. The required four fleet suites passed all
**104 tests**. Python compilation and `git diff --check` also passed.

```bash
python -m pytest -q -ra -o addopts='' tests/demo/test_factory_bg_rd.py tests/demo/test_neural_bg.py tests/demo/test_background_provenance.py tests/demo/test_background_codec_smoke.py tests/demo/test_hnerv_latent_packet.py tests/demo/test_background_smoke_specs.py
python -m pytest -q -ra -o addopts='' tests/demo/test_background_smoke_specs.py
python -m pytest -q tests/experiments/test_resource_claims.py tests/experiments/test_gpu_fleet.py tests/experiments/test_job_monitor.py tests/experiments/test_fleet_inbox.py
```

No GPU inference correctness,
LPIPS installation, checkpoint architecture/configuration, historical checkpoint
identity, training/hold-out provenance, prediction drift or measured temporal
latent saving is established by these CPU tests.

Read-only management/spec receipts, with their checksum index, are outside
the code tree at `/private/tmp/pointstream-background-recovery-20261004/`.
Original evidence remains under the external data root; each job ID above
identifies its preserved run. The local recovery receipts are management
observations, not exported model results.

## Remaining gates and recommended next action

| Stage | Decision |
|---|---|
| B0 preview reuse | Existing job completion and snapshot provenance verified; manifest/panels/native artifact review still incomplete. No historical checkpoint hashes may be invented from current files. |
| B1 provenance | Blocked by failed inventory gates and unreconciled preparation cost. Recover completed part files/ledger first, hash only missing identities, and obtain a fresh passing fleet doctor before considering another bounded request. |
| B2 codec pairs | Blocked until actual checkpoints and selected inputs are verified. Then start only with primary HT-L pretrained/fine-tuned at QP 21; preserve the remaining common and per-stage budgets. |
| B3 drift | Blocked until a valid B2 LD pair exists. No 64/128-frame follow-up is authorized here. |
| B4 latent packaging | Array packet format tested; trained decoder/source independence and actual RD untested. No learned entropy coder, optimizer or retraining is justified. |
| B5 selection | Inconclusive. Source/checkpoint/inference drift, training degradation and representation limits remain distinguishable hypotheses, not resolved causes. |

The next useful action is read-only recovery of the existing stage artifacts
and exact cost ledger, followed by the missing B1 identities only if the
original preparation and allocation budgets can still accommodate them.
Do not resubmit the previous specifications or silently reset their budgets.
