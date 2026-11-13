# K5d: qualify CUDA context memory accounting before the SVT16 pilot

Status: exact-source CPU qualification is complete; GPU runtime is unqualified.
Scientific a71 is frozen as source bb717 / archive e051 / config 5fc, with
source-only bridge1f371 and sealed cache21ce independently accepted.
Native job `20261113T122511Z-797deaed` passed its four modes, CPU accounting and
cleanup; proof7e6fd454 is published and ACKed. Receiver job
`20261113T124019Z-6af48d82` passed five actual private-file denials, native/wire/
YUV/RGB equality, frozen CPU-only receipt verification and scoped cleanup;
proof6e882f1f is published and ACKed. The [native index](../experiments/k5d-inputs/svt-a71-797deaed-native-CPU-acceptance-20261113.json)
and [receiver index](../experiments/k5d-inputs/svt-a71-6af48d82-CPU-denial-acceptance-20261113.json)
retain exact provenance. Qualification1012 CPU is fully charged.

Two local metadata attempts stopped before output or remote action: first a3CPU
cap, then a launcher hard2 limit that prevented the helper from setting3.
Both10CPU charges remain. Root corrected the launcher (soft2/hard40 with default
SIGXCPU; child3), retained one numerical-library thread, and used a new output.
The third allocation10 passed: .780028 helper/validators +.048366 driverCPU.
Frozen production source admission and both normalized validator returns agree.
All30CPU is charged; reserve is now120. GPU pilot is released for ONE publication after accepted final preflight.
Final false package:
`/private/tmp/pointstream-SVT-a71-first16-stock-FINAL-FALSE-20261113-ROOT-singlethread-correct-limit`;
raw ecae6bea / canonical09483746, deadline13:29:21.937243UTC.

The original shared-SVT job `20261113T114623Z-c6b01236` failed before models,
with zero accepted quality points. Its card is closed to automatic retries.
This is the separate PLAN subitem for a source repair and fresh qualification;
the original dual-codec comparison remains open.

## Goal and decision rule

Establish whether explicit, capped CUDA context initialization permits the
unchanged strict memory check to observe the real campaign PID on the claimed
GPU before model loading. Only after new exact-source CPU qualification and
root review may one frozen, bounded real16 attempt test the repair. It must
retain all three original SVT points (CRF48/55/62, preset4), both arms, fresh
restricted CPU receiving, native/wire/pixel provenance, full-frame and ROI
metrics, both independent judges, and existing exposure/absence/reference rules.
Global six-point defaults remain unchanged. No 48-frame or full-study adapter
or scale release follows from this card.

The hypothesis is lazy CUDA context initialization: no real driver memory row
exists before the first allocation. Competing explanations are incorrect PID
lineage or query visibility. A matching real row after initialization supports
the first hypothesis; another missing/mismatched row stops for diagnosis.
Never synthesize a zero row or weaken PID, foreign-process or memory guards.

## Retained failure and proposed source delta

Diagnostic 79617 passed source admission, then `before_models` rejected the
missing actual GPU memory row on assigned gpu6 Ada1 UUID
`GPU-ef96fc4f-134e-4c01-290e-9a6c8bbef553`. No accepted native/quality result
was produced. The old 848 CPU/600 GPU phase retains its full conservative charge
on failure. Cleanup audit29077 completed: two captured PIDs absent, one group
empty, owned shared-GPU and FQDN CPU tokens absent; the same worker context
was unchanged except CPU accounting. Root closeout582bb644 is accepted; sole failed terminal event7c0c1189bc545d2082bd6846b4331f40 is acknowledged.

Scientific candidate `a71dc6e27743efaa4fb5bbe2e8c8b387ae51ba02` is on independent
branch `codex/svt-cuda-context-scientific-7a9f-20261113`, worktree
`/private/tmp/pointstream-svt-cuda-context-scientific-7a9f-20261113`. Its parent is
exact7a9f; the two-file patch is byte-identical to PR393/head93ebb351
(SHA256 da80bed4543021b45eb063fd9765ae4f85674ea5558f8a8692a917ef0fc42eb9),
excluding inherited control changes. The candidate is immutable. Four setup lines
allocate one CUDA uint8 element **after** the allocator cap, synchronize,
release the probe and then reset peak statistics. Strict memory-row, campaign
PID and foreign-process checks remain unchanged. This is context setup, not
a threshold, model, training or decoder-permission change. PR393 passed full CI
and merged at c4ab531a (12:01:22 UTC); documentation PR394 merged at 12:09:48 UTC.
Runtime qualification remains
unproven and is required before promotion.

## Qualification and stop conditions

1. Complete configured checks and CI, select the exact final source, then
   freeze and independently verify its archive, environment and source bridge.
2. Although all 31 receiver source grants are identical, existing strict
   execution-identity/config/producer/receipt joins require **fresh native and
   actual CPU receiver-denial proofs**. Independent report 69b86104 records this
   requirement. Do not relabel native 6bf, receiver c8 or older proofs for new code.
3. Review the new config identity, actual native tools/pixels, five private-file
   denials, launcher interruption, scoped teardown, complete resource forecast
   and fresh worker/GPU context before any separately released real16 attempt.
4. The one real16 attempt must pass the actual memory/PID guards and all original
   scientific gates. Missing rows, mismatch, OOM, cap violation, partial results,
   missing metrics, exhausted budget or unknown publication stops for root
   review; preserve the actual ID and artifacts, with no automatic replay.

A pass establishes only the prospectively scoped correctness/feasibility result.
No precise GPU-span, speed, extrapolation or scale claim is authorized. Full
lifecycle cost includes setup, staging, validation, publication and teardown.
No paper, PRESLEY, training or paid API work is released.

## Prospective resources

The total envelope stays **600 GPU-seconds / 2000 CPU-seconds**. Exact-source
qualification1012 is accepted. All three local metadata allocations retain their
charges; remaining hold is120 CPU/0 GPU. Read-only preflight16 completed; the final
822 CPU/600 GPU pilot is prospectively charged and released once:

| Stage | CPU-seconds | GPU-seconds |
| --- | ---: | ---: |
| Source preparation, charged/accepted | 100 | 0 |
| Native qualification, charged/accepted | 327 | 0 |
| Actual CPU receiver qualification, charged/accepted | 585 | 0 |
| First local metadata freeze, failed/charged | 10 | 0 |
| Second metadata, failed launcher/charged from reserve | 10 | 0 |
| Corrected metadata, passed/charged from reserve | 10 | 0 |
| Global queue and same-worker preflight, precharged | 16 | 0 |
| Remaining real16 lifecycle, released/precharged once | 822 | 600 |
| Reserve | 120 | 0 |
| Total | 2000 | 600 |

The separate 300 CPU local code/check allocation is already precharged and is not
part of this envelope. Complete qualification costs include local callers,
Git, SSH, remote Python, doctor/probe children, archive staging/publication,
monitoring, independent audits and proof/ACK closeout. No caller is hidden in
the scientific cap. These remain prospective envelopes, not a measured fit;
each concrete stage needs exact pins, a complete forecast and root release.
No extra user allocation or GPU release is implied. The primary exclusive
2c6d route remains separate:1160 CPU/900 GPU held, metadata 50 CPU precharged but
UNRUN, false plan 463f50d0. See the [continuation](../continuation-2026-11-13.md)
and [ledger](../experiments/submission-research-budget-20261104.json).

## Conditional roadmap after a real16 pass

Only after that actual gate passes, consider one opt-in SVT study port covering
48 and registered full clips together, while preserving global six-point
defaults. Qualify one final source and its16/48 calibration; use a real48
three-point/full-metric cost oracle. Then confirm the two fixed full-context
clips, game9/Clip5 (292 frames) and game10/Clip12 (586 frames), without selecting
clips after quality results. The registered 30-clip/5675-frame study remains
conditional on complete stage costs and remaining programme/Ball budgets.
Parts stay within 45 minutes; negative outcomes remain reported. No adapter,
48-frame, full-clip or 30-clip execution is released now.
