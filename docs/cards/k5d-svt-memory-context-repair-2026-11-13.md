# K5d: qualify CUDA context memory accounting before the SVT16 pilot

Status (November 13, 11:58 UTC): build-only candidate, not runtime qualified.
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
a threshold, model, training or decoder-permission change. CI and runtime
qualification are prerequisites, not inferred from this candidate commit.

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

The ledger reserves **600 GPU-seconds / 2000 CPU-seconds**, entirely UNRELEASED:

| Stage | CPU-seconds | GPU-seconds |
| --- | ---: | ---: |
| Clean source and fresh CPU qualification | 818 | 0 |
| One real16 lifecycle, prospective estimate | 848 | 600 |
| Reserve | 334 | 0 |
| Total | 2000 | 600 |

The separate 300 CPU local code/check allocation is already precharged and is not
part of this new hold. These are prospective envelopes, not a measured fit;
each concrete stage needs exact pins, a complete forecast and root release.
No extra user allocation or GPU release is implied. The primary exclusive
2c6d route remains separate:1160 CPU/900 GPU held, metadata 50 CPU precharged but
UNRUN, false plan 463f50d0. See the [continuation](../continuation-2026-11-13.md)
and [ledger](../experiments/submission-research-budget-20261104.json).
