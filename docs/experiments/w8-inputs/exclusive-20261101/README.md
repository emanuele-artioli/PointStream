# Approved exclusive W8 scoring: preparation

The user approved a 450 GPU-s / 450 workload CPU-s inclusive scoring cap inside
the unchanged W8 3,600/3,600 totals. Previous attempts charged 67 GPU-s and
67 CPU-s; prior preparation charged 20 CPU-s. The proposed reservation is
67 prior + 280 campaign + 30 monitoring/cleanup + 10 independent audit + 30
all preparation/final publication = **417 CPU-s**, including those prior 20;
the GPU maximum is **377 s**. Reservations are not expenditures. Do not replay
an uncertain request, or launch another without a remaining-budget decision.

`score-spec-candidate.json` is normalized but deliberately expired. It retains
the exact 80-tick scoring inputs and selection SHA256
`4f5af8c90f5fd909abac2aed123917795f388b3961e31e18cb2e0f306aa7acb0`,
uses gpu2/5/6 with standard exclusive admission and contention stop, one CPU,
deferred failed publication, and 40/195/10-second smoke/full/validator with a
280-second campaign cap. After the guard policy implementation is reviewed,
source and worker releases frozen, fresh queue/host preflight succeeds, and a
future deadline is set, normalize again and separately record the resulting
canonical spec digest and source identity. The candidate's expired deadline is
not a submission deadline. Actual smoke, control/identity validation, the
unchanged W1 projection and remaining-time projection must pass before full.
Exclusive admission does not guarantee speed or prevent later contention.

## Independent full audit, after a successful full stage

Root records the new job ID, exported full job directory, `full/published.tar`
SHA256 and canonical normalized spec digest independently before running the
following. Input/output directories are immutable source and fresh destination,
respectively. Replace only the four explicitly marked placeholder values; do
not guess or reuse the earlier failed job. Run in the checkout root on a
machine with the pinned analysis dependencies. This fresh process has a hard
10 CPU-second limit, includes Python imports, and uses one BLAS/OpenMP thread.
An external wall deadline must also be imposed by the invoking supervisor; if
it expires or the audit fails, stop and charge the work, preserving partial
output. Do not count this as a pass merely because a directory was created.

```bash
(
  ulimit -t 10
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  python -m experiments.tennis.w9_full_audit \
    --job '<EXPORTED_NEW_JOB_DIRECTORY>' \
    --manifest docs/experiments/w8-inputs/amendment-20261101/selection.json \
    --hybrid /private/tmp/w9-inbox/20261031T180054Z-b9927a25/full/published.tar \
    --windows /private/tmp/w9-real-parser-inputs/540156339f8577c0b629e0e65403e2547a90fc4383afeff3597b6f96807ec0ed \
    --out-dir '<NEW_NONEXISTENT_AUDIT_DIRECTORY>' \
    --archive-sha256 '<NEW_FULL_ARCHIVE_SHA256>' \
    --spec-sha256 '<NEW_NORMALIZED_SPEC_DIGEST>'
)
```

The selection hash above, hybrid archive SHA256
`bf05e2630d898a2ec68452f81407e063b0f0ad0490365c41d53a3988f5adb299`,
and windows SHA256
`540156339f8577c0b629e0e65403e2547a90fc4383afeff3597b6f96807ec0ed`
are checked by the audit against the normalized spec. It also checks successful
full execution, archive/spec identity, exact selection and 80 ticks, recomputes
all four per-tick control metrics for each of three controls against original
raw hybrid data, and reruns the full W1 validator. Preserve and independently
hash its `audit.json`; a successful exit and `passed: true` are required.
The audit allocates 10 CPU-s, not 10 seconds of additional GPU time. A comparable
previous W9 audit used 0.122 CPU-s on this Mac; this does not guarantee a new
run's timing or correctness.

## Merge and decision

Within the remaining preparation allocation, call
`experiments.tennis.w8_amendment_merge.screen` with the verified local W9 inbox
`/private/tmp/w9-inbox`, the independently audited prior repair below, a trial
record naming the new full archive and freshly hashed audit, and the exact new
normalized spec digest. The prior repair record is:

```json
{
  "archive": "/private/tmp/w9-gpu-pilot-artifacts/full/published.tar",
  "sha256": "28aeb71d62c64d642f1c5157af6bca691cbb4f6f0c3ebdfdea94887a88177ba4",
  "audit": "docs/experiments/w9-inputs/repair-audits/nad000.json",
  "audit_sha256": "61d38bdef643cb7f5326fd58fefc7a402e8c6c749642a0cfa05a3b67ae45b11d"
}
```

The new trial record has the same four keys with its own freshly recorded paths
and hashes. `screen` verifies the previous W9 repair, original anchors and new
independent audit, keeps the prior `s4_o0` repair, and adds only the selected
stream. It evaluates all four families and both foreground metrics at the fixed
actual W8 rate **46.315062222222224 kbps**, using raw points without cached
old-target results or extrapolation. Write its complete output to a new file,
record its identity and provenance, and review its eight cells. Only a strict
pass permits the original W8 gates to resume subject to their original budget;
fail, unresolved, missing evidence, or any audit/control mismatch stops them.
