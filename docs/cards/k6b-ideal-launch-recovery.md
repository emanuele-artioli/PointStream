# Ideal-player launch recovery — proposed, no new GPU authority

The user-approved ideal-player diagnostic failed before its campaign imported:
job20261028T071835Z-2964e9ef omitted the fleet launcher from its selected source
snapshot. No correctness/cost smoke, metric inference or full run occurred.
The [launch report](../k6b-ideal-player-launch-report-2026-10-28.md) preserves
its error, identities and conservative60s charge. The original3600s cap has
1190s left. The failed request will not be replayed or edited.

Concrete repair: selected workload snapshots now automatically include the
complete native campaign/salvage runtime. CPU checks must verify campaign and
supervisor under the native host interpreter, plus the oracle under the frozen
packed runtime, on both compatible hosts before any new submission.

Proposed authority is exactly ONE fresh request, up to1100GPU-s inside the
1190s remainder: smoke200/full550/validator100/all overhead250,90unallocated.
No new footage, renderer, training, generation, conversion or weight changes.
Use the already frozen input468250a7 and runtime32e68872. Retain ALL scientific
scope, four policies, per-frame byte/metric controls, sensitivity thresholds,
8000MiB allocation+8192margin and separated cold/warm/warm cost gate from the
[original oracle card](k6b-ideal-player-bound.md). Original source RGB is labeled
oracle-only; a positive outcome cannot be called a codec win. No retries after
any failure or gate rejection. Stop at the local decision or resource limit.

Create a fresh absolute deadline only after acceptance. Inspect the failed ID
canonically and all eligible hosts first. Use one native bounded observation,
record ID/deadline/budget/paths before waiting, and preserve every artifact.
No recurring model timer. No request exists for this proposal.
