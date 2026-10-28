# Experiments: persistent prediction

### 2026-10-28 — Accepted direction and bounded feasibility search

- Human requested plan persistence, then authorized the recommended search and
  tests. New ceilings3600GPU-s/3600CPU-s; all earlier studies remain closed.
- Hypothesis: persistent decoded memory plus explicitly coded corrections can
  exploit foreground/background repetition without expensive exact mattes or
  mandatory generative replacement. Competing explanation: ordinary native
  SVT already exploits this redundancy better; selector/geometry and correction
  streams erase any advantage. A second explanation is loss in low-resolution
  decoded history, so this finite pilot cannot bound better feature predictors.
- Scope, three-axis search, literature, configurations and stop rules are in
  [the accepted card](../cards/persistent-prediction.md). No GPU job submitted
  yet; charged0/3600GPU-s. CPU audit/preflight ongoing.
- Prior relevant findings: model discounts already tested on egocentric split
  did not recover LPIPS; tennis rendered-pixel wins granted dynamic mattes and
  background. Recent paid background LPIPS+230.28%; fixed rectangles oversized.
  The new pilot must not repeat these as hidden decoder assumptions.
- Native execution uses fleet, packed environment and host-local scratch;
  preserve prior publications and failed evidence read-only. No recurring timer.

### 2026-10-28 — CPU audit and first frozen request

- CPU accounting COMPLETE: [saved audit](../persistent-accounting-2026-10-28.json).
  Clean-actual fixed-pixel joint side equality ceiling1.619035kbps; even granting
  ALL static overhead leaves real dynamic KPS3.53kbps. Static discounts alone
  fail this finite recipe. This does not price a different representation.
- PnP-VCVE source compatibility audit: released H.264 metadata/checkpoint path
  does not establish an AV1/DCVC restoration test. No inference/training or
  model download; compatibility stop, not a measured quality failure.
- All six hosts inspected; compatible A6000 poolgpu2/gpu3 has capacity, no
  existing PointStream GPU claims; doctor passed. All-history manager status
  query is slow/NFS-limited; retained its original handle, not replayed. New
  predictor entrypoint cannot duplicate any of the terminal historical jobs.
- Exactly ONE new request20261028T181634Z-05c49090 submitted at
  ffa4b4642f9ba7b919e65d65e941c458f60c3960, initially pending. SnapshotSHA
  a20c7ca001e5b410b57425af839dfdb6f91d7871a674b5ee06fa7e748026e48b;
  normalized specSHA3b0ac006c628e9845b9e4bd2d1613d3657b39a4de47ebe3e8024e831d96d904d.
  Receipt/private/tmp/persistent-memory-submission.json. Resource cap3000GPU-s
  within new3600ceiling; absolute deadline2026-10-28T19:30:00Z. No retry.
- Source selection: experiments,src,tools/datasets/k4_inputs.py,k4b_inputs.py,
  plus native fleet launcher dependencies; no selected patches/untracked files.
- Four focused wire/geometry/residual tests and three document checks pass;
  configured ruff and mypy pass. Full local suite running separately.
- Pilot compares a restricted960x544 SVT grid, NOT the full multi-resolution
  conventional/neural envelope. Even an apparent win would need fair expanded
  anchors. Smoke quality is not accepted evidence.
- Initial canonical status: running on gpu2/A6000 UUID
  GPU-fc7e8fff-75be-4984-a185-fe8ce63edcd9, supervisor2920397.
  Native bounded wait returned expected shared-GPU contention event
  5a6af197d83e5461977921462ced3cfa; acknowledged once. Timings remain
  contaminated and are used only for resource limits. One resumed native wait,
  handle44096, max2700s, writes/private/tmp/persistent-memory-wait-terminal.json.
  This is attended execution: automatic model completion wakeup is unverified;
  no scheduled model callback, heartbeat or paid bridge is installed.
- Full local suite has two unchanged platform failures: sandbox-denied ps in
  process suspension test (passes outside sandbox), and protected /home dataset
  symlink resolution absent on macOS. Focused tests pass; Linux CI remains gate.
