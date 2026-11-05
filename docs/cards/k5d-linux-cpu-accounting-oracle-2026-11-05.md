# K5d synthetic Linux CPU-accounting oracle — 2026-11-05

Goal: qualify only whether the current subreaper wrapper waits for a direct child, a forked child, and a double-forked orphan and accounts their CPU on Linux. This is not evidence for native encoder, CUDA, timeout teardown, or production admission.

Hypothesis: all three task-owned synthetic processes finish, are reaped, and their independent per-process resource totals are covered by the wrapper's waited-child delta. Competing explanation: orphan adoption, nested wait accounting, or log attribution can omit CPU or leave a process alive. The independent parent RUSAGE_CHILDREN delta must equal the wrapper's child delta; three distinct actual PIDs must disappear, and the orphan must have been adopted. Require pinned source files, no GPU and no Torch. A failed check stops the oracle; retain failure and artifacts. No timeout/interruption or escaped-session experiment is included because cleanup is not independently qualified.

Budget: CPU-only fleet campaign at most 100 workload CPU-seconds plus 80 CPU-seconds external review; no GPU allocation or GPU computation. Smoke and full each run the same small bounded synthetic workload, and the smoke has a substantive validator before full. Host restricted to gpu2 with one CPU, subject to current all-host and job inspection and independent root review of source and spec before submission. Fleet's ordinary stage, deadline and CPU controls remain in force. No production release, source bridge approval, or changes to numerical components are implied.

The draft spec is `/private/tmp/pointstream-k5d-cpu-oracle-spec-20261105.json`. Record the actual job ID and source identity here before waiting if submitted. Status: source/spec under review; no job submitted.
