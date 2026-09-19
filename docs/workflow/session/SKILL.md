---
name: pointstream-session
description: Route a PointStream task, prepare a scoped dispatch, or report/close a completed work session.
---

# PointStream Session Workflow

This skill holds what is specific to PointStream: area routing, evidence reuse,
and the reporting a codec result must carry. The host `session` skill holds
everything that is not project-specific.

---

## 1. Modes of Operation

### Mode A: Route & Dispatch

Dispatch and report contracts: host `session` skill.

When initiating or handing off a task:
1. **Extract Outcome**: Identify the requested goal, explicit constraints, and potential ambiguity.
2. **Consult Area Context**: Read [PLAN.md](../../../PLAN.md) and the single relevant area document in `docs/areas/`.
3. **Formulate Dispatch**:
   - **Objective & Area**: Name the functional area and stable action ID (e.g. `CODEC-ACT-01`).
   - **Revisions**: Pinned code commit and evidence baseline.
   - **Scope**: Allowed files, worktree path, and branch name.
   - **Inputs & Outputs**: Explicit configs, manifests, or interfaces.
   - **Acceptance Criteria**: Concrete commands and verification gates (see [setup verification](../../setup.md#4-verification)).
   - **Bounds & Controls**: Pre-run bounds for experiment runs; null controls.

For new research work, use [hypothesis-driven experiment design](../experiment-design.md).
Before scheduling any new run, inspect existing result indexes, folder classifications,
manifests, saved decodes and prior validity decisions. Record the exact question,
matching artifact IDs, what can be reused/rescored, and the remaining evidence gap
in the decision card. Check actual configs rather than filenames. Missing timing
alone does not require repeating a rate/quality experiment: link statistically
adequate representative timing for compatible workloads/hardware, or profile
only the missing stratum. Never upgrade old evidence merely because code was
repaired. Launch only the smallest probe that fills the documented gap. The
[older submission dispatch](submission-search.md) is historical repair context;
do not automatically rerun its waves or budget.

### Long jobs

Use [the long-job workflow](../long-jobs.md). Use an already supplied reporting
cadence; otherwise ask once. The September evaluation campaign uses actionable
events, without periodic chat digests. Delegate health checks and ten-minute
logging to the script;
wake the agent only for requested digests or actionable events. Use bounded
pilot/confirmation gates before expensive codec stages. Do not create a periodic
agent polling loop. Training subset stages remain a protocol until its evaluator
is restored.

### Mode B: Execute & Report
When executing work and reporting results:
- **For Documentation & Refactoring Tasks**:
  - Changed files and rationale.
  - Verification appropriate to the changed files; distinguish completed checks from unavailable checks.
  - Area document updates and next actions.
- **For Research & Experiment Tasks**:
  - Exact command, configuration manifest, and input sequence.
  - Encoder binaries, versions, and presets.
  - **Three-axis metrics**: Size (bytes/bitrate), quality (PSNR, SSIM, VMAF, LPIPS), and execution runtime (encode/decode FPS).
  - Pre-run bounds check: Report each alarm and its resolution or unresolved status; report uncertainty at the independent source level.
  - Explicit citable conclusion (label Gate A results exploratory; scope confirmed claims to the Gate B protocol).

### Mode C: Close & Continue
When completing a session:
1. **Update Area State**: Update the owning area document in `docs/areas/` with current status and mark action completed/advanced.
2. **Clean Boundary**:
   - If work is finished: fetch fresh `origin/main`, verify merge ancestry plus unique commits/diff and clean tracked/untracked status before considering retirement. Ask before removing a worktree that may host a paused session. Never force removal or bypass Git refusal with `rm -rf`. Do not run `scripts/cleanup_merged_worktrees.sh` (`INFRA-ACT-01`). Host `git-clean-merged-worktrees` is allowed only on merged, clean trees.
   - If work remains: provide a concise continuation prompt in the chat response. State the unresolved decision and resume command. Do not create a standalone completed-session report or permanent root handoff document; keep decisions in the area and details in the PR.
