---
name: pointstream-session
description: Route a PointStream task, prepare a scoped dispatch, or report/close a completed work session.
---

# PointStream Session Workflow

This skill standardizes task routing, execution reporting, and clean session boundaries across all AI coding assistants (Codex, Claude, Cursor, and VS Code + Antigravity).

---

## 1. Harness & Model Routing

Routing preferences reflect **working strengths and token constraints**, not rigid capability limits:

| Harness | Preferred Tasks | Constraints / Guidance |
|---|---|---|
| **Codex** | Deep analysis, cross-system architectural design, complex experiment adjudication, paper synthesis | Strong reasoning; token budget is scarce: dispatch bounded prompts with precise evidence anchors. |
| **Claude** | Deep technical refactoring and multi-component integration | High reasoning depth; token budget is scarce: avoid open-ended chat or paper prose generation. |
| **Cursor** | Feature development, component coding, parallel worktrees | Fast generation; needs clear interfaces and contracts: best for implementing bounded module functions. |
| **VS Code + Antigravity** | Bounded execution, bulk refactors, documentation overhauls, test execution, plot/artifact generation | Fast iteration and large context; escalate unresolved scientific debates to a deeper analytical session. |

### Model Selection
- **Mechanical edits & typo fixes**: Fast model, low reasoning effort.
- **Bounded component implementation**: Standard model, medium reasoning effort.
- **Complex contracts & scientific decisions**: Flagship reasoning model, high reasoning effort.

---

## 2. Modes of Operation

### Mode A: Route & Dispatch
When initiating or handing off a task:
1. **Extract Outcome**: Identify the requested goal, explicit constraints, and potential ambiguity.
2. **Consult Area Context**: Read [PLAN.md](../../../PLAN.md) and the single relevant area document in `docs/areas/`.
3. **Formulate Dispatch**:
   - **Objective & Area**: Name the functional area and stable action ID (e.g. `CODEC-ACT-01`).
   - **Revisions**: Pinned code commit and evidence baseline.
   - **Scope**: Allowed files, worktree path, and branch name.
   - **Inputs & Outputs**: Explicit configs, manifests, or interfaces.
   - **Acceptance Criteria**: Concrete commands and verification gates (`ruff`, `mypy`, tests).
   - **Bounds & Controls**: Pre-run bounds for experiment runs; null controls.

### Mode B: Execute & Report
When executing work and reporting results:
- **For Documentation & Refactoring Tasks**:
  - Changed files and rationale.
  - Verification results (link validation, lint, contract layer checks).
  - Area document updates and next actions.
- **For Research & Experiment Tasks**:
  - Exact command, configuration manifest, and input sequence.
  - Encoder binaries, versions, and presets.
  - **Three-axis metrics**: Size (bytes/bitrate), quality (PSNR, SSIM, VMAF, LPIPS), and execution runtime (encode/decode FPS).
  - Pre-run bounds check: Confirm no alarm fired; report standard errors.
  - Explicit citable conclusion (never claim victory without Gate B confirmation).

### Mode C: Close & Continue
When completing a session:
1. **Record Result**: Commit changes onto the task branch, push, and open/update a PR.
2. **Update Area State**: Update the owning area document in `docs/areas/` with current status and mark action completed/advanced.
3. **Check CI**: Watch GitHub Actions run via `gh run watch <id>` and inspect `gh run view <id> --log-failed` if failures occur.
4. **Clean Boundary**:
   - If work is finished: check whether worktree can be cleanly retired per AGENTS cleanup rules.
   - If work remains: provide a concise continuation prompt in the chat response. Do not create a permanent root handoff document.

---

## 3. Validation Walkthroughs

### Scenario 1: Routine Typo / Docstring Fix
- **Action**: Fix a typo in `src/components/motion/trajectory.py`.
- **Handling**: Execute directly. No dispatch prompt ceremony, no architecture debate. Run `ruff check` and commit to branch.

### Scenario 2: Disputed Codec Win
- **Action**: An experiment run reports PointStream beating AV1 by 0.5 dB at 200 kbps, but encode time is 100× slower.
- **Handling**: Route to deep analysis (Codex/Claude). Flag the three-axis rule: speed cannot be omitted. Assert two-sided pre-run bounds, verify whether background plate was properly amortized, and document as an exploratory finding pending Gate B confirmation.

### Scenario 3: Completed Docs Overhaul
- **Action**: Retiring legacy plans after area documents are created.
- **Handling**: Update `PLAN.md` area table, verify all 116 files recorded in `docs/history/retired-documents.tsv`, verify links, run CI checks, and provide clean summary.
