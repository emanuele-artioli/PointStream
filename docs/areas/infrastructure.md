# Infrastructure Area

**Evidence Revision**: Reconciled through PR #68 (`fa00a12`) and PR #73.
**Owned Scope**: Environments, CI/GitHub Actions, worktree lifecycle, runner integration, local caches, hardware profiling.

---

## 1. Current State

### Environment & Startup Performance
PointStream runs on a shared remote Linux GPU server with an NFS-backed home directory. On **gpu6** (commit `bc09184`, September 2026), process startup and import latency were measured under clean conditions (`PYTHONNOUSERSITE=1`, explicit `PYTHONPATH`):

| Operation | n | Mean ± SE (s) | Range (s) | Notes |
|---|---:|---:|---:|---|
| `git status --short` | 5 | 0.057 ± 0.034 | 0.020–0.195 | Fast metadata walk |
| `git ls-files` | 5 | 0.0051 ± 0.0001 | 0.0046–0.0053 | Memory cached |
| `rg --files` (source dirs) | 5 | 0.0147 ± 0.0006 | 0.0134–0.0168 | Fast local traversal |
| System Python no-op | 5 | 0.0227 ± 0.0006 | 0.0215–0.0243 | Base interpreter overhead |
| PointStream Python no-op | 5 | 0.0455 ± 0.0113 | 0.0330–0.0905 | Conda env startup |
| `import sqlite3` | 3 | 0.0440 ± 0.0074 | 0.0353–0.0587 | Enforces ABI compliance |
| `import sqlite3` then `torch` | 3 | 2.101 ± 0.620 | 1.350–3.331 | PyTorch import tax |
| `import sqlite3` then `src.runner` | 3 | 0.529 ± 0.117 | 0.406–0.764 | Runner does **not** load Torch |

### Worktree Cleanup Audit
PR #68 introduced `scripts/cleanup_merged_worktrees.sh`. The documentation audit in PR #73 flagged critical safety hazards in this helper:
- Suppresses error codes from `git fetch`, `status`, and `log`.
- Falls back from `git worktree remove` to `rm -rf` when git refuses to delete a dirty worktree.
- Performs irreversible remote-ref pruning (`git remote prune origin`).

**Operational Directive**: Do **not** execute `scripts/cleanup_merged_worktrees.sh` in automated workflows until `INFRA-ACT-01` is completed.

---

## 2. Key Decisions & Evidence Anchor

| Topic | PR / Commit | Decision & Status |
|---|---|---|
| External data separation | #32 (`d436b02`), #35 (`6c4fa10`) | `paths.py` resolver routes data to `.ps-data-root`; zero symlinks in repo. |
| Worktree recovery | #53 (`86653ef`) | Strict branch-and-worktree isolation protocol established. |
| CI automation | #23, #44 | Integrated `ruff`, `mypy`, layer boundaries, and synthetic tier tests. |
| Host profiling | #73 | Measured process startup and import times; confirmed runner does not load Torch. |

---

## 3. Next Actions

| ID | Status | Dependencies | Source | Description & Acceptance Criteria |
|---|---|---|---|---|
| `INFRA-ACT-01` | Ready | None | #68, #73 | **Repair worktree cleanup helper**: Refactor `scripts/cleanup_merged_worktrees.sh` to halt on any git refusal, verify clean working tree against `origin/main`, remove the `rm -rf` fallback, and drop remote pruning. Acceptance: Script refuses to delete unmerged or dirty worktrees and passes unit test. |
| `INFRA-ACT-02` | Ready | None | Host rules | **Host-local cache enforcement**: Configure local caching (`MYPY_CACHE_DIR=/tmp/mypy-$USER`, etc.) in CI and runner scripts. Acceptance: Zero mypy cache files written to NFS home. |
| `INFRA-ACT-03` | Closed / Archived (D1/D6) | None | `plans/DEFERRED.md` | **Static typing and test pollution**: Mypy passes cleanly across all 350 source files; tests isolated from global environment. |
