# BP48 — recovery safety before the reference pilot

Owner: Codex. Date: 3 September 2026. This is an engineering acceptance report,
not a codec comparison. Gate A remains open; no broad E1 run was launched.

## Changes reviewed and repaired

Based on Cursor's `e08641c` (single `--point`) and `fadf9f6` (scene recovery,
progress, per-scene late-frame scores, corrected background-codec provenance).
The follow-up starts at `a155df2` on `codex/recovery-safety`.

- The public runner now rejects checkpoint reuse after a change to configuration,
  full source arrays, object inputs, context IDs, injected implementation identity,
  or Python source code. An exclusive lock prevents concurrent writers.
- Snapshots are flushed, hashed and atomically published. Missing, changed,
  incomplete and legacy snapshots fail closed. Use a new output directory; do
  not edit an identity record to make old data load.
- Saved scenes retain real scoped quality reports, delivered byte counts,
  reconstruction frames, client/encoder symmetry and byte ledgers.
- Canonical canvas preparation is checkpointed separately. Stream recovery
  restores reference history, canvas coordinates and context groups before
  backend preparation. It does not repeat the offline prepass.
- An approved regression exposed an empty-history bug when interruption happened
  after preparation but before scene one. The loader now restores empty arrays
  explicitly; this case and after-scene interruption both pass.
- Progress covers the whole runner, including identity checks, preparation,
  recovery and final scoring. Scene stage timings remain available.
- Timing accumulates all attempts. A hard interruption leaves the total unknown:
  `run_seconds=null`, `run_seconds_lower_bound` labelled explicitly, and
  `timing_complete=false`. It never presents the retry alone as full cost.
- The largest interval without a durable checkpoint is recorded. An exceeded or
  unverified hourly budget makes the E1 point unusable for batch expansion.

## Verified locally

User approved the test cases before test code was added (`test-design` workflow).
The targeted runner/recovery suite passes 34 tests. It includes real small AV1
background streams interrupted between a static scene and a pan, with both
shared and changed contexts. Resumed pixels, quality, accounting and serialized
background continuation match an uninterrupted run exactly. It also checks
pre-scene recovery, heartbeat coverage, incompatible inputs, cumulative timing,
hard-interruption uncertainty and corrupt snapshots.

Full default suite: **1,077 passed, 4 skipped, 84 deselected, 3 expected failures**.
Local log: `/tmp/pointstream-recovery-validation.log`.
Coverage: **80.33%**, above the unchanged 77% CI policy, below the unchanged 81%
local buffer. The default local coverage command therefore exits nonzero despite
all test assertions passing. No tests were added to move that threshold.
The preserved BP47 integration coverage data, reported from its own unchanged
worktree at `68a03dc`, is 79.76%: this branch is +0.57 percentage points. This
comparison includes Cursor's changes and these repairs; it is not a delta from
Cursor's `fadf9f6` alone. Neither run meets the separate local buffer.

Full mypy: no issues in 334 files. Repository-wide ruff and import-direction
checks pass. GitHub PR checks must also pass before merge; inspect the PR's
current head, not an older successful run.

Reproduce from the recovery worktree, with caches on local disk:

```sh
export PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1
export MYPY_CACHE_DIR=/tmp/mypy-recovery RUFF_CACHE_DIR=/tmp/ruff-recovery
conda run -n pointstream --no-capture-output ruff check
conda run -n pointstream --no-capture-output mypy --config-file pyproject.toml
conda run -n pointstream --no-capture-output python -m src.contracts.layers
PYTEST_ADDOPTS='-o cache_dir=/tmp/pytest-recovery-full' COVERAGE_FILE=/tmp/pointstream-recovery.coverage \
  conda run -n pointstream --no-capture-output python scripts/check_coverage_gate.py
```

## Limits and next gate

Scene-level recovery is not codec-internal resume and is not proof of the native
hourly budget. A single preparation/encode/scoring phase can still exceed an
hour; the recorded budget check detects this, rather than promising otherwise.
A native rerun must verify the maximum gap before expanding. If it exceeds the
budget, stop and scope a smaller checkpointable unit with Codex.

Generation-enabled checkpoint recovery is rejected because model/RNG state is
not saved. Injected implementations must supply a stable `checkpoint_identity`;
their opaque internal state is not automatically resumable. The supported E1
path keeps generation off. Restored chunk bags contain the artifacts needed for
result assembly, not every intermediate DAG artifact.

## Next session: bounded execution, then review

1. **Codex:** verify the recovery PR's checks/review and merge when ready. Do
   not delete remote branches or remove another session's worktree.
2. **Cursor:** from the merged code, rerun only native `bg-crf51` on the BP47
   48-frame `alcaraz_highlights` scene_000 + scene_028 inputs in a NEW output
   directory. Verify full source hashes against BP47, record bounds before
   results, run detached, retain the log and checkpoint timing journal. Check
   per-scene quality deltas, complete accounting and maximum checkpoint gap.
   A successful per-point resume is not an interrupted-scene recovery test.
3. **Cursor, after that operational gate:** one slowest-available-preset
   AV1/VVC runtime/recovery pilot on the exact same decoded source hashes.
   Pin executable path/version and effective preset (SVT-AV1 0; determine the
   slowest supported VVenC preset from the actual binary). Respect the matched
   continuous-context and independent-segment access patterns. Conventional
   codecs may also reuse references across scene joins in the continuous arm;
   do not force a fresh keyframe there unless PointStream resets context too.
4. **Codex:** review the pilot's complete size/quality/time record, failures,
   metric controls and checkpoint gaps before authorizing curves or broad E1.

Cursor must return commits, exact commands, full source hashes, tool versions,
output/log paths, submitted/succeeded/failed counts, all timing fields and
checkpoint-gap verdicts, per-scene quality alarms, and an explicit go/no-go.
If a native codec call cannot meet the recovery budget, report it rather than
launching a longer batch. No win is established by this engineering work.
