#!/bin/bash
# Submit W5 specs one after another from this worktree (commit first). Usage: submit.sh SPEC.json [SPEC.json ...]
# Each submission's output goes to $LOGDIR/<spec name>.log; the job ID is the "<stamp>-<hash>" printed there.
set -u
ROOT=/Users/manu/Desktop/PointStream/.claude/worktrees/agent-ae50d9755e58f7ac5
LOGDIR=/private/tmp/claude-501/-Users-manu-Desktop-PointStream--claude-worktrees-zealous-nightingale-52e721/cb149cac-af80-43b3-89f1-5416be3aa053/scratchpad/w5
mkdir -p "$LOGDIR"
cd "$ROOT" || exit 1
for spec in "$@"; do
  name=$(basename "$spec" .json)
  "$ROOT/scripts/ps-fleet" submit --source-worktree "$ROOT" --snapshot-path src --snapshot-path experiments \
    --snapshot-path tools --snapshot-path pyproject.toml "$ROOT/$spec" > "$LOGDIR/$name.log" 2>&1
  echo "$name exit $?" >> "$LOGDIR/submitted.txt"
done
