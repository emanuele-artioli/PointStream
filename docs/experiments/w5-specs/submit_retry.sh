#!/bin/bash
# Submit each W5 spec until it is published. A refused submission ("no fresh heartbeat") or a snapshot-extraction
# timeout leaves an unpublished request (no spec.json in the inbox directory), which is safe to resubmit under a new ID;
# if an errored attempt did publish (spec.json present), it is recorded and not repeated.
# Usage: submit_retry.sh SPEC.json [SPEC.json ...]; results are appended to $LOGDIR/published.txt.
ROOT=/Users/manu/Desktop/PointStream/.claude/worktrees/agent-ae50d9755e58f7ac5
LOGDIR=/private/tmp/claude-501/-Users-manu-Desktop-PointStream--claude-worktrees-zealous-nightingale-52e721/cb149cac-af80-43b3-89f1-5416be3aa053/scratchpad/w5
INBOX=/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox
mkdir -p "$LOGDIR"
cd "$ROOT" || exit 1
for spec in "$@"; do
  name=$(basename "$spec" .json)
  for attempt in 1 2 3 4 5 6 7 8; do
    "$ROOT/scripts/ps-fleet" submit --source-worktree "$ROOT" --snapshot-path src --snapshot-path experiments \
      --snapshot-path tools --snapshot-path pyproject.toml "$ROOT/$spec" > "$LOGDIR/$name.log" 2>&1
    rc=$?
    id=$(grep -o '20[0-9]\{6\}T[0-9]\{6\}Z-[0-9a-f]\{8\}' "$LOGDIR/$name.log" | head -1)
    if [ $rc -eq 0 ]; then
      echo "$name published $id attempt $attempt" >> "$LOGDIR/published.txt"; break
    fi
    if [ -n "$id" ] && ssh -o BatchMode=yes -o ConnectTimeout=20 gpu3 "test -f $INBOX/$id/spec.json"; then
      echo "$name published-after-error $id attempt $attempt" >> "$LOGDIR/published.txt"; break
    fi
    echo "$name attempt $attempt unpublished ($(tail -c 120 "$LOGDIR/$name.log" | tr '\n' ' '))" >> "$LOGDIR/published.txt"
    sleep 240
  done
done
