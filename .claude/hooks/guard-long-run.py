#!/usr/bin/env python3
"""PreToolUse/Bash hook: nudge when a long job may not be checkpointing.

SSH to this host drops a couple of times a day. A training run that only saves
at the end of an epoch can lose hours to a dropped connection, and the loss is
silent — the job is simply gone. This looks at commands that launch a known
training entry point and warns if nothing on the command line suggests
checkpointing is configured.

Advisory: it prints and exits 0. Checkpoint cadence usually lives in a config
file the command line does not name, so blocking here would be wrong far more
often than it would be right. The real enforcement belongs in the training
scripts themselves, which refuse to start without a writable checkpoint dir.
"""

import json
import re
import sys

TRAINING_ENTRY_POINTS = (
    "train_controlnet",
    "train_pix2pix",
    "train_spade4tennis",
    "train_campaign",
)

# Anything that plausibly names a checkpoint setting, in a flag or a config path.
CHECKPOINT_HINTS = re.compile(
    r"checkpoint|ckpt|save[-_]?(every|steps|interval|freq)|resume|--out[-_]weights",
    re.IGNORECASE,
)

DETACHED = re.compile(r"\bnohup\b|\bsetsid\b|&\s*$|\bdisown\b")


def main():
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return

    command = (payload.get("tool_input") or {}).get("command", "")
    if not command:
        return
    if not any(entry in command for entry in TRAINING_ENTRY_POINTS):
        return

    notes = []
    if not CHECKPOINT_HINTS.search(command):
        notes.append(
            "nothing on this command line names a checkpoint setting — confirm the "
            "config checkpoints at least hourly, and that resume has been tested"
        )
    if not DETACHED.search(command):
        notes.append(
            "this looks attached to the shell; an SSH drop would take the run with it "
            "(use run_in_background, or `setsid nohup ... < /dev/null &`)"
        )

    if notes:
        print("Long-run check: " + "; ".join(notes), file=sys.stderr)


if __name__ == "__main__":
    main()
