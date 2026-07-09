#!/usr/bin/env python3
"""PreToolUse guard for POINTSTREAM: block destructive rm against protected dirs.

Reads the Claude Code hook JSON on stdin. If a Bash command actually runs
`rm` against the whole outputs/ or assets/ tree (hours of GPU pipeline runs,
the tennis dataset / raw 4K sources, and the model-weight symlinks), it
denies the call and points at removing a specific outputs/<timestamp>/
directory instead. Everything else is allowed through.

Deleting a single outputs/<timestamp>/ run dir stays allowed. The word "rm"
or a protected name merely appearing inside a string literal (echo/printf/
git commit -m) does NOT trigger a block — only an rm command whose target is
a protected tree does.
"""
import json
import re
import shlex
import sys

PROTECTED = {"outputs", "assets"}
# Command separators that start a new simple-command context.
SEGMENT_SPLIT = re.compile(r"&&|\|\||[;&|\n]")


def _deny(dirname: str) -> None:
    reason = (
        f"Blocked: this rm would delete the whole '{dirname}/' tree, which is "
        "expensive or impossible to regenerate (GPU pipeline outputs / dataset "
        "and raw sources / weight symlinks). If you meant to discard one run, "
        "delete a specific outputs/<timestamp>/ directory instead. See CLAUDE.md."
    )
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }))


def _normalize(arg: str) -> str:
    """Reduce an rm argument to the top-level dir it would wipe, or '' if it
    targets something deeper (a specific subdir/file, which is allowed)."""
    a = arg.strip().lstrip("./").rstrip("/")
    a = re.sub(r"/\*$", "", a)  # outputs/* -> outputs
    return a


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except Exception:
        return 0  # never block on unparseable input
    if data.get("tool_name") != "Bash":
        return 0
    cmd = (data.get("tool_input") or {}).get("command", "")
    if not cmd:
        return 0

    for segment in SEGMENT_SPLIT.split(cmd):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            tokens = segment.split()
        # Strip leading env-assignments and sudo so we find the real command.
        i = 0
        while i < len(tokens) and (tokens[i] == "sudo" or "=" in tokens[i]
                                   and re.match(r"^\w+=", tokens[i])):
            i += 1
        if i >= len(tokens):
            continue
        if tokens[i].split("/")[-1] != "rm":
            continue  # this simple-command is not rm
        # Inspect rm's non-flag arguments.
        for arg in tokens[i + 1:]:
            if arg.startswith("-"):
                continue
            if _normalize(arg) in PROTECTED:
                _deny(_normalize(arg))
                return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
