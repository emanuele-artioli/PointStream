"""Hooks wired from the subagent definitions in .claude/agents/.

`spawn NAME...` (PreToolUse on Agent): a subagent may launch only the
named, cheaper agents, at their own model. Claude Code ignores a type
list on `Agent` inside a subagent definition and lets the caller
override the child's model, so this hook is what keeps delegation going
downhill: opus-designer -> sonnet-builder/haiku-runner -> haiku-runner.
Blocking exits with code 2, the reason on stderr.

`failed` (PostToolUseFailure): tells a runner that a step failed and
that it stops and reports unless the brief expected the failure.
"""

from __future__ import annotations

import json
import sys


def spawn_block(payload: dict, allowed: list[str]) -> str | None:
    if payload.get("tool_name") != "Agent":
        return None
    tool_input = payload.get("tool_input") or {}
    kind = tool_input.get("subagent_type") or "general-purpose"
    if kind not in allowed:
        return (
            f"This subagent may launch only {', '.join(allowed)}, not {kind} "
            "(docs/sessions.md#subagents-inside-a-session). Do the work yourself, "
            "or return it to the main session in your report."
        )
    if tool_input.get("model"):
        return (
            f"Launch {kind} without a model override: it runs at the model its "
            "definition sets."
        )
    return None


def failure_context(payload: dict) -> str:
    error = str(payload.get("error") or "the tool reported an error").strip()
    first = error.splitlines()[0] if error else "the tool reported an error"
    return (
        f"Step failed ({payload.get('tool_name', 'tool')}: {first}). "
        "If the brief expected this result, continue. Otherwise stop now: do not "
        "retry or work around it. Report which step failed, the command, and "
        "this error."
    )


def main(argv: list[str]) -> int:
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        return 0
    if not isinstance(payload, dict) or not argv:
        return 0
    if argv[0] == "spawn":
        message = spawn_block(payload, argv[1:])
        if message:
            print(message, file=sys.stderr)
            return 2
        return 0
    if argv[0] == "failed":
        print(json.dumps({
            "hookSpecificOutput": {
                "hookEventName": "PostToolUseFailure",
                "additionalContext": failure_context(payload),
            }
        }))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
