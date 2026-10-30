"""PreToolUse hook: stop whole-file reads of long files and logs.

A whole long file or log fills the context and stays there until the next
compaction. This hook blocks:

- Read without `limit` on a file over LIMIT_LINES lines (LOG_LIMIT_LINES for logs);
- a Bash `cat`/`less`/`more`/`bat` whose output reaches the session, on such a file.

Claude Code (.claude/settings.json) and Codex (.codex/hooks.json) both run it;
Codex has no Read tool, so there only the Bash check applies.

Blocking exits with code 2; the message on stderr tells the session what to
do instead (search, a line range, or the tail plus error lines). In a main
session (no `agent_type` in the payload) it also suggests handing the file to
a subagent. Reads with an explicit range, piped or redirected output, and
short files pass.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path

LIMIT_LINES = 1000
LOG_LIMIT_LINES = 300
LOG_SUFFIXES = {".log", ".out", ".err"}
# The Read tool shows these as images or pages, not as text lines.
MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".pdf"}
PRINTERS = {"cat", "less", "more", "bat"}
DELEGATE_HINT = (
    " If you need more than a few facts from it, brief a runner or builder subagent "
    "with the question instead (AGENTS.md, subagent tiers)."
)
# Below this size no file can exceed the line limits worth counting.
MIN_BYTES_TO_COUNT = 8_000


def line_limit(path: Path) -> int:
    return LOG_LIMIT_LINES if path.suffix.lower() in LOG_SUFFIXES else LIMIT_LINES


def count_lines(path: Path) -> int | None:
    try:
        if not path.is_file() or path.stat().st_size < MIN_BYTES_TO_COUNT:
            return None
        with path.open("rb") as handle:
            return sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1 << 20), b""))
    except OSError:
        return None


def too_long(path: Path) -> int | None:
    """Line count when the file is over its limit, else None."""
    lines = count_lines(path)
    return lines if lines is not None and lines > line_limit(path) else None


def advice(path: Path, lines: int) -> str:
    if path.suffix.lower() in LOG_SUFFIXES:
        how = (
            f"show the tail and the error lines, e.g. `tail -n 40 {path}` and "
            f"`grep -nE 'Error|Traceback|WARN' {path} | head -n 40`"
        )
    else:
        how = (
            f"search it (`grep -n PATTERN {path}`) or read a line range "
            f"(`sed -n '1,200p' {path}`, or Read with offset and limit)"
        )
    return f"{path} has {lines} lines; reading it whole would fill the context. Instead, {how}. (AGENTS.md, Context and compaction)"


def printed_files(command: str) -> list[str]:
    """Files a printer command sends to the session's output."""
    files: list[str] = []
    for pipeline in re.split(r"&&|\|\||;|\n", command):
        stages = [stage for stage in pipeline.split("|")]
        last = stages[-1].strip()
        try:
            tokens = shlex.split(last)
        except ValueError:
            continue
        if not tokens or Path(tokens[0]).name not in PRINTERS:
            continue
        if any(token.startswith((">", "<<")) or token in {">", ">>", "<<"} for token in tokens[1:]):
            continue
        files.extend(token for token in tokens[1:] if not token.startswith("-"))
    return files


def check(payload: dict) -> str | None:
    """The block message, or None to allow."""
    message = _block(payload)
    if message and not payload.get("agent_type"):
        message += DELEGATE_HINT
    return message


def _block(payload: dict) -> str | None:
    tool = payload.get("tool_name")
    data = payload.get("tool_input") or {}
    cwd = Path(payload.get("cwd") or ".")
    if tool == "Read":
        if data.get("limit") is not None or not data.get("file_path"):
            return None
        path = Path(data["file_path"])
        if path.suffix.lower() in MEDIA_SUFFIXES:
            return None
        lines = too_long(path)
        return advice(path, lines) if lines is not None else None
    if tool == "Bash":
        for name in printed_files(str(data.get("command", ""))):
            path = Path(name).expanduser()
            path = path if path.is_absolute() else cwd / path
            lines = too_long(path)
            if lines is not None:
                return advice(Path(name), lines)
    return None


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        return 0
    message = check(payload) if isinstance(payload, dict) else None
    if message:
        print(message, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
