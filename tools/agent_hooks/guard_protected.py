"""PreToolUse hook for Codex: the blocks Claude Code has in its settings.

.claude/settings.json denies edits inside Models and Datasets and the
commands `git push --force` and `git reset --hard`. Codex has no such
settings, so .codex/hooks.json runs this hook on `apply_patch` and `Bash`:

- a patch that adds, updates, deletes or moves a file under the
  `Models`/`Datasets` links or their targets;
- a force push (`--force`, `-f`, or a `+ref` refspec; `--force-with-lease`
  is allowed) or `git reset --hard`.

Blocking exits with code 2, the reason on stderr.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
from pathlib import Path

STORES = ("/home/itec/emanuele/Models", "/home/itec/emanuele/Datasets")
LINKS = ("Models", "Datasets")
PATCH_PATH = re.compile(r"^\*\*\* (?:Add|Update|Delete) File: (.+)$|^\*\*\* Move to: (.+)$", re.M)


def protected(path: str, cwd: Path) -> bool:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        if candidate.parts and candidate.parts[0] in LINKS:
            return True
        candidate = cwd / candidate
    text = str(candidate)
    if any(text == store or text.startswith(store + "/") for store in STORES):
        return True
    try:
        resolved = str(candidate.resolve())
    except OSError:
        return False
    return any(resolved == store or resolved.startswith(store + "/") for store in STORES)


def git_block(command: str) -> str | None:
    for part in re.split(r"&&|\|\||;|\||\n", command):
        try:
            tokens = shlex.split(part)
        except ValueError:
            continue
        while tokens and re.fullmatch(r"\w+=.*", tokens[0]):
            tokens = tokens[1:]
        if len(tokens) < 2 or Path(tokens[0]).name != "git":
            continue
        args = tokens[1:]
        while len(args) >= 2 and args[0] in {"-C", "-c"}:
            args = args[2:]
        if not args:
            continue
        if args[0] == "push" and any(
            t in {"--force", "-f"} or (t.startswith("-") and not t.startswith("--") and "f" in t[1:])
            or (t.startswith("+") and len(t) > 1)
            for t in args[1:]
        ):
            return "Force pushes are blocked (AGENTS.md: history on shared branches is never rewritten); use --force-with-lease on your own branch, or ask the user."
        if args[0] == "reset" and "--hard" in args[1:]:
            return "`git reset --hard` is blocked: it discards work. Use `git stash` or `git checkout -B BRANCH REF` on a clean tree, or ask the user."
    return None


def check(payload: dict) -> str | None:
    if payload.get("tool_name") == "Bash":
        return git_block(str((payload.get("tool_input") or {}).get("command", "")))
    if payload.get("tool_name") != "apply_patch":
        return None
    patch = str((payload.get("tool_input") or {}).get("command", ""))
    cwd = Path(payload.get("cwd") or ".")
    for match in PATCH_PATH.finditer(patch):
        path = (match.group(1) or match.group(2)).strip()
        if protected(path, cwd):
            return (
                f"{path} is in Models or Datasets, which agents never edit (AGENTS.md, Storage). "
                "Write outputs under Datasets through the experiment code, not by patching files."
            )
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
