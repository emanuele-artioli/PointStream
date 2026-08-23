# POINTSTREAM — Claude Code

@.claude/project-core.md

`AGENTS.md` at the repo root is this project's real, hand-edited rule set, read
directly by Cursor, Antigravity, Codex and Copilot's cloud agent. Claude Code
does not read `AGENTS.md`, so this file imports a generated slice instead:
**`.claude/project-core.md` is `AGENTS.md` minus the `host-rules` block**
(already loaded from `~/.claude/CLAUDE.md`).

Add nothing here that is not specific to Claude Code: project rules belong in
`AGENTS.md`, host-wide rules in `~/.agent-rules/AGENTS.md`. Generated files
(`.claude/project-core.md`, `.cursor/rules/cursor-harness.mdc`,
`.github/copilot-instructions.md`) are produced by `tools/sync_agent_rules.py`
— edit `AGENTS.md` and re-run it.
