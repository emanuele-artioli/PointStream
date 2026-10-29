# Sessions: root and item sessions

PointStream's work runs as one **root session** and several **item
sessions** (user, 2026-10-22).

- The **root session** may run in the cloud, without access to the GPU
  hosts. It reads PLAN.md, writes items and their rules, starts item
  sessions from cards, reads their reports, and decides what comes next
  with the user. It owns PLAN.md's "Where PointStream stands" and the
  dispatch ledger.
- An **item session** runs locally, where `scripts/ps-fleet` reaches
  gpu1–gpu6. It does one PLAN item, within the goal and scope its card
  states, and reports back.

## What a card holds

A card's prompt stands alone:

- **Goal:** the decision the item has to reach, by a rule in
  the item's file in `docs/experiments/` (written by the root session, or
  as the item's first step).
- **Scope:** what is in, what is out, and the GPU budget.
- **Stop conditions:** the goal is reached; the budget is spent; a
  decision belongs to the user; or the session runs out of ideas.
- **Branch:** the item's own, from `origin/main`.
- **Report address:** the root session's ID.

## How an item session works

1. Read AGENTS.md, the PLAN item and its file in `docs/experiments/`.
2. Work on the item's branch. Edit only the item's own PLAN.md text and
   its own file in `docs/experiments/` (append new entries at its end;
   create it for a new item). Leave
   "Where PointStream stands" and the ledger to the root session. This
   keeps parallel branches from conflicting.
3. Check existing fleet jobs before submitting: other item sessions share
   the pool.
4. Questions the user must answer go to the user in the item session, and
   are mentioned in the report.
5. When a stop condition holds, commit, push the branch, open the pull
   request with auto-merge (AGENTS.md), and send the report.
6. Once the pull request has merged and the report is sent, tell the user
   the session is ready to be archived. Sessions never archive themselves
   or each other: the user may want to give a finished session a new task.

## The dispatch ledger

The root session updates its ledger row at every event, not in batches:
card sent, report received, pull request merged, item stopped. A fresh root
session must be able to carry on from PLAN.md alone, so nothing it needs
lives only in its conversation. The startup hook flags rows still marked
queued whose branch already has a merged pull request
([hooks](#hooks)).

## Asking the user

Rules for when to ask are in AGENTS.md ("Deciding and asking"). The user
switches between sessions and cannot be expected to remember this one, so a
question carries its own context. No fixed template; cover what applies:

- what this work is for, in terms of the paper or the demo;
- what is done and what was found, with the numbers that matter;
- the options, with what each costs (GPU time, work) and risks;
- your recommendation and why;
- exactly what you need from the user.

Explain every internal name (item letters, part numbers, script names) the
first time it appears. Sessions may reason in their own labels; the user
should not have to decode them.

## Compaction

A session cannot run `/compact` itself. At a phase boundary (a pilot done, a
report sent, a decision made), once the state is written to files, tell the
user it is a good moment to compact and give a one-line focus. The focus
names where the state is written, the few things that live only in the
conversation, and what to drop:

> State is in the H5p report and its experiments entry. Keep job 4812
> (running), 3.1 of 8 GPU-hours spent, and the open question on crop size.
> Drop logs and the dead ends already recorded.

Automatic compaction (at 80% in cloud sessions) is the safety net. When the
next phase is a new task, a fresh session from a card is better than
compacting. Cache note: every step re-reads the whole context, cached at
about a tenth of the input price for an hour; a compaction rewrites only the
summary. Keep waits under the hour (`ps-fleet wait` returns within 45
minutes) so the cache survives them. Local desktop sessions were checked on
2026-10-23: all cache writes use the 1-hour lifetime.

## Hooks

`.claude/settings.json` (Claude Code) and `.codex/hooks.json` (Codex) run
these hooks from `tools/agent_hooks/` (standard library only, tested in
`tests/test_agent_hooks.py`):

- **SessionStart** (`session_start.py`): after a compaction, prints the
  branch, uncommitted files, last commits and a reminder to re-check fleet
  jobs by ID, so the session works from live state rather than the summary
  alone. On startup and resume it prints only stale ledger rows, and nothing
  when there are none. It does not query the fleet: that is slow on the
  shared drive, and job IDs are in the experiments entries.
- **PreToolUse** (`guard_large_read.py`): blocks a whole-file Read of a file
  over 1000 lines, and a `cat`/`less`/`more` whose output reaches the
  session, on such a file; logs (`.log`, `.out`, `.err`) at 300 lines. The
  message says what to do instead. Ranged reads and piped or redirected
  output pass.
- **PreToolUse, Codex only** (`guard_protected.py`): blocks patches into
  `Models`/`Datasets`, force pushes (`--force-with-lease` passes) and
  `git reset --hard`, which Claude Code's settings deny natively.

## The report

Sent to the root session with `SendMessage` (the root session appears in
`ListAgents`; use the ID from the card). If it cannot be reached, end the
session's last reply with the report so the user can forward it. The same
text goes into the item's entry in `docs/experiments/` as its outcome.

```markdown
## Report: <item> (<date>)
- Stopped because: goal reached | budget spent | user decision needed | out of ideas
- Decision by the rule: pass | fail | unclear | not reached, in one line
- Numbers: the rule's statistic per item, with job IDs, code revision,
  report path and sha256
- Budget: GPU-hours used of the budget
- Branch and commits: <branch> @ <sha>; tests and checks run
- Deviations from the card or the rule, and why
- Open for the user: the questions, if any
- Proposed next step, and what it would cost
```

## Choosing a model per card

Cards say how hard their item is, so cheaper models take the easier ones.

- **Opus:** writing a rule or oracle design, new environments, new
  pipelines, or reading an ambiguous result.
- **Sonnet:** applying a written rule with existing code (judges, reports,
  re-scoring saved outputs), small fleet runs.
- **Haiku:** mechanical, fully specified work: collecting files, rendering
  crops for a visual check, refreshing a table.

## Subagents inside a session

The same three tiers exist as Claude Code subagents in
[`.claude/agents/`](../.claude/agents/): `opus-designer` (Opus, medium
effort), `sonnet-builder` (Sonnet, high) and `haiku-runner` (Haiku, high).
A card picks the session's model; a subagent takes one piece of a
session's work and returns a short report.

**What a subagent costs.** It starts cold: its own prompt, the brief, the
full AGENTS.md and CLAUDE.md files and git status, but none of the
conversation. It then re-reads whatever files it needs. Its report lands
in the main context and is re-read on every later step. The total is
roughly: startup + the brief + its own exploration + the report + any
re-checking the main session does. Delegating saves tokens only when the
work it absorbs (reads, logs, edit-test loops) is large next to the
brief and the report, and when the cheaper model's rate outweighs the
re-reading. An Opus subagent under an Opus session never saves on rate;
use it for context isolation, not for savings.

**Delegate** when the task:

- is self-contained: a brief of a paragraph or two covers it;
- would flood the main context: many files, long logs, many
  edit-and-test rounds;
- fits a cheaper tier than the session's model; or
- runs independently in parallel with other work.

**Keep in the main session** a task that:

- takes a handful of tool calls;
- needs the conversation's context, so the brief would retell it;
- is a step you would re-verify in full anyway;
- needs back-and-forth with the user.

**Briefs.** Name the goal, the files or area, what done looks like, the
GPU budget (zero unless stated) and whether to commit. A `haiku-runner`
brief lists every step and its expected result; Haiku stops at the first
surprise instead of improvising. Each agent caps its report length and
cannot spawn further subagents, so the tree stays one level deep.

**Escalate, don't retry.** If a `haiku-runner` stops, either finish the
step in the main session or re-brief it to `sonnet-builder`; do not
re-send the same brief.

## Codex

Codex sessions follow the same protocol. What differs:

- **Reports and cards.** Codex cannot message other sessions or show cards.
  An item session ends its last reply with the report and writes the same
  text into its experiments entry, then pushes; the user forwards it, or
  the root session reads it from the pushed branch. The root session writes
  a card as a prompt the user pastes into a new Codex session.
- **Model per card.** Codex runs `gpt-6.1-sol` (AGENTS.md); the tier
  becomes reasoning effort: Opus work at `high`, Sonnet work at `medium`,
  Haiku work at `low`.
- **Waiting on the fleet.** Use the native heartbeat in
  [fleet](fleet.md#monitoring-from-the-submitting-chat), or
  `scripts/ps-fleet wait` as a Claude Code session does.
- **Compaction.** `/compact` and automatic compaction work as in Claude
  Code; the SessionStart hook runs after both. Codex's prompt cache is
  automatic and its lifetime is not set by us (minutes, not guaranteed an
  hour), so compacting before a long wait pays off more than in Claude Code.
- **Hooks and blocks.** `.codex/hooks.json` runs the same SessionStart and
  large-read hooks, plus `guard_protected.py`, which blocks what Claude
  Code's settings deny. Codex loads project hooks only once the project is
  trusted and the hooks are reviewed with `/hooks`.
- **Archiving.** As in Claude Code, a session says it is ready to be
  archived; the user archives it.
