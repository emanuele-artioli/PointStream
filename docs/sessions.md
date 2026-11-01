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

Whoever owns a row updates it at every event, not in batches: card
written, session started, report sent, pull request merged, item stopped.
Without a root session (from 2026-10-24) each item session owns its own row
([Codex](#codex)). A fresh session must be able to carry on from PLAN.md
alone, so nothing it needs lives only in a conversation. The startup hook
flags rows still marked queued or card ready whose branch already has a
merged pull request ([hooks](#hooks)).

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

## Reporting and waiting

Agreed with the user on 2026-10-29: output tokens cost, and the user cannot
read a stream of intermediate reports.

- **Work silently.** Report once when a task or phase ends: outcome first,
  about ten lines. Item sessions still send [the report](#the-report).
- **Interrupt only for a change:** a decision the user must make, a blocker,
  a budget about to be exceeded, or a result that changes the plan.
- **A wake that changes nothing ends in one line.** Subagent completions,
  fleet waits and keepalive pings each resume the session, and each turn ends
  with a message. When nothing changed for the user, that message is one line
  ("H5p: 3/5 jobs done, nothing to do").
- **Progress lives in files:** the item's `docs/experiments/` entry and its
  card row, updated at each event. The user reads them when they want, and
  they survive compaction.

### Keeping the cache warm

Claude Code writes a one-hour prompt cache; the hour counts from the last
request. Rewriting an expired cache costs about 2x the context's input price,
reading it about 0.1x (Anthropic's multipliers as of 2026-10-29), so a ping
costs about a twentieth of a rewrite. At 45-50 minute spacing, keepalives
cost less than one rewrite for waits up to about 14 hours.

- **Fleet waits** need no keepalive: each `scripts/ps-fleet wait` ends within
  45 minutes and resumes the session inside the cache hour.
- **Waiting on the user (Claude Code only).** When a turn ends waiting for
  the user, schedule one one-shot wakeup about 50 minutes later (`send_later`
  in cloud sessions; verify a local scheduler's one-shot semantics before
  relying on it). A ping does no other work, ends in one line and arms only
  the next ping. Stop the chain 8 hours after the user's last message. When
  the user writes, cancel the pending ping. One-shot pings cannot loop if
  inference hits a usage limit.
- **Codex** gets no keepalives: its cache lifetime is not exposed, and a chat
  kept 98% of its cache after 118 minutes (below).

## Compaction

A session cannot run `/compact` itself. At a phase boundary (a pilot done, a
report sent, a decision made), once the state is written to files, tell the
user it is a good moment to compact and give a one-line focus. The focus
names where the state is written, the few things that live only in the
conversation, and what to drop:

> State is in the H5p report and its experiments entry. Keep job 4812
> (running), 3.1 of 8 GPU-hours spent, and the open question on crop size.
> Drop logs and the dead ends already recorded.

If the phase delegated work to subagents, suggest running the
`explain-usage` skill before compacting, while the phase is still in the
context. It shows where the phase's tokens went, so the user can check
whether delegating paid off and adjust the
[delegation rules](#subagents-inside-a-session).

Automatic compaction (at 80% in cloud sessions) is the safety net. When the
next phase is a new task, a fresh session from a card is better than
compacting. Cache behavior differs by harness. Claude Code's local check on
2026-10-23 found one-hour cache writes; this is not a Codex guarantee.
45-minute bounded stages avoid forgotten work and, in Claude Code, also resume
the session inside the cache hour.

Codex's desktop cache lifetime is not exposed here. On 2026-10-24 this T1 chat
(`gpt-6.1-sol`, medium) resumed after about 118 minutes: the first new inference
reported 137472 cached input tokens out of 139911 (98.3%). This is observed
cache accounting, not proof of physical KV residency or a guaranteed TTL. The
public API guide documents a 30-minute default minimum for GPT-5.6 and later,
with a 12-hour API option; neither identifies this app's actual setting.
Source: [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching).
In Codex, do not wake a model to keep its cache warm; Claude Code does so only
as in [Keeping the cache warm](#keeping-the-cache-warm). Either way, save
durable continuation state and use the [completion-first monitoring policy](fleet.md#monitoring-from-the-submitting-chat).

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
effort), `sonnet-builder` (Sonnet, high) and `haiku-runner` (Haiku,
medium). Codex's equivalents are in the table in AGENTS.md. A card picks
the session's model; a subagent takes one piece of a session's work and
returns a short report.

**What a subagent costs.** It starts cold: its own prompt, the brief, the
full AGENTS.md and CLAUDE.md files and git status, but none of the
conversation. It then reads the files its task needs, some of which the
main session may already have read; a brief that carries the facts
(paths, line ranges, numbers, the command) saves it that. Its report lands
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
brief lists every step and its expected result, including an expected
failure ("exit 1: no matches"). Each agent caps its report length.

**Stopping at the first surprise.** Three things hold `haiku-runner` to
it: its prompt says so; a `PostToolUseFailure` hook
(`tools/agent_hooks/subagent_guard.py failed`) adds a reminder to stop
and report after every failed step; and `maxTurns: 25` bounds the rest.
A result that runs but differs from the expected one is caught only by
the prompt, and by the main session comparing the report with the brief.

**Delegating downhill.** `opus-designer` may launch `sonnet-builder` and
`haiku-runner`; `sonnet-builder` may launch `haiku-runner`; `haiku-runner`
launches nothing. Claude Code cannot restrict this in a subagent's
definition (it ignores type lists there and lets the caller override the
child's model), so a `PreToolUse` hook (`subagent_guard.py spawn`) blocks
any other type and any model override. Each level summarizes the one
below, so detail is lost at every hop: nest only when the sub-work is
verbose and fully briefable, at most three launches per agent.

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
- **Model per card.** Use the model and effort together from AGENTS.md's
  tier table. With the desktop collaboration tool, explicit overrides require
  `fork_turns="none"`; full-history forks inherit the parent configuration.
  Give an isolated delegate a complete scoped brief. An accepted override is
  not proof of its effective model when runtime telemetry is unavailable.
- **Waiting on the fleet.** Use the
  [completion-first policy](fleet.md#monitoring-from-the-submitting-chat):
  verified completion wakeups, otherwise one self-expiring deadline check.
  A Codex background session handle does not establish automatic model wakeup.
  No recurring LLM heartbeat or cache keepalive.
- **Compaction.** `/compact` and automatic compaction work as in Claude
  Code; the SessionStart hook runs after both. Cache expiry is not loss of
  saved conversation history. Compact when the context needs it, not on an
  assumed cache timer.
- **Hooks and blocks.** `.codex/hooks.json` runs the same SessionStart and
  large-read hooks, plus `guard_protected.py`, which blocks what Claude
  Code's settings deny. Codex loads project hooks only once the project is
  trusted and the hooks are reviewed with `/hooks`.
- **Archiving.** As in Claude Code, a session says it is ready to be
  archived; the user archives it.
- **Without a root session** (from 2026-10-24): each item session keeps
  its own row of PLAN.md's dispatch ledger (`running` when it starts,
  `reported` with its outcome when it stops) and its item's bullet in
  "Where PointStream stands", so PLAN.md stays the single record.

### Codex validation

First desktop audit: **2026-11-01**, during MoQSplat reproduction work. The
installed local CLI reports `codex-cli 0.152.0`; the desktop runtime version
was not exposed, so the CLI version does not identify the desktop harness.

- Requested a designer (`gpt-6.1-sol`, medium), builder (`gpt-6.1-terra`, medium)
  and runner (`gpt-6.1-terra`, low), each with an isolated factual brief.
  The tool accepted all overrides. Every delegate reported that its actual
  model and effort were unavailable. Effective tier identities remain
  **unverified**; this is not evidence of different models being selected.
- The frozen fork-audit runner stopped at its first DNS failure without
  retrying. In a separate one-step deliberate `false` probe it reported exit1
  and stopped without further tools. This verifies observed prompt discipline
  in those two turns, not general enforcement by a hook.
- The available collaboration API exposes no role-specific hook field.
  Role-configuration hook enforcement was not demonstrated. Keep the frozen
  brief's stop-at-first-surprise rule; do not describe Claude's agent-identity
  hook as active under this Codex interface.
- Reviewed the tier table, delegation criteria and large-read hook. Scoped
  briefs and compact reports worked. The old Codex paragraph incorrectly
  reduced tiers to reasoning effort on one model; it now references the
  model-and-effort table. Full-history forks cannot carry explicit overrides
  in this interface, so cheaper-tier delegates need isolated briefs.

Repeat these checks when the harness exposes effective-model telemetry or
role-specific hooks, or after an interface change:

1. **Models.** One trivial delegate per tier: `gpt-6.1-sol` at `medium`,
   `gpt-6.1-terra` at `medium` and at `low`. Record the model and effort
   the delegate actually reports.
2. **Runner discipline.** A runner given a frozen launch-and-report brief
   with one deliberately failing step stops there and reports, without a
   workaround. Claude Code enforces this with a hook
   (`tools/agent_hooks/subagent_guard.py failed`, wired from
   `.claude/agents/haiku-runner.md`). The same script cannot simply be
   reused: Codex's `PostToolUse` does run after a non-zero Bash exit and
   can add `additionalContext`, but its tool events carry no agent
   identity and its hooks cannot be scoped to one subagent (Codex hooks
   docs, read 2026-10-29), so the reminder would also reach the main
   session. Check whether an agent role's config file can carry its own
   hooks; if not, say whether the runner's prompt alone holds.
3. **Parity review.** Read the tier table and the delegation rule in
   AGENTS.md, [Subagents inside a session](#subagents-inside-a-session),
   and the delegation hint in `tools/agent_hooks/guard_large_read.py`
   (which also runs under Codex). Note whatever does not hold for Codex.

Record the date, the Codex version and the findings here, fix what is
wrong in a PR, and delete the "Not yet validated in Codex" paragraph from
AGENTS.md.
