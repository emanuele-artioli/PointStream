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
  experiments.md (written by the root session, or as the item's first
  step).
- **Scope:** what is in, what is out, and the GPU budget.
- **Stop conditions:** the goal is reached; the budget is spent; a
  decision belongs to the user; or the session runs out of ideas.
- **Branch:** the item's own, from `origin/main`.
- **Report address:** the root session's ID.

## How an item session works

1. Read AGENTS.md, the PLAN item and its experiments.md entries.
2. Work on the item's branch. Edit only the item's own PLAN.md text and
   its own experiments.md entries (append new entries at the end). Leave
   "Where PointStream stands" and the ledger to the root session. This
   keeps parallel branches from conflicting.
3. Check existing fleet jobs before submitting: other item sessions share
   the pool.
4. Questions the user must answer go to the user in the item session, and
   are mentioned in the report.
5. When a stop condition holds, commit, push the branch, and send the
   report.

## The report

Sent to the root session with `SendMessage` (the root session appears in
`ListAgents`; use the ID from the card). If it cannot be reached, end the
session's last reply with the report so the user can forward it. The same
text goes into the item's experiments.md entry as its outcome.

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
