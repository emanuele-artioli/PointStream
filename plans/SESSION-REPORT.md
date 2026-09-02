# Work-session contract

Every dispatched session owns one roadmap row, one branch and one worktree.
The brief must name allowed files. Coordination documents are merged to the
shared branch before another wave depends on them.

## Choosing the harness

- **Codex / Claude:** scientific question, experiment design, architecture,
  bounds, alarm diagnosis, cross-workstream integration, claim wording, final
  paper synthesis.
- **Cursor:** bounded Python implementation, tests, extraction scripts, codec
  adapters, reproducibility automation.
- **VS Code + Antigravity:** routine multi-file edits, plots/tables, manuscript
  mechanics, batch setup and result collation.
- **Detached shell job:** long experiment execution after a high-level session
  has approved the bounds and command.

If a routine session discovers that the scientific question or metric must
change, it stops and reports the fork instead of deciding it.

## Prompt header

Every session prompt states:

1. roadmap ID and one-sentence objective;
2. starting commit, branch/worktree, allowed files;
3. exact inputs and output directory;
4. assumptions already decided;
5. pre-run bounds and required nulls, when measuring;
6. acceptance tests and commands;
7. what is deliberately out of scope;
8. the report path and commit/PR expected.

## Required final report

Write the report into the brief's **Delivered** section or its named report file:

- outcome first: complete, partial, failed or blocked;
- commit and PR, plus files changed;
- commands/tests run and their exact outcomes;
- input manifest, sample count and failures;
- tool paths, versions, checkpoints and configuration;
- size, declared quality metrics, encode time and decode time;
- per-video values, mean, standard error and curve overlap;
- output paths and a reproduction command;
- bounds/nulls, every alarm, and how each was closed or left open;
- conclusion licensed by the result, and claims it does not license;
- the next decision or dependency.

A report that says only “tests pass” or only supplies an output path is
incomplete. A tolerant batch that exits zero must also report submitted,
successful and failed entry counts.
