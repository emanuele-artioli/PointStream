# Workstream briefs

**A session reads `AGENTS.md`, `PLAN.md`, and exactly one of these.** A
workstream that cannot be described in one brief is scoped too broadly to hand to
one session. Files not listed under a brief's "owns" belong to another stream: if
you need a change there, say so in your report rather than making it.

## Live

| Brief | Owns | Status |
|---|---|---|
| `BP8-appearance-conditioning.md` | generation loaders, training script | landed unmerged — **honest negative** |
| `BP9-probe-harness.md` | `experiments/probe/**` | landed unmerged `phase-bp/bp9` |
| `P1-paper-catchup.md` | the paper repo only | landed unmerged `phase-p/p1` |
| `C3-runner.md` | `src/runner/**` | landed unmerged `phase-c/c3` |
| `DEFERRED.md` | — | real work deliberately not now |

**The one thing to know before reading anything else:** the generative engines
do not use appearance and lose to pasting the keyframe forward. That survived
re-examining Animate-Anyone, wiring a real IP-Adapter, and retraining pose
ControlNet with a same-track reference (epoch 10: **11.18 dB** vs **11.47 dB**
letterbox floor). Quality flagship is unset. Any plan that assumes a working
generator is out of date.

**Cursor Wave-3 report for Claude:** `plans/wave3-report.md` (copy at
`/tmp/cursor-report-wave3.md`). Parse block at the top.

## Done

`done/` holds seventeen finished briefs, each ending with a *Delivered* section
recording what landed and what was measured. History, not instructions —
`done/README.md` indexes them.

**`done/BP5-roster-decision.md`'s roster verdict is void.** It was reached on a
probe that scored engines against their own conditioning image.

## Not yet written

Nothing in this tree. Wave 3 reported 2026-08-23; nothing merged to `main`.
Option C (what the paper claims is transmitted) is the finding BP8 left, not a
new code stream.
