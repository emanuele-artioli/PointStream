# Workstream briefs

**A session reads `AGENTS.md`, `PLAN.md`, and exactly one of these.** A
workstream that cannot be described in one brief is scoped too broadly to hand to
one session. Files not listed under a brief's "owns" belong to another stream: if
you need a change there, say so in your report rather than making it.

## Live

| Brief | Owns | Status |
|---|---|---|
| `BP8-appearance-conditioning.md` | generation loaders, training script | **critical path** |
| `BP9-probe-harness.md` | `experiments/probe/**` | parallel with BP8 |
| `P1-paper-catchup.md` | the paper repo only | parallel with everything |
| `DEFERRED.md` | — | real work deliberately not now |

**The one thing to know before reading anything else:** the generative engines
were trained without appearance as an input and lose to pasting the keyframe
forward. `PLAN.md` §2.6 has the evidence; `BP8` has the options. Any plan that
assumes a working generator is out of date.

## Done

`done/` holds seventeen finished briefs, each ending with a *Delivered* section
recording what landed and what was measured. History, not instructions —
`done/README.md` indexes them.

**`done/BP5-roster-decision.md`'s roster verdict is void.** It was reached on a
probe that scored engines against their own conditioning image.

## Not yet written

`C3` — the runner: one run path, one accounting implementation, quality
evaluation mandatory on every path. It needs `C1` and `C2`, which are both done,
so it is writable whenever a slot is free.
