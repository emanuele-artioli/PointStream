# Workstream briefs

One file per workstream. **A session reads `AGENTS.md`, `PLAN.md`, and exactly
one of these** — that is the whole context it should need, and a workstream that
cannot be described in one brief is scoped too broadly to hand to one session.

Each brief states what the stream owns exclusively, the contract it implements,
what to build, the traps specific to it, and what "done" means. Files not listed
under "owns" belong to another stream: if you need a change there, say so in your
report rather than making it.

| Brief | Owns | Catalogue rows |
|---|---|---|
| `B1-codec.md` | encoding, region control | 14 |
| `B2-perception.md` | scene, detection, selection, tracking, pose, segmentation | 1–4, 8, 9 |
| `B3-generation.md` | generators, appearance, motion | 12, 5, 6 |
| `B4-background-rigid.md` | background model, rigid objects | 11, 10 |
| `B5-transport-temporal.md` | payload serialization, temporal policy | 15, 7 |
| `B6-metrics.md` | quality measurement | 16 |
| `B7-domain.md` | domain profiles | domain |

**Phase B is done.** Each `B*.md` brief above now ends with a *Delivered* section
recording what actually landed and what is still outstanding — read that before
assuming a stream is finished.

## Phase B′ — the live work

Split for parallel sessions. **A wave starts only once every stream it depends on
has reported back**, and all streams in a wave launch together.

### Wave 1 — five independent streams

| Brief | Owns | Why it is independent |
|---|---|---|
| `BP1-probe-set.md` | `assets/probe_set`, probe tooling | Data, no model code |
| `BP2-region-metrics.md` | `src/components/metrics/**` | Metrics only |
| `BP3-generator-loaders.md` | ControlNet family, pix2pix, spade, upscale | Disjoint files from BP4 |
| `BP4-flagship-candidates.md` | `animate_anyone.py`, `mofa.py`, new engines | Disjoint files from BP3 |
| `BP6-related-work.md` | the **paper repo** only | Different git repo entirely |

`BP3` and `BP4` both touch `src/components/generation/` but **own disjoint
files**. Neither edits the registry table in `__init__.py` without saying so in
its report — that is the one contention point between them.

### Wave 2

| Brief | Depends on | Owns |
|---|---|---|
| `BP5-roster-decision.md` | all of Wave 1 | the probe harness, the invariants, the roster |

### Also live

| File | What |
|---|---|
| `DEFERRED.md` | Real work deliberately not now: mypy in tests, SAM3, the AVC region arm, MOFA |

Phase C briefs are written once `BP5` fixes the roster — the residual and
reconstruction work (C1) is largely independent and is the natural candidate to
join a later wave.
