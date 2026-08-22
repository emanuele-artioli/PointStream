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
| `BP-engine-roster.md` | **B′ — wiring generator weights, fixing the roster** | 12 |

**Phase B is done.** Each `B*.md` brief now ends with a *Delivered* section
recording what actually landed and what is still outstanding — read that before
assuming a stream is finished. `BP-engine-roster.md` is the live one.
