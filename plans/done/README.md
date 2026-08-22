# Completed briefs — history, not instructions

**Nothing here is live work.** These are the briefs for streams that finished,
kept because each ends with a *Delivered* section recording what actually landed,
what was measured, and what was left outstanding. That is worth reading when you
need to know *why* something is the way it is; it is not worth reading to decide
what to do next.

**For what to do next, read `../README.md` and `../../PLAN.md`.**

| Brief | Stream | Landed |
|---|---|---|
| `B1-codec.md` … `B7-domain.md` | Phase B — the sixteen component axes | 2026-08-22 |
| `BP1-probe-set.md` | Probe set v2 rebuild | 2026-08-22 |
| `BP2-region-metrics.md` | Region-scoped scoring | 2026-08-22 |
| `BP3-generator-loaders.md` | ControlNet family, pix2pix, SPADE, upscale loaders | 2026-08-22 |
| `BP4-flagship-candidates.md` | Animate-Anyone wired, StableAnimator wrapped | 2026-08-22 |
| `BP5-roster-decision.md` | First roster probe — **its conclusion is superseded**, see `PLAN.md` §2.6 | 2026-08-22 |
| `BP6-related-work.md` | Related work repaired; merged into the paper repo | 2026-08-22 |
| `BP7-merge-and-align.md` | Pose alignment fix and the Wave-1 merge | 2026-08-22 |
| `C1-reconstruction-residual.md` | Reconstruction and the residual spectrum | 2026-08-22 |
| `C2-encoder-pipeline.md` | The stage DAG | 2026-08-22 |
| `D-cleanups.md` | mypy 66→0; AVC `addroi` verified a no-op under `-qp` | 2026-08-22 |

**One warning.** `BP5-roster-decision.md` reached a roster conclusion —
"ControlNet holds both quality slots" — on a probe that scored engines against
their own conditioning image. On the real coding task those engines lose to a
static copy. The brief is kept for its harness design and its bounds discipline;
**its roster verdict is void.** See `PLAN.md` §2.6 and `../BP8-appearance-conditioning.md`.
