# Archived briefs — completed, superseded or parked

This directory preserves evidence and decisions, not current dispatch instructions.
Not every archived task finished: BP19 and BP28 are parked, and P1 is superseded
by `../PAPER-NEXT.md`. Historical status paragraphs describe their original date.
For current work, read `../README.md` and `../../PLAN.md`.

| Brief | Stream | Landed |
|---|---|---|
| `B1-codec.md` … `B7-domain.md` | Phase B — the sixteen component axes | 2026-08-22 |
| `BP1-probe-set.md` | Probe set v2 rebuild | 2026-08-22 |
| `BP2-region-metrics.md` | Region-scoped scoring | 2026-08-22 |
| `BP3-generator-loaders.md` | ControlNet family, pix2pix, SPADE, upscale loaders | 2026-08-22 |
| `BP4-flagship-candidates.md` | Animate-Anyone wired, StableAnimator wrapped | 2026-08-22 |
| `BP5-roster-decision.md` | First roster probe — **its conclusion is superseded**, see `plans/done/RESEARCH-HISTORY.md` §2.6 | 2026-08-22 |
| `BP6-related-work.md` | Related work repaired; merged into the paper repo | 2026-08-22 |
| `BP7-merge-and-align.md` | Pose alignment fix and the Wave-1 merge | 2026-08-22 |
| `C1-reconstruction-residual.md` | Reconstruction and the residual spectrum | 2026-08-22 |
| `C2-encoder-pipeline.md` | The stage DAG | 2026-08-22 |
| `D-cleanups.md` | mypy 66→0; AVC `addroi` verified a no-op under `-qp` | 2026-08-22 |
| `BP9-probe-harness.md` | static-copy made a permanent probe arm | 2026-08-23 |
| `BP10-appearance-pathway.md` | **VOID** — its "≥ +3 dB = ReferenceNet works" gate certifies a paste | 2026-08-23 |
| `BP11-headroom-and-currency.md` | superseded by `BP13` then `BP20` | 2026-08-23 |
| `BP12-clip-mode-roster.md` | clip mode, LPIPS ranking, null control — **every engine loses to a pasted keyframe** | 2026-08-23 |
| `BP13-motivating-headroom.md` | headroom harness; its *number* was synthetic and is superseded by `BP20` | 2026-08-23 |
| `BP16-ci-signal.md` | CI un-redded after twelve consecutive failures; two real config faults fixed | 2026-08-23 |
| `BP17-caption-channel.md` | the trained caption channel made reachable — and worth nothing measurable | 2026-08-23 |
| `BP18-appearance-identity-metric.md` | `reid` + `palette`, calibrated on ground-truth pairs at 17.1σ | 2026-08-23 |
| `BP20-headroom-real-ladder.md` | **the premise holds**: a player is ~1% of pixels and 17–24% of the bitrate on real 4K | 2026-08-23 |

**Three more warnings.**

`BP10-appearance-pathway.md` is **void**. Its bands classify a pasted keyframe as
having a working ReferenceNet — a paste scores +4.45 dB and +0.285 LPIPS on that
test, the top of the scale, with no network at all. Kept for the record of how
the mistake was found. See `plans/done/RESEARCH-HISTORY.md` §2.10.

`BP13-motivating-headroom.md`'s design is sound and its *number* is not: it
measured a 96×128 synthetic court. `BP20` replaced it on real 4K. The one thing
from BP13 that survived contact with real content is its alarm — flat fill
**understates** the prize, so "flat is an upper bracket" is void on both.

`BP12-clip-mode-roster.md`'s verdict stands and is the current roster: **no
generative engine beats pasting the keyframe**, and the best of the eight is a
non-generative upscaler. Do not re-derive it.

**One older warning.** `BP5-roster-decision.md` reached a roster conclusion —
"ControlNet holds both quality slots" — on a probe that scored engines against
their own conditioning image. On the real coding task those engines lose to a
static copy. The brief is kept for its harness design and its bounds discipline;
**its roster verdict is void.** See `plans/done/RESEARCH-HISTORY.md` §2.6 and `../BP8-appearance-conditioning.md`.
