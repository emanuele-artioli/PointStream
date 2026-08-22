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

### Wave 1 — five independent streams ✅ reported 2026-08-22

Branches: `phase-bp/bp1` … `bp4` (from `phase-b/integrate`), paper `phase-bp/bp6`.
Merged into `phase-bp/integrate` by BP7. Wave 2 reported 2026-08-22 (unmerged).

| Brief | Owns | Why it is independent |
|---|---|---|
| `BP1-probe-set.md` | `assets/probe_set`, probe tooling | Data, no model code |
| `BP2-region-metrics.md` | `src/components/metrics/**` | Metrics only |
| `BP3-generator-loaders.md` | ControlNet family, pix2pix, spade, upscale | Disjoint files from BP4 |
| `BP4-flagship-candidates.md` | `animate_anyone.py`, `mofa.py`, new engines | Disjoint files from BP3 |
| `BP6-related-work.md` | the **paper repo** only | Different git repo entirely |

`BP3` and `BP4` both touch `src/components/generation/` but **own disjoint
files**. Neither edited the registry table in `__init__.py`. Apply both
entries in one edit at Wave-1 merge (implementations live on those two
branches; registering them on `phase-b/integrate` alone would name modules
that are not there yet):

```python
# After pose-controlnet. Same OpenPose ControlNet, sparse-trajectory control.
_add(
    "trajectory-controlnet",
    "src.components.generation.controlnet:ControlNetGenerator",
    summary=(
        "ControlNet OpenPose driven by a rendered trajectory image. "
        "Same backbone as pose-controlnet; the control image changes."
    ),
    capabilities=(
        appearance(APPEARANCE_COMPRESSED_IMAGE)
        | motion(MOTION_SPARSE_TRAJECTORIES)
        | {CAP_PER_FRAME}
    ),
    requires=frozenset({CONDITION_MOTION_FIELD, CONDITION_APPEARANCE}),
    aliases=("trajectory-render",),
    defaults={"variant": "trajectory"},
)

# Replace the animate-anyone summary: not a single match.
# assets/dataset/pointstream_aa_meta.json is 7 matches, 114 tracks.

_add(
    "stable-animator",
    "src.components.generation.stable_animator:StableAnimatorGenerator",
    summary=(
        "StableAnimator pose-to-video. Adapter Apache-2.0 on HF card "
        "FrancisRing/StableAnimator (checked 2026-08-22); inference needs "
        "SVD-XT (Stability AI, not bundled). GitHub code is MIT."
    ),
    capabilities=_PER_FRAME_IMAGE_POSE | {CAP_TEMPORAL_SEQUENCE},
    requires=frozenset({CONDITION_POSE, CONDITION_APPEARANCE}),
    aliases=("stableanimator", "stable_animator"),
)
```

### Wave 1.5 — integration, sequential

| Brief | What |
|---|---|
| `BP7-merge-and-align.md` | Fix the pose alignment, re-run BP3's numbers, merge the four branches, apply the registry entries |

**Not parallel.** It touches every Wave-1 branch and the shared registry. It also
fixes a live fault: 5 of 12 probe clips have colour frames and no skeleton
(`PLAN.md` §2.3), so Wave 2 cannot rank engines until it lands.

### Wave 2 — four parallel streams ✅ reported 2026-08-22

Branches: `phase-bp/bp5`, `phase-c/c1`, `phase-c/c2`, `phase-d/cleanups` (from `phase-bp/integrate` `18bf21e`). Not merged. C3 has not started.

| Brief | Depends on | Owns | Head |
|---|---|---|---|
| `BP5-roster-decision.md` | `BP7` | the probe harness, the invariants, the roster | `36e511d` |
| `C1-reconstruction-residual.md` | `BP7` merge only | reconstruction, the residual spectrum | `8ce3450` |
| `C2-encoder-pipeline.md` | `BP7` merge only | the stage DAG, the encoder | `26533b0` |
| `D-cleanups.md` | nothing | mypy in tests, the AVC region arm | `8470211` |

`C1` and `C2` do not depend on the roster — they are pipeline structure, and
which generator wins does not change them. `C3` (the runner) waits for both.

### Also live
| File | What |
|---|---|
| `DEFERRED.md` | Real work deliberately not now: mypy in tests, SAM3, the AVC region arm, MOFA |

Phase C briefs are written once `BP5` fixes the roster — the residual and
reconstruction work (C1) is largely independent and is the natural candidate to
join a later wave.
