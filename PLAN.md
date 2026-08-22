# POINTSTREAM — the plan

*What the project is, where it stands, and what happens next. Read this plus
`AGENTS.md` plus the one brief in `plans/` for your workstream — that is the
whole context a session should need.*

---

## 1. What this is

An object-centric semantic video codec. Each salient object is transmitted as an
**appearance representation** established once, a **motion representation**
describing how that appearance evolves, and a **temporal policy** saying how
densely the motion is actually sent. A background model and an optional
corrective residual complete the payload. The client reconstructs frames
generatively.

The current cycle rewrites this into a platform where every component — codec,
detector, pose, segmenter, generator, appearance and motion representation,
background method, residual, transport, metric, and the task domain itself — is a
config choice, and the only code a new component needs is the wrapper that
satisfies its contract.

Target: **ACM TOMM, September 30.**

## 2. Status

| Phase | State | Next action |
|---|---|---|
| A — contracts and concepts | ✅ done | — |
| B — components | ✅ **done**, one gap | Generation cannot load weights — see §2.1 |
| **B′ — the engine roster** | ⬜ **next** | Wire weights, validate on the probe set (§6) |
| C — pipeline and runner | ⬜ | Blocked on B′ |
| D — experiments layer | ⬜ | Blocked on C |
| E — experiments and paper | ⬜ | Ordered by §7 |

**Code.** `src/contracts/` is complete and green. `src/components/` now covers all
sixteen axes: ~8.6k lines of source, ~3.3k of tests, 52 registered backends of
which 48 construct. 392 contract and component tests pass, plus 13 integration
tests that drive real tools. `ruff` and `python -m src.contracts.layers` are
clean. `mypy` reports 61 errors, all in `tests/components/`, none in `src/`.

All three Phase-B gates pass: `av1` + `yuv444p` raises `CodecConstraintError`
rather than silently emitting yuv420p; every codec rung has a region arm or a
recorded reason it cannot; `python -m src.components` lists every backend on
every axis.

**Quality measurement works.** This was the standing blocker and it is closed at
the component level: VMAF runs through libvmaf (97.4 on identical frames, 28.9 on
degraded), PSNR, SSIM and LPIPS all return real numbers, and FVMD correctly
refuses a single frame. BD-rate with an overlap check is implemented in
`src/components/metrics/bd_rate.py`. What is *not* yet true is §7 P0 item 1 — a
tier config producing those numbers end to end — because there is no pipeline.

**Paper.** Introduction, Related Work, System Design, Future Work written.
Evaluation is a skeleton of `GOAL`/`HOLE` markers waiting for results.
Conclusion absent until there are results to conclude from.

### 2.1 The one real gap: generation cannot load weights

Every `_load_model` and `_load_pipeline` under `src/components/generation/` is an
unconditional `raise RuntimeError(...has no pipeline loaded...)`. Ten generators
are registered, the conditioning contracts and pairing validation are correct and
tested, and **only `upscale-refine` — the non-generative bicubic baseline —
actually produces pixels.** The tests pass because they inject a fake pipeline.

The weights are on disk and unused: `assets/weights/pose-controlnet` (ten
fine-tuned epochs), `seg-controlnet` (seven), `ip-adapter-controlnet` (ten), full
`stable-diffusion-v1-5`, `pix2pix_generator.pt`, `spade4tennis_lite_generator.pt`.

This is the socket built without anything plugged into it, and it is what B′
exists to close.

### 2.2 Known environment limits

- **SAM3 cannot load.** `torch.nn.attention` does not exist in torch 2.2.2. This
  blocks the SAM3 detector comparison (§7 P1 item 10) unless a second env is
  built. Both the detector and segmenter entries fail construction and say so.
- **RF-DETR is not installed.** It needs `transformers>=5.1`; this env pins
  4.46.3. Registered, honest about it, not usable.
- **MOFA-Video is a candidate, not an integration.** Its SVD weights are
  Stability-AI-licensed and not bundled, so construction refuses by design. §6.2
  says what replaces it for the trajectory arm.

---

## 3. Architecture

Enough here that seven parallel sessions do not make conflicting decisions.
Everything deeper is in `src/contracts/`, which is authoritative, or in the
paper's System Design section, which explains why.

### Layers

```
  contracts/    protocols, schemas, capabilities, config — imports nothing heavy
      ▲
  components/   one package per axis, each with a registry table
      ▲
  pipeline/     stage DAG, encoder, decoder — never knows which backend was chosen
      ▲
  runner/       ONE run path: chunk loop, routing, accounting, evaluation
      ▲
  experiments/  matrices, sweeps, campaigns — consumes runner as a library
```

**Dependencies point inward, always**, enforced by
`python -m src.contracts.layers`. The arrangement being replaced had experiment
scripts shell out to the CLI and scrape stdout, which is what an unchecked
boundary decays into.

### The ablation lattice

**Every component is optional, and the residual absorbs whatever the disabled
stages would have handled.** Turn off detection but keep the background: metadata
shrinks, encode time drops, the residual grows to carry the players. Turn
everything off, residual included, and what remains is the source video.

Three things follow, and they are why the architecture is shaped this way:

- Component ablations share **one currency** — BD-rate against a common anchor,
  measured identically for every component (§5). Payload alone is not the
  currency: two corners never land at the same quality, so a byte count on its
  own compares nothing.
- Alternative encodings of the same thing are directly comparable.
- The whole-frame baseline is not a special mode; it is a corner of the lattice.

No stage may be structurally required except codec, transport and metrics.
Graceful degradation to the baseline codec is a property of the architecture, not
a routing special case.

### Quality is always measured

There is no configuration where correctness can be assumed. The residual always
carries some coarseness, and generative inference is statistical — encoder-side
and client-side generation are not guaranteed to produce identical pixels.

So **every run reports quality**. Symmetry between the two sides is a design goal
*verified by measurement*, not a guarantee asserted by construction.
Deterministic stages get bit-identity checks; generative stages get closeness
measurement.

### The pairing constraint

Appearance and motion representations are independent axes, but not every
combination is decodable — a generator declares what it accepts, and config
validation rejects a pair nothing can decode, naming what would work. This is
what stops the design sprawling into combinations nothing implements.

---

## 4. The component catalogue

Every row optional unless marked. "When off" says where the work goes instead —
which is what makes the lattice measurable. Specs live in `plans/`.

| # | Component | When off | Brief |
|---|---|---|---|
| 1 | Scene classification | one span; no semantic-vs-fallback routing | B2 |
| 2 | Subject detection | subjects land in the residual | B2 |
| 3 | Subject selection | every detection treated as salient | B2 |
| 4 | Tracking / identity | no appearance reuse across frames | B2 |
| 5 | Appearance representation | generator has no appearance cue | B3 |
| 6 | Motion representation | object static after appearance; motion to residual | B3 |
| 7 | Temporal policy | every frame fully processed | B5 |
| 8 | Pose estimation | motion must be trajectories or encoded video | B2 |
| 9 | Segmentation | compositing falls back to heuristic masks | B2 |
| 10 | Rigid objects | rackets and balls land in the residual | B4 |
| 11 | Background model | background lands in the residual | B4 |
| 12 | Generation | subjects land in the residual | B3 |
| 13 | Residual | nothing corrects generation error | C1 |
| 14 | Codec | **required** whenever anything is transmitted | B1 |
| 15 | Transport | **required** | B5 |
| 16 | Metrics | **never fully off** — PSNR always runs | B6 |

---

## 5. Evaluation

**The paper's Evaluation section is the living trace.** Each element below has a
`GOAL`/`HOLE` marker pair there; a result lands by clearing its `HOLE` and adding
a `CLAIM(id): src=` line naming a real `outputs/` path. A result that turns out
weak either moves to an appendix or is deleted before submission — it does not
accumulate in a side file.

### The currency is BD-rate, not a byte count

**Two configurations will not land at the same bitrate, and will not land at the
same quality.** Comparing them at one operating point each therefore compares
nothing: a corner that spends more bytes and scores better has told you only that
more bytes buy more quality. This is the same error as comparing a
region-of-interest arm against a baseline at matched QP.

So **every comparison sweeps a rate ladder and compares curves.** Each lattice
corner is run at several residual coarsenesses (and, where relevant, several
codec rate points), producing a rate–distortion curve. Configurations are then
compared by **Bjøntegaard delta rate (BD-rate)** against a common anchor — the
average bitrate difference at equal quality, integrated over the overlapping
quality range — with BD-PSNR/BD-VMAF reported alongside.

This changes what "pays for itself" means, and it is the definition to implement:

> A component earns its place if enabling it **improves BD-rate against the same
> anchor**. Not if it reduces payload at one operating point, and not if it looks
> better.

Consequences that must reach the code:

- **A single run is never a result.** The unit of evaluation is a swept curve.
  An experiment harness that produces one point per configuration is producing
  something uncomparable.
- **Curves must overlap in quality** for BD-rate to be defined. Report the
  overlap range; a comparison over a sliver of shared quality is weak evidence
  and should say so.
- **A point comparison is valid only under dominance** — one arm better on both
  axes. Where that holds, say so and use it; where it does not, BD-rate is the
  only honest answer.
- **With the residual absent there is no guarantee**, so the comparison against
  the baseline codec is not "did we reconstruct exactly" but "where does this
  configuration's RD curve sit relative to the anchor's". That is the question
  every residual-free corner has to answer.

In priority order:

1. **PointStream against the codec ladder** — AVC/HEVC/AV1/VVC, with and without
   region arms, **at matched rate**. Rate, quality *and encode time* on the same
   axes. The core claim. Verified never run: every sweep in
   `outputs/codec_baselines/` has a null PointStream point.
2. **The ablation lattice** — each component off in turn, each swept across a
   rate ladder, compared by BD-rate against the all-off corner.
3. **The residual-coarseness curve** — absent through fine, plus a lossless
   ceiling calibration.
4. **Object representation** — keypoints versus sparse trajectories versus
   per-object encoded video, on identical objects, identical appearance, backbone
   held fixed. No published work does this comparison.
5. **Generalization** — the general/DAVIS profile, and pretrained versus
   fine-tuned generation across both domains.
6. **Real-time versus compute-unbounded operating points**, no minimum resolution
   or framerate imposed; whatever is achieved is reported.
7. **Perceptual and temporal metrics** — LPIPS and FVMD where they earn their
   place, alongside always-on PSNR.

Three honesty constraints, recorded in the paper as `NOTE` markers: the
encode/decode speed reality gets reported rather than omitted; dataset labels are
model-generated rather than human ground truth and deep annotation covers a
subset of scenes; and baselines get region control wherever the encoder supports
it, or beating them proves nothing.

---

## 6. Build phases

**Phase B — components.** ✅ Done. Seven parallel workstreams, one brief each in
`plans/`; each brief now carries a *Delivered* section recording what landed.

### Phase B′ — the engine roster

The narrative comes first. **The paper decides which experiments matter, the
experiments decide which models we need, and only those models get built.** This
is the rule that keeps us from spending September on runs nothing cites. Every
model below is here because a named `GOAL` in the paper's Evaluation section
cannot be answered without it — and a model no `GOAL` needs does not get wired,
however easy it would be.

**We do not need every generator to work. We need a flagship, plus alternatives
that differ along an axis the paper actually measures.**

#### 6.1 What the paper asks of the generator

| Paper slot | What it demands of the roster |
|---|---|
| `subsec:eval-ladder` | **One** engine good enough to put a real RD curve against the codec ladder |
| `subsec:eval-object` | Keypoints vs sparse trajectories vs encoded video **with the backbone held fixed** |
| `subsec:eval-general` | Tennis *and* DAVIS, pretrained *and* fine-tuned |
| `subsec:eval-operating` | A compute-unbounded point *and* something fast enough to have a real-time point at all |
| `subsec:eval-metrics` | Temporal coherence worth measuring with FVMD |

#### 6.2 The roster that follows

The binding constraint is `subsec:eval-object`: *"with the generative backbone
fixed across arms."* Only one family can satisfy it, and that decides the
flagship.

| Role | Engine | Why the narrative needs it |
|---|---|---|
| **Flagship** | **ControlNet on SD-1.5** (pose / seg / ip-adapter variants) | The only family where the backbone genuinely stays fixed while the conditioning signal changes — which is what makes `eval-object` a representation result rather than a model result. Three variants are already fine-tuned on disk. |
| Temporal arm | **Animate-Anyone** | The only engine with real temporal modelling, so the only one that makes an FVMD claim meaningful. Its score stays scoped to the single match it was fine-tuned on, every time it is quoted. |
| Speed rung | **pix2pix** | One forward pass, no diffusion loop. Without it `eval-operating` has no real-time point to report. |
| Floor | **upscale-refine** | Already works. The cheap non-generative baseline every generative arm must beat, and the control that tells us whether generation is buying anything at all. |
| Domain control | **SPADE4Tennis** | Tennis-specific; useful only as a contrast to the general backbone. Wire it if cheap, drop it if not. |

**The trajectory arm does not need MOFA-Video.** MOFA is licence-blocked, and
routing around it also *improves* the experiment: render sparse trajectories as a
control image into the same ControlNet backbone the keypoint arm uses. That keeps
`eval-object`'s "backbone fixed" promise literally true, which a MOFA-vs-ControlNet
comparison never could — it would have confounded representation with model.

#### 6.3 Validate small before scaling

Two stages, in this order, because the expensive one is only worth running once
the cheap one says the roster is real.

- **B′.1 — wire and probe.** Implement the loaders, then drive each engine on
  `assets/probe_set` (the existing minimal set) for a starting number per engine:
  PSNR, SSIM, VMAF, LPIPS on a handful of scenes. Purpose is **triage, not
  results** — which engines produce a plausible frame at all, and how far apart
  they are. Nothing measured here is citable and nothing here gets a `CLAIM`
  line.
- **B′.2 — decide the roster.** From those numbers, confirm or replace the
  flagship. Expect the fine-tuned ControlNet variants to beat the pretrained
  backbone on tennis and lose on DAVIS; expect Animate-Anyone to look strong on
  its own match and poor elsewhere. Both of those are *findings the paper wants*,
  not failures — `eval-general` exists precisely to report that gap.
- **Only then** prepare the full dataset and let §7 drive the real sweeps.

**Bound before believing (§8 applies here).** These checkpoints are lightly
trained, some on a single video, some not trained by us at all. A first-pass VMAF
in the **25–45** band is the expected outcome and is a *pass* for B′.1: it means
the engine runs and the pipeline is honest. Below ~15 suspect a broken inference
path before concluding the model is weak — that exact mistake has already been
made once, when ControlNet's 0.11 VMAF was read as a model result and was in fact
a broken path. Above ~70 on a first pass, suspect the reference.

**Improving the numbers is a separate decision, taken after B′.2**, and in this
order of cost: parameter tuning first, then fine-tuning on our own dataset, then
swapping a model. Do not start any of them before the roster is fixed.

**Phase C — pipeline and runner.**
- *C1 reconstruction and residual*: decompose `SynthesisEngine` (512 lines doing
  panorama resolution, pose unrolling, ball rendering, generative dispatch, CUDA
  determinism and OOM fallback) and `ResidualCalculator` (967 lines). Implement
  the residual coarseness spectrum including absent. Deliver bit-identity tests
  for deterministic stages.
- *C2 encoder pipeline*: DAG built from the enabled stage set, with the skips
  that make a reduced corner genuinely cheaper rather than nominally.
- *C3 runner*: one run path — single-chunk becomes the degenerate case of
  full-match. One accounting implementation instead of two. Quality evaluation
  mandatory on every path.

**Phase D — experiments layer.** Rebuilt to consume `runner/` as a library.
Lattice-driven matrices; the codec sweep actually *invoking* PointStream rather
than reading a pre-existing summary.

---

## 7. Delivery order

Nothing below P0 blocks a submission. **The order is set by the paper, not by
the code** — an item is here because a `GOAL` marker in the Evaluation section
depends on it, and it is prioritised by how much the paper loses without it.
When a result lands and changes what the narrative can claim, this order is
re-read rather than followed blindly.

**P0 — without these there is no paper**
1. Quality measurement working at all — a tier config producing real numbers.
   *Half done: the metrics compute (§2), the tier config cannot run yet.*
2. PointStream against the codec ladder, including region arms.
3. The residual-coarseness curve.
4. The core ablation lattice.
5. A working generative engine, or an honest scoped negative result.
   *This is B′, and it is the true critical path — items 2, 3, 4 and 6 are all
   blocked behind it.*
6. Generalization on the general/DAVIS profile.
7. Evaluation and Conclusion sections; abstract reconciled with what was measured.

**P1 — strongly strengthens**
8. Perceptual and temporal metrics. 9. Object-representation comparison — the most
novel item. 10. Detector comparison including SAM3. 11. Temporal-policy ablation.
12. Encode-time comparison against VVC.

**P2 — only if time remains**
13. Appearance-representation comparison. 14. Keypoint-schema richness.
15. JPEG quality versus downscaling. 16. Open-vocabulary versus hand-written
selection. 17. Animate-Anyone full retrain. 18. Background-layer ladder.
19. Football as a third domain.

**Out of scope, named as future work in the paper:** MOS user study;
variable-keypoint training-regime robustness; representations with no decoder.

---

## 8. Verification

Per stream before merge: `ruff` and `mypy` clean; the required-behaviour suite
passes; tests cover plausible misuse, not line coverage.

**The required-behaviour suite** replaces a percentage gate, because a percentage
gate is satisfiable by padding and this one is not. It asserts: bit-identity for
deterministic stages; every lattice corner produces a runnable pipeline; config
rejects unknown keys; codec constraint violations raise; an undecodable
appearance/motion pair is rejected; no layer imports outward; every registered
backend constructs; every domain profile round-trips; every weight a shipped
config names resolves; every run emits at least one quality metric.

The ~436 pre-rewrite tests are untouched and test modules Phase B and C delete.
They die with their modules; no separate culling is needed.

**The suite does not exist yet.** `tests/invariants/` is a three-test stub that
skips for want of a run summary. Most of the assertions above need Phase C, but
two were checkable at the end of B and were not written: *every registered
backend constructs* and *every weight a shipped config names resolves*. B′ owns
writing both, since B′ is where weights start mattering.

Phase gates:

- **After B — ✅ passed 2026-08-22.** `libsvtav1` + `yuv444p` raises
  `CodecConstraintError` rather than silently emitting yuv420p; every codec rung
  has a region arm or a recorded reason it cannot; `python -m src.components`
  lists every registered backend on every axis. Verified by driving each, not by
  reading the code.
- **After B′:** every engine on the §6.2 roster loads real weights and returns a
  frame that is not the input; each has a probe-set number recorded; the roster
  is fixed in writing with the reason each engine is on it.
- **After C — the decisive gate:** `config/tier_fast.yaml` and
  `config/tier_quality.yaml` each produce a run summary carrying **real PSNR,
  SSIM, VMAF and LPIPS numbers**; a residual-absent run completes and reports its
  measured quality drop; the all-off corner reduces to the source video. None of
  this works today in any configuration.
- **After D:** a codec sweep invokes PointStream itself and emits one
  rate-quality-time table containing every arm.
- **During §7:** every cited run has empty `invariant_failures`.
