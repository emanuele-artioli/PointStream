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
| **B — components** | ⬜ **next** | Seven parallel briefs in `plans/`. Start B3, it is largest |
| C — pipeline and runner | ⬜ | Blocked on B |
| D — experiments layer | ⬜ | Blocked on C |
| E — experiments and paper | ⬜ | Ordered by §7 |

**Code.** `src/contracts/` is complete and green — 190 tests, ruff and mypy
clean. The property to preserve: **it imports nothing heavy**, so a configuration
validates on a machine with no torch, cv2 or ffmpeg. Tensors are described by a
structural protocol rather than imported; registry targets are lazily-resolved
import strings.

**Paper.** Introduction, Related Work, System Design, Future Work written.
Evaluation is a skeleton of `GOAL`/`HOLE` markers waiting for results.
Conclusion absent until there are results to conclude from.

**Left wired but unused, deliberately.** Per-axis registry modules are
pre-created empty so parallel streams do not contend on a shared table.
`config.validate_backends` is a third validation pass taking registries as
arguments — keeping `contracts` free of heavy imports — and is a no-op until they
are populated. Wiring it is each stream's job.

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

**Phase B — components.** Seven parallel workstreams, one brief each in
`plans/`. B3 is largest; start it first. Each owns its files exclusively and
reports rather than reaching into another stream's.

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

Nothing below P0 blocks a submission.

**P0 — without these there is no paper**
1. Quality measurement working at all — a tier config producing real numbers.
2. PointStream against the codec ladder, including region arms.
3. The residual-coarseness curve.
4. The core ablation lattice.
5. A working generative engine, or an honest scoped negative result.
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

Phase gates, each a real check that fails today:

- **After B:** `libsvtav1` + `yuv444p` raises rather than silently emitting
  yuv420p; every codec rung has a region arm or a documented reason it cannot;
  one command lists every registered backend on every axis.
- **After C — the decisive gate:** `config/tier_fast.yaml` and
  `config/tier_quality.yaml` each produce a run summary carrying **real PSNR,
  SSIM, VMAF and LPIPS numbers**; a residual-absent run completes and reports its
  measured quality drop; the all-off corner reduces to the source video. None of
  this works today in any configuration.
- **After D:** a codec sweep invokes PointStream itself and emits one
  rate-quality-time table containing every arm.
- **During §7:** every cited run has empty `invariant_failures`.
