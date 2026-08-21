# POINTSTREAM — rebuild as a configurable research platform

*The operational plan for the whole project, and the first thing to read when
picking it up. Written to be split across subagent sessions.*

*How this document is organized: **§2** is the current status — what is done and
what is next. **§3** collects constraints learned the hard way. **§4** holds
concepts spanning more than one component; a session needs all of it. **§5** says
which documents carry truth and who reads them. **§6** is the per-component
specification, one section per component or tightly-coupled group — a session
reads only its own rows. **§9** builds all the code; **§10** decides what gets run
and written, in what order. Anything about scheduling lives in §9 or §10 and
nowhere else.*

---

## 1. Context

PointStream is an object-centric semantic video codec. The encoder transmits
semantic understanding — salient-object appearance and motion, a background model
— plus an optional corrective residual; the client reconstructs frames with
generative models. Tennis is the current domain, deliberately constrained.

The code works but was written largely by an older model, and it shows: backends
are picked by substring matching scattered across modules, one axis silently
rewrites another, the generative interface is uniform in signature but not in
meaning, and the two entry points have mirror-image gaps that mean **the shipped
tier ladder cannot currently produce a single quality number**.

The decision for this cycle: **the pipeline is the lasting asset.** Full rewrite
into a platform where every axis — codec, detector, pose, segmenter, generator,
appearance and motion representation, background method, residual, transport,
metric, and the task domain itself — is a config choice, and the only code a new
component needs is the wrapper that makes it satisfy the agreed interface.
Cleanups that change generated pixels are in scope; existing generative results
are superseded and re-baselined.

**Nothing in the existing codebase is assumed correct.** Where this plan says a
feature "exists," that means there is prior art to read, not a foundation to
trust — every such component goes through the same rewrite and its own
verification.

Target is **September 30**, with a checkpoint one week in to judge from actual
progress rather than an estimate made before starting. Reworking code is fast;
experiments are slow — hence the split between §9 and §10.

---

## 2. Status

*Update this table as work lands. It is the answer to "where are we", and the
reason this document rather than a separate handoff file is what a new session
reads first.*

| Phase | State | Notes |
|---|---|---|
| Prerequisites | ✅ done | Tree committed, CI restored, literature review folded into the paper |
| **A** — contracts and concepts | ✅ done | `src/contracts/` complete; 190 tests; paper's concept sections written |
| **B** — components | ⬜ not started | Seven parallel streams; B3 (generation) is the largest, start it first |
| **C** — pipeline and runner | ⬜ not started | The decisive gate lives here: a tier config producing real quality numbers |
| **D** — experiments layer | ⬜ not started | Rebuilt to consume `runner/` as a library |
| Delivery (§10) | ⬜ not started | P0 first; nothing below P0 blocks a submission |

**Code.** `src/contracts/` is complete and green — errors, registry, capability
vocabulary, layers and their import check, keypoint schemas, codec ladder,
conditioning, metrics, object stream, domain profiles, stage lattice, strict
config parsing, run configuration. ruff and mypy clean. The property to preserve:
**it imports nothing heavy**, so a whole configuration validates on a machine
with no torch, cv2 or ffmpeg. Tensors are described by a structural protocol
rather than imported, and registry targets are lazily-resolved import strings.

**Paper.** Introduction, Related Work, System Design and Future Work are written.
Evaluation and Conclusion are correctly absent — they need results. The abstract
carries a `HOLE` naming the experiment that would let a number back in. Reviewer
themes 3 and 8 are closed on text; six others advanced.

**Two things Phase A left wired but unused, deliberately.** Per-axis registry
modules are pre-created empty so parallel streams do not contend on a shared
table; and `config.validate_backends` is a third validation pass that takes
registries as arguments — keeping `contracts` free of heavy imports — and is a
no-op until they are populated. Wiring it is each stream's job.

---

## 3. Hard-won constraints

*Things that cost real time once. Each is phrased as the rule that prevents a
repeat, not the story of the failure.*

**Encoders lie by omission.** SVT-AV1 accepts `-pix_fmt yuv444p`, returns
success, and emits yuv420p. Every residual encode since that knob was added
silently requested a format it never got, and the ablation built on it measured
nothing. `contracts/codecs.py` declares what each encoder honours and rejects the
rest — extend that table rather than trusting a flag.

**Two builds of one tool is a trap.** The conda environment carried `svt-av1
1.4.1`, shadowing the system 1.8.0 on `PATH`; only 1.8 has `--roi-map-file` at
all, so testing the wrong one reads as "region control does not work" for reasons
unrelated to region control. Resolve and record a tool's *path and version*, not
just its name.

**Matched QP is not matched rate.** Our own first ROI measurement compared two
arms at the same QP, so the region arm also spent more bytes — more bits buying
more quality is not a result. `codecs.assert_matched_rate_control` encodes the
rule. A prior project lost a published table to the harsher version of this.

**Bound before believing, and record why a bound moved.** State a plausible best
and worst case *before* reading a result. When the ROI bound fired an alarm, the
bound was wrong — derived in QP units when AV1's offsets are q_index, roughly
four per QP step — not the measurement. Recording why a bound was revised matters
as much as recording the bound.

**Weight symlinks go stale silently.** Seven links under `assets/weights/` were
dangling, including every default the config names, because models moved into a
subdirectory. Ultralytics quietly auto-downloads replacements into the repo root
when this happens. A check that every weight a config names actually resolves
belongs in the behaviour suite.

**Never add a test to raise a coverage number.** The old suite carries ~2,100
lines in files named `*_coverage*`. The percentage gate that motivated them is
replaced by a required-behaviour suite — a named list of properties that must
hold, which padding cannot satisfy.


---

## 4. Architecture

### 4.1 Layers and dependency direction

```
  ┌─ contracts/ ─────── pure, dependency-free, THE SHARED CONTEXT ───────┐
  │  schemas · per-axis protocols · capability vocabulary · registry     │
  │  config schema · StageLattice · DomainProfile · KeypointSchema       │
  │  ObjectStream · AppearanceRepresentation · MotionRepresentation      │
  │  TemporalPolicy · MetricSpec · CodecCapabilities                     │
  │  ConditioningPlan · ReconstructionPlan                               │
  │  no torch, no cv2, no ffmpeg — importable for validation alone       │
  └──────────────────────────────────────────────────────────────────────┘
                                ▲ implemented by
  ┌─ components/ ──── one package per axis, each with a registry table ──┐
  │  domain · scene · temporal · detection · selection · tracking        │
  │  appearance · motion · pose · segmentation · rigid                   │
  │  background · generation · residual · codec · transport · metrics    │
  └──────────────────────────────────────────────────────────────────────┘
                                ▲ composed by
  ┌─ pipeline/ ──── knows contracts, never which backend was chosen ─────┐
  │  stage DAG · encoder · decoder · reconstruction                      │
  └──────────────────────────────────────────────────────────────────────┘
                                ▲ driven by
  ┌─ runner/ ──── ONE run path ──────────────────────────────────────────┐
  │  chunk loop · scene routing · accounting · evaluation · invariants   │
  └──────────────────────────────────────────────────────────────────────┘
                                ▲ consumed as a library by
  ┌─ experiments/ ── matrices · sweeps · campaigns · dataset tooling ────┘
```

**Dependencies point inward, always.** Today `scripts/benchmark_matrix.py` shells
out to `python src/main.py` and scrapes the summary from stdout, and
`scripts/benchmark_player_backends.py` passes CLI flags the CLI stopped accepting
— it would exit(2).

This rule is **enforced by code, not by prose**: the layer list lives in
`contracts/`, and a CI check walks the import graph and fails on any outward
import. No document can go stale relative to it, because no document is the
authority (§5).

### 4.2 The ablation lattice — every component is optional

This is the organizing principle, and it is what makes the platform a research
instrument rather than just tidy code.

**Every stage can be switched off, and the residual absorbs whatever the disabled
stages would have handled.** Turn off player detection but keep the background
model: metadata shrinks, encode time drops, and the residual grows to carry the
players. Turn off the background model too: the residual grows again. Turn
everything off — residual included — and there is nothing left but the source
video. The Whole-Frame Residual Baseline is not a special comparison mode; it is
simply a corner of the lattice.

**The residual is one component like any other.** It happens to be the most
consequential switch, because it is the only one whose job is to correct everyone
else's errors — but it is not architecturally privileged.

That gives, for free:

- **Component ablations in one uniform currency.** Does racket tracking pay for
  itself? Run it on and off; the change in total payload *is* the answer,
  measured identically for every component.
- **Like-for-like comparison of alternative encodings of the same thing.**
- **A continuous compute/bandwidth/quality surface** rather than a few hand-built
  tiers.

Architecturally: no stage may be structurally required. Each declares what it
produces and consumes; the DAG is built from the enabled set; anything a disabled
stage would have contributed simply isn't predicted, so it lands in the residual
if there is one and in the error if there isn't. **Graceful degradation to the
baseline codec is a property of the architecture, not a routing special case.**

### 4.3 Quality is always measured

There is no configuration in which correctness can be assumed rather than
measured, for two independent reasons:

1. **The residual always carries some coarseness.** A truly lossless residual is
   possible but heavy and slow, so every practical configuration quantizes.
2. **Generative inference is statistical.** Even with identical inputs, seeds and
   code, encoder-side and client-side generation are not guaranteed to produce
   identical pixels. Determinism can be pursued; it cannot be assumed.

So **quality measurement is mandatory in every configuration** — an architectural
requirement, not an optional evaluation step. Every run reports quality, always.
Three consequences form the paper's evaluation spine:

- **With the residual absent**, quality measurement is the *only* way to know how
  much each like-for-like alternative degraded, and therefore which encoding of a
  salient object or background is actually better.
- **With the residual present**, sweeping its coarseness gives the rate/quality
  curve the paper needs.
- **With a fine residual, when client and server really did generate the same
  thing**, quality should come out near-flawless. *That measurement is the proof
  the architecture works* — it is the claim, not an assumption behind it.

Symmetry between encoder and decoder is therefore a **design goal verified by
measurement**, not a guarantee asserted by construction. Still worth engineering
hard: one `ReconstructionPlan` derived once and handed to both sides, rather than
each side building its own from config and hoping they agree (today's design, and
already the source of one real bug). Deterministic stages — panorama warping,
residual arithmetic — can and should be bit-identity tested. Generative stages get
closeness measurement instead.

**A lossless residual stays in the plan as a ceiling calibration**, run once or
twice rather than as an operating point: it establishes the upper bound of what
the architecture can achieve, which makes every coarser point interpretable.

### 4.4 The object stream — the central abstraction

This unifies what the current code treats as unrelated special cases. **Every
salient object is described by three independent choices:**

1. **An appearance representation** — what it looks like, established once.
2. **A motion representation** — how that appearance evolves.
3. **A temporal policy** — how densely that motion is actually sent, versus
   interpolated.

All three raise or lower object metadata, they compose freely, and each is a
lattice axis with its own variants (specified in §6.3). The client needs a decoder
capable of handling the particular combination.

The insight that makes this powerful: **a skeleton is a motion representation.**
Sending per-frame keypoints is functionally the same act as sending motion vectors
— both describe how an established appearance should be transformed. So keypoints
are not privileged, and objects without a skeleton are not a separate problem
needing a separate design. Symmetrically, a compressed image is only one way to
carry appearance; a JPEG is itself an appearance vector, just a two-dimensional one
encoded with a DCT.

**The pairing constraint.** Not every combination is decodable. Structure-only
representations such as Canny edges are a tempting extra appearance option, but
they discard colour and texture entirely, so nothing can recover appearance from
them — Canny belongs on the conditioning side, never as the appearance carrier.
More generally, a representation is usable only if some registered generator can
consume it.

So each generator **declares which appearance representations and which motion
representations it accepts**, and config validation rejects an unmatched
combination with a message naming what would work. This is the capability system
doing real work: it keeps the axes genuinely independent where they are, and makes
the places they are coupled explicit and checked rather than discovered at
runtime. It is also what stops this design becoming a combinatorial rabbit hole —
the implemented set is whatever has a decoder, and everything else is named as
future work rather than half-built.

### 4.5 Domain versus components

**A `DomainProfile` declares the semantics** — what is being modelled: which
object classes are salient (tennis: players, racket, ball; general: whichever
humans are present), which `KeypointSchema` applies where keypoints apply at all,
the camera-motion assumption, and what scene classification means here.

**Components are interchangeable implementations that satisfy those needs.** In
any human domain we care about people — but whether YOLO26, SAM3 or RF-DETR
extracts them is an independent axis. The domain says *what*; components say
*how*; config picks both separately.

**Camera-motion assumption.** This is not about camera hardware. It is about
whether the background can be modelled as a single warpable plane. A tennis
broadcast camera is near-static with pan/tilt/zoom, so successive frames relate by
a homography and a panorama background is valid. Freely-moving handheld footage
has parallax, so no single homography exists and a panorama background is
*invalid* — it will produce garbage, quietly. The profile declares which regime it
is in, so the background component either applies or is disabled rather than
silently producing a bad panorama. This matters immediately: DAVIS clips are
largely handheld.

**Two profiles: tennis and general.** General is evaluated on **the DAVIS clips
containing humans**, the direct answer to the most-requested reviewer item
(generalizability, raised by four of five referees). It also opens a second
experiment costing almost nothing extra: **pretrained human-generation models will
likely do reasonably on DAVIS and struggle on tennis**, so measuring
pretrained-versus-fine-tuned across both domains shows exactly how much
domain-specific fine-tuning buys. Football is a later decision (§11).

### 4.6 Two rules the contracts enforce

1. **Cross-axis effects are derived, never string-matched.** Today
   `genai_backend` containing `"canny-controlnet"` reaches into an unrelated
   module and nulls the pose estimator. Instead a generator *declares* what it
   consumes, and the encoder derives which stages to enable and what to transmit.
2. **Constraints are checked data, not prose.** `libsvtav1` silently ignores the
   pixel-format knob and always emits yuv420p — documented in comments, enforced
   nowhere, so every residual encode since `residual_pix_fmt` was added has
   silently requested a format it never got.

---

## 5. Documents, truth, and who reads what

**Three documents, each with one job, and no fourth.**

| Document | Job | Where |
|---|---|---|
| **This plan** | Status, architecture, component specs, phases, priorities | `PLAN.md` |
| **`src/contracts/`** | The machine-checkable truth | `src/contracts/` |
| **The paper** | The conceptual record: findings in its sections, secondary findings in its appendices | `67a9ea6275d3d9785ce57026/` (separate git repo) |

Structural facts that could go stale in prose are not written in prose at all:
the layer list lives in `contracts/` and is enforced by the import-direction
check (§4.1), and each module is described by its own docstring and registry
entry, next to the code they describe.

**Plan and paper are updated together**, as ideas land, get implemented, and
produce results. A finding goes to the paper; a change in what is done or next
goes to §2 here. Adding a fourth document to track status was tried and reverted
— it duplicated this one, which is exactly the drift these three exist to
prevent.

**`contracts/` — the machine-checkable truth.** Protocols, schemas, capability
declarations, config validation, the layer list. If code and prose disagree, this
wins, because CI runs it.

**The paper — the conceptual record and the long-term trace.** This plan
carries a lot of original design that exists nowhere else, and implementation is
exactly the phase where such ideas get lost. So the concept-bearing sections are
written before the results exist.

This is not a new process: the manuscript already has a marker convention built
for it (`STATUS` / `GOAL` / `HOLE` / `NOTE` / `NEXT` / `CLAIM(id): src=`), which
exists precisely so a section can record intent, missing data, and provenance
without any of it reading as a claim. Used as designed, the paper becomes a living
trace of what has been done and what is left.

Which sections: **Introduction, System Design / Method, and Future Work** carry
concepts and can be written without results. Every mechanism that is designed but
unproven gets a `NOTE()` or `HOLE()` marker; a `CLAIM(id): src=` line appears only
once a real `outputs/` path backs it. **Evaluation and Conclusion** are written
from results, last.

There is a payoff beyond not losing ideas: **the contracts package and the Method
section describe the same design twice, in two languages.** Written together, each
becomes a check on the other — a mechanism too vague to write down in prose is
usually too vague to have specified correctly in code, and the reverse.

### Appendices — where secondary findings go

**Every appendix is its own `.tex` file under `appendices/`, `\input{}` from
`main.tex`.** Nobody has to read them to read the paper, and nothing has to be
thrown away to keep the paper focused.

This is the release valve that keeps all three documents clean. Work generates
findings that are real and worth preserving but are not the contribution: a
negative result, a survey of what an encoder does and does not support, a
reproducibility note about a version that behaves differently. Left in the plan
they bog down every agent that reads it; left in the main text they dilute the
argument; deleted, they get rediscovered the hard way. An appendix takes them,
and the main text refers to it in a sentence.

Working rule: **when a result is interesting but not load-bearing, write the
sentence in the main text and the substance in an appendix.** There is no budget
on how many appendices or how long they run.

**Keeping the paper honest is a standing obligation, not a one-time cleanup.** No
unmeasured quantitative claim may appear anywhere; every mechanism that is
designed but unproven carries a `NOTE()` or `HOLE()`; a `CLAIM(id): src=` line
appears only when a real `outputs/` path backs it. When a claim is downgraded by
new evidence, the downgrade is written where the claim was, not appended
somewhere else.

**And this plan (`PLAN.md`) — the operational document.** Phases, workstreams, delivery
order, status. Diffable at a glance, editable by the harness, and what subagent
sessions are dispatched from. Source of truth for *what we are doing next*, and
deliberately **not** a place for results, related work, or narrative — those have
homes above, and a plan that accumulates them stops being usable by the agents it
exists to drive.

### What each subagent session reads

| Document | Role for a component session |
|---|---|
| This plan, §4 and its own §6 section | **Required.** Cross-cutting architecture, plus its own specification. |
| `contracts/` for its axis | **Required.** The interface it must satisfy. |
| The paper's Method / System Design section | **Required as design reference** — the same design in prose, and often clearer about *why*. |
| The paper's Evaluation section | **Not read.** Results-dependent and irrelevant to building a component. |

Sessions **write back** to the paper: when an implementation diverges from what the
Method section describes, the session updates the markers in the same pass rather
than leaving prose and code to drift. That drift is the one real risk of writing
method text ahead of implementation; the mitigation is this rule plus a standing
re-read of the Method section at every phase gate.

---

## 6. The component catalogue

Every component is optional unless marked otherwise, per §4.2. "When off" states
where the work goes instead — which is what makes the lattice measurable.

| # | Component | Variants | When off | Spec |
|---|---|---|---|---|
| 1 | **Scene classification** | HSV-histogram + point-anchored motion; none | whole input is one span; no semantic-vs-fallback routing | §6.2 |
| 2 | **Subject detection** | YOLO26, SAM3, RF-DETR | no subjects found; they land in the residual | §6.2 |
| 3 | **Subject selection** | open-vocabulary class prompt, ad-hoc heuristic, all-detections | every detection treated as salient, spectators included | §6.2 |
| 4 | **Tracking / identity** | tracker + recovery policy, per-frame only | no cross-frame identity, so no appearance reuse | §6.2 |
| 5 | **Appearance representation** | compressed image (JPEG quality *q* / downscale *s*), diffusion latent, image embedding | generator has no appearance cue | §6.3 |
| 6 | **Motion representation** | keypoints, motion vectors, encoded video (B/P frames) | object static after appearance is established; motion lands in the residual | §6.3 |
| 7 | **Temporal policy** | metadata sparsity at threshold *t*; generation sparsity; pipeline sparsity; none | every frame fully processed — maximum metadata and compute, minimum prediction error | §6.3 |
| 8 | **Pose estimation** | DWPose, YOLO-pose, none | no keypoints; motion representation must be vectors or video | §6.2 |
| 9 | **Segmentation** | YOLO-seg, SAM3, none | compositing falls back to heuristic masks | §6.2 |
| 10 | **Rigid objects** | per-class strategies: racket convex-hull + wrist anchoring, ball difference-based or segmentation-based, none | rigid objects land in the residual | — |
| 11 | **Background model** | panorama-full, panorama-delta, none; sidecar codec jpeg/png/roi-video | background lands in the residual | — |
| 12 | **Generation** | ControlNet variants, pix2pix, SPADE4Tennis, Animate-Anyone, **MOFA-Video** (trajectory-conditioned), upscale-refine, none | subjects land in the residual | §6.4 |
| 13 | **Residual** | lossy at coarseness *c*; none; lossless as ceiling calibration | nothing corrects generation error; quality rests entirely on generation | §4.3 |
| 14 | **Codec** | AVC, HEVC, AV1, VVC × {ROI, no-ROI} | required whenever any video stream is transmitted | §6.1 |
| 15 | **Transport** | disk (payload serialization split from transport medium) | required | — |
| 16 | **Metrics** | PSNR, SSIM, VMAF, LPIPS, FVMD | **never fully off** — at least PSNR always runs | §6.5 |

The all-off corner of rows 1–13 leaves the source video, which is the baseline.

### 6.1 Codec (row 14) — the ladder and ROI

| Rung | Driver | Invocation |
|---|---|---|
| **AVC** | ffmpeg | `libx264` — the speed rung, what makes a real-time target reachable |
| **HEVC** | binary | `kvazaar` — speed/quality trade-off, and the proven ROI comparator |
| **AV1** | binary | `SvtAv1EncApp` — quality |
| **VVC** | ffmpeg | `libvvenc` — quality, and the anchor reviewers asked for |

One HEVC library only. Two rungs are driven as **standalone binaries rather than
ffmpeg sub-encoders**, because that is the only way to reach their
region-of-interest surfaces — so driving binaries is a structural requirement of
this axis, not an implementation detail.

Capabilities are declared, checked data — supported pixel formats, preset scale,
rate control, ROI mechanism, losslessness — so an illegal combination raises at
config validation instead of being silently substituted. Already implemented in
`src/contracts/codecs.py`; a component stream extends it with command building.

**ROI is a first-class requirement**, because the evaluation must show this system
beating baselines *that are themselves allowed region control*. Current state:
AV1 and HEVC expose real delta-QP maps, VVC exposes none and will need the
in-house pixel-domain arm, and ffmpeg's `addroi` path is unverified. Detail and
measurements live in the paper's ROI appendix; `experiments/verify_codec_roi.py`
is the harness.

**Hard rule, enforced by `codecs.assert_matched_rate_control`:** an ROI arm and
its baseline must use identical rate control at matched rate. A prior project
compared fixed-QP ROI arms against target-bitrate baselines on an encoder that
overshoots by 30–45%, producing a "24.9–40.2% saving" that was entirely the
overshoot. Our own first ROI measurement fell into the softer version of the same
trap — matched QP, so the ROI arm simply spent more bytes — which is why matched
*bitrate* comparison is the outstanding work on this axis.

**In-house ROI** — degrading non-salient blocks in the pixel domain before
encoding — gives every rung an ROI arm regardless of encoder support, and is
required for VVC. Prior art in `/home/itec/emanuele/presley`
(`src/presley/components/roi.py`, `encode_utils.py`).

### 6.2 Perception — scene, detection, selection, tracking, pose, segmentation (rows 1–4, 8, 9)

Registry replaces substring matching. Track-recovery logic currently living on
`Yolo26Detector` and inherited by `YoloEDetector` gets decomposed into a composed
policy, so a non-YOLO backend can reuse it.

Backends: **YOLO26** (default *and* fallback comparator — ubiquitous, with
genuinely fast variants, which matters for the real-time rung), **SAM3**
(`assets/weights/sam3.pt` is on disk; supersedes the SAM2 the reviewer suggested),
and **RF-DETR**.

**Subject selection (row 3) is itself an experiment.** Open-vocabulary detectors
like YOLOE and SAM3 accept a class prompt directly — asking for "tennis player"
may be enough on its own, with no bespoke selection code. But not every detector
supports open vocabulary, and it is unknown how good the ones that do actually
are. So both paths stay: an open-vocabulary prompt variant and an ad-hoc heuristic
variant (the current `heuristics.py` logic, which separates players from ball kids
and crowd). Having both gives a real comparison — *does built-in open-vocabulary
detection match hand-written domain heuristics?*

**Scene classification (row 1) — scoped deliberately.** Its only current use is
routing: point scenes to the semantic pipeline, interludes straight to the
fallback codec. That is a real use and it answers a reviewer question, but it is
not obviously worth more investment than it already has, and for short DAVIS clips
it may be irrelevant. Keep it as an optional lattice row, measure whether routing
pays, and do not build more on it without a reason.

**Environment:** keep one conda env for the whole project. RF-DETR is not
currently installed; attempt it in the existing `pointstream` env first, and only
if it genuinely cannot coexist with the pinned versions, split into the smallest
possible set of envs and document exactly why.

#### Keypoint schema

**Decided and implemented** in `src/contracts/keypoints.py`: the canonical
internal schema for humans is COCO-WholeBody-133, with name-based projection to
and from COCO-17, OpenPose-18 and AP-10K-17, and every keypoint carrying a
present flag so partial schemas are first-class rather than zero-filled.

The one thing a component stream must respect: **the canonical schema is internal,
not what goes on the wire.** What is transmitted is derived from what the chosen
generator consumes. Sending 133 joints to a conditioner that reads 18 is wasted
payload, and payload is the ranking currency.

Rationale, the schema landscape, and how richness trades against rate live in the
paper's Method section.


### 6.3 The object stream — appearance, motion, temporal policy (rows 5–7)

Implements the abstraction in §4.4.

#### Appearance representations (row 5)

| Representation | What is sent | Decoder needed |
|---|---|---|
| **Compressed image** | the object crop as JPEG at quality *q*, or downscaled by factor *s* | any image-conditioned generator — img2img ControlNet, pix2pix, Animate-Anyone's reference path |
| **Diffusion latent** | the crop encoded to the generator's own VAE latent space | a diffusion generator, which consumes latents natively — compact, no format mismatch |
| **Image embedding** | a CLIP/IP-Adapter-style appearance vector, a few KB | IP-Adapter-conditioned diffusion — a strategy that already exists in this codebase |

The two degradation knobs on the compressed-image route — **JPEG quality** versus
**downscaling** — are not equivalent: one discards high-frequency detail through
quantization, the other through resolution. Which serves generative reconstruction
better is an open and cheap question.

#### Motion representations (row 6)

| Representation | What is sent | Applies to |
|---|---|---|
| **Keypoints / skeleton** | per-frame pose vector | objects with a stable skeleton — humans, animals |
| **Sparse trajectories** | a handful to ~100 tracked points per clip, expanded to dense motion by the decoder | *any* object, no skeleton required |
| **Encoded video (B/P frames)** | the object crop as a literal video after its keyframe | any object — the classical codec answer, applied per object |

For humans, **all three apply**, which gives a clean controlled comparison of the
paper's core idea against both a generic alternative and the classical baseline,
on identical objects with identical appearance.

**Sparse trajectories, not dense flow** — settled by the literature check. The
flow-conditioned animation literature splits into models consuming *dense*
per-pixel flow (Motion-I2V, OnlyFlow, FloVD) and models consuming *sparse*
trajectories which they expand to dense motion internally (MOFA-Video, DragNUWA,
Tora, Image Conductor, ATI). Dense flow is no cheaper to transmit than classical
block motion vectors, which would defeat the purpose. Sparse trajectories are
**structurally the same size as a skeleton** — for scale, the generative
face-coding literature transmits 16–100 motion parameters per frame against our
17–133 keypoints — and the sparse-to-dense expansion is exactly the generative
decoder's job.

**MOFA-Video is a candidate generator, not a commitment.** It goes into the
registry as one more optional entry in row 12, and it earns its place on the same
terms as everything else: does it shrink the payload by more than it costs, and
what does it do to quality. It is the leading candidate because it is ECCV 2024
with public checkpoints, is SVD-based with adapter-only training, and consumes
exactly the sparse-trajectory signal we would want to transmit — but if it loses
to a ControlNet variant or to plain upscale-refine, it loses, and the lattice will
say so. DragNUWA and ObjCtrl-2.5D are alternative entries on the same axis; Tora's
weights are gated under a non-commercial-leaning licence. Expect SVD-class cost,
roughly 4–7 s per frame on an A100 — decode-side only and nowhere near real time,
which the compute axis of the lattice will report like any other cost.

**The classical baseline has a name to cite:** MPEG-4 Part 2 object/sprite coding
(Video Object Planes, background sprite coded once, foreground objects as their
own streams) is the direct ancestor of the encoded-video representation. Modern
ROI neural codecs evaluated on DAVIS are its learned counterpart.

### Related work

**Lives in the paper**, not here — see its Related Work section. It covers
generative face video coding, the trajectory- and flow-conditioned animation
literature, and object-based coding precedent, and states what is distinct
about this work.

### 6.4 Generation (row 12)

The worst defect in the current code. `BaseGenAIStrategy.generate` takes
`dense_dwpose_tensor`, which carries a **pose** for some backends, a **binary mask
or canny image** for others, and a **`(pose, mask)` tuple** for multi-controlnet —
with the compositor string-matching the backend name to decide which.
`controlnet_engine.py:590-604` contains in-code comments admitting the confusion.
Temporal capability is detected via `isinstance(strategy, AnimateAnyoneStrategy)`,
so any new temporal backend must subclass Animate-Anyone to be recognized.

Replaced by a typed conditioning bundle with separately-typed optional fields and
declared capabilities — including the appearance and motion representations each
generator accepts (§4.4). ~10 strategies rebuilt. The ~40-line pose-rescale block
copy-pasted across four ControlNet classes gets deduplicated and corrected.

Two new variants complete the §6.3 comparison. **MOFA-Video** is the
trajectory-conditioned entry — appearance plus sparse trajectories → object in
motion — registered like any other backend, declaring
`appearance:compressed-image` and `motion:sparse-trajectories`, and judged on the
same terms as everything else: does it shrink the payload by more than it costs,
and what does it do to quality. It is the leading candidate, not a commitment; if
it loses to a ControlNet variant the lattice will say so. **Upscale-refine** is
the other: no diffusion, just upsampling and refinement of a low-resolution
appearance, and the cheap baseline every generative model has to beat.

**Animate-Anyone** joins as a first-class variant, with a caveat that travels with
every number it produces: **its fine-tuned checkpoint was trained on scenes from a
single tennis match, not on general tennis players.** It has seen one match's
appearance distribution, so it is not a fair general model and any score it posts
is scoped accordingly. It also cannot enter the training campaign as things stand
— no variant branch, no checkpoint-path handling, and critically no entry in
`eval_checkpoint.py`'s `ARCH_CHOICES`, so the campaign could not score it even if
it trained; its trainer lives in an external vendored package rather than this
repo. **Make it evaluable and score the existing fine-tune first** (cheap); commit
to a stage-1 + stage-2 retrain only if that justifies the GPU days.

### 6.5 Metrics (row 16)

Pluggable and tiered, so development stays fast and expensive metrics appear only
where they earn their place. **At least PSNR always runs** (§4.3).

| Tier | Metric | When |
|---|---|---|
| Fast | **PSNR** | always on; the default during development and tests |
| Traditional | **VMAF**, SSIM | video-quality scoring for headline tables |
| Perceptual | **LPIPS** | generated content, where PSNR misleads |
| Temporal | **FVMD** | temporal-coherence claims (reviewer theme 2) |

**FVMD rather than FVD** — Fréchet Video Motion Distance is the better fit for the
temporal-coherence question reviewers actually asked about. The existing FVD
wiring is prior art to read, not to keep by default.

LPIPS exists in the codebase but is wired only into checkpoint evaluation, never
into pipeline evaluation — a gap mis-reported as closed twice.

The perceptual and temporal tiers together replace the MOS study; the user study
becomes stated future work.

---

## 7. Tests — trim hard, change what the gate measures

Measured: **72 files, 10,766 lines, 436 test functions — and 2,112 lines of that
(about a fifth) sit in files named `*_coverage*`**, written to move the gate number
rather than assert behaviour. A test that exists only to raise a coverage number is
a defect, because it makes the gate lie. Only 10 files carry
`integration`/`slow`/`invariants` markers; 25 are mock-heavy enough that they
substantially test their own mocks.

Given the two real purposes — catch bugs while developing, flag weird results
during experiments:

- **Delete the coverage-padding tier** (~2,100 lines).
- **Drop the percentage coverage gate.** On a single-maintainer research codebase
  it incentivizes exactly the padding being deleted. Replace with a
  **required-behaviour suite**: bit-identity for deterministic stages, every
  lattice corner produces a runnable pipeline, config rejects unknown keys, codec
  constraint violations raise, an undecodable appearance/motion pair is rejected,
  no layer imports outward, every registered backend constructs, every domain
  profile round-trips, every run emits at least one quality metric. A gate that
  cannot be satisfied by padding.
- **Keep and extend the `invariants` suite.** It audits `outputs/` for citability
  rather than testing code — purpose #2 exactly, and the most valuable thing in
  the current suite.
- **Rewrite the rest alongside their modules**, at roughly a third the volume.

Expected: ~10.7k lines → ~4k, with materially better signal.

---

## 8. Dataset sequencing

Deliberately *not* fixing the whole dataset first — the cost is uncertain and it
is not yet on the critical path.

1. **Minimal dataset first.** Just enough correctly-prepared material to exercise
   the pipeline realistically end to end: a handful of tennis tracks, plus a few
   DAVIS human clips for the general profile.
2. **Stand up the pipeline and run it over that**, confirming every lattice corner
   behaves as expected and the numbers are plausible.
3. **Only then repair the full dataset** — regenerate `_skeleton` with absolute
   filenames (verified still broken: `djokovic_federer/scene_009/track_0076` has
   colour frames 151→509 against skeletons 000→355, so ControlNet-pose training
   pairs are 33% wrong and 23% missing), and raise `--max-scenes` above 10 to lift
   deep-annotation coverage past its current 15%. Measure the cost on one video
   before committing to all seven.

Where each step lands in the schedule is §7.

---

## 9. Build phases

Phases A–D build **all** the code. What gets *tested and written up*, and in what
order, is §8 — the two are deliberately separate, because code is fast and
experiments are slow.

### Prerequisites — done

Working tree committed, CI restored (lint, typecheck, behaviour suite,
import-direction check), literature review complete and folded into the paper.


### Phase A — Contracts and concepts — **done**

`src/contracts/` is complete and green: errors, registry, capability vocabulary,
layers and their import check, keypoint schemas, codec ladder, conditioning,
metrics, object stream, domain profiles, stage lattice, strict parsing, and the
run configuration. 190 tests; ruff and mypy clean; **the package imports nothing
heavy**, so a whole configuration validates on a machine with no torch, cv2 or
ffmpeg installed.

The paper purge is done: unsupported abstract claim removed with a `HOLE` naming
the experiment that would restore it, the rejected submission archived behind a
README, `RESEARCH_LOG.md` split into Standing and History.

Two things Phase B inherits rather than rediscovers:

- **Per-axis registry modules are pre-created empty**, so parallel streams fill
  their own file without touching a shared table.
- **Config validation has a third pass waiting**: `config.validate_backends`
  takes registries as arguments and checks that every named backend exists and
  that the chosen appearance/motion pairing has a generator able to decode it. It
  is a no-op until registries are populated — wiring it is each stream's job.


### Phase B — Components (Aug 25–Sep 5, parallel worktrees)

| Stream | Rows | Notes |
|---|---|---|
| B1 | 14 | codec ladder, capabilities, ROI arms incl. in-house, matched-rate rule |
| B2 | 1–4, 8, 9 | registry, recovery decomposition, YOLO26 / SAM3 / RF-DETR, open-vocab vs heuristic selection, keypoint schema |
| B3 | 12, 5, 6 | conditioning bundle, ~10 strategies, motion-vector animator, upscale-refine, appearance representations, Animate-Anyone evaluable |
| B4 | 11, 10 | background, rigid objects — as optional lattice rows |
| B5 | 15, 7 | transport serialization/medium split; temporal policy incl. pipeline sparsity |
| B6 | 16 | tiered metric registry, LPIPS into pipeline eval, FVMD |
| B7 | domain | tennis and general/DAVIS profiles |

B3 is the largest; start it first with the strongest agent.

### Phase C — Pipeline and runner (Sep 5–12, parallel)

- **C1 Reconstruction + residual (row 13).** Decompose `SynthesisEngine` (512
  lines doing panorama resolution, pose unrolling, ball rendering, GenAI dispatch,
  CUDA determinism and OOM fallback) and `ResidualCalculator` (967 lines).
  Implement the residual coarseness spectrum including absent and the lossless
  ceiling. **Deliverable: bit-identity tests for deterministic stages**, plus
  encoder-vs-decoder closeness measurement for generative ones (§4.3).
- **C2 Encoder pipeline.** DAG built from the enabled stage set; the stage skips
  that make a reduced lattice corner genuinely cheaper, not just nominally.
- **C3 Runner.** One run path — single-chunk becomes the degenerate case of
  full-match, since the latter already owns more structure and the reverse would
  reload model weights per sub-chunk. One accounting implementation instead of
  two. Quality evaluation mandatory on every path. Invariants working across every
  lattice corner, plus a quality-coverage invariant.

Dataset step 2 (§8) is the Phase C exit gate.

### Phase D — Experiments layer (Sep 12–16, parallel)

Rebuilt to consume `runner/` as a library. Lattice-driven ablation matrices; codec
sweep with the PointStream arm actually *invoked* rather than read from a
pre-existing summary; training campaign including Animate-Anyone. Dataset step 3
(§8) runs here, before the experiments that need it.

---

## 10. Delivery order — what gets run and written, and when

From Sep 16 the work is experiments and paper, ordered strictly by this list.
Nothing below P0 blocks a submission. Jobs run detached with hourly checkpointing.
**GPUs are assumed available, and more than one server can be used — including
several at once**, which the sweeps parallelize across cleanly. Each result gets a
plausible best/worst bound written down *before* the number is read.

### P0 — without these there is no paper

1. **Quality measurement working at all.** The tier ladder producing real
   PSNR/SSIM/VMAF/LPIPS numbers. Everything else depends on it.
2. **PointStream against the codec ladder, including ROI arms.** The core claim.
   Verified never run: all three sweeps in `outputs/codec_baselines/` have
   `pointstream_point: null`, and grep finds zero traces anywhere under
   `outputs/`. Without ROI arms a reviewer calls it a strawman.
3. **The residual-coarseness curve**, absent through fine, plus the lossless
   ceiling — the rate/quality story.
4. **The core ablation lattice** — does the semantic path pay for itself, per
   component, in one uniform currency.
5. **A working generative engine, or an honest scoped negative result.**
6. **Generalization on the general/DAVIS profile.** Raised by four of five
   reviewers on a rejected paper; effectively mandatory for resubmission.
7. **Evaluation and Conclusion sections**, and the abstract reconciled with what
   was actually measured. (Introduction, Method and Future Work already landed in
   Phase A.)

### P1 — strongly strengthens; addresses named reviewer themes

8. Perceptual and temporal metrics (LPIPS, FVMD) as the MOS replacement.
9. **Object-representation comparison** — keypoints versus motion vectors versus
   encoded video, on identical objects with identical appearance. The most novel
   idea in the plan, and the one most likely to distinguish the resubmission from
   the rejected version.
10. Detector comparison including SAM3 (reviewer theme 7).
11. Temporal-policy ablation — the row-7 axis, never measured.
12. Scene-classification and shadow write-ups — text only, no experiment needed.
13. Encode-time comparison against VVC.

### P2 — run only if time remains

14. Appearance-representation comparison: compressed image versus diffusion latent
    versus image embedding.
15. Keypoint-schema richness ablation (start with the free degradation
    measurement).
16. Appearance degradation: JPEG quality versus downscaling.
17. Open-vocabulary versus hand-written subject selection.
18. Animate-Anyone full retrain.
19. Background-layer ladder verdict.
20. Football as a third domain.

### Explicitly out of scope, named as future work in the paper

- MOS user study.
- Variable-keypoint training-regime robustness across architectures (§6.2).
- Appearance and motion representations for which no decoder exists yet.

---

## 11. Checkpoint — around August 27

One week in, reassess against actual progress: how much of Phases A–B landed, how
far down the §10 list the experiment plan realistically reaches, whether football is
worth adding, and whether the full paper or a narrowed one is the right target.
Decide then, with data rather than estimates.

---

## 12. Verification

Per stream before merge: `ruff` and `mypy` clean; required-behaviour suite passes;
tests cover plausible misuse (unknown backend, unsupported codec/pix-fmt pair, a
generator declaring conditioning nothing supplies, an appearance/motion pair no
generator can decode, a domain profile missing a keypoint schema, a panorama
background requested under a parallax camera-motion assumption, a temporal policy
asked to interpolate across a scene cut, a motion representation requiring
keypoints on an object class that has no skeleton).

Phase gates — each a real check that fails today:

- **After A:** config validation runs without importing torch; a typo'd config key
  raises and names the closest legal key; the import-direction check passes; the
  Method section and the contracts describe the same mechanisms; no retracted
  claim remains reachable as current.
- **After B:** `libsvtav1` + `yuv444p` raises instead of silently emitting
  yuv420p; every codec rung has a working ROI arm or a documented reason it
  cannot; one command lists every registered backend on every axis.
- **After C — the decisive gate:** `config/tier_fast.yaml` and
  `config/tier_quality.yaml` each produce a run summary carrying **real PSNR /
  SSIM / VMAF / LPIPS numbers**; a residual-absent run completes and reports its
  measured quality drop; the all-off lattice corner reduces to the source video;
  deterministic-stage bit-identity passes. None of these work today in any
  configuration.
- **After D:** a codec sweep invokes PointStream itself and emits one
  rate-quality-time table containing every arm.
- **During §10:** every cited run has empty `invariant_failures`, and
  `pytest -m invariants` passes against the live `outputs/` tree.
