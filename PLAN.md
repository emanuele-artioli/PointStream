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
| B — components | ✅ **done** | Merged-ready on `phase-b/integrate` (still unmerged to main) |
| **B′ — the engine roster** | Wave 3 in flight | Quality flagship **unset** until pose-ref ControlNet posts a coding-task number. Wave-2 roster (ControlNet holds both slots) is **void** — self-reconstruction. |
| C — pipeline and runner | ⬜ | `C1`/`C2` can start with Wave 2; `C3` after both |
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

### 2.1 Generation loaders — Wave 1 merged, numbers re-run on aligned pairs

The Phase-B socket is plugged. Tests that inject `_FakePipe` remain; the loaders
now load real weights when not injected. Registry entries for
`trajectory-controlnet` (alias `trajectory-render`) and `stable-animator` are
applied; the Animate-Anyone summary is 7 matches / 114 tracks, not a single
match. Driven 2026-08-22 on `cuda:0` (gpu6, both cards idle). Seed **42**.
These are triage numbers, not results.

**Comparison backbone**, re-run on the aligned probe set after the pose-offset
fix. Clip: `assets/probe_set/clips/alcaraz_perricard/scene_006/track_0196`
(one of the five that had 48 colour / 0 skeleton under filename reconstruction;
crop `frame_000641.png` pairs with skeleton `frame_000119.png`). One frame,
PSNR vs letterboxed appearance, 20 ControlNet steps, output ≠ input on every
engine. Bounds were written to `outputs/bp7-psnr-bounds.txt` before generate.
Log: `outputs/bp7-aligned-probe.log`.

| Engine | Checkpoint | PSNR vs letterboxed appearance | Notes |
|---|---|---|---|
| pose-controlnet | epoch **10** | 19.0 dB | `cuda:0`. Bounds 12–32, expect 14–28. Not identity. |
| seg-controlnet | epoch **7** | 20.1 dB | `cuda:0`. Same band. Mask is from the crop, not the skeleton. |
| ip-adapter-controlnet | epoch **10** + `h94/IP-Adapter` | 11.0 dB | `cuda:0`. Bounds worst 10 (txt2img) / best 28. Still on that floor; not identity. Not tuned. |
| pix2pix | `pix2pix_generator.pt` | 18.5 dB (CPU) | Bounds 10–30. |
| spade4tennis | `spade4tennis_lite_generator.pt` | 15.1 dB (CPU) | Bounds 10–30. Cheap blob; wired because it was cheap. |
| trajectory-render | pose-controlnet epoch **10** | 19.0 dB | `cuda:0`. Within 0.03 dB of pose; four synthetic sticks still barely move it. |

BP3's earlier numbers (pose 20.3, seg 21.7, ip-adapter 11.3, pix2pix 16.6,
spade 14.5, trajectory 20.5) were on
`alcaraz_highlights/.../track_0002`, which starts at source frame 0, so that
run never saw the offset. Deltas here are 0.3–1.9 dB on a *different clip*,
not a jump from aligning a mismatched pose. **BP3's "smeared but recognisable"
was not the filename-offset fault.** Alignment still had to land: five of
twelve probe clips had no skeleton at all, and any later ranking that used
those clips would have been measuring missing pose, not model quality.

**Quality flagship** (`phase-bp/bp4`): Animate-Anyone loads
`~/Models/AnimateAnyone/profiles/finetuned_tennis` (7 matches, 114 tracks — not
one). 3 DDIM steps melted (9.65 dB, below the 12 dB floor). 20 steps: 14.0 dB
in-set, 5.0 GiB, 32.5 s warm. StableAnimator is wrapped; SVD-XT not bundled
(§2.4). Sparse2Dense still has no public code.

### 2.3 The probe set — rebuilt 2026-08-22 on `phase-bp/bp1`

The inherited v1 view was unusable. Using it naively produced silently wrong
results rather than an error. Two faults were diagnosed; only the first
survived measurement.

**1. The manifest and the materialised view named different clips.** That is
the fault that actually fires. `manifest.json` seeded 12 clips;
`training_view/` symlinked an entirely *different* set. Zero of the 12
manifest-named tracks were in the view (driven: 17 verifier violations on the
v1 snapshot at `assets/probe_set.broken-v1/`). A harness reading the manifest
found nothing; one reading the view evaluated an unseeded, unrecorded
selection.

**2. Two naming conventions in one track group — the live fault, still open.**
Both earlier diagnoses were wrong, in opposite directions, and the truth is
worse than either. Verified 2026-08-22 by listing every track directory:

| Directory | Naming | Count |
|---|---|---|
| crop, `_canny`, `_pose_body`, `_pose_racket` | **global source frame ids** | consistent |
| `_skeleton` | **track-local, zero-based** | 50/50 sampled |

So a track that starts at source frame 29 has its crop at `frame_000029.png` and
its *pose* for that same instant at `frame_000000.png`. **50 of 114 tracks (44%)
carry this offset**; the other 64 align only because they happen to start at
source frame 0. Frame counts always match — it is a pure re-indexing.

The v1 measurement that found "5 of 12 clips missing every frame" was measuring
`_skeleton` and was **correct for the pose channel**. The re-measurement that
found "0/12 missing" was measuring the colour crop and was **correct for that
channel**. Neither noticed the two disagree.

**This survived into the first v2 tree and is now closed.** `materialize.py`
copied the crop strictly (raising on a missing frame) but copied every
conditioning directory with the *same* global `source_id` under
`if src.is_file()` — silently skipping what it cannot find. Result, verified
on the snapshot at `assets/probe_set.broken-v2-unaligned/`:

| Clips | Colour frames | Skeleton frames |
|---|---|---|
| 7 of 12 | 48 | 48 |
| **5 of 12** | 48 | **0** |

The five are `alcaraz_perricard/scene_006/track_0196`,
`alcaraz_ruud/scene_004/track_0257`, `alcaraz_ruud/scene_004/track_0297`,
`federer_djokovic/scene_001/track_0071`,
`sinner_alcaraz/scene_012/track_0058` — the same five as v1.

**Fix, landed on `phase-bp/integrate`.** Every channel is resolved by the
frame's *position in the track* (sorted `frame_*.png` lists, pair by index),
never by reconstructing a filename. The verifier asserts each conditioning
directory has the same frame count as the crop — it failed on the unaligned
v2 snapshot (5 clips, `_skeleton has 0 frames, crop has 48`) and passes on
the rematerialised tree. Driven: all 12 clips have 48 colour frames **and**
48 skeleton frames. `assets/dataset` was not renamed.

The underlying `assets/dataset` was never the problem — all 12 named tracks
exist there, with crops, canny, `pose_body`, `pose_racket`, skeleton,
keypoints, captions and metadata, plus the 15 GB of 4K source in
`assets/raw_4k`.

**Rebuilt as `pointstream.probe_set.v2` on `phase-bp/bp1`, alignment-fixed on
`phase-bp/integrate`.** Track-local indexing, `global_offset` plus
`global_frame_ids` (two of twelve windows are not contiguous in source
numbering). The clips view is written first; the manifest is walked off that
tree. Same seed (`20260712`) kept the same 12 tracks; 576 colour frames and
576 skeleton frames. The verifier fails on the v1 snapshot and on the
unaligned v2 snapshot, and passes on `assets/probe_set`. Locked 5-train /
2-held-out split asserted.

Still outstanding, not this stream: `scripts/eval_checkpoint.py` still treats
`frame_ids` as dataset filename numbers when pointed at `assets/dataset`;
with a v2 manifest it would load the start of the track, not the selected
window. `scripts/select_probe_set.py` now delegates to
`python -m experiments.probe_set` and is not the regenerator.

### 2.4 Known environment limits

- **SAM3 cannot load.** `torch.nn.attention` does not exist in torch 2.2.2. This
  blocks the SAM3 detector comparison (§7 P1 item 10) unless a second env is
  built. Both the detector and segmenter entries fail construction and say so.
- **RF-DETR is not installed.** It needs `transformers>=5.1`; this env pins
  4.46.3. Registered, honest about it, not usable.
- **MOFA-Video is a candidate, not an integration.** Its SVD weights are
  Stability-AI-licensed and not bundled, so construction refuses by design. §6.2
  says what replaces it for the trajectory arm.
- **StableAnimator is the same SVD class.** Wrapper on `phase-bp/bp4`. HF card
  `FrancisRing/StableAnimator` is Apache-2.0 (checked 2026-08-22); GitHub code
  is MIT; inference needs SVD-XT (Stability AI Community / SVD research licence)
  and InsightFace `antelopev2`, neither bundled. Cannot ship as a flagship until
  SVD is cleared. Live inference was not run: leftover VRAM ~11.6 GiB, VAE decode
  wants ~16 GiB.
### 2.5 Animate-Anyone has seen the held-out videos

Verified 2026-08-22 from `assets/dataset/pointstream_aa_meta.json`. The probe set
holds out `alcaraz_highlights` and `djokovic_zverev`. Animate-Anyone's
fine-tuning set contains **both**: 20 tracks from the first, 16 from the second,
out of 114 total across 7 videos.

**So for Animate-Anyone there is currently no held-out data at all**, and any
number it posts on the held-out split is an in-training number wearing the wrong
label. This bears directly on `subsec:eval-general`, which exists to separate
what fine-tuning buys from what a pretrained backbone already delivers.

Three options, and the choice is a decision for `BP5`, not an assumption:

1. **Re-split** so the held-out videos are ones AA never saw — cheapest, but AA
   saw 7 of the 7 videos we have, so this needs new source material.
2. **Report AA as in-domain only** and let a pretrained engine carry the
   held-out arm. Honest, and costs nothing.
3. **Retrain AA** on a proper split — explicitly out of scope (§7 P2 item 17).

Option 2 is the default unless someone argues otherwise. Whatever is chosen, the
paper says which, because an unlabelled in-training score is the kind of thing a
reviewer finds.

### 2.6 Coding-task numbers (BP8) — Wave-2 roster is void

Wave 2 ranked engines on **self-reconstruction** (score against the conditioning
image). That ranking is not a coding result. Re-measured on the coding task:
appearance from track-local frame 0, pose from frame 24, score against frame 24,
12 clips, seed 42, 20 steps, independently letterboxed 512 canvases, `cuda:0`.
Pairing is by position in the sorted `frame_*.png` lists.

| Arm | Object PSNR | vs static | Path | In-domain? |
|---|---|---|---|---|
| **static copy** (paste keyframe, no model) | **11.47 dB** (this letterbox run; published shared-geometry paste was **11.82**) | floor | `outputs/bp8-coding-task/` | n/a |
| pose-controlnet (Wave-2 / published) | 11.20 dB | **−0.62** vs 11.82 | not this driver | n/a |
| seg-controlnet (published) | 11.01 dB | **−0.81** vs 11.82 | not this driver | n/a |
| Animate-Anyone, 20 DDIM, letterbox | **8.96 dB** | −2.51 vs 11.47 | ReferenceNet fed a non-blank, non-pose reference. Scheduler v-pred / trailing / zero-SNR. `outputs/bp8-coding-task/animate-anyone-letterbox.json` | **yes** — AA saw both held-out videos (§2.5) |
| real IP-Adapter + stock OpenPose | **8.90 dB** | −2.56 vs 11.47 | Not the mislabelled `ip-adapter-controlnet` dir. `outputs/bp8-coding-task-ipadapter/` | n/a |
| pose-ref step 5950 | 11.31 dB | −0.16 vs 11.47 | `outputs/bp8-coding-task-pose-ref-step5950/` | n/a |
| pose-ref epoch 1 | 11.33 dB | −0.13 vs 11.47 | `outputs/bp8-coding-task-pose-ref-epoch1/` | n/a |
| pose-ref epoch 2 | 11.25 dB | −0.21 vs 11.47 | `outputs/bp8-coding-task-pose-ref-epoch2/` | n/a |
| pose-ref epoch 3 | 11.24 dB | −0.23 vs 11.47 | `outputs/bp8-coding-task-pose-ref-epoch3/` | n/a |
| pose-ref epoch 4 | 11.24 dB | −0.23 vs 11.47 | `outputs/bp8-coding-task-pose-ref-epoch4/` | n/a |

An engine at or below the static-copy floor is **not using appearance**, in those
words. AA and IP-Adapter both lose to doing nothing. Tuning (guidance / steps /
strength) is ruled out: the models were not trained with appearance as an input,
except AA, whose ReferenceNet path was driven and still lost.

**Option A in flight — mid-train still below the floor.** Pose ControlNet is
being fine-tuned with a same-track reference painted under the skeleton
(`pose-ref-controlnet`). Smoke (driven, not read): `residual_delta=2.69` on two
ControlNet forwards that differed only in the reference; composed control dump
at `outputs/bp8-pose-ref-smoke/`. Train: `cuda:0`, init
`pose-controlnet/checkpoint-epoch-10`, 10 epochs, batch 4, dataset 35713, log
`outputs/bp8-pose-ref-train/train.log`. Coding-task evals of epoch checkpoints
ran on `cuda:1` (same protocol: seed 42, 12 clips, letterbox, bounds written
first). Bounds: `outputs/bp8-pose-ref-train/bounds.json` **before** generate.
Success is **≥ 12.82 dB** object-scoped (11.82 + 1). Below 11.82 it still is
not using appearance. Epochs 1–4 all landed in **11.24–11.33 dB** (static-copy
letterbox floor 11.47 on this driver). Epoch 3/4 are slightly worse than epoch
1. That band is evidence the reference paint is not teaching identity; epoch 10
is the number that decides the roster, not a reason to train longer in this
session.

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
is what keeps September from filling up with runs nothing cites.

**When a component has no marker to serve, that is a question, not a verdict.**
Raise it rather than silently dropping it — the paper is a work in progress and
does not yet contain every `GOAL` we will want. A component with no home may be
telling us the evaluation is missing something.

**We do not need every generator to work.** We need two flagships and a small
set of alternatives that differ along axes the paper measures.

#### 6.1 What the paper asks of the generator

| Paper slot | What it demands |
|---|---|
| `subsec:eval-ladder` | The **best** engine we have, to put PointStream's strongest RD curve against the codec ladder |
| `subsec:eval-object` | Keypoints vs sparse trajectories vs encoded video, **backbone held fixed** |
| `subsec:eval-general` | Tennis *and* DAVIS, pretrained *and* fine-tuned |
| `subsec:eval-operating` | A compute-unbounded point *and* something fast enough to have a real-time point |
| `subsec:eval-metrics` | Temporal coherence worth measuring with FVMD |

#### 6.2 Two flagships, because two questions are being asked

`eval-ladder` and `eval-object` want different things. Wave 2 assigned both
quality slots to ControlNet on **self-reconstruction** numbers. That assignment
is **void**. On the coding task (§2.6) every trained generator loses to pasting
the keyframe.

| Role | Engine | Serves | Why |
|---|---|---|---|
| **Quality flagship** | **Unset.** Pose-ref ControlNet is still training (epoch 10 decides). Epochs 1–4 did not beat static copy. | `eval-ladder` | AA 8.96 dB in-domain, real IP-Adapter 8.90 dB, published pose/seg ControlNet 11.20 / 11.01, pose-ref epochs 1–4 **11.33 / 11.25 / 11.24 / 11.24** — all at or below the 11.82 dB static-copy floor (this driver's letterbox floor 11.47). An engine at or below that floor is not using appearance. |
| **Comparison backbone** | ControlNet on SD-1.5, **pose-ref first** once epoch 10 has a coding-task number | `eval-object` | Still the only family where the backbone stays fixed while the conditioning changes. Mid-train pose-ref is in the 11.2–11.3 dB band; do not cite Wave-2 object dB. |
| Temporal / FVMD | Animate-Anyone (in-domain only) | `eval-metrics` | Only temporal engine that ran. ReferenceNet path was driven and is not the fault; 8.96 dB still loses to static copy. In-domain only (§2.5 option 2). |
| Speed rung | pix2pix | `eval-operating` | One forward pass. Not re-measured on the coding task this wave. |
| Floor | **static copy of the keyframe** | all | 11.47 dB object-scoped on independent letterbox (published 11.82 on shared-geometry paste). The non-generative control that says whether generation uses appearance. |

**Do the two flagship roles collapse?** Not yet: there is no quality flagship
until pose-ref (or a later engine) beats 11.82 dB by ~1 dB on the coding task.
If pose-ref does, the comparison backbone *is* the quality engine and the roles
share a family without sharing a row. If it does not, Option C (change what the
paper claims is transmitted) is a reported finding, not an escape.

**The trajectory arm does not need MOFA-Video.** MOFA is licence-blocked, and
routing around it improves the experiment: render sparse trajectories as a
control image into the same ControlNet backbone the keypoint arm uses. That makes
`eval-object`'s "backbone fixed" promise literally true, which a
MOFA-vs-ControlNet comparison never could.

#### 6.2.1 Which existing engines survive, and why

Wave-2 "ControlNet holds both quality slots" was decided on self-reconstruction
and is withdrawn. Survival below is about *what the paper still needs*, not
about who won a probe that scored against the conditioning image.

| Engine | Verdict | Coding-task object PSNR |
|---|---|---|
| **static copy** | **Keep — the appearance floor** | **11.47 dB** (letterbox) / **11.82 dB** (published shared-geometry). Generation that cannot beat this is not using appearance. |
| **ControlNet family** | **Keep as comparison backbone; quality flagship pending pose-ref epoch 10** | pose 11.20 / seg 11.01 (published, lose to static). `ip-adapter-controlnet` directory is a mislabelled seg ControlNet — do not load it. Real IP-Adapter 8.90 dB, **not using appearance**. Pose-ref epochs 1–4: 11.33 / 11.25 / 11.24 / 11.24 dB, all **not using appearance**. |
| **pix2pix** | **Keep** | Not re-measured this wave. Still the one-forward-pass rung. |
| **upscale-refine** | **Keep** | Not re-measured this wave. Distinct from static copy: this stretches; static copy pastes the letterboxed keyframe. |
| **Animate-Anyone** | **Keep as temporal incumbent, not quality flagship** | **8.96 dB**, 20 DDIM, in-domain. ReferenceNet path driven (reference not blank, not the pose canvas, non-zero latent). Still **not using appearance**. |
| **SPADE4Tennis** | **Keep as a domain-specialisation control** | Not re-measured on the coding task this wave. |
| **MOFA-Video** | Stays dropped | Licence-blocked. |
| **StableAnimator** | Wrapped, not shipped | Generate refuses (SVD-XT). Cannot be flagship. |

**What gets added:** StableAnimator is wrapped on `phase-bp/bp4` but **cannot
be the shipped flagship** until SVD-XT is licence-cleared — the Apache-2.0
claim applied to the adapter card, not the inference stack (see §2.4).
MTVCrafter is still a candidate *motion representation* (4D/SMPL tokens), not a
drop-in generator; not wrapped this wave. Sparse2Dense still has no public code
or weights (rechecked 2026-08-22).

#### 6.3 What the 2026 literature says we should consider

Surveyed 2026-08-22. Two findings matter and both are actionable.

**Our engines are old.** Animate-Anyone is repeatedly outperformed in recent
comparisons, showing face and body distortion where newer work does not.
Candidates worth probing for the quality-flagship slot, in order of how cheaply
we could adopt them:

| Candidate | Why | Cost |
|---|---|---|
| **StableAnimator** | Best reported identity preservation (CSIM) and FVD in its class. Adapter Apache-2.0 on HF; inference needs SVD-XT (not bundled) — **same licence class as MOFA**, so "cheapest real upgrade" was wrong. Wrapped; live run blocked on VRAM and the pinned env. | high (licence) |
| **MTVCrafter** | SOTA on TikTok, +65% FID-VID over second best. Tokenises raw 4D motion rather than 2D pose images — *directly relevant to our motion-representation axis*. | medium |
| DisPose / Animate-X / StableAnimator++ | Incremental over the above | defer |

**The other new work is not a competitor system — it is a corner of our lattice.**
This was initially mis-read here, and the correction matters because it is the
stronger position. These systems all share one construction: code a reference
frame conventionally, send a compact per-frame motion signal, and synthesise the
rest. In this project's vocabulary that is an *appearance representation* plus a
*motion representation* plus a generator — with detection, selection, tracking,
rigid objects, background, residual and codec fallback all switched off.

| Work | As a lattice corner | What it does not have |
|---|---|---|
| **Sparse2Dense** (DCC 2026) | appearance = VVC-coded key-reference frame; motion = sparse **3D** keypoints; generator = keypoint-aware multi-task net | one subject; no background model, no non-person objects, no residual, no fallback |
| **T-GVC** | appearance = coded keyframe; motion = semantically weighted sparse trajectories; generator = training-free steered diffusion | whole-frame rather than per-object; one motion representation; no corrective channel |
| **ReGenVC** | appearance = neurally coded first frame; motion = per-frame pose keypoints | talking heads; single subject |

So the positioning is not "we compete with these". It is: **each of these
corresponds to a single configuration of the lattice this paper defines, and
PointStream is the framework in which such a configuration is one cell among
many** — with the components they lack, and with the representation comparison
none of them runs. `GVC-RT` is the exception and sits outside this framing; it
bears on `eval-operating`, where our measured speed is poor, and should be cited
plainly.

Say *corresponds to*, never *is a special case of ours*. These are independent
systems whose designs happen to land on corners we also define; claiming they are
instances of our framework would be both wrong and rude.

**Can we plug them in?** Architecturally yes — Sparse2Dense satisfies our
generator contract exactly. Practically, **no public code or weights were found**
for Sparse2Dense (verify once more before concluding), and T-GVC is an unreviewed
preprint. What *is* adoptable is the idea: **3D keypoints as a motion
representation**, against our current 2D COCO-17 stored as WholeBody-133. That is
a candidate arm for `eval-object` and costs a keypoint schema, not a new model.

**Bound before believing:** Sparse2Dense's 74.5% BD-rate against VVC is the
state of the art on an easier problem. A PointStream BD-rate substantially
better than that on harder content is an **alarm**, not a triumph — check the
anchor, the overlap range and the region scoping before believing it.

#### 6.4 Measure the part we generate, not the whole frame

**A frame-level score hides a broken object.** A reconstruction whose background
is perfect and whose player is mangled still posts a respectable frame PSNR,
because the player occupies a small fraction of the pixels. Every evaluation must
therefore be **scoped to the region the component under test produced**:

- object generation → scored on the **object crop / mask**, not the frame;
- background modelling → scored on the **background region**, with objects excluded;
- the whole frame → reported *as well*, never *instead*.

This is an architectural requirement, not an evaluation preference, and it is why
`BP2-region-metrics.md` exists. Today `src/components/metrics/` has no concept of
a region at all.

#### 6.5 Metric discipline: PSNR to develop, the rest to publish

**PSNR is the internal check.** It is fast, always comparable, and enough to
answer "did this engine run and produce something plausible". VMAF, SSIM, LPIPS
and FVMD are slow, and computing them during triage buys nothing.

- **Triage and development:** PSNR only, region-scoped.
- **Paper results:** the full set, once the roster is fixed and the runs are the
  ones we intend to cite.

A triage number is never citable and never gets a `CLAIM` line.

#### 6.6 Validate small before scaling

- **B′.1 — fix the probe set.** ✅ Rebuilt as v2 on `phase-bp/bp1`, pose
  alignment fixed on `phase-bp/integrate` (§2.3). The verifier fails on the
  v1 snapshot and on the unaligned v2 snapshot, and passes on the rematerialised
  tree. All 12 clips have 48 colour frames and 48 skeleton frames.
- **B′.2 — wire the loaders and probe.** Loaders wired on `phase-bp/bp3` /
  `phase-bp/bp4` (§2.1). The cross-engine probe is Wave 2 (`BP5`).
- **B′.3 — fix the roster in writing**, with the reason each engine holds its
  slot. **Only then** prepare the full dataset.

Improving numbers is a separate decision taken after B′.3, in this order of
cost: parameter tuning, then fine-tuning on our data, then swapping a model.
Tuning an engine that is about to be dropped is the most expensive way to waste
September.

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
- **After B′:** the probe-set verifier fails on the old set and passes on the
  rebuilt one; a perfect-background/destroyed-object reconstruction posts a good
  frame PSNR and a bad object-scoped PSNR; every engine on the §6.2 roster loads
  real weights and returns a frame that is not its input; the roster is fixed in
  writing with the reason each engine holds its slot.
- **After C — the decisive gate:** `config/tier_fast.yaml` and
  `config/tier_quality.yaml` each produce a run summary carrying **real PSNR,
  SSIM, VMAF and LPIPS numbers**; a residual-absent run completes and reports its
  measured quality drop; the all-off corner reduces to the source video. None of
  this works today in any configuration.
- **After D:** a codec sweep invokes PointStream itself and emits one
  rate-quality-time table containing every arm.
- **During §7:** every cited run has empty `invariant_failures`.
