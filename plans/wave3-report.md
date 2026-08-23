# Cursor → Claude — Wave 3 report

**Written:** 2026-08-23T07:44+02:00 on host `gpu6`.
**Why this file exists:** Claude's Wave-3 prompt asked Cursor to run four streams
and report back. This is that report. Numbers are triage unless a `CLAIM` names
an `outputs/` path in the paper repo. Nothing is merged to `main`. Nothing is
pushed. GPUs are idle; the pose-ref train has exited.

Read this, then `AGENTS.md`, then the living `PLAN.md` on the main checkout
(status table + §2.3 were updated in the working tree, **uncommitted**).

---

## PARSE BLOCK

```
WAVE: 3
DATE: 2026-08-23
HOST: gpu6
MAIN: /home/itec/emanuele/pointstream  branch=main  HEAD=7499faf  ahead_of_origin=10
MAIN_UNCOMMITTED: PLAN.md plans/README.md plans/C3-runner.md plans/wave3-report.md
MERGED_TO_MAIN: no
PUSHED: no
GPU_JOBS: none
QUALITY_FLAGSHIP: unset
OPTION_A: failed
OPTION_B_AA: failed  object_psnr=8.96  verdict=not_using_appearance
OPTION_B2_IPADAPTER: failed  object_psnr=8.90  verdict=not_using_appearance
STATIC_COPY_PUBLISHED: 11.82  # PLAN.md §2.3 shared-geometry paste, 12 clips seed 42
STATIC_COPY_LETTERBOX: 11.47  # BP8/BP9 independent 512 letterbox, same task
SUCCESS_BAR_DB: 12.82  # 11.82 + 1; beating floor by <1 dB is not a result
POSE_REF_EPOCH10: 11.18  verdict=not_using_appearance  series=11.11-11.33
OPTION_C: reported_finding_not_started  # change what the paper claims is transmitted
WAVE2_ROSTER: void
```

---

## One sentence

No generative engine on the roster uses appearance: Animate-Anyone, a real
IP-Adapter, and a ten-epoch pose-ref ControlNet retrain all lose to pasting the
keyframe forward. The pipeline runner and the probe harness now exist anyway.
The paper's evaluation hole already names the checkpoint property; it does not
yet record that retraining failed.

---

## Streams (verified)

| ID | Brief | Worktree | Branch | Tip | Verified |
|---|---|---|---|---|---|
| A | `plans/BP8-appearance-conditioning.md` | `/home/itec/emanuele/pointstream-wt/bp8` | `phase-bp/bp8` | `ace5cc5` | driven coding-task JSON + train exited |
| B | `plans/BP9-probe-harness.md` | `/home/itec/emanuele/pointstream-wt/bp9` | `phase-bp/bp9` | `8ec61bb` | 16 harness tests re-run pass |
| C | `plans/C3-runner.md` | `/home/itec/emanuele/pointstream-wt/c3` | `phase-c/c3` | `940a602` | 11 runner tests re-run pass |
| D | `plans/P1-paper-catchup.md` | `/home/itec/emanuele/pointstream-wt/p1` | `phase-p/p1` (paper repo) | `dfe7b49` | grep: 11.82 dB only in `HOLE` comments |

Base for A/B: `phase-bp/integrate` `18bf21e`.
C3 merge-base: C1 `8ce3450` + C2 `26533b0` then runner. Paper base: `f967e0c` (BP6 already merged there).

**Correction to Claude's Wave-3 prompt:** Waves 1 and 2 are **not** on `main`.
`main` still has no `src/components/`. Do not merge/rebase code onto `main`.

Worktree `assets/` is a symlink to `/home/itec/emanuele/pointstream/assets`.
Do not commit the symlink, `assets.gitkeep-tree/`, or `assets/weights/.gitkeep`
deletion.

---

## A — BP8 (critical path). Honest negative.

Coding task everywhere below: appearance = track-local frame 0, pose = frame 24,
score vs frame 24, 12 probe clips, seed **42**. Object-scoped PSNR.
`citable: false` on every JSON.

Success bound written **before** generate:
`pointstream-wt/bp8/outputs/bp8-pose-ref-train/bounds.json` and each eval dir's
`bounds.json`. Working appearance-conditioned model must be **≥ 12.82 dB**.
At or below the floor = **not using appearance**.

### Driven numbers

| Arm | Object PSNR | vs letterbox floor 11.47 | Path under `pointstream-wt/bp8/outputs/` |
|---|---|---|---|
| static copy (this letterbox driver) | **11.47** | floor | `bp8-coding-task/` |
| static copy (published, shared-geometry) | **11.82** | — | PLAN.md §2.3, not this driver |
| Animate-Anyone 20 DDIM letterbox | **8.96** | −2.51 | `bp8-coding-task/animate-anyone-letterbox.json` |
| AA stretch (diagnostic) | 7.57 | −4.39 | `bp8-coding-task-stretch/` |
| real IP-Adapter + stock OpenPose | **8.90** | −2.56 | `bp8-coding-task-ipadapter/` |
| pose-ref step 5950 | 11.31 | −0.16 | `bp8-coding-task-pose-ref-step5950/` |
| pose-ref epoch 1 | 11.33 | −0.13 | `bp8-coding-task-pose-ref-epoch1/` |
| epoch 2 | 11.25 | | `...-epoch2/` |
| epoch 3 | 11.24 | | `...-epoch3/` |
| epoch 4 | 11.24 | | `...-epoch4/` |
| epoch 5 | 11.18 | | `...-epoch5/` |
| epoch 6 | 11.19 | | `...-epoch6/` |
| epoch 7 | 11.18 | | `...-epoch7/` |
| epoch 8 | 11.11 | | `...-epoch8/` |
| epoch 9 | 11.26 | | `...-epoch9/` |
| **pose-ref epoch 10** | **11.18** | −0.28 | `bp8-coding-task-pose-ref-epoch10/` |

AA: ReferenceNet was fed a non-blank, non-pose reference (driven, not read).
Scheduler DDIM v-prediction / trailing / zero-SNR. In-domain: AA saw both
held-out videos. Verdict **not using appearance**.

IP-Adapter: **not** the mislabelled `assets/weights/ip-adapter-controlnet`
directory (that is a seg ControlNet). Wired against stock SD-1.5. Still
**not using appearance**.

Pose-ref: `include_reference` used; control = reference painted under skeleton.
Smoke: `residual_delta=2.69` on two ControlNet forwards that differed only in
the reference (`outputs/bp8-pose-ref-smoke/`). Train: `cuda:0`, init
`pose-controlnet/checkpoint-epoch-10`, 10 epochs, batch 4, 35713 samples, 8929
steps/epoch, hourly checkpoints. Log:
`pointstream-wt/bp8/outputs/bp8-pose-ref-train/train.log`. Final weights:
`assets/weights/pose-ref-controlnet/checkpoint-epoch-10` (also copied to
`diffusion_pytorch_model.safetensors` in that dir, 06:38). **Series is
flat-to-down from epoch 1.** The reference entered the net and did not teach
identity. No measurement alarm fired (all means inside 8–35 dB).

**Do not retrain this recipe.** **Do not tune guidance/steps/strength.**
**Do not treat Wave-2 16 dB self-reconstruction as a coding result.**

Code on `phase-bp/bp8` also: independent letterbox of AA appearance vs pose;
real IP-Adapter loader; `scripts/train_controlnet.py --include-reference`;
`scripts/bp8_coding_task.py`.

---

## B — BP9 probe harness

`experiments/probe/**` on `phase-bp/bp9`. Appearance from keyframe, conditioning
and scoring reference from frame 0+offset. Offsets **8, 16, 24, 32**; headline
24. Static copy is a permanent arm. Engine at or below floor labelled
**not using appearance**. `self_reconstruction_psnr` recorded, listed in
`ranking_ignores`, never the rank key.

Driven static copy: `pointstream-wt/bp9/outputs/bp9-static-copy/summary.json`
seed 42, 12 clips, CPU. Headline offset 24: **11.47 dB** object, **13.68 dB**
frame. Frame is higher than PLAN.md §2.3's 8.90 because this harness scores the
512 letterboxed canvas. Object-scoped is the triage number.

Bounds rewritten against the floor, written before generate. One per-clip
outside the 8–16 expected *mean* band (`sinner_alcaraz/.../track_0058` offset 8,
5.91 dB) — letterbox geometry, not an alarm; recorded in `bounds.py` revision 1.

Tests: `tests/components/test_probe_harness.py` — 16 passed (re-run).

---

## C — C3 runner

Brief: `plans/C3-runner.md` (copied to main checkout, uncommitted).
Code: `src/runner/**` on `phase-c/c3` `940a602`. One importable `run()`.
Single-chunk is the same loop. Both quality views required:
`QualityReport` (recon vs source) and DAG `ART_QUALITY` (delivered vs source).
One `sizes_bytes` ledger. Generation off: generator never constructed.
Encoder-side `dispatch` uses the same `GeneratorRef` as client `reconstruct`.
Did not edit C1/C2. Tests: `tests/runner` — 11 passed (re-run).

`assert_coherent` stays in contracts; runner passes `GeneratorRef.requires` into
`Encoder.build`.

---

## D — P1 paper (separate git repo)

`/home/itec/emanuele/pointstream-wt/p1` branch `phase-p/p1`.

```
dfe7b49 design: lattice and residual construction claims now cite tests
4e164a7 appendix: AVC addroi is a no-op under QP, so that arm is pixel-domain
f5f499d eval: record the appearance-unconditioned engines and the methodology they force
```

`HOLE(sec:evaluation)` names ControlNet trained without a reference image, not
July's 15.8 VMAF. **11.82 / 11.01 / 11.20 live only in that HOLE comment**, not
in a results table. Paper does **not** yet know Option A (pose-ref retrain)
failed — that landing is still owed, as `NOTE`/`HOLE` prose, not a table.

---

## Prior waves, still unmerged (do not delete)

| Branch | Tip | What |
|---|---|---|
| `phase-b/integrate` | `5910203` | Phase B components |
| `phase-bp/integrate` | `18bf21e` | B′ loaders, probe-set alignment |
| `phase-bp/bp5` | `36e511d` | roster probe — **verdict void** |
| `phase-c/c1` | `8ce3450` | reconstruction + residual |
| `phase-c/c2` | `26533b0` | encoder DAG |
| `phase-d/cleanups` | `8470211` | mypy 0; AVC addroi no-op under QP |

---

## Open questions (for a human / Claude, not for another silent retrain)

1. **Option C wording.** The architecture still *transmits* appearance. These
   checkpoints do not *use* it. That is a scoped negative about the current
   engines, or a claim change if we stop promising identity. P1 already forbids
   writing it as a lattice failure. Decide the sentence before any more GPU.
2. **Merge order.** Code still lives on many branches. `main` is plan-only.
   A merge to a single integrate branch is a session of its own.
3. **Phase D** can consume `runner/` as a library on all-off / residual-only
   corners **without** a working generator. A quality-flagship RD curve cannot.
4. **Do not** upgrade torch in `pointstream` for SAM3; that is still `DEFERRED.md` D2.

---

## Immediate next (Claude)

1. Read this file. Fold Option A's failure into the paper as `NOTE`/`HOLE`
   prose (not a results table) if taking the paper next.
2. Do not launch another ControlNet retrain on painted-reference pose.
3. If merging: worktrees above; never onto `main` until `src/components/` is
   supposed to live there.
4. If running experiments: all-off and residual-only through `src.runner.run`
   on `phase-c/c3`. Generative corners will lose to static copy until Option C
   or a genuinely new appearance-conditioned engine.

## Landmarks

- This report: `plans/wave3-report.md`
- Living briefs: `plans/README.md`
- BP8 numbers also in worktree `pointstream-wt/bp8/PLAN.md` §2.6 / §6.2
- Probe: `pointstream-wt/bp9/experiments/probe/`
- Runner: `pointstream-wt/c3/src/runner/run.py`
- Paper: `pointstream-wt/p1/sections/evaluation.tex`
- Env: `conda run -n pointstream --no-capture-output <cmd>`
- Never `pip install` into `pointstream`
