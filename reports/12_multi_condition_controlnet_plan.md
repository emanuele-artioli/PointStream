# Multi-Condition ControlNet (fused Canny + Pose + Seg + Reference) — Plan
*Status: Active | Last updated: 2026-07-13 | Code: `scripts/train_controlnet.py`, `scripts/train_campaign.py`, `src/decoder/`*

## Scope

Report 5 proposed a unified Multi-ControlNet ("the ultimate goal"): fuse
Canny, DWPose skeletons, YOLO segmentations, and a reference-image prompt
into one conditioning stream, instead of the separate single-condition
ControlNets we fine-tuned. Report 7 §4 deferred it post-core, and report
10 kept it deferred "unless single-condition ControlNet is the
bottleneck". **Decision 2026-07-13 (user): un-deferred.** The G2 campaign
currently compares pix2pix / SPADE4Tennis-lite / *pose-only* ControlNet —
the multi-condition model was never wired in and must join the comparison.

**Why it's cheap in bandwidth terms (the paper argument):** every
condition is derivable on the decoder from data the encoder already
transmits — keypoints are in the `ActorPacket`s, the segmentation and
Canny maps can be recomputed client-side from the deterministic
`SynthesisEngine` output, the reference crop comes from the panorama /
first-keyframe actor texture, and the caption is static per scene. So the
multi-condition model adds **zero metadata bytes**; under the Residual
Guarantee it pays for itself iff it shrinks the residual at all. Any
implementation choice that would require transmitting a *new* per-frame
stream breaks this argument and is out of scope.

**Executor discipline:** phases in order; **HARD STOP** gates halt the
work and get written into this report's findings log (create one at the
bottom as findings accrue, standard format). ruff/mypy/fast-pytest before
any GPU run; commit before long runs. Pin `seed`, image size (512), and
inference step count in every eval so runs are comparable.

## Existing assets (verify these paths before starting; if one is missing, STOP and report)

- Fine-tuned single-condition ControlNets in `assets/weights/`
  (pose / canny / seg / ip-adapter — see report 10's checkpoint census,
  plus fresher ones under `outputs/campaign/g2_overnight/checkpoints/`).
- Dataset conditions per track, already materialized:
  `assets/dataset/<video>/segmentations/scene_*/track_XXXX` (RGBA actor
  crops) with sibling dirs `track_XXXX_skeleton`, `track_XXXX_canny`,
  `track_XXXX_caption` — exactly what `ControlNetDataset` in
  `scripts/train_controlnet.py` already globs.
- Probe protocol: `assets/probe_set/manifest.json` (12 clips, 5 training
  videos), `scripts/eval_checkpoint.py`, `scripts/train_campaign.py`.
  Held-out videos `alcaraz_highlights`, `djokovic_zverev` must never
  appear in training data (`verify_data_root_excludes_probe_set` guards
  the probe split; the held-out split is guarded by using
  `assets/probe_set/training_view` as `--data-root`, nothing else).

## Phase A — Composition without training (1 session, mostly inference)

Compose the **already fine-tuned** single-condition ControlNets with
diffusers' `MultiControlNetModel` (list of ControlNets + per-model
`controlnet_conditioning_scale`), plus IP-Adapter for the reference crop.

1. Add a `multi-controlnet` GenAI backend next to the existing
   `canny-controlnet` engine in `src/decoder/` (same interface, same
   deterministic seeding path; config key `genai-backend:
   multi-controlnet`, condition weights as config keys, not CLI flags).
2. Extend `scripts/eval_checkpoint.py` with a variant kind that loads the
   composition (no new checkpoint format needed — it references the
   existing single-condition dirs).
3. Score on the probe set at the same settings as the campaign's
   `controlnet_pose` eval (image size, steps, seed) and put the rows in
   the same JSONL so `rank_variants` maths applies.
4. Sweep only the conditioning scales, coarsely: pose/canny/seg each in
   {0.5, 1.0}, reference (IP-Adapter scale) in {0.3, 0.6} — 16 combos max,
   probe-set only, 10 inference steps to keep it to hours not days.

**Gate A (decision, not stop):** best composition vs best single
condition on the probe set. If it wins on ≥ 2 of {LPIPS, SSIM, VMAF} →
Phase B is justified. If it *loses badly* (> 10 % worse composite), the
usual cause is conditioning misalignment, not a dead idea — run the
debug checklist below before concluding anything. Either way, write the
findings entry with the table.

**Debug checklist for a bad Phase A result (work through in order,
attach evidence for each):**
- [ ] Dump the actual condition images side by side for 3 probe frames
      (pose render, canny map, seg mask, reference crop) — are they
      aligned to the same square padding/crop the pipeline uses
      (`pad_to_square` fill=0 vs 255 differs per condition in
      `ControlNetDataset` — the inference path must match)?
- [ ] Are all ControlNets the *fine-tuned* ones, not the stock
      lllyasviel checkpoints? Print the loaded paths.
- [ ] Same VAE/UNet/scheduler as the single-condition eval?
- [ ] Try dropping one condition at a time — a single misaligned
      condition usually explains the whole gap.

## Phase B — One trained multi-condition model (campaign variant)

Two candidate designs; **B2 is the default** unless Phase A's evidence
argues otherwise:

- **B1 — joint fine-tune of the MultiControlNet stack.** Faithful to
  report 5's proposal but trains 3× ControlNet params; VRAM-heavy (the
  spade OOM on this shared 48 GB GPU is a warning), and inference cost
  is 3 ControlNet forward passes per step.
- **B2 — single ControlNet, fused RGB condition (default).** Render one
  3-channel control image per frame: **R = canny map, G = skeleton
  render, B = seg mask**; reference photometrics stay with IP-Adapter at
  inference. One model, same VRAM and inference cost as any single
  ControlNet, trivially fits the existing training/eval plumbing.

Implementation for B2:
1. `ControlNetDataset`: add `condition_type: "fused"` that loads the
   three sibling dirs and stacks them into one RGB condition (identical
   resize/pad path for all three — see debug checklist above).
2. `scripts/train_controlnet.py`: accept `fused`; init from the
   fine-tuned canny ControlNet (closest input statistics), not from
   scratch.
3. `scripts/train_campaign.py`: add variant `controlnet_fused`
   (kind `controlnet`, condition_type `fused`). **Fairness rule:** a
   late-joining variant must first be trained to the same cumulative
   epoch budget as the current rung target before it may be ranked
   against the incumbents — never rank a 5-epoch model against
   20-epoch ones.
4. Decoder: the `multi-controlnet` backend from Phase A gains a
   `fused` mode (one ControlNet, composite condition rendered
   client-side from transmitted semantics).

**Gate B1 (HARD STOP):** before the full training run, overfit 16 frames
from one track for 200 steps and confirm the sampled output is not
gray/noise and the loss decreased — the standard mock-first smoke this
repo requires. **Gate B2 (HARD STOP):** training loss over the real run
must trend down; if any epoch's sample grid (`--sample-dir`) is visually
constant or "deep-dream" noise (report 5's known failure mode), stop and
report rather than training through it.

## Phase C — Residual-Guarantee verdict (the only verdict that counts)

Probe-set metrics justify continuing; the *paper claim* requires: run the
pipeline on eval clips with `genai-backend: multi-controlnet` (fused) vs
the incumbent backend, same config otherwise, and compare
`size(metadata) + size(residual)` via the benchmark harness
(`python -m scripts.benchmark_matrix run <spec.yaml>`, new spec in
`config/benchmarks/`). Since the condition adds no metadata bytes, the
test is simply whether the residual shrinks. Report the pays-for-itself
table like every other component.

## Self-verification checklist (every phase)

- [ ] Loaded-weights paths printed and pasted into the findings entry (stock-vs-finetuned mixups are silent otherwise).
- [ ] Seed + steps + image size pinned and identical across compared variants.
- [ ] Sample frames dumped and eyeballed before any metric is computed.
- [ ] Eval rows appended to the campaign JSONL (never a side spreadsheet).
- [ ] No training data from `alcaraz_highlights` / `djokovic_zverev`; `--data-root` is the materialized `training_view`.
- [ ] Batch size chosen for the *shared* GPU: check `nvidia-smi` free memory first; start at 4 with gradient accumulation, never 16 at 512 px (that's what OOM-killed spade4tennis on 2026-07-13).

## Findings log

*(append dated entries here as phases complete, standard format)*
