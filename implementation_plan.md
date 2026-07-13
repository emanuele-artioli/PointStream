# Multi-Condition ControlNet Phase A

*Reviewed and corrected 2026-07-13 — the original draft had two errors that
would have broken Phase A. Read this whole file before writing code; both
corrections below come from actually checking `assets/weights/` and
`scripts/train_controlnet.py`'s dataset code, not from re-reading report 12.*

The current training campaign has been halted, the ControlNet checkpoint recovered, the orchestrator fixed to prevent `--epochs 0` overwrites, and the restarted campaign verified to be running 20 real epochs.

While the campaign runs on the GPU, we will implement **Phase A** of the Multi-Condition ControlNet plan (`reports/12_multi_condition_controlnet_plan.md`). Phase A evaluates the composition of *already fine-tuned* single-condition ControlNets (Canny, Seg, Pose, Reference) to determine if joint conditioning provides a tangible residual benefit without training a new fused model.

## Open Questions (corrected — the original "none at this time" was wrong; both of these were findable in 30 seconds by listing `assets/weights/`)

1. **The fine-tuned Canny ControlNet is not named "canny-controlnet".** `assets/weights/` contains
   `custom-controlnet`, `pose-controlnet`, `seg-controlnet`, `ip-adapter-controlnet` — no directory
   literally named "canny". Checked `assets/weights/custom-controlnet/config.json`:
   `"_name_or_path": "lllyasviel/control_v11p_sd15_canny"` — **`custom-controlnet` is the fine-tuned
   Canny checkpoint.** Use this exact mapping:
   - pose    → `assets/weights/pose-controlnet`
   - canny   → `assets/weights/custom-controlnet`   (NOT "canny-controlnet" — that path doesn't exist)
   - seg     → `assets/weights/seg-controlnet`
   - reference → `assets/weights/ip-adapter-controlnet` (see point 2 — this is NOT a diffusers IP-Adapter)
   Before loading any of these in code, assert the directory exists and log the resolved path —
   report 12's own debug checklist flags "loaded stock instead of fine-tuned" as the most likely
   failure mode; a wrong directory name is exactly how that happens silently (diffusers'
   `from_pretrained` would either error on a bad local path or, worse, resolve it as a HuggingFace
   Hub repo id and silently download stock weights instead).

2. **"ip-adapter-controlnet" is not a diffusers IP-Adapter — it is a fourth ControlNetModel.**
   Checked `scripts/train_controlnet.py`'s `ControlNetDataset.__getitem__`: for
   `condition_type == "ip-adapter"`, the conditioning image is `cond_tensor =
   self.cond_transform(img)` — i.e. the training script fed the **target actor image itself**
   (padded/resized) as the ControlNet condition, using OpenPose-architecture ControlNet weights
   as the base (`defaults["ip-adapter"] = "assets/weights/control_v11p_sd15_openpose"` in the same
   file). This was trained as a reference-image-as-ControlNet-condition trick via the exact same
   `train_controlnet.py --condition-type ip-adapter` pathway as the other three — it is
   architecturally a `ControlNetModel`, not the diffusers-native IP-Adapter format
   (`pipe.load_ip_adapter(...)`, image-prompt cross-attention, `.bin`/`.safetensors` from
   `h94/IP-Adapter`). **Do not call `pipe.load_ip_adapter()` on this checkpoint — it will fail or
   silently do the wrong thing.** Treat "reference" as a **fourth condition inside the same
   `MultiControlNetModel` list**, alongside pose/canny/seg, with its conditioning image built the
   same way training built it: the reference actor crop, RGBA-composited onto black, padded to
   square (fill=0), resized to the pipeline's working resolution — mirror
   `ControlNetDataset.__getitem__`'s `img`/`cond_transform` path exactly so train/inference are
   consistent. Report 12's text ("reference via IP-Adapter") is corrected by this finding; that
   report should be updated to match once Phase A code lands.

## Proposed Changes

### `src/decoder/genai_compositor.py` (or similar strategy registry)
- Ensure the strategy factory can load `multi-controlnet`.

### `src/decoder/controlnet_engine.py`

#### [MODIFY] [controlnet_engine.py](file:///home/itec/emanuele/pointstream/src/decoder/controlnet_engine.py)
- **NEW Class:** `MultiControlNetStrategy(BaseGenAIStrategy)`
- **Initialization:** Load `MultiControlNetModel` from diffusers containing **four** fine-tuned ControlNets — Pose, Canny, Seg, and Reference — using the exact paths in "Open Questions" item 1 above. Log each resolved path at init time so a wrong/stock load is visible in the run log, not discovered later from a bad eval score.
- **Pipeline:** Load `StableDiffusionControlNetPipeline` with the four-model `MultiControlNetModel`. No `IPAdapter` injection — see item 2 above.
- **Inference (`generate` method):** Render the skeleton (pose), extract the Canny edge map, load the Seg mask, and build the reference condition image (target-image-as-condition, same preprocessing as training — see item 2) from the deterministic synthesis outputs. Combine these into a list of four conditioning images, order matching the `MultiControlNetModel` construction order.
- **Config:** Expose a `controlnet_conditioning_scale` array of length 4 (pose, canny, seg, reference) to sweep in the evaluation step. Drop `ip_adapter_scale` — it doesn't apply here (item 2).
- **Alignment check (do this before the first real eval, not after a bad score):** confirm all four conditioning images use the same padding convention as their respective training preprocessing (`pad_to_square(..., fill=0)` for pose/canny/seg per `train_controlnet.py`, and the same RGBA-composite-then-pad path for reference) and the same target resolution as the pipeline's working size. A mismatch here is report 12's predicted most-likely failure mode for a bad Phase A score — verify it up front instead of waiting to hit it.

### `scripts/eval_checkpoint.py`

#### [MODIFY] [eval_checkpoint.py](file:///home/itec/emanuele/pointstream/scripts/eval_checkpoint.py)
- Add a new variant kind `multi-controlnet` that bypasses the single-checkpoint path and directly instantiates `MultiControlNetStrategy`.
- Provide CLI overrides to support sweeping conditioning scales: pose/canny/seg each in {0.5, 1.0}, reference in {0.3, 0.6} — the exact grid from report 12 Phase A (16 combos max).
- **Use the same probe manifest, seed, `--eval-steps 10`, and `--eval-img-size` as the campaign's `controlnet_pose` rung-0 eval** (`outputs/campaign/g2_overnight/`) so results land in the same JSONL and are directly comparable via `rank_variants` — do not invent a separate eval protocol for this variant.

## Verification Plan

### Automated Tests
- Run `ruff check src/decoder scripts/eval_checkpoint.py`
- Run `mypy src/decoder scripts/eval_checkpoint.py`
- Run `pytest` on the decoder logic.

### Manual Verification
- Before any real eval: print/log the four resolved checkpoint paths from "Open Questions" item 1 and confirm by eye they point at `custom-controlnet`/`pose-controlnet`/`seg-controlnet`/`ip-adapter-controlnet`, not a HuggingFace Hub id.
- Run a single-frame dummy generation with the `multi-controlnet` strategy to verify shapes, dtypes, and scale bindings don't crash.
- Dump the four conditioning images for 2-3 probe frames side by side and eyeball alignment (padding, resolution, content) before trusting any score from them.
- Run `python scripts/eval_checkpoint.py` on the probe set with the `multi-controlnet` variant and append its results to the JSONL.
- Perform the Phase A scale sweep as documented in report 12 (16 combos max).
- **Gate A (from report 12):** compare best composition vs the campaign's best single-condition ControlNet result on the probe set. Win = beats it on ≥2 of {LPIPS, SSIM, VMAF}. If it loses badly, work through report 12's debug checklist (condition alignment, fine-tuned-vs-stock paths, same VAE/UNet/scheduler, drop-one-condition ablation) before concluding anything about the architecture — write the result into report 12's findings log either way, using real numbers, not just "seems worse."
