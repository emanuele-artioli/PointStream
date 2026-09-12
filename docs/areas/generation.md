# Generation Area

**Evidence Revision**: Reconciled through PR #28 (`66da545dcf`) and PR #72.
**Owned Scope**: Generative backends (`src/components/generation/`), ControlNet, IP-Adapter, Animate-Anyone, SVD, generator dispatch.

---

## Current audit — 2026-09-12

Keep generation OFF as the current control policy, not an architecture ranking.
[The audit](../history/antigravity-audit-2026-09-12.md) finds sparse first-frame
changes, near-identical whole-frame null scores, unhashed loaded pose inputs and
a reproduced checkpoint-factory verification gap. `GEN-ACT-08` is only partially
resolved at integration boundaries; complete the relevant `EVAL-ACT-11` repairs
before model comparison. The historical “strictly superior” / fully verified
phrasing below is scoped to the tested sparse diagnostic and superseded as a
general conclusion. Next session follows [the handoff](../workflow/session/evaluation-handoff.md).

## 1. Current State

PointStream is architected so that generative synthesis is a modular, optional stage (`STAGE_GENERATION` in `src/contracts/lattice.py`).

### Key Empirical Finding
Earlier benchmark reports favored **pasted reference keyframes over the tested generative configurations** on objective fidelity metrics (PSNR and SSIM). These are scoped historical findings, not a validated comparison of every backend or a prediction about future training:
- Generative models introduced spatial hallucination, boundary bleeding, and frame-to-frame temporal flicker.
- In-domain fine-tuning improved perceived realism in isolated crops but degraded whole-frame fidelity compared to reference pasting.
- **Operating Policy (revised 2026-09-09)**: The audited Gate A/B sweeps and shipped tiers configure generation OFF. Earlier comparisons favor pasted references in their tested settings; this does not establish that future trained generators cannot help. Permit bounded generator validation and training before a generation-free win. Select by whole-codec rate–distortion and runtime, including residual demand and transmitted model data, rather than crop realism alone.

---

## 2. Key Decisions & Evidence Anchor

| Topic | PR / Commit | Decision & Status |
|---|---|---|
| Engine Roster | #20 (`fd36b4b338`), #27 (`8d9a4b1669`), #28 (`66da545dcf`) | Evaluated ControlNet, IP-Adapter, Animate-Anyone against pasted reference. |
| Generator Interface | #28 (`66da545dcf`) | Established `GeneratorRef` contract and disabled-by-default behavior. |
| Negative Result | Paper & #28 | Formally documented as a paper finding: pasted reference was the stronger control in the tested configurations; broader claims require renewed validated comparisons. |

---

## 3. Next Actions

| ID | Status | Dependencies | Source | Description & Acceptance Criteria |
|---|---|---|---|---|
| `GEN-ACT-01` | Previously deferred (D2) | Gate A passed | `plans/DEFERRED.md` | **SAM3 segmentation evaluation**: Assess SAM3 in an isolated conda environment with newer PyTorch support. Acceptance: Verify whether segmentation quality improves crop boundary precision without breaking current environment pins. |
| `GEN-ACT-02` | Previously deferred (D4) | Gate A passed | `plans/DEFERRED.md` | **SVD temporal consistency test**: Evaluate Stable Video Diffusion components under isolated license and runtime boundaries. Acceptance: Strict evaluation against the pasted-keyframe control under matched bitrates. |
| `GEN-ACT-03` | Closed / Archived | None | `plans/DEFERRED.md` (D7) | **Training stop rules**: Unconstrained training remains deferred; bounded pilots are allowed under GEN-ACT-04 without prior generation-free parity. |
| `GEN-ACT-04` | Ready | Validated evaluator before training; no generator-free win prerequisite | 2026-09-09 residual/generation audit | **Generator readiness and bounded training**: Inventory actual checkpoints, verify native temporal inference and conditioning, restore the retired training evaluator, then compare a bounded pilot by total coded rate–distortion/runtime. See [worker brief](../workflow/session/generator-readiness.md). |

## PR #88 generator audit — 2026-09-10

The one-epoch pix2pix/SPADE4Tennis-lite pilot is **not a defensible architecture ranking or proof of adequate training**. Both logs show a completed one-epoch run (pix2pix reports 16,272 images and batch 8); SPADE also has an OOM retry log. There is no recorded hyperparameter sweep, convergence study, paired untrained/pasted baseline or source-level uncertainty. The default commands use learning rate 0.0002; pix2pix uses pixel loss weight 100, while this custom SPADE variant uses pixel/VGG/feature-matching weights 10 each. Those differences need model-specific tuning, not forced identical hyperparameters. The campaign state contains only these two variants; the default driver also lists ControlNet, and Animate-Anyone training is explicitly unwired. Omitted models were not shown inferior.

More fundamentally, `evaluate_checkpoint` reads resized actor crops, installs an artificial central-quarter bbox/mask, silently fills missing sources with grey, and calls its results full reconstructed clips. The current full-dataset/probe-manifest pairing has **9/12 tracks with zero requested local-numbered source files**, while their global-numbered source files exist. The manifest distinguishes these coordinates. The archived campaign does not capture enough input-path/configuration provenance to certify which data were scored; invalidate its promotion/pruning decision and rerun after repairing this path. The evaluator also silently falls back from an empty held-out-video selection to all probes; this manifest has no probes from its listed held-out videos. Development track holdouts must not be described as independent-source confirmation.

Other blockers: an empty/missing-metric aggregate can be accepted as successful; temporal frame-difference error is mislabeled FVD; device/seed parameters are accepted without being applied in the evaluator; checkpoint identity uses filename/size/mtime rather than a content hash. Ranking fixed-QP residual bytes does not enforce matched final quality or complete payload cost. The campaign halts with one survivor instead of training the selected candidate to its allocated final budget. Do not resume its pruned state as a justified selection.

Animate-Anyone's smoke script exercises sequence inference and conditioning sensitivity, but does not itself launch or compare two fresh processes. A claimed fresh-process determinism result requires separate immutable evidence. Sensitivity alone does not establish a useful predictor. Readiness-screen all registered backend families, collapse aliases, and record an explicit reason for deferring each untested candidate.

| ID | Status | Dependencies | Source | Description & Acceptance Criteria |
|---|---|---|---|---|
| `GEN-ACT-05` | Ready; blocks ranking | EVAL-ACT-06 / CODEC-ACT-05 integration | PR #88 audit | Repair dataset coordinates, missing-data rejection, real-frame evaluation, checkpoint identity and controls. Re-evaluate both candidates; restore SPADE to the pool. |
| `GEN-ACT-06` | Inventory ready; training gated on evaluator | GEN-ACT-05 | PR #88 audit | Multi-fidelity model/configuration search: existing weights first, meaningful resource rungs and per-architecture hyperparameters, matched-fidelity total wire cost/runtime, learning curves and confirmation-safe splits. Follow [submission search](../workflow/session/submission-search.md). |

Repair update — 2026-09-10: coordinate resolution, missing-input rejection, whole-codec evaluation, SHA-256 checkpoint identity, matched configuration controls and the single-survivor campaign continuation have been integrated and tested. `manifests/candidate_inventory.json` is intentionally marked inventory-only: its readiness labels and runtime figures are not selection evidence without linked immutable artifacts. No candidate has been newly ranked or trained by the repaired path.

Recovery closeout — 2026-09-10: PR #88 merged as `2b7c2b0` with CI passing. A three-frame CPU smoke of the immutable pix2pix checkpoint passed its non-source-fallback and shuffled-pose sensitivity controls. This only establishes basic backend readiness; it is neither a quality comparison nor a runtime result.

GPU diagnostic matrix (gpu5, 16 frames, pix2pix, residual QP 32, artifact `diagnostic-pix2pix-rq32.json`): **invalid for generator comparison**. Generation-off and generation-on without residual produced identical PSNR/SSIM/VMAF because `_finish_chunk` converted appearance-conditioned objects into pasted-reference placements (`is_gen` inferred from empty `supplied_crop`). Generation-on added only ~180 bytes; client time was unchanged. Encoder-side generation timing of 0.7–1.2 s is not evidence that generated pixels reached the client. Residual-on still moved quality on the paste arm (~+8 dB PSNR-Y); that does not validate pix2pix. Reports lacked checkpoint SHA, delivered hashes, and a shuffled-conditioning control. Preserve the artifact; do not cite it as a model result. Staged training remains unlaunched until a repaired path shows generation changing delivered frames and residual demand.

## Foreground experiment priority — 2026-09-11

`GEN-ACT-07` — Planned; broad model search follows the component/headroom diagnosis
in `EVAL-ACT-08` and `CODEC-ACT-06`, rather than a required generation-free win.
Follow [resource-dependent frontiers](../workflow/experiment-design.md#4-foreground-models-test-resource-dependent-frontiers).
Start from pasted-reference and ready-model controls, with small-model and
diffusion advantages stated as falsifiable hypotheses under named client budgets.
Charge conditioning, weights/updates and matched-quality correction; measure
latency, throughput and peak memory. Preserve older trials without reusing their
invalid rankings. SPADE/training remains off pending a newly scoped experiment
card. Acceptance: a supported Pareto tradeoff or a bounded rejection/inconclusive
result that determines the next experiment; no family-wide claims from one pilot.

`GEN-ACT-08` — **Complete** (PR #93, `4a55573` / `e9f781f`). Enforced fail-closed client checkpoint resolution against transmitted SHA-256 digest (`resolve_client_checkpoint`), full effective configuration identity, declared control validation, and dynamic clip start frame resolution without synthetic fallback.

`GEN-ACT-09` — **Complete** (Wave 2 Diagnostic Matrix, artifacts `outputs/development-recovery/diagnostic-pix2pix-alcaraz.json` SHA-256 `ccfaa34b8ceca48409839c3baefec20e91f87a0425aea6da2b7429c37ed2fa50` and `diagnostic-pix2pix-federer.json` SHA-256 `407148416bf1455ccfb68cc025a2ff31899689ebd0ffc722fd4f0dddc085af25`).
- **Validity Criteria Met**: Generator comparison validity is strictly `true` for both development scenes; `delivered_pixels_changed` is `true`; `gen_on_vs_paste_hash_match` is `false`; `shuffled_conditioning_changed_pixels` is `true`; `wire_reconciliation` matched 100% of serialized bytes.
- **Instrument Verification**: Resolved skeleton pose alignment by track position (matching crop global IDs to 0-based skeleton index), eliminating the previous indexing mismatch.
- **Empirical Findings on pix2pix (16 frames @ 4K)**:
  - *Alcaraz*: Pasted reference (`gen_off_res_off`) achieves 34.55 dB PSNR-Y / 90.81 VMAF for 523,610 B. pix2pix without residual (`gen_on_res_off`) yields 33.50 dB PSNR-Y (-1.05 dB) / 90.56 VMAF (-0.25) while costing 957,945 B (+434,335 B, primarily pose conditioning metadata). With residual (QP 32), pasted reference (`gen_off_res_on`) reaches 42.62 dB / 93.35 VMAF for 600,656 B, while pix2pix (`gen_on_res_on`) achieves 38.88 dB (-3.74 dB) / 93.16 VMAF for 1,035,763 B.
  - *Federer*: Pasted reference (`gen_off_res_off`) achieves 30.46 dB PSNR-Y / 78.13 VMAF for 627,735 B. pix2pix without residual (`gen_on_res_off`) yields 29.82 dB (-0.64 dB) / 77.75 VMAF for 1,282,341 B (+654,606 B). With residual, pasted reference (`gen_off_res_on`) reaches 39.76 dB / 90.01 VMAF for 796,646 B, while pix2pix (`gen_on_res_on`) reaches 37.40 dB (-2.36 dB) / 89.50 VMAF for 1,451,803 B.
- **Decision**: Verified that the client generator executes faithfully and alters reconstructed frames as conditioned, but pix2pix under current uncompressed pose conditioning degrades whole-codec rate–distortion against pasted reference. Generator remains OFF for the headline compression ladder.

