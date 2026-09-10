# Generation Area

**Evidence Revision**: Reconciled through PR #28 (`66da545dcf`) and PR #72.
**Owned Scope**: Generative backends (`src/components/generation/`), ControlNet, IP-Adapter, Animate-Anyone, SVD, generator dispatch.

---

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
