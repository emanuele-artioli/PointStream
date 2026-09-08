# Generation Area

**Evidence Revision**: Reconciled through PR #28 (`c3d0b2e`) and PR #72.
**Owned Scope**: Generative backends (`src/components/generation/`), ControlNet, IP-Adapter, Animate-Anyone, SVD, generator dispatch.

---

## 1. Current State

PointStream is architected so that generative synthesis is a modular, optional stage (`STAGE_GENERATION` in `src/contracts/lattice.py`).

### Key Empirical Finding
Extensive empirical benchmarking across multiple generative models demonstrated that **no generative engine outperformed a simple pasted reference keyframe** on objective rate–distortion metrics (PSNR and SSIM):
- Generative models introduced spatial hallucination, boundary bleeding, and frame-to-frame temporal flicker.
- In-domain fine-tuning improved perceived realism in isolated crops but degraded whole-frame fidelity compared to reference pasting.
- **Operating Policy**: In all Gate A competitive sweeps and shipped tier configurations (`fast`, `balanced`, `quality`), **generation is configured OFF**. The primary codec win must come from semantic background reuse, sparse appearance crops, and corrective residuals, not generative hallucination.

---

## 2. Key Decisions & Evidence Anchor

| Topic | PR / Commit | Decision & Status |
|---|---|---|
| Engine Roster | #20 (`34b5c77`), #27 (`9991275`), #28 (`c3d0b2e`) | Evaluated ControlNet, IP-Adapter, Animate-Anyone against pasted reference. |
| Generator Interface | #28 (`c3d0b2e`) | Established `GeneratorRef` contract and disabled-by-default behavior. |
| Negative Result | Paper & #28 | Formally documented as a paper finding: pasted reference keyframe is the gold standard for temporal video codec fidelity. |

---

## 3. Next Actions

| ID | Status | Dependencies | Source | Description & Acceptance Criteria |
|---|---|---|---|---|
| `GEN-ACT-01` | Previously deferred (D2) | Gate A passed | `plans/DEFERRED.md` | **SAM3 segmentation evaluation**: Assess SAM3 in an isolated conda environment with newer PyTorch support. Acceptance: Verify whether segmentation quality improves crop boundary precision without breaking current environment pins. |
| `GEN-ACT-02` | Previously deferred (D4) | Gate A passed | `plans/DEFERRED.md` | **SVD temporal consistency test**: Evaluate Stable Video Diffusion components under isolated license and runtime boundaries. Acceptance: Strict evaluation against the pasted-keyframe control under matched bitrates. |
| `GEN-ACT-03` | Closed / Archived | None | `plans/DEFERRED.md` (D7) | **Training stop rules**: Policy established that no unconstrained model training is executed before rate–distortion parity is reached. |
