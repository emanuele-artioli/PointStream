# Codec Area

**Evidence Revision**: Implementations through PR #85; gate interpretations audited 2026-09-09.
**Owned Scope**: `src/components/background/`, `src/components/appearance/`, `src/components/motion/`, `src/components/residual/`, `src/pipeline/codec/`.

---

## 1. Current State

PointStream structures video coding into distinct semantic components:
1. **Background**: A reusable high-resolution canvas / plate transmitted once or periodically.
2. **Appearance**: Sparse foreground actor crops transmitted as reference keyframes.
3. **Motion**: Trajectories, 2D bounding boxes, and pose/keypoints driving actor deformation.
4. **Residual**: Optional corrective high-frequency residual signal for high-fidelity reconstruction.
5. **Fallback**: Explicit coded conventional fallback when scene dynamics violate semantic assumptions.

### Empirical Benchmarks and Bottlenecks
In the Gate A 48-frame native run (#69), PointStream payload was dominated by two elements:
- **Background Plate Bitfloor**: `src/components/background/stream.py` hardcoded `StreamCodec` to `libaom-av1`. The archived #71/#72 probe reported 268,949 bytes at CRF 63 on one canvas/build. This is a configuration-specific observation, not proof of a universal AV1 floor; reproduce the command and reconstruction quality before attributing the cause.
  - Benchmarked alternatives on the same 4K canvas:
    - `libaom-av1 (CRF 63)`: 268,949 bytes (262.6 KB)
    - `SVT-AV1 (QP 63)`: 41,162 bytes (40.2 KB)
    - `VVC libvvenc slower (QP 63)`: 5,347 bytes (5.2 KB)
    - `VVC libvvenc slower (QP 55)`: 14,313 bytes (about 14.0 KiB; archived #72 value)
- **Actor Appearance Crops**: The pre-#75 implementation used baseline JPEG (`CompressedImageAppearance` in `src/components/appearance/compressed.py`), spending 4–25 KB per crop with ringing artifacts below quality 40. WebP encoding was implemented in #75 and used by #83/#85; standalone matched-quality crop benchmarking remains needed. The proposed 35–50% reduction is unverified at matched quality on this corpus; it is not an established result.

---

## 2. Key Decisions & Evidence Anchor

| Topic | PR / Commit | Decision & Status |
|---|---|---|
| Sidecar encoders | #36 (`09c92880c4`) | Standalone AV1 and VVC intra sidecar encoders implemented. |
| Background canvas | #45 (`ecebd9b97d`), #50 (`bb5d17445f`) | Canonical offline background stitched representation established. |
| Fallback mechanism | #52 (`68a03dc542`) | Explicit conventional video fallback path added for ineligible segments. |
| Native 48-frame run | #69 (`648325b`) | Measured whole-system rate/quality; identified background & appearance byte floor. |
| Competitive regime | #70–#72 | Proposed VVC/SVT background sidecars and WebP/AVIF appearance crops. |
| VVC background streaming | #75, #80 | Implemented VVC low-delay background streaming (`StreamCodec.VVC`) with periodic intra refresh (`-period 1`) for causal prefix stability. Resolves `CODEC-ACT-01`. |
| WebP appearance crops | #75 | Implemented OpenCV WebP foreground actor reference compression (`AppearanceFormat.WEBP`). Resolves `CODEC-ACT-02`. |
| Gate A Rate Ladder | `outputs/gate-a-vvc-webp-n96-run2/report.json` | Completed development sweep; competitive pass superseded by 2026-09-09 evaluation audit. |
| Gate B Procedure Freeze | `src/contracts/frozen_procedure.py` | Locked down C0–C3 rate ladder configuration, VVC low-delay background streaming with intra refresh (`-period 1`), WebP appearance crops, zero generation/residual lattice settings, and metric thresholds/bounds for confirmation. |
| Gate B pilot | `outputs/gate-b-confirmation/report.json` | Execution completed, but competitive confirmation and fair speed claims are not established; see evaluation audit. |
| Residual transport repair & wire reconciliation | Overnight recovery (`3d05ce7`) | Restored serialized residual transport in `src/runner/client.py`; implemented full-range signed mapping $[-255, 255]$ in `src/pipeline/residual/`; true predictor base preserved; wire envelope byte reconciliation enforced. Resolves `CODEC-ACT-03`, `CODEC-ACT-05`. |
| High-fidelity residual verification (H0) | Overnight recovery | Verified H0 residual rung (AV1 QP 42, 1:1 scale, zero gating) on 4K tennis clip: achieved PSNR-Y 44.07 dB, SSIM 0.9883, VMAF 94.77 with 228,474 B residual payload and exact ledger reconciliation. |

---

## 3. Next Actions

| ID | Status | Dependencies | Source | Description & Acceptance Criteria |
|---|---|---|---|---|
| `CODEC-ACT-01` | Complete | None | #70, #71, #75, #80 | **Connect VVC intra / SVT-AV1 to background stream**: Implemented in `src/components/background/stream.py` with causal prefix intra-refresh stability. |
| `CODEC-ACT-02` | Complete | None | #71, #72, #75 | **Add WebP/AVIF appearance crops**: Implemented WebP encoding in `src/components/appearance/compressed.py` with matched-quality savings and client decode round trip. |
| `CODEC-ACT-03` | Complete | `CODEC-ACT-01`, `02` | #72, overnight recovery (`3d05ce7`) | **Payload ledger simplification**: Streamline byte allocation tracking across components. Transmitted bytes reconcile to serialized wire payload; `len(bitstream) == sizes.residual` enforced. |
| `CODEC-ACT-04` | Previously deferred (D5) | None | `plans/DEFERRED.md` | **Coded fallback verification**: Verify behavior when semantic tracking fails. Acceptance: Clean switch to conventional intra/inter coding without crash or pipeline desynchronization. |
| `CODEC-ACT-05` | Complete | Coordinate EVAL-ACT-06 | 2026-09-09 audit, overnight recovery (`3d05ce7`) | **Residual fidelity and transport**: Restored serialized client residuals, implemented full-range $[-255, 255]$ signed mapping, preserved true server predictor base $P_s$, verified independent client output scoring without source frames, and measured high-fidelity H0 (44.07 dB PSNR-Y / 94.77 VMAF). |
