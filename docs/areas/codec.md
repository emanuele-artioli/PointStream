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

---

## 3. Next Actions

| ID | Status | Dependencies | Source | Description & Acceptance Criteria |
|---|---|---|---|---|
| `CODEC-ACT-01` | Complete | None | #70, #71, #75, #80 | **Connect VVC intra / SVT-AV1 to background stream**: Implemented in `src/components/background/stream.py` with causal prefix intra-refresh stability. |
| `CODEC-ACT-02` | Complete | None | #71, #72, #75 | **Add WebP/AVIF appearance crops**: Implemented WebP encoding in `src/components/appearance/compressed.py` with matched-quality savings and client decode round trip. |
| `CODEC-ACT-03` | Ready | `CODEC-ACT-01`, `02` | #72 | **Payload ledger simplification**: Streamline byte allocation tracking across components. Acceptance: All transmitted bytes reconcile to the serialized payload, including background, appearance, motion, residual, fallback, headers, and container/metadata overhead; do not hide overhead inside a component saving. |
| `CODEC-ACT-04` | Previously deferred (D5) | None | `plans/DEFERRED.md` | **Coded fallback verification**: Verify behavior when semantic tracking fails. Acceptance: Clean switch to conventional intra/inter coding without crash or pipeline desynchronization. |
| `CODEC-ACT-05` | Ready | Coordinate EVAL-ACT-06 | 2026-09-09 code audit | **Residual fidelity and transport**: Gate A/B explicitly disabled residuals; shipped tiers enable them. Serialized client residuals remain unsupported; lossy uint8 offset clips large signed differences. Preserve the actual coded stream, validate full-range correction and independent output scoring, and measure high-fidelity residual-on curves. See [worker brief](../workflow/session/repair-residual-transport.md). |

PR #88 recovery repairs now complete the current CODEC-ACT-05 implementation: actual residual streams/full-range mapping, client-originated delivered scoring, serialized generation conditioning/reference data and complete envelope reconciliation are covered by integration tests and a native residual smoke test. This validates transport plumbing, not high-fidelity rate–distortion performance; run the residual-on ladder through the repaired client before claiming a codec result. See [evaluation audit](evaluation.md#pr-88-audit--2026-09-10) and [next dispatch](../workflow/session/submission-search.md).

## Current background decision — 2026-09-12

`CODEC-ACT-06` delivered a useful component prototype, not verified package
transport or a 94% matched-quality saving. Its geometry is estimated in bytes
while original float64 mappings drive rendering; source/cache identity and metric
scope need attention. The old table below is retained as exploratory observations,
including stage-only timings. `CODEC-ACT-07` remains open: test the user's still /
panorama / cleaned-video and separate fill axes under one bounded card, then
measure full-codec headroom before selecting a production representation. See
[audit](../history/antigravity-audit-2026-09-12.md) and
[handoff](../workflow/session/evaluation-handoff.md).

## Background experiment priority & findings — 2026-09-11

`CODEC-ACT-06` — **Complete** (PR #96, `0bdf0ef`, artifacts under `outputs/development-recovery/wave2-background-probe/`).
Evaluated three background representations on the 48-frame Federer sequence (`federer_djokovic/scene_007`, 48 frames @ 4K 24 fps) using a common foreground-removed frame stack (visible background preserved bit-identically, player mask filled via temporal composite plate, 0 uncovered holes):

| Representation | QP | Payload (B) | Side Data (B) | Total (B) | PSNR-Y Vis (dB) | SSIM Vis | Enc Time (s) | Dec Time (s) |
|---|---|---|---|---|---|---|---|---|
| still_frame0 | 47 | 31,804 | 10 | 31,814 | 20.06 | 0.8158 | 2.05 | 4.18 |
| registered_panorama | 47 | 30,626 | 1,742 | 32,368 | 26.22 | 0.9541 | 2.25 | 8.31 |
| cleaned_video | 47 | 74,178 | 10 | 74,188 | 31.18 | 0.9824 | 6.97 | 7.00 |
| still_frame0 | 32 | 126,184 | 10 | 126,194 | 20.02 | 0.8173 | 2.43 | 4.22 |
| registered_panorama | 32 | 118,157 | 1,742 | 119,899 | 26.97 | 0.9643 | 2.61 | 8.02 |
| cleaned_video | 32 | 364,212 | 10 | 364,222 | 38.98 | 0.9967 | 9.31 | 7.07 |

- **Verdict: Hypothesis SUPPORTED.** Still frame 0 is fundamentally limited by uncompensated camera motion (~40 px pan), pinning visible PSNR at ~20 dB and SSIM at 0.817 regardless of rate.
- **Registered panorama resolves this deficit efficiently**: Camera homographies gain +6.2 dB (QP 47) and +7.0 dB (QP 32), boosting SSIM to 0.954–0.964. Total package cost is 32.4 kB at QP 47 (saving 94% vs the legacy 529 kB plate).
- **Cleaned video** reaches higher quality (+12 dB over panorama at QP 32), but incurs a 3.0x byte multiplier and 3.6x encode time.
- **Historical Wave 2 proposal (qualified by audit above)**: investigate panorama integration; selection requires actual wire and matched-final-quality evidence.
