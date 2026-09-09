# Codec Area

**Evidence Revision**: Reconciled through PR #72 (`bc09184`) and PR #73.
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
- **Actor Appearance Crops**: Currently uses baseline JPEG (`CompressedImageAppearance` in `src/components/appearance/compressed.py`), spending 4–25 KB per crop with ringing artifacts below quality 40. Native OpenCV WebP encoding is a candidate to benchmark. The proposed 35–50% reduction is unverified at matched quality on this corpus; it is not an established result.

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
| Gate A Rate Ladder | outputs/gate-a-vvc-webp-n96-run2 | Characterized C0–C3 across 192 frames @ 4K 24 fps (48.7–124.4 kB / 49.9–127.4 kbps) establishing competitive win below AV1 bitrate floor and beating VVC low-rate collapse. |

---

## 3. Next Actions

| ID | Status | Dependencies | Source | Description & Acceptance Criteria |
|---|---|---|---|---|
| `CODEC-ACT-01` | Complete | None | #70, #71, #75, #80 | **Connect VVC intra / SVT-AV1 to background stream**: Implemented in `src/components/background/stream.py` with causal prefix intra-refresh stability. |
| `CODEC-ACT-02` | Complete | None | #71, #72, #75 | **Add WebP/AVIF appearance crops**: Implemented WebP encoding in `src/components/appearance/compressed.py` with matched-quality savings and client decode round trip. |
| `CODEC-ACT-03` | Ready | `CODEC-ACT-01`, `02` | #72 | **Payload ledger simplification**: Streamline byte allocation tracking across components. Acceptance: All transmitted bytes reconcile to the serialized payload, including background, appearance, motion, residual, fallback, headers, and container/metadata overhead; do not hide overhead inside a component saving. |
| `CODEC-ACT-04` | Previously deferred (D5) | None | `plans/DEFERRED.md` | **Coded fallback verification**: Verify behavior when semantic tracking fails. Acceptance: Clean switch to conventional intra/inter coding without crash or pipeline desynchronization. |
