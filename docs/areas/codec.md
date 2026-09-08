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
- **Background Plate Bitfloor**: `src/components/background/stream.py` hardcoded `StreamCodec` to `libaom-av1`. In `libaom-av1`, intra 4K frames encounter an artificial bitfloor refusing to quantize below ~260 KB (CRF 63 yielded 268,949 bytes).
  - Benchmarked alternatives on the same 4K canvas:
    - `libaom-av1 (CRF 63)`: 268,949 bytes (262.6 KB)
    - `SVT-AV1 (QP 63)`: 41,162 bytes (40.2 KB)
    - `VVC libvvenc slower (QP 63)`: 5,347 bytes (5.2 KB)
    - `VVC libvvenc slower (QP 55)`: 14,336 bytes (14.0 KB)
- **Actor Appearance Crops**: Currently uses baseline JPEG (`CompressedImageAppearance` in `src/components/appearance/compressed.py`), spending 4–25 KB per crop with ringing artifacts below quality 40. Native OpenCV WebP encoding (`cv2.imencode('.webp', crop, [cv2.IMWRITE_WEBP_QUALITY, q])`) achieves 35–50% bitrate reduction with improved edge preservation.

---

## 2. Key Decisions & Evidence Anchor

| Topic | PR / Commit | Decision & Status |
|---|---|---|
| Sidecar encoders | #36 (`6975dc9`) | Standalone AV1 and VVC intra sidecar encoders implemented. |
| Background canvas | #45 (`fbb463d`), #50 (`32729a9`) | Canonical offline background stitched representation established. |
| Fallback mechanism | #52 (`83a7587`) | Explicit conventional video fallback path added for ineligible segments. |
| Native 48-frame run | #69 (`648325b`) | Measured whole-system rate/quality; identified background & appearance byte floor. |
| Competitive regime | #70–#72 | Proposed VVC/SVT background sidecars and WebP/AVIF appearance crops. |

---

## 3. Next Actions

| ID | Status | Dependencies | Source | Description & Acceptance Criteria |
|---|---|---|---|---|
| `CODEC-ACT-01` | Ready | None | #70, #71 | **Connect VVC intra / SVT-AV1 to background stream**: Update `src/components/background/stream.py` to route through `libvvenc` or SVT-AV1. Acceptance: 4K background plate encodes to <15 KB at acceptable PSNR; decodes cleanly in receiver pipeline. |
| `CODEC-ACT-02` | Ready | None | #71, #72 | **Add WebP/AVIF appearance crops**: Add WebP/AVIF encoder in `src/components/appearance/compressed.py`. Acceptance: Crop byte size reduced by ≥30% at matched crop PSNR; no regression in client assembly. |
| `CODEC-ACT-03` | Proposed | `CODEC-ACT-01`, `02` | #72 | **Payload ledger simplification**: Streamline byte allocation tracking across components. Acceptance: Every frame payload maps strictly to background, appearance, motion, or residual bytes without unallocated overhead. |
| `CODEC-ACT-04` | Previously deferred (D5) | None | `plans/DEFERRED.md` | **Coded fallback verification**: Verify behavior when semantic tracking fails. Acceptance: Clean switch to conventional intra/inter coding without crash or pipeline desynchronization. |
