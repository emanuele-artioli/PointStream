# Evaluation Area

**Evidence Revision**: Reconciled through PR #69 (`648325b`) and PR #72 (`bc09184`).
**Owned Scope**: `src/pipeline/reconstruction/quality.py`, `experiments/tier/`, `src/contracts/lattice.py`.

---

## 1. Current State

The Gate A 48-frame native run (#69) completed the first full-system rate–distortion measurement on real tennis footage. While validating pipeline integrity, it confirmed that Gate A is **not passed yet** under the legacy configuration due to:
1. Large `libaom` background plate under the tested settings (archived probe; not a codec-wide lower bound).
2. JPEG-compressed foreground appearance crops (~4–25 KB per actor per keyframe).
3. Slow metric evaluation writing uncompressed PNGs to disk for VMAF computation.

### Two-Tier Metric Protocol (#72)
To accelerate the configuration search while maintaining rigorous publication standards:
- **Tier 1 (Exploration & Tuning)**: Compute **PSNR only** directly in memory via NumPy (`<0.05` s per frame). This avoids disk I/O, prevents process timeouts, and allows rapid sweeping of quantization parameters and keyframe intervals.
- **Tier 2 (Paper Evidence)**: Run the full multithreaded metric suite—**PSNR-Y, SSIM, VMAF (`libvmaf` with `n_threads=16`), and LPIPS**—strictly on frozen winning candidate configurations.

### Amortization Hypothesis
Because the high-resolution background plate is transmitted once per scene, its effective bitrate contribution scales inversely with scene length:
$$\text{Bitrate}_{\text{background}} = \frac{\text{Plate Size (bytes)} \times 8 \times \text{FPS}}{\text{Number of Frames}}$$
With a 14.3 KB VVC intra background plate:
- **48 frames**: $14.3\text{ KB} / 48 = 0.30\text{ KB/frame}$
- **96 frames**: $14.3\text{ KB} / 96 = 0.15\text{ KB/frame}$
- **192 frames**: $14.3\text{ KB} / 192 = 0.075\text{ KB/frame}$

Evaluating over longer sequences (96 and 192 frames) is a core hypothesis for establishing a rate–distortion win against conventional temporal inter-coding.

---

## 2. Key Decisions & Evidence Anchor

| Topic | PR / Commit | Decision & Status |
|---|---|---|
| Synthetic Tier Tests | #23 (`ca0f75af30`) | Synthetic 3-frame tier path test established as CI regression gate. |
| Ladder Plumbery | #65 (`91b33e623f`), #66 (`606cf53893`) | Sweep infrastructure and anchor pairing harness created. |
| 48-Frame Native Run | #69 (`648325b`) | Full-system baseline evaluated. Identified background/appearance bottlenecks. |
| Fast Eval Strategy | #71, #72 | Two-tier protocol adopted; piped FFmpeg streaming proposed. |
| In-Memory Metric Acceleration | #77, #78, #79, #81 | Thread-local SSIM scratch buffers (176× speedup), Y4M piped VMAF streaming (80× speedup), and streamed closeness (memory down to <500 MB). |
| Gate A Tier 2 Evaluation | outputs/gate-a-vvc-webp-n96-run2 | Confirmed 192-frame (8.0s @ 4K 24 fps) rate ladder C0–C3 with PSNR-Y, SSIM, VMAF. Verified winning operating regime below AV1 bitrate floor and beating VVC low-rate perceptual collapse. |

---

## 3. Next Actions

| ID | Status | Dependencies | Source | Description & Acceptance Criteria |
|---|---|---|---|---|
| `EVAL-ACT-01` | Complete | None | #71, #72, #79 | **Piped in-memory metric computation**: Replaced `_write_png_clip` disk writes with direct stdin streaming to ffmpeg Y4M rawvideo and enabled `n_threads=16`. Measured 80× speedup on 4K clips with bit-identical scores to reference. |
| `EVAL-ACT-02` | Complete | `CODEC-ACT-01`, `CODEC-ACT-02`, `EVAL-ACT-01` | #72, #81 | **Amortization & rate sweep (Tier 1 PSNR)**: Fixed virtual memory exhaustion via streamed closeness; completed 192-frame sweeps on multi-scene 4K video. |
| `EVAL-ACT-03` | Complete | `EVAL-ACT-02` | #72, outputs/gate-a-vvc-webp-n96-run2 | **Tier 2 full-metric confirmation**: Evaluated PSNR-Y, SSIM, VMAF across C0–C3 ladder. 0 alarms, pre-registered rot bounds verified, null controls passed, decode speed ~13.5 fps on CPU. |
| `EVAL-ACT-04` | Ready (previously D-CODEC-PRESETS) | None | `plans/DEFERRED.md` | **Anchor preset standardization**: Document exact FFmpeg command lines, presets, and versions for AV1 (`libsvtav1`/`libaom`) and VVC (`libvvenc`). Acceptance: Explicit, reproducible anchor scripts checked into repository. |
