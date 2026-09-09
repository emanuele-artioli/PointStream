# Evaluation Area

**Evidence Revision**: Reconciled through PR #69 (`648325b`) and PR #72 (`bc09184`).
**Owned Scope**: `src/pipeline/reconstruction/quality.py`, `experiments/tier/`, `src/contracts/lattice.py`.

---

## 1. Current State

### Bounded codec pilot controller (draft PR #82)

`experiments/jobs/codec.py` adds pilot, longer-clip confirmation, and final stages
around paired anchor/PointStream ladders. QP and joint JPEG/QP payload spacing can
widen only inside an explicit policy. Worker timeouts, saved decisions and
fail-closed evidence checks stop expensive stages when pilots are invalid or
uninformative. Outputs remain exploratory and uncitable. Implementation is
pending new regression-test approval and CI; no GPU result is claimed.

The [long-job protocol](../workflow/long-jobs.md) also records the proposed
scene-sanity / same-video / cross-video / frozen-test training progression.
The old training campaign evaluator remains retired and is not launch-ready.

The Gate A 48-frame native run (#69) completed the first full-system rate–distortion measurement on real tennis footage. While validating pipeline integrity, it confirmed that Gate A is **not passed yet** under the legacy configuration due to:
1. High intra bitfloor in `libaom` background plate (~262 KB).
2. Uncompressed JPEG foreground appearance crops (~4–25 KB per actor per keyframe).
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
| Synthetic Tier Tests | #23 (`d3d7890`) | Synthetic 3-frame tier path test established as CI regression gate. |
| Ladder Plumbery | #65 (`6bdf2c3`), #66 (`9a1e0b5`) | Sweep infrastructure and anchor pairing harness created. |
| 48-Frame Native Run | #69 (`648325b`) | Full-system baseline evaluated. Identified background/appearance bottlenecks. |
| Fast Eval Strategy | #71, #72 | Two-tier protocol adopted; piped FFmpeg streaming proposed. |

---

## 3. Next Actions

| ID | Status | Dependencies | Source | Description & Acceptance Criteria |
|---|---|---|---|---|
| `EVAL-ACT-05` | In progress | Test-scope approval | Bounded pilot controller | Validate spacing, result identity, budget and promotion gates; then run a calibrated small real pilot under a separately chosen run policy. |
| `EVAL-ACT-01` | Ready | None | #71, #72 | **Piped in-memory metric computation**: Replace `_write_png_clip` disk writes with direct stdin streaming (`-f rawvideo -pix_fmt rgb24 ...`) to `ffmpeg` and enable `n_threads=16` for `libvmaf`. Acceptance: Metric calculation time reduced by ≥5× without altering score values on reference clips. |
| `EVAL-ACT-02` | Blocked | `CODEC-ACT-01`, `EVAL-ACT-01` | #72 | **Amortization & rate sweep (Tier 1 PSNR)**: Sweep QP ladders across 48, 96, and 192 frames with VVC/SVT background plate. Acceptance: Establish operating range where PointStream PSNR exceeds AV1/VVC anchors at matched bitrate. |
| `EVAL-ACT-03` | Blocked | `EVAL-ACT-02` | #72 | **Tier 2 full-metric confirmation**: Run PSNR-Y, SSIM, VMAF, and LPIPS on the winning configuration from `EVAL-ACT-02`. Acceptance: Complete three-axis report (size, quality, speed) with two-sided pre-run bounds and null controls. |
| `EVAL-ACT-04` | Ready (previously D-CODEC-PRESETS) | None | `plans/DEFERRED.md` | **Anchor preset standardization**: Document exact FFmpeg command lines, presets, and versions for AV1 (`libsvtav1`/`libaom`) and VVC (`libvvenc`). Acceptance: Explicit, reproducible anchor scripts checked into repository. |
