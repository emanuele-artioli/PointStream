# Module Scorecard: [Module Name]

- **Owner Lane**: [e.g. Antigravity Background / Antigravity Foreground / Cursor / Codex]
- **Source Scope**: [e.g. `src/components/background/`]
- **Input Artifact**: [schema and path, e.g. `raw_frames: 3840x2160x3 uint8`, `masks: npy bool`]
- **Output Artifact**: [schema and path, e.g. `canvas: WebP/VVC`, `homographies: float32`]
- **Last Evaluated**: [YYYY-MM-DD]
- **Current Verdict**: [RETIRE | ACTIVE_SEARCH | SATISFIED_FREEZE]

---

## 1. Triad Definitions

| Arm | Implementation | Rationale |
|---|---|---|
| **Null** | [Description of absent or baseline trivial implementation] | Baseline cost when module performs zero work |
| **Current** | [Description of shipped operational component] | Current production performance |
| **Oracle** | [Idealized ground truth or zero-cost reference] | Theoretical upper performance bound |

---

## 2. Whole-Codec Rate-Distortion Impact

Measurements at fixed rest-of-pipeline against paired VVC/AV1 conventional anchors.

### Short Horizon (48 frames @ 24 fps, e.g. `federer_djokovic/scene_007`)

| Arm | Module Bytes | Total Codec Bytes | PSNR-Y (dB) | SSIM | VMAF | Anchor Marginal Bytes (VVC) | Delta vs Anchor |
|---|---|---|---|---|---|---|---|
| **Null** | | | | | | | |
| **Current** | | | | | | | |
| **Oracle** | | | | | | | |

- **Headroom (Oracle - Current)**: [Bytes saved / Quality delta]
- **Anchor Clearance**: [Does Current or Oracle beat Anchor at this horizon?]

### Long Horizon (192 frames @ 24 fps, e.g. `alcaraz_highlights/scene_000` or `federer_djokovic/scene_007`)

| Arm | Module Bytes | Total Codec Bytes | PSNR-Y (dB) | SSIM | VMAF | Anchor Marginal Bytes (VVC) | Delta vs Anchor |
|---|---|---|---|---|---|---|---|
| **Null** | | | | | | | |
| **Current** | | | | | | | |
| **Oracle** | | | | | | | |

- **Headroom (Oracle - Current)**: [Bytes saved / Quality delta]
- **Anchor Clearance**: [Does Current or Oracle beat Anchor at this horizon?]

---

## 3. Decision Rule & Next Action

- **Criteria**:
  - If $\text{Oracle} \approx \text{Null}$: **RETIRE**. Module cannot justify its complexity at any horizon.
  - If $\text{Current} \approx \text{Null} \ll \text{Oracle}$: **ACTIVE_SEARCH**. High headroom, current implementation underperforming. Cap budget and test alternative in registry.
  - If $\text{Current} \approx \text{Oracle}$ (within 5% rate/quality): **SATISFIED_FREEZE**. Stop spending compute and agent effort.
  - If All-Oracle still loses to Anchor: Decomposition failure at this horizon.
- **Next Action**: [Concrete 1-sentence next action for this module's owner]
