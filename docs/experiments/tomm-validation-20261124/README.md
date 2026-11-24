# TOMM Validation Sprint (2026-11-24)

Pre-registered diagnostics and pilots executed before locking Week 1–2 GPU compute.
Local V1, V2, V4 audits completed on 2026-11-24 (`pose-gap-audit.json`, `temporal-audit.json`);
V3 campus inventory verified on `gpu1.itec.aau.at`.

## Pre-Registered Fleet CPU Experiments (V6, V7, V8 — Completed)

### V6 — Temporal Self-Flow, Context-Seam, and tLPIPS Diagnostic (`tomm_temporal_self_flow.py`)
- **Question:** How much of PointStream's higher source-flow temporal residual on Shanghai `match102` (`27.18` vs. `19.77` AV1 / `18.19` DCVC-UF) comes from baseline motion-damping vs. genuine frame-to-frame synthesis jitter/flicker or 24-frame context-window seams?
- **Hypothesis:** Under each arm's *own* estimated optical flow (`self_flow_error`) and frame-to-frame perceptual change (`tLPIPS`), PointStream's flicker penalty shrinks relative to source-flow error, while still showing measurable frame-to-frame synthesis noise compared to AV1/DCVC-UF; 24-frame context seams (`t >= 20, t % 20 in {0..4}`) are a minor contributor (`<10%` delta) compared to continuous synthesis noise.
- **Competing explanation:** Self-flow and `tLPIPS` reveal severe frame-to-frame perceptual flicker or periodic 24-frame seam jumps that are hidden by DISTS/LPIPS spatial averages.
- **Decision rule:** Run on all 240 native frames (478 player transitions) of Shanghai `match102` (`av1-320-warm`, `roi-640-warm`, `dcvc-480-q34`, `joint-crf50`, plus `source`), with `tLPIPS` evaluated on 48 transitions (`t = 5, 10, ..., 235`) + 11 context boundary ticks (`t = 20, 40, ..., 220`). If PointStream self-flow MAE or `tLPIPS` is >1.5× DCVC-UF, prioritize pose smoothing + `overlap=8` in the V5 GPU pilot; if self-flow shows mostly spatial offset rather than flicker, pair both metrics in the paper's perception–distortion analysis.
- **Budget & Scale:** CPU only (`cpu_threads=1` via `single_cpu_campaign`), smoke `frames=10`, full `frames=240`. Actual full runtime: `67.21 s`.
- **Job & Receipt:** `20261124T041924Z-97b539b0` on `gpu5` (revision `d119c241`) -> [v6-self-flow-summary.json](v6-self-flow-summary.json). *(Prior failed attempts: `20261124T040900Z-cde87fd0` missing `tools/`, `20261124T041312Z-88c89a13` missing `"checks"` key in `validation.json`.)*
- **Observed Results (240 frames, 478 player transitions, 94 tLPIPS crops):**

| Arm | Source-Flow MAE | Self-Flow MAE | `decoded_delta / source_delta` | Self-Flow Seam / Interior | `tLPIPS` Mean (n=94) | `tLPIPS` Seam / Interior |
|---|---:|---:|---:|---:|---:|---:|
| `av1-320-warm` | `19.77` | `4.76` | `6.88 / 17.52` (`0.39×`) | `4.91 / 4.71` | `0.0363` | `0.0429 / 0.0343` |
| `roi-640-warm` | `20.12` | `4.34` | `6.51 / 17.52` (`0.37×`) | `5.02 / 4.13` | `0.0255` | `0.0246 / 0.0257` |
| `dcvc-480-q34` | `18.19` | `6.43` | `7.87 / 17.52` (`0.45×`) | `6.52 / 6.40` | `0.0486` | `0.0434 / 0.0502` |
| `joint-crf50` (PointStream) | `27.18` | **`16.88`** (`2.63×` DCVC) | `20.23 / 17.52` (`1.15×`) | `17.97 / 16.55` (`+8.6%`) | **`0.0883`** (`1.82×` DCVC) | `0.0948 / 0.0864` (`+9.8%`) |

- **Conclusion (Stabilization Gate Triggered):** Both `self_flow_error` (`2.63×` DCVC-UF) and `tLPIPS` (`1.82×` DCVC-UF) exceed the `1.5×` threshold. While 24-frame context seams add ~`8.6%–9.8%` extra error, interior frames (`16.55` self-flow, `0.0864` `tLPIPS`) already contain genuine per-frame synthesis jitter rather than merely spatial alignment offset vs. source. **Action:** Elevate temporal pose smoothing + `CONTEXT_OVERLAP=8` + correlated latent-noise check inside the Week 1 **V5** GPU pilot.

---

### V7 — Symmetric Player Pose Machine-Task Benchmark (`tomm_player_pose_machine.py`)
- **Question:** Does PointStream's paid 17-joint stream preserve downstream player-pose utility at ~17.8 kbps where post-decode vision pose estimation on AV1/DCVC-UF degrades?
- **Hypothesis:** Running an independent non-YOLO11x pose estimator (**YOLO26x-Pose**, cross-checked on uncompressed source against **YOLO26l-Pose**) on ~17.8 kbps decoded AV1/DCVC-UF suffers sharp drops in detection recall, PCK@0.2, OKS, and ankle/hip localization—especially on the 97-px far player—while parsing PointStream's direct transmitted joints from `players.pscp` (`3ebc2e45...`) preserves high PCK/OKS and sub-10-px court-plane localization at <2 ms/frame CPU.
- **Competing explanation:** YOLO26x-Pose still recovers accurate skeletons from 17–18 kbps DCVC-UF or AV1-ROI pixels (especially with top-down crops), or PointStream's quantized transmitted joints (`q=4`) have poor anatomical agreement with YOLO26x on uncompressed source.
- **Decision rule:** Evaluate 48 uniformly spaced frames (`96` player observations) of Shanghai `match102`. Smoke (`frames=4`) MUST verify YOLO26x-vs-YOLO26l agreement on uncompressed source (`PCK@0.2 >= 0.70`). Promote to full multi-match evaluation (Sydney, Halle, + new matches in Track A3) if direct transmitted joints achieve higher PCK@0.2 and ankle localization than both post-decode AV1 and DCVC-UF on the 97-px far player or overall.
- **Budget & Scale:** CPU only (`cpu_threads=1`), smoke `frames=4`, full `frames=48`.
- **Job & Receipt:** `20261124T041946Z-ae31989b` on `gpu6` (revision `d119c241`) -> [v7-pose-machine-summary.json](v7-pose-machine-summary.json). *(Prior attempt `20261124T041334Z-7074c97d` lacked `onnxruntime` in packed env.)*
- **Reference Sanity & Timing:**
  - Uncompressed source agreement (`YOLO26x-Pose` vs. `YOLO26l-Pose`, `n=96`): `100.0%` recall, `0.9917` PCK@0.1, `0.9975` PCK@0.2, `0.9863` OKS, `1.43 px` ankle error, `1.50 px` hip error.
  - Decode/inference latency on CPU: Direct `players.pscp` parse `429.55 ms` for 240 frames (**`1.79 ms/frame`**) vs. full-frame `YOLO26x-Pose` **`4,702 ms/frame`** (`2,627×` slower) and oracle 2-crop `YOLO26x-Pose` **`1,372 ms/frame`** (`767×` slower).
- **Observed Results — Full-Frame Post-Decode `YOLO26x-Pose` vs. Direct Transmitted Joints (n=96):**

| Arm | Overall Recall | Near / Far Recall | Overall PCK@0.2 (Near / Far) | Overall OKS (Near / Far) | Ankle Err px (Valid) | Hip Err px (Valid) |
|---|---:|---|---:|---:|---:|---:|
| `av1-320-warm` (~17.7 kbps) | `6.3%` | `12.5% / 0.0%` | `0.038` (`0.077 / 0.000`) | `0.026` (`0.053 / 0.000`) | `22.26` (`6`) | `15.07` (`6`) |
| `roi-640-warm` (~17.7 kbps) | `0.0%` | `0.0% / 0.0%` | `0.000` (`0.000 / 0.000`) | `0.000` (`0.000 / 0.000`) | `—` (`0`) | `—` (`0`) |
| `dcvc-480-q34` (~17.4 kbps) | `50.0%` | `100.0% / 0.0%` | `0.472` (`0.945 / 0.000`) | `0.398` (`0.796 / 0.000`) | `8.36` (`48`, near only) | `10.81` (`48`) |
| `joint-crf50` decoded video (~17.8 kbps) | `84.4%` | `100.0% / 68.8%` | `0.836` (`0.993 / 0.679`) | `0.760` (`0.907 / 0.614`) | `4.30` (`81`; far `2.95`) | `5.09` (`81`) |
| **PointStream Direct Joints (`players.pscp`)** | **`100.0%`** | **`100.0% / 100.0%`** | **`0.951`** (`0.959 / 0.943`) | **`0.895`** (`0.924 / 0.866`) | **`4.05`** (`96`; far `3.80`) | **`3.71`** (`96`) |

- **Observed Results — Oracle Top-Down SAM-Crop `YOLO26x-Pose` Control (n=96):**
  - Even with oracle bounding-box localization, AV1 (`4.2%` recall, `0.022` OKS), AV1-ROI (`6.3%` recall, `0.012` OKS), and DCVC-UF (`58.3%` overall / **`16.7%` far recall**, `0.410` OKS, **`17.10 px` far ankle error**) fail severely on the 97-px far player, whereas PointStream decoded video reaches `100.0%` recall (`0.970` PCK@0.2, `0.876` OKS, `4.86 px` ankle) and PointStream direct joints achieve `100.0%` recall (`0.951` PCK@0.2, `0.895` OKS, `4.05 px` ankle, `3.71 px` hip) in `1.79 ms/frame`.
- **Conclusion (Gate Passed Decisively):** Promote symmetric **ball + player pose machine access** to **Track A3** (replicate V7 on Sydney `match127`, Halle `match105`, and newly added external matches, plus a small manual keypoint verification on uncompressed source to close `HOLE(journal-anatomy)`).

---

### V8 — MIT RacketVision Badminton Shuttlecock Oracle (`tomm_badminton_oracle.py`)
- **Question:** Does the paid PSBL coordinate+streak trajectory representation transfer beyond tennis and 120-fps table tennis to **badminton shuttlecocks** (different object shape, aerodynamics, 25-fps broadcast motion blur) at a 20.4 kbps ceiling?
- **Hypothesis:** Across 6 deterministically sampled official `test` clips from MIT-licensed `RacketVision/badminton.tar`, adding a paid shuttlecock stream (charging all position, visibility, appearance, wrapper, and checksum bytes) reduces `64×64` crop DISTS at equal/matched total bitrate (`<= 20.4 kbps` vs. best affordable ordinary SVT-AV1 across `320×180` and `640×360`) on >=5/6 clips.
- **Competing explanation:** White shuttlecocks against bright court lines or erratic smash blur cause the fixed synthetic streak to hurt or tie crop DISTS vs. AV1, or sparse 25-fps visibility reduces the benefit.
- **Decision rule:** Salted-SHA256 deterministic sample of 6 official `test` clips (`>=240` frames, `>=15` visible ball annotations). Smoke (`clips=1`), full (`clips=6`). Report per-clip and pooled crop DISTS, crop PSNR, central `12×12` PSNR, whole-frame PSNR, and serialization parity. If >=4/6 clips improve crop DISTS at matched `<=20.4 kbps`, promote a full **Badminton + Table Tennis real-tracker transfer study** (Track A4) in Week 2; otherwise diagnose whether tracked-tile mode is needed for shuttlecocks.
- **Budget & Scale:** CPU only (`cpu_threads=1`), smoke `clips=1`, full `clips=6`. Actual full runtime: `1,031.07 s`.
- **Job & Receipt:** `20261124T042008Z-249c8936` on `gpu6` (revision `d119c241`) -> [v8-badminton-oracle.json](v8-badminton-oracle.json). *(Prior attempt `20261124T041356Z-17059923` missing `"checks"` key in `validation.json`; native tools recorded in receipt: `SVT-AV1 v4.2.0`, `ffmpeg 8.1.2`, `dav1d 1.5.4`.)*
- **Observed Results (6 MIT RacketVision `test` clips @ `<= 20.4 kbps`):**
  - **Pooled mean (`6` clips):** PointStream `dists64 = 0.2401` at `20.19 kbps` vs. same-res AV1 `0.2542` at `20.09 kbps` (**`-5.5%` DISTS**, **5/6 wins**; `same_base = 0.2545`) vs. best-affordable `{320, 640}` AV1 `0.2685` at `20.24 kbps` (**`-10.6%` DISTS**, **5/6 wins**).

| Clip ID | Visible Marks | Payload (B) | Max Quant Err Src (px) | PointStream (w, CRF, kbps) | PS `dists64` | AV1 Same-Res `dists64` | AV1 Best `dists64` | Result |
|---|---:|---:|---:|---|---:|---:|---:|---|
| `match114_000` | 40 | 164 | `1.40` | `640`, 67.5, `20.40` | **`0.2346`** | `0.2559` (`20.19`) | `0.2846` (`320`, `20.38`) | Win |
| `match133_000` | 48 | 194 | `1.37` | `640`, 65.25, `20.23` | **`0.2197`** | `0.2281` (`20.00`) | `0.2828` (`320`, `20.29`) | Win |
| `match11_000` | 45 | 184 | `1.36` | `640`, 64.25, `20.10` | **`0.2072`** | `0.2490` (`19.87`) | `0.2513` (`640`, `20.33`) | Win |
| `match135_000` | 49 | 198 | `1.38` | `320`, 61.75, `20.35` | **`0.3131`** | `0.3202` (`20.12`) | `0.3202` (`320`, `20.12`) | Win |
| `match125_000` | 48 | 194 | `1.40` | `640`, 67.0, `20.11` | `0.23066` | **`0.23058`** (`20.31`) | **`0.23058`** (`640`, `20.31`) | Near-tie (-241 B) |
| `match102_000` | 39 | 161 | **`151.00`** | `640`, 67.5, `19.95` | **`0.2352`** | `0.2413` (`20.02`) | `0.2413` (`640`, `20.02`) | Win *(anomalous CSV)* |

- **Data-Hygiene Catch (`match102_000`):** 5/6 clips show normal sub-2-px quantization parity (`max_quant_err_src_px <= 1.40 px`), whereas `match102_000` contains an out-of-frame raw CSV label that was clamped during quantization (`151.00 px`), dropping `psnr64_db` from `20.88 dB` to `18.80 dB`. Excluding `match102_000`, `4/5` clean clips still win cleanly (with the 5th clip `match125_000` at `+0.000075` DISTS while saving 241 bytes and beating its own `same_base` CRF 67.0 `0.2366 -> 0.2307`).
- **Conclusion (Gate Passed `5/6 >= 4/6`):** Greenlights **Track A4** (MIT RacketVision Badminton + Table Tennis transfer study in Week 2), with raw CSV bounds sanitization added upfront so out-of-frame ground-truth labels are filtered/flagged cleanly before encoding or scoring.
