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

---

## Wave 1–3 Overnight Execution & Final Receipts (2026-11-24 / 2026-11-25)

### V5 — Nine-Configuration Speed–Quality Pareto & Stabilization Study (`tomm_v5_pareto_pilot.py`, `tomm_v5_score.py`)
- **GPU Jobs:**
  - `20261124T112219Z-fe23784a` (`gpu6` RTX 6000 Ada, shared, `partial.tar` sha256 `0943b0216e06a2b45bc356c723053844dad7ed921255e119986253e0256fe1ca`; ~0.67 GPU-h): completed all 7 Pareto configs across `match102_near` & `match102_far` (48 frames each = 96 observations).
  - `20261124T121722Z-89fd9610` (`gpu6` RTX 6000 Ada, COMPLETE, `published.tar` sha256 `de6d27b2e21452b31b830f49ce6818df9aa501afef5b459a0017d70c49b1e575`; ~0.25 GPU-h, `timing_contaminated: true`): [v5-stab-summary.json](v5-stab-summary.json) recorded `ddim30_512_ov4` `674.47 s` (`353.47 + 321.00`), `dpm10_512_ov8_smooth` `206.51 s` (`76.04 + 130.47`, **`3.27×` faster**), `dpm10_512_ov8_smooth_corr05` `208.32 s` (`77.50 + 130.82`, `3.24×` faster).
  - Exclusive timing spec (`v5-timing-exclusive-spec.json`, using new `--units` flag in `experiments/tennis/tomm_v5_pareto_pilot.py`) was prepared and validated (`ruff`), but not submitted because all Ada/A6000 hosts (`gpu2`, `gpu3`, `gpu5`, `gpu6`) remained continuously occupied by foreign GPU compute processes; shared Ada runtimes + exact step NFE ratios are reported transparently.
- **CPU Scorer Jobs:**
  - `20261124T122451Z-30419ed5` (COMPLETE on `gpu3`, 7-config interim score).
  - `20261124T210801Z-0ab311f9` (COMPLETE on `gpu1`, `571.99 s` scoring / `12.3 min` total) -> [v5-pareto-scores.json](v5-pareto-scores.json) (all 9 configs scored uniformly on `n_dists=96`, `n_flow=94`, `tlpips_n=22`):

| Configuration | Masked DISTS | Masked PSNR (dB) | `self_flow_error` | Seam / Interior `self_flow` | `tLPIPS` Mean | `tLPIPS` Seam / Interior | Speedup / Ada Wall Time |
|---|---:|---:|---:|---:|---:|---:|---|
| `ddim30_512_ov4` (base) | `0.3032` | `12.87` | `17.45` | `17.46 / 17.45` | `0.0593` | `0.0792 / 0.0549` | `1.00×` (`674.47 s*`) |
| `dpm15_512_ov4` | `0.3036` | `12.80` | `17.96` | `17.66 / 18.04` | `0.0580` | `0.0767 / 0.0538` | `2.0×` NFE |
| `dpm10_512_ov4` | `0.3035` | `12.88` | `17.69` | `17.67 / 17.69` | `0.0586` | `0.0757 / 0.0547` | `3.0×` NFE |
| `unipc10_512_ov4` | `0.3053` | `12.87` | `17.62` | `17.77 / 17.58` | `0.0598` | `0.0776 / 0.0558` | `3.0×` NFE |
| `dpm6_512_ov4` | `0.3075` | `13.09` | `16.87` | `16.93 / 16.85` | `0.0555` | `0.0706 / 0.0522` | `5.0×` NFE |
| `dpm10_384_ov4` | `0.3109` | `12.85` | `15.81` | `16.41 / 15.65` | `0.0553` | `0.0767 / 0.0505` | `3.0×` NFE, `0.54×` px |
| `dpm10_256_ov4` | `0.3329` | `12.04` | `15.53` | `17.58 / 14.97` | `0.0632` | `0.0944 / 0.0563` | `3.0×` NFE, `0.24×` px |
| `dpm10_512_ov8_smooth` | `0.3041` | `12.92` | `17.51` | `17.20 / 17.59` | `0.0571` | **`0.0596`** / `0.0566` | **`3.27×`** (`206.51 s*`) |
| `dpm10_512_ov8_smooth_corr05` | `0.5610` | `6.67` | `36.54` | `34.68 / 37.04` | `0.0663` | `0.0744 / 0.0645` | `3.24×` (`208.32 s*`) |

- **Takeaways:** `dpm10_512_ov8_smooth` achieves **`3.27×` speedup** within `+0.0009` DISTS while cutting chunk-boundary `tLPIPS` seams by **`24.8%`** (`0.0792 -> 0.0596`, shrinking the seam-over-interior jump from `+44.3%` to `+5.3%`); `dpm6_512_ov4` reaches `5×` NFE reduction within `+0.0043` DISTS; naive correlated initial noise (`corr05`) degrades fidelity (`0.5610` DISTS).

### Track A1 / Halle / A3 / A4 / A6 Completion Receipts
- **A1.2 Unmeasured TEST Court Audit (`20261124T112632Z-9732c726`, COMPLETE on `gpu5`):** `published.tar` sha256 `281a9f82dfb3d9fc1ac1c3295398466288a2aa05fa8425986a34e629f25fc66e` -> [a1-unmeasured-courts.json](a1-unmeasured-courts.json): **5/5 `court_eligible: True`** (`match107` 240/240 @ 60 fps, `match131` 228/240 @ 25 fps, `match104` 240/240 @ 60 fps, `match130` 240/240 @ 25 fps, `match126` 197/240 @ 25 fps). Companion `test_joint_features` attempts on `match107`/`match131` (`20261124T120458Z-672328b9`, `20261124T120746Z-9fe8835f`, `20261124T121814Z-3fbcbf68`) OOMed below `>= 26 GiB` (`sam3.1_multiplex.pt` peak `~24.8 GiB`).
- **Halle `match105` Status:** Sender verified (`20261123T115652Z-1c5a006d`) & far player generated (`20261123T120635Z-5a1a9720`); prior near-player exclusive runs (`20261124T112312Z-7c3f01bf`, `20261124T210849Z-924b8f4b`) contended, `20261124T235423Z-61291fae` (`smoke=44`) hit `2680 s` forecast cap, `20261125T001028Z-a88734ca` (`--full-cap 3600`) violated `common_receiver_job.py` (`<=2700`), `20261125T001821Z-cc12a19c` (`smoke=64`, `gpu5` 97% contended) timed out at 550 s, and `20261125T003436Z-83275e97` (`smoke=64`, `gpu3` RTX A6000) failed smoke (`exit code 1`) under shared cluster contention; near-player synthesis stopped without further retry per budget rules. External baselines COMPLETE (`gpu2`): `20261124T235517Z-70597ed6` (`av1-640-280`, 13.70 kbps, `0.5744` DISTS, `22.81` dB -> [halle-score-av1-640-280.json](halle-score-av1-640-280.json)), `20261124T235624Z-f6db4a58` (`av1-640-160`, 123.98 kbps, `0.3058` DISTS, `26.62` dB -> [halle-score-av1-640-160.json](halle-score-av1-640-160.json)), `20261124T235708Z-b242a19d` (`av1-320-160`, 41.85 kbps, `0.4105` DISTS, `22.54` dB -> [halle-score-av1-320-160.json](halle-score-av1-320-160.json)), `20261124T235811Z-1a2c100e` (`dcvc-480-q24`, 12.15 kbps, `0.4671` DISTS, `23.32` dB -> [halle-score-dcvc-480-q24.json](halle-score-dcvc-480-q24.json)).
- **A1.2 5-Match Pose Extension — COMPLETE (12/12 TEST Matches Measured, 0 Unmeasured):**
  - Feature extraction: `match104` (`20261125T000108Z-5fb93d8a`), `match130` (`20261125T000212Z-5de76453`), `match126` (`20261125T000312Z-3f097251`) COMPLETE on `gpu3`.
  - Crop pose audits: `match107` (`20261124T235912Z-a823f726`, 240/240 paired), `match131` (`20261125T000001Z-ac70134c`, 221/240 paired, max gap 8), `match104` (`20261125T001539Z-52580342`, 240/240 paired vs. 26/240 whole-frame far), `match130` (`20261125T002427Z-66a24291`, 240/240 paired on 72-px far player vs. 69/240 whole-frame), `match126` (`20261125T003605Z-bac4c818`, 240/240 paired) -> [coverage-summary-12.json](coverage-summary-12.json), [pose-gap-audit.json](pose-gap-audit.json), [native-player-coverage.pdf](native-player-coverage.pdf).
  - Across all 11 court-eligible TEST matches (Monte-Carlo `match119` routed to `Conv`): **7/11** achieve full 240/240 paired coverage raw (`2,490/2,640 = 94.3%`), rising to **10/11** (`2,607/2,640 = 98.8%`) under the `<= 10`-frame interior interpolation bridge (`11/11` at 19-frame bridge).
- **A3.1 Sydney `match127` Pose Machine Task (`20261124T095725Z-43f16084`, COMPLETE):** [a3-sydney-pose-machine-summary.json](a3-sydney-pose-machine-summary.json): `YOLO26x-pose` gets `0.000` recall (`ordinary-320-warm`, `roi-320-floor`), `0.4896` (`dcvc-480-q26`: `0.9792` near, `0.000` far, `0.4045` OKS), `0.7917` (`joint-crf70`: `1.000` near, `0.5833` far, `0.7315` OKS), vs. direct `PSCP` **`1.000` recall**, **`0.9527` PCK@0.1**, **`0.9164` OKS** in `104.0 ms` CPU (vs. `3.74 s` full-frame `YOLO26x-pose`).
- **A4.1 Multi-Sport 24-Clip Transfer (`20261124T095801Z-9c00d626`, `20261124T112406Z-8ee77f9e`, COMPLETE):** [a4-badminton-12.json](a4-badminton-12.json) (`11/12` wins vs. both AV1 same-res & best; `0.24175` vs. `0.25080` / `0.26048` DISTS64; `21.684` vs. `21.310` dB) and [a4-tabletennis-12.json](a4-tabletennis-12.json) (`12/12` vs. same-res, `10/12` vs. best; `0.29387` vs. `0.30930` / `0.31278` DISTS64; `20.296` vs. `20.000` dB), zero out-of-bounds failures.
- **A6 Mode Selector & Amortization (12/12 Measured):** [a6-mode-selector-amortization.json](a6-mode-selector-amortization.json) (`warm_20.4`: 5 Conv / 0 Conv+Ball / 7 Full-Obj; `warm_26.0`: 1 Conv / 1 Conv+Ball / 10 Full-Obj; `cold_26.0`: 1 Conv / 11 Conv+Ball / 0 Full-Obj; `cold_200.0`: 1 Conv / 4 Conv+Ball / 7 Full-Obj).
- **MIT Visual Plate & Supplementary Comparison Video:** `experiments/tennis/tomm_build_mit_media.py` (CPU on `gpu1`) built synchronized Sydney (`match127`, players t=115, ball t=68) & Shanghai (`match102`, t=95) crop plate (`external-native-crops.pdf` in manuscript) and `1920x1080` H.264 $2\times 2$ comparison video ([supplementary-comparison-mit.mp4](supplementary-comparison-mit.mp4), `3,709,191 B`).
- **Cumulative GPU Budget Ledger:** ~2.5 GPU-h total spent against `6.0 GPU-h` cap (~0.92 V5 + ~0.55 Halle contended/smoke attempts + ~0.55 A1 5-match features/crop + ~0.48 prior smokes/tests).

### Final Validation Closures (2026-11-25)
- **Shanghai `17.713 kbps` `av1-640-warm` Pose Recall (`20261125T071808Z-4ef2f004`, `gpu5` CPU):** [`v7-shanghai-ordinary640-summary.json`](v7-shanghai-ordinary640-summary.json) measured full-frame `YOLO26x-pose` `detect_recall = 0.0104` (**0.010**, 1/96; `0.0208` near / `0.000` far), `pck_01 = 0.0008`, `oks = 0.0021`, `direct_joints = 1.0`.
- **Halle `match105` Near-Player COMPLETE (`20261125T131927Z-2d59f826`, `gpu6` RTX 6000 Ada, uncontended 24.5 min):** All 240 frames rendered (`render_and_save_seconds = 960.4 s`, `published.tar` sha256 `cd2e4e07453abd204161ffff12e13b871b632744d982daa2881ef1092e8e931b`, `generation.json` sha256 `c8123ea42ca6fb731b992a9cca8eaf4a774d0ca26511919a3918084460a87a94`). Prior exclusive retry `20261125T130031Z-800363b3` passed 64-frame smoke (`373.1 s`, forecast `1,903.3 s`) before contention stopped `full`.
- **Halle `match105` Full Compositor Finish (`20261125T134808Z-6caa0351`, in flight):** `halle-match105-finish.json` (`common_finish_al028_job`), compositing `20261123T120635Z-5a1a9720` (`far`) + `20261125T131927Z-2d59f826` (`near`) + sender `20261123T115652Z-1c5a006d` (`crf50`, `15.27 kbps` warm).
- **Sydney AVIF Q90 Extraction (`20261125T073908Z-b6fcc7b0`, `gpu6`):** Verified exact `avifenc -q 90 --speed 8 --jobs 1` SHA-256 parity with `sydney-crf70.json` and staged `sydney-q90-refs.tar`.
- **Matched Latency, Rendered Sydney Q90 & Rendered 5-Frame Gap (`20261125T075312Z-67505393` on `gpu5` Ada shared `timing_contaminated: true`, 24.8 min; scored by `20261125T082819Z-f14375d2` on `gpu6` CPU, 5.8 min):** [`matched-checks-gpu-summary.json`](matched-checks-gpu-summary.json) + [`matched-checks-scores.json`](matched-checks-scores.json).
  - Shanghai 48 frames (2 players): `ddim30_512_ov4` `564.51 s` (`1.00x`, `0.3032` DISTS, `0.0593` `tLPIPS`, `0.0792` seam); `dpm10_512_ov8_smooth` `196.19 s` (**2.88x faster**, `0.3041` DISTS, `0.0571` / `0.0595` seam); `dpm6_512_ov8_smooth` `122.57 s` (**4.61x faster**, `0.3070` DISTS, `0.0529` / `0.0607` seam); `dpm10_384_ov8_smooth` `97.76 s` (**5.77x faster**, `0.3084` DISTS, `0.0572` / `0.0687` seam).
  - Sydney 24 frames (`n=48`): `ref_q90_uninterrupted` (`0.2987` DISTS, `14.68 dB`) matches `ref_lossless_uninterrupted` (`0.2991` DISTS, `14.70 dB`).
  - Sydney 24-frame gap-5 (`gap=[9,14)`): `ref_lossless_gap5_bridge` (`0.2993` overall / `0.3022` in-gap DISTS) preserves uninterrupted fidelity (`0.2991` / `0.3023`), outperforming `ref_lossless_gap5_hold` (`0.3018` overall / `0.3124` in-gap).
