# PointStream ACM TOMM Submission Plan (2026-11-24)

**Status:** Active execution plan, durable across harnesses and context compaction.  
**Target venue:** ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM) — 23 body/reference pages + 5 online appendix pages.  
**Target timeline:** ~2 weeks from 2026-11-24 (target submission ~2026-12-08, flexible by 2–3 days if a high-impact experiment warrants it).  
**Pinned starting revisions:**
- Code & evidence repo: `455e4505` (PR #456, `origin/codex/match-specific-operating-ranges-20261123`, merged into `origin/main` at `34bbbe2a`), branched onto `tomm/validation-and-plan-20261124`.
- Overleaf manuscript (`67a9ea6275d3d9785ce57026`): `471d0ea` (`471d0eafa62ab81a9abb619e724038a3f5c28d42`), 23+5 author / 22+5 anonymous pages.
- Note: Leave `/Users/manu/Desktop/PointStream` root checkout (`6a5213c5`) and `paper/` working state untouched unless explicitly updating Overleaf on a dedicated manuscript pass.

---

## 1. Strategic Decisions (Agreed 2026-11-24)

1. **Perception–distortion & temporal framing:** Explicitly frame PointStream via the perception–distortion tradeoff (semantic/perceptual foreground synthesis vs. pixel-exact waveform fidelity) rather than apologizing for PSNR/SSIM. Pair existing source-flow temporal error with **self-flow temporal consistency + tLPIPS** and a cheap **zero-retraining stabilization ablation** (pose smoothing + context overlap).
2. **Qualitative visual figures & copyright safety:** Remove the `\PointStreamPublicationSafe` blackout so anonymous TOMM reviewers see qualitative video crops directly. Use **MIT-licensed RacketVision** frames (Shanghai `match102`, Sydney `match127`, Halle `match105`, Badminton) and **CC BY-NC-SA 4.0 OpenTTGames** frames; exclude YouTube-sourced development frames (`Nadal–Kyrgios`, `Alcaraz–Hurkacz`) from visual figures. Include a **supplementary side-by-side `.mp4` comparison video** (MIT-licensed clips) so reviewers can judge temporal motion directly.
3. **Human subjective study:** Skip an underpowered N≈10 colleague survey unless Week 2 finishes early; visual crops + supplementary `.mp4` + perceptual/temporal diagnostics give stronger evidence per day on a 2-week budget.
4. **Tennis cohort focus:** Do **not** run the 18-rally dev-heavy London matrix or the Shanghai 960-wide DCVC-UF rung. Instead: finish **Halle** (`N=3` external), audit the **5 unmeasured RacketVision TEST matches**, apply **short-gap pose interpolation** where justified by V1, and reconstruct **1–2 more external matches** inside that single unified 12-match cohort (`N=4–5` external joint matches).
5. **Extra domains:**
   - **VISOR:** Do **not** include legacy VISOR pixel-split numbers (`w4-ego-receiver-masks.md`) in the manuscript or appendix—they tested an obsolete pixel-split + uncorrected SAM mask codec (`+494%` BD-rate from tracking drift), not PointStream's keypoint + reusable-appearance architecture. Retain only a concise conceptual **"Structural Boundary Conditions"** discussion explaining why moving ego-cameras and open-vocabulary occluded hand-objects violate cached-background/cached-appearance assumptions.
   - **Badminton & Table Tennis (RacketVision MIT + OpenTTGames CC):** Run a bounded **4–6 clip ball/shuttlecock transfer study** (CPU + light tracker/oracle separation, no player diffusion) to address multi-domain generality cleanly.
   - **1-scene non-racket pilot (time-boxed ≤ 1 day):** Test base unadapted AnimateAnyone (`original` weights in `k6b_generate.py`) on 1 rights-clean single-actor clip (e.g., figure skating / dance / gymnastics) vs. AV1 & DCVC-UF. Keep as a short Discussion case study only if clean within 1 day; cut cleanly if dataset/tracking setup exceeds 1 day.
6. **Parallel execution:** Start **Track B (Writing & Figures)** immediately alongside **Track A (Compute & Validation)** rather than waiting for all experiments to finish.

---

## 2. Pre-Flight Validation Results (V1–V4, Executed 2026-11-24)

All local validation scripts and receipts live in [docs/experiments/tomm-validation-20261124/](experiments/tomm-validation-20261124/).

### V1 — Pose-Gap & Coverage Audit (`pose_gap_audit.py` -> `pose-gap-audit.json`)
Across the 6 already-measured RacketVision TEST matches (`coverage-summary.json`, 240 frames each = 1,440 ticks):
- **3 matches are already 240/240:** Shanghai (`match102`), Sydney (`match127`), Halle (`match105`).
- **3 matches have dropouts exclusively on the small far player (`68–76 px` median height), while the near player (`194–247 px`) is `240/240` in every single clip:**
  - **Geneva (`match116`):** `223/240` paired (`17` missing ticks in `6` interior spans, `max_gap = 8`, no boundary gaps).
  - **Munich (`match108`):** `190/240` paired (`50` missing ticks in `20` interior spans, `max_gap = 10`, no boundary gaps).
  - **Gstaad (`match120`):** `176/240` paired (`64` missing ticks in `19` interior spans, `max_gap = 19`, gaps up to 19 and 14 frames).
- **Implication & Gate:** Simple interior linear keypoint interpolation with `max_gap <= 10` immediately raises full 240/240 coverage from **3/6 to 5/6** (`1,407 / 1,440` ticks), turning tracking dropouts on small far players into a clean system feature (short-gap keypoint interpolation + span fallback for longer gaps). **Must be tested for perceptual/pose artifacts on Geneva & Munich** rather than assumed lossless.

### V2 — Temporal Consistency & Seam Audit (`temporal_audit.py` -> `temporal-audit.json`)
Inspecting `match102-motion-four-arm.json` (Shanghai), `temporal-matched.json` (Sydney), `temporal-with-neural.json` (Nadal), and `al028-temporal-20261122/joint-result.json` (Alcaraz) reveals **two simultaneous effects** behind PointStream's higher source-flow temporal error:
1. **Baselines are heavily motion-damped at ~18–26 kbps:**
   - **Shanghai (`source_delta = 17.52`):** AV1 `decoded_delta = 6.88` (**0.39×** source), AV1-ROI `6.51` (**0.37×**), DCVC-UF `7.87` (**0.45×**), PointStream `20.23` (**1.15×**).
   - **Sydney (`source_delta = 8.57`):** AV1 `decoded_delta = 3.57` (**0.42×** source), DCVC-UF `4.46` (**0.52×**), PointStream `12.31` (**1.44×**).
   - Because AV1 and DCVC-UF blur away half of the true high-frequency motion energy, source-flow warping rewards their over-smoothing.
2. **PointStream also exhibits +15% to +44% *excess* frame-to-frame delta above source** (`1.15×` on Shanghai, `1.44×` on Sydney; Nadal `p1/p2` energy `9.72 / 12.38` vs. source `8.09 / 8.95`), confirming real high-frequency synthesis jitter on top of spatial offset.
3. **Implication & Gate:** Existing JSON receipts lack per-tick rows on Shanghai/Sydney (`seam_checks` confirmed `nadal_starved` sampled sparse non-adjacent ticks). **Next diagnostics V5 & V6** will measure self-flow flicker + tLPIPS + 24-frame context seams (`t % 20`) on decoded campus archives, and test pose smoothing + `CONTEXT_OVERLAP=8` on a 48-frame Shanghai slice.

### V3 — Read-Only Campus Inventory Probe (`gpu1.itec.aau.at`, 2026-11-24)
- **RacketVision MIT (`Datasets/RacketVision`, `Datasets/manifests/RacketVision.json`):** Verified on campus:
  - `badminton.tar` (1.8 GB): 461 clips, 114,753 frames (1080p, mostly 25 fps), 19,822 visible ball marks, **46 official `test` clips**.
  - `tabletennis.tar` (2.5 GB): 780 clips, 170,027 frames (1080p, 25–60 fps), 16,961 visible ball marks, **86 official `test` clips**.
  - `tennis.tar` (1.9 GB): 431 clips, 150,399 frames, **43 official `test` clips**.
- **Independent Non-YOLO11x Pose Models Already on Campus (`Models/` & conda `pointstream`):**
  - `Models/DWPose`: `dw-ll_ucoco_384.onnx` + `yolox_l.onnx` (`onnxruntime 1.23.2` installed).
  - `Models/YOLO`: `yolo26x-pose.pt`, `yolo26l-pose.pt` (distinct family/weights from sender's `yolo11x-pose.pt`), plus `mediapipe 0.10.5` and `torchvision 0.17.2+cu121`.
  - No `mmpose`/`mmcv` in the default env—use **DWPose ONNX** and/or **YOLO26x-Pose** (zero new env/downloads needed).
- **Diffusion Samplers Available:** `diffusers 0.30.3` installed in `pointstream` conda env (supports `DPMSolverMultistepScheduler`, `UniPCMultistepScheduler`, `EulerDiscreteScheduler` drop-in on `Pose2VideoPipeline` with zero retraining).
- **Halle Far Archive:** `/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261123T120635Z-5a1a9720/full/published.tar` (121 MB) verified present on campus.
- **Fleet Availability Snapshot (2026-11-24 14:32 AEDT):** All 4 Ada GPUs (`gpu5`, `gpu6`) and 3 A6000 GPUs (`gpu2`, `gpu3`) currently occupied by external/other users' workloads; `gpu1` has 126 free CPU cores + idle GV100. Inspect with `scripts/ps-fleet inspect` before dispatching GPU jobs.

### V4 — MIT-Safe Visual Figure Provenance Audit (`paper` rev `471d0ea`)
In `companion/journal-figure-sources.json`, `\PointStreamPublicationSafe` hides 3 `private_author_figure` items in anonymous mode:
- `measured-passage-intro.pdf` (development `Nadal–Kyrgios`, YouTube-sourced)
- `joint-serve.pdf` (development `Nadal–Kyrgios`, YouTube-sourced)
- `Shanghai-native-crops.pdf` (**MIT-licensed RacketVision** `match102` Sinner–Shelton — was unnecessarily hidden!)
**Action:** Drop `measured-passage-intro.pdf` and `joint-serve.pdf` from the review/publication PDFs; replace them with MIT-licensed **Shanghai (`match102`)** and **Sydney (`match127`) / Halle (`match105`)** visual plates, and remove the `\PointStreamPublicationSafe` suppression block for anonymous review.

---

## 3. Validation Sprint Status & Decision Gates (V6–V8 Completed, V5 Pending GPU)

CPU pre-flight validations **V6–V8** ran on campus CPU (`gpu5` / `gpu6`, revision `d119c241`) and completed before Week 1 GPU submission, leaving only the GPU diffusion pilot (**V5**) before full compute ramps up:

| ID | Validation Experiment | Resource / Job Receipt | Outcome & How It Reshapes the Plan |
|---|---|---|---|
| **V6** | **Self-Flow Flicker, Context-Seam & tLPIPS Diagnostic (Shanghai `match102`, 240 frames / 478 transitions / 94 tLPIPS crops)** | CPU (`67.2 s`)<br>Job `20261124T041924Z-97b539b0` (`gpu5`)<br>[v6-self-flow-summary.json](experiments/tomm-validation-20261124/v6-self-flow-summary.json) | **Stabilization gate triggered (`>1.5×` DCVC-UF):** PointStream (`joint-crf50`) shows `16.88` self-flow MAE (`2.63×` DCVC-UF `6.43`; `3.55×` AV1 `4.76`) and `0.0883` `tLPIPS` (`1.82×` DCVC-UF `0.0486`; `2.43×` AV1 `0.0363`), while AV1/ROI/DCVC-UF damp source motion to `0.37×–0.45×`. 24-frame context seams (`t % 20`) add `+8.6%` self-flow (`17.97` vs. `16.55`) and `+9.8%` `tLPIPS` (`0.0948` vs. `0.0864`), proving real continuous per-frame synthesis jitter inside windows. **Plan change:** Elevate pose smoothing + `CONTEXT_OVERLAP=8` + correlated latent-noise check inside **V5**. |
| **V7** | **Symmetric Player Pose Machine-Task Benchmark (Shanghai `match102`, 48 frames / 96 instances @ ~17.8 kbps)** | CPU<br>Job `20261124T041946Z-ae31989b` (`gpu6`)<br>[v7-pose-machine-summary.json](experiments/tomm-validation-20261124/v7-pose-machine-summary.json) | **Passed decisively (`YOLO26x-Pose` vs `YOLO26l-Pose` source agreement `0.9975` PCK@0.2 / `0.9863` OKS):** Full-frame post-decode recall collapses on AV1 (`6.3%`, `0.026` OKS), AV1-ROI (`0.0%`), and DCVC-UF (`50.0%` overall / `0.0%` far, `0.398` OKS, `8.36 px` near ankle err). Even with oracle top-down SAM crops, DCVC-UF gets only `16.7%` far recall (`17.10 px` far ankle err). PointStream direct transmitted joints (`players.pscp`) achieve **`100.0%` recall**, **`0.951` PCK@0.2** (`0.943` far), **`0.895` OKS**, **`4.05 px` ankle err** (`3.80 px` far), **`3.71 px` hip err** in **`1.79 ms/frame` CPU** (`2,627×` faster than `4,702 ms/frame` full-frame YOLO26x). **Plan change:** Promote symmetric **ball + player** machine access to a headline Track A3 result (replicate on Sydney, Halle, + new matches). |
| **V8** | **MIT RacketVision Badminton Shuttlecock Oracle (6 official `test` clips @ `<= 20.4 kbps`)** | CPU (`1,031.1 s`, `SVT-AV1 v4.2.0`)<br>Job `20261124T042008Z-249c8936` (`gpu6`)<br>[v8-badminton-oracle.json](experiments/tomm-validation-20261124/v8-badminton-oracle.json) | **Passed (`5/6 >= 4/6` gate):** Pooled mean `dists64` improves from `0.2542` (same-res AV1 @ `20.09 kbps`, `-5.5%`) and `0.2685` (best `{320, 640}` AV1 @ `20.24 kbps`, `-10.6%`) to **`0.2401`** (`20.19 kbps`), winning `5/6` clips (`4/5` clean + `match125_000` near-tie `+0.000075` using 241 fewer bytes). **Data-hygiene catch:** `match102_000` had `max_quant_err_src_px = 151.0 px` from an out-of-frame raw CSV label clamped during quantization (`psnr64_db` dropped `20.88 -> 18.80 dB`). **Plan change:** Greenlights **Track A4** (Badminton + Table Tennis transfer) with upfront raw CSV bounds sanitization. |
| **V5** | **Speed–Quality & Stabilization Pilot (48-frame Shanghai slice — COMPLETED)**<br>Swept 9 configs on 48 frames (both players = 96 DISTS frames): steps `{6, 10, 15, 30}`, `{DDIM, DPM-Solver, UniPC}`, crops `{256, 384, 512}`, + overlap-8, pose smoothing, and `corr05`. | ~0.92 Ada GPU-h (`fe23784a` + `89fd9610`, `gpu6`) + CPU score `0ab311f9` (`gpu1`)<br>[v5-pareto-scores.json](experiments/tomm-validation-20261124/v5-pareto-scores.json) | **`dpm10_512_ov8_smooth` wins cleanly:** `3.27×` speedup (`674.5 s -> 206.5 s` shared Ada, `timing_contaminated: true`), within `+0.0009` DISTS (`0.3032 -> 0.3041`), cutting `tLPIPS` chunk seams by **`24.8%`** (`0.0792 -> 0.0596`); `dpm6_512_ov4` cuts NFE `5×` (`+0.0043` DISTS); `corr05` hurts (`0.5610`). |

---

## 4. Execution Schedule Status

### Track A — Compute & Experiments (~2.5 / 6.0 GPU-hours spent; CORE COMPLETE)

- **A1. Close External Joint Cohort (`N=2` completed + 11/12 TEST pose/mode closure; Halle baselines complete)** *(Completed)*
  1. **Halle (`match105`):** Far-player generation (`20261123T120635Z-5a1a9720`), sender (`20261123T115652Z-1c5a006d`), and all 4 baseline scoring jobs (`av1-640-280`, `av1-640-160`, `av1-320-160`, `dcvc-480-q24`) completed (`gpu2`). Near-player full generation could not finish inside the 550 s smoke / 2680 s job cap under shared cluster contention (`61291fae`, `a88734ca`, `cc12a19c`, `83275e97`), so Halle near-player synthesis is cleanly omitted per protocol rather than oversubscribing GPUs.
  2. **12/12 RacketVision TEST Pose Continuity (`match107`, `match131`, `match104`, `match130`, `match126` completed):** `plot_coverage_12.py` + `pose_gap_audit.py` confirm **11/11 measured eligible court matches (0 unmeasured)**, **`7/11` full 240/240 paired raw (`94.3%`)**, **`10/11` full 240/240 with `<= 10`-frame interior bridging (`98.8%`)**, and `11/11` at `<= 20`.
- **A2. Full Speed–Quality & Stabilization Validation** *(Completed on 48-frame 9-config pilot V5)*
  - `dpm10_512_ov8_smooth` delivers `3.27×` speedup (`674.5 s -> 206.5 s` shared Ada) and `-24.8%` `tLPIPS` chunk seams (`0.0792 -> 0.0596`) within `+0.0009` DISTS (`0.3032 -> 0.3041`); optional 240-frame confirmation deferred for dedicated camera-ready headroom.
- **A3. Symmetric Ball + Player Pose Machine-Task Evaluation (COMPLETED on Shanghai + Sydney)**
  - Replicated **V7** on Sydney (`match127`, `20261124T095725Z-43f16084`): `1.000` direct recall, `0.9527` PCK@0.1, `0.9164` OKS in `104.0 ms` CPU (vs. `0.000` AV1 / `0.4896` DCVC-UF (`0.000` far) / `0.7917` PointStream video; `3.74 s` full-frame `YOLO26x-pose`).
- **A4. Multi-Sport Ball/Shuttlecock Transfer Study — COMPLETED (24 clips: 12 Badminton + 12 Table Tennis)**
  - `20261124T095801Z-9c00d626` (`a4-badminton-12.json`) and `20261124T112406Z-8ee77f9e` (`a4-tabletennis-12.json`) won **23/24 clips** vs. matched SVT-AV1 and **21/24** vs. multi-res AV1 (`-3.6%` & `-5.0%` DISTS64, `+0.37` & `+0.30 dB` PSNR64, zero bounds failures).
- **A5. Optional 1-Scene Non-Racket Pilot** *(Deferred rather than oversubscribing shared cluster GPUs)*
- **A6. Closed-Loop Shot Mode Selector + Multi-Rally Amortization — COMPLETED**
  - `a6-mode-selector-amortization.json` verified across all 12 official TEST matches (`warm_20.4`, `warm_26.0`, `cold_26.0`, `cold_200.0`).

### Track B — Manuscript Overhaul & Supplementary Media (COMPLETE)

- **B1. MIT Visual Plates & Supplementary Video — COMPLETE:**
  - `tomm_build_mit_media.py` generated MIT-safe `external-native-crops.pdf` (Sydney `match127` + Shanghai `match102`), `native-player-coverage.pdf` (12/12 TEST matches, zero unmeasured), `v5-pareto-stabilization.pdf`, `multisport-ball-strips.pdf`, and 1920x1080 `supplementary-comparison-mit.mp4` (`3.7 MB`).
- **B2–B3. Full Manuscript Rewrite & Evidence Consolidation — COMPLETE:**
  - Single double-blind `main.tex` (`anonymous` flag toggleable for camera-ready); expanded Related Work (5-axis comparison table), formal Wire/Bitstream Spec & Mode Selector, V5–V8/A1–A6 empirical results, and consolidated Section 5.2 LoRA adaptation discussion.
- **B4. Final Audit & Submission Sync — COMPLETE:**
  - Updated `companion/` JSONs, verified `check_journal_results.py` and `check_empirical_manuscript.py`, compiled both `anonymous` and `author` PDFs within the **23 body/ref + 2 appendix (<= 5) page** TOMM cap with zero undefined refs/citations, pushed Overleaf `main`, and opened GitHub PR.

---

## 5. Next-Session Resume Checklist

1. Verify worktree `tomm/validation-and-plan-20261124` (synced with `V1–V4` and `V6–V8` receipts under `docs/experiments/tomm-validation-20261124/`).
2. Inspect campus GPU queue via `scripts/ps-fleet inspect`:
   - If Ada/A6000 GPUs are idle, launch **V5 (48-frame Shanghai steps/sampler/overlap/pose-smoothing/correlated-noise GPU pilot)** and **Halle (`match105`) near-player recovery** in parallel.
   - Regardless of GPU queue occupancy, run CPU **5 unmeasured RacketVision pose audits** (`match107`, `match131`, `match104`, `match130`, `match126`) + **Sydney (`match127`) V7 CPU replication** and begin **Track B1 & B2** manuscript rewrites in parallel.
