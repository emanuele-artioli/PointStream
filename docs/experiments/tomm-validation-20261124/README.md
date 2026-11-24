# TOMM Validation Sprint (2026-11-24)

Pre-registered diagnostics and pilots executed before locking Week 1–2 GPU compute.
Local V1, V2, V4 audits completed on 2026-11-24 (`pose-gap-audit.json`, `temporal-audit.json`);
V3 campus inventory verified on `gpu1.itec.aau.at`.

## Pre-Registered Fleet CPU Experiments (V6, V7, V8)

### V6 — Temporal Self-Flow, Context-Seam, and tLPIPS Diagnostic (`tomm_temporal_self_flow.py`)
- **Question:** How much of PointStream's higher source-flow temporal residual on Shanghai `match102` (`27.18` vs. `19.77` AV1 / `18.19` DCVC-UF) comes from baseline motion-damping vs. genuine frame-to-frame synthesis jitter/flicker or 24-frame context-window seams?
- **Hypothesis:** Under each arm's *own* estimated optical flow (`self_flow_error`) and frame-to-frame perceptual change (`tLPIPS`), PointStream's flicker penalty shrinks relative to source-flow error, while still showing measurable frame-to-frame synthesis noise compared to AV1/DCVC-UF; 24-frame context seams (`t >= 20, t % 20 in {0..4}`) are a minor contributor (`<10%` delta) compared to continuous synthesis noise.
- **Competing explanation:** Self-flow and `tLPIPS` reveal severe frame-to-frame perceptual flicker or periodic 24-frame seam jumps that are hidden by DISTS/LPIPS spatial averages.
- **Decision rule:** Run on all 240 native frames (478 player transitions) of Shanghai `match102` (`av1-320-warm`, `roi-640-warm`, `dcvc-480-q34`, `joint-crf50`, plus `source`), with `tLPIPS` evaluated on 48 transitions (`t = 5, 10, ..., 235`) + 11 context boundary ticks (`t = 20, 40, ..., 220`). If PointStream self-flow MAE or `tLPIPS` is >1.5× DCVC-UF, prioritize pose smoothing + `overlap=8` in the V5 GPU pilot; if self-flow shows mostly spatial offset rather than flicker, pair both metrics in the paper's perception–distortion analysis.
- **Budget & Scale:** CPU only (`cpu_threads=2`), smoke `frames=10` (<=120 s), full `frames=240` (<=480 s), total cap 600 s.
- **Job ID:** *(pending submission)*

### V7 — Symmetric Player Pose Machine-Task Benchmark (`tomm_player_pose_machine.py`)
- **Question:** Does PointStream's paid 17-joint stream preserve downstream player-pose utility at ~17.8 kbps where post-decode vision pose estimation on AV1/DCVC-UF degrades?
- **Hypothesis:** Running an independent non-YOLO11x pose estimator (**DWPose ONNX** `yolox_l` + `dw-ll_ucoco_384`, verified on uncompressed source against **YOLO26x-Pose**) on ~17.8 kbps decoded AV1/DCVC-UF suffers sharp drops in detection recall, PCK@0.2, OKS, and ankle/hip localization—especially on the 97-px far player—while parsing PointStream's direct transmitted joints from `players.pscp` (`3ebc2e45...`) preserves high PCK/OKS and sub-10-px court-plane localization at <1 ms/frame CPU.
- **Competing explanation:** Modern DWPose still recovers accurate skeletons from 17–18 kbps DCVC-UF or AV1-ROI pixels, or PointStream's quantized transmitted joints (`q=4`) have poor anatomical agreement with DWPose/YOLO26x on uncompressed source.
- **Decision rule:** Evaluate 48 uniformly spaced frames (`96` player observations) of Shanghai `match102`. Smoke (`frames=4`) MUST verify DWPose-vs-YOLO26x agreement on uncompressed source (`PCK@0.2 >= 0.70`) to guarantee keypoint index alignment. Promote to full multi-match evaluation (Sydney, Halle, + new matches in Track A3) if direct transmitted joints achieve higher PCK@0.2 and ankle localization than both post-decode AV1 and DCVC-UF on the 97-px far player or overall.
- **Budget & Scale:** CPU only (`cpu_threads=4`), smoke `frames=4` (<=180 s), full `frames=48` (<=540 s), total cap 720 s.
- **Job ID:** *(pending submission)*

### V8 — MIT RacketVision Badminton Shuttlecock Oracle (`tomm_badminton_oracle.py`)
- **Question:** Does the paid PSBL coordinate+streak trajectory representation transfer beyond tennis and 120-fps table tennis to **badminton shuttlecocks** (different object shape, aerodynamics, 25-fps broadcast motion blur) at a 20.4 kbps ceiling?
- **Hypothesis:** Across 6 deterministically sampled official `test` clips from MIT-licensed `RacketVision/badminton.tar`, adding a paid shuttlecock stream (charging all position, visibility, appearance, wrapper, and checksum bytes) reduces `64×64` crop DISTS at equal/matched total bitrate (`<= 20.4 kbps` vs. best affordable ordinary SVT-AV1 across `320×180` and `640×360`) on >=5/6 clips.
- **Competing explanation:** White shuttlecocks against bright court lines or erratic smash blur cause the fixed synthetic streak to hurt or tie crop DISTS vs. AV1, or sparse 25-fps visibility reduces the benefit.
- **Decision rule:** Salted-SHA256 deterministic sample of 6 official `test` clips (`>=240` frames, `>=15` visible ball annotations). Smoke (`clips=1`), full (`clips=6`). Report per-clip and pooled crop DISTS, crop PSNR, central `12×12` PSNR, whole-frame PSNR, and serialization parity. If >=4/6 clips improve crop DISTS at matched `<=20.4 kbps`, promote a full **Badminton + Table Tennis real-tracker transfer study** (Track A4) in Week 2; otherwise diagnose whether tracked-tile mode is needed for shuttlecocks.
- **Budget & Scale:** CPU only (`cpu_threads=4`), smoke `clips=1` (<=180 s), full `clips=6` (<=600 s), total cap 780 s.
- **Job ID:** *(pending submission)*
