# BP57 — Confirmation Acquisition and Shot Pilot Report

**Auditor**: Antigravity  
**Date**: 2026-09-04  
**Worktree**: `/home/itec/emanuele/pointstream-bp57`  
**Branch**: `antigravity/bp57-confirmation-acquisition-pilot`  
**Base Commit**: `a59934a` (Merge pull request #62 from emanuele-artioli/codex/bp53-bp54-closeout)  
**Assigned Brief**: [`plans/BP57-confirmation-acquisition-pilot.md`](BP57-confirmation-acquisition-pilot.md)  
**Committed Manifest**: [`manifests/bp57-acquisition-pilot.json`](../manifests/bp57-acquisition-pilot.json)  

---

## 1. Executive Summary & Outcome

- **Outcome**: **COMPLETE** (Bounded physical acquisition and lightweight shot audit succeeded for both approved provisional sources within all operational caps; zero failures, zero retries, zero alarms).
- **Structured Manifest Totals**:
  - `submitted`: 2
  - `succeeded`: 2
  - `failed`: 0
  - `combined_downloaded_bytes`: 735398842 (701.33 MiB = 0.685 GiB vs. 10.0 GB cap, 6.85% utilized)
  - `retries`: 0
  - `alarms`: []
- **Core Results**:
  - **Fetched Intervals**:
    - Source 1 (AO 2024 Final, Sabalenka–Zheng): **39m 00s** (33:00–72:00, Set 2 complete) vs. 45m limit.
    - Source 2 (US Open 2023 Final, Gauff–Sabalenka): **35m 00s** (37:00–72:00, Set 2 complete) vs. 45m limit.
  - **Decoded Parameters & Scope**:
    - Source 1: Decodes at $1920 \times 1080$ / $25$ fps constant.
    - Source 2: Decodes at $1280 \times 720$ / $30000/1001$ fps constant (~29.97 fps).
    - *Native Resolution Scope*: The acquired sources decode at $1920 \times 1080$ / $25$ fps and $1280 \times 720$ / $30000/1001$ fps; they cannot independently confirm a native-4K-specific claim. Final eligibility depends on the frozen regime, prior-use audit, and later annotation checks.
  - **Shot Continuity & Candidate Yield**:
    - AO 2024 Set 2: 172 intervals $\ge 2.0$s; **77 court-like candidate intervals (automatic rally candidates)**, **23 candidates $\ge 16.0$s** (384 frames @ 24 fps).
    - US Open 2023 Set 2: 203 intervals $\ge 2.0$s; **148 court-like candidate intervals (automatic rally candidates)**, **25 candidates $\ge 16.0$s** (384 frames @ 24 fps).
    - *Visual Verification*: The term **"visually verified two-player interval"** is strictly reserved for the explicitly sampled shots that were actually inspected via contact sheets (3 shots in AO 2024, 4 shots in US Open 2023; see §4.2).
  - **Prior-Use & Overlap Audit Status**:
    - No identified match overlap with the known development assets, subject to the final prior-use audit.
  - **Confirmation Status**: Both sources remain **provisional candidates** (`confirmation_eligible=false`), not certified confirmation matches. Active experiment manifests and paper repository are untouched.

---

## 2. Authority, Bounds & Policy Adherence

| Rule / Constraint | Limit / Requirement | Observed / Enforced Value | Compliance Status |
|---|---|---|---|
| Permitted Sources | Only `8EShbWpBm_0` and `PH6VpEfTMVQ` | Exactly these two sources fetched | **PASS** |
| Max Duration per Source | $\le 45$ minutes per source | AO: 39m 00s; US Open: 35m 00s | **PASS** (under cap) |
| Max Combined Download Payload | $\le 10.0$ GB combined | 701.33 MiB (0.685 GiB) | **PASS** (6.85% of cap) |
| Stream Selection | Prefer video-only SDR, native resolution/fps | AO: 1080p25 SDR (137, 1920x1080/25 fps); US Open: 720p30 SDR (136, 1280x720/30000/1001 fps). Cannot independently confirm a native-4K-specific claim; final eligibility depends on the frozen regime, prior-use audit, and later annotation checks. | **PASS** |
| Downloader Interval Enforcement | Truly bounded interval; no whole-match fetch | Verified via range headers / ffmpeg stream copy | **PASS** |
| Data Storage Location | Configured external data root off-tree | `assets/confirmation_raw/bp57/` in `pointstream-data` | **PASS** (zero media in Git) |
| Model Operations | No annotation, training, or confirmation encodes | Strictly metadata & lightweight shot checks | **PASS** |
| Manifest Safety | No modifications to active manifests | `manifests/bp46_long_tennis_scenes.json` untouched | **PASS** |

---

## 3. Provenance & Bitstream Verification

### 3.1 Source 1: Australian Open 2024 Women's Singles Final
- **Match**: Aryna Sabalenka vs. Qinwen Zheng (6-3, 6-2)
- **Official Record**: [WTA Tour 901/LS001](https://www.wtatennis.com/tournament/901/australian-open/2024/scores/LS001)
- **YouTube Video ID**: `8EShbWpBm_0`
- **Official Uploader**: Australian Open (`@australianopen`, Channel `UCeTKJSW1NTAkf27nNmjWt5A`)
- **Acquired Slice**: `assets/confirmation_raw/bp57/ao2024_w_final_set2_raw.mp4`
- **File Size**: 564,336,240 bytes (538.19 MiB)
- **SHA-256 Checksum**: `fc82b25f71b0972ecfbcb861e72185eb70a4cb3b636254d096d34c988a3b6b12`
- **Exact Fetched Interval**: 33:00.00 to 72:00.00 (1980.0s to 4320.0s; 2340.04s duration = 39m 00s)
- **Selection Rationale**: Captures complete second set from opening service game through championship point and conclusion; selected purely on match structure, blind to PointStream rate–quality.
- **`ffprobe` Observed Stream Parameters**:
  - Container Format: `mov,mp4,m4a,3gp,3g2,mj2`
  - Video Codec: `h264` (High profile, level 40, `avc1 / 0x31637661`)
  - Pixel Format: `yuv420p`
  - Dimensions: $1920 \times 1080$ (SAR 1:1, DAR 16:9)
  - Frame Rate: `25/1` (25.000 fps constant)
  - Total Decoded Frames: 58,593 frames
  - Color Primaries / Space / Transfer: `bt709` / `bt709` / `bt709` (color range: `tv`)
  - Container Bitrate: 1,929 kb/s
  - Decoded Resolution & Scope: Stream decodes at $1920 \times 1080$ / $25$ fps constant. Cannot independently confirm a native-4K-specific claim; final eligibility depends on the frozen regime, prior-use audit, and later annotation checks.

### 3.2 Source 2: US Open 2023 Women's Singles Final
- **Match**: Coco Gauff vs. Aryna Sabalenka (2-6, 6-3, 6-2)
- **Official Record**: [WTA Tour 903/LS001](https://www.wtatennis.com/tournament/903/us-open/2023/scores/LS001)
- **YouTube Video ID**: `PH6VpEfTMVQ`
- **Official Uploader**: US Open Tennis Championships (`@usopen`, Channel `UCXbboag48Qlr78zzz6SkzkQ`)
- **Acquired Slice**: `assets/confirmation_raw/bp57/usopen2023_w_final_set2_raw.mp4`
- **File Size**: 171,062,602 bytes (163.14 MiB)
- **SHA-256 Checksum**: `a626689e284760a51f9249cca9d9811269a05b0e77a9b0e0623303e155a8d66f`
- **Exact Fetched Interval**: 37:00.00 to 72:00.00 (2220.0s to 4320.0s; 2100.05s duration = 35m 00s)
- **Selection Rationale**: Captures complete second set (Gauff comeback set, 6-3); selected purely on match structure, blind to PointStream rate–quality.
- **Resolution & Upscaling Audit**: Web metadata exposed 1080p formats `248-sr` and `399-sr` carrying YouTube server-side AI super-resolution tags (`xtags=sr=1`). In strict compliance with BP57 instruction (*"Do not upscale a 1080p source and call it native 4K; record native resolution/fps"*), native broadcast upload format `136` (720p 29.97fps AVC SDR) was acquired.
- **`ffprobe` Observed Stream Parameters**:
  - Container Format: `mov,mp4,m4a,3gp,3g2,mj2`
  - Video Codec: `h264` (Main profile, level 31, `avc1 / 0x31637661`)
  - Pixel Format: `yuv420p`
  - Dimensions: $1280 \times 720$ (SAR 1:1, DAR 16:9)
  - Frame Rate: `30000/1001` (29.970 fps constant)
  - Total Decoded Frames: 63,014 frames
  - Color Primaries / Space / Transfer: `bt709` / `bt709` / `bt709` (color range: `tv`)
  - Container Bitrate: 651 kb/s
  - Decoded Resolution & Scope: Stream decodes at $1280 \times 720$ / $30000/1001$ fps constant. Cannot independently confirm a native-4K-specific claim; final eligibility depends on the frozen regime, prior-use audit, and later annotation checks.

---

## 4. Lightweight Shot & Camera Continuity Audit

Scene boundary detection was executed using FFmpeg 7.1.1 native scene change detection (`select='gt(scene,0.35)',metadata=print`). All contiguous intervals $\ge 2.0$s were analyzed for court surface color consistency, line visibility, and camera stability.

### 4.1 Quantitative Shot Breakdown

| Metric | Source 1: AO 2024 Final (Sabalenka–Zheng) | Source 2: US Open 2023 Final (Gauff–Sabalenka) | Combined Total |
|---|---|---|---|
| Total Intervals $\ge 2.0$s | 172 | 203 | 375 |
| **Court-Like Candidate Intervals (Automatic Rally Candidates)** | **77** (44.8%) | **148** (72.9%) | **225** (60.0%) |
| Borderline Mixed Shots | 40 (23.3%) | 22 (10.8%) | 62 (16.5%) |
| Disqualified Non-Court Shots | 55 (32.0%) | 33 (16.3%) | 88 (23.5%) |
| **Candidate Yield $\ge 16.0$s (384f @ 24fps)** | **23** | **25** | **48** |
| **Candidate Yield $\ge 8.0$s (192f @ 24fps)** | **36** | **80** | **116** |
| **Candidate Yield $\ge 4.0$s (96f @ 24fps)** | **62** | **125** | **187** |
| **Candidate Yield $\ge 2.0$s (48f @ 24fps)** | **77** | **148** | **225** |

> [!NOTE]
> The aggregate numbers above reflect automated heuristic filtering ("court-like candidate intervals" / "automatic rally candidates") based on FFmpeg scene cut and surface color metrics. The designation **"visually verified two-player interval"** is strictly reserved for the explicitly sampled shots that were actually inspected via contact sheets in §4.2 below.

### 4.2 Top Candidate Windows (Visually Verified Two-Player Intervals via Contact Sheets)

#### Source 1: AO 2024 Final
1. **Shot 057** (`685.20s` – `711.00s`; `11:25.20` – `11:51.00` in Set 2):
   - **Duration**: **25.80 seconds** (645 native frames; 619 frames @ 24fps).
   - **Visual Verification**: **Visually verified two-player interval**: Perfect continuous elevated baseline perspective of Rod Laver Arena; both Sabalenka and Zheng visible and actively exchanging baseline groundstrokes across all 16 contact sheet samples. Zero camera cuts or zooms.
   - **Supported Spans**: 48, 96, 192, 384 frames.
2. **Shot 050** (`615.76s` – `637.36s`; `10:15.76` – `10:37.36` in Set 2):
   - **Duration**: **21.60 seconds** (540 native frames; 518 frames @ 24fps).
   - **Visual Verification**: **Visually verified two-player interval**: Clean unbroken point play; Zheng serving, Sabalenka returning; both players tracked on court throughout.
   - **Supported Spans**: 48, 96, 192, 384 frames.
3. **Shot 175** (`2289.84s` – `2313.16s`; `38:09.84` – `38:33.16` in Set 2):
   - **Duration**: **23.32 seconds** (583 native frames; 559 frames @ 24fps).
   - **Visual Verification**: **Visually verified two-player interval**: Championship-game rally; two players visible, full court coverage.

#### Source 2: US Open 2023 Final
1. **Shot 109** (`955.25s` – `982.65s`; `15:55.25` – `16:22.65` in Set 2):
   - **Duration**: **27.39 seconds** (821 native frames; 657 frames @ 24fps).
   - **Visual Verification**: **Visually verified two-player interval**: Unbroken elevated baseline camera on Arthur Ashe Stadium court; Gauff and Sabalenka engaging in extended 18+ stroke rally; both players visible on opposite baselines across all 16 sample frames. Zero cuts.
   - **Supported Spans**: 48, 96, 192, 384 frames.
2. **Shot 031** (`221.72s` – `248.65s`; `03:41.72` – `04:08.65` in Set 2):
   - **Duration**: **26.93 seconds** (807 native frames; 646 frames @ 24fps).
   - **Visual Verification**: **Visually verified two-player interval**: Gauff serving to Sabalenka; long crosscourt rally; two players fully visible.
   - **Supported Spans**: 48, 96, 192, 384 frames.
3. **Shot 171** (`1560.39s` – `1587.42s`; `26:00.39` – `26:27.42` in Set 2):
   - **Duration**: **27.03 seconds** (810 native frames; 648 frames @ 24fps).
   - **Visual Verification**: **Visually verified two-player interval**: High-intensity baseline exchange; zero director switches.
   - **Supported Spans**: 48, 96, 192, 384 frames.
4. **Shot 202** (`1917.58s` – `1944.31s`; `31:57.58` – `32:24.31` in Set 2):
   - **Duration**: **26.73 seconds** (801 native frames; 641 frames @ 24fps).
   - **Visual Verification**: **Visually verified two-player interval**: Set-clinching rally; clean baseline broadcast angle.

### 4.3 Borderline & Disqualified Shots (Failure Modes)
- **Disqualified Non-Court Shots (88 total across both videos)**:
  - Player closeups (e.g. server bouncing ball, returner waiting).
  - Changeover bench rest and towel-off periods (e.g. AO shot 76, 20s).
  - Crowd / spectator reaction shots or stadium exterior pans (e.g. AO shot 82, 41s).
  - Graphic replay sequences (Hawk-Eye ball-tracking graphics, slow-motion replays).
- **Borderline Mixed Shots (62 total)**:
  - Sequences where broadcast director used a slow cross-dissolve between a player closeup and point play without an abrupt hard cut (e.g. AO shot 14, where court play occupies frames 9–16 but frames 1–8 are a closeup).
  - Shots where camera temporarily zoomed in on the winner during point termination before cutting away.

---

## 5. Provenance & Compilation Overlap Audit

1. **Carlos Alcaraz Quarantine Maintained**:
   - Zero Carlos Alcaraz involvement in either acquired match (Sabalenka vs. Zheng, Gauff vs. Sabalenka).
   - No identified match overlap with the known development assets, subject to the final prior-use audit.
2. **Development / Training Asset Separation**:
   - Verified against `assets/dataset/pointstream_aa_meta.json` (114 tracks):
     - Aryna Sabalenka: 0 prior tracks.
     - Qinwen Zheng: 0 prior tracks.
     - Coco Gauff: 0 prior tracks.
   - No identified match overlap with the known development assets, subject to the final prior-use audit (checked against `federer_djokovic`, `alcaraz_perricard`, `alcaraz_ruud`, `djokovic_federer`, `djokovic_zverev`, `sinner_alcaraz`, `alcaraz_highlights`).
3. **Rights & Redistribution Restrictions**:
   - Both sources are publicly hosted broadcast matches under the Standard YouTube License.
   - Raw video media resides strictly in the external data root (`assets/confirmation_raw/bp57/` under `pointstream-data`).
   - No raw video media is added to the Git repository or the paper repository.

---

## 6. Proposed Next Steps & Cost Estimation

> [!NOTE]
> Per BP57 instructions, these next steps are proposed for subsequent authorized waves; they were **not executed** in this pilot.

1. **Extraction & Silhouette Segmentation (Gate B Phase 2)**:
   - Target: Top 2 visually verified two-player intervals per source (e.g. AO 2024 shots 57 & 50; US Open 2023 shots 109 & 31).
   - Extract source frames at 24.0 fps working rate (total 4 scenes $\times$ 384 frames = 1,536 frames).
   - Run automated two-player detection and silhouette tracking using existing ByteTrack / BoT-SORT on YOLOv8x / SAM.
   - Estimated compute: ~15–25 minutes GPU wall clock.
   - Estimated intermediate storage: ~5.0 GB (PNG frames, masks, and canonical background canvas).
2. **Invariant Validation**:
   - Execute `experiments.long_scenes.verify --strict-confirmation` on extracted candidate scenes.
   - Verify paste-back MAE $\le 2.0$, canvas growth $\le 2.5\times$, and consecutive MAD $\le 10.0$ across 48, 96, 192, and 384 frames.
3. **Confirmation Corpus Integration**:
   - Promote verified scenes to `confirmation_videos` in experiment manifests only after all invariant and annotation checks pass. The acquired sources decode at 1920x1080/25 fps and 1280x720/30000/1001 fps; they cannot independently confirm a native-4K-specific claim. Final eligibility depends on the frozen regime, prior-use audit, and later annotation checks.
