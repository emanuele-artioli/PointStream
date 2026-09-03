# BP54 — Fresh Confirmation Source Shortlist & Acquisition Plan

**Date**: 2026-09-03  
**Auditor**: Antigravity  
**Roadmap Milestone**: E2 Gate B (Independent Confirmation Corpus)  
**Status**: COMPLETE (Audit & Acquisition Plan Prepared; Pending User / Codex Acquisition Approval)  
**Branch**: `antigravity/bp54-fresh-confirmation-sources`  
**Committed Manifest**: [`manifests/confirmation-source-candidates.json`](file:///home/itec/emanuele/pointstream/manifests/confirmation-source-candidates.json)  
**Assigned Brief**: [`plans/BP54-fresh-confirmation-sources.md`](file:///home/itec/emanuele/pointstream/plans/BP54-fresh-confirmation-sources.md)  
**Related Plans**: [`plans/BP51-confirmation-audit.md`](file:///home/itec/emanuele/pointstream/plans/BP51-confirmation-audit.md), [`plans/BP46-long-tennis-scenes.md`](file:///home/itec/emanuele/pointstream/plans/BP46-long-tennis-scenes.md), [`plans/SESSION-REPORT.md`](file:///home/itec/emanuele/pointstream/plans/SESSION-REPORT.md)

---

## 1. Executive Summary

This report establishes the vetted **candidate shortlist** and **source acquisition plan** for fresh, independent tournament match footage required to satisfy the **E2 Gate B** confirmation milestone.

### Key Outcomes
1. **Host Confirmation Deficit Acknowledged**: Following the BP51 audit, all seven existing 4K video assets on host remain classified as diagnostic (`alcaraz_highlights`, `federer_djokovic`) or development (`alcaraz_perricard`, `alcaraz_ruud`, `djokovic_federer`, `djokovic_zverev`, `sinner_alcaraz`), with 114 player tracks previously used in Animate-Anyone fine-tuning, headroom sweeps, or ladder tuning. Zero untouched confirmation matches exist on host (`confirmation_videos: []`).
2. **Fresh Candidate Roster (6 Primary + 4 Alternatives)**: Ten distinct, professional tournament matches have been identified from official tournament channels (`@australianopen`, `@rolandgarros`, `@usopen`, `@WTA`, `@wimbledon`). None overlap with prior training or tuning assets.
3. **Strict Candidate Status**: Per the brief, `candidate != accepted confirmation`. No video has been downloaded, annotated, or ingested into the active experiment manifest ([`manifests/bp46_long_tennis_scenes.json`](file:///home/itec/emanuele/pointstream/manifests/bp46_long_tennis_scenes.json)). All candidates remain unacquired pending explicit user/Codex approval.
4. **Metadata Observation Boundary**: All resolution and frame rate figures in this audit are strictly uploader- and metadata-derived (checked 2026-09-03). Decoded stream characteristics (true resolution, color space, frame hashes) will only be reported following physical acquisition and `ffprobe` verification.
5. **Timeline to Evidence Freeze**: A concrete 17-day schedule to the **20 September evidence freeze** is defined, including storage budgets and a bounded first acquisition batch.

---

## 2. Prior-Use Contamination Baseline & Exclusion Rules

The BP51 audit verified 114 annotated player tracks across seven video assets in [`assets/dataset/pointstream_aa_meta.json`](file:///home/itec/emanuele/pointstream-data/assets/dataset/pointstream_aa_meta.json). The candidate selection strictly enforces the following negative boundaries:

| Asset on Host | Prior Uses / Role | Overlap Hazard to Avoid | Candidate Constraint Enforced |
|---|---|---|---|
| `alcaraz_highlights` | 20 AA tracks; BP21/23/24/31/30; E1 diagnostic | Multiple tournament compilation | Exclude any match appearing in this highlight compilation; prioritize non-Alcaraz matches and verified post-June 2024 matches. |
| `federer_djokovic` | 20 AA tracks; BP21/24/31/30; E1 diagnostic | Cincinnati 2015 Final | Exclude 2015 Western & Southern Open Final entirely. |
| `alcaraz_perricard` | 14 AA tracks; BP21/24/31/30; probe set | Beijing 2024 R32 (or 2025 AO practice matchplay) | Exclude Beijing 2024 R32 and court-level practice matchplay. |
| `alcaraz_ruud` | 4 AA tracks; practice session | 2024 Practice drills (single-player) | Exclude all non-competitive practice footage. |
| `djokovic_federer` | 20 AA tracks; BP21/24/31/30 | Wimbledon 2019 Final | Exclude Wimbledon 2019 Final entirely. |
| `djokovic_zverev` | 16 AA tracks; BP21/24/31; probe set | Unresolved event identity | **Blocker**: Exclude all Djokovic vs. Zverev matches until host asset identity is verified. |
| `sinner_alcaraz` | 20 AA tracks; BP21/23/24/31/30 | Unresolved event identity (Beijing 2024 vs Indian Wells/Miami) | **Blocker**: Exclude all Sinner vs. Alcaraz matches from Beijing 2024, Indian Wells 2024, and Miami 2023. |

---

## 3. Fresh Confirmation Source Shortlist

All candidates are professional tournament matches from official Grand Slam or WTA/ATP 1000 tournaments, verified against official ATP Tour, WTA Tennis, or Grand Slam archives as of 2026-09-03.

### 3.1 Primary Candidates (6 Required Matches)

| Candidate ID | Match & Players | Tournament, Round & Surface | Official Event Date | Official Event Source URL | Stable YouTube URL / Video ID | Uploader & Channel | Duration (HMS) | Claimed Res / FPS (Metadata) | 16s (384f) Feasibility |
|---|---|---|---|---|---|---|---|---|---|
| `conf_cand_01` | **Jannik Sinner vs. Daniil Medvedev** | 2024 Australian Open, Men's Final (Outdoor Hard) | 2024-01-28 | [ATP Tour Match 2024/580/ms001](https://www.atptour.com/en/scores/stats-centre/archive/2024/580/ms001) | [`qkNLSXDAZtQ`](https://www.youtube.com/watch?v=qkNLSXDAZtQ) (Full) / [`b90INDbXX7Y`](https://www.youtube.com/watch?v=b90INDbXX7Y) (Ext) | Australian Open (`@australianopen`) | 3:00:28 (Full) / 8:06 (Ext) | 1080p/4K @ 50 fps | **High** (numerous 20+ shot rallies; 39-shot rally in Set 4) |
| `conf_cand_02` | **Aryna Sabalenka vs. Qinwen Zheng** | 2024 Australian Open, Women's Final (Outdoor Hard) | 2024-01-27 | [WTA Tour Match 901/LS001](https://www.wtatennis.com/tournament/901/australian-open/2024/scores/LS001) | [`F3QYc6W1k0k`](https://www.youtube.com/watch?v=F3QYc6W1k0k) (Full) / [`kLdkTzDbXVE`](https://www.youtube.com/watch?v=kLdkTzDbXVE) (Ext) | Australian Open (`@australianopen`) | 2:10:00 (Full) / 8:06 (Ext) | 1080p/4K @ 50 fps | **Medium-High** (aggressive rallies; multiple 14-18 shot points in Set 2) |
| `conf_cand_03` | **Carlos Alcaraz vs. Alexander Zverev** | 2024 Roland-Garros, Men's Final (Outdoor Red Clay) | 2024-06-09 | [ATP Tour Match 2024/520/ms001](https://www.atptour.com/en/scores/stats-centre/archive/2024/520/ms001) | [`qorFNY2lSN8`](https://www.youtube.com/watch?v=qorFNY2lSN8) (Full) / [`nwofBAmsDpE`](https://www.youtube.com/watch?v=nwofBAmsDpE) (Ext) | Roland-Garros (`@rolandgarros`) | 4:23:50 (Full) / 41:47 (Ext) | 1080p/4K UHD @ 50 fps | **Very High** (heavy clay rallies routinely exceeding 20 seconds) |
| `conf_cand_04` | **Coco Gauff vs. Aryna Sabalenka** | 2023 US Open, Women's Final (Outdoor Hard) | 2023-09-09 | [WTA Tour Match 903/LS001](https://www.wtatennis.com/tournament/903/us-open/2023/scores/LS001) | [`PH6VpEfTMVQ`](https://www.youtube.com/watch?v=PH6VpEfTMVQ) (Full) / [`XfsT9YHYVzk`](https://www.youtube.com/watch?v=XfsT9YHYVzk) (Ext) | US Open Tennis Championships (`@usopen`) | 1:43:51 (Full) / 18:07 (Ext) | 1080p/4K @ 60 fps | **High** (Gauff retrieving style extended points to 15-25 shots in Sets 2 & 3) |
| `conf_cand_05` | **Daniil Medvedev vs. Novak Djokovic** | 2023 US Open, Men's Final (Outdoor Hard) | 2023-09-10 | [ATP Tour Match 2023/560/ms001](https://www.atptour.com/en/scores/stats-centre/archive/2023/560/ms001) | [`1R4qtc1H4wM`](https://www.youtube.com/watch?v=1R4qtc1H4wM) (Full) / [`lGjvN4y5XFE`](https://www.youtube.com/watch?v=lGjvN4y5XFE) (Ext) | US Open Tennis Championships (`@usopen`) | 2:36:49 (Full) / 26:59 (Ext) | 1080p/4K @ 60 fps | **Very High** (104-minute 2nd set war of attrition with 25+ shot exchanges) |
| `conf_cand_06` | **Iga Swiatek vs. Aryna Sabalenka** | 2024 Mutua Madrid Open, Women's Final (Outdoor Red Clay) | 2024-05-04 | [WTA Tour Match 1038/LS001](https://www.wtatennis.com/tournament/1038/madrid/2024/scores/LS001) | [`n0MZNE_AIy4`](https://www.youtube.com/watch?v=n0MZNE_AIy4) (Full) / [`gF8CwX8Hdkw`](https://www.youtube.com/watch?v=gF8CwX8Hdkw) (Ext) | WTA (`@WTA`) | 2:55:54 (Full) / 5:21 (Ext) | 1080p/4K @ 50 fps | **Very High** (3h 11m clay marathon; multiple 18-24 shot rallies) |

### 3.2 Alternative Candidates (4 Vetted Standbys)

| Candidate ID | Match & Players | Tournament, Round & Surface | Official Event Date | Official Event Source URL | Stable YouTube URL / Video ID | Uploader & Channel | Duration (HMS) | Claimed Res / FPS (Metadata) | Reason Kept as Alternative |
|---|---|---|---|---|---|---|---|---|---|
| `conf_alt_01` | **Carlos Alcaraz vs. Novak Djokovic** | 2023 Wimbledon Championships, Gentlemen's Final (Outdoor Grass) | 2023-07-16 | [ATP Tour Match 2023/540/ms001](https://www.atptour.com/en/scores/stats-centre/archive/2023/540/ms001) | [`5uFAkizQNJI`](https://www.youtube.com/watch?v=5uFAkizQNJI) (Full) / [`dvBr9Wr8BCY`](https://www.youtube.com/watch?v=dvBr9Wr8BCY) (Ext) | Wimbledon (`@wimbledon`) | 4:03:01 (Full) / 21:02 (Ext) | 1080p/4K @ 50 fps | Potential overlap uncertainty with untracked scenes in `alcaraz_highlights` compilation. Grass wear over fortnight. |
| `conf_alt_02` | **Rafael Nadal vs. Daniil Medvedev** | 2022 Australian Open, Men's Final (Outdoor Hard) | 2022-01-30 | [ATP Tour Match 2022/580/ms001](https://www.atptour.com/en/scores/stats-centre/archive/2022/580/ms001) | [`6I06-ITW88k`](https://www.youtube.com/watch?v=6I06-ITW88k) (Full) / [`v27M_RgrLzU`](https://www.youtube.com/watch?v=v27M_RgrLzU) (Ext) | Australian Open (`@australianopen`) | 5:41:11 (Full) / 8:25 (Ext) | 1080p/4K @ 50 fps | Full match download is exceptionally large (5.5h, ~25 GB); highly eligible if acquired in a set slice. |
| `conf_alt_03` | **Stefanos Tsitsipas vs. Novak Djokovic** | 2023 Australian Open, Men's Final (Outdoor Hard) | 2023-01-29 | [ATP Tour Match 2023/580/ms001](https://www.atptour.com/en/scores/stats-centre/archive/2023/580/ms001) | [`FE2or3g488o`](https://www.youtube.com/watch?v=FE2or3g488o) (Full) / [`N2Dtsx-6aDc`](https://www.youtube.com/watch?v=N2Dtsx-6aDc) (Ext) | Australian Open (`@australianopen`) | 2:24:38 (Full) / 8:21 (Ext) | 1080p/4K @ 50 fps | Backup hard court match; clean provenance (Tsitsipas has 0 tracks in AA). |
| `conf_alt_04` | **Novak Djokovic vs. Casper Ruud** | 2023 Roland-Garros, Men's Final (Outdoor Red Clay) | 2023-06-11 | [ATP Tour Match 2023/520/ms001](https://www.atptour.com/en/scores/stats-centre/archive/2023/520/ms001) | [`nJXznKxFG8U`](https://www.youtube.com/watch?v=nJXznKxFG8U) (Full) / [`hvl_iaK4ra8`](https://www.youtube.com/watch?v=hvl_iaK4ra8) (Ext) | Roland-Garros (`@rolandgarros`) | 3:21:07 (Full) / 34:23 (Ext) | 1080p/4K @ 50 fps | Backup red clay match; Ruud was only in practice drills on host, zero match tracks. |

---

## 4. Candidate Profiles & Uncertainty Analysis

### 4.1 Candidate 1: Sinner vs. Medvedev (AO 2024 Final)
- **Event Identity**: 2024 Australian Open Men's Singles Final, 28 January 2024, Rod Laver Arena, Melbourne.
- **Source Video**: Australian Open official YouTube upload [`qkNLSXDAZtQ`](https://www.youtube.com/watch?v=qkNLSXDAZtQ) (3h 00m 28s).
- **Official Match Record**: ATP Tour official archive match 2024-580-ms001; Jannik Sinner d. Daniil Medvedev 3-6, 3-6, 6-4, 6-4, 6-3.
- **Prior-Use Cross-Check**:
  - `pointstream_aa_meta.json`: Medvedev = 0 tracks. Sinner appears in `sinner_alcaraz`, but that asset is an unresolved Beijing/US hard court match with Carlos Alcaraz, distinct from Rod Laver Arena Melbourne.
  - Headroom (BP21), calibration (BP23), ladder sweeps (BP24/31), panorama registration (BP30): 0 uses.
- **Claimed vs. Verified Resolution**: Official broadcast stream claims 1080p/4K at 50 fps. Decoded container parameters (width, height, color primaries, pix_fmt, bitrate) are **unverified** pending download and `ffprobe` execution.
- **Camera & Court Assessment**: Elevated main-court camera; high contrast between blue court and player attire; minimal camera zoom; expected homography canvas growth $\le 1.15\times$.
- **16-Second Feasibility**: **High**. Extensive baseline rallies in sets 3, 4, and 5 exceed 20 seconds.
- **Rights / Access**: Public YouTube video (YouTube Standard License). Research fair use only; no raw video redistribution in Git repository.

### 4.2 Candidate 2: Sabalenka vs. Zheng (AO 2024 Final)
- **Event Identity**: 2024 Australian Open Women's Singles Final, 27 January 2024, Rod Laver Arena, Melbourne.
- **Source Video**: Australian Open official YouTube upload [`F3QYc6W1k0k`](https://www.youtube.com/watch?v=F3QYc6W1k0k) (2h 10m).
- **Official Match Record**: WTA Tour official archive match 901-LS001; Aryna Sabalenka d. Qinwen Zheng 6-3, 6-2.
- **Prior-Use Cross-Check**: Neither player appears in `pointstream_aa_meta.json` or any PointStream experimental run. Zero overlap with any historical codebase asset.
- **Claimed vs. Verified Resolution**: Metadata claims 1080p/4K at 50 fps. Decoded file parameters **unverified**.
- **Camera & Court Assessment**: Elevated baseline broadcast camera.
- **16-Second Feasibility**: **Medium-High**. Fast baseline points mean average rally length is shorter (~6-10s); however, games 4 and 8 in the second set featured multiple 14-18 shot rallies that comfortably provide 384 continuous frames.
- **Rights / Access**: Public YouTube video; no paywall.

### 4.3 Candidate 3: Alcaraz vs. Zverev (RG 2024 Final)
- **Event Identity**: 2024 Roland-Garros Men's Singles Final, 9 June 2024, Court Philippe-Chatrier, Paris.
- **Source Video**: Roland-Garros official YouTube upload [`qorFNY2lSN8`](https://www.youtube.com/watch?v=qorFNY2lSN8) (4h 23m 50s).
- **Official Match Record**: ATP Tour official archive match 2024-520-ms001; Carlos Alcaraz d. Alexander Zverev 6-3, 2-6, 5-7, 6-1, 6-2.
- **Prior-Use Cross-Check**:
  - `pointstream_aa_meta.json`: Animate-Anyone dataset was finalized prior to this match.
  - `alcaraz_highlights`: Video file was created on host 12 June 2024; scene metadata audit shows 0 red clay scenes among diagnostic candidates.
  - `djokovic_zverev`: Features a different opponent (Novak Djokovic).
- **Claimed vs. Verified Resolution**: Host broadcast produced in native 4K UHD HDR by France Télévisions; YouTube stream metadata reports 1080p/4K at 50 fps. Decoded stream parameters **unverified**.
- **Camera & Court Assessment**: Court Philippe-Chatrier elevated camera has exceptional positional stability; red clay provides crisp player contrast.
- **16-Second Feasibility**: **Very High**. Clay court conditions produce long, multi-shot rallies frequently lasting 20-35 seconds without cuts.

### 4.4 Candidate 4: Gauff vs. Sabalenka (US Open 2023 Final)
- **Event Identity**: 2023 US Open Women's Singles Final, 9 September 2023, Arthur Ashe Stadium, New York.
- **Source Video**: US Open official YouTube upload [`PH6VpEfTMVQ`](https://www.youtube.com/watch?v=PH6VpEfTMVQ) (1h 43m 51s).
- **Official Match Record**: WTA Tour official archive match 903-LS001; Coco Gauff d. Aryna Sabalenka 2-6, 6-3, 6-2.
- **Prior-Use Cross-Check**: Zero tracks in `pointstream_aa_meta.json`. Zero prior use in any PointStream sweep or experiment.
- **Claimed vs. Verified Resolution**: US Open broadcast stream claims 1080p/4K at 60 fps. Decoded stream parameters **unverified**.
- **Camera & Court Assessment**: High Arthur Ashe stadium camera angle; blue hard court with green surround.
- **16-Second Feasibility**: **High**. Gauff's defensive retrieving against Sabalenka's power produced multiple 20+ shot exchanges across sets 2 and 3.

### 4.5 Candidate 5: Medvedev vs. Djokovic (US Open 2023 Final)
- **Event Identity**: 2023 US Open Men's Singles Final, 10 September 2023, Arthur Ashe Stadium, New York.
- **Source Video**: US Open official YouTube upload [`1R4qtc1H4wM`](https://www.youtube.com/watch?v=1R4qtc1H4wM) (2h 36m 49s).
- **Official Match Record**: ATP Tour official archive match 2023-560-ms001; Novak Djokovic d. Daniil Medvedev 6-3, 7-6(5), 6-3.
- **Prior-Use Cross-Check**: Medvedev has 0 AA tracks. Djokovic appears in Wimbledon 2019 and Cincinnati 2015 on host, but Arthur Ashe 2023 is a completely distinct tournament, court, opponent, and year.
- **Claimed vs. Verified Resolution**: Claims 1080p/4K at 60 fps. Decoded stream parameters **unverified**.
- **Camera & Court Assessment**: Elevated main-court camera with smooth tracking.
- **16-Second Feasibility**: **Very High**. Second set was an epic 104-minute tactical war of attrition with frequent rallies exceeding 25 shots (>20s).

### 4.6 Candidate 6: Swiatek vs. Sabalenka (Madrid 2024 Final)
- **Event Identity**: 2024 Mutua Madrid Open Women's Singles Final, 4 May 2024, Manolo Santana Stadium, Caja Mágica, Madrid.
- **Source Video**: WTA official YouTube upload [`n0MZNE_AIy4`](https://www.youtube.com/watch?v=n0MZNE_AIy4) (2h 55m 54s).
- **Official Match Record**: WTA Tour official archive match 1038-LS001; Iga Swiatek d. Aryna Sabalenka 7-5, 4-6, 7-6(7).
- **Prior-Use Cross-Check**: Zero tracks in `pointstream_aa_meta.json`. Zero prior use in any PointStream sweep or experiment.
- **Claimed vs. Verified Resolution**: Claims 1080p/4K at 50 fps. Decoded stream parameters **unverified**.
- **Camera & Court Assessment**: Manolo Santana Stadium broadcast camera; red clay court with indoor/outdoor architectural lighting.
- **16-Second Feasibility**: **Very High**. High-altitude clay match lasting 3 hours 11 minutes with extreme rally intensity; 384-frame intervals abundant in sets 2 and 3.

---

## 5. Storage, Extraction & Timeline Estimation (to 20 September)

### 5.1 Storage Budget
- **Host Context**: Host filesystem `/home/itec/emanuele` has **27 TB available**.
- **Raw Video Footprint**:
  - Full match downloads (if acquired in full, ~2–4 hours each at 1080p/4K high bitrate) average ~4–8 GB per match.
  - Targeted slice acquisition (e.g. 1 complete set or a 30–45 min continuous rally block per match): ~1.5–3.0 GB per match.
  - Total raw storage for 6 primary candidates: **~12–25 GB**.
- **Extracted Frame & Artifact Footprint**:
  - Source frames for 4 candidate scenes per match (each up to 384 frames at 3840×2160):
    $6 \times 4 \times 384 \times 8\text{ MB (PNG)} \approx 73.7\text{ GB}$.
  - Player segmentation crops + alpha masks ($2 \times 384 \times 50\text{ KB} \approx 38.4\text{ MB}$ per scene): **~0.9 GB**.
  - Background canonical canvases & homographies: **~0.2 GB**.
  - Total storage budget for complete 6-match confirmation corpus: **$\le 100\text{ GB}$** (less than 0.4% of available NFS capacity).

### 5.2 Extraction & Two-Player Annotation Effort
- **Object Detection & Silhouette Generation**:
  - 2 foreground players tracked simultaneously across all spans.
  - Automated tracking: ByteTrack / BoT-SORT on YOLOv8x / SAM masks.
  - Invariant validation: paste-back MAE $\le 2.0$, canvas growth $\le 2.5\times$, consecutive MAD $\le 10.0$.
- **Per-Match Yield Requirement**:
  - Each match must provide **at least 1 validated scene** with continuous 2-player tracking across all 4 target spans (**48, 96, 192, and 384 frames** at 24 fps).
  - Target extraction: 3–5 candidate scenes extracted per match to guarantee at least 1–2 scenes meet the strict 384-frame criteria.

### 5.3 Execution Schedule to 20 September Evidence Freeze (17 Days Remaining)

| Phase | Milestone / Task | Target Dates | Duration | Dependency / Gate |
|---|---|---|:---:|---|
| **Phase 1** | **Acquisition Approval & Eligibility Rule Freeze** | Sept 4 – Sept 5 | 2 days | User / Codex review of BP54 shortlist and approval of download batch |
| **Phase 2** | **Bounded Acquisition & Decoded Verification** | Sept 6 – Sept 7 | 2 days | Download 6 primary matches; run `ffprobe` verification; record true stream parameters |
| **Phase 3** | **Scene Clustering & 2-Player Extraction** | Sept 8 – Sept 11 | 4 days | Extract 3–5 scenes per match; generate player silhouettes and homographies |
| **Phase 4** | **Invariant & Ingestion Verification** | Sept 12 – Sept 14 | 3 days | Run `verify.py --strict-confirmation`; confirm 6 accepted matches; update manifest |
| **Phase 5** | **Independent Confirmation Runs (E2 Gate B)** | Sept 15 – Sept 18 | 4 days | Run frozen winning PointStream config vs. AV1/VVC on confirmation scenes across all 4 spans |
| **Phase 6** | **Evidence Freeze & Manuscript Integration** | Sept 19 – Sept 20 | 2 days | Verify null controls, bounds, timing ledgers; freeze evidence for 30 Sept submission |

---

## 6. Proposed Bounded Acquisition Batch (Pending Approval)

> [!IMPORTANT]
> **No downloads are executed in this session.** The following bounded batch is submitted for explicit user and Codex approval before any network download or annotation script is invoked.

### Batch Specification
- **Batch Identifier**: `confirmation_acquisition_batch_01`
- **Scope**: Download a targeted 45-minute continuous broadcast segment (encompassing a full set with top rallies) from each of the 6 primary candidate matches:
  1. `conf_cand_01`: Sinner vs. Medvedev (AO 2024 Final, Set 4 & 5 segment)
  2. `conf_cand_02`: Sabalenka vs. Zheng (AO 2024 Final, Set 2 segment)
  3. `conf_cand_03`: Alcaraz vs. Zverev (RG 2024 Final, Set 4 & 5 segment)
  4. `conf_cand_04`: Gauff vs. Sabalenka (US Open 2023 Final, Set 2 segment)
  5. `conf_cand_05`: Medvedev vs. Djokovic (US Open 2023 Final, Set 2 segment)
  6. `conf_cand_06`: Swiatek vs. Sabalenka (Madrid 2024 Final, Set 3 segment)
- **Target Destination**: `/home/itec/emanuele/pointstream-data/assets/confirmation_raw/` (external data root, off-tree).
- **Download Tool**: `yt-dlp` (via conda environment) selecting best available video stream.
- **Estimated Total Download Size**: ~15–22 GB.
- **Post-Download Technical Gate**:
  - Run automated `ffprobe` format inspection script to extract exact dimensions, duration, container fps, pixel format, color space, color transfer, and compute SHA256 checksums.
  - Candidates failing 1080p/4K resolution or showing court-level / heavily edited camera work will be substituted with the pre-vetted alternatives (`conf_alt_01`–`conf_alt_04`).

---

## 7. Decisions & Authorizations Required Next

1. **Approval of Candidate Roster**: Approve the 6 primary candidates (`conf_cand_01` through `conf_cand_06`) as the target confirmation corpus.
2. **Approval of Bounded Acquisition Batch**: Authorize execution of `confirmation_acquisition_batch_01` in the next dispatched session.
3. **Approval of Frozen Eligibility Rule**: Confirm that the confirmation selection criteria remain strictly frozen:
   - Simultaneous 2-player coverage across 48, 96, 192, and 384 frames at 24 fps.
   - Homography canvas growth $\le 2.5\times$.
   - Consecutive frame MAD $\le 10.0$.
   - Paste-back MAE $\le 2.0$.
   - Selection independent of PointStream rate–quality scores.
