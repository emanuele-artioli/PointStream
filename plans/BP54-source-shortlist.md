# BP54 — Fresh Confirmation Source Shortlist & Provenance Triage

**Date**: 2026-09-03 (Updated 2026-09-04)  
**Auditor**: Antigravity  
**Roadmap Milestone**: E2 Gate B (Independent Confirmation Corpus)  
**Status**: COMPLETE (Shortlist & Provenance Triage Prepared; Pending Physical Acquisition & Scene-Level Audit)  
**Branch**: `antigravity/bp54-fresh-confirmation-sources`  
**Committed Manifest**: [`manifests/confirmation-source-candidates.json`](file:///home/itec/emanuele/pointstream/manifests/confirmation-source-candidates.json)  
**Assigned Brief**: [`plans/BP54-fresh-confirmation-sources.md`](file:///home/itec/emanuele/pointstream/plans/BP54-fresh-confirmation-sources.md)  
**Related Plans**: [`plans/BP51-confirmation-audit.md`](file:///home/itec/emanuele/pointstream/plans/BP51-confirmation-audit.md), [`plans/BP46-long-tennis-scenes.md`](file:///home/itec/emanuele/pointstream/plans/BP46-long-tennis-scenes.md), [`plans/SESSION-REPORT.md`](file:///home/itec/emanuele/pointstream/plans/SESSION-REPORT.md)

---

## 1. Scope & Methodology: Shortlist vs. Completed Provenance Audit

> [!IMPORTANT]
> **This document is a candidate shortlist and provenance triage, NOT a completed or certified provenance audit.**
> A candidate is not an accepted confirmation match (`candidate != accepted confirmation`).
> Acceptance into the confirmation corpus requires:
> 1. Physical video acquisition into the data root;
> 2. `ffprobe` bitstream verification (true container dimensions, pixel format, frame rate, and color primaries);
> 3. Visual shot-by-shot de-duplication against the 71 scenes of `alcaraz_highlights.mp4`;
> 4. Invariant verification via `experiments.long_scenes.verify --strict-confirmation`.

In this triage phase:
- Zero videos were downloaded or annotated.
- Active experiment manifests ([`manifests/bp46_long_tennis_scenes.json`](file:///home/itec/emanuele/pointstream/manifests/bp46_long_tennis_scenes.json)) remain untouched.
- All URL availability claims are corroborated via YouTube's official oEmbed API (checked 2026-09-04).
- All video format, container bitrate, exact frame rate, and duration claims are explicitly treated as **claimed in web metadata but strictly unverified** until local decoding.
- All rally length and camera continuity claims are unverified tournament report estimates until shot-boundary extraction runs.

---

## 2. Provenance Classification: Clean vs. Unresolved Compilation Risk

The BP51 audit verified 114 annotated player tracks across seven video assets in [`assets/dataset/pointstream_aa_meta.json`](file:///home/itec/emanuele/pointstream-data/assets/dataset/pointstream_aa_meta.json). The candidate selection strictly enforces the following negative boundaries:

### 2.1 Host Assets to Avoid
- `federer_djokovic`: 20 AA tracks; Cincinnati 2015 Final.
- `alcaraz_perricard`: 14 AA tracks; Beijing 2024 R32 (and AO 2025 practice matchplay).
- `alcaraz_ruud`: 4 AA tracks; 2024 Practice drills (single-player only).
- `djokovic_federer`: 20 AA tracks; Wimbledon 2019 Final.
- `djokovic_zverev`: 16 AA tracks; unresolved tournament event (suspected ATP Finals / Olympics / Masters).
- `sinner_alcaraz`: 20 AA tracks; unresolved tournament event (Beijing 2024 vs Indian Wells / Miami).
- `alcaraz_highlights`: 20 AA tracks; multi-tournament compilation across 71 scenes.

### 2.2 Compilation Overlap Principles
1. **Unresolved overlap is a blocker, not permission to assume independence** (per BP51/BP54 rules).
2. **Absence of clay in diagnostic scenes is NOT clearance**: The 6 diagnostic scenes extracted from `alcaraz_highlights.mp4` represent only 78.4 seconds of an 8-minute 14-second compilation (71 total scenes). The remaining 65 scenes have not been exhaustively identified.
3. **File timestamps do NOT exclude prior footage**: A file modification time on host (e.g. 12 June 2024) cannot rule out footage from matches that occurred prior to or around that date (such as the Roland-Garros final on 9 June 2024).
4. **Alcaraz Quarantine**: Because `alcaraz_highlights` is an unresolved compilation centered on Carlos Alcaraz, **any match featuring Carlos Alcaraz carries an unresolved compilation overlap risk** until every scene of `alcaraz_highlights.mp4` is visually audited against official match points.

### 2.3 Candidate Provenance Classification
- **Provisional Clean (8 Candidates)**: Matches with zero Carlos Alcaraz involvement, zero overlap with historical development assets, and verified organizer provenance (e.g. WTA and ATP finals featuring Sinner, Medvedev, Sabalenka, Zheng, Gauff, Djokovic, Swiatek, Ruud, Tsitsipas, Nadal).
- **Unresolved Compilation Risk (2 Candidates)**: Matches featuring Carlos Alcaraz (`conf_alt_01` Alcaraz vs Zverev RG 2024, `conf_alt_02` Alcaraz vs Djokovic Wimbledon 2023). Both are blocked from confirmation acceptance pending scene-by-scene visual audit of `alcaraz_highlights.mp4`.
- **Definitively Contaminated (0 Candidates)**: None of the 10 shortlisted matches are among the 7 contaminated host assets.

---

## 3. Fresh Confirmation Source Shortlist

To ensure the primary acquisition cohort is 100% free of compilation overlap risk, the primary cohort is composed exclusively of **Provisional Clean** matches, while Alcaraz matches are placed in the **Unresolved Compilation Risk** standby cohort.

### 3.1 Primary Cohort (6 Provisional Clean Matches)

All 6 matches have zero Alcaraz involvement, zero prior training use, and verified official broadcast streams:

| Candidate ID | Match & Players | Tournament, Surface & Round | Date | Official Event Record Source | Corroborated Source Video URL / ID | Uploader & Channel | Claimed Duration (HMS) | Claimed Res / FPS (Metadata) | 16s (384f) Feasibility |
|---|---|---|---|---|---|---|---|---|---|
| `conf_cand_01` | **J. Sinner vs. D. Medvedev** | 2024 Australian Open, Men's Final (Outdoor Hard) | 2024-01-28 | [ATP Tour 2024/580/ms001](https://www.atptour.com/en/scores/stats-centre/archive/2024/580/ms001) | [`qkNLSXDAZtQ`](https://www.youtube.com/watch?v=qkNLSXDAZtQ) (Full) / [`b90INDbXX7Y`](https://www.youtube.com/watch?v=b90INDbXX7Y) (Ext) | Australian Open (`@australianopen`) | 3:00:28 (Full) / 8:06 (Ext) | 1080p/4K @ 50 fps *(unverified)* | **High** (numerous 20+ shot rallies; 39-shot rally in Set 4) |
| `conf_cand_02` | **A. Sabalenka vs. Q. Zheng** | 2024 Australian Open, Women's Final (Outdoor Hard) | 2024-01-27 | [WTA Tour 901/LS001](https://www.wtatennis.com/tournament/901/australian-open/2024/scores/LS001) | [`8EShbWpBm_0`](https://www.youtube.com/watch?v=8EShbWpBm_0) (Full) / [`kLdkTzDbXVE`](https://www.youtube.com/watch?v=kLdkTzDbXVE) (Ext) / [`yh8VupWjKO0`](https://www.youtube.com/watch?v=yh8VupWjKO0) (Cond) | Australian Open (`@australianopen`) | 1:43:22 (Full) / 8:06 (Ext) / 20:08 (Cond) | 1080p/4K @ 50 fps *(unverified)* | **Medium-High** (aggressive rallies; multiple 14-18 shot points in Set 2) |
| `conf_cand_03` | **C. Gauff vs. A. Sabalenka** | 2023 US Open, Women's Final (Outdoor Hard) | 2023-09-09 | [WTA Tour 903/LS001](https://www.wtatennis.com/tournament/903/us-open/2023/scores/LS001) | [`PH6VpEfTMVQ`](https://www.youtube.com/watch?v=PH6VpEfTMVQ) (Full) / [`XfsT9YHYVzk`](https://www.youtube.com/watch?v=XfsT9YHYVzk) (Ext) | US Open Tennis Championships (`@usopen`) | 1:43:51 (Full) / 18:07 (Ext) | 1080p/4K @ 60 fps *(unverified)* | **High** (Gauff retrieving style extended points to 15-25 shots in Sets 2 & 3) |
| `conf_cand_04` | **D. Medvedev vs. N. Djokovic** | 2023 US Open, Men's Final (Outdoor Hard) | 2023-09-10 | [ATP Tour 2023/560/ms001](https://www.atptour.com/en/scores/stats-centre/archive/2023/560/ms001) | [`1R4qtc1H4wM`](https://www.youtube.com/watch?v=1R4qtc1H4wM) (Full) / [`lGjvN4y5XFE`](https://www.youtube.com/watch?v=lGjvN4y5XFE) (Ext) | US Open Tennis Championships (`@usopen`) | 2:36:49 (Full) / 26:59 (Ext) | 1080p/4K @ 60 fps *(unverified)* | **Very High** (104-minute 2nd set war of attrition with 25+ shot exchanges) |
| `conf_cand_05` | **I. Swiatek vs. A. Sabalenka** | 2024 Mutua Madrid Open, Women's Final (Outdoor Red Clay) | 2024-05-04 | [WTA Tour 1038/LS001](https://www.wtatennis.com/tournament/1038/madrid/2024/scores/LS001) | [`n0MZNE_AIy4`](https://www.youtube.com/watch?v=n0MZNE_AIy4) (Full) / [`gF8CwX8Hdkw`](https://www.youtube.com/watch?v=gF8CwX8Hdkw) (Ext) | WTA (`@WTA`) | 2:55:54 (Full) / 5:21 (Ext) | 1080p/4K @ 50 fps *(unverified)* | **Very High** (3h 11m clay marathon; multiple 18-24 shot rallies) |
| `conf_cand_06` | **N. Djokovic vs. C. Ruud** | 2023 Roland-Garros, Men's Final (Outdoor Red Clay) | 2023-06-11 | [ATP Tour 2023/520/ms001](https://www.atptour.com/en/scores/stats-centre/archive/2023/520/ms001) | [`nJXznKxFG8U`](https://www.youtube.com/watch?v=nJXznKxFG8U) (Full) / [`hvl_iaK4ra8`](https://www.youtube.com/watch?v=hvl_iaK4ra8) (Ext) | Roland-Garros (`@rolandgarros`) | 3:21:07 (Full) / 34:23 (Ext) | 1080p/4K @ 50 fps *(unverified)* | **Very High** (heavy clay rallies exceeding 16s in Set 1; clean non-Alcaraz clay match) |

*Note on URL correction for `conf_cand_02`*: Previously proposed URL `F3QYc6W1k0k` returned HTTP 404 upon automated validation. It has been corrected to `8EShbWpBm_0` (Full Match, 1h 43m 22s), corroborated via oEmbed alongside `kLdkTzDbXVE` (Extended Highlights) and `yh8VupWjKO0` (Condensed Match).

### 3.2 Standby & Alternative Cohort (4 Matches)

| Candidate ID | Match & Players | Tournament, Surface & Round | Date | Classification | Corroborated Source Video URL / ID | Uploader & Channel | Claimed Duration (HMS) | Reason / Risk Profile |
|---|---|---|---|---|---|---|---|---|
| `conf_alt_01` | **C. Alcaraz vs. A. Zverev** | 2024 Roland-Garros, Men's Final (Outdoor Red Clay) | 2024-06-09 | **Unresolved Compilation Risk** | [`qorFNY2lSN8`](https://www.youtube.com/watch?v=qorFNY2lSN8) (Full) / [`nwofBAmsDpE`](https://www.youtube.com/watch?v=nwofBAmsDpE) (Ext) | Roland-Garros (`@rolandgarros`) | 4:23:50 (Full) / 41:47 (Ext) | **Blocked**: Alcaraz match. Absence of clay in diagnostic scenes does not clear remaining 65 scenes in `alcaraz_highlights.mp4`. Held in reserve pending compilation audit. |
| `conf_alt_02` | **C. Alcaraz vs. N. Djokovic** | 2023 Wimbledon Championships, Gentlemen's Final (Outdoor Grass) | 2023-07-16 | **Unresolved Compilation Risk** | [`5uFAkizQNJI`](https://www.youtube.com/watch?v=5uFAkizQNJI) (Full) / [`dvBr9Wr8BCY`](https://www.youtube.com/watch?v=dvBr9Wr8BCY) (Ext) | Wimbledon (`@wimbledon`) | 4:03:01 (Full) / 21:02 (Ext) | **Blocked**: Alcaraz match. Wimbledon 2023 is prime compilation material. Unresolved overlap with `alcaraz_highlights` blocks confirmation acceptance. |
| `conf_alt_03` | **R. Nadal vs. D. Medvedev** | 2022 Australian Open, Men's Final (Outdoor Hard) | 2022-01-30 | **Provisional Clean** | [`6I06-ITW88k`](https://www.youtube.com/watch?v=6I06-ITW88k) (Full) / [`v27M_RgrLzU`](https://www.youtube.com/watch?v=v27M_RgrLzU) (Ext) | Australian Open (`@australianopen`) | 5:41:11 (Full) / 8:25 (Ext) | Clean backup. Full match is exceptionally large (5.5h, ~25 GB); eligible if acquired as a single-set slice. |
| `conf_alt_04` | **S. Tsitsipas vs. N. Djokovic** | 2023 Australian Open, Men's Final (Outdoor Hard) | 2023-01-29 | **Provisional Clean** | [`FE2or3g488o`](https://www.youtube.com/watch?v=FE2or3g488o) (Full) / [`N2Dtsx-6aDc`](https://www.youtube.com/watch?v=N2Dtsx-6aDc) (Ext) | Australian Open (`@australianopen`) | 2:24:38 (Full) / 8:21 (Ext) | Clean backup. Tsitsipas has 0 tracks in AA. Hard court night session. |

---

## 4. Substantiated Source Evidence (oEmbed Corroboration)

All 21 candidate URLs were systematically tested and corroborated via YouTube's official public oEmbed API on **2026-09-04T07:27:52Z**. The exact response evidence is embedded in [`manifests/confirmation-source-candidates.json`](file:///home/itec/emanuele/pointstream/manifests/confirmation-source-candidates.json):

```text
✓ conf_cand_01_full (qkNLSXDAZtQ): "Jannik Sinner v Daniil Medvedev Full Match | Australian Open 2024 Final" (Australian Open)
✓ conf_cand_01_ext  (b90INDbXX7Y): "Jannik Sinner v Daniil Medvedev Extended Highlights | Australian Open 2024 Final" (Australian Open)
✓ conf_cand_02_full (8EShbWpBm_0): "Qinwen Zheng v Aryna Sabalenka Full Match | Australian Open 2024 Final" (Australian Open)
✓ conf_cand_02_ext  (kLdkTzDbXVE): "Qinwen Zheng v Aryna Sabalenka Extended Highlights | Australian Open 2024 Final" (Australian Open)
✓ conf_cand_02_cond (yh8VupWjKO0): "Qinwen Zheng v Aryna Sabalenka Condensed Match | Australian Open 2024 Final" (Australian Open)
✓ conf_cand_03_full (PH6VpEfTMVQ): "Coco Gauff vs. Aryna Sabalenka Full Match | 2023 US Open Final" (US Open Tennis Championships)
✓ conf_cand_03_ext  (XfsT9YHYVzk): "Coco Gauff vs. Aryna Sabalenka Extended Highlights | 2023 US Open Final" (US Open Tennis Championships)
✓ conf_cand_04_full (1R4qtc1H4wM): "Daniil Medvedev vs. Novak Djokovic Full Match | 2023 US Open Final" (US Open Tennis Championships)
✓ conf_cand_04_ext  (lGjvN4y5XFE): "Daniil Medvedev vs. Novak Djokovic Extended Highlights | 2023 US Open Final" (US Open Tennis Championships)
✓ conf_cand_05_full (n0MZNE_AIy4): "The Greatest Match of 2024!? 🤯 | 2024 Madrid Final | Iga Swiatek vs Aryna Sabalenka" (WTA)
✓ conf_cand_05_ext  (gF8CwX8Hdkw): "Iga Swiatek vs. Aryna Sabalenka | 2024 Madrid Final | WTA Match Highlights" (WTA)
✓ conf_cand_06_full (nJXznKxFG8U): "Djokovic vs Ruud 2023 Men's final Full Match | Roland-Garros" (Roland-Garros)
✓ conf_cand_06_ext  (hvl_iaK4ra8): "Novak makes it 23 🏆 Djokovic vs Ruud extended highlights | Roland-Garros 2023" (Roland-Garros)
✓ conf_alt_01_full  (qorFNY2lSN8): "Alcaraz vs Zverev 2024 Men's final Full Match | Roland-Garros" (Roland-Garros)
✓ conf_alt_01_ext   (nwofBAmsDpE): "A 5-set thriller for a first RG title ⚔️ Alcaraz vs Zverev extended highlights | Roland-Garros 2024" (Roland-Garros)
✓ conf_alt_02_full  (5uFAkizQNJI): "A FINAL FOR THE AGES | Carlos Alcaraz vs Novak Djokovic Full Match | Wimbledon 2023" (Wimbledon)
✓ conf_alt_02_ext   (dvBr9Wr8BCY): "Carlos Alcaraz vs Novak Djokovic: Extended Highlights | Wimbledon 2023 Final" (Wimbledon)
✓ conf_alt_03_full  (6I06-ITW88k): "Rafael Nadal v Daniil Medvedev Full Match (Final) | Australian Open 2022" (Australian Open)
✓ conf_alt_03_ext   (v27M_RgrLzU): "Rafael Nadal v Daniil Medvedev Extended Highlights (Final) | Australian Open 2022" (Australian Open)
✓ conf_alt_04_full  (FE2or3g488o): "Stefanos Tsitsipas v Novak Djokovic Full Match | Australian Open 2023 Final" (Australian Open)
✓ conf_alt_04_ext   (N2Dtsx-6aDc): "Stefanos Tsitsipas v Novak Djokovic Extended Highlights | Australian Open 2023 Final" (Australian Open)
```

### Unverified Format, Container & Rally Claims
To maintain rigorous research honesty, the following items are formally tagged as **strictly unverified** in this shortlist:
- **Resolution & Frame Rate**: Broadcast uploads claim 1080p / 4K UHD at 50.0 or 60.0 fps in web metadata. However, actual decoded container dimensions (`width`, `height`), pixel format (`pix_fmt`), color primaries (`bt709` vs `bt2020`), and codec (`av01`, `vp09`, `h264`) cannot be verified without physical acquisition and `ffprobe` execution.
- **Duration**: Video lengths reported above are container estimates from YouTube metadata. Exact frame counts at 24.0 fps working rate remain unverified.
- **Rally Continuity**: Feasibility assessments ("High", "Very High") are qualitative estimates based on published match reports (e.g. shot counts). The existence of unbroken 384-frame (16.0s) point-play camera coverage without director cuts or zooms can only be verified after running automated scene segmentation on the decoded frames.

---

## 5. Storage, Extraction & Timeline Estimation (to 20 September)

### 5.1 Storage Budget
- **Host Context**: Host filesystem `/home/itec/emanuele` has **27 TB available**.
- **Raw Video Footprint**:
  - Full match downloads average ~4–8 GB per match at high bitrate.
  - Targeted slice acquisition (e.g. 1 complete set or a 30–45 min continuous broadcast segment per match): ~1.5–3.0 GB per match.
  - Total raw storage for 6 primary candidates: **~12–25 GB**.
- **Extracted Frame & Artifact Footprint**:
  - Source frames for 4 candidate scenes per match (each up to 384 frames at 3840×2160):
    $6 \times 4 \times 384 \times 8\text{ MB (PNG)} \approx 73.7\text{ GB}$.
  - Player segmentation crops + alpha masks ($2 \times 384 \times 50\text{ KB} \approx 38.4\text{ MB}$ per scene): **~0.9 GB**.
  - Background canonical canvases & homographies: **~0.2 GB**.
  - Total storage budget for complete 6-match confirmation corpus: **$\le 100\text{ GB}$** (< 0.4% of available NFS capacity).

### 5.2 Execution Schedule to 20 September Evidence Freeze (16 Days Remaining)

| Phase | Milestone / Task | Target Dates | Duration | Dependency / Gate |
|---|---|---|:---:|---|
| **Phase 1** | **Acquisition Approval & Eligibility Rule Freeze** | Sept 4 – Sept 5 | 2 days | User / Codex review of BP54 shortlist and approval of download batch |
| **Phase 2** | **Bounded Acquisition & Decoded Verification** | Sept 6 – Sept 7 | 2 days | Download 6 primary clean matches; run `ffprobe` verification; record true stream parameters |
| **Phase 3** | **Scene Clustering & 2-Player Extraction** | Sept 8 – Sept 11 | 4 days | Extract 3–5 scenes per match; generate player silhouettes and homographies |
| **Phase 4** | **Invariant & Ingestion Verification** | Sept 12 – Sept 14 | 3 days | Run `verify.py --strict-confirmation`; confirm 6 accepted matches; update manifest |
| **Phase 5** | **Independent Confirmation Runs (E2 Gate B)** | Sept 15 – Sept 18 | 4 days | Run frozen winning PointStream config vs. AV1/VVC on confirmation scenes across all 4 spans |
| **Phase 6** | **Evidence Freeze & Manuscript Integration** | Sept 19 – Sept 20 | 2 days | Verify null controls, bounds, timing ledgers; freeze evidence for 30 Sept submission |

---

## 6. Proposed Bounded Acquisition Batch (`confirmation_acquisition_batch_01`)

> [!IMPORTANT]
> **No downloads are executed in this session.** The following bounded batch is submitted for explicit user and Codex approval before any network download or annotation script is invoked.

### Batch Specification
- **Batch Identifier**: `confirmation_acquisition_batch_01`
- **Scope**: Download a targeted 45-minute continuous broadcast segment (encompassing a full set with top extended rallies) from each of the 6 primary **provisional clean** matches:
  1. `conf_cand_01`: Sinner vs. Medvedev (AO 2024 Final, Set 4 & 5 segment)
  2. `conf_cand_02`: Sabalenka vs. Zheng (AO 2024 Final, Set 2 segment; URL `8EShbWpBm_0`)
  3. `conf_cand_03`: Gauff vs. Sabalenka (US Open 2023 Final, Set 2 segment)
  4. `conf_cand_04`: Medvedev vs. Djokovic (US Open 2023 Final, Set 2 segment)
  5. `conf_cand_05`: Swiatek vs. Sabalenka (Madrid 2024 Final, Set 3 segment)
  6. `conf_cand_06`: Djokovic vs. Ruud (RG 2023 Final, Set 1 segment)
- **Target Destination**: `/home/itec/emanuele/pointstream-data/assets/confirmation_raw/` (external data root, off-tree).
- **Download Tool**: `yt-dlp` (via conda environment) selecting best available video stream.
- **Estimated Total Download Size**: ~15–22 GB.
- **Post-Download Technical Gate**:
  - Run automated `ffprobe` format inspection script to extract exact dimensions, duration, container fps, pixel format, color space, color transfer, and compute SHA256 checksums.
  - Candidates failing resolution or showing court-level / heavily edited camera work will be substituted with pre-vetted clean alternatives (`conf_alt_03` Nadal vs Medvedev AO 2022, `conf_alt_04` Tsitsipas vs Djokovic AO 2023).

---

## 7. Decisions & Authorizations Required Next

1. **Approval of Primary Clean Roster**: Approve the 6 primary clean candidates (`conf_cand_01` through `conf_cand_06`) as the target confirmation acquisition cohort.
2. **Quarantine of Alcaraz Matches**: Confirm that Carlos Alcaraz matches (`conf_alt_01` RG 2024, `conf_alt_02` Wimbledon 2023) remain quarantined in standby under `unresolved_compilation_risk` until an exhaustive scene-by-scene visual audit of `alcaraz_highlights.mp4` is completed.
3. **Approval of Bounded Acquisition Batch**: Authorize execution of `confirmation_acquisition_batch_01` in the next dispatched session.
4. **Approval of Frozen Eligibility Rule**: Confirm that the confirmation selection criteria remain strictly frozen:
   - Simultaneous 2-player coverage across 48, 96, 192, and 384 frames at 24 fps.
   - Homography canvas growth $\le 2.5\times$.
   - Consecutive frame MAD $\le 10.0$.
   - Paste-back MAE $\le 2.0$.
   - Selection strictly independent of PointStream rate–quality scores.
