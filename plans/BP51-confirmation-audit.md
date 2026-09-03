# BP51: Confirmation Split & Contamination Audit Report

**Date**: 2026-09-03  
**Status**: COMPLETE  
Review repair: strict confirmation now rejects matching development/diagnostic
events under different filenames and missing/malformed prior-use audit fields.
Match labels are case/whitespace normalized; this is not automatic alias or
compilation resolution. Acquisition still needs verified event identity.
Targeted checks: 26 tests passed, 1 data-dependent test skipped; lint and mypy
passed. Existing fixture confirms genuinely clean independent matches still pass.
The committed interval records remain unchanged. Historical output mirrors are
not rewritten; use the current checkout's manifest for split decisions.

**Branch**: `antigravity/bp51-confirmation-split`  
**Historical dispatch**: [done/HANDOFF-BP51-confirmation-split.md](done/HANDOFF-BP51-confirmation-split.md)
**Related Plans**: [`plans/BP46-long-tennis-scenes.md`](file:///home/itec/emanuele/pointstream/plans/BP46-long-tennis-scenes.md), [`plans/ROADMAP.md`](file:///home/itec/emanuele/pointstream/plans/ROADMAP.md)

---

## 1. Executive Summary

A comprehensive contamination audit of all 4K raw video assets available on host was conducted to evaluate their suitability for the independent confirmation corpus (**E2 Gate B**).

### Central Findings
1. **Host Contamination**: All seven existing 4K video assets on host have been previously used in generative model fine-tuning (Animate-Anyone), headroom sweeps (BP21), metric calibration (BP23), ladder sweeps (BP24/BP31), panorama registration (BP30), or probe set triage.
2. **Accepted Confirmation Count**: **0 of 6 independent confirmation matches** are accepted in the committed manifest ([`manifests/bp46_long_tennis_scenes.json`](file:///home/itec/emanuele/pointstream/manifests/bp46_long_tennis_scenes.json)). `confirmation_videos` is strictly `[]`.
3. **Partition Classification**: The 37 scenes previously assigned to confirmation are reclassified as `development_candidate` across 5 development videos (`alcaraz_perricard`, `alcaraz_ruud`, `djokovic_federer`, `djokovic_zverev`, `sinner_alcaraz`).
4. **Diagnostic Readiness Unaffected**: The 16 diagnostic scenes across `alcaraz_highlights` and `federer_djokovic` and 1 fallback control (`alcaraz_highlights/scene_006`) remain 100% verified, extracted, and cached with 0 violations. **The diagnostic parameter search (E1) is fully unblocked.**

---

## 2. Audit Evidence & Training Count Verification

Training metadata was audited against [`assets/dataset/pointstream_aa_meta.json`](file:///home/itec/emanuele/pointstream-data/assets/dataset/pointstream_aa_meta.json), containing **114 annotated player tracks**:

| Video Asset | Exact AA Tracks | Prior Experimental Use | Event Identity | Source Type | Status |
|---|:---:|---|---|---|:---:|
| `alcaraz_highlights` | 20 | BP21 (`scene_000`, `scene_010`), BP23 calibration (`scene_000`), BP24/31 sweeps, E1 diagnostic | Multiple tournaments | Compilation | **Contaminated (Diagnostic)** |
| `federer_djokovic` | 20 | BP21 (`scene_001`, `scene_003`), BP24/31 sweeps, BP30 panorama tuning, E1 diagnostic | Cincinnati 2015 Final | Tournament Broadcast | **Contaminated (Diagnostic)** |
| `alcaraz_perricard` | 14 | BP21 (`scene_002`, 3.27% area), BP24/31 sweeps, BP30 (88 scenes), triage probe set (`scene_006`) | Beijing 2024 R32 | Tournament Broadcast | **Contaminated (Development)** |
| `alcaraz_ruud` | 4 | Animate-Anyone fine-tuning; single-player annotation only (0 overlap) | 2024 Practice Session | Practice Session | **Contaminated & Ineligible** |
| `djokovic_federer` | 20 | BP21 (`scene_003`, QP 31 substitution), BP24/31 sweeps, BP30 (224 scenes) | Wimbledon 2019 Final | Tournament Broadcast | **Contaminated (Development)** |
| `djokovic_zverev` | 16 | BP21 (`scene_002`, 10x concentration bound), BP24/31 sweeps, probe set conflict (§2.8) | Ambiguous / Unresolved | Tournament Broadcast | **Contaminated & Unresolved** |
| `sinner_alcaraz` | 20 | BP21 (`scene_001`), BP23 calibration anchor, BP24/31 sweeps, BP30 (53 scenes) | Ambiguous / Unresolved | Tournament Broadcast | **Contaminated & Unresolved** |
| **Total** | **114** | — | — | — | **0 Confirmation Matches** |

---

## 3. Provenance & Contamination Profiles by Video

### 3.1 `alcaraz_highlights`
- **Match / Title**: Carlos Alcaraz Broadcast Highlights
- **Event**: Multiple ATP / Grand Slam tournaments (unresolved match boundaries)
- **Source Type**: `compilation_highlights`
- **Prior Use**: 20 tracks in Animate-Anyone fine-tuning; BP21 headroom sweeps (`scene_000`, `scene_010`); BP23 metric calibration anchors; BP24/BP31 ladder sweeps; BP30 panorama tuning; designated E1 diagnostic corpus (5 point camera scenes + 1 crowd control).
- **Confirmation Eligibility**: Ineligible (`is_contaminated=True`, compilation source, diagnostic role).

### 3.2 `federer_djokovic`
- **Match / Title**: Roger Federer vs. Novak Djokovic
- **Event**: 2015 Western & Southern Open (Cincinnati Masters), Men's Singles Final (`verified`)
- **Source Type**: `tournament_broadcast`
- **Prior Use**: 20 tracks in Animate-Anyone fine-tuning; BP21 headroom sweeps (`scene_001`, `scene_003`); BP24/BP31 ladder sweeps; BP30 panorama registration; designated E1 diagnostic corpus (10 smooth-pan scenes).
- **Confirmation Eligibility**: Ineligible (`is_contaminated=True`, diagnostic role).

### 3.3 `alcaraz_perricard`
- **Match / Title**: Carlos Alcaraz vs. Giovanni Mpetshi Perricard
- **Event**: 2024 China Open (Beijing), Men's Singles Round of 32 (`verified`)
- **Source Type**: `tournament_broadcast`
- **Prior Use**: 14 tracks in Animate-Anyone fine-tuning; BP21 headroom sweeps (`scene_002`, where 3.27% player area fired the upper bound); BP24/BP31 ladder sweeps; BP30 panorama registration (88 scenes); baseline triage probe set (`scene_006/track_0196`).
- **Confirmation Eligibility**: Ineligible (`is_contaminated=True`, used in design and tuning; reclassified to `development_candidate`).

### 3.4 `alcaraz_ruud`
- **Match / Title**: Carlos Alcaraz & Casper Ruud Practice Session
- **Event**: 2024 Practice Session (`unresolved`)
- **Source Type**: `practice_session`
- **Prior Use**: 4 tracks in Animate-Anyone fine-tuning.
- **Confirmation Eligibility**: Ineligible (`is_contaminated=True`, non-competitive practice drills, single-player tracks with 0 simultaneous 2-player overlap; reclassified to `development_candidate`).

### 3.5 `djokovic_federer`
- **Match / Title**: Novak Djokovic vs. Roger Federer
- **Event**: 2019 Wimbledon Championships, Gentlemen's Singles Final (`verified`)
- **Source Type**: `tournament_broadcast`
- **Prior Use**: 20 tracks in Animate-Anyone fine-tuning; BP21 headroom sweeps (`scene_003`, where empty-bitstream behaviour forced QP 31 substitution); BP24/BP31 ladder sweeps; BP30 panorama registration (224 scenes).
- **Confirmation Eligibility**: Ineligible (`is_contaminated=True`, used in sweeps and panorama tuning; reclassified to `development_candidate`).

### 3.6 `djokovic_zverev`
- **Match / Title**: Novak Djokovic vs. Alexander Zverev
- **Event**: Unresolved tournament event (metadata does not distinguish ATP Finals / Olympics / Masters) (`unresolved`)
- **Source Type**: `tournament_broadcast`
- **Prior Use**: 16 tracks in Animate-Anyone fine-tuning; BP21 headroom sweeps (`scene_002`, where 0.011 plate saving fired the 10× concentration floor bound); BP24/BP31 ladder sweeps; historical probe set conflict (Research History §2.8).
- **Confirmation Eligibility**: Ineligible (`is_contaminated=True`, unresolved match identity, used in design sweeps; reclassified to `development_candidate`).

### 3.7 `sinner_alcaraz`
- **Match / Title**: Jannik Sinner vs. Carlos Alcaraz
- **Event**: Unresolved tournament event (metadata does not distinguish Beijing 2024 Final from Indian Wells / Miami) (`unresolved`)
- **Source Type**: `tournament_broadcast`
- **Prior Use**: 20 tracks in Animate-Anyone fine-tuning; BP21 headroom sweeps (`scene_001`); BP23 metric calibration anchor (`scene_001` vs `alcaraz_highlights/scene_000`); BP24/BP31 ladder sweeps; BP30 panorama tuning (53 scenes).
- **Confirmation Eligibility**: Ineligible (`is_contaminated=True`, unresolved match identity, used in calibration anchor; reclassified to `development_candidate`).

---

## 4. Code & Manifest Integration

### 4.1 Schema (`experiments/long_scenes/schema.py`)
- Added `VideoProvenance` dataclass defining structured provenance metadata (`match_name`, `event`, `event_status`, `source_type`, `prior_use`, `is_contaminated`, `contamination_reasons`, `confirmation_eligible`).
- Updated `SceneRecord.role` valid set to explicitly include `development_candidate`.
- Updated `ManifestPayload` to include `provenance` dictionary and `development_videos` list.

### 4.2 Manifest (`manifests/bp46_long_tennis_scenes.json`)
- Top-level `confirmation_videos` set to `[]`.
- Top-level `development_videos` set to `["alcaraz_perricard", "alcaraz_ruud", "djokovic_federer", "djokovic_zverev", "sinner_alcaraz"]`.
- Added authoritative `provenance` object for all 7 assets with exact training counts and prior uses.
- Updated all 37 non-diagnostic scenes from `role: "confirmation"` to `role: "development_candidate"`.
- Preserved 100% of interval records, frame hashes, coordinates, canvas growth, and paste-back measurements.

### 4.3 Verifier (`experiments/long_scenes/verify.py`)
- Added provenance and contamination validation:
  - Candidates listed in `confirmation_videos` are rejected if they lack provenance, have `is_contaminated=True`, non-empty `prior_use`, `event_status != "verified"`, `source_type != "tournament_broadcast"`, or duplicate an already-seen match identity.
  - Rejection cleanly registers as a confirmation deficit without blocking diagnostic readiness.
- Invariant separation:
  - Clean exit `0` on standard invocation:
    `status="DIAGNOSTIC_READY_CONFIRMATION_INCOMPLETE"`, verdict=`"diagnostic inputs ready; confirmation corpus incomplete (0 of 6 independent confirmation matches accepted)"`.
  - Non-zero exit `1` with `--strict-confirmation`: raises `ManifestValidationError` explicitly citing confirmation corpus deficits.

### 4.4 Extraction Pipeline (`experiments/long_scenes/extract.py`)
- Embedded `PROVENANCE` constant mapping in extraction pipeline.
- Configured candidate scene roster to assign `role: "development_candidate"`.
- Pipeline output guarantees `confirmation_videos: []` and preserves split roles on any future extraction run.

---

## 5. Test Suite & Verification Results

All 20 unit tests in [`tests/experiments/test_long_scenes.py`](file:///home/itec/emanuele/pointstream/tests/experiments/test_long_scenes.py) pass cleanly:

```bash
# 1. Manifest verification (default diagnostic-ready exit 0)
conda run -n pointstream python -m experiments.long_scenes.verify
# Output:
# Manifest verification: diagnostic inputs ready; confirmation corpus incomplete (0 of 6 independent confirmation matches accepted)
# Status: DIAGNOSTIC_READY_CONFIRMATION_INCOMPLETE (Diagnostic: READY, Confirmation: INCOMPLETE)

# 2. Strict confirmation check (fails as expected with exit 1)
conda run -n pointstream python -m experiments.long_scenes.verify --strict-confirmation
# Output:
# Manifest verification FAILED:
# confirmation corpus incomplete (1 deficit(s)):
#   - expected >= 6 independent confirmation matches, got 0 accepted: []

# 3. Unit test suite
PS_DATA_ROOT=/home/itec/emanuele/pointstream-data POINTSTREAM_DATA_TESTS=1 conda run -n pointstream pytest tests/experiments/test_long_scenes.py -v
# Output: 20 passed in 15.81s

# 4. Code quality & contracts
conda run -n pointstream ruff check experiments/long_scenes tests/experiments/test_long_scenes.py
# Output: All checks passed!
MYPY_CACHE_DIR=/tmp/mypy-pointstream-bp51 conda run -n pointstream mypy --config-file pyproject.toml experiments/long_scenes tests/experiments/test_long_scenes.py
# Output: Success: no issues found in 6 source files
conda run -n pointstream python -m src.contracts.layers
# Output: Import direction: OK — no outward imports.
```

---

## 6. Fresh Confirmation Acquisition Brief (E2 Gate B)

To satisfy the E2 Gate B independent confirmation milestone, a fresh confirmation corpus must be ingested according to the following requirements:

### Ingestion Criteria:
1. **Match Count**: At least **6 distinct, independent tournament matches** (no compilations, no practice sessions, no duplicate matches). The same players may meet in different events; repeated pairings are not automatically duplicate matches.
2. **Untouched Provenance**: Must have **0 prior use** in Animate-Anyone fine-tuning, headroom sweeps, ladder parameter searches, metric calibration, or background panorama tuning (`is_contaminated=False`, `prior_use=[]`).
3. **Verified Event Identity**: Exact tournament name, year, round, and player names verified against official ATP/WTA match records (`event_status="verified"`).
4. **Broadcast Quality**: Native 4K UHD tournament broadcast camera footage with continuous court coverage (`source_type="tournament_broadcast"`).
5. **Span Coverage**: Every confirmation match must have at least one valid scene providing continuous 2-player tracking across all 4 target spans (**48, 96, 192, and 384 frames** at 24.0 fps) with homography canvas growth $\le 2.5\times$ and paste-back MAE $\le 2.0$.
