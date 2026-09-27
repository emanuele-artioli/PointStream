# Data Area

## Unified SAM3.1 observation pilot — 2026-09-27

Implemented the shared observation path in `src/contracts/observation.py` and
`src/components/perception/`, registered DWPose beside YOLO, integrated the
SAM3.1 multiplex backend, and added the frozen 3-scene × 16-frame development
selection in [`manifests/sam31_pilot_v1.json`](../../manifests/sam31_pilot_v1.json).
Dataset preparation writes source/frame/object identities, typed racket geometry,
transforms, model-specific views, and artifact digests outside the code tree.
Runtime and offline tracking policies stay distinct.

The completed bounded audit is job `20260927T100838Z-ad9440f9` under
`/home/itec/emanuele/pointstream-data/outputs/sam31-unification/pilot-v1-20260927-run-05`.
It used SAM3.1 source revision `2345a4ad109ac29c569da749c91d84f10dc08c40`,
checkpoint SHA-256 `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`,
and gpu5 RTX 6000 Ada UUID `GPU-07aa7586-7116-8c7b-1e9c-e5dbdc74b162`. The code
snapshot records HEAD `768945c55e6a5a33d817be1e150824344d909db2`, selected tracked
patch digest `d7dd6782a75adddc3b70704038c979682acf85c6b9824824cb9d4e09a8253373`,
and snapshot SHA-256 `83eed424b2d7d512019793b9cc067278e28c96c9fc4c3f50ba98534c6028d7c8`.
The dispatch logs and `audit.json` are the record for the exact command, full
source identities, environment, and timings. SAM3.1 peaked at 17,200,521,728
allocated and 21,864,906,752 reserved bytes.

The audit emitted 317 view records and hashed 3,045 unique sample files. Player
coverage was 192/192 observations. Racket coverage was 63/96; the panning scene
had 0/16 racket masks, and the small-object scene had 32/48. Visual review found
unresolved racket failures in all three scenes. The explicit decisions in
[`manifests/sam31_pilot_visual_review_v1.json`](../../manifests/sam31_pilot_visual_review_v1.json)
mark every racket-only and joint view ineligible; player views remain available
as development samples. The completed external run's per-scene counts and
artifacts are in `audit.json`, `dataset_manifest.json`, and `views.jsonl`.

The review overlay has not yet been written into the external run: fleet rejected
the bounded finalization launch during host/path revalidation, including a DNS
resolution failure, so no remote files changed. The checked-in review decisions
and tested finalizer are ready to apply when a compatible host is reachable. The
current external manifest remains inactive and predates those sample-level
quarantine annotations. This pilot is from exposed development
material, does not regenerate eligible train/validation splits, and its client
payload did not carry pose or racket geometry. Therefore the exported training
conditioning is not yet proven identical to decoded client conditioning, and no
training entrypoint has been redirected. The reserved confirmation sources were
excluded. Run the audit through the fleet after inspecting all six hosts; keep
its output under `PS_DATA_ROOT` and finalize sample hashes and visual quarantine
with:

```bash
python -m experiments.jobs.fleet inspect --hosts gpu1 gpu2 gpu3 gpu4 gpu5 gpu6
python -m experiments.jobs.fleet launch \
  --gpu-memory-mib 128 --cpu-threads 1 --budget-hours 0.05 \
  --require-path /home/itec/emanuele/pointstream-data/outputs/sam31-unification/pilot-v1-20260927-run-05/audit.json \
  --require-command /home/itec/emanuele/.conda/envs/pointstream/bin/python \
  --include-untracked scripts/finalize_sam31_manifest.py \
  --include-untracked manifests/sam31_pilot_visual_review_v1.json \
  -- \
  /home/itec/emanuele/.conda/envs/pointstream/bin/python scripts/finalize_sam31_manifest.py \
  --run-dir /home/itec/emanuele/pointstream-data/outputs/sam31-unification/pilot-v1-20260927-run-05 \
  --visual-review manifests/sam31_pilot_visual_review_v1.json
```

After revalidation succeeds, this is a bounded CPU finalization step; it does not
rerun inference or make the manifest active. See the codec and generation area
notes for transport scope and backend limitations.

## Coordinator source reservation — 16 September 2026

[The reservation manifest](../../manifests/evaluation_20260916_coordinator_confirmation_reservation.json)
pins all three acquired media hashes and all existing score-free scene bounds.
Independent hashing matched acquisition. This reserves the source set; it does
not freeze the still-unselected codec procedure or authorize scoring. Native 4K
claims are excluded; uncertainty must use three matches, not scenes or frames.

The old `development_provenance_unused` field does not establish exposure: code
assigns a contamination variable and its candidate-ID predicate does not audit
these IDs. Exposure/replay checks and exact decodable PTS remain open. Preserve
the original eligibility evidence. Final execution must pin metrics, rate
ladder, codec/adaptation/install budgets, source-count verifier and failure rules
after development acceptance. No confirmation quality scores were computed.


## E03B score-free confirmation eligibility — 16 September 2026

Reserved trio hashes still match acquisition. Timestamp origin remains the E03A
stream-copy preroll record. Content scenes were frozen from 2 fps thumbnail MAD
without codec scores (`scores_computed=false`). Pointer:
`manifests/evaluation_20260916_e03b_confirmation_eligibility.json` with full
bounds in `outputs/evaluation-20260914/e03b/run-20260916-federer007/confirmation_eligibility.json`.
All three windows are 1080p and cannot confirm a native 4K claim. AO 2024 and
US Open 2023 share a tournament-year with already-scored Gate B matches (different
matches). Coordinator freeze is still required before any confirmation scoring.

## Current acquisition / integration review

Acquisition accepted complete: three media files and report/log digests verified
against Cursor #104 head `7e4da6e` acquisition pointers. All are 1080p, reserved,
and now have score-free scene/overlap records; codec scores stay untouched. E03A checks actual local timestamp
origin after stream-copy cuts, event overlap and score-free scene eligibility;
no repeat downloads. Higher-resolution confirmation cannot be claimed from these
assets. The older zero-acquired counts below are historical.

## Coordinator follow-up — E01/E02

Coordinator accepts three fresh matches as the target. E01R acquired three
bounded windows (no scores). Eligible confirmation count remains 0 pending
event/overlap/freeze checks. Pointer:
`manifests/evaluation_20260914_e01r_acquisition.json`. Stream-copy windows
begin on non-keyframes; I-frame preroll is in
`manifests/evaluation_20260915_e03a_confirmation_timestamps.json`. The
source-count policy does not fall back to exposed Gate B sources. Training
selectors exclude validation blocks and reserved confirmation IDs. Proposed
practical quality criterion uses calibrated development severe-degradation
anchors, not VMAF 20 as a floor.

## Current campaign — 14 September 2026

E01/E07 follow the [campaign](../workflow/session/evaluation-campaign/plan.md).
The user permits fewer than six independent confirmation matches to meet the
submission objective. Six remains preferred; select and record the largest
feasible untouched set prospectively, with explicit small-sample limitations and
matching manifest/verifier policy before scoring. Old source exposure remains.
Secondary-domain evaluation stays required, after the tennis advantage.

**E01 split (E01R).** Seven BP46 videos remain development. The two Gate B
matches are already scored and are not a holdout. Three reserved matches now
have unscored 15-minute windows on disk (`acquired=3`, `eligible=0`).
`required_matches=3` with small-sample limits in
`manifests/evaluation_20260914_source_count_policy.json`. Scene easy/hard splits
use camera MAD, not quality scores. Coordinator must still freeze eligibility
before any confirmation scoring.

**Evidence Revision**: Reconciled through PR #63 (`5ca87b2ee4`) and PR #72.
**Owned Scope**: Data acquisition, manifest definitions, eligibility criteria, sequence splits, domain definitions.

---

## 1. Current State

PointStream processes video sequences that exhibit salient foreground objects moving against a coherent, reconstructible background:
- **Primary Diagnostic Domain**: 4K broadcast tennis footage. Features camera pans, static court backgrounds, high player motion, and ball trajectories.
- **Data Isolation**: All raw video, intermediate frames, and large feature files live strictly outside the tracked git repository under `.ps-data-root` (see [docs/setup.md](../setup.md)). YouTube-derived source media is not redistributed.
- **Sequence Lengths**:
  - Initial tests: 8 frames (quick unit checks).
  - Gate A diagnostic: 48 frames (~2 seconds at 25 fps).
  - Extended amortization sequences: 96 and 192 frames (identified in #63 for Gate A competitive evaluation).

---

## 2. Key Decisions & Evidence Anchor

| Topic | PR / Commit | Decision & Status |
|---|---|---|
| External data root | #32 (`d436b02`), #35 (`420c3bec4a`) | Isolated `.ps-data-root` resolver; no symlinks in repo. |
| Acquisition audit | #56 (`09727a4f16`), #57 (`77f30ecbe0`) | Validated diagnostic corpus manifests and integrity hashes. |
| Domain candidates | #60 (`f6f4f72cd9`) | Screened secondary domain candidates for Gate D. |
| Long sequences | #63 (`5ca87b2ee4`) | Characterized 96- and 192-frame tennis sequences for background amortization. |

---

## 3. Next Actions

| ID | Status | Dependencies | Source | Description & Acceptance Criteria |
|---|---|---|---|---|
| `DATA-ACT-01` | Ready | None | #63 | **Long sequence manifest packaging**: Package manifests and frame bounding boxes for 96- and 192-frame tennis scenes. Acceptance: Verified clips available to `load_tier_clip` without missing frame annotations. |
| `DATA-ACT-02` | Proposed | None | #60 | **Secondary domain eligibility screening**: Screen candidates (surveillance, conferencing) for camera stability and foreground isolation. Acceptance: Documented eligibility report and rights-cleared sample manifest for Gate D. |
| `DATA-ACT-03` | Previously deferred (BP46) | `DATA-ACT-01` | `plans/BP46-long-tennis-scenes.md` | **Camera pan boundary verification**: Verify background plate stitching across camera sweeps. Acceptance: Zero canvas seam artifacts on extreme pan sequences. |

## 4. Confirmation Protocol

The seven existing sources are development data, not a clean seven-video training partition: `manifests/bp46_long_tennis_scenes.json` records prior model fitting, sweeps, calibration, and panorama work, and has no accepted confirmation videos. The six-match target concerns new independent matches; six scenes from one match do not count as six sources.

Distinguish three operations:

- **Shared model training and codec selection:** use development data. Reserve validation data for hyperparameters and rate/quality choices; never choose the winner using final test scores.
- **Per-video encoding:** fitting a background or a video-specific model on the very frames being compressed is legitimate when it is part of the fixed encoding algorithm. Count transmitted parameters, references, metadata and adaptation time; a standalone decoder must not read source frames or untransmitted fitted state. Offline use of future frames must be declared and matched in comparisons. This is the setting used by [NeRV](https://proceedings.nips.cc/paper/2021/file/b44182379bf9fae976e6ae5996e13cd8-Paper.pdf).
- **Generalization evaluation:** hold out at the level named by the claim. New matches require match-level separation. Disjoint scenes within every source can support a narrower “new scenes from known broadcasts” claim, but shared courts, players, lighting, and replay footage make them dependent. Use contiguous scene blocks, temporal gaps, replay/duplicate exclusion, separate validation and final test blocks, and source-level uncertainty. Do not retrospectively call already-inspected scenes untouched. Group separation follows the rationale in [grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).

Recommended path: use all seven already-exposed sources for Gate A development and any shared-model fitting; preserve the fresh match candidates for final confirmation. Audit their actual prior use before accepting them. Optional within-source holdouts are supplementary, with the narrower claim explicit. The September 14 user decision permits a smaller fresh-match count. Record the
selected count and uncertainty limitations prospectively and use matching
manifest/verifier policy before scoring; if no fresh sources are available,
report the deficit rather than relabel exposed content.

Grouped outer cross-validation can use each source for training in other folds and testing once, with tuning confined to inner folds. It costs multiple fits and cannot erase prior manual design exposure to these seven videos; here it is a robustness analysis, not fresh confirmation.

Next action `DATA-ACT-04` (Partial; confirmation eligibility remains open): audited candidate exposure and match identity, reserved source IDs and scene/time bounds in `manifests/gate_b_confirmation.json`, verified SHA256 integrity hashes, and launched Gate B confirmation under frozen procedure.

Audit 2026-09-09: PR #85 scored two sources, not the six independent matches required by the standing gate. It used 48 frames per source at 1080p/720p. These sources are now observed. Preserve their exposure history and label any score-driven redesign as development; do not reset them to untouched confirmation. Verify event identities, prior use, frame/shot bounds and full freeze identity for fresh sources before the next final test.
