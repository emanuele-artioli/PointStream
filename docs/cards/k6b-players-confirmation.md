# Card: tennis-player confirmation (8 GPU-hours approved; provenance gate unresolved)

Effort: high. Report to the requesting chat and append to
`docs/experiments/k6b-players.md`; do not launch chats or recurring monitors.

## Goal
Decide whether frozen step-5 AnimateAnyone player rendering transfers to unused
240-frame windows of Alcaraz–Hurkacz and Medvedev–Struff under a frozen ≥95%
court-registration gate. This tests a same-match conditional hypothesis;
it does not establish independent-match generalization. Step 5 remains failed.

## Scope and stop conditions
User approval (2026-10-24): **8 GPU-hours hard ceiling**, all smoke/failed/retry
work included, contingent on CPU provenance validation. **0 spent**. CPU source
checks found no original Alcaraz/Medvedev videos or verified extraction receipts
in the inspected archive/records; source-frame mapping remains unresolved.
Stop at that gate: no GPU runs or new generations until it clears. No fitting,
Wan-Animate or other task is included. The authorization/outcome continuation
is appended to the experiment entry. Stop on unresolved overlap, validator failure, budget
exhaustion, mathematical futility or required user judgment. No automatic
replacement, tuning, refit, or expansion of scope.

Branch: `codex/k6b-player-confirmation-plan`, isolated from main.
Read AGENTS.md, PLAN.md, docs/sessions.md and the 5b report plus the appended
“CPU preparation, awaiting approval” entry in docs/experiments/k6b-players.md.
That entry contains source hashes, complete window identities, validator contract,
comparison semantics, decision rule, cost basis and remaining provenance gates.

## Frozen cohort and rule

| Clip | Native inclusive frames | fps | K8b fits | Source SHA256 |
|---|---|---:|---:|---|
| alcaraz_hurkacz/031 | 9–248 | 25 | 240/240 | `054e4535638d8dfa3dee69258f45c231aaeab2241b91309d449be7b0934cef76` |
| alcaraz_hurkacz/013 | 93–332 | 25 | 239/240 | `1af169200b28659c503499979863b6219b0d331f532962ac69fb505d13c4976c` |
| alcaraz_hurkacz/023 | 108–347 | 25 | 239/240 | `88345a9deac75a4b6180daa56a2ec7cda2afea25babb5b549b937105bca45d1a` |
| alcaraz_hurkacz/016 | 153–392 | 25 | 234/240 | `305a369d74882a628ff7efd6b49cf3f75c56a985d78ffd798863542df20551bd` |
| alcaraz_hurkacz/008 | 141–380 | 25 | 240/240 | `355e4b0a3408e4c8f049baadf749c937e52930f57df5018867e5baa0cdd49e54` |
| alcaraz_hurkacz/047 | 3–242 | 25 | 240/240 | `7bb3a53e3b416d3b84c0344c99b29d1374f8c49b54f970034f4f29a95609e99b` |
| medvedev_struff/014 | 74–313 | 50 | 240/240 | `a2898b5cdfb463f59b5084fddfe23a2442cf2ba5feaaddb9d5414873e52dd0be` |
| medvedev_struff/037 | 229–468 | 50 | 240/240 | `97372d45ba53068078fdc85300a550221b2c4bd31a3074958b52081326e3feeb` |
| medvedev_struff/002 | 218–457 | 50 | 240/240 | `49cc09d072d740bd4d5ed02fdd24f2b3edbdc5fb16a660f66f1e8148ff8be14a` |
| medvedev_struff/017 | 86–325 | 50 | 240/240 | `64cc4bc0a0712b5daa6bc00c5ed592b31a0f8333cb36a4caf5b05d24f9e39873` |
| medvedev_struff/029 | 18–257 | 50 | 240/240 | `013979d403ec9ae3e48ce3dc49bbf261b37fb6a3dd78af71d5b8d81419612e8f` |
| medvedev_struff/021 | 126–365 | 50 | 240/240 | `fa2da44fdc64d6d95c9a111953dd6d16a088cb534dc7336a98d3cb2557d30b4a` |

Columns: clip, native inclusive frame interval, native fps, K8b fits/240,
source SHA256. Decode sequentially, never reinterpret @target as a start.
Selection: centered window of each unused N≥240 clip, exclude every old
57-window manifest clip; take first six per match ordered by
sha256("pointstream-k6b-confirm-v1:<match>/<clip>"). No scores informed selection.

Gate: ≥228/240 K8b `fits=true` rows (≤6 px), finite nonsingular homography;
missing/invalid=false. This is K6's definition, not the older ≤2 px
`registered` flag. No height, cut or pose-success filter. **115 px is
exploratory only.** All selected windows pass; therefore this sample cannot
validate gate discrimination or fallback benefits.

## Audit and pending prerequisites
These are two familiar match clusters. Clips were reviewed for K8b registration;
no new generator/anchor scores have been opened. Selected clips are disjoint
from all old manifest clip IDs and their in-clip references. The original-profile
base and both match-adapter dependency chains are verified in the preserved
job records and their hashes are recorded in the entry; recheck actual staged
bytes against those receipts before dispatch. Pre-reset finetuned_tennis
has unknown track overlap and is forbidden here. Verify original-source PTS
mapping and exact/near-duplicate replay overlap with every old training,
development and reference clip on CPU before any GPU spending. Separate clip
hashes alone do not prove source-frame independence. Known judge training overlap
blocks judging; unknown upstream pretraining exposure remains disclosed.
If a check fails, stop and revise the card before opening quality, not after.

## Future execution, only after approval
Use existing step-5 C base plus frozen matching adapters/guider, 4-bit,
guidance 2.0, 512×784 crops, 30 DDIM steps, existing deterministic seeds
and 4 px velocity keypoints every fourth frame, no fitting. Same first fitted source
reference per player, granted and unbilled. **Exact target matte R is granted
and unbilled**; this cannot establish end-to-end codec savings. Carry appearance
oracle and reference warp diagnostically. Re-encode SVT-AV1/DCVC-UF anchors on
identical windows, masks, scoring indices and duration convention. Bill adapter
and trained guider over 2 h, report unbilled and base-shipping assumptions too.
Use achievable anchor envelopes without extrapolation; DCVC encode/decode/judge
stays in one job on one GPU. Start three settings per codec as fixed in the
entry; complete all potentially hull-improving K6 settings before deciding,
within budget. No fabricated BD-rate from one method operating point.

Decision: mean of LPIPS/DISTS win shares at billed rate ≥75%, minimum six
gated windows with three per match. With all 12, ≥18/24 window-metrics passes.
Below billed bar but ≥75% unbilled goes to user; below unbilled bar fails.
Report pooled/gated, each metric/window/match, unbilled and missingness. No
interim pass; futility stop only if all remaining wins cannot reach the bar.
115 px results cannot rescue failure. No inference that frames are independent.

Implement and test the substantive validator from the experiment entry before
GPU dispatch (only after the unresolved source gate clears): hashes and dependency/overlap joins, PTS/frame/fps exactness,
227/228 gate boundaries and invalid homographies, identical mask/index sets,
keypoint round trip, generated coverage/nonflatness, reference membership,
byte/rate recomputation, real codec round trips, actual device/kernels and
finite/bracketed metrics; corruption fixtures must be rejected.

Staging ceiling: smoke 0.3 h (≤600 s each), first six 3.4 h (three per match),
remaining six 3.4 h, reserve 0.9 h; 8 h total. Expected renders 4.2 h plus
1–2 h prep/anchors and 0.6–1 h judging. Pilot must demonstrate scale fits budget.
Every stage ≤45 min, single-window render shards, save each completed item.
Use compatible Ada gpu5/gpu6 pool, pinned packs/locks and SHA256 inputs. Inspect
existing jobs/hosts; persist IDs, budget, deadline and output paths before one
bounded fleet wait. No recurring monitors. Preserve evidence read-only and
write new outputs under Datasets. Run project checks, commit/push scoped work,
report here. No GPU execution may begin until the user approves the budget.
