# 8. Evidence ledger, retractions and source audit

Audit date: 28 September 2026. Baseline Mac code revision:
`086ae9035bc0ff7b552c05eb04904224f9abb32a`. Documentation is being rebuilt on
`codex/paper-documentation-20260928`. New prose is not a new experiment.

**No confirmed PointStream win against AV1/VVC or the newly selected generative
competitors is established here.** “Documented measurement” below means the
checked-in experiment record was inspected; external run artifacts were not
rehashed/redecoded in this documentation task. Only E00 can recertify reuse.

## Sources and authority

| Source | What was recovered | Limits |
|---|---|---|
| [Refine PointStream core idea](thread://01a0d490-13ce-7833-801c-3af6034eac1c?hostId=remote-ssh-discovered%3Agpu3) | All three pages of available turns inspected for user messages, plans and final assessments. Presley→GenStream→tennis rationale; JE10/11 rejection; prior metric/GPU/conditioning audits; shared SAM3.1/DWPose/racket plan. | A remote chat's past working tree and local-only commits are historical, not the current Mac checkout. Its old host execution rules do not supersede current AGENTS. |
| Mac code and docs at the baseline above | Source, test and protocol inventory; recent foreground and perception status | Code presence is not real-checkpoint qualification; existing untracked/active demo edits are outside this change. |
| Sibling manuscript `21d887c` | Motivation, methods, evaluation caveats and HOLE/CLAIM markers; clean local main at audit | Local text is older than some September code-side references to Overleaf. No fetch/sync performed. Do not assume it is latest remote paper. |
| Local Presley `07b46952`; GenStream `05ffd90a` | Primary manuscript source for lineage and receiver assumptions | Publication status and underlying experiments not recertified; see [introduction](01-introduction.md). |
| Primary papers and author repositories | Results, comparator lists, available artifact instructions | [Related work](02-related-work.md) separates inspected releases from actual replication. |

Current primary records are recoverable from immutable source links, even after
retiring the append-only area notes:
[foreground campaign](https://github.com/emanuele-artioli/PointStream/blob/086ae9035bc0ff7b552c05eb04904224f9abb32a/docs/workflow/session/evaluation-campaign/20260924-foreground-campaign.md),
[background campaign](https://github.com/emanuele-artioli/PointStream/blob/086ae9035bc0ff7b552c05eb04904224f9abb32a/docs/workflow/session/evaluation-campaign/20260923-background-campaign.md),
[data record](https://github.com/emanuele-artioli/PointStream/blob/086ae9035bc0ff7b552c05eb04904224f9abb32a/docs/areas/data.md),
[generation record](https://github.com/emanuele-artioli/PointStream/blob/086ae9035bc0ff7b552c05eb04904224f9abb32a/docs/areas/generation.md),
[evaluation record](https://github.com/emanuele-artioli/PointStream/blob/086ae9035bc0ff7b552c05eb04904224f9abb32a/docs/areas/evaluation.md),
[historical findings](https://github.com/emanuele-artioli/PointStream/blob/086ae9035bc0ff7b552c05eb04904224f9abb32a/docs/history/findings.md).

## Documented development measurements

Unless absolute, artifact paths below are under external
`/home/itec/emanuele/pointstream-data/`; they are not instructions to create
repository `outputs/` or `assets/` directories. Exact row/run identities are in
the linked source records. Preserve native binaries, code patches, masks and
frame hashes when reopening them.

| Evidence | Scope / reported measurement | Supported interpretation / next check |
|---|---|---|
| BP21 headroom, `outputs/bp21-headroom/report.json` | Eight 4K, 48-frame scenes from six matches; reported foreground-removal BD-rate savings 14.2–18.3% across VVC/AV1/HEVC/AVC, with standard errors in [motivation](03-motivating-example.md) | Changed-target, own-source conventional-coding diagnostic. Not direct player bitrate attribution or same-source PointStream gain. Inspect ladders/overlap/mask provenance and preserve registered alarms. |
| 23 September background campaign | Perricard cleaned 86,894 B vs 104,482 B source leaves 17,588 B; Alcaraz cleaned 58,476 B vs 65,149 B leaves 6,673 B; Federer cleaned 108,192 B vs 112,295 B leaves 4,103 B | Fixed operating-point budgets, not the BP21 BD-rate statistic. Foreground, side information and corrections still need payment. Exact region qualities in background record. |
| 22 September measured Federer, `outputs/modular/measured-tennis-codec/current-short.json` | C1 plate+crops 466,166 B /31.84 weighted dB vs VVC 112,295 B /24.59 weighted dB; unified residual 759,913 B /33.21 weighted dB | Higher weighted quality at far greater bytes; no rate advantage. Restrict to this 48-frame window and its original metric. |
| `outputs/modular/warp-residual/federer007.json` | Registered intra plate 58,814 B +1,728 B maps; warp residual 107,005 B at 29.56 background dB; corrected foreground arm 179,996 B /21.26 weighted dB | Does not dominate anchor. A global warp/correction cost is observed; causal attribution to crowd, parallax or court error needs isolated masks/controls. |
| 23–24 September foreground continuation, `outputs/modular/` row ledgers indexed in campaign | Best capped Alcaraz 61,301 B /21.298 weighted dB vs 65,149 B /25.366. Temporal object color/alpha 193,195–258,631 B /29.126–31.029 weighted dB | No tested 48-frame point clears its own cap and historical weighted quality simultaneously. Different generation/transport choices require new full-client tests. |
| AnimateAnyone charged 16-frame screen, same campaign | FG 12.264 dB vs articulated paste 16.301 dB; target-alpha oracle 10.193 dB | Negative for this checkpoint/path, not every animation model or a claimable full 48-frame codec comparison. Verify original vs reproduction checkpoint identity. |
| Pix2Pix exact-video diagnostic, same campaign | Raw 256 px PSNR 21.45 vs shuffled 10.01; ~208 MiB adapted checkpoint | Capacity/conditioning diagnostic. Per-clip weights are rate-bearing; no generalizing/full-codec win. |
| Motion/crop control, 22 September | One AV1 crop +COCO17 total 136,228 B, pose 4,794 B, crop 1,962 B; rest mostly background | Pose compression alone cannot recover this particular deficit. Classical affine warp control, not generative inference. |
| E06 saved report `outputs/evaluation-20260914/e06/run-20260916-federer007-perframe-bbox/probe_report.json` | Prior losing whole-codec configuration; recorded SHA256 `52c0179c724469e56a1dedeafd8695536b44623cd0a63c48bf8128a49f8b5e33` | Rehash before reuse; residual utility does not imply total advantage. Missing timing/packing qualifications remain separate. |

## Shared perception pilot

The 27 September record identifies job `20260927T100838Z-ad9440f9`, external run
`outputs/sam31-unification/pilot-v1-20260927-run-05`, SAM source
`2345a4ad109ac29c569da749c91d84f10dc08c40`, checkpoint SHA256
`0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`, and gpu5
RTX 6000 Ada UUID `GPU-07aa7586-7116-8c7b-1e9c-e5dbdc74b162`.
Recorded snapshot HEAD `768945c55e6a5a33d817be1e150824344d909db2` plus selected
patch digest `d7dd6782a75adddc3b70704038c979682acf85c6b9824824cb9d4e09a8253373`
produced snapshot SHA256 `83eed424b2d7d512019793b9cc067278e28c96c9fc4c3f50ba98534c6028d7c8`.

The report lists 317 view records, 3,045 unique hashed sample files, player
coverage 192/192 and racket coverage 63/96. Coverage is not segmentation accuracy.
The [checked-in visual decisions](../../manifests/sam31_pilot_visual_review_v1.json)
quarantine racket-only and joint views in all three scenes. The external
finalizer did not run successfully at the recorded attempt; the manifest was
inactive, and no training entry point was redirected. Recheck external state
before retrying—do not assume a later process has or has not finalized it.

The pilot payload did not transport pose/racket geometry, so its training
conditioning does not yet establish encode→decode→render parity. It used exposed
development frames, not a regenerated eligible training/validation split.
DWPose CPU fallback after a missing CUDA provider is not GPU inference evidence.
E02 preserves the shared code and repairs these explicit gaps rather than
creating another disconnected preprocessing script.

## Retractions and claims that must not return

| Claim or shortcut | Finding / disposition |
|---|---|
| 65.4%/32.5% modular ladder “wins” | Withdrawn 22 September: constant-table runner and grey strips, not encoding. Do not cite these figures as measured results. |
| JE10 “definitive Pareto win” | Source chat verified arithmetic 64,840 B vs 65,149 B and selected object files, but runner reconstructed from decoded background/source tracks/masks in memory; complete persisted motion/silhouette payload and fresh client not established. A candidate on one Alcaraz/VVC point, not confirmed or AV1-wide. Historical remote artifacts not reverified here. |
| JE11 “multi-point amortization” | Source chat found repeated-cost extrapolation, no distinct successive-point inputs/per-point quality. Keep as analytical projection only. |
| JE05 “court line drift proves camera-pan cause” | Line-mask pixel PSNR is not measured line displacement or a causal diagnosis. Require E01/E05. |
| JE07 silhouette headline | Source chat found conflicting result/summary values. Reconcile original artifacts before promoting its IoU/rate. |
| Early perceptual/model rankings | Pre-Aug23 uncalibrated VGG labeled LPIPS, reversed VMAF inputs, invalid framewise AA calls; other self-image scoring and missing-source substitutions also invalidated earlier probes. Do not pool with repaired results. |
| “Paste beats all generators” | Checkpoint/protocol-specific at best. Local manuscript text contradicts its own listed PSNR and later LPIPS rows. Rerun compatible controls; do not exclude a whole family. |
| Training target==reference | Shortcut found in old training paths; repaired reference policy does not retrospectively qualify legacy weights. Audit source/target identity and checkpoint provenance. |
| “Unseen tennis” on previously fine-tuned videos | Seven-video exposure prevents this claim. Split by untouched match and audit all adaptation. |
| Circular plate/client reconstruction | Historical source-derived client plate or caches invalidate source-free reconstruction. Require E04 isolation. |
| ROI option accepted means ROI active | Historical AVC build ignored side data. Verify actual encode behavior, not parser acceptance. |
| Empty output with exit 0 | FFmpeg/libvvenc failures documented; reject empty files and preserve actual native fallback command/version. |
| Constant-table candidate evaluator as neural benchmark | `experiments/modular/neural_benchmark.py` thresholds supplied JSON values; it does not run model inference. No replicated neural result follows from its verdict alone. |
| Larger cached match always amortizes | Recurring residual/refresh cost can eliminate all margin. Measure E09; a smaller one-time plate does not solve a per-frame deficit. |

## Manuscript reconciliation before integration

The audited local manuscript has source markers but still contains unsupported
live/never-worse language and “smaller payload is outcome-safe” logic. Its
evaluation says static copy beats generators on PSNR despite reporting 11.82 dB
vs 12.03–12.21; another row gives AA LPIPS .570 vs same-offset static .582. These
contradictions require source/protocol reconciliation, not choosing the favorable
sentence. [Local evaluation](../../../67a9ea6275d3d9785ce57026/sections/evaluation.tex)
and [design](../../../67a9ea6275d3d9785ce57026/sections/system_design.tex)
are audited at `21d887c` only.

Code-side old paper notes refer to a different Overleaf-era `fc65fb8` containing
withdrawn ladder claims. Before edits, inspect the paper repository and remote
state under its own rules; preserve local/user work and do not force-sync.
No manuscript changes are made by this documentation rewrite.

## Evidence promotion record

Each new accepted row must name card ID, source selection and exposure, immutable
run location, code and selected patches, actual stream/decoded-frame hashes,
metric versions/regions, native tools, model deployment policy, hardware/time
eligibility and limitations. Record failed searches and unavailable artifacts.
Promote only the exact claim supported: implementation, infrastructure smoke,
development measurement, or frozen confirmation. Completion of one category
does not imply the next. [Evaluation](06-evaluation.md) defines the prospective
metric/access contract; [experiment cards](07-experiment-plan.md) define the
missing evidence.
