# W7: script map of the hybrid pipeline for one window (2026-10-30, from specs and code; not re-run)

Reference chain: Al 031@129 and Med 014@194 (two of the twelve 2026-10-25 windows), all in `experiments/tennis/`.
Job IDs are the fleet jobs that produced the saved inputs of the real-base judge (`docs/experiments/w5-specs/judge.json`).

| Stage | Module (command) | Saved job / output | Window-specific parts |
|---|---|---|---|
| Source staging | copy clip to `tennis/k6b-confirm-20261024-execution/sources/<match>__<clip>.mp4`, sha256 checked (`manifest.json`, `manifest-v2.json`) | CPU staging | `k6b_confirm.FROZEN` and `check_manifest` reject any cohort but the 12 frozen windows |
| Features (SAM 3.1, YOLO11x-pose) | `k6b_confirm features --windows <clip>@<centre frame>` | `20261024T174553Z-ecc5ca36` | same cohort check |
| Anchors, keypoints, K6 inputs | `k6b_confirm_run --kind anchors` wraps `k6 run` (SVT, DCVC anchors, 4 px keypoints) | `20261024T201059Z-29586984` | cohort check |
| Prepare (fit data, tracks, references) | `k6b prepare-match`, `k6b_confirm_run --kind prepare` | prepare jobs | cohort check |
| Generation (adapter, 4-bit, guidance 2.0, 512x784) | `k6b_generate generate` (`gen-match5`), `k6b_confirm_run --kind generate` | `tennis/k6b-reference-quality-20261026T093850Z/raw/generated-full-{al,med}-0.tar` | FULL-reference recipe |
| Reference bank (12 crops per match, SVT preset 8 CRF 20, half-size) | `k6b_reference_video`, `k6b_reference_prepare`, `k6b_reference_presets` | same directory (`full-p1-q20-prepared.tar`) | bank = the 12 conditioning crops of the match (my reading: 6 confirmation windows x 2 players) |
| Side-information rates (KPS1/KPS2, adapter, bank) | `k6b_rate_budget`, `k6b_rate_streams`, `k6b_rate_transport`, `w2_pose_rate` | `20261025T082944Z-29f021a5` | smoke list names Al031/Med014/Al016 |
| Paid base (scale 8/4/2 SVT, plain CRF 63) | `k6b_native_base` (`--al --med --frames`) and `k6b_rgb_diagnostic` (GPU decode) | `20261027T072713Z-2888cefe`, `20261027T074823Z-135421ee` | `SOURCES` tuple hard-codes Al031 (start 9) and Med014 |
| Receiver masks (pose-prompted SAM) | `k6b_receiver_sam`, `k6b_receiver_frame` (every-frame fresh prompt, the version used) | `tennis/k6b-receiver-frame-20261026T192107Z` | windows in the frozen archives |
| Compose and score (whole frame and R, LPIPS, DISTS, 80 ticks) | `k6b_real_base_compose`, `k6b_real_base_judge` (`--cost-model separated-v1 --no-early-stop`) | `20261027T211451Z-f6a754eb`; W5 step 1 `20261030T072024Z-10c9606e` | `--gen-al/--gen-med/--al/--med`, `PILOTS`, `SOURCE_SHA` |
| W1/W5 anchors: R, S masks; SVT half/quarter, +20/+40 QP outside R; scoring | `w1_masks`, `w1_encode`, `w1_score`, `w1_roi` (`Window`), `w1_specs`, `w5_specs`, `w5_report` | W5: masks `…192450Z-bc8195a6`, encodes `…072109Z-eaff2f20`, `…072138Z-4c6ec4e5`, `…072250Z-581e1ff7`, `…072402Z-837f2dac`, score `20261030T091552Z-fe376abe` | `w1_roi.AL/MED` constants (window id, packet dir, source sha, start) |
| DCVC-UF anchor | `w3_neural_baseline`, `w5_dcvc` | W3/W3b jobs | Al and Med triples hard-coded |
| Per-match adapter fit, base | `k6b_fit fit` (base: `--all`, rank 64; adapters: `--match`, rank 4/4-bit) | `20261023T194934Z-b0dcbb19`, `…221650Z-3d7cbe2f`, `…204452Z-d5c90446` | per-match data tars |

Missing for a new window: (1) a generic window/cohort manifest: `k6b_confirm`, `k6b_confirm_run`, `k6b_native_base`,
`k6b_real_base_judge`, `w1_roi` all pin Al031/Med014 or the 12 frozen windows by constant; (2) features, K6 inputs, prepare and
generation jobs for the six new windows (none exist); (3) masks and the receiver SAM pass for them; (4) paid base encodes, rates and DCVC
points at the new hybrid's rate; (5) `w1_roi.Window` entries (source sha, yuv sha, rates) for each; (6) for Tier 2: scene scan,
registration, adapter fit on the first half and its reference bank (all GPU). Whether the 10 other confirmation windows ever ran
the receiver-mask and real-base stages was not checked (only Al031 and Med014 are in the judge's inputs).
