# Registered background reconstruction

Completed CPU study on one offline prepared development window: 48 RGB frames, 640×360, 12 fps. Raw report: `gpu3:/home/itec/emanuele/pointstream-data/audits/background-full-a74efa6-20261001/results/report.json`; SHA256 `57a938ec13d8dbbf2d531803695bc0f3ac7197bff11465c9bb2bfa9c573311ca`. Frozen code `a74efa6`; exact source/parent, worker/native and receiver identities are in the accompanying JSON.

Every package charges coded plates, original appearance, masks, maps, placement metadata and archive headers. Reset arms repeat complete packages. All eight fresh receivers produce 48 frames. Independent arithmetic checks reproduce pooled and mean-frame Y-PSNR; physical byte sums and all48 common-anchor source hashes match.

| Arm | Complete B | Whole Y dB | Parent FG dB | Parent BG dB | Plates |
|---|---:|---:|---:|---:|---:|
| first_identity_q32 | 60,607 | 17.085301 | 12.360420 | 17.120441 | 1 |
| median_identity_q32 | 44,809 | 19.610861 | 12.360420 | 19.688175 | 1 |
| first_registered_q32 | 62,129 | 20.338002 | 12.360420 | 20.432869 | 1 |
| median_registered_q32 | 52,149 | 21.374750 | 12.360420 | 21.500486 | 1 |
| median_registered_q44 | 35,770 | 21.338987 | 12.360420 | 21.463526 | 1 |
| median_registered_perframe_q32 | 113,867 | 21.515489 | 29.617633 | 21.500486 | 1 |
| median_registered_reset12_q32 | 197,074 | 25.001143 | 12.360420 | 25.321523 | 4 |
| median_registered_reset24_q32 | 98,419 | 23.136101 | 12.360420 | 23.335418 | 2 |

Every arm is strictly dominated by a sampled AV1 point and a sampled VVC point. AV1 support begins above the strongest candidate; no AV1 BD-rate is inferred. Median construction, registration and resets improve this window’s background quality at measured costs; per-frame appearance improves the parent-mask foreground diagnostic. These regions are parent masks, not independent semantic truth. Single-plate SVT CQP and continuous native recipes differ; equal quantizers do not imply equal effort. Native source extraction, held-out exposure, task accuracy and perceptual qualification remain unresolved. Receiver guards describe observed Python reads/native argv rather than an OS sandbox.
