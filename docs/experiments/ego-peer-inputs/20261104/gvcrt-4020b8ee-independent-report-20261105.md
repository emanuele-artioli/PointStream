# GVC guarded 16-frame peer: independent audit

Request `20261105T121902Z-4020b8ee` completed. A separate gpu6 CPU2 audit with CUDA hidden, a 100-second wall cap and one-CPU affinity ran the frozen revision's actual full-stage validator. It passed source archive and tree, spec, smoke gate, full dispatch and all substantive validator checks: staged identities, model checkpoints, sender/receiver coverage and parity, native stream/manifest and physical bytes, source isolation and denials, source-frame identities, scoring rows, device evidence and paired baseline recomputation. The audit produces no claim that individual kernel names were traced. No GPU model was rerun for this audit. The result is one fixed-QP3 point, not a rate–distortion curve or BD-rate result.

The source is the first 16 frames of `tracknet/game10/Clip3`, 1280×720 at 30 fps; all 16 labels are visible, so false-positive performance on absent frames cannot be estimated. Metrics are compared against original-source RGB; crop metrics use the registered visible-ball crops. LPIPS and DISTS are lower-is-better; PSNR is higher-is-better. The score's actual first-parameter checks and before/after physical device observations support the recorded Torch CUDA provider; they do not establish that every parameter was independently checked. Judges' training overlap remains unknown. A previous interrupted raw-source screen on the same calibration reported WASB 16/16 at 5 and 10 pixels and BallTrack 14/16 at 5, 15/16 at 10 ([record](../../k5d-ball-benchmark.md)); it is separate from this audit and did not establish native quality.

| Codec / setting | Native kb/s | Wire kb/s | PSNR dB | Full LPIPS | Full DISTS | Crop LPIPS | Crop DISTS | WASB 10px | BallTrack 10px |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GVC 3 | 338.550 | 382.155 | 21.106 | 0.1061 | 0.0861 | 0.1018 | 0.1731 | 6/16 | 9/16 |
| dcvc 0 | 123.660 | 124.695 | 28.784 | 0.1122 | 0.1334 | 0.2163 | 0.2212 | 1/16 | 2/16 |
| dcvc 21 | 313.140 | 314.175 | 33.446 | 0.0420 | 0.0653 | 0.1415 | 0.1845 | 5/16 | 6/16 |
| dcvc 42 | 691.890 | 692.925 | 36.149 | 0.0184 | 0.0320 | 0.0765 | 0.1448 | 12/16 | 12/16 |
| svt 48 | 862.215 | 863.235 | 36.735 | 0.0147 | 0.0230 | 0.0497 | 0.1233 | 15/16 | 14/16 |
| svt 55 | 615.420 | 616.440 | 35.392 | 0.0239 | 0.0345 | 0.1211 | 0.1672 | 9/16 | 9/16 |
| svt 62 | 325.245 | 326.265 | 32.334 | 0.0501 | 0.0602 | 0.1668 | 0.1991 | 4/16 | 3/16 |

At the GVC native rate, the saved achievable anchor envelope interpolates PSNR 33.712 dB versus GVC 21.106 dB, full DISTS 0.0586 versus 0.0861 and crop LPIPS 0.1351 versus 0.1018. These are equal-rate context for one point; no general codec ordering follows. GVC has 22,570 native bytes plus a 2,907-byte JSON manifest (25,477 wire bytes). DCVC/SVT wire figures include their own framing, so the overhead conventions differ; the native and wire columns make this explicit.

The isolated GPU probe observed exactly one RTX 6000 Ada (8.9), UUID `GPU-dbbb295f-6156-1133-8658-841f3f5b17d0`, matching dispatch, claimed visibility and independent scorer observation. The sender and receiver each recorded strict I/P checkpoint coverage and native decode parity. The selected backend recorded `disable_fused=true` and `customized_cuda_inference=false`; this asserts the intended eager path with actual CUDA device/model execution, not a profiled kernel trace. Runtime: Python 3.12.14, PyTorch 2.10.0+cu128, CUDA 12.8, FFmpeg 8.1.2. Smoke's complete 1.5× forecast was 265.705 seconds and passed the 500-second and campaign gates. The supervisor and campaign PIDs 299075/299082 were absent after termination; the exclusive GPU claim was absent and gpu6 CPU allocation map was empty. Other users' work was not touched.

Provenance: source revision `9d56c5e181ef1e38dd3977af71453cf6506781cf`; source identity `22aba7ddd69734099194bc756272ee0c1836f96b9de7e27e7aede1d53b8662df`; source archive `4dd3d3f1210ccdc3e31e27d53888670e75f1df2e265214c82ec3962998afd36e`; normalized spec `d8585a9140e14eecf1de6ea220225f972474da64dca47e7a28d1e6b44c56146f`. Staged model, native bridge, entropy extension, frame, scorer, metric and output hashes are preserved in the result and provenance files below.

- [detail receipt](gvcrt-4020b8ee-details-20261105.json): SHA-256 `072afc65faf7e28d358242ca024ae7d1e30465fec63271cafa7aa9a4ed06b7b5`.
- [provenance receipt](gvcrt-4020b8ee-provenance-20261105.json): SHA-256 `6c9c630dd5893696dcf44ec192c7e6f49912905bbbdea8f6cd93e0c84f68755f`.
- [validator receipt](gvcrt-4020b8ee-independent-audit-20261105.json): SHA-256 `6eb736fffa654dc9c9c4fb28a0dce31ea8e8097778219bc2c731fb5ea714d0a3`.
- [Audit script](../../../../tools/gvcrt/audit_4020b8ee.py): SHA-256 `3164b4fc67d9bf6fd97ad4ac97290df76e153e3406aa56888641a692418f97d2`.
