# Lossy-mask transport diagnostics

All six lossy E06 arms were compared with their original delivered parent union masks across all 48 prepared 640×360 frames. All 19 packet-file identities were checked before decoding; no source RGB, segmentation model or annotation file was read. These are posthoc changes to parent model outputs, not task accuracy or agreement with human truth.

Raw report: `gpu3:/home/itec/emanuele/pointstream-data/audits/mask-diag-results-08eab82-20261001/mask-diagnostics.json`; 115,536 B; SHA256 `0332da42d2b21fdf8f001b220d2db10e27c818a2669876315318536142039041`. Exact input packet report SHA256 `9318c324bbe520a309e4495d5700dcec715b91477d1bd1dbb54449e74a565d5f`, packet code `bfab96c3549ba6ed9acfb709caf796e07a8ee67d`; diagnostic code `08eab828236005184914c5336ad26db32d3d49c6`. CPU1 on gpu3, no GPU; detached monitor completed exit 0 and released its claim.

Both parent arms contain the same 45,097 original positive mask pixels. FN and FP percentages below each use that fixed denominator; pooled IoU uses summed intersection/union, not a mean of frame ratios. Changed-raster share uses all 11,059,200 frame pixels. All 48 frame counts, package identities and original parent identities are retained in the companion JSON. No nonempty parent frame becomes empty.

| Appearance | Scale | Pooled parent-mask IoU | FN pixels (%) | FP pixels (%) | Changed raster (%) | Whole Y delta (dB) |
|---|---:|---:|---:|---:|---:|---:|
| First reference | 2 | 0.870169 | 3,107 (6.890) | 3,158 (7.003) | 0.056650 | -0.001120 |
| First reference | 4 | 0.677443 | 8,572 (19.008) | 8,819 (19.556) | 0.157254 | -0.003224 |
| First reference | 8 | 0.458197 | 16,769 (37.184) | 16,728 (37.093) | 0.302888 | -0.005273 |
| Per-frame crop | 2 | 0.870169 | 3,107 (6.890) | 3,158 (7.003) | 0.056650 | -0.003394 |
| Per-frame crop | 4 | 0.677443 | 8,572 (19.008) | 8,819 (19.556) | 0.157254 | -0.012999 |
| Per-frame crop | 8 | 0.458197 | 16,769 (37.184) | 16,728 (37.093) | 0.302888 | -0.030917 |

Scale eight changes only 0.302888% of the complete raster but removes 37.1843% of the original positive mask pixels and adds 37.0934% relative to that same denominator. Its pooled parent-mask IoU is 0.458197, despite a first-reference whole-frame Y-PSNR decrease of only 0.005273 dB. The global score therefore conceals substantial changes to this parent mask geometry. These counts do not measure downstream tracking, recognition, perceptual quality or correctness against independent annotations.

Independent checks reproduce every pooled IoU and FN/FP denominator from all 48 retained per-frame counts. The original packages and every lossy package are identified by complete bytes/SHA256 in the JSON. Native extraction and held-out exposure are not newly certified.
