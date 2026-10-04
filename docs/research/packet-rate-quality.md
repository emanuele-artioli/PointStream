# Complete packet packing and matched-source native anchors

The completed frozen `bfab96c3549ba6ed9acfb709caf796e07a8ee67d` campaign evaluates nineteen registered retained/generated package rows and ten continuous native anchor rows on the same 48-frame 640×360 prepared RGB source at 12 fps. Each source-file receipt, full RGB hash and all 48 frame hashes agrees between the two reports. The source is a retained Federer development cache; native MP4 extraction, exposure and independent task truth are not newly qualified. Encoders receive the whole prepared window, and receivers consume persisted packages or native streams using preinstalled decoding software. Package bytes include the entire physical archive. Anchor bytes include their physical stream and persisted deployment manifest. No source-derived appearance or adapted model is free.

An independent arithmetic check retains all 29 rows and all 48 per-frame luma MSE values per row. Recomputing pooled MSE, pooled PSNR and mean per-frame PSNR reproduces the stored fields exactly in this audit environment. Mean dB and pooled MSE dB are separate quantities. Each native probe reports the expected 48 frames and 640×360 raster. Original/lossless rows preserve their own current parent RGB output hash. These checks audit the saved reports; they do not rerun encoding or independently replay native media.

| Original parent | Original bytes | Batched PSM1 bytes | Batched RLE bytes | Pooled Y-PSNR, all three (dB) |
|---|---:|---:|---:|---:|
| First reference, correction off | 117,861 | 22,492 | 20,065 | 20.680077 |
| First reference, correction on | 195,878 | 100,410 | 97,983 | 24.133154 |
| Per-frame crops, correction off | 180,955 | 84,224 | 81,799 | 20.799722 |
| Per-frame crops, correction on | 253,122 | 156,293 | 153,866 | 24.150988 |

These lossless adapter gains are relative to each original envelope, not proof of a compression advantage. The already retained lossless first-reference RLE floor is smaller than the generic adapter at 17,581 bytes, with identical current reconstructed pixels.

| First-reference correction-off mask policy | Complete bytes | Pooled whole-frame Y-PSNR (dB) |
|---|---:|---:|
| Retained lossless floor | 17,581 | 20.680077 |
| Sample/repeat scale2 RLE | 16,676 | 20.678956 |
| Sample/repeat scale4 RLE | 15,046 | 20.676852 |
| Sample/repeat scale8 RLE | 14,246 | 20.674804 |

The smallest scale8 package saves 3,335 bytes (18.969%) against the retained lossless floor, while whole-frame pooled Y-PSNR declines by 0.005273 dB. This is a measured cost/whole-frame-quality trade-off on this prepared target. It does not establish a small foreground error, perceptual equivalence or preserved task utility: those were not measured. Lossy masks are only applied to correction-off inputs; stale correction-bearing packets are rejected. All registered scale2/4/8 rows remain in the compact data.

Every one of the nineteen candidate rows is strictly dominated in complete physical bytes and whole-frame pooled Y-PSNR by at least one of the five observed VVC anchor points. For example, VVC QP58 has 7,721 complete bytes and 20.819903 dB, versus 14,246 bytes and 20.674804 dB for the smallest candidate. This is a bounded observed comparison, not a statement that every semantic codec or every quality definition loses.

| Native quantizer | AV1 complete bytes / pooled Y-PSNR (dB) | VVC complete bytes / pooled Y-PSNR (dB) |
|---|---:|---:|
| 32 | 251,556 / 37.116629 | 158,246 / 31.860989 |
| 44 | 122,855 / 33.486624 | 50,136 / 27.073141 |
| 52 | 76,620 / 31.383595 | 15,178 / 22.943484 |
| 58 | 51,940 / 29.969258 | 7,721 / 20.819903 |
| 63 | 26,727 / 27.736537 | 4,162 / 18.837757 |

Candidate pooled-quality support is 20.674804–24.150988 dB; observed AV1 support is 27.736537–37.116629 dB. They are disjoint, so no AV1 BD-rate is reported or extrapolated. AV1 uses libaom, CPU-used6, CRF controls; VVC uses libvvenc medium, QP controls and 10-bit YUV420 encoding. The common scorer operates on decoded RGB converted to uint8 BT.601 luma. Equal numerical CRF/QP values do not imply equal quality or encoder effort.

[Compact numerical/provenance record](packet-rate-quality.json) preserves all nineteen package rows, ten anchor rows, exact physical packet/stream/manifest hashes, observed dominators, source/frame identity, native binary/version and environment. Full reports remain external at the recorded paths. Packet report SHA-256: `9318c324bbe520a309e4495d5700dcec715b91477d1bd1dbb54449e74a565d5f`; anchor report SHA-256: `3d71547869d18b9180c51defe74bc9e79e9db44ac08b778fcef9dc61cd93d0fa`. The report-arithmetic follow-up itself used local `/Users/manu/miniconda3/bin/python`3.12.7, NumPy2.0.0 and OpenCV5.0.0. This differs from the recorded remote measurement environment and supplies no timing result.

## General adapter and additional component qualification

The installed `src.runner.packet_packing` module is an opt-in boundary: `pack_client_envelope(payload, mask_codec="rle", batch_masks=True)` creates the charged archive, and `unpack_client_envelope(packet)` restores the ordinary client envelope. Pass the latter to `reconstruct_serialized_client(..., require_compressed=True)`. `mask_scale=2`, `4` or `8` is explicitly lossy and rejects correction-bearing inputs. The original native streams and geometry remain charged and intact; delivery rate is the packed archive length, never the expanded receiver intermediate. This pass leaves active development consumers unchanged; integration can use the reviewed API.

A same-entry-point smoke preceded a separate complete three-window lossless foreground-component campaign at `a0f70a7`. On Alcaraz scene000, scene010, and Alcaraz--Perricard scene007, complete package bytes change from 649,798/1,175,424/3,202,610 to 167,228/70,486/203,855 using batched RLE. All 16 decoded 4K frames per window have exactly the pinned historical RGB hashes. These 74.265%, 94.003%, 93.635% savings concern foreground-only packages whose missing background stays black, not a complete reconstruction advantage. The original, batched PSM1, and batched RLE variants were all retained. [Compact receipts](guided-packing-results.json) preserve identities, physical bytes, source-frame IDs and fresh receiver/native-argument receipts.
