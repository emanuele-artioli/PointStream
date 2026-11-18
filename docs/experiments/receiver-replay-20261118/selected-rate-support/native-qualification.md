# Native base qualification at the selected three rates

The six selected cold rates share three distinct base streams. The decision is whether the selected Linux FFmpeg produces exactly the previously frozen native pixel samples for each paid 960×544 IVF, in order, and whether the declared conversion yields 240 complete 1920×1080 RGB canvases. Compare the entire YUV hash against the independent saved original native control, and bind each run to the actual player carrier, manifest and joint ball sidecar. Inspect every timestamp, each plane and the hashes of every canvas. A repeated subset of canvas conversions must produce identical pixels. A mismatch stops the qualification at that base.

The local follow-up used 119.41 of its 600 aggregate CPU-seconds. Reserve at most 300 of the remainder, including a same-code first-four-frame conversion smoke and single-thread all-frame confirmation. Allow 135 seconds wall for this check in the remaining 156.16-second wall allowance. Set FFmpeg, OpenCV and numerical runtime to one thread and disable GPU access. Use the campus Linux SAM environment, its archived executable and CPU-only source-independent inputs. Preserve new logs, planes and native frame hashes in a new private directory. This check covers the native canvas entering the compositor.

## Result

The declared Linux FFmpeg 8.1.2 in the packed SAM environment decoded each actual paid base into **188,006,400 bytes** of planar 8-bit 4:2:0. The full YUV file matched the previously archived independent decoder’s hash for each CRF. Every IVF contained 240 units with PTS 0…239 in order at 50 Hz. The limited-range BT.709 conversion followed by the declared OpenCV bicubic resize produced 240 RGB24 canvases of 1920×1080 per rate. Repeating conversion at ticks 0, 11, 120 and 239 reproduced each corresponding canvas exactly.

| CRF | Cold player / joint kbps | YUV SHA-256 prefix | RGB canvas sequence SHA-256 prefix |
|---:|---:|---|---|
| 60 | 743.42 / 775.28 | ee5354e6edce | 4363c4ba383a |
| 48 | 920.86 / 952.72 | 4f549befc144 | 23715f8e0472 |
| 36 | 1,254.88 / 1,286.74 | dd8acda561d7 | 3a1ab0d1bc93 |

The run on gpu5 consumed 22.26 CPU-seconds and 21.57 wall-seconds. The receipt records the executable and packet identities, command, Y/U/V sample geometry and hashes, whole-stream hashes, and hashes of all per-frame records. The three frame-level receipts are preserved under the private selected-rate archive in `native-plane-receipts/crf{36,48,60}.json`. The original native planes are also available locally in the existing per-rate native controls. This conventional base check requires no source pixels at runtime. It did not exercise the source-denied learned receiver, masks, fallback decisions, ball composition or foreground quality measurement at these rates. [Linux CPU receipt](native-report.json) · [Reproduction script](qualify_native.py)

The result pins the exact script retained in the private campus run directory. The repository reproduction script spells the same SHA-256 helper as a named function for lint.
