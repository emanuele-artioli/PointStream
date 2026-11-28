# Experiments: Multi-view CPU player receiver

### 2026-11-28 — Native 240-frame CPU receiver

This addresses the out-of-plane turn failure in the TOMM receiver-cost comparison. The operating point is the existing 240-frame cleaned AV1 base and PSCP pose stream. The comparison holds both payloads and scoring support fixed and varies only the cold appearance cache.

**Hypothesis.** Extra decoded appearance views reduce moving least squares (MLS) errors during body turns because a more similarly oriented crop exposes the relevant clothing and body surfaces. Competing explanations are unreliable torso orientation from 2D keypoints, articulation differences within the same orientation, and matte errors from comparing later views with the shot's first base frame.

**Upper bound.** An unrestricted sender with a source-aligned per-frame appearance and exact target support can reproduce the scored pixels, setting the DISTS floor at zero and strictly clearing the quality trigger. It can reproduce any finite-cache output as well. This bound relaxes the appearance budget and says nothing about receiver speed. The existing Shanghai SAM-matte control is a separate diagnostic. Its DISTS is 0.34341106538971267 and pose recall is 40/48, using the original single view. [Native receiver experiment](tomm-realtime-warp-20261128.md)

**Decision rule.** Start with a 16-frame two-player smoke on the real carrier. Repeat the receiver from cold initialization for every tested K and require pixel max absolute difference of zero. The full Shanghai pilot compares K=1, K=2 and K=3 over 240 frames. Promote a candidate only if Shanghai receiver throughput is at least 25.0 fps on one CPU and masked DISTS decreases by at least 0.010 or YOLO26x-pose recall at OKS 0.50 increases by at least 0.10 absolute against the paired K=1 arm. If both qualify, select the lower DISTS, then higher recall, then smaller cache. Run only the winning K on Sydney and Halle. Report each measured throughput independently of the broadcast cadence. If neither qualifies, stop at Shanghai. Historical Shanghai K=1 is 0.3536325709273418 DISTS, 25/48 recall and 30.823449367566838 fps under the earlier timing boundary. [Historical Shanghai receipt](tomm-warp-native240-match102-scores.json)

**Method.** Keep the original paid reference as view 0. The sender uses decoded track poses to select progressively more diverse subsequent frames using signed horizontal shoulder and hip spread normalized by torso length. The additional AVIF Q90 crops and their tick bindings are serialized as a separate cold PSR attachment, bound to the original carrier hash. Extra references use box geometry already derivable from PSCP. Receiver view selection uses decoded poses with a switching margin and two consecutive eligible frames before switching, so each frame uses one warp per player. At shot initialization, the receiver decodes the added crops and runs seeded GrabCut against the first cleaned AV1 base frame. Measure AVIF payload bytes and the complete cold attachment including framing. Verify the original carrier bytes and warm accounting are identical by construction.

**Budget.** Zero GPU-hours. No receiver SAM, including the determinism smoke. Use one pinned CPU thread for every fleet job. Per match, allow 400 seconds for the smoke, 2400 seconds for the scored run and 30 seconds per validation, with a 2900-second job budget. Stage 2 is conditional. Report receiver timing including base decode/upscale, carrier and cache decoding, matte extraction, view selection, warping and compositing. Exclude the offline sender and judges. Use the same immutable source, feature supports and judge weights as the existing native benchmark.

## Fleet ledger

| Match | Fleet job | Frozen revision | Status |
|---|---|---|---|
| Shanghai, K=1/2/3 | `20261128T122416Z-4f6db8c2` | `c2f1863770e29612ac993367fdaea3046235ca78` | Smoke rejected on gpu1 |

The Shanghai deadline is 2026-11-28 18:00 UTC. The publication root is
`/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261128T122416Z-4f6db8c2/`;
retrieve and validate the score JSON from `smoke/` and `full/`. The committed [job specification](tomm-mls-multiview-match102-spec.json) pins the input, environment and declared resource caps.
Snapshot SHA256 is `539ec0f4bdcaff039c974b75c3e2996a5a771cd1a5f38d9a41201c8b8e0ccf1e` with an empty worktree patch.
The pool passed the [fleet doctor](../fleet.md#one-entry-point) using check `1920c785c6ec420b8f9079277b70ca49`.
Archived SAM masks supply the existing scoring support. They are read by the judge, and the new receiver path never receives a matte from them.

Local synthetic verification used real AVIF Q90 encode/decode, two players, all 16 frames, both added cache sizes, and a constructed orientation change. It exercised the cold binding and zero warm bytes, rejected a different carrier hash, switched views and obtained pixel max absolute difference zero. The native smoke and 240-frame job provide the broadcast measurements.

Repository-wide Ruff and mypy passed with the CI-pinned tools after switching single-channel OpenCV scalar arguments to one-element tuples. A local pixel comparison confirmed the typed call form preserved the GrabCut matte and composite output exactly. The common packet, packet finish and pose-rate tests passed. The full suite encountered an inherited experiment-card format failure on `tomm-backbone-mimicmotion-20261128.md`. The macOS process-suspension test also failed under sandbox restrictions on `ps` and passed when rerun with process inspection available.

The first native smoke produced exact repeated pixels for all three arms and zero warm-byte additions. AVIF Q90 payload and total cold increments were 41,087 B and 41,552 B for K=2 and 80,373 B and 81,100 B for K=3. The validator blocked full scoring because Ultralytics changed PyTorch's thread setting to eight during the CPU judge, despite verified one-core affinity, empty CUDA visibility and no GPU claim. The retry pins the judge's thread setting and verifies it before each timed arm. It targets gpu5, the historical Shanghai timing host, and reduces the cap to 2,650 seconds to account for the failed smoke within the Shanghai campaign budget.
