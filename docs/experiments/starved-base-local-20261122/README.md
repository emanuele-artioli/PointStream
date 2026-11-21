# Quarter-scale local base experiment

Decision: on the opened Nadal–Kyrgios 240-frame development rally, determine whether reducing the base to 240 × 136 at CRF 36, 48 and 60 yields a complete player-and-ball stream in the starved bitrate range and how its fixed receiver signals and native baselines compare. Inputs are the authenticated 960 × 544 source and GrabCut-cleaned, donor-filled YUV frames from the prior study. Both arms are downscaled by the same fixed area filter and encoded in SVT-AV1, then decoded at 50 fps. Use the same 240 ticks, same CRFs, and count IVF file bytes. Obtain lossless player conditioning, sidecar and manifest from the prior genuine 240-tick carrier. Rebind without changing conditioned arrays, canvas, tile payload, native timestamps or reference images. Reconstruct regional images at 12 previously reviewed ticks if original generated crops and their masks are available. Score only the actual paid base plus receiver composite, and use the original 12 region definitions.

The hypothesis is that a lower running base cost makes the separate representation viable at 20–125 kbps when the appearance reference can be reused. The competing explanation is that the base, ball and keypoints impose too high a running floor or that boundary defects erase the player benefit. Report actual complete cold-stream bytes and a separately labeled optimistic warm scenario that amortizes already-paid references only. The scenario is arithmetic, not an observed live return. AVIF reference substitutions are byte-only estimates unless their corresponding real player renderings are measured. A single opened match and model-conditioned scores do not prove a broad codec claim.

Budget is at most 900 local CPU-seconds and 30 minutes elapsed, with no GPU. First run 24 real source ticks through the actual encoder/decoder for each source arm at CRF48; validate 24 complete decoded frames, 50 fps and file accounting. Forecast six 240-frame cases at 1.5 times observed per-frame smoke CPU plus 120 CPU-seconds serialization reserve. Stop before the full run if the forecast exceeds budget. For each full stream, validate all 240 ticks, decoder output size, IVF timing and real carrier round trip. An exact 24-tick receiver packet is not a valid profile because the existing serialized player format requires 240 ticks. In full, fail if conditioning or ball sidecar is changed.

Attended compute-supervisor task `f87c3a0252cd48b3a7fe6b465f3956dd` was prepared with its private local runtime at `/private/tmp/ps-starved-agentctl-runtime-20261122` and state at `/private/tmp/ps-starved-agentctl-state-20261122`. The spec is `/private/tmp/ps-starved-base-local-spec-20261122.json`. It stages a 24-tick smoke, a second 24-tick pilot and 240-tick full only after supervision validates each. The 1,500-second wall supervisor ceiling includes setup; the scientific budget remains 900 aggregate CPU-seconds. The worker rejects a smoke whose forecast exceeds that ceiling. No output is overwritten.

The first task stopped before any computation because the source checkout's bundle traversal encountered missing Git object `9ec3cba07316ef4d05e20f4ef0bf8f937ae7502f`. Recovery uses independently committed local snapshot `d786915` of the exact current source and experiment tree plus worker, separate from the shared checkout. The corrected attended task ID is `5fee5626313a4204b3174f961da3bfd6`, runtime `/private/tmp/ps-starved-agentctl-runtime-b-20261122`, spec `/private/tmp/ps-starved-base-local-snapshot-spec-20261122.json`. It uses the identical predeclared input hashes, stages and budget. Inspect this task before any retry.

Task `5fee5626313a4204b3174f961da3bfd6` stopped at the start of smoke because the sandbox denied the supervisor's process inspection (`ps`). Its task-owned worker was terminated by the supervisor and the stage output is empty. A scoped escalation allows the required process monitor. The same frozen snapshot/spec was started as task `3f16cff383094e069c82a9bed688c876` under `/private/tmp/ps-starved-agentctl-runtime-c-20261122`. Earlier preparations consumed no measured encode results.

Task `3f16cff383094e069c82a9bed688c876` failed in Python imports before accessing inputs or encoding. The isolated snapshot lacked `tools.datasets`, imported by the existing packet module. Added the source-tree `tools` directory to that independent snapshot as revision `09d4734b4b365882f4faef6a3aeb886379823892`, keeping the worker unchanged. The replacement task is `9897d39530594990a4b5cc4ab12ceaf9` under `/private/tmp/ps-starved-agentctl-runtime-d-20261122`.

The base task succeeded. The two 24-frame preflights consumed 2.08 and 2.00 CPU-seconds, and the six 240-frame encodes plus three actual packet/sidecar builds consumed 7.83 CPU-seconds. Original SVT-AV1 CRF 36/48/60 costs 77.89/46.39/27.92 kbps. Clean bases cost 63.21/39.80/25.34 kbps, complete cold packets 507.21/483.80/469.33 kbps, and idealized removal of the already-paid 242,317 reference bytes leaves 103.34/79.94/65.47 kbps. The full receipt is `/private/tmp/agentctl-artifacts/Emanueles-MacBook-Pro.local/9897d39530594990a4b5cc4ab12ceaf9/output/full-quarter/report.json`. One 4.8-second native rally does not demonstrate warm reference validity or visual superiority.

The fixed 12-reviewed-frame regional composition is supervised separately as task `4a8859cb65354d3ebb3859ab96a00e4d` under `/private/tmp/ps-starved-regional-runtime-20261122`. It uses the existing receiver's authenticated per-frame player generations and masks, the actual decoded ball sidecar and the frozen source human masks. Both development frame and packet provenance must pass before scoring. The smoke and pilot each process one reviewed tick and enforce a 900 CPU-second forecast before the twelve-tick stage. It shares the original overall CPU cap.

Regional composition `4a8859cb65354d3ebb3859ab96a00e4d` succeeded for all twelve ticks using the real old player renderings, verified 240-tick masks and the newly rebound encoded ball tile. The six-arm crop archive and hashes are under its `output/full-regional` directory. Existing project masked DISTS scorer now runs as supervised task `e1d361fbc1914cba83b51ad1966c512b` under `/private/tmp/ps-starved-score-runtime-20261122`, with one-tick smoke and pilot before twelve ticks. It uses the pinned VGG and source-reviewed masks, excludes racket, and is not a new trained model.

DISTS task `e1d361fbc1914cba83b51ad1966c512b` stopped during import before scoring because the metric environment lacked OpenCV. The prior successful scorer used the OpenCV environment's Python site alongside the same Torch 2.8 and TorchMetrics 1.9; a direct import check verified those exact versions. Snapshot revision `cb10bd66e70dc7f1d6502d16f11877331156d310` binds that site explicitly. The replacement supervised task is `e553f65f36ab48cc87f06b823b4df87b` under `/private/tmp/ps-starved-score-runtime-b-20261122`.

### Appearance-payload arithmetic

The same two source reference images were measured previously at AVIF Q46/65/78, with actual paired image payloads 24,463/44,179/63,915 bytes. If those image payloads replaced the 242,317 lossless-reference bytes without changing any other cost, the estimated cold-start totals would be:

| Clean base CRF | Q46 image payload | Q65 image payload | Q78 image payload |
|---|---:|---:|---:|
| 36 | 144.12 kbps | 176.98 kbps | 209.87 kbps |
| 48 | 120.71 kbps | 153.57 kbps | 186.46 kbps |
| 60 | 106.25 kbps | 139.10 kbps | 172.00 kbps |

These numbers are substitution arithmetic, using the measured clean packet's non-reference bytes plus measured AVIF payload. AVIF framing may differ, and the current full packet accepts lossless reference formats. They are not serialized deliveries. Generated pixels and DISTS in the experiment use the original lossless references. The pending blinded Q65 review, and Q46/Q78 rendered outputs if needed, determine appearance acceptability. The 32 kbps tracked ball sidecar constrains the cheapest configuration; reducing base bytes cannot by itself reach 20 kbps while retaining this sidecar.

### Source-reviewed DISTS results

The supervised DISTS task `e553f65f36ab48cc87f06b823b4df87b` succeeded, including a source self-comparison. The scorer used the existing VGG DISTS implementation, 12 original reviewed ticks, each body mask and a 64×64 source-centered ball crop. The clean rows use the actual clean decoded base, authenticated old lossless-reference generated player crops and SAM masks, and decoded ball tiles. The plain rows are only the corresponding conventional decoded frames. Lower DISTS is better.

| Mode | CRF | Actual cold kbps | Ideal reference-already-present kbps | Player 1 | Player 2 | Ball crop |
|---|---:|---:|---:|---:|---:|---:|
| Plain quarter SVT-AV1 | 36 | 77.89 | 77.89 | .4363 | .4443 | .3718 |
| Plain quarter SVT-AV1 | 48 | 46.39 | 46.39 | .4527 | .4602 | .3808 |
| Plain quarter SVT-AV1 | 60 | 27.92 | 27.92 | .4859 | .4964 | .3931 |
| Clean + players + tiles | 36 | 507.21 | 103.34 | .2521 | .2682 | .2676 |
| Clean + players + tiles | 48 | 483.80 | 79.94 | .2517 | .2679 | .2727 |
| Clean + players + tiles | 60 | 469.33 | 65.47 | .2517 | .2669 | .2762 |

Under the idealized already-installed reference scenario, the 65.47 kbps clean composite beats the 77.89 kbps plain encode on all three region means. A higher-rate plain encode is not a matched-rate proof, and stronger 65.47 kbps AV1/ROI/DCVC anchors have not yet been scored. Only one opened match and 12 frames were scored. The real cold delivery is 469.33 kbps, so this result motivates a real reference-reuse comparison and cannot itself establish an operational warm rate. Metrics record lossless-reference images; the AVIF estimates above have no corresponding visual score.

[Base and packet receipts](base-receipt.json) · [Crop identities](crops-receipt.json) · [Per-tick DISTS and metric identity](dists-receipt.json). Across base encoding, regional extraction and scoring, smoke, pilot and full stages recorded 102.15 CPU-seconds, excluding the tile-decoder subprocess during regional extraction. The measured forecast and runtime were within the 900 CPU-second cap. The generated arrays and source material remain in their private local locations.

Full-frame RGB PSNR on the same 12 reviewed ticks is being measured by the exact regional composition path under attended local supervisor task `5f7b5044bc3745a283da0039b16671b8`. It retains the original anchor and sidecar byte accounting, and the three-stage source-identity/forecast checks.

The full-frame task succeeded and recorded 10.87 parent-process CPU-seconds, excluding its short tile-decoder subprocess. On the same 12 source-reviewed ticks, pooled full-frame RGB PSNR was 24.26/24.12/23.73 dB for plain CRF36/48/60 and 23.98/23.89/23.60 dB for the respective cleaned, player-composited and tiled streams. Thus the 65.47 kbps modeled reference-already-present clean CRF60 is 0.66 dB below the 77.89 kbps plain CRF36 while improving all three object DISTS means. [Full-frame receipt](fullframe-receipt.json)
