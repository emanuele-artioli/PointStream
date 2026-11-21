# Low-rate joint codec study

## Decision and design

The lead question already recorded at the top of PLAN is whether the complete player-and-ball codec beats ordinary and region-of-interest coding at low bitrate. The November 18 priority table did not preserve a concrete low-rate target. This study tests that possibility directly. The November 19 690.14 versus 336.18 kbps cold-start comparison is one 50-fps, 4.8-second operating point. It does not establish the result at lower rates or after valid reuse.

Use the opened Alcaraz–Hurkacz 028 W7 window first, because it already has the clearest historical positive. The source SHA-256 is `c51dace59349914da401356984de4e83da658c878011db3f54ff233077152934`, 240 frames at 25 fps over 9.6 s. Register three indicative total-rate targets **20, 50 and 125 kbps**, spanning the 19.24 kbps historical operating point and the low end of the new rally's existing conventional ladder. Targets are initial rate-discovery goals, not a requirement to force all codecs to exact values. Record actual bytes and accept measured bracketing rates; never extrapolate quality or BD-rate.

Compare the same source, cadence, frame indices, crop and color geometry with (1) generated players and paid ball stream on the clean base, (2) each player and ball component separately, and (3) plain SVT-AV1, player-plus-ball ROI SVT-AV1 and DCVC-UF. For each candidate include foreground extraction and repairs, background initialization and updates, poses, coordinates, ball pixels, references, packets, manifests and applicable adaptation assets. Show cold cost and steady-state scenarios separately. Models and any shared receiver resources must have the same availability assumptions for all competitors. If the paid object components alone exceed a proposed rate, state that infeasibility rather than substituting a free stream. Where the real player receiver or source-aligned ball track is unavailable, finish the attainable anchor and byte feasibility stages and keep the combined result pending.

The prior experiment scores every third frame only. For a genuine 25-fps stream qualify outputs at all 240 frame indices, including temporal behavior. Score matched decoded frames on independently reviewed player masks and ball positions, plus the complement and full frame. Report player-region DISTS and LPIPS, ball-crop DISTS, ball detection recall and false detections, full-frame DISTS/LPIPS and a temporal diagnostic. Report the registered 70/30 full-canvas regional metric with its exact masks and definition. No detection threshold may be selected separately per method. Report all anchor rates and foreground and background tradeoffs. Use BD-rate only across actual common quality and the project envelope/PCHIP policy, and report equal-rate quality as a secondary result.

The first development decision is whether a paid candidate shows an observable player-and-ball gain at any supported matched total rate without an unacceptable visual or temporal failure. Retain wins and costs by operating point. If the 20 kbps combined point is infeasible or misses its bracket, keep the measured achievable range and diagnose the component cost before proposing changes. Use frozen rules and unopened physical matches for confirmation. A single opened window, shorter cadence or modeled reuse remains development evidence.

## Stages and cost gates

1. **Input and arithmetic audit, CPU only.** Check source and label identities, usable sender/player and ball traces, all native tool versions, existing 25-fps sampled outputs, reference and model availability, and receiver state. Reproduce the cost sensitivity below. List actual artifact paths and SHA-256 values before choosing reusable outputs. The installed renderer's checkpoint, dependencies and full-frame receiver must be qualified for the exact source, not inferred from the newer 50-fps test.
2. **One-window CPU rate pilot.** Starting from the authenticated source, adapt the existing `docs/experiments/native-control-20261118/run.py` and matching ROI-map workflow for this window. The existing script hard-codes the Nadal source identity, 50-fps cadence, dimensions and CRF settings, so it cannot be run on Al028 unchanged. Qualify the 25-fps source and geometry, source-matched player and ball tracks, and CRF/rate controls in a narrow worker before using it. First encode/decode a short prefix for one plain and one actual two-player-and-ball ROI setting and verify geometry, timestamps, full decode and region allocation. Then use at most three settings to bracket target rates. The existing same-resolution 240-frame six-encode benchmark used 371 CPU-seconds plus smoke on a Mac. Differences in source, resolution and settings require a new measured forecast. Budget proposal is 600 local aggregate CPU-seconds for smoke and three full initial encodes, plus a 120-second audit margin. Stop if the smoke forecast or available inputs cannot support that cap. No GPU.
3. **Receiver and neural smoke.** Before a full matrix, smoke the exact real player receiver, source-derived masks and DCVC-UF on one short matched span. Verify full device/provider and kernel identity, complete output, the same-frame pixel and timestamp alignment, actual rate, memory and elapsed GPU/CPU use. Use a smoke projection with margin to size the full run and a separate approved GPU allocation. The previous same-rally DCVC job cost 377.49 GPU-seconds including setup, two settings and scoring on shared Ada, while a single 960×544 encode/decode took about 27–29 seconds. Those numbers are context, not a bound for this different source.
4. **Scale and validation.** Only after the smoke meets its resource and correctness gate, run one complete development window at the frozen settings, saving each decoded stream and all transmitted byte receipts. Check native 25-fps timestamps and all-frame output. Evaluate at least three matched rate points where the complete candidate and controls actually overlap. Then freeze the chosen codec and reuse policy before unseen-match confirmation. Run a small source-aligned blind motion review to catch flicker, missing ball, racket or body artifacts that metrics miss.

No GPU study is submitted by this preparation. The rate targets do not authorize exceeding any existing experiment budget.

For a measured reuse result, initialize the receiver once on a continuous passage, then charge each actual refresh and adaptation update, every fallback shot, all signaling, and all presented frames through the next return to court. Report how much of the charged elapsed time displays PointStream-coded play. A two-hour scenario in the arithmetic table remains a scenario until an actual reference lifetime is measured.

## What the historical low-rate player win means

The W7 result is real positive evidence for **players**, on one Alcaraz–Hurkacz window with a same-match adapter and the two-hour modeled asset reuse. The judge evaluated 80 sampled frames from a 240-frame, 25-fps source. At the corrected modeled rate **19.2398 kbps**, player DISTS was **0.2420**, while DCVC's interpolated value at that same modeled rate was **0.3030**. Player LPIPS was **0.3042** versus **0.3841**. DCVC was the strongest available player anchor at that rate, and its actual rate samples bracket the interpolation. Whole-frame DISTS was **0.1750** versus DCVC **0.1207**. This is a pointwise player-region comparison, with no ball-channel or complete operational receiver claim. [W7/W9 audit](../submission-ladder-chapter14-demo-audit-20261115.md) · [Repaired anchor data](../w9-inputs/w7-repaired-2h.json)

The corrected accounting models a 12-crop PNG bank from two observed crops, 934,098 bytes, and 1,488,252 bytes for the match adapter plus guider. Its projected **19.2398 kbps** at two hours assumes the bank and adapter remain valid throughout that interval. A prior bug undercounted the modeled bank fourfold and was corrected. Operational framing, actual reference lifetime, all-tick motion and model initialization remain to be qualified. The intended split used a seen match with evaluation clips outside the known fit manifests. A later source-code audit found unresolved upstream conditioning and overlap checks, so the strength of the untouched-footage claim needs requalification. Disclose the same-match adapter. The newer clean-court result is a different source, cadence, composition and receiver, so the two outcomes answer different questions.

## Reference reuse sensitivity

These are explicit arithmetic scenarios from saved byte receipts, not measured long-duration reconstructions. Hold the 4.8-second rally's non-reference bytes and visual output fixed, spread its **242,317 bytes of two lossless reference images** over the listed use time, and pay the other **171,764 bytes** every 4.8 seconds. This idealized calculation assumes unchanged valid references, excludes unseen reference refreshes and adds no new model or adapter bytes. The raw cold bill is **414,081 bytes**. The historical column spreads its modeled bank and adapter plus guider over the same period while holding all other historical terms fixed. It does not demonstrate those assumptions. Historical scores apply to their original sampled-frame output only.

| Assumed reuse | Current rally modeled kbps | W7 player modeled kbps |
|---|---:|---:|
| 4.8 seconds | 690.14 | 4,053.80 |
| 30 seconds | 350.89 | 662.51 |
| 1 minute | 318.58 | 339.53 |
| 10 minutes | 289.50 | 48.85 |
| 30 minutes | 287.35 | 27.31 |
| 1 hour | 286.81 | 21.93 |
| 2 hours | 286.54 | 19.24 |

Even a hypothetical two-hour reference reuse leaves the current CRF48 running-cost floor at about **286.27 kbps**, before any uncounted model delivery or reference updates. Lower rate requires changing and evaluating the actual base or component streams. The receipt and exact script bind these calculations to the input SHA-256 values. The 30-minute and one-hour historical values match previously saved W9 all-assets records. [Reproducible sensitivity](reuse-sensitivity.json)

The local availability check found no Al028 source-named video in the searched `/Users/manu/Datasets` paths. This is a filename search rather than a content-hash audit of every file. The historical source, generation, transport and judge archive identifiers are documented in the W7/W9 audit. Recover and hash-check those inputs before writing a concrete fleet specification. The current repository has the old scores and DCVC curve, so the cost analysis above is runnable now, but the newly proposed all-frame comparison is not yet an executable submission.

## Prepared native-control worker

The recovered campus copy of the original W7 source is
`/home/itec/emanuele/Datasets/archive/pre-reset-2026-10-05/tennis_games/alcaraz_hurkacz/028.mp4`.
A campus SHA-256 check identified the original `c51dace5…23152934`. A read-only
November 22 `ffprobe` on gpu1 confirmed 1920×1080, 25/1 fps and 492 frames. The
source remains on campus. The historic W7 selection likewise gives 492 frames,
1920×1080, 25 fps, and start frame 0. `@120` names a prepared window row.
The first 240 source frames are the scientific window.

[`native_control.py`](native_control.py) is a plain SVT-AV1 25-fps entrypoint,
adapted from the real November 18 control and using the shared `svtav1.code`
encoder/decoder. It verifies the input digest and original geometry, frame
count and cadence, decodes only the selected 24- or 240-frame prefix, and
validates each IVF header, frame timestamp, frame size, complete byte count and
decoded-frame count. It produces the source and decode hashes, exact commands,
tool versions, CPU time, complete rates and honest lower/upper rate support.
It records unsupported targets explicitly and does not extrapolate. Scale 4
uses the historical 480×272 geometry, while scale 2 uses 960×544.

A campus run must first be packaged through the required fleet workflow with
a substantively validating smoke and separately assigned CPU budget. Do not
run the following as an unsupervised campus job. These show the entrypoint
arguments to place in that fleet specification, after validating the selected
environment and fresh task-owned output paths:

```text
PYTHONPATH=. python docs/experiments/starved-joint-20261122/native_control.py \
  --source /home/itec/emanuele/Datasets/archive/pre-reset-2026-10-05/tennis_games/alcaraz_hurkacz/028.mp4 \
  --output <fresh-smoke-dir> --frames 24 --scale 4 --crfs 48

PYTHONPATH=. python docs/experiments/starved-joint-20261122/native_control.py \
  --source /home/itec/emanuele/Datasets/archive/pre-reset-2026-10-05/tennis_games/alcaraz_hurkacz/028.mp4 \
  --output <fresh-full-dir> --frames 240 --scale 4 --crfs 36 48 60
```

The full candidate is admitted only when the real smoke and lifecycle
overhead project within the assigned budget. Its measured three rates may
require an additional prospectively justified setting. Frame-level source
masks, ball tracks and ROI controls remain separate inputs to recover and
authenticate. The entrypoint implements the plain anchor, with no inference
about the ROI result or a combined codec quality result.

A local synthetic test source exercised the 24-frame smoke and 240-frame
three-CRF encode/decode, full IVF accounting, timestamp checks, and empty,
single-point and bracketing rate cases. The synthetic results are
infrastructure checks and are not scientific evidence. Real al028 has not
been encoded or scored by this new entrypoint.

### Prepared CPU fleet submission

The staged fleet-data copy of al028 and the packed 2026-10-09 PointStream
environment both passed read-only SHA-256 checks on gpu1. The prepared
[specification builder](build_fleet_spec.py) takes a future, timezone-qualified
deadline and a fresh output filename. It produces a CPU-only, single-affinity,
no-GPU specification with a 300-second smoke, 900-second full-stage allowance,
120-second validator allowance and 1,700-second total planning ceiling.
Those numbers are proposed caps, not measured timing or released budget.

```text
python docs/experiments/starved-joint-20261122/build_fleet_spec.py \
  --deadline <future-ISO-8601-with-timezone> --output <fresh-spec.json>
```

The specification stages the source and packed environment by SHA-256, then
runs [the fleet module](../../../experiments/tennis/starved_native_fleet.py)
from a frozen source snapshot. It requires the historical W7 SVT-AV1, ffmpeg
and dav1d binary hashes. Each stage writes intermediates to its fresh local
scratch. Only IVF streams and the report are published back into the campus
data root. Source frames and decoded YUV stay in scratch. The smoke validator
rechecks the source and worker bindings, CPU isolation, frame and cadence
metadata, encoded byte accounting, rates and supported brackets. It performs a
fresh independent decode and checks its hash against the worker decode.
It projects the full stage using 1.5 times the 24-frame elapsed cost scaled
to three full encodes, plus 120 seconds of setup margin, and refuses promotion
if the result exceeds the full allowance. The complete stage independently
checks each stream and decode in the production worker.

The builder passed the project's `validate_spec` schema check. A synthetic
native SVT stream passed the actual smoke validator and a deliberately
corrupted published IVF was rejected. The exact campus staging path and
environment hashes are verified, but the fleet module has not been executed
on campus. Before submission, inspect existing jobs and all eligible hosts,
verify workers support CPU-only single-affinity jobs, freeze and review the
source revision, establish the per-job budget and refresh the deadline. No
job was submitted as part of this preparation.
