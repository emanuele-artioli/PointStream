# Gate A — bounded long-context coherent-rate search

Status: **design frozen for implementation review; no experiment launch is
authorized by this brief.** Agent owns adjudication. Routine implementation and
execution may be dispatched only after Gates 0–2 below pass. The full-frame
search ends on 10 September 2026.

## Question and pass rule

On eligible native-resolution broadcast-tennis footage at 24 fps, does an
offline/buffered PointStream configuration amortise its fixed background and
appearance cost over 48/96/192/384-frame contexts enough to beat **both** AV1
and VVC at ultra-low rate?

Primary quality is full-frame VMAF. Gate A passes at one predeclared duration
and access pattern only if PointStream has either:

1. negative VMAF BD-rate against each anchor over an overlapping VMAF interval
   at least 5 VMAF points wide, using at least four usable points per curve; or
2. strict low-rate-boundary dominance against each anchor: fewer total coded
   bytes than that anchor's smallest decodable point and VMAF at least as high.

The same PointStream curve must beat both anchors. A projected crossover, a
single unmatched point, a win on a metric selected after reading the run, or a
win caused only by resetting an anchor does not pass. Encode/decode time is
reported but is not a Gate-A pass condition.

## Gate 0 — freeze one measurement identity

Create a new branch/worktree from code `main` at
`1f934fa7e8ef2b93f8b5f96d92ea2ab89d8aaaf5`. Implementation will necessarily
change timing and sweep code, so the **measured implementation identity is the
reviewed implementation commit**, not `1f934fa`. Before any native encode,
write an immutable `identity.json` under a new output root
`outputs/gate-a-long-context-2026-09-05/` containing:

- full implementation commit and SHA-256 digest from
  `experiments.tier.low_rate_checkpoint.implementation_digest()`;
- clean/dirty state and a SHA-256 for every dirty tracked file (a dirty
  measurement is reviewable but cannot silently claim the commit identity);
- exact driver, metric, codec-wrapper and config-file SHA-256 values;
- data-root path and selected manifest-record SHA-256;
- video, ordered scenes, context ID, eligibility labels, native width/height,
  24 fps, colour conversion (`RGB source -> yuv420p anchors -> RGB decode`),
  and frames per scene;
- per-prefix RGB SHA-256 for 48, 96, 192 and 384 frames, computed over contiguous
  uint8 RGB bytes plus shape/dtype; no extraction, resampling or frame dropping;
- binary realpaths, file SHA-256 where readable, and complete version strings.

Development source is frozen to `alcaraz_highlights`, ordered scenes
`scene_000` (near-static) then `scene_028` (smooth pan), context
`alcaraz_highlights_main_court`, native 3840x2160, 24 fps. The known 48-frame
prefixes must reproduce RGB SHA-256
`388665774c91f980c3bf0e329d6f4e3bd7123398e99e9192854540723cc60fd6` and
`e2491f5772cab6d89bd8f32af5d691e97dcde1df3a060aa831f9c7a2371d9aeb`, and
selected-record manifest SHA-256
`840c298776ededa1ff5786be3be299ea24968cf754e3aacbf747541ecb2cb2d6`.
Any mismatch stops the run. The longer prefix hashes become part of the frozen
identity before results are scored. A missing 384-frame contiguous prefix stops
that duration; it is never filled from another shot.

Each duration has a distinct identity and checkpoint directory. Source,
implementation, preset, access pattern, rate ladder, timing schema or metric
identity changes require a new output root. Existing BP45–BP56 results are
context only and are never loaded as ranked points.

## Gate 1 — instrument timing without changing coded output

Retain BP55's three disjoint boundaries:

- `encoder_seconds`: already-decoded source input to complete transmitted
  payload, including analysis, canonical preparation, encoder-side synthesis,
  appearance/motion/background/correction coding and metadata assembly;
- `client_seconds`: a byte-only independent client from payload and metadata to
  delivered full-resolution frames, including decode, warp/restoration,
  foreground reconstruction and correction; it receives no source pixels or
  encoder-side objects;
- `evaluation_seconds`: metric controls, VMAF/Y-PSNR/SSIM, diagnostics and
  serialization, excluded from both codec clocks.

Also record cold initialization, steady-state component totals, full attempt
wall, preparation, checkpoint I/O, recovery/lost-work lower bound, host, GPU,
CPU and peak memory. Synchronize GPU work at every timed boundary. Do not add
overlapping component times or infer a missing clock by subtraction.

The timing patch passes only if controlled-clock tests prove the boundaries,
metrics do not change codec clocks, generation-off is covered, the independent
client exactly reproduces the pre-patch delivered frames, and a 48-frame
BP56-seed regression produces byte-identical payloads and identical quality.
Checkpoint state must exclude nondeterministic clock values.

## Gate 2 — verify tools, metrics and null controls

Resolve installed tools afresh. Primary AV1 and VVC presets are the slowest
valid presets actually encoded by a new codec-floor probe, never the convenience
defaults (`av1=10`, `vvc=faster`). Expected AV1 is SVT-AV1 preset 0. VVC must
record its actual encoder/decoder and selected full-reference/slowest valid
configuration; do not assume a preset name from an old probe. Both anchors use
the same ordered source frames, native size, 24 fps and yuv420p conversion.

For each codec/duration/rate, run:

- `continuous`: concatenate the two ordered scenes and permit inter prediction
  across their boundary;
- `segmented`: encode each scene independently and sum bytes and codec time.

Continuous is the headline control because PointStream reuses one compatible
background context across the boundary. Segmented is reported as the reset-cost
diagnostic. If eligibility/product wording later requires independent shots,
segmented becomes headline **before** reading that comparison.

Probe legal QP endpoints and the existing sparse walk, coarsest first. Retain
the lowest decodable point and select the smallest subset of at least four
usable, sufficiently monotone points spanning the PointStream VMAF range.
Add only adjacent legal QPs needed to create overlap or resolve a possible
crossover. Every output must be non-empty and decode to the exact frame count,
size and colour convention.

Before native points, run identical, mild, severe and unrelated metric anchors
on the same working resolution. Required order is identical > mild > severe and
mild > unrelated for VMAF/Y-PSNR/SSIM. VMAF identical must be [95,99] and
unrelated [0,40]. Run a shuffled-frame temporal null and report its full-frame
scores. Run PointStream object-stream-off in the same session. Run conventional
fallback against the matching anchor and require rate ratio [0.95,1.05] and
absolute VMAF difference <=1.0. A control failure stops before curve ranking.

## Coherent PointStream rate ladder

Generation and correction stay off for all Gate-A points: prior work has not
shown that either earns its bytes, so their invariant zero-byte contribution is
explicit rather than presented as a swept channel. Canonical background,
`libaom-av1 -usage good -cpu-used 4 -lag-in-frames 0 -bf 0`, native background
transport scale, one appearance reference per track, and existing sparse motion
representation are fixed. The BP56 seed is rung C1.

Run four ordered rungs, coarsest first:

| Rung | background CRF | appearance JPEG | appearance downscale | motion max points |
|---|---:|---:|---:|---:|
| C0 | 63 | 25 | 4 | 8 |
| C1 (BP56 seed) | 63 | 40 | 2 | 16 |
| C2 | 57 | 55 | 2 | 24 |
| C3 | 51 | 70 | 1 | 32 |

This is one semantic-quality ladder: background, appearance and motion fidelity
are nondecreasing from C0 to C3. C0/C1 share background CRF deliberately to
measure whether foreground allocation creates a useful second low-rate point;
they are not claimed to move the background ledger. For each adjacent rung,
total bytes must not fall by more than 5%; background bytes must be nondecreasing
when CRF changes; appearance and metadata bytes must change in the intended
direction or the driver records a pre-ranking alarm. If fewer than four distinct
usable rates result, add at most one predeclared refinement between the offending
rungs; do not open a Cartesian grid.

Report total bytes and bytes/frame plus background, appearance, motion/metadata,
correction and fallback ledgers. The ledger must balance exactly. Fit fixed cost
plus per-additional-frame slope only after at least three valid durations, with
uncertainty; do not call it an asymptote or marginal rate.

## Bounds written before results

Store these two-sided bands in `bounds-before-run.json`. Values outside a band
are alarms, not findings; revision needs a dated reason and a new identity.

- PointStream and anchor coded bytes: `(0, raw RGB bytes + 1 MiB]`; decoded
  shape/frame count must match exactly.
- VMAF `[0,98]`, Y-PSNR `[8,55]` dB, SSIM `[0,1]`.
- Scene-local last-minus-first: VMAF `[-25,+8]`, Y-PSNR `[-8,+3]` dB.
- VMAF BD-rate against each anchor `[-90,+300]%`; outside this deliberately
  broad search band suggests units, ordering or curve bugs.
- Each clock is finite and nonnegative; encoder/client/evaluation are each no
  greater than attempt wall plus 1 s timer tolerance. Attempt wall must cover
  their non-overlapped union. Encode/decode cannot be null on a ranked point.
- At each fixed duration, finer rungs may invert total rate or VMAF by at most
  5% of that curve's span. Endpoint inversion in both rate and quality is an
  alarm.
- Continuous anchor bytes should be no more than 1.05x segmented bytes. A
  larger value is retained but stops ranking until access-pattern construction
  is checked.

Report n (frames, scenes and source videos) and standard error for per-frame
quality diagnostics. Gate A remains development `n=1` video and licenses no
generalization.

## Adaptive execution, checkpoints and budget

One long-lived driver pays imports once. It runs detached, logs a progress line
at least every 10 minutes, and checks available GPU/CPU jobs without killing
unknown processes. Durable state is written atomically before and after every
control, codec/access/QP point and PointStream rung, and after every completed
scene/chunk. Maximum durable-progress gap is 3,599 s. Resume verifies the full
identity fingerprint and charges all known attempt wall and lost work exactly
once.

A single non-resumable encoder/scorer subprocess has a 55-minute timeout. The
48-frame timing probe must show that each longer subprocess is conservatively
projected below 55 minutes before that duration is admitted. If not, stop and
implement a semantics-preserving resumable boundary; do not raise the timeout.
Point-level checkpoints alone do not clear an operation that can lose more than
an hour.

Hard total budget: **120 cumulative machine-hours and 96 elapsed wall-hours**
from first native control, with at most 48 machine-hours on PointStream, 56 on
slow anchors and 16 on controls/scoring/retries. Reserve 15% of each pool before
starting a point. A timed-out or failed attempt is charged. No more than one
retry per identity, and only after recording a concrete fault and repair.

Advance sequentially by duration: 48 -> 96 -> 192 -> 384 frames per scene.
At each duration: controls, anchor floor/refinement, C0–C3, then adjudication.
Do not start the next duration unless all points are usable, ledgers/clocks and
controls pass, the checkpoint gap passes, budget remains, and either (a) the
PointStream/anchor gap improves with duration or (b) a plausible crossover is
within the next duration's measured amortisation range.

Stop the whole search immediately on identity mismatch, wrong decode shape,
unbalanced ledger, failed metric/fallback/client control, null timing, checkpoint
gap >=3600 s, budget exhaustion, two failures of one identity, or an unexplained
bound alarm. Also stop after a valid Gate-A pass (freeze the winning duration
and curve for Gate B), or after a valid 384-frame failure. If no crossover is
present by 10 September, record the negative full-frame search and activate the
predeclared salient-object-quality thesis; do not add generator or broad rate
axes here.

## Required artifacts and review

The new output root must contain immutable identity, bounds, tool probe, metric
and null controls, per-point checkpoints, budget ledger, heartbeat/job log,
balanced per-point records, curve comparison with overlap, failures/skips, and a
final report carrying size, quality and all timing fields together. Submitted,
succeeded and failed counts must reconcile; process exit 0 is not sufficient.

Before launch, Codex reviews the implementation diff, tests, exact generated
identity/bounds documents, dry-run plan and budget projection. Only a 48-frame
source/hash + fake/lightweight codec dry run is permitted before that review.
No native AV1/VVC/PointStream curve point is authorized by this document alone.
