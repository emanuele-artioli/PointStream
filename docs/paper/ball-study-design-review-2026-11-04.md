# Independent ball study: design review, 2026-11-04

**Status: CPU preparation and data/exposure audit are authorized; new GPU resource approval is pending.** K5c's
48-frame game 8 clip is development evidence, not confirmation. Its separate
WASB image judge found 42–44/48 streaks versus 0–8/48 native, with 45/48 source
qualification; all frames were visible positives. The physical BALL containers
were 237–254 bytes over 1.6 seconds. These are same-setting observations, not
matched-rate gains, a negative-frame test, a realism test or a complete codec
result. K5b's original +1% full-frame PSNR BD-rate guard failed; the user's
subsequent decision to retain the streak did not change that result. The closed
K5c allocations are 463/900 GPU-s and 1,413/1,800 CPU-s; the remainder is not a
new study allocation.

## Decision and sampling

Question: at equal transmitted rate, does the existing billed streak preserve
independently judged ball content without unacceptable image artifacts? Competing
explanations include a bright detectable mark, a misplaced or duplicated ball,
selection of positive frames, overfitting to a few matches, and costs or native
quality omitted from the comparison. First freeze a manifest and exposure audit:
match, clip, nonoverlapping interval, frame hashes, label states, previous
research use and all models' reported training exposure. All 30 TrackNet game 8–10 clips were previously scored in tracker selection
(`20261020T214833Z-bf224666`) and a CRF 62 probe. A larger TrackNet test is
therefore a retrospective benchmark with that exposure disclosed, not an
unscored or selection-naive confirmation. A newly selected interval is not a
new match; author-reported model splits do not prove checkpoint training history. Inspect available data before committing counts. In particular,
do not claim 4–8 independent games without showing they exist. If available
matches have been historically scored, describe confirmation as new held-out
windows from previously studied matches, not independent matches. No judge may
have trained on frames it judges; unresolved exposure means a provenance-limited
finding, not a clean claim. Include source qualification separately for each
window and report failures and exclusions rather than selecting only successes.

Use two development windows to select fixed parameters and diagnose denominators,
controls, rate range and runtime. One may be the historical window; ensure the
other adds distinct footage and includes negative or occluded states if eligible.
Freeze all processing, thresholds, window selection and analysis before 4–8
confirmation windows across as many eligible matches as actually exist. A frozen
TrackNet selection can be prospective relative to this new study choice only,
not independent of historical development. New-clip generalization requires
an alternative dataset with independently verified availability, split, labels
and exposure (such as OpenTTGames or RacketVision if the data audit establishes
eligibility). If none qualifies, report the retrospective engineering benchmark
and do not imply external generalization. Windows,
not frames, are the basic result unit; report per-match clusters, per-window
results, pooled frame counts and uncertainty with match clustering where the
number of independent matches permits it. With few matches, show their outcomes
and avoid a precise population or significance claim. The 48-frame prefix is
nested in its larger window and is not a separate sample. No adaptive replacement
of unfavorable or unqualified windows.

Proposed primary endpoint: per-window change in independently judged 10-pixel
visible-ball recall at equal rate against the strongest achievable native anchor,
with source recall, 5-pixel recall and position error as secondary outcomes. Fix
the rate range and a minimum meaningful improvement before confirmation; treat
unsupported overlap as inconclusive, not a win. At every registered frame retain
visible, confirmed absent, occluded/ambiguous and unknown denominators; only
visible labels enter recall, verified absent frames enter false detections per
frame, and occluded/unknown frames are reported separately. Report duplicate or
misplaced marks and trajectory errors, discontinuity and temporal flicker;
count missing detections rather than calculating error only among successes.
The protocol requires judges not trained on judged frames and source qualification;
the pilot's 80% within 10 pixels is a minimum feasibility gate, not evidence of
accuracy on synthetic marks. Prespecify a minimum number of absent frames and
report uncertainty; if unavailable, negative behavior is unresolved.

## Equal-rate fidelity and controls

Encode each window with the existing native DCVC-UF and SVT-AV1 settings at three
spread points first, adding only points needed for a shared operating range. Use
actual native stream plus fully framed BALL bytes and duration, same resolution,
frame rate, decoder and image transforms. Build each anchor's achievable upper
concave envelope, compare only supported overlapping quality/rate regions, and
use PCHIP BD-rate for suitable continuous metrics. Interpolate recall cautiously:
a small discrete count need not define a meaningful smooth BD-rate; give
per-window equal-rate comparisons and native points, never extrapolate. Report
full-frame PSNR and ball-centered fixed-label crop PSNR, LPIPS and DISTS at equal
rate, with a common source-defined crop and same crop processing for all arms.
A zero or insufficient overlap, invalid/nonmonotonic curve, or unsupported crop
metric yields an explicit inconclusive result. Preserve K5b's original guard and
separately report any newly preregistered acceptance criterion; same-setting
PSNR nondecrease does not pass a BD-rate guard. Native, streak and any control
must pay their own real transport bytes.

On development windows at one decision-relevant rate, compare the existing
streak against native, a disk, a brightness-matched mark, and a seeded random
position control with the same permissible visible geometry and an explicitly
billed packet. Include the source image and a labeled-position/source-pixel
oracle for interpretation; an oracle with source pixels or privileged labels is
not a deployable codec. Define the controls precisely before outcome review:
brightness and support matching, random seed and spatial exclusion rules,
handling of boundaries and negatives, and whether an appearance is fitted.
These controls test whether the judge rewards a bright mark or placement rather
than realistic ball content. Run only the most discriminating controls on the
frozen confirmation, not the entire rate sweep. Use independent image-only
predictions on full frames, then score against labels outside inference; never
crop around transmitted position before prediction. Report false alarms on
negative frames, and visual strips including source, native and streak at matched
rate, including failures, ghosts and occlusions. Realism needs a separately
validated measure: fixed source-defined crop LPIPS/DISTS assess fidelity, while
blinded randomized human side-by-side ratings or a qualified held-out realism
metric can test appearance. Do not call independent detector recall realism.
Small or dependent crop sets cannot support a stable KID claim without its own
sample-size and reference-set qualification.

The existing per-frame fitted renderer is unchanged, so its numerical path can
be evaluated without a new representation oracle. If proposing a new renderer,
tracker, coding representation or model, write its decision rule first and run
the cheapest oracle that *dominates the proposed alternatives at that operating
point*, including their rate treatment; a labeled or source-pixel visualization
alone is not automatically such a bound. Verify dominance on development samples
when not guaranteed by construction. A failed oracle narrows the configuration
and stops the gated build; a different configuration requires its own oracle.

## Receiver, accounting and minimal implementation

The pilot rereads native and BALL packets but performs its work in a process with
source access; it does not demonstrate source isolation. A small separate
receiver entrypoint should accept only the native stream, BALL packet, declared
geometry/timing and explicitly identified native decoder/model resources, and
write reconstructed frames. No source frames, labels, fitting outputs or encoder
memory may be visible to it. Decode the packet and render with the existing
`ball.decode_frames` / `ball.render` path, compare byte-for-byte or with a
predeclared deterministic tolerance to the existing numerical path, and fail on
mismatched frame counts, geometry, hashes or incomplete packets. Test the
receiver in a source-inaccessible directory or container, and prove the test by
removing access to source/labels; do not merely omit their CLI arguments. The
judge accepts reconstructed images only; the separate scorer joins image-only
predictions to labels after inference. Freeze production WASB/BallTrack and
codec resources, versions, hashes, execution provider and transforms. Do not
feed fitted coordinates, labels or renderer metadata to receiver or judge.
Encoder fitting and its source-dependent PSNR guard remain encoder-side and are
part of the documented method, not secrets the decoder may require.

Measure on-wire native plus BALL framing and metadata for each clip. Report cold
and warm deployment separately: one-time decoder/model binaries and weights,
model/version identifiers, initialization and any amortization assumptions;
shared installed resources must not be silently counted as free for a cold
receiver. Include any manifest/framing needed to reconstruct and distinguish
actual transmitted bytes from static installation bytes. Include encoding and
decoding wall time and resource use, without charging the judge as part of the
codec; report judging separately.

A minimal worker comprises a manifest/identity and exposure validator, native
encoder and existing fit/packet producer, isolated receiver, image-only judge,
label-only scorer, and a frozen analysis stage. Reuse existing metric and hull
implementations after checking their semantics; do not fork numerical fitting.
CPU tests should exercise packet/native parity and corruption, denial of source
and label access, prediction input boundaries, all label kinds and denominator
accounting, duration/physical-byte arithmetic, matched-rate overlap and no
extrapolation, deterministic seeded controls and output completeness. Fixtures
establish behavior only; real component jobs establish evidence. Retain images,
predictions, packet files, hashes and validation receipts for independent audit.

## Stages, cost and stopping

No further GPU allocation follows from the old caps. Before requesting a new
budget, complete CPU-only manifest and provenance work, review worker and
validator, and state the proposed fresh GPU, aggregate workload CPU, wall-time
and storage ceilings. A real representative smoke must establish device/provider,
packet isolation, frame/label counts, independent inference, both codec paths,
metric resources and process containment. Project each remaining stage with
measured startup, scaling and controls, conservative margin, staging,
validation, monitoring and teardown; if it cannot fit, stop and revise before
promotion. A bounded two-window pilot determines range and cost; freeze the
confirmation only after feasibility and scientific review. Split all jobs into
at-most-45-minute checkpointed parts and stop as soon as a registered decision
is made. Source qualification failure, missing identity/exposure, bad decoder
parity, incomplete frames, missing negative denominators needed for a claim,
unsupported equal-rate overlap or budget uncertainty narrows or stops the
relevant claim; retain failures and do not silently substitute data or models.

The old 48-frame six-setting command took about 227.5 s on a single CPU, with
262 s allocation-to-terminal on Ada. A naive fivefold frame-count extrapolation
alone is about 1,138 s for 240 frames per window, before additional controls,
metrics, startup and variability; two such windows would start around 2,275 s
of that simple work estimate. These are illustrative lower-bound scenarios,
not forecasts or a guarantee: fitting and inference may scale differently,
codecs have fixed costs, device contention and new components can dominate.
Measure a representative smoke and bounded pilot before assigning a ceiling;
existing pilot timing neither authorizes nor proves feasibility of this study.
