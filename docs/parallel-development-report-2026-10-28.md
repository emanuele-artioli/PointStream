# Parallel development batch: no validated baseline win

All five requests are terminal. Camera compensation improves the static
reference, but its LPIPS BD-rate is **+230.28%** versus the original achievable
SVT envelope over their overlapping quality range. The pixel method has valid
full results at much higher rates; existing anchor coverage cannot establish
its relative rate-distortion performance. DCVC stops on real decode
nondeterminism, and the independent SVT comparison stops on its cost projection.
No complete PointStream win or TOMM readiness is established.

The new batch conservatively charges **2400/7200 GPU-seconds**, leaving4800
unused. Dedicated CPU analysis is charged120/3600CPU-seconds conservatively.
The prior study remains separately closed at2980/3600GPU-seconds,620unused.
No budget is borrowed between these ledgers. No further run is launched.

## Accepted full measurements

Al is Alcaraz–Hurkacz031@129, Med is Medvedev–Struff014@194. Each encode
processes240nativeframes (25/50fps respectively); full-image metrics score
80ticks0:240:3 at1920x1080. These are familiar development windows, not untouched
confirmation. Lower LPIPS/DISTS is better. No training or generation occurred.

| Construction | Window | CRF | Total kbps | LPIPS | DISTS |
|---|---|---:|---:|---:|---:|
| Static reference + residual + frozen overlay | Al |45|216.607129|0.143709|0.081906|
| Same | Al |55|125.634629|0.173960|0.094040|
| Same | Al |63|52.920462|0.224457|0.110906|
| Paid camera compensation + residual + overlay | Al |45|206.052129|0.135084|0.078526|
| Same | Al |55|108.182962|0.158170|0.090225|
| Same | Al |63|46.733796|0.179777|0.100966|
| Paid rectangular player tiles + base | Al |32|1183.439167|0.110700|0.063885|
| Same | Al |45|524.614167|0.115737|0.070511|
| Same | Al |58|236.074167|0.124850|0.081532|
| Paid rectangular player tiles + base | Med |32|1852.233333|0.304007|0.129396|
| Same | Med |45|841.065000|0.307220|0.134099|
| Same | Med |58|357.325000|0.312056|0.140537|

Background referenceCRF35, residuals at960x544/preset4. Identity control pays
for all camera changes in its residual. Motion uses ORB2000features and frozen
RANSAC/fit rules, transmitted float32homographies, linear plane warps with
explicit borders/chroma mapping, and residuals constructed using the same
decoded geometry as the receiver. All240fits pass; motion packet7581bytes is
charged at every operating point. Complete reference/residual/header/geometry
bytes plus the old optimistic overlay side5.607129kbps are retained. This
foreground reuse remains a diagnostic with incomplete final receiver wire and
installation accounting; it is not a delivered full codec.

Pixel tiles deliver every rectangle position/size, complete base and two native
player streams, with no secret decoder mask or retained pose/reference/adapter
charges. Full-window maximum crop sizes are Al1216x684/1146x676 and
Med600x522/456x434, fixed before quality outcomes. Source-mask outliers make
these fixed tiles large. This test does not bound adaptive crops or other
pixel representations. All header/base/tile bytes are counted.

## Comparison and scientific limits

Native SVT's original three Al rates are10.1975/17.130833/33.685kbps. The
33.685kbps encode measures LPIPS0.139336/DISTS0.109729. Every static point,
and the two cheaper motion points, lose LPIPS by>0.001 to that measured cheaper
encode. This is direct finite dominance, not extrapolated anchor quality.
The high motion point improves both distances at206.052kbps, but its equal-rate
anchor is unmeasured; it is not an accepted pointwise win. Al-only background
results are not a two-window aggregate; Med background was not run.

Post-hoc CPU curve analysis follows the existing protocol: candidate Pareto
front, achievable upper-concave SVT anchor hull, PCHIP log-rate integrated only
over intersecting quality. This supplements, and does not replace, the frozen
pointwise gate. Static LPIPS BD-rate is+296.83%; motion is+230.28%, remaining
positive at+224.34% to+236.28% under candidate-distance sensitivity+/-0.001.
Static DISTS is+63.17% on an unusually thin0.001178-distance overlap; that
narrow result is descriptive and should not be treated as a broad DISTS loss.
Motion DISTS has no quality overlap: its distances are lower than all measured
anchors, at higher rates. Neither pixel metric overlaps the original Al anchor's
quality range. No BD-rate, equal-rate win, or universal failure is inferred for
these nonoverlapping comparisons. Med has no new accepted baseline curve.

Read the raw [integrity audit](parallel-development-audit-2026-10-28.json) and
[curve analysis](parallel-development-curves-2026-10-28.json). Integrity PASS
means stream framing, complete byte accounting, full score arrays/means,
geometry, historical controls and saved support replays pass. It does not mean
the scientific criterion passes, nor independently recompute the neural metrics.
All available original Al source RGB identities match; Med's full native source
YUV identity is verified. Background controls reproduce exactly.
Pixel full-base controls remain within the predeclared1e-4 tolerance.

## Baseline stops: correctness and cost, not quality results

DCVC-UF HT-L, shared A6000, passes the24-frame AlQP0/scale4 native check. At
QP21/scale2, independently decoded passes disagree on24/24frames, and the first
frame differs from the encoder reconstruction. Container identities and counts
match; peak allocated740.2MiB. This is a genuine correctness stop, not timing
variation, memory exhaustion or a reported quality loss. No full metrics or
SVT comparator ran in that request. The cause of nondeterminism is not proven;
sharing is an observed condition, not an established cause. Exact reports,
bitstreams and decoded pixels are preserved. No DCVC retry occurred.

The independent SVT-only completion retains original ROI43/53/63 atscale4,
and adds one native-resolution whole-frameCRF63/preset4 coverage point per
window, frozen before its outcome. It passes all smoke correctness, source,
neural-control, byte and memory checks. Smoke command112.474733s succeeds.
Its conservative full projection1195.292118s exceeds the900s full allowance,
so full evaluation never launches. This is an absolute cost gate; there is no
warm-timing-variation veto. Its24-frame/3-tick scores are not accepted full
quality evidence. No further request follows this failed gate.

The next useful baseline plan is to prepare native encodes on CPUs and reserve
GPUs for judging, after resolving DCVC reproducibility on a qualified execution
configuration. That revised placement has not been implemented or run here.
The CPU follow-up also records fixed-pixel side ceilings and confirmation-split
prerequisites; no audited untouched source manifest has been frozen.

## Jobs, sources and budget

| Job | Outcome | Source Git | GPU | Admitted s | Through terminal s | Charged GPU-s |
|---|---|---|---|---:|---:|---:|
|20261028T143341Z-c50f6395|full complete|140b56ea|gpu2 A6000|436.652|483.641|500|
|20261028T143713Z-a3ae0af0|full complete, two windows|140b56ea|gpu3 A6000|899.895|935.706|950|
|20261028T143856Z-ae2c81a7|native correctness failure|140b56ea|gpu2 A6000|134.691|206.331|250|
|20261028T150051Z-445e051d|cost-gate stop, no full|0c1381c4|gpu2 A6000|168.452|185.398|200|
|20261028T152201Z-244ab9ae|motion full complete|31421e6e|gpu2 A6000|454.452|490.484|500|

Charges round canonical admission-through-terminal closure upward to50seconds,
including conservative control/salvage overhead. Shared execution timings are
cost records only, not performance benchmarks. Task totals: background1000/2700,
pixels950/2700, baselines450/1800. The usage interruption occurred after jobs
ended; its elapsed model-idle time is not GPU time. Resume inspected the saved
motion ID and did not replay anything.

Exact input paths/SHA256s, runtime, commands and frozen budgets are in
[specifications](experiments/parallel-specs/) and the
[append-only batch record](experiments/parallel-development.md). Runtime32e68872,
SVT4.2.0, dav1d1.5.4, FFmpeg8.1.2, Torch2.10.0+cu128, OpenCV4.10.0;
LPIPS7be5be79/DISTS397923af backbones unchanged. Exact native paths, versions,
binary hashes and GPU UUIDs are retained in dispatch/summary metadata.

Normal successful transfer removes its temporary source tar. All three source
archives were reconstructed from their exact Git revisions and selected paths,
and match native snapshot SHA256s211b4e69/3d4aa208/1d46ff03 byte-for-byte.
No selected patches or untracked research files entered these snapshots.

All10recorded supervisor/campaign PIDs are absent, both qualified hosts have
no PointStream GPU claims, and completion/decision/contention events are
delivered/acknowledged. No task-owned recurring monitor or new compute remains.
Existing workers and other users' processes were preserved.

## Preservation and verification

Read-only evidence archive:
`/home/itec/emanuele/Datasets/pointstream-data/tennis/parallel-development-20261028-244ab9ae/evidence.tar`,
SHA256`71900d0da6b7076b1ba4978da3663a1f5ec8a3f710f503cfa26fa2b524e20d5e`,
100720640bytes/105members. Destination hash and every member identity verified;
archive and all five canonical publications protected0444. It includes raw
accepted publications, unaccepted SVT smoke and DCVC failure data, native
metadata, exact source archives, independent audit/curve code and results,
input/control identities and resource/event closure records. Original large
input packs, models and datasets remain untouched at their canonical paths.

Implementation [PR270](https://github.com/emanuele-artioli/PointStream/pull/270)
merged with CI passing, mergeadeab13c. The initial CI documentation-header
failure was corrected; it did not affect executed science. Root reviewed the
native paths and wire contracts, not just test results. Final Ruff and configured
mypy pass (268source files),22focused/document checks pass. The archived portable
auditor replay is byte-identical to the recorded full audit. Result PR and Linux
CI are recorded at publication. User's latest Luna6/max,
precise launch/report delegation policy is now written into AGENTS.md.
