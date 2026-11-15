# Experiments: native-footage ball-control intake

### 2026-11-15 — retrospective candidates and minimal compatible profile

**Decision:** the existing RacketVision footage supports a small native-footage
ball-control study, with explicit retrospective exposure. It cannot supply a
pristine held-out confirmation. No new model, encoding or fleet job was launched
by this intake. The [machine manifest](submission-ladder-ball-intake-20261115.json)
records exact source/label/model identities and the selected native labels.
ROOT's metadata ceiling is1800CPU seconds/GPU0 under the12-hour authority ending
November15 11:07:34UTC. Metadata reads used gpu6 after gpu2 access failed.

## Source identity and research exposure

The3 candidates are outside the old G1 eight-clip tennis selection. However,
**all43 RacketVision tennis test clips were already used for K5b tracker
selection**, in successful jobs `20261020T220058Z-fc8980da`,
`20261020T214901Z-0d339486`, and `20261020T214907Z-eefaac4b` (14/15/14 rallies;
[historical record](k5b.md)). New native codec reconstruction conditions can be
measured, but “untouched” and independent selection-holdout claims are unavailable.

Fresh inspection of `annotations/dataset_info.json`
(SHA`c79f8b25…9cccfa`) finds only pseudonymous match/clip IDs, media paths,
geometry/cadence and annotation-count fields. There are no per-clip source URLs,
YouTube IDs, player names or tournament identities. The count fields are zero for
these clips despite their nonempty native CSVs, so use actual CSV rows instead.
The [official paper](https://arxiv.org/html/2511.17045v1) reports professional
YouTube broadcast collection; the [official repository](https://github.com/OrcustD/RacketVision)
links packaged media/labels without a per-candidate URL map in the inspected
README. This bounded negative search does not prove provenance is unrecoverable.
Overlap with TrackNet games1–10 remains **unknown**, including possible overlap
with WASB's training broadcasts. Pseudo match IDs do not establish disjointness.

An existing eligible fleet-root archive is
`pointstream-data/tennis/k5b-2026-10-20/inputs/racketvision-tennis.tar`,
SHA`9b1ea8c4…d6a3820`; no new source copy is necessary. The previous intake
actually hashed each selected native MP4 and CSV and matched the original
member-checksum sidecar. Full identities are retained in the manifest.

## Frozen metadata-only windows

For each previously selected candidate, enumerate every contiguous64-frame
window having at least8 visible and2 not-visible original CSV labels. Choose
minimum `SHA256("pointstream-ladder-ball-window-v1:<match>_000:<start>")`.
Selection used no new image-model or codec outcomes.

| Native source | Full frames / duration | Full visible / not-visible | Selected inclusive frames | Selected visible / not-visible / unknown |
|---|---:|---:|---|---:|
| match11_000 |249 /9.96s|46 /4|35–98|12 /2 /50|
| match126_000 |240 /9.60s|39 /11|91–154|11 /3 /50|
| match12_000 |249 /9.96s|48 /2|11–74|12 /2 /50|

There are67,106,50 qualifying windows respectively; no full-clip fallback was
needed. All3 native videos are1920×1080,25fps,BT.709 limited range.

The downloaded dataset README and native CSV contract define zero-based frame
numbers, native1080p coordinates, visibility1=visible and0=not visible. **Not
visible does not necessarily mean physically absent or out of frame.** Unlike
TrackNet's explicit out-of-frame labels, these cannot establish absence safety
without source review. Unlisted frames are unknown, not negatives. Use only
`all/<match>/csv/000_ball.csv`, never interpolated trajectories. Preserve every
frame in decoding/full-frame scoring and use native annotations only for task
and region denominators.

## Interface and model constraints

Existing `k5b_trackers.Zoo.transform` uses global1280×720 source coordinates;
`BallTrack.run/run_gpu` likewise scale predictions to global1280×720, regardless
of input-array size. Directly feeding1080p and treating results as native
coordinates is invalid. The frozen K5d encoder also assumes TrackNet ZIP input
and30fps, while its source oracle assumes dense TrackNet rows and1280×720 bounds.
Do not relabel those frozen jobs.

The existing RacketVision reader already uses a compatible policy: area-downscale
the video to1280×720, scale labels by2/3, and retain unknown labels as−1. Its
returned native fps must replace the encoder's hardcoded30. BallTrack's selected
window median is noncausal; no real-time claim follows.

Freshly read Models manifests bind BallTrack tensors SHA`b052479e…563fec`
(original checkpoint`00d707b9…972342`), source archive`5bec7ca5…d961ff`, and WASB
checkpoint`9d391239…42b4bb`, source archive`657fb5e0…e87a55`. Existing staged paths
are in the machine manifest; actual job staging must authenticate the payloads.
BallTrack is author-reported RacketVision-trained and supplies sender positions,
so its scores are coupled diagnostics. WASB's authors report TrackNet training
games1–7/test8–10; optimizer inputs and RacketVision broadcast overlap remain
unverified. The distinct model alone does not certify training independence.

## Cheapest next-stage proposal

After ROOT reviews the egocentric result, start only with `match11_000`'s selected
64 frames and a thin native-video/sparse-label/fps adapter. Use the existing720p
tracker convention and preserve25fps for all arms. This is a **native-footage720p
control**, not a native1080p advantage. A source-only16→64 correctness/cost screen
can be bounded at600GPU/1200CPU seconds, with a1.5 forecast margin. Require
WASB source recall≥80% on the12 explicit positives at10px, report not-visible
responses, and preserve failed qualification. These2 negative labels cannot
qualify absence safety. Do not substitute candidates based on favorable scores.

Only after source/device/kernel/coordinate/cost gates, separately budget the
three-point native SVT comparison plus complete-billed streak, disk and coded
pixel controls. Use evaluation labels only for scoring; source image predictions
seed the sender. Require source-denied packet decoding and retain all-frame
metrics. A1080p route would additionally need explicit judge preprocessing,
coordinate backscaling and appearance/ROI scale decisions; it is not this
minimal next-stage proposal.
