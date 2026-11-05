# W7 Matroska decoder-profile feasibility (proposal, not submitted)

This is a new, separately identified transport sensitivity, not a pass of
job `20261104T221616Z-49f19251` or its failed strict color-metadata gate. It
neither scores quality nor changes the frozen original IVF family or its
analysis. The 100 CPU-second / 0 GPU-second phase reservation includes at most
90 seconds of single-CPU fleet lifecycle and 10 seconds external preparation
and audit. Each stage is bounded to 35 seconds, validator 10 seconds; any
failure, unexplained identity or projected budget overrun stops the campaign.
The old phase remains failed and the complete IVF family unresolved.

**Hypothesis:** a single predeclared video-only, stream-copy Matroska delivery
fits strictly below each corrected two-hour target and can reproduce the
historical native YUV without semantic changes. **Competing explanation:**
container overhead, missing cadence or native decode differences prevent one
or both lower brackets. No per-window format or parameter search is permitted.

The entire Matroska file is billed, not the AV1 payload or the temporary IVF
used internally by the native decoder. On the 240-frame 50-fps clips, the
physical file must be at most **25,351 bytes** for med000 and **25,209 bytes**
for med019. The selected receiver interpretation is the original scorer's
limited-range BT.709 with cubic upscale; it is explicitly chosen, not a claim
about the original bitstream's missing color primaries/transfer/matrix tags.
The oracle publishes both raw probe color records, including absent tags.
It requires original and new exact rational presentation times, the 240-frame
AV1 960×544 8-bit YUV420 geometry, and whole decoded native YUV hashes equal
to the historical record. The new decoder consumes only the new Matroska:
ffmpeg packet-copies it to a temporary IVF and native dav1d decodes that IVF.
No neural model, CUDA, altered frames, fabricated original metadata, or
codec quality scoring is permitted. Any stage exceeding the byte threshold
or an identity, cadence or decode check is negative for this exact profile.

Inputs, binaries and source provenance are fixed to the two archives and packed
environment in `remux-transport-oracle-20261104.json` and the hashes in
`experiments/tennis/w7_remux_job.py` plus the dav1d hash in
`experiments/tennis/w7_mkv_profile.py`. Smoke uses med000 and the full stage
med019. The smoke validator checks the published file, archived commands,
frame probes and native decode digests and requires its 1.5× observed
full-stage projection and setup reserve to fit the remaining campaign budget.
The full output needs a separate independent audit: the fleet validates smoke
only. Actual timestamps and environment records remain part of the result.

Construct a concrete prospective spec only after review and with a real future
UTC deadline:

```text
python -m experiments.tennis.w7_mkv_profile --make-spec --deadline ACTUAL_ISO_UTC_DEADLINE
```

The generator binds the existing real source paths and rejects a missing,
naive or past deadline. Its output is a proposal, not a fleet admission or
submission. A pass alone does not authorize further work: a separate design,
budget and forecast must establish physically comparable new-contract rates
and unchanged pixel/score identities across all 53 relevant scored Tier-1 SVT
points, then score at least the four previously unscored CRF 70/47 points and
controls before recalculating comparisons. If color conversion or pixels
change, affected old quality scores also require recomputation. Even after
success, the result is a transport-aware sensitivity until the comparison
contract is independently reviewed; med000's DCVC retry remains dropped.
