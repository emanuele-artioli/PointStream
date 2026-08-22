# D — Deferred cleanups

**Wave 2, parallel.** Cheap, independent, and each unblocks something small. A
good use of a spare slot; not a critical-path stream.

**Owns exclusively:** `tests/components/**` type annotations,
`src/components/codec/roi.py` verification.
**Read first:** `plans/DEFERRED.md`.

## D1 — The 61 mypy errors (do this one first)

`mypy --config-file pyproject.toml` reports 61 errors, **all** in
`tests/components/`, none in `src/`. Mostly `dict[str, object]` where a typed
protocol is expected, plus `type: ignore` comments carrying the wrong error code.

Distribution: `test_background.py` 24, `test_rigid.py` 15, `test_domain.py` 13,
`test_temporal.py` 4, and one or two each in `test_metrics.py`,
`test_segmentation.py`, `test_generation.py`, `test_detection.py`.

**Why it matters:** `AGENTS.md` requires mypy clean before merge, so this is the
one thing between the B′ tree and a tidy merge to main. Every session that runs
mypy meanwhile has to know these are pre-existing — which is exactly the noise
that hides a real error later.

**Trap:** fix the types, do not weaken them. Adding `# type: ignore` to silence
an error is not a fix, and neither is loosening a protocol to accept `object`.
The tests are testing contract shape; that is the point of them.

## D3 — Verify the AVC region arm

`src/components/codec/roi.py` records ffmpeg's `addroi` filter for AVC as
**unverified**. AV1 and HEVC have native delta-QP maps that have been driven;
AVC's has not.

**Why it matters:** `NOTE(sec:evaluation)` item (c) commits the paper to giving
every baseline region control *wherever its encoder supports it*. If `addroi`
works and we did not use it, the AVC comparison is weaker than it claims; if it
does not work, the paper must say so rather than leave it silent.

**What to do:** encode one clip with and without the filter and confirm the
bitstream actually differs **in the labelled region** — not merely that the file
size changed, which a global quality shift would also produce. Record the ffmpeg
path and version; this host has carried two builds of the same encoder with
different capabilities.

**Bound before believing:** if the region-controlled arm shows no measurable
difference in the labelled region, that is a *finding* — `addroi` is a no-op in
this build — and it goes in the appendix, not in the bin.

## Not in this stream

**D2 (SAM3)** needs a second conda env with newer torch and is a day of work, not
an afternoon. **Never install into the `pointstream` env** — several forked
models are version-sensitive and a stray upgrade breaks them silently. If someone
picks it up, it is its own stream with its own brief.

**D4 (MOFA)** stays dropped; the rendered-trajectory arm replaces it.

## Done when

- `mypy --config-file pyproject.toml` is clean, with no new ignores.
- The AVC region arm is verified or recorded as a no-op, with encoder path and
  version.
- `ruff`, tests pass.
