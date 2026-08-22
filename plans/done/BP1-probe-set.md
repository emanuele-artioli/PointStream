# B′1 — Rebuild and verify the probe set

**Wave 1. Blocks every number B′ would otherwise produce.** Independent of the
other Wave-1 streams; start it immediately.

**Owns exclusively:** `assets/probe_set/**` (regenerating it),
`experiments/probe_set/**`, `tests/components/test_probe_set.py`.
**Read first:** `PLAN.md` §2.3, which records exactly how it is broken.

## The defect

`assets/probe_set` was inherited from the pre-rewrite implementation and never
checked. It is broken in two independent ways, and **using it naively yields
silently wrong frames rather than an error** — see `PLAN.md` §2.3 for the
evidence.

1. **Manifest and view disagree.** Zero of the 12 manifest-named tracks appear in
   `training_view/`; the view materialised a different, unseeded selection.
2. **Two coordinate systems.** The manifest stores *global video frame indices*;
   the extracted PNG sequences are numbered *track-locally from zero*. 5 of 12
   clips miss every frame; the other 7 would return frames nobody selected.

The underlying `assets/dataset` is intact and rich — all 12 named tracks exist,
with crops, canny, `pose_body`, `pose_racket`, skeleton, keypoints, captions and
metadata. **The data is fine. The view and the manifest are not.**

## What to build

**A regenerator that cannot produce this class of fault**, plus a verifier that
runs in CI.

- **One coordinate system, named in the schema.** Pick track-local indexing —
  it is what the extracted data actually uses — and record the global offset
  alongside it so the mapping is recoverable. Bump the schema version; the
  current `pointstream.probe_set.v1` is not trustworthy and should not be
  silently reused.
- **The manifest is generated from what was materialised, never written
  independently of it.** The two disagreeing is the root cause, and it is only
  preventable by having one source.
- **Selection stays seeded and reproducible.** Same seed, same clips. Record the
  seed, the selection rule, and the train/held-out split (currently 5 training
  videos, 2 held out — preserve that separation, it is what keeps `eval-general`
  honest).
- **Keep it minimal.** ~12 clips is the right size for triage. This set exists to
  answer "does the engine run", not to produce results.

## The verifier — this is the part that matters

A test that fails loudly on the exact faults above:

- every manifest clip resolves to a real directory of real frames;
- every named frame **exists**, checked against the files rather than assumed;
- frame counts match what the manifest claims;
- **the view contains the tracks the manifest names, and no others** — this is
  the check that would have caught the original bug;
- the held-out videos do not appear in the training view.

Run it against the *current* probe set first and watch it fail. A verifier that
passes on the broken input is not a verifier.

## Traps specific to this stream

**Do not regenerate from `assets/raw_4k` unless you must.** The 4K source is
15 GB and re-extraction is slow. Everything needed is already in
`assets/dataset`; this is a view-building job, not an extraction job.

**Symlinks are fine, silence is not.** The view is a symlink farm and that is a
reasonable design — it avoids duplicating a large dataset. The fault was never
the symlinks; it was that nothing checked they pointed at what the manifest
claimed. Keep the farm, add the check.

**A broken symlink and a wrong symlink are different failures.** The current set
has *zero* broken symlinks and is still completely wrong. Checking `-e` is not
enough; check identity.

## Done when

- The verifier fails on the old probe set and passes on the new one.
- All 12 clips resolve to real frames, in one documented coordinate system.
- The manifest is generated from the materialised view, not written beside it.
- The train/held-out split is preserved and asserted.
- `ruff`, `mypy`, tests pass.
