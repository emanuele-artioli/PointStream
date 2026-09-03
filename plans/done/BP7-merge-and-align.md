# B′7 — Merge Wave 1, and fix the pose alignment

**Wave 1.5 — sequential, one agent, do this before Wave 2 launches.** It is the
integration step, not a parallel stream: it touches every Wave-1 branch and the
shared registry, so two agents doing it at once would collide.

**Owns:** the merge, `src/components/generation/__init__.py`,
`experiments/probe_set/materialize.py`, `experiments/probe_set/verify.py`.
**Read first:** `plans/done/RESEARCH-HISTORY.md` §2.1 and §2.3.

## 1. Fix the pose alignment — do this before the merge, not after

`plans/done/RESEARCH-HISTORY.md` §2.3 has the evidence. In `assets/dataset`, a track's directories use
**two different frame-naming conventions**:

| Directory | Naming |
|---|---|
| crop, `_canny`, `_pose_body`, `_pose_racket` | **global** source frame ids |
| `_skeleton` | **track-local**, zero-based |

50 of 114 tracks (44%) carry the offset; the rest align only because they start
at source frame 0.

`materialize.py` resolves conditioning frames by global filename under
`if src.is_file()` and skips silently, so **5 of 12 v2 clips have 48 colour
frames and 0 skeleton frames**, and the verifier passed because it checks colour
frames only.

**The fix already exists in this repo — copy its approach.**
`src/shared/tennis_dataset.py:95-110` pairs crop and skeleton **sequentially by
sorted order**, with the comment *"Pair them sequentially by order"*. That is why
every trained checkpoint here is sound. Do the same in the materializer: resolve
each channel by the frame's **position in the track**, never by reconstructing a
filename.

Then extend the verifier: **every conditioning directory must have the same frame
count as the crop.** That single assertion would have caught this.

**Do not rename anything in `assets/dataset`.** See §3 below.

## 2. Re-run BP3's five numbers on aligned pairs

`plans/done/RESEARCH-HISTORY.md` §2.1 records pose-ControlNet at 20.3 dB, "smeared but recognisable".
That smearing is what misaligned conditioning looks like, so those numbers are
suspect. Re-run all five comparison-backbone engines on the fixed probe set,
same seed (42), same checkpoint epochs, and replace the table.

If the numbers move materially, say so plainly — it is a useful finding about
the measurement, not an embarrassment.

## 3. Merge the four code branches

Merge `phase-bp/bp1`, `bp2`, `bp3`, `bp4` into `phase-b/integrate` (or a fresh
`phase-bp/integrate`). Then **apply the registry entries in one edit** —
`trajectory-controlnet`, `stable-animator`, and the Animate-Anyone summary
update. BP3 and BP4 deliberately left
`src/components/generation/__init__.py` alone and recorded the entries in
`plans/done/README.md`; applying them earlier would have registered modules that were
not yet on the tree.

The paper branch `phase-bp/bp6` merges separately, in its own repo.

## Traps specific to this stream

**Do not "fix" `assets/dataset` by renaming skeleton files.** It is derived data,
15 GB of source behind it, and the layout is what every trained checkpoint and
`pointstream_aa_meta.json` were built against. Read-time normalisation is
non-destructive and is robust to any further convention drift in a dataset that
is model-generated and not fully trusted. Renaming is a one-way door for no gain.

**`tennis_dataset.py`'s docstring is wrong even though its code is right.** Lines
26-27 claim crop and skeleton share `frame_ZZZ.png` naming. They do not; the code
survives because it pairs by position. **Fix that docstring** — the next person
who "corrects" the code to match the comment breaks every dataset loader here.

**Check whether anything else resolves conditioning by filename.** Consumers of
`_skeleton` include `src/encoder/actors/builder.py`,
`src/shared/synthesis_engine.py`, `scripts/train_controlnet.py`,
`scripts/eval_checkpoint.py`. Report what you find; do not fix the pre-rewrite
ones, which Phase C deletes.

**`scripts/select_probe_set.py` still writes v1** and must not be used as the
regenerator. Either delete it or make it call the v2 path.

## Done when

- Every channel resolves by position in track; no filename reconstruction.
- The verifier asserts conditioning frame counts match the crop, and fails on the
  current v2 tree before it passes on the fixed one.
- All 12 clips have 48 colour frames **and** 48 skeleton frames.
- BP3's five numbers are re-run on aligned pairs and `plans/done/RESEARCH-HISTORY.md` §2.1 updated.
- The four branches are merged and the registry entries applied in one edit.
- `ruff`, `mypy`, tests, layer check pass.
