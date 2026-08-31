# If you are resuming the wave-8 A–D session, read this first

Written 2026-08-31 by the BP30 session, which ran in parallel with yours in
`/home/itec/emanuele/pointstream-w8-e`. Your four streams all merged (PRs
#34–#38) and nothing here disputes them. This exists because **your worktrees
are pinned to commits from before that**, and one of my changes lands inside a
file stream D owns.

## Your worktrees are stale, and main has moved a long way

| worktree | branch | sits at | main is at |
|---|---|---|---|
| `pointstream-w8-a` | `wave8/plate-codec-sweep` | `5b4ae94` | `65c7540`+ |
| `pointstream-w8-b` | `wave8/intra-sidecar` | `d85839c` | |
| `pointstream-w8-c` | `wave8/low-rate` | `1f6d7a4` | |
| `pointstream-w8-d` | `wave8/panorama` | `1fb0b85` | |

Since those commits: PR #39 (BP30 stream component), #40 (probe-set repair),
#41 (stream wired into the runner). **Rebase onto `origin/main` before touching
anything**, or you will be editing against a tree three merges old.

## The one thing that can be silently undone

**Stream D's `make_background` in `src/runner/stages.py` changed, and the change
looks like a stylistic move.** It used to call `bind_background(ctx.config)`
*inside* the per-chunk body. It now binds once, outside:

```python
model = _bound_background(ctx)          # once per run

def background_stage(bag): ...          # reuses it across chunks
```

That is load-bearing, not tidying. `background.method: panorama-stream` carries
the previous scene's reconstruction so the next plate can be coded against it;
rebinding per chunk hands every scene a fresh empty stream and every scene pays
a full keyframe. **The amortisation would be configured, reported in the ledger,
and absent** — and nothing about the output would look wrong, because every
payload is still a valid plate and every reconstruction a real image. Only the
byte counts move, in the flattering direction.

If you resume from a stale w8-d and re-apply your own version of that function,
you will revert it. `tests/runner/test_background_stream_stage.py` will fail if
you do — one test greps the stage body specifically to catch this.

## What else is new that touches your area

- **`panorama-stream` is a fourth background method**, registered alongside
  `panorama-full`, `panorama-delta`, `none`. `BackgroundConfig` gained
  `reference_mode`, `keyframe_interval`, `stream_codec`, `stream_crf`.
- **`panorama-delta` was *not* unimplemented**, contrary to what
  `plans/BP30-background-stream.md` §4 and `PLAN.md` §2.21 said. It implements
  pixel subtraction. Both mechanisms are kept and named apart so the lattice can
  compare them; see `PLAN.md` §2.23.
- **`SizesBytes.panorama` is a *marginal* cost under `panorama-stream`.**
  Summing across chunks is still right, and right *because* chunk 0's keyframe
  is in the sum. Do not treat the mean per-chunk figure as the cost of a plate.
- **The probe set was dangling and is repaired** (PR #40). If you saw
  `test_probe_set.py` failing, that was not yours and is fixed. Do not
  `regenerate` it — that reselects clips and would move the probe set out from
  under every result measured on it; use `python -m experiments.probe_set
  repair-links`.
- **`mypy` now covers `experiments/`.** If your branch has type errors there,
  they were previously invisible and CI will now catch them.

## One finding that changes how to read your own results

The background lever's spread across five videos is **0.294 to 0.624** — larger
than every effect measured inside it. BP30 twice drew a conclusion from a single
video that inverted at five, including which reference mode to recommend. Any
BP29 result of yours measured on one clip carries the same exposure; the
harnesses take `--video`, and `experiments/tier/scene_plates.py` will enumerate
point-class scenes for any video in the dataset (`djokovic_federer` has 224).
