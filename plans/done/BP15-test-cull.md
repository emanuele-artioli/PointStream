# B′15 — Retire the pre-rewrite tree and its 433 tests

**Owns:** `src/encoder/**`, `src/decoder/**`, `src/shared/**`, and the 69
top-level `tests/test_*.py` files.

## The numbers

| Origin | Files | Tests |
|---|---|---|
| **pre-rewrite** (`tests/*.py`) | 69 | **433** |
| contracts | — | 195 |
| components | — | 295 |
| pipeline | — | 110 |
| runner | — | 11 |
| invariants | — | 12 |
| | | **~1056** |

`plans/done/RESEARCH-HISTORY.md` §8 already anticipated this: *"The ~436 pre-rewrite tests are untouched
and test modules Phase B and C delete. They die with their modules; no separate
culling is needed."* 433 is that number. **The plan was right; the deletion just
has not happened**, because Phase C landed without removing what it replaced.

**623 rewrite tests across 16 component axes, the contracts, the pipeline and the
runner is not bloat** — that is roughly 40 per axis with misuse cases. The 433
are the removable half, and they are dead weight now: they slow every run, and
two of them are already `xfail` for pollution nobody will chase (`DEFERRED.md`
D6).

## The boundary is nearly clean

Only **three** modules in the pre-rewrite tree are still imported by new code:

| Module | Lines | Used by |
|---|---|---|
| `src/shared/torch_dtype.py` | 137 | `components/generation/controlnet.py` |
| `src/shared/spade4tennis_arch.py` | 138 | `components/generation/spade.py` |
| `src/decoder/animate_anyone_runtime.py` | 465 | `components/generation/animate_anyone.py` |

Everything else — 24 files and 6378 lines of `src/encoder`, most of
`src/decoder` (4175), most of `src/shared` (4782) — has no inbound edge from the
new tree.

## What to do

1. **Move the three modules** into the new tree, under `src/components/generation/`
   or a small `src/shared/` successor that respects the layer check. Keep their
   tests, ported.
2. **Confirm nothing else imports the old tree** — `src/main.py` does, so decide
   whether it is replaced by the runner CLI or retired with the rest.
3. **Delete the rest, with their tests**, in one commit per subtree so a mistake
   is easy to read and revert.
4. **Re-run the required-behaviour suite** and the layer check after each.

## Traps

**Read before deleting.** These modules are prior art we have already mined
twice — the two-naming-convention discovery came out of `tennis_dataset.py`, and
that file's *correct* positional pairing is the pattern the probe set now
follows. Grep for anything the new tree should inherit before removing a file,
and say what you took.

**`tennis_dataset.py` is still live for training** (`scripts/train_controlnet.py`
imports it) and must survive this cull, or move with the training code.

**Do not delete a test to make the suite green.** The two `xfail`ed AA tests
(D6) go when their module goes, not before, and `D5`'s architectural test is not
part of this cull at all.

**A smaller suite is not the goal; an honest one is.** If a pre-rewrite test
covers behaviour the new tree also has and does not test, port the test rather
than dropping it — and say which ones you ported.

## Done when

- The three shared modules live in the new tree with their tests.
- `src/encoder`, and the retired parts of `src/decoder` and `src/shared`, are
  gone with their tests.
- `python -m src.contracts.layers` is clean and the suite is green without the
  D6 `xfail`s.
- The report says what was ported rather than deleted, and why.

## Delivered

Wave 3 Stream B on `wave3/bp15-cull`. Encoder / `src.main` / the three generation
moves were already on main (`14d7ef2`, `6390e3e`). This stream did the remainder:
`src/decoder/**` and `src/shared/**`. PR:
https://github.com/emanuele-artioli/PointStream/pull/19

**Ported:** nothing. `eval_checkpoint.py` still has to call
`src.decoder.genai_compositor.build_genai_strategy` — that is the Residual
Guarantee for eval (same classes the old decoder ran). The new
`BaseFrameGenerator.generate(ConditioningBundle, *, seed, device, params)` is a
different API. Porting it would be a rewrite, not a mechanical move, so
compositing stays.

**Deleted (decoder, `3eece7d`):**
- `src/decoder/decoder_renderer.py` — no rewrite inbound, no training/eval
  script. Only `tests/test_decoder_genai_debug_parity.py` called it.
- `src/decoder/compositor.py` (`ResidualCompositor`) — only `DecoderRenderer`
  used it. The new runner has `src.pipeline.residual.signal`.
- `tests/test_decoder_genai_debug_parity.py` — goes with `DecoderRenderer`
  (that was the failure mode).

**Tagged and left (decoder):** compositing package, `genai_compositor` shim,
`controlnet_engine` / `pix2pix_engine` / `spade4tennis_engine`,
`attention_injection`. Callers: `scripts/eval_checkpoint.py` and
`tests/components/test_spade4tennis.py`.

**Deleted (shared, `dd8c2ae`):**
- `src/shared/synthesis_engine.py` + `tests/test_synthesis_engine_coverage.py`
  + the two `SynthesisEngine` tests in `tests/test_panorama_encoder.py`.
  Panorama encoder tests stay; they test `src.transport.panorama_encoder`.
- `src/shared/mask_codec.py` + `tests/test_mask_codec.py` +
  `tests/test_mask_codec_coverage.py`. Only `DecoderRenderer` imported the
  module.
- `src/shared/profiling.py` + `tests/test_profiling.py`. Same.
- `src/shared/track_id.py`. Only `DecoderRenderer`. The `track_id` asserts in
  `tests/test_coverage_utilities.py` went with it; the dtype assert stays.

**Tagged and left (shared):** `schemas`, `interfaces`, `tags` (`src.transport.disk`);
`dwpose_draw` (`animate_anyone_runtime` and compositing); `video_io`,
`experiment_evaluation`, `fvd`, `lpips_metric`, `config` (eval / hnerv /
codec-baseline scripts); `tennis_dataset` (pix2pix/spade training,
`eval_checkpoint.pad_to_square`, `debug_dataloader` — the two-naming-convention
docstring is still on the class); `geometry`, `player_extraction`,
`racket_heuristic`, `scene_classification` (`process_dataset`); `hnerv_arch`;
`genai_debug` (compositing); `invariants` (`tests/invariants/test_outputs_tree.py`).
`train_controlnet.py` was not edited. It does not import `tennis_dataset`; it
copies the positional-pairing comment and has its own dataset class.

**`benchmark_mask_codecs` (`aecd107`):** deleted the script and
`tests/test_benchmark_mask_codecs.py`. It subprocessed `-m src.main`, which is
gone. Rewriting it against the new runner is another stream's pipeline work.

**Coverage gate:** still 77. Deleting those tests did not drop CI below the
floor, so the gate was not lowered.

**CI watched green:**
- decoder `3eece7d`: [32746645150](https://github.com/emanuele-artioli/PointStream/actions/runs/32746645150)
- shared `dd8c2ae`: [32747406198](https://github.com/emanuele-artioli/PointStream/actions/runs/32747406198)
- benchmark `aecd107`: [32747869455](https://github.com/emanuele-artioli/PointStream/actions/runs/32747869455)

**Outside this stream's files:**
- Unexpected live rewrite import: `src.components.generation.animate_anyone_runtime`
  still imports `src.shared.dwpose_draw`. There is no skeleton-drawing
  replacement under `src.components.generation` (`pose.py` is letterbox only),
  so `dwpose_draw` is tagged, not moved.
- `tests/components/test_spade4tennis.py` still constructs
  `src.decoder.spade4tennis_engine.Spade4TennisStrategy`.
- `src.components.scene.hsv` mentions `src.shared.scene_classification` in a
  docstring only; it does not import it.
