# B'24 — Put a real encoder in the codec stage

**This is the single highest-value open item in the project.** `PLAN.md` §7 P0
items **2, 3 and 4** are all stacked behind it: the codec ladder with region
arms, the residual-coarseness curve, and the rate half of the ablation lattice.
Half the remaining P0 list unblocks the moment this lands.

**Owns:** `src/runner/stages.py` (codec stage), `src/pipeline/encoder/**`,
`src/contracts/lattice.py`, `config/tier_*.yaml` codec keys, `experiments/tier/**`.
**Read first:** `AGENTS.md`, `PLAN.md` §2.16 and §3, `plans/done/BP23-first-tier-run.md`,
`plans/done/C3-runner.md`, `src/runner/stages.py:192` (`make_codec`).

## The problem, precisely

`src/runner/stages.py` line 7 says it plainly: *"Codec / transport / metrics are
identity-roundtrip."* `make_codec` passes pixels through. Consequences measured
in BP23:

- Every byte count in `outputs/bp23-tier/report.json` is **pixel payload, not
  coded bytes**. The raw panorama plate alone is 24.9 MB — 95% of `tier_fast`'s
  entire figure.
- `transport_to_source_ratio` is **not a compression ratio** and must not be
  reported as one.
- **BD-rate cannot be computed at all**, because there is no rate axis.

`src/components/codec/**` already wraps real encoders and BP21 drove four of them
(AVC/HEVC/AV1/VVC) to real bitstreams on 4K. **The encoders work.** What is
missing is the binding between the runner's codec stage and those components.

## What to do

1. **Decide the boundary and write it into `PLAN.md` §3 before coding.** The
   question is what the codec stage receives and returns: coded bytes plus a
   decoded reconstruction, or a bitstream handle the transport stage owns. C3
   deliberately left this as identity; do not silently reverse that — state the
   contract.
2. **Bind the codec components** into `make_codec` so `codec.name` and the rung
   parameters in `config/tier_*.yaml` reach a real encoder. Resolve encoder
   binaries **by path and version**, not by name — this host has carried two
   builds of the same encoder with different capabilities.
3. **Make the size ledger count coded bytes**, and keep the pixel-payload number
   as a separate, clearly-named field. Do not overwrite the BP23 numbers; add
   alongside, so the change in meaning is visible.
4. **Fix `STAGE_CODEC.optional_inputs`** in `src/contracts/lattice.py` to declare
   `generated-frames`. BP23 reported this rather than patching it: without it the
   DAG may order the codec before the generator, and a generation-on /
   residual-off corner cannot deliver a reconstruction.
5. **Re-run the BP23 tier ladder** and report the same table with real coded
   bytes beside the old pixel-payload figures.

## Bounds — write to `outputs/bp24-encoder/bounds-before-run.json` first

- **A real codec rung must produce fewer coded bytes than the source.** BP23's
  `tier_quality` "residual" figure of 37,919,751 bytes against a 199,065,600-byte
  source is a 5.2x pixel-payload ratio; a real encoder at the same quality should
  be **far** below that. A coded size within a few percent of the pixel payload
  means the encoder is not actually running — check before believing.
- **The tier ordering must survive.** `fast` < `balanced` < `quality` in both
  bytes and quality. An inversion is an alarm.
- **Quality must not move much** when only the accounting changes. If binding a
  real encoder shifts PSNR by more than ~1 dB at matched settings, the codec is
  being applied somewhere it was not before — investigate.

## Traps

- **A flag existing is not a feature working.** Drive each codec name and prove
  the output changed in the way the option claims. BP23 found 27 of 32 config
  fields reaching nothing precisely because nobody had checked.
- **Do not import from `src/shared` or `src/decoder`.** `src/pipeline` and
  `src/runner` currently import nothing from either, which is what keeps `BP22`
  able to delete them in parallel. Report, do not import.
- **Do not report a compression ratio until the ledger counts coded bytes.**

## Done when

- The codec stage runs a real encoder, and the boundary contract is written in
  `PLAN.md` §3.
- The BP23 ladder is re-run with coded bytes, alongside the pixel-payload figures.
- `STAGE_CODEC.optional_inputs` declares `generated-frames`.
- A required-behaviour test asserts a codec rung produces coded bytes smaller
  than the source and that the tier ordering holds.
- The report says explicitly whether P0 items 2 and 3 are now unblocked.
