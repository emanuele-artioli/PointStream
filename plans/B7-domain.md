# B7 — Domain profiles

**Owns exclusively:** `src/components/domain/**` and its tests.
**Implements:** `src/contracts/domain.py` — `DomainProfile`, `SalientClass` and
`CameraMotion` are already defined there.

## What to build

Two profiles, and the dataset plumbing each needs.

**tennis** — players, racket, ball; whole-body human schema; a broadcast camera
that pans, tilts and zooms, so a homography holds and a panorama is valid.
Extract this from what is currently hardcoded.

**general** — whichever humans are present; evaluated on **the DAVIS clips
containing humans**; a free-moving handheld camera, so panorama backgrounds are
invalid and the residual carries the background.

Football is **not** in scope. It is a later decision.

## Why the general profile earns its place

It is the direct answer to the most-requested reviewer item — generalizability,
raised by four of five referees. It also opens a second experiment at almost no
extra cost: pretrained human-generation models will likely do reasonably on
DAVIS and struggle on tennis, so measuring pretrained-versus-fine-tuned across
both domains shows exactly what domain-specific fine-tuning buys.

## Traps specific to this stream

**Camera motion is not about hardware.** It is about whether the background can
be modelled as one warpable plane. Declaring it wrong does not produce a slightly
worse panorama — it produces a quietly incoherent one.

**The domain says *what*, components say *how*.** In any human domain we care
about people; whether YOLO26, SAM3 or RF-DETR extracts them is an independent
axis. Do not let a profile name a backend.

**Salient-object selection is code, not data.** Deciding that *these* people are
players and *those* are spectators cannot be expressed as profile parameters. The
profile names which selector to use; it does not encode the rule.

## Dataset sequencing

Build the **minimal** dataset first — a handful of tennis tracks plus a few DAVIS
human clips, enough to exercise the pipeline end to end. Full dataset repair
comes later and is not this stream's job.

## Done when

- Both profiles resolve by name and round-trip.
- A panorama requested under the general profile is rejected with a usable
  message.
- Minimal datasets for both profiles exist and a run completes on each.
- `ruff`, `mypy`, tests pass; import direction clean.
