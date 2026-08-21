# B5 — Transport and temporal policy

**Owns exclusively:** `src/components/{transport,temporal}/**` and their tests.
**Implements:** `BaseTransport` and the temporal policy in `src/contracts/`.

## What to build

**Transport, split in two.** `DiskTransport` currently both *serializes* the
payload (msgpack metadata, JPEG sidecars, residual file) and *moves bytes*. Split
them, so a network transport would not have to reimplement serialization.

**Temporal policy**, at three composing levels:

- *Metadata sparsity* — send motion only at keyframes, interpolate between.
  **Exists**: the payload encoder emits a keyframe event only when pose delta
  exceeds a threshold, and reconstruction rebuilds a dense track.
- *Generation sparsity* — run the generator only at keyframes and interpolate its
  output client-side. **Exists**, with a preroll window kept residual-only.
- *Pipeline sparsity* — skip detection, segmentation and pose entirely on
  low-motion frames. **Missing, and this is where the real encode-time saving
  is**: today sparsity governs only what is transmitted and generated, while
  perception still runs on every frame.

Also missing: **motion-adaptive thresholds.** The delta threshold is a fixed
constant; it should derive from measured scene motion, so a slow rally and a fast
exchange do not get the same keyframe density.

## Traps specific to this stream

**The decision must travel in the payload.** The encoder honours it by skipping
stages and reconstruction honours it by interpolating — if each side recomputes
it independently, they drift. That drift is silent and shows up as a quality loss
nobody can attribute.

**Neither existing mechanism has ever been ablated.** Treat the existing code as
prior art to read, not as a working foundation.

**Never interpolate across a discontinuity.** A scene cut is not smooth motion.

## Done when

- Serialization and transport medium are separable.
- All three sparsity levels are selectable and compose.
- A reduced temporal setting is genuinely cheaper to encode, not just nominally.
- `ruff`, `mypy`, tests pass; import direction clean.
