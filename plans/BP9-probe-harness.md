# B′9 — Fix what the probe measures, permanently

**Parallel with `BP8`.** Small, and every future generation number depends on it.

**Owns exclusively:** `experiments/probe/**`, `tests/components/test_probe*.py`.
**Read first:** `PLAN.md` §2.6.

## The defect

`experiments/probe/run.py` passes `frame.appearance_rgb` as the **conditioning**
and the same array as the **scoring reference**. Every engine was asked to
reproduce the image it was handed. Two things follow:

- **It is not the coding task.** PointStream establishes appearance once and
  drives later frames from motion. The question is: given appearance from a
  keyframe and pose from frame *N*, reconstruct frame *N*.
- **It contradicts the harness's own check.** `differs_from_input` fails an
  output that did not change, while the score punishes exactly the changing. A
  diffusion model cannot satisfy both.

## What to build

**Score against a later frame than the appearance came from.** Appearance from a
keyframe (frame 0 by default), conditioning from frame *N*, reference frame *N*.
Make the keyframe/target offset an explicit, recorded parameter — it is a real
axis (how far can one appearance carry?) and will become an experiment later.

**Keep the static-copy baseline as a permanent arm.** Paste the keyframe forward,
no model, score it exactly like an engine. It costs nothing, it is the floor any
generator must beat, and it is what exposed `PLAN.md` §2.6. Measured today:
**11.82 dB object-scoped, 8.90 dB frame** at offset 24 on the 12 probe clips.

**Make it a gate, not a decoration.** An engine scoring at or below the
static-copy floor is reported as **not using appearance**, in those words, rather
than as a low-ranking engine. That is the sentence that would have saved this
round.

## Traps specific to this stream

**Do not delete the self-reconstruction number — relabel it.** It is a useful
diagnostic (how much does this engine alter an image it was given?), it is simply
not a coding result. Report it as `self_reconstruction_psnr` and never rank on it.

**Keep the bounds discipline.** `experiments/probe/bounds.py` writes plausible
ranges before generating, which is right and should stay. **Rewrite the bounds
for the new task** — the old ones were calibrated on self-reconstruction and will
fire wrongly. Anchor them on the static-copy floor instead of on absolute dB.

**One frame per clip is thin.** The current harness scores `frame_index=24` only.
Take several offsets per clip so a single unlucky frame cannot decide an engine.

**PSNR stays the triage metric** (`PLAN.md` §6.5). Do not add VMAF or LPIPS here.

## Done when

- Appearance comes from a keyframe and the reference is a later frame, with the
  offset recorded.
- The static-copy baseline runs as a permanent arm on every probe.
- An engine at or below that floor is reported as not using appearance.
- Bounds are re-derived against the floor, written before generating.
- More than one frame per clip is scored.
