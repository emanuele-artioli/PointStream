# B′13 — The motivating example, measured

**Owns:** `experiments/headroom/**`, and the paper's
`sections/problem.tex` motivating example.

## Why this exists

The motivating example used to carry **numbers**: the bitrate of coding a person
as appearance + keypoints and the background as panorama + homographies, against
a traditional codec, under the assumption that a generator could restore the
video. It proved the headroom for both foreground and background, and it
motivated the whole design.

The rewrite turned it into prose. **It now motivates nothing**, and it drags in
residuals and other mechanisms that belong in the Method, not in a motivating
example. Restore the numbers, keep the mechanism out.

## The two measurements

### FG — what do the players cost the codec?

Encode a clip normally; encode it again with the players **removed**, and
difference the bitrate **at matched quality**.

**Removal must be thorough.** Blurring leaves structure, and the codec still
spends bits on it — that under-measures the saving. In increasing fidelity:

1. **Background-plate inpaint** — replace the player region with the background
   model's plate for that frame. Best: it is what PointStream would actually
   transmit.
2. Temporal median of the same region across the scene, where the plate is
   unavailable.
3. Flat fill as a *lower* bound only, and say so — a flat region is cheaper than
   any real alternative, so it over-states the saving.

Report at least (1) and (3): they bracket the truth.

**Expect the players to be expensive relative to their area.** They are ~2% of
the pixels but essentially the only complex motion in the frame — the rest is
slow camera pan and a static court, which inter-prediction handles very well.
A saving far larger than 2% is the expected result, not an alarm; a saving
*near* 2% would be the alarm, and would mean the codec is already spending its
bits elsewhere.

### BG — what does the panorama buy?

The same question for the other half of the claim: code the background
conventionally, versus transmitting a panorama plate once plus per-frame
homographies. Difference the cost over a scene.

This is the half the current text drops entirely, and it is the half that covers
~98% of the pixels.

## Bounds, written before running

- **FG ≥ 25% of the bitrate** — the premise is strong; proceed.
- **10–25%** — real but modest; the background and residual carry the paper.
- **< 10%** — the premise is weak and we must know now, in time to redirect.
- **BG**: a panorama plate plus homographies should cost **orders of magnitude**
  less than coding the background per frame over a scene. If it does not, the
  panorama component does not pay for itself and the lattice will say so.

## Traps

**Matched quality, not matched file size** (`PLAN.md` §5). Removing the players
changes quality as well as rate; comparing raw sizes measures nothing.

**Report the region quality separately.** Per `PLAN.md` §6.4, and because the
players are what a viewer watches: a frame-level number will barely move when the
player region is destroyed, which is exactly why it cannot carry this argument
alone.

**This is a headroom argument, not a result.** It says what is available *if* a
generator works. It must not be written as though PointStream achieved it. The
Evaluation section reports what we achieved; this section says what is on the
table.

**Keep residuals out of the motivating example.** They are a Method mechanism.
The motivating example answers one question: how much is there to win?

## Done when

- FG and BG headroom numbers exist for at least one scene per domain, with the
  removal method named and the bracketing bounds reported.
- `sections/problem.tex` carries them, with `CLAIM` lines naming `outputs/` paths.
- The section motivates the design again, and mentions no mechanism.
