# Two procedures used in the tennis implementation

## 1. Find camera views eligible for object coding

Input is a shot-change score and a court-registration result per native frame. Output is a sequence of labeled, half-open frame intervals. Tennis uses court geometry as its eligibility cue. A different domain can supply a different cue.

```text
Cut the sequence wherever the scene-change score reaches 0.25.
For each resulting shot:
    Mark frames whose court model fits within 6 px of the observed lines.
    Group adjacent equal fit states into runs.
    While several runs remain and a run is shorter than one second:
        Merge the shortest short run into its longer neighbor.
        Coalesce adjacent runs that now share a state.
    For each remaining interval:
        If at least 80% of its original frames fit, mark it court-eligible.
        Otherwise mark it for conventional coding.
```

For equal lengths the implementation picks the earlier run or neighbor. Durations round to native frame ticks. A court-eligible shot can still be a replay or between points. On the recorded court / close-up / court passage, the separate historical active-play selector classified the three spans as other / interlude / point. Report active-play labels and actual routing when making a switching claim. [Shot splitting and interval scoring](../../../../experiments/tennis/k8b.py) · [Court registration](../../../../experiments/tennis/court.py) · [Continuous passage](../../../experiments/continuous-return-20261118/README.md)

## 2. Deliver tracked ball pixels at native cadence

Input is the native video and an optional sender track location at each tick. Output is a complete native base stream and, when used, an aligned tile stream with in-frame placements. This is the tracked-pixel mode. The trajectory-rendering experiment uses a different representation.

```text
Encode the full base video at its allocated setting.
If any frame has a tracked location:
    For each tick:
        Start a neutral 64 × 64 tile. If a location exists, align a 32 × 32
        window to the chroma grid, clip its origin and copy it into the center.
    Encode the tile sequence, including the neutral ticks, at native cadence.
Transmit the base, tile payload if present, placements and packet headers.
For each receiver tick, decode the base and advance any tile stream.
    If there is a placement, paste its decoded 32 × 32 center on the base.
    Emit the resulting frame. With a missing placement, emit the base frame.
```

The receiver can use a placement directly as tile-location metadata. Centering and edge clipping affect its relationship to the original ball coordinate. The 64 × 64 source-centered evaluation crop has a different role from the 32 × 32 pasted area. Read the exact sender positions, payload and decoded baseline to evaluate the fixed-budget gain. [Packet writer and receiver](../../../../src/pointstream/tracked_patch.py) · [Matched-rate evidence](../ball/README.md)
