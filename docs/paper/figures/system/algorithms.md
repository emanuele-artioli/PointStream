# Three procedures used in the tennis implementation

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

## 2. Reconstruct players from appearance and keypoints

The evaluated common-player mode covers one continuous 240-frame track per player. On the server, the player packet carries the base video, appearance reference per run, compressed keypoints and the information needed to recover the crop placement and timing. A pinned common renderer is installed at the receiver. This procedure uses the conditioning parsed from the delivered packet.

```text
For each player track, code its appearance reference and keypoint sequence.
Pack these with the base, track timing, geometry and declared player order.
For each admitted track, decode its reference, keypoints, boxes and seed.
    For each native tick, rasterize its decoded pose in the player crop.
    Render the sequence conditioned on that reference and the pose sequence.
For each tick, initialize the display canvas from the decoded base.
    Visit visible players in the packet's far-to-near order.
    Prompt the segmenter on each generated crop using paid pose geometry.
    If support or the predicted mask is empty, retain the underlying pixels.
    Otherwise project the mask and composite in-frame pixels at the crop box.
Emit the composed frame; an optional paid ball update follows the players.
```

Keep model installation, appearance transmission, refresh and reuse assumptions visible beside the achieved rate. The renderer does not receive source target RGB or human masks. The fallback above applies to missing support; a corrupt packet, missing scheduled crop or mismatched model is an error. The historical sampled-window comparison has a different adaptation and rate contract. [Conditioning and native generation](../../../../experiments/tennis/common_receiver_generate.py) · [Compositor and fallback](../../../../experiments/tennis/common_player_finish.py) · [Audited 240-frame replay](../../../experiments/receiver-replay-20261118/independent-idle-20261118/README.md)

## 3. Send a ball location with optional pixels

The sender has an ordinary base video and a sequence of optional locations. An application can read the transmitted locations in both modes. The optional tile replaces a small region in the display. The rendered-trajectory mode has its own appearance syntax.

```text
Encode the base video at the chosen setting.
If the mode is video plus coordinates:
    Pack the base with all coordinates and explicit missing symbols.
    At the receiver, return that sequence to the application.
    Emit the unmodified decoded base for playback and finish.
For each tick in tracked-pixel mode:
    Align the tracked 32 × 32 region to the chroma grid and image edges.
    Copy it into a 64 × 64 tile, or use a neutral tile if missing.
Send the base, coded tiles when any location exists, and placements.
For each received base frame:
    Read its placement and advance the tile stream if present.
    If a placement exists, paste the decoded 32 × 32 center at that origin.
    Return the placement to the application and emit the resulting frame.
```

In the measured video-plus-coordinates variant, the sender packs the ordinary AV1 stream with the sender’s existing position sequence. The application can read coordinates while the viewer sees the decoded AV1 frames unchanged. The tile mode uses its placement as location metadata. [Packet, all-frame equality and per-frame locations](../../../experiments/ball-basel-confirmation-20261122/metadata-control.json) Centering and edge clipping affect its relationship to the original ball coordinate. The 64 × 64 source-centered evaluation crop has a different role from the 32 × 32 pasted area. Read the exact sender positions, payload and decoded baseline to evaluate the fixed-budget gain. [Packet writer and receiver](../../../../src/pointstream/tracked_patch.py) · [Matched-rate evidence](../ball/README.md)
