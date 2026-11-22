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
For each player track:
    Load its cached appearance or decode the transmitted reference.
    Decode the track geometry, timing, crop boxes and random seed.
    If the packet carries conditioning rasters, decode those rasters.
    Otherwise rasterize the decoded keypoints at each native tick.
    Generate the crop sequence using this appearance and conditioning.
For each tick, read the composition order for the delivered mode.
    Start with the decoded base. Visit each layer in order.
        For a player, segment the generated crop using the paid pose.
            If support is nonempty, project and paste the masked crop.
        For a ball layer, apply the decoded ball update.
    Present the frame.
```

The initial serve reconstructs its exact conditioning rasters from the paid keypoints. The return-shot profile separately carries losslessly packed conditioning because rasterizing its decoded keypoints did not reproduce the generator inputs exactly. The fitted trajectory is drawn before the players in the measured joint serve and fitted return. The tracked-pixel return draws its tile after the players. Both orders, including occlusion, are part of their respective receiver definitions. [Indexed return packets and paid conditioning](../../../experiments/cumulative-joint-20261122/README.md)

Keep model installation, appearance transmission, refresh and reuse assumptions visible beside the achieved rate. The renderer does not receive source target RGB or human masks. The fallback above applies to missing support; a corrupt packet, missing scheduled crop or mismatched model is an error. The historical sampled-window comparison has a different adaptation and rate contract. [Conditioning and native generation](../../../../experiments/tennis/common_receiver_generate.py) · [Compositor and fallback](../../../../experiments/tennis/common_player_finish.py) · [Audited 240-frame replay](../../../experiments/receiver-replay-20261118/independent-idle-20261118/README.md)

## 3. Send a ball location with optional pixels

The sender has an ordinary base video and a sequence of optional locations. An application can read the transmitted locations in both modes. The optional tile replaces a small region in the display. The rendered-trajectory mode has its own appearance syntax.

```text
Encode the base at the selected rate and send its bitstream.
If the display uses unchanged video:
    Send optional ball centers. Present the ordinary decoded frames.
If the display uses a fitted trajectory:
    Send positions, missing symbols and fitted appearance parameters.
    At its layer, render at each valid position using the decoded motion.
If the display uses tracked pixels:
    Align each 32 × 32 tile origin with the chroma grid and image edges.
    Place the source region in a 64 × 64 tile, or a neutral tile if missing.
    Encode tiles when any placement exists. Send tiles and placements.
    At its layer, advance the tile stream for each tick.
        At valid placements paste the decoded 32 × 32 tile center.
Expose available positions or tile origins to the application.
```

In the measured video-plus-coordinates variant, the sender packs the ordinary AV1 stream with the sender’s existing position sequence. The application can read coordinates while the viewer sees the decoded AV1 frames unchanged. The tile mode uses its placement as location metadata. [Packet, all-frame equality and per-frame locations](../../../experiments/ball-basel-confirmation-20261122/metadata-control.json) Centering and edge clipping affect its relationship to the original ball coordinate. The 64 × 64 source-centered evaluation crop has a different role from the 32 × 32 pasted area. Read the exact sender positions, payload and decoded baseline to evaluate the fixed-budget gain. [Packet writer and receiver](../../../../src/pointstream/tracked_patch.py) · [Matched-rate evidence](../ball/README.md)
