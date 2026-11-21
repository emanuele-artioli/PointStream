# Can filling the missing player edges rescue the cleaned rally?

On the 12 already source-reviewed development ticks, compare the actual 690.14 kbps GrabCut joint replay and the actual 336.18 kbps AV1-plus-ball replay with the original broadcast. This is a component diagnostic at two known delivery costs. Its purpose is to discriminate between an edge-only repair and a repair that changes the generated player interiors as well.

Before scoring, freeze the same reviewed body masks, receiver geometry and the exact three RGB images per tick. For each player, grant the receiver free, perfect source pixels everywhere that player's reviewed body extends beyond its own generated mask. Leave every other pixel alone. Pixel squared error is nonnegative and additive, so the resulting body RGB MSE is a lower bound for *any* repair restricted to those missing pixels, evaluated on this fixed composition. Use the arithmetic mean of the 12 per-frame body MSEs for each player, and record the pooled pixel calculation separately. A local pass only justifies testing the paid correction. If even perfect repair leaves a player's error above native AV1 plus ball, include player interiors in the next diagnostic before implementing a boundary channel. DISTS, missing equipment and unobserved background are separate endpoints. The source-only oracle can never be transmitted at the displayed rate.

Use the stored receiver projection code, actual mask archives, full-precision decoded PNGs, and fixed source and packet identities. Smoke on the first actual frame in the same pass, check its omitted counts against the earlier independent mask audit, and stop at 90 local CPU-seconds or two minutes elapsed. No campus compute. Only JSON receipts enter git.

## Observed body fidelity

| Full-resolution RGB MSE, mean over 12 frames | Player 1 | Player 2 |
|---|---:|---:|
| Cleaned base with current generated players | 2,114.0 | 2,097.7 |
| Same reconstruction, with perfect source pixels in all missing body-mask areas | 1,620.1 | 1,603.9 |
| Lower-rate native video plus tracked ball | 187.3 | 240.4 |

The actual receiver projection matched the independent missing-pixel inventory at all 24 player-frame observations. Exact replacement recovers 23–24% of the current body MSE. Even with perfect edge repairs, the unchanged player interiors leave more body pixel error than the lower-rate fallback. A correction confined to these pixels cannot close this MSE gap. Include the interior rendition in the next low-rate comparison. This test does not quantify a viewer's preference, neural feature distances or the effect of changing the generator, source material, bitrate or reuse horizon. Rackets have no independent mask in this rally.

Run `PYTHONPATH=. python docs/experiments/receiver-body-repair-20261122/check.py --output NEW_JSON_PATH` with the archived inputs specified by its arguments. The [measurement receipt](result.json) records both averaging conventions, input and script hashes, CPU time, and integer squared errors for each source frame. The [paid packet totals](../clean-court-grabcut-20261119/delivery-receipt.json) and [perceptual score comparison](../clean-court-regional-screen-20261118/README.md) describe the two real reconstructions.
