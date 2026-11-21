# Pipeline choice across bandwidth

The paper figure should answer which complete stream best serves the viewer or machine task at the available bitrate. The plotted objective and all initialization, updates, references, poses, coordinates, masks, repairs and fallback bytes must follow one shared contract. A reference lifetime or model delivery assumption changes that contract and therefore gets its own measured comparison.

## Available figures

| Artifact | Existing evidence | Use in the paper |
|---|---|---|
| [Rate and regional quality](rate-quality.pdf) | 17 same-rally points, 12 source-reviewed frames | Draft three-panel comparison for players and ball. Actual points are plotted. Lines join encodes within the same family. Generated alternatives have one setting each. The dotted staircase picks the best existing decode that fits each bandwidth ceiling. |
| [Measured bandwidth choices](bandwidth-choice.pdf) | Exact same 17 streams and three object metrics | Compact companion showing how the selected family changes with budget and task. Every transition is an actual stored bitrate. |
| [Earlier low-rate player comparison](historical-low-rate.pdf) | Four DCVC-UF encodes and the corrected W7 player point | Supports the motivation for a starved-rate player study at 25 fps. PointStream's rate assumes two-hour asset reuse. |

The first two use 240 decoded frames over 4.8 seconds, cold lossless player references, and complete delivery bytes on Nadal–Kyrgios. The quality sample is the fixed 12 reviewed ticks. The two body metrics use independently reviewed person masks. The ball metric uses a 64 × 64 source-centered crop. Lower DISTS is better. These plots expose the tradeoff between regional objectives. The bandwidth-choice bars evaluate the stored library under a bandwidth constraint. An ABR implementation also needs independently measured switch, state initialization, reference refresh and transition costs.

The last figure uses Alcaraz–Hurkacz 028, a 240-frame 25-fps window with 80 sampled scoring ticks, and a modeled two-hour reference-bank and adapter lifetime. Its PointStream diamond and the 50-fps cold-start points have different protocols. The later all-frame study will measure the useful operating range under shared accounting.

## Data to finish the submission figure

| Missing plot element | Reuse first | Targeted measurement |
|---|---|---|
| Common foreground and whole-frame objective | All same-source decoded AV1, ROI and native-plus-ball streams, reviewed samples and original scoring code | Score the registered foreground/complement objective, whole-frame quality and temporal diagnostics on identical supports and ticks. Do this once per unique decoded stream. Freeze that same objective before testing selection. |
| Contemporary DCVC-UF curve | The historical same-rally job and saved rate probe in the linked receipt | Recover exact real decodes if they contain the required frames and valid byte bills, then score the new supports. Otherwise encode only the settings needed to bracket the current candidate rates. The old 80-tick regional scores cannot supply the present 12-tick body/ball values. |
| Starved-rate PointStream/AV1/ROI/DCVC comparison | Authenticated Alcaraz–Hurkacz 028 and archived model, transport and judge inputs | Run the already prepared [20/50/125 kbps study](../../../experiments/starved-joint-20261122/README.md). Begin with the byte floor and a smoke, then add only settings that establish overlapping actual rates. Keep references, background and ball policy explicit. |
| PointStream curve with generated players | Current fixed generated crops and measured carrier components | At three overlapping rates, construct and decode the complete chosen background/player/ball representation. A single measured setting remains a point. Changing reference quality requires the actual render for that ladder rung. |
| Receiver reuse and transitions | Existing cross-shot court-reference test, source-denied player replay and fallback codec | Count one initialized continuous passage including returns, refreshes and mode changes, then report the actual reused lifetime. Show cold and measured reuse curves alongside their denominators. |
| Evidence of a useful mode selector | Development library and explicit per-task objective | Freeze the mode and reference policy, evaluate new physical matches, and compare the chosen stream with each fixed codec at the same total rate. Measure additional state/switch bytes and temporal discontinuities before reporting ABR gains. |

These six rows are the data dependencies of one selection figure. A new experiment must name the row and the missing point or interval it supplies. More bitrate settings are justified only by absent support or a decision boundary. The selection figure becomes a contribution when the measured gain persists under a frozen policy on new matches.

## Reproduction and checks

Run `MPLBACKEND=Agg python docs/paper/figures/selection/build.py` from any working directory with Matplotlib and NumPy. Supply `--output DIRECTORY` to write a separate rendering. The builder checks equality of the source, label, region geometry, quality implementation and metric weights for the matched-rate addition. It checks the frozen numeric input hashes in [sources.json](sources.json) and retains their inventory in [data.json](data.json). The 17 rows consist of five plain AV1, five ROI AV1, four AV1-plus-ball and three generated-player compositions. The categorical choice is computed from finite measured points without quality interpolation. The plot renderer uses no GPU or source footage.

The historical same-rally DCVC record is [prior-dcvc.json](../../../experiments/receiver-replay-20261118/selected-rate-support/prior-dcvc.json), job `20261031T181055Z-28301343`, archive SHA-256 `2dd9fd2e821b9cbfce6de838166ac5f77d0ff0dafc156b75fa9d2a4f7a0d888f`. Its recorded purpose is comparability, with a different regional definition and 80 scoring ticks. The current plotted points and all measured selection breakpoints are available in [data.json](data.json).
