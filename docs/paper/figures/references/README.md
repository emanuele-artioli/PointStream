# A reference ladder shared across players

[Figure PDF](reference-ladder.pdf) · [Preview](reference-ladder.png) · [Experiment and receipt](../../../experiments/reference-ladder-20261122/README.md)

**Caption.** Each codec uses one setting for both players at each of three total two-image budgets. Points show actual bytes, before packet framing, against RGB fidelity for each source reference. Higher PSNR means a closer still image. The dotted verticals are the 25, 45 and 65 kB budgets. Lines guide the eye between measured configurations.

| Budget for both images | JPEG setting / bytes | WebP setting / bytes | AVIF setting / bytes |
|---|---:|---:|---:|
| 25 kB | Q16 / 24,967 | Q62 / 24,680 | Q46 / 24,463 |
| 45 kB | Q45 / 44,437 | Q84 / 44,232 | Q65 / 44,179 |
| 65 kB | Q70 / 64,498 | Q90 / 62,338 | Q78 / 63,915 |

AVIF achieved the highest still-image PSNR for each player at every tested budget. The Q46 payloads are also the same size as the older Q45 encodes. These two images nominate shared rungs for a motion test. At run time the reference budget comes from the session rate, valid reference lifetime, refreshes and other stream bytes. A short or interrupted session gives a different available image budget. A reference already present at the receiver has a different switching cost from a new reference.

The next figure panel needs blinded motion quality, complete reference and refresh bytes, and player-region distortion. The [Q65 campus pilot](../../../experiments/common-reference-20261122/figure-run/README.md) rendered the common setting for both players on the same 24 native frames. Score the blind comparison against the previously reviewed controls. Rendering success alone does not qualify a rung. If the middle rung passes, check the lower rung and only test a higher rung if motion or a rate decision justifies it. Freeze the resulting ladder across unseen players. For a claim that AVIF is the best *motion* choice, include a payload-matched WebP candidate. Rebuild the still-image panels with `python docs/paper/figures/references/plot.py`.

### When to send a reference

Use the still-image comparison in the appendix. For the main experiment, choose among keeping a valid cached reference, sending one shared AVIF rung for both players, or using conventional coding. Define the remaining budget in **bytes** over a conservative predicted use interval: available link bytes minus the chosen base, poses, ball, headers, model initialization, expected updates and fallback reserve. Also require that any new reference arrive by its playback deadline. Estimate the use interval from information available at the sender when it chooses (observed time until previous changes, shot type, cache identity and buffer). Recompute on a cut, invalidation, bandwidth change or return. The measured future duration is reserved for retrospective scoring.

For each feasible choice use a development lookup of rendered player quality and the opportunity cost of spending those bytes on the rest of the video. Pick the lowest predicted total distortion among admissible choices. Keep the valid cache if a refresh adds too little measured quality to justify its bytes. If no qualified object-coded choice fits, use the conventional candidate. For example, at 20 kbps even **all** of a 9.6-second budget is only 24,000 bytes. The lowest tested AVIF pair costs 24,463 image bytes before transport or a single video frame. The same reference can be affordable on a longer visit, after prefetch that was charged earlier, or from a valid cache.

**Smallest policy test.** Complete the blinded Q65 motion review first. If it passes, test Q46 at the lower byte cap; if it fails, test Q78 against the accepted control. A decision-changing upper rung can be tested later if Q65 passes. Record per-player rendered distortion and temporal artifacts, complete startup bytes, time to first frame, actual cache hits and refreshes. For one short start and one observed return at the planned total-rate budgets, compare the resulting shared policy with always-refresh at a fixed quality, always-reuse while valid, and conventional coding. Include the actual bytes and quality of all fallback frames. Choose rungs and duration estimates on development footage, then freeze them for held-out players and matches. A crossover that persists on fresh passages earns a main-text panel of chosen rung, realized bitrate and displayed quality. Until then the policy is a design to test.
