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
