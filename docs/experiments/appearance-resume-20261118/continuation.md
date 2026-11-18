# JPEG reference pilot, 18 November 2026

Q70 `20261118T105102Z-4c8cf5bb` was published with a 900-second, one-GPU, one-core campus cap and 2026-11-18T11:13:35.579183+00:00 UTC deadline. Q40 is conditional on Q70 integrity, closed accounting, and a fresh 900 GPU-second / 1,350 CPU-second reservation. The archived matched controls and scope are in the [approved protocol](../player-reference-pilot-20261117.md). Raw pilot work is preserved at `/private/tmp/ps-appearance-20261118`.

Initial client ID `20261118T104949Z-5af762ee` failed DNS before remote prepare. Canonical status returned unpublished or unknown. Independent gpu1 and management gpu2 checks both found no job directory on the live shared filesystem. No campaign, claim, or GPU charge was created for that ID. All five previously unconfirmed input files were published with mkdir and no-overwrite extraction after the original namespace was found absent, then hash-verified against the frozen expected inputs.

The blinded review uses seed `20261118`, fixed before seeing Q70 or Q40 output. The live Q70 bounded wait uses `scripts/ps-fleet wait 20261118T105102Z-4c8cf5bb --max-seconds 1370`, with local handle 20350.

Q70 passed the worker and separate output validator, including Player 2 smoke/full equality, with a 337-second forecast. Both registered 24-tick crops are present. Creation through observed teardown cost 420 GPU-seconds and 870 CPU-seconds. Q40 `20261118T105900Z-07f1b885` was subsequently published at the same resource caps through deadline 2026-11-18T11:22:47Z.

Q40 bounded wait uses `scripts/ps-fleet wait 20261118T105900Z-07f1b885 --max-seconds 1390`, local handle 32856.

Both Q40 players passed independent output validation, including smoke/full bit equality on Player 2. Q40 consumed 437 GPU-seconds and 887 CPU-seconds through verified teardown. The two campus jobs used 857 of 1,800 GPU-seconds and 1,757 of 2,700 CPU-seconds. Source and blinded candidates for each player were delivered at [the author review](</Users/manu/Documents/PointStream-review/2026-11-18-reference-appearance/q70-q40/review.html>) at 50 and 10 fps. All 16 presentations have 24 frames. Author judgments are pending. Those judgments determine reference eligibility on this development source. Both terminal events have been acknowledged.
