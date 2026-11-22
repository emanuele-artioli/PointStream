# Al028 low-rate neural display recovery

Before inspecting scores, recover the already encoded original 240-frame Al028 scale-2 QP6, 18, 30 physical packets, at 7.6542, 14.5717, 28.8292 kbps. Compare fixed ticks 10,30,…,230 on the existing 24 source-proposal body regions and whole-frame RGB PSNR, using the same source, crop padding, metric and resize as QP24/QP36. This narrows the actual-rally crossover; it does not establish another match or human-validated quality.

Hypothesis: lower-rate neural controls may beat or lose to the physical joint stream. Competing explanation: downscaled neural quality may deteriorate faster than the unchanged generated players. Preserve all results. Recover exact 960×544 I420 frames from the native packets, verifying all 240 independently decoded frame hashes against four original producer passes; two 240-frame GPU replay stages and 8 GiB shared allocation, forecast gate 1.5× smoke +30s <180s. Run QP30 first as representative; admit remaining two only after successful identity gate. Each job ≤430 GPU seconds, deadline 06:15 UTC. No re-encoding, no new QP search or human labels.

QP30 representative submitted as `20261122T053633Z-18b9214d`, snapshot `9b4939880e7f918ee1323294a7eeaa0f90d33d6ceb6a719d9a5a291280ade68f`, state pending.

QP30 representative `20261122T053633Z-18b9214d` failed the full native smoke on gpu6: the actual decoder did not pass the check requiring both replay passes to reproduce all 240 original producer hashes. The original packet identity passed; smoke salvage contains no decoded frames or report, so the magnitude and source of disagreement are unknown. Shared device contention was recorded. QP6/18 were not submitted under the preregistered gate. No quality result has been produced.

Independent decision: test QP18 as one representative from the separate original producer, retaining the same exact four-pass gate; if it passes, score it before optionally admitting QP6 from that archive. If it fails, stop. QP30 remains unavailable.

QP18 independent representative submitted as `20261122T053856Z-de98ec06`, snapshot `420438c8e9a059ec53555b0859ca93620d2e04fc0b6c8de2d374783703f6a8f9`, pending.

QP18 `20261122T053856Z-de98ec06` had a genuine successful replay and wrote all 12 frames, but the fleet job failed because its spec mistakenly named the Q36 validator, which rejected `--qp`. Preserve the failed attempt. Correct the validator module and retry exactly QP18 under unchanged intrinsic-hash and budget gates; if it succeeds, score and submit QP6.

Corrected QP18 retry `20261122T054250Z-5803d65d` submitted pending; spec SHA256 `17f8e263ea65dc511ea53248d4101a54154d376733a3344f21d33a1bb773d47e`.

Independent local audit of the first QP18 smoke archive SHA256 `97aea2da399da97e9728c2c970957ce3cd874b5e777bab0f68ad6358e07266f0` using the corrected validator passed all checks: ticks, 240 frames, packet, 6-GiB child allocator, smoke forecast, selected bytes and per-frame identities. The fleet job remains failed due solely to wrong validator invocation.

Corrected QP18 retry `20261122T054250Z-5803d65d` failed the scientific all-frame producer identity gate on gpu6 under shared contention; no decoded frames or difference report were salvaged. Accordingly QP6 was not submitted. The earlier independently validated smoke output is an exact recovery but the retry contradicts reproducibility; it is not promoted as a successful fleet job.

Archival QP18 display scored on the 12 fixed source ticks and 24 source-proposal body regions: DISTS 0.39473863, RGB PSNR 29.3870 dB at 14.57167 kbps. QP24 matched all 24 historical individual scores exactly (max error zero). See `q18-body-scores.json`; this recovers known producer pixels and does not claim successful fresh reproducibility. QP6 remains unrun.
