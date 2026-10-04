# Foreground smoke decision

Diagnostic only. Nothing here is a trained encoder or a claim against AV1.
GPU allocation used: **0 minutes**. No job was submitted.

## Why the live stages did not run

`ps-fleet doctor` did not pass: gpu2 and gpu6 timed out, and gpu4 is locked for measurements. `workers start` was not used. Existing workers on gpu3 and gpu5 were already fresh, so they were left running.

A five-minute fleet heartbeat cannot be registered from this session. The plan forbids an unmonitored GPU job, so no audit, fit, profile, or hold-out compare was submitted. gpu3's RTX A6000 was idle at inspection time (0% utilization, no compute process, about 48 GiB free). That is capacity, not a completed smoke.

The local interpreter has OpenCV and NumPy and does not have PyTorch, so the objective tests were skipped rather than executed.

## Gates

| Gate | Status | Evidence | GPU minutes |
|---|---|---|---|
| F1 crop credibility | blocked | No contact sheet from the three recordings. Fixture audit keeps labels pending and does not invent joint scores. | 0 |
| F2 decoder-only conditioning | passed | `tests/demo/test_foreground_contract.py`. Poisoned source boxes/joints do not change decode or the crop window. AV1 and generator slots are the same decoded track ids. A track id absent from the packet cannot be scored. | 0 |
| F3 opacity objective | inconclusive | `demo/models/foreground_objective.py` implements the masked RGB, balanced alpha, and composite loss without editing `hand_objective.py`. PyTorch tests did not run. No 8-crop checkpoint re-render. | 0 |
| F4 corrected SPADE can learn | blocked | F1 has no reviewed visible-hand split. `demo/experiments/foreground_fit.py` caps an arm at 120 steps and 120 seconds and was not executed. | 0 |
| F5 dominant runtime stage | blocked | No ONNX profile on the 16 saved images. Local timing helpers only check units and the 41.67 ms line. | 0 |
| F6 temporal packets at full precision | inconclusive | Synthetic 8-frame packets round-trip in raw, zlib, and residual modes with identical codes. The hold-out cut at index 120 was not read. No savings claim. | 0 |

## Answers

The crops are not yet credible. No sampled frame from the factory recordings was reviewed.

Conditioning can be decoder-only. The version-2 packet carries track id, handedness, presence, the box, and 21 joints in 1/16-pixel units. The decoder does not read the original box or joints.

Whether corrected SPADE learns these crops is unanswered. There is no fit.

Whether opacity is fixed on real checkpoints is unanswered. The new loss penalizes a missing foreground alpha and an outside leak on tensors, and that behavior was not executed here.

Which runtime stage dominates is unanswered.

Whether temporal coding helps at unchanged precision is unanswered on the hold-out. The codec can represent the same quantized codes with a residual, which is the precondition for that measurement, not the measurement.

## Next experiments

1. Re-run doctor, and if it passes, submit only the F1 audit (budget at most 120 seconds, at most 24 pixel candidates per recording) on an idle A6000 or 6000 Ada. Hypothesis: the permissive pose selection mixes visible hands with non-hands, which the sheets can confirm before any fit. Requires a registered five-minute fleet heartbeat.

2. Only after a human labels at least eight fit crops and four validation crops from different source seconds, run the three-arm 120-step smoke. Hypothesis: the corrected objective reduces outside alpha error relative to the legacy arm on those same crops. That result would still not justify a full training run.
