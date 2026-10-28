# Ideal-player full evaluation — timing secondary, COMPLETE2026-10-28

The user explicitly directed launch if timing variation was the only reason
full evaluation did not start. It was the only rejected check in repaired
job20261028T073233Z-36f21aa5. Its byte controls, historical LPIPS/DISTS, CUDA
and memory checks all passed. This human instruction supersedes the old
25% variability veto and the earlier no-retry stop for one new request.

Keep the same frozen input468250a7, environment32e68872, both weights,
four policies, Al80ticks/scale4 and all scientific decisions from the original
ideal-player card. No training, generation, extra footage or altered pixels.

Use --timing-secondary explicitly in the saved spec and worker summary.
Record variability as secondary metadata; estimate with the slower warm
sample, fixed setup/cold excess and35% safety, and retain publication costs.
Do not gate on warm ratios. Numeric validity, complete traces, controls,
SHA identities, CUDA, memory and projection strictly below550s still bind.
Existing saved evidence projects about223.270s including publication; this is not a
benchmark or guaranteed runtime. A fresh fleet smoke exercises the revised
entrypoint before automatic full promotion.

One new request<=1100GPU-s within1120s remaining of original3600s:
smoke200/full550/validator100/overhead250,20unallocated. Hard deadlines and
per-frame checkpointing remain. Stop at the local appearance decision or a
real correctness/resource failure; no new renderer or training is authorized.
Positive result is only local oracle headroom, never a codec win.

## Result

Job20261028T080453Z-34e50ec2 completed all80ticks. Independently audited
finite NO-GO: both ideal supports lose both whole-frame metrics at original
total rate, robust+/-0.001. Charged250s; cumulative2730/3600s,870unused.
Decision made; stop. [Full report](../k6b-ideal-player-full-report-2026-10-28.md).
