# Sydney AVIF Q90 startup accounting

Hypothesis: the paid Sydney reference images admit a smaller AVIF Q90
envelope, with unchanged warm packet size. Competing explanation: the
higher-cadence receiver's reference crops compress poorly or the required
framing consumes the savings. Decision: an image-only scenario is useful
if the validated complete Q90 envelope is at least 25% smaller, preserves
both role identities and crop geometry, and the rebound warm packet keeps
every stream section and the total wire length. A later rendered comparison
must requalify fidelity before replacing the observed cold operating point.
Budget: 120 CPU seconds and no GPU time. Use Sydney's verified CRF 70 cache
and warm packet from sender job 20261122T134208Z-0e12634b as resolved by the
saved sender receipt. Halle requires corresponding locally available bytes.

The validated Q90 envelope was 59,776 B versus 149,450 B (60.0% smaller).
Rebinding the cache identity preserved all warm sections and the 12,890 B
packet size. Over four seconds the modeled cold rate is 145.33 kbps, versus
324.68 kbps for the original reference. The decoded appearance crops scored
48.45 and 47.05 dB against their original images. The real renderer still
needs a fidelity comparison. Halle's local export contains only receipts,
so its Q90 byte count remains unmeasured.

Run with Python plus the project dependencies and libavif 1.4.2:

```sh
PYTHONPATH=. python docs/experiments/tomm-reference-price-20261125/measure.py CACHE WARM_PACKET NEW_SCRATCH --revision REVISION
```
