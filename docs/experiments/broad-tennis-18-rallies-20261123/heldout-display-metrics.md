# Complete-display metrics for the six London deliveries

The foreground-by-bandwidth figure needs the whole-picture cost at the same
budget, including cutaways. Before any independent regional labels arrive,
audit the physical ball deliveries already selected at 20.4 and 33 kbps. Use
the saved ordinary encode at the same coded width and common ceiling, plus
the exact AV1 base carried by the ball delivery as an ablation. Eleven pairs
use 640-wide video and one uses 320. A coarse encoding
grid can leave large differences in achieved rates. In that case compare at
the labeled cap and show the actual bytes. The same-base ablation establishes
only the impact of adding the render, including its extra bytes.

Hypothesis: the added rendering has little effect on complete-picture
similarity. The competing explanation is that the changed quantizer of the
ball display's base accounts for the reported quality gap. Measure RGB PSNR
and local luma SSIM on every 960×544 frame, check decoded RGB hashes against
the earlier receipts, and verify exact fallback. Separately report
conventional and court phases. Average frames within a point and points
equally within each test fixture. No object localization or segmentation
claims follow from these scores. A local smoke covers one real point, followed
by both rates for all six in at most ten CPU minutes. No GPU or new encode.

The [complete 904-frame audit](heldout-display-metrics.json) took 172 CPU-seconds. Adding the ball rendering
to its identical base changed RGB PSNR by less than 0.003 dB and luma SSIM
by less than 0.00008 in every delivery. The coarse AV1 grid produced two
larger gaps, 1.35 dB and 1.28 dB, when the ball mode's base stepped down to
an encode using many fewer bytes. For the other ten point-by-cap pairs the
AV1 base was identical and the ball delivery used 1.5–5.0% more bytes.

The two coarse-grid gaps motivate one targeted follow-up. Keep the saved ball
packets. Adjust only the ordinary AV1 quantizer at 640 pixels wide until its
total bytes are within 0.5% of the ball delivery, using at most seven
quarter-CRF probes in each existing bracket and no GPU. If the achievable
granularity is coarser, report the exact gap. This determines how much of the
apparent whole-picture loss came from the large disparity in spending.

The two ordinary re-encodes matched to within 0.042% of the ball packet's
total bytes. The remaining whole-picture PSNR losses were 0.089 dB on point
22 and 0.273 dB on point 108 (luma SSIM differences −0.00237 and −0.00737).
These are the measured whole-picture tradeoffs on those two deliveries.
The ball's regional fidelity and localization still require independent
source labels. [Match-rate recheck and exact bytes](heldout-rate-recheck.json)
