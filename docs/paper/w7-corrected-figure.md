# W7 corrected player-gap figure

Suggested caption: **Corrected W7 player-region gaps on untouched windows.** Each dot is the best resolved anchor's player-region score minus the hybrid's at its modeled total rate; positive values favor the hybrid for the metric shown (lower LPIPS and DISTS are better). Diamonds and bars give the mean and two-sided 95% t interval across windows in each tier. Parentheses give the number of resolved anchor families out of four; unresolved or missing families are excluded, so these results are conditional and can favor the hybrid. Tier 1 contains six windows from familiar matches; Tier 2 contains three windows from an unseen match with its own fitted adapter. The Tier 2 LPIPS interval crosses zero, so the original two-metric Tier 2 rule is inconclusive. Rates use the corrected 12-crop modeled bank and adapter-plus-guider amortized over two hours; reference reuse is assumed and some operational framing remains unmeasured. These are not complete-bitstream claims.

The figure is generated without model execution by `python -m experiments.tennis.w7_corrected_figure --out-dir OUTPUT_DIRECTORY`. It reads the published corrected W7 artifacts, checks their SHA256 before plotting, and plots their stored per-window gaps and tier means and intervals without recomputation. Input provenance:

- `docs/experiments/w9-inputs/w7-tier1-billing-corrected.json`: SHA256 `62376a4f3dcbe406d2065e54c12ebcbc0d2579aeb97c243cbf843bdf15a66e1f`.
- `docs/experiments/w9-inputs/w7-tier2-billing-corrected.json`: SHA256 `a60ea553077203bd3bbf894843bbe826d613c7c1effdab371ddad0b7b539d979`.

Figure interpretation and correction provenance: [W9 experiment entry](../experiments/w9-adapter-break-even.md).
