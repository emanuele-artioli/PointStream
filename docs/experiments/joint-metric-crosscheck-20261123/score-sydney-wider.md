# Scoring the wider Sydney receiver samples

Add `--review-samples` to the registered `med_mixed_rung` command. It copies
one physical smoke tick and the twelve frozen review ticks for a complete
240-frame replay into `review-samples.zseq`. The validator checks each selected
RGB digest against the complete receiver receipt and publishes
`review-sample-validation.json` in the full stage. It includes
`checks.review_rung_sha256`, `checks.review_manifest_sha256`, and
`checks.review_sample_sha256` in its result. The original command preserves
its previous behavior when this option is absent.

For each completed replay, retrieve only `rung.json`, `review-samples.json`
and `review-samples.zseq` from its authenticated full stage. Use the
`checks.review_rung_sha256` from the full stage validation in the command
below. For an earlier run with the sample export before the full validation
receipt was added, calculate the SHA256 of its completed `full/rung.json`
on campus. The job's gate file always binds its smoke receipt. Run the scorer
in the existing CPU environment used for the matched metric
cross-check. Reuse the original Sydney review directory, official ball CSV,
and exactly the same frozen metric weights. The standard invocation scores all
twelve source instants. Run once with `--smoke` and a separate output to check
the first source instant and model bindings. Adding more `--rung DIR SHA`
arguments scores up to three physical replays on the same source support.

```sh
python docs/experiments/joint-metric-crosscheck-20261123/score_sydney_wider.py \
  --review "$sydney_review" --ball "$sydney_official_ball_csv" \
  --alex "$alexnet_checkpoint" \
  --rung "$retrieved_rung_dir" "$validated_rung_sha" \
  --output "$new_metrics_json"
```

Each result gives the actual cached and cold prices. It uses the earlier
DISTS, spatial AlexNet LPIPS, luma SSIM and RGB PSNR routines on the source
player masks and ball crops. It reports complementary-background and
whole-picture RGB PSNR separately. Pool the two players, then give the player
mean and ball equal weight for the foreground summary. The sampled metric
panel does not evaluate temporal fidelity. Use the existing scored Sydney
control bank on the same twelve source instants for the primary
foreground-DISTS comparisons. Report each budget whose affordable control
does not yet have a matched LPIPS or PSNR score.

Four archived controls have the matched spatial scores in
[`sydney-wider-anchor-metrics.json`](sydney-wider-anchor-metrics.json).
The two ordinary AV1 points are 83.354 and 135.248 kbps, the neural point is
191.730 kbps and the effective ROI point is 34.154 kbps. The saved 40.502 kbps
neural control supplies DISTS alone. Reproduce the four-control receipt with
`score.py --case sydney-wider-anchors --ticks 12` using the original Sydney
review and the native per-control `arm.json` directories for `--arms`.
The per-frame `background_psnr` field averages PSNR on the complement of
the same two player masks and ball crop as the wider rerenders.
