# Scoring the wider Sydney receiver samples

Add `--review-samples` to the registered `med_mixed_rung` command. It copies
one physical smoke tick and the twelve frozen review ticks for a complete
240-frame replay into `review-samples.zseq`. The validator checks each selected
RGB digest against the complete receiver receipt and includes
`checks.review_rung_sha256`, `checks.review_manifest_sha256`, and
`checks.review_sample_sha256` in its result. The original command preserves
its previous behavior when this option is absent.

For each completed replay, retrieve only `rung.json`, `review-samples.json`
and `review-samples.zseq` from its authenticated full stage. Use the
`checks.review_rung_sha256` from the validated fleet result in the command
below. Run in the existing CPU environment used for the matched metric
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
