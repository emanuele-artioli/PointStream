# Demo internal audit, 3 October 2026

This audit follows Cursor sessions `a5d37a69-b392-4fb2-afdd-cce0b7c612fc`,
`388bee31-79d4-41e9-83da-fd5e42889908`, and
`7332ff3e-3e00-4b7f-bfd0-0cea79e3429b`. It inspects the local experimental
code and saved remote JSON records. It does not train a model, change a
training selection, or establish a baseline win. Existing evidence stays
unchanged. The main PointStream campaign has not been merged into this work.

## Foreground findings

The current SPADE/pix2pix ranking is a ranking of these runs, not their
architectures. `jobs/hand-rd/train/report.json` records early stopping at
26/7 epochs for factory001/factory002 SPADE and 13/9 for pix2pix. These
networks were trained with different objectives: SPADE uses masked appearance
and an outside-alpha penalty; pix2pix uses whole-image RGB L1. There is no
adversarial loss in this trainer.

`demo/models/hand_objective.py` does not supervise positive alpha inside the
target hand. Zero alpha everywhere satisfies its matte term. RGB outside the
target region is also unconstrained by that loss. The foreground scorers in
`score_holdout_generators.py` and `fair_crop_av1.py` discard the predicted alpha
and score raw RGB against a black-matted target. This is a material mismatch
between training and evaluation. It can contribute to visible background
bleed, but does not prove that all appearance/geometry errors are caused by
that mismatch. Correct alpha supervision and decoder-side compositing need a
fresh smoke before retraining or rescoring; old checkpoints/results must stay
separate.

`choose_hand` chooses the best returned skeleton without rejecting low mask
agreement or degenerate geometry. Training then relies on `selected`, plus a
minimum crop-mask area. That is not a semantic hand validity test.

The stored RTMW training candidates after the row-level `aisle`/`look` flags,
before the final crop-area and last-second split, contain:

| Source | Selected candidates | Fewer than half the joints inside mask | Fewer than 8 distinct joint positions |
|---|---:|---:|---:|
| clip 1 | 1534 | 347 | 166 |
| clip 3 | 539 | 162 | 53 |
| factory002 | 2608 | 291 | 65 |

Distinct positions here mean coordinates rounded to 0.1 pixel. These are
diagnostic flags, not measured non-hand counts or an approved rejection rule.
4586 of 4681 stored mean confidence values exceed 1, and none of these hand
records stores per-joint confidence. Do not interpret 0.25 as a calibrated
probability threshold for these records. Whole-body heads return 21 locations
even for a bad crop, so counting returned locations alone is insufficient.

There is also a packet/scorer selection mismatch. `segment_delta_payload`
keeps at most two sorted detections. Both foreground scorers iterate all
detections and fall back to original, untransmitted landmarks when a decoded
entry is absent. AV1 side tracks retain the last crop per side. Remote
`jobs/hand-rd/holdout/*/poses.json` records:

| Hold-out | Frames | Frames with more than 2 detections | Frames with duplicate side labels |
|---|---:|---:|---:|
| clip 1 | 300 | 3 | 66 |
| clip 3 | 300 | 22 | 93 |
| factory002 | 299 | 124 | 169 |

The decoder returns joints but not the transmitted box/identity to the scorer;
the scorer uses the original box. Both rate and distortion must be regenerated
after adopting one common detection/track selection and using only decoded
packet fields for conditioning. No original-coordinate fallback is valid.

Both generators use the same packet. A hand at 8-bit x/y precision uses 42
coordinate bytes plus box/flags before temporal compression. Training a more
specific appearance generator does not automatically reduce that packet.
First remove redundant detections and establish stable tracks. Then evaluate
temporal prediction, crop-local geometry, and learned/bone-based joint
prediction before dropping joints. Report added headers, residuals, appearance
setup, and geometric error. Revisit quantization alongside the main campaign
only after the demo's internal decoder/scorer contract is correct.

## Runtime findings

The hold-out pose timer includes JPEG/mask reads, connected components, pose
calls, crop construction, and bookkeeping. It is not a pure pose inference
timer. It requests CUDA without recording/asserting the actual session
provider, unlike the separate GPU timing smoke. Profile those stages on the
same representative input, preload images for a compute-only measurement,
check each session provider, and batch the crop calls before considering a
smaller pose head or distillation. SAM propagation alone is about 3 fps in the
saved run, so fixing pose alone cannot establish 24 fps for that pipeline.

## Background findings and provenance

Current remote DCVC HEAD is
`cbdae87a5445114cdc7f48816da63ea80bdeac40`, with tracked changes to
`src/layers/extensions/inference/dmc_common.cpp` and `train_video.py`, and an
untracked CUTLASS tree. This is a current inventory, not proof of the exact
historical scoring revision. The saved score JSON does not identify checkpoint
hashes. Reconcile the scored `s1/ckpt.pth.tar` with the stage-status `net`
tensors and logs before interpreting LD/HT-S failure or HT-L completion.
An epoch belongs to its stage; `status_epo19` alone does not establish a
combined epoch count. Preserve the source diff and hashes in new runs.
Reading the saved Torch ZIP metadata confirms embedded epochs 19 and 10 in
the factory001 HT-L stage-0/stage-1 status files respectively. The current
stage-1 checkpoint is 482521204 bytes. A bounded SHA-256 read timed out on the
shared filesystem, so no checkpoint hash or tensor equivalence is claimed.

DCVC-UF's QP is a learned index across lambdas 1–768 in this experiment;
higher indices can mean better quality and more bits. This is not an inverted
AV1-style quality curve. The current scoring path uses RGB PSNR for both
codecs. Compare exactly the same cuts: common 1/8/32-frame cuts are available;
HT long cuts use 296 frames while the full factory001 AV1 cuts use 300.

The HNeRV latent-only Huffman estimates on the 300-frame factory001 cuts are
78.8/76.2 kbps at 18.07/20.91 dB. The estimator does not predict across frames
and omits code-table/header packaging. Its `total_bits` also includes decoder
weights; this is different from steady-state latent rate. A lossless temporal
latent packet experiment can reduce rate but cannot improve reconstruction
quality. Compare actual packaged bytes first, then a lossy latent sweep, with
decoder setup reported separately. Do not call the current result a bound on
temporally coded latents.

`render_bg_visual_snapshot.py` is a bounded qualitative preview on an existing
eight-frame segment. It currently labels HNeRV with `total_bits`, so any panel
must explain that setup is included rather than compare that number directly
with steady-state streams. It is not a full-stream evaluation or deployment.

## Paths and preservation

Verified canonical data root: `/home/itec/emanuele/Datasets/pointstream-data`.
The old `/home/itec/emanuele/pointstream-data` is a symlink. It remains needed:
the main checkout's `.ps-data-root` and numerous active demo scripts still use
the old path. Do not remove it until active path consumers are migrated;
retain historical evidence files unchanged.

The confirmation worktree is registered at
`/home/itec/emanuele/.codex/worktrees/pointstream-confirmation`, and both Git
pointers use that canonical path. Its HEAD is
`5009f68c7855981bcb92d8bc1314a631eac47eb9`; its uncommitted
`manifests/second_domain_generalization.json` is preserved. Git itself does not
need the old alias, but other active consumers must be checked before unlinking
it. Never remove the target worktree.
The old confirmation alias was subsequently removed after a bounded active
configuration/process reference check returned no references. The target HEAD
and dirty manifest were verified unchanged; the manifest SHA-256 is
`6bce284070ee3e90b2b585020085505baca22874eb72f9cbec12de32e160d16a`.
Recoverable link metadata was saved at
`Datasets/pointstream-data/jobs/demo-path-audit/confirmation-alias-20261003T102627Z.json`.
The data alias remains in place.

## Promotion order

1. Audit crops with stratified sheets and geometry/mask/temporal flags. Preserve
   partial/occluded hands and ambiguous cases; do not silently select easy
   hold-out crops or label model confidence as ground truth.
2. Fix the packet-to-scorer selection/geometry contract and SPADE
   alpha/compositing mismatch. Smoke on a small verified hand set before any
   full training. Keep training validation distinct from final hold-outs.
3. Profile and optimize the existing pose/segmentation path, then evaluate
   smaller models/distillation if needed. Report measured throughput honestly.
4. Reconcile background checkpoints and inspect matching decoded frames.
   Compare pretrained and fine-tuned codecs before spending more training.
5. Measure temporal hand packets and HNeRV latent streams with actual headers,
   stable identities, and rate/quality/latency at matching segment lengths.
6. Integrate candidate foreground/background encoders on all three full
   hold-outs, compare to full-frame AV1, then rebuild the website. Later fuse
   the main PointStream campaign's compression work; do not merge it now.
