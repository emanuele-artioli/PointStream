# Experiments: JPEG reference appearance pilot

### 2026-11-17 — approved; local preparation, campus execution gated

The user approved the bounded pilot in reply to `call_117522ab2b614ee1ad6c883fe399fbeb`
and separately instructed **Continue locally for now** in reply to
`call_c137ccb5bb484268bababd3a2425b66d`. The [live release ledger](player-reference-pilot-20261117/release.json)
records30GPU-minutes/45CPU-minutes, quality70 first and quality40 conditional.
No campus probe, publication retry or GPU job is scheduled during local work.
The original proposal remains unchanged as a historical proposal. Budget
approval persists; network restoration does not require the same approval again.

The previous motion allocation is closed at 3,318 GPU-s and 6,318 CPU-s; its unused balance is not transferable. Both full 240-tick common-tennis players received author judgments of Acceptable. The next hypothesis is that the same common model tolerates cheaper JPEG references while maintaining acceptable source-aligned body and pose. The competing explanation is that reference damage destroys that fidelity. This is an appearance/reference pilot, not a complete paid codec or rate–quality test.

Freeze exactly JPEG quality 70 and 40 before any GPU result, on the already opened Nadal–Kyrgios source. Preserve the same weights, poses, boxes, seed, 512×784, 30 steps, CFG 3.5 and context 24/overlap 4. Use schema-1 24-native-tick runs: Player 2 first as smoke, then Player 2 and Player 1 in the full stage. Require the same player's smoke/full crops to be bit-exact. Compare with existing lossless 24-tick outputs at matched context; no 44/240-prefix comparison or longer-motion inference. Preserve original player identities when labelling. Both players are one source and no significance is claimed. Fresh LPIPS/DISTS, training, new sources, tuning, model search and bulk downloads are excluded.

Each quality is a separate900-second request with one CPU core: real smoke300s, full450s, validator60s. The approved aggregate maximum is **1,800 GPU-s and2,700 CPU-s**, including900 externalCPU-s. Each smoke must pass substantive existing input/output and CUDA/model checks and the measured admission forecast `1.25*(load + 2*(worker_wall-load)) + 90`, at most450s and at most remaining native campaign time minus60s. Historical shared24-tick renders of74.17/83.15s plus about24s loading support the bounded pilot, not a completion guarantee. Stop on failed validation, cost gate, ambiguity or exhausted budget; no automatic retry. Stages are at most45minutes, each request15minutes. Refresh the deadline and perform normal fleet admission only after campus access and publication verification.

Promotion requires a durable randomized source-aligned human comparison of source, lossless and JPEG outputs, including body/pose and actual reference bytes. A uniform JPEG quality is eligible only if **both players** are judged acceptable; uncertainty is not automatic acceptance. Choose the eligible quality with the smallest actual total REF1 byte count. A single-player success is a narrowed scope and a possible future variant, not pooled promotion or authorization for tuning. There is no arbitrary 10% floor and no codec-win inference. A failed setting narrows the claim and does not authorize new tuning. The prepared viewer below must pass actual-publication checks before delivery; the old generic preview may renumber ordered units and must not be used for identity labels. Native encoder/decoder versions and all input, output, job, GPU and code identities must be recorded at execution. Offline target-frame references and geometry remain disclosed.

Admit quality70 first. Admit quality40 only after accepted full output and
closed actual accounting from the first request, with at least900GPU-s and
1,350CPU-s remaining for its whole reservation, including450externalCPU-s.
Charge each externalCPU reservation in full; include setup, failures and final
task teardown conservatively. If the second reservation no longer fits, retain
the first result and stop. Two settings are not guaranteed by the budget cap.

The [proposal](player-reference-pilot-20261117/proposal.json) is not authority to execute. Local CPU preparation authenticates the original packet/profile and conditioning NPZs, encodes actual REF1 JPEGs, decodes them into new NPZ references and preserves other arrays. Its manifest records source, old/new reference, decoded RGB, NPZ and archive hashes and bytes. The archive and configs are local pending review; remote paths in prospective specs are destinations, not claims of publication.

The superseded first local preparation used OpenCV 5.0.0. The v2 preparation used OpenCV 4.10.0 (`k6b_rate_streams.encode_reference` and `decode_reference`). The original REF1 payloads total 366,161 bytes (Player 2: 88,412; Player 1: 277,749). Quality 70 totals 65,120 bytes (31,595 / 33,525); quality 40 totals 41,075 bytes (20,517 / 20,558). These are reference payload costs only. Independent CPU array comparison confirms every non-reference NPZ array is unchanged and each replacement reference equals the decoded REF1 pixels. These input sizes do not establish generated quality.

## Audit and review readiness

The matched lossless control is the common-tennis full publication from job `20261116T124347Z-407d92df`, local `/private/tmp/pointstream-player-replay-results-20261116/tennis_only/publish`; its `receipt.json` SHA256 is `fdb6155adc50582c434836c4e6ec8be2152a32c364f4de41ad11ab181564fcc4`. Original config `docs/experiments/player-requalification-20261116/config.json` SHA256 `eb67e4a972cde36895addef8db306c963423f814a31be7273a31c696ce5e98c6`, conditioning `/private/tmp/pointstream-preview-conditioning-20261116`, checkpoint `/private/tmp/pointstream-player-replay-results-20261116/resources/tennis-base@fit-100.pt`, and source `/private/tmp/pointstream-player-replay-results-20261116/source.mp4` are the local audit inputs. The cached lossless full publication passed the existing substantive output validator against these inputs. Its source identity is `b7d887c0e2ba15ecc4f6e831d23942b2eb0857e40385c840ea4249d037433f1d`; source conversion and exact reference alignment are rechecked at review time against **lossless** conditioning, never against the intentionally changed JPEG reference.

`python -m experiments.tennis.player_reference_review` accepts this control config/hash/conditioning/checkpoint/publication and source, plus one or two completed `--variant CONFIG SHA CONDITIONING SMOKE FULL` groups, `--seed`, and a new local `--output`. It runs the existing substantive validation for both real stages of each variant and the full control, requires Player2's smoke/full plan identities and output arrays bit-exact, and checks both players' source, settings, ticks, seeds, pose, boxes and all non-reference conditioning arrays against the lossless control. It then generates durable local HTML with original Player2/Player1 labels and randomized A/B or A/B/C candidates, native50fps and explicitly slowed10fps. A one-variant review is marked partial; an absent setting remains untested if quality40 cannot be admitted. The separate `reveal.json` contains the mapping, completed qualities and costs; record judgments before opening it. This is a blind development aid, not an independently administered study. No server is needed. Real JPEG publications do not yet exist, so the new audit and viewer have only synthetic tests; they must pass on the actual publications before review or promotion.

Use `q70/spec-template.json` and `q40/spec-template.json` in the local preparation directory as prospective specs. Their deliberately invalid `REPLACE_AFTER_RELEASE` deadline prevents dispatch; after network/publication gates pass, set a fresh timezone-aware deadline in a new spec copy and rerun fleet schema validation and normal live admission. Earlier `spec-unreleased.json` files with a far-future placeholder are superseded and must not be submitted. Templates stage the same exact recipe, conditioning archive, base, match and eight pinned weight resources as the existing schema-1 worker, with the specified300/450/60/900-second caps and smoke cost gate. Freeze the GPU producer at merged revision `55ba06290978bb5e25347031a1b64c1c66107981`; a clean detached copy is retained at `/private/tmp/pointstream-reference-pilot-source-55ba0629`. Record this SHA and source archive SHA when preparing the actual submission. Later local packet/viewer changes do not change this experiment. The smoke validator cannot check future full output; the independent audit above is mandatory after each completed request.

The GPU worker consumes locally decoded Mac/OpenCV RGB arrays from the prepared NPZs; it does not decode REF1 itself. Linux receiver REF1 decoding parity is unproven and is not required for this appearance pilot, but must be established before any codec claim. The v2 manifest records OpenCV version, platform and a SHA256 of the complete OpenCV build information alongside each transform and its inputs; v2 configs preserve the original tick-0 reference and offline lookahead provenance. The v2 `spec-template.json` files have invalid deadlines, and their exact paths and hashes are in the v2 manifest. Earlier local preparation directories and their manifests are preserved historical artifacts and are superseded.

### Local readiness and campus access

The [independent final review](player-reference-pilot-20261117/final-review.json)
passes the corrected actual inputs and scientific protocol. Configured Ruff,
Mypy and focused tests pass; the reference branch's full suite passed2,669tests
with39skips before the disjoint packet implementation was merged. Final CI
checks the combined branch. No GPU work has started.

The [frozen v2 manifest](player-reference-pilot-20261117/prepared-manifest.json)
and [verification](player-reference-pilot-20261117/verification.json) retain exact
hashes. A verified durable copy is at
`/Users/manu/Documents/PointStream-prepared/2026-11-17-reference-pilot-v2`.

An input publication attempt timed out; subsequent fresh SSH connections to
gpu1, gpu5 and gpu6 timed out too. Its
[unknown state](player-reference-pilot-20261117/input-publication-attempt.json)
must be inspected after restored connectivity against the
[expected files](player-reference-pilot-20261117/input-publication-expected.json).
Do not replay the unknown write, delete or overwrite any partial dataset file,
or dispatch while inputs are unverified. The failed connection does not show
a GPU computation failure or establish whether remote writes completed.
