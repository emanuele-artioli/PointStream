# Experiments: K4b

Entries for this item, oldest first. The protocol, the entry template and
the decisions are in [experiments.md](../experiments.md).

### 2026-10-16 — K4b.1: identifying the two players (rule)
- Question (PLAN K4b, item 1). K4's motion rule (the two tracks that travel
  furthest) mislabels a player who stands still (2 of 9 broadcast windows).
  Which content-blind rule picks the two players, judged against players
  marked by hand rather than by the rule's own output?
- Data: frames from broadcast scene clips K4 did not measure, chosen
  content-blind (lowest sha256 of "pointstream-k4b:<match>/<clip>/<frame>"),
  at least 48 frames from at least 8 clips across the 5 matches, court
  views only once K8 is adopted. Person instances from SAM 3.1's "person"
  and "tennis player" prompts, as K4's segmenter.
- Marks: a local review page shows each frame with its person instances
  numbered; the user clicks the two players (or marks one or none visible).
  These marks are the labels; no other labels exist for broadcast players.
- Candidates (content-blind, each a fixed function of the clip):
  (a) motion in body heights (K4's amendment 1); (b) horizontal motion,
  with the clip's last second discounted; (c) a racket mask touching the
  person on most frames; (d) inside the court outline (K8's homography;
  evaluated once K8 is adopted); (e) keypoint motion; (f) the combinations
  (c)+(d), (a)+(d), (a)+(c).
- Decision rule (fixed before marking). Per candidate: the share of marked
  frames where both marked players are its two picks (mask IoU ≥ 0.5 each;
  frames with one marked player count on that one). The candidate with the
  highest share is adopted if it reaches 0.95, ties to the simpler rule;
  it is then applied once to every K4 window as amendment 2, with both
  earlier definitions' numbers kept. If none reaches 0.95, the table is
  reported and the user decides.
- Hypothesis: (d) or (c)+(d) wins: players are the people inside the court
  and holding rackets, while ball kids and line judges stand outside the
  lines or hold none.
- Budget: SAM 3.1 instances on the fleet (smoke ≤ 600 s, ≤ 45 min per
  stage); candidates on CPU; the user's marking about 15 minutes.
- Marking-stage implementation (2026-10-16, `codex/k4b-players`; rule
  above unchanged). Zero-based decimal frame indices and clip stems as in
  K4. Enumerate all frames in all 159 scene clips in the five `MATCHES`
  directories; exclude K4's ten broadcast clips in their entirety, including
  the outro card. Take the smallest **global** hash prefix meeting all
  three minima, without a content/court filter: 48 frames, 40 clips,
  all five matches (alcaraz/hurkacz 3, medvedev/struff 7, rublev/auger 20,
  djokovic/federer 7, djokovic/zverev 11). Selection is metadata-only;
  `tools/datasets/k4b_inputs.py` preserves the complete inventory, exclusion
  list, hashes, original clip identities and conversion commands.
  Outputs: `Datasets/pointstream-data/tennis/k4b-2026-10-16/`.
- This stage obtains review instances only through K4's
  `Sam31SequenceSegmenter`, offline bidirectional,
  both prompts, K4's limited-range 1080p 4:2:0 conversion and 960×540
  quality-95 JPEG path. It does not estimate clip-level motion from sparse
  review frames. Full temporal clip features for candidates (a), (b), (c),
  (e), (a)+(c) and scoring await the marks; (d) and its combinations await
  adoption of K8. No candidate or K4 amendment is applied at this stop.
  Raw masks of both prompts survive; overlapping prompt masks (IoU ≥ .8)
  share a review number, with the first mask as its scoring mask and all
  source IDs recorded. Page has an original-image toggle, explicit visible
  count, missing-instance flag, per-frame confirmation, browser storage and
  JSON export/import bound to selection and instance-result SHA256s.
- Marking-stage budget: one fleet job, smoke first 3 hash-ranked frames
  through the same path ≤ 600 s, full 48 frames ≤ 2,700 s, validator ≤ 300 s,
  total ceiling 3,600 s, compatible Ada/A6000 pool, shared execution (no
  timing claim). GPU outputs and frame/mask identities must pass the module
  validator for **both** smoke and full before the page is cited. The
  correctness smoke gates production of the page, not scientific adoption.
  Initial CPU preparation failed before conversion because an assumed
  ffmpeg path was absent; its empty archive is retained. Preparation v2
  uses the probed `/opt/local/bin/ffmpeg` and a separate `inputs-v2/`.
  First fleet request `20261016T085803Z-4152bf09` (`32ec13c8`, not evidence)
  was cancelled while pending after review found its validator expected
  local scratch to survive publication. The replacement validator reads
  `published.tar`, like K4's, and is tested after removing scratch.
  Request `20261016T090018Z-00e9fcb6` (`bd38cad9`) was also cancelled
  before allocation: its conservative 24,000 MiB reservation plus the
  sharing margin no longer fitted the available GPUs. The same revision
  and inputs were submitted at 16,000 MiB (recorded SAM smoke peak 11.3 GiB;
  this stage has one frame per session), without increasing the budget.
  Both cancelled requests' revisions were read from their `ready.json`.
- **Rejected marking attempt** `20261016T090231Z-64b221c7`, gpu5 RTX 6000 Ada, shared;
  revision **`bd38cad9bfdc4f082a3a39621a88083851dc500e`**, read from
  `ready.json` (no selected changes; empty patch SHA256 `e3b0c442…b855`),
  snapshot `87f25e2a…bacb4`, source identity `028dc031…da0`, spec
  `be5462a7…b5ce4`. Exact spec `marking-job-16g.json`; selection
  `inputs-v2/selection.json` SHA256 `5e79df8406407fbeeab6ef1ed2a990f39dad6393ecb1109d1fdbe65693fb4088`,
  selected converted frames archive `3af558001d71fbd06eff3d366aafe853de9b1e84c8775ab96c30d30dd3bcf161`.
  Environment: K4's `pointstream-20261009T214002Z.tar.gz`, SHA256
  `32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661`;
  SAM checkpoint `0567debe…41cb6`. CPU preparation used ffmpeg
  `n7.1.1-56-gc2184b65d2` at `/opt/local/bin/ffmpeg`; exact preparation
  sources, checksums, command and tool versions are retained in
  `preparation-source.tar` and `preparation-provenance.json`. There are
  no native video encodes in this marking stage.
- This attempt's one-frame sessions produced **zero instances on all 48
  frames** (peak 6.17 GiB). The smoke and local full validators passed
  integrity checks but failed to detect this functional failure; the
  subsequent output review caught it. It is not a usable marking page
  or scientific evidence. The archived output and the read-only root
  `index.html` are retained as failed artifacts, not offered for marking.
  SAM's pinned `sam3/model_builder.py:1185` enables masklet confirmation;
  `sam3_multiplex_base.py:251` requires three consecutive detections.
  One-frame video sessions cannot satisfy that gate.
- Correction, with sampling and decision rule unchanged: `inputs-v3/`
  reuses v2's frozen inventory and selection. Around each selected frame,
  eight **real consecutive** frames are decoded (start `frame - 4`,
  clamped to the clip's valid span), through exactly K4's conversion and
  JPEG path. Both prompts propagate on those short sequences with the
  original confirmation settings; only the originally selected frame's
  instances enter the page. All-empty smoke/full outputs now fail
  `nonempty_model_output`, and real contiguous context is checked. This
  supplies masks for manual marking, not the later full-clip candidate
  features. A regression covers published-archive validation after scratch
  removal, mask tampering, and rejection of all-empty model output.
  Replacement budget ≤ 3,000 s (smoke ≤ 600, full ≤ 2,100, validator ≤ 300),
  keeping total GPU execution within the original 3,600 s ceiling.
- Corrected job `20261016T091155Z-7b19ac9a`: revision
  **`dc4d4fdd8421cb9dd53178cf39e922e99e7802ef`**, from `ready.json`,
  no selected patch (empty patch SHA256 `e3b0c442…b855`), snapshot
  `7b9a32f7…3d38a`, source `836c4a3f…7f40b`, spec `fa2ec81f…8b5a0`.
  Inputs `inputs-v3/selection.json` SHA256
  `7d1a22f62ac03c514b5f47dbc8c4c3864d9e6b92d461ab90175d2b520c6c23ef`,
  context archive `3bf99973505a7fb77c68d7155fd56b584c62e3751489f87a0a161a2b46f269e6`;
  original frame IDs, ranks and source-video SHA256s checked equal to v2.
  Exact spec `marking-context-job.json`; preparation source/checksums and
  command `preparation-v3-source.tar`, `preparation-v3-provenance.json`.
  Environment, checkpoint, GPU compatibility and original sampling rule
  unchanged. Page to mark is the corrected `review/index.html`, not the
  failed attempt's root `index.html`.
- **Marking page ready; stopped for user marks.** Corrected job completed
  on gpu5 RTX 6000 Ada (shared). Both the 3-frame smoke and **48-frame
  full output** pass the module validator locally, all 16 checks. Full
  `published.tar` SHA256
  `2dc9acb6541224f9f72af8ceb9c781308c3e62d19fcb819515736e4b7da261ee`;
  `instances.json` `89b78156e85b62741326a5ff2dc8a020ac58934f5dba0ad33a158c887b40f731`;
  self-contained page `2a05d0eaf6a0a480e2caddd5d7714df72376c2f1606847506f22bfbfd49f0223`.
  334 numbered instance-frames across 48 review frames (none empty);
  all 48 original review JPEGs byte-identical to the frozen one-frame
  attempt's originals. Peak allocated GPU memory 10.70 GiB, within the
  16,000 MiB declaration. This is resource/marking validation, not evidence
  that a player-identification rule reaches 0.95; no timing claim.
  Durable page:
  `/home/itec/emanuele/Datasets/pointstream-data/tennis/k4b-2026-10-16/review/index.html`.
  Companion `instances.json` and `provenance.json` identify the raw masks
  in the job's full `published.tar`. Local validation receipts are
  `context-smoke-local-validation.json` and `context-full-local-validation.json`.
  Browser QA on synthetic frames covered two/one/none, confirmation and
  persistence; the real complete page was checked at 0/48 confirmations.
  No real frames were marked by the agent. Ruff, mypy and the full test
  suite pass (560 passed, 19 skipped). Monitoring paused after completion.
  Next: freeze exact candidate implementations before reading the returned
  labels, compute their clip features and score by the unchanged rule.
  No rule is selected or scored yet; (d) waits for adopted K8 homography.
  Any winner is reported for review of K4's amendment before applying it
  to K4's windows, as required by this task.
- Labels returned (2026-10-16): immutable
  `marks-05f513052625.json`, SHA256
  `05f5130526254316c078003a3a66398c97aa33027ac56519e0f267429eba50ba`.
  Only schema, page identities and completeness were inspected initially:
  correct review/selection identities, 48 entries, 48 confirmed. Individual
  selections, visibility, missing flags and notes were not inspected before
  freezing `experiments/tennis/k4b_rules.py` and its focused tests.
- Candidate implementation freeze (before opening selections). All use
  tracks present on ≥ half the clip frames and choose at most two.
  (a) K4's sum of consecutive centroid distances / median mask height.
  (b) sum of consecutive absolute horizontal displacements in SAM pixels,
  zero weight for the last second. (c) one-pixel dilation of the racket
  mask at 960×540 for contact; require contact on **more than half all clip
  frames**, rank by contact fraction and abstain for unqualified tracks.
  (e) sum of mean displacements of jointly confident (≥ .5) YOLO11x-pose
  joints on consecutive fitted frames / median mask height (K4's native
  whole-frame pose, image size 1280). (a)+(c) ranks first by contact on
  most frames, then (a), filling remaining picks with (a). No other
  combinations without (d) are prescribed. Track ties use canonical
  prompt/native-ID order; candidate ties follow (a), (b), (c), (e), (a)+(c),
  putting the simpler alternatives first. Duplicate prompt tracks are
  matched one-to-one when IoU ≥ .8 on > half their co-visible frames,
  covering ≥ half the shorter track; prefer the person mask.
  Scoring uses injective IoU ≥ .5 matches for two marked players, only
  the marked one for one-player frames, and no visible picked masks on
  zero-player frames. A visible player without a labelled instance cannot
  earn an IoU match and is never silently excluded. The 0.95 bar, sample,
  metric, candidate list and deferral of (d) are unchanged. These functions
  do not load annotations to select any constants.
  A cheap proposal oracle is run first; it bounds **only the review's
  existing masks**, not re-segmentation of the full clips, which could
  return a previously missing instance. Full-clip feature extraction
  covers 30,825 frames across the same 40 clips; it is not approximated
  by the review's eight-frame contexts.
- Label audit after the freeze (`466860d`): 37 two-player, 10 one-player,
  one zero-player frame; no missing-instance flags or notes. All 48 pass
  schema, ID, confirmation and visibility checks. The proposal oracle is
  48/48, scoped to selecting the review's exact proposals. Audit artifact
  `label-audit-05f513052625.json` preserves this check and its per-frame rows.
- Full-clip feature stage preflight (before GPU runs). Hypothesis: contact
  plus motion improves over centroid travel when a player stands still;
  competing explanations are SAM racket misses, camera motion, pose misses
  and track fragmentation. Frozen formulas and human marks are not tuned.
  Every one of the 30,825 native source frames is converted once by K4's
  ffmpeg path and used for SAM person/player/racket and YOLO11x-pose.
  The input manifest excludes human selections. Pose runs before SAM and
  releases its model; SAM's official CPU video/state offload is enabled
  for long clips. The smoke compares all class/frame unions to native
  non-offloaded inference at IoU ≥ .999; offload is part of config provenance.
  Defaults for existing SAM callers remain unchanged. Full-clip native
  SAM IDs are retained; cross-prompt aliases and scalar features are CPU
  operations, with raw SAM masks, raw pose detections and scoring-frame
  masks saved so the validator can recompute them independently.
  Pilot axis is clip length, not quality or thresholds: minimum
  `medvedev_struff/007` (64 frames), median `djokovic_zverev/004` (407),
  maximum `djokovic_zverev/000` (3,975). Pilot promotion requires valid
  full artifacts and evidence the longest clip fits one ≤45-minute stage
  and the remaining clips fit the phase budget. No clip is truncated for
  the final score. Each job has a ≤600 s prefix smoke, substantive
  validator, SHA256-identified original video/weights/environment/manifest,
  ≤2,700 s full stage, 300 s validation reserve, and a deadline.
  GPU compatibility: the already-verified A6000/Ada pool, all six hosts
  eligible by model filter, 12,500 MiB allocator limit (recorded SAM peak
  11.3 GiB; CPU-offload pilot must substantiate long-clip use), shared
  execution with no timing claims, 40 GiB scratch. Phase ceiling 8 GPU-hours
  and 10 h wall, including pilots/smokes; record actual usage and reserve
  remaining job allowances before subsequent submissions. Failures or an
  unattainable stage/phase bound stop promotion, without enlarging the budget.
  Complete focused tests cover offload flags/provenance, the five rules,
  strict majority, gaps and tail discount, joint visibility, duplicate
  prompts, target-mask export, injective IoU matching and annotation identity.
- Feature pilot inputs: `features-v1/manifest.json`, SHA256
  `4db6651100efc5c450525b154d7f82e9b1b53de47e6699f02aaf2a25739a48b5`;
  40 immutable original-video identities (3,269,018,181 bytes total),
  hard-linked under `features-v1/inputs/`. Exact per-clip specs are alongside
  the manifest. Environment/weights reuse K4's pinned artifacts. Deadline
  `2026-10-17T01:00:00Z`, inside the 10 h wall ceiling from preparation.
  Submitted first, before any long-clip promotion:
  `20261016T151219Z-f30bd26d` (`medvedev_struff/007`, full allowance 300 s,
  total 1,200 s), and `20261016T151309Z-259d7651` (`djokovic_zverev/004`,
  full 446 s, total 1,346 s). Both `ready.json` records identify
  **`ebd0024d79c591fd70bcf868210fe609152ef498`**, snapshot
  `2d882a9c…257e8`, source `dc7ca0b3…b1235`, empty selected patch
  `e3b0c442…b855`; specs `342753ea…53840` and `98efd543…157a7`.
  The then-untracked scorer was explicitly excluded from both snapshots;
  it is not a dependency of feature extraction. No feature result is cited
  until its full output passes the module validator locally.
- Both first pilots failed in the **native comparison** before CPU-offload
  propagation or full promotion. Same diagnosed cause on gpu3 and gpu2:
  a 324 MiB activation allocation exceeded the 12,500 MiB per-process cap
  (11.34 GiB already allocated plus 619 MiB reserved; 12.21 GiB allowed),
  while physical memory remained available. These are failed configuration
  smokes, not evidence about any candidate. Saved logs and artifacts remain
  in the original jobs. Correction: 16,000 MiB allocator cap and 4,096 MiB
  sharing margin, giving a 20,096 MiB admission reservation versus the old
  20,692 MiB; more actual allocator headroom with physical sharing headroom
  still reserved. All input pixels, models, parity threshold, candidate
  formulas, stage timeouts, deadline and 8 GPU-hour phase ceiling unchanged.
  Revised specs are preserved separately in `features-v1/specs-16g.tar`.
  Re-run the short pilot alone first, then promote only from valid evidence.
- The 16,000 MiB short attempt `20261016T152206Z-e078b538` also failed
  during the native comparison (gpu2, before full promotion). Its
  `ready.json` revision is `04808b97db8b1137c0490f109a7695d95081af1a`,
  snapshot `54e80bea…96346`, spec `edd289ea…ba6fc`, empty selected patch.
  A 1.27 GiB detection-head activation exceeded the allocator cap; physical
  memory was available. No candidate outcome was produced. Revised-spec
  archive SHA256 `7773a5949ddbfba8ccbbeb9c4927fa9412840f3da24e1dd434c9307dd88d6701`.
- Revised memory approach, before another smoke: upstream
  `model_builder.py:1191` batches **16** frames for grounding; those
  detector activations caused both cap failures. Explicitly use grounding
  batches of eight, with video/state offload and the same 16,000 MiB cap.
  This changes batching, not temporal sampling: every native source frame
  still runs through the same detector/tracker and fixed candidate formulas.
  The official attribute is asserted effective and included in config
  provenance; default callers retain the original batch. The production
  smoke is still all first 48 frames. Its parity control uses the first 12
  real frames, default batch 16 (effective 12) versus configured batch 8,
  comparing every class/frame union at the unchanged IoU ≥ .999 threshold.
  Both parity mask sets and the comparison are preserved even if it fails.
  No memory cap, time allowance, deadline, phase budget, decision rule or
  human mark is changed by this revision.
- Before admission, cancelled `20261016T153440Z-1b94df2e` (terminal
  cancelled; no stage ran). Snapshot revision `2d2c5fda4203e8dfa499a404fcaa1d000cb4527b`;
  the queued request was also rejected at gpu5 admission for insufficient
  shared-memory headroom. Source inspection found that the pinned top-level
  multiplex `init_state` supports video offload but not tracker-state offload.
  The compatibility adapter had silently filtered the latter flag. Corrected
  it to reject unsupported true requests explicitly (regression tested), and
  request **video offload only**, retaining GPU tracker state. The underlying
  tracker's separate CPU-state option is not exposed by the top-level API;
  no adapter is added to change it. Therefore the longest-clip pilot must
  establish state-memory feasibility before promotion. Earlier references to
  video/state offload describe the requested, not effective, placement.
  Corrected specs preserved as `features-v1/specs-video-only.tar`, SHA256
  `3df90280f2308b23b8ed305a7f2794f4253cc1d643a5d1efa4a13cd681141c6a`.
  Grounding batch eight, parity IoU ≥ .999, all-frame sampling and all budget
  ceilings remain unchanged. Cancellation consumes no GPU-stage allowance.
- Corrected short pilot `20261016T154053Z-91a66242`, gpu3 RTX A6000:
  actual `ready.json` revision `84ec4a3edfa98c0f5778e523e9f0f8b1ef7fe9a3`,
  snapshot `3a1880ea…367e4`, source `97832f14…6c37e`, spec
  `3ee88af7…0459b`, empty selected patch. The immutable spec's generic
  “CPU offload” basis is unchanged from the batch-eight archive; the code
  and recorded effective configuration explicitly request video only.
  No full output or score is cited yet. Local checks: ruff and configured
  mypy pass; pytest 572 passed, 19 skipped (rerun with process-inspection
  permission after the sandbox denied `ps` in an existing contention test).
- The corrected smoke failed its parity gate, before production/full:
  minimum class/frame union IoU **0.9265116431832364**, below .999
  (123 native reference instances on 12 frames). Both RLE sets and
  `parity.json` survive in the job's `smoke/partial.tar`; the failure is
  not a candidate score. Batch eight is rejected. Restore **native batch
  16**, retain video-only CPU offload and the identical parity gate.
  Reserve 24,000 MiB allocator memory plus the unchanged 4,096 MiB sharing
  margin, admitting only GPUs with sufficient headroom; this targets the
  diagnosed native detector activation peak. Timeouts, deadlines, the
  8 GPU-hour phase ceiling, formulas and human marks remain unchanged.
  Specs saved separately as `features-v1/specs-native16.tar`, SHA256
  `a5d55d0b9e56b3765bce7fee955cb0641981d1a49827a6ebbfa43b3a01631264`.
  Re-run short alone; no medium/long/full-corpus promotion from failures.
- Native-batch pilot `20261016T154524Z-9d4d461c` (gpu3 A6000), actual
  ready revision `796933bb91d3828075eb2cfe9a1e0875ced2606b`, passed the
  12-frame offload parity comparison exactly (minimum union IoU **1.0**,
  123 reference instances); failed in the subsequent 48-frame production
  path, before full. A 1.27 GiB request hit the 23.44 GiB allocator cap
  with 20.35 GiB allocated and 3.07 GiB reserved but unused. Preserve its
  partial masks/parity and logs. Test the diagnosed fragmentation remedy
  `PYTORCH_ALLOC_CONF=expandable_segments:True`, with the same native
  batch, 24,000 MiB cap, input pixels, model, gate and all time ceilings.
  Record actual expandable CUDA segments and require them in validation;
  a successful short full and length pilots remain prerequisites to scaling.
- Expandable-allocator short pilot `20261016T154944Z-4081668c`, gpu3
  A6000, actual ready revision `3494a8b1c888c6daeb90c0cabd748fa3fa655754`:
  smoke and full complete. Full **locally validated**, all 17 checks true:
  64/64 frames, original source coverage, raw SHA256s, pinned CUDA SAM and
  pose, effective native batch, actual expandable segments, recomputed
  features and exact target masks. Full peak allocation 22.1047 GiB.
  Smoke parity min IoU 1.0. Smoke execution 154.750 s, full 120.802 s;
  GPU-claim interval 325.555 s. Shared timing is not a performance claim.
  Local receipt SHA256 `bb3cc62943dd42e6a761920d48b239e46358cc056d86afee0f20b95eca5f1be8`.
  These are configuration/full-coverage checks, not a candidate table.
- Length promotion is median only: 407 frames of `djokovic_zverev/004`.
  The initial 446 s full allowance has inadequate margin given the short
  stage's fixed loading cost plus per-frame propagation. Use a separately
  preserved median spec with full 900 s, smoke 600 s, validator 300 s,
  total 1,800 s; the 8 GPU-hour phase ceiling and deadline do not change.
  Do not submit the maximum or remaining corpus until median evidence
  establishes feasible stage and phase bounds. Prior failed smoke stage
  time is 390.260 s; charge complete GPU-claim intervals in the phase ledger.
- Median `20261016T155820Z-40b6c9d9`, gpu3 A6000, actual ready revision
  `f58bb8f1890aba12bffa6a7c2bfb8d079f8195be`: smoke passes, full fails
  at frame 160/407 of the first prompt, after 164.207 s. A 1.27 GiB
  request exceeds the 23.44 GiB cap with 21.92 GiB allocated and 344.50 MiB
  unused reserve. This demonstrates growing state pressure; no full result
  or candidate score is cited.
- Next memory-only correction uses the pinned lower-level tracker's
  **existing** `init_state(offload_state_to_cpu=True)` option. The top-level
  adapter still rejects unsupported requests; a separate narrow worker
  adapter forwards the native tracker arguments, verifies each returned
  state's `storage_device == cpu` and flag, and records counts and API
  version. No compute kernels, masks, frame sampling or tracking rules are
  altered. Effective config SHA256 includes this adapter. The existing
  native-versus-offloaded IoU ≥ .999 smoke now also gates tracker-state
  placement; native and offloaded mask archives are retained. Reset the
  initialization counters before production, so parity alone cannot pass
  the production-placement check. Tests verify forwarding, actual storage,
  unsupported API rejection and silently ignored requests. Repeat the
  bounded median pilot directly: its 48-frame correctness smoke precedes
  the 407-frame full stage, with identical 600/900/300 s limits and phase
  ceiling. No further corpus promotion until it and the scale bound pass.
  Source inspection also finds top-level `cached_frame_outputs` retains
  per-frame **boolean** masks on GPU independently of tracker-state storage.
  The same placement adapter stores this cache's already-computed masks on
  CPU after the native cache filter, preserving the original GPU output
  dictionary used by propagation/postprocessing. It changes storage only;
  no masks are recomputed. Record and validate every cached mask's CPU
  placement; adapter version 2 identifies both routes in the effective
  config. Regression test confirms the native output is unmodified and the
  cache copy moves to CPU. The native-mask parity gate covers this route too.
  gpu3 has 384 GiB CPU RAM available at inspection, for long-clip video/state
  storage; the maximum-length pilot must substantiate actual feasibility.
  Parity is strengthened before this promotion: require identical native
  class/track-ID sets and per-instance IoU ≥ .999, as well as class-union
  IoU ≥ .999. The validator recomputes these from both saved RLE archives;
  a regression covers an identity swap with identical union masks. This
  protects motion-based candidates from an offload route that might preserve
  only aggregate foreground. Decision/scoring thresholds remain unchanged.
  Median CPU-state spec SHA256
  `e02b425dfe10a660966f21941b21236b2ddb8955232bdde1eb7053279f3c34cd`.
- CPU-state attempt `20261016T161135Z-a8b70442`, gpu3 A6000, actual
  ready revision `58e9b2215203fbacffc1c8d5f59ea8bb04d3f18c`, failed the
  smoke before parity/full: upstream `video_tracking_multiplex._merge`
  casts the new CUDA tensor's dtype but omits its CPU destination device.
  The neighboring `_append` similarly concatenates without a device transfer.
  Add two guarded, in-memory compatibility replacements: transfer the new
  tensor to the destination tensor's device before assignment/concatenation.
  `_merge` retains its original dtype cast; `_append` retains native dtype
  promotion. No package/environment file is modified. The exact original
  expressions must occur once or installation fails. Save both unified
  diffs, original/revised source SHA256s and patch checksums before GPU
  propagation; include them in effective configuration identity and full
  validation. CPU-only regression reproduces mixed-device rejection and
  checks repaired values, dtype semantics and destination. The native
  mask/track parity gate remains mandatory, followed by the bounded median
  full stage. No candidate formula, score threshold or phase budget changes.
- Corrected CPU-state pilot `20261016T161909Z-5e827d31`, gpu3 A6000,
  actual ready revision `079bf6186f283ca1088817bee0016633481c8a1b`:
  smoke passes fleet validation and full median processing starts. Native
  versus CPU-storage parity: both minimum union and per-instance IoU 1.0,
  identical track-ID sets, 132 native instances. Smoke production records
  3/3 initialized tracker states and 1,125/1,125 cached masks on CPU.
  Compatibility-patch set SHA256
  `1692bdc34ece9f1dc48ec7b6b9354f97c69ba6f282c3648477651a0588f41019`.
  Native package archive remains immutable. No full result is cited before
  local validation. Local full suite after this correction: 575 passed,
  19 skipped; ruff and configured mypy pass.
- Preparing maximum-length confirmation only after local median validation:
  the shared A6000's observed propagation pace does not establish a ≤45 min
  all-prompt bound for 3,975 frames. Prefer the verified Ada class for that
  pilot, checking all hosts and claims again. Add an operational (not scoring)
  smoke scale gate: separately time imports, actual production preparation,
  SAM loading, all-prompt propagation, and CPU export. Exclude the native
  parity control; scale per-frame work to the full clip with 25% margin,
  retain measured fixed load/import costs, add 60 s fixed reserve. Save the
  estimate and its components; validator recomputes it and blocks promotion
  if it exceeds an explicitly supplied full allowance (at most 2,700 s).
  This changes neither masks nor the frozen identification rule. A passing
  estimate only permits the bounded maximum pilot, whose complete output
  still needs local validation and a feasible remaining-phase budget before
  any corpus run. All phase ceilings and deadlines remain unchanged.
- Median full `20261016T161909Z-5e827d31` is now **locally validated**:
  all 22 checks true, 407/407 native frames, recomputed scalar features and
  scoring masks, actual CPU states/cache and checksummed compatibility
  receipts. Full peak 21.5914 GiB; full execution 638.061 s; complete GPU
  claim 850.956 s. Receipt SHA256
  `f15fea18e714481826c327208e30bb011a92fbcfa0f9ddfda48cecf6cc597bc4`.
  Total completed phase claims 2,306.386 s; no other phase job is running.
  Full corpus and maximum remain unsubmitted at this point.
- Interruption: after median completion, automatic approval review could
  not run further read-only inspections because its account usage limit
  was exhausted. No command rejected by that review executed. After the
  stated reset time, the authorized fleet watch succeeded, confirmed all
  jobs terminal, and local validation resumed. The idle monitor was paused
  and the completion event acknowledged; no job was replayed. Absolute
  deadline `2026-10-17T01:00:00Z` and 8 GPU-hour ceiling remain unchanged.
- Maximum pilot specification file `features-v1/maximum-ada.json`, file
  SHA256 `8089be02891006995b91f314dbdc70df91672028b835c874e0b57b8f5a15fe51`:
  original 600/2,700/300 s stage/validation limits and 3,600 s reservation,
  Ada filter only, every host checked. gpu5's two Ada GPUs had 28,521 MiB
  free each and no PointStream claims; gpu5/gpu6 had 504/439 GiB available
  CPU RAM. GPU time reservation plus completed usage is 5,906.386 s,
  below the unchanged 28,800 s ceiling. The smoke forecast separates YOLO
  loading and the first eight-frame prediction batch from subsequent
  steady work, so one-time warmup is not multiplied by clip length. A
  regression checks this distinction and rejects a prefix without any
  post-warmup pose batch. The 25% margin and 60 s reserve stay fixed.
- Maximum pilot `20261016T200442Z-17bfb750` (recorded at takeover; the
  fleet's own record): gpu5 RTX 6000 Ada, shared, revision `6754141`
  from `ready.json`. Its 48-frame smoke passed every check but its own
  scale gate: forecast 5,348 s for the 3,975-frame clip against the
  2,700 s allowance (SAM 43.8 s for 48 frames with three prompts, peak
  21.46 GiB), so it was not promoted. Not evidence. Phase claims so far
  about 2,460 s of the 8 GPU-hour ceiling.
- Takeover (2026-10-20). Codex stopped here; a Claude session continues
  on `claude/k4b-players` (from `codex/k4b-players`, `origin/main`
  merged). Everything above stays as recorded.
- **Amendment 1 (user, 2026-10-20): features over a bounded window per
  marked frame, not the whole clip.** Written before any windowed run.
  - *Window.* 240 frames, K4's window length (K4 measured the first 240
    frames of each clip, and amendment 2 applies the chosen rule to those
    windows). One window per marked frame `f` of a clip of `n` frames:
    start `min(max(f − 120, 0), max(n − 240, 0))`, length `min(240, n)`.
    48 windows, 10,444 frames instead of 30,825; the 10 clips shorter
    than 240 frames are taken whole. Placement reads only the marked
    frame's index, chosen by hash before marking, and the clip's length,
    so it is content-blind and the same for every candidate. Every frame
    of the window is processed; nothing is subsampled.
  - *Why the candidates measure the same thing.* Each candidate is a fixed
    function of the frames it is given: presence on at least half of
    them, motion and contact summed over them. Amendment 2 will give it
    K4's 240-frame windows, so scoring it on 240-frame windows measures
    the rule as it will be applied. The whole-clip features (up to 3,975
    frames) measured an input the rule never sees in K4. Candidate (b)'s
    "last second discounted" becomes the window's last second, the only
    end a rule sees on a K4 window; its frozen formula already zeroes the
    last second of its input, so `k4b_rules` is unchanged. The frozen
    formulas (`RULES`, revision `466860d`), the duplicate-prompt merge,
    the injective IoU ≥ 0.5 scoring, the 0.95 bar, the marks and the
    deferral of (d) and its combinations to K8 are unchanged.
  - *SAM runs natively.* 240 frames is what K4 ran with no offload (three
    prompts, offline bidirectional; peaks 23.9–25.9 GiB under a 32,000
    MiB declaration on A6000). This path therefore drops video and
    tracker-state offload, the tracker-state adapter, the two in-memory
    replacements of upstream SAM functions, the grounding-batch control
    and the scale forecast (they stay in git history, `6754141`). SAM
    runs at its defaults (grounding batch 16), as in K4; the
    expandable-segment allocator setting stays (it changes storage, not
    results). YOLO11x-pose (1280) runs on every window frame first and is
    released before SAM loads, as before.
  - *Jobs.* One run per part; SAM and YOLO load once per part; each
    window's masks, poses and features are published as soon as it ends,
    so a stop keeps the finished windows. The smoke is the part's first
    window whole (240 frames), so it measures the real peak; the full
    stage runs every window of the part. Pilot: three 240-frame windows,
    one per source format (`djokovic_zverev/004@381`, 2160p60;
    `djokovic_federer/069@305`, 2160p50; `alcaraz_hurkacz/006@387`,
    1080p25). Its windows count in the score (same code and
    specification apart from the selection, as in K4). Promotion: the
    pilot's full output validates locally, its peak is within the
    declaration, and its rate puts each remaining part under 2,700 s.
    Then the other 45 windows in parts sized from that rate, in parallel
    on compatible GPUs.
  - *Validator.* Window placement recomputed from the manifest; every
    window frame masked and posed; SAM pinned, flash attention, default
    configuration (no offload, batch 16); person masks found; features and
    target masks recomputed from the saved masks and poses; frozen rules.
  - *Budget.* About 10,444 frames at the maximum smoke's 0.9 s per frame:
    2.6 GPU-hours of SAM, plus loading and pose. Ceiling 4 GPU-hours and
    6 h wall for this stage, within the 8 GPU-hour phase ceiling with
    what is spent. Shared mode, Ada and A6000 only, 32,000 MiB declared
    with a 4,096 MiB margin; every stage ≤ 45 min.
  - *Pilot* `20261020T184104Z-0af937d9`: gpu6 RTX 6000 Ada, shared
    (`timing_contaminated`), revision `fa7a4fb` from `ready.json`; specs
    and plan in `features-v2/` (`pilot.json` `ae7d56e7…`, `windows.json`
    `dc0e83ed…`), manifest `features-v1/manifest.json` (`4db66511…`). The
    smoke (one whole window) passed the fleet validator; the full stage
    (3 windows, 720 frames) passes all 18 checks locally. SAM native
    (flash attention, batch 16, default configuration); peak 27.3 GiB
    against the 31.25 GiB declaration; SAM 95–107 s per window and pose
    plus JPEGs about 21 s (shared timings, not evidence). Promoted: six
    parts of 7–8 windows (1,564–1,660 frames, about 16 min each at the
    pilot's rate), split longest-first by `tools/datasets/k4b_feature_inputs.py`.
  - *Parts*, all at `fa7a4fb` (`ready.json`). Part 3
    `20261020T185418Z-4dfa86ac` (8 windows, 1,612 frames, peak 27.8 GiB)
    and part 6 `20261020T185438Z-129dc1a3` (7 windows, 1,660 frames, peak
    27.3 GiB) completed on gpu6 RTX 6000 Ada; both full stages pass all
    18 checks locally. Parts 1, 2, 4 and 5 (`…185405Z-f78cf558`,
    `…185411Z-28392fd4`, `…185424Z-4a3e41d9`, `…185430Z-46b3f755`) were
    admitted one after another on gpu2 (A6000) and each ran out of
    physical memory in its smoke: another user's process grew to 34.4
    GiB after admission, leaving about 13 GiB (our allocator cap was 31.25
    GiB). Contention, not evidence. Resubmitted with the same
    specification apart from the host pool (gpu2 left out while that
    process holds 35 GB on both its GPUs): `part{1,2,4,5}-retry.json`,
    jobs `20261020T214345Z-b3c719eb`, `…214351Z-9fe1e191`,
    `…214357Z-3cf9965f`, `…214404Z-c4775c13` at `0477bfb` (code as
    `fa7a4fb`; only this entry changed). All four completed (gpu5 and
    gpu6, RTX 6000 Ada); each full stage passes all 18 checks locally.
    Every one of the 48 windows (10,444 frames) is now covered by a
    locally validated full stage; peaks 27.3–27.8 GiB.
- **Outcome (2026-10-21), amendment 1, the user's 48 marks** (37
  two-player, 10 one-player, 1 no-player frames). Jobs: pilot
  `20261020T184104Z-0af937d9` and parts 3, 6 at `fa7a4fb`; parts 1, 2,
  4, 5 (retries) at `0477bfb`, all from `ready.json`. Report
  `features-v2/score-2026-10-20/k4b-report.json` (`2a9a26fd…`), index
  `index-full.json` (`10e61aa5…`), local validation receipts beside
  them; marks `05f51305…`, review `instances.json` `89b78156…` and masks
  `published.tar` `2dc9acb6…` as recorded above.

  | candidate | correct / 48 | share |
  |---|---:|---:|
  | (a) motion in body heights | 30 | 0.625 |
  | (b) horizontal motion, window's last second discounted | 39 | 0.813 |
  | (c) racket contact on most frames | 32 | 0.667 |
  | (e) keypoint motion | 26 | 0.542 |
  | (a)+(c) | 42 | **0.875** |
  | (d), (c)+(d), (a)+(d) | pending K8 | |

  The window features' proposal oracle is 48/48: on every marked frame
  the window's SAM masks contain each marked player (IoU ≥ 0.5), so the
  misses are the rules' choices, not missing masks.
- Decision (by the rule): **no candidate reaches 0.95**; nothing is
  adopted and K4 is not amended. The table goes to the user. (d) and its
  combinations are pending: K8's v0 court list covers `nadal_kyrgios`
  only, and none of the marked frames come from it, so they wait for
  K8's homographies on the five marked matches.
- Reading (not a test; the rules stay frozen). (a)+(c)'s six misses: on
  five two-player frames it picks one marked player (IoU 0.97–0.98) and
  another person as the second, or on one frame two other people; the
  sixth is the one frame with no player visible, which any rule that picks
  tracks fails. (b) is right on 2 of those 6 and (a)+(c) on 5 of (b)'s
  9 misses (44 of 48 frames have one of the two right). Misses
  concentrate on rublev_auger-aliassime and djokovic_federer. The
  pattern, a second person mistaken for the far player, is what the
  court outline (d) is meant to remove, and the no-player frame is a
  shot K8 should drop. Hypothesis "(d) or (c)+(d) wins" stays open.
- Budget: 7,209 s of stage execution (2.0 GPU-hours: pilot, six parts
  and the four failed smokes, from each stage's `execution.json`), inside
  the 4-hour ceiling; about 4.5 h wall, most of it queueing for a GPU
  with 36 GB free.
- (d) and its combinations, defined before any (d) score is computed
  (2026-10-21; K8b adopted, homographies for the five matches:
  `k8b-2026-10-20/review/k8b-report.json` `8230bf1a…3314` for
  alcaraz_hurkacz, djokovic_zverev and medvedev_struff,
  `k8b-2026-10-20/k4b-matches/registration/k8b-report.json` `eb7b34b9…e3e0`
  for djokovic_federer and rublev_auger-aliassime; model metres, x across
  the court 0–10.97, y along it 0–23.77, to 1080p pixels). Frozen as
  `k4b_court.RULES_D` (version 1), on the same windows, canonical tracks
  and SAM masks as the other candidates, CPU only:
  - A track-frame is *registered* when K8's entry for that clip frame
    (window start + t) has `fits` true (error ≤ 6 px); other frames are
    not read. The *foot point* is the mask's lowest row at its centroid's
    x, at 1080p; it maps through the inverse homography to the court
    plane. *Inside*: x in [−1, 11.97] m and y in [−4, 27.77] m (the doubles
    court, 1 m beyond each sideline and 4 m behind each baseline; ITF's
    minimum run-off is 3.66 m at the sides and 6.40 m at the ends, where
    line judges and ball kids stand). A point behind the camera is outside.
  - Feature d: the share of the track's registered frames that are
    inside (0 with no registered frame); *inside on most*: d > 0.5.
  - (d): eligible tracks (presence ≥ 0.5, as every candidate) inside on
    most, by d, ties by track key (as v1); fewer than two qualify, fewer
    picks, as (c). (c)+(d): eligible tracks by inside on most, then racket
    on most, then d, then the racket share c. (a)+(d): by inside on most,
    then (a).
  - Scored as before (injective IoU ≥ 0.5 on the marked frame; bar 0.95;
    ties to the simpler rule: a, b, c, d, e, then (a)+(c), (c)+(d), (a)+(d)).
    Two tables: all 48 marked frames, and the marked frames K8 registers
    (court views, as this entry's data rule asks once K8 is adopted).
    The court-view table decides; the all-frames table is reported beside
    it.
- Outcome with (d) (`experiments.tennis.k4b_court`, CPU, local; the
  validated features of the seven jobs above; `features-v2/court-score-2026-10-21/`
  `k4b-court-report.json` `d593c879…`). K8 registers 37 of the 48 marked frames as
  court views (36 with two players, 1 with one); the 11 others are the
  no-player frame, five medvedev_struff one-player close-ups and five
  rublev_auger-aliassime shots (003, 006, 054, 062 with one player, 063
  with two).

  | candidate | court views (37) | all frames (48) |
  |---|---:|---:|
  | (a) | 23 (0.622) | 30 (0.625) |
  | (b) | 31 (0.838) | 39 (0.813) |
  | (c) | 25 (0.676) | 32 (0.667) |
  | (d) inside the court | 34 (0.919) | 36 (0.750) |
  | (e) | 19 (0.514) | 26 (0.542) |
  | (a)+(c) | 33 (0.892) | 42 (0.875) |
  | **(c)+(d)** | **36 (0.973)** | 45 (0.938) |
  | (a)+(d) | 35 (0.946) | 42 (0.875) |

  The v1 candidates' all-frames counts reproduce the table above exactly.
- Decision (by the rule): **(c)+(d) is adopted** (0.973 ≥ 0.95 on court
  views; no other candidate reaches the bar). The hypothesis held. Its
  application to K4's windows is amendment 2 (below, not yet run).
- Reading. (c)+(d)'s one court-view miss, rublev_auger-aliassime/002@422:
  only one track's foot point falls inside on most registered frames, so
  the second pick is another person. (d) alone misses it and two windows
  where a marked player stands inside on 34–42% of frames
  (rublev_auger-aliassime/040@363, 048@481); the racket rule recovers those
  two. On all 48 frames (c)+(d) misses three: that frame, the no-player
  frame and rublev_auger-aliassime/063@337, a two-player shot K8 does not
  register, so no track is inside and the racket fills the picks. Both
  are shots K8 sends to the pixel coder, where no identification is
  needed. Scope: 37 court-view frames from five matches; the margins
  (1 m, 4 m) were fixed from ITF's run-off before scoring and not tuned.
- Amendment 2 (to K4, not yet run): K4's broadcast windows need K4b's
  window features (SAM 3.1 with the person prompt; K4 kept no masks) and
  K8b's registration, which covers all 10 of K4's broadcast clips
  (medvedev_struff/042, the outro card, and rublev_auger-aliassime/038
  register on no frame, rublev_auger-aliassime/052 on 32 of 240); K4's
  players, bits and fit are then recomputed with (c)+(d)'s players, with
  amendment 1's and the first definition's numbers kept. TrackNet has no
  court registration (other courts, K8 not run there), so (d) cannot be
  applied to it as defined.
  Folded into K6 (2026-10-21; the user left the order to the session):
  after K4b.2 and K4b.3 every K6 gate condition passes whatever K4's two
  mislabelled windows hold, so a standalone re-run decides nothing; K6
  includes K4's 9 broadcast windows and reports K4's numbers with the
  adopted players there (PLAN K4b, K6).

### 2026-10-21 — K4b.2: pose on the far player (crop-size sweep)
- Question (PLAN K4b, item 2). K4's (b) failed on the far player alone:
  YOLO11x-pose on the whole frame at 1280 px fits players under 10% of
  the frame height on 0–50% of frames and those 15% and taller on
  97–100%. Does a top-down estimate on each player's crop, enlarged, fit
  the far player, and at which crop size?
- Players: identified by hand, not by a rule, so this does not wait for
  K4b.1's decision or K8. K4b.1's 48 windows (240 frames per marked
  frame, 10,444 frames, the five broadcast matches) already carry SAM 3.1
  masks and canonical tracks (`publish/<window>/sam-masks.rle`,
  `features.json`, from the seven locally validated jobs). On each
  window's marked frame, each player the user marked takes the canonical
  track whose mask matches it (IoU ≥ 0.5, injective; 48/48 match, K4b.1's
  proposal oracle), and that track is the player on every frame of the
  window (its person mask where both prompts have one, as K4b.1). The
  no-player frame's window has no players. *Far player*: in a window with
  two marked players, the one with the smaller median mask height; the
  other is the *near player*.
- Settings (YOLO11x-pose, ultralytics 8.4.6, confidence 0.25, as K4).
  *Whole frame*: K4's setting (1280 px), read from K4b.1's published
  detections; no GPU. *Crop ×s*, s = 1, 2, 4: the player's mask box at
  1920×1080, grown by 20% of its height on every side and clipped to the
  frame, resized by s (bicubic; s = 1 untouched), its long side capped at
  1280 px (K4's input; only near players reach it), padded to a multiple
  of 32 and run at that size, so ultralytics does not rescale it.
  Keypoints map back to the frame.
- Measures, per setting and player-frame, as K4 (`k4.pose_analysis`'s
  definitions): *fitted* when a detection has ≥ 0.5 of its box inside the
  mask's box (H1's matching; for a crop, only that crop's detections);
  *inside*: share of confident keypoints (≥ 0.5) inside the mask dilated
  by 8 px. Reported for all player-frames, the far and near players, and
  K4's height bins (6–10%, 10–15%, 15–25%, ≥ 25% of the frame height;
  under 6% beside them). *Synthetic far player* (fit says nothing of
  accuracy): every 8th frame of each near player whose median height is
  ≥ 15% of the frame and whose ×1 crop is fitted, reduced (area filter)
  to the same window's far player's median height, then run through each
  crop setting; OKS (COCO sigmas, the native mask's area) against the
  near player's own ×1 keypoints confident ≥ 0.5; unfitted counts as
  OKS 0. *Latency*: GPU milliseconds per crop at each setting (batch 16,
  synchronized), the whole frame beside it. Raw keypoints are saved (K4b.3
  reads them).
- Decision rule (fixed before any run). A crop setting *passes* when, on
  the 48 windows' hand-identified players: (1) fitted ≥ 0.90 of all
  player-frames and ≥ 0.90 of far-player frames; (2) median inside ≥ 0.9
  for both; (3) ≥ 0.90 of synthetic far players at OKS ≥ 0.5. The
  smallest passing s is chosen (cheapest). If one passes, PLAN K4b.2 is
  done: K4's (b) is re-read with it (here on hand-identified players; on
  K4's windows once K4b.1's identification is adopted), and K4b.3 runs on
  its keypoints. If none passes, stage 2 runs DWPose (`Models/DWPose`,
  `dw-ll_ucoco_384.onnx`, body 17 of COCO-WholeBody) on the same crops
  under the same three conditions, after an environment audit and a
  first-run smoke of its execution provider; then super-resolution of the
  crop; fine-tuning only if all fail.
- Hypothesis: whole frame fits about 0.5 of far-player frames and ≥ 0.97
  of near ones (K4); ×2 lifts the far player to ≥ 0.9 and ×4 adds nothing;
  near players are unchanged; inside stays 1.0; synthetic OKS ≥ 0.5 on
  most samples at ×2 and ×4, fewer at ×1.
- Competing explanations: (1) the far player is missed for occlusion or
  motion blur, not size: fit would not rise with s, and the height bins
  show it. (2) Enlarged crops give confident wrong keypoints: inside and
  the synthetic OKS check it; the reduction does not reproduce broadcast
  compression of a real far player, so OKS is an upper bound there. (3)
  A SAM track drifts to another person within the window: review images
  (the marked frame and one content-blind frame per window, every
  setting's skeleton on the crop) check it. (4) The crop's margin admits
  another person: the matching to the player's own mask decides, as K4.
- Budget: no SAM; one window conversion per window (K4b.1's, 83 s for
  all), YOLO on about 20,000 player crops per setting and the synthetic
  samples. Smoke: the part's first window (≤ 600 s). Pilot: K4b.1's three
  pilot windows, ≤ 45 min. Full: the other 45 windows in two parts of
  ≤ 45 min. Ada or A6000 (YOLO11x-pose's table rows; Ada asserted in
  K4b.1's jobs), shared, 12 GB declared. Ceiling 1 GPU-hour and 4 h wall.
- Jobs (all RTX 6000 Ada, gpu6, shared; environment
  `pointstream-20261009T214002Z`, ultralytics 8.4.6, YOLO11x-pose
  `013c4354…85f8`; inputs `pointstream-data/tennis/k4b-2026-10-16/pose-v1/`
  `inputs.tar` `3ae0106e…5e81`, built by `tools/datasets/k4b_pose_inputs.py`
  from K4b.1's seven validated jobs and the marks `05f51305…`). Pilot
  `20261021T090639Z-c2c32f37` (`1e9dc6b`; 3 windows, 720 frames; 177 s;
  `published.tar` `8bba8170…`); part 1 `20261021T091816Z-6a1e0668` (`c5ac3b5`;
  22 windows, 4,832 frames; 1,326 s; `552f3109…`); part 2
  `20261021T091821Z-1b022f25` (`c5ac3b5`; 23 windows, 4,892 frames; 1,245 s;
  `32e79fdd…`). `c5ac3b5` differs from `1e9dc6b` only in the validator's OKS
  comparison (within 1e-9: an ulp differs between the hosts' and macOS's
  numpy) and the `report` command. Peak 8.75 GiB of 12 GB. Every full output
  passes the validator's 9 checks locally (`pose-v1/report-2026-10-21/local-validation/`):
  each setting's fit and inside, and every synthetic OKS, recomputed from the
  published masks and keypoints. Both parts saw GPU contention from other
  users (action continue), so the latencies are indicative only.
- Outcome (`k4b_pose report`, 48 windows, 19,062 hand-identified
  player-frames: 8,860 far, 8,860 near, 1,342 in one-player windows;
  `k4b2-report.json` `bd87835b…91fb`):

  | setting | fitted, all | fitted, far | under 6% (113) | 6–10% (3,257) | 10–15% (3,609) | ≥ 15% (12,083) | synthetic OKS ≥ 0.5 (1,109) | median OKS |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | whole frame (K4) | 0.812 | 0.595 | 0.009 | 0.036 | 0.916 | 0.997 | | |
  | crop ×1 | **0.997** | **0.993** | 0.584 | 0.995 | 1.000 | 1.000 | 0.991 | 0.967 |
  | crop ×2 | 1.000 | 1.000 | 1.000 | 0.999 | 1.000 | 1.000 | 0.994 | 0.982 |
  | crop ×4 | 0.996 | 0.999 | 0.991 | 0.997 | 0.997 | 0.995 | 0.994 | 0.974 |

  Median keypoints inside the mask: 1.0 in every setting and group.
  Latency (shared, contended GPU): 21.6 ms per crop averaged over the three
  sizes, 39.6 ms per whole frame (median).
- Decision (by the rule): **crop ×1 passes** (all three sizes pass; the
  smallest is chosen). The far player is fitted: K4's (b) re-read on
  hand-identified players goes from 0.812 (whole frame) to 0.997 with
  median inside 1.0, above the 0.90 bar, so K6's gate condition (b) passes
  here; on K4's own windows it is re-read once K4b.1's identification is
  adopted. DWPose, super-resolution and fine-tuning are not needed.
  K4b.3 (keypoint rate) runs on the ×1 keypoints.
- Reading. The hypothesis held except that ×1 was already enough: the
  failure was the whole-frame input (1920 px shown to YOLO at 1280, so a
  60–100 px far player became 40–70 px), not the player's size in the
  source. ×2 adds the 113 player-frames under 6% of the height (0.58 →
  1.0). Competing explanation (1) does not hold (fit rises from 0.04 to
  0.99 with the crop alone in the 6–10% bin). (2), confident wrong
  keypoints: the synthetic far player agrees with its own native keypoints
  (median OKS 0.97–0.98), and on the far player 99.6% of ×1 detections are
  at least half the mask's height and 96.6% have ≥ 12 confident keypoints
  (whole frame 95.0%). The fit measure itself is loose on large players:
  enlarged crops of near players and of close-up players (one-player
  windows) get partial detections more often (near at ×4: 33% of detections
  under 0.7 of the mask's height, whole frame 8%; one-player windows at ×1
  25% against 0.5%), while K4's matching still counts them fitted; a
  review image (`rublev_auger-aliassime/002@422`, t = 175, ×4) shows a
  small skeleton inside the player's box. So the estimator to carry
  forward is ×1 crops for players under 15% of the height and either
  setting above (whole frame is as good there); K4b.3 and K6 read the
  keypoints, not this fit count. (3), track drift, and (4), others in the
  crop: the review images checked (the pilot's 12 and four from
  rublev_auger-aliassime) show the marked player each time, including with
  an umpire inside the crop. Scope: 48 broadcast windows from five
  matches; the synthetic check reduces near players without broadcast
  compression of a real far player, so its OKS is an upper bound.
- Budget: 2,905 s of stage execution (0.81 GPU-hours with the three
  smokes), inside the 1 GPU-hour ceiling; about 2 h wall from the rule.

### 2026-10-21 — K4b.3: keypoint rate against keypoint error
- Question (PLAN K4b, item 3). What does it cost to send the players'
  keypoints at each precision, and how far do the decoded keypoints move?
  K4 measured one point (9.4 kbps for two players, first differences at
  0.25 px, 22% of SVT-AV1's player bits at CRF 62) and kept no keypoints;
  K6 needs the curve to choose its operating point.
- Data: K4b.2's ×1 crop keypoints (the chosen setting), the 48 windows'
  hand-identified players, from the three locally validated K4b.2 stages
  (19,062 player-frames; 8,860 far, 8,860 near). CPU only, local.
- Codes (per player track, YOLO's 17 COCO keypoints, x and y): quantization
  step q = 1, 2, 4, 8 px at 1080p; prediction *previous* (first
  differences, K4's) or *constant velocity* (2·previous decoded −
  the one before; previous on a track's second frame); keypoint rate
  every frame, every 2nd or every 4th frame, the skipped frames decoded by
  linear interpolation between sent frames (held after the last). Closed
  loop: residuals are taken against decoded values. A frame where the
  player is not fitted ends the track; the next fitted frame starts a new
  one, its values sent absolutely (log2 of 1920/q and 1080/q bits). Per
  keypoint and sent frame, one visibility symbol (confidence ≥ 0.5). Rate:
  empirical entropy of the residual symbols and of the visibility symbols,
  each pooled over all windows per code (adaptive coding's optimistic
  bound, as K4), plus the absolute starts, in kbps for two players at each
  window's frame rate.
- Error, against the uncoded keypoints, over every fitted frame and the
  keypoints confident there: displacement in px at 1080p (median, p95) and
  in shares of the player's box height; OKS against the uncoded keypoints
  (COCO sigmas, the mask's area), for far and near players.
- Decision rule (fixed before computing). The code handed to K6 is the
  lowest-rate code whose median OKS against the uncoded keypoints is
  ≥ 0.95 for both far and near players (the coding error then sits well
  inside COCO's tolerance at the player's own scale); its rate is reported
  against K4's SVT-AV1 player bits on broadcast (41.9 kbps at CRF 62,
  155 kbps at CRF 48; K4's windows, not these, so a reference, not a
  test). The whole curve goes to K6, which judges the error on the
  rendering. K4's gate (c) is not re-read here: its bits come from other
  windows.
- Hypothesis: q = 2 px is free in error (YOLO's keypoints come at 1–2 px
  precision, K4) and costs a third less than 1 px; constant velocity saves
  little (K4: 27–67% of differences already zero) and loses on jitter;
  every 2nd frame passes at ≥ 50 fps; the chosen code costs 3–6 kbps for
  two players.
- Competing explanation: YOLO's frame-to-frame jitter, not motion, sets
  both the rate and the error floor of interpolation; the error at q = 1
  every frame against the uncoded keypoints, and the share of zero
  residuals, show it.
- Budget: CPU minutes, no fleet.
- Outcome (`experiments.tennis.k4b_rate`, local, 13 s; the three
  K4b.2 stages above, each locally validated; 24 codes;
  `pose-v1/rate-2026-10-21/k4b3-report.json` `89276994…`). kbps for two
  players (1.82 players per frame on average); OKS median and 5th
  percentile against the uncoded keypoints; displacement in px at 1080p:

  | code | kbps | far OKS (p05) | near OKS (p05) | px median / p95 |
  |---|---:|---:|---:|---:|
  | q 1, previous, every frame (K4's kind) | 15.0 | 0.998 (0.995) | 1.000 (0.999) | 0.40 / 0.60 |
  | q 2, velocity, every frame | 11.7 | 0.991 | 0.999 | 0.80 / 1.20 |
  | q 4, velocity, every frame | 9.5 | 0.963 (0.925) | 0.994 (0.986) | 1.59 / 2.40 |
  | q 2, velocity, every 2nd | 6.6 | 0.989 (0.913) | 0.998 (0.940) | 0.90 / 4.39 |
  | **q 4, velocity, every 4th** | **3.3** | 0.958 (0.813) | 0.988 (0.858) | 1.79 / 8.07 |
  | q 8, velocity, every 4th | 2.6 | 0.891 | 0.974 | 3.03 / 8.56 |

  18 of the 24 codes pass the bar; every q = 8 code fails on the far
  player. Constant velocity saves 2–8% over first differences at equal
  error. Absolute starts are 1–5% of the bits.
- Decision (by the rule): **q = 4 px, constant velocity, every 4th frame,
  3.3 kbps for two players** goes to K6 as the cheapest code within the
  bar, with the whole curve. Against K4's SVT-AV1 player bits on broadcast
  (other windows, a reference) it is 7.8% at CRF 62 and 2.1% at CRF 48;
  K4's 22% came from 9.4 kbps.
- Reading. The bar holds the median, not the tail: skipping frames
  costs the fast limbs. At every 4th frame the p95 error is 8 px (near
  player 9.5 px; players in one-player windows, mostly close-ups, 46 px),
  against 2.4 px for the same quantization every frame (9.5 kbps). So the
  frame rate is the lever whose error K6 must judge on the rendering:
  swinging arms are what a viewer watches. Hypothesis: q = 2 is not free
  but cheap in error (0.8 px median, 22% fewer bits than q = 1); velocity
  saves little, as expected; every 2nd frame passes; the chosen code is at
  the low end of 3–6 kbps. Competing explanation: jitter sets the floor
  at q = 1 (18% zero residuals, 4.2 bits per value; the crops give YOLO
  finer keypoints than K4's whole frames, 27–67% zeros there), so
  keypoints at full precision cost more here (15.0 kbps) than K4's 9.4.
  Scope: YOLO11x-pose's 17 keypoints on ×1 crops of hand-identified
  players in 48 broadcast windows; the rate is a pooled-entropy bound
  without a real entropy coder.
