# Experiments: K5d — expanded ball benchmark

### 2026-11-04 — registered preparation and engineering checks

- Goal: complete a real packet/receiver, matched-rate regional-fidelity and
  independently judged task study, including absent frames, beyond the
  48-positive-frame prototype. Main-paper contribution, with supporting controls
  and negative results in the appendix/supplement, within 23+5 pages.
- Hypothesis: spatially coded appearance/motion preserves ball-region fidelity
  and independent task recall more efficiently than spending all bytes on the
  native video. Competing explanations: bright-mark recognition, duplicate or
  misplaced balls, omitted framing, judge/tracker leakage and known-benchmark
  selection. Neither task recall nor crop metrics alone proves realism.
- Evidence population: all 30 full TrackNet games 8–10 clips, 5,675 frames:
  5,472 visible, 197 explicit out-of-frame, 6 occluded. All30 were scored in
  tracker selection; report retrospective benchmark exposure. Deterministic
  calibration uses game9/Clip5 and game10/Clip3. Each original frame appears
  once in the primary benchmark, including old nested prefixes. Source identities,
  label hashes and exact selection are in [k5d-inputs/study-selection-20261104.json](k5d-inputs/study-selection-20261104.json).
  RacketVision tennis test also has selection exposure and sparse labels;
  it does not supply an untouched dense confirmation set. OpenTTGames test
  videos are a possible cross-sport transfer set, subject to exact 240-frame
  context joins and source qualification; prior camera-motion use is disclosed.
- Frozen components: existing BallTrack encoder positions, existing per-frame
  fit/disk/streak primitives and codebooks; no training or post-test tuning.
  WASB receives output images alone, ground truth only enters the separate
  scorer. Its authors report tennis training on games 1–7 and test 8–10;
  optimizer logs were not independently audited. No documented table-tennis
  WASB checkpoint or training split is invented. A source-judge failure narrows
  the task claim; it does not invalidate correctly joined raw-image fidelity
  measurements. The fidelity route may proceed independently only after its
  own development correctness and complete-cost gates. No after-test judge switch.
- Primary fidelity report: crop DISTS and LPIPS PCHIP BD-rate against the pooled
  achievable native envelope, per clip and each native family, with actual
  transported byte counts. Full-frame PSNR and perceptual metrics remain beside
  the task result. Minimum overlapping spans fixed before inference: 0.005 for
  DISTS/LPIPS and 0.1 dB for PSNR; two distinct valid points are necessary, three
  initial settings per codec. Add a point only where support is missing. No
  extrapolation; disjoint curves yield supported-rate ordering or unresolved
  coverage, never a fabricated BD number. The old +1% full-frame PSNR BD guard
  remains failed on the original four windows and is evaluated/reported under
  that same contract; pointwise PSNR nondecrease is distinct.
- Primary task report: per-clip 10-pixel visible recall at supported equal rates;
  5-pixel recall, missed/wrong positions, trajectory discontinuity and explicitly
  absent-frame false detections beside it. Descriptive success for this new
  task endpoint requires at least 75% of all 30 registered clips to have qualified
  source inference and supported mean recall gain at least 0.10 over the native
  envelope at the two fixed low operating settings, DCVC-UF QP 0 +streak and
  SVT-AV1 CRF 62 +streak. Both actual candidate rates must be bracketed by the
  measured native support; their mean difference is the per-clip endpoint.
  Lack of support is not a pass. Pooled excess false detections must be no more than 0.05 per explicit
  absent frame, with all 197 negatives retained and per-clip counts shown. This
  is a prespecified benchmark criterion, not population significance or a
  retrospective pass of the old 6/8 whole-contract rule. Unknown/occluded labels
  never become negatives; report source qualification failure and exclusion
  coverage. Frame counts are not independent sample counts. Show three source
  game outcomes and per-clip results; no precise independent-match CI claim.
- Source feasibility gate: WASB recall at least 80% of visible source labels
  within 10 pixels on every qualified clip, all registered frame IDs preserved;
  failures stay in the coverage denominator, no replacement. Cross-sport source
  qualification uses the existing tennis WASB only; failing it stops that
  transfer claim without switching judges after seeing results.
- Controls: on calibration only, at one fixed rate compare native, billed disk
  and streak, and preregistered wrong-position/brightness stress marks with
  explicit diagnostic status and bytes where a coding arm is claimed. Privileged
  source-pixel/label-placement oracles are diagnostics, never deployable coding
  results. PSNR guards on coding arms remain unchanged; diagnostic stress marks
  cannot be relabeled as admissible codecs. A source-centered fixed 64-pixel
  crop is common to all arms on visible labels; independent full-frame judge
  never sees crop coordinates or the transport. Freeze exact control geometry,
  random seeds and reference crop implementation before the pilot outcome.
- Wire/receiver: PSBL v1 retains exact native IVF/PSDC and canonical BALL symbols,
  dimensions, constant rational fps/count, codec/resource ID, lengths and checksum.
  Every physical byte counts. Generic installed decoder/models are identified
  and reported separately, not amortized as a match-specific adaptation.
  The packet parser/render API is CPU-tested; actual native decoder parity,
  packet-only process input and source/label/saved-output file denials must pass
  on real components before source-hidden/full-receiver claims. New framing is
  a different convention from historical payload-only/SVT framing omissions.
- Budget: GPU 0 authorized for this new study. Proposal is part of
  [the staged research envelope](../cards/submission-research-2026-11-04.md),
  20 GPU-h /24 workload CPU-h total, including a 2 GPU-h /3 CPU-h initial feasibility release;
  no allocation is granted by this entry. Old K5c 463 GPU-s /1,413 CPU-s is closed.
  Cheap local engineering checks may use at most 120 CPU-s, zero GPU, under the
  current reversible code/preparation scope; they are no new quality result.
- Local engineering decision: preserve the six actual 48-frame native+BALL
  records read-only, repack/unpack PSBL and verify exact native hash, canonical
  symbols, frame geometry and total byte sum. No source-frame or model inference,
  re-encoding, new quality metric or statistical promotion. Local Homebrew FFmpeg
  is currently broken by a missing libx265.215 dependency; do not modify the
  personal installation or claim fresh native decoding from this check.
- Stops: bad identity/split/frame join, incomplete or nonfinite metrics, failed
  native or packet parity, source/judge exposure violation, denied-path leak,
  unsupported overlap, unfit all-in forecast or exhausted budget. Retain failed
  controls/cost gates and narrow the claim. New methodological choices after
  test results need a new prospective rule; no automatic training or brute force.
- Jobs: none; no new GPU request submitted. Register exact frozen source/spec/
  input/model/tool/device identities before admission and persist the ID before
  one bounded canonical fleet wait. Every waited stage within 45 minutes; split
  the 30-clip benchmark into independently published clip shards and stop when
  the decision is made.

### 2026-11-04 — actual saved-component wire check (CPU only)

- Code `85588f048b20427f32c2b67ca88e4335c343ab87`; command
  `python -m experiments.tennis.k5d_wire_audit` with the exact preserved K5c
  archive SHA256 `d40a9f7e0a7d2878d01010d4e128467bbdc174f9a899e007a555b9febe256d32`.
- PASS: all six actual native containers and BALL symbol payloads roundtrip
  exactly through PSBL v1, including declared 48-frame geometry, resource ID,
  checksum and full physical-byte sum. The native files are opaque retained
  bytes; no model, native decoder, source image or metric was run. This is an
  engineering framing result, not codec-quality or source-isolation evidence.
- Recorded process-time interval 0.020504 CPU-s, excluding import/startup. Reserve
  a conservative 1 CPU-s inside the local 120 CPU-s engineering ceiling. GPU 0.
  Original model/data/results remain untouched; the old 48 frames are not new
  statistical observations. Receipt:
  [wire-framing-audit-20261104.json](k5d-inputs/wire-framing-audit-20261104.json).
- Software preparation: PSBL parsing/render parity, equal-rate pooled-envelope
  analysis and all-frame denominator rules, real source-oracle CLI, and strict
  W6 gate have focused checks. Receiver-only CLI is prepared separately but
  has not decoded a real native stream; filesystem denial, full encoder/fit/
  metric orchestration and fleet admission remain incomplete. W8 diagnostic
  is per-condition preparation, not a submitted or fully admitted supervisor.
- No new GPU allocation or submission. The staged resource question is pending;
  inference remains blocked until explicit approval.

### 2026-11-04 — historical source evidence recovered before new inference

- Exact selection archive `20261020T214833Z-bf224666`, source `7590a774`, was
  recovered read-only. It contains all 30 source frame/label/prediction records.
  WASB-step1 records 5,051/5,472 visible hits at 10 pixels; 27/30 clips clear the 80%
  source screen. The failures are game10/Clip11 (25/39), game10/Clip8 (40/64),
  game9/Clip7 (69/87). All remain in registered coverage.
- A source-negative calibration screen was fixed in the root's audit brief
  before opening these recovered counts: at least 100 explicit negatives and
  false rate at most 0.05. Recorded source false detections 10/197=0.0507614
  fail that proposed screen. This is an instrument-calibration failure, not a
  negative codec-quality result or a revised historical rule. No threshold is
  relaxed and no alternative judge is selected after this outcome. Strong
  negative-frame/realism claims remain blocked. Relative detector counts may
  still be reported with this calibration and their label-based limitations.
- The raw-JPEG OpenCV loader/label/count functions are AST-identical to `7590a774`;
  Zoo's inference and transforms are unchanged, while its constructor added
  explicit candidate-name selection. Archive, weight and environment identities
  match the recorded family. This supports historical source screening and
  possible reuse after actual input/config/runtime binding; it does not prove
  exact current inference or parity to K5c's FFmpeg YUV420 source path. The new
  source-oracle explicitly uses raw JPEG, so these records must not be rejected
  solely because the different K5c contract used YUV. Actual decoded-image
  judging and source controls must share the declared new transform.
- The CRF 62 probe archive exists and was hashed, but retained JSON records only;
  its code removed native stream work directories. No bitstreams are recovered
  from the inspected archive. This search cannot prove absence elsewhere.
- Frozen source/weight/archive identities and per-clip counts are preserved in
  [source-evidence-reuse-20261104.json](k5d-inputs/source-evidence-reuse-20261104.json).
  No new source/model/codec run, GPU use or closed-budget transfer. Primary
  regional fidelity and byte-efficiency remain testable independently under
  their development/cost gates; the detector-based interpretation is narrower.

### 2026-11-04 — staged allocation approved

The user approved the [program card](../cards/submission-research-2026-11-04.md):
20 GPU-hours /24 workload CPU-hours inclusive, first 2 GPU-hours /3 CPU-hours for
feasibility. Old allocations remain closed. The new [budget ledger](submission-research-budget-20261104.json)
records 300 CPU-s administrative/pre-admission allocation, 0 GPU use, no reservations
or new jobs at this checkpoint. Root reviews actual module/validator/source/
input/runtime/device correctness and complete forecasts before submission.
No larger study is admitted by approval alone. No training or 200 GPU-hour full
pipeline run is included. Uncertain submissions are inspected, never repeated.

### Prospective calibration admission — approved phase

Root reviewed real six-point encoder, packet receiver, full-frame/fixed64px crop
metrics and native/zero-BALL/candidate byte ledgers. New frozen entrypoint
`k5d_fleet`: game10/Clip3,16-frame full-metrics smoke then188-frame fullclip.
Inclusive phase ceiling3300GPU-s/3600CPU-s within initialrelease; oneCPU,
exclusive qualified GPU, smoke600s/full2400s/campaign3300s. Full forecast uses
1.25 times measured smoke cost scaled188/16 plus60s and a further60s campaign
reserve. Failure stops promotion; no filesystem-isolation claim. Source ZIP
is a new non-overwriting hardlink (device58,inode37089793,2564111322bytes)
to the original TrackNet ZIP, not a copy/overwrite; staged verifier must check
the frozen full SHA. Source revision derives from authenticated fleet dispatch.
No inference or job ID at this freeze checkpoint.

Submitted calibration job **20261104T222053Z-0061cb27**, source c34ddd0290327a203d5bab38492fc1eeef42c7e11bcbfce3efccfd923f132233 from bf6208c5380f537429b4e49bed39e1c22cdb7b8f, spec b54c6986a0b343695709fa3b631d89859a162c3e0211c8bef08cfe6c79d370ff. Phase3300GPU/3600CPU; inspect canonical status before action, never replay.

Prospective judge-sensitivity controls frozen before opening calibration outcome: k5d-inputs/calibration-controls-contract-20261104.json; same188frameClip3 SVT62, exact retainednative/streak, same-track exposure0 disk withPSNRguard, diagnostic64pxhorizontalshift andwhite-streak colour. Fullimagejudges/registeredlabels only; no realismcertification, no tuning. Phase reserved600GPU/900CPU; no executionuntil realinputparity andcostfit.

**Calibration interrupted, not a quality negative.** Canonical request
20261104T222053Z-0061cb27 is terminal contended on gpu5; foreignGPU PID358756
triggered frozen stop policy. Its16frame raw-source screen completed
(WASB16/16 at5/10px; BallTrack14/16 at5px,15/16 at10px), but no complete
native/wire/metric record or cost promotion exists. Parent349596 andforeignPID
were both gone at recovery; no foreign process was signalled by this session.
Receipt k5d-inputs/calibration-contention-receipt-20261104.json preserves
status/staging/source identities. Charge conservative450GPU-s/650CPU-s
(inclusive, above259.4s oneCPU supervisor lifecycle), release unused reservation.
Rootprospectively reserves3300GPU/3600CPU for one fresh same-contract request
after fresh host/alljobchecks; this is not a replay of the stopped request.
Initial feasibility ceiling remains7200GPU/10800CPU, oldallocationsclosed.

Submitted fresh same-contract calibration **20261104T223611Z-ed28cf8e**, source 82effb7975c304090cdd748a90e0365be5fb7e02ee24ece3c0238e034f574948 from 237ae2145ad2a3b86e20e3899b4250a559b3a42b, spec b54c6986a0b343695709fa3b631d89859a162c3e0211c8bef08cfe6c79d370ff. Same frozen encoder asbf6208; concurrent benchmarkencoder edits excluded. Complete1577request canonical snapshot+all6host probes found bothgpu2A6000idle; gpu5busy/gpu6CPUheadroom0. Phase3300GPU/3600CPU, initialreleasebudgetstillfits; neverreplayold0061cb27.

**ed28coststop, valid16-frame fullpath.** Allsixpoints completedwithfull
metrics/judges/native+PSBL/zeroBALL identitiesandfreshpacketparity; nativeUF
sm80 onA6000, bothmodelsGPU, oneCPU, no contention. Pilotc4686467…fff12;
publishedtar dc1bc14133cf210589f2452785fbaf6d4905e91aae05fdc6d7ac0f25da8ec066.
208.447644s command ledto3121.575s conservative188projection>2400stagecap.
No fullpromotion. Independentselectedaudit at
k5d-inputs/calibration-ed28-smoke-independent-audit-20261104.json.
Charge450GPU/650CPU inclusive above370.388s lifecycle; totalCPUunmeasured.
Prospective nextphase1800GPU/2100CPU is a smaller16/48 sameclip timingpilot
onprovenbitidenticalnumericalpath; oldgate staysfailed, qualitysettingsunchanged.
Onlyafter two correctuncontaminatedframecounts andconservativecompletecost
fit may a fresh188confirmation be admitted; nocorpus/realismclaimfromsmokes.

Root admitted prospective16/48 cost instrumentation after focused correctness tests and exact ed28 contract review. No188/full30 promotion is authorized by this cost48-only receipt. Inclusive1800GPU/2100CPU reserved; host inspection00:05UTC found gpu2 idle A6000 capacity.

Fresh cost16/48 request **20261105T000550Z-d8b5c76b** submitted from cd669a22; immutable source receipt at `/private/tmp/pointstream-cost16-48-submit-20261105.json`. Spec SHA-256 acd2859c1aff5f2f118e262bf528afa90e4bea7612da68d7b6a5fead94e23a02. Inclusive1800GPU/2100CPU; deadline2026-11-05T00:45:21.789990+00:00. Re-read canonical status before acting; never replay.

### OpenTTGames CPU frame-index witness admission —2026-11-05

Root reviewed the pureCPU witness and six focused tests pass. Metadata links are new hardlinks matching source device/inode/size on gpu2, receipt `opentt-join-input-links-20261105.json`; originals are untouched. New bounded test2-only request uses the already reserved300CPU/0GPU (270s oneCPU fleet+30external incl preparation/audit), not seven-video or judge admission. Exact120Hz fullPTS, sequential/seek RGB parity and3labeled images required; any pass remains REVIEW_REQUIRED for independent image/coordinate verification. Full corpus contexts remain frozen.

FreshCPUjoin request **20261105T001308Z-4ae20579**, revision 73e5b8c63e7cf6c6db24e09e0eb1f808aee6e352, source d543b220b1653b09de29ee7e356633927c2202582ad8d88368536d6d7220c7ce, spec 221dc26a477efcebf6b99c895ae8ffe9e3d4c180ad2b9565bf136078a79e6a0d. Entire300CPU/0GPU, test2only; inspectcanonicalstatusbeforenextaction, neverreplay.

### Accepted cost review and prospective188 stage —2026-11-05

Independent cost audit `cost-d8b5-independent-audit-20261105.json` SHA66cbbd2ef1480524b0327effb9c760d7ef726abe40e338c3246c434bc9640d66 checks16exactoldcontract and48completepoint/frame/metric/judge receipts and filehashes. No independent full48 pixeldecode or judgeinference was repeated. Command205.769s/339.863s imply fixed138.722s plus4.190427s/frame; childrenCPU is cumulative, not summed. All1800GPU/2100CPU phase ceiling charged conservatively. Fresh188-only receipt389f3af98638f30f4d2aad6334f37aaa66392d2e71528137108253dbf2086d32 and larger2500GPU/2800CPU reservation authorize only one stagedconfirmation after real16 exactparity/cost gate; full1750/campaign2400≤40min, post-full audit required. The longest586-frame command projects2594s beforeoverhead, so full30awaits frozencheckpointparity/cost evidence. Prototypecheckpoint helpers are unqualified code, not experimental results.

### One table-tennis source-only admission —2026-11-05

Root reviewed original1085–1087 PNGs from CPUjob4ae20579: visible white ball agrees with zero-based labels (189,539),(154,536),(117,534);35/37pixel displacement distinguishes an offset. Reviewed join receipt7ae07b38… and evidence7fc2cdb1… were published by exclusivecreate at`Datasets/pointstream-data/tennis/k5d-20261105-inputs/` and rehashed. The earlier CPUjob exhausted its270s campaign after valid witness/validator, and no fullno-workmarker ran. Its gate reports 262.773s elapsed from campaign setup through smoke publication and validation, not command execution time; the smoke execution duration was not present in the locally retained status. The reviewed join receipt retains the earlier misattribution of 262.773s to the command and an unverified claim that it exceeded its 210s stage limit; this correction does not change that immutable receipt or the budget-exhausted outcome. Preserve that coststop; no model or source-qualification conclusion follows. Before GPUadmission a real CPU-only OpenCV/FFmpeg witness check matched all3 rawRGB hashes (auditb8e762f6…,1.622CPU-s inside prospective sourcephase external140CPU). Source worker independently repeats this before Zoo/model. Root admits onlytest2 context1066..1305,16timing smoke/full240 source-judge, fixedtennisWASB andall85visible/6absent/149unknown: recall10evalpx≥.8 andfalse-per-negative≤.05, no resampling120fps or GT/crops tojudge. Entire240GPU/360CPU withininitialrelease; complete measured≤120s full forecast required; no fullcross-sportcoding or other6videojoins. Eight focusedtests andfullRuffpass.

Fresh188confirmation request **20261105T003756Z-cc450baf**, revision b26e5b43702a21c37bcb4357223e0e6a25e1bec5, source c212f85b429f3a1abbca925de25736334fd06ffe9e821fac7798aaa99703fda7, spec 2d58571180db6b7d6e1685da78a36d7bc14a1f72ade5f7b56016ae631789bf6e. Entire2500GPU/2800CPU largerstage2 only; no full30release. Readcanonicalstatusbeforenextaction; neverreplay.

Fresh test2 source-only request **20261105T003930Z-a4067c30**, revision 5018210928e892e2029064615d3571544c55e37e, source d11fb244de97152d65f7a35f38095176f41fc560bd64c0da42aac9ccc1cf2d50, spec ab3fe7365fc37d57716cb053ce2a0c806d22763de7d202113b709537848228c7. Entire240GPU/360CPU, measured16timing gate thenfull240 onlyifforecastfits. No cross-sportcoding claim; inspectcanonicalstatusbeforenextaction, neverreplay.

Independent source audit: **20261105T003930Z-a4067c30** completed without a
mechanical error, but the frozen tennis WASB judge did not qualify on this one
OpenTT test_2 context: 0/85 visible hits within 10 evaluation pixels (required
≥80%), with 0/6 false positives (required ≤5%); 149 unknown frames are excluded.
The reviewed three-frame OpenCV/FFmpeg join agrees. This is a negative result for
this judge and context, not proof that all cross-sport approaches fail. The next
cheap test is a separately justified independent judge or source-correctness
check with frozen exposure and budget; no cross-sport coding or training is
admitted. [Independent audit](k5d-inputs/opentt-test2-source-a406-independent-audit-20261105.json).

### Checkpoint equivalence admission — 2026-11-05

The bounded phase and stop conditions are recorded in the research card. Preparation's complete native contexts and joint fit are unchanged; scoring replays existing packets and preserves whole-clip judge context. All results must match d8b5's full-schema 16/48 reports exactly. No corpus admission follows from code or CPU fixtures. The SHA-bound prospective prepare decomposition (`53b15df0…`) subtracts only 24 nonoverlapping outer scoring events from each accepted command: remaining times169.770/235.418s; predicted full48 with margin/reserve393.127s. Per-point startup is measured by its own smoke. Root reserves3,400GPU/4,350CPU under the approved larger-stage ceiling and will freeze an independent source checkout for every job in this phase.

Fresh checkpoint prepare16/48 request **20261105T005745Z-d72ef178**, frozenrevision ae7bc1c89532d589feac4dfd2379fbea5dee3dfb, source 7d38b2bd54a7a712ac5cbf653fab6ce164373bafba5ea58394cc868aa4f11469, spec e60b534424fc2c7521527b496ef67b9bc124937bf13ce34cde82fab8e083f45c. Frozen sourcecheckout `/private/tmp/pointstream-checkpoint-frozen-20261105` remainsunchanged forscore/assembly. Inclusiveprepare700GPU/900CPU. No corpusadmission; readcanonicalstatusbeforeaction andneverreplay.

### 188-frame Clip3 confirmation saved-artifact audit — 2026-11-05

Job **20261105T003756Z-cc450baf** completed on exclusive RTX A6000; full `published.tar` SHA-256 `622fd68d5380b97449a09872bc15402117dd9c3e7a2576a757cd6aff8fd090d7`. Independent saved-artifact audit [`confirmation188-cc450baf-independent-audit-20261105.json`](k5d-inputs/confirmation188-cc450baf-independent-audit-20261105.json), SHA-256 `c6a1f88e1971c70bd320a83f290aae190c3f05c438005e727b3907c45bbcbb34`, checks the purpose-limited 188-frame gate `389f3af9...`, frozen validator, six native and paired PSBL ledgers, recorded producer/receiver parity, 2,256 metric rows and 4,512 judge rows; it independently recomputes each judge count and pooled RGB PSNR from saved SSE. Source qualification passes on this clip: 188 visible labels, no absent or unknown labels; source WASB has 174/188 and BallTrack 183/188 hits within 10 pixels. No false-positive rate can be estimated here. Checkpoint training logs remain unaudited.

Across the six operating points, baseline versus streak 10-pixel WASB hits range 1–127 versus 160–177/188, and BallTrack 2–153 versus 182–183/188. Streak pooled RGB PSNR is 0.0076–0.0217 dB higher than its paired baseline; the four saved perceptual means also decrease at each point. These are descriptive paired operating-point results with added wire bytes, not a matched-rate or corpus result. The old 188-frame naive cost failure and historical four-window guard outcome are not converted into passes; three retrospective exclusions and the 10/197 negative-frame calibration failure remain. Nested 16/48-frame prefixes are not additional clips.

The recorded smoke and full commands took 203.65 s and 897.03 s, respectively; no contention was recorded, claims were released and both task PIDs and process group were absent on inspection. Validation inspects saved bytes, rows and parity receipts; it does not independently decode, inspect source pixels, run perceptual models or rerun judges, and it does not establish OS-level source denial. No full-corpus release follows from this confirmation.

Fresh prepare repair1 request **20261105T012016Z-7f12c20a**, frozenrevision 550e85898acfada12dace83a81ac1e48ae6266cb, source 2d8ce3c31460a745628910d22c20e2af158b5e89533f346f3a4a34c1c7fcffd9, spec da0d6e81c853e0164143ef5855b2cc42439c5765c186178cfed878f8fb567a1d. Frozencheckout `/private/tmp/pointstream-checkpoint-repair1-frozen-20261105` forallscores/assembly; oldfrozencheckout intact. Newprepare700GPU/900CPU; exact16parity andcostgatebefore48 remain. Inspectcanonicalstatusbeforenextaction; neverreplay.

## Fresh coalesced-source equivalence release (2026-11-05)

Independent implementation review accepted commit f7695b70 for a new bounded prepare16/48 and one all-six scoring job. This replaces six unlaunched point-score jobs; no prepared result is silently transferred across whole-source identities. Frozen checkout: `/private/tmp/pointstream-coalesced-frozen-20261105`. Preparation ceiling700 GPU/900 CPU seconds and subsequent scoring ceiling700 GPU/900 CPU seconds include assembly. Failed first preparation700/900 and successful repair700/900 are separately charged conservatively; cumulative2800/3600 is below prior4100/5250 approval. The new preparation uses the pinned measured cost gate, 200/420-second stages and650-second campaign. The new scorer requires actual fresh checkpoint archive hashes, exact16/48 equality with d8b5, and measured forecast before48. Full30 admission remains pending real parity and complete cost evidence. No per-point score job was submitted.

Fresh coalesced-source prepare submitted as `20261105T020925Z-97a4cf02`; source SHA256 `59844fa9daa863ff64f09e16e02a564454fa3c4598288cbdcd3dbc65aed24423`, canonical spec SHA256 `93a9923c782869e5cfc73cbe58cc8831c422f5fe2b1527d66ed72af1e4f73c85`; pending16/48 qualification.

Sensitivity controls: the previous 188-frame `calibration-controls-spec-20261105.json` is **SUPERSEDED** and rejected by the pilot worker. The unsubmitted [16/48-frame cost pilot](k5d-inputs/calibration-controls-pilot-20261105.md) uses the fixed five arms and whole-frame judges, with a 550-second campaign inside a 600 GPU-second / 700 CPU-second inclusive pilot allocation. A separate 188-frame run remains contingent on SHA-bound pilot cost evidence, full forecast and remaining budget; no new outcome-based selection is allowed.

Control16/48costpilot submitted as `20261105T021434Z-3ec159d8`; source `6840271ba7b4179a044184e7db5dea357c1702dd093f335d812f3f5fb077b15a`, spec `9f83edb1ec4db02ce7d9125c0b96fea4d1b4eefada8eacafddb3ac5d4f61efc5`, frozen48046314; inclusive600GPU700CPU, original188goal remainsconditional.

### 2026-11-05 — Measured prepare cost-cap revision

Job97a4cf02 completed196.731-second16frame preparation with exactnative/source/reconstruction parity and zero recordedcontention. Its449.206-second conservative fullforecast exceeded420, so48correctlydidnotrun. Revised frozenff60bb3d uses220/500-second stages and780campaign,850GPU1050CPUinclusive, retaining1.5slowdownmargin+40reserve+20remainingreserve. Three previous700/900prepareceilings charged separately; newprepare+futurecoalescedscore yields3650GPU4650CPU cumulative≤prior4100/5250cap. No replay or replacement of the oldfailedgate. Rootreviewedstrictspec500max and31focusedtestspassed beforedispatch.

Sized-source prepare job `20261105T021805Z-f518beaa`, source `c25d77329c6fe85be4396e51abb794ee6805736a8486b5061c9f8ebcaba7c4d6`, spec `b9726296d634425eed0467a716acbaac6708fcc59ccba616cfa829ecf674507c`; frozen `/private/tmp/pointstream-coalesced-sized-frozen-20261105`.

Same-source sixpointcoalescedscore job `20261105T023544Z-408e8453`, source `c25d77329c6fe85be4396e51abb794ee6805736a8486b5061c9f8ebcaba7c4d6`, spec `213c2248f72c660914edbb9a4051f4a835062d2511cee2a4959546ad346fb8f9`; frames16/48all6, budget700GPU900CPU, exactd8metrics/judgesandcurrentforecastrequired.

Finalcontrol16/188job `20261105T023619Z-0e2a610f`, source `c6bbb5ccd64555321997ecefe0cabc4aa32b1ef37e66740e0e944132e20d405f`, canonicalspec `38aefa3a49ae8e98ea6dc087adcc40b50b92c196e0a6d5ea4b6aeace0ee2c7f7`, frozen2cc35643; inclusive600GPU800CPU, samef438costreceipt andfivearms, no appearancequalificationorcorpusclaim.

Phase k5d-coalesced-permission-repair-20261105 job `20261105T025241Z-fe451f9e`, revision `7640ed34696ec93c1a61fd3666d29828eea38e46`, source `8a3b51c487cf9b456e304fdf24300f230ece3f3c7b2c8569ab6681fbd8166f13`, canonicalspec `fcfa508d37f0bc93c0213b868314cc925820dbea715774427dfa18a70573b47a`; inclusive850GPU/1050CPU seconds. No replay or implied scientificpass.

Standalone coalescedscore job `20261105T032433Z-1423f92a`, source `8a3b51c487cf9b456e304fdf24300f230ece3f3c7b2c8569ab6681fbd8166f13`, spec `c017b76ab52a281d1444b6e6b457d9987024a6996dc2bf85fb8ca7501a387d4e`, frozen7640ed34,gpu2-only/nootherregisteredactivejobatadmission; inclusive1200GPU1400CPU220/750/1100. Same producer identityfe451, all6native/metrics/judges and48cost gate pending.


### Accepted shared scoring and pending expanded release — 2026-11-05

Job1423f92a is complete, with all six16/48 native/wire/receiver/metric/judge arrays matching d8b5; saved-artifact audit `k5d-inputs/coalesced-score-1423-independent-audit-20261105.json` SHA09f7d8761c95c83b8c000cae9cc1f86d648e3ad415481b880b34240baee7421d. The whole2050GPU2450CPUprepare/scorephase is charged conservatively. WASB is independent of the encoder; BallTrack is encoder-model consistency. The actualcc450source audit confirms the encoder already uses automatic BallTrack coordinates, so the duplicate automatic-input proposal is superseded.

The proposedH3 CPU-readiness cost probe b23db6b8 was never published because source upload timed out. Its partial archive is preserved; no readiness timing or full30admission follows. The additional24GPUh/28CPUh proposal remains unapproved, and the original20/24programme ceiling remains authoritative. Exact actualsource bridge, readinesscost,292/586scalechecks and complete remainingforecast are still required before full30.

Fresh CPU-only readiness request `20261105T050105Z-e9104e5e`, actualH3fc8source d9825bd5, published with exactbufferedtransfer; gpu2CPU0,300CPUinclusive,noGPU. Priorb23unpublishedpreserved. Source/assets smokepassed; full586/checkpointcost notyetaccepted. Canonicalboundedwait600seconds, noproductionadmission.

Synthetic Linux CPU-accounting oracle `20261105T122208Z-a62ad658` submitted CPU-only to gpu2, pending; 100 workload CPU-seconds plus 80 external reserved within the original programme allocation. Frozen revision `b96e81bd021e2bb36fe78c4f461b4a8a4512830c`, source `8df50d964f7af8283c0d7d05695af6885e77f1dd667d09f9b7261577bca97060`, snapshot `7cca0cdd84cf2b145718ccfbeb451e55106d6ec8348a7800bf806772b3809108`, normalized spec `311e078b159bc6bce04ab0b908e2ac3dd9e49e4558934ce043c13c157a605170`; deadline 2026-11-05 12:56:18 UTC. The [card](../cards/k5d-linux-cpu-accounting-oracle-2026-11-05.md) specifies the decision and limits; no CUDA, native-tool teardown, source bridge or production admission follows from this synthetic run. Root owns the bounded wait; never replay an uncertain submission.

Synthetic accounting oracle `a62ad658` completed and passed independent saved-artifact audit ([receipt](k5d-inputs/linux-cpu-accounting-a62ad658-audit-20261105.json), SHA256 `0231a157b52e13f87431823c2817ea16d614a14cf3d41c6bcac698ee902c2478`): the direct, forked and adopted double-forked synthetic children were reaped, and both stages' separately sampled waited-child CPU agreed with the wrapper counter. Command CPU was 0.217693/0.212744 seconds; all recorded task PIDs and supervisor process group were absent and GPU2 CPU allocations empty on later inspection. Whole-lifecycle CPU is unmeasured, so the recommended conservative charge is the full180CPU/0GPU reservation. This is not interruption, escaped-session, native or GPU qualification and does not admit production.

Synthetic PID namespace teardown oracle `20261105T124406Z-a6149bf3` submitted CPU-only to gpu2, pending; 100 CPU-seconds campaign and 80 external reserved within the original programme ceiling, no GPU. Frozen revision `7bdb05ee436815f757926ff84c0a413f89098503`, source `1d29c0f3af8c33ce20ac6a7862aa7465cbbb4ee52568497687834c79ac77177d`, snapshot `d23d0620962a745036751d63372740648a1603f29257eefa37229e1f4aaf3c34`, normalized spec `c34f6e31ce28628084652f8fb5367aaab196cc94cc94defc19cf9d9966101a70`; deadline 2026-11-05 13:18:02 UTC. The [card](../cards/k5d-pid-namespace-teardown-oracle-2026-11-05.md) specifies the four synthetic exits, proof and limitations. Root owns the bounded wait; no automatic retry or production admission.

Synthetic PID namespace oracle `a6149bf3` **failed smoke**: the SIGTERM-case launcher did not exit within three seconds; its error path killed its task-owned launcher. The exact mechanism is unobserved. The earlier normal and timeout functions returned before SIGTERM but their complete proof was not published, and SIGKILL was not reached. Later inspection found all nine recorded namespace-member PIDs and the supervisor PID/process group absent, with no remaining gpu2 CPU allocation. Saved source/spec, log, partial scratch markers and limitations are in the [audit](k5d-inputs/linux-pid-teardown-a6149bf3-audit-20261105.json), SHA256 `c11056dba889f3db94f22ee626579e4bf6b96c15c585ea50305d07a2127cb9e0`. Recommend conservatively charging the entire180CPU/0GPU reservation; whole-lifecycle CPU is unmeasured. No retry, production use or native/CUDA conclusion.

Synthetic exec-preserved PID namespace escalation oracle `20261105T130124Z-684bc795` submitted CPU-only to gpu2, pending; 100 CPU-seconds campaign plus80 external reserved, zero GPU, within original allocation. Revision `aec74889709fe08f2c58abf97fe5873a787ad61e`, source `b3c215dc440a42c17ccecda3f29e1d435241705e4ac39efc295c95e22a902c84`, snapshot `af2ef759bf892c3d958048352200dd5fc5b21f35bc55fbeab33141b30580e050`, normalized spec `7df2cb24594be4d8544d12e0bd911f1c6f0ad3758c9215e5d22dcfdc974fd2a1`; deadline 2026-11-05 13:24:37 UTC. The [card](../cards/k5d-pid-escalation-wrapper-oracle-2026-11-05.md) records its decision rule and limitations. The earlier immediate-SIGTERM failure remains unchanged. Root owns bounded wait; no retry or production admission.

Synthetic exec-preserved PID namespace escalation oracle `684bc795` completed and passed independent saved-artifact audit ([receipt](k5d-inputs/linux-pid-escalation-684bc795-audit-20261105.json), SHA256 `f44597cd9b6c0fe887d99c1b65bf7638116401bdcbcc6cf40f5861d377ea9965`). Each stage validated normal exit, timeout/SIGTERM with 0.1-second grace and actual SIGKILL escalation, and direct SIGKILL, including a double-forked new-session member. Recorded member disappearance was 0.033–0.236 seconds; all32 recorded case PIDs and both supervisor PIDs were absent later, supervisor group absent and gpu2 CPU allocation empty. Stage elapsed0.926/0.988s is not a lifecycle CPU measurement; recommend full180CPU/0GPU reservation charge. This synthetic result does not qualify native or CUDA workloads, nested namespaces, or production; it does not undo `a6149bf3`'s immediate-SIGTERM failure.

Native CPU/PID proof request `20261105T224658Z-97705347` was **not published**: the client reported an unknown submission after its bounded source-archive transfer timed out. Canonical exact-ID status says unpublished or unknown; independent read-only inspection found only `events/` and a 7,158,849-byte partial `source.tar`, with no ready, status, dispatch or spec file. The reviewed full archive is 16,076,800 bytes. Preserve the partial upload; no worker, native, GPU or validator result is established, and no automatic retry is allowed. Candidate source checkout `/private/tmp/pointstream-k5d-native-frozen-20261105`, revision `279b7ff9c0ec412a3b774df4078d6352e811959c`, expected source tree `277546e02ada2e4a928a4b09b63f54e67dba56f88f4f0a8cadb5106c241e14ae`, expected snapshot `c7afd4bb806dc6f2f8ce5aea48fd4ba4b469ecf5f45c1748e740292893f4ee7b`, normalized spec `8fc07d732191dc199019043628dcad58192e911079c69af142ca775f35c0abfb`, deadline 2026-11-05 23:15 UTC. Root owns reconciliation of the 180 CPU-second inclusive reservation, which includes 20 CPU-seconds reserved and charged for metadata; no GPU was requested. This is not parity, containment, native or production evidence.
