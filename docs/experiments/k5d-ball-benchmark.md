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

Root reviewed original1085–1087 PNGs from CPUjob4ae20579: visible white ball agrees with zero-based labels (189,539),(154,536),(117,534);35/37pixel displacement distinguishes an offset. Reviewed join receipt7ae07b38… and evidence7fc2cdb1… were published by exclusivecreate at`Datasets/pointstream-data/tennis/k5d-20261105-inputs/` and rehashed. The earlier CPUjob exhausted its270s campaign after valid witness/validator; command262.773s exceeded210s requested stage limit, and no fullno-workmarker ran. Preserve that coststop; no model or source-qualification conclusion follows. Before GPUadmission a real CPU-only OpenCV/FFmpeg witness check matched all3 rawRGB hashes (auditb8e762f6…,1.622CPU-s inside prospective sourcephase external140CPU). Source worker independently repeats this before Zoo/model. Root admits onlytest2 context1066..1305,16timing smoke/full240 source-judge, fixedtennisWASB andall85visible/6absent/149unknown: recall10evalpx≥.8 andfalse-per-negative≤.05, no resampling120fps or GT/crops tojudge. Entire240GPU/360CPU withininitialrelease; complete measured≤120s full forecast required; no fullcross-sportcoding or other6videojoins. Eight focusedtests andfullRuffpass.
