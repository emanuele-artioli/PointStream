# Experiments: Chapter 14 player demo audit

### 2026-11-15 — Story website player comparison (read-only audit)

**Identity and decision.** The user identified Chapter 14 at `https://story-emanueleartioli.emanuartioli.workers.dev/` as the intended player demo. The root session inspected its browser UI and reported the videos `/media/pointstream-story-v3/hybrid.mp4` and `/media/pointstream-story-v3/base.mp4`, an 80-scored-frame W7 snapshot, 67 hash-verified displayed frame pairs, and original 18.46 vs 14.12 kbps labels. These website observations have not independently been tied by this audit to the bytes in W7 archives. The separately recorded local egocentric-hands audit concerns a different site and is not evidence about Chapter 14.

**Important accounting correction.** The 18.4614183 kbps historical hybrid value matches W7's original al028 judge, but W9 found its modeled 12-crop reference bank was underbilled fourfold: four alternative encodings per crop were mistakenly counted as four crops. The corrected two-hour modeled hybrid rate is **19.2398333 kbps**, not 18.46; see [W7 correction](w7-untouched.md#2026-10-31--reference-bank-billing-correction-supersedes-the-historical-closure), [W9 audit](w9-adapter-break-even.md#critical-correction-to-w7-reference-bank-billing-2026-10-31) and [corrected per-window result](w9-inputs/w7-tier1-billing-corrected.json). Original rendered frames and quality values are unchanged. At two hours the modeled correction is about +0.7784 kbps; the 12-crop reuse assumption and omitted operational framing remain unverified. The 14.12 kbps base and corrected hybrid are different-rate points; their direct visual comparison does not establish an equal-rate win or BD-rate. Do not call this a complete operational bitrate.

**Source and pipeline.** W7's exact matched score identifies `al028` as `alcaraz_hurkacz/028@120`; the selection records source `alcaraz_hurkacz/028`, SHA256 `c51dace59349914da401356984de4e83da658c878011db3f54ff233077152934`, 25 fps, 240 native frames, 9.6 seconds. W7 selects 80 every-third-frame generation/scoring ticks (8⅓ fps); the `@120` identifies the prepared window row, while the selection entry records start 0 within source clip 028. The score alone does not authenticate the site's exact assets. The frozen W5 recipe uses a quarter-scale SVT-AV1 base, KPS2 pose coordinates, per-match 4-bit adapter, modeled 12-image reference bank, receiver pose-prompted SAM masks and generated player composites. The base generator is AnimateAnyone `original` with a tennis rank-64 LoRA and an Al match-specific adapter `ah-C`; see [W7 rule and first-window record](w7-untouched.md#2026-10-30--w7-the-foreground-first-claim-on-untouched-footage-rule). This is generated players rather than a demonstrated RealESRGAN stage. Searches for RealESRGAN in `experiments/tennis` and `src/pointstream` found none; that does not exclude another component or an uninspected website export process.

The native base job is `20261030T142923Z-f6699fe5`, code `aaf6ae4a`, archive SHA256 `9744754d6acd62f7d872e804b5af09b00de1187f100062f9369e526b3e285239`. Generation `20261030T150651Z-a58b2ce2` ran on RTX 6000 Ada with two player units, archive SHA256 `d88f1c79cd78f292dc70105540cde4b659a4726138d34de02f343b4bb9edbd91`. Rate transport `20261030T150852Z-d4e7cd41` archive SHA256 `8b0871ab633edcf4c77cb810cf2d3617a2a9593f0aa51af451c2a2ec21e30607`. Judge `20261030T185245Z-e51a4dec` ran on Ada, archive SHA256 `cbc26962935b3542b0b7cf77f724010de19323fedd6fe02cb822ec6e43e75837`. The judge's controls confirmed the no-player composite equals the paid base and repeat plain-base scoring agrees.

**Observed metrics, not preference.** Original judge player-region DISTS was 0.4961 for the 14.12 kbps paid base and 0.2420 for the hybrid at its original 18.46 kbps modeled rate; whole-frame DISTS was 0.2002 and 0.1750 respectively. Player-region LPIPS was 0.7180 and 0.3042. These are different-rate, one-window results on 80 sampled frames. Correcting rate does not change their pixels or scores. W7's subsequent same-rate interpolated analyses use independently scored anchor families and have coverage caveats; the corrected results and later repairs must be consulted rather than the superseded historical pooled verdict. The corrected al028 result initially had only DCVC bracketed; see W9 for later repairs. No human preference study was identified in the inspected W7 and W6 records. Separate W6 found greater temporal flicker than SVT in its own full-rate experiment; it is a related limitation, not a measurement of this particular website clip.

**Next step.** Authenticate the Chapter 14 export: identify site source/revision, SHA256 of the served MP4s, export command, mapping of the 67 displayed pairs to the judge archive's 80 timestamps and exact source hashes. Locate existing decoded native anchors at a genuinely comparable corrected total rate and an authenticated reference; verify alignment and presentation before a blinded randomized preference pilot using saved outputs. The current two served previews can support only a clearly labeled exploratory preference on these different-rate displayed pictures. No new encoding, GPU job, site write, media mutation or human measurement was performed in this audit.

### 2026-11-15 — Public manifest identity join and corrected matched comparison

ROOT inspected Chapter14 in the browser, including court close-up, base versus
generated players, whole-picture/player DISTS and bank-reuse controls. The
visible player reconstruction is substantially clearer in the inspected
snapshot; this is an informal inspection, not a preference measurement.
The source-linked [public measurement manifest](https://story-emanueleartioli.emanuartioli.workers.dev/data/tennis-media-v3.json)
was retrieved as metadata only: 233,724 bytes, SHA256
`4769759f48441472619b49990bf595e860eb37914efd160f46197985ab2f4d7a`.
The [retained identity receipt](submission-ladder-chapter14-web-manifest-20261115.json)
records its exact W7 source, judge revision, GPU, archive/summary hashes,
display hashes claimed by the exporter and the original metric values.

The manifest binds the site to judge `e51a4dec` and archive `cbc269...`, with
the same source SHA and exact quality means as the saved W7 records. It declares
67 displayed ticks at native indices0,3,...,198 (8.04seconds) and80 judged ticks
over the full9.6seconds. The display proxies are lossy1280×720 AVC, while the
research scores use1920×1080. This verifies the metadata join, not an independent
hash/download/reassembly of all served video frames. No remote video was downloaded.

The repaired [W9 two-hour comparison](w9-inputs/w7-repaired-2h.json) supplies the
rate-controlled context absent from the website's base/hybrid illustration:

| al028 output | Modeled kbps | Player LPIPS | Player DISTS | Full LPIPS | Full DISTS |
|---|---:|---:|---:|---:|---:|
| W7 actual decoded base |14.1208|0.717960|0.496144|0.293668|0.200172|
| W7 hybrid, W9 corrected price |19.2398|0.304181|0.242025|0.284529|0.175013|
| Best bracketed anchor (DCVC), interpolated at corrected price |19.2398|0.384051|0.302989|0.145873|0.120687|

Lower is better. The player improvement is already strongly reflected in
LPIPS and DISTS, including against the bracketed same-modeled-rate anchor.
The hybrid loses to that anchor on full-frame similarity. These are pointwise
quality comparisons, not BD-rate reductions, a physically encoded exact-rate
point or a human-preference result. They do not establish the new70/30 weighted
objective. Every-third-frame generation/scoring does not qualify all25fps frames
or temporal stability. The model, same-match adaptation, reference reuse and
receiver dependence still require their original qualifications.

The W9 repaired record includes all four anchor families for this al028 window;
its DCVC anchor is best for both player metrics. Other Med windows still have
missing families. Window-level historical intervals cannot be presented as
independent-match population inference. Tier2 remains inconclusive under its
original two-metric rule; neither the website clip nor the conditional Tier1
result repairs that limitation.

**Priority correction:** retain the actual generative-player positive evidence
explicitly in the paper and next design review. Re-evaluate this saved route
before assuming the new rectangle construction is the best next experiment.
The next bounded audit should establish source-independent receiver state,
actual bank/adaptation/refresh/framing costs and temporal behavior, alongside a
source-spread human pilot of saved matched anchors. The site's18.46kbps and
bank-to-clip legend require correction against W9; no website edit or deployment
was performed. Metric mismatch is a possible general issue, but it is not the
explanation needed for this particular player-region observation.
