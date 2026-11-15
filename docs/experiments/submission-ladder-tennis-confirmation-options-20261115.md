# Experiments: installed tennis confirmation options

### 2026-11-15 — source-only eligibility audit and native-label mapping

Author-review eligibility draft. No confirmation outcome, cohort promotion, codec run, model inference or GPU use.

## Decision and scope

RacketVision TRAIN remains a possible installed source pool. The exposure of all 43 TEST clips does not by itself exclude TRAIN, and permitted encoder-training overlap must be disclosed rather than confused with judge contamination. The installed original sparse ball labels can support an explicitly **Level 1 ball-only** confirmation at annotated times; they do not require new dense foreground masks for that narrower endpoint.

However, an author split or `match_id` does not establish a new underlying physical match. Source identity, conservative cross-archive deduplication and learned-judge eligibility must be established before reserving three groups. No evaluation windows or quality outcomes were selected in this audit. The separate whole-foreground/egocentric Level 2 confirmation still needs its own annotation and reference gates.

## Actual installed inputs

The companion [machine receipt](submission-ladder-tennis-confirmation-options-20261115.json) binds exact installed manifest/sidecar hashes, all 431 official tennis CSV member hashes, original split members, native-header diagnostics and raw audit receipts. The full 2,009,118,720-byte tennis archive was not rehashed; selected original MP4 members were freshly hashed against its installed original member sidecar. Do not interpret sidecar-only uniqueness for every video as fresh whole-corpus payload authentication.

| Split | Rallies / author IDs | Native frames, official metadata | Official point-label rows | Visible / nonvisible | Unlisted, unknown frames |
|---|---:|---:|---:|---:|---:|
| TRAIN | 350 / 350 | 119,547 | 17,494 | 16,337 / 1,157 | 102,053 |
| VAL | 38 / 38 | 16,422 | 1,900 | 1,782 / 118 | 14,522 |
| TEST | 43 / 43 | 14,430 | 2,150 | 1,947 / 203 | 12,280 |

Every actual official CSV was read and authenticated. Label frame indices are unique and in range, with valid coordinates for visible entries. Neither interpolated trajectories nor merged model predictions substitute for official annotations. `Visibility=0` means not visible; it does not establish physical absence. Unlisted native frames remain unknown.

All author IDs and sidecar video hashes are distinct across the three splits. This excludes exact author-ID collisions and recorded byte duplicates, but not different rallies, crops, transcodes or reuploads of one physical match. The original annotation metadata contains no source URL, event, date, player-pair or tournament field. The tennis metainfo contains only `[1080,1920,3]`. A scan of 67 small official code/text archive members found no original broadcast identity map.

The [official RacketVision paper](https://arxiv.org/html/2511.17045v1) describes professional YouTube broadcasts and sampled manual point/visibility annotations. Its release date and split names are not dates or independent identities of the depicted matches. Counts above come from actual installed files, not rounded paper sampling descriptions.

## Native cadence and colour

Three lexicographic TRAIN clips were header/schema diagnostics only, not candidate reservations. `match172_000` has 250 native frames; `match173_000` and `match174_000` each have 249. All three are original archived H.264, 1920×1080, 25/1 fps, 8-bit yuv420p, limited range with BT.709 matrix/transfer/primaries. The latter two streams start at PTS 0.04 s. Official labels index decoded frame order from zero; a nonzero stream start PTS must not shift those indices.

TRAIN metadata also includes 60, 29.97, 30 and 50 fps clips. A future selected cohort must bind each actual native stream and conversion independently; historical 30 fps TrackNet conversion settings cannot be silently reused. Container encoder tags and MP4 packaging dates are not event provenance.

## Development exposure and judges

The bounded repository audit found actual tracker-selection results on all 43 TEST clips, including jobs `20261020T220058Z-fc8980da`, `20261020T214901Z-0d339486` and `20261020T214907Z-eefaac4b`; G1 used eight TEST clips. [K5b](k5b.md) records the 350/38/43 split and adoption of the authors' RVD-trained BallTrack checkpoint. It states a training protocol, but this audit does not establish an actual PointStream optimizer execution over each TRAIN frame. Nor does failure to find a TRAIN scoring log establish that no development exposure occurred.

Existing named broadcast development sources include Djokovic–Zverev, Alcaraz–Hurkacz, Medvedev–Struff, Djokovic–Federer, Rublev–Auger-Aliassime and Nadal–Kyrgios; TrackNet source games also require joining rather than assuming independence. See [K4](k4.md), [K6b](k6b-players.md) and the [source-only annotation pilot](submission-ladder-annotation-pilot-20261114.md). A new `match_id` with one of these player pairs is conservatively merged with that exposure group when dates cannot be verified.

For the proposed perceptual-only endpoint, the encoder's BallTrack training overlap is permitted with disclosure; no new BallTrack/WASB task accuracy claim is needed. This does **not** waive the independent learned-judge rule. The frozen [LPIPS](https://github.com/richzhang/PerceptualSimilarity) and [DISTS](https://arxiv.org/abs/2004.07728) judges have ImageNet-based backbones and additional learned calibration/quality parameters. The relevant provenance gate covers all learned weights, not only an ImageNet-2012 cutoff. The saved accepted score binds exact backbone/head hashes in the machine receipt; this audit did not load their models. An original event date after every frozen learned-weight release, or direct documented exclusion from all judge training sources, would resolve this conservatively. A 2025 dataset release or recent transcode does not.

## Conditional sparse-label Level 1 protocol

After eligibility, freeze three canonical physical source groups and source windows mechanically, before codec outputs or quality are inspected. Freeze the source hashes, full native decoded sequence, per-frame timing/colour conversion, official label hashes and selection seed. Keep the actual fixed PTP candidate and automatic serialized sender placement; official positions may define evaluation crops, never encoder placement.

Retain every decoded frame. Evaluate the predeclared 64×64 point crop at every official visible annotated time in each frozen window, retain nonvisible annotation records with undefined crop endpoint, and mark all unlisted times unknown. Never map unknowns to empty foreground. Compute full-frame metrics over all decoded frames and charge the complete packet against its complete duration. A 16-frame diagnostic cannot use a 48-frame packet as a promoted full-window quality result. Report source-group dependence and uncertainty; sparse times from one rally are correlated observations.

This tests the narrow ball-neighbourhood perceptual endpoint only. It supplies no dense-mask, whole-foreground, task, absence, real-time or universal rate–distortion claim. Preserve unsupported BD-rate intervals and historical spatial-map LPIPS naming.

## Source-only visual fallback: three concrete leads

Before inspecting any image, the fallback froze ascending SHA256 ranks of `pointstream-tennis-confirmation-v1:<match_id>_<round_id>` among TRAIN clips with at least 64 native frames. It decoded frame 0 of the first 12 and of all 43 exposed TEST clips, with one CPU core and no model. Original MP4 member hashes were checked against the original sidecar. Native PNGs, selection, full commands and frame hashes are retained in a new private SSD directory; the machine receipt binds the retrieved 55-frame archive and contact sheets. This is source-provenance inspection, not evaluation-window reservation.

The first 12 ranked IDs, in order, were `match408`, `match209`, `match174`, `match99`, `match410`, `match87`, `match89`, `match284`, `match382`, `match269`, `match357`, `match235`, all round `000`. Their scoreboards/events yield several different physical-match leads. Three particularly concrete leads are:

| Installed TRAIN clip | Native frame-0 evidence | Official event corroboration | Native frames / official visible times |
|---|---|---|---:|
| `match209_000` | DIMITROV / SHEVCHENKO; ABN AMRO indoor court | Rotterdam, 16 February 2024; [event organizer's next-day report](https://www.abnamro-open.nl/en/news/headlines/51th-edition/dimitrov-and-de-minaur-on-repeat) confirms this quarterfinal | 249 / 44 |
| `match269_000` | EVANS / TIAFOE; WASHINGTON, D.C.; Mubadala Citi branding | Washington 2023 quarterfinal; [ATP event report](https://www.atptour.com/en/news/tiafoe-evans-fritz-thompson-washington-dc-2023) and [official Tennis TV highlights](https://www.youtube.com/watch?v=jnZT6irpaf0), published 4 August 2023 | 250 / 50 |
| `match235_000` | SHELTON / MENSIK; INDIAN WELLS branding; first set 4–6 | Indian Wells, 8 March 2024; [official ATP day-three notes](https://www.atptour.com/-/media/5699814da282484abaa9d387b00268ab.pdf), page 4, list their first meeting | 570 / 41 |

These joins are **AI visual inferences supported by primary event records**, not human identity review, original source-URL recovery or exact-frame matching to an official broadcast. The official records verify that the matches occurred; they do not by themselves prove which uploaded broadcast supplied these clips. Different pairings make the three leads distinct from one another. All same-player-pair footage must be conservatively merged if event dates remain uncertain; distinct clip IDs never split such groups.

No same pair appears among the 38 legible TEST frame-0 scoreboards or the named broadcast development ledger. The five unresolved TEST identities are `match1_000`, `match13_000`, `match118_000`, `match120_000` and `match129_000`; first frames lack a readable pair or usable scoreboard. The complete original physical identity map of TrackNet `game*` sources was not established in this bounded audit. A repository Markdown search for these candidate pair names found no prior named development entry, but unindexed IDs or unlogged use remain possible. Thus **three concrete source-identity candidates were found; zero are promoted to strictly verified new-match confirmation yet**. This is a much narrower remaining gap than “TRAIN has no usable labels” or “all installed RVD is ineligible.”

The inferred 2023/2024 events also offer a conservative way to close learned-judge eligibility: establish the exact frozen LPIPS/DISTS learned-weight lineage and its release/training cutoff before these events. The 2018/2020 method dates make this plausible; this audit does not silently equate publication date with an authenticated release date for every weight byte. No future task-detector judge is required for this perceptual-only endpoint.

## Cheapest next scientific decision

Perform a small **source-only reference review**, using the already extracted images and primary event records: confirm the three identities, resolve the five unknown TEST/TrackNet source identities sufficiently to rule out the same three player pairs, and record prior PointStream use plus exact frozen judge lineage. No new dense ball annotation, encoder experiment or GPU scoring is necessary for this gate. If identities cannot be resolved conservatively, merge the uncertain exposure groups or leave those candidates unqualified; do not inspect quality to decide which source to keep.

Once that gate passes, mechanically reserve three eligible source groups/windows and their complete frame/label manifests before any confirmation outcomes. If only unknown-provenance author IDs can be established, any later measurement must remain additional scoped descriptive evidence, not the requested new-match confirmation. Do not reject every TRAIN source merely because encoder training may overlap; disclose that overlap beside the result.

## Resources and verification

The metadata/official-code audit used 3.117 measured remote CPU seconds. The 55-frame source-only fallback used 22.821 CPU seconds and 23.504 wall seconds; contact-sheet generation used 3.693 CPU seconds. Total measured remote Python and child CPU was **29.632 seconds**, within the shared 600-CPU-second authority, with zero GPU seconds. These figures exclude local text processing, SSH transfer and browser-service work. All executions completed successfully; two DNS failures occurred before the fallback script started. No unknown job was replayed, source was edited, model was loaded, quality was scored or evaluation window was reserved.

JSON syntax, reported counts, selected source hashes, 55 PNG hashes and Markdown whitespace were checked locally. Full-video corpus hashing, human identity confirmation, full TrackNet provenance and exact learned-weight release lineage remain explicit limits.

## Follow-up: source and judge gates now support scoped admission

This follow-up supersedes the unresolved-gate conclusion above. **Recommend admitting the three concrete source groups for a prospectively fixed, perceptual-only Level 1 confirmation relative to the recorded PointStream development ledger**, subject to the primary agent's independent review of the source-only evidence and acceptance of the provenance limits below. This is an admission recommendation, not a human-reviewed identity certification, universal unseen-data claim or completed experiment. No windows have been reserved and no confirmation outcomes exist.

For only the five unreadable TEST clips, the first, median and last *official visible-labeled* native frame indices were selected mechanically before inspection. The [private follow-up contact sheet](/private/tmp/pointstream-tennis-confirmation-followup-contact-20261115.jpg) shows:

| TEST ID | Native indices | Source-only facts | Exclusion relative to the three candidate events |
|---|---|---|---|
| `match1_000` | 9,269,590 | Western & Southern/Cincinnati branding and purple-green outdoor stadium; pair unreadable | Distinct from Rotterdam, Indian Wells and Washington |
| `match13_000` | 14,123,290 | Western & Southern/Cincinnati, HOLCIM US signage and the same distinctive outdoor stadium; pair unreadable | Distinct candidate events |
| `match118_000` | 9,352,590 | TORINO and Nitto ATP FINALS indoor arena; pair unreadable | Distinct candidate events |
| `match120_000` | 9,305,590 | GSTAAD clay court; later scoreboard GASTON / THIEM | Distinct event and pair |
| `match129_000` | 9,146,255 | Itau event logo, turquoise outdoor multi-court setting, doubles play; pair unreadable | Strong Miami venue inference; independently review this event exclusion |

These are AI observations, not human annotations. Unknown pairs remain unknown. The exclusions use the visibly different events, rather than invented player identities or inferred dates. The machine receipt binds all 15 actual source indices, native PNG hashes, official CSV hashes, source MP4 hashes and decoding commands. Native PNGs remain in the new SSD directory `/local/users/emanuele/pointstream/checks/tennis-confirmation-followup-20261115T-f5c393` on gpu1.

The actual TrackNet archive was freshly hashed as `bb45372d011ef4092d78173169c8ad6280d7dbdb9998efaf2b1c72a63ca1dd18`, matching the installed manifest: 2,564,111,322 bytes and 20,036 ZIP members. All stored member dates fall in 2017–2019. More decisively, its original `Dataset/Readme.docx`, member SHA256 `add605016272a954fd70f9da76788a5963437e63804053811fd55cd4ab1fca93`, contains the physical source table that the first audit had missed:

| TrackNet game | Original README source identity |
|---|---|
| 1 | Federer–Djokovic, Shanghai 2014 semifinal |
| 2 | Djokovic–Monfils, Eastbourne 2017 final |
| 3 | Murray–Djokovic, Doha 2017 final |
| 4 | ITF Future CIT 2014 final; players unnamed |
| 5 | Checa–Rumenov; event/year unnamed |
| 6 | Ferrer–Fognini, Buenos Aires 2014 final |
| 7 | Federer–Kyrgios, Laver Cup 2017 |
| 8 | Sharapova–Vinci, 2017 first round |
| 9 | Sharapova–Halep, US Open 2017 |
| 10 | Jung–Hong, 2017 Universiade final |

The [original 2019 TrackNet paper](https://arxiv.org/html/1907.03698) independently establishes the old ten-video dataset lineage. The actual original README, source names, old member dates and installed archive hash support chronological/source exclusion of the 2023/2024 candidate events; identifying every TrackNet image is unnecessary for that decision. The public acquisition chain is explicitly a third-party Drive mirror, and no independent original 2019 full-archive SHA256 was found. ZIP timestamps alone are editable and are not presented as cryptographic proof of an immutable 2019 publication. The README says 19,842 frames, whereas actual installed labels have 19,835; preserve that documented discrepancy.

Every learned judge component now has a concrete pre-event lineage:

| Frozen component | Exact installed identity / historical evidence |
|---|---|
| LPIPS AlexNet linear heads | SHA256 `df73285e…`; 6,009 bytes; Git blob `1df9dfe62abb1fc89cc7f82b4e5fe886c979708e` exactly matches [official commit c33f89e9](https://github.com/richzhang/PerceptualSimilarity/commit/c33f89e9f46522a584cf41d8880eb0afa982708b), 6 September 2020 |
| DISTS learned alpha/beta | SHA256 `f5e65c96…`; 12,288 bytes; Git blob `68fd96923642e8c416274430685c77661a39d66d` exactly matches [official commit da720e8a](https://github.com/dingkeyan93/DISTS/commit/da720e8aba80f2addfb25400d522e630a135416d), 15 April 2020 |
| LPIPS AlexNet backbone | Actual frozen full SHA256 `7be5be79…`; precise `alexnet-owt-7be5be79.pth` URL appears in [official TorchVision v0.12.0 code](https://github.com/pytorch/vision/blob/v0.12.0/torchvision/models/alexnet.py), released [10 March 2022](https://github.com/pytorch/vision/releases/tag/v0.12.0) |
| DISTS VGG16 backbone | Actual frozen full SHA256 `397923af…`; precise `vgg16-397923af.pth` URL appears in [official TorchVision v0.4.0 code](https://github.com/pytorch/vision/blob/v0.4.0/torchvision/models/vgg.py), released [8 August 2019](https://github.com/pytorch/vision/releases/tag/v0.4.0) |

The small installed learned-parameter files were read as bytes and checked against Git **tree metadata**, without deserializing tensors or downloading weights. Thus the LPIPS heads and DISTS alpha/beta chronology is an exact byte-identity result, not an inference from method publication dates. Backbone lineage binds the saved full hashes to the official historical checkpoint URL and its published eight-hex hash prefix; no separate historical full-SHA256 publication was obtained. The older TorchVision v0.4 AlexNet URL is `4df8aa71`, so it was specifically **not** used to justify the frozen `7be5be79` checkpoint. Together these records support temporal exclusion of the candidate 2023/2024 footage from these frozen judges, under the documented official-weight provenance assumption.

The cheapest next action is primary-agent review of the three candidate frame-0 identities and the five later-frame event exclusions, then an explicit eligibility admission record and mechanical source-window reservation. No model call, fresh quality result, dense mask painting or dataset download is needed to make that decision. Retain the possible RVD encoder-training overlap disclosure and scope freshness to the recorded PointStream development/selection sources; this audit cannot prove the absence of undocumented use.

The follow-up used 32.783 measured remote CPU seconds for frames/hash inspection/contact generation plus 0.156 for the original README/manifest. The cumulative measured remote CPU total is **62.572 seconds of the same 600-second authority**, with GPU use zero. Metadata-only official web retrievals and local text processing are excluded from that measured remote subtotal. All handles completed; no outcomes were opened, models loaded, weights downloaded or windows reserved. The full follow-up and metadata-response hashes are appended to the companion JSON, preserving the initial audit receipts.

A final actual-header check binds the **candidate** clips, rather than extrapolating from the three initial schema examples: `match209_000` is 249 frames at 25/1 fps, start PTS 0.04 s, duration 9.96 s; `match269_000` is 250 frames at 25/1 fps, start 0, duration 10 s; **`match235_000` is 570 frames at 60/1 fps**, start 0, duration 9.5 s. All three actual streams are H.264, 1920×1080, yuv420p, limited-range BT.709 matrix/transfer/primaries. Preserve the native 60 fps sequence for the Indian Wells clip and use its actual native label indices; neither a shared 25 fps assumption nor historical 30 fps conversion is valid. Header probes add 0.312 measured CPU seconds, bringing the final remote subtotal to **62.883/600 CPU seconds**, GPU zero.

## Final mapping gate: passed; ready for the primary agent to freeze windows

The primary agent independently inspected the candidate and five TEST event sheets and conditionally admitted the three source identities, subject to this native-label mapping gate. **That remaining gate passes.** The source-only [mapping contact sheet](/private/tmp/pointstream-tennis-label-mapping-contact-20261115.jpg) has nine rows (first/middle/last official visible labels per candidate) and three columns (native index `Frame−1`, `Frame`, `Frame+1`), all with the same original official point. Each crop is a nearest-neighbour enlargement of a 128×128 native diagnostic neighbourhood, with a red ring around the point; it is not a new label or an evaluated perceptual crop. The ball or its motion trace agrees with the official coordinate at the centre-column native index in all nine sampled cases. Adjacent frames show motion and do not motivate an index offset. This is AI source-alignment inspection, not human annotation validation or exhaustive label-accuracy certification.

The actual official code in the authenticated `RacketVision-c44af2a.tar` supplies the decisive index contract:

- `source/DataPreprocess/extract_frames.py`, member SHA256 `8ac15b056b9a735534bda752fb990e90af0fd58c63713875f9893c5c58e1f4e7`, lines 26–33: initialize `frame_idx=0`, read every successive `cap.read()`, write that frame as `0000.jpg`, `0001.jpg`, and so on. There is no fps normalization or temporal sampling.
- `source/BallTrack/dataset/uball.py`, member SHA256 `6674d128e980eb378994a2797800e1eca6271fa25ac5910f9ebcf286c3c68c2d`, lines 111–127: read CSV `Frame` into `fids`, set `curr_i=fid`, and use that exact zero-padded integer as the target native JPEG filename. No ±1 conversion occurs. Earlier sequence context is a separate sliding step.
- `source/DataPreprocess/generate_dataset_info.py`, member SHA256 `6d5bbec5d047799224ac2af9dfe3f8e0041d96ab94031882213b845eefedaa84`, lines 18–28: metadata cadence and frame count come from the actual video capture, rather than a global 25 fps assumption.

Actual CSV ranges independently agree: `match209_000` uses indices 9–239 with strides 4/5; `match269_000` uses 9–240 with strides 4/5; **`match235_000` uses 9–560 with strides 11/12**. The latter extends beyond a hypothetical 25 fps version of its 9.5-second clip. Its native indices 9,278,560 align visually; the two 25 fps clips were checked at 9,126,239 and 9,126,240 respectively. The original CSVs retain 50 sampled rows each, not 50 consecutive video frames. Native MP4 start PTS 0.04 for `match209_000` does not change the zero-based sequence index.

For the primary agent's proposed rule `first N=ceil(2.56×native_fps) frames beginning at decoded index 0`, the prospective counts are:

| Source group | Proposed native window | Actual duration | Official visible | Official not-visible | Unlisted unknown |
|---|---|---:|---:|---:|---:|
| Dimitrov–Shevchenko, `match209_000` | `[0,64)` at 25 fps | 2.56 s | 9 | 3 | 52 |
| Evans–Tiafoe, `match269_000` | `[0,64)` at 25 fps | 2.56 s | 12 | 0 | 52 |
| Shelton–Mensik, `match235_000` | `[0,154)` at 60 fps | 2.566667 s | 11 | 2 | 141 |

These are counts for a **proposed, not yet reserved** window. All 282 native frames remain required; only 32 have official visible point endpoints, five are officially not-visible and 245 remain unknown. Sparse visible endpoints are not an all-frame task label set. If the primary agent freezes an area-resized 1280×720 RGB view, original native points must be scaled by exactly 2/3 and the perceptual crop geometry must be explicitly fixed in that view. Preserve native cadence, all frames and complete duration; do not silently use 25/30 fps, treat missing labels as empty, or conflate a native 64-pixel crop with a 720-view 64-pixel crop.

The mapping job used 41.540 measured CPU seconds and 41.594 wall seconds. Final cumulative measured remote CPU is **104.424/600 seconds**, GPU zero. Source PNGs/crops and full official-code texts are preserved in `/local/users/emanuele/pointstream/checks/tennis-label-mapping-20261115T-c459f2` on gpu1; raw receipt and hashes are embedded in the companion JSON. No codec was run, judge loaded, perceptual outcome inspected, source edited, new annotation created or window reserved. The primary agent can now record final scoped admission and freeze its mechanical window manifest once, before outcomes.
