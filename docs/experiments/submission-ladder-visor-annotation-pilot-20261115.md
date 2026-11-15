# Experiments: reserved VISOR source-only annotation pilot

### 2026-11-15 — 18 native source frames prepared; human review pending

**Ready for source-only human review, not confirmation.** The
[machine receipt](submission-ladder-visor-annotation-pilot-20261115.json) binds18
lossless PNG source renderings, official annotation proposals and exact source
video/member/frame identities. No new human decisions, model inference, codec
outcome, quality metric, training or propagation was produced. The provisionally
reserved cohorts and64-frame windows from the
[intake](submission-ladder-confirmation-intake-20261115.md) remain unchanged.

## Mechanical selection and proposal tiers

Each window contributes six frames: every official manual anchor in the window,
both endpoints, then farthest-gap filling with earliest-index ties. Selection
uses native indices and source metadata, never codec or mask-quality outcomes.
All indices below are **zero-based native decoded frames**; the corresponding
released VISOR frame is **index+1**, verified through the official50fps mapping.

| Video | Native indices | Initial proposal tiers |
|---|---|---|
| P35_105 |175,184,193,212,225,238|two official human, two dense-interpolated, two unknown/no proposal|
| P12_101 |217,238,249,260,271,280|four official human, two dense-interpolated|
| P11_103 |229,241,254,266,279,292|two official human, four dense-interpolated|

The eight human-anchor proposals use original1920×1080 polygons. Eight dense
proposals are rasterized on their854×480 release canvas and nearest-neighbor
resized to native geometry. These are distinct provenance tiers. The two unknown
P35 frames (225,238) have **no annotation**: invisible all-zero asset placeholders
only satisfy the generic review input interface, are never shown as overlays,
and are not empty ground truth. They require complete human redraw or explicit
empty confirmation. Reject/uncertain cases must remain in the review record.

Three additional hash-selected blind controls hide available proposals:
P35_105@184, P12_101@238 and P11_103@241. Thus five frames have no overlay asset;
13 overlays are shipped. The displayed instruction specifies visible hands/
forearms and the released active-object target set for each video, exact
zero-based/one-based frame mapping, proposal provenance and unknown flags.
Corrections replace the whole foreground union; unrelated scene objects are
excluded. No agent filled in the reviewer name, polygons, decisions or approvals.

P11's same-frame human/dense contradiction anchors229 and279 are both included,
using their human proposals. P12's uncertain tap visibility on261–270 is **not**
resolved by the mechanically selected pilot; it remains uncertain, not proved
missing. This18-frame pilot cannot repair the full192-frame reference set or
establish confirmation eligibility. Official source annotations and proposed
masks remain proposals until the actual review disposition is incorporated.

## Artifacts and reproducibility

New immutable archive (54,446,080B):
`/home/itec/emanuele/Datasets/pointstream-data/ladder-autonomy20261114/annotation-visor-confirmation-20261115T0320/pilot.tar`.
SHA256: `7c83d70167e3b803535b6dc1cbe78be30b903ef1dc1dd76af57500cb63b101be`.
Remote write/readback and local transfer hashes pass. The exact selected source
closure is separately preserved read-only in the adjacent
`annotation-visor-confirmation-20261115T0320-source-proof/source.tar`, SHA256
`5ebd23351a3888c277a5ad9e7c397ed2ec64a4ddb0aab361b7c0693f8a3b8247`.
Neither completed publication was overwritten.

Review digest:
`5d67a28c53714e410dd4ec0d77a765b1a7181575d17abe69a48e9d753fd755b8`.
It differs from the Tennis pilot digest, so their local browser drafts use
different hash-bound storage keys on the same HTTP origin.

Durable local review:
`/Users/manu/Documents/PointStream-review/annotation-pilot-2026-11-14/review/visor-confirmation-pilot/index.html`.
Live URL: `http://127.0.0.1:17691/visor-confirmation-pilot/index.html`.
Only this fresh subdirectory was added. The old Tennis HTML hash remains
`14d3bd320c21d5c6cb4dfa82bd73e741052d49d8087dab083905185141630200`;
both old/new URLs returnedHTTP200 with exact matching file hashes.
All media remain local/source-review material, without external redistribution.

Export reproducer: `python -m tools.datasets.visor_annotation_pilot --help`.
The checked source closure uses canonical VISOR parsing/rasterization, known
packed FFmpeg (`env32e`, exact SHA in receipt), and known-safe annotation UI.
All three original MP4s and selected official sparse/dense/mapping members were
rehash-verified. Native frame selection preserves geometry and50fps cadence;
full-range BT.709 YUV is explicitly rendered as RGB8 and stored losslessly in
PNG, with no resizing or generated frame. These RGB PNGs are annotation
renderings, not a new codec or pixel-identity claim against native YUV planes.

## CPU budget, browser verification and helper recovery

ROOT released at most2400CPU seconds,0GPU. A single CPU-affined gpu6 export,
with CUDA hidden, completed in160.712682 command wall seconds and155.401820
self+child CPU seconds including archive publication/readback. The source,
cache and intermediates used the verified separate host-local filesystem under
`/local/users/emanuele/pointstream/annotation-visor-confirmation-20261115T0320`;
TMP/TMPDIR/TEMP/Python cache locations were there, never server`/tmp`. Individual
FFmpeg calls were bounded300s and the complete command2400s. These are scoped
preparation costs, not encoder speed or exact whole-administration accounting.

Two focused selection/proposal tests, configured Ruff and configured Mypy
(631source files) pass. Actual export verifies source geometry, frame count,
all selected asset hashes and annotation tiers. Firefox visual QA observed the
manual proposal UI, source PNG rendering, unknown-frame blind/no-overlay state
and explicit redraw/empty requirement. Navigation only was saved; human marking
time and references remain pending. Browser dwell includes QA, not human effort.

The previous HTTP helper was unexpectedly absent before its recorded deadline;
its child19189 and wrapper19188 were both absent and the port had no listener.
No scientific computation was replayed. The source-only helper was recovered
on the **same root, origin and unchanged absolute deadline** using a detached
one-shot wrapper: **wrapperPID46003, childPID46006**, expiry
**2026-11-15 11:07:34UTC** (**08:07:34 America/Sao_Paulo**). ParentPID1 and HTTP
readiness were observed after the launcher exited. A separate **03:22:50UTC**
check after the prior agent final verified that both exact processes remained
alive, with parentPID1 ownership, and both URLs still returnedHTTP200 with the
same content hashes. The parent's intervening non-escalatedcurl000/exit7 did
not establish a process death: localhost access must be checked with the
appropriate sandbox permission. No second restart, launchd registration or
deadline extension was performed. This is observed persistence, not a promise
against logout, reboot or explicit termination. There is no KeepAlive, renewal
chain, model automation or recurring LLM call. It terminates only its own child
at expiry.

`/private/tmp/pointstream-annotation-pilot-20261114/server-state.json` and
`detached-server.log` retain current helper identity, exact command and deadline;
the preceding helper state is preserved separately. Drafts are in browser
storage under the HTTP origin. Opening `file://.../index.html` has a different
storage origin and does not automatically recover HTTP-origin drafts. All
source files, old Tennis drafts and annotations were preserved.
