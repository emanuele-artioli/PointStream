# Experiments: matched ball-objective ROI control proposal

### 2026-11-15 — cheapest missing conventional comparison

**Recommendation:** first test a3-rate conventional ROI curve on the accepted
48-frame `tracknet/game10/Clip3` development source, before another native-footage
tracker qualification. No model/encoding/GPU job was launched by this900CPU-cap
read-only intake. Exact provenance is in the [manifest](submission-ladder-ball-roi-control-20261115.json).

The real producer payload is
`jobs/fleet/inbox/20261113T210157Z-41ee26e3/full/published.tar`,
1,327,800,320bytes, freshly SHA-verified
`9074362c3f62ec18fba75acb3393c6a0803d907ad235a57bce98eb050c30b128`.
The15MB immutable preparation archive is audit records, **not checkpoint pixels**.
The producer contains original native/paidPSBL streams, authenticated receiver
YUV/RGB and source-qualification labels/source-BallTrack predictions. Its
`publish/checkpoint.json` hashes `1536eb02…aa9d03`.

The original raw inputYUV is no longer at its recorded scratch path. Reconstruct
it from the exact first48 original JPEGs in the existing TrackNet ZIP, checking
every JPEG SHA from `publish/source-qualification.json`, then use the pinned
`k5d_encoder.convert_rgb` conversion. Require66,355,200bytes and golden SHA
`a40b172b7a3e72c0fcf75dff15584564e3d220ffa6f1653439d045387f2d9963`
before encoding. This needs CPU preparation, not new tracker inference.

The native1280×720/30fps/preset4 SVT points areCRF48/55/62. **Complete IVF bytes**
are90,949/63,879/34,146; checkpoint `bytes`90341/63271/33538 omit608bytes of
IVF framing and must not price the anchor. Paid streakPSBL bytes are
91,272/64,177/34,421.

ROOT froze the initial newROI family before anyROI result: **CRF35/48/62,
0inside source-GT64×64crop-touching blocks,+40outside**. This is an explicitly
privileged conventional control, not an automatic deployable sender or a
universal bound. It supersedes the provisional+20 transmitted-center proposal;
that optional automatic control remains separately labeled. Build the rectangles
with the existing crop round/clamp convention and generate maps through
`w1_roi.roi_text(regions,1280,720,-40,outside=True)`.

Reuse `svtav1.encode_command/code`, cloning the accepted native command and adding
only `--roi-map-file`, paths and selectedCRF. Require an unchanged nativeCRF62
control SHA `cd79e397…757224b`. The current ladder encoder is useful plumbing,
but adds colour headers absent from this family; its complete runner also adds
new packet framing and uses different regional metrics. It cannot be run
unchanged and called a matched historical control.

Score against **original JPEG RGB**, using `windows.rgb_frames` and
`k5d_encoder.metrics`: full-frame metrics plus separately cropped source64×64
LPIPS/DISTS under the original weights. Full-image feature masks or the new70/30
regional objective are not substitutes. First reproduce a saved native/streak
metric control. Price ROI as all nativeIVF bytes, with AV1 ROI signalling included;
encoder-only map coordinates need not be transmitted separately. Compare with
actual paidPSBL bytes, supported common-quality intervals and full-frame costs.

A paid diskCRF62 point is already exactly joinable: old control job
`20261105T021434Z-3ec159d8` contains `disk.psbl`34,413bytes, SHA`ba4fc24a…96a777`.
Its streak packet SHA`90e8d04c…46a3c95` is **identical** to the accepted primary
SVT62 streak. Cached full/crop metric arrays exist, but require source/metric
identity and numerical parity checks before reuse. This is one disk point,
not a3-rate disk curve. The292-frame accepted conventional clip is a later
retrospective replication option. The initial48 are all visible: this control
supplies no absence or held-out claim.

The [saved-point pricing check](submission-ladder-ball-price-basis-20261115.json)
confirms the accepted manuscript already uses physicalIVF rates454.745/319.395/
170.73kbps. Recomputed fullLPIPS remains+1.2158968%. A separately labeled
payload-only SVT-anchor counterfactual gives+1.2118185%; no sign reversal or
correction to the accepted report follows. Historical reports remain unchanged.

Proposed separate stages: CPU reconstruction/encoding cap1200CPU seconds and
GPU scoring cap900GPU seconds, subject to ROOT freeze. A16-frame complete smoke
must forecast48-frame execution plus publication at1.5 margin and fixed setup
reserve before full. Reuse the original functional CPU decoder and metric
provenance; no new tracker/source qualification or isolation stack is needed
for this competing-anchor measurement. Source-denial claims retain their
original scope.
