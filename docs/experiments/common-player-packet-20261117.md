# Experiments: common-player packet contract

### 2026-11-17 — CPU implementation and retained-input check

**Implemented locally; no additional GPU or training release.** This is a
packet/conditioning contract, not a complete learned video receiver. The
approved motion replay's outputs and budgets remain separate.

The new `experiments.tennis.common_player_packet` module preserves the old W8
format and adds explicit common-player and native-only profiles. It carries
actual native IVF, compact KPS2 tracks and paid PNG references, with framing,
installed-resource identities, per-run references, timing and composition order.
Unpacking and inspection read the bounded embedded profile; an external expected
profile is optional. [Inspection of the real carrier](common-player-packet-20261117/embedded-profile-audit.json)
also succeeds without external content-specific configuration.
Keypoints count as ongoing frame data; only envelope bytes count as window
framing. Native-only mode has no semantic payload or generator dependencies.
Every tick has declared coverage, with uncovered ticks using the native base.

The contract declares coded dimensions separately from the coordinate/output
canvas and pins RGB cubic resampling. Retained IVF is480x272 while player
coordinates use1920x1080. Same-unit continuous KPS2 tracks cannot cross cuts;
separate shot units and paid references can preserve complete coverage. Gaps
reset runs and use independent paid references. Shared geometry/visibility
constant drift fails closed.

The [final actual retained-input audit](common-player-packet-20261117/retained-input-final-audit.json)
authenticates the existing W8 packet/profile and explicitly recodes KPS2
metadata/order without changing decoded ticks, coordinates or visibility. For
both players, references, reference boxes, all240pose maps, placement boxes,
run IDs and seeds exactly match the inputs used by the approved motion replay.
An installed local native decoder also decodes all240coded-resolution frames.
No new generator, mask model or quality metric was run. The check used about
1.79parent CPU-s and0.14child CPU-s.

| Serialized component | Bytes |
|---|---:|
| Native base plus KPS2 |23,547|
| Lossless references |366,161|
| Envelope/framing |2,273|
| Match-specific adaptation |0|
| Total |391,981|

This carrier's cold4.8s bill is653.30kbps. That is a cost diagnostic, not a
codec gain. References dominate this short window despite the installed common
model. Before an expensive matched-quality study, resolve cheaper reference
coding or demonstrate actual longer reuse and refreshes; duration arithmetic
alone does not establish reuse. Changing coded references changes conditioning
and needs its own declared bounded test. The next study still requires effective
ROI/DCVC and relevant enhancement controls, all-frame scoring and human viewing.

Verification:33focused tests, configured Ruff/Mypy (672files) and the final full
suite pass:2,699passed/39skipped, with one existing fork warning. An earlier
sandboxed full-suite attempt failed because its process-group test could not
invoke `ps`; the final full run had that OS access. No test was weakened.
Independent code review caught inner-profile and expansion assumptions before
publication. The new profile checks actual W2 coding settings and bounded run
plans before decoding, bounds transport/window and integer-prefix expansion,
and validates PNG dimensions before allocating image pixels. The
[final review](common-player-packet-20261117/code-review.json) passed, and valid
retained packets still reproduce identical conditioning and carrier bytes.

The [CPU profile](common-player-packet-20261117/cpu-profile.json) and
[final audit script](common-player-packet-20261117/audit-final-script.py) preserve the
local recipe. Inputs remain the original immutable W8 artifacts; local paths
are validation caches. The audit script refuses an existing output directory.
The native binary identity is for this local CPU check; the GPU receiver's
complete installed resources and environment still need freezing/validation.

Remaining work: actual source-denied common-model generation, receiver masks,
native-to-canvas reconstruction/compositing, resource-failure behavior, a joint
paid ball path and matched-rate evaluation. Header validation in the library
does not establish AV1 decodability; the separate real decode check establishes
only the tested native stream. No complete-decoder or rate-quality claim follows.

### JPEG packet compatibility (local CPU)

The separate `common-tennis-player-jpeg-v1` recipe accepts only nominal sender
qualities 70 and 40. The quality field is encoder metadata; it does not prove
which quantization tables are present. Exact REF1 payload hashes identify the
experimental inputs. The profile declares a `reference_decoder` resource: the
SHA256 of the UTF-8 OpenCV build information measured at decode time, a
configuration fingerprint rather than a binary or runtime identity. The
preparation build was Mac OpenCV 4.10.0 with hash
`bade1ed132afc491c4cadda9ea0590a3d0745e16807f765450d3d3d4152a780b`.
This pins that build, not Linux pixel parity. Before OpenCV allocation the
transport checks a bounded, single-scan baseline 8-bit three-component JPEG
header and 512×784 dimensions; this structural check does not prove full
entropy validity. The JPEG profile also binds each reference name to the
expected decoded RGB SHA256, checked against the actual decoded bytes. For
the pilot these expected hashes must come from the independently frozen prepared
manifest. A self-declared hash does not authenticate the sender or payload.
PNG v1 and native-only profiles remain separate.
This is byte accounting and conditioning transport only; it does not establish
a learned receiver or a rate–quality result.
