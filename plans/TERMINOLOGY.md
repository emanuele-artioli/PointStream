# PointStream terminology

Use the **preferred term** in plans, reports, plots and the paper. Existing code
and stored results may retain a legacy identifier until a bounded compatibility
change is safe. Do not perform a repository-wide rename during the September
experiment campaign.

| Preferred term | Legacy term / identifier | Meaning |
|---|---|---|
| **background reference image** | plate | One reusable image representing the scene background. Use “background panorama” only when it is actually wider than the source frame. |
| **independent background coding** | `panorama-full` | Encode a separate background reference image for each scene. |
| **predictive background-sequence coding** | `panorama-stream`, stream mode, cross-scene AV1 streaming | Encode compatible scene background images as frames of one predictive video sequence so later backgrounds can reuse earlier ones. |
| **standalone background-image codec** | sidecar codec | The codec used when one background reference image is encoded independently. “Sidecar” did not explain what it encoded. |
| **background-sequence codec/quality** | `stream_codec`, `stream_crf` | Codec and quality control used by predictive background-sequence coding. |
| **standalone background quality** | `intra_qp` | Intended quality control for independent intra/background images. It is currently unwired; remove it or connect and verify it. |
| **canonical background canvas** | common/union canvas | Shared dimensions, origin and coordinate system for a compatible group of scene backgrounds. |
| **background context** | — | Scenes whose backgrounds are compatible enough to share a canonical canvas and predictive coding, normally the same camera/view/venue. |
| **scene length** | span | Number of consecutive frames represented by one scene background model. State frames and seconds. |
| **rate--quality sweep** | ladder | Several encoder configurations producing a rate--distortion curve. |
| **operating point** | rung | One measured configuration on a rate--quality sweep. |
| **system preset** | tier / tier config | A named bundle of PointStream component settings. |
| **experiment configuration** | arm | One side/configuration of a controlled comparison. |
| **reference codec** | anchor | AV1, VVC, DCVC-RT or another comparator encoding the same source sequence. |
| **reference-codec end-to-end time** | anchor wall-clock time | Time to encode and decode the source with the reference codec. It is not directly the same workload as full PointStream processing. |
| **per-additional-frame payload slope** | two-point marginal estimate | Fitted change in bytes per added frame after at least three scene lengths. The old number used only 8 and 16 frames and is provisional. |
| **late-frame quality drop check** | quality-drift alarm | Last-frame minus first-frame quality. A large drop suggests reconstruction degrades over the scene. |
| **component ablation matrix** | ablation lattice | Controlled configurations that turn core components on/off and measure their contribution. “Lattice” overstated free composability because some combinations are constrained. |
| **conventional fallback control** | all-off lattice corner | Source video encoded entirely with the conventional fallback path. The current shortcut must become an explicit valid configuration. |
| **compression opportunity bound** | headroom | Measured amount of conventional bitrate attributable to/removable from a region under an oracle-like intervention. It is not achieved system performance. |
| **background payload fraction** | background share / plate share | Background-reference bytes divided by total transmitted bytes. It does not show whether the total rate is competitive. |
| **reconstruction model candidates** | engine roster | Candidate foreground reconstruction/generation methods. |
| **foreground object** | actor / subject / player | Use “object” generically and “player” for the tennis-domain instance. Keep API names only where compatibility requires them. |
| **correction signal** | residual | Pixel-domain correction between the receiver reconstruction and target. “Residual” is standard and may remain in technical passages after definition. |
| **transmitted bytes** | payload / wire cost | Bytes actually sent. Estimates must be labelled estimates rather than wire costs. |

## Phrases to avoid

- “Win by construction.” Say that the representation removes a named form of
  repetition and then report whether its overhead preserves the advantage.
- “Live” or “real-time” for an offline prepass or an unprofiled configuration.
- “High fidelity,” “efficient,” “low bitrate,” or “significant saving” without
  an adjacent measured result and scope.
- “Generator” for a paste, upscaler or deterministic renderer.
- “The background is the problem” without a scene length; its fraction falls as
  the one-time background cost is amortized.

## Code migration rule

Misleading **behavioral** names are fixed before publication:

- the reported background codec must be the codec that actually produced bytes;
- `intra_qp` must either reach its encoder and have an effect test, or be
  removed;
- plots and JSON schemas use the preferred labels.

Cosmetic identifier renames wait until after the evidence freeze unless they are
small, compatibility-preserving aliases. Stored result keys remain readable.
