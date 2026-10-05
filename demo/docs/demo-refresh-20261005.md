# Demo refresh — 5 October 2026

The user confirmed three recordings from two factories: Factory001 clip1 and
clip3, plus Factory002. Stable website IDs map as follows:

| ID | Training recording | Demo interval |
| --- | --- | --- |
| clip_01 | clip_01_factory001_worker001_00001 | Existing last10s holdout, start1189.933s |
| clip_02 | factory002_worker001_00000 | Existing last10s holdout, start1189.767s;299frames |
| clip_03 | clip_03_factory001_worker001_00000 | Existing last10s holdout, start423.433s |

Immutable source/checkpoint/anchor manifests are under
`/home/itec/emanuele/Datasets/pointstream-data/jobs/demo-refresh/20261005T130522Z`.
These original videos and saved results remain read-only. A new export must load
the matching factory's trained foreground checkpoint and saved training anchors,
not the older single overfit checkpoint or anchors mined from the holdout.
The initial rebuild retains the existing pix2pix foreground implementation;
this is not an architecture ranking or a new training run. DCVC/HNeRV background
qualification remains separately incomplete and is not silently substituted into
the six-rung AV1-background PointStream recipe.

Required camera ladders are PointStream and native AV1 at180,240,360,540,720,
1080p. All website mask families (DINOv3, RTMPose, SAM3.1 and YOLOE) must be
regenerated against these exact intervals at every applicable rung. Source IDs,
actual frame counts, source/model/anchor hashes, native tools and measured rates
must agree before switching the public site. Old aggregate scores and alignment
numbers cannot be relabelled as new measurements. Pose-model agreement is not
human ground truth. Model setup and browser-preview sizes stay separate from
native streaming payload sizes.

A bounded same-path smoke precedes each full stage. The live site remains
unchanged until the complete export and mask set pass identity, real-decode,
frame-count, ladder-completeness and browser checks. Failed or partial runs stay
external for diagnosis; they never trigger publication. No paused Gate-A job is
resumed, and no Claude work is changed.

## Campaign and publication gate

Camera job `20261005T131343Z-9013609e` completed its same-path smoke,
validator and full export on gpu5. Mask requests are independently gated:
YOLOE `20261005T133458Z-93087ca1`, SAM3.1 `20261005T133519Z-f3bb6020`,
RTMPose `20261005T133741Z-1b916224`, DINOv3 `20261005T134656Z-b219552a`.
Source-transfer acknowledgement timeouts require exact read-only preparation
reconciliation before publishing the original request; they never permit a
replacement launch. GPU3's pre-existing empty CPU mutex has no ownership
metadata and is not removed. Admission can use gpu5 normally.

DINOv3 architecture code is pinned under `/home/itec/emanuele/Models/DINOv3`
at `6876159a11b4df116f30f667f8c9888617df0751`, with the existing local weights.
Strict loading rejects partial state dictionaries and unrelated weight fallbacks.
RTMPose mask generation uses explicit existing local detector/pose ONNX paths.

`demo.pitch.release` assembles a new external release from completed camera
and mask stages, checks all 111 videos with native ffprobe, checks keypoint
lengths, binds sources and weight hashes, and records every public file hash.
`demo.pitch.publish_site` refuses partial or changed releases. GitHub Pages
assembles into a clean temporary directory from that inventory, excluding old
unreferenced clips and plots. The inspector retains comparison, resolution,
mask, playback and crop controls while deriving current rates and model
agreement from the release. Historical latency, LPIPS, IoU and task-truth
claims are not carried over. DINOv3's PCA visualization and AV1 mask-preview
rates are explicitly distinguished from native feature/coordinate payloads.

The first DINO smoke stopped before extraction because the default Torch lacks
`torch.amp.custom_fwd`. Its full stage did not run. The revised configuration
uses the already installed `pointstream-neural` interpreter (Torch2.5.1+cu121,
OpenCV4.11.0), records the actual inference versions and retains strict loading.
The original failed request remains intact. Its 200-second conservative charge
is deducted from the original3900-second allowance:3700seconds remain, with a
2400-second full-stage estimate and the original absolute deadline retained.
No shared runtime was upgraded and no old execution was replayed.

Visual inspection of the refreshed camera previews shows black rectangular
foreground regions, especially on Factory002. The RGB-only pix2pix model uses
box feathering rather than a learned alpha; the current reconstruction also has
crop/aspect qualification work remaining. This release is an honest prototype
comparison, not a claim that only throughput remains. At240p, Factory002's
PointStream model-agreement hand recall is19.6% vs AV1's73.7%, and MPJPE is65.3px
vs22.9px. Preserve these negative results. Quality, alpha/crop behavior and
receiver qualification should lead the next discussion before timing optimization.
