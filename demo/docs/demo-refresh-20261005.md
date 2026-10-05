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
