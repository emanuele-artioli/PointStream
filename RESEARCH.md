# Research brief

Can sending points and appearance improve the parts of a video that matter at a given total bitrate? A conventional stream, ROI coding, tracked pixels, rendered trajectories and pose-conditioned synthesis are all credible choices. The [paper figure plan](PLAN.md) defines how to choose between them.

## Evidence we should build on

- **Viewers.** The tracked-pixel ball channel saved 26.28% at equal ball-crop DISTS across three matches. At roughly equal 303 kbps on one rally it improved the ball crop with a small measured whole-frame cost. The earlier player study found lower player-region DISTS and LPIPS on three windows from a familiar match under modeled reuse. [Ball figures](docs/paper/figures/ball/README.md) · [Player comparison](docs/paper/figures/selection/historical-low-rate.pdf)
- **Machines.** Measure what an image-only receiver can recover from the displayed video and what a receiver can read directly from the point stream. Use independent source labels for the task. Ball detection has supporting observations. Player keypoint quantization and direct body-mask overlap establish different, narrower results. [Player task figures](docs/paper/figures/player-tasks/README.md) · [VCM research and datasets](docs/vcm.md)
- **System.** Background reconstruction, reuse and shared reference ladders explain where the bits go. Charge every channel and use the same timeline, supports and reference policy before comparing complete pipelines. [Court comparison](docs/paper/figures/background/README.md) · [Reference ladder](docs/paper/figures/references/README.md)

## Submission decision

Build the evaluation around the figures that demonstrate a claim on matched streams. Use an object-specific claim when only that object has been confirmed. Report training exposure, reuse, whole-frame and temporal costs in the corresponding evaluation. The optional equipment and egocentric results keep their own scope.

<a id="main-hypothesis"></a><a id="model-and-adaptation-choices"></a><a id="boundaries-that-keep-the-result-defensible"></a><a id="what-not-to-repeat"></a><a id="audit-and-submission-state"></a>

[Previous research brief, model history and audit trail](https://github.com/emanuele-artioli/PointStream/blob/5424b0452412cdb0b95ecb0838d5edcc760d2f6a/RESEARCH.md) · [Experiment protocol](docs/experiments.md#protocol)
