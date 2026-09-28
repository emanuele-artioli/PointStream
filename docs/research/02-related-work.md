# 2. Related work and baseline selection

Research/artifact inspection: 28 September 2026. This chapter records primary
papers and author repositories, not reproduced PointStream comparisons. A code
link, README command, or advertised checkpoint is **artifact availability**, not
proof of runnable inference, actual bitstream coding, or reproduced accuracy.
The [experiment plan](07-experiment-plan.md) makes those separate gates.

## Comparison families

PointStream's primary question is broadcast tennis with transmitted references,
camera/background representation, players, rackets and a ball. Face codecs are
the historical foundation; human-body codecs are closer representation peers;
general-video generative codecs are the strongest domain-independent opponents.
Hybrid learned/conventional methods and distortion-oriented neural codecs are
necessary controls. A new content type alone does not establish a new method.

| Work / venue | Representation and tested regime | Paper's comparison set | Published result and limitation | Artifact inspection / priority |
|---|---|---|---|---|
| [MTTF](https://arxiv.org/html/2410.10171v1), 2024 preprint; author repository labels TCSVT 2026 | Compact temporal motion factors; foreground/background generation; faces and moving bodies, including TEDTalk 384×384 | VVC VTM22.2; MRAA, TPSM, CFTE, LIA in the explicit comparison section | Preprint moving-body single-resolution table reports 65.96% rate-DISTS saving over VVC; multi-resolution 69.35%. Not broadcast-tennis or a PSNR result. Some captions differ from the comparison text; verify final journal version before replication. | [Encoder/decoder, arithmetic coder, checkpoint link](https://github.com/xyzysz/Extreme-Human-Video-Compression-with-MTTF). Priority: closest semantic competitor. Human matting and checkpoint access need smoke. Learned motion features are **not interchangeable with DWPose input**. |
| [GLC-video](https://arxiv.org/html/2505.16177v1), TCSVT 2025 | General-video coding in a generative VQ-VAE latent space; 96-frame tests | HEVC/HM, VVC/VTM, DCVC-FM, PLVC | Reports 65.3% average rate-DISTS saving over PLVC; acknowledges lower PSNR/MS-SSIM than non-generative codecs and residual flicker. | [Video scripts and model release instructions](https://github.com/jzyustc/GLC). Inference priority; inspected test path uses estimated bits without a persisted stream. See the [intake audit](09-baseline-intake.md). CVPR 2024 GLC was image coding; cite the journal extension for video. |
| [GVC-RT](https://arxiv.org/html/2608.04891v1), accepted ACM MM 2026 according to authors | General video; fast generative latent coding with LFQ-based training | HM16.25, VTM17.0, DCVC-FM, DCVC-RT, PLVC, GLC-video | Reports 12.4% DISTS / 48.8% LPIPS BD-rate saving over GLC-video; 123.1/55.1 encode/decode fps at 1080p on RTX 4090. Hardware/protocol-specific author measurements. | [Inference, checkpoints and real-stream mode](https://github.com/semcomm/GVC-RT). High priority for quality/computation tradeoffs; training release pending at inspection. |
| [S²VC](https://openaccess.thecvf.com/content/CVPR2026/papers/Xue_Single-step_Diffusion-based_Video_Coding_with_Semantic-Temporal_Guidance_CVPR_2026_paper.pdf), CVPR 2026 | Single-step diffusion with semantic and temporal guidance; UVG, HEVC-B, MCL-JCV | HM, VTM, ECM, DCVC-FM, DCVC-RT, PLVC; DiffVC from reported data | Proceedings reports 51.62% average DISTS bitrate saving over prior perceptual method PLVC. Do not substitute a differing arXiv-version number. | [Official inference and checkpoint pairs](https://github.com/onedc-codec/s2vc_official); ≥24 GB VRAM recommended for 1080p. Secondary priority. Evaluation reads bpp from filenames: audit how it is obtained and persist/count complete streams. |
| [PLVC](https://www.ijcai.org/proceedings/2022/0214.pdf), IJCAI 2022 | Recurrent conditional GAN for general-video perceptual compression | HM16.20 low-delay P, DVC/OpenDVC, HLVC, M-LVC, RLVC and prior GAN codec; additional distortion comparisons | Perceptual metrics/user study favor PLVC in tested regimes; conventional distortion and perception tradeoffs remain. | [Weights and RLVC-based inference instructions](https://github.com/RenYang-home/PLVC). Legacy TF1.12, tensorflow-compression1.0 and HiFiC I-frames raise setup cost. Use if newer baselines fail availability gates, not simply because easier to beat. |
| [HDAC](https://goluck-konuko.github.io/static/paper.pdf), ICIP 2022 | Face animation fused with an auxiliary low-rate HEVC stream | DAC, HEVC, VVC | Reports >30% average BD-rate gains over HEVC and similar performance to VVC on conferencing data; extends animation-only rate range. | [Author GFVC integration with checkpoint links](https://github.com/Goluck-Konuko/GFVC), [animation-codec family](https://github.com/Goluck-Konuko/animation-based-codecs). Useful hybrid precedent, face-domain comparison only unless adaptation is explicit. |
| [SEVC](https://openaccess.thecvf.com/content/CVPR2025/papers/Bian_Augmented_Deep_Contexts_for_Spatially_Embedded_Video_Coding_CVPR_2025_paper.pdf), CVPR 2025 | Low-resolution coded spatial references, including emerging objects and large motion | VTM13.2 LDB, DCVC-HEM/DC/FM | Author table reports PSNR BD-rate reductions vs VTM of 17.5% HEVC-B,27.7% MCL-JCV,33.2% UVG,12.5% USTC-TD; 96 frames/IP−1. | [Weights and explicit real encoder/decoder](https://github.com/EsakaK/SEVC). Relevant control if emerging-object robustness is claimed. Not primarily a perceptual generative codec. |
| [GNVC-VD](https://arxiv.org/html/2512.05016v2), CVPR 2026 | Video diffusion transformer prior; sequence-level generative refinement | HEVC, VVC, DCVC-FM/RT, PLVC, GLC-video | Reports perceptual gains below 0.03 bpp and improved temporal consistency. Version/metric-specific reported curves, not local replication. | [Repository](https://github.com/CUC-MIPG/GNVC-VD) contained only a Coming Soon README. Literature comparator, not execution-ready. |
| [GIViC](https://openaccess.thecvf.com/content/ICCV2025/papers/Gao_GIViC_Generative_Implicit_Video_Compression_ICCV_2025_paper.pdf), ICCV2025 | Video-specific implicit diffusion representation; random access, YUV420, GOP32 | HM18.0, VTM20.0, AV1 libaom3.0.2, DCVC-DC/FM, PNVC, NVRC | Reports 15.94%,22.46%,8.52% BD-rate gains over VTM,DCVC-FM,NVRC on UVG. Its random-access optimization protocol differs from streaming. | Runnable official artifact not verified; project-page access failed. Lower priority unless per-video fitting becomes the PointStream claim. |
| [Sparse2Dense](https://arxiv.org/html/2509.23169v1), 2025 preprint inspected | Sparse 3D keypoints for human-video synthesis and vertex prediction | VTM22.2, MRAA,FV2V,TPSM,CFTE,LIA,MTTF,IMT,IHVC | Reports 74.54% DISTS BD-rate reduction vs VVC on 30 TEDTalk sequences, with limitations at higher rates. Geometry supervision is model-derived. | No runnable release verified. Required prior art for joint video/geometry claims; not evidence for a new tennis baseline implementation. |
| [SemConf](https://doi.org/10.1145/3712678.3721884), NOSSDAV 2025 | Multiparty semantic video conferencing | **Unverified**: full experimental text inaccessible in this search | Venue/title verified through [author publication list](https://mypage.cuhk.edu.cn/academics/wangfangxin/publications.html); no numerical result imported. | No public runnable code/checkpoint located. Do not invent its anchor list or make it a deadline-critical replication dependency. |
| [Neural Wrapping](https://openaccess.thecvf.com/content/CVPR2025/papers/Khan_Perceptual_Video_Compression_with_Neural_Wrapping_CVPR_2025_paper.pdf), CVPR 2025 | Learned pre/postprocessing around conventional video coding; gaming footage | AV1,VVC, neural coding and perceptual preprocessing comparators | Reports 18.5% average BD-rate saving over objective quality scores and improved MOS; metric/setting-specific. | No runnable official release verified. Relevant to hybrid design and simpler enhancement controls. |

All percentages above are the respective authors' reports, with different
datasets, anchors, access patterns and metrics. They cannot be ranked against
one another or subtracted from PointStream's historical values.

## Foundations, adjacent work, and novelty boundaries

- FOMM (NeurIPS 2019), Face-vid2vid (CVPR 2021), CFTE (DCC 2022), MRAA
  (CVPR 2021), TPSM (CVPR 2022), and LIA (ICLR 2022) motivate motion-driven
  reconstruction. Animation papers are not automatically codecs. The
  [GFVC software](https://github.com/Berlin0610/Awesome-Generative-Face-Video-Coding)
  supplies coding adaptations for FOMM/CFTE/FV2V; distinguish that adaptation
  from the original method and verify reference/feature bytes.
- The [GFVC survey](https://arxiv.org/abs/2506.07369) describes feature
  representations, evaluation and standardization. The available common
  software is a better starting point than writing a new face benchmark.
- [Interactive human-body coding](https://arxiv.org/html/2505.16152v1) and
  Sparse2Dense preclude a blanket “first beyond talking heads” contribution.
- [Txt2Vid](https://arxiv.org/abs/2106.14014) transmits text for talking-head
  audiovisual reconstruction; its quality-of-experience objective and speech
  assumptions differ from preserving a tennis event. Its
  [code](https://github.com/tpulkit/txt2vid) is contextual, not a primary opponent.
- [Spatiotemporal Diffusion Priors](https://cgl.ethz.ch/publications/papers/paperRel25b.php),
  PCS 2025, uses sparse bidirectional flow and diffusion interpolation. The
  [VoRTeC September 2026 preprint](https://arxiv.org/html/2609.02291v1) studies
  fast one-step flow-based coding. General-video generation and accelerated
  decoding are occupied research directions. Their artifacts were not
  qualified for this campaign.
- [DCVC](https://github.com/microsoft/DCVC) provides FM(CVPR 2024), RT(CVPR 2025)
  and UF(CVPR 2026) code/model links. Choose a verified runnable variant with
  matching access constraints, and state its date. FM/RT connect directly to
  most papers above; do not label either the latest universal neural SOTA.
- Presley and GenStream are prior work, not PointStream ablations; see the
  [introduction](01-introduction.md) for their exact receiver assumptions.
  Classic sprite/background reuse and conventional reference-picture tools
  also constrain novelty. The availability and configuration of a particular
  composite-reference implementation must be verified, not presumed from the
  codec standard name. Do not assert that AV1/VVC must forget the court at
  every point.

## The stage on which PointStream seeks an advantage

The proposed contribution is a **measured tennis-specific tradeoff**: preserve
court geometry and ball/racket motion while allocating generative capacity to
deformable appearance, under explicit total-rate and client-resource budgets.
This is a hypothesis. It must survive whole-frame generative competitors and
simple coded/warped foreground controls, not merely a face model's failure on
out-of-domain tennis.

Minimum planned baseline panel: AV1 and VVC implementation-specific ladders;
one qualified DCVC variant; GLC-video and GVC-RT; MTTF on a supported-domain
sanity test and clearly labeled tennis transfer/adaptation. Add S²VC if the
diffusion artifact passes its smoke; add SEVC if novel-object behavior is central.
Unavailable baselines remain listed with reasons. No paper in this table has
yet been **replicated by this documentation task**.

For each selected method first reproduce a short supported-domain case, then
run the same tennis input/window as PointStream. Freeze adaptation budgets and
shared/prepaid model assumptions before comparisons. Use a public metric
implementation with a pinned revision; new task metrics require independent
labels and calibration. Baseline quality failures, setup failures and OOM are
distinct outcomes.

## Search limitations

Search covered author repositories, arXiv/full papers, CVF proceedings, IJCAI
proceedings and author publication pages for face, human-body, general-video,
hybrid and implicit coding. Primary sources above support the imported claims.
SemConf full text and GIViC project access were unsuccessful; several checkpoints
are advertised on external storage but were not downloaded. A targeted search
did not identify a directly matching tennis court/player/racket/ball codec.
That is **not** a novelty proof or exhaustive literature review. Refresh
artifact availability and verify final publication versions at baseline intake.
