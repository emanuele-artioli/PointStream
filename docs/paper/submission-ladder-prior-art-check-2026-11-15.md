# Focused check of the regional-allocation novelty claim

A limited primary-source search on November 15, 2026 found no basis for treating regional allocation or combining ordinary codecs with ROI pre/postprocessing as new by itself. This is a targeted check, not an exhaustive novelty search or a performance comparison.

- [Meuel et al., ICME 2016](https://www.tnt.uni-hannover.de/papers/data/1162/Meuel_et_al_Codec_Independent_ROI_Video_Coding_using_a_Joint_Pre-_and_Postprocessing_Framework_ICME2016.pdf) already combine codec-independent ROI preprocessing with receiver reconstruction of background from prior transmitted content. Its background/motion construction differs from the present additive patch profile.
- [Perugachi-Diaz et al., BMVC 2022](https://arxiv.org/abs/2203.01978) establish ROI control within neural video coding, including synthetic training masks. Foreground priority alone cannot distinguish PointStream.
- [Różek et al., EURASIP JIVP 2025, article 18](https://link.springer.com/article/10.1186/s13640-025-00682-3) study object-ROI retargeting for video coding for machines. Their task-oriented results do not establish perceptual dominance on PointStream's sources, and PointStream has not implemented that method as a baseline.

Inference for this submission: complete accounting and a working receiver are prerequisites. A distinctive research claim must instead survive the simple ROI, coded-pixel, disk/streak and source-level controls, with an objective-specific interpretation. The new controls can support an empirical contribution; they cannot establish superiority to unimplemented prior methods. No new experiment or change to the frozen confirmation cohort follows from this search.
