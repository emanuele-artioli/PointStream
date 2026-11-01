# Citation relevance review — 2026-11-01

This is a bounded primary-source review for the current empirical, object-centric video-coding manuscript, not a novelty proof or a comparison experiment. The existing `sections/related_work.tex` already cites ELVIS (`elvis`) and GenStream (`genstream2025`) in its prediction, objects, and restoration discussion. Their existing proceedings DOIs are respectively `10.1145/3712678.3721881` and `10.1145/3746027.3758153`. Author versions: [ELVIS](https://arxiv.org/abs/2512.14185) and [GenStream](https://arxiv.org/abs/2609.18634).

**PRESLEY — include.** Emanuele Artioli, Farzad Tashtarian, and Christian Timmerer, “Perceptual Refinement of an End-to-End Video Streaming Pipeline via Generative AI Layers,” [arXiv:2609.19215v1](https://arxiv.org/abs/2609.19215v1), 2026; arXiv DOI `10.48550/arXiv.2609.19215` is listed as pending registration on the checked record. The record describes a submission to ACM TOMM, not an established published journal article. PRESLEY extends ELVIS by selectively degrading lower-priority regions, signaling block-level strengths, and using conditioned generative restoration. That is directly relevant to the paper's altered-target and restoration-cost discussion, though it is not PointStream's independent object stream and has not been replicated here. Place one method-only sentence immediately after the existing ELVIS description, with no inherited numerical advantage:

> PRESLEY extends ELVIS by selectively degrading low-priority regions, signaling the degradation strengths, and restoring them with conditioned generative models.

The [full text, §3.2](https://arxiv.org/html/2609.19215v1), clarifies that protected foreground still passes through lossy coding; the generative layer leaves the transmitted foreground unchanged across restoration variants. Do not repeat the abstract's unqualified “foreground fidelity bit-exact” as if it meant equality to the original source.

Suggested version-specific reference:

```bibtex
@misc{artioli2026presley,
  author={Artioli, Emanuele and Tashtarian, Farzad and Timmerer, Christian},
  title={Perceptual Refinement of an End-to-End Video Streaming Pipeline via Generative AI Layers},
  year={2026},
  howpublished={arXiv:2609.19215v1},
  url={https://arxiv.org/abs/2609.19215v1}
}
```

**MoQSplat — omit from the current focused survey.** Emanuele Artioli, Mohammadreza Ghafari, Md Tariqul Islam, Farzad Tashtarian, Christian Rothenberg, and Christian Timmerer, “MoQSplat: Adaptive Progressive Streaming of 3D Gaussian Splatting via MoQ,” [arXiv:2609.18624v1](https://arxiv.org/abs/2609.18624v1), 2026, DOI `10.48550/arXiv.2609.18624`. The arXiv record reports acceptance at IEEE MMSP 2026. The [author-laboratory record](https://athena.itec.aau.at/2026/07/moqsplat-adaptive-progressive-streaming-of-3d-gaussian-splatting-over-moq/) uses the title variant “over MoQ”; no proceedings DOI was verified in this bounded search. Object-aware progressive 3D Gaussian delivery is adjacent immersive transport, not an empirical 2D foreground video codec or evaluated PointStream comparator. It could support a future explicitly scoped discussion of spatial 3D delivery, but its inclusion here would not improve the present coding comparison.

**TIGAS — omit from the current focused survey.** The seven authors on [arXiv:2605.08699v1](https://arxiv.org/abs/2605.08699v1), “Thin-Client Interactive Gaussian Adaptive Streaming over HTTP/3,” 2026, are Emanuele Artioli, Philipp Fößl, Daniele Lorenzi, Farzad Tashtarian, Mahdi Dolati, Cheng-Hsin Hsu, and Christian Timmerer; arXiv DOI `10.48550/arXiv.2605.08699`. The [author-laboratory ACM MM 2026 acceptance record](https://athena.itec.aau.at/2026/07/thin-client-interactive-gaussian-adaptive-streaming-over-http-3-accepted-in-acm-mm26/) additionally lists Shao-Yang Hung, who is absent from the cited arXiv v1 record. A future citation must use the actual selected version's verified author list; do not silently combine them. No conference DOI was verified in this bounded search. TIGAS remotely renders 3D Gaussian scenes and adaptively transports view-dependent 2D projections; that is not the manuscript's object-decomposed video coding or a tested codec baseline. It would be relevant to a separately scoped remote-rendering discussion.

These decisions reflect relevance to the current manuscript rather than authorship. Searches may miss later publication versions or additional records; recheck bibliographic status at submission.
