# PointStream

**PointStream** is an object-centric semantic video codec targeting ultra-low-bitrate broadcast sports and structured human-object scenes (ACM TOMM submission). Across cut-segmented shots, a source-only router selects per interval between:
- **`Conv`** — conventional SVT-AV1 video for off-court views and cutaways;
- **`Conv+Ball`** — a cleaned background base stream plus a lightweight parametric ball trajectory payload for direct machine coordinate readout (`~300 fps` on CPU) and rendered visual restoration; and
- **`Full-Obj`** — a cached reference appearance initialization (`.psr`), cleaned background base, 17-joint COCO player pose tracks (`.pscp`), and ball stream reconstructed at the receiver.

Historical development logs, retired scratch cards, and early exploration notes prior to the submission clean are preserved on `origin` at annotated tag [`archive/pre-tomm-clean-2026-11-25`](https://github.com/emanuele-artioli/PointStream/releases/tag/archive/pre-tomm-clean-2026-11-25).

---

## Core Submission Contributions

1. **Unified multi-mode codec and source-only router:** Cut-aware packetization (`Conv`, `Conv+Ball`, `Full-Obj`) charging every header, background, pose, ball, and initialization byte on one unified bitstream timeline, with source-only mode routing across all twelve held-out RacketVision TEST tennis matches (0 unmeasured matches; 11/12 court-eligible, 10/11 player-continuous under `<=10`-frame interior bridging).
2. **External joint foreground reconstruction & operating-range characterization:** Unadapted external 240-frame full-HD receivers on Shanghai (`match102`, 29.97 fps, `17.78 kbps` warm -> `0.295` vs. `0.399` SVT-AV1 / `0.352` DCVC-UF foreground DISTS) and Sydney (`match127`, 60 fps, `25.78 kbps` warm -> `0.277` vs. `0.406` / `0.349` DISTS), alongside complete bitrate ladders mapping exact crossover boundaries where pixel codecs overtake generative reconstruction (`~118.9 kbps` Shanghai, `~191.7 kbps` Sydney), full temporal coherence analysis (`self_flow` / `tLPIPS`), and 24-frame rendered startup compression (AVIF Q90 cuts Sydney startup `60.0%` from `149,450 B` to `59,776 B` within `0.0004` DISTS) and 5-frame pose bridging (`0.2993` vs. `0.2991` uninterrupted DISTS).
3. **Twelve-match ball coding & zero-shot multi-sport transfer:** Across twelve salted-hash RacketVision TEST tennis matches (402 visible labels, zero target tuning), `Conv+Ball` achieves **`-21.7%` mean BD-rate** vs. adaptive-width SVT-AV1 (wins in 12/12 same-resolution pairs) and generalizes zero-shot to 24 held-out RacketVision TEST clips in badminton and table tennis (winning 23/24 against same-resolution SVT-AV1 and 21/24 against multi-resolution AV1).
4. **Dual human-and-machine consumption & speed–quality Pareto optimization:** Direct `PSCP`/ball payload readout recovers `1.000` pose availability (`0.925`–`0.953` PCK@0.1 in `0.43`–`1.79 ms/frame` CPU) vs. `0.000`–`0.010` matched-rate SVT-AV1 full-frame `YOLO26x-pose` recall (`1/96` on `17.71 kbps` Shanghai `640-warm`), while matched RTX 6000 Ada benchmarks show 10-step DPM-Solver + overlap-8 + pose smoothing runs **`2.88x` faster** (`196.2 s` vs. `564.5 s` DDIM-30) within `+0.0009` DISTS and `-24.8%` seam `tLPIPS`, and 6-step / `384px` reach **`4.61x` / `5.77x`** speedups (`+0.0038` / `+0.0052` DISTS).

---

## Reproducing Paper Numbers (Offline, No GPU Needed)

All numerical tables and plots in the ACM TOMM paper can be reproduced offline from checked-in receipts.

### 1. External joint operating-range ladders (Shanghai & Sydney)

```sh
card=docs/experiments/joint-operating-range-20261123
shanghai=docs/experiments/shanghai-wider-20261123
PYTHONPATH=. python "$card/compare.py" OUTPUT_DIRECTORY \
  --shanghai-wider "$shanghai/q23-metrics.json" "$shanghai/q17-metrics.json" "$shanghai/q640-metrics.json" \
  --ordinary "$card"/ordinary-*.json --roi "$card"/foreground-roi-*.json
```

### 2. Twelve-match ball metric suite

```sh
python docs/experiments/ball-metric-suite-20261123/summarize.py OUTPUT.json \
  docs/experiments/ball-metric-suite-20261123/match*-result.json
```

### 3. Validation campaign receipts (Pareto, pose machine, multi-sport, matched checks)

Inspect [docs/experiments/tomm-validation-20261124/README.md](docs/experiments/tomm-validation-20261124/README.md) for job IDs, SHA-256 receipts, and scoring commands covering:
- `v7-shanghai-ordinary640-summary.json` (matched `17.713 kbps` SVT-AV1 pose recall `0.010`),
- `halle-match105-finish-summary.json`, `halle-joint-scores.json`, `halle-score-joint-crf50.json`, `halle-motion-joint-crf50.json`, & `halle-pose-machine-summary.json` (complete unadapted Halle `match105` 240-frame synthesis, 12-frame joint FG DISTS/LPIPS/PSNR, 480-observation body DISTS, and 48-frame YOLO26x pose machine),
- `matched-checks-gpu-summary.json` & `matched-checks-scores.json` (48-frame matched timing, rendered Sydney Q90, rendered 5-frame gap bridge),
- `a4-badminton-12.json` & `a4-tabletennis-12.json` (24-clip zero-shot multi-sport transfer),
- `coverage-summary-12.json`, `pose-gap-audit.json`, & `a6-mode-selector-amortization.json` (12/12 TEST coverage & routing).

---

## Repository Layout & Local Checks

- `src/` — Core codec modules, segmentation (`python -m src.segmentation --help`), quality/metrics, container serialization.
- `experiments/` — Reproducible fleet workers, scoring pipelines, and evaluation harnesses.
- `docs/experiments/` — Frozen experiment cards, specs, and verified JSON receipts backing every paper claim (`% CLAIM(...)`).
- `scripts/ps-fleet`, `docs/fleet.md`, `docs/experiments.md`, `docs/resources.md`, `docs/components.md` — Fleet execution tooling and protocol specs.
- `tests/` — Unit, documentation, and contract test suites.

### Local development checks (CI gates)

```bash
ruff check
mypy --config-file pyproject.toml
python -m pytest
```

Quick segmentation check:

```bash
python -m src.segmentation run --backend yoloe-26n --domain tennis --source CLIP --out OUT
python -m src.segmentation --help
```

*(Note: `Models` and `Datasets` are external cluster symlinks excluded from git, ruff, mypy, and pytest.)*

---

## Prioritized Future Work

1. **Egocentric hand-object (VISOR) pilot:** Evaluate PointStream's object-centric decomposition, hand pose continuity, and cached appearance reuse on dense egocentric hand-object interaction in a dedicated follow-up session.
2. **Controlled human subjective preference study:** Run blinded ITU-T P.910 / pairwise preference trials comparing PointStream `Full-Obj` and `Conv+Ball` against SVT-AV1 and DCVC-UF at matched ultra-low bitrates.
3. **Real-time diffusion receiver (distilled / TensorRT):** Replace 10–30 step UNet inference with consistency/step-distilled models, FP8/TensorRT engines, and temporally recurrent KV caching to close the gap toward real-time 30–60 fps decoding.
4. **Dedicated racket/equipment keypoints & extended external cohort:** Add explicit racket/equipment landmarks beyond COCO body joints and expand full multi-rally `Full-Obj` evaluation beyond Shanghai, Sydney, and Halle to London Olympic clips and additional grass/clay rallies.
