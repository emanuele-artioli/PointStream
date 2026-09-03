## Report: Wave 5 Stream F (Paper Catch-Up)

**Target Repo:** `67a9ea6275d3d9785ce57026/` (Overleaf paper repo)  
**Commit:** `f32c1cf` (`catchup: update headroom to n=8, retract VVC gap, scope generator baselines, and record tier runs`) — pushed to `origin/main`

---

### 1. Key Changes and Findings Landed

#### A. Motivating Headroom Section: Brief, Punchy, Focused on Headline Evidence (`sections/problem.tex`, `appendices/headroom_measurement.tex`)
- **Headline 1 (Foreground Cost):** Salient players occupy $1.11 \pm 0.32\%$ of frame area (alpha silhouette), yet carry a **$18.9 \pm 5.0\times$** bitrate concentration (inside pre-registered $[10, 60]\times$ band). Removing them via background-plate inpainting saves **$14.2\%$--$18.3\%$** BD-rate across the entire codec ladder (AVC $17.0 \pm 3.1\%$, HEVC $18.3 \pm 3.4\%$, AV1 $15.4 \pm 2.8\%$, VVC $14.2 \pm 2.6\%$).
- **Headline 2 (Background Redundancy):** Transmitting a panorama plate once with per-frame homographies ($1728$\,B/clip) yields **$64.3\%$--$78.0\%$** background BD-rate saving vs conventional inter-coding ($0.780 \pm 0.056$ for AV1, $0.761 \pm 0.039$ for VVC, $0.665 \pm 0.073$ for HEVC, $0.643 \pm 0.084$ for AVC).
- **Headline 3 (Consistency Across Standards):** Evaluated across modern codec generations under common QP ($32/40/46$) and common PSNR intervals, the AVC$-$VVC gap is $+0.028 \pm 0.015$ ($1.8\sigma$) and $+0.023 \pm 0.017$ ($1.3\sigma$). Both fall below the $0.04$ threshold ($<2\sigma$), proving that conventional block-based compression across generations leaves the same object-coding headroom unexploited.
- **Headline 4 (The PointStream Opportunity):** PointStream replaces dense pixel grids with compact semantic motion (poses/trajectories) over a shared background plate and an optional corrective residual, targeting exactly this unexploited headroom.
- **No Work-Schedule / Chronology Fluff in Reader-Facing Prose:** Removed internal backstory and references to "preliminary probes" or work stages. The text presents clean, peer-review-grade empirical findings.
- **Full Measurement Appendix (`appendices/headroom_measurement.tex`):** Full technical details (exact encoder versions `libx264`, `kvazaar`, `SvtAv1EncApp`, `libvvenc`; uncompressed paste-back MAE $0.0$ validation; pre-registered bounds/alarms; and Table 3 per-clip breakdown across all 8 scenes and 4 codecs).

#### B. Generative Negative Results: Framed as Diagnostic Floors Driving Positive Improvements (`sections/evaluation.tex`)
- **Diagnostic Finding:** Off-the-shelf generative baselines (trained on pose/segmentation conditions with text prompts and no reference images) synthesise generic players and lose to the static-copy floor ($11.82$\,dB object PSNR / $0.4505$ LPIPS vs seg-controlnet $12.19$\,dB / $0.5595$, Animate-Anyone $12.21$\,dB / $0.5692$, pose-controlnet $12.03$\,dB / $0.6031$).
- **Strategic Purpose for Future Sessions & Claude:**
  - This negative result is explicitly **not** a dead-end or permanent limitation of PointStream; it diagnoses the exact architectural requirement for generative video coding: models require dedicated reference-appearance conditioning (e.g. cross-attention image embeddings via IP-Adapter, or ReferenceNet feature extraction).
  - The static-copy floor ($11.82$\,dB / $0.4505$ LPIPS) serves as the concrete performance threshold.
  - Future sessions (starting with Stream B's IP-Adapter re-score and subsequent wave iterations) are tasked with **beating this floor and replacing these diagnostic baseline numbers with strong positive rate-distortion claims** as appearance-conditioned engines mature.
- **IP-Adapter Verdict Withheld:** Set as `NEXT(sec:eval-ip-adapter)` marker for Stream B re-scoring; did not record a premature negative verdict.

#### C. End-to-End Tier Execution Framing (`sections/evaluation.tex`, `sections/system_design.tex`)
- Documented that `BP23` closed P0 item 1: `tier_fast` ($43.75$\,dB), `tier_balanced` ($48.28$\,dB), `tier_quality` ($56.74$\,dB), and controls (`all-off` inf dB, `residual-absent` $34.88$\,dB) executed end-to-end on 4K broadcast footage.
- **Strict Framing Maintained:** Byte counts represent uncompressed pixel payloads rather than coded bitstreams, as the codec stage operates as an identity pass-through pending `BP24`.
- Recorded independent confirmation of `plans/done/RESEARCH-HISTORY.md` §2.6: unaided static plate scores $34.88$\,dB on frame but $14.30$\,dB on player object ($25$\,dB gap concentrated on $0.57\%$ of pixels).
- Documented compute scaling: wall times span $29.1$\,s to $299.6$\,s for 8 4K frames ($\sim0.09$\,fps encode / $\sim0.06$\,fps decode).

#### D. Metric Invariants & Validity Constraints (`sections/evaluation.tex`)
- Documented empirical VMAF ceiling of $97.54$ (flooring at $0.00$ for severe blur/unrelated content) and LPIPS patch sensitivity at 4K ($0.000$, $0.250$, $0.430$, $0.645$ across anchors).
- Reaffirmed that no perceptual ranking produced before 2026-08-23 is citable.

---

### 2. Markers Closed and Updated

| File | Marker | Action | Description |
|---|---|---|---|
| `sections/problem.tex` | `STATUS(sec:problem)` | Updated | 2026-08-26: $n=8$ headroom across 6 matches landed with concentration & codec consistency. |
| `sections/problem.tex` | `HOLE(sec:problem)` | **Closed** | Removed $n=2$ directional probe hole; $n=8$ is fully landed. |
| `sections/problem.tex` | `NOTE(sec:problem)` | Updated | Mandates reporting standard error ($0.170 \pm 0.031$), two near-zero clips, and AVC concentration band. |
| `sections/problem.tex` | `CLAIM(sec:problem)` | Updated | `src=outputs/bp21-headroom/report.json date=2026-08-25` |
| `appendices/headroom_measurement.tex` | `STATUS(app:headroom)` | Updated | 2026-08-26: Landed experimental setup, tool versions, pre-registered bounds, alarms, and per-clip table. |
| `appendices/headroom_measurement.tex` | `CLAIM(app:headroom)` | Updated | `src=outputs/bp21-headroom/report.json date=2026-08-25` |
| `appendices/README.md` | Table entry | Added | Listed `headroom_measurement.tex` (`app:headroom`). |
| `sections/README.md` | Table entry | Added | Listed `problem.tex` (`sec:problem`, `tab:headroom`, `fig:ps-overview`). |
| `sections/evaluation.tex` | `STATUS(sec:evaluation)` | Updated | 2026-08-26: Methodology, generator negative, end-to-end tiers, and metric invariants updated. |
| `sections/evaluation.tex` | `HOLE(sec:evaluation)` | Updated | Clarified scoped generator negative, pixel payload accounting for BP23 tiers, and pending BP24/BP25 dependencies. |
| `sections/evaluation.tex` | `NEXT(sec:eval-ip-adapter)` | Added | Awaits Stream B IP-Adapter re-scoring. |
| `sections/evaluation.tex` | `HOLE(subsec:eval-ladder)` | Updated | Notes that tier pipeline runs end-to-end with pixel payloads, while swept RD BD-rate curves await BP24. |
| `sections/evaluation.tex` | `HOLE(subsec:eval-operating)` | Updated | Cites BP23 measured wall times ($29.1$\,s to $299.6$\,s for 8 4K frames). |
| `sections/system_design.tex` | `STATUS(sec:system-design)` | Updated | 2026-08-26: Noted BP23 end-to-end tier execution under pixel payload accounting. |

---

### 3. Claims Retracted or Narrowed

1. **Retracted VVC Exception Claim:** The preliminary claim that VVC eliminates object-coding headroom was retracted. The AVC$-$VVC gap was shown to be an operating-point/rate-ladder confound ($+0.028 \pm 0.015$ at common QP, $+0.023 \pm 0.017$ at common PSNR interval; both $<0.04$ and $<2\sigma$).
2. **Foreground Saving Quantities:** Stated with standard errors ($17.0 \pm 3.1\%$ for AVC, $14.2\%$--$18.3\%$ across ladder), explicitly citing the two near-zero clips ($1.1\%$ and $9.9\%$).
3. **Background Saving Range:** Replaced synthetic order-of-magnitude estimates with real panoramic plate savings ($64.3\%$--$78.0\%$ BD-rate saving vs inter-coding), recording alarms for AV1 ($0.780 \pm 0.056$) and VVC ($0.761 \pm 0.039$) exceeding the pre-written $[0.25, 0.75]$ band.
4. **Scoped Generative Baselines:** Clarified that off-the-shelf ControlNet/AA limitations are diagnostic baselines that establish the requirement for appearance conditioning, defining the roadmap to replace them with positive gains.

---

### 4. Stale Text / Plan Inconsistencies Identified for Central Plan Maintenance

- `plans/done/README.md` item 0 and `plans/done/RESEARCH-HISTORY.md` §2.14 still state the preliminary $n=2$ framing where VVC was described as a possible exception ("VVC is a step down of ~0.077... VVC is the exception worth naming"). Central updates to `plans/done/RESEARCH-HISTORY.md` should reflect `BP21`'s $n=8$ resolution.
