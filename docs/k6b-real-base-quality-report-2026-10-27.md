# Actual-base player enhancement: audited negative result

The frozen player-overlay recipe fails the registered whole-frame rule, even
under its optimistic rate accounting. Foreground quality improves, but the
additional side rate buys less whole-frame quality than the achievable SVT
anchor envelope. The prescribed arithmetic early stop is valid; no second
actual-base window was needed to decide this recipe.

## Decision and coverage

Job `20261027T211451Z-f6a754eb` completed with exit0 after scoring all80saved
ticks of Al031@129. Both original granted-matte and saved-prediction controls
were first reproduced on all80ticks in BOTH development windows. Med014@194's
actual-base metrics remain **unscored**, not losses. The source windows are
240frames at25fps; saved ticks are0,3,...,237. There is no full240semantic-wire
claim and no untouched-test claim.

At the frozen global scale4, base rate17.130833kbps + side floor5.607129kbps
= total22.737962kbps. Lower metric values are better.

| Metric, Al only | Hybrid at total rate | SVT envelope at SAME total rate | Strict result | Al BD-rate |
|---|---:|---:|---|---:|
| Full-frame LPIPS | 0.275786611 | 0.234056553 | loss | +31.247321% |
| Full-frame DISTS | 0.169612437 | 0.164586054 | loss | +17.331683% |
| Fixed-foreground LPIPS | 0.345875574 | 0.612627101 | win | undefined: no quality overlap |
| Fixed-foreground DISTS | 0.252551242 | 0.453136982 | win | undefined: no quality overlap |

Foreground distances are better at every rate within the measured common
rate interval15.804629–33.685000kbps; their quality ranges do not overlap,
so no foreground BD-rate is reported and no extrapolation is used.

All four central decisions survive the fixed+/-0.001 sensitivity. Both full
metrics lose in the first window, so even two Med wins would allow at most2/4,
below the required3/4. The full-frame AND foreground rule therefore FAILS.
No two-window mean, win fraction or missing-window loss is fabricated.
BD-rate is independently recomputed with PCHIP log-rate integration over only
the overlapping quality range against the achievable anchor hull. Positive
BD-rate is worse. Figures and raw arrays are in the evidence archive below.

## Independent verification

The CPU metadata/score audit passes55checks: complete80-tick arrays and means,
both historical controls within1e-4, source/composite identities, source YUV,
base-frame hashes, native IVF/bootstrap/side-rate accounting, central decisions,
BD-rate and arithmetic stop validity.

A second CPU process replays the actual frozen component using only decoded
paid bases, KPS geometry, saved generator crops and saved receiver masks. It
reconstructs all240Al hybrid images and matches all480base/hybrid hashes.
Pixels outside predicted support remain exactly equal to the paid base. It
verifies all210code files and all six input archives it consumes; it uses no
source video or target RGB. This is native component replay, not an independently
implemented renderer or a complete wire decoder. Replay cost79.090832wall/
76.897103CPU-seconds, peakRSS1987.19MiB, zero GPU. Global client/setup CPU cost
was not completely measured.

Full published archive SHA256:
`e346844d20671e5dfd78828ec61d262aa13109bc1be9a1b421873facdc86daa1`.
It matches the native publication receipt before local auditing.

## Provenance

- Job: `20261027T211451Z-f6a754eb`, gpu2 GPU1, RTX A6000 capability8.6,
  UUID`GPU-40e9a544-b3d6-e5f3-a736-7020a6401bfc`. Foreign process305859 was
  preserved; shared timing is resource cost only, not speed evidence.
- Frozen code: `01888c76ce2dd2d7ffbb5ceb63f9a28e774a41f1` (PR259), selected
  `src`, `experiments`, `tools`, `pyproject.toml`; no patches or untracked files.
  Empty selected patch SHA256`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Source archive:`0c153bba964116f8fd4e3a359224d8b9b78ddfa5a1f9193a4d7e127a0437a7ef`;
  native source identity:`be7dfb1989b2609686941b980497f02fa9f381c49318a5daf5d1b8b3068de9b9`.
- Normalized spec:`33474777347537343ecf092ac9ede1ca5f515ba16926caa020d49c4253d420dd`.
  All13input identities and the exact command are retained in spec/dispatch.
- Packed runtime:`32e68872700e4c3ffbee38425740f5f57ecf6d827a6a05329942c2ac10534661`,
  Python3.12.14, Torch2.10.0+cu128, CUDA12.8, cuDNN91002. PeakTorch GPU3.52214GiB.
  Actual CUDA metric networks and the original source/composite gates are used.
- Native base encoder: SVT-AV1`SvtAv1EncApp`v4.2.0, CRF63, preset4,240frames,
  keyint-1; decoder`dav1d`1.5.4, single thread; FFmpeg8.1.2. Original binaries
  were under`/dev/shm/emanuele-pointstream/cache/<runtime-sha>/tree/bin/`.
  Their exact paths, checksums, six encode/decode commands,240timestamps and
  container-byte counts are in the retained native base manifest.
- Encoder SHA:`4cec951d20d1b72c4c226b337b00521ad32893885931f34d9e74cfbed53a25b4`;
  dav1d SHA:`e7e2fb1818ed790606a50ed33d9a0c5bf44f57528cb673a86ec37e33fc96c7e6`;
  FFmpeg SHA:`7556f7249282a43dfaa02b776d4d6ad8b01a8865bfbca790a018a9c5d98103d2`.
- Original outputs:
  `/home/itec/emanuele/Datasets/pointstream-data/jobs/fleet/inbox/20261027T211451Z-f6a754eb/full/`.
  Independent evidence archive:
  `/home/itec/emanuele/Datasets/pointstream-data/tennis/k6b-real-base-quality-20261027-f6a754eb/evidence.tar`.
  It contains raw published scores, native metadata, analysis scripts, input
  provenance, independent audits and the figure. Its separate manifest gives
  checksums. Preserve originals and failed scratch; no cleanup was performed.

## Cost and closure

The fresh three-tick smoke passed all17checks. Native separated timing projected
1511.906996s worker +137.144188s publication =1649.051184s <2000s. The actual
full worker took1006.600970s because the registered scientific early stop fired;
this is NOT a measured two-window full-run cost.

Current admitted interval1793137556.7579308–1793138833.7093961 =1276.951465s.
Through terminal reconciliation1318.620690s; conservatively charge1320s.
Prior charges580+120+60+270=1030s, study total **2350/3600GPU-s**
(0.652778GPU-hour), **1250s unspent**. All waits ended, events acknowledged,
no recurring model monitor or automatic next experiment. Background model
callbacks and model cache lifetime remain unknown.

| Request | Terminal state | Conservative GPU-seconds |
|---|---|---:|
| 20261027T072713Z-2888cefe | COMPLETE, CPU base recovery | 0 |
| 20261027T074823Z-135421ee | COMPLETE, GPU RGB conversion | 370 |
| 20261027T080644Z-9dba0803 | FAILED original cost gate | 210 |
| 20261027T165235Z-f58a59af | CANCELLED before scientific stage | 120 |
| 20261027T180347Z-0f65e6f6 | FAILED import before scoring | 60 |
| 20261027T181542Z-b22436b0 | FAILED120s smoke limit | 270 |
| 20261027T211451Z-f6a754eb | COMPLETE, scientific early stop | 1320 |
| Total | All study requests terminal | 2350 |

Evidence archive SHA256: `335e1b00c778ce04e68203b10a9dff81ab1ad8a671908132dc18cf910d90a462`.

## Post-hoc diagnostic, not a changed decision

Keeping all measured pixels/curves fixed but making EVERY term except the
current KPS packet free still gives Al full LPIPS BD-rate+18.841831% and full
DISTS+5.737559%. Thus packaging or longer bank/adaptation amortization alone,
with the same KPS wire and pixels, does not repair this window's BD-rate result.
This is an optimistic CPU accounting bound, not a delivered codec experiment;
it does not rule out a different pose coder, renderer, base policy or objective.

At scale4 the maximum extra rate for equal full-LPIPS quality is only
0.784306kbps; the current KPS packet alone costs3.53kbps. For DISTS the extra
rate ceiling is4.604087kbps. With ALL side terms free, Al BD-rates would be
-2.467586% LPIPS/-14.134389% DISTS. These exploratory counterfactuals cannot be
substituted for the registered total-rate outcome or extrapolated to Med.

## Research disposition

Close this fixed recipe. Do not complete expensive receiver serialization,
launch another generator, or reinterpret the foreground-only gain as an
end-to-end or fair-ROI win. Learned tensor/bootstrap/framing costs remain
incomplete, DCVC and fair ROI/pixel controls are absent, and development
training/reference/source independence remains unresolved.

Before funding any new renderer, the next proposed question is a finite
ideal-player appearance bound on the same paid base and operating point.
Its [separate proposed card](cards/k6b-ideal-player-bound.md) requires a new
research decision; this completed experiment authorizes no further GPU work.
