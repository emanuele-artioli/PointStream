# Experiments: evaluation code re-audit

### 2026-11-15 — Historical player decisions and corrected native anchors

Decision question: which saved tests can guide development, and which require
same-input requalification before another expensive decision? This is source
review plus bounded synthetic CPU checks, not a rerun of historical GPU quality.
Budget: local CPU only; zero GPU time, training, downloads or mutations of source
artifacts. Starting checkout: `7f39e87e98934670c83c36c5be43478629aafdc4`.
Hypothesis: several historical simplifications limit scope but do not invalidate
all player evidence. Competing explanation: rate, mask or coverage defects alter
the decisive cells. No new scientific claim is promoted by these checks.

#### Concrete defects and prospective repairs

1. **K6b unsupported-rate fallback is a real code defect.** Previously
   `k6b.won()` called `k1.at_rate()` and substituted the *lowest-rate* anchor
   whenever it returned `None`, at either end of the measured support. A CPU
   example with anchors `(1 kbps, LPIPS .5)` and `(10 kbps, .1)` awarded a
   20-kbps, LPIPS-.3 candidate a win against .5. The affordable 10-kbps endpoint
   already beats that candidate. Below 1 kbps the fallback instead compares
   against a stream the candidate budget cannot buy. Neither is equal-rate
   interpolation. The new implementation delegates to the existing
   `k6b_confirm_validate.strict_won`: both unsupported ends return
   `resolved=False, won=None`; aggregation explicitly refuses an unresolved
   decision. Bracketed interpolation and measured endpoint comparisons are
   unchanged. This does not erase or rewrite historical reports. A separately
   labelled affordable-endpoint dominance test is legitimate; it must not be
   presented as measured equality of rates.
2. **Rational FPS was incorrectly billed in W7's rate helper.**
   `w7_rates.rates()` read only the numerator of `[numerator, denominator]`.
   A 100-byte packet over 240 frames at 30000/1001 fps was charged 100 kbps
   instead of 0.09990009990009989 kbps. It now uses the complete rational FPS
   for keypoints and window-local bank sensitivity. Historical W7 transport
   writes `[25,1]` or `[50,1]`, so those accepted charges are unchanged. This is
   a latent generalization defect, not a reason to retract W7. Tests cover
   both integer cadences and 30000/1001.

Regressions in `tests/test_k6b.py` test both unsupported ends, exact measured
endpoints and refusal to average unresolved billed/unbilled cells.
`tests/test_w7_billing_corrected.py` tests rational-duration accounting.
Focused verification: `python -m pytest -q tests/test_k6b.py
tests/test_w7_billing_corrected.py tests/test_w7_specs.py
tests/test_w7_analysis.py tests/tennis/test_w7_mkv_envelope.py` completed
successfully locally (86passed,2skipped). Configured `ruff check` and
`git diff --check` pass. Configured mypy in the current local environment
fails with22errors in9files, all outside this patch (659files checked);
this run does not establish their baseline cause. Full-suite verification and
resolution of that environment/check blocker remain the integrating session's
responsibility. No model output was regenerated.

ROOT integration subsequently resolved the environment/check question using the
existing pinned `/private/tmp/pointstream-ladder-env`: configured mypy passes
all659files, configured Ruff passes, and the complete suite reports2,628passed/
37skipped. The ambient22errors are not reproduced in the pinned environment.
Raw scientific receipts and old results remain unchanged; software verification
does not establish which old comparison cells triggered the discovered defect.

#### What the generic 50% result actually measures

`k6b.judge_window()` scores each generated composite and the K6 warp/ref
controls against the same decoded source frame. `window_truth()` authenticates
the original clip SHA256. The fixed scoring support S is R plus all K6 oracle
drawn masks; it is shared across compared images, rather than chosen separately
to hide each generator's errors. LPIPS is a spatial-map mean on that support;
DISTS uses the shared padded support crop and custom masking. These are
historical regional objectives, not independently authenticated standard scalar
LPIPS or a racket/ball task test. The composite uses the source-derived Telea
fill and granted source matte R; generated crops are pasted only inside R.
The keypoint-only rate omits that matte, the reference and the background.
Thus this is an optimistic regional appearance test, not a complete codec.

The report takes equal-frame means, then an equal mean of the LPIPS and DISTS
win shares. The recorded generic result is six of twelve window-metric cells,
not a probability that a random sports video will work. Three of six windows
win both metrics. With two unrendered windows winning both, ten of the original
sixteen cells is below the frozen twelve-cell pass requirement: the stopping
arithmetic is correct **conditional on the six observed cells being valid**.
The six were selected by completed rendering coverage under the spent budget;
the two omitted windows were not silently counted as measured failures.

I did not locate the raw historical K6/K6b report JSONs or their anchor hulls
in this checkout. The documented report addresses point into external saved
results. Consequently this audit cannot identify which, if any, of those twelve
cells used the endpoint fallback, cannot certify a revised percentage, and
does not claim the 50% result has reversed. The minimum necessary next check is
the saved per-cell rate-support audit below. The raw generated/source metrics
remain useful observations even if a comparison cell needs relabelling.

**Subsequent ROOT saved-input check:** the original generic report was retrieved
read-only at its recorded SHA256 `acd27678226bd3208422b56bbb6ae26bbc60b9828e9152cf36315c37cc42ca43`,
and the referenced K6 anchor report matched
`22cb71c741d2870dfd3f646ca55f6ffbed16fdc275f86174f5103cd8635ee0ad`.
All12 generic cells are inside supported rates and retain their original labels
(six wins); the endpoint bug did not activate in that result. Three further
original report hashes matched the documented step3/4/5 records; all60 checked
adaptation cells retain their labels. See [generic receipt](k6b-original-saved-support-reaudit-20261115.json)
and [adaptation receipt](k6b-adaptation-saved-support-reaudit-20261115.json).
This resolves the saved-rate-support question, not profile coverage, mask/source
identity or upstream preprocessing. No report or generation was overwritten.

One further validator weakness remains: `judge_window()` records
`s_masks_match_k6`, but `validate_judge()` checks equal pixel counts and warp/ref
LPIPS reproduction, not that recorded exact mask-hash equality. Equal counts
alone do not prove equal pixels. Later actual-base scoring explicitly checks
both R and S hashes. No observed historical mask mismatch was established here.
The generator's missing-profile fallback and unit/frame coverage also require
receipt inspection before attributing every score to the requested renderer;
coverage greater than .5 is not complete output identity.

#### Actual-base pairing and early stopping

`k6b_real_base_judge.run()` builds the three scale bases and their paired
hybrids from the same native source, scores all six images against the same
truth, and verifies source RGB frame hashes and fixed R/S hashes. It scores
saved ticks (normally 0:240:3), not all 240 decoded frames. The complete IVF,
KPS and bootstrap bytes are divided by the full 240-frame native duration;
the same duration applies to anchor and hybrid. This is consistent for a
saved-tick quality experiment, but cannot certify between-tick reconstruction.
Bank/adapter amortization is a model of repeated use, not a measured receiver
bootstrap lifecycle. Source masks used to define evaluation support are not
automatically a paid decoder matte; receiver masks come from the saved
prediction path.

The legacy early stop is evaluated only after finishing a complete window:
both LPIPS and DISTS lost in either full-frame or foreground family at scale4
make the registered three-of-four criterion impossible for that window.
Breaking the outer loop skips later windows, not difficult frames inside the
current window. W5's `--no-early-stop` explicitly obtains the skipped second
window. This legacy stop cannot be repurposed as evidence of a complete W7
tier; full validators expect the declared window/tick coverage.

#### Billing corrections, omissions and the latest native-IVF path

The original W7 bank mistake is already corrected: eight `reference_files`
entries were two physical crops with four codec alternatives each. Dividing
PNG bytes by eight undercharged the intended 12-crop bank by fourfold.
`w7_rates` now counts PNG entries and verifies sums/uniqueness.
`w7_billing_corrected` SHA-verifies immutable archives, shifts the scale4 total
rate, and recomputes comparisons from the original scored policy rows. Its
finite-L branch replaces the old bank+adapter charge, leaving recurring base,
KPS and bootstrap charges alone. Quality is reused, not regenerated. Twelve
crops with these sizes supporting quality for L seconds remains an assumption.

Old `w7_analysis.analyse_window()` intentionally uses every *resolved* available
family and lists others as excluded; it does not require the complete intended
family set before returning a positive group. A missing med000 DCVC or
unbracketed ROI policy can weaken the anchor. It also makes mismatching
judge/score summaries warnings rather than hard admission errors. These are
limits of using that helper standalone; immutable input/source audits must
precede it. `w1_roi.bracketing_pair()` requires a strictly lower point and an
upper point at or above target: even the lowest exact endpoint is unresolved.
That conservative frozen historical policy is distinct from a bug or from the
new affordable-endpoint policy. It never extrapolates.

The latest accepted Tier1 analysis is **`w7_mkv_envelope.py`**, not the old W7
JSON or the first invalid rerate report. Its source path requires 53 historical,
18 control and four new roles, checks native-YUV/IVF/score identities, and removes
only four proven duplicate streams. The native-YUV proof chain is in
`w7_native_yuv_final.py`. Primary costs are complete original IVF; actual MKV
delivery costs are separate sensitivity. `achievable_at_rate()` correctly
returns no affordable anchor below minimum and the *best* affordable endpoint
above maximum, explicitly labelled. Between knots it uses the established
modeled hull interpolation. Coverage is per foreground metric, with all six
windows required for the pooled interval. The old strict-bracket family view
remains separate. These checks address the earlier endpoint/coverage defect;
do not apply the newly found K6b fallback bug to this repaired path.

The accepted record `w7-inputs/ivf-mkv-corrected-tier1-report-20261105.json`
retains positive pooled player gaps; med000 DCVC is still absent, the Medvedev
subgroup is inconclusive, and all six whole-frame gaps favor anchors. The
unseen-match Tier2 result is separate and its LPIPS interval is inconclusive.
This review inspected the implementation and accepted report lineage; it did
not download or independently replay the 75 native streams. Modelled envelope
interpolation is not physical time-sharing proof, and deterministic ±.001
stress is not a GPU-distribution confidence interval.

Newer `codec_ladder.analysis`, `ball_roi_control.analyse_score` and
`src.pointstream.evaluation.anchor_sensitivity` use literal native IVF versus
complete candidate packets, require complete registered receiver coverage,
and integrate log-rate PCHIP only on overlapping quality support. Null overlap
does not become a BD-rate. Legacy spatial-map and separately authenticated
standard scalar LPIPS must remain distinct. These current paths cannot
retroactively validate old source identities or expand the player claim.

#### Bounded same-input requalification before development decisions

1. CPU first, no encoding or training: retrieve the two saved generic judge
   archives and original K6 report with full SHA256s; freeze source revision,
   converter/network identities, judged ticks, source/mask hashes, native stream
   bytes and every historical rate denominator. Preserve old files read-only.
   Emit one new twelve-cell table with rate support, bracket/endpoint status,
   requested and actual renderer, coverage and original scalar margins.
2. Recompute only CPU interpolation and frozen decisions. Report the generic
   six-window percentage and original eight-window attainable bounds separately.
   Unresolved cells stay unresolved; a changed result must identify the precise
   old fallback cell. For W7, independently replay the accepted *corrected*
   native-IVF CPU report with its pinned numerical runtime, retain original
   families and missing DCVC, and reproduce controls/duplicates/whole-frame
   losses. No fresh GPU scoring is needed if authenticated score joins pass.
3. Only if source, mask, metric or renderer identity fails, freeze a bounded
   same-input scorer replay before spending GPU time: one representative window
   with plain base, no-player identity, source-self controls and original
   renderer outputs, then complete only affected saved windows. Keep every
   stage below45minutes, with an explicit released budget and output directory.
   Require all registered frames for a new complete-codec decision, actual paid
   reference/matte/bootstrap assets, fixed shared metrics and complete bytes.
4. Use this requalification to decide whether a *specific* development route
   deserves a new pilot. It cannot turn opened old windows into confirmation,
   validate generalization to other sports, or justify another training sweep.

Stop after the receipts classify the disputed cells and the supported decision
is clear. If archived inputs are unavailable, retain that exact limitation;
do not replace missing evidence with a cheap stand-in or a broad invalidation.
