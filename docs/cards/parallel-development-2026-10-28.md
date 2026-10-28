# Approved parallel development batch

Human approval: “Go ahead with these parallel tasks.” (2026-10-28).
This is a NEW allocation, separate from the closed 3600-GPU-second study
(2980 spent, 620 unspent). Do not borrow between ledgers or count old evidence
as a new run. No training, generation, downloads or paper submission authorized.

## Goal and resources

Run independent development diagnostics concurrently, then report all results
together. Maximum cumulative new GPU reservation: **7200 seconds**, divided
into temporal-background 2700, pixel-corrections 2700 and baselines 1800.
Each includes setup, smoke, full, validation and publication. Separate CPU
analysis cap: 3600 CPU seconds. Encoding done during a GPU reservation is
charged to that task's GPU allocation. No unapproved budget transfers or retries
of scientifically failed configurations. Infrastructure uncertainty is inspected,
never replayed. No recurring model monitors.

All input files and tool/weight versions are frozen by SHA256 before dispatch;
source revisions and exact stage commands accompany each job. Use fleet resource
admission, compatible host pools and host-local scratch. Timing variation is
secondary, while correctness, finite output, memory, byte accounting, deadlines
and total budgets remain hard gates. Each wait/stage ends within 45 minutes.

## Independent tasks

1. **Temporal background, 2700 GPU-s.** Test a paid compressed reference with
   explicitly specified camera transform and coded residuals, at three frozen
   rate settings. Identity camera motion is a restricted static-reference
   candidate, not an arbitrary court panorama bound. Pay reference, transforms,
   residuals, initialization and any retained foreground overlay side data.
   Compare actual candidate pixels; an original-pixel substitution cannot
   establish a universal LPIPS/DISTS upper bound.
2. **Pixel corrections, 2700 GPU-s.** Deliver compressed player rectangles over
   the decoded base at three frozen rate settings. Charge geometry, payload,
   headers and all decoder inputs; original masks may inform encoder cropping
   but cannot be secret decoder inputs. Remove only side terms actually replaced.
   A zero-payload rate relaxation may gate only those exact reconstructed pixels.
3. **Baselines, 1800 GPU-s.** Whole-frame DCVC-UF and a fair SVT ROI comparator
   on the same source/color/timing path; three initial operating points. Existing
   valid encodes may be reused with complete identities. An encoder-only source
   mask ROI is explicitly labeled and provides no free decoder pixels. A failed
   codec/precondition is reported, never silently substituted.
4. **CPU gap/accounting, 3600 CPU-s, no GPU.** Reuse saved raw scores to invert
   the achievable hull, quantify fixed-pixel side-rate ceilings and robust
   quality gaps, and audit confirmation split prerequisites. No new quality
   measurement or mathematical bound on arbitrary methods. Do not claim an
   untouched manifest exists without auditing footage and training/judge overlap.

## Cohort and decisions

Development windows: Al031@129, Med014@194 (the existing two-window identities
must be verified from the frozen study; do not infer timestamps from labels).
Native 240 frames, original per-window fps and source conversion; score the
same 80 ticks 0:240:3 at full 1920x1080. Smoke uses a representative prefix
through the same native paths, including every tested policy. If exact second
window inputs are unavailable or cost projection fails, stop or report the
bounded first-window diagnostic; never imply a two-window result.

Quality: whole-frame LPIPS and DISTS, lower is better, against the achievable
anchor envelope within overlap. A promising setting must beat BOTH by more
than 0.001 using the SAME policy/rate. Report each window separately; do not
mix metrics/settings or extrapolate. Three points support BD-rate only when
quality overlap and PCHIP conditions are met. This exploratory batch selects
directions; it cannot be relabeled untouched confirmation or TOMM readiness.

Stop on failed correctness/cost gate, scientific infeasibility for the tested
construction, exhausted allocation or completed decision. Do not continue an
appearance-only renderer under the closed old grant. Report jobs, SHA256s,
native tools, raw results, caveats and actual charged time together after all
tasks terminate; preserve completed evidence read-only.
