# Ada calibration budget revision — November 13

Status: prospective code build only, 300 workload CPU-seconds / zero GPU.
Root allocates this from unallocated Ball/programme headroom within the approved
24 GPU-hour / 28 workload CPU-hour programme and Ball 64,800 GPU / 72,000 CPU
ceilings. Earlier charged ceilings and all remaining holds are retained.

The actual final-source Ada witness d1bb8926 passes the six original isolated
decoder/native pixel checks. Its 16-frame namespace wall time is 227.148307 s.
The unchanged conservative calibration formula gives 1.5*3*wall+50 = 1072.167382 s,
which fails the frozen 580-second gate. Existing receiver timers cannot establish
enough frame-independent overhead to subtract it. Preserve this failed forecast;
do not run another old-source calibration merely to reproduce the known failure.

Build the minimum prospective calibration budget revision: keep the formula and
all source, isolation, device, fidelity, quality and actual-evidence checks;
allow at most 1200 seconds for the full 48-frame namespace, 1220 for its fleet
stage, and a finite 1800-second campaign. Smoke remains at most 600 seconds and
all stages remain below 45 minutes. Complete remaining-time checks must include
the declared full-stage and reporting/validator reserves; constructor and worker
must agree. Tests must cover old-cap rejection, new-cap admission, insufficient
remaining time, and the maximum new cap. Do not loosen scientific rules.

Work in an independent checkout from current main. Required checks and focused
tests are inside this 300-CPU build allocation; stop or revise before exhaustion.
Commit/push/open a CI-gated PR after root review. Do not update the frozen ae892
worktree or alter its accepted evidence.

This card authorizes no remote execution, model, new scientific qualification,
calibration or study. A changed source/archive needs new exact-source acceptance
before calibration. The separate queue-controller candidate remains unreleased;
client PR383 is merged but does not establish complete orchestration correctness.
Root must review the complete source/preparation/qualification/admin/observer/
calibration forecast and resource protocol before any further compute release.

The registered 30-clip study remains the goal only if measured paired 16/48
prepare/score costs and all overhead fit both unallocated programme and Ball
headroom. The predeclared fallback is the two registered full-context clips
game9/Clip5 and game10/Clip12, with all six points and independent metrics;
selection must not depend on new quality outcomes. Preserve failed feasibility
and negative outcomes beside any narrower claim.


Implementation0118786c is committed in isolated checkout /private/tmp/pointstream-ada-calibration-budget-20261113. Root reviewed worker/test diff: no scientific-rule change; full-stage remaining-time reserve is1350s (1220+30+100). ExactCI Ruff0.11.2 andmypy1.15 pass594files; Python3.12.7 focused25testsPASS. Wrong-toolchain fullpytest was stopped andretained, no fullsuitePASS claimed; exact fullsuite delegated toPR CI because projected210CPU didnotfitremaininglocalallocation. Measuredchecks165.384362CPU, conservativewholebuild180/300CPU, retainfull300charge. Report /private/tmp/pointstream-Ada-resource-window-20261113-new/scientific-audit/budget-build-checks/build-report.json. Pushedbranch; no runtime/qualification/GPU release.


PR384 passed fullCI (3m17s) andmerged2c6d8581fbd64279dde554ca1402047d27ab7fa2. That exactcommit is nowfrozen in /private/tmp/pointstream-ada-calibration-qualified-source-20261113; source6f898f75/archive5ca2d194/confige478. Newsourcequalification is tracked in [its card](k5d-Ada-calibration-source-qualification-2026-11-13.md); oldae892/d1bb evidence remains unchanged. The source/cache stage passed independently; native CPU and actual SVT CPU-denial evidence passed their scoped root reviews. No new GPU/calibration release.
