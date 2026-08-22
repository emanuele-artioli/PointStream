# Deferred — real work, deliberately not now

Things that are genuinely worth doing, are not blocking the submission, and
would otherwise be lost. Each says what it is, why it waits, and what it costs.

Nothing here is a nice-to-have wish: if an item stops being true, delete it.

---

## D1 — 61 mypy errors in the component tests

**What.** `mypy --config-file pyproject.toml` reports 61 errors, **all** in
`tests/components/`, none in `src/`. Mostly `dict[str, object]` where a typed
protocol is expected, plus `type: ignore` comments carrying the wrong error code.

Distribution: `test_background.py` 24, `test_rigid.py` 15, `test_domain.py` 13,
`test_temporal.py` 4, and one or two each in `test_metrics.py`,
`test_segmentation.py`, `test_generation.py`, `test_detection.py`.

**Why deferred.** It blocks nothing that produces a result, and the source tree
is already clean.

**Why it still matters.** `AGENTS.md` requires mypy clean before merge, so this
is the one thing standing between `phase-b/integrate` and a tidy merge to main.
Every session that runs mypy meanwhile has to know these are pre-existing, which
is exactly the kind of noise that hides a real error later.

**Cost.** An hour or two, mechanical. Good first task for a spare parallel slot.

---

## D2 — SAM3 cannot load

**What.** `ModuleNotFoundError: No module named 'torch.nn.attention'`. Torch
2.2.2 is too old; SAM3 needs a newer torch. Both `detection/sam3` and
`segmentation/sam3` fail construction, honestly and with a stated reason.

**Why deferred.** It blocks `PLAN.md` §7 **P1 item 10** (detector comparison
including SAM3) and nothing in P0. YOLO26 loads and runs, so the pipeline has a
working detector and segmenter.

**Why it still matters.** SAM3 is the open-vocabulary arm, and P1 item 16
(open-vocabulary versus hand-written selection) depends on it too. Losing both
costs the paper a comparison a reviewer may well ask for.

**Cost, and the trap.** *Do not upgrade torch in the `pointstream` env.* Several
forked models here are version-sensitive and a stray upgrade breaks them
silently — this is a standing host rule. The fix is a **second conda env** with
newer torch, and SAM3 invoked across the process boundary. That is a day of
work, not an afternoon, which is precisely why it waits.

---

## D3 — The AVC region arm is unverified

**What.** `roi.py` records ffmpeg's `addroi` filter for AVC as unverified. AV1
and HEVC have native delta-QP maps that have been driven; AVC's has not.

**Why deferred.** AVC is the speed rung, not the quality anchor, and the
region-controlled comparison that matters most is against AV1 and HEVC.

**Why it still matters.** `NOTE(sec:evaluation)` item (c) commits the paper to
giving every baseline region control *wherever its encoder supports it*. If
`addroi` works and we did not use it, the AVC comparison is weaker than it
claims; if it does not work, the paper must say so rather than leave it silent.

**Cost.** An afternoon: encode one clip with and without, confirm the bitstream
actually differs in the labelled region.

---

## D4 — MOFA-Video, and the trajectory arm

**What.** Registered as a candidate; construction refuses because its SVD weights
are Stability-AI-licensed and not bundled.

**Why deferred, and probably permanently.** `PLAN.md` §6.2 routes around it by
rendering sparse trajectories into the ControlNet backbone, which is a *better*
experiment — it keeps `eval-object`'s backbone genuinely fixed, which a
MOFA-vs-ControlNet comparison could not.

**Revisit only if** the rendered-trajectory arm turns out to be a strawman, in
which case a licence-cleared trajectory model becomes worth the trouble.
