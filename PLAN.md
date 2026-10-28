# Plan

Each workstream is scoped to one session. Items that do not depend on each
other run in parallel sessions; each item names what it waits for (user,
2026-10-16). This file is the source of truth: a session card only starts
an item already written here, and nothing exists only as a card. Work runs
in two phases
([decision](docs/experiments.md#2026-10-06-egocentric-first-racket-sports-second)):

- **Phase 1, egocentric hand-object video.** VISOR is the only adopted dataset
  with native video and dense human masks of the whole foreground: hands with
  forearms, and active objects. It carries the end-to-end codec evaluation.
- **Phase 2, racket sports**, once phase 1 has a result. No racket dataset has
  dense foreground masks, ball positions and video together. Players and the
  ball are evaluated in separate stages, each on the dataset that labels it.
  Together the stages estimate the performance on a fully labelled dataset.
  If phase 2 is not finished by submission, the paper presents it as future
  work.

Every dataset is used for what it labels ([resources](docs/resources.md#what-each-dataset-is-used-for)).
The environment audit (step 0) comes before the first component that needs a
new dependency.

## Where PointStream stands (2026-10-28)

**Parallel development batch CLOSED/REPORTED (2026-10-28):** human authorized
three concurrent GPU diagnostics plus CPU gap/accounting. New separate cap
7200 GPU-s (temporal background2700, paid pixel corrections2700,
DCVC/ROI baselines1800) and3600CPU-s. Charged2400/7200GPU-s,4800unused;
CPUanalysis charged120/3600s conservatively. The old2980/3600ledger remains closed.
Read [batch card](docs/cards/parallel-development-2026-10-28.md) and
[live batch record](docs/experiments/parallel-development.md) for exact frozen
settings, IDs and outcomes. All five requests terminal, no claims/processes/timer.
No old job restarted. No training/generation. Read the
[consolidated report](docs/parallel-development-report-2026-10-28.md): motion
improves static background but LPIPS BD-rate+230.28% on overlap; pixel full
Al/Med points beyond measured anchor coverage; DCVC native decode failure,
SVT independent cost stop with no full evaluation. No validated baseline win.
Background covered Al; pixels covered both development windows. Untouched confirmation
and TOMM readiness are not established. Future delegated job launch/report tasks
use GPT-6 Luna at max effort with precise narrow instructions (user,2026-10-28);
root owns experiment design/review/decisions.

The root session keeps this section and the dispatch ledger below; item
sessions update their own items and their files in `docs/experiments/`
([sessions](docs/sessions.md)).

### Resume here after the root chat is archived

Read [the 2026-10-27 continuation](docs/continuation-2026-10-27.md) first. It
contains the current scientific stop, exact job/artifact identities, remaining
budget, storage state and the next decision. The existing study chat is
`01a16e7a-e752-78f3-b372-c338d8cdbf46` ("Prepare tennis confirmation study");
its code and reports are merged. It is idle. Do not create duplicate studies or
restart any terminal job. Pace active sessions to conserve tokens; do not launch
several reasoning turns automatically. All resumed requests are terminal; no research job or recurring model monitor
is running. Read the current ledger below before taking action.

**Current scientific position:** no complete PointStream win over both required
baselines has been established. The fixed player-overlay recipe now has an
[audited real-base negative result](docs/k6b-real-base-quality-report-2026-10-27.md).
At equal total rate on Al031@129, full LPIPS/DISTS lose (+31.25%/+17.33% BD-rate),
although fixed-foreground quality wins. Both full cells lose robustly, making
registered >=3/4 impossible. The prescribed early stop leaves Med actual-base
metrics UNSCORED, not losses. Historical granted-background4/4 results do not
transfer into a whole-frame paid-base win. Only80/240saved ticks were evaluated;
these are familiar development windows, not untouched confirmation.

**Study closed:** final job20261027T211451Z-f6a754eb is COMPLETE with a valid
scientific early stop. Independent score/rate audit passed55checks, and CPU
replay matched all240hybrids/480image hashes. Both control windows completed
all80ticks. The registered recipe cost2350GPU-s; including the follow-up diagnostics below,
the shared ceiling ledger is now **2980/3600GPU-s**, **620s unspent**.
All waits ended and events were acknowledged. No new GPU request is authorized
by this result; do not complete receiver serialization or try another generator
to evade the failed rule. Source01888c76/spec33474777 remain immutable.

**Ideal-player full evaluation COMPLETE — finite NO-GO:** user-directed
job20261028T080453Z-34e50ec2 scored all80Al ticks after making timing
variation secondary. Controls exact; CPU audit400images/160oracle compositions
passes. Even source pixels inside R lose full LPIPS0.270927 vs SVT0.234057
and DISTS0.167818 vs0.164586 at same total22.737962kbps, robust+/-0.001.
The other fixed support also loses both. Do not fund an appearance-only
renderer at this point without a new dominating background/rate bound.
That stage closed at2730/3600GPU-s; the later diagnostic below updates the ledger. Read
[the full report](docs/k6b-ideal-player-full-report-2026-10-28.md); no active job.

**CPU follow-up COMPLETE (2026-10-28):** fixed-pixel rate inversion finds
only0.668734kbps robust total-side allowance for the actual scale4 overlay,
or1.230365kbps even with source pixels inside R. Current KPS3.53 and other
side2.077129 each exceed the latter. Three lossless wrappers all grow KPS.
Read [the feasibility report](docs/k6b-side-feasibility-report-2026-10-28.md).
The subsequent paid clean-background test is complete below.

**Paid clean-background full evaluation COMPLETE — two-metric gate FAILED:**
CPU preparation20261028T091736Z-bf9f97d2 and single GPU request
20261028T100153Z-27f39d64 completed. The same request's first publication was
recovered after a transfer timeout/usage interruption; no replacement or retry.
All80Al ticks scored, original source/score controls exact; independent audit
passes240goldenRGB checks/160new pixel reconstructions and raw rate/decision.
Complete base saves898bytes (20557to19659), total21.989629kbps. Source-pixel
oracle LPIPS0.268848 loses to SVT0.240532, while DISTS0.166890 beats0.168336.
Formal MIXED/UNRESOLVED; neither new policy wins both by>0.001. Close this
fixed construction, not arbitrary future backgrounds/renderers. Read
[the full report](docs/k6b-paid-clean-background-report-2026-10-28.md).
New250GPU-s charge; global **2980/3600GPU-s**, **620s unused**. Native jobs/
waits ended, events acknowledged, own PIDs gone; no more GPU authority or timer.
DCVC, fair pixel ROI, complete wire and untouched evaluation remain missing;
no complete baseline win or TOMM readiness is established.

**Infrastructure:** large inactive `/var/tmp` files were relocated to separate
host-local `/local/users/emanuele` storage on all six hosts, retaining NFS as
source of truth. The shared shell defaults are active and verified. Live
application originals remain intentionally protected. Fleet manager safeguards now run on gpu2/gpu3 after the
verified scoped refresh; the other four workers remain unchanged. See [storage closure](docs/storage-maintenance-2026-10-27.md).
The detailed historical findings follow; they do not supersede this resume state.

### Historical findings and completed components

- **Protocol (2026-10-22):** two tracks, fidelity and experience. J
  (2026-10-22) re-judged the queued results on experience: H3e, FoundHand
  and K6 `ref` fail, H3a/H3a2 were not re-judged (outputs not kept). The
  binding check is time: objects driven by coded keypoints change 1.3–6 ×
  as much as the source between frames, and K1f's bound (`warp`) fails it
  too. J's jitter follow-up (2026-10-22, PR #218): smoothing the coded
  keypoints at the decoder (σ = 1, 2, 4 frames) does not make the bound
  pass, and neither do smoothed *uncoded* keypoints (2.6–2.9 × the
  source's change against 1.25). Either the estimator's per-frame noise
  (1.32% of the box, against the source's own motion of 2.24%) or K1c's
  compositing (Telea fill, drawn outline) causes it. An identity warp
  would separate the two (≤ 0.3 GPU-hours; open for the user).
- **Hands, keypoints (K1f, K6b):** the bound (appearance granted) beats
  the pixel hull on 30 of 30 held-out items by LPIPS and DISTS. No real
  renderer keeps it: FoundHand zero-shot fails (17%), a bank of in-window
  references fails by one item-metric (67% at K = 16), and the buildable
  bank chosen outside the window fails (2c, 12.5% at K = 16–64; free,
  25%). References from other moments miss the forearm, sleeve and
  lighting. The reference route closes at a buildable rate. Rung 3
  (2026-10-23, user approved): FoundHand fitted 20 min on the window's
  own frames, weights unbilled, **passes at 83%** (LPIPS 2 of 3, DISTS
  3 of 3; 75% clean of G, at the bar; margins 0.006–0.010 on the hard
  items). Rung 4 (2026-10-23, user's rule: the fit moved to other shots
  of the video, delivered beforehand) **fails, 25%** (0% clean of G):
  none of rung 3's gain transfers, and it lands on rung 2's zero-shot.
  **Hands close on fidelity at the keypoint rate**; K1f's bound stays a
  bound. Every hand result also granted R, which the decoder does not
  have (H3b2: 72–170 kbps).

- **Egocentric pipeline (H5):** runs end to end on the 4 pilot items
  (`src/pointstream/`, `experiments/visor/h5.py`). Against fairly
  configured off-the-shelf codecs, venue models unbilled, F granted:
  weighted LPIPS −22% (4 of 4). At equal adaptation (H5q, closed
  2026-10-21) against a whole-frame DCVC-UF adapted the same way, neither
  billed: **DISTS wins** (weighted −15%, frame −22%, 4 of 4; also against
  the full fair envelope), weighted PSNR wins (−11%), **LPIPS loses** (+6%
  at best). The LPIPS cost is structural (F coded in both streams) and
  follows F's share of the frame, median 24% on egocentric video; it
  wins below about 10% (P09_106, 2%: −15%). Billing the mask F costs
  most of the margin (H5m). **H5p 4–5 done (2026-10-22): billing F
  removes every win.** With F coded (lzma, 36–116 kbps), weighted DISTS
  reads +93% (1 of 4) against the venue-adapted DCVC-UF, and every
  metric loses against every anchor. F̂ itself costs no quality; the
  mask's rate does, at rates where the venue codec needs 13–26 kbps. A
  carrier inside the foreground stream would have to cost about 2 kbps,
  and an exact F at zero bits still loses LPIPS (+16%). H5p 4c measured
  one (a flat key colour outside K, 5-byte segment): F̃ is as accurate as
  lzma's, but the key's border costs 26–129% more foreground rate where F
  is large, so weighted DISTS reads +1% (1 of 4) even as an optimistic
  bound; on P09_106 (F 2%) the key is nearly free and reads −30%. H5p 4
  closes as a fail: no F coder tried keeps the egocentric win once F is
  paid for. The key carrier is the candidate for small-F domains. The crops for the
  visual check are ready (user). The egocentric claim is the user's to
  reframe; small-F domains (tennis) are where F's rate matters least.
  H5p 4d (2026-10-24): binary shape coding costs 10.9–24.8 kbps and
  still fails (+16.6% weighted DISTS); decoder corrections cost 13.3–61.3
  kbps at one point per item. The full decoder curve is blocked by native
  decode instability; exact free masks still do not fix LPIPS.
- **Background (G):** a scene model adapted to the venue, sent beforehand
  and unbilled (payback hours reported), plus a per-video enhancer route
  (G6).
- **Foreground (H):** hands and handled objects are coded pixels, DCVC-UF
  at ¼ or ½ plus the generic enhancer G_HO. Keypoints lose on egocentric
  video (H1, H3, H4, K1, K3).
- **Tennis (K):** the natural small-foreground domain (players a few
  percent of the frame). The ball mode wins (K5); court-view shots are adopted
  as K8b (2026-10-21): on 69 held-out shots from three broadcasts,
  precision and recall are 1.00, with a registration median of 0.77 px and
  p95 of 2.71 px; homographies are published for all five of K4b's
  matches. K4b done (2026-10-21): players found by racket contact plus the
  court outline, the far player fitted on ×1 crops, keypoints 3.3 kbps for
  two, so K6's gate passes. K5b done: BallTrack adopted as tracker and
  judge; the ball mode kept with the streak renderer (user). K6 done
  (2026-10-21): with true appearance at the coded pose the players'
  keypoints save 95% of the pixel coders' rate, but a reference frame
  warped to the pose loses to every pixel coder. K6b step 2 (2026-10-22):
  AnimateAnyone zero-shot, one unbilled reference and R granted as its
  matte, wins 3 of 6 windows by LPIPS and by DISTS (50%, fail). It wins
  all three Djokovic windows (up to 73% saving) and loses both Alcaraz
  windows and Rublev's. Step 3 (2026-10-23, user approved): a LoRA
  fitted 7.5 min per window on its own frames **passes at 83%** (LPIPS 4
  of 6, DISTS 6 of 6; rate savings 35–84% where it wins), reference,
  matte and LoRA (49 MB) unbilled. Step 4 (a fit per match, billed) is
  Step 4 (2026-10-23): a rank-4 LoRA per match (1.55 MB, 8-bit LZMA,
  1.73 kbps over 2 h), fitted on other shots of the match, **fails**:
  67% unbilled, 58% billed (losses: alcaraz_hurkacz/045 and
  rublev_auger-aliassime/052). About 60% of step 3's gain transfers
  (hands: none), and payback is 0.4–2.1 h where it wins. **The
  generative route for players closes at the keypoint rate by the
  rule**, with the matte and reference still granted. Step 5
  (2026-10-24, user approved): a tennis base LoRA (K6's windows
  excluded, unbilled) plus a 4-bit rank-4 match LoRA with the pose
  guider (1.65 kbps over 2 h) **fails at 67%, billed and unbilled**.
  Billing no longer costs a window. The same two windows (045, 052) lose
  in steps 3–5, so each recipe change moves the margins, not the score.
  The frozen **clip-local diagnostic (2026-10-25)** on 12 new archive
  windows from the familiar Alcaraz and Medvedev broadcasts reaches
  **23/24 billed window–metric wins** (LPIPS 11/12, DISTS 12/12;
  descriptive bar 18/24), and 24/24 unbilled. Billing here covers the
  match adapter/guider and keypoints, with a common shipped tennis base;
  references and exact target matte remain granted and unbilled.
  Charging the 48,878,567-byte base over 2 h adds 54.309519 kbps:
  0/12 bracketed cells win, and the other 12 are unbracketed, unresolved.
  This is not a fully charged codec saving or independent-source
  confirmation, and does not overturn step 5's historical failure.
  Original-source/replay independence remains unresolved; every window
  passes registration, so gate discrimination is untested and 115 px
  remains exploratory. All 120 anchor points and 956 mask identities
  validated; 24 terminal GPU jobs used **6.995428306 of 8 GPU-hours**.
  Report commit `72228816`, [PR #236](https://github.com/emanuele-artioli/PointStream/pull/236)
  (merged in `da6a42b1`); no jobs remain active.
  The 2026-10-26 two-window compression pilot preserves actual quality with
  unmasked video-coded reference banks: 4/4 strict metric cells, robust to
  ±0.001 renderer sensitivity. Segmented alpha and black-key references each
  reach only 1/4; dense foreground carriers each 0/4. This does not establish
  >=18/24 on the full cohort. Exact dynamic matte is still granted: charging
  its measured packet removes all four pilot wins. The new pilot used
  4.411019/4.5 GPU-hours, including failures and recovery; no jobs remain.
  Joint saved-render follow-up (2026-10-26): pairing the new compressed-RGB
  generations with paid1/8,1/16,1/32 mattes gives0/4 at every scale, with the
  exact/granted control reproduced at4/4. Small RGB boundary ramps damage shape
  even before compression; separate luma ramps are accurate but not cheaper
  on the bounded excerpts. The pilot stops; it does not imply a full-cohort
  bound or global compression impossibility. It used0.103706/0.5 new GPU-hours.
  Step 6 (Wan-Animate 14B, zero-shot first) remains outlined and waits for
  the user. K6c after K6b; K7 (HiEve) later.
- **Paper (I):** notes on what to state, the negative results, the order
  of writing; the 34-item final runs wait for the figure list.

### Dispatch ledger

The single record of open work: one row per item session, with its card,
branch and status. Cards are prompts in [docs/cards](docs/cards/README.md)
that the user pastes into a new session (Claude Code or Codex). From
2026-10-24 there is no standing root session: each item session sets its
own row to `running` when it starts and to `reported` with its outcome when
it stops, and updates its item's bullet in "Where PointStream stands". A row
marked `card ready` is waiting for the user to start it.

| item | card | branch | model / effort | status |
|---|---|---|---|---|
| J | experience-track re-judging | `claude/j-experience-rejudge` @ `a7cec43` | Sonnet | reported 2026-10-22: 3 fail (time, content), H3a/H3a2 not kept; 1.25 GPU-hours of 4 |
| J, jitter follow-up | coded keypoints' jitter, smoothing oracle | `claude/j-experience-rejudge` @ `6a1b09a`, PR #218 | Sonnet | reported 2026-10-22: smoothing fails; estimator or compositing; 0.53 GPU-hours of 1 |
| H5p 4–5 | F in the foreground stream; visual check | `claude/h5p-f-stream` @ `3b11a36` | Opus | reported 2026-10-22: 4a fail (F billed +93% wDISTS), 4b not built (LPIPS), crops ready; 2.6 GPU-hours of 4 |
| H5p 4c | key carrier for F (user's request) | `claude/h5p-f-stream` @ `41c37c8`, PR #220 | Opus | reported 2026-10-22: fail at the gate (+1%, 1 of 4; P09_106 −30%); 1.38 GPU-hours of 3 |
| K6b hands 2c | buildable reference bank | `claude/k6b-hand-bank` @ `505f5a7` | Opus | reported 2026-10-22: fail (12.5%; free bank 25%); 2.6 GPU-hours of 3.5 |
| K6b hands 3 | overfit oracle (user approved) | `claude/k6b-hand-bank` @ `3059660` | Opus | reported 2026-10-23: pass (83%; 75% clean of G), R granted; visual check supports it; 3.1 GPU-hours of 4 |
| K6b hands 4 | the fit moved to other shots (user's rule) | `claude/k6b-hand-bank` @ `fd83eb7`, PR #221 | Opus | reported 2026-10-23: fail (25%; 0% clean of G); 1.3 GPU-hours of 3. Hands closed on fidelity |
| K6b hands 3m | rung 3 with a decoder-side matte | — | Opus | withdrawn 2026-10-23: rung 4 failed with the matte granted |
| K6b players 2 | AnimateAnyone zero-shot oracle | `claude/k6b-players-zeroshot` @ `340d8d5` | Opus | reported 2026-10-22: fail (50%); 4.75 GPU-hours of 4 (overrun approved) |
| K6b players 3 | overfit oracle (user approved) | `claude/k6b-players-zeroshot` @ `03a3df8` | Opus | reported 2026-10-23: pass (83%), matte and LoRA unbilled; 4.8 GPU-hours of 6 |
| K6b players 4 | match fit, billed | `claude/k6b-players-zeroshot` @ `20eb7c3` | Opus | reported 2026-10-23: fail (67% unbilled, 58% billed); 7.0 GPU-hours of 7 |
| K6b players 5 | tennis base, guider, 4 bits (user approved) | `claude/k6b-players-zeroshot` @ `d0a00e5` | Opus | reported 2026-10-24: fail (67% billed and unbilled); 10.0 GPU-hours of 11; historical failure retained in the merged experiment record |
| K6m | the players' matte on tennis | — | Sonnet | card dismissed 2026-10-24: folded into T1 |
| K6b players 5b | why 045 and 052 lose; a content-blind scope rule (CPU) | `codex/k6b-players-5b` | Codex high | reported 2026-10-24: no cut in 052; registration gate 80% vs 67% pooled (hypothesis); 0 of 0 GPU-hours |
| K6b players confirmation | unused same-match windows, CPU preparation | `codex/k6b-player-confirmation-plan` | Codex high | closed; preparation superseded by completed clip-local diagnostic below (PR #236), 6.995428306/8 GPU-hours; no running preparation/anchor job ([card](docs/cards/k6b-players-confirmation.md)) |
| T1 | the split on tennis, F billed | `codex/t1-tennis-split` | Codex high | reported 2026-10-24: primary oracle +10.00% BD weighted DISTS, 3/8 wins; billed split loses all metrics/windows; 3.12338 of 4 GPU-hours |
| K6b players 6 | Wan-Animate zero-shot (reference-conditioned) | `codex/k6b-players-6` | Codex high | deferred; finish actual-base feasibility decision first; no new model budget or dispatch authorized ([card](docs/cards/k6b-players-6.md)) |
| H5p 4d | a cheaper foreground mask: binary shape coding, F at the client | `codex/h5p-4d-mask`, PR #233 | Codex high | reported 2026-10-24: shape target fails (+16.6% wDISTS); decoder diagnostic 13.3–61.3 kbps, full curve blocked by native decode; 0.3171 GPU-hours of 2 |
| K6b players clip-local diagnostic | [frozen 12-window study](https://github.com/emanuele-artioli/PointStream/blob/72228816/docs/cards/k6b-players-confirmation.md) | `codex/k6b-player-confirmation-plan` @ `72228816`, PR #236 | Codex high | reported 2026-10-25: 23/24 billed wins with common base, reference and matte granted; base-charged 0/12 bracketed wins, 12 unresolved; 6.995428306 of 8 GPU-hours; all jobs terminal; PR #236 merged 2026-10-25 after explicit user auto-merge approval |
| K6b players rate-budget audit | saved 12-window side-information headroom, CPU only | `codex/k6b-player-confirmation-plan` | Codex high | reported 2026-10-25: tested lossless route fails (0/24; 22 bracketed, 2 unresolved but dominated); even free mattes leave 17/24 with real keypoints + PNG bank 2 h; CPU job `20261025T082944Z-29f021a5` complete, 0 new GPU-hours; deployable alternatives unresolved ([card](docs/cards/k6b-player-rate-budget.md)) |
| K6b players compressed mattes | [CPU-first saved-render test](docs/cards/k6b-player-compressed-matte.md) | `codex/k6b-compressed-matte`, PR #239 | Codex high | reported 2026-10-25: CPU complete; all 12 mask rates/coverage and two real neural smoke frames; full quality exceeds CPU cap; CPU phase0 GPU, proposed0.75 GPU-h judging subsequently approved; half-size JPEG reference pricing~90% below PNG, generation quality untested |
| K6b players matte GPU judging | [saved-render quality continuation](docs/cards/k6b-player-compressed-matte.md) | `codex/k6b-matte-judging`, PR #240 | Codex high | reported 2026-10-25: all scales0/8 measured wins after4 complete windows; best possible16/24<18, frozen stop reached;0.167674940 of0.75 GPU-h; all jobs terminal, no fitting/generation |
| K6b players reference/alpha video | [bounded coding/quality pilot](docs/cards/k6b-player-reference-video.md) | `codex/k6b-reference-quality-pilot`, PR #242; prior #241 merged | Codex high | reported 2026-10-26: actual unmasked compressed references 4/4 (passes >=3/4); alpha/black each1/4; dense each0/4; exact-matte-paid all0/4; no full12 or independent-source claim; 4.411019/4.5 new GPU-h, CPU2647.199/3600s, no jobs remain |
| K6b players joint matte/edge pilot | [saved-render joint test and CPU mechanism](docs/cards/k6b-joint-matte-edge.md) | `codex/k6b-joint-matte-edge-pilot`, PR #243 | Codex high | reported 2026-10-26: actual paid1/8,1/16,1/32 mattes each0/4; exact control4/4; RGB ramps lose shape, luma ramps not cheaper, none promoted; 0.103706/0.5GPU-h, CPU721.525/3600s; no full12/end-to-end/independence claim, no active jobs |
| K6b players receiver silhouette | [raw-generation/correction input audit](docs/cards/k6b-receiver-silhouette.md) | `codex/k6b-receiver-silhouette-audit`, PR #244 | Codex high | reported 2026-10-26: prior H5p4d related, exact tennis question untested; each actual raw unit saves80/240 RGB ticks, all240 correction test stops at input gate;0/0.75 new GPU-h, no inference/submission; sparse receiver profile or raw recovery remains a distinct next choice |
| K6b players sampled silhouette | [80-tick scope amendment](docs/cards/k6b-receiver-silhouette.md) | `codex/k6b-receiver-sampled-pilot`, PR #245 | Codex high | reported 2026-10-26: actual zero-matte YOLOE26x profile1/4<3/4; private fresh-reset predictions reproduce/all160 judged frames scored; exact control4/4; cumulative0.163059668/0.75GPU-h, CPU2334.860723/3600s closed; all240 corrections/wire and end-to-end evidence remain unresolved, all jobs terminal |
| K6b players pose-prompted SAM | [sampled receiver tracker](docs/cards/k6b-receiver-sam.md) | `codex/k6b-receiver-sam-pilot` | Codex high | reported 2026-10-26: actual pose-prompted SAM2/4<3/4; Med004 recovered but Al006 empty51/80; private reset/control checks pass; sampled XOR122.62/212.46kbps costs more than direct sampled R; conservative0.177777778/0.75GPU-h and1429.969250/3600CPU-s CLOSED, one terminal job; no full240/end-to-end claim |
| K6b players independent-frame SAM | [last narrow receiver profile](docs/cards/k6b-receiver-frame.md) | `codex/k6b-receiver-frame-pilot`, PR #247 | Codex high | reported 2026-10-26: actual4/4strict wins (frozen+.0013/4),0/320empty player masks; read-isolated resets/control pass; conservative0.477777778/0.75GPU-h and2329.366896/3600CPU-s CLOSED; fullyaccounted/fresh-eval cards prepared only, no integration/full240/confirmation claim |
| K6b native base recovery | [recovery card](docs/cards/k6b-base-recovery-encode.md) | `codex/k6b-native-base-attended`, PR #251 | Codex | closed 2026-10-27: CPU job `20261027T072713Z-2888cefe` complete, six native 240-frame SVT streams and retained YUV; 184.54109/1800 workload CPU-s, no GPU |
| K6b real-base quality | [actual-base pilot](docs/cards/k6b-real-base-quality-pilot.md) | `codex/k6b-real-base-player-quality`, PR #252 | Codex | stopped 2026-10-27: conversion `20261027T074823Z-135421ee` complete; quality `20261027T080644Z-9dba0803` failed cost gate, full never launched; original smoke controls pass both windows; no accepted quality/BD-rate; conservative580/3600 GPU-s,3020 unused, revised plan required |
| Host-local storage safety | [maintenance report](docs/storage-maintenance-2026-10-27.md) | `codex/local-storage-safety-20261027`, PR #253 | Codex | merged 2026-10-27: inactive files moved/verified on six hosts, shared shell defaults active; live originals retained; fleet manager rollout stopped at health gate before changing workers |
| Root continuation | [2026-10-27 handover](docs/continuation-2026-10-27.md) | `codex/root-handover-20261027`, [PR #254](https://github.com/emanuele-artioli/PointStream/pull/254) | Codex | reported: final handover for user-requested root-chat archive after merge; no new session or research job launched |
| Root real-base cost revision | [prospective same-budget plan](docs/cards/k6b-real-base-cost-revision.md) | `codex/real-base-cost-plan`, PR #255 merged | Codex | human accepted 2026-10-27: smoke240/full2600/validator120/overhead60 within3020s remaining; original quality/identity gates retained |
| Root approved execution preflight | [accepted cost plan](docs/cards/k6b-real-base-cost-revision.md) | `codex/real-base-cost-execution` | Codex | 2026-10-27: validator and frozen spec prepared; local ruff/mypy and16focused tests pass; stopped before submission at current fleet health (gpu3 20s probe failure, stale compatible worker heartbeats); no new GPU cost |
| Fleet health recovery and approved real-base run | [recovery report](docs/fleet-health-recovery-2026-10-27.md) | `codex/real-base-approved-run`; repair PR #257 merged | Codex | 2026-10-27: final20261027T211451Z-f6a754eb COMPLETE; fresh separated smoke passes, full stops scientifically after Al loses both whole-frame metrics; audited55checks+240CPUhybrids;2350/3600s charged,1250unspent; no active wait/GPU job |
| Ideal-player appearance bound | [completed finite bound](docs/cards/k6b-ideal-timing-secondary.md) | `codex/ideal-player-full-result` | Codex | COMPLETE20261028T080453Z-34e50ec2; all80ticks, exact controls, independently audited finite NO-GO both metrics/both ideal supports;2730/3600s charged,870unused; item stopped |
| Player joint side-rate feasibility | CPU saved-score inversion and three lossless packet wrappers | `codex/player-rate-feasibility` | Codex | COMPLETE2026-10-28; zero GPU; robust actual/ideal_R side ceilings0.668734/1.230365kbps; all three packet wrappers grow; follow-up paid clean-background test now complete |
| Paid clean-background oracle | [new finite background proposal](docs/cards/k6b-paid-clean-background.md) | `codex/paid-clean-background` | Codex | COMPLETE2026-10-28; CPUbf9f97d2 and sameGPU27f39d64 audited; all80ticks; formal MIXED, two-metric gate FAILED;250s charge,2980/3600total,620unused; stopped |

**T1: the split on tennis, F billed.** Fixed-grid oracle fails weighted DISTS: +10.00% mean BD-rate, 3/8 wins even with foreground payload free.
The fully billed split loses every metric/window; 3.12338 of 4 GPU-hours used ([report](docs/experiments/t1.md#2026-10-24--report-t1-the-billed-tennis-pixel-split)).

**K6b players 5b and 6.** Steps 3–5 lose the same two windows:
5b's CPU diagnosis (2026-10-24) finds **no cut** in 052: one oblique court
view, 32/240 registered frames and seven judged. A ≥95% registration gate
reads 80% beside 67% pooled; adding ≥115 px reads 100% on four windows,
but both gates are unconfirmed hypotheses. 045 shows appearance/view and
racket errors, with no isolated causal explanation; step 5's fail stands.
Confirmation preparation (2026-10-24): 12 new same-match windows frozen;
all pass registration, so gate discrimination is untested. Recorded fit/reference
clips are disjoint. CPU recovery found splitter logic; the completed 61-clip
audit found no exact/screened duplicates. Original-source identity/replay
independence remains unresolved. The user approved the clip-local diagnostic,
which subsequently completed at 6.995428306/8 GPU-hours (PR #236); the historical
preparation and anchor jobs are terminal. Reference and matte were granted in
that diagnostic. Later component and actual-base results are summarized in the
resume section above ([card](docs/cards/k6b-players-confirmation.md)).
Step 6 tries Wan-Animate, conditioned on the same unbilled reference as
step 2 (zero-shot means no fitting, not no appearance); speed is
secondary while quality is tested (user, 2026-10-24).

**H5p 4d: a cheaper foreground mask.** Every mask coder tried on the
egocentric pilot (lossless coding of downscaled frames XORed with the
previous one, a video codec on the signed distance, a key colour) costs
36–116 kbps, and billed it removes the split's win. The ceiling is H5p 4b:
the exact F at zero bits wins weighted DISTS (−15%) and still loses
weighted LPIPS (+16%), so a mask can at best restore the DISTS win, within
about 2 kbps. 4d tests two further routes: binary shape coding with motion
prediction (MPEG-4's context-based arithmetic coding; CPU), and F derived
at the client from the decoded streams by the same segmenter (no bits, or
a small correction stream). Measured 2026-10-24: shape coding fails
the rate target and DISTS rule; SAM plus exact corrections also misses 2 kbps
on the single-point diagnostic. Full decoder-side BD-rate is unresolved
after native repeatability failures; [record](docs/experiments/h5p.md).

**Handover (2026-10-24).** The Claude Code root session
(`session_01StNQ6dvxYRHhnMbQymNgNf`) stopped when the weekly Claude
tokens ran out. No root session replaces it: item sessions keep this
ledger themselves (above). Open for the user: (1) how to reframe the
egocentric claim (proposal: a negative result with its cause, F's rate,
and the split's win where F is small, P09_106; tennis carries the
positive claim); (2) whether to run step 6 after 5b; (3) the identity-warp
check (J's caveat), only if the experience track stays a goal. Branches
with work not on main: `claude/k6b-players-zeroshot` (step 5's outcome,
`d0a00e5`). PR #218 (J) was open at handover.

## Done

- **A. Data acquisition and archive** (2026-10-06). All six datasets are under
  `Datasets`, with immutable manifests and smoke reads in `Datasets/manifests/`.
  The pre-reset data is in `Datasets/archive/pre-reset-2026-10-05/` with its
  own manifest. Tooling: `tools/datasets/`.
- **0. Environment audit, first wave** (2026-10-06). One environment for
  SAM 3.1, YOLOE-26, the VISOR and HOT3D-Clips readers, HaMeR and WiLoR,
  SVT-AV1 and DCVC-UF (the chosen neural codec), with the conflicts, locks and
  packed archive in [resources](docs/resources.md#environments) and per-GPU
  results in the [model–GPU table](docs/fleet.md#modelgpu-table). Open for H:
  HaMeR or WiLoR, decided on HInt test and HOT3D.

## Phase 1: egocentric

### B1. VISOR adapter and evaluation set (done 2026-10-07)

Done (2026-10-07). Narrative: [history](docs/history.md#b1-visor-adapter-and-evaluation-set-done-2026-10-07).

### B2b. Baseline follow-ups (agreed 2026-10-07, after B2)

B2's baselines made fair and stronger before anything is compared with them.
Each item gets an entry in its `docs/experiments/` file before its fleet run.
1. **Scoring on the GPU for every codec.** SVT-AV1 jobs encode on the host's
   CPU but claim a GPU for scoring (B2 scored them on CPU: 227 s per point
   against 6.5 s). B2's CPU scores stay valid; they are not redone.
2. **Clean timing, then an equal time budget.** Time DCVC-UF's encode and
   decode as upstream does (CUDA events around the model calls), apart from
   the worker's verification (second decode, per-frame hashes) and its CPU
   4:2:0 conversion, which move out of the timed region. Time SVT-AV1 per
   preset on the same host class, isolated, with a fixed thread count. Then
   SVT-AV1 gets the slowest preset whose encode time per window fits DCVC-UF's
   on the same host, and that preset replaces preset 4 if it differs. CRF only
   sets the rate. SVT-AV1 has no GPU path; NVENC AV1 (Ada only) is a separate
   hardware encoder, at most an extra reference point.
3. **Lower rates by resolution.** CRF stops at 70, so SVT-AV1 also encodes
   720p, 540p and 360p versions, upscaled to 1080p by a fixed filter before
   scoring; each item's curve is the convex hull over resolutions. DCVC-UF
   gets the same ladder if its QP 0 floor is not low enough.
4. **ROI SVT-AV1.** `--roi-map-file` lowers the quantizer inside the VISOR
   foreground (an oracle: the dataset's masks, not a segmenter's). A pilot
   picks the offset. A segmenter-mask variant follows D1.
5. **DCVC-UF full-range check.** One pilot item coded with limited-range
   input, scored against the same reference.
6. **Secondary studies (CPU).** Spatial and temporal complexity per frame and
   inside the foreground (VCA v2, `/opt/local/bin/vca`; ffmpeg's `siti` as a
   cross-check) against each item's DCVC-UF advantage; SVT-AV1 with temporal
   filtering off (`--enable-tf 0`) on a few items.
7. **Why SVT-AV1's ROI cannot raise a region's quality** (deferred; a
   possible secondary contribution, after the main path). In 4.2.0's CRF
   mode a negative quantizer offset adds bytes without raising the region's
   PSNR, while a positive one lowers it as expected (B2b J3, J3b, probe).
   Hypothesis: the encoder derives its rate-distortion trade-off (lambda)
   from the frame's quantizer, not the segment's, so a finer segment
   quantizer buys coefficients that rate-distortion optimization does not
   turn into quality. First read SVT-AV1's ROI/segmentation and lambda code
   (minutes); then one bounded probe that tests what the code suggests (e.g.
   offsets with a matched lambda, or another encoder's ROI); write it up only
   if the mechanism is confirmed.
**Done when** the B2 curves are redrawn with the fair SVT-AV1 preset, the
low-rate ladder and the ROI variant, from recorded jobs.

Status (2026-10-07): items 1, 2 (timing), 5 and 6 are done, and item 4's pilot
chose the ROI offset (+32 on background blocks; lowering the foreground's
quantizer does nothing in SVT-AV1's CRF mode)
([outcome](docs/experiments/b2b.md#2026-10-07--b2b-fair-baselines-timing-equal-time-roi-range-complexity)).
DCVC-UF HT-L encodes a window in 1.9 s, faster than any SVT-AV1 preset on 32
cores. By the user's decision the equal-time preset is matched to
PointStream's encode time instead, so the equal-time re-encode, the full ROI
run and the low-rate ladder wait for PointStream's first timing and rates, and
then run together from J2's time-per-preset table.

### B1b. Filling VISOR's missing hands and objects (done 2026-10-08)

Done (2026-10-08). Narrative: [history](docs/history.md#b1b-filling-visors-missing-hands-and-objects-done-2026-10-08).

### B2. Baseline rate-distortion on VISOR (done 2026-10-07)

Done (2026-10-07). Narrative: [history](docs/history.md#b2-baseline-rate-distortion-on-visor-done-2026-10-07).

### G. Background (user, 2026-10-08)

The thesis: a viewer barely looks at the background, and most of it repeats.
After a warm-up stretch of video, the client receives one representation of
everything the camera has seen; from then on each background frame is
rebuilt from it with only the camera's pose (GenStream did this with a
hand-made 3D model; PointStream builds the representation from the video
itself). With a camera that rotates in place there is no parallax, so the
representation is a panorama (planar, cylindrical or spherical) rather than
3D Gaussian splatting. A second candidate stores the warm-up background in
the weights of a neural codec and then sends one latent per view. The
foreground occludes the background: a panorama merges views and fills what
one frame hides; a neural model needs clean frames, from the panorama or from
generative inpainting. PRESLEY's lesson stands: the background only has to be
good enough not to distract. Each step below is one session; its outcome
decides the next.

#### G1. Camera motion and coverage audit (next)

Done (2026-10-08). Narrative: [history](docs/history.md#g1-camera-motion-and-coverage-audit-next).

#### G2. Background evaluation protocol

Superseded by G5–G6 (the background became a scene-adapted neural model). Narrative: [history](docs/history.md#g2-background-evaluation-protocol).

#### G3. Panorama background

Superseded: G1 ruled out a rotation-only panorama on VISOR. Narrative: [history](docs/history.md#g3-panorama-background).

#### G4. Clean background frames

Superseded by G6's fill (ProPainter). Narrative: [history](docs/history.md#g4-clean-background-frames).

#### G1d. Depth-aware warps on VISOR (done 2026-10-09)

Done (2026-10-09). Narrative: [history](docs/history.md#g1d-depth-aware-warps-on-visor-done-2026-10-09).

#### G1e. Reference refresh (done)

Done (2026-10-09). Narrative: [history](docs/history.md#g1e-reference-refresh-done).

#### G5. Neural background model

Done (2026-10-09). Narrative: [history](docs/history.md#g5-neural-background-model).

#### G5b. Scene-adapted neural background (user, 2026-10-10)

Done (2026-10-10). Narrative: [history](docs/history.md#g5b-scene-adapted-neural-background-user-2026-10-10).

#### G5c. Scene model at scale (agreed with the user, 2026-10-10)

Done (2026-10-12): the recipe and model size G6 uses. Narrative: [history](docs/history.md#g5c-scene-model-at-scale-agreed-with-the-user-2026-10-10).

#### G6. The background in the first working PointStream (plan, 2026-10-12)

What H5 needs from G: for each of evaluation set v2's 34 windows (1080p,
4:2:0, 50 or 59.94 frames/s, 4 s), the background's bitstream, its decoded
frames and decode time, at several rates, so the foreground session can
composite and B2's harness can score the whole frame. G5c worked on 10
stretches at 960×540 and 10 frames/s; three gaps lie between. Two routes
go in (user, 2026-10-12): the scene model (DCVC-UF adapted, recipe L) and
the venue enhancer after SVT-AV1 (G5d). Cheapest first:
1. *Interface, agreed with the foreground session before any run.* Input:
   the window's 1080p frames and a per-frame foreground mask F (the pixels
   the foreground stream delivers; the encoder fills F with ProPainter and
   codes the filled frames). Output per rate point: bytes, decoded 1080p
   frames, decode milliseconds per frame; once per venue: the model, its
   shipped size and payback. Composite at the client: foreground pixels
   over the decoded background on F. Scoring: B2's harness (weighted PSNR
   and LPIPS on dataset masks) plus every metric as BD-rate against
   SVT-AV1 and DCVC-UF on the whole frame. Open: the mask source at the
   encoder (dataset masks for the first run, predicted masks after).
2. *Resolution.* Does a scene model trained at 960×540 help at 1080p?
   Two options, one smoke each on the pilots' windows: code the background
   at 1080p with the model as trained (patches are resolution-free), or
   code it at 540p and upsample at the client (the viewer barely looks at
   it; PRESLEY). Rate at equal quality on V decides.
3. *Frame rate.* The windows are 5–6× denser in time than the stretches'
   10 frames/s, so motion per frame is smaller and DCVC-UF's temporal
   prediction does more of the work. Training sequences at the window's
   native rate (footage extracted at native rate, shorter segments) against
   the 10 frames/s model, on the pilots' windows.
4. *Scene footage for every window.* Step 3's pipeline (SAM 3.1 and EPIC
   detections, which cover exactly the 34 videos, then ProPainter) on the
   window's own video outside the window and a guard, and the kitchen's
   other recordings for the venue-only variants. About 34 footage jobs.
5. *Both routes on all 34 windows,* against SVT-AV1 and DCVC-UF coding
   the same filled background; published for the foreground session's
   compositor, with rate allocation between foreground and background
   taken as the concave hull over their rate combinations, per metric.
Budget: steps 2–3 about 4 GPU-hours; step 4 about 25; step 5 about 40
(34 windows × two routes). Steps 2–3 run first and decide the settings
of 4–5; each stage stays under 45 minutes per job.
*Steps 2–3 done (2026-10-12, [experiments](docs/experiments.md), G6):*
SVT-AV1 codes the background at 1080p; the scene model at 960×540,
upsampled at the client; at the windows' native rate it beats SVT-AV1
on LPIPS_V, DISTS_V and KID_V on both pilots (−91% and −45% on
LPIPS_V). The frame-rate rule missed by 1 point and the user treated it
as passed: native-rate and 1080p training are deferred to the final
runs as ablations.
*Interface, agreed with the foreground session (2026-10-12):* the
background codes the whole 1080p frame with F filled, so its decoded
frames are valid everywhere. F, first run: B2's dataset foreground (the
union of B1b's `visor_dense_sam_fill` instances), filled exactly (no
dilation; the foreground blends linearly over an 8 px ring outside F, so
true background must survive there). The mask is granted in the first
run, stated, and reported as its own line (its cost, 72–170 kbps for a
raw outline per H3b2, is an open H5 item). Rates: about 10 kbps to
2 Mbps, at least five log-spaced points per route (the allocation is the
concave hull over foreground and background combinations). Published per
window: decoded frames (raw 4:2:0 or FFV1; DCVC-UF decodes differ across
GPU classes) and a manifest (window id and frame range, F's mask-set
sha256, and per rate point bytes, bitstream sha256, per-frame decoded
sha256s, decode ms per frame, GPU class, job id, `git_head`).
**Done when** both routes' background streams for all 34 windows are
published with the interface above and the foreground session has
composited them.
*Pilot done (2026-10-12, [experiments](docs/experiments.md), G6 steps
4–5 pilot):* the per-window pipeline runs end to end on the pilots'
windows and publishes both routes at the agreed interface. The enhancer
after SVT-AV1 beats it on LPIPS_V, DISTS_V and KID_V on both (LPIPS_V
better at every rate | −83%); the window's own footage makes the same
scene model as the stretch (±3% LPIPS_V at matched rate). *Deferred
(user, 2026-10-12):* the other 32 windows run once, with the final
experiments, after the paper's draft fixes the figures they feed.
*Item 1 done (2026-10-13, [experiments](docs/experiments.md), G6 item
1):* with F filled exactly, the scene model is trained on footage masked
the same way (`footage --mask exact`; LPIPS_V −11% | −12% against G1's
footage) and codes at 1080p in SVT-AV1's range, 540p below it (LPIPS_V
against SVT-AV1 −97% | −33%; DISTS_V and KID_V still worse on P06_03).

### H. Foreground (user, 2026-10-08)

The thesis: a pixel codec spends most of its bits on what moves, and people
and the objects they handle move in ways computer vision already models.
GenStream sent a skater as skeleton keypoints and rendered the person at the
client. PointStream does the same per object: an appearance reference sent
once (like the background's warm-up), then compact motion parameters per frame
(hand pose, an object's rigid motion), rendered and composited over the
background at the client. Whatever the model cannot explain (an object first
seen, food being cut, a failed estimate) falls back to coded pixels, so every
frame decodes. VISOR's hands include the forearm, which MANO does not model,
and the hands carry most of the viewer's attention. Each step is one session.
G and H are independent until H5, so their sessions can run side by side.

#### H1. Foreground motion and representation audit (done 2026-10-08)

Done (2026-10-08). Narrative: [history](docs/history.md#h1-foreground-motion-and-representation-audit-done-2026-10-08).

#### H2. Hand-pose estimators: HaMeR against WiLoR (done 2026-10-10)

Done (2026-10-10): WiLoR; the pose stream at 1.93 kbps. Narrative: [history](docs/history.md#h2-hand-pose-estimators-hamer-against-wilor-done-2026-10-10).

#### H3. Hand and arm rendering

Done (2026-10-13): hands are coded pixels (DCVC-UF at ¼/½ plus an enhancer, since H4c G_HO); keypoint rendering loses (H3b, H3e). Narrative: [history](docs/history.md#h3-hand-and-arm-rendering).

#### H4. Handled objects (rescoped after H1, user 2026-10-08)

Done (2026-10-16): handled objects are coded pixels with the hands (H4a, H4b fail; H4c adopts G_HO). Narrative: [history](docs/history.md#h4-handled-objects-rescoped-after-h1-user-2026-10-08).

#### H5. First working PointStream

Handled objects and hands (decided by H4c, 2026-10-16): DCVC-UF HT-L off
the shelf at ¼ or ½ resolution plus G_HO, one generic enhancer for both,
applied on the foreground boxes and sent with the decoder (networks:
`h4c-2026-10-16` jobs `20261016T152552Z-c3d9ed14` (¼) and
`20261016T153801Z-03f2a819` (½), one per QP). It replaces G on the hands.

G's background and H's foreground in one bitstream with every byte counted:
references once, then camera pose, hand pose and object motion per frame,
plus the fallback pixels. Scored with B2's harness on evaluation set v2
(weighted PSNR on dataset masks, with and without B1b's fill) against B2's
curves, and timed end to end, which picks B2b's equal-time SVT-AV1 preset.
**Done when** PointStream's curves and encode and decode times on all 34
items come from recorded jobs, beside B2's.
*Pilot done (2026-10-17, [experiments](docs/experiments.md), H5):* the
pipeline runs end to end on B2's 4 pilot items: G6's two background routes,
one DCVC-UF foreground stream of F (¼, ½, full; G on the hands), the
compositor and a container with every byte counted
(`src/pointstream/`, `experiments/visor/h5.py`). With F granted, rule (a)
passes: weighted LPIPS −15% against the pooled envelope of B2's preset-4
SVT-AV1 and DCVC-UF (CI −25 to −5, 3 of 4 items); frame LPIPS −38% and
DISTS −48% (4 of 4); whole-frame PSNR loses (+70%) and weighted PSNR
splits. The neural background carries the gain; the foreground stream
is worse per bit than whole-frame DCVC-UF where F is large. Billing F
at its lossless cost (0.5–2.9 Mbps) flips every comparison. Encoder
163–166 s per window (ProPainter 77%), decoder 95–133 s (this
pipeline's CPU stages): **the equal-time SVT-AV1 preset is 1** (B2b's
re-encode, ladder and ROI run can go). Before the 34-item run (about 200
GPU-hours, final-runs-once): B2b at preset 1, a mask coder, and the
foreground's per-bit cost on large F.
*H5m (2026-10-17):* F's class map coded at 1/s and billed (16–116 kbps at
s = 8–16) does not keep the weighted-LPIPS margin (hull over s −4%, 2 of 4
items; rule −10%, 3 of 4); frame LPIPS −21% and DISTS −28% survive on 4 of
4. Next for F: re-encode the foreground with K and the fill from F̂ (the
measured streams were coded for exact F), then F carried by the
foreground stream itself. *G_HO (H4c) in both foreground slots
(2026-10-20):* weighted LPIPS −29% (CI −38 to −19, 4 of 4), DISTS −23%
(4 of 4), with F granted.

*H5p, pilot-scale checks before any 34-item run (user, 2026-10-20: the
34-item run waits for the paper's figures; plan these instead).* On the
4 pilot items, each with its rule written before its run, about 10
GPU-hours in all:
Order agreed with the user (2026-10-20): F's rate test moved up, before
the fair anchors.
1. *Venue models' payback (CPU, minutes).* **Done**: the scene model
   ships as a 50 MB 4-bit change. Billed per video, rule (a) fails (−9%;
   the curve is the enhancer route alone); billed per kitchen it passes,
   −25% on 4 of 4. The scene model repays in 0.4–3.3 h of stream, each
   enhancer in 10–324 s. The routes stand as per-kitchen models, pending
   a cross-video check (a model fitted on the kitchen's other recordings,
   scored on the pilot window; GPU).
2. *F as a video (CPU oracle).* **Done, fails**: SVT-AV1 (F's signed
   distance or its step in Y, the class in U) never matches lzma at s = 8
   for less (1.4–2.4× dearer at equal accuracy). It halves the rate only
   for a slightly worse F̂, and even then, re-billed optimistically,
   weighted LPIPS reaches −8% (3 of 4), short of rule (a). lzma stays;
   F carried in the foreground stream becomes the main route.
3. *Fair anchors.* **Done, fails.** Against SVT-AV1 at preset 1 and
   DCVC-UF off the shelf, both with a 720/540/360p ladder, plus a
   whole-frame DCVC-UF adapted to the venue the same way, PointStream with
   its models billed per kitchen needs 74% more rate on weighted LPIPS (0
   of 4).
   - The ladder reaches 8–80 kbps, where the billed 50 MB scene model
     loses. Break-even is 10–30 h of stream per scene model, against a
     kitchen's 0.65–3 h on disk.
   - At equal adaptation, neither model billed, the venue-adapted
     whole-frame codec beats PointStream on LPIPS (weighted +16%, frame
     +15%). PointStream wins on frame DISTS (−17%, 4 of 4) and weighted
     PSNR (−11%).
   - The background's gain is mostly adaptation, not the split.
   - Decided (user, 2026-10-20): venue models are sent beforehand and not
     billed, so the venue-adapted anchor is unbilled too. The paper
     reports the hours of stream a venue model needs to repay itself
     (10–30 h here). Smaller venue models are a later optimization, once
     there is a win.
4. *F, GPU.* Re-encode the foreground with K from F̂ (lzma at s = 8 and
   16) with G_HO in the slots (H5m had G), and rescore; then an oracle for
   F carried by the foreground stream itself (no MASK segment).
   **Done (2026-10-22, [experiments](docs/experiments.md)), fails.**
   - 4a: with F billed (lzma at s = 8, the stream built for F̂, the
     per-point configuration), weighted DISTS against the venue-adapted
     DCVC-UF reads +93% (1 of 4), where F granted reads −15% (4 of 4).
     Every metric loses, against the venue codec and the pooled envelope.
     s = 16 is not run: its upper bound fails too (+24%, 1 of 4).
   - F̂ costs no quality (mask bytes removed: −17%, as granted). The loss
     is the mask's 36–116 kbps, at the low rates where the venue codec
     reaches 13–26 kbps.
   - 4b: exact F at zero bits fails rule (a) on weighted LPIPS (+16%, 1
     of 4), so F in the foreground stream is not built for H5q's loss. On
     DISTS it passes. A post hoc bound puts any carrier's budget at about
     2 kbps (5 bytes per frame), 7–20× under every F coder measured.
   - 4c (the user asked for the measured carrier, 2026-10-22): a key
     colour outside K, F̃ recovered at the client (IoU 0.95–0.99). The
     key's edge costs +26% to +129% of the foreground stream where F is
     large; its optimistic bound reads weighted DISTS +1% (1 of 4), so
     the gate fails and it is not scored. Only where F is 2% of the
     frame (P09_106) is the key free (−30%).
   - **H5p 4 closes as a fail:** no F coder tried (lzma, video, key)
     keeps the split's DISTS win at equal adaptation once F is paid
     for; the split wins only where F is small.
5. *Visual check (CPU).* Side-by-side crops of PointStream and the
   anchors at matched rates, for the user's review.
   **Done (2026-10-22):** 16 crops (4 items, about 200 kbps and 1 Mbps,
   frames 60 and 180), PointStream billed as in 4a against the venue
   codec and SVT-AV1 preset 1 at their best point at or under its rate;
   in 4a's report archive, for the user's review.
Scores stay LPIPS, DISTS and PSNR (a model trained on a metric may be
judged on it, user 2026-10-20).

*H5q, the split's win at equal adaptation (user, 2026-10-20: "as long as
there's a way to win, we should take it").*
- The target: PointStream against the venue-adapted whole-frame DCVC-UF,
  neither model billed. Rule (a) on weighted LPIPS (mean −10% or better, 3
  of 4 items below zero), and against the pooled fair envelope too. Today
  it reads +16% (1 of 4). DISTS and PSNR are reported beside it.
- The levers only the split has, cheapest test first; each one gets its
  rule written before its run:
  1. *A venue-adapted foreground coder.* PointStream adapts only its
     background; the anchor adapts the whole frame, hands included. Use
     recipe L's fine-tune of DCVC-UF HT-L on the own-video footage's
     foreground input (F and its ring, the rest push-pull padded, at ¼
     and ½), with G_HO after it. Oracle first: fit on the scored window
     itself, the memorization upper bound. If even that does not turn the
     LPIPS row negative, the lever is dropped.
  2. *A denser allocation grid.* The hull over (foreground, background)
     pairs already splits the rate between F and the background. Add
     points only where the hull is sparse near the anchor's curve.
  3. *F carried in the foreground stream* (PLAN H5p 4). This removes the
     MASK segment, which only matters once F is billed.
- Venue models are sent beforehand and unbilled, on both sides. Payback
  hours are reported. Smaller models come later (user, 2026-10-20).
- *Where it loses (2026-10-20, from H5p 3's per-region scores,
  [experiments](docs/experiments/h5q.md#2026-10-20--h5q-1-the-venue-model-codes-the-foreground-equal-adaptation)):*
  below about 200 kbps PointStream's foreground already beats the venue
  anchor's and its background loses; above it the foreground loses, and
  at QP 63 it equals off-the-shelf whole-frame DCVC-UF while the venue
  model halves its LPIPS. The high-rate gap is the adaptation alone.
  Lever 1's cheapest form therefore needs no training: the window's venue
  model (the anchor's own checkpoint) codes the foreground.
- *Lever 1 done, fails (2026-10-21):* +19% (1 of 4). It closes the
  high-rate foreground gap on every item, but G_HO after a venue decode
  loses at ¼ and ½; choosing the coder per point (post hoc) reaches +6%
  (1 of 4), with weighted DISTS −15% and frame DISTS −22% (4 of 4). A
  region decomposition puts the remaining LPIPS loss in the
  **background**: PointStream with the anchor's background at the same
  total rate would pass (−13%, 4 of 4); with the anchor's foreground,
  −1%. The venue model also fails to decode QP 63 at ½ on 2 of 3 items.
- *The background stream is not worse per bit* (it beats the anchor's
  background region below about 300 kbps); the split's loss is
  structural, paying for F twice, and follows F's share of the frame
  (P02_02 50%: +19%; P09_106 2%: −15%). Even a free fill in the
  background stream (CPU estimate) reaches −5% (2 of 4), short of the
  rule; lever 2 adds a few points, lever 3 matters once F is billed.
- *H5q 2, the fill measured (user, 2026-10-21), fails:* a static plate in
  F saves 13–39% of the background where F is half the frame (P02_02)
  and nothing where it is small (moving F makes a frozen plate costly;
  ProPainter's fill is already cheap for the scene model). With the
  measured saving, P01_107 +9%, P02_02 +11%, P03_10 +13%: the rule
  cannot pass.
- **H5q closed on LPIPS (2026-10-21).** At equal adaptation the split
  loses on weighted LPIPS (+16% as built, +6% best per point) and wins on
  weighted DISTS (−15%), frame DISTS (−22%), weighted PSNR (−11%), and
  on LPIPS where F is small. **The egocentric claim is the user's to
  reframe**; next on the critical path: H5p 4–5.

#### K. Where keypoints win (analysis, user 2026-10-13)

Analysis and finished items: [history](docs/history.md#k-where-keypoints-win-analysis-user-2026-10-13).
In short: keypoints lost on egocentric hands and objects every time they
replaced pixels, and win where pixels collapse on small, fast, structured
objects (the tennis ball).
- K1. Done (2026-10-16): no low-rate win for egocentric hands against
  DCVC-UF + G; the claim is dropped (user, 2026-10-17).
- K2. Done (2026-10-13): [docs/vcm.md](docs/vcm.md), the machine-judge protocol.
- K3. Done, closed (2026-10-17): keypoint-conditioned coding does not pay
  for hands at any keypoint rate (K3b: K3's gain was all appearance).
- K4. Done (2026-10-13): players fail H1's screen on the far player (fit
  0.875, keypoints 22% of SVT-AV1's player bits at CRF 62); K4b next.
- K5. Done (2026-10-13): **a ball mode is built** (−57% / −38% BD-rate on
  ball-region PSNR over DCVC-UF / SVT-AV1, for 0.23 kbps); K5b next.
- K6. *Player oracle* (after K4; gate passed with K4b, 2026-10-21: the far
  player fitted by ×1 crops, players found by (c)+(d), keypoints 3.3–15
  kbps; runs on K4's 9 broadcast windows among its own, which is K4b.1's
  amendment 2): H3b for bodies, perfect appearance at
  the coded body pose with what the pose cannot describe charged; whether
  the outline problem shrinks at broadcast scale. A per-player appearance
  model is amortized over a match (hours), the condition H3d found
  decisive. The racket in a fixed grip re-tests H4a's grip pose. The
  question it answers is the total: keypoints plus appearance against the
  pixel coders' envelope (SVT-AV1, DCVC-UF), swept over the rendering
  resolution, so K4's keypoint share (22% of SVT-AV1's player bits at
  CRF 62) is judged there, not by H1's 10% screen.
  Done (2026-10-21, [entry](docs/experiments/k6.md#2026-10-21--k6-player-oracle-h3b-for-bodies-on-broadcast-tennis);
  K4's 9 broadcast windows, 8 scored from 5 matches; the stopping rule
  met at the first look). **The bound passes and the charged oracle
  fails, so the decision is the user's.**
  - The bound: the true appearance moved to the coded keypoints (K4b.3's
    code, every 4th frame, 3.6 kbps for two players) scores LPIPS 0.075.
    It saves 95% [91%, 98%] of the envelope's rate (SVT-AV1 and DCVC-UF
    from 1080p to ⅛; SVT-AV1 at CRF 62 is 42 kbps for 0.130). Every 2nd
    frame and every frame pass too, so every 4th stays the operating
    point; fast wrists cost +0.07 LPIPS there.
  - Charged: a reference frame of the same player warped to the coded
    pose scores LPIPS 0.46, worse than every pixel point on half of the
    windows. The coding error plays no part in that (uncoded 0.460). The
    outline is off by 16–26% missed and 32–46% overshoot at every player
    size, and the fixed racket grip fails, as H4a's did.
  - So keypoints carry the pose cheaply, and drawing the appearance is
    the open problem; retrieval plus a 2D warp is ruled out.
  - Amendment 2, with the adopted players: players are 91–93% of the
    foreground's bits and fitted on 0.989 of frames, and keypoints are
    10.3% of the players' SVT-AV1 bits at CRF 62 (K4: 22%).
  - Open for the user: whether to give a learned per-player renderer
    (pose-conditioned, amortized over a match) its own oracle, try
    keypoints plus a coded residual, or keep players as pixels in the
    tennis pipeline. Decided (user, 2026-10-21): K6b, then K6c.
- K6b. *Learned per-player renderer, oracle* (user, 2026-10-21). No
  pose-conditioned generator has been trained on PointStream's footage
  (H3e ran one off the shelf on hands). K6's `ref` was a 2D warp of one
  reference frame, not a learned model, so K6 rules out retrieval plus
  warp only. The bar comes from K6, at 3.6 kbps for two players (K4b.3's
  code, every 4th frame): break-even is about LPIPS 0.29 (DCVC-UF at
  4.3 kbps, 0.294), and the bound, perfect appearance at the coded pose,
  is 0.075. Judged by K6's judges against K6's envelope on K4's 8 scored
  windows.
  Steps, cheapest first. Each later step is taken only if the one before
  leaves the decision open, and each has its rule and GPU budget written
  before it runs (user, 2026-10-21: no training campaign that a cheaper
  check could have discarded).
  1. Audit. Reference-conditioned pose-to-person models (an appearance
     image plus a pose sequence in, video out) and per-subject models,
     with weights, licences, inputs, size and inference cost per frame.
     Small models were preferred at first (H3e's Wan2.1-14B took 27 min
     for 81 frames); from 2026-10-24 speed is secondary while quality is
     what is being tested (user).
  2. Zero-shot oracle, inference only. The audited model is given one
     reference crop of each player (granted, unbilled) and the coded
     keypoints. Its interface is K6c's, so a pass here decides K6c's
     route with no training at all.
  3. Overfit oracle, only if step 2 fails or no zero-shot model exists.
     A renderer (or a LoRA on step 2's model) is fitted on the window's
     own frames, H3d's oracle form. It has seen the frames it is judged
     on, so it bounds every renderer fitted elsewhere in the match. If it
     misses the bar, the generative route closes for bodies, at the cost
     of a few short fits.
  4. Match fit, only if step 3 passes: fitted on other shots of the same
     match, judged on K6's windows and paid for (amortized, payback hours
     reported, as G5b).
  Audit, first part (2026-10-21). A tennis fine-tune of
  Moore-AnimateAnyone (SD 1.5 ReferenceNet, pose guider, motion module;
  Apache-2.0) is already in `Models/AnimateAnyone/profiles/finetuned_tennis`,
  beside the original weights. It was trained in two stages in Feb 2026,
  before the reset, on 20 player tracks from all seven archived tennis
  videos, so it has seen K6's matches; the overlap is recorded and the
  clean subset reported (AGENTS.md).
  Pre-reset diagnostics are not evidence by today's rules, but they bear
  on the choice:
  - On 16 Alcaraz frames it lost 4.0 dB foreground PSNR to an articulated
    paste, with the target's alpha granted too.
  - It responded to its inputs: shifted poses and swapped references
    degraded it.
  - It smears the racket: COCO-17 has no racket.
  - Off-the-shelf generators lost to a pasted keyframe on player LPIPS
    (pre-reset BP10).
  Step 2 runs both profiles, the original and the tennis fine-tune, in
  K6's harness. That needs Moore-AnimateAnyone's source and an
  environment with its diffusers pin (the archived runtime pinned
  diffusers 0.30.3 and imported an external checkout). Neither is on the
  hosts today, and the current environments have no diffusers. So step 2
  starts with an environment audit and a packed environment. Its
  composite grants R as the generator's matte, the analogue of `warp`'s
  granted outline, so it bounds the generator with its own matte.
  **Step 2, zero-shot (2026-10-22, experiments.md "K6b (players) 2"):
  fails.** Moore-AnimateAnyone (the released weights) with one unbilled
  reference crop per player and the coded keypoints, inside the granted R,
  wins 3 of 6 windows on LPIPS and 3 of 6 on DISTS (score 50%; the two
  unjudged windows could not have lifted it to 75%). It is far better than
  K6's `ref` warp (LPIPS over S 0.21–0.38 against 0.39–0.55) and beats the
  envelope on both metrics on the three Djokovic windows (saving 2–73%),
  losing both Alcaraz windows (by 0.004–0.058) and the Rublev close-up.
  The bound is 0.05–0.12. The environment is `pointstream-jcv` extended
  (huggingface_hub < 1.0, conflict 10); first run on Ada recorded. Scope
  (user, before any score): the released profile alone decides; the
  tennis fine-tune was a diagnostic on one window, and no better there.
  4.75 GPU-hours, over the 4-hour ceiling because parallel parts slowed
  each other (reported to the user before judging). Next: step 3, the
  overfit oracle; its rule is written (ceiling 6 GPU-hours) and it waits
  for the user's approval.
  **Step 3, overfit oracle (2026-10-23, experiments.md "K6b (players)
  3"): passes.** AnimateAnyone fitted on each window's own frames (a LoRA
  on both UNets' attention and the pose guider, 7.5 minutes: the pilot's
  learning curve saturated there) wins 4 of 6 windows on LPIPS and 6 of 6
  on DISTS (score 83%), saving 35–84% of the envelope's rate on four
  windows. It improves every window over the zero-shot step (LPIPS
  0.185–0.300 against 0.211–0.381); it loses LPIPS on
  `alcaraz_hurkacz/045` by 0.003 (its fit got the fewest steps) and on the
  Rublev close-up. The fit saw the frames it is judged on, and its
  reference, matte and LoRA are unbilled, so it bounds a match fit. 4.8
  GPU-hours of the 6 approved. Next: step 4 (a fit on other shots of the
  same match, billed and amortized). **Its rule is written (2026-10-23,
  experiments.md "K6b (players) 4")**: one LoRA per match on up to 6 of
  K4b.1's other windows, the pose guider frozen, 8-bit adapter bytes
  billed over H = 2 hours (an assumption: the archive holds scene clips),
  a pilot over ranks {64, 16, 4} on `djokovic_federer`, ceiling 7
  GPU-hours. Size decides: step 3's adapter would cost 54 kbps over 2 h.
  **Step 4, match fit billed (2026-10-23): fails.** The pilot chose rank
  4 and a 10-minute fit (1.55 MB at 8 bits, 1.73 kbps over 2 h; it scored
  as rank 64 and as fp16). On the 6 windows, `gen-match` wins 4 of 6 on
  LPIPS and 4 of 6 on DISTS unbilled (67%), 3 and 4 billed (58%). It
  recovers about 60% of step 3's gain over zero-shot and pays back in
  0.4–2.1 hours where it wins, but loses `alcaraz_hurkacz/045` (fewest
  training crops; its fit failed the coarse `loss_falls` check) and the
  Rublev close-up. By the rule, the generative route for players closes
  at the keypoint rate, and step 3's extra windows came from fitting the
  judged frames. About 7.0 GPU-hours, at the ceiling. Not closed (user,
  2026-10-23): **step 5's rule is written (experiments.md "K6b (players)
  5")**. Three cumulative arms (step 4's recipe; all of the match's
  K4b.1 windows plus the pose guider trained; the same on a tennis-wide
  base LoRA that is part of the decoder) are chosen on two development
  windows that K6 scored and steps 2–4 never judged (`djokovic_zverev/008`,
  `medvedev_struff/033`). Then 4-bit adapters and guidance 2.0 are tried
  at render time, and the 6 windows are judged billed by step 4's rule,
  beside a per-player Djokovic model billed over 20 hours. Ceiling 11
  GPU-hours. Waits for the user's approval. Step 6 (outlined): a newer
  generator (Wan-Animate), after an environment audit.
  Training discipline for steps 3 and 4: a correctness smoke in minutes;
  then a learning-curve pilot on held-out frames at fixed fractions of the
  step budget; the scaled fit only if that curve, extrapolated, can reach
  the bar within budget (compute-preflight); every stage at most 45
  minutes and checkpointed.
- K6c. *General player codec* (user, 2026-10-21; after K6b passes). One
  model shared across players, sent once with the decoder. Per player it
  receives an appearance image or feature vector, sent once per match or
  shot and billed, then the coded pose sequence. Scope: court-view shots
  (K8b). Other shots (close-ups, crowd, replays) go to the pixel coder,
  which on them is PointStream's own route (the scene-adapted pixel coder
  with an enhancer, as H5/G6), not the generator. Whether that route beats
  the baselines there is measured with the full tennis pipeline. The codec
  may be hybrid, generating only the subsets where it pays, with pixels
  elsewhere. Subdivisions, each judged by K6's envelope on that subset:
  1. the far player (under 15% of the frame height), where pixel coders
     collapse first and a generator's errors are smallest in pixels;
  2. the body generated, with the racket arm or the racket alone as coded
     pixels or residual (K6: the fixed grip fails);
  3. between-point walking against rallies (motion and pose novelty);
  4. other people (officials and ball kids, 5.8% of SVT-AV1's broadcast
     bits at CRF 62 in K4) synthesized too, if it pays. Line judges barely
     move and are cheap as pixels, while ball kids run.
- K1c. *K6's bound on hands* (user, 2026-10-21: plan the cheap check).
  H3b's hand bound charged MANO's outline. K6's `warp` grants the outline
  (the frame's own pixels moved to the coded keypoints, only the coding
  error charged), and on bodies that is the difference between a 95%
  saving and a loss. Run K6's `warp` on H3b's 4 pilot items with H2's
  coded 2D joints, against H3d's envelope (DCVC-UF + G, SVT-AV1) on LPIPS
  over R. One fleet job of under 45 min (LPIPS and a CPU warp; the
  windows, masks and joints are recorded). If even this bound loses,
  hands are closed for every renderer. If it passes, hands get K6b's
  ladder. It does not wait for K6b.
  **Outcome (2026-10-21, `k1c-report.json` `085bb948…`): to the user.**
  On R at 1.93 kbps the bound beats the SVT-AV1 + DCVC-UF + G hull on 2 of
  4 items by LPIPS (P01_107 saves 96%, P03_10 78%), and on 4 of 4 by DISTS.
  The two LPIPS losses (P02_02, P09_106) are where H2 has no fit for 40%
  and 27% of R. Those pixels are charged as fill, so the shortfall is the
  estimator's coverage plus the coding error, not the outline. Options:
  close hands by LPIPS's letter, or give fitted hands K6b's ladder after
  a coverage fix (H2b's hybrid, or a pixel coder for unfitted hands, with
  its own bound first).
  **K1d (user, 2026-10-21: "let's try it"):** WiLoR on every masked hand
  without an H2 fit (`k1d-2026-10-21/report/k1c-report.json` `6427ea8f…`).
  Coverage is full, but the result is the same: 2 of 4 by LPIPS (P01_107
  and P03_10 save 96%; P09_106 misses by 0.003, P02_02 by 0.07) and 4 of 4
  by DISTS, so it goes to the user. The mask-box fits jitter (coded shifts
  of 24–64 px) and double the rate on the two hard items.
  **Decision (user, 2026-10-21):** the decision criterion is the mean
  over LPIPS and DISTS of the share of items won, ≥ 75%. It applies to
  later runs and, disclosed as post-hoc, to K1c and K1d, which pass under
  it (75%). Hands go to K6b's ladder. K1e (hand-only boxes) is running.
  Then K1f: the scope (an encoder-measurable predictor, fixed on the
  pilot) tested on B2's other 30 items, with the pixel envelope coded
  there (up to 8 GPU-hours, approved).
  **K1f outcome (2026-10-22, `k1f-report.json` `2f6e4291…`):** on 22
  held-out items, the hand bound (appearance granted, 21 coded keypoints
  at about 2 kbps, every masked hand fitted) is below the SVT-AV1 + DCVC-UF
  + G hull on **22 of 22 by LPIPS and by DISTS**. Median LPIPS on R is
  0.084 against the hull's 0.175, and on 14 items it beats every pixel
  point up to about 45 kbps. No scope is needed. K1f's envelope path
  reproduces H3d's recorded points on P01_107. **Over all 30 held-out
  items (2026-10-22, `k1f-report-30.json` `d31b1fc2…`): 30 of 30 by
  LPIPS and by DISTS** (median 0.086 against 0.174). Next: K6b's ladder for
  hands (a renderer that has to earn this appearance).
  **K6b for hands, audit (2026-10-22, experiments.md "K6b (hands) 1"):**
  the renderer is local: it draws a hand crop from a reference crop and
  the coded keypoints, inside the granted R. That avoids H3e's scene
  drift. FoundHand (reference plus 2D keypoints, 256², released; it has
  seen VISOR, so the overlap is recorded) is the zero-shot candidate, and
  K6's `ref` warp of the same reference is the floor. AnimateAnyone stays
  the body candidate. Next: rung 2, the zero-shot oracle on positions 0–7
  of K1f's order (rule written; ceiling 3 GPU-hours). Before it: the
  FoundHand source and weights into `Models`, and an environment check.
  **Rung 2, zero-shot (2026-10-22, experiments.md "K6b (hands) 2"):
  fails.** FoundHand with one unbilled reference crop and the coded
  keypoints wins 0 of 3 held-out items by LPIPS and 1 of 3 by DISTS
  (score 17%; a 4th item, not run, could not have changed it). Its LPIPS
  on R is 0.28–0.52 against the hull's 0.17–0.26 and the bound's
  0.06–0.10. A warp of the reference does as well or better. The 4
  items at 50 steps were the user's decision when 8 did not fit 3 GPU-hours.
  Next: rung 3, the overfit oracle (FoundHand fitted on the window's own
  frames). Its rule is written, with a ceiling of 4 GPU-hours, and it
  waits for the user's approval. If it fails, the generative route
  closes for hands.
  **Two tracks (user, 2026-10-22):** fidelity (encode the original) and
  experience (the same experience despite pixel differences: realism by
  KID, same content by the object's estimator, same over time by
  flicker). The protocol is drafted in
  [experiments](docs/experiments.md#protocol), with the thresholds set
  (user, 2026-10-22). Results closed on fidelity are untested on experience. The
  re-judging queue is H3e, FoundHand, H3a/H3a2, and K6 `ref`: item J.
  **Rung 2b, a bank of free references (2026-10-22): fails by one
  item-metric.** Each frame is warped from the in-window reference nearest
  in coded pose. With 16 references the score is 67% (P22_107 and P25_09
  won on both metrics; P06_108 lost DISTS 0.209 against 0.206); K = 1 and
  4 score 33% and 50%. Quality rises steeply with K. Next for hands, on
  fidelity: the buildable bank (K up to 64, chosen outside the window,
  coded and billed once per video). It needs keypoints outside the
  windows, so it is a task of its own. Rung 3 waits behind it.
  **Rung 2c, the buildable bank (2026-10-22, experiments.md "K6b (hands)
  2c", `k6bh2c-report.json` `3e46782b…`): fails, and the free bank fails
  too.** Up to 64 references per hand, chosen from VISOR-masked frames
  outside the window (WiLoR on 529–1,723 frames per video), coded as
  SVT-AV1 stills with polygon masks and billed once per video (0.3–2.5
  kbps), score 12.5% at K = 16, 32 and 64 (LPIPS 0 of 4, DISTS 1 of 4), and
  25% unbilled. Clean of G's footage: 0%. LPIPS on R is 0.23–0.50 against
  the hull's 0.14–0.25. The bits are not the problem, and neither is K
  (16 → 64 moves LPIPS by at most 0.03). References from other moments of
  the video do not carry the forearm, sleeve and lighting: the warp covers
  46–91% of R, against 95–97% for the window's own pixels. With 2b, the
  reference route for egocentric hands closes at a buildable rate. Next on
  fidelity: rung 3 (the overfit oracle, rule written, 4 GPU-hours), the
  last open rung for hands, waiting for the user.
  **Rung 3, the overfit oracle (user approved, 2026-10-22; experiments.md
  "K6b (hands) 3: overfit oracle (run)", `k6bh3-report.json` `07ac515b…`):
  passes, 83%.** FoundHand fine-tuned for 20 min on each window's own frames
  (weights unbilled) draws the hand from the coded keypoints at about 2 kbps
  below the SVT-AV1 + DCVC-UF + G hull on R: LPIPS 2 of 3 (P06_108 0.162
  against 0.172, P22_107 0.174 against 0.236; P25_09 loses, 0.291 against
  0.260), DISTS 3 of 3. Clean of G's footage: 75%, at the bar. The fit
  roughly halves rung 2's zero-shot gap (LPIPS down 0.11–0.23), and the
  pilot's held-out curve was still falling at 30 min. The margins on the
  hard items are thin (0.006–0.010). It is an oracle: fitted on the frames
  it is judged on, unbilled, and FoundHand may have seen the footage. Next by
  the rule: rung 4, a fit on other shots of the same video, billed. A visual
  check (user, 2026-10-23; P06_108 re-fitted, 0.163 / 0.180) shows the
  source's pose, skin and lighting with softer texture, blur where fingers
  cross an object, and flicker 25% above the source's. **Rung 4 (entry "K6b
  (hands) 4", `k6bh4-report.json` `3690a96f…`): fails, 25%.** The same fit
  on 240 frames from other shots of the video (outside the window, fit
  unbilled) lands on rung 2's zero-shot numbers (P06_108 0.321 / 0.268,
  P22_107 0.283 / 0.194 LPIPS / DISTS on R), and none of rung 3's gain
  transfers. Stopped after 2 of 3 items, once decided. Hands on fidelity
  close at the keypoint rate; K1f's bound stays a bound, and hands go to the
  experience track (item J).
- K7. *HiEve audit* (user, 2026-10-14: open a domain if it serves
  PointStream; [docs/vcm.md](docs/vcm.md#datasets-vcm-and-fcm-evaluate-on)).
  HiEve (crowded events, surveillance-style footage; manual 14-joint poses, tracks
  and actions; CC BY-NC-SA 4.0) holds the five videos MPEG's FCM conditions
  use for tracking, and their labels are public. K4's audit on them: people
  per frame, their share of pixels and of SVT-AV1's bits, the camera's
  stillness, a body-pose estimator's fit against HiEve's poses, and the
  keypoint rate against the bits on those people. Decides whether the body
  mode (K6) is also judged there, by MOTA with JDE and pose AP against
  VTM and HM through CompressAI-Vision, on MPEG's own videos. CPU first;
  the download (registration may be needed) is the first step.
Follow-ups from K4–K5 (user, 2026-10-14), in the order they gate each
other. Each has its rule written before its run, as everywhere.
- K8. *Court-view shots.* Split a match at its cuts (ffmpeg scene score)
  and send to PointStream only the shots that show the court; close-ups,
  crowd shots, replays and graphics go to the pixel coder. The old
  pipeline (`9cf3012`, `scene_classification.py`) called low-motion
  scenes points, which let a close-up and an outro card into K4's
  windows. Here a shot is a court view when its court lines register to
  the court model (the `Tennis_courts` reference) with a homography; the
  same homography gives the court outline that K4b uses to find the
  players and the per-frame camera registration a panning broadcast
  camera needs (K4: still on 35% of a match). Shared by foreground and
  background: G's per-shot background needs the registration, H/K the
  court outline and the shot list. Judged on the archived matches' scene
  clips, reviewed by eye, and on the whole match's shots.
  **Not adopted (2026-10-20)**, scored on a stratified sample of 82
  shots marked by the user ([entry](docs/experiments/k8.md#2026-10-16--k8-court-view-shots-and-camera-registration-rule)).
  Precision is 1.00 (all 41 predicted court shots are court), with a
  registration median of 0.91 px on them. Duration recall is 0.34
  [0.26, 0.45]: most court time is missed. The misses fall into three
  classes. 11 of 17 sampled misses fail only the 80% frame rule. 6 have
  wrong or no fits. 6 shots are mixed (missed cuts). A one-frame decode
  offset is fixed in code. Open for the user: whether to revise the
  registration (temporal tracking through a shot, a shot rule on the
  median error) and the cuts, under a new rule judged on unmarked shots.
  Or K4b's candidate (d) could use only the 41 precise predicted-court
  shots.
  **K8b adopted (2026-10-21)**
  ([entry](docs/experiments/k8b.md#2026-10-20--k8b-court-recall-in-shade-and-across-venues-rule)).
  The changes: a line detector relative to the local background (works in
  shade), hypotheses from both detectors, line-swap re-seating, a
  centerline ruler, a clutter guard, and a 6 px shot rule. On 69 held-out
  shots of three broadcasts the user's marks and the system agree on
  every shot. Precision is 1.00 (44/44, lower bound 0.934) and recall
  1.00; the registration median is 0.77 px and the p95 2.71 px. Per-frame
  homographies for those 70 clips are published for K4b (d) and G.
  djokovic_federer and rublev_auger-aliassime are registered too
  (2026-10-21), so all five of K4b's matches are covered. nadal_kyrgios's
  whole match (v1 of the court-shot list, with a spot-check of its
  predicted court shots) waits until the full tennis pipeline needs it
  (user, 2026-10-21).
- K4b. *Players before K6* (re-reads K4's (b) and the K6 gate).
  1. Identification. Candidate rules, content-blind: motion in body
     heights (K4's amendment), horizontal motion with the clip's end
     discounted, a racket mask touching the person on most frames,
     inside the court outline (K8), keypoint motion, and combinations.
     Chosen on players marked by hand on windows K4 did not measure (a
     review page where the user marks the two players on a few dozen
     frames; RacketVision has no player labels), then applied once to
     every window as amendment 2, with both earlier definitions' numbers
     kept. Scored (2026-10-21, [entry](docs/experiments/k4b.md#2026-10-16--k4b1-identifying-the-two-players-rule);
     features on a 240-frame window per marked frame, user's amendment):
     no candidate reaches 0.95. Best (a)+(c) 0.875 (42/48), then (b)
     0.813, (c) 0.667, (a) 0.625, (e) 0.542. The misses mostly take a
     second person for the far player. With K8b's homographies
     (2026-10-21): **(c)+(d) adopted**, 36/37 (0.973) on the marked frames
     K8 registers as court views (0.938 on all 48); (d) 0.919, (a)+(d)
     0.946. Amendment 2 is folded into K6 (2026-10-21, user delegated the
     call): with K4b.2 and K4b.3, K6's gate passes on every condition
     whatever K4's two mislabelled windows hold ((a) 86–89% and robust to
     them in K4; (b) 0.997; keypoints 8–36% of SVT-AV1's player bits), so a
     standalone re-run of K4's audit (1–2 GPU-hours) decides nothing. K6
     runs the adopted pipeline (K8 court shots, (c)+(d) players, ×1 crops,
     K4b.3's codes) and includes K4's 9 broadcast windows, so K4's numbers
     with the adopted players come from K6's run, beside both earlier
     definitions'. TrackNet has no court registration.
  2. Pose on the far player. Top-down on each SAM player crop, enlarged:
     a cheap sweep of the crop size (three sizes, native to about 4×)
     measuring fit and latency, then a top-down model (DWPose, weights in
     `Models/DWPose`) if the sweep falls short, super-resolution of the
     crop last; fine-tuning only if all fail (no tennis pose labels;
     SportsPose or pseudo-labels from fitted near players). Players under
     10% of the frame height are fitted on 0–50% today, 15% and taller on
     97–100%. Done (2026-10-21, [entry](docs/experiments/k4b.md#2026-10-21--k4b2-pose-on-the-far-player-crop-size-sweep)):
     **crop ×1 passes**; on 48 windows' hand-identified players, fitted
     0.997 (far 0.993, whole frame 0.812 / 0.595), synthetic OKS ≥ 0.5 on
     0.991. K4's (b) passes with it; DWPose not needed. K4b.3 reads its
     keypoints.
  3. Keypoint rate against keypoint error: quantization 1, 2, 4, 8 px,
     constant-velocity prediction, a lower keypoint frame rate with
     interpolation; raw keypoints saved this time (K4 kept histograms
     only). Which error is acceptable is K6's to judge on the rendering. Done (2026-10-21, [entry](docs/experiments/k4b.md#2026-10-21--k4b3-keypoint-rate-against-keypoint-error)):
     q = 4 px, constant velocity, every 4th frame, **3.3 kbps for two
     players** (median OKS 0.96 far, 0.99 near; p95 8 px on fast limbs);
     every frame 9.5 kbps (p95 2.4 px). The curve goes to K6.
- K5b. *Ball mode.* (1) The renderer's oracle: from the coded position,
  velocity and a per-window colour and size sent once, a sharp disk and
  an elongated blur, composited over the decodes and billed; scored as K5,
  and shown side by side with the source at equal rate for the user's
  preference (a sharp ball that does not look like broadcast video may be
  preferred). (2) An encoder-side ball tracker trained on TrackNet's and
  RacketVision's ball labels (PLAN's ball stage); it is also the judge,
  since an off-the-shelf detector finds 46% of the source's balls and
  SAM 3.1's text prompt 56%.
  Outcomes (2026-10-21, [entries](docs/experiments.md)): (1) no renderer
  adopted: a per-window appearance keeps 10–20% of the pixel oracle's
  gain (BD about 0% against the −25% bar); the diagnostic points to
  per-frame position and appearance. The user's rendering preference
  (disk or streak) is pending. (2) RacketVision's BallTrack adopted off
  the shelf as tracker and judge (0.918 on K5's TrackNet games, never in
  its training; 0.881 on RacketVision test; 0.022 false detections per
  frame); no training needed. Through SVT-AV1 at CRF 62 every tracker
  finds under 3% of the balls.
- K5b 1b. *Per-frame ball*: K5b 1's oracle with the
  encoder refining each position by fit (±4 px, billed through the same
  residual code) and sending the appearance per frame (setting and
  colour predicted from the previous frame, billed); same decodes, judges
  and rule; positions from BallTrack as well as the labels, so the ball
  mode is measured end to end. Judges (user, 2026-10-21): BallTrack
  detection and crop DISTS. Outcome (2026-10-21, [entry](docs/experiments.md)):
  not adopted, on rate: the drawn ball restores detection (0.01 → 0.90 at
  DCVC-UF's 13 kbps; 5–20× less rate than the pixel coders for 80% of the
  source's balls) and passes crop DISTS (streak −27% to −42%), but its
  ~1 kbps payload fails the frame-PSNR guard (+1.7% to +2.8% against
  +1%). Decision (user, 2026-10-21): the ball mode is kept as a win
  despite the guard (about 1 kbps is small; the ball matters in tennis),
  the guard's failure disclosed; the **streak** is the default renderer
  (the disk often looks unrealistic; the streak matches DCVC-UF where
  DCVC-UF keeps the ball and keeps it visible far more often). A cheap
  check (not evidence) finds colour coded at step 8 saves about 20% of
  the payload for 0.3% of the fit's gain; a codec default to set before
  the final runs, not a reason to rerun now.
Datasets: still-camera sets (TrackNet, RacketVision, OpenTTGames) are
fine where a decision needs labels, with the scope stated in the paper;
broadcast matches answer what a real broadcast costs.
Order: H5 continues on egocentric (a working pipeline with what is
chosen: G's background, DCVC-UF + G_HO on hands and objects, H4c), since its
bitstream, compositor and harness carry over to tennis. K4–K5 and K2 run
beside it in their own sessions; K1 and K3 are cheap and egocentric; K7
follows K4's method once K4 has run.

### After a working pipeline

Optimisation, training and extensions, ordered by what G and H show they need.
VVC (VVenC, `/opt/local/bin/vvencapp` on the hosts) joins the baselines here
(user, 2026-10-08: SVT-AV1 is enough until then).

#### H3d follow-ups (deferred, user 2026-10-13)

H3d adopted DCVC-UF with a generic super-resolution network (G) for the
hand pixels; optimizing it waits for a full pipeline. Deferred, each with
its own oracle when taken up:
1. *Person network with a fallback* (best of both). K beat G where the
   person looked as usual (−43%, −55%) but recoloured P02_02's gloves toward
   skin. A per-person network that defers to G on content unlike its
   footage (by a novelty test on the decoded hand, or by sending a one-bit
   choice per segment, billed) would keep K's gain without its failures.
2. *Longer videos for the per-video network.* Fitted outside the window,
   V kept about half of the oracle's gain (−37% free of bits) from 1,152
   frames of 1.7–3.5-minute videos, and its bits sank it. Longer videos give
   more of the person's own footage (expected to retain more of the oracle's
   gain) and more seconds to amortize the network over. Measure both on
   long recordings, e.g. Egocentric-10K (archived on disk,
   `Datasets/archive/pre-reset-2026-10-05/`).
3. *G at scale:* more footage and recordings, a longer schedule, network
   size; then G on all 34 items.

#### D1. Segmentation benchmark on VISOR and EgoHOS

Moved after a working pipeline (user, 2026-10-08): SAM 3.1 already segments
well enough to build on, and choosing a faster segmenter is an optimisation that needs the
pipeline's own timing.

SAM 3.1 (text and prompted) and YOLOE-26 against the labels, per class, accuracy
against speed. EgoHOS is scored on single images only. On VISOR: evaluation set
v2, the "hold the first frame" floor (foreground J 0.459) as the table's first
row, and every number with and without B1b's fill
([Using VISOR](docs/resources.md#using-visor)).
**Done when** the table is reproducible from a recorded job.

#### E1. Handled-object proposer

A hand-object proposer trained on VISOR (contact relations) or EgoHOS (object
orders) prompts SAM 3.1. Cross-tested on the other dataset and on HOT3D object
masks. VISOR training follows [Using VISOR](docs/resources.md#using-visor):
train split only, drift ≤ 1, no frames with a hand gap as targets.
**Done when** cross-dataset numbers decide whether to commit to it.

#### F1. Training-data export

VISOR foreground crops and masks per instance, plus background frames with the
foreground removed, written by `python -m src.segmentation dataset` with
provenance and each mask's drift; training admits drift of at most 1 frame.
Background frames and segmenter targets skip frames with a hand gap (B1b
found no hand fill fit for them; excluding keeps 75% of admitted train
frames, [Using VISOR](docs/resources.md#using-visor)).

## Phase 2: racket sports

### C. SAM 3.1 from label prompts

Players and rackets from RacketVision boxes and keypoints; the ball from point
prompts. OpenTTGames masks check the objects: whether SAM finds the right
players, table and scoreboard. Their 320×128 resolution cannot grade borders.

### Players stage

Foreground and background encoding with players as the foreground, scored on
OpenTTGames. Its camera is fixed, so this is the favourable case for a
panorama background.

### Ball stage

Ball tracker trained or fine-tuned on RacketVision, cross-tested on OpenTTGames
and TrackNet (the only dense ball labels). A parametric ball trajectory is the
candidate encoding.

### D2. Segmentation benchmark on racket sports

As D1, per dataset and class, with the ball scored by point metrics.

## Throughout

### J. Experience-track re-judging (user, 2026-10-22)

Results closed on fidelity are untested on the experience track (realism
by KID, same content by the object's estimator, same over time by
flicker; [protocol](docs/experiments.md#protocol)). J judges the four
queued results on it, cheapest first, each against the anchor hull at its
own rate. The rule, judges and budget are in [docs/experiments/j.md](docs/experiments/j.md), "J:
experience-track re-judging (rule)".

1. H3e V1, from its saved generations and judge archives (CPU, Inception
   and WiLoR only).
2. FoundHand `gen` (K6b hands rung 2), re-run on positions 0–3 of K1f's
   order keeping the crops (≤ 3 GPU-hours).
3. H3a/H3a2, only if their renders were kept; otherwise recorded as not
   re-judged.
4. K6 `ref` on players (CPU warp, YOLO pose).

A result passes when at least 75% of its items pass all three checks.
A pass reopens that route on the experience track only, and its next
step goes to the user. A fail closes it on both tracks.
**Outcome (2026-10-22): all four routes stay closed.** The user kept
the drafted time reading. H3e V1 fails on content (P01_107 and P03_10
above SVT-AV1 alone at every point). FoundHand `gen` fails on time (2 of
2 items scored, which decides; energy 4–6 × the source's). K6 `ref`
fails on time (8 of 8 windows). H3a/H3a2 were not re-judged (outputs not
kept). Every decision came from content or time against SVT-AV1's points
alone, so the held-out KID reference sets were not built. Beside: K1f's
bound (`warp`) fails time on both hand items too, so the coded
keypoints' own jitter is the binding problem on this track. Whether to
measure and smooth it is open for the user. GPU: 0.49 + 0.76 hours.
**J2 (2026-10-22, user: "go ahead"): the estimator is the cause.** Decoder-side
smoothing of the coded keypoints never makes `warp` pass time (energy
2.8–3.0 × the source's at best), and neither does removing the
estimator's own noise without coding (2.6–2.9 ×). Next, if pursued: an
identity warp to separate the estimator from K1c's compositing (fill,
outline). GPU: 0.53 hours.

### I. Paper

Rescope with a fresh paper repository state: egocentric first, racket sports as
the second domain or as future work. Write the dataset and evaluation-protocol
sections (weighted PSNR on dataset masks, provenance tiers) and the hand-pose
estimator comparison from H. Set the venue and
submission date. Wherever a model is delivered beforehand and not billed
(G5b's scene model), report beside the rate the playback time after which its
bytes are repaid (user, 2026-10-10; G5b's pilot: 2.8–14 h for 241 MB).
Results to state when writing (user, 2026-10-13), if they still hold in
the final experiments: a per-video network (NAS) does not pay for itself on
short egocentric videos (H3d: 1.7–3.5 minutes; even 86 KB costs more than
it saves at the hands' rates), so networks are better delivered once; and
the generic network's gain on the hands is perceptual (LPIPS and DISTS
better, PSNR worse, a slight colour shift, no invented texture), as G5d's
enhancer on the background.
An observation for the tennis paper (user, 2026-10-21; K5b 2): pixel
coders erase the ball at low rate, measured with a valid ball detector.
RacketVision's BallTrack finds 0.938 of TrackNet's balls on games 8–10
(0.918 on K5's games, none in its training); through SVT-AV1 preset 8 at
CRF 62 (median 84 kbps, frame PSNR 33–34 dB) it finds 0.026, and every
other tracker of WASB's zoo under 0.03. K5's YOLO26x figure (0.02) came
from a detector that found only 46% of the source's balls. DCVC-UF,
measured with BallTrack in K5b 1b: 0.01 of the balls at QP 0 (13 kbps),
0.15 at QP 21, 0.67 at QP 42 (118 kbps); with the ball mode's streak
(about 1 kbps) 0.90–0.92 at every point.
A figure for the tennis paper, if space allows (user, 2026-10-21): the
ball at equal rate, source | DCVC-UF | + disk | + streak, on two frames
the user chose from K5b 1b's side-by-side clips (the far player's ball,
where DCVC-UF loses it and the disk looks unrealistic): tracknet/game3/Clip3
at DCVC-UF QP 42, frames 35 and 71 of the window (ball-following 3× crop;
matched to the user's screenshots, job `20261021T084417Z-048cc9d4`);
regenerated from the final experiments' streams, as every figure is.
Negative results to report (user, 2026-10-13), with their numbers:
parametric motion fails for egocentric objects (H1's homographies, H4a's
grip pose on HOT3D motion capture, H4b's bank of views), and keypoints
lose to pixels on egocentric hands at equal quality (H3b, H3e) while
carrying the pose for less rate (H3c2). Possibly two papers, in order, if
the final results hold (user, 2026-10-13): one where keypoints beat the
standard codecs (tennis), one on the learned pixel coders that beat the
keypoints on egocentric video. Order of writing (user, 2026-10-13): a
definitive pipeline; a first draft complete but for final touches and the
figures' final experiments; the final experiments (run once, after the
figure list is fixed); final touches before submission. The framing waits
for the pipeline and K4–K6.
Ablations to report (user, 2026-10-21), tennis players: (1) how the
two players are found among everyone on screen. Content-blind candidate
rules frozen before the user marked players on 48 frames (motion in body
heights, horizontal motion, racket contact, inside the court outline from
K8's homography, keypoint motion, and combinations), scored on the marks:
racket contact plus inside the court wins (0.973 on court views; K4's
motion rule 0.62), and the shot filter (K8) is
part of it, since off-court shots carry the misses (K4b.1). (2) How the
far player's pose is found: whole-frame YOLO fits it on 0.595 of frames,
a top-down crop at native size on 0.993 (K4b.2; 0.04 → 0.99 for players
6–10% of the frame height). Method to state with them: frames chosen
content-blind, rules frozen before the labels were opened, and the marks
the only labels. (3) What the keypoints cost (K4b.3): quantization
(1, 2, 4, 8 px), prediction (previous, constant velocity) and keypoint
frame rate (every frame, 2nd, 4th, interpolated) against the error, from
15.0 kbps (0.4 px median) to 3.3 kbps (1.8 px median, but 8 px p95 on
fast limbs); quantization is cheap in error, frame rate is the lever
whose error shows on swinging arms, which K6 judges on the rendering.
[K4b.1](docs/experiments/k4b.md#2026-10-16--k4b1-identifying-the-two-players-rule),
[K4b.2](docs/experiments/k4b.md#2026-10-21--k4b2-pose-on-the-far-player-crop-size-sweep),
[K4b.3](docs/experiments/k4b.md#2026-10-21--k4b3-keypoint-rate-against-keypoint-error).
A point to argue (user, 2026-10-13), scoped to the evidence when written:
conventional codecs are one-size-fits-all, rigid in what they code and tuned
to pixel fidelity rather than to what people see; a learned codec's
strength is less its margin over AV1 than how much further it goes when
domain knowledge directs it. Evidence so far: DCVC-UF off the shelf is
−63% against SVT-AV1 on the hands (H3c), and directed by PointStream
(region rate split by its own entropy model, coding at ¼ and ½, a generic
enhancer) a further −22% (H3d); the scene-adapted background model (G5b,
G5c) is the same pattern. Limits to state with it: retraining DCVC-UF
itself on the domain gained little and did not transfer (H3c: 14% on the
test windows, 151% worse on other participants), so the direction works
through what surrounds the codec (scale, regions, enhancers, scene
adaptation) more than through retraining it; and SVT-AV1 with the same
enhancer was unclear (H3d oracle: −19%, synthesized texture), which
supports, weakly, that the learned codec composes better with learned
parts.
Equal adaptation, as it stands (H5p 3, H5q 1; 2026-10-21): against a
whole-frame DCVC-UF adapted to the venue the same way, neither model
billed, PointStream loses on LPIPS (+16%; +6% with the best foreground
per point) and wins on DISTS (weighted −15%, frame −22%, 4 of 4) and
weighted PSNR (−11%). H5q decided (2026-10-21): no LPIPS win at
equal adaptation. Reporting (user, 2026-10-21): every metric is reported
with its result, DISTS (and PSNR where it wins) may lead and LPIPS's loss
is stated beside it; no account of which metric was planned as the
headline is needed. State the metrics' disagreement; that the split's
LPIPS cost is structural (F coded in both streams) and grows with F's
share of the frame (P09_106, 2%: −15%; P02_02, 50%: +19%); and that the
foreground alone matches the venue codec at high rates once adapted.
Checked (2026-10-16, [analysis](docs/experiments/h3d.md#2026-10-16--does-the-learned-codec-compose-better-with-a-learned-enhancer-analysis-of-h3ds-oracle)):
over its own base, the same enhancer gains about as much on SVT-AV1 as on
DCVC-UF (LPIPS −73% against −76%; PSNR −35% against −12%), so "the
learned codec composes better" is not supported. The argument to make
instead: DCVC-UF is the better base (−63%, H3c), and learned parts
directed by domain knowledge add large gains on top of a codec. A G
fitted on SVT-AV1 decodes would only refine this and is not planned.
A candidate secondary contribution (user, 2026-10-21): the court-view
classifier and registration (K8b) and the hand-marked shot labels.
Evidence: 69 held-out shots from three broadcasts, with the user's marks
and the system agreeing on every shot (registration p95 2.71 px), plus
K8's 82 marked nadal_kyrgios shots. Before claiming it: (1) a related-work
check, since broadcast court registration from lines is an established
topic and the novelty has to be stated against it (shade-robust
detection, line-swap re-seating, a clutter guard), not assumed; (2) the
labels are per-shot classes for 151 shots of four broadcasts, not
per-frame court keypoints, so they are better called a released evaluation
set than a dataset, and their size is stated; (3) only labels and clip
identifiers can be released, since the footage is broadcast material. It
may also be framed as the component that makes the tennis pipeline
automatic, rather than as a contribution of its own.
