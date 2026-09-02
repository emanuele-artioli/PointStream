# BP46 — Long eligible tennis-scene manifest

**Roadmap ID:** D1  
**Preferred harness:** Cursor or VS Code with Antigravity.  
**Outcome:** validated 2/4/8/16-second first-domain inputs for the low-rate search
and later six-video confirmation.

## Read

`AGENTS.md`, `plans/ROADMAP.md` §§3–4,
`plans/TERMINOLOGY.md`, the existing BP21 clip-cache reader and dataset path
contracts. Do not read raw files serially; batch on this NFS host.

## Eligibility rule

Record features first; do not silently cherry-pick by result:

- duration available at 24 fps;
- camera/background-motion statistic and panorama canvas growth;
- camera/view/background-context identity;
- number and known class of foreground objects;
- object size, separation, occlusion and track continuity;
- paste-back alignment check;
- whether the scene should route to PointStream or conventional fallback.

The search set may favor clearly eligible content. The confirmation rule is
frozen before selecting the six-video set.

## Deliverables

- a machine-readable manifest with source ID, timestamps, hashes, scene lengths,
  context ID and eligibility features;
- two diagnostic videos spanning near-static and smooth-pan eligible cases;
- at least six independent videos/matches reserved for confirmation;
- one high-motion ineligible control;
- extraction/validation command and submitted/succeeded/failed counts.

Do not commit or redistribute source video. YouTube-derived footage remains
under the external data root. The artifact may publish code, configuration,
hashes and legally permitted metadata only.

## Validation

- exact frame counts at 48/96/192/384 frames;
- dimensions, frame rate and colour metadata recorded;
- object tracks cover the requested interval;
- source frames, masks, motion and appearance references align;
- context IDs do not group unrelated cameras/backgrounds;
- failures stay in the manifest with reasons.

## Completion report

Follow `plans/SESSION-REPORT.md`. This work reports a corpus/manifest, not a
codec result and not a winning claim.
