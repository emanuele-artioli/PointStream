# Annotation workflow

## Dedicated editors for future tasks

The user asked on November 15, 2026 to carry forward the findings of
[Explore annotation platforms](codex://threads/01a1deda-cc0c-70e1-b45a-9d4f60a8644c).
Start future annotation tasks with that session's shortlist before extending
the bespoke review editor. Its findings were based on documentation, not a
successful application trial on PointStream footage.

| Candidate | Role identified in that session | Qualification still needed |
|---|---|---|
| CVAT | First shared-workspace candidate: video, masks/polygons, tracks and skeletons; REST/Python SDK can import saved model predictions. | Deployment, exact exports and editing/recovery. Do not assume Community includes the documented Online/Enterprise SAM2 tracker, or a verified native SAM 3.1/MCP integration. |
| X-AnyLabeling | First desktop alternative: editable geometry and documented SAM 3.1 multiplex video integration through a companion server. | The documented workflow uses 1008 input resolution, forward propagation and one selected target per point-prompt session. Test tiny balls, mid-clip correction, interrupted propagation and checkpoint recovery; editor MCP control was not verified. |
| Label Studio | Alternative for geometry plus classification/review questions, with an official HumanSignal MCP server and custom ML backends. | Edition restrictions: the session found the newer video polygon/skeleton interface in Enterprise/Starter Cloud. MCP task/prediction access does not establish arbitrary vertex editing. |
| FiftyOne | Optional dataset curation and model-comparison layer; local MCP server and CVAT integration. | Preserve our metadata through the integration; keypoint editing was not established. Distinguish the local server from paid enterprise agents/gateways. |

The session also identified Supervisely/V7 as commercial alternatives. Check
current capabilities, licence and edition constraints before adoption. A new
annotation platform or business is outside PointStream's submission path;
an adapter to an existing editor is the first integration to evaluate.

Sources retained from the session:
[CVAT model SDK](https://docs.cvat.ai/docs/api_sdk/sdk/auto-annotation/),
[CVAT skeletons](https://docs.cvat.ai/docs/annotation/manual-annotation/shapes/skeletons/),
[CVAT tracker](https://docs.cvat.ai/docs/annotation/auto-annotation/segment-anything-2-tracker/),
[X-AnyLabeling SAM 3.1 guide](https://github.com/CVHub520/X-AnyLabeling/blob/main/examples/interactive_video_object_segmentation/sam3/README.md),
[Label Studio video interface](https://labelstud.io/tags/videovector),
[HumanSignal MCP](https://github.com/HumanSignal/label-studio-mcp-server),
[FiftyOne local MCP](https://github.com/voxel51/fiftyone-mcp-server),
[FiftyOne–CVAT integration](https://docs.voxel51.com/integrations/cvat.html).

## Small adoption trial

Use the saved 18-frame VISOR pilot and a few fixed tennis examples containing
racket keypoints, blurred balls and absent objects. Reuse saved model outputs;
model inference is separate from editor qualification. Keep the existing
review package and decisions intact. No application has yet passed this trial,
and this record launches no installation, external upload, paid service or GPU job.

1. Define the target before editing: people/hands and directly manipulated
   objects for the revised VISOR pilot. Specify sleeves/forearms, occlusion,
   uncertainty and empty foreground explicitly. Held objects can remain relevant
   while stationary; a text segmentation is not proof of interaction.
2. Import exact source frames and model proposals into a disposable local task.
   Keep codec outputs and quality scores hidden. Use the same frozen frames and
   initial proposals in each editor; retain the original blind assignments in
   the audit record rather than treating visible re-review as a fresh blind test.
3. Exercise instance selection/deselection, mask and polygon correction, holes
   and disconnected regions, keypoint movement/visibility and persistent IDs.
   Test undo, draft recovery, navigation and interruption before scaling.
4. Export and independently compare frame index/hash/geometry, object IDs,
   masks and keypoint coordinates/visibility with the imported data and edits.
   Preserve raster masks losslessly; make any polygon/raster conversion and its
   loss explicit. Keep unsupported metadata in a versioned sidecar bound to the
   source and export hashes; missing or unreviewed labels must not become empty.
5. Record setup and active correction time separately, missed objects, remaining
   geometry errors, export fidelity and recovery failures. Select the editor
   after this bounded comparison. Propagation is a later, separately validated
   step; propagated but unreviewed frames remain a separate provenance tier.

## Reference and provenance contract

Keep source → exact frame → model/checkpoint/prompt → proposal → human edit →
reviewer decision → dataset release traceable. API/MCP writes create proposals,
not human approvals. Preserve rejected/uncertain decisions, explicit confirmed
empty frames, reviewer identity and original model outputs. Approval applies to
the final complete displayed mask, not a rough guidance polygon.

The first genuine VISOR pilot produced nine uncertain and nine rejected cases,
zero approved references, and a reported 15–30 minutes of active triage. Its
79-minute browser elapsed time is not active annotation time. The revised
[source-only proposals](experiments/submission-ladder-visor-action-revision-20261115.md)
remain unapproved. Moving them into another editor cannot promote them to gold
or change historical metric targets. Final references are human-reviewed,
model-assisted annotations; original dataset masks and unreviewed propagation
keep their distinct provenance.
