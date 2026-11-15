"""Combine preregistered official/SAM proposals for ROOT's source-only V2 review.

No mask is gold. Rejected rough human polygons supply query boxes only and are
never copied into proposals or approved references. This creates a NEW private
review directory; publication into the live server is a separate ROOT decision.
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from tools.datasets.annotation_review import prepare, load_review, validate_decisions

EXPECTED_HUMAN = "9a31519973bbd3984af1a049f3c503d761fc251f75fe8bb908d1228a118da754"
TASK = {"P35_105": "saucepan", "P12_101": "tap", "P11_103": "pan"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_mask(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as image:
        pixels = np.asarray(image)
        if (
            image.format != "PNG"
            or pixels.shape != (1080, 1920)
            or pixels.ndim != 2
            or not set(np.unique(pixels)) <= {0, 255}
        ):
            raise ValueError("binary native-geometry mask required")
    return pixels > 0


def portable(review_root: Path) -> None:
    """Preserve review metadata/digest while inlining display assets, no fetch/CORS."""
    review = load_review(review_root)
    assets = {}
    import base64

    for row in review["frames"]:
        for name in (row["source"], row["overlay"]):
            if name is not None:
                assets[name] = (
                    "data:image/png;base64,"
                    + base64.b64encode((review_root / name).read_bytes()).decode()
                )
    html = (review_root / "index.html").read_text()
    html = html.replace("i.src=path;", "i.src=portableReviewAssets[path]||path;")
    prefix = "<script>const portableReviewAssets=" + json.dumps(assets) + ";</script>"
    (review_root / "review-standalone.html").write_text(
        html.replace("<!doctype html>", "<!doctype html>" + prefix, 1)
    )
    (review_root / "OFFLINE.txt").write_text(
        "Open review-standalone.html for all source/proposal images without an HTTP server.\n"
        "HTTP and file origins store drafts separately. Export completed decisions from the origin used.\n"
        "Approve only the FINAL displayed mask and complete corrections. Rejected rough polygons are prompts, not references.\n"
    )


def combine(
    filtered_root: Path,
    old_review: Path,
    human_path: Path,
    output: Path,
    *,
    sam_archive: Path | None = None,
    sam_sha256: str | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if sha(human_path) != EXPECTED_HUMAN:
        raise ValueError("human feedback differs from frozen input")
    old = load_review(old_review)
    human = json.loads(human_path.read_bytes())
    statuses = validate_decisions(old, human)
    if any(row["status"] == "approved" for row in statuses):
        raise ValueError("This revision expects no prior approved references")
    filtered = json.loads((filtered_root / "filtered-proposals.json").read_text())
    by_id = {row["id"]: row for row in filtered["frames"]}
    if set(by_id) != {row["id"] for row in old["frames"]} or len(by_id) != 18:
        raise ValueError("fixed18 source cohort differs")
    sam_rows: dict[tuple[str, str], dict[str, Any]] = {}
    sam_masks: dict[str, bytes] = {}
    if sam_archive is not None:
        if sam_sha256 is None or sha(sam_archive) != sam_sha256:
            raise ValueError("SAM publication identity mismatch")
        with tarfile.open(sam_archive) as archive:
            names = archive.getnames()
            if len(names) != len(set(names)) or any(not m.isfile() for m in archive.getmembers()):
                raise ValueError("unexpected SAM archive")
            reader = archive.extractfile("result.json")
            if reader is None:
                raise ValueError("missing SAM result")
            raw = json.load(reader)
            if raw["count"] != 18:
                raise ValueError("full18 source-only proposal stage required")
            for row in raw["records"]:
                key = (row["id"], row["prompt"])
                if key in sam_rows:
                    raise ValueError("duplicate SAM source/query")
                sam_rows[key] = row
                for instance in row["instances"]:
                    if "mask" in instance:
                        name = "masks/" + instance["mask"]
                        handle = archive.extractfile(name)
                        if handle is None:
                            raise ValueError("missing SAM mask")
                        data = handle.read()
                        if hashlib.sha256(data).hexdigest() != instance["mask_sha256"]:
                            raise ValueError("SAM mask changed")
                        sam_masks[name] = data
    expected = {
        (f"{r['video']}_native_{r['native_index_0based']}", prompt)
        for r in by_id.values()
        for prompt in ("person", "hand", "object")
    }
    if sam_archive is not None and set(sam_rows) != expected:
        raise ValueError("all18 fixed sources and three frozen queries required")
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    manifest_rows, receipt_rows = [], []
    for ordinal, prior in enumerate(old["frames"]):
        row = by_id[prior["id"]]
        source = old_review / prior["source"]
        if (
            sha(source) != prior["source_sha256"]
            or row["source_png_sha256"] != prior["source_sha256"]
        ):
            raise ValueError("source PNG differs")
        official_path = filtered_root / row["proposal"]
        if sha(official_path) != row["proposal_sha256"]:
            raise ValueError("official filtered mask changed")
        mask = read_mask(official_path.read_bytes())
        source_key = f"{row['video']}_native_{row['native_index_0based']}"
        task_exists = any(
            i["name"] == TASK[row["video"]]
            and i["keep_candidate"]
            and i["native_annotation_canvas_pixels"] > 0
            for i in row["instance_inventory"]
        )
        included, ignored, failures = [], [], []
        actor_pixels = 0
        for prompt in ("person", "hand", "object"):
            raw = sam_rows.get((source_key, prompt))
            if raw is None:
                failures.append(prompt + ": unavailable")
                continue
            if raw["source_sha256"] != prior["source_sha256"]:
                raise ValueError("SAM source PNG identity differs")
            for instance in raw["instances"]:
                if "mask" not in instance:
                    failures.append(prompt + ":" + instance["status"])
                    continue
                pixels = read_mask(sam_masks["masks/" + instance["mask"]])
                use = prompt != "object" or not task_exists
                entry = dict(
                    prompt=prompt,
                    mask=instance["mask"],
                    mask_sha256=instance["mask_sha256"],
                    status=instance["status"],
                    score=instance["score"],
                )
                if use:
                    mask |= pixels
                    included.append(entry)
                    if prompt != "object":
                        actor_pixels += int(pixels.sum())
                else:
                    ignored.append(
                        {
                            **entry,
                            "reason": "official allowed task object exists; frozen fallback condition false",
                        }
                    )
        # Preserve official fallback, but make a complete actor-model failure an
        # explicit incomplete proposal rather than an inferred empty reference.
        requires_correction = not mask.any() or actor_pixels == 0
        proposal = inputs / f"combined-{ordinal}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(proposal)
        manifest_rows.append(
            dict(
                id=prior["id"],
                source_group=prior["source_group"],
                frame_index=prior["frame_index"],
                blind=False,
                requires_complete_correction=requires_correction,
                source=dict(path=str(source), sha256=prior["source_sha256"]),
                proposal=dict(path=proposal.name, sha256=sha(proposal)),
            )
        )
        receipt_rows.append(
            dict(
                id=prior["id"],
                source_sha256=prior["source_sha256"],
                official_proposal_sha256=row["proposal_sha256"],
                combined_proposal_sha256=sha(proposal),
                task_official_exists=task_exists,
                included_SAM=included,
                ignored_SAM=ignored,
                failures=failures,
                requires_complete_correction=requires_correction,
                visible=True,
                prior_human_status=next(r["status"] for r in statuses if r["id"] == prior["id"]),
                prior_polygons_used_as_pixels=False,
            )
        )
    manifest = dict(version=2, review_id="visor-action-targets-review18-v2", frames=manifest_rows)
    manifest_path = inputs / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    review = prepare(manifest_path, output / "review")
    instructions = (
        "<p><strong>Foreground target:</strong> visible people/hands and objects actively moved or manipulated. "
        "Include the tap when interacted with. Exclude unrelated towels and static whole sink/oven/stove. "
        "These filtered/model masks are proposals; verify every boundary and missing actor/object. "
        "If drawing corrections, draw the complete foreground union: corrections replace the entire proposal. "
        "All 18 proposals are visible; earlier hidden overlays were blind review conditions.</p>"
    )
    index = output / "review" / "index.html"
    index.write_text(
        index.read_text().replace('<p id="draft-status"', instructions + '<p id="draft-status"', 1)
    )
    portable(output / "review")
    # Keep every raw instance, including masks ignored by the frozen union rule.
    if sam_archive is not None:
        shutil.copyfile(sam_archive, output / "raw-SAM-proposals.tar")
    official_copy = output / "filtered-official"
    official_copy.mkdir()
    shutil.copyfile(
        filtered_root / "filtered-proposals.json", official_copy / "filtered-proposals.json"
    )
    for row in filtered["frames"]:
        shutil.copyfile(filtered_root / row["proposal"], official_copy / Path(row["proposal"]).name)
    policy = (
        Path(__file__).resolve().parents[2]
        / "docs/experiments/visor-action-policy-v2-20261115.json"
    )
    shutil.copyfile(policy, output / "frozen-action-policy.json")
    result = dict(
        schema="pointstream.visor.action-review.v2",
        state="ROOT_review_required_before_UI_publication",
        prior_review_sha256=old["review_sha256"],
        human_feedback_sha256=EXPECTED_HUMAN,
        filtered_proposals_sha256=sha(filtered_root / "filtered-proposals.json"),
        SAM_archive_sha256=sam_sha256,
        policy_sha256=sha(policy),
        review_sha256=review["review_sha256"],
        count=18,
        blind_count=0,
        new_reference_annotations=0,
        decisions_download="annotation-decisions-v2.json",
        frames=receipt_rows,
        combination="filtered official hands/task object + all default SAM person/hand instances + SAM object only if official allowed object absent",
        source_only=True,
        rough_polygon_pixels_copied=False,
    )
    (output / "revision-provenance.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
