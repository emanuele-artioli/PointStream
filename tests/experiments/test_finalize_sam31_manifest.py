from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.finalize_sam31_manifest import finalize_run


def test_finalizer_records_every_view_digest_without_activating_pilot(tmp_path: Path) -> None:
    run_dir = tmp_path / "pilot"
    run_dir.mkdir()
    view_dir = run_dir / "views"
    adapter_dir = run_dir / "adapters"
    view_dir.mkdir()
    adapter_dir.mkdir()
    appearance = view_dir / "appearance.png"
    mask = view_dir / "mask.png"
    condition = view_dir / "condition.png"
    adapter = adapter_dir / "control.png"
    for path, content in (
        (appearance, b"appearance"),
        (mask, b"mask"),
        (condition, b"condition"),
        (adapter, b"adapter"),
    ):
        path.write_bytes(content)
    sample = {
        "sample_id": "scene:1:player:0:player",
        "appearance_path": str(appearance),
        "mask_path": str(mask),
        "condition_path": str(condition),
        "model_adapter_paths": {"controlnet_condition_rgb": str(adapter)},
    }
    (run_dir / "views.jsonl").write_text(json.dumps(sample) + "\n")
    (run_dir / "observations.jsonl").write_text("{}\n")
    (run_dir / "poses.jsonl").write_text("{}\n")
    (run_dir / "observation_masks.npz").write_bytes(b"masks")
    manifest = {
        "schema": "pointstream.dataset-manifest.v1",
        "active": False,
        "samples": [sample.copy()],
        "artifacts": {
            "training_views_jsonl": {"path": str(run_dir / "views.jsonl"), "sha256": "stale"},
            "observations_jsonl": {"path": str(run_dir / "observations.jsonl"), "sha256": ""},
            "pose_observations_jsonl": {"path": str(run_dir / "poses.jsonl"), "sha256": ""},
            "mask_arrays_npz": {"path": str(run_dir / "observation_masks.npz"), "sha256": ""},
        },
    }
    (run_dir / "dataset_manifest.json").write_text(json.dumps(manifest))
    (run_dir / "audit.json").write_text(
        json.dumps({"status": "complete_bounded_development_pilot", "artifacts": {}})
    )

    result = finalize_run(run_dir)

    updated = json.loads((run_dir / "dataset_manifest.json").read_text())
    sample_after = updated["samples"][0]
    assert updated["active"] is False
    assert sample_after["appearance_sha256"] == hashlib.sha256(b"appearance").hexdigest()
    assert sample_after["mask_sha256"] == hashlib.sha256(b"mask").hexdigest()
    assert sample_after["condition_sha256"] == hashlib.sha256(b"condition").hexdigest()
    assert sample_after["model_adapter_sha256"]["controlnet_condition_rgb"] == hashlib.sha256(b"adapter").hexdigest()
    assert updated["artifacts"]["training_views_jsonl"]["sha256"] == result["views_sha256"]
    report = json.loads((run_dir / "audit.json").read_text())
    assert report["training_conditioning_transport_scope"]["training_views_identical_to_decoded_client_conditioning"] is False
    assert report["active_dataset_manifest_changed"] is False
    assert "visual contact-sheet review has not been applied to sample eligibility" in report["promotion"]["reasons"]


def test_visual_review_quarantines_racket_and_joint_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "pilot"
    run_dir.mkdir()
    rows = []
    samples = []
    for role in ("player", "racket", "joint"):
        files = {}
        for kind in ("appearance", "mask", "condition"):
            path = run_dir / f"{role}_{kind}.png"
            path.write_bytes(f"{role}-{kind}".encode())
            files[f"{kind}_path"] = str(path)
        row = {
            "sample_id": f"scene-a:{role}",
            "source_id": "scene-a",
            "view": role,
            "eligible_for_training": True,
            **files,
        }
        rows.append(row)
        samples.append(row.copy())
    (run_dir / "views.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    for name, content in (("observations.jsonl", "{}\n"), ("poses.jsonl", "{}\n"), ("observation_masks.npz", "masks")):
        (run_dir / name).write_text(content)
    manifest = {
        "schema": "pointstream.dataset-manifest.v1",
        "active": False,
        "samples": samples,
        "artifacts": {
            name: {"path": str(run_dir / path), "sha256": ""}
            for name, path in (
                ("training_views_jsonl", "views.jsonl"),
                ("observations_jsonl", "observations.jsonl"),
                ("pose_observations_jsonl", "poses.jsonl"),
                ("mask_arrays_npz", "observation_masks.npz"),
            )
        },
    }
    (run_dir / "dataset_manifest.json").write_text(json.dumps(manifest))
    (run_dir / "audit.json").write_text(
        json.dumps({"status": "complete_bounded_development_pilot", "artifacts": {}})
    )
    review_path = run_dir / "review-input.json"
    review_path.write_text(
        json.dumps(
            {
                "schema": "pointstream.sam31-visual-review.v1",
                "review_date": "2026-09-27",
                "method": "contact sheet",
                "sources": [
                    {
                        "source_id": "scene-a",
                        "quarantine_roles": ["racket", "joint"],
                        "reasons": ["visible racket failure remains unresolved"],
                    }
                ],
            }
        )
    )

    finalize_run(run_dir, visual_review_path=review_path)
    finalize_run(run_dir, visual_review_path=review_path)

    updated = json.loads((run_dir / "dataset_manifest.json").read_text())
    by_role = {sample["view"]: sample for sample in updated["samples"]}
    assert by_role["player"]["eligible_for_training"] is True
    assert by_role["racket"]["eligible_for_training"] is False
    assert by_role["racket"]["eligible_before_visual_review"] is True
    assert by_role["racket"]["training_exclusion_reason"] == "visual_review_unresolved_racket_observation"
    assert by_role["joint"]["eligible_for_training"] is False
    review_copy = run_dir / "visual_review.json"
    assert updated["artifacts"]["visual_review_json"]["sha256"] == hashlib.sha256(
        review_copy.read_bytes()
    ).hexdigest()
    assert updated["visual_review"]["source_sha256"] == hashlib.sha256(
        review_path.read_bytes()
    ).hexdigest()
    report = json.loads((run_dir / "audit.json").read_text())
    assert "racket and joint samples with unresolved visible-object failures are quarantined" in report["promotion"]["reasons"]
    assert updated["active"] is False
