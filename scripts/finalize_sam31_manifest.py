"""Finalize hashes and evidence scope in a completed external SAM3.1 audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import sys
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_text_atomic(path: Path, value: str) -> None:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _apply_visual_review(
    rows: list[dict[str, Any]], review: dict[str, Any]
) -> dict[str, Any]:
    if review.get("schema") != "pointstream.sam31-visual-review.v1":
        raise ValueError("unsupported SAM3.1 visual-review schema")
    decisions = review.get("sources")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError("visual review must contain at least one source decision")
    by_source: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        source_id = str(decision.get("source_id", ""))
        roles = decision.get("quarantine_roles")
        reasons = decision.get("reasons")
        if not source_id or source_id in by_source:
            raise ValueError(f"invalid or duplicate visual-review source {source_id!r}")
        if not isinstance(roles, list) or not roles or any(
            role not in {"racket", "joint"} for role in roles
        ):
            raise ValueError(f"invalid quarantine_roles for {source_id}")
        if not isinstance(reasons, list) or not reasons or any(not str(item).strip() for item in reasons):
            raise ValueError(f"visual review needs explicit reasons for {source_id}")
        by_source[source_id] = decision

    seen_sources: set[str] = set()
    quarantined_by_source: dict[str, dict[str, int]] = {}
    for row in rows:
        source_id = str(row.get("source_id", ""))
        decision = by_source.get(source_id)
        if decision is None:
            continue
        seen_sources.add(source_id)
        role = str(row.get("view", row.get("object_class", "")))
        if role not in decision["quarantine_roles"]:
            continue
        counts = quarantined_by_source.setdefault(source_id, {})
        counts[role] = counts.get(role, 0) + 1
        row["eligible_before_visual_review"] = bool(
            row.get("eligible_before_visual_review", row.get("eligible_for_training", False))
        )
        row["eligible_for_training"] = False
        row["training_exclusion_reason"] = "visual_review_unresolved_racket_observation"
        row["visual_review"] = {
            "status": "quarantined",
            "reason_code": "racket_observation_unresolved_visual_failure",
            "reasons": list(decision["reasons"]),
        }
    missing = set(by_source) - seen_sources
    if missing:
        raise ValueError(f"visual-review sources have no matching samples: {sorted(missing)}")
    return {
        "schema": review["schema"],
        "review_date": review.get("review_date"),
        "method": review.get("method"),
        "quarantined_samples_by_source": quarantined_by_source,
    }


def finalize_run(
    run_dir: Path,
    *,
    repo_root: Path | None = None,
    visual_review_path: Path | None = None,
) -> dict[str, Any]:
    """Add file digests and correct the report's measured transport scope."""
    run_dir = run_dir.resolve(strict=True)
    if repo_root is not None and run_dir.is_relative_to(repo_root.resolve()):
        raise ValueError("audit outputs must remain outside the code repository")
    audit_path = run_dir / "audit.json"
    manifest_path = run_dir / "dataset_manifest.json"
    views_path = run_dir / "views.jsonl"
    if not all(path.is_file() for path in (audit_path, manifest_path, views_path)):
        raise FileNotFoundError("run directory lacks completed audit, dataset manifest, or views")

    audit = _read_json(audit_path)
    manifest = _read_json(manifest_path)
    if audit.get("status") != "complete_bounded_development_pilot":
        raise ValueError("refusing to finalize an incomplete audit")
    if manifest.get("active") is not False:
        raise ValueError("refusing to finalize a manifest already marked active")

    rows = [
        json.loads(line)
        for line in views_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {str(row["sample_id"]): row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("view manifest contains duplicate sample identities")
    review_record: dict[str, Any] | None = None
    review_digest: str | None = None
    review_source_digest: str | None = None
    review_source_path: str | None = None
    if visual_review_path is not None:
        visual_review_path = visual_review_path.resolve(strict=True)
        review = _read_json(visual_review_path)
        review_record = _apply_visual_review(rows, review)
        review_source_path = str(visual_review_path)
        review_source_digest = _sha256(visual_review_path)
        review_copy_path = run_dir / "visual_review.json"
        _write_text_atomic(
            review_copy_path,
            json.dumps(review, indent=2, sort_keys=True) + "\n",
        )
        review_digest = _sha256(review_copy_path)
        review_record["source_path"] = review_source_path
        review_record["source_sha256"] = review_source_digest
    manifest_rows = manifest.get("samples")
    if not isinstance(manifest_rows, list) or {str(row["sample_id"]) for row in manifest_rows} != set(by_id):
        raise ValueError("dataset manifest and views.jsonl sample identities differ")

    checked_files: set[Path] = set()
    for row in rows:
        for field in ("appearance_path", "mask_path", "condition_path"):
            path = Path(row[field])
            if not path.is_absolute():
                path = (run_dir / path).resolve(strict=True)
            else:
                path = path.resolve(strict=True)
            row[field.removesuffix("_path") + "_sha256"] = _sha256(path)
            checked_files.add(path)
        adapter_paths = row.get("model_adapter_paths", {})
        if not isinstance(adapter_paths, dict):
            raise ValueError(f"model_adapter_paths must be an object for {row['sample_id']}")
        adapter_hashes: dict[str, str] = {}
        for name, raw_path in adapter_paths.items():
            path = Path(raw_path)
            if not path.is_absolute():
                path = (run_dir / path).resolve(strict=True)
            else:
                path = path.resolve(strict=True)
            adapter_hashes[str(name)] = _sha256(path)
            checked_files.add(path)
        row["model_adapter_sha256"] = adapter_hashes
    row_by_id = {str(row["sample_id"]): row for row in rows}
    manifest["samples"] = [row_by_id[str(row["sample_id"])] for row in manifest_rows]

    artifact_hashes: dict[str, str] = {}
    for name, artifact in manifest.get("artifacts", {}).items():
        raw_path = artifact["path"] if isinstance(artifact, dict) else artifact
        path = Path(raw_path)
        if not path.is_absolute():
            path = (run_dir / path).resolve(strict=True)
        else:
            path = path.resolve(strict=True)
        if name != "dataset_manifest_json":
            artifact_hashes[str(name)] = _sha256(path)
            if isinstance(artifact, dict):
                artifact["sha256"] = artifact_hashes[str(name)]
    manifest["artifact_hashes"] = artifact_hashes
    if review_record is not None:
        review_copy_path = run_dir / "visual_review.json"
        manifest.setdefault("artifacts", {})["visual_review_json"] = {
            "path": str(review_copy_path),
            "sha256": review_digest,
        }
        manifest["artifact_hashes"]["visual_review_json"] = review_digest
        manifest["visual_review"] = {
            **review_record,
            "path": str(run_dir / "visual_review.json"),
            "sha256": review_digest,
        }
    _write_text_atomic(
        views_path,
        "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows) + "\n",
    )
    views_hash = _sha256(views_path)
    manifest["artifact_hashes"]["training_views_jsonl"] = views_hash
    manifest["artifacts"]["training_views_jsonl"]["sha256"] = views_hash
    manifest["manifest_finalization"] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": Path(__file__).name,
        "script_sha256": _sha256(Path(__file__).resolve()),
        "python_version": sys.version,
        "sample_count": len(rows),
        "sample_file_count": len(checked_files),
        "views_jsonl_sha256": views_hash,
        "dataset_active": False,
    }
    _write_json_atomic(manifest_path, manifest)

    audit["codec_evidence_scope"] = (
        "PointStream semantic client transport measured JPEG appearance references, bbox metadata, "
        "and exact PSM1 mask roundtrip. Pose/motion payloads and native video codec execution "
        "were not exercised."
    )
    audit["training_conditioning_transport_scope"] = {
        "appearance_and_masks": "decoded through the fresh-process client path",
        "pose_and_racket_geometry": "derived from retained unquantized observations",
        "pose_or_geometry_transmitted": False,
        "training_views_identical_to_decoded_client_conditioning": False,
    }
    audit.setdefault("artifacts", {})["training_views_sha256"] = views_hash
    audit["artifacts"]["dataset_manifest_sha256"] = _sha256(manifest_path)
    audit["artifacts"]["sample_artifact_count"] = len(checked_files)
    audit["active_dataset_manifest_changed"] = False
    promotion_reasons = [
        "pose and racket geometry are not transmitted in the measured client payload",
        "the development pilot is not a regenerated eligible training split",
    ]
    if review_record is None:
        promotion_reasons.insert(0, "visual contact-sheet review has not been applied to sample eligibility")
    else:
        promotion_reasons.insert(
            0,
            "racket and joint samples with unresolved visible-object failures are quarantined",
        )
    audit["promotion"] = {"eligible": False, "reasons": promotion_reasons}
    if review_record is not None:
        audit["visual_review"] = {
            **review_record,
            "path": str(run_dir / "visual_review.json"),
            "sha256": review_digest,
        }
        audit.setdefault("artifacts", {})["visual_review_sha256"] = review_digest
    audit["manifest_finalization"] = manifest["manifest_finalization"]
    _write_json_atomic(audit_path, audit)
    return {
        "audit_path": str(audit_path),
        "dataset_manifest_path": str(manifest_path),
        "views_sha256": views_hash,
        "dataset_manifest_sha256": _sha256(manifest_path),
        "audit_sha256": _sha256(audit_path),
        "sample_count": len(rows),
        "sample_artifact_count": len(checked_files),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--visual-review", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            finalize_run(
                args.run_dir,
                repo_root=args.repo_root,
                visual_review_path=args.visual_review,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
