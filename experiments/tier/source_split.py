"""Development/validation/confirmation split and training scene selectors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SCHEMA_ID = "pointstream.campaign_source_split.v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO_ROOT / "manifests" / "evaluation_20260914_source_split.json"


def load_source_split(path: Path | None = None) -> dict[str, Any]:
    data = json.loads((path or DEFAULT_PATH).read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA_ID:
        raise ValueError(f"unsupported source-split schema: {data.get('schema')!r}")
    return data


def _scene_key(video: str, scene: str) -> tuple[str, str]:
    return (str(video).strip(), str(scene).strip())


def excluded_training_blocks(split: dict[str, Any] | None = None) -> set[tuple[str, str]]:
    """Validation and confirmation scenes that training selectors must drop."""
    data = split or load_source_split()
    blocked: set[tuple[str, str]] = set()
    for item in data.get("validation", {}).get("preferred_blocks") or []:
        if item.get("video") and item.get("scene"):
            blocked.add(_scene_key(item["video"], item["scene"]))
    for extra in data.get("training_selector", {}).get("exclude_scenes") or []:
        blocked.add(_scene_key(extra["video"], extra["scene"]))
    return blocked


def confirmation_match_ids(split: dict[str, Any] | None = None) -> set[str]:
    data = split or load_source_split()
    ids = set(data.get("confirmation_reserved_unacquired", {}).get("candidates") or [])
    for src in data.get("exposed_confirmation_not_holdout", {}).get("sources") or []:
        if src.get("candidate_id"):
            ids.add(str(src["candidate_id"]))
        if src.get("source_id"):
            ids.add(str(src["source_id"]))
    return ids


def training_scene_allowed(
    video: str,
    scene: str,
    *,
    match_id: str | None = None,
    split: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Return whether a scene may enter shared-model training.

    Validation blocks are excluded. Confirmation and exposed Gate B IDs are
    excluded even if a caller passes a development video name by mistake.
    Replay/compilation videos stay development-only when marked contaminated.
    """
    data = split or load_source_split()
    if _scene_key(video, scene) in excluded_training_blocks(data):
        return False, "validation_block"
    reserved = confirmation_match_ids(data)
    if video in reserved or scene in reserved or (match_id and match_id in reserved):
        return False, "confirmation_or_exposed_holdout"
    for item in data.get("development", {}).get("videos") or []:
        if item.get("video") != video:
            continue
        if item.get("use") == "do_not_select_for_codec_scenes_until_route_is_pointstream":
            return False, "ineligible_development_asset"
        if item.get("contaminated") and data.get("grouping", {}).get("replay_excerpt_rule"):
            return True, "development_contaminated_ok_for_training"
        return True, "development"
    return False, "unknown_video"
