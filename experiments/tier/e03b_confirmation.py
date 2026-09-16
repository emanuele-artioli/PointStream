"""Score-free confirmation eligibility for the reserved trio.

Uses acquired files and acquisition hashes. Never computes codec quality
scores. Scene bounds come from thumbnail consecutive MAD only.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from experiments.long_scenes.extract import PROVENANCE
from experiments.tier.e03b_persist import sha256_path
from src.components.codec.tools import resolve_ffmpeg
from src.contracts import paths as ps_paths

REPO_ROOT = Path(__file__).resolve().parents[2]
ACQUISITION_POINTER = REPO_ROOT / "manifests" / "evaluation_20260914_e01r_acquisition.json"
TIMESTAMP_POINTER = REPO_ROOT / "manifests" / "evaluation_20260915_e03a_confirmation_timestamps.json"
CANDIDATE_POINTER = REPO_ROOT / "manifests" / "confirmation-source-candidates.json"
SOURCE_SPLIT = REPO_ROOT / "manifests" / "evaluation_20260914_source_split.json"

THUMB_FPS = 2.0
THUMB_WIDTH = 160
THUMB_HEIGHT = 90
SCENE_MIN_S = 4.0
MAD_FACTOR = 3.0


def _sha256_file(path: Path) -> str:
    return sha256_path(path)


def _candidate_index() -> dict[str, dict[str, Any]]:
    payload = json.loads(CANDIDATE_POINTER.read_text(encoding="utf-8"))
    index: dict[str, dict[str, Any]] = {}
    for group in ("primary_candidates", "alternative_candidates"):
        for item in payload.get(group) or []:
            index[str(item["candidate_id"])] = item
    return index


def _development_events() -> list[dict[str, str]]:
    split = json.loads(SOURCE_SPLIT.read_text(encoding="utf-8"))
    events: list[dict[str, str]] = []
    for video in split["development"]["videos"]:
        events.append(
            {
                "video": str(video["video"]),
                "match_identity": str(video.get("match_identity") or ""),
            }
        )
    for source in split.get("exposed_confirmation_not_holdout", {}).get("sources") or []:
        events.append(
            {
                "video": "exposed_gate_b",
                "match_identity": str(source.get("match_name") or source.get("source_id")),
                "source_id": str(source.get("source_id")),
                "candidate_id": str(source.get("candidate_id") or ""),
            }
        )
    return events


def _event_overlap(candidate: dict[str, Any], development: list[dict[str, str]]) -> dict[str, Any]:
    identity = candidate.get("match_identity") or {}
    event = str(identity.get("event") or "")
    tournament = str(identity.get("tournament") or "")
    year = identity.get("year")
    match_name = str(identity.get("match_name") or "")
    hits: list[dict[str, str]] = []
    for row in development:
        text = " ".join(str(value) for value in row.values()).lower()
        reasons: list[str] = []
        if tournament and tournament.lower() in text:
            reasons.append("tournament_token")
        if match_name and match_name.lower() in text:
            reasons.append("same_match_name")
        if year and str(year) in text and tournament.lower() in text:
            reasons.append("tournament_year")
        if reasons:
            hits.append({"against": row.get("match_identity") or row.get("video") or "", "reasons": ",".join(reasons)})
    exact_match = any("same_match_name" in item["reasons"] for item in hits)
    for row in development:
        source_id = str(row.get("source_id") or "")
        if source_id.startswith("bp57_ao") and tournament.lower() == "australian open" and year == 2024:
            hits.append({"against": source_id, "reasons": "exposed_gate_b_same_tournament_year"})
        if source_id.startswith("bp57_usopen") and "us open" in tournament.lower() and year == 2023:
            hits.append({"against": source_id, "reasons": "exposed_gate_b_same_tournament_year"})
    return {
        "event": event,
        "tournament": tournament,
        "year": year,
        "match_name": match_name,
        "exact_match_overlap": exact_match,
        "tournament_or_token_hits": hits,
        "independent_match": not exact_match,
    }


def sample_thumbnails(video_path: Path, *, ffmpeg: str, duration_s: float) -> tuple[np.ndarray, float]:
    """Low-resolution 2 fps RGB thumbnails. Not a quality score."""
    raw = subprocess.check_output(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-an",
            "-vf",
            f"fps={THUMB_FPS},scale={THUMB_WIDTH}:{THUMB_HEIGHT}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ]
    )
    frames = np.frombuffer(raw, dtype=np.uint8)
    usable = (frames.size // (THUMB_HEIGHT * THUMB_WIDTH * 3)) * THUMB_HEIGHT * THUMB_WIDTH * 3
    stack = frames[:usable].reshape(-1, THUMB_HEIGHT, THUMB_WIDTH, 3)
    return stack, THUMB_FPS


def consecutive_mad(frames: np.ndarray) -> np.ndarray:
    if len(frames) < 2:
        return np.zeros(0, dtype=np.float64)
    diffs = np.abs(frames[1:].astype(np.int16) - frames[:-1].astype(np.int16))
    return diffs.mean(axis=(1, 2, 3)).astype(np.float64)


def scene_bounds_from_mad(
    mads: np.ndarray,
    *,
    fps: float,
    min_duration_s: float = SCENE_MIN_S,
    factor: float = MAD_FACTOR,
) -> list[dict[str, Any]]:
    """Content cuts from MAD spikes. No codec metric is involved."""
    if mads.size == 0:
        return []
    median = float(np.median(mads))
    scale = float(np.median(np.abs(mads - median))) or 1.0
    threshold = median + factor * scale
    cuts = [0]
    for index, value in enumerate(mads, start=1):
        if float(value) >= threshold:
            cuts.append(index)
    cuts.append(int(mads.size) + 1)
    scenes: list[dict[str, Any]] = []
    for start, end in zip(cuts, cuts[1:]):
        t0 = start / fps
        t1 = end / fps
        if t1 - t0 < min_duration_s:
            continue
        piece = mads[start : max(start, end - 1)]
        scenes.append(
            {
                "t_start_s": round(t0, 4),
                "t_end_s": round(t1, 4),
                "duration_s": round(t1 - t0, 4),
                "mean_consecutive_mad": round(float(piece.mean()) if piece.size else 0.0, 4),
            }
        )
    return scenes


def evaluate_reserved_sources(*, ffmpeg: str | None = None) -> dict[str, Any]:
    acquisition = json.loads(ACQUISITION_POINTER.read_text(encoding="utf-8"))
    timestamps = json.loads(TIMESTAMP_POINTER.read_text(encoding="utf-8"))
    stamp_by_id = {item["candidate_id"]: item for item in timestamps["sources"]}
    candidates = _candidate_index()
    development = _development_events()
    ffmpeg_path = ffmpeg or resolve_ffmpeg().path
    data_root = ps_paths.assets().parent
    sources_out: list[dict[str, Any]] = []
    eligible = 0
    for source in acquisition["sources"]:
        candidate_id = str(source["candidate_id"])
        rel = Path(str(source["path"]))
        path = data_root / rel if not rel.is_absolute() else rel
        expected = str(source["sha256"])
        actual = _sha256_file(path) if path.is_file() else ""
        hash_ok = actual == expected
        stamp = stamp_by_id.get(candidate_id, {})
        cand = candidates.get(candidate_id, {})
        overlap = _event_overlap(cand, development) if cand else {"independent_match": False, "note": "missing_candidate_row"}
        provenance_contaminated = any(
            item.get("is_contaminated") and candidate_id.startswith("unused") for item in PROVENANCE.values()
        )
        width = int(source.get("width") or 0)
        height = int(source.get("height") or 0)
        native_4k = width >= 3840 and height >= 2160
        scenes: list[dict[str, Any]] = []
        mad_alarm = None
        if hash_ok and path.is_file():
            thumbs, fps = sample_thumbnails(path, ffmpeg=ffmpeg_path, duration_s=float(source.get("duration_s") or 0))
            mads = consecutive_mad(thumbs)
            scenes = scene_bounds_from_mad(mads, fps=fps)
            if not scenes:
                mad_alarm = "no_content_scene_lasting_4s"
        blockers: list[str] = []
        if not hash_ok:
            blockers.append("acquisition_hash_mismatch_or_missing_file")
        if not stamp:
            blockers.append("timestamp_origin_unrecorded")
        if overlap.get("exact_match_overlap"):
            blockers.append("exact_match_overlap_with_development_or_exposed")
        if not scenes:
            blockers.append(mad_alarm or "content_scene_bounds_missing")
        is_eligible = not blockers
        if is_eligible:
            eligible += 1
        sources_out.append(
            {
                "candidate_id": candidate_id,
                "path": str(path),
                "acquisition_sha256": expected,
                "on_disk_sha256": actual,
                "hash_match": hash_ok,
                "bytes": int(source.get("bytes") or 0),
                "width": width,
                "height": height,
                "fps": source.get("fps"),
                "native_4k": native_4k,
                "cannot_confirm_native_4k_claim": not native_4k,
                "timestamp_origin": {
                    "container_start_s": stamp.get("container_start_s"),
                    "first_frame": stamp.get("first_frame"),
                    "first_iframe": stamp.get("first_iframe"),
                    "keyframe_preroll_s": stamp.get("keyframe_preroll_s"),
                },
                "event_overlap": overlap,
                "content_scenes": scenes,
                "n_content_scenes": len(scenes),
                "frozen_scene_bounds": scenes,
                "confirmation_eligible": is_eligible,
                "eligibility_blockers": blockers,
                "scores_computed": False,
                "development_provenance_unused": provenance_contaminated,
            }
        )
    report = {
        "schema": "pointstream.campaign_confirmation_eligibility.v1",
        "doc_role": "e03b_score_free_reserved_source_eligibility",
        "campaign": "evaluation-20260914",
        "contract_revision": "e01r-20260914",
        "acquisition_pointer": str(ACQUISITION_POINTER.relative_to(REPO_ROOT)),
        "timestamp_pointer": str(TIMESTAMP_POINTER.relative_to(REPO_ROOT)),
        "counts": {
            "acquired": len(sources_out),
            "confirmation_eligible": eligible,
            "scores_computed": False,
        },
        "claim_restriction": "All three acquired windows are 1080p; they cannot confirm a native 4K claim.",
        "method": {
            "quality_scores": False,
            "scene_detector": "consecutive_mad_on_2fps_160x90_thumbnails",
            "min_scene_s": SCENE_MIN_S,
            "overlap_rule": "exact match_name overlap fails; tournament token hits are recorded, not automatic failure",
        },
        "sources": sources_out,
    }
    return report


def write_eligibility(report: dict[str, Any], dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return dest
