"""Unit tests for BP46 long eligible tennis-scene manifest and tooling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from experiments.long_scenes.extract import (
    measure_motion_and_panorama,
)
from experiments.long_scenes.schema import (
    MAX_CANVAS_GROWTH,
    MAX_CONSECUTIVE_MAD,
    PASTE_MAE_MAX,
    SCHEMA_ID,
    TARGET_SPANS,
)
from experiments.long_scenes.verify import ManifestValidationError, verify_manifest


def _build_valid_manifest() -> dict[str, Any]:
    return {
        "schema": SCHEMA_ID,
        "created_utc": "2026-09-02T18:00:00Z",
        "source_data_root": "/home/itec/emanuele/pointstream-data",
        "target_spans": list(TARGET_SPANS),
        "diagnostic_videos": ["alcaraz_highlights", "federer_djokovic"],
        "confirmation_videos": [
            "alcaraz_perricard",
            "alcaraz_ruud",
            "djokovic_federer",
            "djokovic_zverev",
            "federer_djokovic",
            "sinner_alcaraz",
        ],
        "ineligible_controls": ["alcaraz_highlights/scene_006"],
        "summary": {
            "submitted_scenes": 3,
            "pointstream_eligible_scenes": 2,
            "conventional_fallback_scenes": 1,
            "succeeded_by_span": {"48_frames": 2, "96_frames": 2, "192_frames": 2, "384_frames": 2},
            "failed_by_span": {"48_frames": 1, "96_frames": 1, "192_frames": 1, "384_frames": 1},
        },
        "scenes": [
            {
                "video": "alcaraz_highlights",
                "scene": "scene_000",
                "t_start": 0.0,
                "t_end": 18.0,
                "duration": 18.0,
                "cluster": "cluster_point",
                "context_id": "alcaraz_highlights_main_court",
                "role": "diagnostic_near_static",
                "source_metadata": {
                    "video_file": "alcaraz_highlights.mp4",
                    "width": 3840,
                    "height": 2160,
                    "source_fps": 59.94,
                    "working_fps": 24.0,
                    "pix_fmt": "yuv420p",
                    "color_space": "bt709",
                    "color_primaries": "bt709",
                    "color_transfer": "bt709",
                    "sha256": "abc123mock",
                },
                "eligibility": {
                    "duration_24fps_frames": 434,
                    "camera_motion": {
                        "consecutive_mad": 0.33,
                        "vs_first_frame_mad": 1.36,
                        "last_vs_first_mad": 1.50,
                    },
                    "panorama": {
                        "canvas_width": 3840,
                        "canvas_height": 2160,
                        "growth_factor": 1.00,
                        "registration_ok": True,
                    },
                    "objects": {
                        "num_objects": 2,
                        "object_class": "player",
                        "player_pixel_fraction": 0.008,
                        "min_separation_px": 650.0,
                        "has_occlusion": False,
                        "track_continuity": True,
                    },
                    "paste_back": {
                        "convention": "extract_24_frame_id",
                        "opaque_mae": 0.0,
                        "threshold": 2.0,
                        "passes_threshold": True,
                    },
                    "route": "pointstream",
                    "ineligibility_reasons": [],
                },
                "intervals": {
                    str(span): {
                        "frame_count": span,
                        "start_frame": 38,
                        "end_frame": 38 + span,
                        "status": "eligible",
                        "frame_hashes": {"first": "h1", "last": "h2"},
                        "paste_back_mae": 0.0,
                        "canvas_growth": 1.00,
                        "failure_reasons": [],
                    }
                    for span in TARGET_SPANS
                },
            },
            {
                "video": "federer_djokovic",
                "scene": "scene_001",
                "t_start": 0.0,
                "t_end": 15.0,
                "duration": 15.0,
                "cluster": "cluster_point",
                "context_id": "federer_djokovic_main_court",
                "role": "diagnostic_smooth_pan",
                "source_metadata": {
                    "video_file": "federer_djokovic.mp4",
                    "width": 3840,
                    "height": 2160,
                    "source_fps": 59.94,
                    "working_fps": 24.0,
                    "pix_fmt": "yuv420p",
                    "color_space": "bt709",
                    "color_primaries": "bt709",
                    "color_transfer": "bt709",
                    "sha256": "def456mock",
                },
                "eligibility": {
                    "duration_24fps_frames": 361,
                    "camera_motion": {
                        "consecutive_mad": 1.85,
                        "vs_first_frame_mad": 2.88,
                        "last_vs_first_mad": 3.20,
                    },
                    "panorama": {
                        "canvas_width": 4000,
                        "canvas_height": 2200,
                        "growth_factor": 1.06,
                        "registration_ok": True,
                    },
                    "objects": {
                        "num_objects": 2,
                        "object_class": "player",
                        "player_pixel_fraction": 0.009,
                        "min_separation_px": 720.0,
                        "has_occlusion": False,
                        "track_continuity": True,
                    },
                    "paste_back": {
                        "convention": "extract_24_frame_id",
                        "opaque_mae": 0.0,
                        "threshold": 2.0,
                        "passes_threshold": True,
                    },
                    "route": "pointstream",
                    "ineligibility_reasons": [],
                },
                "intervals": {
                    str(span): {
                        "frame_count": span,
                        "start_frame": 0,
                        "end_frame": span,
                        "status": "eligible",
                        "frame_hashes": {"first": "h1", "last": "h2"},
                        "paste_back_mae": 0.0,
                        "canvas_growth": 1.06,
                        "failure_reasons": [],
                    }
                    for span in TARGET_SPANS
                },
            },
            {
                "video": "alcaraz_highlights",
                "scene": "scene_006",
                "t_start": 100.0,
                "t_end": 105.0,
                "duration": 5.0,
                "cluster": "cluster_other",
                "context_id": "alcaraz_highlights_crowd_side",
                "role": "control_ineligible",
                "source_metadata": {
                    "video_file": "alcaraz_highlights.mp4",
                    "width": 3840,
                    "height": 2160,
                    "source_fps": 59.94,
                    "working_fps": 24.0,
                    "pix_fmt": "yuv420p",
                    "color_space": "bt709",
                    "color_primaries": "bt709",
                    "color_transfer": "bt709",
                    "sha256": "abc123mock",
                },
                "eligibility": {
                    "duration_24fps_frames": 120,
                    "camera_motion": {
                        "consecutive_mad": 15.2,
                        "vs_first_frame_mad": 35.0,
                        "last_vs_first_mad": 40.0,
                    },
                    "panorama": {
                        "canvas_width": 3840,
                        "canvas_height": 2160,
                        "growth_factor": 3.5,
                        "registration_ok": False,
                    },
                    "objects": {
                        "num_objects": 0,
                        "object_class": "player",
                        "player_pixel_fraction": 0.0,
                        "min_separation_px": 0.0,
                        "has_occlusion": False,
                        "track_continuity": False,
                    },
                    "paste_back": {
                        "convention": "unknown",
                        "opaque_mae": 999.0,
                        "threshold": 2.0,
                        "passes_threshold": False,
                    },
                    "route": "conventional_fallback",
                    "ineligibility_reasons": ["cluster_other is not point camera", "consecutive_mad > 10.0"],
                },
                "intervals": {
                    str(span): {
                        "frame_count": span,
                        "start_frame": 0,
                        "end_frame": min(120, span),
                        "status": "ineligible",
                        "frame_hashes": {},
                        "paste_back_mae": 999.0,
                        "canvas_growth": 3.5,
                        "failure_reasons": ["cluster_other is not point camera"],
                    }
                    for span in TARGET_SPANS
                },
            },
        ],
    }


def test_schema_constants() -> None:
    assert SCHEMA_ID == "pointstream.long_scenes.v1"
    assert TARGET_SPANS == (48, 96, 192, 384)
    assert PASTE_MAE_MAX == 2.0
    assert MAX_CANVAS_GROWTH == 2.5
    assert MAX_CONSECUTIVE_MAD == 10.0


def test_manifest_verification_accepts_valid_manifest() -> None:
    manifest = _build_valid_manifest()
    res = verify_manifest(manifest)
    assert res["status"] == "VERIFIED_PASS"
    assert res["num_scenes"] == 3
    assert len(res["confirmation_videos"]) >= 6


def test_manifest_verification_rejects_missing_diagnostic() -> None:
    manifest = _build_valid_manifest()
    manifest["diagnostic_videos"] = ["alcaraz_highlights"]  # only 1 video
    with pytest.raises(ManifestValidationError, match="diagnostic"):
        verify_manifest(manifest)


def test_manifest_verification_rejects_conflated_contexts() -> None:
    manifest = _build_valid_manifest()
    # Conflate context across different videos
    manifest["scenes"][1]["context_id"] = manifest["scenes"][0]["context_id"]
    with pytest.raises(ManifestValidationError, match="groups unrelated source videos"):
        verify_manifest(manifest)


def test_manifest_verification_rejects_ineligible_without_reasons() -> None:
    manifest = _build_valid_manifest()
    manifest["scenes"][2]["eligibility"]["ineligibility_reasons"] = []
    with pytest.raises(ManifestValidationError, match="must record ineligibility_reasons"):
        verify_manifest(manifest)


def test_manifest_verification_rejects_wrong_span_frame_count() -> None:
    manifest = _build_valid_manifest()
    manifest["scenes"][0]["intervals"]["48"]["frame_count"] = 50
    with pytest.raises(ManifestValidationError, match="frame_count"):
        verify_manifest(manifest)


def test_measure_motion_and_panorama_static() -> None:
    frames = np.ones((8, 100, 100, 3), dtype=np.uint8) * 128
    mot, pano = measure_motion_and_panorama(frames)
    assert mot.consecutive_mad == 0.0
    assert mot.vs_first_frame_mad == 0.0
    assert mot.last_vs_first_mad == 0.0
    assert pano.growth_factor == 1.0


def test_load_long_scene_clip_verified() -> None:
    from experiments.long_scenes.loader import LongSceneError, load_long_scene_clip

    # Valid scene and span
    clip = load_long_scene_clip("alcaraz_highlights", "scene_028", 48)
    assert clip.n_frames == 48
    assert clip.frames.shape == (48, 2160, 3840, 3)
    assert clip.masks.shape == (48, 2160, 3840)
    assert len(clip.objects) == 2
    assert clip.paste_back_mae == 0.0

    # Non-existent scene raises LongSceneError
    with pytest.raises(LongSceneError, match="not registered"):
        load_long_scene_clip("alcaraz_highlights", "scene_999", 48)

    # Ineligible control raises LongSceneError
    with pytest.raises(LongSceneError, match="not eligible"):
        load_long_scene_clip("alcaraz_highlights", "scene_006", 48)

