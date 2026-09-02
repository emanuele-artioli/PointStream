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
    conf_vids = [
        "alcaraz_perricard",
        "alcaraz_ruud",
        "djokovic_federer",
        "djokovic_zverev",
        "sinner_alcaraz",
        "tournament_match_6",
    ]
    diag_vids = ["alcaraz_highlights", "federer_djokovic"]

    scenes: list[dict[str, Any]] = [
        # Diagnostic near-static
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
                    "start_frame": 0,
                    "end_frame": span,
                    "status": "eligible",
                    "frame_hashes": {"first": "h1", "mid": "h2", "last": "h3"},
                    "paste_back_mae": 0.0,
                    "canvas_growth": 1.00,
                    "failure_reasons": [],
                }
                for span in TARGET_SPANS
            },
        },
        # Diagnostic smooth-pan
        {
            "video": "alcaraz_highlights",
            "scene": "scene_010",
            "t_start": 40.0,
            "t_end": 55.0,
            "duration": 15.0,
            "cluster": "cluster_point",
            "context_id": "alcaraz_highlights_main_court",
            "role": "diagnostic_smooth_pan",
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
                "sha256": "def456mock",
            },
            "eligibility": {
                "duration_24fps_frames": 360,
                "camera_motion": {
                    "consecutive_mad": 1.45,
                    "vs_first_frame_mad": 2.50,
                    "last_vs_first_mad": 3.10,
                },
                "panorama": {
                    "canvas_width": 3950,
                    "canvas_height": 2160,
                    "growth_factor": 1.04,
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
                    "frame_hashes": {"first": "h1", "mid": "h2", "last": "h3"},
                    "paste_back_mae": 0.0,
                    "canvas_growth": 1.04,
                    "failure_reasons": [],
                }
                for span in TARGET_SPANS
            },
        },
        # Diagnostic smooth-pan (video 2: federer_djokovic)
        {
            "video": "federer_djokovic",
            "scene": "scene_001",
            "t_start": 0.0,
            "t_end": 20.0,
            "duration": 20.0,
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
                "sha256": "fed123mock",
            },
            "eligibility": {
                "duration_24fps_frames": 480,
                "camera_motion": {
                    "consecutive_mad": 1.85,
                    "vs_first_frame_mad": 2.80,
                    "last_vs_first_mad": 3.40,
                },
                "panorama": {
                    "canvas_width": 3960,
                    "canvas_height": 2160,
                    "growth_factor": 1.05,
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
                    "frame_hashes": {"first": "h1", "mid": "h2", "last": "h3"},
                    "paste_back_mae": 0.0,
                    "canvas_growth": 1.05,
                    "failure_reasons": [],
                }
                for span in TARGET_SPANS
            },
        },
        # Ineligible control
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
    ]

    # Add confirmation scenes for all 6 independent confirmation videos
    for vid in conf_vids:
        scenes.append({
            "video": vid,
            "scene": "scene_001",
            "t_start": 0.0,
            "t_end": 20.0,
            "duration": 20.0,
            "cluster": "cluster_point",
            "context_id": f"{vid}_main_court",
            "role": "confirmation",
            "source_metadata": {
                "video_file": f"{vid}.mp4",
                "width": 3840,
                "height": 2160,
                "source_fps": 59.94,
                "working_fps": 24.0,
                "pix_fmt": "yuv420p",
                "color_space": "bt709",
                "color_primaries": "bt709",
                "color_transfer": "bt709",
                "sha256": f"sha256_{vid}",
            },
            "eligibility": {
                "duration_24fps_frames": 480,
                "camera_motion": {
                    "consecutive_mad": 1.1,
                    "vs_first_frame_mad": 2.2,
                    "last_vs_first_mad": 2.8,
                },
                "panorama": {
                    "canvas_width": 3900,
                    "canvas_height": 2160,
                    "growth_factor": 1.02,
                    "registration_ok": True,
                },
                "objects": {
                    "num_objects": 2,
                    "object_class": "player",
                    "player_pixel_fraction": 0.008,
                    "min_separation_px": 700.0,
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
                    "frame_hashes": {"first": "h_f", "mid": "h_m", "last": "h_l"},
                    "paste_back_mae": 0.0,
                    "canvas_growth": 1.02,
                    "failure_reasons": [],
                }
                for span in TARGET_SPANS
            },
        })

    return {
        "schema": SCHEMA_ID,
        "created_utc": "2026-09-02T18:00:00Z",
        "source_data_root": "/home/itec/emanuele/pointstream-data",
        "target_spans": list(TARGET_SPANS),
        "diagnostic_videos": diag_vids,
        "confirmation_videos": conf_vids,
        "ineligible_controls": ["alcaraz_highlights/scene_006"],
        "summary": {
            "submitted_scenes": len(scenes),
            "pointstream_eligible_scenes": len(scenes) - 1,
            "conventional_fallback_scenes": 1,
            "succeeded_by_span": {f"{s}_frames": len(scenes) - 1 for s in TARGET_SPANS},
            "failed_by_span": {f"{s}_frames": 1 for s in TARGET_SPANS},
        },
        "scenes": scenes,
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
    assert res["verdict"] == "diagnostic inputs and confirmation corpus fully verified"
    assert len(res["confirmation_videos"]) == 6
    assert len(res["diagnostic_videos"]) == 2


def test_manifest_verification_rejects_missing_diagnostic() -> None:
    manifest = _build_valid_manifest()
    manifest["diagnostic_videos"] = ["alcaraz_highlights"]  # only 1 video (< 2)
    with pytest.raises(ManifestValidationError, match="expected >= 2 diagnostic videos"):
        verify_manifest(manifest)


def test_manifest_verification_rejects_split_isolation_overlap() -> None:
    manifest = _build_valid_manifest()
    # Violate split isolation: put a confirmation video into diagnostic_videos
    manifest["diagnostic_videos"].append("alcaraz_perricard")
    with pytest.raises(ManifestValidationError, match="split isolation violated"):
        verify_manifest(manifest)


def test_manifest_verification_rejects_role_membership_mismatch() -> None:
    manifest = _build_valid_manifest()
    # Confirmation scene with non-confirmation role
    manifest["scenes"][4]["role"] = "diagnostic_smooth_pan"
    with pytest.raises(ManifestValidationError, match="has non-confirmation role"):
        verify_manifest(manifest)


def test_manifest_verification_rejects_conflated_contexts() -> None:
    manifest = _build_valid_manifest()
    # Conflate context across different videos
    manifest["scenes"][4]["context_id"] = manifest["scenes"][0]["context_id"]
    with pytest.raises(ManifestValidationError, match="groups unrelated source videos"):
        verify_manifest(manifest)


def test_manifest_verification_rejects_ineligible_without_reasons() -> None:
    manifest = _build_valid_manifest()
    manifest["scenes"][3]["eligibility"]["ineligibility_reasons"] = []
    with pytest.raises(ManifestValidationError, match="must record ineligibility_reasons"):
        verify_manifest(manifest)


def test_manifest_verification_rejects_interval_missing_hashes() -> None:
    manifest = _build_valid_manifest()
    manifest["scenes"][0]["intervals"]["48"]["frame_hashes"] = {"first": "h1", "last": "h2"}  # missing mid
    with pytest.raises(ManifestValidationError, match="missing frame_hashes"):
        verify_manifest(manifest)


def test_manifest_verification_rejects_wrong_span_frame_count() -> None:
    manifest = _build_valid_manifest()
    manifest["scenes"][0]["intervals"]["48"]["frame_count"] = 50
    with pytest.raises(ManifestValidationError, match="frame_count"):
        verify_manifest(manifest)


def test_manifest_verification_reports_diagnostic_ready_when_confirmation_incomplete() -> None:
    manifest = _build_valid_manifest()
    # Make one confirmation video ineligible at 384 frames
    target_vid = "tournament_match_6"
    for s in manifest["scenes"]:
        if s["video"] == target_vid:
            s["intervals"]["384"]["status"] = "ineligible"
            s["intervals"]["384"]["failure_reasons"] = ["mock fail"]
    res = verify_manifest(manifest)
    assert res["status"] == "DIAGNOSTIC_READY_CONFIRMATION_INCOMPLETE"
    assert res["verdict"] == "diagnostic inputs ready; confirmation corpus incomplete"
    assert res["diagnostic_status"] == "READY"
    assert res["confirmation_status"] == "INCOMPLETE"
    assert len(res["confirmation_deficits"]) > 0

    # Strict confirmation raises ManifestValidationError
    with pytest.raises(ManifestValidationError, match="confirmation corpus incomplete"):
        verify_manifest(manifest, strict_confirmation=True)


def test_measure_motion_and_panorama_static() -> None:
    frames = np.ones((8, 100, 100, 3), dtype=np.uint8) * 128
    mot, pano = measure_motion_and_panorama(frames)
    assert mot.consecutive_mad == 0.0
    assert mot.vs_first_frame_mad == 0.0
    assert mot.last_vs_first_mad == 0.0
    assert pano.growth_factor == 1.0


@pytest.mark.skipif(
    __import__("os").environ.get("POINTSTREAM_DATA_TESTS") != "1",
    reason="requires external YouTube-derived footage; set POINTSTREAM_DATA_TESTS=1",
)
def test_load_long_scene_clip_verified() -> None:
    from experiments.long_scenes.loader import LongSceneError, load_long_scene_clip

    # Valid scene and span
    clip = load_long_scene_clip("alcaraz_highlights", "scene_028", 48)
    assert clip.n_frames == 48
    assert clip.frames.shape == (48, 2160, 3840, 3)
    assert clip.masks.shape == (48, 2160, 3840)
    assert len(clip.objects) == 2
    assert clip.paste_back_mae == 0.0
    assert clip.is_eligible is True

    # Non-existent scene raises LongSceneError
    with pytest.raises(LongSceneError, match="not registered"):
        load_long_scene_clip("alcaraz_highlights", "scene_999", 48)

    # Ineligible control raises LongSceneError by default
    with pytest.raises(LongSceneError, match="not eligible"):
        load_long_scene_clip("alcaraz_highlights", "scene_006", 48)

    # Ineligible control can be loaded when allow_ineligible=True
    fallback_clip = load_long_scene_clip("alcaraz_highlights", "scene_006", 48, allow_ineligible=True)
    assert fallback_clip.n_frames == 48
    assert fallback_clip.is_eligible is False
    assert fallback_clip.route == "conventional_fallback"
    assert len(fallback_clip.failure_reasons) > 0
    # Small-resolution smoke only: actual ineligible content and recorded route
    # through the production fallback path, not a rate-quality comparison.
    from src.runner.fallback import deliver_fallback
    from src.contracts.config import FallbackConfig
    from src.contracts.codecs import RateControl
    source = np.ascontiguousarray(fallback_clip.frames[:, ::12, ::12])
    delivered = deliver_fallback(
        source, FallbackConfig(codec="av1", preset="0", rate_control=RateControl.QP, rate=63),
        route=fallback_clip.route,
    )
    assert delivered.trip.frames.shape == source.shape
    assert delivered.transport_bytes == delivered.trip.size_bytes + len(delivered.routing_header)
