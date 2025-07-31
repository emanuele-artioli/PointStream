"""
Tests for Stage 4: Foreground Representation.
"""
import pytest
from pathlib import Path
from pointstream.pipeline import stage_01_analyzer, stage_02_detector, stage_03_background, stage_04_foreground

@pytest.fixture
def test_video_path() -> Path:
    """Provides the path to the test video."""
    path = Path(__file__).parent / "data" / "DAVIS_stitched.mp4"
    if not path.exists():
        pytest.skip(f"Test video not found at {path}")
    return path

def test_run_foreground_pipeline_streaming(test_video_path):
    """Tests the main orchestrator function of the foreground representation stage."""
    stage1_gen = stage_01_analyzer.run_analysis_pipeline(str(test_video_path))
    stage2_gen = stage_02_detector.run_detection_pipeline(stage1_gen)
    stage3_gen = stage_03_background.run_background_modeling_pipeline(stage2_gen, test_video_path.stem)
    stage4_gen = stage_04_foreground.run_foreground_pipeline(stage3_gen, str(test_video_path))

    results = list(stage4_gen)

    assert len(results) > 0, "Foreground pipeline did not yield any scenes."

    # Check all scenes for correct final structure
    for scene in results:
        assert "foreground_objects" in scene
        assert "detections" not in scene, "The 'detections' key should be removed from the final output."

    processed_scene = next((s for s in results if s.get("foreground_objects")), None)
    if processed_scene:
        fg_objects = processed_scene["foreground_objects"]
        assert isinstance(fg_objects, list)
        
        first_obj = fg_objects[0]
        assert "track_id" in first_obj
        assert "appearance_path" in first_obj
        assert Path(first_obj["appearance_path"]).exists()
        assert "keypoint_type" in first_obj
        assert "keypoints" in first_obj
        assert len(first_obj["keypoints"]) > 0