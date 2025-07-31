"""
Tests for Stage 3: Background Modeling.
"""
import pytest
from pathlib import Path
from pointstream.pipeline import stage_01_analyzer, stage_02_detector, stage_03_background

@pytest.fixture
def test_video_path() -> Path:
    """Provides the path to the test video."""
    path = Path(__file__).parent / "data" / "DAVIS_stitched.mp4"
    if not path.exists():
        pytest.skip("Test video not found at tests/data/DAVIS_stitched.mp4")
    return path

def test_run_background_pipeline_streaming(test_video_path):
    """
    Tests the main orchestrator function of the background modeling stage.
    """
    # Arrange: Chain all pipeline stages together
    video_stem = test_video_path.stem
    stage1_gen = stage_01_analyzer.run_analysis_pipeline(str(test_video_path))
    stage2_gen = stage_02_detector.run_detection_pipeline(stage1_gen)
    stage3_gen = stage_03_background.run_background_modeling_pipeline(stage2_gen, video_stem)

    # Act: Consume the final generator
    results = list(stage3_gen)

    # Assert
    assert len(results) > 0, "Background pipeline did not yield any scenes."

    # Find a processed STATIC or SIMPLE scene to check its contents
    processed_scene = next(
        (s for s in results if s["motion_type"] in ["STATIC", "SIMPLE"]), None
    )

    if processed_scene:
        assert "background_image_path" in processed_scene, "Scene missing 'background_image_path' key."
        assert processed_scene["background_image_path"] is not None, "Background image path should not be None for a processed scene."
        
        bg_path = Path(processed_scene["background_image_path"])
        assert bg_path.exists(), f"Background image file was not created at {bg_path}"
        
        assert "camera_motion" in processed_scene, "Scene missing 'camera_motion' key."
        assert isinstance(processed_scene["camera_motion"], list), "'camera_motion' should be a list."
        assert len(processed_scene["camera_motion"]) > 0, "Camera motion list should not be empty."