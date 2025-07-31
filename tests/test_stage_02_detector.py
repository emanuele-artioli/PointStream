"""
Tests for Stage 2: Foreground Object Detection and Tracking.
"""
import pytest
from pathlib import Path
from pointstream.pipeline import stage_01_analyzer, stage_02_detector

@pytest.fixture
def test_video_path() -> Path:
    """Provides the path to the test video."""
    path = Path(__file__).parent / "data" / "DAVIS_stitched.mp4"
    if not path.exists():
        pytest.skip("Test video not found at tests/data/DAVIS_stitched.mp4")
    return path

def test_run_detection_pipeline_streaming(test_video_path):
    """
    Tests the main orchestrator function of the detection stage with streaming.
    """
    # Arrange
    scene_analysis_generator = stage_01_analyzer.run_analysis_pipeline(str(test_video_path))
    
    # Act
    detection_results = list(stage_02_detector.run_detection_pipeline(scene_analysis_generator))

    # Assert
    assert len(detection_results) > 0, "Detection pipeline did not yield any scenes."

    processed_scene = detection_results[0]
    assert "detections" in processed_scene, "Scene is missing 'detections' key after Stage 2."
    # FIX: The 'frames' key SHOULD exist after Stage 2 for Stage 3 to use.
    assert "frames" in processed_scene, "The 'frames' key should be passed through Stage 2."