"""
Tests for Stage 1: Scene Analysis and Motion Classification.
"""
import pytest
from pathlib import Path
from pointstream.pipeline import stage_01_analyzer

@pytest.fixture
def test_video_path() -> Path:
    """Provides the path to the test video."""
    path = Path(__file__).parent / "data" / "DAVIS_stitched.mp4"
    if not path.exists():
        pytest.skip(f"Test video not found at {path}")
    return path

def test_run_analysis_pipeline_streaming(test_video_path):
    """Tests the main orchestrator function of the analysis stage with streaming."""
    scene_analysis_generator = stage_01_analyzer.run_analysis_pipeline(str(test_video_path))
    results = list(scene_analysis_generator)

    assert len(results) > 0, "The analysis pipeline did not yield any scenes."

    for i, scene in enumerate(results):
        assert "scene_index" in scene, f"Scene {i} missing 'scene_index'"
        assert "start_frame" in scene, f"Scene {i} missing 'start_frame'"
        assert "end_frame" in scene, f"Scene {i} missing 'end_frame'"
        assert "motion_type" in scene, f"Scene {i} missing 'motion_type'"
        assert scene["motion_type"] in ["STATIC", "SIMPLE", "COMPLEX"]
        assert "frames" in scene, f"Scene {i} missing 'frames' key."
        assert len(scene["frames"]) > 0, f"Scene {i} has an empty 'frames' list."