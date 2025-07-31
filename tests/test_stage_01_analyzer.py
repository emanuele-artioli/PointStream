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
        pytest.skip("Test video not found at tests/data/DAVIS_stitched.mp4")
    return path

def test_run_analysis_pipeline_streaming(test_video_path):
    """
    Tests the main orchestrator function of the analysis stage with streaming.
    This version is more robust and checks all yielded scenes.
    """
    # Arrange: Create the Stage 1 generator
    scene_analysis_generator = stage_01_analyzer.run_analysis_pipeline(str(test_video_path))

    # Act: Consume the generator into a list to check its full output
    results = list(scene_analysis_generator)

    # Assert
    # 1. Ensure that the pipeline produced at least one scene
    assert len(results) > 0, "The analysis pipeline did not yield any scenes."

    # 2. Check the structure of EVERY yielded scene
    for i, scene in enumerate(results):
        print(f"Testing Scene {i}: {scene['start_frame']}-{scene['end_frame']}")
        assert "scene_index" in scene
        assert "start_frame" in scene
        assert "end_frame" in scene
        assert "motion_type" in scene
        assert scene["motion_type"] in ["STATIC", "SIMPLE", "COMPLEX"]
        
        # The 'frames' key should exist at this point, before being popped in Stage 2
        assert "frames" in scene, f"Scene {i} is missing the 'frames' key."
        assert isinstance(scene["frames"], list), f"Scene {i} 'frames' is not a list."
        assert len(scene["frames"]) > 0, f"Scene {i} has an empty 'frames' list."

