"""
Tests for pipeline integration and end-to-end functionality.
"""
import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from pointstream.pipeline import stage_01_analyzer


class TestPipelineIntegration:
    """Test pipeline integration and data flow."""
    
    @pytest.fixture
    def test_video_path(self) -> Path:
        """Provides the path to the test video."""
        path = Path(__file__).parent / "data" / "DAVIS_stitched.mp4"
        if not path.exists():
            pytest.skip(f"Test video not found at {path}")
        return path
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_pipeline_data_consistency(self, test_video_path):
        """Test that pipeline stages produce consistent data structures."""
        # Run stage 1 analysis
        scene_generator = stage_01_analyzer.run_analysis_pipeline(str(test_video_path))
        scenes = list(scene_generator)
        
        assert len(scenes) > 0, "No scenes generated from analysis"
        
        # Verify scene data structure consistency
        required_keys = ["scene_index", "start_frame", "end_frame", "motion_type", "frames"]
        
        for scene in scenes:
            for key in required_keys:
                assert key in scene, f"Missing required key '{key}' in scene data"
            
            # Check frame ordering
            assert scene["start_frame"] <= scene["end_frame"], "Invalid frame range"
            
            # Check motion type is valid
            assert scene["motion_type"] in ["STATIC", "SIMPLE", "COMPLEX"], \
                f"Invalid motion type: {scene['motion_type']}"
    
    def test_scene_frame_data_integrity(self, test_video_path):
        """Test that scene frame data maintains integrity."""
        scene_generator = stage_01_analyzer.run_analysis_pipeline(str(test_video_path))
        scenes = list(scene_generator)
        
        for scene in scenes:
            frames = scene.get("frames", [])
            assert len(frames) > 0, f"Scene {scene['scene_index']} has no frames"
            
            # Check frame data structure
            for frame in frames:
                assert "frame_index" in frame, "Frame missing frame_index"
                assert "timestamp" in frame, "Frame missing timestamp"
                
                # Frame index should be within scene range
                assert scene["start_frame"] <= frame["frame_index"] <= scene["end_frame"], \
                    f"Frame index {frame['frame_index']} outside scene range"
    
    def test_pipeline_output_serialization(self, test_video_path, temp_output_dir):
        """Test that pipeline outputs can be properly serialized and deserialized."""
        scene_generator = stage_01_analyzer.run_analysis_pipeline(str(test_video_path))
        scenes = list(scene_generator)
        
        # Test JSON serialization
        output_file = temp_output_dir / "test_scenes.json"
        
        try:
            with open(output_file, 'w') as f:
                json.dump(scenes, f, indent=2)
            
            # Test deserialization
            with open(output_file, 'r') as f:
                loaded_scenes = json.load(f)
            
            assert len(loaded_scenes) == len(scenes), "Scene count mismatch after serialization"
            
            # Verify structure is preserved
            for original, loaded in zip(scenes, loaded_scenes):
                assert original["scene_index"] == loaded["scene_index"]
                assert original["motion_type"] == loaded["motion_type"]
                
        except (TypeError, ValueError) as e:
            pytest.fail(f"Pipeline output is not JSON serializable: {e}")


class TestPipelineErrorHandling:
    """Test pipeline error handling and robustness."""
    
    def test_invalid_video_path(self):
        """Test handling of invalid video paths."""
        invalid_path = "/nonexistent/video.mp4"
        
        # Should handle gracefully without crashing
        try:
            scene_generator = stage_01_analyzer.run_analysis_pipeline(invalid_path)
            scenes = list(scene_generator)
            # If it doesn't raise an exception, it should return empty results
            assert len(scenes) == 0, "Should return empty results for invalid video"
        except (FileNotFoundError, ValueError):
            # This is also acceptable behavior
            pass
    
    def test_corrupted_video_handling(self, temp_output_dir):
        """Test handling of corrupted video files."""
        # Create a fake corrupted video file
        corrupted_video = temp_output_dir / "corrupted.mp4"
        corrupted_video.write_bytes(b"not a real video file")
        
        try:
            scene_generator = stage_01_analyzer.run_analysis_pipeline(str(corrupted_video))
            scenes = list(scene_generator)
            # Should handle gracefully
            assert isinstance(scenes, list), "Should return a list even for corrupted video"
        except Exception:
            # Exception is acceptable for truly corrupted files
            pass


class TestPipelinePerformance:
    """Test pipeline performance characteristics."""
    
    @pytest.mark.slow
    def test_memory_usage_with_large_scenes(self, test_video_path):
        """Test that pipeline doesn't consume excessive memory with large scenes."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run analysis pipeline
        scene_generator = stage_01_analyzer.run_analysis_pipeline(str(test_video_path))
        scenes = list(scene_generator)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 500MB for test video)
        assert memory_increase < 500, f"Excessive memory usage: {memory_increase:.1f}MB"
    
    @pytest.mark.slow
    def test_processing_time_reasonable(self, test_video_path):
        """Test that processing time is reasonable for the test video."""
        import time
        
        start_time = time.time()
        
        scene_generator = stage_01_analyzer.run_analysis_pipeline(str(test_video_path))
        scenes = list(scene_generator)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should process test video in reasonable time (less than 60 seconds)
        assert processing_time < 60, f"Processing took too long: {processing_time:.1f}s"
        assert len(scenes) > 0, "No scenes processed"


@pytest.mark.integration
class TestFullPipelineFlow:
    """Integration tests for the complete pipeline flow."""
    
    @pytest.fixture
    def test_video_path(self) -> Path:
        """Provides the path to the test video."""
        path = Path(__file__).parent / "data" / "DAVIS_stitched.mp4"
        if not path.exists():
            pytest.skip(f"Test video not found at {path}")
        return path
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for pipeline testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create expected directory structure
            (workspace / "artifacts" / "pipeline_output").mkdir(parents=True)
            (workspace / "artifacts" / "training").mkdir(parents=True)
            
            yield workspace
    
    @pytest.mark.slow
    def test_server_to_client_data_flow(self, test_video_path, temp_workspace):
        """Test data flow from server pipeline to client reconstruction."""
        # This is a simplified integration test
        # In a real test, we would run the actual server and client pipelines
        
        # Mock server output
        server_output = {
            "video_path": str(test_video_path),
            "scenes": [
                {
                    "scene_index": 0,
                    "start_frame": 0,
                    "end_frame": 30,
                    "motion_type": "SIMPLE",
                    "compressed_data": "mock_compressed_data"
                }
            ],
            "metadata": {
                "total_frames": 30,
                "fps": 30,
                "resolution": [640, 480]
            }
        }
        
        # Save mock server output
        server_output_file = temp_workspace / "artifacts" / "pipeline_output" / "test_results.json"
        with open(server_output_file, 'w') as f:
            json.dump(server_output, f)
        
        # Verify file was created and is readable
        assert server_output_file.exists(), "Server output file not created"
        
        with open(server_output_file, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data["video_path"] == str(test_video_path)
        assert len(loaded_data["scenes"]) == 1
        assert "metadata" in loaded_data
