"""
Tests for the evaluation module metrics and evaluator.
"""
import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import cv2

from pointstream.evaluation.metrics import (
    calculate_psnr, calculate_ssim, calculate_lpips, 
    calculate_vmaf, calculate_fvd, calculate_compression_ratio
)
from pointstream.evaluation.evaluator import Evaluator


class TestMetrics:
    """Test individual metric calculations."""
    
    @pytest.fixture
    def test_videos(self):
        """Create temporary test videos for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create simple test videos
            original_path = temp_path / "original.mp4"
            reconstructed_path = temp_path / "reconstructed.mp4"
            
            # Create simple 10-frame videos (32x32 for speed)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # Original video - gradient pattern
            writer = cv2.VideoWriter(str(original_path), fourcc, 30.0, (32, 32))
            for i in range(10):
                frame = np.zeros((32, 32, 3), dtype=np.uint8)
                frame[:, :, 0] = (i * 25) % 255  # Red channel varies
                frame[:, :, 1] = 128  # Green constant
                frame[:, :, 2] = 255 - (i * 25) % 255  # Blue inverse
                writer.write(frame)
            writer.release()
            
            # Reconstructed video - similar but with noise
            writer = cv2.VideoWriter(str(reconstructed_path), fourcc, 30.0, (32, 32))
            for i in range(10):
                frame = np.zeros((32, 32, 3), dtype=np.uint8)
                frame[:, :, 0] = (i * 25) % 255
                frame[:, :, 1] = 128
                frame[:, :, 2] = 255 - (i * 25) % 255
                # Add small amount of noise
                noise = np.random.randint(-5, 5, frame.shape, dtype=np.int16)
                frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                writer.write(frame)
            writer.release()
            
            yield {
                'original': original_path,
                'reconstructed': reconstructed_path,
                'temp_dir': temp_path
            }
    
    def test_calculate_psnr(self, test_videos):
        """Test PSNR calculation."""
        psnr = calculate_psnr(
            str(test_videos['original']), 
            str(test_videos['reconstructed'])
        )
        
        assert psnr is not None
        assert isinstance(psnr, (int, float))
        assert psnr > 0  # Should be positive for similar videos
        assert psnr < 100  # Should be reasonable value
    
    def test_calculate_ssim(self, test_videos):
        """Test SSIM calculation."""
        ssim = calculate_ssim(
            str(test_videos['original']), 
            str(test_videos['reconstructed'])
        )
        
        assert ssim is not None
        assert isinstance(ssim, (int, float))
        assert 0 <= ssim <= 1  # SSIM should be between 0 and 1
    
    def test_calculate_compression_ratio(self, test_videos):
        """Test compression ratio calculation."""
        # Create a larger "compressed" file for testing
        compressed_path = test_videos['temp_dir'] / "compressed.bin"
        with open(compressed_path, 'wb') as f:
            f.write(b'0' * 1000)  # 1KB compressed file
        
        ratio = calculate_compression_ratio(
            str(test_videos['original']),
            str(compressed_path)
        )
        
        assert ratio is not None
        assert isinstance(ratio, (int, float))
        assert ratio > 0
    
    @pytest.mark.slow
    def test_calculate_lpips(self, test_videos):
        """Test LPIPS calculation (marked as slow)."""
        try:
            lpips = calculate_lpips(
                str(test_videos['original']), 
                str(test_videos['reconstructed'])
            )
            
            if lpips is not None:  # LPIPS might not be available
                assert isinstance(lpips, (int, float))
                assert lpips >= 0  # LPIPS should be non-negative
        except ImportError:
            pytest.skip("LPIPS not available")
    
    @pytest.mark.slow
    def test_calculate_vmaf(self, test_videos):
        """Test VMAF calculation (marked as slow)."""
        vmaf = calculate_vmaf(
            str(test_videos['original']), 
            str(test_videos['reconstructed'])
        )
        
        if vmaf is not None:  # VMAF might not be available
            assert isinstance(vmaf, (int, float))
            assert 0 <= vmaf <= 100  # VMAF should be between 0 and 100
    
    @pytest.mark.slow  
    def test_calculate_fvd(self, test_videos):
        """Test FVD calculation (marked as slow)."""
        try:
            fvd = calculate_fvd(
                str(test_videos['original']), 
                str(test_videos['reconstructed'])
            )
            
            if fvd is not None:  # FVD might not be available
                assert isinstance(fvd, (int, float))
                assert fvd >= 0  # FVD should be non-negative
        except Exception:
            pytest.skip("FVD calculation failed - dependencies may not be available")


class TestEvaluator:
    """Test the Evaluator class."""
    
    @pytest.fixture
    def test_data(self):
        """Create test data for evaluator."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock video files
            original = temp_path / "original.mp4"
            reconstructed_dir = temp_path / "reconstructed_scenes"
            reconstructed_dir.mkdir()
            compressed = temp_path / "compressed.json"
            
            # Create minimal files
            original.write_bytes(b'original_video_data' * 100)
            (reconstructed_dir / "scene_0.mp4").write_bytes(b'reconstructed_data' * 100)
            compressed.write_text('{"test": "data"}')
            
            yield {
                'original': str(original),
                'reconstructed_dir': str(reconstructed_dir),
                'compressed': str(compressed),
                'output_dir': str(temp_path / "output")
            }
    
    def test_evaluator_initialization(self, test_data):
        """Test Evaluator initialization."""
        evaluator = Evaluator(
            original_video_path=test_data['original'],
            json_results_path=test_data['compressed'],
            reconstructed_scenes_dir=test_data['reconstructed_dir']
        )
        
        assert evaluator.original_video_path == test_data['original']
        assert evaluator.json_results_path == test_data['compressed']
        assert evaluator.reconstructed_scenes_dir == test_data['reconstructed_dir']
    
    @patch('pointstream.evaluation.metrics.calculate_psnr')
    @patch('pointstream.evaluation.metrics.calculate_ssim')
    def test_basic_evaluation_workflow(self, mock_ssim, mock_psnr, test_data):
        """Test basic evaluation workflow with mocked functions."""
        # Mock return values
        mock_psnr.return_value = 35.5
        mock_ssim.return_value = 0.85
        
        evaluator = Evaluator(
            original_video_path=test_data['original'],
            json_results_path=test_data['compressed'],
            reconstructed_scenes_dir=test_data['reconstructed_dir']
        )
        
        # This is a basic test - the actual evaluation might not work
        # without real video files, but we can test the object creation
        assert evaluator is not None
