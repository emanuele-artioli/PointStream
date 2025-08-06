"""
Main Evaluator Class

Coordinates comprehensive evaluation of PointStream pipeline results.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import time
from dataclasses import dataclass
import pandas as pd

from .metrics import CompressionMetrics, QualityMetrics, VideoFrameExtractor

logger = logging.getLogger(__name__)

@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    
    # Compression metrics
    original_size_mb: float
    compressed_size_mb: float
    json_size_mb: float
    compression_ratio: float
    space_savings_percent: float
    bits_per_pixel: float
    
    # Quality metrics
    avg_psnr: float
    avg_ssim: float
    avg_lpips: float
    vmaf_score: float
    fvd_score: float
    
    # Pipeline metrics
    num_scenes: int
    num_objects: int
    processing_time: float
    reconstruction_time: float
    
    # Video info
    original_resolution: str
    original_fps: float
    total_frames: int
    duration_seconds: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'compression_metrics': {
                'original_size_mb': self.original_size_mb,
                'compressed_size_mb': self.compressed_size_mb,
                'json_size_mb': self.json_size_mb,
                'compression_ratio': self.compression_ratio,
                'space_savings_percent': self.space_savings_percent,
                'bits_per_pixel': self.bits_per_pixel,
            },
            'quality_metrics': {
                'avg_psnr': self.avg_psnr,
                'avg_ssim': self.avg_ssim,
                'avg_lpips': self.avg_lpips,
                'vmaf_score': self.vmaf_score,
                'fvd_score': self.fvd_score,
            },
            'pipeline_metrics': {
                'num_scenes': self.num_scenes,
                'num_objects': self.num_objects,
                'processing_time': self.processing_time,
                'reconstruction_time': self.reconstruction_time,
            },
            'video_info': {
                'original_resolution': self.original_resolution,
                'original_fps': self.original_fps,
                'total_frames': self.total_frames,
                'duration_seconds': self.duration_seconds,
            }
        }


class Evaluator:
    """Main evaluator for PointStream pipeline."""
    
    def __init__(self, original_video_path: str, json_results_path: str, 
                 reconstructed_scenes_dir: str, vmaf_model: str = "vmaf_v0.6.1"):
        self.original_video_path = original_video_path
        self.json_results_path = json_results_path
        self.reconstructed_scenes_dir = reconstructed_scenes_dir
        self.vmaf_model = vmaf_model
        
        self.compression_metrics = CompressionMetrics()
        self.quality_metrics = QualityMetrics()
        self.frame_extractor = VideoFrameExtractor()
        
    def evaluate_pipeline(self, max_frames_for_quality: int = 100, 
                         compute_fvd: bool = True) -> EvaluationResults:
        """
        Perform comprehensive evaluation of the PointStream pipeline.
        
        Args:
            max_frames_for_quality: Maximum frames to use for quality metrics
            compute_fvd: Whether to compute FVD (computationally expensive)
            
        Returns:
            EvaluationResults object with all metrics
        """
        logger.info("Starting comprehensive pipeline evaluation...")
        
        # Load pipeline results
        with open(self.json_results_path, 'r') as f:
            pipeline_data = json.load(f)
        
        # Get file sizes
        original_size = Path(self.original_video_path).stat().st_size
        json_size = Path(self.json_results_path).stat().st_size
        
        # Calculate total reconstructed size
        reconstructed_scenes = list(Path(self.reconstructed_scenes_dir).glob("*.mp4"))
        total_reconstructed_size = sum(scene.stat().st_size for scene in reconstructed_scenes)
        
        # Get video info
        video_info = self.frame_extractor.get_video_info(self.original_video_path)
        
        # Calculate compression metrics
        compression_ratio = self.compression_metrics.calculate_compression_ratio(
            original_size, total_reconstructed_size
        )
        space_savings = self.compression_metrics.calculate_space_savings(
            original_size, total_reconstructed_size
        )
        bits_per_pixel = self.compression_metrics.calculate_bits_per_pixel(
            total_reconstructed_size, 
            video_info['width'], 
            video_info['height'], 
            video_info['frame_count']
        )
        
        # Calculate quality metrics
        quality_results = self._evaluate_quality(
            self.original_video_path, 
            reconstructed_scenes, 
            max_frames_for_quality,
            compute_fvd
        )
        
        # Extract pipeline metrics from timestamps if available
        processing_time = 0.0
        reconstruction_time = 0.0
        if 'processing_time' in pipeline_data:
            processing_time = pipeline_data['processing_time']
        
        # Extract pipeline metrics
        num_scenes = len(pipeline_data['scenes'])
        num_objects = sum(len(scene.get('foreground_objects', [])) for scene in pipeline_data['scenes'])
        
        # Create results object
        results = EvaluationResults(
            # Compression metrics
            original_size_mb=original_size / (1024 * 1024),
            compressed_size_mb=total_reconstructed_size / (1024 * 1024),
            json_size_mb=json_size / (1024 * 1024),
            compression_ratio=compression_ratio,
            space_savings_percent=space_savings,
            bits_per_pixel=bits_per_pixel,
            
            # Quality metrics
            avg_psnr=quality_results['avg_psnr'],
            avg_ssim=quality_results['avg_ssim'], 
            avg_lpips=quality_results['avg_lpips'],
            vmaf_score=quality_results['vmaf_score'],
            fvd_score=quality_results['fvd_score'],
            
            # Pipeline metrics
            num_scenes=num_scenes,
            num_objects=num_objects,
            processing_time=processing_time,
            reconstruction_time=reconstruction_time,
            
            # Video info
            original_resolution=f"{video_info['width']}x{video_info['height']}",
            original_fps=video_info['fps'],
            total_frames=video_info['frame_count'],
            duration_seconds=video_info['frame_count'] / video_info['fps'] if video_info['fps'] > 0 else 0,
        )
        
        logger.info("Pipeline evaluation completed successfully")
        return results
    
    def _evaluate_quality(self, original_video: str, reconstructed_scenes: List[Path], 
                         max_frames: int = 100, compute_fvd: bool = True) -> Dict[str, float]:
        """Evaluate video quality metrics."""
        logger.info("Calculating quality metrics...")
        
        # Extract frames from original video
        original_frames = self.frame_extractor.extract_frames(original_video, max_frames=max_frames)
        
        if not original_frames:
            logger.error("Could not extract frames from original video")
            return {
                'avg_psnr': 0.0,
                'avg_ssim': 0.0, 
                'avg_lpips': 0.0,
                'vmaf_score': 0.0,
                'fvd_score': 0.0
            }
        
        # Collect all reconstructed frames
        all_reconstructed_frames = []
        for scene_path in sorted(reconstructed_scenes):
            scene_frames = self.frame_extractor.extract_frames(str(scene_path), max_frames=50)
            all_reconstructed_frames.extend(scene_frames)
        
        if not all_reconstructed_frames:
            logger.error("Could not extract frames from reconstructed videos")
            return {
                'avg_psnr': 0.0,
                'avg_ssim': 0.0,
                'avg_lpips': 0.0, 
                'vmaf_score': 0.0,
                'fvd_score': 0.0
            }
        
        # Calculate frame-wise quality metrics
        psnr_scores = []
        ssim_scores = []
        lpips_scores = []
        
        # Sample frames for comparison
        min_frames = min(len(original_frames), len(all_reconstructed_frames))
        comparison_frames = min(50, min_frames)  # Limit for performance
        
        frame_indices = np.linspace(0, min_frames - 1, comparison_frames, dtype=int)
        
        for i in frame_indices:
            orig_frame = original_frames[i % len(original_frames)]
            recon_frame = all_reconstructed_frames[i % len(all_reconstructed_frames)]
            
            # Resize to same size for comparison
            if orig_frame.shape != recon_frame.shape:
                recon_frame = self._resize_frame(recon_frame, orig_frame.shape[:2])
            
            # Calculate metrics
            psnr_score = self.quality_metrics.calculate_psnr(orig_frame, recon_frame)
            ssim_score = self.quality_metrics.calculate_ssim(orig_frame, recon_frame)
            lpips_score = self.quality_metrics.calculate_lpips(orig_frame, recon_frame)
            
            psnr_scores.append(psnr_score)
            ssim_scores.append(ssim_score)
            if lpips_score >= 0:  # Only include valid LPIPS scores
                lpips_scores.append(lpips_score)
        
        # Calculate VMAF (requires reconstructed video file)
        vmaf_score = 0.0
        if reconstructed_scenes:
            # Use first scene for VMAF calculation as example
            first_scene = reconstructed_scenes[0]
            vmaf_score = self.quality_metrics.calculate_vmaf(original_video, str(first_scene), self.vmaf_model)
        
        # Calculate FVD
        fvd_score = 0.0
        if compute_fvd:
            fvd_score = self.quality_metrics.calculate_fvd(
                original_frames[:min(len(original_frames), 20)],
                all_reconstructed_frames[:min(len(all_reconstructed_frames), 20)]
            )
        
        return {
            'avg_psnr': np.mean(psnr_scores) if psnr_scores else 0.0,
            'avg_ssim': np.mean(ssim_scores) if ssim_scores else 0.0,
            'avg_lpips': np.mean(lpips_scores) if lpips_scores else 0.0,
            'vmaf_score': vmaf_score if vmaf_score >= 0 else 0.0,  # Keep 0 for failed VMAF
            'fvd_score': fvd_score if fvd_score >= 0 else 0.0,    # Keep 0 for failed FVD
        }
    
    def _resize_frame(self, frame: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Resize frame to target shape."""
        import cv2
        return cv2.resize(frame, (target_shape[1], target_shape[0]))
    
    def generate_report(self, results: EvaluationResults, output_path: str):
        """Generate comprehensive evaluation report."""
        report = f"""
# PointStream Pipeline Evaluation Report

## Compression Performance
- **Original Size**: {results.original_size_mb:.2f} MB
- **Compressed Size**: {results.compressed_size_mb:.2f} MB
- **JSON Data Size**: {results.json_size_mb:.2f} MB
- **Compression Ratio**: {results.compression_ratio:.2f}:1
- **Space Savings**: {results.space_savings_percent:.1f}%
- **Bits per Pixel**: {results.bits_per_pixel:.4f}

## Video Quality Metrics
- **Average PSNR**: {results.avg_psnr:.2f} dB
- **Average SSIM**: {results.avg_ssim:.4f}
- **Average LPIPS**: {results.avg_lpips:.4f}
- **VMAF Score**: {results.vmaf_score:.2f}
- **FVD Score**: {results.fvd_score:.2f}

## Pipeline Statistics
- **Number of Scenes**: {results.num_scenes}
- **Number of Objects**: {results.num_objects}
- **Processing Time**: {results.processing_time:.2f} seconds
- **Reconstruction Time**: {results.reconstruction_time:.2f} seconds

## Video Information
- **Resolution**: {results.original_resolution}
- **Frame Rate**: {results.original_fps:.2f} fps
- **Total Frames**: {results.total_frames}
- **Duration**: {results.duration_seconds:.2f} seconds

## Performance Summary
The PointStream pipeline achieved a **{results.compression_ratio:.2f}:1** compression ratio 
with **{results.space_savings_percent:.1f}%** space savings while maintaining a PSNR of 
**{results.avg_psnr:.2f} dB** and SSIM of **{results.avg_ssim:.4f}**.
"""
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Evaluation report saved to: {output_path}")
    
    def generate_table(self, results: EvaluationResults) -> pd.DataFrame:
        """Generate publication-ready table."""
        data = {
            'Metric': [
                'Original Size (MB)',
                'Compressed Size (MB)', 
                'Compression Ratio',
                'Space Savings (%)',
                'Bits per Pixel',
                'PSNR (dB)',
                'SSIM',
                'LPIPS',
                'VMAF',
                'FVD',
                'Number of Scenes',
                'Number of Objects',
                'Processing Time (s)',
                'Reconstruction Time (s)'
            ],
            'Value': [
                f"{results.original_size_mb:.2f}",
                f"{results.compressed_size_mb:.2f}",
                f"{results.compression_ratio:.2f}:1",
                f"{results.space_savings_percent:.1f}%",
                f"{results.bits_per_pixel:.4f}",
                f"{results.avg_psnr:.2f}",
                f"{results.avg_ssim:.4f}",
                f"{results.avg_lpips:.4f}",
                f"{results.vmaf_score:.2f}",
                f"{results.fvd_score:.2f}",
                f"{results.num_scenes}",
                f"{results.num_objects}",
                f"{results.processing_time:.2f}",
                f"{results.reconstruction_time:.2f}"
            ]
        }
        
        return pd.DataFrame(data)
    
    def save_results(self, results: EvaluationResults, output_path: str):
        """Save results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(results.to_dict(), f, indent=2)
        logger.info(f"Results saved to: {output_path}")
    
    def generate_summary_table(self, results: EvaluationResults, output_path: str):
        """Generate CSV summary table."""
        df = self.generate_table(results)
        df.to_csv(output_path, index=False)
        logger.info(f"Summary table saved to: {output_path}")
    
    def generate_latex_table(self, results: EvaluationResults, output_path: str):
        """Generate LaTeX table for publication."""
        df = self.generate_table(results)
        latex_content = df.to_latex(index=False, escape=False, caption="PointStream Pipeline Evaluation Results", label="tab:pointstream_results")
        
        with open(output_path, 'w') as f:
            f.write(latex_content)
        logger.info(f"LaTeX table saved to: {output_path}")
    
    def generate_markdown_report(self, results: EvaluationResults, output_path: str):
        """Generate markdown evaluation report."""
        self.generate_report(results, output_path)
