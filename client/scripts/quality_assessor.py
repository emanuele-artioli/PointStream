#!/usr/bin/env python3
"""
Quality Assessor

This module assesses the quality of reconstructed videos compared to original videos.
Provides metrics like SSIM, PSNR, LPIPS, and optionally VMAF.
"""

import logging
import time
import numpy as np
import cv2
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import subprocess

try:
    from ...utils.decorators import track_performance
    from ...utils import config
    # Optional imports for quality metrics
    try:
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr
        SKIMAGE_AVAILABLE = True
    except ImportError:
        SKIMAGE_AVAILABLE = False
        logging.warning("scikit-image not available, using basic quality metrics")
    
    try:
        import lpips
        LPIPS_AVAILABLE = True
    except ImportError:
        LPIPS_AVAILABLE = False
        logging.warning("LPIPS not available, perceptual quality metrics disabled")
        
except ImportError as e:
    logging.error(f"Failed to import PointStream utilities: {e}")
    raise


class QualityAssessor:
    """
    Assesses quality of reconstructed videos against original videos.
    
    This component provides various quality metrics to evaluate
    the effectiveness of the reconstruction pipeline.
    """
    
    def __init__(self):
        """Initialize the quality assessor."""
        # Configuration
        self.enable_ssim = config.get_bool('video_reconstruction', 'compute_ssim', True)
        self.enable_psnr = config.get_bool('video_reconstruction', 'compute_psnr', True)
        self.enable_lpips = config.get_bool('video_reconstruction', 'compute_lpips', True)
        self.enable_vmaf = config.get_bool('video_reconstruction', 'compute_vmaf', False)
        
        # Initialize LPIPS model if available
        self.lpips_model = None
        if LPIPS_AVAILABLE and self.enable_lpips:
            try:
                self.lpips_model = lpips.LPIPS(net='alex')  # or 'vgg'
                logging.info("   📊 LPIPS model loaded")
            except Exception as e:
                logging.warning(f"Failed to load LPIPS model: {e}")
                self.lpips_model = None
        
        logging.info("📊 Quality Assessor initialized")
        logging.info(f"   SSIM: {self.enable_ssim and SKIMAGE_AVAILABLE}")
        logging.info(f"   PSNR: {self.enable_psnr and SKIMAGE_AVAILABLE}")
        logging.info(f"   LPIPS: {self.enable_lpips and self.lpips_model is not None}")
        logging.info(f"   VMAF: {self.enable_vmaf}")
    
    @track_performance
    def assess_video_quality(self, original_video: str, 
                           reconstructed_video: str) -> Dict[str, Any]:
        """
        Assess quality of reconstructed video against original.
        
        Args:
            original_video: Path to original video
            reconstructed_video: Path to reconstructed video
            
        Returns:
            Quality assessment results
        """
        start_time = time.time()
        
        logging.info(f"📊 Assessing video quality:")
        logging.info(f"   Original: {original_video}")
        logging.info(f"   Reconstructed: {reconstructed_video}")
        
        # Load videos
        original_frames = self._load_video_frames(original_video)
        reconstructed_frames = self._load_video_frames(reconstructed_video)
        
        if not original_frames or not reconstructed_frames:
            return {
                'error': 'Failed to load video frames',
                'processing_time': time.time() - start_time
            }
        
        # Align frame counts (use minimum)
        min_frames = min(len(original_frames), len(reconstructed_frames))
        original_frames = original_frames[:min_frames]
        reconstructed_frames = reconstructed_frames[:min_frames]
        
        logging.info(f"📺 Comparing {min_frames} frames")
        
        # Compute quality metrics
        quality_results = {}
        
        # Frame-by-frame SSIM and PSNR
        if (self.enable_ssim or self.enable_psnr) and SKIMAGE_AVAILABLE:
            ssim_scores, psnr_scores = self._compute_frame_metrics(
                original_frames, reconstructed_frames
            )
            
            if self.enable_ssim and ssim_scores:
                quality_results['ssim'] = {
                    'mean': np.mean(ssim_scores),
                    'std': np.std(ssim_scores),
                    'min': np.min(ssim_scores),
                    'max': np.max(ssim_scores),
                    'scores': ssim_scores
                }
            
            if self.enable_psnr and psnr_scores:
                quality_results['psnr'] = {
                    'mean': np.mean(psnr_scores),
                    'std': np.std(psnr_scores),
                    'min': np.min(psnr_scores),
                    'max': np.max(psnr_scores),
                    'scores': psnr_scores
                }
        
        # LPIPS (perceptual distance)
        if self.enable_lpips and self.lpips_model is not None:
            lpips_scores = self._compute_lpips_scores(original_frames, reconstructed_frames)
            if lpips_scores:
                quality_results['lpips'] = {
                    'mean': np.mean(lpips_scores),
                    'std': np.std(lpips_scores),
                    'min': np.min(lpips_scores),
                    'max': np.max(lpips_scores),
                    'scores': lpips_scores
                }
        
        # VMAF (if enabled and available)
        if self.enable_vmaf:
            vmaf_score = self._compute_vmaf_score(original_video, reconstructed_video)
            if vmaf_score is not None:
                quality_results['vmaf'] = vmaf_score
        
        # Basic pixel difference metrics
        mse_score, mae_score = self._compute_pixel_difference_metrics(
            original_frames, reconstructed_frames
        )
        quality_results['mse'] = mse_score
        quality_results['mae'] = mae_score
        
        processing_time = time.time() - start_time
        
        result = {
            'quality_metrics': quality_results,
            'frame_count': min_frames,
            'processing_time': processing_time,
            'original_video': original_video,
            'reconstructed_video': reconstructed_video
        }
        
        # Log summary
        self._log_quality_summary(quality_results)
        
        logging.info(f"✅ Quality assessment completed in {processing_time:.2f}s")
        return result
    
    def _load_video_frames(self, video_path: str) -> List[np.ndarray]:
        """Load frames from video file."""
        if not Path(video_path).exists():
            logging.error(f"Video file not found: {video_path}")
            return []
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
        finally:
            cap.release()
        
        logging.info(f"📽️  Loaded {len(frames)} frames from {Path(video_path).name}")
        return frames
    
    def _compute_frame_metrics(self, original_frames: List[np.ndarray], 
                             reconstructed_frames: List[np.ndarray]) -> Tuple[List[float], List[float]]:
        """Compute SSIM and PSNR for each frame pair."""
        ssim_scores = []
        psnr_scores = []
        
        for i, (orig, recon) in enumerate(zip(original_frames, reconstructed_frames)):
            try:
                # Ensure same size
                if orig.shape != recon.shape:
                    recon = cv2.resize(recon, (orig.shape[1], orig.shape[0]))
                
                # Convert to grayscale for SSIM/PSNR
                orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
                recon_gray = cv2.cvtColor(recon, cv2.COLOR_BGR2GRAY)
                
                # SSIM
                if self.enable_ssim:
                    ssim_score = ssim(orig_gray, recon_gray, data_range=255)
                    ssim_scores.append(ssim_score)
                
                # PSNR
                if self.enable_psnr:
                    psnr_score = psnr(orig_gray, recon_gray, data_range=255)
                    psnr_scores.append(psnr_score)
                    
            except Exception as e:
                logging.warning(f"Failed to compute metrics for frame {i}: {e}")
                continue
        
        return ssim_scores, psnr_scores
    
    def _compute_lpips_scores(self, original_frames: List[np.ndarray], 
                            reconstructed_frames: List[np.ndarray]) -> List[float]:
        """Compute LPIPS (perceptual distance) scores."""
        if self.lpips_model is None:
            return []
        
        import torch
        
        lpips_scores = []
        
        for i, (orig, recon) in enumerate(zip(original_frames, reconstructed_frames)):
            try:
                # Ensure same size
                if orig.shape != recon.shape:
                    recon = cv2.resize(recon, (orig.shape[1], orig.shape[0]))
                
                # Convert to tensors and normalize to [-1, 1]
                orig_tensor = torch.from_numpy(orig).float().permute(2, 0, 1) / 127.5 - 1.0
                recon_tensor = torch.from_numpy(recon).float().permute(2, 0, 1) / 127.5 - 1.0
                
                # Add batch dimension
                orig_tensor = orig_tensor.unsqueeze(0)
                recon_tensor = recon_tensor.unsqueeze(0)
                
                # Compute LPIPS
                with torch.no_grad():
                    lpips_score = self.lpips_model(orig_tensor, recon_tensor).item()
                    lpips_scores.append(lpips_score)
                    
            except Exception as e:
                logging.warning(f"Failed to compute LPIPS for frame {i}: {e}")
                continue
        
        return lpips_scores
    
    def _compute_vmaf_score(self, original_video: str, reconstructed_video: str) -> Optional[float]:
        """Compute VMAF score using external tool."""
        try:
            # This requires VMAF to be installed and accessible
            # You might need to install libvmaf and compile FFmpeg with VMAF support
            cmd = [
                'ffmpeg',
                '-i', reconstructed_video,
                '-i', original_video,
                '-lavfi', '[0:v][1:v]libvmaf=log_fmt=json:log_path=/tmp/vmaf_output.json',
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Parse VMAF result from JSON
                import json
                with open('/tmp/vmaf_output.json', 'r') as f:
                    vmaf_data = json.load(f)
                    return vmaf_data.get('pooled_metrics', {}).get('vmaf', {}).get('mean')
            
        except Exception as e:
            logging.warning(f"VMAF computation failed: {e}")
        
        return None
    
    def _compute_pixel_difference_metrics(self, original_frames: List[np.ndarray], 
                                        reconstructed_frames: List[np.ndarray]) -> Tuple[float, float]:
        """Compute basic pixel difference metrics (MSE, MAE)."""
        mse_scores = []
        mae_scores = []
        
        for orig, recon in zip(original_frames, reconstructed_frames):
            # Ensure same size
            if orig.shape != recon.shape:
                recon = cv2.resize(recon, (orig.shape[1], orig.shape[0]))
            
            # Compute MSE and MAE
            mse = np.mean((orig.astype(np.float32) - recon.astype(np.float32)) ** 2)
            mae = np.mean(np.abs(orig.astype(np.float32) - recon.astype(np.float32)))
            
            mse_scores.append(mse)
            mae_scores.append(mae)
        
        return np.mean(mse_scores), np.mean(mae_scores)
    
    def _log_quality_summary(self, quality_metrics: Dict[str, Any]):
        """Log quality assessment summary."""
        logging.info("📊 Quality Assessment Summary:")
        
        if 'ssim' in quality_metrics:
            ssim_mean = quality_metrics['ssim']['mean']
            logging.info(f"   SSIM: {ssim_mean:.4f} (higher is better, max=1.0)")
        
        if 'psnr' in quality_metrics:
            psnr_mean = quality_metrics['psnr']['mean']
            logging.info(f"   PSNR: {psnr_mean:.2f} dB (higher is better)")
        
        if 'lpips' in quality_metrics:
            lpips_mean = quality_metrics['lpips']['mean']
            logging.info(f"   LPIPS: {lpips_mean:.4f} (lower is better)")
        
        if 'vmaf' in quality_metrics:
            vmaf_score = quality_metrics['vmaf']
            logging.info(f"   VMAF: {vmaf_score:.2f} (higher is better, max=100)")
        
        mse = quality_metrics.get('mse', 0)
        mae = quality_metrics.get('mae', 0)
        logging.info(f"   MSE: {mse:.2f} (lower is better)")
        logging.info(f"   MAE: {mae:.2f} (lower is better)")
    
    def assess_frame_quality(self, original_frame: np.ndarray, 
                           reconstructed_frame: np.ndarray) -> Dict[str, float]:
        """
        Assess quality of a single frame pair.
        
        Args:
            original_frame: Original frame
            reconstructed_frame: Reconstructed frame
            
        Returns:
            Quality metrics for the frame pair
        """
        metrics = {}
        
        # Ensure same size
        if original_frame.shape != reconstructed_frame.shape:
            reconstructed_frame = cv2.resize(reconstructed_frame, 
                                           (original_frame.shape[1], original_frame.shape[0]))
        
        # SSIM and PSNR
        if SKIMAGE_AVAILABLE:
            orig_gray = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
            recon_gray = cv2.cvtColor(reconstructed_frame, cv2.COLOR_BGR2GRAY)
            
            if self.enable_ssim:
                metrics['ssim'] = ssim(orig_gray, recon_gray, data_range=255)
            
            if self.enable_psnr:
                metrics['psnr'] = psnr(orig_gray, recon_gray, data_range=255)
        
        # MSE and MAE
        mse = np.mean((original_frame.astype(np.float32) - reconstructed_frame.astype(np.float32)) ** 2)
        mae = np.mean(np.abs(original_frame.astype(np.float32) - reconstructed_frame.astype(np.float32)))
        
        metrics['mse'] = mse
        metrics['mae'] = mae
        
        return metrics
    
    def create_quality_report(self, assessment_results: List[Dict[str, Any]], 
                            output_path: Path) -> bool:
        """
        Create a comprehensive quality report.
        
        Args:
            assessment_results: List of quality assessment results
            output_path: Path to save report
            
        Returns:
            Success status
        """
        try:
            import json
            
            # Aggregate results
            report = {
                'summary': {
                    'total_assessments': len(assessment_results),
                    'successful_assessments': sum(1 for r in assessment_results if 'quality_metrics' in r)
                },
                'assessments': assessment_results,
                'generated_at': time.time()
            }
            
            # Calculate overall statistics
            all_ssim = []
            all_psnr = []
            all_lpips = []
            
            for result in assessment_results:
                metrics = result.get('quality_metrics', {})
                
                if 'ssim' in metrics:
                    all_ssim.append(metrics['ssim']['mean'])
                if 'psnr' in metrics:
                    all_psnr.append(metrics['psnr']['mean'])
                if 'lpips' in metrics:
                    all_lpips.append(metrics['lpips']['mean'])
            
            # Add aggregate statistics
            if all_ssim:
                report['summary']['overall_ssim'] = {
                    'mean': np.mean(all_ssim),
                    'std': np.std(all_ssim),
                    'min': np.min(all_ssim),
                    'max': np.max(all_ssim)
                }
            
            if all_psnr:
                report['summary']['overall_psnr'] = {
                    'mean': np.mean(all_psnr),
                    'std': np.std(all_psnr),
                    'min': np.min(all_psnr),
                    'max': np.max(all_psnr)
                }
            
            if all_lpips:
                report['summary']['overall_lpips'] = {
                    'mean': np.mean(all_lpips),
                    'std': np.std(all_lpips),
                    'min': np.min(all_lpips),
                    'max': np.max(all_lpips)
                }
            
            # Save report
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logging.info(f"📊 Quality report saved: {output_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to create quality report: {e}")
            return False
