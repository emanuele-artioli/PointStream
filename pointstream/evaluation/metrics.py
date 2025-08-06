"""
Video Quality and Compression Metrics Calculator

Provides comprehensive quality assessment including PSNR, SSIM, VMAF, LPIPS, and FVD.
"""

import cv2
import numpy as np
import torch
import torch.hub
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import subprocess
import tempfile
import json
import urllib.request
import scipy.linalg
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class CompressionMetrics:
    """Calculate compression-related metrics."""
    
    @staticmethod
    def calculate_compression_ratio(original_size: int, compressed_size: int) -> float:
        """Calculate compression ratio."""
        return original_size / compressed_size if compressed_size > 0 else float('inf')
    
    @staticmethod
    def calculate_space_savings(original_size: int, compressed_size: int) -> float:
        """Calculate space savings percentage."""
        return ((original_size - compressed_size) / original_size) * 100 if original_size > 0 else 0.0
    
    @staticmethod
    def calculate_bits_per_pixel(file_size_bytes: int, width: int, height: int, frames: int) -> float:
        """Calculate bits per pixel."""
        total_pixels = width * height * frames
        total_bits = file_size_bytes * 8
        return total_bits / total_pixels if total_pixels > 0 else 0.0


class QualityMetrics:
    """Calculate video quality metrics."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._init_lpips()
        
    def _init_lpips(self):
        """Initialize LPIPS model."""
        try:
            import lpips
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
            self.lpips_available = True
        except ImportError:
            logger.warning("LPIPS not available. Install with: pip install lpips")
            self.lpips_available = False
    
    def calculate_psnr(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate Peak Signal-to-Noise Ratio."""
        try:
            return psnr(original, reconstructed, data_range=255)
        except Exception as e:
            logger.error(f"Error calculating PSNR: {e}")
            return 0.0
    
    def calculate_ssim(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate Structural Similarity Index."""
        try:
            if len(original.shape) == 3:  # Color image
                ssim_values = []
                for i in range(original.shape[2]):
                    s = ssim(original[:, :, i], reconstructed[:, :, i], data_range=255)
                    ssim_values.append(s)
                return np.mean(ssim_values)
            else:  # Grayscale
                return ssim(original, reconstructed, data_range=255)
        except Exception as e:
            logger.error(f"Error calculating SSIM: {e}")
            return 0.0
    
    def calculate_lpips(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculate Learned Perceptual Image Patch Similarity."""
        if not self.lpips_available:
            return -1.0
        
        try:
            # Convert to tensors and normalize to [-1, 1]
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            if len(original.shape) == 2:  # Grayscale
                original = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB)
                reconstructed = cv2.cvtColor(reconstructed, cv2.COLOR_GRAY2RGB)
            
            original_tensor = transform(original).unsqueeze(0).to(self.device)
            reconstructed_tensor = transform(reconstructed).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                lpips_score = self.lpips_model(original_tensor, reconstructed_tensor)
            
            return lpips_score.item()
        except Exception as e:
            logger.error(f"Error calculating LPIPS: {e}")
            return -1.0
    
    def calculate_vmaf(self, original_video: str, reconstructed_video: str, model: str = "vmaf_v0.6.1") -> float:
        """Calculate VMAF score using dedicated vmaf tool, with FFmpeg fallback."""
        # First try the dedicated vmaf tool (more reliable)
        vmaf_score = self._calculate_vmaf_tool(original_video, reconstructed_video)
        
        if vmaf_score >= 0:
            return vmaf_score
        
        # Fallback to FFmpeg VMAF
        vmaf_score = self._calculate_vmaf_ffmpeg(original_video, reconstructed_video, model)
        
        if vmaf_score >= 0:
            return vmaf_score
        
        # Final fallback to frame-based quality estimation
        logger.info("VMAF unavailable, using frame-based quality estimation")
        return self._calculate_frame_quality_estimate(original_video, reconstructed_video)
    
    def _calculate_vmaf_tool(self, original_video: str, reconstructed_video: str) -> float:
        """Calculate VMAF using dedicated vmaf command-line tool."""
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                ref_y4m_path = Path(tmpdir) / 'reference.y4m'
                dist_y4m_path = Path(tmpdir) / 'distorted.y4m'
                log_file_path = Path(tmpdir) / 'vmaf_log.json'
                
                # Get frame counts and synchronize
                ref_info = self._get_video_info_ffprobe(original_video)
                dist_info = self._get_video_info_ffprobe(reconstructed_video)
                min_frames = min(ref_info["frame_count"], dist_info["frame_count"])
                
                logger.info(f"Synchronizing videos to {min_frames} frames for VMAF comparison")
                
                # Convert to Y4M format
                self._convert_to_y4m(original_video, str(ref_y4m_path), min_frames)
                self._convert_to_y4m(reconstructed_video, str(dist_y4m_path), min_frames)
                
                # Run VMAF tool
                vmaf_cmd = [
                    'vmaf', 
                    '--reference', str(ref_y4m_path),
                    '--distorted', str(dist_y4m_path),
                    '--output', str(log_file_path),
                    '--json',
                    '--threads', str(4)
                ]
                
                result = subprocess.run(vmaf_cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    logger.warning(f"VMAF tool failed: {result.stderr}")
                    return -1.0
                
                # Parse results
                with open(log_file_path, 'r') as f:
                    vmaf_data = json.load(f)
                
                return float(vmaf_data['pooled_metrics']['vmaf']['mean'])
                
        except Exception as e:
            logger.warning(f"Error with VMAF tool: {e}")
            return -1.0
    
    def _get_video_info_ffprobe(self, video_path: str) -> Dict[str, Any]:
        """Get video info using ffprobe."""
        cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate,nb_frames',
            '-of', 'json', video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        info = json.loads(result.stdout)['streams'][0]
        
        frame_count_str = info.get('nb_frames', '0')
        if not frame_count_str.isdigit() or int(frame_count_str) == 0:
            count_cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-count_frames', '-show_entries', 'stream=nb_read_frames',
                '-of', 'csv=p=0', video_path
            ]
            result = subprocess.run(count_cmd, capture_output=True, text=True, check=True)
            frame_count = int(result.stdout.strip())
        else:
            frame_count = int(frame_count_str)
        
        return {
            "width": int(info['width']),
            "height": int(info['height']),
            "frame_rate": eval(info['r_frame_rate']),
            "frame_count": frame_count
        }
    
    def _convert_to_y4m(self, input_path: str, output_path: str, frame_limit: int):
        """Convert video to Y4M format."""
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-vframes', str(frame_limit),
            '-pix_fmt', 'yuv420p',
            '-f', 'yuv4mpegpipe', output_path
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    
    def _calculate_vmaf_ffmpeg(self, original_video: str, reconstructed_video: str, model: str) -> float:
        """Try to calculate VMAF using FFmpeg."""
        try:
            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as tmp_file:
                vmaf_log = tmp_file.name
            
            cmd = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error',
                '-i', reconstructed_video,
                '-i', original_video,
                '-lavfi', f'[0:v][1:v]libvmaf=model={model}:log_path={vmaf_log}:log_fmt=json',
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.warning(f"FFmpeg VMAF calculation failed: {result.stderr}")
                return -1.0
            
            # Parse VMAF log
            with open(vmaf_log, 'r') as f:
                vmaf_data = json.load(f)
            
            # Clean up temporary file
            Path(vmaf_log).unlink()
            
            # Extract mean VMAF score
            frames = vmaf_data.get('frames', [])
            if frames:
                vmaf_scores = [frame['metrics']['vmaf'] for frame in frames if 'vmaf' in frame['metrics']]
                return np.mean(vmaf_scores) if vmaf_scores else -1.0
            
            return -1.0
            
        except Exception as e:
            logger.warning(f"Error calculating VMAF with FFmpeg: {e}")
            return -1.0
    
    def _calculate_frame_quality_estimate(self, original_video: str, reconstructed_video: str) -> float:
        """Calculate a VMAF-like quality estimate using frame comparison."""
        try:
            # Extract frames from both videos
            extractor = self._get_video_frame_extractor()
            orig_frames = extractor.extract_frames(original_video, max_frames=30)
            recon_frames = extractor.extract_frames(reconstructed_video, max_frames=30)
            
            if not orig_frames or not recon_frames:
                return -1.0
            
            quality_scores = []
            min_frames = min(len(orig_frames), len(recon_frames))
            
            for i in range(min_frames):
                orig_frame = orig_frames[i].astype(np.float32)
                recon_frame = recon_frames[i % len(recon_frames)]
                
                # Resize if needed
                if orig_frame.shape != recon_frame.shape:
                    recon_frame = cv2.resize(recon_frame, (orig_frame.shape[1], orig_frame.shape[0]))
                
                recon_frame = recon_frame.astype(np.float32)
                
                # Calculate MSE
                mse = np.mean((orig_frame - recon_frame) ** 2)
                
                # Convert MSE to PSNR-like score
                if mse > 0:
                    psnr = 20 * np.log10(255.0 / np.sqrt(mse))
                    # Convert PSNR to VMAF-like scale (0-100)
                    # Rough mapping: PSNR 20-40 dB -> VMAF 0-100
                    vmaf_like = max(0, min(100, (psnr - 20) * 5))
                    quality_scores.append(vmaf_like)
                else:
                    quality_scores.append(100.0)  # Perfect match
            
            return np.mean(quality_scores) if quality_scores else -1.0
            
        except Exception as e:
            logger.warning(f"Error calculating frame quality estimate: {e}")
            return -1.0
    
    def _get_video_frame_extractor(self):
        """Get VideoFrameExtractor instance (defined below)."""
        return VideoFrameExtractor()
    
    def calculate_fvd(self, original_frames: List[np.ndarray], reconstructed_frames: List[np.ndarray]) -> float:
        """Calculate Frechet Video Distance using I3D features."""
        try:
            if not original_frames or not reconstructed_frames:
                logger.warning("No frames available for FVD calculation")
                return -1.0
            
            if len(original_frames) < 2 or len(reconstructed_frames) < 2:
                logger.warning("Not enough frames for FVD calculation (need at least 2)")
                return -1.0
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info("Loading I3D model for FVD...")
            model = self._get_fvd_model(device)
            
            # Convert frames to tensors and resize to 224x224 (I3D input size)
            ref_tensor = self._prepare_frames_for_fvd(original_frames, device)
            dist_tensor = self._prepare_frames_for_fvd(reconstructed_frames, device)
            
            # Extract I3D features
            logger.info("Extracting I3D features for reference video...")
            real_features = self._extract_i3d_features(ref_tensor, model, device)
            
            logger.info("Extracting I3D features for reconstructed video...")
            fake_features = self._extract_i3d_features(dist_tensor, model, device)
            
            # Calculate FVD using Frechet distance
            mu_real = np.mean(real_features, axis=0)
            mu_fake = np.mean(fake_features, axis=0)
            
            # Handle case where we have too few samples for covariance
            if real_features.shape[0] < 2 or fake_features.shape[0] < 2:
                # Fallback to simple MSE
                return float(np.mean((mu_real - mu_fake) ** 2))
            
            sigma_real = np.cov(real_features, rowvar=False)
            sigma_fake = np.cov(fake_features, rowvar=False)
            
            # Ensure matrices are 2D
            if sigma_real.ndim == 0:
                sigma_real = np.array([[sigma_real]])
            if sigma_fake.ndim == 0:
                sigma_fake = np.array([[sigma_fake]])
            
            m = np.square(mu_fake - mu_real).sum()
            s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False)
            fid = np.real(m + np.trace(sigma_fake + sigma_real - 2 * s))
            
            return float(fid)
            
        except Exception as e:
            logger.error(f"Error calculating FVD: {e}")
            return -1.0
    
    def _get_fvd_model(self, device):
        """Downloads and loads the I3D model for FVD calculation."""
        detector_url = 'https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1'
        
        try:
            with urllib.request.urlopen(detector_url) as f:
                model = torch.jit.load(f).to(device)
        except urllib.error.URLError as e:
            # Fallback to cached version if download fails
            cached_file = Path(torch.hub.get_dir()) / 'checkpoints' / 'i3d_torchscript.pt'
            if cached_file.exists():
                logger.warning(f"Download failed ({e}), using cached model.")
                model = torch.jit.load(str(cached_file)).to(device)
            else:
                raise e
        
        model.eval()
        return model
    
    def _prepare_frames_for_fvd(self, frames: List[np.ndarray], device) -> torch.Tensor:
        """Prepare frames for FVD calculation (resize to 224x224)."""
        resized_frames = []
        for frame in frames:
            # Resize to 224x224 for I3D
            resized = cv2.resize(frame, (224, 224))
            resized_frames.append(resized)
        
        # Convert to tensor (T, H, W, C)
        return torch.from_numpy(np.stack(resized_frames))
    
    def _extract_i3d_features(self, video_frames_tensor, model, device):
        """Extract I3D features from video frames."""
        # video_frames_tensor is (T, H, W, C)
        # I3D model expects (B, C, T, H, W) and values in [-1, 1]
        
        video_tensor = video_frames_tensor.to(device)
        # Permute to (T, C, H, W), scale to [-1, 1]
        video_tensor = video_tensor.permute(0, 3, 1, 2).float() / 127.5 - 1.0

        features = []
        with torch.no_grad():
            # Process in chunks of 16 frames
            for i in tqdm(range(0, video_tensor.size(0), 16), desc="Extracting FVD features"):
                batch = video_tensor[i:i+16]
                
                # The model requires T > 1. If last batch is small, skip.
                if batch.size(0) > 1:
                    # Add batch dimension and permute to (B, C, T, H, W)
                    batch = batch.permute(1, 0, 2, 3).unsqueeze(0)
                    features.append(model(batch).cpu().numpy())

        if not features:
            raise ValueError("No valid feature batches extracted")
        
        return np.concatenate(features)


class VideoFrameExtractor:
    """Extract frames from video files for quality assessment."""
    
    @staticmethod
    def extract_frames(video_path: str, max_frames: int = 50) -> List[np.ndarray]:
        """Extract frames from video file."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            logger.error(f"Could not read video: {video_path}")
            return frames
        
        # Sample frames evenly
        frame_indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        cap.release()
        return frames
    
    @staticmethod
    def get_video_info(video_path: str) -> Dict[str, Any]:
        """Get video information."""
        cap = cv2.VideoCapture(video_path)
        
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        
        cap.release()
        return info


# Convenience functions for easy access to metrics
def calculate_psnr(original_video: str, reconstructed_video: str) -> Optional[float]:
    """Calculate PSNR between two videos."""
    try:
        metrics = QualityMetrics()
        extractor = VideoFrameExtractor()
        
        original_frames = extractor.extract_frames(original_video)
        reconstructed_frames = extractor.extract_frames(reconstructed_video)
        
        if not original_frames or not reconstructed_frames:
            return None
        
        psnr_values = []
        for orig, recon in zip(original_frames, reconstructed_frames):
            psnr_val = metrics.calculate_psnr(orig, recon)
            if psnr_val is not None:
                psnr_values.append(psnr_val)
        
        return np.mean(psnr_values) if psnr_values else None
    except Exception as e:
        logger.error(f"Error calculating PSNR: {e}")
        return None


def calculate_ssim(original_video: str, reconstructed_video: str) -> Optional[float]:
    """Calculate SSIM between two videos."""
    try:
        metrics = QualityMetrics()
        extractor = VideoFrameExtractor()
        
        original_frames = extractor.extract_frames(original_video)
        reconstructed_frames = extractor.extract_frames(reconstructed_video)
        
        if not original_frames or not reconstructed_frames:
            return None
        
        ssim_values = []
        for orig, recon in zip(original_frames, reconstructed_frames):
            ssim_val = metrics.calculate_ssim(orig, recon)
            if ssim_val is not None:
                ssim_values.append(ssim_val)
        
        return np.mean(ssim_values) if ssim_values else None
    except Exception as e:
        logger.error(f"Error calculating SSIM: {e}")
        return None


def calculate_lpips(original_video: str, reconstructed_video: str) -> Optional[float]:
    """Calculate LPIPS between two videos."""
    try:
        metrics = QualityMetrics()
        extractor = VideoFrameExtractor()
        
        original_frames = extractor.extract_frames(original_video)
        reconstructed_frames = extractor.extract_frames(reconstructed_video)
        
        if not original_frames or not reconstructed_frames:
            return None
        
        lpips_values = []
        for orig, recon in zip(original_frames, reconstructed_frames):
            lpips_val = metrics.calculate_lpips(orig, recon)
            if lpips_val is not None:
                lpips_values.append(lpips_val)
        
        return np.mean(lpips_values) if lpips_values else None
    except Exception as e:
        logger.error(f"Error calculating LPIPS: {e}")
        return None


def calculate_vmaf(original_video: str, reconstructed_video: str) -> Optional[float]:
    """Calculate VMAF between two videos."""
    try:
        metrics = QualityMetrics()
        return metrics.calculate_vmaf(original_video, reconstructed_video)
    except Exception as e:
        logger.error(f"Error calculating VMAF: {e}")
        return None


def calculate_fvd(original_video: str, reconstructed_video: str) -> Optional[float]:
    """Calculate FVD between two videos."""
    try:
        metrics = QualityMetrics()
        extractor = VideoFrameExtractor()
        
        original_frames = extractor.extract_frames(original_video)
        reconstructed_frames = extractor.extract_frames(reconstructed_video)
        
        if not original_frames or not reconstructed_frames:
            return None
        
        return metrics.calculate_fvd(original_frames, reconstructed_frames)
    except Exception as e:
        logger.error(f"Error calculating FVD: {e}")
        return None


def calculate_compression_ratio(original_file: str, compressed_file: str) -> Optional[float]:
    """Calculate compression ratio between two files."""
    try:
        import os
        original_size = os.path.getsize(original_file)
        compressed_size = os.path.getsize(compressed_file)
        
        metrics = CompressionMetrics()
        return metrics.calculate_compression_ratio(original_size, compressed_size)
    except Exception as e:
        logger.error(f"Error calculating compression ratio: {e}")
        return None
