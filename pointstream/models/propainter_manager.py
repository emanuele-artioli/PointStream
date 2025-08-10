"""
ProPainter integration with frame-level fallback to OpenCV.
"""
import os
import sys
import cv2
import numpy as np
import torch
import subprocess
import tempfile
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .. import config
from ..utils.adaptive_metrics import calculate_mask_complexity_score, get_metrics_cache


class ProPainterManager:
    """Manages ProPainter with intelligent fallback to OpenCV."""
    
    def __init__(self):
        self.propainter_path = Path(config.PROPAINTER_PATH)
        self.model_loaded = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics_cache = get_metrics_cache()
        
    def is_available(self) -> bool:
        """Check if ProPainter is available."""
        if not config.ENABLE_PROPAINTER:
            return False
        
        inference_script = self.propainter_path / "inference_propainter.py"
        return self.propainter_path.exists() and inference_script.exists()
    
    def inpaint_scene_chunk(self, frames: List[np.ndarray], 
                           frame_masks: List[np.ndarray],
                           content_type: str = "general") -> List[np.ndarray]:
        """
        Inpaint a scene chunk with ProPainter and frame-level OpenCV fallback.
        
        Args:
            frames: List of video frames
            frame_masks: List of binary masks for each frame
            content_type: Content type for threshold learning
            
        Returns:
            List of inpainted frames
        """
        if not frames or not frame_masks:
            return frames
            
        print(f"  -> Inpainting chunk of {len(frames)} frames...")
        
        # Step 1: Analyze mask complexity for the chunk
        chunk_complexity = np.mean([
            calculate_mask_complexity_score(mask) for mask in frame_masks
        ])
        print(f"     -> Chunk mask complexity: {chunk_complexity:.3f}")
        
        # Step 2: Decide whether to try ProPainter for the chunk
        use_propainter = (
            self.is_available() and 
            len(frames) >= config.MIN_FRAMES_FOR_PROPAINTER and
            chunk_complexity > config.PROPAINTER_COMPLEXITY_THRESHOLD
        )
        
        if use_propainter:
            print("     -> Attempting ProPainter for temporal coherence...")
            try:
                # Try ProPainter on the entire chunk
                inpainted_frames = self._inpaint_with_propainter(frames, frame_masks)
                
                # Validate results and fallback individual frames if needed
                final_frames = self._validate_and_fallback_frames(
                    frames, frame_masks, inpainted_frames, content_type
                )
                
                return final_frames
                
            except Exception as e:
                print(f"     -> ProPainter failed for chunk ({e}), falling back to OpenCV")
                return self._inpaint_all_with_opencv(frames, frame_masks, content_type)
        else:
            print("     -> Using OpenCV for chunk (low complexity or ProPainter unavailable)")
            return self._inpaint_all_with_opencv(frames, frame_masks, content_type)
    
    def _validate_and_fallback_frames(self, original_frames: List[np.ndarray], 
                                     masks: List[np.ndarray], 
                                     propainter_frames: List[np.ndarray],
                                     content_type: str) -> List[np.ndarray]:
        """Validate ProPainter results and fallback individual bad frames to OpenCV."""
        print("     -> Validating ProPainter results...")
        
        final_frames = []
        opencv_fallbacks = 0
        
        for i, (orig, mask, pp_frame) in enumerate(zip(original_frames, masks, propainter_frames)):
            # Simple quality check: look for artifacts or incomplete inpainting
            if self._is_frame_quality_acceptable(orig, mask, pp_frame):
                final_frames.append(pp_frame)
            else:
                # Fallback this specific frame to OpenCV
                print(f"        -> Frame {i}: ProPainter quality poor, using OpenCV fallback")
                opencv_frame = self._opencv_inpaint(orig, mask)
                final_frames.append(opencv_frame)
                opencv_fallbacks += 1
        
        print(f"     -> ProPainter succeeded: {len(final_frames) - opencv_fallbacks}/{len(final_frames)} frames")
        if opencv_fallbacks > 0:
            print(f"     -> OpenCV fallbacks: {opencv_fallbacks} frames")
        
        return final_frames
    
    def _is_frame_quality_acceptable(self, original: np.ndarray, 
                                   mask: np.ndarray, 
                                   inpainted: np.ndarray) -> bool:
        """Simple quality check for inpainted frame."""
        try:
            # Check if inpainted region has reasonable variation
            mask_area = mask > 0
            if np.sum(mask_area) == 0:
                return True  # No mask, nothing to check
            
            inpainted_region = inpainted[mask_area]
            
            # Check for reasonable color variation (not all black/white/same color)
            std_dev = np.std(inpainted_region)
            if std_dev < 5:  # Very low variation suggests poor inpainting
                return False
            
            # Check for extreme values
            if np.mean(inpainted_region) < 10 or np.mean(inpainted_region) > 245:
                return False
            
            return True
            
        except Exception:
            return False  # If check fails, assume poor quality
    
    def _inpaint_all_with_opencv(self, frames: List[np.ndarray], 
                                frame_masks: List[np.ndarray], 
                                content_type: str) -> List[np.ndarray]:
        """Inpaint all frames with OpenCV."""
        print("     -> Using OpenCV for all frames...")
        
        inpainted_frames = []
        for i, (frame, mask) in enumerate(zip(frames, frame_masks)):
            inpainted_frame = self._opencv_inpaint(frame, mask)
            inpainted_frames.append(inpainted_frame)
        
        return inpainted_frames
    
    def _propainter_single_frame(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Use ProPainter for a single frame with temporal context."""
        # For single frame, we create a short sequence for temporal consistency
        # Duplicate frame 3 times to give ProPainter temporal context
        temp_frames = [frame.copy() for _ in range(3)]
        temp_masks = [mask.copy() for _ in range(3)]
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            
            # Save as video files
            video_path = tmp_path / "input.mp4"
            mask_path = tmp_path / "mask.mp4" 
            output_path = tmp_path / "output.mp4"
            
            self._save_frames_as_video(temp_frames, str(video_path))
            self._save_masks_as_video(temp_masks, str(mask_path))
            
            # Run ProPainter
            success = self._run_propainter_inference(video_path, mask_path, output_path)
            
            if success and output_path.exists():
                # Extract middle frame (index 1) for best quality
                return self._extract_frame_from_video(str(output_path), frame_index=1)
            else:
                raise RuntimeError("ProPainter inference failed")
    
    def _opencv_inpaint(self, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Fallback OpenCV inpainting."""
        if mask is None or np.sum(mask) == 0:
            return frame.copy()
        
        # Ensure mask is single channel uint8
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = mask.astype(np.uint8)
        
        # Use TELEA algorithm for better quality than NS
        return cv2.inpaint(frame, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    
    def _save_frames_as_video(self, frames: List[np.ndarray], output_path: str):
        """Save frames as MP4 video."""
        if not frames:
            return
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
        
        for frame in frames:
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            out.write(frame)
        out.release()
    
    def _save_masks_as_video(self, masks: List[np.ndarray], output_path: str):
        """Save masks as MP4 video."""
        if not masks:
            return
        
        height, width = masks[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height), isColor=False)
        
        for mask in masks:
            if len(mask.shape) == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            out.write(mask)
        out.release()
    
    def _run_propainter_inference(self, video_path: Path, mask_path: Path, 
                                 output_path: Path) -> bool:
        """Run ProPainter inference with timeout."""
        try:
            cmd = [
                "python3", str(self.propainter_path / "inference_propainter.py"),
                "--video", str(video_path),
                "--mask", str(mask_path),
                "--output", str(output_path.parent),
                "--save_fps", "30",
                "--fp16",  # Use half precision for speed
                "--fast"   # Enable fast mode if available
            ]
            
            result = subprocess.run(
                cmd,
                cwd=str(self.propainter_path),
                capture_output=True,
                text=True,
                timeout=config.PROPAINTER_TIMEOUT
            )
            
            return result.returncode == 0
            
        except subprocess.TimeoutExpired:
            logging.warning(f"ProPainter timeout after {config.PROPAINTER_TIMEOUT}s")
            return False
        except Exception as e:
            logging.warning(f"ProPainter error: {e}")
            return False
    
    def _extract_frame_from_video(self, video_path: str, frame_index: int = 0) -> np.ndarray:
        """Extract specific frame from video."""
        cap = cv2.VideoCapture(video_path)
        
        # Seek to specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return frame
        else:
            raise ValueError(f"Could not extract frame {frame_index} from video")


# Global ProPainter manager instance
_propainter_manager = None

def get_propainter_manager() -> ProPainterManager:
    """Get global ProPainter manager instance."""
    global _propainter_manager
    if _propainter_manager is None:
        _propainter_manager = ProPainterManager()
    return _propainter_manager
