#!/usr/bin/env python3
"""
Inpainting Component

This component handles background inpainting using multiple methods:
- Stable Diffusion 2 inpainting model (AI-based, high quality)
- OpenCV TELEA inpainting (traditional, fast)
- OpenCV Navier-Stokes inpainting (traditional, alternative)

The component can be configured to use either method as primary with automatic
fallback to the other method if the primary fails. This allows for optimization
between quality and performance while handling various input scenarios.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional, List, Tuple
from PIL import Image
from decorators import log_step, time_step
import config

try:
    import torch
    from diffusers import StableDiffusionInpaintPipeline
    DIFFUSERS_AVAILABLE = True
    logging.info("Diffusers available for AI inpainting")
except ImportError:
    DIFFUSERS_AVAILABLE = False
    logging.warning("Diffusers not available, using fallback inpainting")


class Inpainter:
    """Multi-method inpainting component for background restoration."""
    
    def __init__(self):
        """Initialize the inpainter with configuration."""
        # Primary inpainting method
        self.primary_method = config.get_str('inpainting', 'primary_method', 'stable_diffusion')
        
        # Model configuration
        self.model_name = config.get_str('inpainting', 'model_name', 'stabilityai/stable-diffusion-2-inpainting')
        self.device = config.get_str('inpainting', 'device', 'auto')
        
        # Generation parameters
        self.guidance_scale = config.get_float('inpainting', 'guidance_scale', 7.5)
        self.num_inference_steps = config.get_int('inpainting', 'num_inference_steps', 20)
        self.strength = config.get_float('inpainting', 'strength', 1.0)
        
        # Image processing parameters
        self.max_image_size = config.get_int('inpainting', 'max_image_size', 512)
        self.padding = config.get_int('inpainting', 'padding', 20)
        
        # Mask processing parameters
        self.dilate_mask = config.get_bool('inpainting', 'dilate_mask', True)
        self.dilate_kernel_size = config.get_int('inpainting', 'dilate_kernel_size', 5)
        self.blur_mask_edges = config.get_bool('inpainting', 'blur_mask_edges', True)
        self.blur_kernel_size = config.get_int('inpainting', 'blur_kernel_size', 3)
        
        # OpenCV inpainting parameters
        self.opencv_method = config.get_str('inpainting', 'opencv_method', 'telea')
        self.opencv_radius = config.get_int('inpainting', 'opencv_radius', 3)
        
        # Fallback parameters
        self.fallback_method = config.get_str('inpainting', 'fallback_method', 'telea')
        self.fallback_radius = config.get_int('inpainting', 'fallback_radius', 3)
        
        # Set device
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() and DIFFUSERS_AVAILABLE else 'cpu'
        
        # Initialize pipeline only if using stable diffusion
        self.pipeline = None
        if self.primary_method == 'stable_diffusion':
            self._initialize_pipeline()
        
        logging.info("Inpainter initialized")
        logging.info(f"Primary method: {self.primary_method}")
        if self.primary_method == 'stable_diffusion':
            logging.info(f"Model: {self.model_name}")
            logging.info(f"Device: {self.device}")
            logging.info(f"Max image size: {self.max_image_size}")
            logging.info(f"Guidance scale: {self.guidance_scale}")
            logging.info(f"Inference steps: {self.num_inference_steps}")
        else:
            logging.info(f"OpenCV method: {self.opencv_method}")
            logging.info(f"OpenCV radius: {self.opencv_radius}")
    
    def _initialize_pipeline(self):
        """Initialize the Stable Diffusion inpainting pipeline."""
        if not DIFFUSERS_AVAILABLE:
            logging.warning("Diffusers not available, using fallback methods only")
            return
        
        try:
            logging.info(f"Loading Stable Diffusion inpainting model: {self.model_name}")
            
            # Try to load with safetensors first, fall back to standard loading
            try:
                self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                    safety_checker=None,  # Disable safety checker for performance
                    requires_safety_checker=False,
                    use_safetensors=True  # Prefer safetensors when available
                )
                logging.info("Loaded model using safetensors")
            except Exception as safetensor_error:
                logging.info("Safetensors format not available, using standard format")
                logging.info("Falling back to standard loading...")
                self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                    safety_checker=None,  # Disable safety checker for performance
                    requires_safety_checker=False,
                    use_safetensors=False  # Use standard pytorch format
                )
                logging.info("Loaded model using standard pytorch format")
            
            # Move to device
            self.pipeline.to(self.device)
            
            # Enable memory efficient attention if available
            if hasattr(self.pipeline, 'enable_attention_slicing'):
                self.pipeline.enable_attention_slicing()
            
            # Enable CPU offload if needed
            if self.device == 'cuda' and hasattr(self.pipeline, 'enable_sequential_cpu_offload'):
                # Only enable if GPU memory is limited
                try:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    if gpu_memory < 8:  # Less than 8GB GPU memory
                        self.pipeline.enable_sequential_cpu_offload()
                        logging.info("Enabled CPU offload due to limited GPU memory")
                except:
                    pass
            
            logging.info("Stable Diffusion inpainting pipeline loaded successfully")
            
        except Exception as e:
            logging.error(f"Failed to load Stable Diffusion pipeline: {e}")
            self.pipeline = None
    
    @log_step
    @time_step(track_processing=True)
    def inpaint_background(self, panorama: np.ndarray, masks: List[np.ndarray], 
                          prompt: str) -> Dict[str, Any]:
        """
        Inpaint the background of a panorama using object masks and text prompt.
        
        Args:
            panorama: Panorama image as numpy array
            masks: List of object masks to inpaint
            prompt: Text prompt describing the desired background
            
        Returns:
            Dictionary containing:
            - inpainted_image: The inpainted panorama
            - method: Method used for inpainting
            - processing_info: Additional processing information
        """
        if panorama is None:
            return {
                'inpainted_image': None,
                'method': 'error',
                'success': False,
                'error': 'no_panorama'
            }
        
        logging.info(f"Inpainting background with {len(masks)} masks")
        logging.info(f"Prompt: '{prompt[:100]}...' ({len(prompt)} chars)")
        
        try:
            # Combine masks
            combined_mask = self._combine_masks(masks, panorama.shape[:2])
            
            if np.sum(combined_mask) == 0:
                logging.info("No mask pixels to inpaint, returning original panorama")
                return {
                    'inpainted_image': panorama.copy(),
                    'method': 'no_inpainting_needed',
                    'success': True,
                    'mask_pixels': 0
                }
            
            # Use primary method first, then fallback if it fails
            try:
                if self.primary_method == 'opencv':
                    result = self._inpaint_with_opencv(panorama, combined_mask)
                elif self.primary_method == 'stable_diffusion' and self.pipeline is not None:
                    result = self._inpaint_with_stable_diffusion(panorama, combined_mask, prompt)
                else:
                    # Fallback to OpenCV if stable diffusion is not available
                    result = self._inpaint_with_opencv(panorama, combined_mask)
                    
            except Exception as primary_error:
                logging.warning(f"Primary inpainting method ({self.primary_method}) failed: {primary_error}")
                logging.info("Attempting fallback method")
                
                # Try the alternative method as fallback
                try:
                    if self.primary_method == 'opencv':
                        # Primary was OpenCV, try stable diffusion as fallback
                        if self.pipeline is not None:
                            result = self._inpaint_with_stable_diffusion(panorama, combined_mask, prompt)
                        else:
                            # No stable diffusion available, use fallback OpenCV settings
                            result = self._inpaint_with_fallback_opencv(panorama, combined_mask)
                    else:
                        # Primary was stable diffusion, use OpenCV as fallback
                        result = self._inpaint_with_opencv(panorama, combined_mask)
                        
                except Exception as fallback_error:
                    logging.error(f"Fallback inpainting method also failed: {fallback_error}")
                    # Return original image as last resort
                    result = {
                        'inpainted_image': panorama.copy(),
                        'method': 'error_both_failed',
                        'success': False,
                        'primary_error': str(primary_error),
                        'fallback_error': str(fallback_error)
                    }
            
            return result
            
        except Exception as e:
            logging.error(f"Inpainting failed: {e}")
            return {
                'inpainted_image': panorama.copy(),  # Return original as fallback
                'method': 'error_fallback',
                'success': False,
                'error': str(e)
            }
    
    def _combine_masks(self, masks: List[np.ndarray], image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Combine multiple masks into a single mask.
        
        Args:
            masks: List of individual object masks
            image_shape: Shape of the target image (height, width)
            
        Returns:
            Combined binary mask
        """
        if not masks:
            return np.zeros(image_shape, dtype=np.uint8)
        
        combined_mask = np.zeros(image_shape, dtype=np.uint8)
        
        for mask in masks:
            if mask is not None and mask.shape[:2] == image_shape:
                # Ensure mask is binary
                binary_mask = (mask > 0).astype(np.uint8)
                combined_mask = np.logical_or(combined_mask, binary_mask).astype(np.uint8)
        
        # Post-process mask
        processed_mask = self._process_mask(combined_mask)
        
        return processed_mask
    
    def _get_mask_bounding_box(self, mask: np.ndarray) -> Optional[Dict[str, int]]:
        """
        Get the tight bounding box around all mask pixels with padding.
        
        Args:
            mask: Binary mask array
            
        Returns:
            Dictionary with bounding box coordinates or None if no mask pixels
        """
        # Find all non-zero pixels
        coords = np.where(mask > 0)
        
        if len(coords[0]) == 0:
            return None
        
        # Get min/max coordinates
        y_min, y_max = np.min(coords[0]), np.max(coords[0])
        x_min, x_max = np.min(coords[1]), np.max(coords[1])
        
        # Add padding
        h, w = mask.shape
        y_min = max(0, y_min - self.padding)
        y_max = min(h, y_max + self.padding)
        x_min = max(0, x_min - self.padding)
        x_max = min(w, x_max + self.padding)
        
        return {
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max,
            'width': x_max - x_min,
            'height': y_max - y_min
        }
    
    def _extract_crop(self, image: np.ndarray, mask: np.ndarray, 
                     crop_info: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        """
        Extract cropped regions from image and mask.
        
        Args:
            image: Source image
            mask: Source mask
            crop_info: Bounding box information
            
        Returns:
            Tuple of (cropped_image, cropped_mask, crop_coordinates)
        """
        x_min, y_min = crop_info['x_min'], crop_info['y_min']
        x_max, y_max = crop_info['x_max'], crop_info['y_max']
        
        # Extract crops
        image_crop = image[y_min:y_max, x_min:x_max].copy()
        mask_crop = mask[y_min:y_max, x_min:x_max].copy()
        
        crop_coords = {
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_max,
            'y_max': y_max
        }
        
        return image_crop, mask_crop, crop_coords
    
    def _paste_crop_back(self, original_image: np.ndarray, inpainted_crop: np.ndarray, 
                        crop_coords: Dict[str, int]) -> np.ndarray:
        """
        Paste the inpainted crop back into the original image.
        
        Args:
            original_image: Original full-size image
            inpainted_crop: Inpainted crop result
            crop_coords: Coordinates where to paste the crop
            
        Returns:
            Full image with inpainted region pasted back
        """
        result = original_image.copy()
        
        x_min, y_min = crop_coords['x_min'], crop_coords['y_min']
        x_max, y_max = crop_coords['x_max'], crop_coords['y_max']
        
        # Ensure crop dimensions match
        expected_h, expected_w = y_max - y_min, x_max - x_min
        crop_h, crop_w = inpainted_crop.shape[:2]
        
        if crop_h != expected_h or crop_w != expected_w:
            # Resize crop to match expected dimensions
            inpainted_crop = cv2.resize(inpainted_crop, (expected_w, expected_h))
        
        # Paste the crop back
        result[y_min:y_max, x_min:x_max] = inpainted_crop
        
        return result

    def _process_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Args:
            mask: Input binary mask
            
        Returns:
            Processed mask
        """
        processed_mask = mask.copy()
        
        # Dilate mask to ensure complete coverage
        if self.dilate_mask and self.dilate_kernel_size > 0:
            kernel = np.ones((self.dilate_kernel_size, self.dilate_kernel_size), np.uint8)
            processed_mask = cv2.dilate(processed_mask, kernel, iterations=1)
        
        # Blur mask edges for smoother transitions
        if self.blur_mask_edges and self.blur_kernel_size > 0:
            processed_mask = cv2.GaussianBlur(
                processed_mask.astype(np.float32), 
                (self.blur_kernel_size, self.blur_kernel_size), 
                0
            )
            # Convert back to binary
            processed_mask = (processed_mask > 0.5).astype(np.uint8)
        
        return processed_mask
    
    def _inpaint_with_stable_diffusion(self, image: np.ndarray, mask: np.ndarray, 
                                     prompt: str) -> Dict[str, Any]:
        """
        Perform inpainting using Stable Diffusion.
        
        Args:
            image: Input image
            mask: Binary mask
            prompt: Text prompt
            
        Returns:
            Inpainting result dictionary
        """
        try:
            # Find bounding box of mask for cropping
            crop_info = self._get_mask_bounding_box(mask)
            
            if crop_info is None:
                # No mask pixels found
                return {
                    'inpainted_image': image.copy(),
                    'method': 'stable_diffusion_no_mask',
                    'success': True,
                    'mask_pixels': 0
                }
            
            # Extract crop
            image_crop, mask_crop, crop_coords = self._extract_crop(image, mask, crop_info)
            
            # Convert to PIL
            pil_image, pil_mask = self._convert_to_pil(image_crop, mask_crop)
            
            # Resize for optimal SD processing if needed
            original_size = pil_image.size
            pil_image, pil_mask = self._resize_for_sd(pil_image, pil_mask)
            
            # Run Stable Diffusion inpainting
            logging.debug(f"Running SD inpainting on {pil_image.size} crop")
            result = self.pipeline(
                prompt=prompt,
                image=pil_image,
                mask_image=pil_mask,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                strength=self.strength
            ).images[0]
            
            # Resize back if needed
            if result.size != original_size:
                result = result.resize(original_size, Image.Resampling.LANCZOS)
            
            # Convert back to numpy
            result_np = np.array(result)
            if len(result_np.shape) == 3:
                result_np = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
            
            # Paste back into original image
            inpainted_image = self._paste_crop_back(image, result_np, crop_coords)
            
            return {
                'inpainted_image': inpainted_image,
                'method': 'stable_diffusion',
                'success': True,
                'crop_size': original_size,
                'mask_pixels': int(np.sum(mask > 0)),
                'processing_size': pil_image.size
            }
            
        except Exception as e:
            logging.error(f"Stable Diffusion inpainting failed: {e}")
            raise
    
    def _inpaint_with_opencv(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """
        Perform inpainting using OpenCV methods.
        
        Args:
            image: Input image
            mask: Binary mask
            
        Returns:
            Inpainting result dictionary
        """
        try:
            # Convert mask to 8-bit
            mask_8bit = (mask * 255).astype(np.uint8)
            
            # Apply OpenCV inpainting using primary method settings
            if self.opencv_method == 'telea':
                inpainted = cv2.inpaint(image, mask_8bit, self.opencv_radius, cv2.INPAINT_TELEA)
                method_name = 'telea'
            elif self.opencv_method == 'navier_stokes':
                inpainted = cv2.inpaint(image, mask_8bit, self.opencv_radius, cv2.INPAINT_NS)
                method_name = 'navier_stokes'
            else:
                logging.warning(f"Unknown OpenCV method: {self.opencv_method}, using Telea")
                inpainted = cv2.inpaint(image, mask_8bit, self.opencv_radius, cv2.INPAINT_TELEA)
                method_name = 'telea_fallback'
            
            return {
                'inpainted_image': inpainted,
                'method': f'opencv_{method_name}',
                'success': True,
                'mask_pixels': int(np.sum(mask > 0)),
                'radius': self.opencv_radius
            }
            
        except Exception as e:
            logging.error(f"OpenCV inpainting failed: {e}")
            raise
    
    def _inpaint_with_fallback_opencv(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
        """
        Perform inpainting using OpenCV fallback settings.
        
        Args:
            image: Input image
            mask: Binary mask
            
        Returns:
            Inpainting result dictionary
        """
        try:
            # Convert mask to 8-bit
            mask_8bit = (mask * 255).astype(np.uint8)
            
            # Apply OpenCV inpainting using fallback settings
            if self.fallback_method == 'telea':
                inpainted = cv2.inpaint(image, mask_8bit, self.fallback_radius, cv2.INPAINT_TELEA)
                method_name = 'telea'
            elif self.fallback_method == 'navier_stokes':
                inpainted = cv2.inpaint(image, mask_8bit, self.fallback_radius, cv2.INPAINT_NS)
                method_name = 'navier_stokes'
            else:
                logging.warning(f"Unknown fallback method: {self.fallback_method}, using Telea")
                inpainted = cv2.inpaint(image, mask_8bit, self.fallback_radius, cv2.INPAINT_TELEA)
                method_name = 'telea_fallback'
            
            return {
                'inpainted_image': inpainted,
                'method': f'opencv_fallback_{method_name}',
                'success': True,
                'mask_pixels': int(np.sum(mask > 0)),
                'radius': self.fallback_radius
            }
            
        except Exception as e:
            logging.error(f"Fallback OpenCV inpainting failed: {e}")
            raise
    
    def _get_mask_bounding_box(self, mask: np.ndarray) -> Optional[Dict[str, int]]:
        """
        Get bounding box coordinates for mask.
        
        Args:
            mask: Binary mask
            
        Returns:
            Dictionary with bounding box coordinates or None if no mask
        """
        mask_coords = np.where(mask > 0)
        if len(mask_coords[0]) == 0:
            return None
        
        min_y, max_y = np.min(mask_coords[0]), np.max(mask_coords[0])
        min_x, max_x = np.min(mask_coords[1]), np.max(mask_coords[1])
        
        # Add padding
        h, w = mask.shape
        min_y = max(0, min_y - self.padding)
        max_y = min(h, max_y + self.padding)
        min_x = max(0, min_x - self.padding)
        max_x = min(w, max_x + self.padding)
        
        return {
            'min_x': min_x,
            'max_x': max_x,
            'min_y': min_y,
            'max_y': max_y
        }
    
    def _extract_crop(self, image: np.ndarray, mask: np.ndarray, 
                     crop_info: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        """
        Extract crop from image and mask.
        
        Args:
            image: Input image
            mask: Input mask
            crop_info: Bounding box information
            
        Returns:
            Tuple of (image_crop, mask_crop, crop_coordinates)
        """
        min_x, max_x = crop_info['min_x'], crop_info['max_x']
        min_y, max_y = crop_info['min_y'], crop_info['max_y']
        
        image_crop = image[min_y:max_y, min_x:max_x]
        mask_crop = mask[min_y:max_y, min_x:max_x]
        
        crop_coords = {
            'x': min_x,
            'y': min_y,
            'width': max_x - min_x,
            'height': max_y - min_y
        }
        
        return image_crop, mask_crop, crop_coords
    
    def _convert_to_pil(self, image: np.ndarray, mask: np.ndarray) -> Tuple[Image.Image, Image.Image]:
        """
        Convert numpy arrays to PIL Images.
        
        Args:
            image: Image as numpy array
            mask: Mask as numpy array
            
        Returns:
            Tuple of (PIL_image, PIL_mask)
        """
        # Convert image
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Convert mask
        mask_8bit = (mask * 255).astype(np.uint8)
        pil_mask = Image.fromarray(mask_8bit, mode='L')
        
        return pil_image, pil_mask
    
    def _resize_for_sd(self, image: Image.Image, mask: Image.Image) -> Tuple[Image.Image, Image.Image]:
        """
        Resize images for optimal Stable Diffusion processing.
        
        Args:
            image: PIL image
            mask: PIL mask
            
        Returns:
            Tuple of resized (image, mask)
        """
        width, height = image.size
        max_dimension = max(width, height)
        
        if max_dimension <= self.max_image_size:
            return image, mask
        
        # Calculate new size maintaining aspect ratio
        scale = self.max_image_size / max_dimension
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Ensure dimensions are divisible by 8 (SD requirement)
        new_width = (new_width // 8) * 8
        new_height = (new_height // 8) * 8
        
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        resized_mask = mask.resize((new_width, new_height), Image.Resampling.NEAREST)
        
        logging.debug(f"Resized for SD: {width}x{height} -> {new_width}x{new_height}")
        
        return resized_image, resized_mask
    
    def _paste_crop_back(self, original_image: np.ndarray, inpainted_crop: np.ndarray, 
                        crop_coords: Dict[str, int]) -> np.ndarray:
        """
        Paste inpainted crop back into original image.
        
        Args:
            original_image: Original full image
            inpainted_crop: Inpainted crop
            crop_coords: Crop coordinates
            
        Returns:
            Full image with inpainted region
        """
        result = original_image.copy()
        
        x = crop_coords['x']
        y = crop_coords['y']
        h, w = inpainted_crop.shape[:2]
        
        # Ensure we don't go out of bounds
        end_y = min(result.shape[0], y + h)
        end_x = min(result.shape[1], x + w)
        
        crop_h = end_y - y
        crop_w = end_x - x
        
        result[y:end_y, x:end_x] = inpainted_crop[:crop_h, :crop_w]
        
        return result
    
    def batch_inpaint(self, panoramas: List[np.ndarray], masks_list: List[List[np.ndarray]], 
                     prompts: List[str]) -> List[Dict[str, Any]]:
        """
        Perform batch inpainting on multiple panoramas.
        
        Args:
            panoramas: List of panorama images
            masks_list: List of mask lists (one list per panorama)
            prompts: List of prompts (one per panorama)
            
        Returns:
            List of inpainting results
        """
        if len(panoramas) != len(masks_list) or len(panoramas) != len(prompts):
            raise ValueError("Input lists must have the same length")
        
        results = []
        
        for i, (panorama, masks, prompt) in enumerate(zip(panoramas, masks_list, prompts)):
            logging.info(f"Batch inpainting {i+1}/{len(panoramas)}")
            result = self.inpaint_background(panorama, masks, prompt)
            results.append(result)
        
        return results
    
    def validate_inputs(self, panorama: np.ndarray, masks: List[np.ndarray], 
                       prompt: str) -> Dict[str, Any]:
        """
        Validate inputs for inpainting.
        
        Args:
            panorama: Panorama image
            masks: List of masks
            prompt: Text prompt
            
        Returns:
            Validation results
        """
        issues = []
        
        # Check panorama
        if panorama is None:
            issues.append("panorama_is_none")
        elif len(panorama.shape) not in [2, 3]:
            issues.append("invalid_panorama_shape")
        elif panorama.size == 0:
            issues.append("empty_panorama")
        
        # Check masks
        if not isinstance(masks, list):
            issues.append("masks_not_list")
        else:
            for i, mask in enumerate(masks):
                if mask is None:
                    issues.append(f"mask_{i}_is_none")
                elif len(mask.shape) != 2:
                    issues.append(f"mask_{i}_invalid_shape")
                elif panorama is not None and mask.shape[:2] != panorama.shape[:2]:
                    issues.append(f"mask_{i}_size_mismatch")
        
        # Check prompt
        if not isinstance(prompt, str):
            issues.append("prompt_not_string")
        elif len(prompt.strip()) < 5:
            issues.append("prompt_too_short")
        elif len(prompt) > 500:
            issues.append("prompt_too_long")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }
