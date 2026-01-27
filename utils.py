"""
Utility functions for PointStream dataset preparation and inference.
"""
import cv2
import numpy as np


def resize_and_pad(img, target_size=512):
    """
    Resize and pad an image to target_size x target_size, maintaining aspect ratio.
    Always resizes so the largest dimension becomes target_size (consistent person size for training).
    
    Args:
        img: Input image (H, W) or (H, W, C)
        target_size: Target dimension for the square output
        
    Returns:
        padded: The resized and padded image (target_size, target_size, ...)
        transform_info: Dict with original dimensions, scale, and padding offsets for reversing
    """
    h, w = img.shape[:2]
    # Always resize so the largest dimension becomes target_size
    scale = target_size / max(w, h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Use INTER_AREA for shrinking, INTER_LINEAR for enlarging
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    
    if len(img_resized.shape) == 3:
        padded = np.zeros((target_size, target_size, img_resized.shape[2]), dtype=img_resized.dtype)
    else:
        padded = np.zeros((target_size, target_size), dtype=img_resized.dtype)
    
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    padded[top:top+new_h, left:left+new_w] = img_resized
    
    # Store transform info for reversing during inference
    transform_info = {
        "orig_h": h,
        "orig_w": w,
        "scale": scale,
        "pad_top": top,
        "pad_left": left,
        "resized_h": new_h,
        "resized_w": new_w
    }
    
    return padded, transform_info


def reverse_resize_and_pad(img, transform_info):
    """
    Reverse the resize_and_pad transformation to restore original dimensions.
    Use this during inference to get outputs at the original crop size.
    
    Args:
        img: The 512x512 (or target_size x target_size) image from the model
        transform_info: Dict containing the transform parameters:
            - orig_h, orig_w: Original dimensions before transform
            - scale: Scale factor that was applied
            - pad_top, pad_left: Padding offsets
            - resized_h, resized_w: Dimensions after resize, before padding
            
    Returns:
        restored: Image at original dimensions (orig_h, orig_w, ...)
    """
    # Extract transform parameters
    orig_h = int(transform_info["orig_h"])
    orig_w = int(transform_info["orig_w"])
    scale = float(transform_info["scale"])
    pad_top = int(transform_info["pad_top"])
    pad_left = int(transform_info["pad_left"])
    resized_h = int(transform_info["resized_h"])
    resized_w = int(transform_info["resized_w"])
    
    # Step 1: Remove padding (crop the content region)
    cropped = img[pad_top:pad_top+resized_h, pad_left:pad_left+resized_w]
    
    # Step 2: Resize back to original dimensions
    # Use INTER_AREA for shrinking, INTER_LINEAR for enlarging
    inv_scale = 1.0 / scale
    interp = cv2.INTER_AREA if inv_scale < 1.0 else cv2.INTER_LINEAR
    restored = cv2.resize(cropped, (orig_w, orig_h), interpolation=interp)
    
    return restored


def paste_crop_to_frame(frame, crop, bbox):
    """
    Paste a crop back into a frame at the specified bounding box location.
    
    Args:
        frame: The original frame (H, W, C)
        crop: The crop to paste (should be at original crop dimensions, 
              i.e., after reverse_resize_and_pad)
        bbox: [x1, y1, x2, y2] coordinates where the crop should be placed
        
    Returns:
        frame: The frame with the crop pasted (modified in-place)
    """
    frame_h, frame_w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    
    # Clamp to frame boundaries
    x1_clamped = max(0, x1)
    y1_clamped = max(0, y1)
    x2_clamped = min(frame_w, x2)
    y2_clamped = min(frame_h, y2)
    
    # Calculate the corresponding region in the crop
    crop_x1 = x1_clamped - x1
    crop_y1 = y1_clamped - y1
    crop_x2 = crop_x1 + (x2_clamped - x1_clamped)
    crop_y2 = crop_y1 + (y2_clamped - y1_clamped)
    
    # Paste the crop
    frame[y1_clamped:y2_clamped, x1_clamped:x2_clamped] = crop[crop_y1:crop_y2, crop_x1:crop_x2]
    
    return frame


def transform_keypoints_to_original(keypoints, transform_info):
    """
    Transform keypoints from 512x512 space back to original crop space.
    
    Args:
        keypoints: List of [x, y] keypoints in 512x512 space
        transform_info: Dict with transform parameters
        
    Returns:
        keypoints_original: Keypoints in original crop space
    """
    pad_top = int(transform_info["pad_top"])
    pad_left = int(transform_info["pad_left"])
    scale = float(transform_info["scale"])
    
    keypoints_original = []
    for kpt in keypoints:
        if kpt[0] == 0 and kpt[1] == 0:
            # Invalid keypoint, keep as-is
            keypoints_original.append([0, 0])
        else:
            # Remove padding offset, then scale back
            x_orig = (kpt[0] - pad_left) / scale
            y_orig = (kpt[1] - pad_top) / scale
            keypoints_original.append([x_orig, y_orig])
    
    return keypoints_original


def keypoints_to_frame_coords(keypoints_crop, bbox):
    """
    Transform keypoints from crop space to frame space.
    
    Args:
        keypoints_crop: Keypoints in crop coordinates
        bbox: [x1, y1, x2, y2] of the crop in frame space
        
    Returns:
        keypoints_frame: Keypoints in frame coordinates
    """
    x1, y1, x2, y2 = bbox
    
    keypoints_frame = []
    for kpt in keypoints_crop:
        if kpt[0] == 0 and kpt[1] == 0:
            keypoints_frame.append([0, 0])
        else:
            keypoints_frame.append([kpt[0] + x1, kpt[1] + y1])
    
    return keypoints_frame
