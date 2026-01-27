import argparse
import os
import cv2
import numpy as np
import pandas as pd
import ast
from tqdm import tqdm

# TODO: Can be sped up by using multiprocessing and GPU.

# COCO keypoint skeleton connections for YOLO pose models
SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],  # Head
    [6, 12], [7, 13], [6, 7],  # Torso
    [6, 8], [8, 10],  # Left arm
    [7, 9], [9, 11],  # Right arm
    [12, 14], [14, 16],  # Left leg
    [13, 15], [15, 17]   # Right leg
]

# Connection colors (BGR format) - color-coded by body part and side
# This helps identify which direction the actor is facing
CONNECTION_COLORS = [
    # Head connections (yellow/orange)
    (0, 165, 255),   # [16, 14] - right ear to right eye
    (0, 165, 255),   # [14, 12] - right eye to right shoulder
    (0, 255, 255),   # [17, 15] - left ear to left eye
    (0, 255, 255),   # [15, 13] - left eye to left shoulder
    (0, 200, 255),   # [12, 13] - shoulder connection
    
    # Torso (white/gray)
    (200, 200, 200), # [6, 12] - left hip to left shoulder
    (200, 200, 200), # [7, 13] - right hip to right shoulder
    (200, 200, 200), # [6, 7] - hip connection
    
    # Left arm (cyan/blue) - ACTOR'S LEFT
    (255, 255, 0),   # [6, 8] - left shoulder to left elbow
    (255, 200, 0),   # [8, 10] - left elbow to left wrist
    
    # Right arm (red/magenta) - ACTOR'S RIGHT
    (0, 0, 255),     # [7, 9] - right shoulder to right elbow
    (128, 0, 255),   # [9, 11] - right elbow to right wrist
    
    # Left leg (cyan/blue) - ACTOR'S LEFT
    (255, 150, 0),   # [12, 14] - left shoulder to left knee (Note: this seems wrong in original)
    (255, 100, 0),   # [14, 16] - left knee to left ankle
    
    # Right leg (red/magenta) - ACTOR'S RIGHT
    (64, 0, 255),    # [13, 15] - right shoulder to right knee (Note: this seems wrong in original)
    (128, 0, 200)    # [15, 17] - right knee to right ankle
]


def resize_and_pad(img, target_size=512):
    """
    Resize and pad an image to target_size x target_size, maintaining aspect ratio.
    
    Args:
        img: Input image
        target_size: Target size for both width and height (default: 512)
    
    Returns:
        Resized and padded image of size (target_size, target_size)
    """
    h, w = img.shape[:2]
    
    # Calculate scale to fit within target size
    scale = min(target_size / w, target_size / h)
    
    # Only downscale if image is larger than target
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        new_w, new_h = w, h
    
    # Create black canvas
    if len(img.shape) == 3:
        padded = np.zeros((target_size, target_size, img.shape[2]), dtype=img.dtype)
    else:
        padded = np.zeros((target_size, target_size), dtype=img.dtype)
    
    # Calculate padding to center the image
    top = (target_size - new_h) // 2
    left = (target_size - new_w) // 2
    
    # Place the image in the center
    padded[top:top+new_h, left:left+new_w] = img
    
    return padded


def overlay_mask_and_crop(frame, mask_path, bbox):
    """
    Overlay mask on frame and crop based on bounding box.
    
    Args:
        frame: Original frame image
        mask_path: Path to the mask PNG file
        bbox: Bounding box [x1, y1, x2, y2]
    
    Returns:
        Cropped and masked image
    """
    # Load mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Warning: Could not load mask from {mask_path}")
        return None
    
    # Apply mask to frame (set background to black)
    masked_frame = frame.copy()
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    masked_frame = cv2.bitwise_and(masked_frame, mask_3ch)
    
    # Crop based on bounding box
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    
    cropped = masked_frame[y1:y2, x1:x2]
    return cropped


def create_skeleton_image(keypoints, bbox, img_size=(512, 512)):
    """
    Create a skeleton image from keypoints.
    
    Args:
        keypoints: List of [x, y] keypoint coordinates (relative to crop)
        bbox: Bounding box [x1, y1, x2, y2] used for the crop
        img_size: Size of output skeleton image (width, height)
    
    Returns:
        Skeleton image as numpy array
    """
    skeleton_img = np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)
    
    if not keypoints or len(keypoints) == 0:
        return skeleton_img
    
    # Convert keypoints to numpy array
    kpts = np.array(keypoints)
    
    # Scale keypoints to output image size
    x1, y1, x2, y2 = bbox
    crop_w = x2 - x1
    crop_h = y2 - y1
    
    if crop_w <= 0 or crop_h <= 0:
        return skeleton_img
    
    scaled_kpts = kpts.copy()
    scaled_kpts[:, 0] = (kpts[:, 0] / crop_w) * img_size[0]
    scaled_kpts[:, 1] = (kpts[:, 1] / crop_h) * img_size[1]
    
    # Draw skeleton connections with color coding
    for i, connection in enumerate(SKELETON):
        pt1_idx, pt2_idx = connection[0] - 1, connection[1] - 1  # Convert to 0-indexed
        
        if pt1_idx < len(scaled_kpts) and pt2_idx < len(scaled_kpts):
            pt1 = tuple(scaled_kpts[pt1_idx].astype(int))
            pt2 = tuple(scaled_kpts[pt2_idx].astype(int))
            
            # Only draw if both points are valid (not [0, 0])
            if pt1 != (0, 0) and pt2 != (0, 0):
                color = CONNECTION_COLORS[i]
                cv2.line(skeleton_img, pt1, pt2, color, 3)  # Thicker line for visibility
    
    # Draw keypoints as small neutral circles (optional - can be removed if not desired)
    for i, kpt in enumerate(scaled_kpts):
        pt = tuple(kpt.astype(int))
        if pt != (0, 0):  # Only draw valid keypoints
            cv2.circle(skeleton_img, pt, 2, (100, 100, 100), -1)  # Small gray dots
    
    return skeleton_img


def main():
    parser = argparse.ArgumentParser(description="Create masked crops and skeleton images")
    parser.add_argument("--experiment_dir", type=str, default="/home/itec/emanuele/pointstream/experiments/20260123_122136_sam_seg", help="Path to the experiment folder")
    parser.add_argument("--max_frames", type=int, default=10, help="Maximum number of frames to process (for debugging)")
    args = parser.parse_args()

    experiment_dir = args.experiment_dir
    
    # Load metadata
    tracking_csv = os.path.join(experiment_dir, "tracking_metadata.csv")
    pose_csv = os.path.join(experiment_dir, "pose_metadata.csv")
    frames_dir = os.path.join(experiment_dir, "frames")
    
    if not os.path.exists(tracking_csv):
        print(f"Error: Tracking metadata not found at {tracking_csv}")
        return
    
    if not os.path.exists(pose_csv):
        print(f"Error: Pose metadata not found at {pose_csv}")
        return
    
    tracking_df = pd.read_csv(tracking_csv)
    pose_df = pd.read_csv(pose_csv)
    
    # Create output directories
    masked_crops_dir = os.path.join(experiment_dir, "masked_crops")
    skeletons_dir = os.path.join(experiment_dir, "skeletons")
    os.makedirs(masked_crops_dir, exist_ok=True)
    os.makedirs(skeletons_dir, exist_ok=True)
    
    print(f"Processing {len(tracking_df)} detections...")
    
    # Group by frame_index to load each frame only once
    if args.max_frames is not None:
        print(f"Limiting to first {args.max_frames} frames for debugging")
        tracking_df = tracking_df[tracking_df['frame_index'] < args.max_frames]
    grouped_tracking = tracking_df.groupby('frame_index')
    
    # Process masked crops
    for frame_idx, group in tqdm(grouped_tracking, desc="Creating masked crops"):
        # Load frame once per group
        frame_filename = os.path.join(frames_dir, f"frame{frame_idx:05d}.png")
        frame = cv2.imread(frame_filename)
        if frame is None:
            print(f"Warning: Could not load frame {frame_filename}")
            continue
        
        # Process all detections for this frame
        for idx, row in group.iterrows():
            det_id = row['id']
            
            # Parse bbox and mask path
            bbox = ast.literal_eval(row['bbox']) if isinstance(row['bbox'], str) else row['bbox']
            mask_path = row['mask_path']
            
            # Create masked crop
            masked_crop = overlay_mask_and_crop(frame, mask_path, bbox)
            # Sometimes masks are 0 by 0 pixels, skip those
            if masked_crop is not None and masked_crop.size > 0:
                # Resize and pad to 512x512
                masked_crop_padded = resize_and_pad(masked_crop, target_size=512)
                output_path = os.path.join(masked_crops_dir, f"frame{frame_idx:05d}_id{det_id}.png")
                cv2.imwrite(output_path, masked_crop_padded)
    
    # Merge pose_df with tracking_df to get class_id information
    pose_with_class = pose_df.merge(
        tracking_df[['frame_index', 'id', 'class_id']], 
        left_on=['frame_index', 'detection_id'], 
        right_on=['frame_index', 'id'], 
        how='left'
    )
    
    # Filter for class_id 0 (tennis players) only
    pose_class_0 = pose_with_class[pose_with_class['class_id'] == 0]
    
    print(f"\nProcessing {len(pose_class_0)} poses (class_id 0 only)...")
    
    # Process skeleton images
    for idx, row in tqdm(pose_class_0.iterrows(), total=len(pose_class_0), desc="Creating skeletons"):
        frame_idx = row['frame_index']
        det_id = row['detection_id']
        
        # Parse keypoints and bbox
        keypoints = ast.literal_eval(row['keypoints']) if isinstance(row['keypoints'], str) else row['keypoints']
        bbox = ast.literal_eval(row['bbox']) if isinstance(row['bbox'], str) else row['bbox']
        
        # Create skeleton image
        skeleton_img = create_skeleton_image(keypoints, bbox)
        
        # Save skeleton
        output_path = os.path.join(skeletons_dir, f"frame{frame_idx:05d}_id{det_id}.png")
        cv2.imwrite(output_path, skeleton_img)
    
    print(f"\nMasked crops saved to: {masked_crops_dir}")
    print(f"Skeleton images saved to: {skeletons_dir}")


if __name__ == "__main__":
    main()
