# python segment_with_sam.py /home/itec/emanuele/Datasets/djokovic_federer/015.mp4 --model_path /home/itec/emanuele/models/sam3.pt

import argparse
import datetime
import os
import cv2
import numpy as np
import torch
from ultralytics.models.sam import SAM3VideoPredictor, SAM3SemanticPredictor

# TODO: To improve ID consistency, check bounding boxes after processing all frames and re-assign IDs based on IoU with previous frames.
# TODO: need a better way than top 2 highest confidence to select players in frame 0, for example I could draw a map of likelihood based on position in the court (players are likely to start near their baselines), what else?
# TODO: refine prompts for better segmentation, ablation test
#TODO: right now we are just segmenting players, we need to segment rackets and ball too.
# TODO: Ultralytics sam3 allows efficiency gains by reusing features, SAM3VideoPredictor already includes this, but could be useful across videos.
# TODO: if it breaks in the middle, can I save the poses so far with try-finally? Can I resume from there?


def resize_and_pad(img, target_size=512):
    """
    Resize and pad an image to target_size x target_size, maintaining aspect ratio.
    Always resizes so the largest dimension becomes target_size (consistent person size for training).
    
    Returns:
        padded: The resized and padded image
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


def main():
    parser = argparse.ArgumentParser(description="Segment video using SAM with text prompts")
    parser.add_argument("--video_path", type=str, default="/home/itec/emanuele/Datasets/djokovic_federer/012.mp4", help="Path to the input video")
    parser.add_argument("--model_path", type=str, default="/home/itec/emanuele/models/sam3.pt", help="Path or name of the SAM model")
    args = parser.parse_args()

    video_path = args.video_path

    # 0. Create experiment folder as timestamped directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directories
    output_dir = f"/home/itec/emanuele/pointstream/experiments/{timestamp}_sam_seg"
    os.makedirs(output_dir, exist_ok=True)
    masked_crops_dir = os.path.join(output_dir, "masked_crops")
    os.makedirs(masked_crops_dir, exist_ok=True)

    # Step 1: Extract frame 0 and get initial bounding boxes
    print("Extracting frame 0 to get initial bounding boxes...")
    cap = cv2.VideoCapture(video_path)
    ret, frame0 = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError("Failed to read frame 0 from video")
    
    # Save frame 0 temporarily for semantic predictor
    frame0_path = os.path.join(output_dir, "frame0_temp.png")
    cv2.imwrite(frame0_path, frame0)
    
    # Initialize semantic predictor for frame 0 only
    overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1288, model=args.model_path, half=True, save=True, compile=None)
    semantic_predictor = SAM3SemanticPredictor(overrides=overrides)
    semantic_predictor.set_image(frame0_path)
    
    # Get detections for tennis players
    names = ["tennis player"]
    print(f"Running semantic segmentation on frame 0 with classes: {names}")
    frame0_results = semantic_predictor(text=names)
    
    # Filter to get 2 most confident class_id 0 detections
    class_0_detections = []
    for idx, det in enumerate(frame0_results[0].boxes):
        cls_id = int(det.cls[0].cpu().numpy())
        if cls_id == 0:
            bbox = det.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
            conf = float(det.conf[0].cpu().numpy())
            class_0_detections.append({"idx": idx, "bbox": bbox, "conf": conf})
    
    # Check if we have at least 2 detections
    if len(class_0_detections) < 2:
        raise ValueError(f"Frame 0 has only {len(class_0_detections)} class_id 0 detection(s). Need at least 2 tennis players.")
    
    # Sort by confidence and take top 2
    class_0_detections.sort(key=lambda x: x["conf"], reverse=True)
    initial_bboxes = [det["bbox"] for det in class_0_detections[:2]]
    
    print(f"Found {len(class_0_detections)} tennis player(s) in frame 0")
    print(f"Using top 2 bounding boxes for tracking:")
    for i, bbox in enumerate(initial_bboxes):
        print(f"  Player {i+1}: {bbox}")
    
    # Clean up temporary frame
    os.remove(frame0_path)
    
    # Step 2: Initialize bbox-based video predictor
    print(f"\nStarting bbox-based tracking on video: {video_path}")
    overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=644, model=args.model_path, half=True, save=False, compile=None)
    predictor = SAM3VideoPredictor(overrides=overrides)
    results = predictor(source=video_path, bboxes=initial_bboxes, stream=True)

    metadata = []
    
    for frame_idx, result in enumerate(results):
        # Save the frame, and each detection's id and bounding box in a common dataframe
        frame = result.orig_img
        frame_height, frame_width = frame.shape[:2]
        frame_data = {"frame_index": frame_idx, "detections": []}

        # Process all detections (should be exactly 2 tracked objects)
        for idx, (det, mask) in enumerate(zip(result.boxes, result.masks)):
            # Both detections are tennis players, so class_id is always 0 TODO: this is limiting if we want to track more classes.
            cls_id = 0
            # Assign stable IDs based on bbox order: first bbox = ID 0, second bbox = ID 1
            det_id = idx
            bbox = det.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
            mask_data = mask.data.cpu().numpy()[0]  # Binary mask
            
            # Create masked crop directly
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_width, x2), min(frame_height, y2)
            
            # Apply mask to frame and crop
            mask_3ch = cv2.cvtColor((mask_data * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            masked_frame = cv2.bitwise_and(frame, mask_3ch)
            masked_crop = masked_frame[y1:y2, x1:x2]
            
            # Save masked crop if valid
            if masked_crop is not None and masked_crop.size > 0:
                masked_crop_padded, transform_info = resize_and_pad(masked_crop, target_size=512)
                id_subfolder = os.path.join(masked_crops_dir, f"id{det_id}")
                os.makedirs(id_subfolder, exist_ok=True)
                crop_path = os.path.join(id_subfolder, f"{frame_idx:05d}.png")
                cv2.imwrite(crop_path, masked_crop_padded)
            else:
                transform_info = None
            
            frame_data["detections"].append({
                "id": det_id,
                "class_id": cls_id,
                "bbox": bbox,
                "transform_info": transform_info
            })
        metadata.append(frame_data)

    # Save metadata in a csv file
    import pandas as pd
    metadata_frame = pd.DataFrame(metadata)
    # Explode detections to have one row per detection
    metadata_frame = metadata_frame.explode('detections').reset_index(drop=True)
    # Expand detection dicts into separate columns
    detections_expanded = metadata_frame['detections'].apply(pd.Series)
    metadata_frame = pd.concat([metadata_frame.drop('detections', axis=1), detections_expanded], axis=1)
    
    # Expand transform_info dict into separate columns for easy access during inference
    if 'transform_info' in metadata_frame.columns:
        transform_expanded = metadata_frame['transform_info'].apply(
            lambda x: pd.Series(x) if isinstance(x, dict) else pd.Series()
        )
        transform_expanded.columns = [f'transform_{col}' for col in transform_expanded.columns]
        metadata_frame = pd.concat([metadata_frame.drop('transform_info', axis=1), transform_expanded], axis=1)
    
    metadata_csv_path = os.path.join(output_dir, "tracking_metadata.csv")
    metadata_frame.to_csv(metadata_csv_path, index=False)
    print(f"Tracking metadata saved to: {metadata_csv_path}")

if __name__ == "__main__":
    main()
