# python segment_with_sam.py /home/itec/emanuele/Datasets/djokovic_federer/015.mp4 --model_path /home/itec/emanuele/models/sam3.pt

import argparse
import json
import os
import cv2
import numpy as np
import torch
from ultralytics.models.sam import SAM3VideoPredictor, SAM3SemanticPredictor

# TODO: To improve ID consistency, check bounding boxes after processing all frames and re-assign IDs based on IoU with previous frames.
# TODO: refine prompts for better segmentation, ablation test
# TODO: Ultralytics sam3 allows exemplar-based segmentation, try that.
# TODO: Ultralytics sam3 allows efficiency gains by reusing features, try that too.
# TODO: if it breaks in the middle, can I save the poses so far with try-finally? Can I resume from there?

def main():
    parser = argparse.ArgumentParser(description="Segment video using SAM with text prompts")
    parser.add_argument("--video_path", type=str, default="/home/itec/emanuele/Datasets/djokovic_federer/012.mp4", help="Path to the input video")
    parser.add_argument("--model_path", type=str, default="/home/itec/emanuele/models/sam3.pt", help="Path or name of the SAM model")
    args = parser.parse_args()

    video_path = args.video_path

    # 0. Create experiment folder as timestamped directory
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directories
    output_dir = f"/home/itec/emanuele/pointstream/experiments/{timestamp}_sam_seg"
    os.makedirs(output_dir, exist_ok=True)
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

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
    # Use larger image size for better initial detection accuracy
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

        # Save frame as PNG
        frame_filename = os.path.join(frames_dir, f"{frame_idx:05d}.png")
        cv2.imwrite(frame_filename, frame)

        # Process all detections (should be exactly 2 tracked objects)
        for idx, (det, mask) in enumerate(zip(result.boxes, result.masks)):
            cls_id = int(det.cls[0].cpu().numpy())
            # Assign stable IDs based on bbox order: first bbox = ID 0, second bbox = ID 1
            det_id = idx
            bbox = det.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
            mask = mask.data.cpu().numpy()[0]  # Binary mask
            
            # Save mask as PNG
            mask_filename = os.path.join(masks_dir, f"{frame_idx:05d}_id{det_id}.png")
            cv2.imwrite(mask_filename, (mask * 255).astype(np.uint8))
            
            frame_data["detections"].append({
                "id": det_id,
                "class_id": cls_id,
                "bbox": bbox,
                "mask_path": mask_filename
            })
        metadata.append(frame_data)

    # Save metadata in a csv file
    import pandas as pd
    metadata_frame = pd.DataFrame(metadata)
    # Explode detections to have one row per detection
    metadata_frame = metadata_frame.explode('detections')
    # Expand detection dicts into separate columns
    detections_expanded = metadata_frame['detections'].apply(pd.Series)
    metadata_frame = metadata_frame.drop('detections', axis=1).join(detections_expanded)
    metadata_csv_path = os.path.join(output_dir, "tracking_metadata.csv")
    metadata_frame.to_csv(metadata_csv_path, index=False)
    print(f"Tracking metadata saved to: {metadata_csv_path}")

if __name__ == "__main__":
    main()
