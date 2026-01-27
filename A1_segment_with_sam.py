# python segment_with_sam.py /home/itec/emanuele/Datasets/djokovic_federer/015.mp4 --model_path /home/itec/emanuele/models/sam3.pt

import argparse
import json
import os
import cv2
import numpy as np
import torch
from ultralytics.models.sam import SAM3VideoSemanticPredictor

# TODO: To improve ID consistency, check bounding boxes after processing all frames and re-assign IDs based on IoU with previous frames.
# TODO: refine prompts for better segmentation, ablation test
# TODO: Ultralytics sam3 allows exemplar-based segmentation, tried by passing first frame players. It stopped detecting rackets and ball though.
# TODO: Ultralytics sam3 allows efficiency gains by reusing features, try that too.
# TODO: if it breaks in the middle, can I save the poses so far with try-finally? Can I resume from there?

def main():
    parser = argparse.ArgumentParser(description="Segment video using SAM with text prompts")
    parser.add_argument("--video_path", type=str, default="/home/itec/emanuele/Datasets/djokovic_federer/015.mp4", help="Path to the input video")
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

    # Initialize semantic video predictor
    overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=640, model=args.model_path, half=True, save=True)
    predictor = SAM3VideoSemanticPredictor(overrides=overrides)
    
    # Set text prompt for "tennis players holding their rackets"
    # We use a singular prompt which the model generalizes to instances
    names = ["tennis player", "racket", "tennis ball"]
    print(f"Setting classes to: {names}")
    
    print(f"Starting tracking on video: {video_path}")
    results = predictor(source=video_path, text=names, stream=True)

    metadata = []
    
    for frame_idx, result in enumerate(results):
        # Save the frame, and each detection's id and bounding box in a common dataframe
        frame = result.orig_img
        frame_height, frame_width = frame.shape[:2]
        frame_data = {"frame_index": frame_idx, "detections": []}

        # Save frame as PNG
        frame_filename = os.path.join(frames_dir, f"frame{frame_idx:05d}.png")
        cv2.imwrite(frame_filename, frame)

        # Filter detections: keep only 2 most confident (first 2) class_id 0 detections
        class_0_indices = []
        for idx, det in enumerate(result.boxes):
            cls_id = int(det.cls[0].cpu().numpy())
            if cls_id == 0:
                class_0_indices.append(idx)
        
        # Determine which indices to keep
        indices_to_keep = set(range(len(result.boxes)))
        if len(class_0_indices) > 2:
            # Remove all but the first 2 class_id 0 detections
            indices_to_remove = class_0_indices[2:]
            indices_to_keep = indices_to_keep - set(indices_to_remove)
        
        # Process only filtered detections
        for idx, (det, mask) in enumerate(zip(result.boxes, result.masks)):
            if idx not in indices_to_keep:
                continue
            cls_id = int(det.cls[0].cpu().numpy())
            det_id = int(det.id.cpu().numpy()) if det.id is not None else -1
            bbox = det.xyxy[0].cpu().numpy().tolist()  # [x1, y1, x2, y2]
            mask = mask.data.cpu().numpy()[0]  # Binary mask
            
            # Save mask as PNG
            mask_filename = os.path.join(masks_dir, f"frame{frame_idx:05d}_id{det_id}.png")
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
