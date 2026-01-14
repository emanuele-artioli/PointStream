# python segment_with_yolo.py /home/itec/emanuele/Datasets/djokovic_federer/015.mp4 --model_path /home/itec/emanuele/models/yoloe-11l-seg.pt

import argparse
import json
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLOE

def main():
    parser = argparse.ArgumentParser(description="Segment video using YOLOE with text prompts")
    parser.add_argument("--video_path", type=str, default="/home/itec/emanuele/Datasets/djokovic_federer/015.mp4", help="Path to the input video")
    parser.add_argument("--model_path", type=str, default="/home/itec/emanuele/models/yoloe-11l-seg.pt", help="Path or name of the YOLOE model")
    args = parser.parse_args()

    video_path = args.video_path

    # 0. Create experiment folder as timestamped directory
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directories
    output_dir = f"/home/itec/emanuele/pointstream/experiments/{timestamp}_yolo_seg"
    os.makedirs(output_dir, exist_ok=True)
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Initialize model
    print(f"Loading model: {args.model_path}")
    model = YOLOE(args.model_path)
    
    # Set text prompt for "tennis players holding their rackets"
    # We use a singular prompt which the model generalizes to instances
    names = ["tennis player", "racket", "tennis ball"] # TODO: refine prompts for better segmentation, ablation test
    print(f"Setting classes to: {names}")
    model.set_classes(names, model.get_text_pe(names))

    print(f"Starting tracking on video: {video_path}")
    # Run tracking with persist=True to maintain IDs across frames
    # user requested: assume cuda availability
    # Optimized: half=True (FP16), retina_masks=True (original resolution masks)
    results = model.track(
        video_path, stream=True, persist=True, retina_masks=True, conf=0.1, iou=0.1, imgsz=3840, device='cuda', half=True, stream_buffer=True
    )

    metadata = []
    
    for frame_idx, result in enumerate(results):
        # Save the frame, and each detection's id and bounding box in a common dataframe
        frame = result.orig_img
        frame_height, frame_width = frame.shape[:2]
        frame_data = {"frame_index": frame_idx, "detections": []}

        # Save frame as PNG
        frame_filename = os.path.join(frames_dir, f"frame{frame_idx:05d}.png")
        cv2.imwrite(frame_filename, frame)

        for det, mask in zip(result.boxes, result.masks):
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
