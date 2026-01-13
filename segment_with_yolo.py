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
    parser.add_argument("video_path", type=str, help="Path to the input video")
    parser.add_argument("--model_path", type=str, default="yoloe-11l-seg.pt", help="Path or name of the YOLOE model")
    args = parser.parse_args()

    video_path = args.video_path

    # 0. Create experiment folder as timestamped directory
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/home/itec/emanuele/pointstream/experiments/{timestamp}_yolo_seg"
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(masks_dir, exist_ok=True)

    # Initialize model
    print(f"Loading model: {args.model_path}")
    model = YOLOE(args.model_path)
    
    # Set text prompt for "tennis players holding their rackets"
    # We use a singular prompt which the model generalizes to instances
    names = ["tennis player", "racket", "tennis ball"]
    print(f"Setting classes to: {names}")
    model.set_classes(names, model.get_text_pe(names))

    print(f"Starting tracking on video: {video_path}")
    # Run tracking with persist=True to maintain IDs across frames
    # user requested: assume cuda availability
    results = model.track(video_path, persist=True, device='cuda', stream=True, imgsz=3840)

    metadata = []
    
    for frame_idx, result in enumerate(results):
        frame_meta = {
            "frame_index": frame_idx,
            "detections": []
        }
        
        # Check if we have detections and masks
        if result.boxes is not None and result.masks is not None and result.boxes.id is not None:
            boxes = result.boxes
            masks = result.masks
            
            # Move data to CPU for processing
            track_ids = boxes.id.int().cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            cls_ids = boxes.cls.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().tolist()
            
            # masks.data contains the mask tensors (N, H, W)
            # We need to resize them to the original image size
            raw_masks = masks.data
            
            if raw_masks is not None:
                orig_h, orig_w = result.orig_shape
                
                for i, track_id in enumerate(track_ids):
                    # Get individual mask and resize
                    mask_tensor = raw_masks[i]
                    mask_np = mask_tensor.cpu().numpy()
                    
                    # Resize mask to original image dimensions
                    # mask_np might be smaller than orig image (e.g. 160x160), so we upsample
                    mask_resized = cv2.resize(mask_np, (orig_w, orig_h))
                    
                    # Create binary mask (0 or 255)
                    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
                    
                    # Save mask image
                    mask_filename = f"mask_frame_{frame_idx:06d}_id_{track_id}.png"
                    mask_path = os.path.join(masks_dir, mask_filename)
                    cv2.imwrite(mask_path, mask_binary)
                    
                    # Store metadata
                    detection_data = {
                        "track_id": track_id,
                        "class_index": cls_ids[i],
                        "class_name": names[cls_ids[i]] if cls_ids[i] < len(names) else "unknown",
                        "confidence": confs[i],
                        "bbox": xyxy[i], # [x1, y1, x2, y2]
                        "mask_file": mask_filename
                    }
                    frame_meta["detections"].append(detection_data)
        
        metadata.append(frame_meta)
        
        if frame_idx % 50 == 0:
            print(f"Processed frame {frame_idx}...")

    # Save final metadata to JSON
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"Processing complete. Metadata saved to {metadata_path}")
    print(f"Masks saved in {masks_dir}")

if __name__ == "__main__":
    main()
