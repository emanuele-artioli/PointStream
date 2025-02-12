import os
import numpy as np
import cv2
import argparse
from ultralytics import SAM, YOLO

def perform_instance_segmentation_sam(input_file, segmented_folder):
    cap = cv2.VideoCapture(input_file)
    model = SAM('sam2.1_b.pt')
    frame_id = 0

    # No object tracking for SAM; we assign new IDs per frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame)
        for mask_idx, mask in enumerate(results[0].masks):
            mask_data = mask.data[0].cpu().numpy().astype(np.uint8)
            object_folder = os.path.join(segmented_folder, f"player_{frame_id}_{mask_idx}")
            os.makedirs(object_folder, exist_ok=True)
            object_img = cv2.bitwise_and(frame, frame, mask=mask_data)
            cv2.imwrite(os.path.join(object_folder, f"{frame_id:04d}.png"), object_img)
        frame_id += 1
    cap.release()

def perform_instance_segmentation_yolo(input_file, segmented_folder):
    model = YOLO('yolo11n-seg.pt')
    results = model.track(
        input_file, 
        # conf=0.5, 
        tracker='bytetrack.yaml', # uses BoT-SORT by default
        imgsz=640, 
        classes=[0], 
        max_det=2, 
        retina_masks=True, 
        stream=True
    )
    frame_id = 0
    for result in results:
        boxes = result.boxes
        masks = result.masks
        frame = result.orig_img
        if boxes is None or masks is None:
            frame_id += 1
            continue
        for mask_data, track_id in zip(masks.data, boxes.id):
            mask_arr = mask_data.cpu().numpy().astype(np.uint8)
            obj_folder = os.path.join(segmented_folder, f"person_{int(track_id)}")
            os.makedirs(obj_folder, exist_ok=True)
            object_img = cv2.bitwise_and(frame, frame, mask=mask_arr)
            cv2.imwrite(os.path.join(obj_folder, f"{frame_id:04d}.png"), object_img)
            # remove object from frame to get background
            frame = cv2.bitwise_and(frame, frame, mask=1-mask_arr)
        # save background
        cv2.imwrite(os.path.join(segmented_folder, f"background_{frame_id:04d}.png"), frame)
        frame_id += 1

def main():
    parser = argparse.ArgumentParser(description='Perform instance segmentation on video.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--segmented_folder', type=str, required=True, help='Folder to save segmented objects.')
    parser.add_argument('--model', type=str, required=True, choices=['sam','yolo'], help='Model to use for segmentation.')
    args = parser.parse_args()

    os.makedirs(args.segmented_folder, exist_ok=True)
    if args.model == 'sam':
        perform_instance_segmentation_sam(args.input_file, args.segmented_folder)
    else:
        perform_instance_segmentation_yolo(args.input_file, args.segmented_folder)

if __name__ == "__main__":
    main()
