import os
import numpy as np
import cv2
import argparse
from ultralytics import SAM, YOLO

def perform_instance_segmentation(detected_folder, segmented_folder, model=None):
    for sub in os.listdir(detected_folder):
        # skip files, only process folders
        detected_person = os.path.join(detected_folder, sub)
        if not os.path.isdir(detected_person):
            continue

        results = model.predict(
            detected_person, 
            conf=0.15, 
            imgsz=640, 
            classes=[0], 
            max_det=1, 
            retina_masks=True, 
            stream=True
        )

        for frame_id, result in enumerate(results):
            masks = result.masks
            frame = result.orig_img
            if masks is None:
                continue
            mask_arr = masks.data[0].cpu().numpy().astype(np.uint8)
            output_folder = os.path.join(segmented_folder, sub)
            os.makedirs(output_folder, exist_ok=True)
            object_img = cv2.bitwise_and(frame, frame, mask=mask_arr)
            cv2.imwrite(os.path.join(output_folder, f"{frame_id:04d}.png"), object_img)

def main():
    parser = argparse.ArgumentParser(description='Perform instance segmentation on video.')
    parser.add_argument('--detected_folder', type=str, required=True, help='Path to the input folder.')
    parser.add_argument('--segmented_folder', type=str, required=True, help='Folder to save segmented objects.')
    parser.add_argument('--model', type=str, required=True, choices=['sam','yolo'], help='Model to use for segmentation.')
    args = parser.parse_args()

    os.makedirs(args.segmented_folder, exist_ok=True)
    if args.model == 'sam':
        perform_instance_segmentation(args.detected_folder, args.segmented_folder, model=SAM('sam2.1_b.pt'))
    else:
        perform_instance_segmentation(args.detected_folder, args.segmented_folder, model=YOLO('yolo11n-seg.pt'))

if __name__ == "__main__":
    main()
