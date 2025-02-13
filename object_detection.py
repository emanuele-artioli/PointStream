import os
import cv2
import argparse
import csv
from ultralytics import YOLO
import numpy as np

def perform_object_detection(input_file, segmented_folder, N_saved=2):
    model = YOLO('yolo11n.pt')
    results = model.track(
        input_file, 
        conf=0.15, 
        # tracker='bytetrack.yaml', # uses BoT-SORT by default
        imgsz=640, 
        classes=[0], 
        retina_masks=True, 
        stream=True
    )
    
    # Create CSV file and write header
    csv_file = os.path.join(segmented_folder, 'box_coordinates.csv')
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['frame_id', 'object_id', 'x1', 'y1', 'x2', 'y2'])
    
    frame_id = 0
    last_known_boxes = {}  # Dictionary to store the last known bounding boxes for each object ID
    
    for result in results:
        boxes = result.boxes
        frame = result.orig_img
        if boxes is None:
            frame_id += 1
            continue
        
        mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255  # Create a white mask
        
        current_boxes = {}
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            obj_id = int(box.id)
            current_boxes[obj_id] = (x1, y1, x2, y2)
            if obj_id <= N_saved:  # Only save the first N_saved objects
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([frame_id, obj_id, x1, y1, x2, y2])
                obj_folder = os.path.join(segmented_folder, f"person_{obj_id}")
                os.makedirs(obj_folder, exist_ok=True)
                object_img = frame[y1:y2, x1:x2]
                cv2.imwrite(os.path.join(obj_folder, f"{frame_id:04d}.png"), object_img)
                
                # Update the mask to cut out the bounding box area
                mask[y1:y2, x1:x2] = 0
        
        # Use the last known bounding boxes for objects not found in the current frame
        for obj_id, (x1, y1, x2, y2) in last_known_boxes.items():
            if obj_id not in current_boxes and obj_id <= N_saved:
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([frame_id, obj_id, x1, y1, x2, y2])
                obj_folder = os.path.join(segmented_folder, f"person_{obj_id}")
                os.makedirs(obj_folder, exist_ok=True)
                object_img = frame[y1:y2, x1:x2]
                cv2.imwrite(os.path.join(obj_folder, f"{frame_id:04d}.png"), object_img)
                
                # Update the mask to cut out the bounding box area
                mask[y1:y2, x1:x2] = 0
        
        # Apply the mask to the frame to remove the bounding box areas
        background = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imwrite(os.path.join(segmented_folder, f"background_{frame_id:04d}.png"), background)
        
        # Update the last known bounding boxes
        last_known_boxes.update(current_boxes)
        
        frame_id += 1

def main():
    parser = argparse.ArgumentParser(description='Perform instance segmentation on video.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--detected_folder', type=str, required=True, help='Folder to save detected images and CSV file.')
    args = parser.parse_args()
    
    perform_object_detection(args.input_file, args.detected_folder)

if __name__ == "__main__":
    main()
