import os
import cv2
import argparse
import csv
from ultralytics import YOLO

def perform_object_detection(input_file, segmented_folder, N_saved=2):
    model = YOLO('yolo11n.pt')
    results = model.track(
        input_file, 
        conf=0.5, 
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
    for result in results:
        boxes = result.boxes
        frame = result.orig_img
        if boxes is None:
            frame_id += 1
            continue
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            obj_id = int(box.id)
            if obj_id <= N_saved:  # Only save the first N_saved objects
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([frame_id, obj_id, x1, y1, x2, y2])
                obj_folder = os.path.join(segmented_folder, f"person_{obj_id}")
                os.makedirs(obj_folder, exist_ok=True)
                object_img = frame[y1:y2, x1:x2]
                cv2.imwrite(os.path.join(obj_folder, f"{frame_id:04d}.png"), object_img)
        frame_id += 1

def main():
    parser = argparse.ArgumentParser(description='Perform instance segmentation on video.')
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--detected_folder', type=str, required=True, help='Folder to save detected images and CSV file.')
    args = parser.parse_args()
    
    perform_object_detection(args.input_file, args.detected_folder)

if __name__ == "__main__":
    main()
