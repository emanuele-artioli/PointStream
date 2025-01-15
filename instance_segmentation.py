import os
import cv2
import csv
import argparse
from ultralytics import SAM, YOLO

def perform_instance_segmentation_sam(frames_folder, frames_segmented_folder, frames_pose_csv, height, width):
    # Load segmentation model
    model = SAM('sam2.1_b.pt')

    # Read bounding boxes from the CSV file
    bboxes = {}
    with open(frames_pose_csv, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        _ = next(csv_reader)  # Skip the header
        for row in csv_reader:
            frame, object_id, object_type, bbox_x1, bbox_y1, bbox_x2, bbox_y2 = row[:7]
            if frame not in bboxes:
                bboxes[frame] = {}
            if object_id not in bboxes[frame]:
                bboxes[frame][object_id] = (float(bbox_x1), float(bbox_y1), float(bbox_x2), float(bbox_y2))

    for frame_file in sorted(os.listdir(frames_folder)):
        frame_name, _ = os.path.splitext(frame_file)
        frame_path = os.path.join(frames_folder, frame_file)
        if not frame_file.endswith(".png"):
            continue

        # Read the frame
        frame = cv2.imread(frame_path)

        # Process each detected object
        segmented_objects_folder = os.path.join(frames_segmented_folder, frame_name)
        os.makedirs(segmented_objects_folder, exist_ok=True)
        for object_id, (x1, y1, x2, y2) in bboxes.get(frame_file, {}).items():
            # Crop the frame based on the bounding box
            cropped_frame = frame[int(y1):int(y2), int(x1):int(x2)]

            # Perform instance segmentation on the cropped frame
            results_seg = model.track(
                cropped_frame, 
                persist=True,
                imgsz=(height//2, width//2), 
                half=True,
                max_det=1,
                classes=[0],
                save=False
            )[0].cpu().numpy()  # Move tensor to CPU before converting to NumPy array

            # Save segmentation result
            segmented_frame = results_seg[0].plot()  # Visualization
            object_output_path = os.path.join(segmented_objects_folder, f'{object_id}.png')
            cv2.imwrite(object_output_path, segmented_frame)

def perform_instance_segmentation_yolo(frames_folder, frames_segmented_folder, frames_pose_csv, height, width):
    # Load segmentation model
    model = YOLO('yolo11n-seg.pt')

    # Read bounding boxes from the CSV file
    bboxes = {}
    with open(frames_pose_csv, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        _ = next(csv_reader)  # Skip the header
        for row in csv_reader:
            frame, object_id, object_type, bbox_x1, bbox_y1, bbox_x2, bbox_y2 = row[:7]
            if frame not in bboxes:
                bboxes[frame] = {}
            if object_id not in bboxes[frame]:
                bboxes[frame][object_id] = (float(bbox_x1), float(bbox_y1), float(bbox_x2), float(bbox_y2))

    for frame_file in sorted(os.listdir(frames_folder)):
        frame_name, _ = os.path.splitext(frame_file)
        frame_path = os.path.join(frames_folder, frame_file)
        if not frame_file.endswith(".png"):
            continue

        # Read the frame
        frame = cv2.imread(frame_path)

        # Process each detected object
        segmented_objects_folder = os.path.join(frames_segmented_folder, frame_name)
        os.makedirs(segmented_objects_folder, exist_ok=True)
        for object_id, (x1, y1, x2, y2) in bboxes.get(frame_file, {}).items():
            # Crop the frame based on the bounding box
            cropped_frame = frame[int(y1):int(y2), int(x1):int(x2)]

            # Perform instance segmentation on the cropped frame
            results_seg = model.predict(
                cropped_frame, 
                imgsz=(height//2, width//2), 
                half=True,
                max_det=1,
                classes=[0],
                save=False
            )[0]

            # Save segmentation result
            # segmented_frame = results_seg[0].plot()  # Visualization
            mask = results_seg[0].masks.data[0].cpu().numpy()  # Get the segmentation mask
            mask = (mask * 255).astype('uint8')  # Convert mask to uint8 format
            mask = cv2.resize(mask, (cropped_frame.shape[1], cropped_frame.shape[0]))  # Resize mask to match cropped frame
            segmented_frame = cv2.bitwise_and(cropped_frame, cropped_frame, mask=mask)  # Apply mask to the cropped frame
            object_output_path = os.path.join(segmented_objects_folder, f'{object_id}.png')
            cv2.imwrite(object_output_path, segmented_frame)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform instance segmentation on video frames.')
    parser.add_argument('--frames_folder', type=str, required=True, help='Path to the folder containing video frames.')
    parser.add_argument('--frames_segmented_folder', type=str, required=True, help='Path to the folder to save segmented results.')
    parser.add_argument('--frames_pose_csv', type=str, required=True, help='Path to the CSV file containing pose estimation data.')
    parser.add_argument('--target_height', type=int, required=True, help='Target processing height for the video frames.')
    parser.add_argument('--target_width', type=int, required=True, help='Target processing width for the video frames.')
    parser.add_argument('--model', type=str, required=True, choices=['sam', 'yolo'], help='Model to use for segmentation.')
    args = parser.parse_args()

    if args.model == 'sam':
        perform_instance_segmentation_sam(args.frames_folder, args.frames_segmented_folder, args.frames_pose_csv, args.target_height, args.target_width)
    elif args.model == 'yolo':
        perform_instance_segmentation_yolo(args.frames_folder, args.frames_segmented_folder, args.frames_pose_csv, args.target_height, args.target_width)
