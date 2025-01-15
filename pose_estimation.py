import os
import cv2
import csv
import argparse
from ultralytics import YOLO

def perform_pose_estimation(frames_folder, frames_pose_folder, frames_pose_csv, height, width):
    # Load YOLO model for pose estimation
    model = YOLO('yolo11n-pose.pt')

    # Open CSV file for writing pose data
    with open(frames_pose_csv, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Updated header to include bounding box coordinates and keypoints
        header = ['frame', 'object_id', 'object_type', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']
        for i in range(17):  # Assuming 17 keypoints
            header.extend([f'keypoint_{i}_x', f'keypoint_{i}_y'])
        csv_writer.writerow(header)

        for frame_file in sorted(os.listdir(frames_folder)):
            frame_name, _ = os.path.splitext(frame_file)
            frame_path = os.path.join(frames_folder, frame_file)
            if not frame_file.endswith(".png"):
                continue

            # Read the frame
            frame = cv2.imread(frame_path)

            # Perform pose estimation
            results_pose = model.track(
                frame, 
                persist=True,
                imgsz=(height//2, width//2), 
                half=True,
                max_det=30,
                classes=[0],
                save=False
            )[0].numpy()

            # Extract and save pose keypoints and bounding boxes to CSV
            for i, id_obj in enumerate(results_pose.boxes.id):
                # Get bounding box coordinates
                x1, y1, x2, y2 = results_pose.boxes.xyxy[i]
                # Get object type
                object_type = results_pose.boxes.cls[i]

                # Prepare row data
                row = [frame_file, int(id_obj), object_type, x1, y1, x2, y2]
                for keypoint in results_pose.keypoints.xy[i]:
                    row.extend(keypoint)

                csv_writer.writerow(row)

                # Plot the whole frame with keypoints and edges
                pose_frame = results_pose.plot(line_width=1, boxes=False, masks=False)

                # Create a folder for each frame
                frame_output_folder = os.path.join(frames_pose_folder, frame_name)
                os.makedirs(frame_output_folder, exist_ok=True)

                # Crop the player image from the whole pose frame
                player_img = pose_frame[int(y1):int(y2), int(x1):int(x2)].copy()

                # Save the player image with keypoints and edges
                player_output_path = os.path.join(frame_output_folder, f'{int(id_obj)}.png')
                cv2.imwrite(player_output_path, player_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform pose estimation on video frames.')
    parser.add_argument('--frames_folder', type=str, required=True, help='Path to the folder containing video frames.')
    parser.add_argument('--frames_pose_folder', type=str, required=True, help='Path to the folder to save pose estimation results.')
    parser.add_argument('--frames_pose_csv', type=str, required=True, help='Path to the CSV file to save pose estimation data.')
    parser.add_argument('--target_height', type=int, required=True, help='Target processing height for the video frames.')
    parser.add_argument('--target_width', type=int, required=True, help='Target processing width for the video frames.')
    args = parser.parse_args()

    perform_pose_estimation(args.frames_folder, args.frames_pose_folder, args.frames_pose_csv, args.target_height, args.target_width)
