import os
import csv
import argparse
from ultralytics import YOLO

# can be sped up with multiprocessing
# tells when no objects are detected, which can be used to avoid sending the image to client
# def perform_pose_estimation(segmented_objects_folder, pose_csv):
#     model = YOLO('yolo11n-pose.pt')

#     with open(pose_csv, mode='w', newline='') as csv_file:
#         csv_writer = csv.writer(csv_file)
#         header = ['frame', 'object_id', 'object_type', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']
#         for i in range(17):
#             header.extend([f'keypoint_{i}_x', f'keypoint_{i}_y'])
#         csv_writer.writerow(header)

#         # For each object subfolder, parse images and run YOLO pose
#         for obj_folder in sorted(os.listdir(segmented_objects_folder)):
#             full_obj_path = os.path.join(segmented_objects_folder, obj_folder)
#             if not os.path.isdir(full_obj_path):
#                 continue

#             # Use folder name (e.g., "object_4" or "object_sam_0_1") as object_id
#             object_id = obj_folder

#             for frame_file in sorted(os.listdir(full_obj_path)):
#                 if not frame_file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                     continue

#                 frame_path = os.path.join(full_obj_path, frame_file)
#                 frame = cv2.imread(frame_path)
#                 if frame is None:
#                     continue

#                 results = model.predict(
#                     frame,
#                     imgsz=640,
#                     classes=[0],
#                     max_det=2, 
#                     retina_masks=True,
#                 )[0]

#                 for i, box in enumerate(results.boxes):
#                     x1, y1, x2, y2 = box.xyxy[0]
#                     object_type = box.cls[0]
#                     row = [frame_file, object_id, object_type, x1.item(), y1.item(), x2.item(), y2.item()]
#                     # Append keypoints
#                     if results.keypoints is not None:
#                         for kp in results.keypoints.xy[i]:
#                             row.extend(kp.tolist())
#                     csv_writer.writerow(row)

#                 # # Optionally save pose visualization
#                 # pose_frame = results.plot(line_width=1, boxes=False, masks=False)
#                 # frame_output_folder = os.path.join(os.path.dirname(pose_csv), obj_folder, 'pose_vis')
#                 # os.makedirs(frame_output_folder, exist_ok=True)
#                 # cv2.imwrite(os.path.join(frame_output_folder, frame_file), pose_frame)

def perform_pose_estimation(segmented_folder, pose_csv):
    model = YOLO('yolo11n-pose.pt')

    # Create CSV file and write header
    with open(pose_csv, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        header = ['player', 'frame']
        for i in range(17):
            header.extend([f'keypoint_{i}_x', f'keypoint_{i}_y'])
        csv_writer.writerow(header)

        for sub in os.listdir(segmented_folder):
            # skip files, only process folders
            detected_person = os.path.join(segmented_folder, sub)
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
                row = [sub, frame_id]
                # Append keypoints
                if result.keypoints is not None:
                    for kp in result.keypoints.xy[0]:
                        row.extend(kp.tolist())
                csv_writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(description='Perform pose estimation on segmented object frames.')
    parser.add_argument('--segmented_folder', type=str, required=True, help='Folder containing subfolders of object frames.')
    parser.add_argument('--pose_csv', type=str, required=True, help='Path to the CSV file to save pose estimation data.')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.pose_csv), exist_ok=True)
    perform_pose_estimation(args.segmented_folder, args.pose_csv)

if __name__ == "__main__":
    main()
