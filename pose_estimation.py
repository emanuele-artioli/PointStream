import os
import cv2
import csv
import argparse
from ultralytics import YOLO

def extract_pose_detections(result):
    """Extract keypoints and bounding boxes for each detected person."""
    boxes = getattr(result, 'boxes', [])
    keypoints_data = getattr(result, 'keypoints', None)
    detections = []
    for i, box in enumerate(boxes):
        obj_id = int(box.id) if box.id is not None else 9999
        bbox = tuple(map(int, box.xyxy[0]))
        conf = float(box.conf)
        keypoints = []
        if keypoints_data is not None and i < len(keypoints_data.xy):
            keypoints = keypoints_data.xy[i].cpu().numpy().tolist()
        detections.append({
            'id': obj_id,
            'conf': conf,
            'bbox': bbox,
            'keypoints': keypoints
        })
    return detections

def remove_objects_from_frame(frame_img, detections):
    """Remove detected people from the frame by setting their bounding boxes to black."""
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        frame_img[y1:y2, x1:x2] = 0

def main():
    parser = argparse.ArgumentParser(description='Perform pose estimation on a video.')
    parser.add_argument('--video_file', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--experiment_folder', type=str, required=True, help='Folder to save keypoints and background.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model')
    parser.add_argument('--model', type=str, default='models/yolo11m-pose.pt', help='Path to the pose model file.')
    args = parser.parse_args()

    model = YOLO(args.model)
    background_folder = os.path.join(args.experiment_folder, 'background')
    os.makedirs(background_folder, exist_ok=True)

    # Create CSV file for writing keypoints
    pose_csv_path = os.path.join(args.experiment_folder, 'keypoints.csv')
    with open(pose_csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write CSV header: frame_id, object_id, confidence, plus pairs of x,y per keypoint
        header = ['frame_id', 'object_id', 'confidence']
        # Assume up to 17 keypoints as in COCO body
        for i in range(17):
            header.extend([f'keypoint_{i}_x', f'keypoint_{i}_y'])
        csv_writer.writerow(header)

        # Perform pose estimation in a streaming fashion
        inf_results = model.track(
            source=args.video_file,
            conf=0.25,
            iou=0.2,
            imgsz=1280,
            device=args.device,
            classes=[0],  # person class
            stream=True,
            persist=True,
            max_det=10
        )

        frame_id = 0
        for frame_id, result in enumerate(inf_results):
            frame_id += 1
            frame_img = result.orig_img
            detections = extract_pose_detections(result)

            # Write keypoints to CSV
            for det in detections:
                row = [
                    frame_id,
                    det['id'],
                    f"{det['conf']:.2f}",
                ]
                # Add keypoints data (x,y)
                for kp in det['keypoints']:
                    row.extend([kp[0], kp[1]])
                csv_writer.writerow(row)

            # Remove detected bounding boxes from the background
            remove_objects_from_frame(frame_img, detections)

            # Save background image
            cv2.imwrite(os.path.join(background_folder, f'{frame_id}.png'), frame_img)

if __name__ == "__main__":
    main()