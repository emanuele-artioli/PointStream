import os
import numpy as np
import subprocess
import cv2
import csv
# import torch
from ultralytics import YOLO
import easyocr

video_name = os.environ["VIDEO_NAME"]
scene_name = os.environ["SCENE_NAME"]
height = int(os.environ["TARGET_HEIGHT"])
input_file = f'/app/input/{video_name}.mp4'

# Detect scenes and split the video into scenes if not already done
scenes_folder = f'/app/scenes/{video_name}'
os.makedirs(scenes_folder, exist_ok=True)
if not os.listdir(scenes_folder):
    scene_threshold = 0.15  # Adjust the threshold as needed
    # First, generate a list of scene change frames
    scene_list_file = os.path.join(scenes_folder, 'scenes.txt')
    subprocess.call([
        'ffmpeg', 
        '-threads', '0',
        '-i', input_file,
        '-vf', f'select=gt(scene\\,{scene_threshold}),metadata=print:file={scene_list_file}',
        '-fps_mode', 'vfr',
        '-frame_pts', 'true',
        '-f', 'null', '-'
    ])

    # Parse the scene list to get frame numbers
    with open(scene_list_file, 'r') as f:
        frames = []
        for line in f:
            if 'pts_time:' in line:
                parts = line.strip().split()
                for part in parts:
                    if part.startswith('pts_time:'):
                        time = float(part.split(':')[1])
                        frames.append(time)

    # Create a list of scene start times
    scene_starts = [0] + frames
    scene_ends = frames + [None]

    # Split the video at scene changes and resize frames
    for idx, (start, end) in enumerate(zip(scene_starts, scene_ends)):
        output_file = f'{scenes_folder}/{idx:03d}.mp4'
        if end is not None:
            duration = end - start
            subprocess.call([
                'ffmpeg', 
                '-threads', '0',
                '-i', input_file,
                '-ss', str(start), '-t', str(duration),
                output_file
            ])
        else:
            subprocess.call([
                'ffmpeg', 
                '-threads', '0',
                '-i', input_file,
                '-ss', str(start),
                output_file
            ])

# Split the requested scene into frames if not already done
frames_folder = f'/app/frames/{video_name}/{scene_name}/{height}p'
os.makedirs(frames_folder, exist_ok=True)
scene_file = f'{scene_name}.mp4'
scene_path = os.path.join(scenes_folder, scene_file)
scene_name, _ = os.path.splitext(scene_file)
if not os.listdir(frames_folder):
    # Extract frames
    subprocess.call([
        'ffmpeg', 
        '-threads', '0',
        '-i', scene_path,
        '-vf', f'scale=-1:{height}',
        os.path.join(frames_folder, '%05d.png')
    ])

# Perform pose estimation on the frames if not already done
frames_pose_folder = f'/app/frames_pose/{video_name}/{scene_name}/{height}p'
frames_pose_csv = f'/app/frames_pose/{video_name}/{scene_name}/poses.csv'
os.makedirs(frames_pose_folder, exist_ok=True)
if not os.listdir(frames_pose_folder):
    # Load YOLO model
    pose_model = YOLO('yolo11n-pose.pt')

    # Open CSV file for writing pose data
    with open(frames_pose_csv, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Updated header to include bounding box coordinates
        csv_writer.writerow(['frame', 'object_id', 'object_type', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'keypoint_id', 'x', 'y'])
        for frame_file in sorted(os.listdir(frames_folder)):
            frame_name, _ = os.path.splitext(frame_file)
            frame_path = os.path.join(frames_folder, frame_file)
            if not frame_file.endswith(".png"):
                continue

            # Read the frame
            frame = cv2.imread(frame_path)

            # Perform pose estimation
            results_pose = pose_model.track(frame, persist=True)[0].numpy()

            # Extract and save pose keypoints and bounding boxes to CSV
            for i, id_obj in enumerate(results_pose.boxes.id):
                # Get bounding box coordinates
                x1, y1, x2, y2 = results_pose.boxes.xyxy[i] # take xywh for width and height
                # Get object type
                object_type = results_pose.boxes.cls[i]

                # Get keypoints for the current object
                for id_kpt, keypoint in enumerate(results_pose.keypoints.xy[i]): # we can also take normalised values to use one pass for different resolutions
                    x_kpt, y_kpt = keypoint
                    csv_writer.writerow([
                        frame_file,        # frame
                        id_obj,            # object_id
                        object_type,       # object_type
                        x1, y1, x2, y2,    # bounding box coordinates
                        id_kpt,            # keypoint_id
                        x_kpt, y_kpt,      # keypoint coordinates
                    ])

            # Save pose estimation result
            pose_frame = results_pose.plot()  # Visualization
            cv2.imwrite(os.path.join(frames_pose_folder, frame_file), pose_frame)

# Perform instance segmentation on the frames if not already done
frames_segmented_folder = f'/app/frames_segmented/{video_name}/{scene_name}/{height}p'
os.makedirs(frames_segmented_folder, exist_ok=True)
if not os.listdir(frames_segmented_folder):
    # Load YOLO model
    segmentation_model = YOLO('yolo11n-seg.pt')
    for frame_file in sorted(os.listdir(frames_folder)):
        frame_name, _ = os.path.splitext(frame_file)
        frame_path = os.path.join(frames_folder, frame_file)
        if not frame_file.endswith(".png"):
            continue

        # Read the frame
        frame = cv2.imread(frame_path)

        # Perform instance segmentation
        results_seg = segmentation_model.track(frame, persist=True)

        # Save segmentation result
        segmented_frame = results_seg[0].plot()  # Visualization
        cv2.imwrite(os.path.join(frames_segmented_folder, frame_file), segmented_frame)

        # Process each detected object
        segmented_objects_folder = os.path.join(frames_segmented_folder, frame_name)
        os.makedirs(segmented_objects_folder, exist_ok=True)
        for i, mask in enumerate(results_seg[0].masks):
            # Ensure the mask is in the correct format
            mask = mask.data[0].numpy().astype(np.uint8)

            # Calculate the aspect ratio of the frame
            frame_aspect_ratio = frame.shape[1] / frame.shape[0]

            # Calculate the target height for the mask to match the aspect ratio
            target_height = int(mask.shape[1] / frame_aspect_ratio)

            # Pad or crop the mask to match the target height
            if target_height > mask.shape[0]:
                # Pad the mask vertically
                pad_height = (target_height - mask.shape[0]) // 2
                mask_padded = np.pad(mask, ((pad_height, target_height - mask.shape[0] - pad_height), (0, 0)), mode='constant', constant_values=0)
            else:
                # Crop the mask vertically
                crop_height = (mask.shape[0] - target_height) // 2
                mask_padded = mask[crop_height:crop_height + target_height, :]

            # Resize the mask to match the dimensions of the frame
            mask_resized = cv2.resize(mask_padded, (frame.shape[1], frame.shape[0]))

            # Extract the object using the resized mask
            object_img = cv2.bitwise_and(frame, frame, mask=mask_resized)

            # Save the object image
            object_output_path = os.path.join(segmented_objects_folder, f'{i}.png')
            cv2.imwrite(object_output_path, object_img)

# Perform jersey number recognition on the frames if not already done
reader = easyocr.Reader(['en'], gpu=False)
jersey_csv = f'/app/frames_segmented/{video_name}/{scene_name}/jersey_numbers.csv'
if not os.path.isfile(jersey_csv):
    # Open CSV file for writing the results
    with open(jersey_csv, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header
        csv_writer.writerow(['image_path', 'jersey_number'])

        # Iterate through subfolders in the parent folder
        for subfolder_name in os.listdir(frames_segmented_folder):
            subfolder_path = os.path.join(frames_segmented_folder, subfolder_name)
            if os.path.isdir(subfolder_path):
                # Process images in this subfolder
                image_files = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                for image_file in image_files:
                    image_path = os.path.join(subfolder_path, image_file)
                    # Read the image
                    image = cv2.imread(image_path)
                    if image is None:
                        continue

                    # Perform OCR on the image
                    result = reader.readtext(image, detail=0)

                    # Extract digits from the OCR result
                    values = []
                    for text in result:
                        digits = ','.join(text)
                        if digits:
                            values.append(digits)

                    # If any numbers were found, write the most probable one to CSV
                    if values:
                        csv_writer.writerow([subfolder_name, image_file, values])

#