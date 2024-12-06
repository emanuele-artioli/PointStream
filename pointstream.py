import os
import subprocess
import cv2
# import torch
from ultralytics import YOLO

video_name = os.environ["VIDEO_NAME"]
scene_name = os.environ["SCENE_NAME"]
height = int(os.environ["TARGET_HEIGHT"])
input_file = f'/app/input/{video_name}.mp4'
scenes_folder = f'/app/scenes/{video_name}'
frames_folder = f'/app/frames/{video_name}/{scene_name}/{height}p'

# Create necessary folders
os.makedirs(scenes_folder, exist_ok=True)
os.makedirs(frames_folder, exist_ok=True)

# Detect scenes and split the video into scenes if not already done
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
        '-c:v', 'libx264', 
        os.path.join(frames_folder, '%05d.png')
    ])

# Load YOLO models
segmentation_model = YOLO('yolo11n-seg.pt')
pose_model = YOLO('yolo11n-pose.pt')

# Perform instance segmentation and pose estimation on the frames if not already done
frames_segmented_folder = f'/app/frames_segmented/{video_name}/{scene_name}/{height}p'
frames_pose_folder = f'/app/frames_pose/{video_name}/{scene_name}/{height}p'
os.makedirs(frames_segmented_folder, exist_ok=True)
os.makedirs(frames_pose_folder, exist_ok=True)
if not os.listdir(frames_segmented_folder):
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
            # Extract the object using the mask
            object_img = cv2.bitwise_and(frame, frame, mask=mask.data[0].numpy())

            # Save the object image
            object_output_path = os.path.join(segmented_objects_folder, f'object_{i}.png')
            cv2.imwrite(object_output_path, object_img)

        # Perform pose estimation
        results_pose = pose_model(frame)
        pose_frame = results_pose[0].plot()  # Visualization

        # Save pose estimation result
        cv2.imwrite(os.path.join(frames_pose_folder, frame_file), pose_frame)