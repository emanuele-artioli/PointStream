import os
import time
import subprocess

video_name = os.environ["VIDEO_NAME"]
scene_name = os.environ["SCENE_NAME"]
height = int(os.environ["TARGET_HEIGHT"])
width = int(os.environ["TARGET_WIDTH"])
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
    # Calculate the elapsed time of this task
    start_time = time.time()
    subprocess.call([
        'python', 'pose_estimation.py',
        '--frames_folder', frames_folder,
        '--frames_pose_folder', frames_pose_folder,
        '--frames_pose_csv', frames_pose_csv,
        '--target_height', str(height),
        '--target_width', str(width)
    ])
    elapsed_time = time.time() - start_time
    print(f"Elapsed time for pose estimation: {elapsed_time}")

# Perform instance segmentation on the frames if not already done
frames_segmented_folder = f'/app/frames_segmented/{video_name}/{scene_name}/{height}p'
os.makedirs(frames_segmented_folder, exist_ok=True)
if not os.listdir(frames_segmented_folder):
    # Calculate the elapsed time of this task
    start_time = time.time()
    model = os.environ.get("SEGMENTATION_MODEL", "yolo")  # Default to 'yolo' if not specified
    subprocess.call([
        'python', 'instance_segmentation.py',
        '--frames_folder', frames_folder,
        '--frames_segmented_folder', frames_segmented_folder,
        '--frames_pose_csv', frames_pose_csv,
        '--target_height', str(height),
        '--target_width', str(width),
        '--model', model
    ])
    elapsed_time = time.time() - start_time
    print(f"Elapsed time for instance segmentation: {elapsed_time}")

# Perform jersey number recognition on the frames if not already done
jersey_csv = f'/app/frames_segmented/{video_name}/{scene_name}/jersey_numbers.csv'
if not os.path.isfile(jersey_csv):
    subprocess.call([
        'python', 'jersey_recognition.py',
        '--frames_segmented_folder', frames_segmented_folder,
        '--jersey_csv', jersey_csv
    ])