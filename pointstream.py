import os
import subprocess

video_name = os.environ["VIDEO_NAME"]
input_file = f'/app/input/{video_name}.mp4'
scenes_folder = f'/app/scenes/{video_name}'
frames_root_folder = f'/app/frames/{video_name}'

# Create necessary folders
os.makedirs(scenes_folder, exist_ok=True)
os.makedirs(frames_root_folder, exist_ok=True)

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

    # Split the video at scene changes
    for idx, (start, end) in enumerate(zip(scene_starts, scene_ends)):
        output_file = f'{scenes_folder}/{idx:03d}.mp4'
        if end is not None:
            duration = end - start
            subprocess.call([
                'ffmpeg', 
                '-threads', '0',
                '-i', input_file,
                '-ss', str(start), '-t', str(duration),
                '-c', 'copy', output_file
            ])
        else:
            subprocess.call([
                'ffmpeg', 
                '-threads', '0',
                '-i', input_file,
                '-ss', str(start),
                '-c', 'copy', output_file
            ])

# Process each scene if not already done
for scene_file in sorted(os.listdir(scenes_folder)):
    scene_path = os.path.join(scenes_folder, scene_file)
    scene_name, _ = os.path.splitext(scene_file)
    frames_folder = os.path.join(frames_root_folder, scene_name)
    os.makedirs(frames_folder, exist_ok=True)
    # Check if frames already exist
    if not os.listdir(frames_folder):
        # Extract frames
        subprocess.call([
            'ffmpeg', 
            '-threads', '0',
            '-i', scene_path,
            os.path.join(frames_folder, '%05d.png')
        ])

