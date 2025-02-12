import os
import subprocess
import sys

video_path = sys.argv[1]
video_name = os.path.splitext(os.path.basename(video_path))[0]
input_file = video_path

# Detect scenes and split the video into scenes if not already done
scenes_folder = os.path.join(os.getcwd(), "scenes", video_name)
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