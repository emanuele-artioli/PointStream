import os
import subprocess
import sys
import csv
import pandas as pd
import numpy as np

def detect_camera_movement(video_path, sensitivity=8):
    '''
    Detect frames where the camera moves using vidstabdetect.
    
    Args:
        video_path: Path to the video file
        sensitivity: Higher values detect smaller movements (1-15)
        
    Returns:
        List of tuples (timestamp, confidence) for camera movements
    '''
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create directory for camera movement data
    camera_dir = os.path.join(os.getcwd(), "camera_movement")
    os.makedirs(camera_dir, exist_ok=True)
    
    # Generate transform file using vidstabdetect
    transform_file = os.path.join(camera_dir, f"{video_name}_transforms.trf")
    subprocess.call([
        'ffmpeg',
        '-threads', '0',
        '-i', video_path,
        '-vf', f'vidstabdetect=stepsize=6:shakiness={sensitivity}:accuracy=15:result={transform_file}',
        '-f', 'null', '-'
    ], stderr=subprocess.DEVNULL)
    
    # Get video frame rate for timestamp conversion
    fps_cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    output = subprocess.check_output(fps_cmd).decode('utf-8').strip()
    
    # Calculate fps (handling fractional fps like 30000/1001)
    if '/' in output:
        num, den = map(int, output.split('/'))
        fps = num / den
    else:
        fps = float(output)
    
    # Parse the transform file to identify camera movements
    camera_movements = []
    prev_motion = 0
    
    with open(transform_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
                
            parts = line.strip().split()
            if len(parts) >= 8:  # Ensure we have enough data
                frame_num = int(parts[0])
                # Extract translation and rotation values
                tx, ty = float(parts[1]), float(parts[2])
                angle = float(parts[3])
                
                # Calculate motion as a combination of translation and rotation
                motion = np.sqrt(tx**2 + ty**2) + abs(angle) * 5
                
                # Detect significant change in motion
                motion_diff = abs(motion - prev_motion)
                if motion_diff > 10.0:  # Threshold for camera movement
                    timestamp = frame_num / fps
                    camera_movements.append((timestamp, motion_diff))
                
                prev_motion = motion
    
    return camera_movements

def detect_scene_changes(video_path, threshold=0.15):
    '''
    Detect scene changes in the video.
    
    Args:
        video_path: Path to the video file
        threshold: Scene change threshold (default: 0.15)
        
    Returns:
        List of tuples (timestamp, confidence) for scene changes
    '''
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    scenes_dir = os.path.join(os.getcwd(), "scenes")
    os.makedirs(scenes_dir, exist_ok=True)
    
    # Generate a file with scene change information
    scene_info_file = os.path.join(scenes_dir, f"{video_name}_scene_info.txt")
    subprocess.call([
        'ffmpeg', 
        '-threads', '0',
        '-i', video_path,
        '-vf', f'select=gt(scene\\,{threshold}),metadata=print:file={scene_info_file}:key=lavfi.scene_score',
        '-f', 'null', '-'
    ], stderr=subprocess.DEVNULL)
    
    # Parse the scene info to get timestamps and scene scores
    scene_changes = []
    with open(scene_info_file, 'r') as f:
        for line in f:
            if 'pts_time:' in line and 'lavfi.scene_score=' in line:
                pts_time = None
                scene_score = None
                
                parts = line.strip().split()
                for part in parts:
                    if part.startswith('pts_time:'):
                        pts_time = float(part.split(':')[1])
                    if part.startswith('lavfi.scene_score='):
                        scene_score = float(part.split('=')[1])
                
                if pts_time is not None and scene_score is not None:
                    scene_changes.append((pts_time, scene_score))
    
    return scene_changes

def generate_event_csv(video_path, scene_changes, camera_movements):
    '''
    Generate a CSV file with all detected events.
    
    Args:
        video_path: Path to the video file
        scene_changes: List of (timestamp, confidence) for scene changes
        camera_movements: List of (timestamp, confidence) for camera movements
        
    Returns:
        Path to the generated CSV file
    '''
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    events_dir = os.path.join(os.getcwd(), "events")
    os.makedirs(events_dir, exist_ok=True)
    
    csv_path = os.path.join(events_dir, f"{video_name}_events.csv")
    
    # Combine all events
    events = []
    
    for timestamp, confidence in scene_changes:
        events.append({
            'timestamp': timestamp,
            'type': 'scene_change',
            'confidence': confidence
        })
    
    for timestamp, confidence in camera_movements:
        events.append({
            'timestamp': timestamp,
            'type': 'camera_movement',
            'confidence': confidence
        })
    
    # Sort events by timestamp
    events.sort(key=lambda x: x['timestamp'])
    
    # Write to CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['timestamp', 'type', 'confidence'])
        writer.writeheader()
        writer.writerows(events)
    
    return csv_path

def cut_video_by_events(video_path, csv_path, output_dir=None, cut_on_scene=True, cut_on_camera=False):
    '''
    Cut the video based on events in the CSV file.
    
    Args:
        video_path: Path to the video file
        csv_path: Path to the CSV file with events
        output_dir: Directory to save the cut segments (default: auto-generated)
        cut_on_scene: Whether to cut on scene changes
        cut_on_camera: Whether to cut on camera movements
        
    Returns:
        Path to the directory with cut segments
    '''
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "segments", video_name)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Read events from CSV
    df = pd.read_csv(csv_path)
    
    # Filter events based on cut preferences
    cut_events = []
    for _, row in df.iterrows():
        if (row['type'] == 'scene_change' and cut_on_scene) or \
           (row['type'] == 'camera_movement' and cut_on_camera):
            cut_events.append(row['timestamp'])
    
    # Add 0 as the first cut point if it's not already there
    if 0.0 not in cut_events:
        cut_events.insert(0, 0.0)
    
    # Sort cut points
    cut_events.sort()
    
    # Get video duration
    duration_cmd = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    video_duration = float(subprocess.check_output(duration_cmd).decode('utf-8').strip())
    
    # Create pairs of start and end times
    segments = []
    for i in range(len(cut_events)):
        start = cut_events[i]
        end = video_duration if i == len(cut_events) - 1 else cut_events[i+1]
        segments.append((start, end))
    
    # Cut the video into segments
    for idx, (start, end) in enumerate(segments):
        output_file = f'{output_dir}/{idx:03d}.mp4'
        duration = end - start
        
        subprocess.call([
            'ffmpeg', 
            '-threads', '0',
            '-i', video_path,
            '-ss', str(start),
            '-t', str(duration),
            '-c:v', 'libx264',
            '-preset', 'fast',
            output_file
        ], stderr=subprocess.DEVNULL)
    
    return output_dir

def main():
    if len(sys.argv) < 2:
        print("Usage: python split_scenes.py video_path [options]")
        print("Options:")
        print("  --scene-threshold=FLOAT   Threshold for scene detection (default: 0.15)")
        print("  --camera-sensitivity=INT  Sensitivity for camera movement (1-15, default: 8)")
        print("  --cut-on-scene=BOOL       Cut on scene changes (default: True)")
        print("  --cut-on-camera=BOOL      Cut on camera movements (default: False)")
        sys.exit(1)
    
    video_path = sys.argv[1]
    
    # Parse additional options
    scene_threshold = 0.15
    camera_sensitivity = 8
    cut_on_scene = True
    cut_on_camera = False
    
    for arg in sys.argv[2:]:
        if arg.startswith('--scene-threshold='):
            scene_threshold = float(arg.split('=')[1])
        elif arg.startswith('--camera-sensitivity='):
            camera_sensitivity = int(arg.split('=')[1])
        elif arg.startswith('--cut-on-scene='):
            cut_on_scene = arg.split('=')[1].lower() == 'true'
        elif arg.startswith('--cut-on-camera='):
            cut_on_camera = arg.split('=')[1].lower() == 'true'
    
    print(f"Processing video: {video_path}")
    print(f"Scene threshold: {scene_threshold}, Camera sensitivity: {camera_sensitivity}")
    print(f"Cut on scene: {cut_on_scene}, Cut on camera: {cut_on_camera}")
    
    # Detect events
    print("Detecting scene changes...")
    scene_changes = detect_scene_changes(video_path, threshold=scene_threshold)
    print(f"Found {len(scene_changes)} scene changes")
    
    print("Detecting camera movements...")
    camera_movements = detect_camera_movement(video_path, sensitivity=camera_sensitivity)
    print(f"Found {len(camera_movements)} camera movements")
    
    # Generate CSV
    print("Generating events CSV...")
    csv_path = generate_event_csv(video_path, scene_changes, camera_movements)
    print(f"CSV generated: {csv_path}")
    
    # Cut video
    print("Cutting video...")
    output_dir = cut_video_by_events(
        video_path, 
        csv_path, 
        cut_on_scene=cut_on_scene, 
        cut_on_camera=cut_on_camera
    )
    print(f"Video segments saved to: {output_dir}")

if __name__ == "__main__":
    main()