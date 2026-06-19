import argparse
import os
import subprocess
import glob
import re
import csv
import sys
import multiprocessing as mp
import math

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_video_duration(video_path):
    cmd_dur = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    try:
        return float(subprocess.check_output(cmd_dur, text=True).strip())
    except Exception:
        return 0.0

def extract_scene_scores(video_path, cache_file, threads=1, chunk_duration=60):
    duration = get_video_duration(video_path)
    if duration == 0:
        print(f"[extract_scene_scores] Could not get duration for {video_path}")
        return video_path, cache_file
        
    last_processed_time = 0.0
    
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            reader = csv.DictReader(f)
            last_row = None
            for row in reader:
                last_row = row
            if last_row is not None:
                last_processed_time = float(last_row['pts_time'])
                print(f"[extract_scene_scores] Resuming {os.path.basename(video_path)} from {last_processed_time:.2f}s")
    else:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['pts_time', 'scene_score'])
            
    current_time = last_processed_time
    
    while current_time < duration:
        end_time = min(duration, current_time + chunk_duration)
        start_time = max(0.0, current_time - 1.0)
        
        print(f"[extract_scene_scores] Processing {os.path.basename(video_path)}: {current_time:.2f}s to {end_time:.2f}s")
        
        cmd = [
            'ffmpeg', '-hide_banner', '-hwaccel', 'cuda', '-threads', str(threads),
            '-ss', str(start_time), '-copyts',
            '-i', video_path,
            '-to', str(end_time),
            '-filter:v', "scale=320:-1,select='gte(scene,0)',metadata=print:key=lavfi.scene_score",
            '-f', 'null', '-'
        ]
        
        process = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
        
        scores = []
        current_time_val = None
        for line in process.stderr:
            time_match = re.search(r'pts_time:([0-9.]+)', line)
            if time_match:
                current_time_val = float(time_match.group(1))
                
            score_match = re.search(r'lavfi\.scene_score=([0-9.]+)', line)
            if score_match and current_time_val is not None:
                t = current_time_val
                score = float(score_match.group(1))
                if t > current_time:
                    scores.append((t, score))
                current_time_val = None
                        
        process.wait()
        
        if scores:
            with open(cache_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(scores)
            current_time = scores[-1][0]
        else:
            current_time = end_time
            
    print(f"[extract_scene_scores] Finished extracting frames for {os.path.basename(video_path)}")
    return video_path, cache_file

def load_scene_scores(cache_file):
    scores = []
    with open(cache_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append((float(row['pts_time']), float(row['scene_score'])))
    return scores

def process_cpu_phase(args):
    video_path, threshold, threads = args
    vname = os.path.splitext(os.path.basename(video_path))[0]
    dataset_dir = os.path.join(REPO_ROOT, 'assets', 'dataset', vname)
    scenes_dir = os.path.join(dataset_dir, 'scenes')
    cache_file = os.path.join(dataset_dir, 'scene_scores.csv')
    
    # Always call this to handle resuming or completing the extraction
    extract_scene_scores(video_path, cache_file, threads)
        
    scores = load_scene_scores(cache_file)
    base_cuts = [t for t, score in scores if score > threshold]
    
    # Save the cuts to a text file for user inspection
    with open(os.path.join(dataset_dir, f'base_cuts_{threshold:.2f}.txt'), 'w') as f:
        for t in base_cuts:
            f.write(f"{t}\n")
    
    start_t = 0.0
    end_t = scores[-1][0] if scores else 0.0
    
    T = [start_t] + base_cuts + [end_t]
    T = sorted(list(set(T)))
    
    os.makedirs(scenes_dir, exist_ok=True)
    return video_path, T, scenes_dir

def classify_scenes(video_path, cuts, out_dir, device):
    """
    Skeleton function for the new decision tree.
    """
    vname = os.path.basename(video_path)
    print(f"[{device}] [classify_scenes] Starting classification for {vname} with {len(cuts) - 1} scenes...")
    
    # TODO: Implement new classification tree from scratch
    # We have `cuts` which are the timestamps separating the scenes
    
    pass

def process_gpu_phase(args):
    video_path, T, scenes_dir, device = args
    classify_scenes(video_path, T, scenes_dir, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process dataset videos and classify scenes.')
    parser.add_argument('--input', required=True, help='Path to input video file or folder containing .mp4 files')
    parser.add_argument('--threshold', type=float, default=0.3, help='Scene detection threshold (default: 0.3)')
    parser.add_argument('--workers-per-gpu', type=int, default=3, help='Number of worker processes per GPU (default: 3)')
    
    args = parser.parse_args()
    input_path = args.input
    
    if not os.path.exists(input_path):
        print(f"Error: Input path '{input_path}' does not exist.")
        sys.exit(1)
        
    videos = []
    if os.path.isdir(input_path):
        videos = glob.glob(os.path.join(input_path, '*.mp4'))
    else:
        if not input_path.lower().endswith('.mp4'):
            print("Warning: Input file does not have .mp4 extension.")
        videos = [input_path]
        
    if not videos:
        print(f"No .mp4 files found to process.")
        sys.exit(0)

    import torch
    num_cpus = os.cpu_count() or 4
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    num_videos = len(videos)

    print(f"\n=======================================================")
    print(f"Hardware Resources:")
    print(f"  CPU Cores: {num_cpus}")
    print(f"  GPUs:      {num_gpus}")
    print(f"Videos to process: {num_videos}")
    print(f"=======================================================\n")

    # Phase 1: CPU-Bound Processing
    # Distribute threads across videos. If fewer videos than CPUs, use multiple threads per video.
    threads_per_video = max(1, math.floor(num_cpus / num_videos))
    cpu_workers = min(num_videos, num_cpus)
    
    print(f"--- PHASE 1: FFmpeg Extraction ---")
    print(f"Using {cpu_workers} parallel workers, allocating {threads_per_video} threads per video.")
    
    cpu_args = [(v, args.threshold, threads_per_video) for v in videos]
    
    results = []
    # Use spawn context to be safe if mixing with PyTorch later
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=cpu_workers) as pool:
        results = pool.map(process_cpu_phase, cpu_args)
        
    # Phase 2: GPU-Bound Processing
    if num_gpus > 0:
        available_gpus = [f"cuda:{i}" for i in range(num_gpus)]
    else:
        available_gpus = ["cpu"]
        
    workers_per_gpu = args.workers_per_gpu
    gpu_workers = min(num_videos, len(available_gpus) * workers_per_gpu)
    
    print(f"\n--- PHASE 2: Scene Classification ---")
    print(f"Using {gpu_workers} parallel workers mapped across devices: {available_gpus}")
    
    gpu_args = []
    for i, res in enumerate(results):
        video_path, T, scenes_dir = res
        device = available_gpus[i % len(available_gpus)]
        gpu_args.append((video_path, T, scenes_dir, device))
        
    with ctx.Pool(processes=gpu_workers) as pool:
        pool.map(process_gpu_phase, gpu_args)
        
    print("\nDataset processing complete!")
