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
    video_path, threads = args
    vname = os.path.splitext(os.path.basename(video_path))[0]
    dataset_dir = os.path.join(REPO_ROOT, 'assets', 'dataset', vname)
    scenes_dir = os.path.join(dataset_dir, 'scenes')
    cache_file = os.path.join(dataset_dir, 'scene_scores.csv')
    
    # Always call this to handle resuming or completing the extraction
    extract_scene_scores(video_path, cache_file, threads)
        
    return video_path, scenes_dir

import math
import statistics
import numpy as np
from scipy.signal import find_peaks
from scipy.cluster.vq import kmeans2

def classify_scenes(video_path, out_dir, device):
    """
    Step 1 classification: calculate intra-scene score statistics, and save first/last frames.
    """
    vname = os.path.basename(video_path)
    
    # Load scene scores to compute valley statistics
    vname_noext = os.path.splitext(vname)[0]
    cache_file = os.path.join(REPO_ROOT, 'assets', 'dataset', vname_noext, 'scene_scores.csv')
    scores_data = load_scene_scores(cache_file)
    
    if not scores_data:
        print(f"[{device}] [classify_scenes] No scores found for {vname}. Skipping.")
        return
        
    times = np.array([t for t, s in scores_data])
    scores = np.array([s for t, s in scores_data])
    
    # Dynamic splitting using Dynamic Prominence
    # Dynamic Prominence: base it on the video's global statistics
    # Use raw scores because scene cuts are single-frame spikes that would be erased by filtering
    dyn_prominence = float(np.median(scores) + np.std(scores))
    dyn_prominence = max(0.01, min(dyn_prominence, 0.2)) # clamp to safe bounds
    
    peaks, properties = find_peaks(scores, prominence=dyn_prominence, distance=15)
    
    # The peaks define the scene cuts. Include the start and end of the video.
    base_cuts = [times[p] for p in peaks]
    
    start_t = 0.0
    end_t = times[-1] if len(times) > 0 else 0.0
    
    # Visual boundary check for false cuts
    # Identify cuts that border a scene shorter than 1.0 second
    suspect_cuts = set()
    for i in range(len(base_cuts)):
        t = base_cuts[i]
        prev_t = base_cuts[i-1] if i > 0 else start_t
        next_t = base_cuts[i+1] if i < len(base_cuts)-1 else end_t
        
        if (t - prev_t < 1.0) or (next_t - t < 1.0):
            suspect_cuts.add(t)
            
    valid_cuts = []
    if len(suspect_cuts) > 0:
        from moviepy.editor import VideoFileClip
        import cv2
        clip = None
        try:
            clip = VideoFileClip(video_path)
            for t in base_cuts:
                if t in suspect_cuts:
                    t1 = max(0.0, t - 0.05)
                    t2 = min(end_t, t + 0.05)
                    try:
                        frame1 = clip.get_frame(t1)
                        frame2 = clip.get_frame(t2)
                        
                        # Process frames at low res for speed
                        frame1 = cv2.resize(frame1, (320, 180))
                        frame2 = cv2.resize(frame2, (320, 180))
                        
                        hsv1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2HSV)
                        hsv2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2HSV)
                        
                        # 2D Histogram (Hue and Saturation)
                        hist1 = cv2.calcHist([hsv1], [0, 1], None, [30, 32], [0, 180, 0, 256])
                        hist2 = cv2.calcHist([hsv2], [0, 1], None, [30, 32], [0, 180, 0, 256])
                        
                        cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                        cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                        
                        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                        
                        # If histograms are highly similar, it's a continuous pan, not a hard cut
                        if similarity > 0.85:
                            continue # Discard cut, fusing the scenes
                    except Exception as e:
                        pass # Keep the cut if extraction fails
                valid_cuts.append(t)
        except Exception as e:
            valid_cuts = base_cuts
        finally:
            if clip is not None:
                clip.close()
    else:
        valid_cuts = base_cuts
        
    T = [start_t] + valid_cuts + [end_t]
    T = sorted(list(set(T)))
    
    dataset_dir = os.path.dirname(out_dir)
            
    print(f"[{device}] [classify_scenes] Starting classification for {vname} with {len(T) - 1} dynamic scenes...")
    
    os.makedirs(out_dir, exist_ok=True)
    
    # First pass: Collect stats for all scenes
    scene_stats = []
    for i in range(len(T) - 1):
        t_start = T[i]
        t_end = T[i+1]
        duration = t_end - t_start
        
        idx_start = np.searchsorted(times, t_start, side='right')
        idx_end = np.searchsorted(times, t_end, side='left')
        scene_scores = scores[idx_start:idx_end]
        
        if len(scene_scores) > 0:
            avg_score = float(np.mean(scene_scores))
            std_score = float(np.std(scene_scores))
            max_score = float(np.max(scene_scores))
            med_score = float(np.median(scene_scores))
            energy_score = float(np.sum(scene_scores))
        else:
            avg_score = 0.0
            std_score = 0.0
            max_score = 0.0
            med_score = 0.0
            energy_score = 0.0
            
        # Pre-filter blank scenes
        if max_score < 0.0001:
            classification = "error_blank"
        else:
            classification = "unknown"
            
        scene_stats.append({
            't_start': t_start,
            't_end': t_end,
            'duration': duration,
            'avg_score': avg_score,
            'std_score': std_score,
            'max_score': max_score,
            'med_score': med_score,
            'energy_score': energy_score,
            'classification': classification
        })
        
    from sklearn.preprocessing import RobustScaler
    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    
    # Filter out blank scenes for clustering
    valid_scenes = [s for s in scene_stats if s['classification'] != "error_blank"]
    
    if len(valid_scenes) >= 6: # Need enough scenes for K=6
        # 1. Feature Extraction & Logarithmic Scaling
        features = np.array([
            [np.log10(s['duration'] + 1e-6), 
             np.log10(s['avg_score'] + 1e-6), 
             np.log10(s['std_score'] + 1e-6), 
             np.log10(s['max_score'] + 1e-6),
             np.log10(s['med_score'] + 1e-6),
             np.log10(s['energy_score'] + 1e-6)] 
            for s in valid_scenes
        ])
        
        # 2. Robust Scaling (IQR based, immune to extreme outliers)
        scaler = RobustScaler()
        scaled_features = scaler.fit_transform(features)
        
        # 3. PCA Dimensionality Reduction
        # Collapses highly correlated motion metrics, balancing Motion vs Duration
        pca = PCA(n_components=2, random_state=42)
        pca_features = pca.fit_transform(scaled_features)
        
        # 4. GMM with Auto-K via BIC
        best_gmm = None
        best_bic = np.inf
        
        max_k = min(6, len(valid_scenes) // 2)
        for k in range(2, max_k + 1):
            gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42, n_init=3)
            gmm.fit(pca_features)
            bic = gmm.bic(pca_features)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
                
        labels = best_gmm.predict(pca_features)
        
        for idx, s in enumerate(valid_scenes):
            s['classification'] = f"cluster_{labels[idx]}"
            
        # Identify the high-variance (replay) cluster to target for merging
        # Replay heuristic: highest un-logged std_score
        cluster_replay_scores = {}
        for k in np.unique(labels):
            c_scenes = [s for i, s in enumerate(valid_scenes) if labels[i] == k]
            score = np.mean([s['std_score'] / (s['duration'] + 1e-6) for s in c_scenes])
            cluster_replay_scores[k] = score
            
        replay_cluster_id = max(cluster_replay_scores, key=cluster_replay_scores.get)
        replay_class_name = f"cluster_{replay_cluster_id}"
        
        # 5. Post-cut merging of adjacent replay scenes
        merged_scenes = []
        i = 0
        while i < len(scene_stats):
            s = scene_stats[i]
            
            if s['classification'] != replay_class_name:
                merged_scenes.append(s)
                i += 1
                continue
                
            # Found a replay scene. Look ahead to merge consecutive replays
            j = i + 1
            while j < len(scene_stats):
                if scene_stats[j]['classification'] == replay_class_name:
                    j += 1
                else:
                    break
                    
            if j > i + 1:
                # Merge scenes i to j-1
                t_start_m = scene_stats[i]['t_start']
                t_end_m = scene_stats[j-1]['t_end']
                duration_m = t_end_m - t_start_m
                
                idx_start_m = np.searchsorted(times, t_start_m, side='right')
                idx_end_m = np.searchsorted(times, t_end_m, side='left')
                merged_scores = scores[idx_start_m:idx_end_m]
                
                if len(merged_scores) > 0:
                    avg_m = float(np.mean(merged_scores))
                    std_m = float(np.std(merged_scores))
                    max_m = float(np.max(merged_scores))
                    med_m = float(np.median(merged_scores))
                    energy_m = float(np.sum(merged_scores))
                else:
                    avg_m = 0.0
                    std_m = 0.0
                    max_m = 0.0
                    med_m = 0.0
                    energy_m = 0.0
                    
                merged_scenes.append({
                    't_start': t_start_m,
                    't_end': t_end_m,
                    'duration': duration_m,
                    'avg_score': avg_m,
                    'std_score': std_m,
                    'max_score': max_m,
                    'med_score': med_m,
                    'energy_score': energy_m,
                    'classification': replay_class_name
                })
                i = j
            else:
                merged_scenes.append(s)
                i += 1
                
        scene_stats = merged_scenes
        
        # 6. Compute cluster edges for JSON and exact final confidences
        valid_merged = [s for s in scene_stats if s['classification'] != "error_blank"]
        unique_clusters = list(set([s['classification'] for s in valid_merged]))
        
        raw_centroids = {}
        for cname in unique_clusters:
            c_scenes = [s for s in valid_merged if s['classification'] == cname]
            if not c_scenes: continue
            
            durs = [s['duration'] for s in c_scenes]
            avgs = [s['avg_score'] for s in c_scenes]
            stds = [s['std_score'] for s in c_scenes]
            maxs = [s['max_score'] for s in c_scenes]
            
            raw_centroids[cname] = {
                "count": len(c_scenes),
                "duration": {"mean": float(np.mean(durs)), "min": float(np.min(durs)), "max": float(np.max(durs))},
                "avg_score": {"mean": float(np.mean(avgs)), "min": float(np.min(avgs)), "max": float(np.max(avgs))},
                "std_score": {"mean": float(np.mean(stds)), "min": float(np.min(stds)), "max": float(np.max(stds))},
                "max_score": {"mean": float(np.mean(maxs)), "min": float(np.min(maxs)), "max": float(np.max(maxs))}
            }
            
        # Recalculate GMM probabilities for the newly merged scenes to get perfect confidence scores
        merged_features_raw = np.array([
            [np.log10(s['duration'] + 1e-6), 
             np.log10(s['avg_score'] + 1e-6), 
             np.log10(s['std_score'] + 1e-6), 
             np.log10(s['max_score'] + 1e-6),
             np.log10(s['med_score'] + 1e-6),
             np.log10(s['energy_score'] + 1e-6)] 
            for s in valid_merged
        ])
        
        merged_scaled = scaler.transform(merged_features_raw)
        merged_pca = pca.transform(merged_scaled)
        
        # We don't refit, just predict probability using the established GMM model
        probas = best_gmm.predict_proba(merged_pca)
        
        for i, s in enumerate(valid_merged):
            cname = s['classification']
            try:
                cluster_id = int(cname.replace("cluster_", ""))
                s['cluster_confidence'] = float(probas[i][cluster_id])
            except:
                s['cluster_confidence'] = 0.0
            
        # Determine heuristics & dynamic threshold
        if raw_centroids:
            # The point cluster should be a major cluster. Find clusters with >= 5% of scenes.
            total_scenes = len([s for s in scene_stats if s['classification'] != "error_blank"])
            valid_cnames = [k for k, v in raw_centroids.items() if v['count'] >= total_scenes * 0.05]
            if not valid_cnames:
                valid_cnames = list(raw_centroids.keys())
                
            point_cluster = min(valid_cnames, key=lambda k: raw_centroids[k]['avg_score']['mean'])
            point_mean = raw_centroids[point_cluster]['avg_score']['mean']
            point_max = raw_centroids[point_cluster]['avg_score']['max']
            
            # Dynamic threshold based on the distribution of the point cluster
            # The threshold is a buffer (1.5x) above the maximum motion found inside the point cluster itself.
            # Fallback to 2.0x mean if the cluster is extremely tight.
            dynamic_threshold = max(point_max * 1.5, point_mean * 2.0)
            
            # Map all clusters based on camera movement relative to the dynamic threshold
            target_mapping = {}
            for k, v in raw_centroids.items():
                if v['avg_score']['mean'] > dynamic_threshold:
                    target_mapping[k] = 'cluster_interlude'
                else:
                    target_mapping[k] = 'cluster_point'
            
            conf_scores = [s.get('cluster_confidence', 0.0) for s in scene_stats if s.get('cluster_confidence', 0.0) > 0]
            if conf_scores:
                threshold = np.percentile(conf_scores, 10)
                threshold = min(threshold, 0.98) # cap at 0.98
            else:
                threshold = 0.98
                
            for s in scene_stats:
                if s['classification'] == "error_blank":
                    continue
                cname = s['classification']
                conf = s.get('cluster_confidence', 0.0)
                
                target_class = target_mapping.get(cname, 'cluster_other')
                if conf >= threshold:
                    s['classification'] = target_class
                else:
                    s['classification'] = 'cluster_other'

    else:
        # Robust fallback for < 6 valid scenes
        raw_centroids = {}
        for s in valid_scenes:
            if s['avg_score'] < 0.005 and s['duration'] > 3.0:
                s['classification'] = "cluster_point"
            else:
                s['classification'] = "cluster_other"
            
        for s in scene_stats:
            s['cluster_confidence'] = 0.0
            

    # Third pass: Extract frames with final classification
    for s in scene_stats:
        if s['classification'] == "error_blank": continue
        avg_s = s['avg_score']
        dur = s['duration']
        std_s = s['std_score']
        t_start = s['t_start']
        t_end = s['t_end']
        max_score = s['max_score']
        classification = s['classification']
        cluster_conf = s.get('cluster_confidence', 0.0)
            
        base_name = f"{t_end:08.3f}_dur_{dur:.1f}_avg_{avg_s:.4f}_std_{std_s:.4f}_max_{max_score:.4f}_class_{classification}_conf_{cluster_conf:.2f}"
        frame_path = os.path.join(out_dir, f"{base_name}.jpg")
        
        t_first = t_start + 0.05
        t_last = max(t_start + 0.1, t_end - 0.05)
        
        subprocess.run([
            'ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-hwaccel', 'cuda',
            '-ss', str(t_first), '-i', video_path,
            '-ss', str(t_last), '-i', video_path,
            '-filter_complex', '[0:v][1:v]vstack=inputs=2',
            '-vframes', '1',
            '-q:v', '2',
            '-y', frame_path
        ], check=False)

    # Write JSON output
    import json
    json_data = {
        "video_name": vname,
        "total_scenes": len(scene_stats),
        "cluster_info": raw_centroids,
        "scenes": [
            {
                "t_start": s['t_start'],
                "t_end": s['t_end'],
                "duration": s['duration'],
                "avg_score": s['avg_score'],
                "std_score": s['std_score'],
                "max_score": s['max_score'],
                "cluster": s['classification'],
                "cluster_confidence": s.get('cluster_confidence', 0.0)
            } for s in scene_stats if s['classification'] != "error_blank"
        ]
    }
    with open(os.path.join(dataset_dir, 'dynamic_cuts.json'), 'w') as f:
        json.dump(json_data, f, indent=4)
        
    print(f"[{device}] [classify_scenes] Finished {vname}")

def process_gpu_phase(args):
    video_path, scenes_dir, device = args
    classify_scenes(video_path, scenes_dir, device)


'''
rm -f assets/dataset/*/scenes/*.jpg
python scripts/dataset_processing.py --input assets/raw_4k
'''
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
    
    cpu_args = []
    for video in videos:
        # We need to skip processing if it's already fully extracted, BUT process_cpu_phase handles that.
        cpu_args.append((video, threads_per_video))
        
    cpu_results = []
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=cpu_workers) as pool:
        for result in pool.imap_unordered(process_cpu_phase, cpu_args):
            cpu_results.append(result)
            
    # Phase 2: GPU-bound classification
    print("\n--- PHASE 2: Scene Classification ---")
    gpu_devices = ['cuda:0', 'cuda:1']
    gpu_workers = 6 # Total GPU workers
    print(f"Using {gpu_workers} parallel workers mapped across devices: {gpu_devices}")
    
    gpu_args = []
    for i, res in enumerate(cpu_results):
        video_path, scenes_dir = res
        device = gpu_devices[i % len(gpu_devices)]
        gpu_args.append((video_path, scenes_dir, device))
        
    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=gpu_workers) as pool:
        pool.map(process_gpu_phase, gpu_args)
        
    print("\nDataset processing complete!")
