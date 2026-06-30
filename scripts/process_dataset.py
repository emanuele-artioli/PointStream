import os
import argparse
import subprocess
import glob
import re
import csv
import sys
import multiprocessing as mp
import math
import tempfile
import cv2
import json
import torch
import numpy as np
from scipy.signal import find_peaks

from src.shared.geometry import get_global_motion
from src.shared.player_extraction import track_persons_iou, match_rackets_to_players
from src.shared.racket_heuristic import interpolate_racket_track

from scripts._compat_patches import apply_compat_patches
apply_compat_patches()

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

def segment_and_fuse_scene(video_path, dataset_dir, scene_idx, t_start, t_end, device, seg_model, debug_video=False):
    import multiprocessing as mp
    ctx = mp.get_context('spawn')
    p = ctx.Process(target=_segment_and_fuse_scene_worker, args=(video_path, dataset_dir, scene_idx, t_start, t_end, device, seg_model, debug_video))
    p.start()
    p.join()
    if p.exitcode != 0:
        print(f"[{device}] Scene segmentation failed for scene {scene_idx}")

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
                    except Exception:
                        pass # Keep the cut if extraction fails
                valid_cuts.append(t)
        except Exception:
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
            if not c_scenes:
                continue
            
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
            except Exception:
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
        if s['classification'] == "error_blank":
            continue
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
        
        if os.path.exists(frame_path):
            continue
            
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
    with open(os.path.join(dataset_dir, 'scene_metadata.json'), 'w') as f:
        json.dump(json_data, f, indent=4)
        
    print(f"[{device}] [classify_scenes] Finished {vname}")


def process_gpu_phase(args):
    video_path, scenes_dir, device, stages, pose_model, debug_video, seg_model, max_scenes = args
    vname = os.path.splitext(os.path.basename(video_path))[0]
    dataset_dir = os.path.join(REPO_ROOT, 'assets', 'dataset', vname)
    json_path = os.path.join(dataset_dir, 'scene_metadata.json')

    if 'classify' in stages:
        if os.path.exists(json_path):
            print(f"[{device}] Classification for {vname} already exists. Skipping.")
        else:
            classify_scenes(video_path, scenes_dir, device)
        
    if 'segment' in stages or 'pose' in stages or 'skeleton' in stages:
        if not os.path.exists(json_path):
            print(f"[{device}] No cuts found for {vname}, skipping downstream.")
            return
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        scenes = data.get('scenes', [])
        point_scenes_processed = 0
        for i, s in enumerate(scenes):
            if s['cluster'] == 'cluster_point':
                if point_scenes_processed >= max_scenes:
                    continue
                point_scenes_processed += 1
                if 'segment' in stages:
                    seg_dir = os.path.join(dataset_dir, 'segmentations', f'scene_{i:03d}')
                    if os.path.exists(seg_dir) and len(os.listdir(seg_dir)) > 0:
                        print(f"[{device}] Segmentations for scene {i} already exist. Skipping.")
                    else:
                        print(f"[{device}] Segmenting scene {i} ({s['t_start']:.2f}s - {s['t_end']:.2f}s)")
                        segment_and_fuse_scene(video_path, dataset_dir, i, s['t_start'], s['t_end'], device, seg_model, debug_video)
                
                if 'pose' in stages:
                    print(f"[{device}] Extracting pose for scene {i}")
                    _extract_pose_worker(video_path, dataset_dir, i, device, pose_model)
                    
                if 'skeleton' in stages:
                    print(f"[{device}] Rendering skeleton for scene {i}")
                    _render_skeleton_worker(video_path, dataset_dir, i, device)





'''
rm -f assets/dataset/*/scenes/*.jpg
python scripts/dataset_processing.py --input assets/raw_4k
'''

def _segment_and_fuse_scene_worker(video_path, dataset_dir, scene_idx, t_start, t_end, device, seg_model, debug_video=False):

    
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    weights_path = os.path.join(REPO_ROOT, 'assets', 'weights', seg_model)
    
    # Save segs in a common 'segmentations' folder
    seg_dir = os.path.join(dataset_dir, 'segmentations', f'scene_{scene_idx:03d}')
    videos_dir = os.path.join(dataset_dir, 'videos')
    os.makedirs(seg_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    
    from ultralytics import YOLO
    model = YOLO(weights_path)
    model.to(device)
    if hasattr(model, 'set_classes'):
        try:
            model.set_classes(["person", "tennis racket", "tennis ball"])
        except Exception:
            pass
            
    fps = 24
    
    with tempfile.TemporaryDirectory() as tmp_frames_dir:
        cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-ss', str(t_start),
            '-i', video_path,
            '-t', str(t_end - t_start),
            '-r', '24',
            os.path.join(tmp_frames_dir, 'frame_%06d.png')
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[{device}] Error extracting frames for scene {scene_idx}: {e}")
            return
            
        frame_files = sorted(glob.glob(os.path.join(tmp_frames_dir, '*.png')))
        if not frame_files:
            return
            
        # We will map each tracked object to a single consistent track ID
        active_tracks = {}
        
        prev_gray = None
        cumulative_dx, cumulative_dy = 0.0, 0.0
        
        global_centroids = {}
        racket_scores = {}
        parsed_dets = []
        cumulative_rackets = 0
            
        for frame_id, fpath in enumerate(frame_files):
            img = cv2.imread(fpath)
            if img is None:
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                dx, dy = get_global_motion(prev_gray, gray)
                cumulative_dx += dx
                cumulative_dy += dy
            prev_gray = gray
                
            h, w = img.shape[:2]
            inf_size = min(max(h, w), 1920)
            
            results = model.predict(img, verbose=False, imgsz=inf_size)
            result = results[0]
            
            frame_dets = []
            
            persons = []
            rackets = []
            other_dets = []
            if result.boxes is not None and len(result.boxes) > 0:
                for i, box in enumerate(result.boxes):
                    cls_idx = int(box.cls[0])
                    cls_name = result.names[cls_idx].lower()
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy().tolist())
                    bbox = (x1, y1, x2, y2)
                    mask_data = None
                    if result.masks is not None:
                        m = np.array(result.masks.data[i].cpu().tolist(), dtype=np.float32)
                        mask_data = (m > 0.5).astype(np.uint8) * 255
                    d = {'cls': cls_name, 'conf': conf, 'bbox': bbox, 'mask': mask_data}
                    if cls_name == 'person':
                        persons.append(d)
                    elif cls_name == 'tennis racket':
                        rackets.append(d)
                    else:
                        other_dets.append(d)
            
            persons, active_tracks = track_persons_iou(persons, active_tracks, frame_id, max_gap=30)
            
            for p in persons:
                tid = p['tid']
                cx = (p['bbox'][0] + p['bbox'][2]) / 2.0
                cy = (p['bbox'][1] + p['bbox'][3]) / 2.0
                g_cx = cx - cumulative_dx
                g_cy = cy - cumulative_dy
                global_centroids.setdefault(tid, []).append((g_cx, g_cy))
            
            match_rackets_to_players(persons, rackets, strategy="accumulate", racket_scores=racket_scores)
            
            frame_dets = persons + rackets + other_dets
            parsed_dets.append(frame_dets)
            cumulative_rackets += len(rackets)
            
            if frame_id == 240 and cumulative_rackets == 0:
                print(f"[{device}] Scene {scene_idx} aborted early: 0 rackets found in first 30s.")
                return
            
        if not parsed_dets:
            return
            
        print(f"[{device}] YOLO prediction finished for scene {scene_idx} ({len(frame_files)} frames).")
        
        movement_scores = {}
        for tid, points in global_centroids.items():
            if len(points) < 5:
                movement_scores[tid] = 0
            else:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                var = np.std(xs) + np.std(ys)
                movement_scores[tid] = float(var)
                
        # Normalize movement scores to combine with racket scores
        max_movement = max(movement_scores.values()) if movement_scores else 1.0
        if max_movement == 0:
            max_movement = 1.0
        
        combined_scores = {}
        all_tids = set(racket_scores.keys()).union(set(movement_scores.keys()))
        for tid in all_tids:
            # Racket score is dominant (each racket assoc is a big deal)
            # Give it a heavy weight so 1 racket assoc > any amount of movement
            r_score = racket_scores.get(tid, 0) * 1000.0
            m_score = (movement_scores.get(tid, 0) / max_movement) * 100.0
            combined_scores[tid] = r_score + m_score

        sorted_tids = sorted(combined_scores.keys(), key=lambda t: combined_scores[t], reverse=True)
        player_tids = set(sorted_tids[:2])
        
        print(f"[{device}] Scene {scene_idx} - Selected TIDs: {player_tids}")
        
        track_crops = {}
        
        for frame_id, fpath in enumerate(frame_files):
            img = cv2.imread(fpath)
            if img is None:
                continue
            dets = parsed_dets[frame_id]
            persons = []
            rackets = []
            
            for det in dets:
                x1, y1, x2, y2 = det['bbox']
                x1, y1 = max(0, x1), max(0, y1)
                
                if det['cls'] == 'person' and det['tid'] in player_tids:
                    persons.append(det)
                elif det['cls'] == 'tennis racket':
                    rackets.append(det)
                    
            fused_tracks = []
            for p in persons:
                px1, py1, px2, py2 = p['bbox']
                
                best_racket = None
                best_iou = 0
                for r in rackets:
                    rx1, ry1, rx2, ry2 = r['bbox']
                    inter_x1 = max(px1, rx1)
                    inter_y1 = max(py1, ry1)
                    inter_x2 = min(px2, rx2)
                    inter_y2 = min(py2, ry2)
                    if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                        r_area = (rx2 - rx1) * (ry2 - ry1)
                        iou = inter_area / r_area
                        if iou > best_iou:
                            best_iou = iou
                            best_racket = r
                            
                fused_tracks.append({
                    'tid': p['tid'],
                    'person': p,
                    'racket': best_racket if best_iou > 0.1 else None,
                    'ball': None 
                })
                
            for track in fused_tracks:
                tid = track['tid']
                p_bbox = track['person']['bbox']
                r_bbox = track['racket']['bbox'] if track['racket'] else None
                
                p_mask = track['person']['mask']
                if p_mask is not None and p_mask.shape != img.shape[:2]:
                    p_mask = cv2.resize(p_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    
                if r_bbox and track['racket']['mask'] is not None:
                    r_mask = track['racket']['mask']
                    if r_mask.shape != img.shape[:2]:
                        r_mask = cv2.resize(r_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    if p_mask is not None:
                        p_mask = cv2.bitwise_or(p_mask, r_mask)
                    else:
                        p_mask = r_mask
                
                min_x = max(0, p_bbox[0])
                min_y = max(0, p_bbox[1])
                max_x = min(img.shape[1], p_bbox[2])
                max_y = min(img.shape[0], p_bbox[3])
                
                if r_bbox:
                    min_x = max(0, min(min_x, r_bbox[0]))
                    min_y = max(0, min(min_y, r_bbox[1]))
                    max_x = min(img.shape[1], max(max_x, r_bbox[2]))
                    max_y = min(img.shape[0], max(max_y, r_bbox[3]))
                
                crop = img[min_y:max_y, min_x:max_x].copy()
                crop_mask = p_mask[min_y:max_y, min_x:max_x] if p_mask is not None else None
                
                if crop_mask is not None:
                    b, g, r_ch = cv2.split(crop)
                    crop_rgba = cv2.merge([b, g, r_ch, crop_mask])
                else:
                    b, g, r_ch = cv2.split(crop)
                    alpha = np.ones((crop.shape[0], crop.shape[1]), dtype=np.uint8) * 255
                    crop_rgba = cv2.merge([b, g, r_ch, alpha])
                    
                track_dir = os.path.join(seg_dir, f'track_{tid:04d}')
                os.makedirs(track_dir, exist_ok=True)
                out_path = os.path.join(track_dir, f'frame_{frame_id:06d}.png')
                cv2.imwrite(out_path, crop_rgba)
                
                if (tid, 'fused') not in track_crops:
                    track_crops[(tid, 'fused')] = []
                
                if r_bbox:
                    rx1, ry1, rx2, ry2 = r_bbox
                    racket_bbox_crop = [
                        (rx1 - min_x),
                        (ry1 - min_y),
                        (rx2 - min_x),
                        (ry2 - min_y)
                    ]
                    racket_mask_points = None
                    if track['racket']['mask'] is not None:
                        contours, _ = cv2.findContours(r_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            c = max(contours, key=cv2.contourArea)
                            if len(c) >= 5:
                                hull = cv2.convexHull(c)
                                max_dist = -1
                                p1, p2 = None, None
                                for i in range(len(hull)):
                                    for j in range(i+1, len(hull)):
                                        pt1 = hull[i][0]
                                        pt2 = hull[j][0]
                                        dist = (pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2
                                        if dist > max_dist:
                                            max_dist = dist
                                            p1 = tuple(pt1)
                                            p2 = tuple(pt2)
                                
                                if p1 and p2:
                                    vx = p2[0] - p1[0]
                                    vy = p2[1] - p1[1]
                                    length = math.hypot(vx, vy)
                                    if length > 0:
                                        ux, uy = vx / length, vy / length
                                        nx, ny = -uy, ux
                                        
                                        max_proj = -float('inf')
                                        min_proj = float('inf')
                                        p3, p4 = None, None
                                        
                                        for pt_arr in c:
                                            pt = pt_arr[0]
                                            proj = pt[0] * nx + pt[1] * ny
                                            if proj > max_proj:
                                                max_proj = proj
                                                p3 = tuple(pt)
                                            if proj < min_proj:
                                                min_proj = proj
                                                p4 = tuple(pt)
                                                
                                        if p3 and p4:
                                            racket_mask_points = {
                                                'p1': (float(p1[0] - min_x), float(p1[1] - min_y)),
                                                'p2': (float(p2[0] - min_x), float(p2[1] - min_y)),
                                                'p3': (float(p3[0] - min_x), float(p3[1] - min_y)),
                                                'p4': (float(p4[0] - min_x), float(p4[1] - min_y)),
                                            }
                else:
                    racket_bbox_crop = None
                    racket_mask_points = None

                track_crops[(tid, 'fused')].append({
                    'frame_id': frame_id,
                    'bbox': (min_x, min_y, max_x, max_y),
                    'racket_bbox_crop': racket_bbox_crop,
                    'racket_mask_points': racket_mask_points
                })

        for (tid, cls_name), items in track_crops.items():
            if len(items) < 5:
                continue
                
            track_meta_file = os.path.join(seg_dir, f'track_{tid:04d}_metadata.json')
            meta_data = []
            for item in items:
                meta_data.append({
                    'frame_id': item['frame_id'],
                    'bbox': item['bbox'],
                    'racket_bbox_crop': item.get('racket_bbox_crop'),
                    'racket_mask_points': item.get('racket_mask_points')
                })
            with open(track_meta_file, 'w') as f:
                json.dump(meta_data, f, indent=2)
                
            if debug_video:
                out_video = os.path.join(videos_dir, f'scene_{scene_idx:03d}_track_{tid:04d}_{cls_name}_debug.mp4')
                cmd = [
                    'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                    '-framerate', str(fps),
                    '-i', os.path.join(seg_dir, f'track_{tid:04d}', 'frame_%06d.png'),
                    '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                    '-c:v', 'libx264',
                    '-preset', 'fast',
                    '-crf', '22',
                    '-pix_fmt', 'yuv420p',
                    out_video
                ]
                subprocess.run(cmd, check=False)


def _extract_pose_worker(video_path, dataset_dir, scene_idx, device, pose_model_name):
    seg_dir = os.path.join(dataset_dir, 'segmentations', f'scene_{scene_idx:03d}')
    videos_dir = os.path.join(dataset_dir, 'videos')
    if not os.path.exists(videos_dir):
        return
    
    import glob
    import json
    from ultralytics import YOLO
    from src.shared.player_extraction import coco17_to_dwpose18
    
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    weights_path = os.path.join(REPO_ROOT, 'assets', 'weights', pose_model_name)
    
    model = None
    
    track_dirs = [d for d in glob.glob(os.path.join(seg_dir, 'track_*')) if os.path.isdir(d) and not d.endswith('_skeleton')]
    for tdir in track_dirs:
        tid = os.path.basename(tdir).split('_')[1]
        out_json = os.path.join(seg_dir, f'track_{tid}_keypoints.json')
        if os.path.exists(out_json):
            continue
            
        if model is None:
            model = YOLO(weights_path)
            model.to(device)
            
        import cv2
        frame_files = sorted(glob.glob(os.path.join(tdir, '*.png')))
        all_kpts = []
        for frame_idx, fpath in enumerate(frame_files):
            frame = cv2.imread(fpath)
            if frame is None:
                all_kpts.append({'frame_id': frame_idx, 'keypoints': None})
                continue
                
            # Drop alpha channel for YOLO
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]
                
            results = model.predict(frame, verbose=False)
            if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                xy = results[0].keypoints.xy[0].cpu().numpy()
                conf = results[0].keypoints.conf[0].cpu().numpy()
                coco17 = np.concatenate([xy, conf[:, None]], axis=-1)
                dw18 = coco17_to_dwpose18(coco17)
                all_kpts.append({
                    'frame_id': frame_idx,
                    'keypoints': dw18.tolist()
                })
            else:
                all_kpts.append({
                    'frame_id': frame_idx,
                    'keypoints': None
                })
            
        with open(out_json, 'w') as f:
            json.dump(all_kpts, f, indent=2)

def _render_skeleton_worker(video_path, dataset_dir, scene_idx, device):
    videos_dir = os.path.join(dataset_dir, 'videos')
    if not os.path.exists(videos_dir):
        return
    
    import glob
    import json
    import numpy as np
    import cv2
    from src.shared.racket_heuristic import render_pose_with_racket
    
    seg_dir = os.path.join(dataset_dir, 'segmentations', f'scene_{scene_idx:03d}')
    track_dirs = glob.glob(os.path.join(seg_dir, 'track_*'))
    for tdir in track_dirs:
        if tdir.endswith('_skeleton'):
            continue
        tid = os.path.basename(tdir).split('_')[1]
        
        meta_path = os.path.join(seg_dir, f'track_{tid}_metadata.json')
        kpt_path = os.path.join(seg_dir, f'track_{tid}_keypoints.json')
        out_skel_dir = os.path.join(seg_dir, f'track_{tid}_skeleton')
        
        if os.path.exists(out_skel_dir):
            continue
            
        if not os.path.exists(meta_path) or not os.path.exists(kpt_path):
            continue
            
        with open(meta_path, 'r') as f:
            meta_data = json.load(f)
            
        with open(kpt_path, 'r') as f:
            kpt_data = json.load(f)
            
        if len(meta_data) != len(kpt_data):
            continue
            
        meta_data = interpolate_racket_track(meta_data)
                        
        from src.shared.racket_heuristic import get_dominant_wrist
        hand_votes = {4: 0, 7: 0}
        for m, k in zip(meta_data, kpt_data):
            kpts = k['keypoints']
            racket_bbox = m.get('racket_bbox_crop')
            if kpts is not None and racket_bbox is not None:
                kpts_np = np.array(kpts, dtype=np.float32)
                wrist_info = get_dominant_wrist(kpts_np, tuple(racket_bbox))
                if wrist_info is not None:
                    hand_votes[wrist_info[2]] += 1
                    
        majority_hand = None
        if hand_votes[4] > hand_votes[7]:
            majority_hand = 4
        elif hand_votes[7] > hand_votes[4]:
            majority_hand = 7

        os.makedirs(out_skel_dir, exist_ok=True)
        for i, (m, k) in enumerate(zip(meta_data, kpt_data)):
            kpts = k['keypoints']
            racket_bbox = m.get('racket_bbox_crop')
            
            # Use original bbox to determine canvas size since crops are native res
            w = m['bbox'][2] - m['bbox'][0]
            h = m['bbox'][3] - m['bbox'][1]
            
            if kpts is not None:
                kpts_np = np.array(kpts, dtype=np.float32)
                if racket_bbox is not None:
                    racket_bbox = tuple(racket_bbox)
                racket_mask_points = m.get('racket_mask_points')
                canvas = render_pose_with_racket(kpts_np, racket_bbox, int(h), int(w), dominant_hand=majority_hand, racket_mask_points=racket_mask_points)
            else:
                canvas = np.zeros((int(h), int(w), 3), dtype=np.uint8)
                
            frame_path = os.path.join(out_skel_dir, f'frame_{i:06d}.png')
            cv2.imwrite(frame_path, canvas)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process dataset videos and classify scenes.')
    parser.add_argument('--input', required=True, help='Path to input video file or folder containing .mp4 files')
    parser.add_argument('--threshold', type=float, default=0.3, help='Scene detection threshold (default: 0.3)')
    parser.add_argument('--workers-per-gpu', type=int, default=3, help='Number of worker processes per GPU (default: 3)')
    parser.add_argument('--seg-model', type=str, default='yolo26x-eg.pt', help='Segmentation model to use')
    parser.add_argument('--pose-model', type=str, default='yolo26x-pose.pt', help='Pose model to use for keypoint extraction')
    parser.add_argument('--max-scenes', type=int, default=10, help='Maximum number of point scenes to process per video')
    parser.add_argument('--stages', nargs='+', default=['classify', 'segment', 'pose', 'skeleton'], help='Stages to run')
    parser.add_argument('--debug-video', action='store_true', help='Generate AVC debug videos for stitched tracks')
    
    args = parser.parse_args()
    input_path = args.input
    stages = args.stages
    pose_model = args.pose_model
    seg_model = args.seg_model
    max_scenes = args.max_scenes
    debug_video = args.debug_video
    
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
        print("No .mp4 files found to process.")
        sys.exit(0)

    import torch
    num_cpus = os.cpu_count() or 4
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    num_videos = len(videos)

    print("\n=======================================================")
    print("Hardware Resources:")
    print(f"  CPU Cores: {num_cpus}")
    print(f"  GPUs:      {num_gpus}")
    print(f"Videos to process: {num_videos}")
    print("=======================================================\n")

    # Phase 1: CPU-Bound Processing
    # Distribute threads across videos. If fewer videos than CPUs, use multiple threads per video.
    threads_per_video = max(1, math.floor(num_cpus / num_videos))
    cpu_workers = min(num_videos, num_cpus)
    
    print("--- PHASE 1: FFmpeg Extraction ---")
    print(f"Using {cpu_workers} parallel workers, allocating {threads_per_video} threads per video.")
    
    cpu_args = []
    for video in videos:
        cpu_args.append((video, threads_per_video))
        
    cpu_results = []
    if 'classify' in stages:
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=cpu_workers) as pool:
            for result in pool.imap_unordered(process_cpu_phase, cpu_args):
                cpu_results.append(result)
    else:
        # Just populate results if skipping CPU phase
        for video in videos:
            vname = os.path.splitext(os.path.basename(video))[0]
            dataset_dir = os.path.join(REPO_ROOT, 'assets', 'dataset', vname)
            scenes_dir = os.path.join(dataset_dir, 'scenes')
            cpu_results.append((video, scenes_dir))
            
    # Phase 2: GPU-bound classification and segmentation
    print("\n--- PHASE 2: Scene Classification ---")
    import torch
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_devices = [f'cuda:{i}' for i in range(max(1, num_gpus))]
    gpu_workers = max(1, num_gpus) * args.workers_per_gpu
    print(f"Using {gpu_workers} parallel workers mapped across devices: {gpu_devices}")
    
    gpu_args = []
    for i, res in enumerate(cpu_results):
        video_path, scenes_dir = res
        device = gpu_devices[i % len(gpu_devices)]
        gpu_args.append((video_path, scenes_dir, device, stages, pose_model, debug_video, seg_model, max_scenes))
    try:
        from multiprocessing.dummy import Pool as ThreadPool
        with ThreadPool(gpu_workers) as pool:
            pool.map(process_gpu_phase, gpu_args)
    except Exception as e:
        print(f"Error during GPU phase: {e}")
        
    print("\n--- PHASE 3: Dataset Metadata Generation ---")
    import json
    from pathlib import Path
    meta_entries = []
    
    # We processed multiple videos. The root is assets/dataset
    dataset_root = os.path.join(REPO_ROOT, 'assets', 'dataset')
    search_pattern = os.path.join(dataset_root, "*", "segmentations", "scene_*", "track_*")
    all_tracks = glob.glob(search_pattern)
    for track_dir_str in all_tracks:
        if track_dir_str.endswith("_skeleton"):
            continue
        track_dir = Path(track_dir_str)
        skel_dir = track_dir.with_name(f"{track_dir.name}_skeleton")
        if not skel_dir.exists():
            continue
        color_frames = list(track_dir.glob("frame_*.png"))
        if len(color_frames) < 2:
            continue
        meta_entries.append({
            "video_path": str(track_dir.resolve()),
            "kps_path": str(skel_dir.resolve())
        })
    
    out_json = os.path.join(dataset_root, "pointstream_aa_meta.json")
    with open(out_json, "w") as f:
        json.dump(meta_entries, f, indent=4)
    print(f"Generated Animate Anyone meta info with {len(meta_entries)} sequences at {out_json}")

    print("\nDataset processing complete!")

