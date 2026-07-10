import os
import argparse
import subprocess
import glob
import sys
import multiprocessing as mp
import math
import tempfile
import cv2
import json
import torch
import numpy as np

from src.shared.geometry import get_global_motion
from src.shared.player_extraction import track_persons_iou, match_rackets_to_players
from src.shared.racket_heuristic import interpolate_racket_track
from src.shared import scene_classification

from scripts._compat_patches import apply_compat_patches
apply_compat_patches()

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def process_cpu_phase(args):
    video_path, threads = args
    vname = os.path.splitext(os.path.basename(video_path))[0]
    dataset_dir = os.path.join(REPO_ROOT, 'assets', 'dataset', vname)
    scenes_dir = os.path.join(dataset_dir, 'scenes')
    cache_file = os.path.join(dataset_dir, 'scene_scores.csv')

    # Always call this to handle resuming or completing the extraction
    scene_classification.extract_scene_scores(video_path, cache_file, threads)

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

    Cut detection and point/interlude clustering are delegated to
    `src.shared.scene_classification` (report 10 Phase 1) so the dataset
    script and the runtime encoder can never diverge on scene boundaries.
    """
    vname = os.path.basename(video_path)

    vname_noext = os.path.splitext(vname)[0]
    cache_file = os.path.join(REPO_ROOT, 'assets', 'dataset', vname_noext, 'scene_scores.csv')

    scores_data = scene_classification.load_scene_scores(cache_file)
    if not scores_data:
        print(f"[{device}] [classify_scenes] No scores found for {vname}. Skipping.")
        return

    scene_stats, raw_centroids = scene_classification.detect_and_classify_scenes(video_path, cache_file)

    dataset_dir = os.path.dirname(out_dir)

    print(f"[{device}] [classify_scenes] Starting classification for {vname} with {len(scene_stats)} dynamic scenes...")

    os.makedirs(out_dir, exist_ok=True)

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
        
    if 'segment' in stages or 'pose' in stages or 'skeleton' in stages or 'canny' in stages or 'caption' in stages:
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
                    
                if 'canny' in stages:
                    print(f"[{device}] Extracting canny edges for scene {i}")
                    _extract_canny_worker(video_path, dataset_dir, i, device)
                    
                if 'caption' in stages:
                    print(f"[{device}] Extracting captions for scene {i}")
                    _extract_caption_worker(video_path, dataset_dir, i, device)





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


def _extract_canny_worker(video_path, dataset_dir, scene_idx, device):
    import glob
    import os
    import cv2
    import numpy as np

    seg_dir = os.path.join(dataset_dir, 'segmentations', f'scene_{scene_idx:03d}')
    track_dirs = [d for d in glob.glob(os.path.join(seg_dir, 'track_*')) if os.path.isdir(d) and not d.endswith('_skeleton') and not d.endswith('_canny')]
    for tdir in track_dirs:
        tid = os.path.basename(tdir).split('_')[1]
        out_canny_dir = os.path.join(seg_dir, f'track_{tid}_canny')
        if os.path.exists(out_canny_dir) and len(os.listdir(out_canny_dir)) > 0:
            continue
            
        os.makedirs(out_canny_dir, exist_ok=True)
        frame_files = sorted(glob.glob(os.path.join(tdir, '*.png')))
        for fpath in frame_files:
            fname = os.path.basename(fpath)
            frame = cv2.imread(fpath)
            if frame is None:
                continue
            if frame.shape[2] == 4:
                frame = frame[:, :, :3]
            
            # Canny edge detection
            # Auto threshold based on median
            v = np.median(frame)
            lower = int(max(0, (1.0 - 0.33) * v))
            upper = int(min(255, (1.0 + 0.33) * v))
            edges = cv2.Canny(frame, lower, upper)
            # Expand to 3 channels
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            out_path = os.path.join(out_canny_dir, fname)
            cv2.imwrite(out_path, edges_bgr)


def _extract_caption_worker(video_path, dataset_dir, scene_idx, device):
    import glob
    import os
    import json
    from PIL import Image

    seg_dir = os.path.join(dataset_dir, 'segmentations', f'scene_{scene_idx:03d}')
    track_dirs = [d for d in glob.glob(os.path.join(seg_dir, 'track_*')) if os.path.isdir(d) and not d.endswith('_skeleton') and not d.endswith('_canny')]
    
    # We only load the model if there's actually work to do
    processor = None
    model = None
    
    for tdir in track_dirs:
        tid = os.path.basename(tdir).split('_')[1]
        out_caption_json = os.path.join(seg_dir, f'track_{tid}_caption.json')
        if os.path.exists(out_caption_json):
            continue
            
        frame_files = sorted(glob.glob(os.path.join(tdir, '*.png')))
        if not frame_files:
            continue
            
        first_frame_path = frame_files[0]
        image = Image.open(first_frame_path).convert('RGB')
        
        if processor is None:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            vlm_id = os.path.join(REPO_ROOT, 'assets', 'weights', 'blip-image-captioning-base')
            processor = BlipProcessor.from_pretrained(vlm_id)
            model = BlipForConditionalGeneration.from_pretrained(vlm_id).to(device)
            
        inputs = processor(image, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=50)
        caption = processor.decode(out[0], skip_special_tokens=True)
        
        base_prompt = "photorealistic tennis player, broadcast sports shot"
        full_prompt = f"{caption}, {base_prompt}"
        
        with open(out_caption_json, 'w') as f:
            json.dump({'caption': full_prompt}, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process dataset videos and classify scenes.')
    parser.add_argument('--input', required=True, help='Path to input video file or folder containing .mp4 files')
    parser.add_argument('--threshold', type=float, default=0.3, help='Scene detection threshold (default: 0.3)')
    parser.add_argument('--workers-per-gpu', type=int, default=3, help='Number of worker processes per GPU (default: 3)')
    parser.add_argument('--seg-model', type=str, default='yolo26x-eg.pt', help='Segmentation model to use')
    parser.add_argument('--pose-model', type=str, default='yolo26x-pose.pt', help='Pose model to use for keypoint extraction')
    parser.add_argument('--max-scenes', type=int, default=10, help='Maximum number of point scenes to process per video')
    parser.add_argument('--stages', nargs='+', default=['classify', 'segment', 'pose', 'skeleton', 'canny', 'caption'], help='Stages to run')
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

