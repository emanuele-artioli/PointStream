import argparse
import os
import glob
import subprocess
import cv2
import numpy as np
import torch
import shutil

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_available_gpu():
    if torch.cuda.is_available():
        return 0
    return -1

def mode_filter(args: argparse.Namespace) -> None:
    """
    Stage 1: filter
    Reads the raw video with ffmpeg at 4fps in native resolution.
    Saves the extracted frames to assets/dataset/<video_name>/frames/
    """
    video_path = args.input
    vname = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = os.path.join(REPO_ROOT, 'assets', 'dataset', vname, 'frames')
    
    if os.path.exists(frames_dir):
        print(f'[filter] Removing existing frames directory: {frames_dir}')
        shutil.rmtree(frames_dir)
    os.makedirs(frames_dir, exist_ok=True)
    
    cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
        '-i', video_path, 
        '-vf', 'fps=8',
        os.path.join(frames_dir, 'frame_%06d.png')
    ]
    
    print(f'[filter] Extracting frames to {frames_dir} at 8 fps...')
    subprocess.run(cmd, check=True)
    print('[filter] Done.')

def mode_segment(args: argparse.Namespace) -> None:
    """
    Stage 2: segment
    Reads frames from the filter stage, passes them to yoloe-26x-seg.pt (at 1080p).
    Tracks up to 5 people, 2 rackets, and 1 ball.
    Saves detection crops and masks as 4-channel RGBA PNGs to assets/dataset/<video_name>/segmentations/
    """
    video_path = args.input
    vname = os.path.splitext(os.path.basename(video_path))[0]
    base_dir = os.path.join(REPO_ROOT, 'assets', 'dataset', vname)
    frames_dir = os.path.join(base_dir, 'frames')
    seg_dir = os.path.join(base_dir, 'segmentations')
    
    if not os.path.exists(frames_dir):
        print(f'[segment] Frames dir not found: {frames_dir}. Run the filter stage first.')
        return
        
    if os.path.exists(seg_dir):
        print(f'[segment] Removing existing segmentations directory: {seg_dir}')
        shutil.rmtree(seg_dir)
    os.makedirs(seg_dir, exist_ok=True)
    
    from ultralytics import YOLO
    
    gpu = get_available_gpu()
    device = f'cuda:{gpu}' if gpu >= 0 else 'cpu'
    weights_path = os.path.join(REPO_ROOT, 'assets', 'weights', 'yoloe-26x-seg.pt')
    
    print(f'[segment] Loading YOLO model from {weights_path} to {device}...')
    model = YOLO(weights_path)
    model.to(device)
    
    if hasattr(model, 'set_classes'):
        try:
            model.set_classes(["person", "tennis racket", "tennis ball"])
        except Exception as e:
            print(f'[segment] Warning: set_classes failed: {e}')

    frame_files = sorted(glob.glob(os.path.join(frames_dir, 'frame_*.png')))
    if not frame_files:
        print('[segment] No frames found in the frames directory.')
        return

    print(f'[segment] Processing {len(frame_files)} frames...')
    empty_frames_streak = 0
    track_history = {} # tid -> (prev_gray, prev_mask)
    movement_scores = {} # tid -> float
    player_tids = set()
    burn_in_counter = 0
    saved_person_files = {} # tid -> [file_path, ...]
    
    for frame_path in frame_files:
        frame_id_str = os.path.splitext(os.path.basename(frame_path))[0].split('_')[1]
        frame_id = int(frame_id_str)
        
        img = cv2.imread(frame_path)
        if img is None: 
            continue
        
        h, w = img.shape[:2]
        inf_size = min(max(h, w), 1920)
        
        results = model.track(img, persist=True, verbose=False, imgsz=inf_size)
        result = results[0]
        
        if result.boxes is None or len(result.boxes) == 0:
            empty_frames_streak += 1
            if empty_frames_streak > 5:
                print(f'[segment] 5 empty frames in a row at frame {frame_id}. Flushing tracker state.')
                # Completely reinitialize the model to flush the tracker safely
                model = YOLO(weights_path)
                model.to(device)
                if hasattr(model, 'set_classes'):
                    try:
                        model.set_classes(["person", "tennis racket", "tennis ball"])
                    except Exception:
                        pass
                empty_frames_streak = 0
                track_history.clear()
                movement_scores.clear()
                player_tids.clear()
                burn_in_counter = 0
                saved_person_files.clear()
            continue
            
        empty_frames_streak = 0
        dets = []
        
        for i, box in enumerate(result.boxes):
            cls_idx = int(box.cls[0])
            cls_name = result.names[cls_idx].lower()
            conf = float(box.conf[0])
            tid = int(box.id[0]) if box.id is not None else -1
            
            if tid == -1: 
                continue 
            
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy().tolist())
            
            mask_data = None
            if result.masks is not None:
                mask_data = result.masks.data[i].cpu().numpy()
                
            dets.append({
                'cls': cls_name, 'conf': conf, 'tid': tid, 
                'bbox': (x1, y1, x2, y2), 'mask': mask_data
            })
            
        filtered_dets = dets
        
        active_player_count = sum(1 for d in filtered_dets if d['tid'] in player_tids and d['cls'] == 'person')
        
        if active_player_count < 2:
            burn_in_counter += 1
        else:
            burn_in_counter = 0
            
        if burn_in_counter == 17:
            sorted_tids = sorted(movement_scores.keys(), key=lambda t: movement_scores[t], reverse=True)
            for tid in sorted_tids:
                if tid not in player_tids:
                    player_tids.add(tid)
                if len(player_tids) >= 2:
                    break
            
            print(f"[segment] Frame {frame_id}: Burn-in complete. Selected player tids: {player_tids}")
            
            for tid, files in saved_person_files.items():
                if tid not in player_tids:
                    for fpath in files:
                        try:
                            os.remove(fpath)
                        except OSError:
                            pass
                    saved_person_files[tid] = []
                    
            movement_scores.clear()
            burn_in_counter = 0
            
        for det in filtered_dets:
            x1, y1, x2, y2 = det['bbox']
            # Bound check
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            crop_img = img[y1:y2, x1:x2]
            
            mask = det['mask']
            if mask is not None:
                if mask.shape != img.shape[:2]:
                    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
                crop_mask = mask[y1:y2, x1:x2]
                crop_mask = (crop_mask * 255).astype(np.uint8)
            else:
                # Fallback solid mask if SAM/Seg fails
                crop_mask = np.ones((y2-y1, x2-x1), dtype=np.uint8) * 255
                
            tid = det['tid']
            if det['cls'] == 'person':
                if active_player_count >= 2 and tid not in player_tids:
                    continue
                    
                gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (128, 128))
                rs_mask = cv2.resize(crop_mask, (128, 128))
                
                if tid in track_history:
                    prev_gray, prev_mask = track_history[tid]
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    valid_mask = cv2.bitwise_and(prev_mask, rs_mask)
                    if np.count_nonzero(valid_mask) > 0:
                        avg_mag = np.mean(mag[valid_mask > 0])
                        movement_scores[tid] = movement_scores.get(tid, 0) + avg_mag
                        
                track_history[tid] = (gray, rs_mask)

            b, g, r_ch = cv2.split(crop_img)
            rgba = cv2.merge([b, g, r_ch, crop_mask])
            
            safe_cls = det['cls'].replace(' ', '_')
            out_name = f"frame_{frame_id:06d}_track_{tid:04d}_cls_{safe_cls}_bbox_{x1}_{y1}_{x2}_{y2}.png"
            out_path = os.path.join(seg_dir, out_name)
            cv2.imwrite(out_path, rgba)
            
            if det['cls'] == 'person':
                saved_person_files.setdefault(tid, []).append(out_path)

    print('[segment] Done.')

def get_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def get_union(boxA, boxB):
    return (
        min(boxA[0], boxB[0]),
        min(boxA[1], boxB[1]),
        max(boxA[2], boxB[2]),
        max(boxA[3], boxB[3])
    )

def mode_fuse(args: argparse.Namespace) -> None:
    """
    Stage 3: fuse
    Fuses broken tracks together based on IoU and temporal distance.
    Fuses person and racket tracks that occur in the same frame into a single player track crop.
    """
    video_path = args.input
    vname = os.path.splitext(os.path.basename(video_path))[0]
    base_dir = os.path.join(REPO_ROOT, 'assets', 'dataset', vname)
    seg_dir = os.path.join(base_dir, 'segmentations')
    frames_dir = os.path.join(base_dir, 'frames')
    
    if not os.path.exists(seg_dir):
        print(f'[fuse] Segmentations dir not found: {seg_dir}')
        return
        
    files = glob.glob(os.path.join(seg_dir, '*.png'))
    
    parsed = []
    for fpath in files:
        fname = os.path.basename(fpath)
        parts = fname.replace('.png', '').split('_')
        try:
            frame_id = int(parts[1])
            tid = int(parts[3])
            bbox_idx = parts.index('bbox')
            cls_name = '_'.join(parts[5:bbox_idx])
            x1, y1, x2, y2 = map(int, parts[bbox_idx+1:])
            parsed.append({
                'path': fpath,
                'fname': fname,
                'frame_id': frame_id,
                'tid': tid,
                'cls': cls_name,
                'bbox': (x1, y1, x2, y2)
            })
        except Exception:
            continue
            
    # Group by track id
    tracks = {}
    for p in parsed:
        tracks.setdefault(p['tid'], []).append(p)
        
    for tid in tracks:
        tracks[tid].sort(key=lambda x: x['frame_id'])
        
    print(f'[fuse] Loaded {len(tracks)} tracks. Commencing stitching...')
    
    tids = sorted(list(tracks.keys()))
    parent = {t:t for t in tids}
    
    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]
        
    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i
            
    for i in range(len(tids)):
        for j in range(i+1, len(tids)):
            tA = tids[i]
            tB = tids[j]
            A_last = tracks[tA][-1]
            B_first = tracks[tB][0]
            
            # Check A_last -> B_first
            if 0 < B_first['frame_id'] - A_last['frame_id'] < 10:
                iou = get_iou(A_last['bbox'], B_first['bbox'])
                if iou > 0.5:
                    union(tA, tB)
                    continue
                    
            A_first = tracks[tA][0]
            B_last = tracks[tB][-1]
            # Check B_last -> A_first
            if 0 < A_first['frame_id'] - B_last['frame_id'] < 10:
                iou = get_iou(B_last['bbox'], A_first['bbox'])
                if iou > 0.5:
                    union(tA, tB)

    # Apply track renames
    merged_tracks = {}
    for p in parsed:
        new_tid = find(p['tid'])
        p['new_tid'] = new_tid
        merged_tracks.setdefault(new_tid, []).append(p)
        
    for p in parsed:
        if p['new_tid'] != p['tid']:
            new_fname = f"frame_{p['frame_id']:06d}_track_{p['new_tid']:04d}_cls_{p['cls']}_bbox_{p['bbox'][0]}_{p['bbox'][1]}_{p['bbox'][2]}_{p['bbox'][3]}.png"
            new_path = os.path.join(seg_dir, new_fname)
            os.rename(p['path'], new_path)
            p['path'] = new_path
            p['tid'] = p['new_tid']

    print(f'[fuse] Stitched tracks down to {len(merged_tracks)} distinct tracks.')
    print(f'[fuse] Commencing person-racket fusion...')
    
    by_frame = {}
    for p in parsed:
        by_frame.setdefault(p['frame_id'], []).append(p)
        
    fusion_count = 0
    for frame_id, items in by_frame.items():
        persons = [x for x in items if x['cls'] == 'person']
        rackets = [x for x in items if 'racket' in x['cls']]
        
        for r in rackets:
            best_p = None
            best_iou = 0
            for p in persons:
                ixA = max(r['bbox'][0], p['bbox'][0])
                iyA = max(r['bbox'][1], p['bbox'][1])
                ixB = min(r['bbox'][2], p['bbox'][2])
                iyB = min(r['bbox'][3], p['bbox'][3])
                interArea = max(0, ixB - ixA) * max(0, iyB - iyA)
                rArea = (r['bbox'][2] - r['bbox'][0]) * (r['bbox'][3] - r['bbox'][1])
                
                # Check how much of the racket is inside the person's bounding box
                overlap_ratio = interArea / float(rArea + 1e-6)
                
                if overlap_ratio > 0.01:
                    if overlap_ratio > best_iou:
                        best_iou = overlap_ratio
                        best_p = p
                        
            if best_p is not None:
                # Fuse best_p and r
                frame_path = os.path.join(frames_dir, f'frame_{frame_id:06d}.png')
                full_img = cv2.imread(frame_path)
                if full_img is None: 
                    continue
                
                p_rgba = cv2.imread(best_p['path'], cv2.IMREAD_UNCHANGED)
                r_rgba = cv2.imread(r['path'], cv2.IMREAD_UNCHANGED)
                
                if p_rgba is None or r_rgba is None: 
                    continue
                if p_rgba.shape[2] != 4 or r_rgba.shape[2] != 4: 
                    continue
                
                # Canvas for the full mask
                full_mask = np.zeros(full_img.shape[:2], dtype=np.uint8)
                
                px1, py1, px2, py2 = best_p['bbox']
                p_mask = p_rgba[:, :, 3]
                try:
                    full_mask[py1:py2, px1:px2] = cv2.bitwise_or(full_mask[py1:py2, px1:px2], p_mask)
                except Exception:
                    pass
                    
                rx1, ry1, rx2, ry2 = r['bbox']
                r_mask = r_rgba[:, :, 3]
                try:
                    full_mask[ry1:ry2, rx1:rx2] = cv2.bitwise_or(full_mask[ry1:ry2, rx1:rx2], r_mask)
                except Exception:
                    pass
                    
                ux1, uy1, ux2, uy2 = get_union(best_p['bbox'], r['bbox'])
                ux1, uy1 = max(0, ux1), max(0, uy1)
                ux2, uy2 = min(full_img.shape[1], ux2), min(full_img.shape[0], uy2)
                
                if ux2 <= ux1 or uy2 <= uy1:
                    continue
                
                u_crop = full_img[uy1:uy2, ux1:ux2]
                u_mask = full_mask[uy1:uy2, ux1:ux2]
                
                b, g, r_ch = cv2.split(u_crop)
                u_rgba = cv2.merge([b, g, r_ch, u_mask])
                
                out_name = f"frame_{frame_id:06d}_track_{best_p['tid']:04d}_cls_player_bbox_{ux1}_{uy1}_{ux2}_{uy2}.png"
                cv2.imwrite(os.path.join(seg_dir, out_name), u_rgba)
                
                try:
                    os.remove(best_p['path'])
                    os.remove(r['path'])
                except Exception:
                    pass
                    
                fusion_count += 1
                persons.remove(best_p)

    print(f'[fuse] Performed {fusion_count} person-racket fusions into "player" tracks.')
    print('[fuse] Done.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pointstream dataset processing pipeline (3-stage).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'mode',
        choices=['filter', 'segment', 'fuse'],
        help='Pipeline stage to run'
    )
    parser.add_argument(
        '--input', required=True,
        help='Path to the source video file'
    )
    parsed = parser.parse_args()

    modes = {
        'filter': mode_filter,
        'segment': mode_segment,
        'fuse': mode_fuse,
    }
    modes[parsed.mode](parsed)
