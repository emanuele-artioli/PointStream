import argparse
import os
import glob
import subprocess
import cv2
import numpy as np
import json
import torch
import shutil
import tempfile
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

def get_available_gpus(min_free_mb=3000):
    try:
        smi = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits']).decode('utf-8')
        gpus = []
        for line in smi.strip().split('\n'):
            idx, free = line.split(', ')
            if int(free) >= min_free_mb:
                gpus.append(int(idx))
        return gpus
    except:
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        return []

# --- SPLIT MODE HELPERS ---

def _adaptive_detect_scene_chunk(chunk_info):
    chunk_path, offset, dur = chunk_info
    width, height = 64, 64
    
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", chunk_path,
        "-s", f"{width}x{height}",
        "-pix_fmt", "gray",
        "-f", "image2pipe",
        "-vcodec", "rawvideo",
        "-"
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    
    prev_frame = None
    frame_idx = 0
    diffs = []
    frame_size = width * height
    
    while True:
        raw = process.stdout.read(frame_size)
        if not raw or len(raw) != frame_size:
            break
            
        frame = np.frombuffer(raw, dtype=np.uint8).astype(np.int16)
        
        if prev_frame is not None:
            diff = np.mean(np.abs(frame - prev_frame))
            diffs.append((frame_idx, diff))
            
        prev_frame = frame
        frame_idx += 1
        
    process.stdout.close()
    process.wait()
    
    if not diffs:
        return []
        
    N = len(diffs) + 1
    fps = N / dur
    frame_duration = dur / N
    
    diff_values = np.array([d for _, d in diffs])
    min_scene_len_frames = int(.5 * fps)
    base_threshold = 10.0
    
    peaks = []
    last_cut = -min_scene_len_frames
    
    for i in range(1, len(diff_values)-1):
        if diff_values[i] > diff_values[i-1] and diff_values[i] > diff_values[i+1]:
            if diff_values[i] > base_threshold:
                start_idx = max(0, i - int(fps))
                end_idx = min(len(diff_values), i + int(fps))
                neighborhood = diff_values[start_idx:end_idx]
                
                sorted_neigh = np.sort(neighborhood)
                if len(sorted_neigh) > 10:
                    local_mean = np.mean(sorted_neigh[:-5])
                else:
                    local_mean = np.mean(sorted_neigh)
                    
                if diff_values[i] > local_mean * 3.0:
                    if i - last_cut >= min_scene_len_frames:
                        peaks.append((diffs[i][0], diff_values[i], local_mean * 3.0))
                        last_cut = i
                        
    # For perfect frame accuracy, offset the timestamp to exactly halfway between the old and new frame
    timestamps_info = [((p * frame_duration) - (frame_duration / 2) + offset, float(d), float(th)) for p, d, th in peaks]
    return timestamps_info

def _extract_clip(task):
    in_path, out_path, start, end = task
    if os.path.exists(out_path): return
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(start), "-to", str(end),
        "-i", in_path, "-map", "0:v:0", "-c:v", "libx264", "-crf", "23", "-preset", "fast", "-an", out_path
    ]
    subprocess.run(cmd)

def mode_split(args):
    raw_dir = "assets/dataset/raw"
    split_dir = "assets/dataset/split"
    os.makedirs(split_dir, exist_ok=True)
    
    videos = glob.glob(os.path.join(raw_dir, "*.mp4"))
    for video in videos:
        vname = os.path.splitext(os.path.basename(video))[0]
        out_folder = os.path.join(split_dir, vname)
        os.makedirs(out_folder, exist_ok=True)
        
        print(f"Splitting {vname}...")
        with tempfile.TemporaryDirectory() as tmpdir:
            # Chunk video losslessly
            chunk_pattern = os.path.join(tmpdir, "chunk_%04d.mp4")
            chunk_cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", video,
                "-f", "segment", "-segment_time", "60", "-c", "copy", chunk_pattern
            ]
            subprocess.run(chunk_cmd)
            
            chunks = sorted(glob.glob(os.path.join(tmpdir, "chunk_*.mp4")))
            chunk_tasks = []
            current_offset = 0.0
            
            for c in chunks:
                # get duration
                dur_cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", c]
                res = subprocess.run(dur_cmd, capture_output=True, text=True)
                dur = float(res.stdout.strip())
                chunk_tasks.append((c, current_offset, dur))
                current_offset += dur
                
            total_dur = current_offset
            scene_cuts_info = []
            print(f"  Detecting scenes across {len(chunks)} chunks using adaptive thresholding...")
            with ProcessPoolExecutor(max_workers=min(len(chunks), 64)) as executor:
                for cuts in executor.map(_adaptive_detect_scene_chunk, chunk_tasks):
                    scene_cuts_info.extend(cuts)
                    
            scene_cuts_info.sort(key=lambda x: x[0])
            
            unique_cuts = []
            last_t = -1.0
            for t, d, th in scene_cuts_info:
                if t - last_t > 0.1:
                    unique_cuts.append((t, d, th))
                    last_t = t
                    
            info_file = os.path.join(out_folder, "scene_cuts.json")
            with open(info_file, "w") as f:
                json.dump([{"timestamp": round(t, 3), "diff": round(d, 2), "threshold": round(th, 2)} for t, d, th in unique_cuts], f, indent=2)
                
            scene_cuts = [t for t, d, th in unique_cuts]
            boundaries = [0.0] + scene_cuts + [total_dur]
            
            extract_tasks = []
            for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                # Trim 0.04 seconds (~2.5 frames at 60fps) from boundaries to guarantee no frame bleeding
                clip_start = start + 0.04 if start > 0.0 else start
                clip_end = end - 0.04 if end < total_dur else end
                
                if clip_end - clip_start > 0.5:
                    out_path = os.path.join(out_folder, f"{i:03d}.mp4")
                    extract_tasks.append((video, out_path, clip_start, clip_end))
                    
            print(f"  Extracting {len(extract_tasks)} clips...")
            with ProcessPoolExecutor(max_workers=32) as executor:
                list(executor.map(_extract_clip, extract_tasks))
        print(f"Finished {vname}.")

# --- FILTER MODE HELPERS ---

def _extract_middle_frame(mp4_path):
    cap = cv2.VideoCapture(mp4_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if total > 0 and fps > 0:
        sec = (total//2)/fps
        cmd = ['ffmpeg', '-v', 'error', '-ss', str(sec), '-i', mp4_path, '-vframes', '1', '-f', 'image2pipe', '-vcodec', 'png', '-']
        try:
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            arr = np.asarray(bytearray(p.stdout), dtype=np.uint8)
            if len(arr) > 0: return (mp4_path, cv2.imdecode(arr, cv2.IMREAD_COLOR))
        except: pass
    return (mp4_path, None)

def mode_filter(args):
    split_dir = "assets/dataset/split"
    to_delete_file = "assets/dataset/to_delete.txt"
    with open(to_delete_file, "w") as out_f:
        for match_dir in sorted(glob.glob(os.path.join(split_dir, "*"))):
            if not os.path.isdir(match_dir): continue
            files = sorted(glob.glob(os.path.join(match_dir, "*.mp4")))
            if not files: continue
            
            print(f"Filtering {os.path.basename(match_dir)}...")
            frames_dict = {}
            with ProcessPoolExecutor(max_workers=64) as executor:
                for path, img in executor.map(_extract_middle_frame, files):
                    if img is not None:
                        frames_dict[path] = cv2.resize(img, (64,64)).astype(np.float32)
            
            valid_files = list(frames_dict.keys())
            if not valid_files: continue
            
            frames = np.array([frames_dict[f] for f in valid_files])
            N = len(frames)
            mse_matrix = np.zeros((N, N))
            for i in range(N):
                mse_matrix[i] = np.mean((frames - frames[i])**2, axis=(1,2,3))
            
            visited = set()
            clusters = []
            for i in range(N):
                if i in visited: continue
                visited.add(i)
                neighbors = set(np.where(mse_matrix[i] < 3000.0)[0])
                if len(neighbors) < 3: continue
                cluster = set([i])
                queue = list(neighbors - {i})
                while queue:
                    p = queue.pop(0)
                    if p not in visited:
                        visited.add(p)
                        p_neighbors = set(np.where(mse_matrix[p] < 3000.0)[0])
                        if len(p_neighbors) >= 3:
                            queue.extend(list(p_neighbors - visited))
                    cluster.add(p)
                clusters.append(list(cluster))
                
            if not clusters:
                point_indices = set()
            else:
                clusters.sort(key=len, reverse=True)
                point_indices = set(clusters[0])
                
            for i, f in enumerate(valid_files):
                if i not in point_indices:
                    out_f.write(f"{f}\n")
    print(f"Written false-positives to {to_delete_file}")

def mode_delete(args):
    fpath = "assets/dataset/to_delete.txt"
    if not os.path.exists(fpath): return
    with open(fpath, "r") as f:
        for line in f:
            p = line.strip()
            if os.path.exists(p): os.remove(p)
    print("Deleted flagged files.")

# --- DETECT MODE HELPERS ---

def compute_iou(boxA, boxB):
    xA, yA = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    xB, yB = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if (areaA + areaB - inter) == 0: return 0
    return inter / float(areaA + areaB - inter)

def _detect_scene(args):
    scene_file, out_folder, gpu_id = args
    os.makedirs(out_folder, exist_ok=True)
    from ultralytics import YOLO
    model = YOLO('assets/weights/yoloe-26x.pt')
    model.to(f'cuda:{gpu_id}')
    
    cap = cv2.VideoCapture(scene_file)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(1, int(fps / 4.0))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % frame_interval == 0:
            results = model.track(frame, persist=True, verbose=False, classes=[0, 32, 38])
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                persons, rackets, balls = [], [], []
                for box in boxes:
                    cls = int(box.cls[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    if cls == 0: persons.append(xyxy)
                    elif cls == 38: rackets.append(xyxy)
                    elif cls == 32: balls.append(xyxy)
                
                if len(persons) == 2:
                    persons.sort(key=lambda b: b[3])
                    player_far, player_near = persons[0], persons[1]
                    
                    valid_rackets = [r for r in rackets if compute_iou(r, player_near) > 0 or compute_iou(r, player_far) > 0]
                    valid_balls = []
                    p1_x, p2_x = (player_near[0]+player_near[2])/2, (player_far[0]+player_far[2])/2
                    for b in balls:
                        bx = (b[0]+b[2])/2
                        if min(p1_x, p2_x) <= bx <= max(p1_x, p2_x): valid_balls.append(b)
                        
                    if len(valid_rackets) <= 2 and len(valid_balls) <= 1:
                        cinfo = {}
                        def save_c(name, b):
                            x1, y1, x2, y2 = max(0, int(b[0])), max(0, int(b[1])), min(frame.shape[1], int(b[2])), min(frame.shape[0], int(b[3]))
                            if x2 > x1 and y2 > y1:
                                cv2.imwrite(os.path.join(out_folder, f"frame_{frame_idx}_{name}.jpg"), frame[y1:y2, x1:x2])
                                cinfo[name] = {"top_left": [x1, y1], "width": x2-x1, "height": y2-y1}
                        
                        save_c("player_far", player_far)
                        save_c("player_near", player_near)
                        for r in valid_rackets:
                            ry = (r[1]+r[3])/2
                            if abs(ry - (player_near[1]+player_near[3])/2) < abs(ry - (player_far[1]+player_far[3])/2):
                                save_c("racket_near", r)
                            else: save_c("racket_far", r)
                        for b in valid_balls: save_c("ball", b)
                        
                        with open(os.path.join(out_folder, f"frame_{frame_idx}_info.json"), "w") as f:
                            json.dump(cinfo, f)
        frame_idx += 1
    cap.release()

def mode_detect(args):
    gpus = get_available_gpus()
    if not gpus:
        print("No free GPUs.")
        return
    
    tasks = []
    split_dir = "assets/dataset/split"
    detected_dir = "assets/dataset/detected"
    gpu_idx = 0
    for m in glob.glob(os.path.join(split_dir, "*")):
        if not os.path.isdir(m): continue
        mname = os.path.basename(m)
        for s in glob.glob(os.path.join(m, "*.mp4")):
            sname = os.path.splitext(os.path.basename(s))[0]
            out = os.path.join(detected_dir, mname, sname)
            tasks.append((s, out, gpus[gpu_idx % len(gpus)]))
            gpu_idx += 1
            
    print(f"Running detection on {len(tasks)} scenes across {len(gpus)} GPUs...")
    workers = len(gpus) * 12
    with ProcessPoolExecutor(max_workers=workers) as executor:
        list(executor.map(_detect_scene, tasks))

# --- SEGMENT MODE HELPERS ---

def _segment_image(args):
    img_path, out_path, gpu_id = args
    if os.path.exists(out_path): return
    from ultralytics import SAM
    model = SAM('assets/weights/sam3.pt')
    model.to(f'cuda:{gpu_id}')
    img = cv2.imread(img_path)
    if img is None: return
    results = model(img, verbose=False)
    if results and len(results) > 0 and results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        if len(masks) > 0:
            best_mask = max(masks, key=lambda m: m.sum())
            cv2.imwrite(out_path, (best_mask * 255).astype(np.uint8))

def mode_segment(args):
    gpus = get_available_gpus()
    if not gpus: return
    
    detected_dir = "assets/dataset/detected"
    masks_dir = "assets/dataset/masks"
    tasks = []
    gpu_idx = 0
    for root, _, files in os.walk(detected_dir):
        for f in files:
            if not f.endswith(".jpg"): continue
            img_path = os.path.join(root, f)
            rel_dir = os.path.relpath(root, detected_dir)
            out_folder = os.path.join(masks_dir, rel_dir)
            os.makedirs(out_folder, exist_ok=True)
            out_path = os.path.join(out_folder, f.replace(".jpg", ".png"))
            tasks.append((img_path, out_path, gpus[gpu_idx % len(gpus)]))
            gpu_idx += 1
            
    print(f"Segmenting {len(tasks)} crops across {len(gpus)} GPUs...")
    with ProcessPoolExecutor(max_workers=len(gpus)*12) as executor:
        list(executor.map(_segment_image, tasks))

# --- FUSE MODE HELPERS ---

def _fuse_image(args):
    img_path, mask_path, out_path = args
    if not os.path.exists(mask_path): return
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None or mask is None: return
    if img.shape[:2] != mask.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    b, g, r = cv2.split(img)
    rgba = cv2.merge([b, g, r, mask])
    cv2.imwrite(out_path, rgba)

def mode_fuse(args):
    detected_dir = "assets/dataset/detected"
    masks_dir = "assets/dataset/masks"
    segmented_dir = "assets/dataset/segmented"
    tasks = []
    for root, _, files in os.walk(detected_dir):
        for f in files:
            if not f.endswith(".jpg"): continue
            img_path = os.path.join(root, f)
            rel_dir = os.path.relpath(root, detected_dir)
            mask_path = os.path.join(masks_dir, rel_dir, f.replace(".jpg", ".png"))
            out_folder = os.path.join(segmented_dir, rel_dir)
            os.makedirs(out_folder, exist_ok=True)
            out_path = os.path.join(out_folder, f.replace(".jpg", ".png"))
            tasks.append((img_path, mask_path, out_path))
            
    print(f"Fusing {len(tasks)} crops using CPU...")
    with ProcessPoolExecutor(max_workers=64) as executor:
        list(executor.map(_fuse_image, tasks))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["split", "filter", "delete", "detect", "segment", "fuse"])
    args = parser.parse_args()
    
    if args.mode == "split": mode_split(args)
    elif args.mode == "filter": mode_filter(args)
    elif args.mode == "delete": mode_delete(args)
    elif args.mode == "detect": mode_detect(args)
    elif args.mode == "segment": mode_segment(args)
    elif args.mode == "fuse": mode_fuse(args)
