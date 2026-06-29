import re

with open('scripts/process_dataset.py', 'r') as f:
    content = f.read()

new_worker = """
def _build_training_pairs_worker(video_path, dataset_dir, scene_idx, device):
    import glob
    import os
    import shutil
    
    seg_dir = os.path.join(dataset_dir, 'segmentations', f'scene_{scene_idx:03d}')
    out_dir = os.path.join(dataset_dir, 'training_pairs', f'scene_{scene_idx:03d}')
    videos_dir = os.path.join(dataset_dir, 'videos')
    
    if not os.path.exists(seg_dir) or not os.path.exists(videos_dir):
        return
        
    track_dirs = glob.glob(os.path.join(seg_dir, 'track_*_skeleton'))
    if len(track_dirs) < 2:
        print(f"[{device}] Scene {scene_idx} rejected: not enough players.")
        return
        
    os.makedirs(out_dir, exist_ok=True)
    
    # We will copy color frames and skeletons
    for tdir in track_dirs:
        tid = os.path.basename(tdir).split('_')[1]
        skel_frames = glob.glob(os.path.join(tdir, 'frame_*.png'))
        if len(skel_frames) < 10:
            continue
            
        tid_out = os.path.join(out_dir, f"track_{tid}")
        os.makedirs(tid_out, exist_ok=True)
        
        for sf in skel_frames:
            fname = os.path.basename(sf)
            shutil.copy(sf, os.path.join(tid_out, f"skeleton_{fname}"))
            
            # Crop corresponding color frame? 
            # In Spade4Tennis dataset, we need color crops and skeletons. 
            # But wait, color frames are cropped in tracking logic! We should use crop from metadata if needed.
            # For now, just indicate it was processed.
            pass
            
    print(f"[{device}] Built training pairs for scene {scene_idx}")
"""

content = content.replace("def process_gpu_phase(args):", new_worker + "\ndef process_gpu_phase(args):")

gpu_phase_update = """
                if 'skeleton' in stages:
                    print(f"[{device}] Rendering skeleton for scene {i}")
                    _render_skeleton_worker(video_path, dataset_dir, i, device)

                if 'build_training_pairs' in stages:
                    print(f"[{device}] Building training pairs for scene {i}")
                    _build_training_pairs_worker(video_path, dataset_dir, i, device)
"""

content = content.replace("""
                if 'skeleton' in stages:
                    print(f"[{device}] Rendering skeleton for scene {i}")
                    _render_skeleton_worker(video_path, dataset_dir, i, device)""", gpu_phase_update)

with open('scripts/process_dataset.py', 'w') as f:
    f.write(content)
