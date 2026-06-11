import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch
from src.encoder.execution_pool import InlineExecutionPool
from src.encoder.actor_components import YoloPoseEstimator, SceneActor
from src.shared.dwpose_draw import draw_dwpose_canvas

scene_person = {
    'djokovic_back': {
        '012': '1', '015': '133', '033': '2327', '036': '2327', '038': '2814', 
        '041': '2814', '044': '3041', '048': '3255', '050': '3460', '055': '5', 
        '057': '268', '064': '648', '072': '900', '076': '966'
    },
    'federer_front': {
        '012': '7', '015': '134', '033': '2363', '036': '2669', '038': '2812', 
        '041': '2896', '044': '3039', '048': '3253', '050': '3455', '055': '2', 
        '057': '252', '064': '650', '072': '901', '076': '967'
    },
    'djokovic_front': {
        '018': '277', '021': '745', '024': '965', '026': '1189', '030': '1860', '079': '1671'
    },
    'federer_back': {
        '018': '274', '021': '743', '024': '961', '026': '1192', '030': '1864', '079': '1670'
    }
}

reverse_lookup = {}
for subset, mapping in scene_person.items():
    for folder, person_id in mapping.items():
        reverse_lookup[(folder, person_id)] = subset

def build_dataset():
    input_root = Path("/home/itec/emanuele/pointstream/Farzad/dataset/Tennis/djokovic_federer_4k60_3840")
    output_root = Path("assets/dataset/pix2pix")
    
    print(f"Loading YoloPoseEstimator...")
    
    estimator = YoloPoseEstimator(model_name="yolov8n-pose.pt") # fallback to nano model or whatever is available, maybe yolo26n-pose.pt but let's use yolov8n-pose.pt as ultralytics auto-downloads it
    pool = InlineExecutionPool()
    
    all_images = []
    
    for folder in sorted(input_root.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name not in ["012", "015", "033", "036", "038", "041", "044", "048", "050", "055", "057", "064", "072", "076", "018", "021", "024", "026", "030", "079"]:
            continue
        
        person_dir = folder / "person"
        if not person_dir.exists():
            continue
            
        for img_path in person_dir.glob("*.png"):
            person_id = img_path.name.split('_')[0]
            if (folder.name, person_id) in reverse_lookup:
                subset = reverse_lookup[(folder.name, person_id)]
                all_images.append((img_path, subset, folder.name))
                
    print(f"Found {len(all_images)} valid images across all subsets to process.")
    
    for img_path, subset, folder_name in tqdm(all_images):
        out_color_dir = output_root / subset
        out_skel_dir = output_root / f"output_task_1_{subset}"
        
        out_color_dir.mkdir(parents=True, exist_ok=True)
        out_skel_dir.mkdir(parents=True, exist_ok=True)
        
        new_filename = f"{folder_name}_{img_path.name}"
        out_color_path = out_color_dir / new_filename
        out_skel_path = out_skel_dir / f"skeleton_{new_filename}"
        
        if out_color_path.exists() and out_skel_path.exists():
            continue
            
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
            
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        def extract(context, deps):
            h, w = deps["image"].shape[:2]
            actor = SceneActor(track_id="dummy", class_name="player", bbox=[0, 0, w, h])
            with_pose = estimator.estimate(deps["image"], actor)
            return with_pose.pose_dw
            
        pose_dw_list = pool.execute(
            tag="gpu", 
            func=extract, 
            context={}, 
            deps={"image": img_rgb}
        )
        
        if pose_dw_list is not None:
            combined = np.zeros((1, 18, 3), dtype=np.float32)
            pose_dw = np.array(pose_dw_list, dtype=np.float32)
            if pose_dw.shape == (18, 3):
                combined[0] = pose_dw
        else:
            combined = np.zeros((1, 18, 3), dtype=np.float32)
            
        h, w = img_rgb.shape[:2]
        canvas = draw_dwpose_canvas(height=h, width=w, people_dw=combined)
        
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        # In dwpose_draw, colors are defined as (R,G,B), but cv2 uses BGR. So canvas is RGB. I need to convert to BGR to write correctly via cv2.imwrite.
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(str(out_color_path), img_bgr)
        cv2.imwrite(str(out_skel_path), canvas_bgr)
        
    print("Dataset generation complete!")

if __name__ == "__main__":
    build_dataset()
