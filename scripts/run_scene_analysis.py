# scripts/run_scene_analysis.py
#
# Runs the optimized, chunk-based streaming pipeline.
# python scripts/run_scene_analysis.py

import sys
import os
import cv2
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from pointstream.pipelines.server import ServerPipeline, StreamProcessor

def main():
    """Main function to execute the script."""
    VIDEO_FILE_PATH = "/home/itec/emanuele/Datasets/input/federer_djokovic.mp4" 

    if not os.path.exists(VIDEO_FILE_PATH):
        print(f"Error: Video file not found at '{VIDEO_FILE_PATH}'")
        return

    try:
        config = {
            # --- Stream Analysis ---
            'scene_threshold': 0.85,
            'min_scene_duration': 25, # In processed frames, not original frames
            'buffer_size': 64,
            
            # --- Performance Optimizations ---
            'frame_skip': 2, # Process 1 of every 2 frames (50fps -> 25fps)
            'processing_resolution': (640, 360), # Downscale to 360p for analysis
            'use_gpu': True, # Set to True to enable GPU acceleration for OpenCV
            
            # --- Classification ---
            'static_prefilter_threshold': 0.01, # Skip optical flow if frames are very similar
            'num_flow_checks': 10, # Number of optical flow calculations per scene
            'optical_flow_threshold': 1.0,
        }
        
        print(f"Initializing streaming pipeline for video: {VIDEO_FILE_PATH}")
        
        stream_processor = StreamProcessor(config)
        server_pipeline = ServerPipeline(config)
        
        cap = cv2.VideoCapture(VIDEO_FILE_PATH)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        with ThreadPoolExecutor() as executor:
            for frame_num in tqdm(range(frame_count), desc="Simulating Stream"):
                ret, frame_bgr = cap.read()
                if not ret: break
                
                completed_scenes = stream_processor.process_frame(frame_num, frame_bgr)
                
                for scene in completed_scenes:
                    executor.submit(server_pipeline.process_scene, scene)

            final_scenes = stream_processor.flush()
            for scene in final_scenes:
                executor.submit(server_pipeline.process_scene, scene)
            
            print("\nWaiting for all scene processing to complete...")

        cap.release()
        print("\n--- Stream Processing Complete ---")

    except (IOError, ValueError) as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
