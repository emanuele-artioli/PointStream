import argparse
import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

# PySceneDetect imports for frame-by-frame processing
from scenedetect.detectors import ContentDetector

from utils import *

def process_segment(frames, video_path, start_frame, end_frame, fps, mode):
    """Orchestrates the analysis pipeline based on the selected mode."""
    if not frames: return
    print(f"\n--- Processing Scene: Frames {start_frame}-{end_frame} (Mode: {mode}) ---")
    
    all_frame_results = []
    prompts = []

    if mode == 'text_prompt':
        keyframe = frames[len(frames) // 2]
        prompts = get_filtered_prompts(keyframe)
        if prompts:
            print(f"  -> Detected subjects: {', '.join(prompts)}")
            all_frame_results = track_objects_in_segment(frames, prompts)
    
    elif mode == 'point_prompt': # We keep the mode name for simplicity
        frame_start, frame_end = frames[0], frames[-1]
        # Use the new box prompt functions
        box_prompts = get_box_prompts_from_frames(frame_start, frame_end)
        if box_prompts:
            prompts = sorted(list(set(p['category'] for p in box_prompts)))
            # Segment with the new box-based function
            all_frame_results = segment_with_boxes(frames, box_prompts)

    if not prompts:
        print("  -> No subjects identified. Skipping detailed analysis.")
        return

    # --- The rest of the function is the same for both modes ---
    scene_type = classify_scene_motion(frames)
    context_str = "-".join([p.replace(" ", "_") for p in prompts])
    output_basename = f"{start_frame}_{end_frame}_{scene_type}_{context_str}"
    
    video_output_path = os.path.join("scenes", f"{output_basename}.mp4")
    extract_video_segment(video_path, start_frame, end_frame, fps, video_output_path)
    
    viz_output_path = os.path.join("scenes", f"{output_basename}_tracking.mp4")
    create_segmentation_video(frames, all_frame_results, viz_output_path, fps)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="PointStream Server - Scene-based Video Clipper")
    argparser.add_argument("--input_video", type=str, required=True)
    argparser.add_argument("--threshold", type=float, default=27.0)
    # --- NEW: Mode selection argument ---
    argparser.add_argument(
        "--mode",
        type=str,
        default="text_prompt",
        choices=["text_prompt", "point_prompt"],
        help="The analysis pipeline to use."
    )
    args = argparser.parse_args()

    os.makedirs("scenes", exist_ok=True)
    video_path = args.input_video
    
    total_frames, fps = get_video_properties(video_path)
    if not total_frames or not fps:
        print("Error: Could not retrieve video properties.")
    else:
        # --- REVISED STREAMING ARCHITECTURE ---
        print(f"Starting live-style processing for '{video_path}'...")
        
        # 1. Initialize the detector and video stream
        detector = ContentDetector(threshold=args.threshold)
        cap = cv2.VideoCapture(video_path)

        scene_buffer = []
        last_cut_frame_num = 0
        frame_num = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # End of video stream: process the final scene in the buffer
                if scene_buffer:
                    process_segment(scene_buffer, video_path, last_cut_frame_num, frame_num - 1, fps, args.mode)
                break

            # 2. Check for a scene cut BEFORE adding the frame to a buffer
            cut_detected = detector.process_frame(frame_num, frame)

            if cut_detected and scene_buffer:
                # A cut was detected AT this frame. The previous scene is now complete.
                # Process the buffer, which contains frames from the previous scene.
                # The end frame is the frame just before the current one.
                process_segment(scene_buffer, video_path, last_cut_frame_num, frame_num - 1, fps, args.mode)

                # 3. Start the new scene's buffer with the current frame
                scene_buffer = [frame]
                last_cut_frame_num = frame_num
            else:
                # No cut was detected, so add the current frame to the ongoing scene buffer
                scene_buffer.append(frame)

            frame_num += 1

        cap.release()
        print(f"\nProcessing complete. Clips and visualizations saved in 'scenes' folder.")
