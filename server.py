import argparse
import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt

# PySceneDetect imports
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

from utils import (
    get_video_properties, 
    extract_frames,
    extract_video_segment,
    classify_scene_motion,
    generate_caption,
    get_filtered_prompts,
    track_objects_in_segment,
    save_frames_as_video
)


def find_scenes(video_path: str, threshold: float = 27.0) -> List[Tuple[int, int]]:
    """Uses PySceneDetect to find all scenes in a video."""
    video = open_video(video_path)
    scene_manager = SceneManager()
    # ContentDetector is robust against fades and other gradual transitions.
    # The threshold is the amount of change in content required to trigger a new scene.
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    print(" -> Detecting scenes with PySceneDetect...")
    scene_manager.detect_scenes(video=video, show_progress=True)
    scene_list = scene_manager.get_scene_list()
    
    # Convert scene list to a list of (start_frame, end_frame) tuples
    return [(s[0].get_frames(), s[1].get_frames() - 1) for s in scene_list]


def create_segmentation_video(
    frames: List[np.ndarray],
    all_frame_results: List[List[Dict[str, Any]]],
    output_path: str,
    fps: float
):
    # This function remains unchanged
    if not any(all_frame_results): return
    h, w, _ = frames[0].shape
    save_frames_as_video(output_path, [frame for frame in frames], fps) # Simplified call
    

def process_segment(frames, video_path, start_frame, end_frame, fps):
    # This function remains unchanged
    if not frames: return
    print(f"\n--- Processing Scene: Frames {start_frame}-{end_frame} ---")
    keyframe = frames[len(frames) // 2]
    caption = generate_caption(keyframe)
    prompts = get_filtered_prompts(caption)
    if not prompts: return
    all_frame_results = track_objects_in_segment(frames, prompts)
    scene_type = classify_scene_motion(frames)
    context_str = "-".join([p.replace(" ", "_") for p in prompts])
    output_basename = f"{start_frame}_{end_frame}_{scene_type}_{context_str}"
    video_output_path = os.path.join("scenes", f"{output_basename}.mp4")
    extract_video_segment(video_path, start_frame, end_frame, fps, video_output_path)
    viz_output_path = os.path.join("scenes", f"{output_basename}_tracking.mp4")
    
    # Generate visualized frames
    visualized_frames = []
    track_colors, color_palette = {}, [plt.cm.viridis(i) for i in np.linspace(0, 1, 20)]
    for i, frame in enumerate(frames):
        draw_frame, overlay = frame.copy(), frame.copy()
        for obj in all_frame_results[i]:
            track_id = obj['track_id']
            if track_id not in track_colors: track_colors[track_id] = color_palette[len(track_colors) % len(color_palette)]
            color_bgr = [c * 255 for c in track_colors[track_id][:3]][::-1]
            h, w, _ = draw_frame.shape
            mask = cv2.resize(obj['mask'], (w, h), interpolation=cv2.INTER_NEAREST)
            overlay[mask > 0.5] = color_bgr
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(draw_frame, contours, -1, color_bgr, 2)
            label = f"ID {track_id}: {obj['prompt']}"
            if contours:
                M = cv2.moments(contours[0])
                if M['m00'] > 0:
                    cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
                    cv2.putText(draw_frame, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        final_image = cv2.addWeighted(draw_frame, 0.7, overlay, 0.3, 0)
        visualized_frames.append(final_image)
    
    save_frames_as_video(viz_output_path, visualized_frames, fps)
    print(f"  -> Saved tracking visualization to '{viz_output_path}'")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="PointStream Server - Scene-based Video Clipper")
    argparser.add_argument("--input_video", type=str, required=True, help="Path to the input video file.")
    argparser.add_argument(
        "--threshold",
        type=float,
        default=27.0,
        help="Threshold for PySceneDetect's ContentDetector. Lower is more sensitive."
    )
    args = argparser.parse_args()

    os.makedirs("scenes", exist_ok=True)
    video_path = args.input_video
    
    total_frames, fps = get_video_properties(video_path)
    if not total_frames or not fps:
        print("Error: Could not retrieve video properties.")
    else:
        # STEP 1: Find all scene boundaries in the entire video.
        scenes = find_scenes(video_path, threshold=args.threshold)
        if not scenes:
            print("No scenes detected. Processing the whole video as one segment.")
            scenes = [(0, total_frames)]

        print(f"\nDetected {len(scenes)} scenes. Starting processing...")

        # STEP 2: Loop through each detected scene.
        for start_frame, end_frame in scenes:
            # STEP 3: Extract the frames for the current scene.
            segment_frames = extract_frames(video_path, (start_frame, end_frame))
            
            # STEP 4: Run our full analysis pipeline on the scene.
            if segment_frames:
                process_segment(segment_frames, video_path, start_frame, end_frame, fps)

        print(f"\nProcessing complete. Clips and visualizations saved in 'scenes' folder.")