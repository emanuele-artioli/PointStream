import argparse
import os
from collections import deque
import cv2
import numpy as np
from typing import List, Dict, Any
import matplotlib.pyplot as plt

from utils import *

def create_segmentation_video(
    frames: List[np.ndarray],
    all_frame_results: List[List[Dict[str, Any]]],
    output_path: str,
    fps: float
):
    """Creates a video with tracked segmentation masks drawn on every frame."""
    if not any(all_frame_results):
        return

    # Generate all visualized frames first
    visualized_frames = []
    track_colors = {}
    color_palette = [plt.cm.viridis(i) for i in np.linspace(0, 1, 20)]

    for i, frame in enumerate(frames):
        # Create a mutable copy for drawing on
        draw_frame = frame.copy()
        overlay = draw_frame.copy()
        
        segmented_objects = all_frame_results[i]
        
        for obj in segmented_objects:
            track_id = obj['track_id']
            if track_id not in track_colors:
                track_colors[track_id] = color_palette[len(track_colors) % len(color_palette)]
            
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
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    cv2.putText(draw_frame, label, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        final_image = cv2.addWeighted(draw_frame, 0.7, overlay, 0.3, 0)
        visualized_frames.append(final_image)
    
    # Use the new, robust FFmpeg saver
    save_frames_as_video(output_path, visualized_frames, fps)
    print(f"  -> Saved tracking visualization to '{output_path}'")

def process_segment(frames, video_path, start_frame, end_frame, fps):
    """Orchestrates the analysis and per-frame tracking of a single video segment."""
    if not frames:
        return

    print(f"\n--- Processing Segment: Frames {start_frame}-{end_frame} ---")
    
    # STEP 1: Get context from keyframe
    keyframe = frames[len(frames) // 2]
    caption = generate_caption(keyframe)
    prompts = extract_prompts_from_caption(caption)
    
    if not prompts:
        # (Handling for no prompts remains the same)
        return

    # STEP 2: Run tracking on the entire segment with a single function call
    all_frame_results = track_objects_in_segment(frames, prompts)
    
    # STEP 3: Save original clip and new tracking visualization
    scene_type = classify_scene_motion(frames)
    context_str = "-".join([p.replace(" ", "_") for p in prompts])
    output_basename = f"{start_frame}_{end_frame}_{scene_type}_{context_str}"
    
    video_output_path = os.path.join("scenes", f"{output_basename}.mp4")
    extract_video_segment(video_path, start_frame, end_frame, fps, video_output_path)

    viz_output_path = os.path.join("scenes", f"{output_basename}_tracking.mp4")
    create_segmentation_video(frames, all_frame_results, viz_output_path, fps)


if __name__ == "__main__":
    # --- Argument Parsing ---
    argparser = argparse.ArgumentParser(description="PointStream Server - Scene-based Video Clipper")
    argparser.add_argument("--input_video", type=str, required=True, help="Path to the input video file.")
    args = argparser.parse_args()

    # --- Execution & Logic ---
    os.makedirs("scenes", exist_ok=True)
    print(f"Starting scene detection for '{args.input_video}'...")

    video_path = args.input_video
    threshold = 0.3 # Threshold for SSIM-based scene change detection

    total_frames, fps = get_video_properties(video_path)
    if not total_frames or not fps:
        print("Error: Could not retrieve video properties.")
    else:
        print(f"Video properties: {total_frames} frames, {fps:.2f} FPS.")

        batch_size = fps # Process video in batches for memory efficiency
        frame_buffer = deque()
        last_cut_frame = 0
        processed_frames_count = 0

        while processed_frames_count < total_frames:
            # 1. Read a large batch of frames for efficient processing
            frame_range = (processed_frames_count, processed_frames_count + batch_size - 1)
            batch_frames = extract_frames(video_path, frame_range)
            if not batch_frames:
                break
            
            frame_buffer.extend(batch_frames)
            print(f"\nReading frames {frame_range[0]} to {frame_range[0] + len(batch_frames) - 1} into buffer...")

            # 2. Run efficient scene detection on the new batch
            # We use a copy of the batch frames for detection to not alter the buffer
            scene_changes = detect_scene_changes(list(batch_frames), threshold=threshold, analysis_window=25)
            
            # Convert local batch indices to global frame indices
            global_cut_indices = sorted([idx + processed_frames_count for idx in scene_changes.keys()])
            
            for cut_frame in global_cut_indices:
                # A segment is found. Get its frames from the start of the buffer.
                segment_len = cut_frame - last_cut_frame
                if segment_len > 0 and len(frame_buffer) >= segment_len:
                    
                    segment_frames = [frame_buffer.popleft() for _ in range(segment_len)]
                    
                    # Process and save the detected segment
                    process_segment(segment_frames, video_path, last_cut_frame, cut_frame, fps)
                    
                    last_cut_frame = cut_frame
            
            processed_frames_count += len(batch_frames)

        # After the loop, process any remaining frames in the buffer as the final scene
        if frame_buffer:
            final_frame_count = last_cut_frame + len(frame_buffer)
            # Process and save the final segment
            process_segment(list(frame_buffer), video_path, last_cut_frame, final_frame_count, fps)
            frame_buffer.clear()

        print(f"\nProcessing complete. Clips and visualizations saved in 'scenes' folder.")